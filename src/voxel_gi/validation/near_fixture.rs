//! Scene identity for matched-history diagnostics. These are host workloads,
//! not branches in the transport or retention algorithms.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fixture {
    CaveTorch,
    HouseSkylight,
    HouseSunPatch,
    HouseDoor,
    HouseEmitter,
    HallwayEmitter,
    HallwayOccluder,
}

impl Fixture {
    pub fn selected() -> Result<Self, String> {
        match std::env::var("VOXEL_GI_NEAR_FIXTURE").as_deref() {
            Err(_) | Ok("cave-torch") => Ok(Self::CaveTorch),
            Ok("house-skylight") => Ok(Self::HouseSkylight),
            Ok("house-sun-patch") => Ok(Self::HouseSunPatch),
            Ok("house-door") => Ok(Self::HouseDoor),
            Ok("house-emitter") => Ok(Self::HouseEmitter),
            Ok("hallway-emitter") => Ok(Self::HallwayEmitter),
            Ok("hallway-occluder") => Ok(Self::HallwayOccluder),
            Ok(value) => Err(format!("Unknown near-history fixture {value:?}")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::CaveTorch => "cave-torch",
            Self::HouseSkylight => "house-skylight",
            Self::HouseSunPatch => "house-sun-patch",
            Self::HouseDoor => "house-door",
            Self::HouseEmitter => "house-emitter",
            Self::HallwayEmitter => "hallway-emitter",
            Self::HallwayOccluder => "hallway-occluder",
        }
    }

    pub fn is_house(self) -> bool {
        matches!(
            self,
            Self::HouseSkylight | Self::HouseSunPatch | Self::HouseDoor | Self::HouseEmitter
        )
    }

    pub fn is_hallway(self) -> bool {
        matches!(self, Self::HallwayEmitter | Self::HallwayOccluder)
    }

    pub fn hallway_offsets(self) -> impl Iterator<Item = i32> {
        crate::structures::hallway::OFFSETS
            .into_iter()
            .filter(move |&d| self != Self::HallwayOccluder || d >= 4)
    }

    pub fn bounds(self, cave_origin: IVec3) -> (IVec3, IVec3) {
        if self.is_hallway() {
            let min = crate::structures::hallway::ORIGIN;
            (
                min,
                min + crate::structures::hallway::SIZE.as_ivec3() - IVec3::ONE,
            )
        } else if self.is_house() {
            (IVec3::new(-6, 10, -24), IVec3::new(6, 17, -12))
        } else {
            (
                cave_origin + IVec3::splat(4),
                cave_origin + IVec3::new(59, 43, 59),
            )
        }
    }

    pub fn view(self) -> (Vec3, Vec3, f32) {
        if self.is_hallway() {
            let offset = std::env::var("VOXEL_GI_HALLWAY_OFFSET")
                .unwrap_or_else(|_| "8".into())
                .parse::<i32>()
                .expect("HALLWAY_OFFSET must be an integer station offset");
            assert!(
                self.hallway_offsets().any(|d| d == offset),
                "unknown hallway station"
            );
            crate::structures::hallway::measurement_view(offset)
        } else if self == Self::HouseSunPatch {
            // Look down at the receiving floor, keeping the roof edit above
            // the frustum. Direct corner sunlight and its bounce are visible
            // without changing any of the captured raster geometry.
            (Vec3::new(0.5, 15.5, -18.), Vec3::new(2.5, 11., -16.5), 40.)
        } else if self.is_house() {
            // Tight upper-corner view excludes the existing low windows and
            // all edited geometry. Both states therefore share raster topology.
            (Vec3::new(0.5, 14., -17.5), Vec3::new(5.5, 17., -12.5), 25.)
        } else {
            let (eye, target) = crate::voxel_gi::validation::cave::view();
            (eye, target, 65.)
        }
    }

    pub fn edits(self, cave_origin: IVec3) -> Vec<IVec3> {
        match self {
            Self::CaveTorch => vec![cave_origin + IVec3::new(5, 10, 56)],
            Self::HouseSkylight => vec![IVec3::new(0, 17, -18)],
            Self::HouseSunPatch => (-19..=-17)
                .flat_map(|z| (-1..=1).map(move |x| IVec3::new(x, 17, z)))
                .collect(),
            Self::HouseDoor => (11..=13).map(|y| IVec3::new(-6, y, -18)).collect(),
            Self::HouseEmitter => vec![IVec3::new(-4, 13, -21)],
            Self::HallwayEmitter => vec![crate::structures::hallway::SOURCE],
            Self::HallwayOccluder => (-2..=2)
                .flat_map(|z| {
                    (0..4).map(move |y| crate::structures::hallway::SOURCE + IVec3::new(1, y, z))
                })
                .collect(),
        }
    }

    /// A conservative box/frustum test: all eight corners must lie outside
    /// the same view half-space. A center-only test could hide edited pixels.
    pub fn edit_is_offscreen(self, voxel: IVec3) -> bool {
        Self::outside_view(self.view(), voxel)
    }

    fn outside_view((eye, target, fov): (Vec3, Vec3, f32), voxel: IVec3) -> bool {
        let forward = (target - eye).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward);
        let tangent = (fov.to_radians() * 0.5).tan();
        let corners: Vec<_> = (0..8)
            .map(|i| {
                let delta = voxel.as_vec3()
                    + Vec3::new((i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32)
                    - eye;
                let z = delta.dot(forward);
                let x = delta.dot(right);
                let y = delta.dot(up);
                [
                    z,
                    z * tangent * 16. / 9. - x,
                    z * tangent * 16. / 9. + x,
                    z * tangent - y,
                    z * tangent + y,
                ]
            })
            .collect();
        (0..5).any(|plane| corners.iter().all(|c| c[plane] < 0.))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hallway_station_views_exclude_source_and_include_receiver() {
        use crate::structures::hallway::*;
        for offset in OFFSETS {
            let view = measurement_view(offset);
            assert!(
                Fixture::outside_view(view, SOURCE),
                "source visible at {offset}"
            );
            assert!(!Fixture::outside_view(view, floor_receiver(offset)));
        }
        for offset in Fixture::HallwayOccluder.hallway_offsets() {
            assert!(
                Fixture::HallwayOccluder
                    .edits(IVec3::ZERO)
                    .iter()
                    .all(|&p| Fixture::outside_view(measurement_view(offset), p))
            );
        }
    }

    #[test]
    fn house_edits_are_outside_the_entire_fixed_view() {
        for fixture in [
            Fixture::HouseSkylight,
            Fixture::HouseDoor,
            Fixture::HouseEmitter,
        ] {
            assert!(
                fixture
                    .edits(IVec3::ZERO)
                    .iter()
                    .all(|&p| fixture.edit_is_offscreen(p))
            );
            assert!(!fixture.edit_is_offscreen(IVec3::new(5, 17, -13)));
        }
        let fixture = Fixture::HouseSunPatch;
        assert_eq!(fixture.edits(IVec3::ZERO).len(), 9);
        assert!(
            fixture
                .edits(IVec3::ZERO)
                .iter()
                .all(|&p| fixture.edit_is_offscreen(p))
        );
        assert!(!fixture.edit_is_offscreen(IVec3::new(3, 10, -16)));
    }
}
