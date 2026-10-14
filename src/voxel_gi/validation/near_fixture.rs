//! Scene identity for matched-history diagnostics. These are host workloads,
//! not branches in the transport or retention algorithms.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fixture {
    CaveTorch,
    HouseSkylight,
    HouseDoor,
    HouseEmitter,
}

impl Fixture {
    pub fn selected() -> Result<Self, String> {
        match std::env::var("VOXEL_GI_NEAR_FIXTURE").as_deref() {
            Err(_) | Ok("cave-torch") => Ok(Self::CaveTorch),
            Ok("house-skylight") => Ok(Self::HouseSkylight),
            Ok("house-door") => Ok(Self::HouseDoor),
            Ok("house-emitter") => Ok(Self::HouseEmitter),
            Ok(value) => Err(format!("Unknown near-history fixture {value:?}")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::CaveTorch => "cave-torch",
            Self::HouseSkylight => "house-skylight",
            Self::HouseDoor => "house-door",
            Self::HouseEmitter => "house-emitter",
        }
    }

    pub fn is_house(self) -> bool {
        self != Self::CaveTorch
    }

    pub fn bounds(self, cave_origin: IVec3) -> (IVec3, IVec3) {
        if self.is_house() {
            (IVec3::new(-6, 10, -24), IVec3::new(6, 17, -12))
        } else {
            (
                cave_origin + IVec3::splat(4),
                cave_origin + IVec3::new(59, 43, 59),
            )
        }
    }

    pub fn view(self) -> (Vec3, Vec3, f32) {
        if self.is_house() {
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
            Self::HouseDoor => (11..=13).map(|y| IVec3::new(-6, y, -18)).collect(),
            Self::HouseEmitter => vec![IVec3::new(-4, 13, -21)],
        }
    }

    /// A conservative box/frustum test: all eight corners must lie outside
    /// the same view half-space. A center-only test could hide edited pixels.
    pub fn edit_is_offscreen(self, voxel: IVec3) -> bool {
        let (eye, target, fov) = self.view();
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
    }
}
