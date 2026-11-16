//! Where GI needs geometry: a page ranking for hosts that choose what to load.
use bevy::platform::collections::HashSet;
use bevy::prelude::*;

use super::world::{PAGE_EDGE, page_distance, pages_within, pages_within_box};

/// How far toward each sun GI traces direct visibility (its sun horizon).
const SUN_HORIZON: f32 = super::tracer::MAX_PATH_DISTANCE;
/// Half a page's diagonal: a page is within this of its centre.
const PAGE_HALF_DIAGONAL: f32 = PAGE_EDGE as f32 * 0.866_025_4;

/// Where GI needs geometry, for hosts that choose which pages to load.
///
/// GI lights the pages a host declares and treats every other page as empty.
/// A host that loads only what its cameras see therefore loses geometry that
/// still shapes the visible light: walls behind the camera bounce and block
/// indirect light, and terrain between the ground and the sun shadows it.
/// This resource ranks pages by GI's need for them, so a host with a loading
/// or memory budget can spend it on the pages GI's rays reach first. Hosts
/// that keep everything nearby loaded can ignore it.
///
/// [`priority`](Self::priority) is a distance in voxels from the place GI
/// needs geometry:
/// - zero for a page holding a camera, growing with the distance from the
///   nearest active camera;
/// - for a page in a sun's shadow column (the receivers around a camera
///   swept toward the sun up to GI's 768-voxel sun horizon), its distance
///   from that column if lower, so the casters of visible shadows rank with
///   the receivers they shadow.
///
/// Load in increasing priority, as far as the budget allows. The world cache's
/// rays end within [`limit`](Self::limit) of a camera, outside the shadow
/// columns. Visible faces farther away trace their own surroundings, which a
/// host that loads what its cameras see mostly holds already.
///
/// GI updates this every frame from its active 3D cameras (their world
/// positions, one world unit per voxel), its suns and its settings, and it
/// changes whenever a camera moves. Re-rank at your own cadence, for example
/// when a camera crosses into another page.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct VoxelGiDemand {
    cameras: Vec<Vec3>,
    suns: Vec<Vec3>,
    receiver_radius: f32,
    reach: f32,
}

impl VoxelGiDemand {
    /// Demand around `cameras`, with suns in the directions `suns` (toward
    /// each sun), world-cache receivers within `receiver_radius` voxels of a
    /// camera (`VoxelGiSettings::cache_radius`) and diffuse rays `reach`
    /// voxels long (`TransportSettings::reach`). GI builds its own; this is
    /// for hosts' tests and tools.
    pub fn new(
        cameras: impl IntoIterator<Item = Vec3>,
        suns: impl IntoIterator<Item = Vec3>,
        receiver_radius: f32,
        reach: f32,
    ) -> Self {
        Self {
            cameras: cameras.into_iter().collect(),
            suns: suns.into_iter().map(Vec3::normalize_or_zero).collect(),
            receiver_radius: receiver_radius.max(0.0),
            reach: reach.max(0.0),
        }
    }

    /// The active cameras' world positions.
    pub fn cameras(&self) -> &[Vec3] {
        &self.cameras
    }

    /// Unit directions toward GI's suns.
    pub fn suns(&self) -> &[Vec3] {
        &self.suns
    }

    /// The world cache's radius around each camera (0 without a cache).
    pub fn receiver_radius(&self) -> f32 {
        self.receiver_radius
    }

    /// How far the world cache's diffuse rays travel from a camera:
    /// `receiver_radius` plus the ray reach, in voxels.
    pub fn limit(&self) -> f32 {
        self.receiver_radius + self.reach
    }

    /// The radius of a sun's shadow column: the receivers' radius, at least
    /// one page (the faces around the camera without a world cache).
    fn column_radius(&self) -> f32 {
        self.receiver_radius.max(PAGE_EDGE as f32)
    }

    /// How much GI needs `page`, as a distance in voxels (lower is sooner),
    /// or `None` before GI has seen an active camera. Shadow-column distances
    /// are measured from the page's centre less half its diagonal, so they
    /// may rank a page up to that much too near.
    pub fn priority(&self, page: IVec3) -> Option<f32> {
        let center = (page * PAGE_EDGE as i32).as_vec3() + Vec3::splat(PAGE_EDGE as f32 * 0.5);
        let column = self.column_radius();
        self.cameras
            .iter()
            .map(|&camera| {
                let mut distance = page_distance(page, camera);
                for &sun in &self.suns {
                    let along = segment_distance(center, camera, camera + sun * SUN_HORIZON);
                    let shadow = (along - PAGE_HALF_DIAGONAL).max(0.0);
                    if shadow <= column {
                        distance = distance.min(shadow);
                    }
                }
                distance
            })
            .reduce(f32::min)
    }

    /// The pages with priority at most `max_priority`, nearest first (ties in
    /// page order). Evaluates every page in the bounding boxes of the camera
    /// spheres and shadow columns that reach this priority: call it when the
    /// ranking needs refreshing, not every frame.
    pub fn pages(&self, max_priority: f32) -> Vec<(IVec3, f32)> {
        if !max_priority.is_finite() || max_priority < 0.0 {
            return Vec::new();
        }
        let mut boxes = Vec::new();
        for &camera in &self.cameras {
            boxes.push(pages_within(camera, max_priority));
            // A column page's centre is within this of the sun segment, so its
            // box is within it of the segment's bounding box.
            let r = max_priority.min(self.column_radius()) + PAGE_HALF_DIAGONAL;
            for &sun in &self.suns {
                let end = camera + sun * SUN_HORIZON;
                boxes.push(pages_within_box(camera.min(end), camera.max(end), r));
            }
        }
        let mut seen = HashSet::new();
        let mut pages = Vec::new();
        for (min, max) in boxes {
            for z in min.z..=max.z {
                for y in min.y..=max.y {
                    for x in min.x..=max.x {
                        let page = IVec3::new(x, y, z);
                        if !seen.insert(page) {
                            continue;
                        }
                        if let Some(priority) = self.priority(page)
                            && priority <= max_priority
                        {
                            pages.push((page, priority));
                        }
                    }
                }
            }
        }
        pages.sort_by(|a, b| {
            a.1.total_cmp(&b.1)
                .then_with(|| a.0.to_array().cmp(&b.0.to_array()))
        });
        pages
    }
}

/// Distance from `point` to the segment from `start` to `end`.
fn segment_distance(point: Vec3, start: Vec3, end: Vec3) -> f32 {
    let along = end - start;
    let length = along.length_squared();
    let t = if length > 0.0 {
        ((point - start).dot(along) / length).clamp(0.0, 1.0)
    } else {
        0.0
    };
    point.distance(start + along * t)
}

/// Copies the inputs only when they changed, so hosts' change detection
/// fires on movement and settings, not every frame. Called when GI reads the
/// active cameras (`pages::track_cameras`); the suns are the previous frame's
/// until this frame's sources synchronize.
pub(super) fn refresh(
    demand: &mut ResMut<VoxelGiDemand>,
    runtime: &super::VoxelGiRuntime,
    settings: &super::VoxelGiSettings,
) {
    let active = || runtime.cameras.iter().copied();
    let suns = || super::sky::sky_light_iter(&runtime.sources).map(|light| light.direction);
    let receiver_radius = settings.cache_radius().unwrap_or(0.0);
    let reach = runtime.transport.reach;
    if demand.cameras.iter().copied().eq(active())
        && demand.suns.iter().copied().eq(suns())
        && demand.receiver_radius == receiver_radius
        && demand.reach == reach
    {
        return;
    }
    let demand = &mut **demand;
    demand.cameras.clear();
    demand.cameras.extend(active());
    demand.suns.clear();
    demand.suns.extend(suns());
    demand.receiver_radius = receiver_radius;
    demand.reach = reach;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page_at(voxel: Vec3) -> IVec3 {
        voxel.div_euclid(Vec3::splat(PAGE_EDGE as f32)).as_ivec3()
    }

    #[test]
    fn pages_rank_by_distance_from_the_nearest_camera() {
        let demand = VoxelGiDemand::new([Vec3::new(16.0, 16.0, 16.0)], [], 64.0, 768.0);
        assert_eq!(demand.priority(IVec3::ZERO), Some(0.0));
        // 100 voxels behind the camera: page z -4 spans -128..-96, so its
        // near face is 112 voxels away.
        assert_eq!(
            demand.priority(page_at(Vec3::new(16.0, 16.0, -100.0))),
            Some(112.0)
        );
        assert_eq!(demand.limit(), 832.0);
        let two = VoxelGiDemand::new(
            [Vec3::splat(16.0), Vec3::new(1016.0, 16.0, 16.0)],
            [],
            64.0,
            768.0,
        );
        assert_eq!(two.priority(IVec3::new(31, 0, 0)), Some(0.0));
        assert_eq!(VoxelGiDemand::default().priority(IVec3::ZERO), None);
    }

    #[test]
    fn shadow_casters_toward_the_sun_rank_with_the_receivers() {
        let camera = Vec3::splat(16.0);
        let sun = Vec3::new(1.0, 1.0, 0.0);
        let demand = VoxelGiDemand::new([camera], [sun], 64.0, 768.0);
        // 400 voxels toward the sun: on the column's axis.
        let caster = page_at(camera + sun.normalize() * 400.0);
        assert_eq!(demand.priority(caster), Some(0.0));
        // The same distance away from the sun ranks by camera distance.
        let away = page_at(camera - sun.normalize() * 400.0);
        assert!(demand.priority(away).unwrap() > 350.0);
        // Beside the column, beyond the receivers' radius: camera distance.
        let beside = page_at(camera + sun.normalize() * 400.0 + Vec3::Z * 200.0);
        assert!(demand.priority(beside).unwrap() > 400.0);
        // Past the sun horizon the column ends.
        let past = page_at(camera + sun.normalize() * 1200.0);
        assert!(demand.priority(past).unwrap() > 1000.0);
    }

    #[test]
    fn listed_pages_are_sorted_and_complete() {
        let camera = Vec3::new(5.0, 40.0, -7.0);
        let demand = VoxelGiDemand::new([camera], [Vec3::new(0.3, 1.0, 0.2)], 64.0, 768.0);
        let pages = demand.pages(96.0);
        assert!(pages.windows(2).all(|w| w[0].1 <= w[1].1));
        assert_eq!(pages[0].1, 0.0);
        assert!(
            pages
                .iter()
                .any(|&(page, p)| page == page_at(camera) && p == 0.0)
        );
        let listed: HashSet<IVec3> = pages.iter().map(|(p, _)| *p).collect();
        // Every page within range, column pages included, is listed.
        for z in -40..=40 {
            for y in -40..=40 {
                for x in -40..=40 {
                    let page = IVec3::new(x, y, z);
                    let wanted = demand.priority(page).unwrap() <= 96.0;
                    assert_eq!(listed.contains(&page), wanted, "{page}");
                }
            }
        }
        assert!(listed.contains(&page_at(
            camera + Vec3::new(0.3, 1.0, 0.2).normalize() * 600.0
        )));
        assert!(demand.pages(f32::NAN).is_empty());
    }

    #[test]
    fn pages_touching_the_range_on_a_page_face_are_listed() {
        // Cameras on page faces and ranges ending on them: a page whose box
        // only touches the sphere has priority equal to the range.
        let at_zero = VoxelGiDemand::new([Vec3::ZERO], [], 64.0, 768.0);
        assert_eq!(at_zero.priority(IVec3::NEG_X), Some(0.0));
        assert!(
            at_zero
                .pages(0.0)
                .iter()
                .any(|&(page, _)| page == IVec3::NEG_X)
        );
        for camera in [
            Vec3::ZERO,
            Vec3::new(32.0, -64.0, 0.0),
            Vec3::new(5.0, 32.0, -7.0),
        ] {
            let demand = VoxelGiDemand::new([camera], [], 64.0, 768.0);
            for range in [0.0, 27.0, 32.0, 64.0] {
                let listed: HashSet<IVec3> = demand
                    .pages(range)
                    .into_iter()
                    .map(|(page, _)| page)
                    .collect();
                let around = page_at(camera);
                for z in -4..=4 {
                    for y in -4..=4 {
                        for x in -4..=4 {
                            let page = around + IVec3::new(x, y, z);
                            let wanted = demand.priority(page).unwrap() <= range;
                            assert_eq!(listed.contains(&page), wanted, "{camera} {range} {page}");
                        }
                    }
                }
            }
        }
    }
}
