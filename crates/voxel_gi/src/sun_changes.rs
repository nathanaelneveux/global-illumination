//! Moving-sun shadow borders as Light retention events. The view tier reports
//! visible faces whose corner sunlight flipped (`report_sun_flips`), with the
//! reflected change; they are binned here into region-sized buckets and
//! published as Light changes, which cut the bounce history the border
//! changes. Edits' own shadows are traced on the GPU as footprints, so these
//! are mostly a moving sun's: they merge and publish a few times per second
//! (publishing each frame made every resident face test them all).
use std::time::{Duration, Instant};

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::geometry::{Bounds, ChangeKind, ChangeWeight, GeometryChange};
use super::world::{REGION_EDGE, TransportScene};

const PUBLISH_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Default)]
pub(crate) struct SunChanges {
    buckets: HashMap<IVec3, (Bounds, f32)>,
    published_at: Option<Instant>,
}

impl SunChanges {
    /// Flipped boxes (min, max, flux): single voxels, or the bound of a
    /// receipt's overflow, binned by their minimum corner.
    pub fn add(&mut self, flips: impl IntoIterator<Item = (IVec3, IVec3, f32)>) {
        for (min, max, flux) in flips {
            let bucket = min.div_euclid(IVec3::splat(REGION_EDGE as i32));
            self.buckets
                .entry(bucket)
                .and_modify(|(bounds, total)| {
                    bounds.min = bounds.min.min(min);
                    bounds.max = bounds.max.max(max);
                    *total += flux;
                })
                .or_insert((Bounds { min, max }, flux));
        }
    }

    /// The merged flips as one Light change, at most every PUBLISH_INTERVAL.
    pub fn take(&mut self, scene: &TransportScene, now: Instant) -> Option<GeometryChange> {
        if self.buckets.is_empty()
            || self.published_at.is_some_and(|at| now - at < PUBLISH_INTERVAL)
        {
            return None;
        }
        self.published_at = Some(now);
        let changes: Vec<_> = self.buckets.drain().map(|(_, change)| change).collect();
        let bounds: Vec<_> = changes.iter().map(|(bounds, _)| *bounds).collect();
        let mut change =
            GeometryChange::from_bounds(scene, &bounds, super::retention::INFLUENCE_RADIUS);
        change.weights = changes
            .into_iter()
            .map(|(bounds, flux)| (bounds, ChangeWeight { kind: ChangeKind::Light, flux }))
            .collect();
        Some(change)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flips_merge_per_region_and_publish_a_few_times_per_second() {
        let scene = TransportScene::default();
        let mut changes = SunChanges::default();
        let now = Instant::now();
        assert!(changes.take(&scene, now).is_none(), "nothing flipped");
        let point = |v: IVec3, flux| (v, v, flux);
        changes.add([point(IVec3::new(1, 2, 3), 0.5), point(IVec3::new(4, 2, 3), 0.25), point(IVec3::new(40, 2, 3), 1.0)]);
        let change = changes.take(&scene, now).unwrap();
        assert_eq!(change.weights.len(), 2, "one bound per region");
        let near = change.weights[&Bounds { min: IVec3::new(1, 2, 3), max: IVec3::new(4, 2, 3) }];
        assert_eq!(near, ChangeWeight { kind: ChangeKind::Light, flux: 0.75 });
        changes.add([(IVec3::ZERO, IVec3::splat(80), 1.0)]);
        assert!(changes.take(&scene, now + Duration::from_millis(100)).is_none(), "merging");
        assert!(changes.take(&scene, now + PUBLISH_INTERVAL).is_some());
    }
}
