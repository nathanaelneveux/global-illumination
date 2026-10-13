//! Experimental, sampling-calibrated statistical retention. This is a change
//! prior, not a transport bound or a convergence certificate. Only weights are
//! capped; measured means, moments, sequences and display priors stay intact.
//!
//! Each receiver page retains its own bounded change journal. Unrelated distant
//! streaming cannot evict local evidence. Evicted *local* evidence folds into a
//! conservative floor instead of silently becoming "unchanged".
use std::collections::VecDeque;

use bevy::{platform::collections::HashMap, prelude::*};

use super::geometry::{Bounds, GeometryChange, NEAR_HISTORY, RELIGHT_RADIUS};

/// Calibrated against tools/quality_profiles/cave-user-2048.json. Recalibrate
/// work against the same fixed appearance goal when changing the estimator.
/// Moving the quality goal itself must also move the baseline work anchor.
pub(super) const QUALITY_PATHS: f64 = 2048.0;
const BASELINE_QUALITY_PATHS: f64 = 2048.0;
const STALE_ERROR: f64 = 0.03;
pub(super) const INFLUENCE_RADIUS: i32 = 278; // ceil(48 / sqrt(.03))
const PAGE_EVENTS: usize = 8;

#[derive(Clone, Copy)]
pub(super) enum Origin {
    Geometry,
    Sunlight,
}

fn floor(origin: Origin, distance_squared: i64) -> u32 {
    let baseline = match origin {
        Origin::Geometry if distance_squared <= i64::from(RELIGHT_RADIUS).pow(2) => NEAR_HISTORY,
        _ => 256,
    };
    (f64::from(baseline) * QUALITY_PATHS / BASELINE_QUALITY_PATHS).ceil() as u32
}

fn limit(origin: Origin, distance_squared: i64) -> u32 {
    let change =
        (f64::from(RELIGHT_RADIUS).powi(2) / (distance_squared as f64).max(2304.0)).min(1.0);
    if change <= STALE_ERROR {
        return u32::MAX;
    }
    // Ceil rounds toward retention, preserving the qualified baseline floor.
    let bias = ((QUALITY_PATHS / 6.0) * STALE_ERROR / (change - STALE_ERROR)).ceil() as u32;
    bias.max(floor(origin, distance_squared))
}

struct Event {
    revision: u64,
    origin: Origin,
    bounds: Vec<Bounds>,
}

#[derive(Default)]
struct PageHistory {
    events: VecDeque<Event>,
    /// Last evicted event at each conservative floor. Keeping separate stamps
    /// prevents an already-consumed old local edit from contaminating new sun
    /// events when the journal next overflows.
    folded: [u64; 2],
}

#[derive(Default)]
pub(super) struct SpatialHistory {
    revision: u64,
    pages: HashMap<IVec3, PageHistory>,
}

impl SpatialHistory {
    pub fn record(&mut self, change: GeometryChange, origin: Origin) {
        if change.nearby.is_empty() {
            return;
        }
        self.revision = self
            .revision
            .checked_add(1)
            .expect("retention revision exhausted");
        for (page, bounds) in change.nearby {
            let history = self.pages.entry(page).or_default();
            if history.events.len() == PAGE_EVENTS {
                let evicted = history.events.pop_front().unwrap();
                let min = page * super::world::PAGE_EDGE as i32;
                let extent = Bounds {
                    min,
                    max: min + IVec3::splat(super::world::PAGE_EDGE as i32 - 1),
                };
                let near = matches!(evicted.origin, Origin::Geometry)
                    && evicted
                        .bounds
                        .iter()
                        .any(|b| b.distance_squared(extent) <= i64::from(RELIGHT_RADIUS).pow(2));
                history.folded[usize::from(!near)] = evicted.revision;
            }
            history.events.push_back(Event {
                revision: self.revision,
                origin,
                bounds,
            });
        }
    }

    /// A page disappears only after its receivers are retired. Recreating it
    /// creates new probes without history, so its metadata can then be dropped.
    pub fn retire(&mut self, page: IVec3) {
        self.pages.remove(&page);
    }

    /// Independent from global scene freshness: a late sunlight result must
    /// still wake a receiver which already refreshed in that geometry revision.
    pub fn query(&self, voxel: IVec3, since: u64) -> (u64, u32) {
        let mut cap = u32::MAX;
        if let Some(page) = self.pages.get(&super::page_of(voxel)) {
            // Most selected receivers already consumed this page's evidence.
            // Its folded stamps necessarily precede its newest explicit event.
            if page
                .events
                .back()
                .is_none_or(|event| event.revision <= since)
            {
                return (self.revision, cap);
            }
            for (index, revision) in page.folded.iter().enumerate() {
                if since < *revision {
                    cap = cap.min(floor(
                        if index == 0 {
                            Origin::Geometry
                        } else {
                            Origin::Sunlight
                        },
                        0,
                    ));
                }
            }
            for event in page.events.iter().rev().take_while(|e| e.revision > since) {
                cap = cap.min(bounds_limit(event.origin, voxel, &event.bounds));
                if cap == floor(Origin::Geometry, 0) {
                    break;
                }
            }
        }
        (self.revision, cap)
    }
}

fn bounds_limit(origin: Origin, voxel: IVec3, bounds: &[Bounds]) -> u32 {
    let mut nearest = i64::MAX;
    for bound in bounds {
        nearest = nearest.min(bound.distance_squared(Bounds::point(voxel)));
        // No other bound can lower this event below its calibrated floor.
        if nearest <= i64::from(RELIGHT_RADIUS).pow(2) {
            return floor(origin, 0);
        }
    }
    limit(origin, nearest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_circuit_queries_match_full_bound_scans() {
        let bounds: Vec<_> = (-16..=16)
            .map(|i| Bounds::point(IVec3::new(i * 31, i * 7, i * 19)))
            .collect();
        for origin in [Origin::Geometry, Origin::Sunlight] {
            for x in -320..=320 {
                let voxel = IVec3::new(x, 32, -48);
                let expected = bounds
                    .iter()
                    .map(|b| limit(origin, b.distance_squared(Bounds::point(voxel))))
                    .min()
                    .unwrap();
                assert_eq!(bounds_limit(origin, voxel, &bounds), expected);
            }
        }
    }

    fn event(page: IVec3, point: IVec3) -> GeometryChange {
        GeometryChange {
            nearby: HashMap::from_iter([(page, vec![Bounds::point(point)])]),
        }
    }

    #[test]
    fn native_rounding_preserves_qualified_curve_and_sun_floor() {
        for (d, expected) in [
            (0, 16),
            (48, 16),
            (96, 256),
            (128, 256),
            (192, 316),
            (256, 1986),
            (384, u32::MAX),
        ] {
            assert_eq!(limit(Origin::Geometry, d * d), expected);
        }
        assert_eq!(limit(Origin::Sunlight, 0), 256);
    }

    #[test]
    fn unrelated_streaming_never_evicts_local_evidence_but_local_overflow_is_conservative() {
        let mut history = SpatialHistory::default();
        history.record(event(IVec3::ZERO, IVec3::new(192, 0, 0)), Origin::Geometry);
        for _ in 0..100 {
            history.record(
                event(IVec3::splat(100), IVec3::splat(3200)),
                Origin::Geometry,
            );
        }
        assert_eq!(history.query(IVec3::ZERO, 0).1, 316);
        let (at, _) = history.query(IVec3::ZERO, 0);
        assert_eq!(history.query(IVec3::ZERO, at).1, u32::MAX);
        for _ in 0..9 {
            history.record(event(IVec3::ZERO, IVec3::ZERO), Origin::Sunlight);
        }
        assert_eq!(history.query(IVec3::ZERO, at).1, 256);
        assert_eq!(history.query(IVec3::ZERO, 0).1, 256);
        history.retire(IVec3::ZERO);
        assert_eq!(history.query(IVec3::ZERO, 0).1, u32::MAX);
    }
}
