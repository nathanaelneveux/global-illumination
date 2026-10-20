//! Whole-range statistical retention. This is a measured change
//! prior, not a transport bound or a convergence certificate. Only weights are
//! capped; measured means, moments, sequences and display priors stay intact.
//!
//! Every change (edit, load, unload, changed sunlight) is applied on the GPU
//! to every resident face in range when it is published, against that face's
//! own current statistics (`apply_events` in voxel_accumulate.wgsl). Nothing
//! is deferred to a face's next selection, so there are no per-page journals
//! and no conservative summaries of evicted events. The CPU keeps only a
//! per-page stamp of the newest event: receivers that have not been traced
//! since are due for an audit regardless of their earlier stability.
use bevy::{platform::collections::HashMap, prelude::*};

use super::geometry::{Bounds, ChangeWeight, GeometryChange};
use super::tracer::HISTORY_LIMIT;

/// Selected by native-material comparisons, not anchored to a near/far floor.
/// Requalify against the fixed appearance goal when changing the estimator.
const DISTANCE_SCALE: i64 = 12;
#[cfg(debug_assertions)]
pub(super) const POLICY_NAME: &str = "whole-range-power4-scale12-v1";

/// Conservative binning reach: beyond this radius H(d) cannot reduce any
/// supported history. This is derived from the statistical ceiling, not a
/// transport/light range. Later changed sunlight contributes its own origins.
pub(super) const INFLUENCE_RADIUS: i32 = {
    let threshold = HISTORY_LIMIT as i64 * DISTANCE_SCALE.pow(4);
    let mut distance = 0_i64;
    while distance.pow(4) < threshold {
        distance += 1;
    }
    distance as i32
};

/// H(d) = round((d / 12)^4), using squared distance without a square root.
/// Geometry and changed illumination share this law. Unlimited is a sentinel
/// so callers do not mark already-consumed/no-cut evidence as a new reset.
pub(super) fn limit(distance_squared: i64) -> u32 {
    debug_assert!(distance_squared >= 0);
    let scaled = distance_squared as f64 / DISTANCE_SCALE.pow(2) as f64;
    let cap = (scaled * scaled).round();
    if cap >= f64::from(HISTORY_LIMIT) {
        u32::MAX
    } else {
        cap as u32
    }
}

/// One published change, ready for the GPU: its bounds with their flux, and
/// their union (an early out). Every face in reach sums each bound's exact
/// change: an aggregate over a whole streaming event (its total flux at the
/// union's distance) overstated changes so far that measured retention cut
/// history during streaming and doubled the hallway's median settle time.
#[derive(Clone)]
pub(crate) struct RetentionEvent {
    pub bounds: Vec<Bounds>,
    pub weights: Vec<ChangeWeight>,
    pub union: Bounds,
}

impl RetentionEvent {
    /// One changed box of known or unknown flux (tests and studies).
    #[cfg(test)]
    pub fn single(bounds: Bounds, weight: ChangeWeight) -> Self {
        Self {
            bounds: vec![bounds],
            weights: vec![weight],
            union: bounds,
        }
    }
}

#[derive(Default)]
pub(super) struct SpatialHistory {
    revision: u64,
    /// Newest event revision per receiver page: wakes receivers that have
    /// not observed it. Their statistics were already capped on the GPU.
    pages: HashMap<IVec3, u64>,
    /// Published events not yet handed to the GPU worker, oldest first.
    pending: Vec<RetentionEvent>,
}

impl SpatialHistory {
    pub fn record(&mut self, change: GeometryChange) {
        if change.nearby.is_empty() {
            return;
        }
        self.revision = self
            .revision
            .checked_add(1)
            .expect("retention revision exhausted");
        let mut bounds: Vec<Bounds> = Vec::new();
        let mut seen = bevy::platform::collections::HashSet::new();
        for (page, changes) in change.nearby {
            self.pages.insert(page, self.revision);
            // A page whose nearest change permits unlimited history gains no
            // cap from it; its stamp still wakes the receivers.
            if limit(changes.minimum_distance_squared) == u32::MAX {
                continue;
            }
            for b in changes.bounds {
                if seen.insert(b) {
                    bounds.push(b);
                }
            }
        }
        if bounds.is_empty() {
            return;
        }
        // Deterministic GPU input regardless of hash iteration order.
        bounds.sort_unstable_by_key(|b| (b.min.to_array(), b.max.to_array()));
        let weights: Vec<_> = bounds
            .iter()
            .map(|b| {
                change
                    .weights
                    .get(b)
                    .copied()
                    .unwrap_or(ChangeWeight::UNKNOWN)
            })
            .collect();
        let union = bounds.iter().fold(
            Bounds {
                min: IVec3::MAX,
                max: IVec3::MIN,
            },
            |u, b| Bounds {
                min: u.min.min(b.min),
                max: u.max.max(b.max),
            },
        );
        self.pending.push(RetentionEvent {
            bounds,
            weights,
            union,
        });
    }

    /// Events for the next GPU batch.
    pub fn take_events(&mut self) -> Vec<RetentionEvent> {
        std::mem::take(&mut self.pending)
    }

    /// A page disappears only after its receivers are retired. Recreating it
    /// creates new probes without history, so its stamp can then be dropped.
    pub fn retire(&mut self, page: IVec3) {
        self.pages.remove(&page);
    }

    /// The newest event at this receiver's page, if it has not observed it.
    pub fn wake(&self, voxel: IVec3, since: u64) -> Option<u64> {
        self.pages
            .get(&super::page_of(voxel))
            .copied()
            .filter(|&revision| revision > since)
    }
}

/// CPU statement of the GPU rule for one receiver: the curve cap from the
/// nearest change, and (measured retention) the predicted luminance change,
/// None if any change in range has unknown flux. Bounds beyond the influence
/// radius contribute nothing.
#[cfg(test)]
pub(super) fn resolve(events: &[RetentionEvent], voxel: IVec3) -> (u32, Option<f32>) {
    let reach = i64::from(INFLUENCE_RADIUS).pow(2);
    let receiver = Bounds::point(voxel);
    let mut cap = u32::MAX;
    let mut change = Some(0.0f32);
    for event in events {
        let outside = event.union.distance_squared(receiver);
        if outside > reach {
            continue;
        }
        let mut exact = change.is_some();
        let mut nearest = i64::MAX;
        for (bounds, weight) in event.bounds.iter().zip(&event.weights) {
            let distance = bounds.distance_squared(receiver);
            if distance > reach {
                continue;
            }
            nearest = nearest.min(distance);
            if exact {
                match weight.change_at(distance, voxel.y) {
                    Some(delta) => *change.as_mut().unwrap() += delta,
                    None => {
                        change = None;
                        exact = false;
                    }
                }
            }
        }
        if nearest != i64::MAX {
            cap = cap.min(limit(nearest));
        }
    }
    (cap, change)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel_gi::geometry::ChangeKind;
    use crate::voxel_gi::geometry::PageChanges;

    fn event(point: IVec3, weight: Option<ChangeWeight>) -> GeometryChange {
        let mut change = GeometryChange {
            nearby: HashMap::from_iter([(
                IVec3::ZERO,
                PageChanges::new(IVec3::ZERO, vec![Bounds::point(point)]),
            )]),
            ..Default::default()
        };
        if let Some(weight) = weight {
            change.weights.insert(Bounds::point(point), weight);
        }
        change
    }

    #[test]
    fn measured_change_follows_flux_height_and_falls_back_when_unknown() {
        let change_at = |point: IVec3, weight: ChangeWeight, receiver: IVec3| {
            let mut history = SpatialHistory::default();
            history.record(event(point, Some(weight)));
            resolve(&history.take_events(), receiver).1
        };
        let edit = ChangeWeight {
            kind: ChangeKind::Edit { voxels: 1 },
            flux: 10.0,
        };
        let change = change_at(IVec3::ZERO, edit, IVec3::new(10, 0, 0)).unwrap();
        assert!((change - 10.0 / (std::f32::consts::PI * 100.0)).abs() < 1e-6);
        // A load whose top solid voxel is below the receiver replaces no sky it saw.
        let load = ChangeWeight {
            kind: ChangeKind::Load { top: 5 },
            flux: 2.0,
        };
        assert_eq!(change_at(IVec3::new(0, 5, 0), load, IVec3::new(4, 12, 0)), Some(0.0));
        let expected = 2.0 * 6.0 / (std::f32::consts::PI * (16 + 25) as f32);
        let below = change_at(IVec3::new(0, 5, 0), load, IVec3::new(4, 0, 0)).unwrap();
        assert!((below - expected).abs() < 1e-6);
        let air = ChangeWeight {
            kind: ChangeKind::Load { top: i32::MIN },
            flux: 2.0,
        };
        assert_eq!(change_at(IVec3::ZERO, air, IVec3::ONE), Some(0.0));
        // Unloads predict nothing; unknown flux falls back to the curve.
        let unload = ChangeWeight {
            kind: ChangeKind::Unload,
            flux: 0.0,
        };
        assert_eq!(change_at(IVec3::ZERO, unload, IVec3::ONE), Some(0.0));
        let mut unknown = SpatialHistory::default();
        unknown.record(event(IVec3::ZERO, None));
        assert_eq!(resolve(&unknown.take_events(), IVec3::new(64, 0, 0)), (809, None));
    }

    #[test]
    fn events_wake_their_pages_once_and_are_handed_over_once() {
        let mut history = SpatialHistory::default();
        history.record(event(IVec3::new(64, 0, 0), None));
        assert_eq!(history.wake(IVec3::ZERO, 0), Some(1));
        assert_eq!(history.wake(IVec3::ZERO, 1), None);
        let events = history.take_events();
        assert_eq!(events.len(), 1);
        assert!(history.take_events().is_empty());
        assert_eq!(resolve(&events, IVec3::ZERO).0, 809);
        history.retire(IVec3::ZERO);
        assert_eq!(history.wake(IVec3::ZERO, 0), None);
        // A page beyond the curve's reach is woken but gets no GPU event.
        let mut far = SpatialHistory::default();
        far.record(GeometryChange {
            nearby: HashMap::from_iter([(
                IVec3::ZERO,
                PageChanges::new(IVec3::ZERO, vec![Bounds::point(IVec3::new(4000, 0, 0))]),
            )]),
            ..Default::default()
        });
        assert_eq!(far.wake(IVec3::ZERO, 0), Some(1));
        assert!(far.take_events().is_empty());
    }

    #[test]
    fn native_rounding_matches_the_selected_whole_range_curve() {
        for (d, expected) in [
            (0, 0),
            (8, 0),
            (16, 3),
            (24, 16),
            (32, 51),
            (48, 256),
            (64, 809),
            (96, 4096),
            (128, 12945),
            (192, u32::MAX),
            (384, u32::MAX),
        ] {
            assert_eq!(limit(d * d), expected);
        }
        assert_eq!(limit(i64::MAX), u32::MAX);
        // Full integer-distance domain parity with the replay equation, not
        // only its example anchors. No square root is needed in production.
        for d2 in 0..=100_000 {
            let expected = ((d2 as f64).sqrt() / 12.0).powi(4).round() as u32;
            assert_eq!(
                limit(d2),
                if expected >= HISTORY_LIMIT {
                    u32::MAX
                } else {
                    expected
                }
            );
        }
    }
}
