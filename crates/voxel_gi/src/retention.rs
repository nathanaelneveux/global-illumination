//! Whole-range statistical retention. This is a measured change
//! prior, not a transport bound or a convergence certificate. Only weights are
//! capped; measured means, moments, sequences and display priors stay intact.
//!
//! Every change (edit, load, unload, changed sunlight) is applied on the GPU
//! to every resident face in range when it is published, against that face's
//! own current statistics (`apply_events` in voxel_accumulate.wgsl). Nothing
//! is deferred to a face's next selection, so there are no per-page journals
//! and no conservative summaries of evicted events. Analytic source changes
//! travel the same way (sources.rs history caps). Any event in reach also
//! revokes a face's stability on the GPU, so GPU selection audits it.
use bevy::prelude::*;

use super::geometry::{Bounds, ChangeWeight, GeometryChange};
use super::tracer::HISTORY_LIMIT;

/// Selected by native-material comparisons, not anchored to a near/far floor.
/// Requalify against the fixed appearance goal when changing the estimator.
///
/// It is the measured rule at one assumed flux, not a separate law: keeping
/// n = c sigma^2 / D^2 samples for a change D = F / (pi d^2) gives
/// n = c pi^2 sigma^2 d^4 / F^2, which is (d / 12)^4 for F = 144 pi sigma
/// (c = 1). An edit of unknown flux is assumed to change a face 12 voxels
/// away by one standard deviation of its samples per sample (it keeps one),
/// at 24 voxels by a quarter (it keeps 16). Known flux replaces the
/// assumption; this applies only when the edit was not measured.
const DISTANCE_SCALE: i64 = 12;

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
/// The GPU's `curve_limit` evaluates it; this CPU statement serves tests.
/// Geometry and changed illumination share this law. Unlimited is a sentinel
/// so callers do not mark already-consumed/no-cut evidence as a new reset.
#[cfg(test)]
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
/// A source change instead caps every face: `source_cap` everywhere, 4
/// within `spheres`.
#[derive(Clone)]
pub(crate) struct RetentionEvent {
    pub bounds: Vec<Bounds>,
    pub weights: Vec<ChangeWeight>,
    pub union: Bounds,
    /// Bounds containing a player's edit: faces near them whose history the
    /// change binds become urgent on the GPU (publication used to wake them).
    pub urgent: Vec<bool>,
    pub source_cap: Option<u32>,
    pub spheres: Vec<(Vec3, f32)>,
}

impl RetentionEvent {
    /// One changed box of known or unknown flux (tests and studies).
    #[cfg(test)]
    pub fn single(bounds: Bounds, weight: ChangeWeight) -> Self {
        Self {
            bounds: vec![bounds],
            weights: vec![weight],
            union: bounds,
            urgent: vec![false],
            source_cap: None,
            spheres: Vec::new(),
        }
    }
}

#[derive(Default)]
pub(super) struct SpatialHistory {
    /// Published events not yet handed to the GPU worker, oldest first.
    pending: Vec<RetentionEvent>,
}

impl SpatialHistory {
    pub fn record(&mut self, change: GeometryChange, edits: &[IVec3]) {
        if let Some(event) = Self::event(change, edits) {
            self.push(event);
        }
    }

    pub fn push(&mut self, event: RetentionEvent) {
        self.pending.push(event);
    }

    /// The retention event for a change: deduplicated, deterministic bounds
    /// and their weights. Pure, so the scene worker builds it (a streaming
    /// build of 768 pages binned ~5k bounds in 12-15 ms of main thread).
    pub fn event(change: GeometryChange, edits: &[IVec3]) -> Option<RetentionEvent> {
        let mut bounds: Vec<Bounds> = Vec::new();
        let mut seen = bevy::platform::collections::HashSet::new();
        for (_, changes) in change.nearby {
            // Reach is tested per face on the GPU: a change can wake faces
            // (an audit) even where it can cap nothing.
            for b in changes.bounds {
                if seen.insert(b) {
                    bounds.push(b);
                }
            }
        }
        if bounds.is_empty() {
            return None;
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
        let urgent = bounds
            .iter()
            .map(|b| edits.iter().any(|&voxel| b.distance_squared(Bounds::point(voxel)) == 0))
            .collect();
        Some(RetentionEvent {
            bounds,
            weights,
            union,
            urgent,
            source_cap: None,
            spheres: Vec::new(),
        })
    }

    /// An analytic source change (sources.rs): every face keeps at most
    /// `cap` samples, at most 4 within any of `spheres` (center, radius).
    pub fn record_source(&mut self, cap: u32, spheres: &[(Vec3, f32)]) {
        let point = Bounds::point(IVec3::ZERO);
        self.pending.push(RetentionEvent {
            bounds: Vec::new(),
            weights: Vec::new(),
            union: point,
            urgent: Vec::new(),
            source_cap: Some(cap),
            spheres: spheres.to_vec(),
        });
    }

    /// Events for the next GPU batch.
    pub fn has_events(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn take_events(&mut self) -> Vec<RetentionEvent> {
        std::mem::take(&mut self.pending)
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
    for event in events.iter().filter(|e| e.source_cap.is_none()) {
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

/// CPU statement of the GPU's source caps for one receiver.
#[cfg(test)]
pub(super) fn source_limit(events: &[RetentionEvent], voxel: IVec3) -> u32 {
    let mut cap = u32::MAX;
    for event in events {
        if let Some(base) = event.source_cap {
            let near = event
                .spheres
                .iter()
                .any(|(center, radius)| voxel.as_vec3().distance_squared(*center) <= radius * radius);
            cap = cap.min(if near { base.min(4) } else { base });
        }
    }
    cap
}

/// Whether any event revokes this receiver's stability on the GPU.
#[cfg(test)]
pub(super) fn woken(events: &[RetentionEvent], voxel: IVec3) -> bool {
    let reach = i64::from(INFLUENCE_RADIUS).pow(2);
    events.iter().any(|event| {
        event.source_cap.is_some() || event.union.distance_squared(Bounds::point(voxel)) <= reach
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::ChangeKind;
    use bevy::platform::collections::HashMap;
    use crate::geometry::PageChanges;

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
            history.record(event(point, Some(weight)), &[]);
            resolve(&history.take_events(), receiver).1
        };
        let edit = ChangeWeight {
            kind: ChangeKind::Edit { voxels: 1, measure: false },
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
        unknown.record(event(IVec3::ZERO, None), &[]);
        assert_eq!(resolve(&unknown.take_events(), IVec3::new(64, 0, 0)), (809, None));
    }

    #[test]
    fn events_are_handed_over_once_and_sources_cap_everywhere() {
        let mut history = SpatialHistory::default();
        history.record(event(IVec3::new(64, 0, 0), None), &[]);
        history.record_source(64, &[(Vec3::new(0.0, 0.0, 300.0), 10.0)]);
        let events = history.take_events();
        assert_eq!(events.len(), 2);
        assert!(history.take_events().is_empty());
        assert_eq!(resolve(&events, IVec3::ZERO).0, 809);
        assert_eq!(source_limit(&events, IVec3::ZERO), 64);
        assert_eq!(source_limit(&events, IVec3::new(0, 0, 295)), 4);
        assert!(woken(&events, IVec3::new(4000, 0, 0)), "source changes wake everyone");
        let mut far = SpatialHistory::default();
        far.record(event(IVec3::new(4000, 0, 0), None), &[]);
        let far = far.take_events();
        assert_eq!(resolve(&far, IVec3::ZERO), (u32::MAX, Some(0.0)));
        assert!(!woken(&far, IVec3::ZERO), "out of reach");
        assert_eq!(source_limit(&far, IVec3::ZERO), u32::MAX);
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
