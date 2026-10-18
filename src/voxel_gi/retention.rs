//! Whole-range statistical retention. This is a measured change
//! prior, not a transport bound or a convergence certificate. Only weights are
//! capped; measured means, moments, sequences and display priors stay intact.
//!
//! Each receiver page retains its own bounded change journal. Unrelated distant
//! streaming cannot evict local evidence. Evicted local evidence folds into a
//! bounded conservative summary, never silently becoming "unchanged".
use std::collections::VecDeque;
use std::sync::Arc;

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
const PAGE_EVENTS: usize = 8;

/// Eight-voxel distance compression, with the final edge at the derived reach.
/// These are conservative journal bins, not steps in the retention equation.
/// Distances rather than sample caps keep summary storage policy-independent.
const BAND_COUNT: usize = (INFLUENCE_RADIUS as usize).div_ceil(8) + 1;
const BAND_STARTS: [i64; BAND_COUNT] = {
    let mut starts = [0; BAND_COUNT];
    let mut index = 0;
    while index < BAND_COUNT - 1 {
        let distance = index as i64 * 8;
        starts[index] = distance * distance;
        index += 1;
    }
    starts[index] = (INFLUENCE_RADIUS as i64).pow(2);
    starts
};

/// Distance bands of evicted events. Pages with no evictions need no band
/// allocation: None means all-zero revisions, including in worker snapshots.
#[derive(Clone, Default)]
struct Summary(Option<Box<Bands>>);

/// Newest revision per distance band and origin, plus the largest flux folded
/// into it (an upper bound for measured retention; never reset, conservative).
#[derive(Clone)]
struct Bands {
    revisions: [[u64; BAND_COUNT]; 2],
    flux: [[f32; BAND_COUNT]; 2],
}

impl Summary {
    fn fold(&mut self, event: &Event) {
        let band = BAND_STARTS.partition_point(|start| *start <= event.page_distance_squared) - 1;
        let bands = self.0.get_or_insert_with(|| {
            Box::new(Bands {
                revisions: [[0; BAND_COUNT]; 2],
                flux: [[0.0; BAND_COUNT]; 2],
            })
        });
        // Loads scale with height above a receiver: fold their maximum.
        let flux = event
            .weights
            .iter()
            .map(|w| match w.kind {
                super::geometry::ChangeKind::Unknown => f32::INFINITY,
                super::geometry::ChangeKind::Load { .. } => {
                    w.flux * super::world::REGION_EDGE as f32
                }
                _ => w.flux,
            })
            .fold(0.0, f32::max);
        let slot = &mut bands.flux[event.origin as usize][band];
        *slot = slot.max(flux);
        bands.revisions[event.origin as usize][band] = event.revision;
    }

    /// The newest revision in each band suffices: if it was consumed, all
    /// older events in that band were too. An old near event cannot poison a
    /// newer far one. Distances round down, so compression never weakens an
    /// unconsumed event's cap relative to its exact receiver distance.
    fn caps(&self, since: u64) -> [u32; 2] {
        let mut caps = [u32::MAX; 2];
        if let Some(bands) = &self.0 {
            for (cap, revisions) in caps.iter_mut().zip(&bands.revisions) {
                if let Some(band) = revisions.iter().position(|revision| *revision > since) {
                    *cap = limit(BAND_STARTS[band]);
                }
            }
        }
        caps
    }

    /// Predicted change from evicted events not yet consumed: each band's
    /// largest flux at that band's (rounded-down) distance.
    fn change(&self, since: u64) -> Option<f32> {
        let Some(bands) = &self.0 else {
            return Some(0.0);
        };
        let mut total = 0.0f32;
        for (revisions, flux) in bands.revisions.iter().zip(&bands.flux) {
            for ((revision, flux), start) in revisions.iter().zip(flux).zip(BAND_STARTS) {
                if *revision > since {
                    if !flux.is_finite() {
                        return None;
                    }
                    total += flux / (std::f32::consts::PI * start.max(1) as f32);
                }
            }
        }
        Some(total)
    }
}

#[derive(Clone, Copy)]
pub(super) enum Origin {
    Geometry,
    Sunlight,
}

/// H(d) = round((d / 12)^4), using squared distance without a square root.
/// Geometry and changed illumination share this law. Unlimited is a sentinel
/// so callers do not mark already-consumed/no-cut evidence as a new reset.
fn limit(distance_squared: i64) -> u32 {
    debug_assert!(distance_squared >= 0);
    let scaled = distance_squared as f64 / DISTANCE_SCALE.pow(2) as f64;
    let cap = (scaled * scaled).round();
    if cap >= f64::from(HISTORY_LIMIT) {
        u32::MAX
    } else {
        cap as u32
    }
}

#[derive(Clone)]
struct Event {
    revision: u64,
    origin: Origin,
    bounds: Arc<Vec<Bounds>>,
    /// Measured-retention weight per bounds entry (parallel to `bounds`).
    weights: Arc<Vec<ChangeWeight>>,
    /// Union of `bounds` with summed flux, for receivers outside it: one
    /// distance instead of hundreds (streaming). Conservative, because the box
    /// is never farther than any member. None if any weight is unknown.
    aggregate: Option<Aggregate>,
    page_distance_squared: i64,
}

#[derive(Clone, Copy)]
struct Aggregate {
    bounds: Bounds,
    /// Summed flux of edits and light changes.
    flux: f32,
    /// Summed load flux per unit of height, and the highest load top.
    load_flux: f32,
    load_top: i32,
}

impl Aggregate {
    fn new(bounds: &[Bounds], weights: &[ChangeWeight]) -> Option<Self> {
        use super::geometry::ChangeKind;
        let mut aggregate = Self {
            bounds: Bounds {
                min: IVec3::MAX,
                max: IVec3::MIN,
            },
            flux: 0.0,
            load_flux: 0.0,
            load_top: i32::MIN,
        };
        for (b, w) in bounds.iter().zip(weights) {
            aggregate.bounds.min = aggregate.bounds.min.min(b.min);
            aggregate.bounds.max = aggregate.bounds.max.max(b.max);
            match w.kind {
                ChangeKind::Unknown => return None,
                ChangeKind::Unload => {}
                ChangeKind::Load { top } => {
                    aggregate.load_flux += w.flux;
                    aggregate.load_top = aggregate.load_top.max(top);
                }
                ChangeKind::Edit { .. } | ChangeKind::Light => aggregate.flux += w.flux,
            }
        }
        Some(aggregate)
    }

    fn change_at(&self, distance_squared: i64, receiver_y: i32) -> f32 {
        let height = (i64::from(self.load_top) + 1 - i64::from(receiver_y))
            .clamp(0, i64::from(super::world::REGION_EDGE)) as f32;
        (self.flux + self.load_flux * height)
            / (std::f32::consts::PI * distance_squared.max(1) as f32)
    }
}

#[derive(Clone, Default)]
struct PageHistory {
    /// Refresh evidence has its own stamp: a page-wide no-cut event needs no
    /// bounds record, but may still wake receivers that have not observed it.
    latest_revision: u64,
    events: VecDeque<Event>,
    summary: Summary,
}

#[derive(Default)]
pub(super) struct SpatialHistory {
    revision: u64,
    pages: HashMap<IVec3, Arc<PageHistory>>,
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
        for (page, changes) in change.nearby {
            // A worker holds immutable page snapshots, never this world map.
            // Copy-on-write clones at most eight event headers and an
            // allocated summary, not bounds.
            let history = Arc::make_mut(self.pages.entry(page).or_default());
            history.latest_revision = self.revision;
            // The page separation lower-bounds every receiver's distance. If
            // even that distance permits unlimited history, this event cannot
            // contribute a cap and must not evict useful exact evidence.
            if limit(changes.minimum_distance_squared) == u32::MAX {
                continue;
            }
            if history.events.len() == PAGE_EVENTS {
                let evicted = history.events.pop_front().unwrap();
                history.summary.fold(&evicted);
            }
            let weights: Vec<_> = changes
                .bounds
                .iter()
                .map(|b| {
                    change
                        .weights
                        .get(b)
                        .copied()
                        .unwrap_or(ChangeWeight::UNKNOWN)
                })
                .collect();
            history.events.push_back(Event {
                revision: self.revision,
                origin,
                aggregate: (changes.bounds.len() > 4)
                    .then(|| Aggregate::new(&changes.bounds, &weights))
                    .flatten(),
                weights: Arc::new(weights),
                bounds: Arc::new(changes.bounds),
                page_distance_squared: changes.minimum_distance_squared,
            });
        }
    }

    /// A page disappears only after its receivers are retired. Recreating it
    /// creates new probes without history, so its metadata can then be dropped.
    pub fn retire(&mut self, page: IVec3) {
        self.pages.remove(&page);
    }

    pub fn snapshot(&self, voxel: IVec3) -> Snapshot {
        Snapshot {
            revision: self.revision,
            page: self.pages.get(&super::page_of(voxel)).cloned(),
        }
    }

    #[cfg(test)]
    pub fn query(&self, voxel: IVec3, since: u64) -> (u64, u32) {
        (self.revision, self.snapshot(voxel).limit(voxel, since))
    }
}

/// Selection only checks the page stamp. Distance scans and the calibrated
/// weight curve execute on the existing tracing worker against this snapshot.
pub(super) struct Snapshot {
    pub revision: u64,
    page: Option<Arc<PageHistory>>,
}

impl Snapshot {
    pub fn pending(&self, since: u64) -> bool {
        self.page
            .as_ref()
            .is_some_and(|page| page.latest_revision > since)
    }

    /// Measured retention: predicted luminance change at this receiver from
    /// every unconsumed event, or None if any has unknown flux (use `limit`).
    pub fn change(&self, voxel: IVec3, since: u64) -> Option<f32> {
        let Some(page) = &self.page else {
            return Some(0.0);
        };
        if page
            .events
            .back()
            .is_none_or(|event| event.revision <= since)
        {
            return Some(0.0);
        }
        let mut total = page.summary.change(since)?;
        let receiver = Bounds::point(voxel);
        for event in page.events.iter().rev().take_while(|e| e.revision > since) {
            if let Some(aggregate) = &event.aggregate {
                let distance = aggregate.bounds.distance_squared(receiver);
                if distance > 0 {
                    total += aggregate.change_at(distance, voxel.y);
                    continue;
                }
            }
            for (bounds, weight) in event.bounds.iter().zip(event.weights.iter()) {
                let distance = bounds.distance_squared(receiver);
                total += weight.change_at(distance, voxel.y)?;
            }
        }
        Some(total)
    }

    pub fn limit(&self, voxel: IVec3, since: u64) -> u32 {
        let mut cap = u32::MAX;
        if let Some(page) = &self.page {
            // Most selected receivers already consumed this page's evidence.
            // Its folded stamps necessarily precede its newest explicit event.
            // Newer wake-only evidence cannot contribute any history cap.
            if page
                .events
                .back()
                .is_none_or(|event| event.revision <= since)
            {
                return cap;
            }
            cap = page.summary.caps(since).into_iter().min().unwrap();
            for event in page.events.iter().rev().take_while(|e| e.revision > since) {
                cap = cap.min(bounds_limit(voxel, &event.bounds));
                if cap == 0 {
                    break;
                }
            }
        }
        cap
    }
}

fn bounds_limit(voxel: IVec3, bounds: &[Bounds]) -> u32 {
    let mut nearest = i64::MAX;
    for bound in bounds {
        nearest = nearest.min(bound.distance_squared(Bounds::point(voxel)));
        // Only zero distance is an unconditional minimum. The old flat-near
        // shortcut would incorrectly erase the curve's within-48 variation.
        if nearest == 0 {
            return 0;
        }
    }
    limit(nearest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_circuit_queries_match_full_bound_scans() {
        let bounds: Vec<_> = (-16..=16)
            .map(|i| Bounds::point(IVec3::new(i * 31, i * 7, i * 19)))
            .collect();
        for x in -320..=320 {
            let voxel = IVec3::new(x, 32, -48);
            let expected = bounds
                .iter()
                .map(|b| limit(b.distance_squared(Bounds::point(voxel))))
                .min()
                .unwrap();
            assert_eq!(bounds_limit(voxel, &bounds), expected);
        }
        // A first bound inside 48 is not a floor; a later nearer bound wins.
        assert_eq!(
            bounds_limit(
                IVec3::ZERO,
                &[Bounds::point(IVec3::X * 32), Bounds::point(IVec3::X * 16)]
            ),
            3
        );
        assert_eq!(bounds_limit(IVec3::ZERO, &[]), u32::MAX);
    }

    fn event(page: IVec3, point: IVec3) -> GeometryChange {
        GeometryChange {
            nearby: HashMap::from_iter([(
                page,
                super::super::geometry::PageChanges::new(page, vec![Bounds::point(point)]),
            )]),
            ..Default::default()
        }
    }

    #[test]
    fn measured_change_follows_flux_height_and_falls_back_when_unknown() {
        use super::super::geometry::{ChangeKind, ChangeWeight};
        let weighted = |point: IVec3, weight: ChangeWeight| {
            let mut change = event(IVec3::ZERO, point);
            change.weights.insert(Bounds::point(point), weight);
            change
        };
        let mut history = SpatialHistory::default();
        history.record(
            weighted(
                IVec3::ZERO,
                ChangeWeight {
                    kind: ChangeKind::Edit { voxels: 1 },
                    flux: 10.0,
                },
            ),
            Origin::Geometry,
        );
        let change = history
            .snapshot(IVec3::new(10, 0, 0))
            .change(IVec3::new(10, 0, 0), 0)
            .unwrap();
        assert!((change - 10.0 / (std::f32::consts::PI * 100.0)).abs() < 1e-6);
        // A load whose top solid voxel is below the receiver replaces no sky it saw.
        let mut loads = SpatialHistory::default();
        loads.record(
            weighted(
                IVec3::new(0, 5, 0),
                ChangeWeight {
                    kind: ChangeKind::Load { top: 5 },
                    flux: 2.0,
                },
            ),
            Origin::Geometry,
        );
        let above = IVec3::new(4, 12, 0);
        assert_eq!(loads.snapshot(above).change(above, 0), Some(0.0));
        let below = IVec3::new(4, 0, 0);
        let expected = 2.0 * 6.0 / (std::f32::consts::PI * (16 + 25) as f32);
        assert!((loads.snapshot(below).change(below, 0).unwrap() - expected).abs() < 1e-6);
        let mut air = SpatialHistory::default();
        air.record(
            weighted(
                IVec3::ZERO,
                ChangeWeight {
                    kind: ChangeKind::Load { top: i32::MIN },
                    flux: 2.0,
                },
            ),
            Origin::Geometry,
        );
        assert_eq!(air.snapshot(IVec3::ONE).change(IVec3::ONE, 0), Some(0.0));
        // Unloads predict nothing; unknown flux falls back to the curve.
        let mut unload = SpatialHistory::default();
        unload.record(
            weighted(
                IVec3::ZERO,
                ChangeWeight {
                    kind: ChangeKind::Unload,
                    flux: 0.0,
                },
            ),
            Origin::Geometry,
        );
        assert_eq!(unload.snapshot(IVec3::ONE).change(IVec3::ONE, 0), Some(0.0));
        let mut unknown = SpatialHistory::default();
        unknown.record(event(IVec3::ZERO, IVec3::ZERO), Origin::Geometry);
        assert_eq!(unknown.snapshot(IVec3::ONE).change(IVec3::ONE, 0), None);
        // Consumed evidence predicts no change.
        assert_eq!(
            history.snapshot(IVec3::ONE).change(IVec3::ONE, 1),
            Some(0.0)
        );
    }

    #[test]
    fn worker_snapshots_preserve_the_selected_epoch_across_publication_and_retirement() {
        let mut history = SpatialHistory::default();
        history.record(event(IVec3::ZERO, IVec3::new(64, 0, 0)), Origin::Geometry);
        let snapshot = history.snapshot(IVec3::ZERO);
        assert!(snapshot.pending(0));
        history.record(event(IVec3::ZERO, IVec3::ZERO), Origin::Sunlight);
        assert_eq!(snapshot.limit(IVec3::ZERO, 0), 809);
        assert_eq!(history.query(IVec3::ZERO, snapshot.revision).1, 0);
        history.retire(IVec3::ZERO);
        assert_eq!(snapshot.limit(IVec3::ZERO, 0), 809);
        assert!(!history.snapshot(IVec3::ZERO).pending(0));
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

    #[test]
    fn unrelated_streaming_never_evicts_local_evidence_but_local_overflow_is_conservative() {
        let mut history = SpatialHistory::default();
        history.record(event(IVec3::ZERO, IVec3::new(64, 0, 0)), Origin::Geometry);
        for _ in 0..100 {
            history.record(
                event(IVec3::splat(100), IVec3::splat(3200)),
                Origin::Geometry,
            );
        }
        assert_eq!(history.query(IVec3::ZERO, 0).1, 809);
        let (at, _) = history.query(IVec3::ZERO, 0);
        assert_eq!(history.query(IVec3::ZERO, at).1, u32::MAX);
        for _ in 0..9 {
            history.record(event(IVec3::ZERO, IVec3::ZERO), Origin::Sunlight);
        }
        assert_eq!(history.query(IVec3::ZERO, at).1, 0);
        assert_eq!(history.query(IVec3::ZERO, 0).1, 0);
        history.retire(IVec3::ZERO);
        assert_eq!(history.query(IVec3::ZERO, 0).1, u32::MAX);
    }

    #[test]
    fn bands_allocate_only_on_overflow() {
        let voxel = IVec3::ZERO;
        let mut candidate = SpatialHistory::default();
        for i in 0..PAGE_EVENTS {
            let point = IVec3::new(199 + i as i32, 0, 0);
            candidate.record(event(voxel, point), Origin::Geometry);
            assert!(candidate.pages[&voxel].summary.0.is_none());
        }
        let snapshot = candidate.snapshot(voxel);
        candidate.record(event(voxel, IVec3::new(212, 0, 0)), Origin::Geometry);
        assert!(candidate.pages[&voxel].summary.0.is_some());
        assert_eq!(snapshot.limit(voxel, 0), u32::MAX);
        assert_eq!(candidate.query(voxel, 0).1, 38416);
        assert_eq!(candidate.query(voxel, 1).1, u32::MAX);
    }

    #[test]
    fn replacing_the_eighth_exact_event_with_its_summary_can_lower_retention() {
        // Page [0,31] is 48 voxels from the change, but this receiver is 79
        // away. Folding one event earlier reduces its distance lower bound.
        let voxel = IVec3::ZERO;
        let mut history = SpatialHistory::default();
        history.record(event(voxel, IVec3::new(79, 0, 0)), Origin::Geometry);
        for _ in 1..PAGE_EVENTS {
            history.record(event(voxel, IVec3::new(200, 0, 0)), Origin::Geometry);
        }
        assert_eq!(history.query(voxel, 0).1, 1878);
        let eight = history.snapshot(voxel);
        let page = Arc::make_mut(history.pages.get_mut(&voxel).unwrap());
        let oldest = page.events.pop_front().unwrap();
        page.summary.fold(&oldest);
        assert_eq!(page.events.len(), PAGE_EVENTS - 1);
        assert_eq!(history.query(voxel, 0).1, 256);
        assert_eq!(eight.limit(voxel, 0), 1878);
    }

    #[test]
    fn one_less_event_is_not_a_fixed_summary_memory_offset() {
        let mut events = VecDeque::new();
        let empty = Event {
            revision: 1,
            origin: Origin::Geometry,
            bounds: Arc::new(Vec::new()),
            weights: Arc::new(Vec::new()),
            aggregate: None,
            page_distance_squared: 0,
        };
        for _ in 0..PAGE_EVENTS {
            events.push_back(empty.clone());
        }
        let allocated = events.capacity();
        events.pop_front();
        assert_eq!(
            events.capacity(),
            allocated,
            "removal does not shrink the ring allocation"
        );
        assert!(std::mem::size_of::<Event>() < std::mem::size_of::<[[u64; BAND_COUNT]; 2]>());
        println!(
            "RETENTION_MEMORY event_header={} bounds_element={} vec_header={} band_payload={} capacity_after_eight={} capacity_after_pop={}",
            std::mem::size_of::<Event>(),
            std::mem::size_of::<Bounds>(),
            std::mem::size_of::<Vec<Bounds>>(),
            std::mem::size_of::<[[u64; BAND_COUNT]; 2]>(),
            allocated,
            events.capacity()
        );
    }

    #[test]
    fn no_cut_events_preserve_wakeups_without_evicting_exact_history() {
        // At the derived binning boundary, no supported weight can be cut.
        for origin in [Origin::Geometry, Origin::Sunlight] {
            let voxel = IVec3::ZERO;
            let distant = IVec3::new(31 + INFLUENCE_RADIUS, 0, 0);
            let distance = Bounds::point(distant).distance_squared(Bounds::page(voxel));
            assert!(distance <= i64::from(INFLUENCE_RADIUS).pow(2));
            assert_eq!(limit(distance), u32::MAX);
            let mut history = SpatialHistory::default();
            history.record(event(voxel, IVec3::new(64, 0, 0)), origin);
            let before = history.snapshot(voxel);
            for _ in 0..32 {
                history.record(event(voxel, distant), origin);
            }
            let snapshot = history.snapshot(voxel);
            assert_eq!(history.pages[&voxel].events.len(), 1);
            assert!(history.pages[&voxel].summary.0.is_none());
            assert_eq!(snapshot.limit(voxel, 0), 809);
            assert_eq!(snapshot.limit(voxel, 1), u32::MAX);
            assert!(snapshot.pending(32));
            assert!(!snapshot.pending(33));
            assert!(
                !before.pending(1),
                "in-flight snapshots must not consume newer wakeups"
            );
            history.retire(voxel);
            assert!(snapshot.pending(1));
            assert!(!history.snapshot(voxel).pending(1));
        }
    }

    #[test]
    fn empty_journals_can_wake_without_history_caps_or_band_storage() {
        for page in [IVec3::ZERO, IVec3::splat(-3)] {
            let voxel = page * 32;
            let mut history = SpatialHistory::default();
            history.record(
                event(page, voxel + IVec3::new(31 + INFLUENCE_RADIUS, 0, 0)),
                Origin::Sunlight,
            );
            assert!(history.pages[&page].events.is_empty());
            assert_eq!(history.pages[&page].events.capacity(), 0);
            assert!(history.pages[&page].summary.0.is_none());
            let snapshot = history.snapshot(voxel);
            assert!(snapshot.pending(0));
            assert_eq!(snapshot.limit(voxel, 0), u32::MAX);
            assert!(!snapshot.pending(snapshot.revision));
        }
    }

    #[test]
    fn page_edge_receivers_keep_evidence_even_when_the_page_origin_is_too_far() {
        let page = IVec3::ZERO;
        let near_side = IVec3::new(31, 0, 0);
        let mut history = SpatialHistory::default();
        history.record(event(page, IVec3::new(200, 0, 0)), Origin::Geometry);
        assert_eq!(history.pages[&page].events.len(), 1);
        assert_eq!(history.query(IVec3::ZERO, 0).1, u32::MAX);
        assert_eq!(history.query(near_side, 0).1, 39339);
    }

    #[test]
    fn bands_conservatively_quantize_distance_not_the_policy() {
        assert!(BAND_STARTS.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(BAND_STARTS[0], 0);
        assert_eq!(BAND_COUNT, 25);
        assert_eq!(
            *BAND_STARTS.last().unwrap(),
            i64::from(INFLUENCE_RADIUS).pow(2)
        );
        let mut previous = 0;
        for d2 in 0..=100_000 {
            let band = BAND_STARTS.partition_point(|start| *start <= d2) - 1;
            let cap = limit(BAND_STARTS[band]);
            assert!(previous <= cap && cap <= limit(d2));
            previous = cap;
        }
        assert_eq!(limit(BAND_STARTS[7]), 474);
        assert_eq!(limit(i64::from(INFLUENCE_RADIUS).pow(2)), u32::MAX);
    }

    #[test]
    fn banded_journal_never_exceeds_exact_history_at_every_epoch() {
        for page in [IVec3::splat(-3), IVec3::ZERO, IVec3::new(5, -2, 1)] {
            let mut candidate = SpatialHistory::default();
            let mut exact = Vec::new();
            for i in 0..40 {
                let point = page * 32 + IVec3::new((i * 59) % 330 - 16, (i * 11) % 80, -12);
                let origin = if i % 3 == 0 {
                    Origin::Geometry
                } else {
                    Origin::Sunlight
                };
                exact.push((origin, point));
                candidate.record(event(page, point), origin);
                for since in 0..=candidate.revision {
                    for x in [0, 15, 31] {
                        for y in [0, 15, 31] {
                            for z in [0, 15, 31] {
                                let voxel = page * 32 + IVec3::new(x, y, z);
                                let full = exact
                                    .iter()
                                    .skip(since as usize)
                                    .map(|(_, p)| {
                                        limit(
                                            Bounds::point(*p)
                                                .distance_squared(Bounds::point(voxel)),
                                        )
                                    })
                                    .min()
                                    .unwrap_or(u32::MAX);
                                let cap = candidate.query(voxel, since).1;
                                assert!(cap <= full);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn consumed_near_bands_do_not_poison_new_far_events_or_worker_snapshots() {
        let voxel = IVec3::ZERO;
        let mut history = SpatialHistory::default();
        history.record(event(voxel, voxel), Origin::Geometry);
        let before = history.snapshot(voxel);
        for _ in 0..20 {
            history.record(event(voxel, IVec3::new(200, 0, 0)), Origin::Geometry);
        }
        let snapshot = history.snapshot(voxel);
        assert_eq!(snapshot.limit(voxel, 0), 0);
        assert_eq!(snapshot.limit(voxel, 1), 38416); // page edge is 169, rounded to 168
        assert_eq!(snapshot.limit(voxel, 21), u32::MAX);
        history.record(event(voxel, voxel), Origin::Sunlight);
        assert_eq!(history.query(voxel, 21).1, 0);
        history.retire(voxel);
        assert_eq!(before.limit(voxel, 0), 0);
        assert_eq!(snapshot.limit(voxel, 1), 38416);
        assert_eq!(history.query(voxel, 1).1, u32::MAX);
    }
}
