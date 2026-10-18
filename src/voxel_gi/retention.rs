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

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub enum SummaryMode {
    /// Conservative comparison: evicted events retain the nearest possible cap.
    Floors,
    /// Default: preserve conservative distance information for evicted events.
    #[default]
    DistanceBands,
}

impl SummaryMode {
    pub(super) fn launch(value: Option<&str>) -> Result<Self, String> {
        match value {
            Some("floors") => Ok(Self::Floors),
            None | Some("distance-bands") => Ok(Self::DistanceBands),
            Some(value) => Err(format!(
                "unknown VOXEL_GI_RETENTION_SUMMARY={value:?}; choose floors or distance-bands"
            )),
        }
    }

    #[cfg(debug_assertions)]
    pub fn name(self) -> &'static str {
        match self {
            Self::Floors => "floors",
            Self::DistanceBands => "distance-bands",
        }
    }
}

#[derive(Clone)]
enum Summary {
    Floors([u64; 2]),
    /// Pages with no evictions need no band allocation. None has the same
    /// meaning as all-zero revisions, including in immutable worker snapshots.
    DistanceBands(Option<Box<Bands>>),
}

/// Newest revision per distance band and origin, plus the largest flux folded
/// into it (an upper bound for measured retention; never reset, conservative).
#[derive(Clone)]
struct Bands {
    revisions: [[u64; BAND_COUNT]; 2],
    flux: [[f32; BAND_COUNT]; 2],
}

impl Summary {
    fn new(mode: SummaryMode) -> Self {
        match mode {
            SummaryMode::Floors => Self::Floors([0; 2]),
            SummaryMode::DistanceBands => Self::DistanceBands(None),
        }
    }

    fn fold(&mut self, event: &Event) {
        let distance = event.page_distance_squared;
        let stamp = match self {
            Self::Floors(stamps) => &mut stamps[event.origin as usize],
            Self::DistanceBands(bands) => {
                let band = BAND_STARTS.partition_point(|start| *start <= distance) - 1;
                let bands = bands.get_or_insert_with(|| {
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
                        super::geometry::ChangeKind::Load { .. } => w.flux * super::world::REGION_EDGE as f32,
                        _ => w.flux,
                    })
                    .fold(0.0, f32::max);
                let slot = &mut bands.flux[event.origin as usize][band];
                *slot = slot.max(flux);
                &mut bands.revisions[event.origin as usize][band]
            }
        };
        *stamp = event.revision;
    }

    /// The newest revision in each band suffices: if it was consumed, all
    /// older events in that band were too. An old near event cannot poison a
    /// newer far one. Distances round down, so compression never weakens an
    /// unconsumed event's cap relative to its exact receiver distance.
    fn caps(&self, since: u64) -> [u32; 2] {
        let mut caps = [u32::MAX; 2];
        match self {
            Self::Floors(stamps) => {
                for (cap, revision) in caps.iter_mut().zip(stamps) {
                    if *revision > since {
                        *cap = limit(0);
                    }
                }
            }
            Self::DistanceBands(Some(bands)) => {
                for i in 0..caps.len() {
                    if let Some(band) = bands.revisions[i].iter().position(|revision| *revision > since) {
                        caps[i] = limit(BAND_STARTS[band]);
                    }
                }
            }
            Self::DistanceBands(None) => {}
        }
        caps
    }

    /// Predicted change from evicted events not yet consumed: each band's
    /// largest flux at that band's (rounded-down) distance. None for floors.
    fn change(&self, since: u64) -> Option<f32> {
        match self {
            Self::Floors(stamps) => stamps.iter().all(|r| *r <= since).then_some(0.0),
            Self::DistanceBands(None) => Some(0.0),
            Self::DistanceBands(Some(bands)) => {
                let mut total = 0.0f32;
                for i in 0..2 {
                    for band in 0..BAND_COUNT {
                        if bands.revisions[i][band] > since {
                            let flux = bands.flux[i][band];
                            if !flux.is_finite() {
                                return None;
                            }
                            total += flux / (std::f32::consts::PI * BAND_STARTS[band].max(1) as f32);
                        }
                    }
                }
                Some(total)
            }
        }
    }
}

/// Demo launch comparison. The library's ledger itself does not read process
/// environment, and an invalid choice must not silently benchmark another arm.
pub(super) fn launch_enabled(value: Option<&str>) -> Result<bool, String> {
    match value {
        // Preserve the previous spelling as an alias, not the old equation.
        None | Some("whole-range" | "calibrated") => Ok(true),
        Some("baseline") => Ok(false),
        Some(value) => Err(format!(
            "unknown VOXEL_GI_RETENTION={value:?}; choose whole-range or baseline (calibrated is an alias)"
        )),
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
    page_distance_squared: i64,
    #[cfg(debug_assertions)]
    audit_sun: Option<Arc<Vec<super::validation::retention_audit::SunEvidence>>>,
}

#[derive(Clone)]
struct PageHistory {
    /// Refresh evidence has its own stamp: a page-wide no-cut event needs no
    /// bounds record, but may still wake receivers that have not observed it.
    latest_revision: u64,
    events: VecDeque<Event>,
    summary: Summary,
    /// Only opt-in watched pages retain this diagnostic shadow journal. It
    /// cannot change the eight-event production journal or its fallback.
    #[cfg(debug_assertions)]
    audit_events: VecDeque<Event>,
    #[cfg(debug_assertions)]
    audit_dropped: u64,
}

impl PageHistory {
    fn new(mode: SummaryMode) -> Self {
        Self {
            latest_revision: 0,
            events: VecDeque::new(),
            summary: Summary::new(mode),
            #[cfg(debug_assertions)]
            audit_events: VecDeque::new(),
            #[cfg(debug_assertions)]
            audit_dropped: 0,
        }
    }
}

#[derive(Default)]
pub(super) struct SpatialHistory {
    revision: u64,
    mode: SummaryMode,
    pages: HashMap<IVec3, Arc<PageHistory>>,
}

impl SpatialHistory {
    pub fn new(mode: SummaryMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    #[cfg(debug_assertions)]
    pub fn summary_mode(&self) -> SummaryMode {
        self.mode
    }

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
            // Copy-on-write clones at most eight production event headers and
            // an allocated summary, not bounds. Opt-in attribution has a larger
            // diagnostic-only sidecar.
            let history = Arc::make_mut(
                self.pages
                    .entry(page)
                    .or_insert_with(|| Arc::new(PageHistory::new(self.mode))),
            );
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
            let weights = changes
                .bounds
                .iter()
                .map(|b| change.weights.get(b).copied().unwrap_or(ChangeWeight::UNKNOWN))
                .collect();
            history.events.push_back(Event {
                revision: self.revision,
                origin,
                weights: Arc::new(weights),
                bounds: Arc::new(changes.bounds),
                page_distance_squared: changes.minimum_distance_squared,
                #[cfg(debug_assertions)]
                audit_sun: change.audit_sun.clone(),
            });
            #[cfg(debug_assertions)]
            if super::validation::retention_audit::watches_page(page) {
                if history.audit_events.len() == 1024 {
                    history.audit_dropped = history.audit_events.pop_front().unwrap().revision;
                }
                history
                    .audit_events
                    .push_back(history.events.back().unwrap().clone());
            }
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
    /// Full-scan diagnostic decomposition. The production cap still comes from
    /// `limit`; raw-corner categories do not alter coalescing or journal order.
    #[cfg(debug_assertions)]
    pub fn explain(
        &self,
        voxel: IVec3,
        since: u64,
    ) -> super::validation::retention_audit::Decision {
        use super::validation::retention_audit::Decision;
        let mut result = Decision::new(since, self.revision);
        if let Some(page) = &self.page {
            result.caps[2..4].copy_from_slice(&page.summary.caps(since));
            for event in page.events.iter().filter(|event| event.revision > since) {
                event.explain(voxel, &mut result);
            }
            if !page.audit_events.is_empty() && since >= page.audit_dropped {
                let mut unfolded = Decision::new(since, self.revision);
                for event in page
                    .audit_events
                    .iter()
                    .rev()
                    .take_while(|e| e.revision > since)
                {
                    event.explain(voxel, &mut unfolded);
                }
                unfolded.cap = unfolded.caps[0].min(unfolded.caps[1]);
                result.unfolded = Some(Box::new(unfolded));
            }
        }
        result.cap = self.limit(voxel, since);
        result
    }

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
        if page.events.back().is_none_or(|event| event.revision <= since) {
            return Some(0.0);
        }
        let mut total = page.summary.change(since)?;
        for event in page.events.iter().rev().take_while(|e| e.revision > since) {
            for (bounds, weight) in event.bounds.iter().zip(event.weights.iter()) {
                let distance = bounds.distance_squared(Bounds::point(voxel));
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

#[cfg(debug_assertions)]
impl Event {
    fn explain(&self, voxel: IVec3, result: &mut super::validation::retention_audit::Decision) {
        let index = usize::from(matches!(self.origin, Origin::Sunlight));
        result.caps[index] = result.caps[index].min(bounds_limit(voxel, &self.bounds));
        for bounds in self.bounds.iter() {
            result.distance_squared[index] =
                result.distance_squared[index].min(bounds.distance_squared(Bounds::point(voxel)));
        }
        if let Some(evidence) = &self.audit_sun {
            for point in evidence.iter() {
                let distance = Bounds::point(point.position).distance_squared(Bounds::point(voxel));
                let index = 4 + point.kind as usize;
                result.caps[index] = result.caps[index].min(limit(distance));
                result.distance_squared[index - 2] =
                    result.distance_squared[index - 2].min(distance);
            }
        }
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
    fn qualified_policy_is_default_and_control_is_explicit() {
        assert_eq!(launch_enabled(None), Ok(true));
        assert_eq!(launch_enabled(Some("whole-range")), Ok(true));
        assert_eq!(launch_enabled(Some("calibrated")), Ok(true));
        assert_eq!(launch_enabled(Some("baseline")), Ok(false));
        assert!(launch_enabled(Some("typo")).is_err());
    }

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
        let mut history = SpatialHistory::new(SummaryMode::DistanceBands);
        history.record(
            weighted(IVec3::ZERO, ChangeWeight { kind: ChangeKind::Edit { voxels: 1 }, flux: 10.0 }),
            Origin::Geometry,
        );
        let change = history.snapshot(IVec3::new(10, 0, 0)).change(IVec3::new(10, 0, 0), 0).unwrap();
        assert!((change - 10.0 / (std::f32::consts::PI * 100.0)).abs() < 1e-6);
        // A load whose top solid voxel is below the receiver replaces no sky it saw.
        let mut loads = SpatialHistory::new(SummaryMode::DistanceBands);
        loads.record(
            weighted(IVec3::new(0, 5, 0), ChangeWeight { kind: ChangeKind::Load { top: 5 }, flux: 2.0 }),
            Origin::Geometry,
        );
        let above = IVec3::new(4, 12, 0);
        assert_eq!(loads.snapshot(above).change(above, 0), Some(0.0));
        let below = IVec3::new(4, 0, 0);
        let expected = 2.0 * 6.0 / (std::f32::consts::PI * (16 + 25) as f32);
        assert!((loads.snapshot(below).change(below, 0).unwrap() - expected).abs() < 1e-6);
        // Unloads predict nothing; unknown flux falls back to the curve.
        let mut unload = SpatialHistory::new(SummaryMode::DistanceBands);
        unload.record(weighted(IVec3::ZERO, ChangeWeight { kind: ChangeKind::Unload, flux: 0.0 }), Origin::Geometry);
        assert_eq!(unload.snapshot(IVec3::ONE).change(IVec3::ONE, 0), Some(0.0));
        let mut unknown = SpatialHistory::new(SummaryMode::DistanceBands);
        unknown.record(event(IVec3::ZERO, IVec3::ZERO), Origin::Geometry);
        assert_eq!(unknown.snapshot(IVec3::ONE).change(IVec3::ONE, 0), None);
        // Consumed evidence predicts no change.
        assert_eq!(history.snapshot(IVec3::ONE).change(IVec3::ONE, 1), Some(0.0));
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
    fn diagnostic_decomposition_matches_production_across_overflow_and_epochs() {
        let mut history = SpatialHistory::default();
        for i in 0..24 {
            history.record(
                event(IVec3::ZERO, IVec3::new(i * 17, 0, 0)),
                if i % 3 == 0 {
                    Origin::Geometry
                } else {
                    Origin::Sunlight
                },
            );
            for since in 0..=history.revision {
                for x in [-300, -48, 0, 48, 128, 192, 256, 384] {
                    let voxel = IVec3::new(x, 0, 0);
                    let snapshot = history.snapshot(voxel);
                    let decision = snapshot.explain(voxel, since);
                    assert_eq!(decision.cap, snapshot.limit(voxel, since));
                    assert_eq!(decision.cap, *decision.caps[..4].iter().min().unwrap());
                }
            }
        }
    }

    #[test]
    fn raw_sun_evidence_is_diagnostic_and_does_not_split_production_events() {
        use super::super::validation::retention_audit::{SunEvidence, SunKind};
        let mut history = SpatialHistory::default();
        let mut change = event(IVec3::ZERO, IVec3::ZERO);
        change.audit_sun = Some(Arc::new(vec![
            SunEvidence {
                position: IVec3::ZERO,
                kind: SunKind::FirstMeasurement,
            },
            SunEvidence {
                position: IVec3::splat(1000),
                kind: SunKind::KnownChange,
            },
        ]));
        history.record(change, Origin::Sunlight);
        let result = history.snapshot(IVec3::ZERO).explain(IVec3::ZERO, 0);
        assert_eq!(result.revision, 1);
        assert_eq!(result.cap, 0);
        assert_eq!(result.caps[4], u32::MAX);
        assert_eq!(result.caps[5], 0);
        assert_eq!(result.caps[0], u32::MAX);
    }

    #[test]
    fn shadow_journal_measures_lost_distance_without_changing_the_cap() {
        let old = Event {
            revision: 1,
            origin: Origin::Geometry,
            bounds: Arc::new(vec![Bounds::point(IVec3::new(128, 0, 0))]),
weights: Arc::new(Vec::new()),
            page_distance_squared: 97 * 97,
            audit_sun: None,
        };
        let latest = Event {
            revision: 2,
            origin: Origin::Geometry,
            bounds: Arc::new(vec![Bounds::point(IVec3::new(200, 0, 0))]),
weights: Arc::new(Vec::new()),
            page_distance_squared: 169 * 169,
            audit_sun: None,
        };
        let mut page = PageHistory {
            latest_revision: 2,
            events: VecDeque::from([latest.clone()]),
            summary: Summary::Floors([1, 0]),
            audit_events: VecDeque::from([old, latest]),
            audit_dropped: 0,
        };
        let snapshot = Snapshot {
            revision: 2,
            page: Some(Arc::new(page.clone())),
        };
        let decision = snapshot.explain(IVec3::ZERO, 0);
        assert_eq!(decision.cap, 0);
        assert_eq!(decision.unfolded.unwrap().cap, 12945);
        assert_eq!(snapshot.limit(IVec3::ZERO, 0), 0);
        page.audit_dropped = 1;
        let snapshot = Snapshot {
            revision: 2,
            page: Some(Arc::new(page)),
        };
        assert!(snapshot.explain(IVec3::ZERO, 0).unfolded.is_none());
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
    fn summary_launch_control_is_explicit() {
        assert_eq!(SummaryMode::launch(None), Ok(SummaryMode::DistanceBands));
        assert_eq!(SummaryMode::default(), SummaryMode::DistanceBands);
        assert_eq!(SummaryMode::launch(Some("floors")), Ok(SummaryMode::Floors));
        assert_eq!(
            SummaryMode::launch(Some("distance-bands")),
            Ok(SummaryMode::DistanceBands)
        );
        assert!(SummaryMode::launch(Some("typo")).is_err());
    }

    #[test]
    fn bands_allocate_only_on_overflow_and_match_the_control_before_it() {
        let voxel = IVec3::ZERO;
        let mut control = SpatialHistory::new(SummaryMode::Floors);
        let mut candidate = SpatialHistory::new(SummaryMode::DistanceBands);
        for i in 0..PAGE_EVENTS {
            let point = IVec3::new(199 + i as i32, 0, 0);
            control.record(event(voxel, point), Origin::Geometry);
            candidate.record(event(voxel, point), Origin::Geometry);
            assert!(matches!(
                candidate.pages[&voxel].summary,
                Summary::DistanceBands(None)
            ));
            for since in 0..=candidate.revision {
                assert_eq!(candidate.query(voxel, since), control.query(voxel, since));
            }
        }
        let snapshot = candidate.snapshot(voxel);
        candidate.record(event(voxel, IVec3::new(212, 0, 0)), Origin::Geometry);
        assert!(matches!(
            candidate.pages[&voxel].summary,
            Summary::DistanceBands(Some(_))
        ));
        assert_eq!(snapshot.limit(voxel, 0), u32::MAX);
        assert_eq!(candidate.query(voxel, 0).1, 38416);
        assert_eq!(candidate.query(voxel, 1).1, u32::MAX);
    }

    #[test]
    fn replacing_the_eighth_exact_event_with_its_summary_can_lower_retention() {
        // Page [0,31] is 48 voxels from the change, but this receiver is 79
        // away. Folding one event earlier reduces its distance lower bound.
        let voxel = IVec3::ZERO;
        let mut history = SpatialHistory::new(SummaryMode::DistanceBands);
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
            page_distance_squared: 0,
            audit_sun: None,
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
            assert!(matches!(
                history.pages[&voxel].summary,
                Summary::DistanceBands(None)
            ));
            assert_eq!(snapshot.limit(voxel, 0), 809);
            assert_eq!(snapshot.limit(voxel, 1), u32::MAX);
            assert!(snapshot.pending(32));
            assert!(!snapshot.pending(33));
            assert!(
                !before.pending(1),
                "in-flight snapshots must not consume newer wakeups"
            );
            let decision = snapshot.explain(voxel, 0);
            assert_eq!(decision.cap, *decision.caps[..4].iter().min().unwrap());
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
            assert!(matches!(
                history.pages[&page].summary,
                Summary::DistanceBands(None)
            ));
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
    fn banded_journal_is_between_floor_control_and_exact_history_at_every_epoch() {
        for page in [IVec3::splat(-3), IVec3::ZERO, IVec3::new(5, -2, 1)] {
            let mut control = SpatialHistory::new(SummaryMode::Floors);
            let mut candidate = SpatialHistory::new(SummaryMode::DistanceBands);
            let mut exact = Vec::new();
            for i in 0..40 {
                let point = page * 32 + IVec3::new((i * 59) % 330 - 16, (i * 11) % 80, -12);
                let origin = if i % 3 == 0 {
                    Origin::Geometry
                } else {
                    Origin::Sunlight
                };
                exact.push((origin, point));
                control.record(event(page, point), origin);
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
                                assert!(control.query(voxel, since).1 <= cap && cap <= full);
                                let explanation = candidate.snapshot(voxel).explain(voxel, since);
                                assert_eq!(cap, *explanation.caps[..4].iter().min().unwrap());
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
        let mut history = SpatialHistory::new(SummaryMode::DistanceBands);
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
