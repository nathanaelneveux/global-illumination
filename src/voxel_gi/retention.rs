//! Sampling-calibrated statistical retention. This is a change
//! prior, not a transport bound or a convergence certificate. Only weights are
//! capped; measured means, moments, sequences and display priors stay intact.
//!
//! Each receiver page retains its own bounded change journal. Unrelated distant
//! streaming cannot evict local evidence. Evicted local evidence folds into
//! revision-stamped distance bands, never silently becoming "unchanged".
use std::collections::VecDeque;
use std::sync::Arc;

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

/// Eight-voxel bands, with exact edges at the near-floor discontinuity and
/// the current prior's no-cut boundary. The summary stores distances, not
/// sample caps: recalibrating estimator work does not bake in old weights.
const BAND_COUNT: usize = 37;
const BAND_STARTS: [i64; BAND_COUNT] = {
    let mut starts = [0; BAND_COUNT];
    let mut index = 0;
    let mut distance = 0;
    while distance <= 272 {
        starts[index] = distance * distance;
        index += 1;
        if distance == 48 {
            index += 1;
            starts[index - 1] = 48 * 48 + 1;
        }
        distance += 8;
    }
    starts[index] = 76800; // 48² / STALE_ERROR
    starts
};

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(super) enum SummaryMode {
    #[default]
    Floors,
    DistanceBands,
}

impl SummaryMode {
    pub fn launch(value: Option<&str>) -> Result<Self, String> {
        match value {
            None | Some("floors") => Ok(Self::Floors),
            Some("distance-bands") => Ok(Self::DistanceBands),
            Some(value) => Err(format!(
                "unknown VOXEL_GI_RETENTION_SUMMARY={value:?}; choose floors or distance-bands"
            )),
        }
    }

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
    DistanceBands(Box<[[u64; BAND_COUNT]; 2]>),
}

impl Summary {
    fn new(mode: SummaryMode) -> Self {
        match mode {
            SummaryMode::Floors => Self::Floors([0; 2]),
            SummaryMode::DistanceBands => Self::DistanceBands(Box::new([[0; BAND_COUNT]; 2])),
        }
    }

    fn fold(&mut self, event: &Event) {
        let distance = event.page_distance_squared;
        let stamp = match self {
            Self::Floors(stamps) => {
                let near = matches!(event.origin, Origin::Geometry)
                    && distance <= i64::from(RELIGHT_RADIUS).pow(2);
                &mut stamps[usize::from(!near)]
            }
            Self::DistanceBands(stamps) => {
                let band = BAND_STARTS.partition_point(|start| *start <= distance) - 1;
                &mut stamps[event.origin as usize][band]
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
                for (i, origin) in [Origin::Geometry, Origin::Sunlight].into_iter().enumerate() {
                    if stamps[i] > since {
                        caps[i] = floor(origin, 0);
                    }
                }
            }
            Self::DistanceBands(stamps) => {
                for (i, origin) in [Origin::Geometry, Origin::Sunlight].into_iter().enumerate() {
                    if let Some(band) = stamps[i].iter().position(|revision| *revision > since) {
                        caps[i] = limit(origin, BAND_STARTS[band]);
                    }
                }
            }
        }
        caps
    }
}

/// Demo launch comparison. The library's ledger itself does not read process
/// environment, and an invalid choice must not silently benchmark another arm.
pub(super) fn launch_enabled(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("calibrated") => Ok(true),
        Some("baseline") => Ok(false),
        Some(value) => Err(format!(
            "unknown VOXEL_GI_RETENTION={value:?}; choose calibrated or baseline"
        )),
    }
}

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

#[derive(Clone)]
struct Event {
    revision: u64,
    origin: Origin,
    bounds: Arc<Vec<Bounds>>,
    page_distance_squared: i64,
    #[cfg(debug_assertions)]
    audit_sun: Option<Arc<Vec<super::validation::retention_audit::SunEvidence>>>,
}

#[derive(Clone)]
struct PageHistory {
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
            // Copy-on-write clones at most eight production event headers, not
            // bounds. Opt-in attribution has a larger diagnostic-only sidecar.
            let history = Arc::make_mut(
                self.pages
                    .entry(page)
                    .or_insert_with(|| Arc::new(PageHistory::new(self.mode))),
            );
            if history.events.len() == PAGE_EVENTS {
                let evicted = history.events.pop_front().unwrap();
                history.summary.fold(&evicted);
            }
            history.events.push_back(Event {
                revision: self.revision,
                origin,
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
            .is_some_and(|page| page.events.back().is_some_and(|e| e.revision > since))
    }

    pub fn limit(&self, voxel: IVec3, since: u64) -> u32 {
        let mut cap = u32::MAX;
        if let Some(page) = &self.page {
            // Most selected receivers already consumed this page's evidence.
            // Its folded stamps necessarily precede its newest explicit event.
            if page
                .events
                .back()
                .is_none_or(|event| event.revision <= since)
            {
                return cap;
            }
            cap = page.summary.caps(since).into_iter().min().unwrap();
            for event in page.events.iter().rev().take_while(|e| e.revision > since) {
                cap = cap.min(bounds_limit(event.origin, voxel, &event.bounds));
                if cap == floor(Origin::Geometry, 0) {
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
        result.caps[index] = result.caps[index].min(bounds_limit(self.origin, voxel, &self.bounds));
        for bounds in self.bounds.iter() {
            result.distance_squared[index] =
                result.distance_squared[index].min(bounds.distance_squared(Bounds::point(voxel)));
        }
        if let Some(evidence) = &self.audit_sun {
            for point in evidence.iter() {
                let distance = Bounds::point(point.position).distance_squared(Bounds::point(voxel));
                let index = 4 + point.kind as usize;
                result.caps[index] = result.caps[index].min(limit(Origin::Sunlight, distance));
                result.distance_squared[index - 2] =
                    result.distance_squared[index - 2].min(distance);
            }
        }
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
    fn qualified_policy_is_default_and_control_is_explicit() {
        assert_eq!(launch_enabled(None), Ok(true));
        assert_eq!(launch_enabled(Some("calibrated")), Ok(true));
        assert_eq!(launch_enabled(Some("baseline")), Ok(false));
        assert!(launch_enabled(Some("typo")).is_err());
    }

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
            nearby: HashMap::from_iter([(
                page,
                super::super::geometry::PageChanges::new(page, vec![Bounds::point(point)]),
            )]),
            ..Default::default()
        }
    }

    #[test]
    fn worker_snapshots_preserve_the_selected_epoch_across_publication_and_retirement() {
        let mut history = SpatialHistory::default();
        history.record(event(IVec3::ZERO, IVec3::new(192, 0, 0)), Origin::Geometry);
        let snapshot = history.snapshot(IVec3::ZERO);
        assert!(snapshot.pending(0));
        history.record(event(IVec3::ZERO, IVec3::ZERO), Origin::Sunlight);
        assert_eq!(snapshot.limit(IVec3::ZERO, 0), 316);
        assert_eq!(history.query(IVec3::ZERO, snapshot.revision).1, 256);
        history.retire(IVec3::ZERO);
        assert_eq!(snapshot.limit(IVec3::ZERO, 0), 316);
        assert!(!history.snapshot(IVec3::ZERO).pending(0));
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
        assert_eq!(result.cap, 256);
        assert_eq!(result.caps[4], u32::MAX);
        assert_eq!(result.caps[5], 256);
        assert_eq!(result.caps[0], u32::MAX);
    }

    #[test]
    fn shadow_journal_measures_lost_distance_without_changing_the_cap() {
        let old = Event {
            revision: 1,
            origin: Origin::Geometry,
            bounds: Arc::new(vec![Bounds::point(IVec3::new(256, 0, 0))]),
            page_distance_squared: 225 * 225,
            audit_sun: None,
        };
        let latest = Event {
            revision: 2,
            origin: Origin::Geometry,
            bounds: Arc::new(vec![Bounds::point(IVec3::new(300, 0, 0))]),
            page_distance_squared: 269 * 269,
            audit_sun: None,
        };
        let mut page = PageHistory {
            events: VecDeque::from([latest.clone()]),
            summary: Summary::Floors([0, 1]),
            audit_events: VecDeque::from([old, latest]),
            audit_dropped: 0,
        };
        let snapshot = Snapshot {
            revision: 2,
            page: Some(Arc::new(page.clone())),
        };
        let decision = snapshot.explain(IVec3::ZERO, 0);
        assert_eq!(decision.cap, 256);
        assert_eq!(decision.unfolded.unwrap().cap, 1986);
        assert_eq!(snapshot.limit(IVec3::ZERO, 0), 256);
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

    #[test]
    fn summary_launch_control_is_explicit() {
        assert_eq!(SummaryMode::launch(None), Ok(SummaryMode::Floors));
        assert_eq!(SummaryMode::launch(Some("floors")), Ok(SummaryMode::Floors));
        assert_eq!(
            SummaryMode::launch(Some("distance-bands")),
            Ok(SummaryMode::DistanceBands)
        );
        assert!(SummaryMode::launch(Some("typo")).is_err());
    }

    #[test]
    fn bands_preserve_monotonic_caps_and_both_policy_boundaries() {
        assert!(BAND_STARTS.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(BAND_STARTS[0], 0);
        assert_eq!(*BAND_STARTS.last().unwrap(), 76800);
        for origin in [Origin::Geometry, Origin::Sunlight] {
            let mut previous = 0;
            for d2 in 0..=100_000 {
                let band = BAND_STARTS.partition_point(|start| *start <= d2) - 1;
                let cap = limit(origin, BAND_STARTS[band]);
                assert!(previous <= cap);
                assert!(floor(origin, d2) <= cap && cap <= limit(origin, d2));
                previous = cap;
            }
        }
        assert_eq!(limit(Origin::Geometry, BAND_STARTS[7]), 256);
        assert_eq!(limit(Origin::Sunlight, 76800), u32::MAX);
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
                                    .map(|(o, p)| {
                                        limit(
                                            *o,
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
            history.record(event(voxel, IVec3::new(287, 0, 0)), Origin::Geometry);
        }
        let snapshot = history.snapshot(voxel);
        assert_eq!(snapshot.limit(voxel, 0), 16);
        assert_eq!(snapshot.limit(voxel, 1), 1986); // page edge is 256 away
        assert_eq!(snapshot.limit(voxel, 21), u32::MAX);
        history.record(event(voxel, voxel), Origin::Sunlight);
        assert_eq!(history.query(voxel, 21).1, 256);
        history.retire(voxel);
        assert_eq!(before.limit(voxel, 0), 16);
        assert_eq!(snapshot.limit(voxel, 1), 1986);
        assert_eq!(history.query(voxel, 1).1, u32::MAX);
    }
}
