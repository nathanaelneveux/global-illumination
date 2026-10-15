//! Shared edit/streaming history policy. Distance controls retained statistical
//! weight, never visibility or whether a receiver must refresh. Plans are built
//! on the scene worker; selection checks only nearby changes in its page bin.
use bevy::math::I64Vec3;
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;

use super::world::{AIR, PAGE_EDGE, REGION_EDGE, TransportScene};
#[cfg(debug_assertions)]
use std::sync::Arc;

pub(super) const RELIGHT_RADIUS: i32 = 48;
pub(super) const NEAR_HISTORY: u32 = 16;
const FAR_HISTORY: u32 = 256;
pub(super) const UNTRACKED_HISTORY: u32 = 64;

/// One worker-side comparison supplies scene publication and spatial history
/// retention. A voxel's optical identity is scene data, regardless of whether
/// that material reflects, emits, or occludes energy.
pub(super) struct SceneChanges {
    pub changed: bool,
    pub transport: Option<GeometryChange>,
}

impl SceneChanges {
    #[cfg(test)]
    pub fn between(previous: &TransportScene, next: &TransportScene) -> Self {
        Self::with_radius(previous, next, RELIGHT_RADIUS)
    }

    pub fn with_radius(previous: &TransportScene, next: &TransportScene, radius: i32) -> Self {
        let changed = next.changed_regions(previous);
        if changed.is_empty() {
            return Self {
                changed: false,
                transport: None,
            };
        }
        let mut occupied_min = IVec3::MAX;
        let mut occupied_max = IVec3::MIN;
        for (&coordinate, region) in &previous.regions {
            if region.any_solid {
                occupied_min = occupied_min.min(coordinate);
                occupied_max = occupied_max.max(coordinate);
            }
        }
        let mut result = Self {
            changed: false,
            transport: None,
        };
        let mut bounds = Vec::new();
        for coordinate in &changed {
            let before = previous.regions.get(coordinate);
            let after = next.regions.get(coordinate);
            result.changed = true;
            let origin = *coordinate * REGION_EDGE as i32;
            let entire = Bounds {
                min: origin,
                max: origin + IVec3::splat(REGION_EDGE as i32 - 1),
            };
            let mut tight = Bounds {
                min: IVec3::MAX,
                max: IVec3::MIN,
            };
            let mut record = |a: u16, b: u16, extent: Bounds| {
                if a == b {
                    return;
                }
                tight.min = tight.min.min(extent.min);
                tight.max = tight.max.max(extent.max);
            };
            let uniform_before = before.map_or(Some(AIR), |r| r.uniform_material());
            let uniform_after = after.map_or(Some(AIR), |r| r.uniform_material());
            if let (Some(a), Some(b)) = (uniform_before, uniform_after) {
                record(a, b, entire);
            } else {
                for z in 0..REGION_EDGE {
                    for y in 0..REGION_EDGE {
                        for x in 0..REGION_EDGE {
                            let local = UVec3::new(x, y, z);
                            record(
                                before.map_or(AIR, |r| r.material(local)),
                                after.map_or(AIR, |r| r.material(local)),
                                Bounds::point(origin + local.as_ivec3()),
                            );
                        }
                    }
                }
            }
            // Residency changes knowledge of air as well as solids. Unknown
            // holes within the previous occupied bounds affect existing paths;
            // outside air-only bookkeeping does not change their integrand.
            let residency = before.is_none() || after.is_none();
            let transport = before.is_some_and(|r| r.any_solid)
                || after.is_some_and(|r| r.any_solid)
                || (coordinate.cmpge(occupied_min).all() && coordinate.cmple(occupied_max).all());
            if transport && (residency || tight.min.cmple(tight.max).all()) {
                bounds.push(if residency { entire } else { tight });
            }
        }
        if !bounds.is_empty() {
            result.transport = Some(GeometryChange::from_bounds(previous, &bounds, radius));
        }
        result
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Bounds {
    pub min: IVec3,
    pub max: IVec3,
}

impl Bounds {
    pub fn distance_squared(self, other: Self) -> i64 {
        let separation = (self.min.as_i64vec3() - other.max.as_i64vec3())
            .max(other.min.as_i64vec3() - self.max.as_i64vec3())
            .max(I64Vec3::ZERO)
            // Beyond the transport horizon no supported retention curve needs
            // an exact distance. Bound arithmetic for disconnected worlds.
            .min(I64Vec3::splat(4096));
        separation.length_squared()
    }

    pub fn point(voxel: IVec3) -> Self {
        Self {
            min: voxel,
            max: voxel,
        }
    }

    pub fn page(page: IVec3) -> Self {
        let min = page * PAGE_EDGE as i32;
        Self {
            min,
            max: min + IVec3::splat(PAGE_EDGE as i32 - 1),
        }
    }
}

/// Worker-built bounds and their minimum separation from the receiver page.
/// Carrying the distance avoids rescanning bounds when publication folds an
/// old event into the bounded retention summary on the main thread.
pub(super) struct PageChanges {
    pub bounds: Vec<Bounds>,
    pub minimum_distance_squared: i64,
}

impl PageChanges {
    #[cfg(test)]
    pub fn new(page: IVec3, bounds: Vec<Bounds>) -> Self {
        let minimum_distance_squared = bounds
            .iter()
            .map(|b| b.distance_squared(Bounds::page(page)))
            .min()
            .unwrap_or(i64::MAX);
        Self {
            bounds,
            minimum_distance_squared,
        }
    }
}

#[derive(Default)]
pub(super) struct GeometryChange {
    /// Existing receiver pages only. New receivers have no history to retain.
    /// Each bin contains just changes within the configured influence radius.
    pub nearby: HashMap<IVec3, PageChanges>,
    /// Optional raw sunlight evidence for diagnostics; never selects a cap.
    #[cfg(debug_assertions)]
    pub audit_sun: Option<Arc<Vec<super::validation::retention_audit::SunEvidence>>>,
}

impl GeometryChange {
    #[cfg(test)]
    pub fn between(previous: &TransportScene, next: &TransportScene) -> Option<Self> {
        SceneChanges::between(previous, next).transport
    }

    pub fn from_bounds(previous: &TransportScene, changes: &[Bounds], radius: i32) -> Self {
        if changes.is_empty() || previous.regions.is_empty() {
            return Self::default();
        }
        // Choose the smaller broad phase. Local edits should not enumerate the
        // resident world, but a huge brush/disconnected world must not enumerate
        // its empty bounding box either. Eight region lookups establish page
        // occupancy; use a conservative cost allowance for those hash probes.
        // distance_squared deliberately saturates at 4096 per axis, so only
        // use geometric range pruning below that exact-distance horizon.
        let local_cells = changes.iter().fold(0_u64, |total, &change| {
            let (min, max) = expanded_page_range(change, radius);
            let size = max.as_i64vec3() - min.as_i64vec3() + I64Vec3::ONE;
            total.saturating_add(
                size.to_array()
                    .into_iter()
                    .fold(1_u64, |a, b| a.saturating_mul(b.max(0) as u64)),
            )
        });
        if (0..4096).contains(&radius)
            && local_cells.saturating_mul(8)
                < (previous.regions.len() as u64).saturating_mul(changes.len() as u64)
        {
            Self::from_local_bounds(previous, changes, radius)
        } else {
            Self::from_resident_bounds(previous, changes, radius)
        }
    }

    fn from_local_bounds(previous: &TransportScene, changes: &[Bounds], radius: i32) -> Self {
        let mut plan = Self::default();
        let mut occupancy = HashMap::new();
        let radius_squared = i64::from(radius).pow(2);
        for &change in changes {
            let (min, max) = expanded_page_range(change, radius);
            for z in min.z..=max.z {
                for y in min.y..=max.y {
                    for x in min.x..=max.x {
                        let page = IVec3::new(x, y, z);
                        let distance = change.distance_squared(Bounds::page(page));
                        if distance > radius_squared {
                            continue;
                        }
                        // Overlapping changes share this negative/positive cache;
                        // repeated empty-space lookups can dominate sunlight deltas.
                        let occupied = *occupancy.entry(page).or_insert_with(|| {
                            (0..8).any(|i| {
                                let offset = IVec3::new(i & 1, (i >> 1) & 1, (i >> 2) & 1);
                                previous
                                    .regions
                                    .get(&(page * 2 + offset))
                                    .is_some_and(|r| r.any_solid)
                            })
                        });
                        if occupied {
                            let entry = plan.nearby.entry(page).or_insert_with(|| PageChanges {
                                bounds: Vec::new(),
                                minimum_distance_squared: distance,
                            });
                            entry.bounds.push(change);
                            entry.minimum_distance_squared =
                                entry.minimum_distance_squared.min(distance);
                        }
                    }
                }
            }
        }
        plan
    }

    fn from_resident_bounds(previous: &TransportScene, changes: &[Bounds], radius: i32) -> Self {
        let pages: HashSet<_> = previous
            .regions
            .iter()
            .filter(|(_, r)| r.any_solid)
            .map(|(r, _)| r.div_euclid(IVec3::splat(2)))
            .collect();
        let mut plan = Self::default();
        let radius_squared = i64::from(radius).pow(2);
        for page in pages {
            let bounds = Bounds::page(page);
            let mut minimum_distance_squared = i64::MAX;
            let nearby: Vec<_> = changes
                .iter()
                .copied()
                .filter(|change| {
                    let distance = change.distance_squared(bounds);
                    if distance > radius_squared {
                        return false;
                    }
                    minimum_distance_squared = minimum_distance_squared.min(distance);
                    true
                })
                .collect();
            if !nearby.is_empty() {
                plan.nearby.insert(
                    page,
                    PageChanges {
                        bounds: nearby,
                        minimum_distance_squared,
                    },
                );
            }
        }
        plan
    }

    fn history_limit(&self, voxel: IVec3) -> u32 {
        if self
            .nearby
            .get(&super::page_of(voxel))
            .is_some_and(|changes| {
                changes.bounds.iter().any(|bounds| {
                    bounds.distance_squared(Bounds::point(voxel))
                        <= i64::from(RELIGHT_RADIUS).pow(2)
                })
            })
        {
            NEAR_HISTORY
        } else {
            FAR_HISTORY
        }
    }
}

/// Inclusive page span in the supported i32 voxel domain. Widen before padding
/// to avoid overflowing near the coordinate limits; exact sphere tests follow.
fn expanded_page_range(change: Bounds, radius: i32) -> (IVec3, IVec3) {
    let edge = I64Vec3::splat(PAGE_EDGE as i64);
    let padding = I64Vec3::splat(i64::from(radius).abs());
    let lower = I64Vec3::splat(i64::from(i32::MIN)).div_euclid(edge);
    let upper = I64Vec3::splat(i64::from(i32::MAX)).div_euclid(edge);
    let min = (change.min.as_i64vec3() - padding)
        .div_euclid(edge)
        .clamp(lower, upper);
    let max = (change.max.as_i64vec3() + padding)
        .div_euclid(edge)
        .clamp(lower, upper);
    (min.as_ivec3(), max.as_ivec3())
}

#[derive(Default)]
pub(super) struct GeometryHistory(super::revision::ChangeHistory<GeometryChange>);

impl GeometryHistory {
    pub fn record(&mut self, from: u64, to: u64, change: GeometryChange) {
        self.0.record(from, to, change);
    }

    /// Distance determines retained weight, not freshness. Combine every
    /// intervening delta; source-revision gaps or evictions use a conservative cap.
    pub fn history_limit(&self, voxel: IVec3, from: u64, to: u64) -> u32 {
        if from == to {
            return UNTRACKED_HISTORY;
        }
        let mut limit = FAR_HISTORY;
        let complete = self.0.visit_since(from, to, |change| {
            limit = limit.min(change.history_limit(voxel));
        });
        if complete {
            limit
        } else {
            limit.min(UNTRACKED_HISTORY)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel_gi::{
        naadf::{linear_index, volume_len},
        world::PAGE_SIZE,
    };

    fn assert_same_bins(a: &GeometryChange, b: &GeometryChange) {
        assert_eq!(a.nearby.len(), b.nearby.len());
        for (page, expected) in &a.nearby {
            let actual = &b.nearby[page];
            assert_eq!(actual.bounds, expected.bounds, "page {page}");
            assert_eq!(
                actual.minimum_distance_squared, expected.minimum_distance_squared,
                "page {page}"
            );
        }
    }

    #[test]
    fn local_binning_matches_resident_scan_with_gaps_edges_and_overlaps() {
        let mut scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            (-8..8)
                .flat_map(|x| {
                    (-2..2).map(move |z| {
                        (
                            IVec3::new(x, -1, z),
                            vec![if (x + z) % 3 == 0 { AIR } else { 1 }; volume_len(PAGE_SIZE)],
                        )
                    })
                })
                .collect(),
            &[],
        );
        // A partially resident page must still be included if one other region
        // is occupied. Empty/missing pages must never gain retention entries.
        scene.regions.remove(&IVec3::new(0, -2, 0));
        let changes = [
            Bounds::point(IVec3::new(-32, -1, 0)),
            Bounds::point(IVec3::new(0, 0, 0)),
            Bounds::page(IVec3::new(1, 0, 1)),
            Bounds::point(IVec3::new(10000, 0, 0)),
        ];
        for radius in [0, 1, 16, 48, super::super::retention::INFLUENCE_RADIUS] {
            for end in 1..=changes.len() {
                let full = GeometryChange::from_resident_bounds(&scene, &changes[..end], radius);
                assert_same_bins(
                    &full,
                    &GeometryChange::from_local_bounds(&scene, &changes[..end], radius),
                );
                assert_same_bins(
                    &full,
                    &GeometryChange::from_bounds(&scene, &changes[..end], radius),
                );
            }
        }
        for radius in [4096, i32::MAX] {
            assert_same_bins(
                &GeometryChange::from_resident_bounds(&scene, &changes, radius),
                &GeometryChange::from_bounds(&scene, &changes, radius),
            );
        }
        let huge = [Bounds {
            min: IVec3::splat(-1_000_000),
            max: IVec3::splat(1_000_000),
        }];
        assert_same_bins(
            &GeometryChange::from_resident_bounds(&scene, &huge, 192),
            &GeometryChange::from_bounds(&scene, &huge, 192),
        );
        for point in [IVec3::MIN, IVec3::MAX] {
            let (min, max) = expanded_page_range(Bounds::point(point), 100);
            assert!(min.cmple(max).all());
            assert!(min.cmpge(IVec3::splat(i32::MIN / PAGE_EDGE as i32)).all());
            assert!(max.cmple(IVec3::splat(i32::MAX / PAGE_EDGE as i32)).all());
        }
    }

    #[test]
    #[ignore = "paired receiver-page binning study; no transport or renderer"]
    fn spatial_binning_cost_study() {
        for count in [256, 4096] {
            let scene = TransportScene::with_page_delta(
                &TransportScene::default(),
                (0..count)
                    .map(|i| {
                        (
                            IVec3::new(i % 64 - 32, 0, i / 64 - 32),
                            vec![1; volume_len(PAGE_SIZE)],
                        )
                    })
                    .collect(),
                &[],
            );
            for changes in [
                vec![Bounds::point(IVec3::new(0, 0, -960))],
                (0..256)
                    .map(|i| Bounds::point(IVec3::new(i % 16, 0, i / 16 - 960)))
                    .collect(),
            ] {
                let mut times: [Vec<f64>; 2] = default();
                for round in 0..24 {
                    let mut results = Vec::new();
                    for mode in [round % 2, 1 - round % 2] {
                        let start = std::time::Instant::now();
                        let result = if mode == 0 {
                            GeometryChange::from_resident_bounds(
                                &scene,
                                &changes,
                                super::super::retention::INFLUENCE_RADIUS,
                            )
                        } else {
                            GeometryChange::from_bounds(
                                &scene,
                                &changes,
                                super::super::retention::INFLUENCE_RADIUS,
                            )
                        };
                        if round >= 4 {
                            times[mode].push(start.elapsed().as_secs_f64() * 1000.0);
                        }
                        results.push(result);
                    }
                    assert_same_bins(&results[0], &results[1]);
                }
                for t in &mut times {
                    t.sort_by(f64::total_cmp);
                }
                println!(
                    "SPATIAL_BINS pages={count} bounds={} full_ms={:.4} local_ms={:.4} exact=true",
                    changes.len(),
                    times[0][10],
                    times[1][10]
                );
            }
        }
    }

    #[test]
    fn edits_loading_and_removal_use_the_same_bounded_distance_policy() {
        let base = TransportScene::with_page_delta(
            &TransportScene::default(),
            (-3..=3)
                .map(|x| (IVec3::new(x, 0, 0), vec![1; volume_len(PAGE_SIZE)]))
                .collect(),
            &[],
        );
        let mut edited = vec![1; volume_len(PAGE_SIZE)];
        edited[linear_index(UVec3::new(0, 8, 8), PAGE_SIZE)] = 2;
        let next = TransportScene::with_page_delta(&base, vec![(IVec3::ZERO, edited)], &[]);
        let change = GeometryChange::between(&base, &next).unwrap();
        for x in [-48, 0, 48] {
            assert_eq!(change.history_limit(IVec3::new(x, 8, 8)), NEAR_HISTORY);
        }
        for x in [-49, 49] {
            assert_eq!(
                change.history_limit(IVec3::new(x, 8, 8)),
                FAR_HISTORY,
                "measure from the changed voxel, not its containing page or region"
            );
        }
        let loaded = TransportScene::with_page_delta(
            &base,
            vec![(IVec3::new(4, 0, 0), vec![1; volume_len(PAGE_SIZE)])],
            &[],
        );
        for change in [
            GeometryChange::between(&base, &loaded).unwrap(),
            GeometryChange::between(&loaded, &base).unwrap(),
        ] {
            assert_eq!(change.history_limit(IVec3::new(80, 8, 8)), NEAR_HISTORY);
            assert_eq!(change.history_limit(IVec3::new(79, 8, 8)), FAR_HISTORY);
        }
        let mut revision = GeometryHistory::default();
        revision.record(4, 5, change);
        assert_eq!(
            revision.history_limit(IVec3::new(49, 8, 8), 4, 5),
            FAR_HISTORY
        );
        assert_eq!(
            revision.history_limit(IVec3::new(49, 8, 8), 3, 5),
            UNTRACKED_HISTORY
        );
        assert_eq!(
            revision.history_limit(IVec3::new(49, 8, 8), 4, 6),
            UNTRACKED_HISTORY
        );
        assert!(GeometryChange::between(&base, &base).is_none());
    }

    #[test]
    fn overlapping_changes_combine_and_old_or_source_gaps_fall_back() {
        let voxel = IVec3::ZERO;
        let mut near = GeometryChange::default();
        near.nearby.insert(
            IVec3::ZERO,
            PageChanges::new(IVec3::ZERO, vec![Bounds::point(voxel)]),
        );
        let mut history = GeometryHistory::default();
        history.record(10, 11, near);
        history.record(11, 12, GeometryChange::default());
        assert_eq!(
            history.history_limit(voxel, 10, 12),
            NEAR_HISTORY,
            "a later distant update cannot erase the earlier near edit"
        );
        assert_eq!(history.history_limit(voxel, 11, 12), FAR_HISTORY);
        assert_eq!(
            history.history_limit(IVec3::splat(1000), 10, 12),
            FAR_HISTORY,
            "multiple streaming deltas must not discard known distant history"
        );
        assert_eq!(history.history_limit(voxel, 11, 13), UNTRACKED_HISTORY);
        for from in 12..21 {
            history.record(from, from + 1, GeometryChange::default());
        }
        assert_eq!(history.history_limit(voxel, 10, 21), UNTRACKED_HISTORY);
        history.record(30, 31, GeometryChange::default());
        assert_eq!(history.history_limit(voxel, 21, 31), UNTRACKED_HISTORY);
    }
}
