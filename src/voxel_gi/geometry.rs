//! Shared edit/streaming history policy. Distance controls retained statistical
//! weight, never visibility or whether a receiver must refresh. Plans are built
//! on the scene worker; selection checks only nearby changes in its page bin.
use bevy::math::I64Vec3;
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;

use super::world::{AIR, PAGE_EDGE, REGION_EDGE, TransportScene};
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
        for coordinate in next.regions.keys().chain(
            previous
                .regions
                .keys()
                .filter(|p| !next.regions.contains_key(*p)),
        ) {
            let before = previous.regions.get(coordinate);
            let after = next.regions.get(coordinate);
            if before.zip(after).is_some_and(|(a, b)| Arc::ptr_eq(a, b)) {
                continue;
            }
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

#[derive(Clone, Copy)]
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
}

#[derive(Default)]
pub(super) struct GeometryChange {
    /// Existing receiver pages only. New receivers have no history to retain.
    /// Each bin contains just changes within the local relight radius.
    pub nearby: HashMap<IVec3, Vec<Bounds>>,
}

impl GeometryChange {
    #[cfg(test)]
    pub fn between(previous: &TransportScene, next: &TransportScene) -> Option<Self> {
        SceneChanges::between(previous, next).transport
    }

    pub fn from_bounds(previous: &TransportScene, changes: &[Bounds], radius: i32) -> Self {
        let pages: HashSet<_> = previous
            .regions
            .iter()
            .filter(|(_, r)| r.any_solid)
            .map(|(r, _)| r.div_euclid(IVec3::splat(2)))
            .collect();
        let mut plan = Self::default();
        let radius_squared = i64::from(radius).pow(2);
        for page in pages {
            let min = page * PAGE_EDGE as i32;
            let bounds = Bounds {
                min,
                max: min + IVec3::splat(PAGE_EDGE as i32 - 1),
            };
            let nearby: Vec<_> = changes
                .iter()
                .copied()
                .filter(|change| change.distance_squared(bounds) <= radius_squared)
                .collect();
            if !nearby.is_empty() {
                plan.nearby.insert(page, nearby);
            }
        }
        plan
    }

    fn history_limit(&self, voxel: IVec3) -> u32 {
        if self
            .nearby
            .get(&super::page_of(voxel))
            .is_some_and(|changes| {
                changes.iter().any(|bounds| {
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
        near.nearby.insert(IVec3::ZERO, vec![Bounds::point(voxel)]);
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
