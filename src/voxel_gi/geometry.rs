//! Shared edit/streaming history policy. Distance controls retained statistical
//! weight, never visibility or whether a receiver must refresh. Plans are built
//! on the scene worker; selection checks only nearby changes in its page bin.
use bevy::math::I64Vec3;
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use std::collections::VecDeque;

use super::world::{PAGE_EDGE, REGION_EDGE, TransportScene};

pub(super) const RELIGHT_RADIUS: i32 = 48;
pub(super) const NEAR_HISTORY: u32 = 16;
const FAR_HISTORY: u32 = 256;
pub(super) const UNTRACKED_HISTORY: u32 = 64;
const MAX_REVISIONS: usize = 8;

#[derive(Clone, Copy)]
struct Bounds {
    min: IVec3,
    max: IVec3,
}

impl Bounds {
    fn distance_squared(self, other: Self) -> i64 {
        let separation = (self.min.as_i64vec3() - other.max.as_i64vec3())
            .max(other.min.as_i64vec3() - self.max.as_i64vec3())
            .max(I64Vec3::ZERO)
            // Only the radius test matters; cap distant axes before squaring
            // so disconnected islands cannot overflow the distance arithmetic.
            .min(I64Vec3::splat(i64::from(RELIGHT_RADIUS) + 1));
        separation.length_squared()
    }

    fn point(voxel: IVec3) -> Self {
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
    nearby: HashMap<IVec3, Vec<Bounds>>,
}

impl GeometryChange {
    pub fn between(previous: &TransportScene, next: &TransportScene) -> Option<Self> {
        let regions = next.transport_changed_regions(previous);
        if regions.is_empty() {
            return None;
        }
        let changes: Vec<_> = regions
            .into_iter()
            .map(|coordinate| {
                let origin = coordinate * REGION_EDGE as i32;
                let entire = Bounds {
                    min: origin,
                    max: origin + IVec3::splat(REGION_EDGE as i32 - 1),
                };
                let (Some(before), Some(after)) = (
                    previous.regions.get(&coordinate),
                    next.regions.get(&coordinate),
                ) else {
                    // Residency also changes knowledge of air, not just solids.
                    return entire;
                };
                // Tight bounds for edits inside known regions: a single block edit
                // keeps the same 48-voxel distance as the explicit urgent path.
                let mut bounds = Bounds {
                    min: IVec3::MAX,
                    max: IVec3::MIN,
                };
                for z in 0..REGION_EDGE {
                    for y in 0..REGION_EDGE {
                        for x in 0..REGION_EDGE {
                            let local = UVec3::new(x, y, z);
                            if before.material(local) != after.material(local) {
                                let voxel = origin + local.as_ivec3();
                                bounds.min = bounds.min.min(voxel);
                                bounds.max = bounds.max.max(voxel);
                            }
                        }
                    }
                }
                bounds
            })
            .collect();
        let pages: HashSet<_> = previous
            .regions
            .iter()
            .filter(|(_, r)| r.any_solid)
            .map(|(r, _)| r.div_euclid(IVec3::splat(2)))
            .collect();
        let mut plan = Self::default();
        let radius_squared = i64::from(RELIGHT_RADIUS).pow(2);
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
        Some(plan)
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

struct GeometryRevision {
    from: u64,
    to: u64,
    change: GeometryChange,
}

#[derive(Default)]
pub(super) struct GeometryHistory {
    revisions: VecDeque<GeometryRevision>,
}

impl GeometryHistory {
    pub fn record(&mut self, from: u64, to: u64, change: GeometryChange) {
        if self.revisions.back().is_some_and(|last| last.to != from) {
            // An interleaved source revision already requires conservative
            // source handling. Older spatial metadata is no longer useful.
            self.revisions.clear();
        }
        if self.revisions.len() == MAX_REVISIONS {
            self.revisions.pop_front();
        }
        self.revisions
            .push_back(GeometryRevision { from, to, change });
    }

    /// Combine every intervening change, never only the latest one. Streaming
    /// can publish several deltas before a receiver runs. Keep a bounded ring
    /// of spatial metadata, not scene snapshots or an unbounded edit journal;
    /// missing transitions conservatively fall back to the previous cap.
    pub fn history_limit(&self, voxel: IVec3, from: u64, to: u64) -> u32 {
        if from == to {
            return UNTRACKED_HISTORY;
        }
        let mut expected = to;
        let mut limit = FAR_HISTORY;
        for revision in self.revisions.iter().rev() {
            if revision.to != expected {
                break;
            }
            limit = limit.min(revision.change.history_limit(voxel));
            if limit == NEAR_HISTORY || revision.from == from {
                return limit;
            }
            expected = revision.from;
        }
        limit.min(UNTRACKED_HISTORY)
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
        assert_eq!(history.revisions.len(), MAX_REVISIONS);
        assert_eq!(history.history_limit(voxel, 10, 21), UNTRACKED_HISTORY);
        history.record(30, 31, GeometryChange::default());
        assert_eq!(history.revisions.len(), 1);
        assert_eq!(history.history_limit(voxel, 21, 31), UNTRACKED_HISTORY);
    }
}
