//! Bounded scene lineage containing region keys, never old voxel payloads.
//! Consumers may skip versions; a missing/forked lineage requires a full diff.
use bevy::prelude::IVec3;
use std::{collections::VecDeque, sync::Arc};

const TRANSITIONS: usize = 8;

#[derive(Clone, Default)]
pub(crate) struct SceneStamp(Arc<()>);

impl SceneStamp {
    fn matches(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

struct RegionChange {
    from: SceneStamp,
    keys: Vec<IVec3>,
}

#[derive(Clone, Default)]
pub(super) struct SceneJournal {
    pub stamp: SceneStamp,
    changes: VecDeque<Arc<RegionChange>>,
}

impl SceneJournal {
    pub fn advance(&self, keys: Vec<IVec3>) -> Self {
        let mut changes = self.changes.clone();
        changes.push_back(Arc::new(RegionChange {
            from: self.stamp.clone(),
            keys,
        }));
        if changes.len() > TRANSITIONS {
            changes.pop_front();
        }
        Self {
            stamp: SceneStamp::default(),
            changes,
        }
    }

    pub fn since(&self, stamp: &SceneStamp) -> Option<Vec<IVec3>> {
        // A same-stamp copy may be a manually constructed/reference scene.
        // Runtime callers already skip identical scene Arcs; prefer a safe
        // comparison over trusting mutable test/host copies as new versions.
        if stamp.matches(&self.stamp) {
            return None;
        }
        let start = self
            .changes
            .iter()
            .position(|change| change.from.matches(stamp))?;
        let mut keys: Vec<_> = self
            .changes
            .iter()
            .skip(start)
            .flat_map(|change| change.keys.iter().copied())
            .collect();
        keys.sort_unstable_by_key(|p| (p.z, p.y, p.x));
        keys.dedup();
        Some(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{AIR, PAGE_SIZE, TransportScene, volume_len};
    use super::*;

    fn full_diff(previous: &TransportScene, next: &TransportScene) -> Vec<IVec3> {
        let mut keys: Vec<_> = previous
            .regions
            .keys()
            .chain(next.regions.keys())
            .filter(
                |key| match (previous.regions.get(*key), next.regions.get(*key)) {
                    (Some(a), Some(b)) => !Arc::ptr_eq(a, b),
                    _ => true,
                },
            )
            .copied()
            .collect();
        keys.sort_unstable_by_key(|p| (p.z, p.y, p.x));
        keys.dedup();
        keys
    }

    #[test]
    fn skipped_versions_expired_history_and_forks_match_complete_comparison() {
        let mut scenes = vec![TransportScene::default()];
        for step in 0..20 {
            let page = IVec3::new(step % 4 - 2, 0, -1);
            let (changed, removed) = if step % 5 == 4 {
                (vec![], vec![page])
            } else {
                (
                    vec![(
                        page,
                        vec![if step % 3 == 0 { AIR } else { step as u16 }; volume_len(PAGE_SIZE)],
                    )],
                    vec![],
                )
            };
            scenes.push(TransportScene::with_page_delta(
                scenes.last().unwrap(),
                changed,
                &removed,
            ));
            let next = scenes.last().unwrap();
            for previous in &scenes[..scenes.len() - 1] {
                assert_eq!(next.changed_regions(previous), full_diff(previous, next));
            }
            assert!(next.journal.changes.len() <= TRANSITIONS);
        }
        assert!(
            scenes
                .last()
                .unwrap()
                .region_delta_since(&scenes[0].stamp())
                .is_none()
        );
        assert!(
            scenes
                .last()
                .unwrap()
                .region_delta_since(&scenes[19].stamp())
                .is_some()
        );
        let fork = TransportScene::with_page_delta(
            &scenes[10],
            vec![(IVec3::ONE, vec![7; volume_len(PAGE_SIZE)])],
            &[],
        );
        assert!(fork.region_delta_since(&scenes[11].stamp()).is_none());
        assert_eq!(
            fork.changed_regions(&scenes[11]),
            full_diff(&scenes[11], &fork)
        );
        let no_op = TransportScene::with_page_delta(&fork, vec![], &[]);
        assert!(no_op.changed_regions(&fork).is_empty());
    }

    #[test]
    fn journal_does_not_keep_replaced_geometry_alive() {
        let first = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, vec![1; volume_len(PAGE_SIZE)])],
            &[],
        );
        let retired = Arc::downgrade(&first.regions[&IVec3::ZERO]);
        let next = TransportScene::with_page_delta(
            &first,
            vec![(IVec3::ZERO, vec![2; volume_len(PAGE_SIZE)])],
            &[],
        );
        drop(first);
        assert!(retired.upgrade().is_none());
        assert_eq!(next.journal.changes.len(), 2);
    }
}
