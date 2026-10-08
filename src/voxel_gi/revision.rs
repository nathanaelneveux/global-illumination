//! Bounded change journals. Consumers that miss retained transitions must use
//! their conservative policy; the newest delta alone never establishes freshness.
use std::collections::VecDeque;

const MAX_REVISIONS: usize = 8;

struct Revision<T> {
    from: u64,
    to: u64,
    change: T,
}

pub(super) struct ChangeHistory<T> {
    revisions: VecDeque<Revision<T>>,
}

impl<T> Default for ChangeHistory<T> {
    fn default() -> Self {
        Self {
            revisions: VecDeque::new(),
        }
    }
}

impl<T> ChangeHistory<T> {
    pub fn record(&mut self, from: u64, to: u64, change: T) {
        if self.revisions.back().is_some_and(|last| last.to != from) {
            self.revisions.clear();
        }
        if self.revisions.len() == MAX_REVISIONS {
            self.revisions.pop_front();
        }
        self.revisions.push_back(Revision { from, to, change });
    }

    /// Visit newest to oldest. False means a gap or eviction prevented a full
    /// chain, even though some deltas may have been visited. No scene is retained.
    pub fn visit_since(&self, from: u64, to: u64, mut visit: impl FnMut(&T)) -> bool {
        if from == to {
            return true;
        }
        let mut expected = to;
        for revision in self.revisions.iter().rev() {
            if revision.to != expected {
                break;
            }
            visit(&revision.change);
            if revision.from == from {
                return true;
            }
            expected = revision.from;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_gaps_eviction_and_wraparound_without_retaining_unbounded_history() {
        let mut history = ChangeHistory::default();
        for from in 0..10 {
            history.record(from, from + 1, from);
        }
        assert_eq!(history.revisions.len(), MAX_REVISIONS);
        let mut visited = Vec::new();
        assert!(history.visit_since(2, 10, |v| visited.push(*v)));
        assert_eq!(visited, (2..10).rev().collect::<Vec<_>>());
        assert!(!history.visit_since(1, 10, |_| {}));
        assert!(history.visit_since(10, 10, |_| panic!("already current")));
        history.record(11, 12, 11);
        assert_eq!(history.revisions.len(), 1);
        assert!(!history.visit_since(10, 12, |_| {}));
        assert!(history.visit_since(11, 12, |_| {}));
        history.record(u64::MAX, 0, 0);
        assert!(history.visit_since(u64::MAX, 0, |_| {}));
    }
}
