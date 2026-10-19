//! Persistent evidence donor lists, one per resident slot. A list is the
//! breadth-first surface walk around its face, which depends only on surface
//! topology: it is rebuilt when the surface within the walk radius changes, or
//! when a face needs more donors than an incomplete list holds. Any shorter
//! need is a prefix (the walk order does not depend on its limit). The GPU
//! picks each face's donor count from its own current history.
//!
//! Lists live in one GPU arena of power-of-two blocks; a per-slot table holds
//! `(start, len)`. Both are updated with word scatters, never rewritten whole.
use bevy::platform::collections::HashMap;

/// Block sizes 1..=1024 donors, the walk's hard bound.
const CLASSES: usize = 11;

#[derive(Clone, Copy)]
struct List {
    start: u32,
    len: u32,
    class: u8,
    /// The walk ran out of surface before its limit: no longer list exists.
    complete: bool,
}

/// Word-scatter records for one target buffer: `[count, (index, value)...]`.
/// A word written twice in one batch keeps its last value: parallel scatter
/// writes to one word would race, so `finish` drops the earlier records.
#[derive(Default)]
pub(super) struct Scatter {
    pub bytes: Vec<u8>,
    pub count: u32,
    records: Vec<(u32, u32)>,
}

impl Scatter {
    fn clear(&mut self) {
        self.records.clear();
        self.bytes.clear();
        self.count = 0;
    }

    fn write(&mut self, index: u32, value: u32) {
        self.records.push((index, value));
    }

    fn finish(&mut self) {
        // Stable: equal indices keep write order, so the last is the latest.
        self.records.sort_by_key(|&(index, _)| index);
        self.bytes.clear();
        self.bytes.extend_from_slice(&0u32.to_le_bytes());
        let mut count = 0u32;
        for (i, &(index, value)) in self.records.iter().enumerate() {
            if self.records.get(i + 1).is_some_and(|next| next.0 == index) {
                continue;
            }
            self.bytes.extend_from_slice(&index.to_le_bytes());
            self.bytes.extend_from_slice(&value.to_le_bytes());
            count += 1;
        }
        self.bytes[..4].copy_from_slice(&count.to_le_bytes());
        self.count = count;
    }
}

#[derive(Default)]
pub(super) struct DonorCache {
    lists: HashMap<u32, List>,
    free: [Vec<u32>; CLASSES],
    /// Arena words in use (high water), in donor entries.
    pub arena_len: u32,
    pub arena: Scatter,
    pub table: Scatter,
    /// Set when the table must be cleared whole (spatial display re-enabled).
    pub reset_table: bool,
    #[cfg(test)]
    contents: HashMap<u32, Vec<u32>>,
}

fn class_of(len: u32) -> usize {
    (len.max(1).next_power_of_two().trailing_zeros() as usize).min(CLASSES - 1)
}

impl DonorCache {
    pub fn begin_batch(&mut self) {
        self.arena.clear();
        self.table.clear();
    }

    pub fn finish_batch(&mut self) {
        self.arena.finish();
        self.table.finish();
    }

    /// Spatial display was off: every list (and the arena) is gone.
    pub fn clear(&mut self) {
        self.lists.clear();
        self.free = Default::default();
        self.arena_len = 0;
        self.reset_table = true;
        #[cfg(test)]
        self.contents.clear();
    }

    /// Whether `slot` needs a fresh walk to supply `donors` donors.
    pub fn needs(&self, slot: u32, donors: u32) -> bool {
        self.lists
            .get(&slot)
            .is_none_or(|list| !list.complete && list.len < donors)
    }

    /// A face joined the walk graph nearby: the list is still valid, but a
    /// walk that ran out of surface might now go further.
    pub fn may_grow(&mut self, slot: u32) {
        if let Some(list) = self.lists.get_mut(&slot) {
            list.complete = false;
        }
    }

    /// Forget a slot's list: removed face, or surface changed nearby.
    pub fn invalidate(&mut self, slot: u32) {
        if let Some(list) = self.lists.remove(&slot) {
            if list.len > 0 {
                self.free[list.class as usize].push(list.start);
            }
            self.table.write(slot * 2 + 1, 0);
            #[cfg(test)]
            self.contents.remove(&slot);
        }
    }

    pub fn store(&mut self, slot: u32, donors: &[u32], complete: bool) {
        self.invalidate(slot);
        let len = donors.len() as u32;
        let class = class_of(len);
        let start = if len == 0 {
            0
        } else {
            self.free[class].pop().unwrap_or_else(|| {
                let start = self.arena_len;
                self.arena_len += 1 << class;
                start
            })
        };
        for (i, &donor) in donors.iter().enumerate() {
            self.arena.write(start + i as u32, donor);
        }
        self.table.write(slot * 2, start);
        self.table.write(slot * 2 + 1, len);
        self.lists.insert(
            slot,
            List {
                start,
                len,
                class: class as u8,
                complete,
            },
        );
        #[cfg(test)]
        self.contents.insert(slot, donors.to_vec());
    }

    #[cfg(test)]
    pub fn donors(&self, slot: u32) -> &[u32] {
        self.contents.get(&slot).map_or(&[], Vec::as_slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(scatter: &Scatter) -> Vec<(u32, u32)> {
        scatter.bytes[4..]
            .chunks_exact(8)
            .map(|c| {
                (
                    u32::from_le_bytes(c[..4].try_into().unwrap()),
                    u32::from_le_bytes(c[4..].try_into().unwrap()),
                )
            })
            .collect()
    }

    #[test]
    fn lists_reuse_blocks_and_prefixes_serve_smaller_needs() {
        let mut cache = DonorCache::default();
        cache.begin_batch();
        cache.store(3, &[7, 8, 9], false);
        cache.finish_batch();
        assert_eq!(cache.arena.count, 3);
        let rewritten = {
            let mut scatter = Scatter::default();
            scatter.clear();
            scatter.write(4, 1);
            scatter.write(2, 5);
            scatter.write(4, 0);
            scatter.finish();
            words(&scatter)
        };
        assert_eq!(rewritten, vec![(2, 5), (4, 0)], "last write wins, once");
        assert_eq!(&cache.arena.bytes[..4], &3u32.to_le_bytes());
        assert_eq!(words(&cache.table), vec![(6, 0), (7, 3)]);
        assert_eq!(cache.arena_len, 4, "three donors take a four-entry block");
        // A shorter need is a prefix; a longer one regathers an incomplete list.
        assert!(!cache.needs(3, 2));
        assert!(!cache.needs(3, 3));
        assert!(cache.needs(3, 4));
        cache.begin_batch();
        cache.store(3, &[7, 8, 9], true);
        cache.finish_batch();
        assert!(!cache.needs(3, 1000), "a complete list cannot grow");
        // Invalidation frees the block for the next list of that class.
        cache.invalidate(3);
        assert!(cache.needs(3, 1));
        cache.store(5, &[1, 2, 3, 4], false);
        assert_eq!(cache.arena_len, 4, "the freed block was reused");
        assert_eq!(cache.donors(5), &[1, 2, 3, 4]);
        cache.finish_batch();
        let table = words(&cache.table);
        assert!(table.contains(&(7, 0)) && table.contains(&(11, 4)));
        cache.begin_batch();
        // An empty walk records a complete, zero-length list without arena use.
        cache.store(6, &[], true);
        assert!(!cache.needs(6, 1));
        assert_eq!(cache.arena_len, 4);
    }
}
