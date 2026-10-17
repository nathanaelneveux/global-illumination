//! GPU lookup from an exposed `(voxel, face)` to its resident measurement slot.
//! Paths terminate at a hit face by reading that face's cached radiance, so
//! the tracer needs the same ownership the worker keeps in its slot map.
//! Open addressing with linear probing and backward-shift deletion; the CPU
//! mirror is authoritative and only changed entries are uploaded.
use super::super::sampling::stream_id;
use super::*;

/// Empty marker: face 7 does not exist.
const EMPTY: [u32; 4] = [0, 0, 0, u32::MAX];
const SLOT_BITS: u32 = 29;
/// GPU probing gives up after this many entries and treats the face as uncached.
pub(super) const MAX_PROBE: usize = 64;

pub(super) struct FaceIndex {
    entries: Vec<[u32; 4]>,
    len: usize,
    dirty: Vec<u32>,
    full_upload: bool,
}

fn home(voxel: IVec3, face: usize, mask: usize) -> usize {
    stream_id(voxel, face as u32) as usize & mask
}

fn entry_key(entry: &[u32; 4]) -> Option<(IVec3, usize)> {
    (entry[3] != u32::MAX).then(|| {
        (
            IVec3::new(entry[0] as i32, entry[1] as i32, entry[2] as i32),
            (entry[3] >> SLOT_BITS) as usize,
        )
    })
}

impl FaceIndex {
    pub(super) fn new() -> Self {
        Self::with_capacity(1024)
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: vec![EMPTY; capacity],
            len: 0,
            dirty: Vec::new(),
            full_upload: true,
        }
    }

    fn mask(&self) -> usize {
        self.entries.len() - 1
    }

    fn find(&self, voxel: IVec3, face: usize) -> Option<usize> {
        let mask = self.mask();
        let mut index = home(voxel, face, mask);
        loop {
            match entry_key(&self.entries[index]) {
                None => return None,
                Some(key) if key == (voxel, face) => return Some(index),
                Some(_) => index = (index + 1) & mask,
            }
        }
    }

    #[cfg(test)]
    pub(super) fn get(&self, voxel: IVec3, face: usize) -> Option<u32> {
        self.find(voxel, face)
            .map(|index| self.entries[index][3] & ((1 << SLOT_BITS) - 1))
    }

    pub(super) fn insert(&mut self, voxel: IVec3, face: usize, slot: u32) {
        assert!(slot < 1 << SLOT_BITS, "resident slot exceeds the face index");
        if let Some(index) = self.find(voxel, face) {
            self.entries[index][3] = (face as u32) << SLOT_BITS | slot;
            self.dirty.push(index as u32);
            return;
        }
        if (self.len + 1) * 2 > self.entries.len() {
            self.grow();
        }
        let mask = self.mask();
        let mut index = home(voxel, face, mask);
        while self.entries[index][3] != u32::MAX {
            index = (index + 1) & mask;
        }
        self.entries[index] = [
            voxel.x as u32,
            voxel.y as u32,
            voxel.z as u32,
            (face as u32) << SLOT_BITS | slot,
        ];
        self.len += 1;
        self.dirty.push(index as u32);
    }

    pub(super) fn remove(&mut self, voxel: IVec3, face: usize) {
        let Some(mut hole) = self.find(voxel, face) else {
            return;
        };
        let mask = self.mask();
        self.len -= 1;
        // Backward-shift deletion keeps every remaining probe chain unbroken
        // without tombstones, so lookups never degrade after streaming churn.
        let mut next = hole;
        loop {
            self.entries[hole] = EMPTY;
            self.dirty.push(hole as u32);
            loop {
                next = (next + 1) & mask;
                let Some((key, key_face)) = entry_key(&self.entries[next]) else {
                    return;
                };
                let ideal = home(key, key_face, mask);
                // The entry may move into the hole only if its ideal position
                // is not cyclically within (hole, next].
                let stays = if hole <= next {
                    hole < ideal && ideal <= next
                } else {
                    hole < ideal || ideal <= next
                };
                if !stays {
                    self.entries[hole] = self.entries[next];
                    self.dirty.push(hole as u32);
                    hole = next;
                    break;
                }
            }
        }
    }

    fn grow(&mut self) {
        let old = std::mem::replace(&mut self.entries, vec![EMPTY; 0]);
        self.entries = vec![EMPTY; old.len() * 2];
        self.len = 0;
        for entry in old {
            if let Some((voxel, face)) = entry_key(&entry) {
                self.insert(voxel, face, entry[3] & ((1 << SLOT_BITS) - 1));
            }
        }
        self.dirty.clear();
        self.full_upload = true;
    }

    /// Header entry 0 holds the probing mask; table entries start at 1.
    pub(super) fn upload(
        &mut self,
        buffer: &mut UploadBuffer,
        device: &RenderDevice,
        queue: &RenderQueue,
    ) -> Result<(), String> {
        let bytes = (self.entries.len() + 1) * 16;
        if buffer.ensure(device, bytes)? || self.full_upload {
            let mut data = Vec::with_capacity(bytes);
            for word in [self.mask() as u32, MAX_PROBE as u32, 0, 0] {
                data.extend_from_slice(&word.to_le_bytes());
            }
            for entry in &self.entries {
                for word in entry {
                    data.extend_from_slice(&word.to_le_bytes());
                }
            }
            queue.write_buffer(&buffer.buffer, 0, &data);
            self.full_upload = false;
            self.dirty.clear();
            return Ok(());
        }
        self.dirty.sort_unstable();
        self.dirty.dedup();
        let mut run_start = 0;
        while run_start < self.dirty.len() {
            let mut run_end = run_start + 1;
            while run_end < self.dirty.len()
                && self.dirty[run_end] - self.dirty[run_end - 1] <= 4
            {
                run_end += 1;
            }
            let first = self.dirty[run_start] as usize;
            let last = self.dirty[run_end - 1] as usize;
            let data: Vec<u8> = self.entries[first..=last]
                .iter()
                .flatten()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            queue.write_buffer(&buffer.buffer, ((first + 1) * 16) as u64, &data);
            run_start = run_end;
        }
        self.dirty.clear();
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn max_probe(&self) -> usize {
        let mask = self.mask();
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                entry_key(entry).map(|(voxel, face)| (index + self.entries.len() - home(voxel, face, mask)) & mask)
            })
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::platform::collections::HashMap;

    #[test]
    fn face_index_matches_a_map_through_insertions_and_removals() {
        let mut table = FaceIndex::with_capacity(16);
        let mut reference = HashMap::new();
        let mut state = 12345u32;
        for step in 0..40_000u32 {
            state = super::super::super::sampling::pcg(state ^ step);
            let voxel = IVec3::new((state % 23) as i32 - 11, (state >> 8) as i32 % 7, -((state >> 16) as i32 % 19));
            let face = (state >> 24) as usize % 6;
            if state & 3 == 0 {
                table.remove(voxel, face);
                reference.remove(&(voxel, face));
            } else {
                table.insert(voxel, face, step);
                reference.insert((voxel, face), step);
            }
            if step % 997 == 0 {
                for (&(voxel, face), &slot) in &reference {
                    assert_eq!(table.get(voxel, face), Some(slot));
                }
                assert_eq!(table.len, reference.len());
            }
        }
        for (&(voxel, face), &slot) in &reference {
            assert_eq!(table.get(voxel, face), Some(slot));
        }
        assert!(table.max_probe() < MAX_PROBE, "probe {}", table.max_probe());
    }
}
