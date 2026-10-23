//! Word-scatter uploads to persistent GPU buffers (slot metadata).

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
    pub(super) fn clear(&mut self) {
        self.records.clear();
        self.bytes.clear();
        self.count = 0;
    }

    pub(super) fn write(&mut self, index: u32, value: u32) {
        self.records.push((index, value));
    }

    pub(super) fn finish(&mut self) {
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
