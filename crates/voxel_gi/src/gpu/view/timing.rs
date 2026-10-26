//! Bounded asynchronous timing receipts. Only timestamps, queue counters and
//! corner-sunlight flips cross back to the CPU per view dispatch, never lighting.
use super::*;
use std::sync::atomic::{AtomicU8, Ordering};

/// Use the same bounded feedback controller as near transport. Dividing a
/// small wave's whole-pass cost by its path count mistakes dispatch overhead
/// for marginal path cost, eventually underfilling or starving every wave.
fn next_wave(current: u32, attempted: u32, milliseconds: f32, target_ms: f32, limit: usize) -> u32 {
    crate::scheduling::next_budget(
        current as usize, attempted as usize, milliseconds, target_ms, limit,
    ).div_ceil(64) as u32 * 64
}

#[cfg(test)]
mod tests {
    use super::next_wave;
    #[test]
    fn costly_minimum_wave_cannot_starve_forever() {
        let mut size = next_wave(128, 128, 8.0, 1.0, 8192);
        assert_eq!(size, 64);
        // A 0.7ms fixed dispatch cost must not trap a 64-path wave at 64.
        for _ in 0..16 {
            size = next_wave(size, size, 0.7, 1.0, 8192);
            assert!((64..=8192).contains(&size));
            assert_eq!(size % 64, 0);
        }
        assert!(size >= 2048);
        assert_eq!(next_wave(4096, 32, 0.5, 1.0, 8192), 4096, "small views do not shrink capacity");
    }

    #[test]
    fn flips_past_the_record_limit_arrive_as_one_bounding_box() {
        use super::super::runtime::{MAX_SUN_FLIPS, SUN_FLIP_BIAS, SUN_FLIP_BYTES};
        use bevy::prelude::IVec3;
        let mut region = vec![0u8; SUN_FLIP_BYTES as usize];
        let mut put = |at: usize, word: u32| region[at..at + 4].copy_from_slice(&word.to_le_bytes());
        put(0, MAX_SUN_FLIPS as u32 + 5);
        put(16, -3i32 as u32);
        put(20, 4);
        put(24, 5);
        put(28, 0.5f32.to_bits());
        let overflow = 16 + MAX_SUN_FLIPS as usize * 16;
        let bias = SUN_FLIP_BIAS as i32;
        for (i, word) in [bias + 10, bias - 2, bias, bias + 7, bias + 9, bias + 1, 128].into_iter().enumerate() {
            put(overflow + i * 4, word as u32);
        }
        let mut flips = Vec::new();
        super::decode_sun_flips(&region, &mut flips);
        assert_eq!(flips.len(), MAX_SUN_FLIPS as usize + 1);
        assert_eq!(flips[0], (IVec3::new(-3, 4, 5), IVec3::new(-3, 4, 5), 0.5));
        assert_eq!(*flips.last().unwrap(), (IVec3::new(-10, 2, 0), IVec3::new(7, 9, 1), 2.0));
    }

    #[test]
    fn view_first_can_spend_its_share_beyond_one_visit_per_owner() {
        let mut size = 8192;
        for _ in 0..16 { size = next_wave(size, size, 1.0, 3.0, 65536); }
        assert_eq!(size, 65536);
        assert!(next_wave(size, size, 9.0, 3.0, 65536) < size);
    }
}

/// A receipt's flip region (`report_sun_flips`): a count, records, then the
/// overflow box, as (min, max, flux) boxes.
fn decode_sun_flips(region: &[u8], flips: &mut Vec<(IVec3, IVec3, f32)>) {
    let word = |at: usize| u32::from_le_bytes(region[at..at + 4].try_into().unwrap());
    let limit = super::runtime::MAX_SUN_FLIPS;
    let reported = u64::from(word(0));
    for slot in 0..reported.min(limit) as usize {
        let at = 16 + slot * 16;
        let voxel = IVec3::new(word(at) as i32, word(at + 4) as i32, word(at + 8) as i32);
        flips.push((voxel, voxel, f32::from_bits(word(at + 12))));
    }
    if reported > limit {
        let overflow = 16 + limit as usize * 16;
        let at = |i: usize| i64::from(word(overflow + i * 4));
        let bias = super::runtime::SUN_FLIP_BIAS;
        let min = IVec3::new((bias - at(0)) as i32, (bias - at(1)) as i32, (bias - at(2)) as i32);
        let max = IVec3::new((at(3) - bias) as i32, (at(4) - bias) as i32, (at(5) - bias) as i32);
        flips.push((min, max, at(6) as f32 / 64.0));
    }
}

pub(super) struct Ticket {
    buffer: Buffer,
    ready: Arc<AtomicU8>,
    reserved: f32,
    stages: u32,
}

pub(super) struct Budget {
    pub queries: Option<wgpu::QuerySet>,
    resolve: Buffer,
    free: Vec<Buffer>,
    pending: Vec<Ticket>,
    allowance: f32,
    wave_size: u32,
    wave_cost: f32,
    pub presentation_ms: f32,
    /// Last receipt's per-stage GPU ms, in dispatch-stage order: discovery,
    /// index, reproject, retained changes, cache load, trace, reconstruct,
    /// cache store, publish, pool prepare, per-pixel tier.
    pub stage_ms: [f32; STAGES],
    pub discovery_ms: f32,
    pub discovery_sum_ms: f64,
    pub discovery_count: u32,
    pub pool_prepare_ms: f32,
    pub pool_scan_ms: f32,
    pub pool_free: u32,
    pub pool_claimed: u32,
    pub pool_exhausted: u32,
    pub pool_retained: u32,
    pub trace_ms: f32,
    pub cache_ms: f32,
    pub cache_loads: u64,
    pub cache_promotions: u64,
    pub active: u32,
    pub completed: u32,
    pub known: u32,
    pub history: f32,
    pub max_history: u32,
    pub uncovered: u32,
    pub displayed: u32,
    pub paused: u32,
    /// Far tier: GPU ms, owned and traced cells, cells with reprojected
    /// history, mean history of owned cells.
    pub pixel_ms: f32,
    pub pixel_owned: u32,
    pub pixel_traced: u32,
    pub pixel_reprojected: u32,
    pub pixel_history: f32,
    /// Handoffs: far cells seeded from receivers, receivers from cells.
    pub pixel_seeded: u32,
    pub receivers_seeded: u32,
    /// Received corner-sunlight flips as (min, max, flux) boxes, taken by the
    /// main world: one voxel each, plus one box bounding any overflow.
    pub sun_flips: Vec<(IVec3, IVec3, f32)>,
    period: f64,
}

/// Dispatch stages with a timestamp pair each.
const STAGES: usize = 11;
const STAMP_BYTES: usize = STAGES * 16;
/// Timestamps, queue counters, pool words and per-pixel counters, then the
/// flip region.
const RECEIPT_BYTES: u64 = STAMP_BYTES as u64 + 48 + 16 + 32;
const TICKET_BYTES: u64 = RECEIPT_BYTES + super::runtime::SUN_FLIP_BYTES;

impl Budget {
    pub fn new(device: &RenderDevice, queue: &RenderQueue) -> Self {
        let queries = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| {
                device
                    .wgpu_device()
                    .create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("Far GI GPU cost"),
                        ty: wgpu::QueryType::Timestamp,
                        count: STAGES as u32 * 2,
                    })
            });
        let resolve = device.create_buffer(&BufferDescriptor {
            label: Some("Far GI timing resolve"),
            size: 256,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let free = (0..3)
            .map(|_| {
                device.create_buffer(&BufferDescriptor {
                    label: Some("Far GI asynchronous timing receipt"),
                    size: TICKET_BYTES,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Self {
            queries,
            resolve,
            free,
            pending: Vec::new(),
            allowance: 0.0,
            wave_size: 1024,
            wave_cost: 1.0,
            presentation_ms: 0.0,
            stage_ms: [0.0; STAGES],
            discovery_ms: 0.0,
            discovery_sum_ms: 0.0,
            discovery_count: 0,
            pool_prepare_ms: 0.0,
            pool_scan_ms: 0.0,
            pool_free: 0,
            pool_claimed: 0,
            pool_exhausted: 0,
            pool_retained: 0,
            trace_ms: 0.0,
            cache_ms: 0.0,
            cache_loads: 0,
            cache_promotions: 0,
            active: 0,
            completed: 0,
            known: 0,
            history: 0.0,
            max_history: 0,
            uncovered: 0,
            displayed: 0,
            paused: 0,
            pixel_ms: 0.0,
            pixel_owned: 0,
            pixel_traced: 0,
            pixel_reprojected: 0,
            pixel_history: 0.0,
            pixel_seeded: 0,
            receivers_seeded: 0,
            sun_flips: Vec::new(),
            period: f64::from(queue.get_timestamp_period()),
        }
    }

    pub fn begin_frame(&mut self, device: &RenderDevice, elapsed: f32, share: f32, view_first: bool) {
        assert!(
            self.queries.is_some(),
            "Far view GI requires TIMESTAMP_QUERY to enforce its transport budget"
        );
        // Poll means service completed callbacks, never wait for GPU progress.
        device
            .poll(wgpu::PollType::Poll)
            .expect("far timing device poll failed");
        let mut index = 0;
        while index < self.pending.len() {
            match self.pending[index].ready.load(Ordering::Acquire) {
                0 => {
                    index += 1;
                    continue;
                }
                1 => {}
                _ => panic!("far GPU timing mapping failed"),
            }
            let ticket = self.pending.swap_remove(index);
            let bytes = ticket.buffer.slice(..TICKET_BYTES).get_mapped_range();
            let stamp = |i: usize| u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
            let costs: [f32; STAGES] = std::array::from_fn(|stage| {
                if ticket.stages & (1 << stage) == 0 {
                    return 0.0;
                }
                (stamp(stage * 2 + 1).saturating_sub(stamp(stage * 2)) as f64 * self.period / 1e6)
                    as f32
            });
            let total: f32 = costs.iter().sum();
            let trace = costs[5];
            let counter = |i: usize| {
                u32::from_le_bytes(bytes[STAMP_BYTES + i * 4..STAMP_BYTES + 4 + i * 4].try_into().unwrap())
            };
            let active = counter(0);
            assert_eq!(
                counter(1),
                0,
                "far receiver hash overflow; no keys may be silently dropped"
            );
            self.active = active;
            self.completed = counter(2);
            self.known = counter(3);
            self.history = counter(4) as f32 / active.max(1) as f32;
            self.max_history = counter(5);
            self.uncovered = counter(6);
            self.displayed = counter(7);
            self.paused = counter(8);
            self.trace_ms = trace;
            self.pixel_ms = costs[10];
            self.presentation_ms = total - trace - costs[10];
            self.pixel_owned = counter(16);
            self.pixel_traced = counter(17);
            self.pixel_history = counter(18) as f32 / counter(16).max(1) as f32;
            self.pixel_reprojected = counter(19);
            self.pixel_seeded = counter(20);
            self.receivers_seeded = counter(21);
            self.stage_ms = costs;
            self.discovery_ms = costs[0] + costs[9];
            self.pool_prepare_ms = costs[9];
            self.pool_scan_ms = costs[0];
            self.pool_free = counter(12);
            self.pool_claimed = counter(13);
            self.pool_exhausted = counter(14);
            self.pool_retained = counter(15);
            self.discovery_sum_ms += f64::from(self.discovery_ms);
            self.discovery_count += 1;
            self.cache_ms = costs[4] + costs[7];
            self.cache_loads += u64::from(counter(10));
            self.cache_promotions += u64::from(counter(11));
            let attempted = counter(9);
            if attempted > 0 {
                // View-first owns the primary lighting budget, not just a
                // distant supplement. Keep the wave bounded while allowing
                // several sequential visits to a small visible working set.
                let target = if view_first { (elapsed * 1000.0 * share).clamp(0.25, 4.0) } else { 1.0 };
                let limit = if view_first { 65536 } else { 8192 };
                self.wave_size = next_wave(self.wave_size, attempted, trace, target, limit);
                self.wave_cost = trace.max(0.01);
            }
            self.allowance += ticket.reserved - trace;
            decode_sun_flips(&bytes[RECEIPT_BYTES as usize..], &mut self.sun_flips);
            drop(bytes);
            ticket.buffer.unmap();
            self.free.push(ticket.buffer);
        }
        self.allowance = (self.allowance + elapsed * 1000.0 * share).min(2.0);
    }

    pub fn reserve(&mut self) -> (u32, Option<Ticket>) {
        let Some(buffer) = self.free.pop() else {
            return (0, None);
        };
        // Bank/repay milliseconds, not fractional workgroups. One bounded
        // wave may overdraw credit; the receipt charges its actual cost.
        let work = if self.allowance > 0.0 { self.wave_size } else { 0 };
        let reserved = if work > 0 { self.wave_cost } else { 0.0 };
        self.allowance -= reserved;
        (
            work,
            Some(Ticket {
                buffer,
                ready: Arc::new(AtomicU8::new(0)),
                reserved,
                stages: 0,
            }),
        )
    }

    pub fn finish(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        counters: &Buffer,
        sun_flips: u64,
        pool: &Buffer,
        pixel_stats: &Buffer,
        mut ticket: Ticket,
        stages: u32,
    ) {
        ticket.stages = stages;
        let stamps = STAMP_BYTES as u64;
        encoder.resolve_query_set(self.queries.as_ref().unwrap(), 0..STAGES as u32 * 2, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &ticket.buffer, 0, stamps);
        encoder.copy_buffer_to_buffer(counters, 0, &ticket.buffer, stamps, 48);
        encoder.copy_buffer_to_buffer(pool, 0, &ticket.buffer, stamps + 48, 16);
        encoder.copy_buffer_to_buffer(pixel_stats, 0, &ticket.buffer, stamps + 64, 32);
        encoder.copy_buffer_to_buffer(
            counters, sun_flips, &ticket.buffer, RECEIPT_BYTES, super::runtime::SUN_FLIP_BYTES,
        );
        // Copied: the flips are delivered, so the region starts over.
        encoder.clear_buffer(counters, sun_flips, Some(super::runtime::SUN_FLIP_BYTES));
        let ready = ticket.ready.clone();
        encoder.map_buffer_on_submit(&ticket.buffer, wgpu::MapMode::Read, ..TICKET_BYTES, move |result| {
            ready.store(if result.is_ok() { 1 } else { 2 }, Ordering::Release)
        });
        self.pending.push(ticket);
    }
}
