//! Bounded asynchronous timing receipts. Only 192 bytes per view dispatch cross
//! back to the CPU: timestamps and queue counters, never lighting.
use super::*;
use std::sync::atomic::{AtomicU8, Ordering};

/// Use the same bounded feedback controller as near transport. Dividing a
/// small wave's whole-pass cost by its path count mistakes dispatch overhead
/// for marginal path cost, eventually underfilling or starving every wave.
fn next_wave(current: u32, attempted: u32, milliseconds: f32, target_ms: f32, limit: usize) -> u32 {
    crate::voxel_gi::scheduling::next_budget(
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
    fn view_first_can_spend_its_share_beyond_one_visit_per_owner() {
        let mut size = 8192;
        for _ in 0..16 { size = next_wave(size, size, 1.0, 3.0, 65536); }
        assert_eq!(size, 65536);
        assert!(next_wave(size, size, 9.0, 3.0, 65536) < size);
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
    pub discovery_ms: f32,
    pub discovery_sum_ms: f64,
    pub discovery_count: u32,
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
    period: f64,
}

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
                        count: 18,
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
                    size: 192,
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
            discovery_ms: 0.0,
            discovery_sum_ms: 0.0,
            discovery_count: 0,
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
            let bytes = ticket.buffer.slice(..192).get_mapped_range();
            let stamp = |i: usize| u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
            let costs: [f32; 9] = std::array::from_fn(|stage| {
                if ticket.stages & (1 << stage) == 0 {
                    return 0.0;
                }
                (stamp(stage * 2 + 1).saturating_sub(stamp(stage * 2)) as f64 * self.period / 1e6)
                    as f32
            });
            let total: f32 = costs.iter().sum();
            let trace = costs[5];
            let counter =
                |i: usize| u32::from_le_bytes(bytes[144 + i * 4..148 + i * 4].try_into().unwrap());
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
            self.presentation_ms = total - trace;
            self.discovery_ms = costs[0];
            self.discovery_sum_ms += f64::from(costs[0]);
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
        mut ticket: Ticket,
        stages: u32,
    ) {
        ticket.stages = stages;
        encoder.resolve_query_set(self.queries.as_ref().unwrap(), 0..18, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &ticket.buffer, 0, 144);
        encoder.copy_buffer_to_buffer(counters, 0, &ticket.buffer, 144, 48);
        let ready = ticket.ready.clone();
        encoder.map_buffer_on_submit(&ticket.buffer, wgpu::MapMode::Read, ..192, move |result| {
            ready.store(if result.is_ok() { 1 } else { 2 }, Ordering::Release)
        });
        self.pending.push(ticket);
    }
}
