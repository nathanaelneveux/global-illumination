//! Bounded asynchronous timing receipts. Only 160 bytes per view dispatch cross
//! back to the CPU: timestamps and queue counters, never lighting.
use super::*;
use std::sync::atomic::{AtomicU8, Ordering};

pub(super) struct Ticket {
    buffer: Buffer,
    ready: Arc<AtomicU8>,
    work: u32,
    reserved: f32,
    stages: u32,
}

pub(super) struct Budget {
    pub queries: Option<wgpu::QuerySet>,
    resolve: Buffer,
    free: Vec<Buffer>,
    pending: Vec<Ticket>,
    allowance: f32,
    cost_per_path: f32,
    pub presentation_ms: f32,
    pub trace_ms: f32,
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
                        count: 14,
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
                    size: 160,
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
            cost_per_path: 0.001,
            presentation_ms: 0.0,
            trace_ms: 0.0,
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

    pub fn begin_frame(&mut self, device: &RenderDevice, elapsed: f32, share: f32) {
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
            let bytes = ticket.buffer.slice(..160).get_mapped_range();
            let stamp = |i: usize| u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
            let costs: [f32; 7] = std::array::from_fn(|stage| {
                if ticket.stages & (1 << stage) == 0 {
                    return 0.0;
                }
                (stamp(stage * 2 + 1).saturating_sub(stamp(stage * 2)) as f64 * self.period / 1e6)
                    as f32
            });
            let total: f32 = costs.iter().sum();
            let trace = costs[4];
            let counter =
                |i: usize| u32::from_le_bytes(bytes[112 + i * 4..116 + i * 4].try_into().unwrap());
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
            let attempted = ticket.work.min(active);
            if attempted > 0 {
                let measured = (trace / attempted as f32).max(1e-5);
                self.cost_per_path = self.cost_per_path * 0.75 + measured * 0.25;
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
        let paths = (self.allowance.clamp(0.0, 1.0) / self.cost_per_path) as u32;
        let work = (paths / 64 * 64).min(8192);
        let reserved = work as f32 * self.cost_per_path;
        self.allowance -= reserved;
        (
            work,
            Some(Ticket {
                buffer,
                ready: Arc::new(AtomicU8::new(0)),
                work,
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
        encoder.resolve_query_set(self.queries.as_ref().unwrap(), 0..14, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &ticket.buffer, 0, 112);
        encoder.copy_buffer_to_buffer(counters, 0, &ticket.buffer, 112, 48);
        let ready = ticket.ready.clone();
        encoder.map_buffer_on_submit(&ticket.buffer, wgpu::MapMode::Read, ..160, move |result| {
            ready.store(if result.is_ok() { 1 } else { 2 }, Ordering::Release)
        });
        self.pending.push(ticket);
    }
}
