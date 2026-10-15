//! Constant-space cost accounting. Main-world system bodies and asynchronous
//! GPU execution overlap; report them separately, never as a summed frame cost.
use bevy::prelude::*;
use std::time::{Duration, Instant};

use super::VoxelGiStats;

const REPORT_INTERVAL: Duration = Duration::from_millis(250);

#[cfg(debug_assertions)]
pub(super) const MAIN_STAGE_NAMES: [&str; 7] = [
    "ingestion",
    "sources",
    "scene_publication",
    "scene_dispatch",
    "trace_publication",
    "trace_dispatch",
    "display_publication",
];

#[derive(Clone, Copy)]
pub(super) enum MainStage {
    Ingestion,
    Sources,
    ScenePublication,
    SceneDispatch,
    TracePublication,
    TraceDispatch,
    DisplayPublication,
}

#[cfg(debug_assertions)]
pub(super) const DISPLAY_STAGE_NAMES: [&str; 7] = [
    "poll",
    "install",
    "topology",
    "snapshot_start",
    "snapshot_advance",
    "values",
    "ledger",
];

#[derive(Clone, Copy)]
pub(super) enum DisplayStage {
    Poll,
    Install,
    Topology,
    SnapshotStart,
    SnapshotAdvance,
    Values,
    Ledger,
}

#[derive(Resource)]
pub(super) struct GiWorkTimings {
    started: Instant,
    frames: u32,
    main: Duration,
    frame_main: [Duration; 7],
    display_enabled: bool,
    frame_display: [Duration; 7],
    gpu_ms: f64,
    gpu_available: bool,
    missing_gpu_timing: bool,
}

impl Default for GiWorkTimings {
    fn default() -> Self {
        Self {
            started: Instant::now(),
            frames: 0,
            main: Duration::ZERO,
            frame_main: [Duration::ZERO; 7],
            display_enabled: false,
            frame_display: [Duration::ZERO; 7],
            gpu_ms: 0.0,
            gpu_available: false,
            missing_gpu_timing: false,
        }
    }
}

impl GiWorkTimings {
    #[cfg(debug_assertions)]
    pub(super) fn enable_display_detail(&mut self) {
        self.display_enabled = true;
    }

    /// Time only the system body, including early returns, not gaps between
    /// systems or asynchronous work. This does not add any GPU synchronization.
    pub(super) fn main_scope(&mut self, stage: MainStage) -> MainThreadScope<'_> {
        let started = Instant::now();
        MainThreadScope {
            started,
            checkpoint: started,
            timings: self,
            stage,
        }
    }

    /// Completed dispatches contribute once, even if their results are stale.
    /// Queue wait/readback time must never masquerade as a GPU timestamp.
    pub(super) fn record_gpu_batch(&mut self, milliseconds: Option<f32>) {
        if let Some(ms) = milliseconds.filter(|ms| ms.is_finite() && *ms >= 0.0) {
            self.gpu_ms += f64::from(ms);
            self.gpu_available = true;
        } else {
            self.gpu_available = false;
            self.missing_gpu_timing = true;
        }
    }

    fn finish_frame_at(&mut self, now: Instant, stats: &mut VoxelGiStats) {
        stats.last_main_stage_ms = self.frame_main.map(|d| d.as_secs_f32() * 1000.0);
        self.frame_main = [Duration::ZERO; 7];
        stats.last_display_stage_ms = self.frame_display.map(|d| d.as_secs_f32() * 1000.0);
        self.frame_display = [Duration::ZERO; 7];
        self.frames += 1;
        if now.saturating_duration_since(self.started) < REPORT_INTERVAL {
            return;
        }
        stats.main_thread_ms_per_frame =
            Some((self.main.as_secs_f64() * 1000.0 / f64::from(self.frames)) as f32);
        stats.gpu_ms_per_frame = (self.gpu_available && !self.missing_gpu_timing)
            .then_some((self.gpu_ms / f64::from(self.frames)) as f32);
        self.started = now;
        self.frames = 0;
        self.main = Duration::ZERO;
        self.gpu_ms = 0.0;
        self.missing_gpu_timing = false;
    }
}

pub(super) struct MainThreadScope<'a> {
    started: Instant,
    checkpoint: Instant,
    timings: &'a mut GiWorkTimings,
    stage: MainStage,
}

impl MainThreadScope<'_> {
    /// Consecutive checkpoints partition display publication without changing
    /// its scheduling. Detailed clocks are opt-in and allocate no per-frame data.
    pub(super) fn display_checkpoint(&mut self, stage: DisplayStage) {
        if self.timings.display_enabled {
            let now = Instant::now();
            self.timings.frame_display[stage as usize] += now.duration_since(self.checkpoint);
            self.checkpoint = now;
        }
    }

    pub(super) fn record_gpu_batch(&mut self, milliseconds: Option<f32>) {
        self.timings.record_gpu_batch(milliseconds);
    }
}

impl Drop for MainThreadScope<'_> {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed();
        self.timings.main += elapsed;
        self.timings.frame_main[self.stage as usize] += elapsed;
    }
}

pub(super) fn finish_frame(mut timings: ResMut<GiWorkTimings>, mut stats: ResMut<VoxelGiStats>) {
    timings.finish_frame_at(Instant::now(), &mut stats);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn individual_frame_costs_clear_without_waiting_for_average_window() {
        let mut timings = GiWorkTimings::default();
        let now = timings.started;
        let mut stats = VoxelGiStats::default();
        timings.frame_main[MainStage::ScenePublication as usize] = Duration::from_millis(20);
        timings.frame_display[DisplayStage::Topology as usize] = Duration::from_millis(12);
        timings.finish_frame_at(now, &mut stats);
        assert_eq!(stats.last_main_stage_ms[2], 20.0);
        assert_eq!(stats.last_display_stage_ms[2], 12.0);
        assert!(stats.main_thread_ms_per_frame.is_none());
        timings.finish_frame_at(now, &mut stats);
        assert_eq!(stats.last_main_stage_ms, [0.0; 7]);
        assert_eq!(stats.last_display_stage_ms, [0.0; 7]);
    }

    #[test]
    fn costs_are_per_frame_and_idle_windows_clear_old_work() {
        let mut timings = GiWorkTimings::default();
        let now = timings.started;
        let mut stats = VoxelGiStats::default();
        timings.main = Duration::from_millis(6);
        timings.record_gpu_batch(Some(4.0));
        timings.finish_frame_at(now, &mut stats);
        assert_eq!(stats.main_thread_ms_per_frame, None);
        timings.finish_frame_at(now + REPORT_INTERVAL, &mut stats);
        assert_eq!(stats.main_thread_ms_per_frame, Some(3.0));
        assert_eq!(stats.gpu_ms_per_frame, Some(2.0));
        timings.finish_frame_at(now + REPORT_INTERVAL * 2, &mut stats);
        assert_eq!(stats.main_thread_ms_per_frame, Some(0.0));
        assert_eq!(stats.gpu_ms_per_frame, Some(0.0));
    }

    #[test]
    fn unavailable_gpu_timestamps_are_not_reported_as_zero_or_partial_cost() {
        let mut timings = GiWorkTimings::default();
        let now = timings.started;
        let mut stats = VoxelGiStats::default();
        timings.record_gpu_batch(None);
        timings.record_gpu_batch(Some(2.0));
        timings.finish_frame_at(now + REPORT_INTERVAL, &mut stats);
        assert_eq!(stats.gpu_ms_per_frame, None);
        timings.record_gpu_batch(Some(3.0));
        timings.finish_frame_at(now + REPORT_INTERVAL * 2, &mut stats);
        assert_eq!(stats.gpu_ms_per_frame, Some(3.0));
    }
}
