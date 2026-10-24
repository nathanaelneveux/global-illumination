//! Frame timing and transport work observations from public GI stats.
//! Deliberately separate from visual ground truth: captures judge lighting.
use voxel_gi::{GI_MAIN_STAGE_NAMES, VoxelGiStats};
use bevy::prelude::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub(super) fn distribution(values: &[f32]) -> Value {
    if values.is_empty() {
        return json!({"count": 0});
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let at = |q: f32| sorted[((sorted.len() - 1) as f32 * q).ceil() as usize];
    json!({"count": sorted.len(), "p50": at(0.5), "p95": at(0.95), "p99": at(0.99), "max": sorted.last()})
}

#[derive(Default)]
pub(super) struct Phase {
    pub frames_ms: Vec<f32>,
    pub excluded_capture_frames: usize,
    selection_ms: Vec<f32>,
    merge_ms: Vec<f32>,
    worker_ms: Vec<f32>,
    gpu_ms: Vec<f32>,
}

#[derive(Default)]
pub(super) struct Metrics {
    pub phases: BTreeMap<&'static str, Phase>,
    work_samples: Vec<Value>,
    last_work_sample: f32,
    last_batch: u64,
    frame_timeline: Vec<Value>,
    previous_gi: [f32; 6],
    frame_timing_enabled: Option<bool>,
}

impl Metrics {
    pub fn record_frame(
        &mut self,
        elapsed: f32,
        phase: &'static str,
        frame_ms: f32,
        capture_excluded: bool,
        stats: &VoxelGiStats,
        previous_host_ms: Option<[f32; 7]>,
    ) {
        if !*self
            .frame_timing_enabled
            .get_or_insert_with(|| std::env::var_os("VOXEL_GI_VALIDATE_FRAME_TIMINGS").is_some())
        {
            return;
        }
        // Associate preceding Main observations, not this Update's work.
        // Time<Real> can use renderer-sent instants under pipelining: these
        // arrays are not an exact partition of its interval. In particular,
        // subtracting them cannot identify a BVW or GPU-only remainder.
        self.frame_timeline
            .push(json!({"at_s":elapsed,"phase":phase,
            "interval_ms":frame_ms,"capture_excluded":capture_excluded,
            "previous_gi_body_ms":self.previous_gi,
            "previous_host_schedule_ms":previous_host_ms}));
        self.previous_gi = stats.last_main_stage_ms;
    }
    /// O(1) observations of publication and lighting progress. A small early
    /// page set can be fully covered while most terrain is still unpublished;
    /// keep its denominator, not a misleading first-100% time.
    pub fn record_work(&mut self, elapsed: f32, phase: &'static str, stats: &VoxelGiStats) {
        if stats.transport_batches != self.last_batch {
            self.last_batch = stats.transport_batches;
            let phase = self.phases.entry(phase).or_default();
            phase.selection_ms.push(stats.last_selection_ms);
            phase.merge_ms.push(stats.last_merge_ms);
            phase.worker_ms.push(stats.last_worker_ms);
            phase.gpu_ms.push(stats.last_trace_batch_ms);
        }
        if elapsed - self.last_work_sample < 0.25 {
            return;
        }
        self.last_work_sample = elapsed;
        self.work_samples.push(json!({
            "elapsed_s": elapsed, "phase": phase,
            "loaded_pages": stats.loaded_pages, "published_probes": stats.surface_probes,
            "coverage": stats.interactive_coverage, "bootstrap": stats.bootstrap_coverage,
            "traced_paths": stats.traced_paths, "transport_batches": stats.transport_batches,
            "pending_probes": stats.pending_probes,
        }));
    }

    pub fn report(&self) -> Value {
        let phases: Vec<_> = self.phases.iter().map(|(name, phase)| {
            json!({
                "phase": name, "frame_ms": distribution(&phase.frames_ms),
                "frames_over_16_7_ms": phase.frames_ms.iter().filter(|ms| **ms > 16.7).count(),
                "frames_over_33_3_ms": phase.frames_ms.iter().filter(|ms| **ms > 33.3).count(),
                "frames_over_50_ms": phase.frames_ms.iter().filter(|ms| **ms > 50.0).count(),
                "excluded_capture_frames": phase.excluded_capture_frames,
                "selection_ms": distribution(&phase.selection_ms), "merge_ms": distribution(&phase.merge_ms),
                "worker_ms": distribution(&phase.worker_ms), "transport_execution_ms": distribution(&phase.gpu_ms),
            })
        }).collect();
        json!({"phases": phases,
            "work_timeline": self.work_samples,
            "gi_main_stage_names": GI_MAIN_STAGE_NAMES,
            "host_schedule_names": super::host_timing::STAGES,
            "frame_timeline": self.frame_timeline})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_intervals_use_previous_gi_work_not_current_update() {
        let mut metrics = Metrics {
            frame_timing_enabled: Some(true),
            ..default()
        };
        let mut stats = VoxelGiStats {
            last_main_stage_ms: [2.0; 6],
            ..default()
        };
        metrics.record_frame(1.0, "startup", 20.0, false, &stats, Some([3.0; 7]));
        stats.last_main_stage_ms = [0.0; 6];
        metrics.record_frame(1.02, "startup", 21.0, true, &stats, Some([4.0; 7]));
        let rows = metrics.report()["frame_timeline"].clone();
        assert_eq!(rows[0]["previous_gi_body_ms"], json!(vec![0.0; 6]));
        assert_eq!(rows[1]["previous_gi_body_ms"], json!(vec![2.0; 6]));
        assert_eq!(rows[1]["capture_excluded"], true);
        assert_eq!(rows[1]["previous_host_schedule_ms"], json!(vec![4.0; 7]));
    }

    #[test]
    fn work_timeline_keeps_denominators_and_does_not_resample_idle_batches() {
        let mut metrics = Metrics::default();
        let mut stats = VoxelGiStats {
            loaded_pages: 3,
            transport_batches: 1,
            interactive_coverage: 1.0,
            bootstrap_coverage: 0.5,
            last_trace_batch_ms: 2.0,
            ..default()
        };
        for elapsed in [0.3, 0.6] {
            metrics.record_work(elapsed, "startup", &stats);
        }
        stats.transport_batches = 2;
        metrics.record_work(0.9, "startup", &stats);
        let report = metrics.report();
        assert_eq!(report["work_timeline"].as_array().unwrap().len(), 3);
        assert_eq!(report["work_timeline"][0]["loaded_pages"], 3);
        assert_eq!(report["work_timeline"][0]["bootstrap"], 0.5);
        assert_eq!(report["phases"][0]["worker_ms"]["count"], 2);
        assert_eq!(report["phases"][0]["transport_execution_ms"]["count"], 2);
    }

    #[test]
    fn no_measurements_are_not_zero_latency_and_tails_survive() {
        assert_eq!(distribution(&[]), json!({"count": 0}));
        assert_eq!(distribution(&[16.0, 16.0, 100.0])["p99"], 100.0);
    }
}
