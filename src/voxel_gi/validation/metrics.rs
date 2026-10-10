//! Bounded field diagnostics, deliberately separate from visual ground truth.
use super::super::{VoxelEnergyField, VoxelGiRuntime, VoxelGiStats};
use bevy::prelude::*;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

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
    near_age_ms: Vec<f32>,
    far_age_ms: Vec<f32>,
    checked_faces: usize,
    unmeasured_faces: usize,
    stale_faces: usize,
    unknown_sky_faces: usize,
    display_steps: Vec<f32>,
    max_dark_positive_step: f32,
    selection_ms: Vec<f32>,
    merge_ms: Vec<f32>,
    worker_ms: Vec<f32>,
    gpu_ms: Vec<f32>,
}

#[derive(Default)]
pub(super) struct Metrics {
    pub phases: BTreeMap<&'static str, Phase>,
    previous_display: HashMap<(IVec3, usize), Vec3>,
    last_sample: f32,
    watch: Option<EditWatch>,
    edit_latency_ms: Vec<f32>,
    superseded_edits: usize,
    untargeted_edits: usize,
    work_samples: Vec<Value>,
    last_work_sample: f32,
    last_batch: u64,
    last_transport_batch: u64,
    frame_timeline: Vec<Value>,
    previous_gi: [f32; 7],
    frame_timing_enabled: Option<bool>,
}

struct EditWatch {
    at: Instant,
    revision: u64,
    targets: Vec<(IVec3, usize)>,
}

impl Metrics {
    pub fn record_frame(
        &mut self,
        elapsed: f32,
        phase: &'static str,
        frame_ms: f32,
        capture_excluded: bool,
        stats: &VoxelGiStats,
    ) {
        if !*self
            .frame_timing_enabled
            .get_or_insert_with(|| std::env::var_os("VOXEL_GI_VALIDATE_FRAME_TIMINGS").is_some())
        {
            return;
        }
        // Time<Real>'s interval ends at this frame's start. Associate the
        // preceding frame's GI bodies, not this Update's work. Pipeline/render
        // overlap still prevents treating the remainder as pure BVW cost.
        self.frame_timeline
            .push(json!({"at_s":elapsed,"phase":phase,
            "interval_ms":frame_ms,"capture_excluded":capture_excluded,
            "previous_gi_body_ms":self.previous_gi}));
        self.previous_gi = stats.last_main_stage_ms;
    }
    /// O(1) observations of publication and lighting progress. A small early
    /// page set can be fully covered while most terrain is still unpublished;
    /// keep its denominator and ingestion backlog, not a misleading first-100% time.
    pub fn record_work(
        &mut self,
        elapsed: f32,
        phase: &'static str,
        stats: &VoxelGiStats,
        runtime: &VoxelGiRuntime,
        field: &VoxelEnergyField,
    ) {
        let batches = stats.transport_batches + stats.sky_only_batches;
        if batches != self.last_batch {
            self.last_batch = batches;
            let phase = self.phases.entry(phase).or_default();
            phase.selection_ms.push(stats.last_selection_ms);
            phase.merge_ms.push(stats.last_merge_ms);
            phase.worker_ms.push(stats.last_worker_ms);
            // Last transport timestamp is not updated by a sky-only batch.
            if stats.transport_batches != self.last_transport_batch {
                self.last_transport_batch = stats.transport_batches;
                phase.gpu_ms.push(stats.last_trace_batch_ms);
            }
        }
        if elapsed - self.last_work_sample < 0.25 {
            return;
        }
        self.last_work_sample = elapsed;
        self.work_samples.push(json!({
            "elapsed_s": elapsed, "phase": phase,
            "loaded_pages": runtime.loaded_pages.len(), "published_probes": field.probes.len(),
            "publishing_pages": field.publishing_pages.len(),
            "dirty_pages": runtime.dirty_pages.len(), "scene_build_pending": runtime.scene_task.is_some(),
            "transport_revision": field.transport_revision, "scene_generation": runtime.generation,
            "coverage": stats.interactive_coverage, "bootstrap": stats.bootstrap_coverage,
            "traced_paths": stats.traced_paths, "transport_batches": stats.transport_batches,
            "pending_probes": stats.pending_probes,
        }));
    }

    pub fn edit(&mut self, field: &VoxelEnergyField, targets: Vec<(IVec3, usize)>) {
        if self.watch.take().is_some() {
            self.superseded_edits += 1;
        }
        if targets.is_empty() {
            self.untargeted_edits += 1;
            return;
        }
        self.watch = Some(EditWatch {
            at: Instant::now(),
            revision: field.transport_revision,
            targets,
        });
    }

    pub fn poll_edit(&mut self, field: &VoxelEnergyField) {
        let Some(watch) = &self.watch else {
            return;
        };
        if field.transport_revision != watch.revision
            && watch.targets.iter().all(|(voxel, face)| {
                field.probes.get(voxel).is_some_and(|probe| {
                    probe.face_mask & (1 << face) != 0
                        && probe.samples[*face] > 0
                        && probe.fresh_mask & (1 << face) != 0
                        && probe.transport_revision == field.transport_revision
                        && probe.face_updated_at[*face].is_some_and(|at| at >= watch.at)
                })
            })
        {
            self.edit_latency_ms
                .push(watch.at.elapsed().as_secs_f32() * 1000.0);
            self.watch = None;
        }
    }

    pub fn sample(
        &mut self,
        field: &VoxelEnergyField,
        eye: Vec3,
        elapsed: f32,
        phase: &'static str,
    ) {
        if elapsed - self.last_sample < 0.25 {
            return;
        }
        self.last_sample = elapsed;
        let stats = self.phases.entry(phase).or_default();
        let now = Instant::now();
        let mut next_display = HashMap::new();
        // At most 1,024 probes, four times a second; no per-frame world scan.
        for voxel in field
            .order
            .iter()
            .step_by(field.order.len().div_ceil(1024).max(1))
        {
            let Some(probe) = field.probes.get(voxel) else {
                continue;
            };
            let near = voxel.as_vec3().distance_squared(eye) <= 32.0 * 32.0;
            for face in 0..6 {
                if probe.face_mask & (1 << face) == 0 {
                    continue;
                }
                stats.checked_faces += 1;
                if probe.display[face].sky.contains(&0) {
                    stats.unknown_sky_faces += 1;
                }
                if probe.samples[face] == 0 || probe.face_updated_at[face].is_none() {
                    stats.unmeasured_faces += 1;
                } else {
                    let age = now
                        .saturating_duration_since(probe.face_updated_at[face].unwrap())
                        .as_secs_f32()
                        * 1000.0;
                    if near {
                        stats.near_age_ms.push(age);
                    } else {
                        stats.far_age_ms.push(age);
                    }
                    if probe.fresh_mask & (1 << face) == 0
                        || probe.transport_revision != field.transport_revision
                        || probe.source_revision != field.source_revision
                    {
                        stats.stale_faces += 1;
                    }
                }
                if near {
                    let value = probe.display[face].indirect;
                    let key = (*voxel, face);
                    if let Some(old) = self.previous_display.get(&key) {
                        stats.display_steps.push((value - *old).abs().max_element());
                        if old.max_element() <= 0.02 {
                            stats.max_dark_positive_step = stats
                                .max_dark_positive_step
                                .max((value - *old).max_element());
                        }
                    }
                    next_display.insert(key, value);
                }
            }
        }
        self.previous_display = next_display;
    }

    pub fn report(&self) -> Value {
        let phases: Vec<_> = self.phases.iter().map(|(name, phase)| {
            json!({
                "phase": name, "frame_ms": distribution(&phase.frames_ms),
                "frames_over_16_7_ms": phase.frames_ms.iter().filter(|ms| **ms > 16.7).count(),
                "frames_over_33_3_ms": phase.frames_ms.iter().filter(|ms| **ms > 33.3).count(),
                "frames_over_50_ms": phase.frames_ms.iter().filter(|ms| **ms > 50.0).count(),
                "excluded_capture_frames": phase.excluded_capture_frames,
                "near_age_ms": distribution(&phase.near_age_ms), "far_age_ms": distribution(&phase.far_age_ms),
                "checked_face_observations": phase.checked_faces,
                "unmeasured_face_observations": phase.unmeasured_faces,
                "stale_measured_face_observations": phase.stale_faces,
                "unknown_sky_face_observations": phase.unknown_sky_faces,
                "near_indirect_display_step": distribution(&phase.display_steps),
                "max_dark_positive_indirect_step": phase.max_dark_positive_step,
                "selection_ms": distribution(&phase.selection_ms), "merge_ms": distribution(&phase.merge_ms),
                "worker_ms": distribution(&phase.worker_ms), "transport_execution_ms": distribution(&phase.gpu_ms),
            })
        }).collect();
        json!({"phases": phases, "edit_first_fresh_measurement_ms": distribution(&self.edit_latency_ms),
            "edits_superseded_before_refresh": self.superseded_edits,
            "edits_without_receivers": self.untargeted_edits,
            "edits_still_pending": usize::from(self.watch.is_some()),
            "work_timeline": self.work_samples,
            "gi_main_stage_names": super::super::telemetry::MAIN_STAGE_NAMES,
            "frame_timeline": self.frame_timeline})
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::world::SurfaceProbe;
    use super::*;

    #[test]
    fn frame_intervals_use_previous_gi_work_not_current_update() {
        let mut metrics = Metrics {
            frame_timing_enabled: Some(true),
            ..default()
        };
        let mut stats = VoxelGiStats {
            last_main_stage_ms: [2.0; 7],
            ..default()
        };
        metrics.record_frame(1.0, "startup", 20.0, false, &stats);
        stats.last_main_stage_ms = [0.0; 7];
        metrics.record_frame(1.02, "startup", 21.0, true, &stats);
        let rows = metrics.report()["frame_timeline"].clone();
        assert_eq!(rows[0]["previous_gi_body_ms"], json!(vec![0.0; 7]));
        assert_eq!(rows[1]["previous_gi_body_ms"], json!(vec![2.0; 7]));
        assert_eq!(rows[1]["capture_excluded"], true);
    }

    #[test]
    fn work_timeline_keeps_denominators_and_does_not_resample_idle_batches() {
        let mut metrics = Metrics::default();
        let mut runtime = VoxelGiRuntime::default();
        let mut field = VoxelEnergyField::default();
        runtime.dirty_pages.insert(IVec3::ZERO);
        field.publishing_pages.insert(IVec3::X);
        let mut stats = VoxelGiStats {
            transport_batches: 1,
            interactive_coverage: 1.0,
            bootstrap_coverage: 0.5,
            last_trace_batch_ms: 2.0,
            ..default()
        };
        for elapsed in [0.3, 0.6] {
            metrics.record_work(elapsed, "startup", &stats, &runtime, &field);
        }
        stats.sky_only_batches = 1;
        metrics.record_work(0.9, "startup", &stats, &runtime, &field);
        let report = metrics.report();
        assert_eq!(report["work_timeline"].as_array().unwrap().len(), 3);
        assert_eq!(report["work_timeline"][0]["dirty_pages"], 1);
        assert_eq!(report["work_timeline"][0]["publishing_pages"], 1);
        assert_eq!(report["work_timeline"][0]["bootstrap"], 0.5);
        assert_eq!(report["phases"][0]["worker_ms"]["count"], 2);
        assert_eq!(report["phases"][0]["transport_execution_ms"]["count"], 1);
    }

    #[test]
    fn no_measurements_are_not_zero_latency_and_tails_survive() {
        assert_eq!(distribution(&[]), json!({"count": 0}));
        assert_eq!(distribution(&[16.0, 16.0, 100.0])["p99"], 100.0);
        let mut metrics = Metrics::default();
        let field = VoxelEnergyField::default();
        metrics.edit(&field, vec![(IVec3::ZERO, 2)]);
        metrics.poll_edit(&field);
        assert_eq!(metrics.report()["edits_still_pending"], 1);
        metrics.edit(&field, vec![(IVec3::ZERO, 2)]);
        assert_eq!(metrics.report()["edits_superseded_before_refresh"], 1);
    }

    #[test]
    fn edit_latency_requires_new_geometry_and_fresh_face_evidence() {
        let mut metrics = Metrics::default();
        let mut field = VoxelEnergyField::default();
        metrics.edit(&field, vec![(IVec3::ZERO, 2)]);
        let mut probe = SurfaceProbe::new(IVec3::ZERO, 4);
        probe.samples[2] = 4;
        probe.face_updated_at[2] = Some(Instant::now());
        field.probes.insert(IVec3::ZERO, probe);
        metrics.poll_edit(&field);
        assert!(metrics.edit_latency_ms.is_empty());
        field.transport_revision = 1;
        field
            .probes
            .get_mut(&IVec3::ZERO)
            .unwrap()
            .transport_revision = 1;
        metrics.poll_edit(&field);
        assert_eq!(metrics.edit_latency_ms.len(), 1);
        assert!(metrics.watch.is_none());
    }

    #[test]
    fn unknown_and_stale_faces_are_reported_not_hidden_by_fast_frames() {
        let mut field = VoxelEnergyField::default();
        let mut probe = SurfaceProbe::new(IVec3::ZERO, 5);
        probe.samples[2] = 4;
        probe.face_updated_at[2] = Some(Instant::now());
        field.transport_revision = 1;
        field.order.push(IVec3::ZERO);
        field.probes.insert(IVec3::ZERO, probe);
        let mut metrics = Metrics::default();
        metrics.sample(&field, Vec3::ZERO, 1.0, "settled");
        let report = metrics.report();
        let phase = &report["phases"][0];
        assert_eq!(phase["unmeasured_face_observations"], 1);
        assert_eq!(phase["stale_measured_face_observations"], 1);
        assert_eq!(phase["unknown_sky_face_observations"], 2);
    }
}
