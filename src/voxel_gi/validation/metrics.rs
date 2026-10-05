//! Bounded field diagnostics, deliberately separate from visual ground truth.
use super::super::VoxelEnergyField;
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
}

struct EditWatch {
    at: Instant,
    revision: u64,
    targets: Vec<(IVec3, usize)>,
}

impl Metrics {
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
            })
        }).collect();
        json!({"phases": phases, "edit_first_fresh_measurement_ms": distribution(&self.edit_latency_ms),
            "edits_superseded_before_refresh": self.superseded_edits,
            "edits_without_receivers": self.untargeted_edits,
            "edits_still_pending": usize::from(self.watch.is_some())})
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::world::SurfaceProbe;
    use super::*;

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
