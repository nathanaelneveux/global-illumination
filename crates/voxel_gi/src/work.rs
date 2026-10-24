//! Compact worker ownership: receiver identity and only the selected face
//! measurements cross threads. Topology and adaptive scheduling stay in the
//! field. Resident mode seeds presentation once, then mirrors the GPU resolve;
//! the CPU backend resolves against the live field at validated publication.
use bevy::prelude::*;
use std::ops::Range;
use std::time::Instant;

use super::{Vec3, world::SurfaceProbe};

#[derive(Default)]
pub(crate) struct TraceWork {
    /// Immutable model snapshot for this batch, shared by both backends.
    pub transport: super::TransportSettings,
    /// Retain measured GPU change packets for the view-owned receiver domain.
    pub view_transport: bool,
    /// RGB-only resident statistics can also be advanced by the render queue.
    pub shared_view_cache: bool,
    pub response_seconds: f32,
    /// Resident history ceiling and recency weighting for tests and harnesses;
    /// zero selects the gameplay defaults (65,536 and a plain running mean).
    pub history_limit: u32,
    pub history_recency: f32,
    /// Measured retention constant c (zero = distance-curve caps).
    pub measured_retention: f32,
    pub probes: Vec<ProbeWork>,
    pub faces: Vec<FaceWork>,
    /// Faces that need a resident slot before they are traced, so every
    /// published face has GPU state (and a seeded presentation) early.
    pub ensure: Vec<EnsureFace>,
    /// Changes published since the previous batch, applied on the GPU to
    /// every resident face in range before this batch's rays.
    pub retention_events: Vec<super::retention::RetentionEvent>,
    /// Pages whose receivers were retired (receiver radius): release their
    /// resident slots and sun corners.
    pub retire_pages: Vec<IVec3>,
    /// GPU selection: a budget of faces the GPU picks itself after the CPU's
    /// traced faces, and what it ranks them by. Picks return in `picked`.
    pub picks: u32,
    pub focus: IVec3,
    /// Never zero (28 bits): GPU picks avoid faces of recent CPU jobs.
    pub batch_id: u32,
    pub transport_revision: u64,
    pub source_revision: u64,
    pub stable_error: f32,
    pub stable_samples: u32,
    pub audit_ms: u32,
    /// Statistics-mirror records the GPU emits this batch.
    pub mirror_budget: u32,
    /// Statistics-mirror records decoded from the readback, with the revision
    /// (low 32 bits) the slot had.
    pub picked: Vec<(FaceWork, u32)>,
    /// This batch's GPU picks: face and the revision tag its slot had.
    pub pick_tags: Vec<(IVec3, usize, u32)>,
}

pub(crate) struct ProbeWork {
    pub voxel: IVec3,
    pub face_mask: u8,
    /// Explicit sky demand, independent of indirect refinement. Unknown corners
    /// request a retry; already measured corners wait for the sky change queue.
    pub sky_requested: bool,
    pub trace_mask: u8,
    pub revision: u64,
    pub source_revision: u64,
    pub transport_revision: u64,
    pub restart: bool,
    /// The newest retention event this receiver observes by being traced
    /// (its page stamp), when one woke it. Its statistics were already
    /// capped on the GPU; publication restarts its stability.
    pub retention: Option<u64>,
    pub sky: [[u32; 4]; 6],
    start: usize,
}

impl ProbeWork {
    /// The header for one GPU-picked face of `probe`, at the batch's
    /// revisions; its face is the only one in the slice publication gets.
    pub fn picked(
        probe: &SurfaceProbe,
        face: usize,
        transport_revision: u64,
        source_revision: u64,
    ) -> Self {
        Self {
            voxel: probe.voxel,
            face_mask: probe.face_mask,
            sky_requested: false,
            trace_mask: 1 << face,
            revision: probe.revision,
            source_revision,
            transport_revision,
            restart: false,
            retention: None,
            sky: [[0; 4]; 6],
            start: 0,
        }
    }

    pub fn faces(&self) -> Range<usize> {
        self.start..self.start + self.trace_mask.count_ones() as usize
    }

    /// Publish only after geometry/source validation. Untouched faces and live
    /// order indices remain authoritative. Resident presentation has already
    /// been resolved by the same filter on the GPU; do not filter it twice.
    pub fn publish(
        &self,
        faces: &[FaceWork],
        probe: &mut SurfaceProbe,
    ) {
        let same_revision =
            probe.transport_revision == self.transport_revision && self.retention.is_none();
        // These faces' merges were accepted: nothing left for the GPU to undo.
        probe.stale_merge &= !self.trace_mask;
        probe.relit_mask &= !self.trace_mask;
        if !same_revision || self.restart {
            probe.fresh_mask = 0;
            probe.progress.settled = 0;
            #[cfg(test)]
            if let Some(stats) = probe.stats.as_mut() {
                stats.stable_batches = [0; 6];
            }
        }
        if self.restart {
            probe.progress.bootstrap = [0; 6];
        }
        #[cfg(test)]
        if self.restart
            && let Some(stats) = probe.stats.as_mut()
        {
            for face in 0..6 {
                if self.trace_mask & (1 << face) == 0 {
                    stats.samples[face] = 0;
                    stats.radiance[face] = Vec3::ZERO;
                    stats.second_moment[face] = Vec3::ZERO;
                    stats.face_updated_at[face] = None;
                }
            }
        }
        #[cfg(test)]
        {
            probe.trace_mask = self.trace_mask;
        }
        probe.source_revision = self.source_revision;
        probe.transport_revision = self.transport_revision;
        for measurement in faces {
            let face = measurement.face;
            measurement.write(probe);
            // The GPU resolved this face's presentation; a face without a
            // resident slot keeps its current display.
            if measurement.resident.y != 0 {
                probe.display[face].indirect = measurement.presentation.truncate();
                probe.display[face].confidence = measurement.presentation.w;
            }
            probe.fresh_mask |= 1 << face;
        }
    }
}

/// A face to give a resident slot (and seed) without tracing it yet.
#[derive(Clone)]
pub(crate) struct EnsureFace {
    pub face: FaceWork,
    pub revision: u64,
    pub source_revision: u64,
}

#[derive(Clone)]
pub(crate) struct FaceWork {
    /// GPU scheduling verdict, independent of optional CPU moments.
    pub gpu_settled: Option<bool>,
    pub voxel: IVec3,
    pub face: usize,
    pub radiance: Vec3,
    pub second_moment: Vec3,
    pub direct: Vec3,
    pub custom_direct: Vec3,
    pub samples: u32,
    /// A change constraint, not a copy of the CPU snapshot's possibly old count.
    /// Resident accumulation applies this to its own latest statistical weight.
    pub history_cap: u32,
    /// The CPU rejected this face's last merged batch: the GPU must undo it.
    pub rollback: bool,
    /// The cap a changed slot revision applies to the GPU history: 16 for a
    /// relit face, 0 on a restart, stricter stale/source limits, else none.
    pub revision_cap: u32,
    /// The receiver's material id (GPU edit measurement uses its albedo).
    pub material: u8,
    pub sequence: u32,
    pub updated_at: Option<Instant>,
    pub presentation: Vec4,
    pub resident: UVec2,
}

impl TraceWork {
    pub fn clear(&mut self) {
        self.probes.clear();
        self.faces.clear();
        self.ensure.clear();
        self.retention_events.clear();
        self.retire_pages.clear();
        self.picked.clear();
        self.pick_tags.clear();
    }

    /// Queue slot allocation for this probe's faces that have none yet.
    pub fn ensure_slots(&mut self, probe: &SurfaceProbe) {
        for face in 0..6 {
            if probe.face_mask & (1 << face) != 0 && probe.resident[face].y == 0 {
                let start = self.faces.len();
                self.push_face(probe, face, false, u32::MAX, u32::MAX);
                let face = self.faces.remove(start);
                // The batch's source revision, not the probe's: an untraced
                // probe still has 0, below any model reset, and its binding
                // would be rejected (no address: the renderer falls back).
                self.ensure.push(EnsureFace {
                    face,
                    revision: probe.revision,
                    source_revision: self.source_revision.max(probe.source_revision),
                });
            }
        }
    }

    fn push_face(
        &mut self,
        probe: &SurfaceProbe,
        face: usize,
        restart: bool,
        stale_limit: u32,
        source_history_limit: u32,
    ) {
        let measured = probe.measured(face);
        self.faces.push(FaceWork {
            gpu_settled: None,
            voxel: probe.voxel,
            face,
            radiance: if restart {
                Vec3::ZERO
            } else {
                measured.radiance
            },
            second_moment: if restart {
                Vec3::ZERO
            } else {
                measured.second_moment
            },
            direct: measured.direct,
            custom_direct: measured.custom_direct,
            samples: if restart {
                0
            } else {
                measured.samples
                    .min(stale_limit)
                    .min(source_history_limit)
            },
            sequence: probe.sequence(face),
            presentation: probe.display[face]
                .indirect
                .extend(probe.display[face].confidence),
            resident: UVec2::ZERO,
            history_cap: if restart {
                0
            } else {
                stale_limit.min(source_history_limit)
            },
            rollback: probe.stale_merge & (1 << face) != 0,
            material: probe.material,
            revision_cap: if restart {
                0
            } else {
                let relit = if probe.relit_mask & (1 << face) != 0 {
                    super::geometry::NEAR_HISTORY
                } else {
                    u32::MAX
                };
                relit.min(stale_limit).min(source_history_limit)
            },
            updated_at: if restart {
                None
            } else {
                measured.updated_at
            },
        });
    }

    /// Preserve the scheduler's history policy without cloning six-face state.
    /// A source reset is also recorded on the header so publication can clear
    /// unselected measurements without touching their presentation history.
    pub fn push(
        &mut self,
        probe: &SurfaceProbe,
        mask: u8,
        transport_revision: u64,
        source_revision: u64,
        restart: bool,
        stale_history_limit: u32,
        source_history_limit: u32,
    ) {
        let mask = mask & probe.face_mask;
        let start = self.faces.len();
        for face in 0..6 {
            if mask & (1 << face) == 0 {
                continue;
            }
            let stale = probe.transport_revision != transport_revision
                || probe.fresh_mask & (1 << face) == 0;
            let stale_limit = if stale { stale_history_limit } else { u32::MAX };
            self.push_face(probe, face, restart, stale_limit, source_history_limit);
        }
        self.probes.push(ProbeWork {
            voxel: probe.voxel,
            face_mask: probe.face_mask,
            sky_requested: mask == 0
                || (0..6).any(|face| {
                    probe.face_mask & (1 << face) != 0 && probe.display[face].sky.contains(&0)
                }),
            trace_mask: mask,
            revision: probe.revision,
            source_revision,
            transport_revision,
            restart,
            retention: None,
            // Filled by SkyCache before return; old presentation never crosses
            // the worker boundary, and stale sky is rejected by the publisher.
            sky: [[0; 4]; 6],
            start,
        });
    }

    #[cfg(test)]
    pub fn from_probes(probes: &[SurfaceProbe], sky_only: bool) -> Self {
        let mut work = Self::default();
        for probe in probes {
            work.push(
                probe,
                if sky_only { 0 } else { probe.trace_mask },
                probe.transport_revision,
                probe.source_revision,
                false,
                super::geometry::UNTRACKED_HISTORY,
                u32::MAX,
            );
            // Reference/benchmark inputs already own their history policy.
            let header = work.probes.last_mut().unwrap();
            // Reference calls explicitly request current sky, including when
            // testing geometry/source changes without the production scheduler.
            header.sky_requested = true;
            header.sky = probe.display.map(|face| face.sky);
            for face in &mut work.faces[header.faces()] {
                face.samples = probe.stats().samples[face.face];
                face.history_cap = u32::MAX;
            }
        }
        work
    }

    #[cfg(test)]
    pub fn copy_measurements(&self, probes: &mut [SurfaceProbe]) {
        for (job, probe) in self.probes.iter().zip(probes) {
            debug_assert_eq!(job.voxel, probe.voxel);
            if job.sky_requested {
                for face in 0..6 {
                    probe.display[face].sky = job.sky[face];
                }
            }
            for face in &self.faces[job.faces()] {
                face.write(probe);
            }
        }
    }
}

impl FaceWork {
    /// The probe already holds a later merge of this face (GPU sequences only
    /// advance; wrapping compare). A statistics-mirror record is classified
    /// before the batch traces, so a CPU job in the same batch is newer.
    pub fn predates(&self, probe: &SurfaceProbe) -> bool {
        (probe.sequence(self.face).wrapping_sub(self.sequence) as i32) > 0
    }

    /// Statistics only reach probes that keep them (`SurfaceProbe::stats`);
    /// the resident address and time always do.
    pub fn write(&self, probe: &mut SurfaceProbe) {
        let face = self.face;
        probe.progress.sequence[face] = self.sequence;
        probe.progress.bootstrap[face] = self.samples.min(super::tracer::BOOTSTRAP_SAMPLES) as u8;
        probe.progress.settled &= !(1 << face);
        if self.gpu_settled == Some(true) {
            probe.progress.settled |= 1 << face;
        }
        #[cfg(test)]
        if let Some(stats) = probe.stats.as_mut() {
            stats.radiance[face] = self.radiance;
            stats.second_moment[face] = self.second_moment;
            stats.direct[face] = self.direct;
            stats.custom_direct[face] = self.custom_direct;
            stats.samples[face] = self.samples;
            stats.sequence[face] = self.sequence;
            stats.face_updated_at[face] = self.updated_at;
        }
        probe.resident[face] = self.resident;
        probe.updated_at = self.updated_at;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slim_progress_reports_bootstrap_and_rejects_older_mirrors_without_allocating_statistics() {
        assert_eq!(
            std::mem::size_of::<super::super::world::ProbeProgress>(),
            32
        );
        let mut full = SurfaceProbe::new(IVec3::ZERO, 4);
        full.stats_mut().samples[2] = 2048;
        full.stats_mut().sequence[2] = 4096;
        let mut work = TraceWork::from_probes(&[full], false);
        work.faces[0].resident = UVec2::new(0, 1);
        work.faces[0].gpu_settled = Some(true);
        let mut slim = SurfaceProbe::new_slim(IVec3::ZERO, 4);
        work.probes[0].publish(&work.faces, &mut slim);
        assert!(slim.stats.is_none());
        assert_eq!(super::super::probe_metrics(&slim, 0), (1, 8, 1, 1));
        assert!(!work.faces[0].predates(&slim));
        work.faces[0].sequence -= 16;
        assert!(work.faces[0].predates(&slim));
        slim.request_refresh();
        assert_eq!(super::super::probe_metrics(&slim, 0), (1, 8, 1, 0));
        slim.update_face_mask(5);
        assert_eq!(slim.bootstrap_samples(0), 0, "new exposure has no samples");
        assert_eq!(
            slim.bootstrap_samples(2),
            8,
            "unchanged face retains progress"
        );
        work.probes[0].restart = true;
        work.probes[0].publish(&[], &mut slim);
        assert_eq!(super::super::probe_metrics(&slim, 0), (2, 0, 0, 0));
        assert!(slim.stats.is_none());
    }

    #[test]
    fn retention_wake_publishes_the_selected_stamp_and_restarts_stability() {
        let mut probe = SurfaceProbe::new(IVec3::ZERO, 63);
        probe.stats_mut().samples = [4096; 6];
        probe.stats_mut().radiance = [Vec3::ONE; 6];
        probe.stats_mut().second_moment = [Vec3::ONE * 2.0; 6];
        probe.stats_mut().sequence = [9000; 6];
        let mut work = TraceWork::default();
        work.push(&probe, 1, 1, 0, false, 256, u32::MAX);
        work.probes[0].retention = Some(7);
        work.faces[0].samples += 16;
        work.faces[0].sequence += 16;
        work.probes[0].publish(&work.faces, &mut probe);
        // Counts come from the GPU, which already applied the change.
        assert_eq!(probe.stats().samples, [272, 4096, 4096, 4096, 4096, 4096]);
        assert_eq!(probe.stats().sequence, [9016, 9000, 9000, 9000, 9000, 9000]);
        assert_eq!(probe.fresh_mask, 1);
        assert_eq!(probe.stats().radiance, [Vec3::ONE; 6]);
        assert_eq!(probe.stats().second_moment, [Vec3::ONE * 2.0; 6]);
    }

    #[test]
    fn long_refinement_history_obeys_geometry_and_source_caps() {
        let mut probe = SurfaceProbe::new(IVec3::ZERO, 4);
        probe.stats_mut().samples[2] = super::super::tracer::HISTORY_LIMIT;
        probe.stats_mut().sequence[2] = 100_000;
        probe.stats_mut().radiance[2] = Vec3::ONE;
        let mut work = TraceWork::default();
        for (revision, geometry_limit, source_limit, restart, expected) in [
            (0, 64, u32::MAX, false, super::super::tracer::HISTORY_LIMIT),
            (1, 64, u32::MAX, false, 64),
            (1, 16, u32::MAX, false, 16),
            (1, 256, u32::MAX, false, 256),
            (
                1,
                u32::MAX,
                u32::MAX,
                false,
                super::super::tracer::HISTORY_LIMIT,
            ),
            (1, 256, 4, false, 4),
            (0, 64, 4, false, 4),
            (1, 256, u32::MAX, true, 0),
            (1, u32::MAX, u32::MAX, true, 0),
        ] {
            work.clear();
            work.push(
                &probe,
                4,
                revision,
                0,
                restart,
                geometry_limit,
                source_limit,
            );
            assert_eq!(work.faces[0].samples, expected);
            assert_eq!(work.faces[0].sequence, 100_000);
            if restart {
                assert_eq!(work.faces[0].radiance, Vec3::ZERO);
            }
        }
    }

    #[test]
    fn relit_faces_carry_an_explicit_cap_never_the_mirrored_count() {
        let mut probe = SurfaceProbe::new(IVec3::ZERO, 5);
        // The CPU never merged this face: its count reads 0.
        probe.stats_mut().samples = [0; 6];
        probe.request_relight();
        let mut work = TraceWork::default();
        work.push(&probe, 5, 0, 0, false, u32::MAX, u32::MAX);
        assert!(
            work.faces
                .iter()
                .all(|f| f.revision_cap == super::super::geometry::NEAR_HISTORY)
        );
        work.probes[0].publish(&work.faces, &mut probe);
        assert_eq!(probe.relit_mask, 0, "an accepted job consumes the relight");
        // A refresh (nearby edit) changes the revision but caps nothing.
        probe.request_refresh();
        let mut work = TraceWork::default();
        work.push(&probe, 5, 0, 0, false, u32::MAX, u32::MAX);
        assert!(work.faces.iter().all(|f| f.revision_cap == u32::MAX));
        let mut work = TraceWork::default();
        work.push(&probe, 5, 0, 0, true, u32::MAX, u32::MAX);
        assert!(work.faces.iter().all(|f| f.revision_cap == 0), "restart");
    }

    #[test]
    fn retaining_history_still_publishes_freshness_and_wakes_stable_faces() {
        let mut probe = SurfaceProbe::new(IVec3::ZERO, 5);
        probe.stats_mut().samples = [4096; 6];
        probe.stats_mut().sequence = [8000; 6];
        probe.stats_mut().radiance = [Vec3::ONE; 6];
        probe.stats_mut().stable_batches = [4; 6];
        let mut work = TraceWork::default();
        work.push(&probe, 4, 1, 1, false, u32::MAX, u32::MAX);
        assert_eq!(work.faces[0].samples, 4096);
        assert_eq!(work.faces[0].sequence, 8000);
        work.probes[0].publish(&work.faces, &mut probe);
        assert_eq!(probe.transport_revision, 1);
        assert_eq!(probe.source_revision, 1);
        assert_eq!(probe.fresh_mask, 4);
        assert_eq!(probe.stats().stable_batches[0], 0);
        assert_eq!(probe.stats().samples[0], 4096);
        assert_eq!(probe.stats().samples[2], 4096);
    }

    #[test]
    fn single_face_work_is_less_than_half_a_full_probe_and_sky_copies_no_history() {
        let bytes = size_of::<ProbeWork>() + size_of::<FaceWork>();
        let full = size_of::<SurfaceProbe>() + size_of::<super::super::world::ProbeStatistics>();
        assert!(bytes < full / 2);
        // A probe without CPU statistics keeps the core: topology, display,
        // revisions and resident addresses.
        assert!(
            size_of::<SurfaceProbe>() < 400,
            "{}",
            size_of::<SurfaceProbe>()
        );
        let mut probe = SurfaceProbe::new(IVec3::ZERO, 63);
        probe.stats_mut().samples = [128; 6];
        probe.stats_mut().sequence = [901; 6];
        let mut work = TraceWork::default();
        work.push(&probe, 4, 1, 0, false, 64, u32::MAX);
        work.push(&probe, 0, 0, 0, false, 64, u32::MAX);
        assert_eq!(work.faces.len(), 1);
        assert_eq!(work.faces[0].face, 2);
        assert_eq!(work.faces[0].samples, 64);
        assert_eq!(work.faces[0].sequence, 901);
        assert_eq!(work.probes[1].faces().len(), 0);
        eprintln!(
            "worker bytes: receiver={}, selected_face={}, full_probe={}",
            size_of::<ProbeWork>(),
            size_of::<FaceWork>(),
            size_of::<SurfaceProbe>()
        );
    }
}
