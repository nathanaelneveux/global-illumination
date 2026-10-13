//! Compact worker ownership: receiver identity and only the selected face
//! measurements cross threads. Display history, topology indices, and adaptive
//! scheduling state stay in the authoritative field until validated publication.
use bevy::prelude::*;
use std::ops::Range;
use std::time::Instant;

use super::{EnergySpectrum, world::SurfaceProbe};

#[derive(Default)]
pub(crate) struct TraceWork {
    /// Immutable model snapshot for this batch, shared by both backends.
    pub transport: super::TransportSettings,
    pub probes: Vec<ProbeWork>,
    pub faces: Vec<FaceWork>,
    pub(super) pending_retention: Vec<RetentionWork>,
}

pub(super) struct RetentionWork {
    probe_index: usize,
    since: u64,
    snapshot: super::retention::Snapshot,
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
    /// Spatial evidence and cap; None preserves the legacy comparison policy.
    pub retention: Option<(u64, u32)>,
    #[cfg(debug_assertions)]
    pub retention_audit: Option<Box<super::validation::retention_audit::Decision>>,
    pub sky: [[u32; 4]; 6],
    start: usize,
}

impl ProbeWork {
    pub fn faces(&self) -> Range<usize> {
        self.start..self.start + self.trace_mask.count_ones() as usize
    }

    /// Publish only after geometry/source validation. Untouched faces and live
    /// order indices remain authoritative; display history never comes back
    /// from a worker. The numerical resolve and stability rules are unchanged.
    pub fn publish(
        &self,
        faces: &[FaceWork],
        probe: &mut SurfaceProbe,
        now: Instant,
        settings: &super::VoxelGiSettings,
    ) {
        let same_revision = probe.transport_revision == self.transport_revision
            && self.retention.is_none_or(|(_, cap)| cap == u32::MAX);
        let previous_stable = probe.stable_batches;
        let previous_display_time = probe.display_updated_at;
        if let Some((revision, cap)) = self.retention {
            // Consume evidence for all faces atomically, including faces not
            // selected in this batch. Partial selections must not lose a cap
            // when the probe-level scene revision advances.
            probe.samples = probe.samples.map(|samples| samples.min(cap));
            probe.retention_revision = revision;
        }
        if !same_revision || self.restart {
            probe.fresh_mask = 0;
            probe.stable_batches = [0; 6];
        }
        if self.restart {
            for face in 0..6 {
                if self.trace_mask & (1 << face) == 0 {
                    probe.samples[face] = 0;
                    probe.radiance[face] = EnergySpectrum::ZERO;
                    probe.second_moment[face] = EnergySpectrum::ZERO;
                    probe.face_updated_at[face] = None;
                }
            }
        }
        probe.trace_mask = self.trace_mask;
        probe.source_revision = self.source_revision;
        probe.transport_revision = self.transport_revision;
        for measurement in faces {
            let face = measurement.face;
            let previous_time = probe.face_updated_at[face].or(previous_display_time);
            let before = probe.radiance[face];
            measurement.write(probe);
            probe.resolve_face(
                face,
                previous_time,
                now,
                settings.display_response_seconds(),
            );
            super::scheduling::record_face_batch(
                probe,
                face,
                before,
                previous_stable[face],
                same_revision,
                now,
                settings,
            );
        }
        probe.display_updated_at = Some(now);
    }
}

#[derive(Clone)]
pub(crate) struct FaceWork {
    pub voxel: IVec3,
    pub face: usize,
    pub radiance: EnergySpectrum,
    pub second_moment: EnergySpectrum,
    pub direct: EnergySpectrum,
    pub custom_direct: Vec3,
    pub samples: u32,
    pub sequence: u32,
    pub updated_at: Option<Instant>,
}

impl TraceWork {
    pub fn clear(&mut self) {
        self.probes.clear();
        self.faces.clear();
        self.pending_retention.clear();
    }

    pub fn defer_retention(&mut self, since: u64, snapshot: super::retention::Snapshot) {
        self.pending_retention.push(RetentionWork {
            probe_index: self.probes.len() - 1,
            since,
            snapshot,
        });
    }

    /// Resolve geometry and late-sun evidence off the game thread, before either
    /// backend merges fresh paths. Source caps were already applied at selection.
    pub fn apply_retention(&mut self) {
        for pending in self.pending_retention.drain(..) {
            let probe = &mut self.probes[pending.probe_index];
            let cap = pending.snapshot.limit(probe.voxel, pending.since);
            #[cfg(debug_assertions)]
            if super::validation::retention_audit::site(probe.voxel).is_some() {
                probe.retention_audit = Some(Box::new(
                    pending.snapshot.explain(probe.voxel, pending.since),
                ));
            }
            probe.retention = Some((pending.snapshot.revision, cap));
            for face in &mut self.faces[probe.faces()] {
                face.samples = face.samples.min(cap);
            }
        }
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
            self.faces.push(FaceWork {
                voxel: probe.voxel,
                face,
                radiance: if restart {
                    EnergySpectrum::ZERO
                } else {
                    probe.radiance[face]
                },
                second_moment: if restart {
                    EnergySpectrum::ZERO
                } else {
                    probe.second_moment[face]
                },
                direct: probe.direct[face],
                custom_direct: probe.custom_direct[face],
                samples: if restart {
                    0
                } else {
                    probe.samples[face]
                        .min(if stale { stale_history_limit } else { u32::MAX })
                        .min(source_history_limit)
                },
                sequence: probe.sequence[face],
                updated_at: if restart {
                    None
                } else {
                    probe.face_updated_at[face]
                },
            });
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
            #[cfg(debug_assertions)]
            retention_audit: None,
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
                face.samples = probe.samples[face.face];
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
    pub fn write(&self, probe: &mut SurfaceProbe) {
        let face = self.face;
        probe.radiance[face] = self.radiance;
        probe.second_moment[face] = self.second_moment;
        probe.direct[face] = self.direct;
        probe.custom_direct[face] = self.custom_direct;
        probe.samples[face] = self.samples;
        probe.sequence[face] = self.sequence;
        probe.face_updated_at[face] = self.updated_at;
        probe.updated_at = self.updated_at;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deferred_caps_run_once_and_preserve_stronger_source_caps() {
        use super::super::{
            geometry::{Bounds, GeometryChange, PageChanges},
            retention::{Origin, SpatialHistory},
        };
        let mut history = SpatialHistory::default();
        history.record(
            GeometryChange {
                nearby: bevy::platform::collections::HashMap::from_iter([(
                    IVec3::ZERO,
                    PageChanges::new(IVec3::ZERO, vec![Bounds::point(IVec3::new(192, 0, 0))]),
                )]),
                ..default()
            },
            Origin::Geometry,
        );
        let mut probe = SurfaceProbe::new(IVec3::ZERO, 4);
        probe.samples = [4096; 6];
        for (source_cap, expected) in [(u32::MAX, 316), (64, 64)] {
            let mut work = TraceWork::default();
            work.push(&probe, 4, 1, 1, false, u32::MAX, source_cap);
            work.defer_retention(0, history.snapshot(probe.voxel));
            assert_eq!(work.faces[0].samples, 4096.min(source_cap));
            work.apply_retention();
            assert_eq!(work.faces[0].samples, expected);
            work.faces[0].samples += 16;
            work.apply_retention();
            assert_eq!(
                work.faces[0].samples,
                expected + 16,
                "resolved work cannot cap fresh samples again"
            );
            assert!(work.pending_retention.is_empty());
        }
    }

    #[test]
    fn spatial_evidence_caps_unselected_faces_and_does_not_consume_later_events() {
        let mut probe = SurfaceProbe::new(IVec3::ZERO, 63);
        probe.samples = [4096; 6];
        probe.radiance = [EnergySpectrum::ONE; 6];
        probe.second_moment = [EnergySpectrum::ONE * 2.0; 6];
        probe.sequence = [9000; 6];
        let mut work = TraceWork::default();
        work.push(&probe, 1, 1, 0, false, 256, u32::MAX);
        work.probes[0].retention = Some((7, 256));
        work.faces[0].samples += 16;
        work.faces[0].sequence += 16;
        work.probes[0].publish(&work.faces, &mut probe, Instant::now(), &default());
        assert_eq!(probe.samples, [272, 256, 256, 256, 256, 256]);
        assert_eq!(probe.sequence, [9016, 9000, 9000, 9000, 9000, 9000]);
        assert_eq!(
            probe.retention_revision, 7,
            "publication must retain selection's stamp, not a later worker-result stamp"
        );
        assert_eq!(probe.fresh_mask, 1);
        assert_eq!(probe.radiance, [EnergySpectrum::ONE; 6]);
        assert_eq!(probe.second_moment, [EnergySpectrum::ONE * 2.0; 6]);
    }

    #[test]
    fn compact_publication_matches_full_probe_reference_for_partial_faces_and_resets() {
        let now = Instant::now();
        let settings = super::super::VoxelGiSettings::default();
        for mask in 1..64 {
            for mode in 0..3 {
                let mut previous = SurfaceProbe::new(IVec3::new(8, 2, 6), 63);
                previous.samples = [128; 6];
                previous.sequence = [1050; 6];
                previous.radiance = [EnergySpectrum::ONE; 6];
                previous.second_moment = [EnergySpectrum::ONE; 6];
                previous.stable_batches = [4; 6];
                previous.face_updated_at = [Some(now - std::time::Duration::from_millis(40)); 6];
                previous.updated_at = previous.face_updated_at[0];
                previous.order_index = 123;
                for display in &mut previous.display {
                    display.indirect = Vec3::splat(0.7);
                    display.confidence = 0.75;
                    display.sky = [27; 4];
                }
                let revision = u64::from(mode != 0);
                let restart = mode == 2;
                let mut work = TraceWork::default();
                work.push(&previous, mask, revision, revision, restart, 64, u32::MAX);
                let mut reference = previous.clone();
                reference.trace_mask = mask;
                if revision != 0 {
                    reference.transport_revision = revision;
                    reference.fresh_mask = 0;
                    reference.stable_batches = [0; 6];
                    for face in 0..6 {
                        if mask & (1 << face) != 0 {
                            reference.samples[face] = 64;
                        }
                    }
                }
                if restart {
                    reference.restart_measurements();
                }
                reference.source_revision = revision;
                for face in &mut work.faces {
                    let index = face.face;
                    assert_eq!(face.samples, reference.samples[index]);
                    assert_eq!(face.radiance, reference.radiance[index]);
                    for sample in 0..4 {
                        super::super::tracer::accumulate_sample(
                            &mut reference,
                            index,
                            EnergySpectrum::new(Vec3::splat(sample as f32 * 0.2), 0.3, 0.6),
                        );
                    }
                    face.radiance = reference.radiance[index];
                    face.second_moment = reference.second_moment[index];
                    face.samples = reference.samples[index];
                    face.sequence = reference.sequence[index];
                    face.updated_at = Some(now);
                    reference.face_updated_at[index] = Some(now);
                }
                reference.updated_at = Some(now);
                reference.resolve_display(&previous, now, settings.display_response_seconds());
                super::super::scheduling::record_batch(&mut reference, &previous, now, &settings);
                // A moved topology index and changed live display are not part
                // of worker ownership. Publication must retain the live index.
                let mut actual = previous;
                actual.order_index = 999;
                work.probes[0].publish(&work.faces, &mut actual, now, &settings);
                assert_eq!(actual.order_index, 999);
                assert_eq!(actual.samples, reference.samples);
                assert_eq!(actual.sequence, reference.sequence);
                assert_eq!(actual.radiance, reference.radiance);
                assert_eq!(actual.second_moment, reference.second_moment);
                assert_eq!(actual.direct, reference.direct);
                assert_eq!(actual.custom_direct, reference.custom_direct);
                assert_eq!(actual.stable_batches, reference.stable_batches);
                assert_eq!(actual.fresh_mask, reference.fresh_mask);
                assert_eq!(actual.face_updated_at, reference.face_updated_at);
                assert_eq!(actual.updated_at, reference.updated_at);
                for face in 0..6 {
                    assert_eq!(
                        actual.display[face].indirect,
                        reference.display[face].indirect
                    );
                    assert_eq!(
                        actual.display[face].confidence,
                        reference.display[face].confidence
                    );
                    assert_eq!(actual.display[face].sky, reference.display[face].sky);
                }
            }
        }
    }

    #[test]
    fn long_refinement_history_obeys_geometry_and_source_caps() {
        let mut probe = SurfaceProbe::new(IVec3::ZERO, 4);
        probe.samples[2] = super::super::tracer::HISTORY_LIMIT;
        probe.sequence[2] = 100_000;
        probe.radiance[2] = EnergySpectrum::ONE;
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
                assert_eq!(work.faces[0].radiance, EnergySpectrum::ZERO);
            }
        }
    }

    #[test]
    fn retaining_history_still_publishes_freshness_and_wakes_stable_faces() {
        let mut probe = SurfaceProbe::new(IVec3::ZERO, 5);
        probe.samples = [4096; 6];
        probe.sequence = [8000; 6];
        probe.radiance = [EnergySpectrum::ONE; 6];
        probe.stable_batches = [4; 6];
        let mut work = TraceWork::default();
        work.push(&probe, 4, 1, 1, false, u32::MAX, u32::MAX);
        assert_eq!(work.faces[0].samples, 4096);
        assert_eq!(work.faces[0].sequence, 8000);
        work.probes[0].publish(&work.faces, &mut probe, Instant::now(), &Default::default());
        assert_eq!(probe.transport_revision, 1);
        assert_eq!(probe.source_revision, 1);
        assert_eq!(probe.fresh_mask, 4);
        assert_eq!(probe.stable_batches[0], 0);
        assert_eq!(probe.samples[0], 4096);
        assert_eq!(probe.samples[2], 4096);
    }

    #[test]
    fn single_face_work_is_less_than_half_a_full_probe_and_sky_copies_no_history() {
        let bytes = size_of::<ProbeWork>() + size_of::<FaceWork>();
        assert!(bytes < size_of::<SurfaceProbe>() / 2);
        let mut probe = SurfaceProbe::new(IVec3::ZERO, 63);
        probe.samples = [128; 6];
        probe.sequence = [901; 6];
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
