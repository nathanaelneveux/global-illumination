//! Bounded adaptive candidate selection; the caller separately reserves a fair
//! all-loaded-world sweep and drains edits/simulation requests first.
use bevy::platform::collections::HashSet;
use bevy::prelude::*;
use std::time::Instant;

use super::{
    field::VoxelEnergyField, settings::VoxelGiSettings, statistics::relative_error_with_floor,
    world::SurfaceProbe,
};

const STABLE_BATCHES: u8 = 4;
// Dark indirect energy is often below the presentation filter's 0.02 floor.
// That floor permitted order-one relative errors before switching to slow
// audits. Refinement needs a tighter, independent tolerance.
const REFINEMENT_ENERGY_FLOOR: f32 = 0.002;

fn required_samples(probe: &SurfaceProbe, face: usize, settings: &VoxelGiSettings) -> u32 {
    let minimum = settings.stable_samples();
    if probe.radiance[face].visible.max_element() < 0.02 {
        // Zero observed variance after a few missed bright paths is not enough
        // evidence in a dark room. This is still a heuristic, not a guarantee
        // that every small light/portal has been sampled.
        minimum.max(256)
    } else {
        minimum
    }
}

fn refinement_error(probe: &SurfaceProbe, face: usize) -> f32 {
    relative_error_with_floor(
        probe.radiance[face],
        probe.second_moment[face],
        probe.samples[face],
        REFINEMENT_ENERGY_FLOOR,
    )
}

/// Faces whose latest measurements passed the adaptive stability checks in
/// this scene revision. An audit becoming due does not itself revoke stability.
/// This is not an idle-work mask or a certified convergence estimate.
pub(crate) fn settled_face_mask(probe: &SurfaceProbe, revision: u64) -> u8 {
    if probe.transport_revision != revision {
        return 0;
    }
    let mut mask = 0;
    for face in 0..6 {
        if probe.face_mask & probe.fresh_mask & (1 << face) != 0
            && probe.stable_batches[face] >= STABLE_BATCHES
        {
            mask |= 1 << face;
        }
    }
    mask
}

/// Candidate-search capacity is not the number of rays emitted. A partially
/// settled window must not collapse the all-world sweep to a handful of sites.
pub(crate) fn next_budget(
    current: usize,
    dispatched: usize,
    elapsed_ms: f32,
    target_ms: f32,
    limit: usize,
) -> usize {
    let ratio = (target_ms / elapsed_ms.max(0.1)).clamp(0.5, 1.25);
    let desired = (dispatched as f32 * ratio) as usize;
    let next = if elapsed_ms <= target_ms {
        current.max(desired)
    } else {
        desired
    };
    next.clamp(64, limit.max(64))
}

pub(crate) fn due_faces(
    probe: &SurfaceProbe,
    revision: u64,
    now: Instant,
    settings: &VoxelGiSettings,
    force: bool,
) -> u8 {
    if !settings.adaptive_updates {
        return probe.face_mask;
    }
    let mut mask = 0;
    for face in 0..6 {
        if probe.face_mask & (1 << face) == 0 {
            continue;
        }
        if force
            || probe.transport_revision != revision
            || probe.fresh_mask & (1 << face) == 0
            || probe.samples[face] < required_samples(probe, face, settings)
            || probe.stable_batches[face] < STABLE_BATCHES
            || probe.face_updated_at[face]
                .is_none_or(|last| now.saturating_duration_since(last) >= settings.audit_interval())
        {
            mask |= 1 << face;
        }
    }
    mask
}

#[cfg(test)]
pub(crate) fn record_batch(
    probe: &mut SurfaceProbe,
    old: &SurfaceProbe,
    now: Instant,
    settings: &VoxelGiSettings,
) {
    for face in 0..6 {
        if probe.face_mask & probe.trace_mask & (1 << face) == 0 {
            continue;
        }
        record_face_batch(
            probe,
            face,
            old.radiance[face],
            old.stable_batches[face],
            probe.transport_revision == old.transport_revision,
            now,
            settings,
        );
    }
}

pub(crate) fn record_face_batch(
    probe: &mut SurfaceProbe,
    face: usize,
    before: super::EnergySpectrum,
    previous_stable_batches: u8,
    same_revision: bool,
    now: Instant,
    settings: &VoxelGiSettings,
) {
    let error = refinement_error(probe, face);
    let after = probe.radiance[face];
    let a = [
        before.visible.x,
        before.visible.y,
        before.visible.z,
        before.infrared,
        before.ionizing,
    ];
    let b = [
        after.visible.x,
        after.visible.y,
        after.visible.z,
        after.infrared,
        after.ionizing,
    ];
    let drift = a
        .into_iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs() / (REFINEMENT_ENERGY_FLOOR + a.abs().max(b.abs())))
        .fold(0.0, f32::max);
    let display_settled = (probe.display[face].indirect - after.visible)
        .abs()
        .max_element()
        <= REFINEMENT_ENERGY_FLOOR * 0.5
            + after.visible.max_element() * settings.stable_error() * 0.5;
    probe.stable_batches[face] = if same_revision
        && display_settled
        && probe.samples[face] >= required_samples(probe, face, settings)
        && error <= settings.stable_error()
        && drift <= settings.stable_error() * 0.5
    {
        previous_stable_batches
            .saturating_add(1)
            .min(STABLE_BATCHES)
    } else {
        0
    };
    probe.face_updated_at[face] = Some(now);
    probe.fresh_mask |= 1 << face;
}

fn priority(probe: &SurfaceProbe, source_revision: u64, now: Instant) -> f32 {
    let age = probe
        .face_updated_at
        .iter()
        .flatten()
        .min()
        .copied()
        .or(probe.updated_at)
        .map_or(10.0, |time| {
            now.saturating_duration_since(time).as_secs_f32()
        })
        .min(10.0);
    let mut error: f32 = 0.0;
    for face in 0..6 {
        if probe.face_mask & (1 << face) != 0 {
            let value = if probe.samples[face] == 0 {
                8.0
            } else {
                refinement_error(probe, face)
            };
            error = error.max(value);
        }
    }
    error + age * 0.1 + f32::from(probe.source_revision != source_revision)
}

/// Retained bounded ranking storage. Ranking order and sweep fairness do not
/// depend on hash iteration; the set only deduplicates the ordered windows.
#[derive(Default)]
pub(crate) struct RankedCandidates {
    seen: HashSet<IVec3>,
    ranked: Vec<(f32, IVec3)>,
}

impl RankedCandidates {
    pub fn clear(&mut self) {
        self.seen.clear();
        self.ranked.clear();
    }

    pub fn pop(&mut self) -> Option<IVec3> {
        self.ranked.pop().map(|(_, voxel)| voxel)
    }

    pub fn refresh(
        &mut self,
        field: &VoxelEnergyField,
        nearby: &[IVec3],
        cursor: &mut usize,
        now: Instant,
        budget: usize,
    ) {
        // Never scan/sort the whole world on a frame. Rotating windows eventually
        // consider every probe, including uncertain non-visible transport bands.
        let window = budget.clamp(64, 1024);
        self.clear();
        let seen = &mut self.seen;
        let candidates = &mut self.ranked;
        for (positions, bonus) in [(&field.order[..], 0.0), (nearby, 0.25)] {
            for index in 0..window.min(positions.len()) {
                let voxel = positions[cursor.wrapping_add(index) % positions.len()];
                if seen.insert(voxel)
                    && let Some(probe) = field.probes.get(&voxel)
                {
                    candidates.push((priority(probe, field.source_revision, now) + bonus, voxel));
                }
            }
        }
        *cursor = cursor.wrapping_add(window);
        candidates.sort_unstable_by(|(a, va), (b, vb)| {
            a.total_cmp(b)
                .then_with(|| (va.x, va.y, va.z).cmp(&(vb.x, vb.y, vb.z)))
        });
    }
}

#[cfg(test)]
fn ranked_candidates(
    field: &VoxelEnergyField,
    nearby: &[IVec3],
    cursor: &mut usize,
    now: Instant,
    budget: usize,
) -> Vec<IVec3> {
    let mut scratch = RankedCandidates::default();
    scratch.refresh(field, nearby, cursor, now, budget);
    scratch.ranked.into_iter().map(|(_, voxel)| voxel).collect()
}

#[cfg(test)]
mod tests {
    use super::super::EnergySpectrum;
    use super::*;

    #[test]
    fn sparse_due_windows_cannot_collapse_search_capacity() {
        let mut budget = 8192;
        for _ in 0..600 {
            budget = next_budget(budget, 64, 0.1, 4.0, 8192);
        }
        assert_eq!(budget, 8192);
        assert_eq!(next_budget(8192, 8192, 8.0, 4.0, 8192), 4096);
        assert_eq!(next_budget(256, 256, 1.0, 4.0, 8192), 320);
    }

    fn stable_probe(now: Instant) -> SurfaceProbe {
        let mut probe = SurfaceProbe::new(IVec3::new(10_000, 0, 0), 5);
        probe.samples = [128; 6];
        probe.radiance = [EnergySpectrum::ONE; 6];
        probe.second_moment = [EnergySpectrum::ONE; 6];
        probe.stable_batches = [STABLE_BATCHES; 6];
        probe.face_updated_at = [Some(now); 6];
        probe.updated_at = Some(now);
        for display in &mut probe.display {
            display.indirect = Vec3::ONE;
        }
        probe
    }

    #[test]
    fn settled_faces_rest_but_audits_edits_and_simulation_requests_wake_them() {
        let now = Instant::now();
        let settings = VoxelGiSettings::default();
        let mut probe = stable_probe(now);
        assert_eq!(due_faces(&probe, 0, now, &settings, false), 0);
        let continuous = VoxelGiSettings {
            adaptive_updates: false,
            ..default()
        };
        assert_eq!(due_faces(&probe, 0, now, &continuous, false), 5);
        assert_eq!(
            due_faces(&probe, 0, now + settings.audit_interval(), &settings, false),
            5
        );
        assert_eq!(
            due_faces(&probe, 1, now, &settings, false),
            5,
            "revision invalidation must work far beyond the edit radius"
        );
        assert_eq!(due_faces(&probe, 0, now, &settings, true), 5);
        probe.stable_batches[2] = 0;
        assert_eq!(
            due_faces(&probe, 0, now, &settings, false),
            4,
            "do not retrace a settled face just because another face is noisy"
        );
    }

    #[test]
    fn settled_mask_counts_only_exposed_fresh_current_faces_and_survives_audit_age() {
        let now = Instant::now();
        let mut probe = stable_probe(now);
        assert_eq!(settled_face_mask(&probe, 0), 5);
        assert_eq!(settled_face_mask(&probe, 1), 0);
        probe.face_updated_at = [Some(now - std::time::Duration::from_secs(10)); 6];
        assert_eq!(
            settled_face_mask(&probe, 0),
            5,
            "audit eligibility is not instability"
        );
        probe.fresh_mask = 1;
        assert_eq!(settled_face_mask(&probe, 0), 1);
        probe.stable_batches[0] = STABLE_BATCHES - 1;
        assert_eq!(settled_face_mask(&probe, 0), 0);
        probe.fresh_mask = 63;
        probe.samples = [256; 6];
        probe.stable_batches = [0; 6];
        assert_eq!(
            settled_face_mask(&probe, 0),
            0,
            "history count alone is not stability"
        );
    }

    #[test]
    fn stability_requires_sample_evidence_all_bands_and_settled_presentation() {
        let now = Instant::now();
        let settings = VoxelGiSettings::default();
        let old = stable_probe(now);
        let mut probe = old.clone();
        probe.trace_mask = 4;
        probe.second_moment[2].infrared = 1000.0;
        record_batch(&mut probe, &old, now, &settings);
        assert_eq!(probe.stable_batches[2], 0);
        assert_eq!(probe.stable_batches[0], STABLE_BATCHES);
        probe = old.clone();
        probe.display[2].indirect = Vec3::ZERO;
        record_batch(&mut probe, &old, now, &settings);
        assert_eq!(
            probe.stable_batches[2], 0,
            "do not freeze visibly unfinished presentation"
        );
        probe = old.clone();
        probe.samples[2] = 4;
        probe.radiance[2] = EnergySpectrum::ZERO;
        probe.second_moment[2] = EnergySpectrum::ZERO;
        record_batch(&mut probe, &old, now, &settings);
        assert_eq!(
            probe.stable_batches[2], 0,
            "a few dark rays are not convergence"
        );
        probe = old.clone();
        probe.radiance[2].ionizing = 50.0;
        probe.second_moment[2].ionizing = 2500.0;
        record_batch(&mut probe, &old, now, &settings);
        assert_eq!(
            probe.stable_batches[2], 0,
            "energy drift must wake even with zero variance"
        );
    }

    #[test]
    fn dark_receivers_need_evidence_before_sparse_audits() {
        let now = Instant::now();
        let settings = VoxelGiSettings::default();
        let mut probe = SurfaceProbe::new(IVec3::ZERO, 4);
        probe.trace_mask = 4;
        probe.samples[2] = 128;
        probe.face_updated_at[2] = Some(now);
        probe.stable_batches[2] = STABLE_BATCHES;
        assert_eq!(
            due_faces(&probe, 0, now, &settings, false),
            4,
            "an early run of black samples must not enter two-second audits"
        );
        let old = probe.clone();
        record_batch(&mut probe, &old, now, &settings);
        assert_eq!(probe.stable_batches[2], 0);

        probe.samples[2] = 256;
        probe.radiance[2] = super::super::EnergySpectrum::new(Vec3::splat(0.002), 0.0, 0.0);
        probe.second_moment[2] = super::super::EnergySpectrum::new(Vec3::splat(0.000128), 0.0, 0.0);
        probe.display[2].indirect = probe.radiance[2].visible;
        assert!(
            super::super::statistics::relative_error(
                probe.radiance[2],
                probe.second_moment[2],
                256
            ) < settings.stable_error(),
            "this dark noisy fixture passed the old presentation-floor test"
        );
        let old = probe.clone();
        record_batch(&mut probe, &old, now, &settings);
        assert_eq!(probe.stable_batches[2], 0);
        probe.samples[2] = 4096;
        for _ in 0..STABLE_BATCHES {
            let old = probe.clone();
            record_batch(&mut probe, &old, now, &settings);
        }
        assert_eq!(
            due_faces(&probe, 0, now, &settings, false),
            0,
            "sufficiently measured dark faces must still be able to rest"
        );
    }

    #[test]
    fn unchanged_workload_has_sparse_audits_instead_of_continuous_batches() {
        let start = Instant::now();
        let settings = VoxelGiSettings::default();
        let mut probe = stable_probe(start);
        let mut face_batches = 0;
        for frame in 0..600 {
            let now = start + std::time::Duration::from_secs_f64(frame as f64 / 60.0);
            let mask = due_faces(&probe, 0, now, &settings, false);
            if mask == 0 {
                continue;
            }
            face_batches += mask.count_ones();
            let old = probe.clone();
            probe.trace_mask = mask;
            record_batch(&mut probe, &old, now, &settings);
        }
        assert_eq!(
            face_batches, 8,
            "two settled faces need four audits each over ten seconds, not 1200 face batches"
        );
    }

    #[test]
    fn uncertainty_in_invisible_bands_is_not_hidden_by_bright_rgb() {
        let now = Instant::now();
        let mut stable = SurfaceProbe::new(IVec3::ZERO, 4);
        stable.samples[2] = 64;
        stable.updated_at = Some(now);
        stable.radiance[2] = EnergySpectrum::new(Vec3::splat(100.0), 1.0, 1.0);
        stable.second_moment[2] = stable.radiance[2] * stable.radiance[2];
        let mut uncertain = stable.clone();
        uncertain.second_moment[2].infrared = 100.0;
        assert!(priority(&uncertain, 0, now) > priority(&stable, 0, now) + 1.0);
        uncertain.second_moment[2].infrared = 1.0;
        uncertain.second_moment[2].ionizing = 100.0;
        assert!(priority(&uncertain, 0, now) > priority(&stable, 0, now) + 1.0);
        assert!(priority(&stable, 1, now) > priority(&stable, 0, now));
    }

    #[test]
    fn candidate_search_is_bounded_and_rotates_across_offscreen_terrain() {
        let mut field = VoxelEnergyField::default();
        for x in 0..3000 {
            let voxel = IVec3::new(x, 0, 0);
            field.order.push(voxel);
            field.probes.insert(voxel, SurfaceProbe::new(voxel, 4));
        }
        let mut cursor = 0;
        let mut visited = HashSet::new();
        for _ in 0..3 {
            let candidates = ranked_candidates(&field, &[], &mut cursor, Instant::now(), 8192);
            assert!(candidates.len() <= 1024);
            visited.extend(candidates);
        }
        assert_eq!(visited.len(), 3000);
    }

    #[test]
    fn ranking_reuses_storage_and_keeps_deterministic_order_with_overlapping_windows() {
        let now = Instant::now();
        let mut field = VoxelEnergyField::default();
        for x in 0..256 {
            let voxel = IVec3::new(x, 0, 0);
            field.order.push(voxel);
            field.probes.insert(voxel, SurfaceProbe::new(voxel, 4));
        }
        // First half overlaps the global window; the near bonus belongs only
        // to sites not already seen there, as in the original ordered search.
        let nearby: Vec<_> = (32..96).map(|x| IVec3::new(x, 0, 0)).collect();
        let mut scratch = RankedCandidates::default();
        let mut cursor = 0;
        scratch.refresh(&field, &nearby, &mut cursor, now, 64);
        let ranked_ptr = scratch.ranked.as_ptr();
        let ranked_capacity = scratch.ranked.capacity();
        let seen_capacity = scratch.seen.capacity();
        let expected: Vec<_> = (64..96)
            .rev()
            .chain((0..64).rev())
            .map(|x| IVec3::new(x, 0, 0))
            .collect();
        for _ in 0..4 {
            assert_eq!(cursor, 64);
            let actual: Vec<_> = std::iter::from_fn(|| scratch.pop()).collect();
            assert_eq!(actual, expected);
            cursor = 0;
            scratch.refresh(&field, &nearby, &mut cursor, now, 64);
            assert_eq!(scratch.ranked.as_ptr(), ranked_ptr);
            assert_eq!(scratch.ranked.capacity(), ranked_capacity);
            assert_eq!(scratch.seen.capacity(), seen_capacity);
        }
        scratch.clear();
        assert!(scratch.pop().is_none());
        assert!(scratch.seen.is_empty());
    }
}
