//! Bounded adaptive candidate selection; the caller separately reserves a fair
//! all-loaded-world sweep and drains edits/simulation requests first.
use bevy::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

use super::{field::VoxelEnergyField, history::relative_error, world::SurfaceProbe};

fn priority(probe: &SurfaceProbe, source_revision: u64, now: Instant) -> f32 {
    let age = probe
        .updated_at
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
                relative_error(
                    probe.radiance[face],
                    probe.second_moment[face],
                    probe.samples[face],
                )
            };
            error = error.max(value);
        }
    }
    error + age * 0.1 + f32::from(probe.source_revision != source_revision)
}

pub(crate) fn ranked_candidates(
    field: &VoxelEnergyField,
    nearby: &[IVec3],
    cursor: &mut usize,
    now: Instant,
    budget: usize,
) -> Vec<IVec3> {
    // Never scan/sort the whole world on a frame. Rotating windows eventually
    // consider every probe, including uncertain non-visible transport bands.
    let window = budget.clamp(64, 1024);
    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
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
    candidates.into_iter().map(|(_, voxel)| voxel).collect()
}

#[cfg(test)]
mod tests {
    use super::super::EnergySpectrum;
    use super::*;

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
}
