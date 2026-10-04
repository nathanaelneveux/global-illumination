//! Presentation history is deliberately separate from transport measurements.
//! Borrowed values never affect sample counts, moments, or simulation queries.
use bevy::prelude::*;
use std::time::{Duration, Instant};

use super::spectrum::EnergySpectrum;
use super::world::{AIR, FACE_OFFSETS, SurfaceProbe, TransportScene};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FaceDisplay {
    pub indirect: Vec3,
    pub sky: Vec3,
    /// Reconstruction weight, not a statistical confidence interval.
    pub confidence: f32,
}

/// Use only immediately adjacent, equally oriented old faces whose exterior
/// air cells remain connected in the NEW scene. A one-axis segment between
/// these face centers crosses only these two air cells, so this exact local
/// visibility test cannot tunnel through a wall or an unloaded region.
pub(crate) fn seed_face<'a>(
    scene: &TransportScene,
    voxel: IVec3,
    face: usize,
    source_revision: u64,
    now: Instant,
    lookup: impl Fn(IVec3) -> Option<&'a SurfaceProbe>,
) -> FaceDisplay {
    let normal = FACE_OFFSETS[face];
    if scene.material_at(voxel + normal) != Some(AIR) {
        return FaceDisplay::default();
    }
    let mut indirect = Vec3::ZERO;
    let mut sky = Vec3::ZERO;
    let mut weight = 0.0;
    for offset in FACE_OFFSETS {
        let donor_voxel = voxel + offset;
        let Some(donor) = lookup(donor_voxel) else {
            continue;
        };
        if donor.face_mask & (1 << face) == 0
            || donor.samples[face] == 0
            || donor.source_revision != source_revision
            || donor.updated_at.is_none_or(|updated| {
                now.saturating_duration_since(updated) > Duration::from_secs(2)
            })
            || scene.material_at(donor_voxel + normal) != Some(AIR)
        {
            continue;
        }
        let display = donor.display[face];
        let w = display.confidence;
        indirect += display.indirect * w;
        sky += display.sky * w;
        weight += w;
    }
    if weight == 0.0 {
        return FaceDisplay::default();
    }
    FaceDisplay {
        indirect: indirect / weight,
        sky: sky / weight,
        confidence: 0.125,
    }
}

/// A per-band heuristic for adaptive work allocation. This is NOT a confidence
/// bound: QMC samples and bounded temporal histories are correlated. Normalize
/// each band independently so bright RGB cannot hide uncertain infrared.
pub(crate) fn relative_error(mean: EnergySpectrum, second: EnergySpectrum, samples: u32) -> f32 {
    if samples < 2 {
        return 1.0;
    }
    let variance = (second - mean * mean).finite_or_zero();
    let means = [
        mean.visible.x,
        mean.visible.y,
        mean.visible.z,
        mean.infrared,
        mean.ionizing,
    ];
    let variances = [
        variance.visible.x,
        variance.visible.y,
        variance.visible.z,
        variance.infrared,
        variance.ionizing,
    ];
    means
        .into_iter()
        .zip(variances)
        .map(|(mean, variance)| (variance / samples as f32).sqrt() / (0.02 + mean.abs()))
        .fold(0.0, f32::max)
        .min(8.0)
}

impl SurfaceProbe {
    /// Run only when a validated worker result is merged. Use current display
    /// history, never a potentially old worker copy. Analytic sun visibility is
    /// deterministic, so it responds immediately rather than being denoised.
    pub(crate) fn resolve_display(&mut self, previous: &Self, now: Instant, response_seconds: f32) {
        self.display = previous.display;
        let elapsed = previous.display_updated_at.map_or(1.0 / 60.0, |time| {
            now.saturating_duration_since(time).as_secs_f32()
        });
        let time_alpha = 1.0 - (-elapsed / response_seconds.max(0.001)).exp();
        for face in 0..6 {
            if self.face_mask & (1 << face) == 0 || self.samples[face] == 0 {
                continue;
            }
            let samples = self.samples[face];
            let uncertainty = relative_error(
                EnergySpectrum::new(self.radiance[face].visible, 0.0, 0.0),
                EnergySpectrum::new(self.second_moment[face].visible, 0.0, 0.0),
                samples,
            );
            // A long scheduling gap is not evidence of convergence. Bound
            // influence by fresh sample evidence AND variance, even after the
            // eight-sample bootstrap milestone. Genuine changes catch up as
            // evidence arrives; there is no permanent brightness clamp.
            let evidence = (samples as f32 / 16.0).min(1.0);
            let alpha = time_alpha.min(evidence / (1.0 + uncertainty));
            let display = &mut self.display[face];
            display.indirect = display
                .indirect
                .lerp(self.radiance[face].visible, alpha.clamp(0.0, 1.0));
            display.sky = self.direct_sky[face];
            display.confidence =
                (samples as f32 / (samples as f32 + 8.0) / (1.0 + uncertainty)).clamp(0.125, 1.0);
        }
        self.display_updated_at = Some(now);
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        field::VoxelEnergyField,
        naadf::{linear_index, volume_len},
        tracer::{TransportSources, trace_probe_batch},
        world::{PAGE_SIZE, VoxelMaterialTransportTable},
    };
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn scene_with(solids: &[IVec3]) -> TransportScene {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        for voxel in solids {
            page[linear_index(voxel.as_uvec3(), PAGE_SIZE)] = 2;
        }
        TransportScene::with_page_delta(&TransportScene::default(), vec![(IVec3::ZERO, page)], &[])
    }

    fn measured(voxel: IVec3, energy: f32, now: Instant) -> SurfaceProbe {
        let mut probe = SurfaceProbe::new(voxel, 4);
        probe.samples[2] = 64;
        probe.radiance[2] = EnergySpectrum::ONE * energy;
        probe.second_moment[2] = probe.radiance[2] * probe.radiance[2];
        probe.updated_at = Some(now);
        probe.display_updated_at = Some(now);
        probe.display[2] = FaceDisplay {
            indirect: Vec3::splat(energy),
            sky: Vec3::splat(energy),
            confidence: 0.9,
        };
        probe
    }

    #[test]
    fn seeds_require_measured_current_same_facing_connected_neighbors() {
        let now = Instant::now();
        let voxel = IVec3::splat(8);
        let neighbor = voxel + IVec3::X;
        let mut donor = measured(neighbor, 2.0, now);
        let scene = scene_with(&[voxel, neighbor]);
        let seed = |donor: &SurfaceProbe, scene: &TransportScene| {
            seed_face(scene, voxel, 2, 0, now, |v| {
                (v == neighbor).then_some(donor)
            })
        };
        assert_eq!(seed(&donor, &scene).indirect, Vec3::splat(2.0));
        assert_eq!(seed(&donor, &scene).confidence, 0.125);
        let blocked = scene_with(&[voxel, neighbor, neighbor + IVec3::Y]);
        assert_eq!(
            seed(&donor, &blocked).confidence,
            0.0,
            "cannot borrow through a thin wall"
        );
        donor.samples[2] = 0;
        assert_eq!(
            seed(&donor, &scene).confidence,
            0.0,
            "synthetic seeds cannot become donors"
        );
        donor.samples[2] = 64;
        donor.face_mask = 8;
        assert_eq!(
            seed(&donor, &scene).confidence,
            0.0,
            "opposite-facing surfaces are incompatible"
        );
        donor.face_mask = 4;
        donor.source_revision = 1;
        assert_eq!(seed(&donor, &scene).confidence, 0.0);
        donor.source_revision = 0;
        donor.updated_at = Some(now - Duration::from_secs(3));
        assert_eq!(
            seed(&donor, &scene).confidence,
            0.0,
            "old measurements cannot seed a new face"
        );
        assert_eq!(
            seed(&donor, &TransportScene::default()).confidence,
            0.0,
            "unknown air is not connected air"
        );
    }

    #[test]
    fn presentation_history_never_becomes_simulation_energy() {
        let voxel = IVec3::splat(8);
        let now = Instant::now();
        let mut probe = measured(voxel, 4.0, now);
        let sequence = probe.sequence;
        probe.invalidate();
        assert_eq!(probe.display[2].indirect, Vec3::splat(4.0));
        assert_eq!(probe.samples, [0; 6]);
        assert_eq!(probe.radiance, [EnergySpectrum::ZERO; 6]);
        assert_eq!(probe.second_moment, [EnergySpectrum::ZERO; 6]);
        assert_eq!(probe.sequence, sequence);
        let mut field = VoxelEnergyField::default();
        field.probes.insert(voxel, probe);
        assert!(field.estimate(voxel, Vec3::Y).is_none());
        let probe = field.probes.get_mut(&voxel).unwrap();
        probe.samples[2] = 4;
        probe.radiance[2] = EnergySpectrum::new(Vec3::ZERO, 10.0, 20.0);
        probe.updated_at = Some(now);
        let estimate = field.estimate(voxel, Vec3::Y).unwrap();
        assert_eq!(
            estimate.incident,
            EnergySpectrum::new(Vec3::ZERO, 10.0, 20.0)
        );
        assert!(field.request_refresh(voxel));
        assert!(field.request_refresh(voxel));
        assert_eq!(field.requested.len(), 1);
        assert!(!field.request_refresh(IVec3::splat(1000)));
    }

    #[test]
    fn first_batch_is_bounded_but_persistent_changes_settle_in_both_directions() {
        let start = Instant::now();
        for (initial, target) in [(0.0, 1.0), (1.0, 0.0)] {
            let mut previous = measured(IVec3::ZERO, initial, start);
            previous.invalidate();
            previous.display_updated_at = Some(start);
            for batch in 1..=8 {
                let mut next = previous.clone();
                next.samples[2] = batch * 4;
                next.radiance[2] = EnergySpectrum::ONE * target;
                next.second_moment[2] = next.radiance[2] * next.radiance[2];
                next.resolve_display(
                    &previous,
                    start + Duration::from_millis(batch as u64 * 50),
                    0.075,
                );
                let value = next.display[2].indirect.x;
                assert!((0.0..=1.0).contains(&value));
                if batch == 1 {
                    assert!((value - initial).abs() <= 0.25);
                }
                if batch == 4 {
                    assert!((value - target).abs() < 0.11);
                }
                if batch == 8 {
                    assert!((value - target).abs() < 0.01);
                }
                previous = next;
            }
        }
    }

    #[test]
    fn long_scheduling_gap_does_not_give_a_noisy_bootstrap_full_weight() {
        let now = Instant::now();
        let previous = measured(IVec3::ZERO, 0.0, now);
        let mut next = previous.clone();
        next.samples[2] = 8;
        next.radiance[2] = EnergySpectrum::ONE;
        next.second_moment[2] = EnergySpectrum::ONE * 32.0;
        next.resolve_display(&previous, now + Duration::from_secs(10), 0.075);
        assert!(next.display[2].indirect.x < 0.2);
        assert_eq!(
            next.radiance[2],
            EnergySpectrum::ONE,
            "only presentation is filtered"
        );
    }

    #[test]
    fn digging_a_sealed_dark_tunnel_never_seeds_unoccluded_sky() {
        let now = Instant::now();
        let mut page = vec![2; volume_len(PAGE_SIZE)];
        for z in 8..12 {
            for y in 8..11 {
                for x in 8..12 {
                    page[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] = AIR;
                }
            }
        }
        let old_scene = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page.clone())],
            &[],
        ));
        let old_voxel = IVec3::new(9, 7, 9);
        let blank = SurfaceProbe::new(old_voxel, 4);
        let materials = Arc::new(VoxelMaterialTransportTable::default());
        let mut donor = trace_probe_batch(
            old_scene,
            TransportSources::default(),
            materials.clone(),
            vec![blank.clone()],
            64,
        )
        .remove(0);
        donor.resolve_display(&blank, now, 0.075);
        assert_eq!(donor.radiance[2], EnergySpectrum::ZERO);
        page[linear_index(old_voxel.as_uvec3(), PAGE_SIZE)] = AIR;
        let scene = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page)],
            &[],
        ));
        let voxel = old_voxel - IVec3::Y;
        let mut previous = SurfaceProbe::new(voxel, 4);
        previous.display[2] = seed_face(&scene, voxel, 2, 0, Instant::now(), |v| {
            (v == old_voxel).then_some(&donor)
        });
        assert!(
            previous.display[2].confidence > 0.0,
            "can inherit the removed front wall"
        );
        assert_eq!(previous.display[2].indirect, Vec3::ZERO);
        for batch in 1..=8 {
            let mut next = trace_probe_batch(
                scene.clone(),
                TransportSources::default(),
                materials.clone(),
                vec![previous.clone()],
                4,
            )
            .remove(0);
            next.resolve_display(&previous, now + Duration::from_millis(batch * 50), 0.075);
            assert_eq!(next.display[2].indirect, Vec3::ZERO);
            assert_eq!(next.radiance[2], EnergySpectrum::ZERO);
            previous = next;
        }
    }

    #[test]
    fn initialization_is_order_independent_and_does_not_spread_synthetic_history() {
        let now = Instant::now();
        let a = IVec3::splat(8);
        let b = a + IVec3::X;
        let c = b + IVec3::X;
        let donors = HashMap::from([(a, measured(a, 3.0, now))]);
        let scene = scene_with(&[a, b, c]);
        let seed = |voxel| seed_face(&scene, voxel, 2, 0, now, |v| donors.get(&v));
        let forward: HashMap<_, _> = [b, c].into_iter().map(|v| (v, seed(v).indirect)).collect();
        let reverse: HashMap<_, _> = [c, b].into_iter().map(|v| (v, seed(v).indirect)).collect();
        assert_eq!(forward, reverse);
        assert!(forward[&b].abs_diff_eq(Vec3::splat(3.0), 1e-6));
        assert_eq!(forward[&c], Vec3::ZERO);
    }
}
