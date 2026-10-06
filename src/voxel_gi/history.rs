//! Presentation history is deliberately separate from transport measurements.
//! Borrowed values never affect sample counts, moments, or simulation queries.
use bevy::prelude::*;
use std::time::Instant;

use super::spectrum::EnergySpectrum;
use super::world::{AIR, FACE_OFFSETS, SurfaceProbe, TransportScene};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FaceDisplay {
    pub indirect: Vec3,
    /// Packed RGBE corners, with zero reserved for unknown visibility.
    pub sky: [u32; 4],
    /// Reconstruction weight, not a statistical confidence interval.
    pub confidence: f32,
}

/// Prefer equally oriented neighbors. When digging exposes an entirely new
/// orientation, fall back to measured faces bordering the same local air
/// pocket. This is a low-confidence display prior, never transport evidence.
/// Air-pocket connections cross at most two known-empty cells in the NEW scene.
/// Placement may also inherit the equally oriented face it just covered: that
/// pre-edit exposed face proves the receiver was air, not an intervening wall.
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
    let mut sky = [0; 4];
    let mut weight = 0.0;
    for offset in FACE_OFFSETS {
        let donor_voxel = voxel + offset;
        let Some(donor) = lookup(donor_voxel) else {
            continue;
        };
        let covered_by_receiver = offset == -normal;
        if !covered_by_receiver && scene.material_at(donor_voxel + normal) != Some(AIR) {
            continue;
        }
        let Some((display, w)) = measured_display(donor, face, source_revision, now) else {
            continue;
        };
        indirect += display.indirect * w;
        // Direct visibility may be reused only at the SAME geometric corner.
        // Face-center averaging would erase sub-voxel contact shadows.
        for (corner, value) in sky.iter_mut().enumerate() {
            for donor_corner in 0..4 {
                if super::sky::corner_position(voxel, face, corner)
                    == super::sky::corner_position(donor_voxel, face, donor_corner)
                    && display.sky[donor_corner] != 0
                {
                    *value = display.sky[donor_corner];
                }
            }
        }
        weight += w;
    }
    if weight == 0.0 {
        let air = voxel + normal;
        for offset in std::iter::once(IVec3::ZERO).chain(FACE_OFFSETS) {
            let donor_air = air + offset;
            if scene.material_at(donor_air) != Some(AIR) {
                continue;
            }
            for (donor_face, donor_normal) in FACE_OFFSETS.into_iter().enumerate() {
                let Some(donor) = lookup(donor_air - donor_normal) else {
                    continue;
                };
                let Some((display, w)) = measured_display(donor, donor_face, source_revision, now)
                else {
                    continue;
                };
                // A face beside the receiver gets precedence over one across
                // the next air cell. Never rotate/reuse direct-sun visibility:
                // that is directional, unlike this approximate diffuse prior.
                let w = w * if offset == IVec3::ZERO { 1.0 } else { 0.5 };
                indirect += display.indirect * w;
                weight += w;
            }
        }
    }
    if weight == 0.0 {
        return FaceDisplay::default();
    }
    FaceDisplay {
        indirect: indirect / weight,
        sky,
        confidence: 0.125,
    }
}

fn measured_display(
    donor: &SurfaceProbe,
    face: usize,
    source_revision: u64,
    now: Instant,
) -> Option<(FaceDisplay, f32)> {
    if donor.face_mask & (1 << face) == 0
        || donor.samples[face] == 0
        || donor.source_revision != source_revision
    {
        return None;
    }
    let updated = donor.face_updated_at[face].or(donor.updated_at)?;
    let age = now.saturating_duration_since(updated).as_secs_f32();
    let display = donor.display[face];
    // Settled faces are intentionally audited sparsely. An arbitrary two-second
    // cutoff made an otherwise valid donor disappear between audit batches.
    // Age reduces preference continuously instead of producing black holes.
    let weight = display.confidence / (1.0 + age / 2.0);
    (weight > 0.0).then_some((display, weight))
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
    #[cfg(test)]
    pub(crate) fn resolve_display(&mut self, previous: &Self, now: Instant, response_seconds: f32) {
        let incoming_sky = self.display.map(|face| face.sky);
        self.display = previous.display;
        for (face, sky) in incoming_sky.into_iter().enumerate() {
            self.display[face].sky = sky;
            if self.face_mask & self.trace_mask & (1 << face) == 0 || self.samples[face] == 0 {
                continue;
            }
            self.resolve_face(
                face,
                previous.face_updated_at[face].or(previous.display_updated_at),
                now,
                response_seconds,
            );
        }
        self.display_updated_at = Some(now);
    }

    /// Resolve one accepted face in place; its display value is still the
    /// authoritative pre-batch history. No six-face worker snapshot is needed.
    pub(crate) fn resolve_face(
        &mut self,
        face: usize,
        previous_time: Option<Instant>,
        now: Instant,
        response_seconds: f32,
    ) {
        let elapsed = previous_time.map_or(1.0 / 60.0, |time| {
            now.saturating_duration_since(time).as_secs_f32()
        });
        let time_alpha = 1.0 - (-elapsed / response_seconds.max(0.001)).exp();
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
        let evidence = samples as f32 / (samples as f32 + 32.0);
        let alpha = time_alpha.min(evidence / (1.0 + uncertainty));
        let display = &mut self.display[face];
        display.indirect = temporal_blend(
            display.indirect,
            self.radiance[face].visible,
            alpha.clamp(0.0, 1.0),
        );
        // Unknown remains unknown in storage. The corner renderer uses
        // available values conservatively without switching shadow modes.
        let confidence =
            (samples as f32 / (samples as f32 + 8.0) / (1.0 + uncertainty)).clamp(0.125, 1.0);
        display.confidence += (confidence - display.confidence) * alpha;
    }
}

/// Brightness-compressed temporal resolve (presentation only). A rare HDR path
/// must not flash an otherwise dark surface. Normalize the two weights instead
/// of clipping energy: persistent changes still reach their full intensity,
/// colors remain a convex RGB blend, and there is no hard brightness ceiling.
/// This is a robust, biased display filter, not a transport estimator.
fn temporal_blend(history: Vec3, measured: Vec3, alpha: f32) -> Vec3 {
    const DISPLAY_SCALE: f32 = 0.25;
    let old_weight = (1.0 - alpha) / (DISPLAY_SCALE + history.max_element());
    let new_weight = alpha / (DISPLAY_SCALE + measured.max_element());
    // Compression must not amplify a transient dark estimate either: limit
    // falling changes to the ordinary evidence/time-weighted EMA response.
    history.lerp(measured, alpha.min(new_weight / (old_weight + new_weight)))
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
    use std::time::Duration;

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
            sky: [super::super::sky::pack(Vec3::splat(energy)); 4],
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
        assert!(
            seed(&donor, &scene).confidence > 0.0,
            "a settled donor cannot disappear just because its audit is due"
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
        let weight = probe.display[2].confidence;
        let sky = probe.display[2].sky;
        probe.invalidate();
        assert_eq!(
            probe.display[2].confidence, weight,
            "editing cannot change retained corner reconstruction weights"
        );
        assert_eq!(probe.display[2].sky, sky);
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
    fn sky_seeding_reuses_only_identical_geometric_corners() {
        let now = Instant::now();
        let voxel = IVec3::splat(8);
        let neighbor = voxel + IVec3::X;
        let scene = scene_with(&[voxel, neighbor]);
        let mut donor = measured(neighbor, 2.0, now);
        donor.display[2].sky = [1, 0, 1, 0];
        let seed = seed_face(&scene, voxel, 2, 0, now, |v| {
            (v == neighbor).then_some(&donor)
        });
        assert_eq!(
            seed.sky,
            [0, 1, 0, 1],
            "shared shadows stay black, other corners stay unknown"
        );
    }

    #[test]
    fn placement_inherits_covered_face_without_copying_its_sun_or_measurements() {
        let now = Instant::now();
        let floor = IVec3::splat(8);
        let placed = floor + IVec3::Y;
        let mut donor = measured(floor, 2.0, now);
        donor.display[2].sky = [super::super::sky::pack(Vec3::ONE); 4];
        let scene = scene_with(&[floor, placed]);
        let prior = seed_face(&scene, placed, 2, 0, now, |v| {
            (v == floor).then_some(&donor)
        });
        assert_eq!(prior.indirect, Vec3::splat(2.0));
        assert_eq!(prior.confidence, 0.125);
        assert_eq!(
            prior.sky, [0; 4],
            "translated corners are not the same sun ray"
        );
        let mut receiver = SurfaceProbe::new(placed, 4);
        receiver.display[2] = prior;
        assert_eq!(receiver.samples[2], 0);
        assert_eq!(receiver.radiance[2], EnergySpectrum::ZERO);
        // Without an old exposed face, this would be borrowing through a wall.
        donor.face_mask = 0;
        assert_eq!(
            seed_face(&scene, placed, 2, 0, now, |v| {
                (v == floor).then_some(&donor)
            })
            .confidence,
            0.0
        );
    }

    #[test]
    fn digging_seeds_every_cavity_orientation_from_the_removed_surface() {
        let now = Instant::now();
        let removed = IVec3::splat(8);
        // The deleted floor block had only a measured top. The new bottom and
        // four side faces should all receive a prior, not just the new bottom.
        let donor = measured(removed, 2.0, now);
        let solids: Vec<_> = FACE_OFFSETS
            .into_iter()
            .filter(|normal| *normal != IVec3::NEG_Y)
            .map(|normal| removed - normal)
            .collect();
        let scene = scene_with(&solids);
        for (face, normal) in FACE_OFFSETS.into_iter().enumerate() {
            if normal == IVec3::NEG_Y {
                continue;
            }
            let seed = seed_face(&scene, removed - normal, face, 0, now, |v| {
                (v == removed).then_some(&donor)
            });
            assert_eq!(seed.indirect, Vec3::splat(2.0), "face {face}");
            assert_eq!(seed.confidence, 0.125);
            assert_eq!(seed.sky, [0; 4], "no invented sky in the cavity");
        }
    }

    #[test]
    fn air_pocket_fallback_rejects_blocked_and_synthetic_donors() {
        let now = Instant::now();
        let voxel = IVec3::splat(8);
        let donor_voxel = voxel + IVec3::Y + IVec3::X;
        let mut donor = measured(donor_voxel, 5.0, now);
        let scene = scene_with(&[voxel, donor_voxel]);
        let seed = |donor: &SurfaceProbe, scene: &TransportScene| {
            seed_face(scene, voxel, 0, 0, now, |v| {
                (v == donor_voxel).then_some(donor)
            })
        };
        // The donor's +Y air is diagonally separated, outside the bounded pocket.
        assert_eq!(seed(&donor, &scene).confidence, 0.0);
        donor.face_mask = 8;
        donor.samples[3] = donor.samples[2];
        donor.display[3] = donor.display[2];
        assert!(seed(&donor, &scene).confidence > 0.0);
        let blocked = scene_with(&[voxel, donor_voxel, voxel + IVec3::X]);
        assert_eq!(seed(&donor, &blocked).confidence, 0.0);
        donor.samples[3] = 0;
        assert_eq!(seed(&donor, &scene).confidence, 0.0);
    }

    #[test]
    fn first_batch_is_bounded_but_persistent_changes_settle_in_both_directions() {
        let start = Instant::now();
        for (initial, target) in [(0.0, 1.0), (1.0, 0.0)] {
            let mut previous = measured(IVec3::ZERO, initial, start);
            previous.invalidate();
            previous.display_updated_at = Some(start);
            for batch in 1..=16 {
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
                if batch == 8 {
                    assert!((value - target).abs() < 0.25);
                }
                if batch == 16 {
                    assert!((value - target).abs() < 0.01);
                }
                previous = next;
            }
        }
    }

    #[test]
    fn isolated_hdr_estimate_does_not_flash_retained_dark_history() {
        let now = Instant::now();
        let mut previous = measured(IVec3::ZERO, 0.0, now);
        previous.invalidate();
        let mut next = previous.clone();
        next.samples[2] = 4;
        next.radiance[2] = EnergySpectrum::new(Vec3::splat(1000.0), 50.0, 80.0);
        // Even a falsely reassuring zero within-batch variance must be bounded.
        next.second_moment[2] = next.radiance[2] * next.radiance[2];
        let measurement = next.radiance[2];
        next.resolve_display(&previous, now + Duration::from_secs(10), 0.075);
        assert!(next.display[2].indirect.max_element() < 0.04);
        assert_eq!(next.radiance[2], measurement);
        for _ in 0..80 {
            next.display[2].indirect =
                temporal_blend(next.display[2].indirect, Vec3::splat(1000.0), 0.5);
        }
        assert!(
            next.display[2]
                .indirect
                .abs_diff_eq(Vec3::splat(1000.0), 0.01),
            "no permanent clamp on real HDR emission"
        );
    }

    #[test]
    fn local_edits_keep_measured_history_but_report_it_stale() {
        let now = Instant::now();
        let voxel = IVec3::ZERO;
        let mut probe = measured(voxel, 1.0, now);
        probe.samples[2] = 128;
        probe.radiance[2] = EnergySpectrum::new(Vec3::ONE, 10.0, 20.0);
        let revision = probe.revision;
        probe.request_relight();
        assert_ne!(
            probe.revision, revision,
            "reject in-flight pre-edit results"
        );
        assert_eq!(probe.samples[2], 16);
        assert_eq!(probe.display[2].indirect, Vec3::ONE);
        let mut field = VoxelEnergyField::default();
        field.probes.insert(voxel, probe);
        let estimate = field.estimate(voxel, Vec3::Y).unwrap();
        assert!(!estimate.transport_current && !estimate.sources_current);
        assert_eq!(estimate.incident.infrared, 10.0);
        let probe = field.probes.get_mut(&voxel).unwrap();
        probe.update_face_mask(5);
        assert_eq!(probe.samples[2], 16, "unchanged face retains its history");
        assert_eq!(probe.samples[0], 0, "new face is genuinely unmeasured");
        assert_eq!(probe.radiance[0], EnergySpectrum::ZERO);
        assert!(field.estimate(voxel, Vec3::X).is_none());
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
