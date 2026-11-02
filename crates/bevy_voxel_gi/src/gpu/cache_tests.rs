//! Cache-terminated transport must converge to the infinite-bounce solution
//! without gaining or losing energy. In a closed furnace whose walls share
//! emission E and reflectance r, every face receives exactly E / (1 - r).
//!
//! A short history reaches that fixed point quickly. The gameplay ceiling
//! (65,536) keeps early, under-bounced cache values in its running mean, so
//! high-reflectance bands approach the fixed point slowly from below.
use super::*;
use crate::research::channels;
use crate::world::{AIR, FACE_OFFSETS, SurfaceProbe};

pub(super) fn all_interior_faces(fixture: &crate::research::ResearchFixture) -> Vec<SurfaceProbe> {
    let mut probes = Vec::new();
    for z in 8..24 {
        for y in 8..24 {
            for x in 8..24 {
                let voxel = IVec3::new(x, y, z);
                if fixture.scene.material_at(voxel).is_none_or(|m| m == AIR) {
                    continue;
                }
                let mut mask = 0;
                for (face, offset) in FACE_OFFSETS.iter().enumerate() {
                    let neighbour = voxel + *offset;
                    let inside = neighbour.cmpge(IVec3::splat(9)).all()
                        && neighbour.cmple(IVec3::splat(22)).all();
                    if inside && fixture.scene.material_at(neighbour) == Some(AIR) {
                        mask |= 1 << face;
                    }
                }
                if mask != 0 {
                    let mut probe = SurfaceProbe::new(voxel, mask);
                    probe.material = 4;
                    probes.push(probe);
                }
            }
        }
    }
    probes
}

#[test]
#[ignore = "native resident furnace: cache feedback converges to E / (1 - r)"]
fn cached_furnace_converges_to_infinite_bounce_series() {
    let fixture = crate::research::furnace();
    let material = fixture.materials.get(4);
    let emitters = Arc::new(EmissionDistribution::build(
        &fixture.scene,
        &fixture.materials,
    ));
    let mut gpu = tests::native_tracer();
    gpu.enable_resident().unwrap();
    let probes = all_interior_faces(&fixture);
    let faces: u32 = probes.iter().map(|p| p.face_mask.count_ones()).sum();
    assert_eq!(faces, 6 * 14 * 14);
    let mut work = TraceWork::from_probes(&probes);
    work.history_limit = 256;
    let expected: Vec<f32> = channels(material.emission)
        .into_iter()
        .zip(channels(material.diffuse))
        .map(|(e, r)| e / (1.0 - r))
        .collect();
    let mut averages = [0.0f64; 5];
    for batch in 0..400 {
        gpu.trace_compact(
            fixture.scene.clone(),
            &fixture.sources,
            &fixture.materials,
            &mut work,
            16,
            &emitters,
        )
        .unwrap();
        if batch % 50 == 49 {
            let mut mean = vec![0.0f64; 5];
            for face in &work.faces {
                for (m, v) in mean.iter_mut().zip(channels(face.radiance)) {
                    *m += f64::from(v) / work.faces.len() as f64;
                }
            }
            eprintln!(
                "CACHED_FURNACE batch={} mean={mean:?} expected={expected:?}",
                batch + 1
            );
            if batch >= 300 {
                for (a, m) in averages.iter_mut().zip(&mean) {
                    *a += m / 2.0;
                }
            }
        }
    }
    for (band, (measured, expected)) in averages.iter().zip(&expected).enumerate() {
        let ratio = *measured as f32 / expected;
        eprintln!(
            "CACHED_FURNACE band={band} measured={measured} expected={expected} ratio={ratio}"
        );
        assert!(
            (ratio - 1.0).abs() < 0.02,
            "band {band}: {measured} != {expected}"
        );
    }
}

/// Feedback lag under the gameplay ceiling: early under-bounced cache values
/// stay in a plain running mean. Recency weighting and shorter caps trade
/// that lag for noise; this records the convergence side of the trade.
#[test]
#[ignore = "native resident furnace: convergence lag by history cap and recency"]
fn cached_furnace_lag_by_history_policy() {
    let fixture = crate::research::furnace();
    let material = fixture.materials.get(4);
    let emitters = Arc::new(EmissionDistribution::build(
        &fixture.scene,
        &fixture.materials,
    ));
    let native = tests::native_tracer();
    let probes = all_interior_faces(&fixture);
    let r = channels(material.diffuse)[2];
    let expected = channels(material.emission)[2] / (1.0 - r);
    for (limit, recency) in [
        (65_536, 1.0),
        (65_536, 1.5),
        (65_536, 2.0),
        (8_192, 1.0),
        (2_048, 1.0),
    ] {
        let mut gpu = GpuTracer::new(native.context.clone()).unwrap();
        gpu.enable_resident().unwrap();
        let mut work = TraceWork::from_probes(&probes);
        work.history_limit = limit;
        work.history_recency = recency;
        let mut line = String::new();
        for batch in 1..=400 {
            gpu.trace_compact(
                fixture.scene.clone(),
                &fixture.sources,
                &fixture.materials,
                &mut work,
                16,
                &emitters,
            )
            .unwrap();
            if [25, 50, 100, 200, 400].contains(&batch) {
                let mean: f64 = work
                    .faces
                    .iter()
                    .map(|f| f64::from(channels(f.radiance)[2]))
                    .sum::<f64>()
                    / work.faces.len() as f64;
                line += &format!(" {batch}:{:.4}", mean / f64::from(expected));
            }
        }
        eprintln!(
            "FURNACE_LAG limit={limit} recency={recency} blue r=0.7 fraction of fixed point:{line}"
        );
    }
}
