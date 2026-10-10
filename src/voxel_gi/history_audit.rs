//! Opt-in policy experiments. Identical fresh moments feed competing histories;
//! none of these virtual service cadences measure interactive frame pacing.
use super::*;
use crate::voxel_gi::geometry::{GeometryHistory, SceneChanges};
use crate::voxel_gi::sources::PointTransportLight;

fn page(scene: &TransportScene) -> Vec<u16> {
    (0..32)
        .flat_map(|z| (0..32).flat_map(move |y| (0..32).map(move |x| IVec3::new(x, y, z))))
        .map(|v| scene.material_at(v).unwrap())
        .collect()
}

fn fresh_batch(gpu: &mut GpuTracer, f: &Fixture, probes: &mut [SurfaceProbe], paths: u32) {
    for p in probes.iter_mut() {
        p.restart_measurements();
    }
    gpu.trace(
        f.scene.clone(),
        &f.sources,
        &f.materials,
        probes,
        paths,
        &Arc::new(EmissionDistribution::build(&f.scene, &f.materials)),
    )
    .unwrap();
}

fn integrate(p: &mut SurfaceProbe, fresh: &SurfaceProbe, cap: u32, now: Instant) {
    p.samples[2] = p.samples[2].min(cap) + fresh.samples[2];
    let alpha = fresh.samples[2] as f32 / p.samples[2] as f32;
    p.radiance[2] += (fresh.radiance[2] - p.radiance[2]) * alpha;
    p.second_moment[2] += (fresh.second_moment[2] - p.second_moment[2]) * alpha;
    p.resolve_face(2, Some(now - Duration::from_millis(50)), now, 0.075);
}

fn relative(a: EnergySpectrum, b: EnergySpectrum) -> f32 {
    channels(a)
        .into_iter()
        .zip(channels(b))
        .map(|(a, b)| (a - b).abs() / (0.002 + b.abs()))
        .fold(0.0, f32::max)
}

#[test]
#[ignore = "native matched-ray receiver material null control"]
fn policy_material_identity_control() {
    let mut gpu = super::super::tests::native_tracer();
    let mut f = fixture(1);
    f.sources.environment = EnergySpectrum::ZERO;
    f.materials.set(6, f.materials.get(2));
    let mut changed = page(&f.scene);
    for p in &f.probes {
        changed[linear_index(p.voxel.as_uvec3(), PAGE_SIZE)] = 6;
    }
    let next = Arc::new(TransportScene::with_page_delta(
        &f.scene,
        vec![(IVec3::ZERO, changed)],
        &[],
    ));
    let mut journal = GeometryHistory::default();
    journal.record(
        0,
        1,
        SceneChanges::between(&f.scene, &next).transport.unwrap(),
    );
    let mut warm = f.probes.clone();
    fresh_batch(&mut gpu, &f, &mut warm, 2048);
    let mut candidates = [warm.clone(), warm];
    for (mode, probes) in candidates.iter_mut().enumerate() {
        for p in probes {
            let cap = journal.history_limit(p.voxel, 0, 1);
            assert_eq!(cap, 16);
            p.samples[2] = if mode == 0 { 0 } else { cap };
            p.display[2].indirect = p.radiance[2].visible;
        }
    }
    let mut fresh = f.probes.clone();
    for p in &mut fresh {
        p.sequence[2] = 2048;
    }
    let start = Instant::now();
    let mut early_error: [Vec<f32>; 2] = default();
    for batch in 1..=64 {
        let mut null = fresh.clone();
        fresh_batch(&mut gpu, &f, &mut fresh, 4);
        let previous = std::mem::replace(&mut f.scene, next.clone());
        fresh_batch(&mut gpu, &f, &mut null, 4);
        f.scene = previous;
        for (a, b) in fresh.iter().zip(&null) {
            assert!(
                relative(a.radiance[2], b.radiance[2]) < 1e-6,
                "material identity alone must not change incident energy"
            );
        }
        for (mode, probes) in candidates.iter_mut().enumerate() {
            for ((p, new), reference) in probes
                .iter_mut()
                .zip(&fresh)
                .zip(f.analytic.as_ref().unwrap())
            {
                integrate(p, new, u32::MAX, start + Duration::from_millis(batch * 50));
                if batch <= 16 {
                    early_error[mode].push(relative(p.radiance[2], *reference));
                }
            }
        }
    }
    eprintln!(
        "POLICY material-null first64-path all-band error p95 reset={} retained16={}",
        percentile(early_error[0].clone(), 95),
        percentile(early_error[1].clone(), 95)
    );
}

#[test]
#[ignore = "native repeated remote history null control"]
fn policy_remote_streaming_control() {
    let mut gpu = super::super::tests::native_tracer();
    let mut f = fixture(1);
    f.sources.environment = EnergySpectrum::ZERO;
    let original = f.scene.clone();
    let mut remote = vec![AIR; volume_len(PAGE_SIZE)];
    remote[0] = 2;
    let extended = Arc::new(TransportScene::with_page_delta(
        &original,
        vec![(IVec3::X * 4, remote)],
        &[],
    ));
    let mut history = GeometryHistory::default();
    history.record(
        0,
        1,
        SceneChanges::between(&original, &extended)
            .transport
            .unwrap(),
    );
    assert!(
        f.probes
            .iter()
            .all(|p| history.history_limit(p.voxel, 0, 1) == 256)
    );
    let mut warm = f.probes.clone();
    fresh_batch(&mut gpu, &f, &mut warm, 8192);
    const CAPS: [u32; 4] = [64, 256, 1024, 65_536];
    let mut candidates = CAPS.map(|_| warm.clone());
    for probes in &mut candidates {
        for p in probes {
            p.display[2].indirect = p.radiance[2].visible;
        }
    }
    let mut fresh = f.probes.clone();
    for p in &mut fresh {
        p.sequence[2] = 8192;
    }
    let start = Instant::now();
    let mut raw: [Vec<f32>; 4] = default();
    let mut variation: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0; fresh.len()]);
    for batch in 0..256 {
        let changes = batch % 8 == 0;
        if changes {
            f.scene = if batch % 16 == 0 {
                extended.clone()
            } else {
                original.clone()
            };
        }
        let mut null = fresh.clone();
        fresh_batch(&mut gpu, &f, &mut fresh, 4);
        if changes {
            let previous = std::mem::replace(&mut f.scene, original.clone());
            fresh_batch(&mut gpu, &f, &mut null, 4);
            f.scene = previous;
            for (a, b) in fresh.iter().zip(null) {
                assert!(
                    relative(a.radiance[2], b.radiance[2]) < 1e-6,
                    "the sealed room cannot see the added remote geometry"
                );
            }
        }
        for (mode, probes) in candidates.iter_mut().enumerate() {
            for (index, ((p, new), reference)) in probes
                .iter_mut()
                .zip(&fresh)
                .zip(f.analytic.as_ref().unwrap())
                .enumerate()
            {
                let old = p.display[2].indirect;
                integrate(
                    p,
                    new,
                    if changes { CAPS[mode] } else { u32::MAX },
                    start + Duration::from_millis((batch + 1) * 50),
                );
                raw[mode].push(relative(p.radiance[2], *reference));
                let compressed = |v: Vec3| v / (Vec3::splat(0.25) + v);
                variation[mode][index] += (compressed(p.display[2].indirect) - compressed(old))
                    .abs()
                    .max_element();
            }
        }
    }
    for (mode, cap) in CAPS.into_iter().enumerate() {
        eprintln!(
            "POLICY repeated-remote cap={cap} all-band-error-p95={} display-total-variation-p95={}",
            percentile(raw[mode].clone(), 95),
            percentile(variation[mode].clone(), 95)
        );
    }
}

#[test]
#[ignore = "native moving analytic source history study"]
fn policy_moving_analytic_source() {
    let mut gpu = super::super::tests::native_tracer();
    let mut f = fixture(1);
    f.probes = f.probes.into_iter().step_by(4).collect();
    let mut shell = page(&f.scene);
    for m in &mut shell {
        if *m == 4 || *m == 5 {
            *m = 2;
        }
    }
    f.scene = Arc::new(TransportScene::with_page_delta(
        &f.scene,
        vec![(IVec3::ZERO, shell)],
        &[],
    ));
    f.materials.set(
        2,
        MaterialTransport {
            diffuse: EnergySpectrum::ONE * 0.6,
            ..default()
        },
    );
    f.sources.environment = EnergySpectrum::ZERO;
    let poses: Vec<_> = (0..16)
        .map(|i| {
            let phase = i as f32 * std::f32::consts::TAU / 16.0;
            PointTransportLight {
                position: Vec3::new(16.0 + 8.0 * phase.sin(), 14.0, 16.0 + 6.0 * phase.cos()),
                intensity: EnergySpectrum::new(Vec3::new(200.0, 40.0, 10.0), 100.0, 50.0),
                range: 64.0,
                custom_visible: Vec3::ZERO,
            }
        })
        .collect();
    let mut refs = Vec::new();
    let mut disagreement = Vec::new();
    for light in &poses {
        f.sources.points = vec![*light];
        let mut reference = [f.probes.clone(), f.probes.clone()];
        for (mode, probes) in reference.iter_mut().enumerate() {
            for p in probes.iter_mut() {
                p.sequence[2] = 1_048_576 + mode as u32 * 65_536;
            }
            fresh_batch(&mut gpu, &f, probes, 16_384);
        }
        refs.push(
            reference[0]
                .iter()
                .zip(&reference[1])
                .map(|(a, b)| {
                    disagreement.push(relative(a.radiance[2], b.radiance[2]));
                    (a.radiance[2] + b.radiance[2]) * 0.5
                })
                .collect::<Vec<_>>(),
        );
    }
    let mut fresh = f.probes.clone();
    f.sources.points = vec![poses[0]];
    fresh_batch(&mut gpu, &f, &mut fresh, 2048);
    let mut candidates = [fresh.clone(), fresh.clone()];
    for probes in &mut candidates {
        for p in probes {
            p.display[2].indirect = p.radiance[2].visible;
        }
    }
    let start = Instant::now();
    let mut raw: [Vec<f32>; 2] = default();
    let mut display: [Vec<f32>; 2] = default();
    let mut steps: [Vec<f32>; 2] = default();
    // One pose per four-path service, four cycles. This is deliberately fast
    // motion in service units, not a claim about the game's light-update rate.
    for batch in 1..=64 {
        let pose = batch % poses.len();
        f.sources.points = vec![poses[pose]];
        fresh_batch(&mut gpu, &f, &mut fresh, 4);
        for (mode, probes) in candidates.iter_mut().enumerate() {
            for ((p, new), target) in probes.iter_mut().zip(&fresh).zip(&refs[pose]) {
                let old = p.display[2].indirect;
                integrate(
                    p,
                    new,
                    [4, 16][mode],
                    start + Duration::from_millis(batch as u64 * 50),
                );
                if batch > 16 {
                    raw[mode].push(relative(p.radiance[2], *target));
                    display[mode].push(
                        (p.display[2].indirect - target.visible).abs().max_element()
                            / (0.002 + target.visible.max_element()),
                    );
                    steps[mode].push(
                        (p.display[2].indirect - old).abs().max_element()
                            / (0.002 + target.visible.max_element()),
                    );
                }
            }
        }
    }
    eprintln!(
        "POLICY analytic reference-disagreement-p95={}",
        percentile(disagreement, 95)
    );
    for mode in 0..2 {
        eprintln!(
            "POLICY analytic cap={} raw-error-p95={} display-error-p95={} display-step-p95={}",
            [4, 16][mode],
            percentile(raw[mode].clone(), 95),
            percentile(display[mode].clone(), 95),
            percentile(steps[mode].clone(), 95)
        );
    }
}

#[test]
#[ignore = "native rare-portal sampling evidence study"]
fn policy_dark_sampling_evidence() {
    let mut gpu = super::super::tests::native_tracer();
    let mut f = fixture(1);
    let mut shell = page(&f.scene);
    for m in &mut shell {
        if *m == 4 || *m == 5 {
            *m = 2;
        }
    }
    shell[linear_index(UVec3::new(16, 20, 16), PAGE_SIZE)] = AIR;
    f.scene = Arc::new(TransportScene::with_page_delta(
        &f.scene,
        vec![(IVec3::ZERO, shell)],
        &[],
    ));
    f.sources.environment = EnergySpectrum::new(Vec3::splat(4.0), 2.0, 1.0);
    let mut missed = [0_usize; 2];
    for replicate in 0..8 {
        let mut fresh = f.probes.clone();
        for p in &mut fresh {
            p.sequence[2] = replicate * 4096;
        }
        let mut seen = vec![false; fresh.len()];
        for batch in 1..=64 {
            fresh_batch(&mut gpu, &f, &mut fresh, 4);
            for (seen, p) in seen.iter_mut().zip(&fresh) {
                *seen |= p.radiance[2].max_component() > 0.0;
            }
            for (i, checkpoint) in [16, 64].into_iter().enumerate() {
                if batch == checkpoint {
                    missed[i] += seen.iter().filter(|&&v| !v).count();
                }
            }
        }
    }
    assert!(
        missed[1] < missed[0],
        "extra discovery rays must find more portal paths"
    );
    eprintln!(
        "POLICY rare-portal zero-observations /800: at64={} at256={} (not convergence, nor a scheduler timing)",
        missed[0], missed[1]
    );
}

/// Exact-zero post-change response isolates the unavoidable cost of retained
/// weight, including repeated caps. Applies independently to each energy band.
#[test]
fn policy_retention_extinction_bounds() {
    for cap in [4, 16, 64, 256, 1024] {
        let mut value = 1.0_f64;
        let mut count = cap;
        let mut once = None;
        let mut repeated = None;
        let mut rolling = 1.0_f64;
        for batch in 1..=4096 {
            value *= count as f64 / (count + 4) as f64;
            count += 4;
            rolling *= cap as f64 / (cap + 4) as f64;
            assert!((value - cap as f64 / count as f64).abs() < 1e-12);
            if value < 0.1 - 1e-12 {
                once.get_or_insert(batch * 4);
            }
            if rolling < 0.1 - 1e-12 {
                repeated.get_or_insert(batch * 4);
            }
        }
        assert!(once.is_some() && repeated.is_some());
        eprintln!(
            "POLICY extinction cap={cap} fresh-paths-below10% once={} recapped-every-batch={}",
            once.unwrap(),
            repeated.unwrap()
        );
    }
}

#[test]
#[ignore = "isolated CPU emission-cache rebuild cost, not gameplay timing"]
fn policy_emission_cache_rebuild_cost() {
    use crate::voxel_gi::{emission::EmissionCache, research::cave};
    let cave = cave();
    let materials = Arc::new(cave.materials);
    // Non-emitting background residency. Scene construction is deliberately
    // outside the timed span; only distribution preparation is measured.
    let pages = (4..68)
        .map(|x| {
            let mut voxels = vec![AIR; volume_len(PAGE_SIZE)];
            voxels[0] = 2;
            (IVec3::new(x, 0, 0), voxels)
        })
        .collect();
    let a = Arc::new(TransportScene::with_page_delta(&cave.scene, pages, &[]));
    let mut edited = vec![AIR; volume_len(PAGE_SIZE)];
    edited[0] = 3;
    let b = Arc::new(TransportScene::with_page_delta(
        &a,
        vec![(IVec3::X * 67, edited)],
        &[],
    ));
    let mut cache = EmissionCache::prepare(None, a.clone(), materials.clone());
    let reference = cache.distribution.clone();
    let mut unchanged = Vec::new();
    let mut rebuilt = Vec::new();
    let mut full_build = Vec::new();
    for i in 0..32 {
        let scene = if i % 2 == 0 { b.clone() } else { a.clone() };
        let start = Instant::now();
        cache = EmissionCache::prepare(Some(cache), scene.clone(), materials.clone());
        rebuilt.push(start.elapsed().as_secs_f32() * 1000.0);
        assert!(Arc::ptr_eq(&reference, &cache.distribution));
        assert_eq!(reference.faces.len(), cache.distribution.faces.len());
        for (a, b) in reference.faces.iter().zip(&cache.distribution.faces) {
            assert_eq!(
                (a.voxel, a.face, a.material, a.probability),
                (b.voxel, b.face, b.material, b.probability)
            );
        }
        let distribution = cache.distribution.clone();
        let start = Instant::now();
        cache = EmissionCache::prepare(Some(cache), scene.clone(), materials.clone());
        unchanged.push(start.elapsed().as_secs_f32() * 1000.0);
        assert!(Arc::ptr_eq(&distribution, &cache.distribution));
        let start = Instant::now();
        let full = EmissionCache::prepare(None, scene, materials.clone());
        full_build.push(start.elapsed().as_secs_f32() * 1000.0);
        assert_eq!(full.distribution.faces.len(), reference.faces.len());
    }
    eprintln!(
        "POLICY emission-cache regions={} faces={} unchanged-ms-p50={} changed-ms-p50={} changed-ms-p95={} full-build-ms-p50={}",
        a.regions.len(),
        reference.faces.len(),
        percentile(unchanged, 50),
        percentile(rebuilt.clone(), 50),
        percentile(rebuilt, 95),
        percentile(full_build, 50)
    );
}
