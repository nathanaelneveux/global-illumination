//! Isolated execution/variance studies against the unchanged production kernel.
//! Candidate shader transformations are test-only and never runtime toggles.
use super::*;
use crate::voxel_gi::world::FACE_OFFSETS;

fn cave_fixture() -> Fixture {
    let cave = crate::voxel_gi::research::cave();
    Fixture {
        name: "many_emitter_cave",
        scene: cave.scene,
        sources: cave.sources,
        materials: cave.materials,
        probes: cave.receivers,
        analytic: None,
    }
}

fn ceiling_fixture() -> Fixture {
    let mut f = fixture(2);
    f.name = "skylight_ceiling";
    f.probes = (6..26)
        .step_by(2)
        .flat_map(|z| {
            (6..26)
                .step_by(2)
                .filter(move |x| !(13..=17).contains(x) || !(13..=17).contains(&z))
                .map(move |x| SurfaceProbe::new(IVec3::new(x, 20, z), 8))
        })
        .collect();
    f
}

fn work(probes: &[SurfaceProbe], sequence: u32) -> TraceWork {
    let mut work = TraceWork::from_probes(probes, false);
    for p in &mut work.probes {
        p.sky_requested = false;
    }
    for face in &mut work.faces {
        face.sequence = sequence;
    }
    work
}

fn trace(
    gpu: &mut GpuTracer,
    f: &Fixture,
    emitters: &Arc<EmissionDistribution>,
    work: &mut TraceWork,
    paths: u32,
) {
    gpu.trace_compact(
        f.scene.clone(),
        &f.sources,
        &f.materials,
        work,
        paths,
        emitters,
        false,
    )
    .unwrap();
}

fn values(work: &TraceWork) -> Vec<EnergySpectrum> {
    work.faces.iter().map(|f| f.radiance).collect()
}

#[test]
#[ignore = "native execution layout study; same paths and five-band moments"]
fn settling_lane_layout_study() {
    lane_layout_study(&[4, 1, 2, 8, 16]);
}

#[test]
#[ignore = "native paired four/eight-lane confirmation"]
fn settling_lane_pair() {
    lane_layout_study(&[4, 8]);
}

#[test]
#[ignore = "native equal-path batch amortization study; no runtime change"]
fn settling_batch_amortization_study() {
    let context = super::super::tests::native_tracer().context;
    let modes = [(16, 8), (32, 8), (64, 8), (32, 16), (64, 16)];
    let mut tracers: Vec<_> = modes
        .iter()
        .map(|&(_, lanes)| {
            let mut gpu = GpuTracer::new(context.clone()).unwrap();
            gpu.fixed_path_lanes = Some(lanes);
            gpu
        })
        .collect();
    for mut f in [fixture(0), cave_fixture(), ceiling_fixture()] {
        let emitters = Arc::new(EmissionDistribution::build(&f.scene, &f.materials));
        f.probes = f.probes.iter().cycle().take(2048).cloned().collect();
        let mut times = vec![Vec::new(); modes.len()];
        let mut workers = times.clone();
        for round in 0..48 {
            for mode in (0..modes.len()).map(|i| (i + round) % modes.len()) {
                let (paths, _) = modes[mode];
                let mut work = work(
                    &f.probes[..32768 / paths as usize],
                    65536 + round as u32 * 64,
                );
                let start = Instant::now();
                trace(&mut tracers[mode], &f, &emitters, &mut work, paths);
                if round >= 8 {
                    times[mode].push(tracers[mode].last_gpu_ms.unwrap());
                    workers[mode].push(start.elapsed().as_secs_f32() * 1000.0);
                }
            }
        }
        for (i, (paths, lanes)) in modes.iter().enumerate() {
            eprintln!(
                "SETTLING_BATCH {}",
                json!({"fixture": f.name,
                "paths_per_face": paths, "lanes": lanes, "total_paths": 32768,
                "gpu_p50_ms": percentile(times[i].clone(),50),
                "worker_p50_ms": percentile(workers[i].clone(),50)})
            );
        }
    }
}

fn lane_layout_study(lanes: &[u32]) {
    let context = super::super::tests::native_tracer().context;
    let mut automatic = GpuTracer::new(context.clone()).unwrap();
    let mut tracers: Vec<_> = lanes
        .iter()
        .map(|&n| {
            let mut gpu = GpuTracer::new(context.clone()).unwrap();
            gpu.fixed_path_lanes = Some(n);
            gpu
        })
        .collect();
    for mut f in [fixture(0), cave_fixture(), ceiling_fixture()] {
        let emitters = Arc::new(EmissionDistribution::build(&f.scene, &f.materials));
        // Tail lanes, short batches, generalized reference loops, and every
        // output band must retain the same estimator before timing it.
        for paths in [1, 7, 16, 64] {
            let mut expected = work(&f.probes[..17], 65536);
            trace(&mut tracers[0], &f, &emitters, &mut expected, paths);
            for gpu in tracers.iter_mut().skip(1).chain([&mut automatic]) {
                let mut actual = work(&f.probes[..17], 65536);
                trace(gpu, &f, &emitters, &mut actual, paths);
                for (a, b) in expected.faces.iter().zip(&actual.faces) {
                    assert_eq!(a.sequence, b.sequence);
                    assert_eq!(a.samples, b.samples);
                    for (a, b) in [a.radiance, a.second_moment, a.direct].into_iter().zip([
                        b.radiance,
                        b.second_moment,
                        b.direct,
                    ]) {
                        for (a, b) in channels(a).into_iter().zip(channels(b)) {
                            assert!(
                                (a - b).abs() <= 1e-5 * (1.0 + a.abs()),
                                "layout changed {a} to {b}"
                            );
                        }
                    }
                }
            }
            assert_eq!(automatic.pipeline_key.2, if paths < 16 { 4 } else { 8 });
        }
        f.probes = f.probes.iter().cycle().take(2048).cloned().collect();
        for paths in [4, 16] {
            let mut times: Vec<Vec<f32>> = vec![vec![]; lanes.len()];
            let mut workers = times.clone();
            for round in 0..48 {
                for mode in (0..lanes.len()).map(|i| (i + round) % lanes.len()) {
                    let mut work = work(&f.probes, round as u32 * paths + 65536);
                    let start = Instant::now();
                    trace(&mut tracers[mode], &f, &emitters, &mut work, paths);
                    if round >= 8 {
                        times[mode].push(tracers[mode].last_gpu_ms.unwrap());
                        workers[mode].push(start.elapsed().as_secs_f32() * 1000.0);
                    }
                }
            }
            for (i, lane) in lanes.iter().enumerate() {
                eprintln!(
                    "SETTLING_LAYOUT {}",
                    json!({"fixture":f.name,"lanes":lane,"faces":f.probes.len(),"paths":paths,
                    "gpu_p50_ms":percentile(times[i].clone(),50),"gpu_p95_ms":percentile(times[i].clone(),95),
                    "worker_p50_ms":percentile(workers[i].clone(),50)})
                );
            }
        }
    }
}

#[test]
#[ignore = "native roulette variance/cost study, no history changes"]
fn settling_roulette_study() {
    let mut baseline = super::super::tests::native_tracer();
    assert_eq!(SHADER.matches("if bounce >= 1u {").count(), 1);
    let shader = SHADER.replace("if bounce >= 1u {", "if bounce >= 3u {");
    let mut candidate = GpuTracer::with_shader(baseline.context.clone(), true, &shader).unwrap();
    for f in [fixture(1), fixture(2), ceiling_fixture(), cave_fixture()] {
        let emitters = Arc::new(EmissionDistribution::build(&f.scene, &f.materials));
        let mut references = Vec::new();
        for (mode, gpu) in [&mut baseline, &mut candidate].into_iter().enumerate() {
            let mut high = work(&f.probes, 4_194_304 + mode as u32 * 1_048_576);
            trace(gpu, &f, &emitters, &mut high, 65536);
            references.push(values(&high));
        }
        let reference: Vec<_> = f.analytic.clone().unwrap_or_else(|| {
            references[0]
                .iter()
                .zip(&references[1])
                .map(|(a, b)| (*a + *b) * 0.5)
                .collect()
        });
        eprintln!(
            "SETTLING_REFERENCE {}",
            json!({"fixture":f.name,"difference":errors(&references[0],&references[1])})
        );
        // Warm short dispatches after high-sample reference passes.
        for _ in 0..8 {
            for gpu in [&mut baseline, &mut candidate] {
                trace(gpu, &f, &emitters, &mut work(&f.probes, 262144), 16);
            }
        }
        let checkpoints = [64, 128, 256, 512, 1024, 2048];
        let mut estimates: [[Vec<EnergySpectrum>; 6]; 2] = default();
        let mut expected = Vec::new();
        let mut peaks: [Vec<f32>; 2] = default();
        let mut times: [Vec<f32>; 2] = default();
        for seed in [0, 65536, 131072, 196608] {
            expected.extend_from_slice(&reference);
            let mut live = [work(&f.probes, seed), work(&f.probes, seed)];
            let mut peak: [Vec<f32>; 2] = std::array::from_fn(|_| vec![0.0; f.probes.len()]);
            for batch in 1..=128 {
                for mode in [batch % 2, 1 - batch % 2] {
                    let gpu = if mode == 0 {
                        &mut baseline
                    } else {
                        &mut candidate
                    };
                    trace(gpu, &f, &emitters, &mut live[mode], 16);
                    times[mode].push(gpu.last_gpu_ms.unwrap());
                    for (i, (value, reference)) in
                        values(&live[mode]).iter().zip(&reference).enumerate()
                    {
                        let luma = |v: Vec3| v.dot(Vec3::new(0.2126, 0.7152, 0.0722));
                        peak[mode][i] = peak[mode][i].max(
                            (luma(value.visible) - luma(reference.visible)).max(0.0)
                                / luma(reference.visible).max(1e-12),
                        );
                    }
                    if let Some(k) = checkpoints.iter().position(|&n| n == batch * 16) {
                        estimates[mode][k].extend(values(&live[mode]));
                    }
                }
            }
            for mode in 0..2 {
                peaks[mode].extend_from_slice(&peak[mode]);
            }
        }
        for mode in 0..2 {
            eprintln!(
                "SETTLING_ROULETTE {}",
                json!({"fixture":f.name,"mode":mode,"gpu_batch_p50_ms":percentile(times[mode].clone(),50),
                "bright_peak_p95":percentile(peaks[mode].clone(),95),"bright_peak_max":percentile(peaks[mode].clone(),100),
                "checkpoints":checkpoints.iter().zip(&estimates[mode]).map(|(n,v)| json!({"paths":n,"error":errors(v,&expected)})).collect::<Vec<_>>()})
            );
        }
    }
}

#[test]
#[ignore = "native exact light-node normal-cone arithmetic study"]
fn settling_light_importance_study() {
    let original = "    var emitter = 0.0;\n    for (var face = 0u; face < 6u; face += 1u) {\n        if (node.normals & (1u << face)) != 0u {\n            emitter = max(emitter, clamp(-dot(face_normal(face), direction) + spread, 0.0, 1.0));\n        }\n    }";
    let replacement = "    let positive = select(vec3(-1.0), -direction, (vec3(node.normals) & vec3(1u, 4u, 16u)) != vec3(0u));\n    let negative = select(vec3(-1.0), direction, (vec3(node.normals) & vec3(2u, 8u, 32u)) != vec3(0u));\n    let projected = max(positive, negative);\n    let emitter = clamp(max(max(projected.x, projected.y), projected.z) + spread, 0.0, 1.0);";
    assert_eq!(SHADER.matches(original).count(), 1);
    let mut baseline = super::super::tests::native_tracer();
    let mut candidate = GpuTracer::with_shader(
        baseline.context.clone(),
        true,
        &SHADER.replace(original, replacement),
    )
    .unwrap();
    let mut f = cave_fixture();
    let emitters = Arc::new(EmissionDistribution::build(&f.scene, &f.materials));
    for paths in [1, 7, 16, 64, 256] {
        let mut a = work(&f.probes, 65536);
        let mut b = work(&f.probes, 65536);
        trace(&mut baseline, &f, &emitters, &mut a, paths);
        trace(&mut candidate, &f, &emitters, &mut b, paths);
        for (a, b) in a.faces.iter().zip(&b.faces) {
            for (a, b) in [a.radiance, a.second_moment, a.direct].into_iter().zip([
                b.radiance,
                b.second_moment,
                b.direct,
            ]) {
                for (a, b) in channels(a).into_iter().zip(channels(b)) {
                    assert!(
                        (a - b).abs() < 1e-5 * (1.0 + a.abs()),
                        "normal-cone rewrite changed {a} to {b}"
                    );
                }
            }
        }
    }
    f.probes = f.probes.iter().cycle().take(2048).cloned().collect();
    let mut times: [Vec<f32>; 2] = default();
    for round in 0..64 {
        for mode in [round % 2, 1 - round % 2] {
            let gpu = if mode == 0 {
                &mut baseline
            } else {
                &mut candidate
            };
            trace(
                gpu,
                &f,
                &emitters,
                &mut work(&f.probes, 65536 + round as u32 * 16),
                16,
            );
            if round >= 8 {
                times[mode].push(gpu.last_gpu_ms.unwrap());
            }
        }
    }
    eprintln!(
        "LIGHT_IMPORTANCE baseline/branchless GPU p50 {}/{}",
        percentile(times[0].clone(), 50),
        percentile(times[1].clone(), 50)
    );
}

/// Fixed-work spatial reconstruction pilot. Compare against the *unfiltered*
/// high-sample corner field, so blurring the target cannot hide approximation
/// error. Missing coplanar neighbors are hard edges, not black samples.
#[test]
#[ignore = "native spatial reconstruction study; no production filtering"]
fn settling_spatial_study() {
    use std::collections::HashMap;
    let mut gpu = super::super::tests::native_tracer();
    for mut f in [fixture(2), ceiling_fixture(), cave_fixture()] {
        let centers: Vec<_> = f.probes.iter().step_by(3).cloned().collect();
        let mut indices = HashMap::new();
        f.probes.clear();
        let axes = |face: usize| match face / 2 {
            0 => [IVec3::Y, IVec3::Z],
            1 => [IVec3::X, IVec3::Z],
            _ => [IVec3::X, IVec3::Y],
        };
        for center in &centers {
            let face = center.face_mask.trailing_zeros() as usize;
            let [u, v] = axes(face);
            for y in -2..=2 {
                for x in -2..=2 {
                    let voxel = center.voxel + u * x + v * y;
                    if f.scene.material_at(voxel).is_none_or(|m| m == AIR)
                        || f.scene.material_at(voxel + FACE_OFFSETS[face]) != Some(AIR)
                    {
                        continue;
                    }
                    indices.entry((voxel, face)).or_insert_with(|| {
                        f.probes.push(SurfaceProbe::new(voxel, 1 << face));
                        f.probes.len() - 1
                    });
                }
            }
        }
        let neighbors: Vec<Vec<_>> = f
            .probes
            .iter()
            .map(|p| {
                let face = p.face_mask.trailing_zeros() as usize;
                let [u, v] = axes(face);
                [u, -u, v, -v]
                    .into_iter()
                    .filter_map(|d| indices.get(&(p.voxel + d, face)).copied())
                    .collect()
            })
            .collect();
        let corners: Vec<Vec<_>> = centers
            .iter()
            .flat_map(|p| {
                let face = p.face_mask.trailing_zeros() as usize;
                let [u, v] = axes(face);
                let indices = &indices;
                [-1, 1].into_iter().flat_map(move |y| {
                    [-1, 1].into_iter().map(move |x| {
                        let mut sites = vec![indices[&(p.voxel, face)]];
                        for delta in [u * x, v * y] {
                            if let Some(&i) = indices.get(&(p.voxel + delta, face)) {
                                sites.push(i);
                            }
                        }
                        if sites.len() > 1
                            && let Some(&i) = indices.get(&(p.voxel + u * x + v * y, face))
                        {
                            sites.push(i);
                        }
                        sites
                    })
                })
            })
            .collect();
        let reconstruct = |values: &[Vec3]| -> Vec<Vec3> {
            corners
                .iter()
                .map(|sites| sites.iter().map(|&i| values[i]).sum::<Vec3>() / sites.len() as f32)
                .collect()
        };
        let filter = |live: &TraceWork, mode: usize| -> Vec<Vec3> {
            live.faces
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let mean = a.radiance.visible;
                    if mode == 0 {
                        return mean;
                    }
                    let mut sum = mean * 4.0;
                    let mut weight = 4.0;
                    let variance = (a.second_moment.visible - mean * mean).max(Vec3::ZERO)
                        / a.samples.max(1) as f32;
                    for &j in &neighbors[i] {
                        let b = &live.faces[j];
                        let other = b.radiance.visible;
                        let other_variance = (b.second_moment.visible - other * other)
                            .max(Vec3::ZERO)
                            / b.samples.max(1) as f32;
                        let w = if mode == 1 {
                            1.0
                        } else {
                            let delta = mean - other;
                            let v = (variance + other_variance).element_sum();
                            v / (v + delta.length_squared() + 1e-12)
                        };
                        sum += other * w;
                        weight += w;
                    }
                    sum / weight
                })
                .collect()
        };
        let emitters = Arc::new(EmissionDistribution::build(&f.scene, &f.materials));
        let mut high = work(&f.probes, 4_194_304);
        // Keep the offline oracle in bounded submissions as well. A dense
        // neighborhood at 65k paths must not become one long GPU dispatch.
        for _ in 0..64 {
            trace(&mut gpu, &f, &emitters, &mut high, 1024);
        }
        let reference = reconstruct(&filter(&high, 0));
        let error = |actual: &[Vec3]| -> (f32, f32) {
            let errors: Vec<_> = actual
                .iter()
                .zip(&reference)
                .map(|(a, b)| (*a - *b).length() / b.length().max(1e-12))
                .collect();
            (percentile(errors.clone(), 95), percentile(errors, 100))
        };
        eprintln!(
            "SETTLING_SPATIAL_BIAS {}",
            json!({"fixture":f.name,"faces":f.probes.len(),"corners":corners.len(),"fixed":error(&reconstruct(&filter(&high,1))),"guided":error(&reconstruct(&filter(&high,2)))})
        );
        let checkpoints = [64, 128, 256, 512, 1024, 2048];
        let mut errors: [[Vec<f32>; 6]; 3] = default();
        let mut peaks: [Vec<f32>; 3] = default();
        let mut gpu_ms = Vec::new();
        for seed in [0, 65536, 131072, 196608] {
            let mut live = work(&f.probes, seed);
            let mut peak: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0; corners.len()]);
            for batch in 1..=128 {
                trace(&mut gpu, &f, &emitters, &mut live, 16);
                gpu_ms.push(gpu.last_gpu_ms.unwrap());
                for mode in 0..3 {
                    let actual = reconstruct(&filter(&live, mode));
                    let luma = |v: Vec3| v.dot(Vec3::new(0.2126, 0.7152, 0.0722));
                    for (i, (a, b)) in actual.iter().zip(&reference).enumerate() {
                        peak[mode][i] =
                            peak[mode][i].max((luma(*a) - luma(*b)).max(0.0) / luma(*b).max(1e-12));
                    }
                    if let Some(k) = checkpoints.iter().position(|&n| n == batch * 16) {
                        errors[mode][k].extend(
                            actual
                                .iter()
                                .zip(&reference)
                                .map(|(a, b)| (*a - *b).length() / b.length().max(1e-12)),
                        );
                    }
                }
            }
            for mode in 0..3 {
                peaks[mode].extend_from_slice(&peak[mode]);
            }
        }
        for mode in 0..3 {
            eprintln!(
                "SETTLING_SPATIAL {}",
                json!({"fixture":f.name,"mode":mode,"gpu_p50_ms":percentile(gpu_ms.clone(),50),"peak_p95":percentile(peaks[mode].clone(),95),"peak_max":percentile(peaks[mode].clone(),100),"error_p95":errors[mode].iter().map(|e|percentile(e.clone(),95)).collect::<Vec<_>>(),"error_max":errors[mode].iter().map(|e|percentile(e.clone(),100)).collect::<Vec<_>>()})
            );
        }
    }
}
