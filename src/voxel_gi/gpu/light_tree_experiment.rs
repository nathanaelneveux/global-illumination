//! Equal-path and equal-GPU-time study. CDF and tree use the same material,
//! visibility and three-bounce model; both evaluate the matching emission MIS.
use super::*;
use crate::voxel_gi::research::{cave, channels};

fn values(probes: &[SurfaceProbe]) -> Vec<EnergySpectrum> {
    probes
        .iter()
        .map(|p| p.radiance[p.face_mask.trailing_zeros() as usize])
        .collect()
}

fn error(values: &[EnergySpectrum], reference: &[EnergySpectrum]) -> (f32, [f32; 5]) {
    let mut errors = Vec::new();
    let mut squares = [0.0; 5];
    for (a, b) in values.iter().zip(reference) {
        for (band, (a, b)) in channels(*a).into_iter().zip(channels(*b)).enumerate() {
            squares[band] += (a - b).powi(2);
            errors.push((a - b).abs() / (0.02 + b.abs()));
        }
    }
    errors.sort_by(f32::total_cmp);
    (
        errors[errors.len() * 95 / 100],
        squares.map(|s| (s / values.len() as f32).sqrt()),
    )
}

#[test]
#[ignore = "native CPU/GPU tree and high-sample many-source study"]
fn many_light_tree_study() {
    let cave = cave();
    let mut gpu = super::tests::native_tracer();
    let source =
        std::fs::read_to_string(std::env::var("VOXEL_GI_BASELINE_SHADER").unwrap()).unwrap();
    let mut baseline = GpuTracer::with_shader(gpu.context.clone(), true, &source).unwrap();
    for kind in 0..3 {
        let mut materials = cave.materials.clone();
        for id in [4, 5, 6, 7] {
            if (kind == 1 && id != 7) || (kind == 2 && id != 4) {
                let mut material = materials.get(id);
                material.emission = EnergySpectrum::ZERO;
                materials.set(id, material);
            }
        }
        let start = Instant::now();
        let tree = Arc::new(EmissionDistribution::build(&cave.scene, &materials));
        let build_ms = start.elapsed().as_secs_f32() * 1000.0;
        let mut cdf = (*tree).clone();
        cdf.tree.clear();
        let cdf = Arc::new(cdf);
        let distributions = [&cdf, &tree];
        let mut cpu = TraceWork::from_probes(&cave.receivers[..5], false);
        let mut native = TraceWork::from_probes(&cave.receivers[..5], false);
        crate::voxel_gi::tracer::trace_work_batch(
            &cave.scene,
            &cave.sources,
            &materials,
            &mut cpu,
            8192,
            &tree,
            &mut default(),
        );
        gpu.trace_compact(
            cave.scene.clone(),
            &cave.sources,
            &materials,
            &mut native,
            8192,
            &tree,
            false,
        )
        .unwrap();
        for (a, b) in cpu.faces.iter().zip(&native.faces) {
            for (a, b) in channels(a.radiance).into_iter().zip(channels(b.radiance)) {
                assert!(
                    (a - b).abs() < 0.005 * (0.02 + a.abs()),
                    "CPU/GPU {a} vs {b}"
                );
            }
        }
        let mut high = [cave.receivers.clone(), cave.receivers.clone()];
        for mode in 0..2 {
            let tracer = if mode == 0 { &mut baseline } else { &mut gpu };
            for p in &mut high[mode] {
                p.sequence = [1_048_576; 6];
            }
            for _ in 0..32 {
                tracer
                    .trace(
                        cave.scene.clone(),
                        &cave.sources,
                        &materials,
                        &mut high[mode],
                        1024,
                        distributions[mode],
                    )
                    .unwrap();
            }
        }
        let reference: Vec<_> = values(&high[0])
            .iter()
            .zip(values(&high[1]))
            .map(|(a, b)| (*a + b) * 0.5)
            .collect();
        eprintln!(
            "LIGHT_TREE_REFERENCE kind={kind} faces={} nodes={} extra_bytes={} build_ms={build_ms:.4} high_difference={:?}",
            tree.faces.len(),
            tree.tree.len(),
            tree.tree.len() * 64,
            error(&values(&high[1]), &values(&high[0]))
        );
        let mut expected = Vec::new();
        let mut estimates: [[Vec<EnergySpectrum>; 3]; 2] = default();
        let mut equal_time: [[Vec<EnergySpectrum>; 3]; 2] = default();
        let mut budget_samples: [[Vec<usize>; 3]; 2] = default();
        let mut times: [Vec<f32>; 2] = default();
        // Warm the actual four-path dispatch after the long reference passes.
        // On Metal the transition otherwise contaminates the first timing
        // budget with long-pass query/clock behavior. Discard these estimates.
        for _ in 0..8 {
            for (mode, distribution) in distributions.iter().enumerate() {
                let tracer = if mode == 0 { &mut baseline } else { &mut gpu };
                tracer
                    .trace(
                        cave.scene.clone(),
                        &cave.sources,
                        &materials,
                        &mut cave.receivers.clone(),
                        4,
                        distribution,
                    )
                    .unwrap();
            }
        }
        for replicate in 0..8 {
            expected.extend_from_slice(&reference);
            let mut probes = [cave.receivers.clone(), cave.receivers.clone()];
            let mut elapsed = [0.0_f32; 2];
            let mut next_budget = [0; 2];
            for group in &mut probes {
                for p in group {
                    p.sequence = [replicate * 4096; 6];
                }
            }
            for batch in 1..=256 {
                for mode in (0..2).map(|m| (m + batch + replicate as usize) % 2) {
                    let tracer = if mode == 0 { &mut baseline } else { &mut gpu };
                    tracer
                        .trace(
                            cave.scene.clone(),
                            &cave.sources,
                            &materials,
                            &mut probes[mode],
                            4,
                            distributions[mode],
                        )
                        .unwrap();
                    if batch > 4 {
                        times[mode].push(tracer.last_gpu_ms.expect("native timing"));
                    }
                    elapsed[mode] += tracer.last_gpu_ms.unwrap();
                    if next_budget[mode] < 3
                        && elapsed[mode] >= [8.0, 16.0, 32.0][next_budget[mode]]
                    {
                        equal_time[mode][next_budget[mode]].extend(values(&probes[mode]));
                        budget_samples[mode][next_budget[mode]].push(batch * 4);
                        next_budget[mode] += 1;
                    }
                    if let Some(k) = [16, 64, 256].iter().position(|&b| b == batch) {
                        estimates[mode][k].extend(values(&probes[mode]));
                    }
                }
            }
        }
        for t in &mut times {
            t.sort_by(f32::total_cmp);
        }
        for mode in 0..2 {
            eprintln!(
                "LIGHT_TREE kind={kind} mode={mode} gpu_batch_p50_ms={:.6} errors_64_256_1024={:?}",
                times[mode][times[mode].len() / 2],
                estimates[mode]
                    .iter()
                    .map(|v| error(v, &expected))
                    .collect::<Vec<_>>()
            );
            assert!(equal_time[mode].iter().all(|v| v.len() == expected.len()));
            eprintln!(
                "LIGHT_TREE_EQUAL_TIME kind={kind} mode={mode} budgets_ms=8,16,32 samples={:?} errors={:?}",
                budget_samples[mode],
                equal_time[mode]
                    .iter()
                    .map(|v| error(v, &expected))
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
#[ignore = "native selection and reverse-PDF oracle"]
fn native_light_tree_selection_pdf_matches_cpu() {
    let cave = cave();
    let emitters = Arc::new(EmissionDistribution::build(&cave.scene, &cave.materials));
    let device = super::tests::native_tracer().context;
    let mut shader = SHADER.replace("fn trace_faces_parallel(", "fn unused_parallel(");
    shader.push_str(
        r#"
@compute @workgroup_size(64)
fn trace_faces_parallel(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x / 4u;
    if id.x % 4u != 0u || i >= params.counts.x { return; }
    let task = tasks[i];
    let normal = face_normal(task.face);
    let position = vec3<f32>(task.voxel) + vec3(0.5) + normal * 0.5;
    // Retain the ordinary dispatch layout without tracing in this oracle.
    if params.counts.y == 0u {
        results[i * 6u] = bitcast<vec4<u32>>(trace_path(position, normal, normal, 0u, 0u).value);
        return;
    }
    let e = tree_emitter(position, normal, sample_1d(91u, task.sequence, 12u));
    let reverse = tree_probability(position, normal, e.voxel, face_normal(e.face));
    for (var j = 0u; j < 6u; j += 1u) { results[i * 6u + j] = vec4(0u); }
    results[i * 6u] = bitcast<vec4<u32>>(vec4(vec3<f32>(e.voxel), f32(e.face)));
    results[i * 6u + 1u] = bitcast<vec4<u32>>(vec4(e.sampling.y, 0.0, 0.0, 0.0));
    results[i * 6u + 4u] = bitcast<vec4<u32>>(vec4(reverse, 0.0, 0.0, 0.0));
}
"#,
    );
    let mut gpu = GpuTracer::with_shader(device, true, &shader).unwrap();
    let probes: Vec<_> = (0..32)
        .flat_map(|sequence| {
            cave.receivers.iter().map(move |p| {
                let mut p = p.clone();
                p.sequence = [sequence; 6];
                p
            })
        })
        .collect();
    let mut work = TraceWork::from_probes(&probes, false);
    for p in &mut work.probes {
        p.sky_requested = false;
    }
    gpu.trace_compact(
        cave.scene.clone(),
        &cave.sources,
        &cave.materials,
        &mut work,
        1,
        &emitters,
        false,
    )
    .unwrap();
    let mut differing_choices = 0;
    for (f, p) in work.faces.iter().zip(probes) {
        let normal = crate::voxel_gi::world::FACE_OFFSETS[f.face].as_vec3();
        let position = f.voxel.as_vec3() + Vec3::splat(0.5) + normal * 0.5;
        let cpu = emitters
            .select_at(
                position,
                normal,
                crate::voxel_gi::sampling::sample_1d(91, p.sequence[f.face], 12),
            )
            .unwrap();
        let chosen_voxel = f.radiance.visible.as_ivec3();
        let chosen_face = f.radiance.infrared as u32;
        differing_choices += usize::from((cpu.voxel, cpu.face) != (chosen_voxel, chosen_face));
        let material = cave.scene.material_at(chosen_voxel).unwrap() as u8;
        let density = emitters.probability_at(
            position,
            normal,
            chosen_voxel,
            chosen_face,
            material,
            &cave.materials,
        );
        assert!((density - f.radiance.ionizing).abs() < 2e-5 * density + 1e-9);
        assert!((density - f.direct.visible.x).abs() < 2e-5 * density + 1e-9);
    }
    assert!(differing_choices <= 1);
    eprintln!(
        "LIGHT_TREE_NATIVE_PDF samples={} differing_choices={differing_choices} forward_reverse_density=true",
        work.faces.len()
    );
}
