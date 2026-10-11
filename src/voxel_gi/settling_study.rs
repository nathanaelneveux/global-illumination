//! Isolated execution/variance studies against the unchanged production kernel.
//! Candidate shader transformations are test-only and never runtime toggles.
use super::*;

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

fn lane_shader(lanes: u32) -> String {
    assert!([1, 2, 4, 8, 16].contains(&lanes));
    let (prefix, parallel) = SHADER.split_once("// Cooperative layout:").unwrap();
    let parallel = parallel
        .replace("/ 4u", &format!("/ {lanes}u"))
        .replace("% 4u", &format!("% {lanes}u"))
        .replace("+= 4u", &format!("+= {lanes}u"))
        .replace("< 4u", &format!("< {lanes}u"));
    format!("{prefix}// Cooperative layout:{parallel}")
}

#[test]
#[ignore = "native execution layout study; same paths and five-band moments"]
fn settling_lane_layout_study() {
    let context = super::super::tests::native_tracer().context;
    let lanes = [4, 1, 2, 8, 16];
    let mut tracers: Vec<_> = lanes
        .iter()
        .map(|&n| {
            let mut gpu = GpuTracer::with_shader(context.clone(), true, &lane_shader(n)).unwrap();
            gpu.faces_per_workgroup = 64 / n;
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
            for gpu in tracers.iter_mut().skip(1) {
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
