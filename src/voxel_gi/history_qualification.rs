//! Native sample tapes for history-policy qualification. Estimator, path streams
//! and targets stay fixed while experiment branches vary only history handling.
use super::*;
use crate::voxel_gi::work::TraceWork;

const BATCH: u32 = 16;
const STEPS: u32 = 512;
const WARM: u32 = 4096;
const REFERENCE_BLOCK: u32 = 32_768;

fn page(scene: &TransportScene) -> Vec<u16> {
    (0..32)
        .flat_map(|z| (0..32).flat_map(move |y| (0..32).map(move |x| IVec3::new(x, y, z))))
        .map(|v| scene.material_at(v).unwrap())
        .collect()
}

fn changed(scene: &TransportScene, page: Vec<u16>) -> Arc<TransportScene> {
    Arc::new(TransportScene::with_page_delta(
        scene,
        vec![(IVec3::ZERO, page)],
        &[],
    ))
}

fn moment(face: &crate::voxel_gi::work::FaceWork) -> Value {
    json!({"mean":channels(face.radiance),"second":channels(face.second_moment)})
}

fn trace(
    gpu: &mut GpuTracer,
    fixture: &Fixture,
    scene: Arc<TransportScene>,
    starts: &[u32],
    paths: u32,
) -> (Vec<Vec<Value>>, f32) {
    let emitters = Arc::new(EmissionDistribution::build(&scene, &fixture.materials));
    let mut work = TraceWork::default();
    for &start in starts {
        for probe in &fixture.probes {
            let mut probe = probe.clone();
            probe.sequence = [start; 6];
            work.push(&probe, probe.face_mask, 0, 0, false, u32::MAX, u32::MAX);
            work.probes.last_mut().unwrap().sky_requested = false;
        }
    }
    gpu.trace_compact(
        scene,
        &fixture.sources,
        &fixture.materials,
        &mut work,
        paths,
        &emitters,
        false,
    )
    .unwrap();
    let elapsed = gpu.last_gpu_ms.unwrap_or(gpu.last_dispatch_ms);
    let values = work
        .faces
        .chunks(fixture.probes.len())
        .map(|faces| faces.iter().map(moment).collect())
        .collect();
    (values, elapsed)
}

fn references(gpu: &mut GpuTracer, f: &Fixture, scene: Arc<TransportScene>) -> Vec<Vec<[f64; 5]>> {
    (0..2)
        .map(|replica| {
            let starts: Vec<_> = (0..4)
                .map(|block| 4_194_304 + replica * 1_048_576 + block * REFERENCE_BLOCK)
                .collect();
            let (blocks, _) = trace(gpu, f, scene.clone(), &starts, REFERENCE_BLOCK);
            let mut result = vec![[0.0; 5]; f.probes.len()];
            for block in blocks {
                for (sum, value) in result.iter_mut().zip(block) {
                    for (band, sum) in sum.iter_mut().enumerate() {
                        *sum += value["mean"][band].as_f64().unwrap() / 4.0;
                    }
                }
            }
            result
        })
        .collect()
}

/// Expand only the first batch after each event into its individual paths.
/// These are the same paths as the corresponding batch, not extra evidence.
/// Keep the rest aggregated to avoid a large single-path tape/readback.
fn individual_event_paths(
    gpu: &mut GpuTracer,
    f: &Fixture,
    scene: Arc<TransportScene>,
    seed: u32,
    repeated: bool,
) -> Value {
    let events: Vec<_> = (0..STEPS)
        .filter(|&step| step == 0 || (repeated && step % 8 == 0))
        .collect();
    let starts: Vec<_> = events
        .iter()
        .flat_map(|&step| (0..BATCH).map(move |path| seed + WARM + step * BATCH + path))
        .collect();
    let (samples, _) = trace(gpu, f, scene, &starts, 1);
    Value::Object(
        events
            .into_iter()
            .zip(samples.chunks(BATCH as usize))
            .map(|(step, paths)| (step.to_string(), json!(paths)))
            .collect(),
    )
}

fn record_case(
    gpu: &mut GpuTracer,
    f: &Fixture,
    before: Arc<TransportScene>,
    after: Arc<TransportScene>,
    name: &str,
    is_null: bool,
    repeated: bool,
) -> Value {
    let reference_before = references(gpu, f, before.clone());
    let reference_after = if Arc::ptr_eq(&before, &after) {
        reference_before.clone()
    } else {
        references(gpu, f, after.clone())
    };
    let mut runs = Vec::new();
    // Three disjoint live prefixes protect against a detector winning by a
    // lucky edit-time ray. References use entirely different sequence blocks.
    for seed in [0, 65_536, 131_072] {
        let (warm, _) = trace(gpu, f, before.clone(), &[seed], WARM);
        // Four preceding batches are an explicit, bounded replay bank. A
        // detector may inspect one or all four, charging every replay path.
        let starts: Vec<_> = (0..STEPS + 4)
            .map(|i| seed + WARM - 4 * BATCH + i * BATCH)
            .collect();
        let (old, old_ms) = trace(gpu, f, before.clone(), &starts, BATCH);
        let (new, new_ms) = if Arc::ptr_eq(&before, &after) {
            (old.clone(), old_ms)
        } else {
            trace(gpu, f, after.clone(), &starts, BATCH)
        };
        let individual_before = individual_event_paths(gpu, f, before.clone(), seed, repeated);
        let individual_after = if Arc::ptr_eq(&before, &after) {
            individual_before.clone()
        } else {
            individual_event_paths(gpu, f, after.clone(), seed, repeated)
        };
        runs.push(json!({"seed":seed,"warm":warm[0],"before":old,"after":new,
            "individual_before":individual_before,"individual_after":individual_after,
            "tape_gpu_ms":[old_ms,new_ms]}));
    }
    let sites: Vec<_> = f
        .probes
        .iter()
        .map(|p| json!({"voxel":p.voxel.to_array(),"face":p.face_mask.trailing_zeros()}))
        .collect();
    json!({"name":name,"null_change":is_null,"repeated":repeated,"sites":sites,
        "references_before":reference_before,"references_after":reference_after,"runs":runs})
}

#[test]
#[ignore = "native deterministic history-policy sample tape; no gameplay timing"]
fn record_history_qualification_tape() {
    let output = std::env::var("VOXEL_GI_HISTORY_TAPE").expect("set VOXEL_GI_HISTORY_TAPE");
    let mut gpu = super::super::tests::native_tracer();
    let mut cases = Vec::new();

    let cave = crate::voxel_gi::research::cave();
    let f = Fixture {
        name: "cave",
        scene: cave.scene,
        materials: cave.materials,
        sources: cave.sources,
        probes: cave.receivers.into_iter().step_by(3).collect(),
        analytic: None,
    };
    cases.push(record_case(
        &mut gpu,
        &f,
        f.scene.clone(),
        f.scene.clone(),
        "cave-null-streaming",
        true,
        true,
    ));

    let mut f = fixture(1);
    f.sources.environment = EnergySpectrum::ZERO;
    f.probes = f.probes.into_iter().step_by(4).collect();
    let before = f.scene.clone();
    let mut dark_page = page(&before);
    for lamp in [IVec3::new(8, 20, 10), IVec3::new(22, 20, 21)] {
        dark_page[linear_index(lamp.as_uvec3(), PAGE_SIZE)] = AIR;
    }
    let dark = changed(&before, dark_page);
    for (name, a, b) in [
        ("emitters-remove", before.clone(), dark.clone()),
        ("emitters-add", dark.clone(), before.clone()),
    ] {
        cases.push(record_case(&mut gpu, &f, a, b, name, false, false));
    }
    // A local lamp edit need not extinguish the entire room. Keep the other
    // source intact so disagreement is tested against a nonzero new target.
    let mut one_lamp = page(&before);
    one_lamp[linear_index(UVec3::new(8, 20, 10), PAGE_SIZE)] = AIR;
    cases.push(record_case(
        &mut gpu,
        &f,
        before.clone(),
        changed(&before, one_lamp),
        "one-emitter-remove",
        false,
        false,
    ));
    let mut covered = page(&before);
    for lamp in [IVec3::new(8, 20, 10), IVec3::new(22, 20, 21)] {
        for offset in crate::voxel_gi::world::FACE_OFFSETS {
            covered[linear_index((lamp + offset).as_uvec3(), PAGE_SIZE)] = 2;
        }
    }
    cases.push(record_case(
        &mut gpu,
        &f,
        before.clone(),
        changed(&before, covered),
        "emitters-occlude",
        false,
        false,
    ));

    // The room is sealed and absorbing: this remote emitter cannot contribute,
    // but it changes the emitter proposal/tree. Equal RNG is not automatically
    // equal sampled paths when a sampling distribution changes.
    let mut remote = vec![AIR; volume_len(PAGE_SIZE)];
    remote[linear_index(UVec3::splat(8), PAGE_SIZE)] = 4;
    let hidden = Arc::new(TransportScene::with_page_delta(
        &before,
        vec![(IVec3::X * 4, remote)],
        &[],
    ));
    cases.push(record_case(
        &mut gpu,
        &f,
        before,
        hidden,
        "hidden-emitter-null",
        true,
        true,
    ));

    let mut f = fixture(2);
    let mut roof = page(&f.scene);
    for z in 13..=17 {
        for x in 13..=17 {
            roof[linear_index(UVec3::new(x, 20, z), PAGE_SIZE)] = 2;
        }
    }
    let closed = changed(&f.scene, roof.clone());
    roof[linear_index(UVec3::new(16, 20, 16), PAGE_SIZE)] = AIR;
    let open = changed(&closed, roof);
    f.probes = [6, 10, 16, 21, 25]
        .into_iter()
        .flat_map(|z| {
            [6, 10, 16, 21, 25]
                .into_iter()
                .filter(move |&x| x != 16 || z != 16)
                .map(move |x| SurfaceProbe::new(IVec3::new(x, 20, z), 8))
        })
        .collect();
    for (name, a, b) in [
        ("skylight-open", closed.clone(), open.clone()),
        ("skylight-close", open, closed),
    ] {
        cases.push(record_case(&mut gpu, &f, a, b, name, false, false));
    }
    for case in &cases {
        eprintln!(
            "HISTORY_TAPE {} sites={}",
            case["name"],
            case["sites"].as_array().unwrap().len()
        );
    }
    let report = json!({"schema":2,"batch_paths":BATCH,"steps":STEPS,"warm_paths":WARM,
        "reference_paths_per_replica":4*REFERENCE_BLOCK,"cases":cases,
        "notes":"Frozen native scenes. Each run has four pre-edit replay batches plus 512 post-edit batches. Repeated null events every eight batches. Individual paths reproduce the first batch after each event, not extra samples. No streaming scheduler or FPS claim. Pair every promotion with real streamed validation."});
    std::fs::write(output, serde_json::to_vec(&report).unwrap()).unwrap();
}
