//! Task serialization oracle and equal-work native comparison. The generic
//! ShaderType encoder is retained only here, not as a production mode.
use super::*;
use crate::voxel_gi::{
    naadf::{linear_index, volume_len},
    world::{AIR, PAGE_SIZE},
};

pub(super) fn legacy_task_bytes(work: &TraceWork, sky: &[IVec3]) -> Vec<u8> {
    let mut tasks: Vec<_> = work
        .faces
        .iter()
        .map(|face| GpuFaceTask {
            voxel: face.voxel,
            face: face.face as u32,
            sequence: UVec4::new(face.sequence, 0, 0, 0),
        })
        .collect();
    tasks.extend(sky.iter().map(|&voxel| GpuFaceTask { voxel, ..default() }));
    ShaderBuffer::from(tasks).data.unwrap()
}

#[test]
fn task_encoding_matches_shader_layout_and_clears_reused_tails() {
    let mut probes = vec![SurfaceProbe::new(IVec3::new(i32::MIN, -17, i32::MAX), 63)];
    probes[0].sequence = [0, 1, u32::MAX, 65536, 0x8000_0000, 12345];
    let work = TraceWork::from_probes(&probes, false);
    let sky = [IVec3::new(-16, 0, 32), IVec3::splat(i32::MIN)];
    let mut bytes = Vec::new();
    for (work, sky) in [
        (&work, &sky[..]),
        (&work, &[][..]),
        (&TraceWork::default(), &sky[..]),
        (&TraceWork::default(), &[][..]),
        (&work, &sky[..1]),
    ] {
        encode_tasks(&mut bytes, work, sky);
        assert_eq!(bytes.len(), (work.faces.len() + sky.len()) * TASK_BYTES);
        if bytes.is_empty() {
            // Encase pads an empty runtime array to one element. Empty work
            // returns before task upload, so no dummy element is needed here.
            assert_eq!(legacy_task_bytes(work, sky), vec![0; TASK_BYTES]);
        } else {
            assert_eq!(bytes, legacy_task_bytes(work, sky));
        }
    }
}

#[test]
#[ignore = "native task upload timing; no concurrent workloads"]
fn task_encoding_benchmark() {
    let mut candidate = super::tests::native_tracer();
    let mut baseline = GpuTracer::new(candidate.context.clone()).unwrap();
    baseline.legacy_task_encoding_for_benchmark = true;
    let mut page = vec![AIR; volume_len(PAGE_SIZE)];
    for z in 0..32 {
        for x in 0..32 {
            page[linear_index(UVec3::new(x, 4, z), PAGE_SIZE)] = 2;
        }
    }
    let scene = Arc::new(TransportScene::with_page_delta(
        &TransportScene::default(),
        (0..8)
            .map(|x| (IVec3::new(x, 0, 0), page.clone()))
            .collect(),
        &[],
    ));
    let sources = super::tests::open_floor_fixture().1;
    let materials = VoxelMaterialTransportTable::default();
    let emitters = Arc::new(EmissionDistribution::default());
    let initial: Vec<_> = (0..32)
        .flat_map(|z| (0..256).map(move |x| SurfaceProbe::new(IVec3::new(x, 4, z), 4)))
        .collect();
    for count in [17, 1024, 8192] {
        let mut prepare = [Vec::new(), Vec::new()];
        let mut worker = [Vec::new(), Vec::new()];
        for pair in 0..44 {
            let mut outputs = [initial[..count].to_vec(), initial[..count].to_vec()];
            for mode in (0..2).map(|offset| (pair + offset) % 2) {
                let gpu = if mode == 0 {
                    &mut baseline
                } else {
                    &mut candidate
                };
                for p in &mut outputs[mode] {
                    p.sequence = [(pair * 4) as u32; 6];
                }
                let mut work = TraceWork::from_probes(&outputs[mode], false);
                // Include cold mixed sky+GI work in correctness checks, then
                // time the common indirect-only path without cached lookups.
                for job in &mut work.probes {
                    job.sky_requested = pair == 0;
                }
                let start = Instant::now();
                gpu.trace_compact(
                    scene.clone(),
                    &sources,
                    &materials,
                    &mut work,
                    4,
                    &emitters,
                    false,
                )
                .unwrap();
                if pair >= 4 {
                    worker[mode].push(start.elapsed().as_secs_f32() * 1e3);
                    prepare[mode].push(gpu.last_prepare_ms);
                }
                work.copy_measurements(&mut outputs[mode]);
            }
            super::upload_tests::compare_probes(&outputs[0], &outputs[1]);
        }
        for mode in 0..2 {
            prepare[mode].sort_by(f32::total_cmp);
            worker[mode].sort_by(f32::total_cmp);
            eprintln!(
                "TASK_ENCODING mode={} faces={} pairs=40 prepare_ms_p50/p95={:.4}/{:.4} worker_ms_p50/p95={:.4}/{:.4} exact=true",
                ["generic", "direct"][mode],
                count,
                prepare[mode][20],
                prepare[mode][37],
                worker[mode][20],
                worker[mode][37]
            );
        }
    }
}
