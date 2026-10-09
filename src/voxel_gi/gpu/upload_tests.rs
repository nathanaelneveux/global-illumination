//! Opt-in native geometry-upload comparisons. Both variants use the same rays;
//! the reference forces the former directory rebuild and full metadata uploads.

use super::*;
use crate::voxel_gi::{
    naadf::{linear_index, volume_len},
    world::{AIR, PAGE_SIZE},
};

fn compare_probes(a: &[SurfaceProbe], b: &[SurfaceProbe]) {
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.radiance, b.radiance);
        assert_eq!(a.second_moment, b.second_moment);
        assert_eq!(a.direct, b.direct);
        assert_eq!(a.custom_direct, b.custom_direct);
        assert_eq!(a.samples, b.samples);
        assert_eq!(a.sequence, b.sequence);
        for face in 0..6 {
            assert_eq!(a.display[face].sky, b.display[face].sky);
        }
    }
}

#[test]
#[ignore = "native edit upload timing; no concurrent workloads"]
fn sparse_geometry_upload_benchmark() {
    let mut candidate = super::tests::native_tracer();
    let mut reference = GpuTracer::new(candidate.context.clone()).unwrap();
    reference.full_geometry_uploads_for_benchmark = true;
    let page = vec![1; volume_len(PAGE_SIZE)];
    let empty = vec![AIR; volume_len(PAGE_SIZE)];
    let pages = (0..16)
        .flat_map(|z| {
            let page = &page;
            let empty = &empty;
            (0..4).flat_map(move |y| {
                (0..16).map(move |x| {
                    (
                        IVec3::new(x, y, z),
                        if y == 0 { page.clone() } else { empty.clone() },
                    )
                })
            })
        })
        .collect();
    let base = Arc::new(TransportScene::with_page_delta(
        &TransportScene::default(),
        pages,
        &[],
    ));
    let mut page = page;
    page[linear_index(UVec3::new(8, 31, 8), PAGE_SIZE)] = AIR;
    let edited = Arc::new(TransportScene::with_page_delta(
        &base,
        vec![(IVec3::ZERO, page)],
        &[],
    ));
    let sources = TransportSources::default();
    let materials = VoxelMaterialTransportTable::default();
    let scenes = [base, edited];
    let emitters = scenes
        .each_ref()
        .map(|s| Arc::new(EmissionDistribution::build(s, &materials)));
    let probes: Vec<_> = (12..20)
        .flat_map(|z| (12..20).map(move |x| SurfaceProbe::new(IVec3::new(x, 31, z), 4)))
        .collect();
    let mut prepare = [Vec::new(), Vec::new()];
    let mut worker = [Vec::new(), Vec::new()];
    let mut uploaded = [Vec::new(), Vec::new()];
    for pair in 0..44 {
        let scene_index = pair % 2;
        let mut outputs = [probes.clone(), probes.clone()];
        for mode in [pair % 2, 1 - pair % 2] {
            let tracer = if mode == 0 {
                &mut reference
            } else {
                &mut candidate
            };
            let before = tracer.uploaded_geometry_bytes;
            let started = Instant::now();
            tracer
                .trace(
                    scenes[scene_index].clone(),
                    &sources,
                    &materials,
                    &mut outputs[mode],
                    4,
                    &emitters[scene_index],
                )
                .unwrap();
            if pair >= 4 {
                worker[mode].push(started.elapsed().as_secs_f32() * 1e3);
                prepare[mode].push(tracer.last_prepare_ms);
                uploaded[mode].push(tracer.uploaded_geometry_bytes - before);
            }
        }
        compare_probes(&outputs[0], &outputs[1]);
    }
    for mode in 0..2 {
        prepare[mode].sort_by(f32::total_cmp);
        worker[mode].sort_by(f32::total_cmp);
        uploaded[mode].sort_unstable();
        eprintln!(
            "GEOMETRY_UPLOAD mode={} regions={} receivers={} pairs=40 prepare_ms_p50/p95={:.3}/{:.3} worker_ms_p50/p95={:.3}/{:.3} bytes_min/max={}/{} exact=true",
            ["full", "delta"][mode],
            scenes[0].regions.len(),
            probes.len(),
            prepare[mode][20],
            prepare[mode][37],
            worker[mode][20],
            worker[mode][37],
            uploaded[mode][0],
            uploaded[mode][39]
        );
    }
}

#[test]
#[ignore = "native sparse upload correctness through buffer growth and recycling"]
fn sparse_geometry_uploads_survive_growth_and_residency_changes() {
    let mut candidate = super::tests::native_tracer();
    let mut reference = GpuTracer::new(candidate.context.clone()).unwrap();
    reference.full_geometry_uploads_for_benchmark = true;
    let materials = VoxelMaterialTransportTable::default();
    let sources = TransportSources::default();
    let mut page = vec![AIR; volume_len(PAGE_SIZE)];
    for z in 0..32 {
        for x in 0..32 {
            page[linear_index(UVec3::new(x, 4, z), PAGE_SIZE)] = 1;
        }
    }
    let mut scene = Arc::new(TransportScene::default());
    for stage in 0..6 {
        let (changed, removed) = match stage {
            0 => (vec![(IVec3::ZERO, page.clone())], vec![]),
            1 => (
                (1..16)
                    .map(|x| (IVec3::new(x, 0, 0), page.clone()))
                    .collect(),
                vec![],
            ),
            2 => (vec![], (1..16).map(|x| IVec3::new(x, 0, 0)).collect()),
            3 => (vec![(IVec3::new(-1, 0, 0), page.clone())], vec![]),
            4 => {
                page[linear_index(UVec3::new(8, 4, 8), PAGE_SIZE)] = AIR;
                (vec![(IVec3::ZERO, page.clone())], vec![])
            }
            _ => (
                vec![(IVec3::ZERO, vec![AIR; volume_len(PAGE_SIZE)])],
                vec![],
            ),
        };
        scene = Arc::new(TransportScene::with_page_delta(&scene, changed, &removed));
        let emitters = Arc::new(EmissionDistribution::build(&scene, &materials));
        let initial: Vec<_> = (12..20)
            .map(|x| SurfaceProbe::new(IVec3::new(x, 4, 8), 4))
            .collect();
        let mut outputs = [initial.clone(), initial];
        reference
            .trace(
                scene.clone(),
                &sources,
                &materials,
                &mut outputs[0],
                4,
                &emitters,
            )
            .unwrap();
        candidate
            .trace(
                scene.clone(),
                &sources,
                &materials,
                &mut outputs[1],
                4,
                &emitters,
            )
            .unwrap();
        compare_probes(&outputs[0], &outputs[1]);
    }
    eprintln!(
        "Native geometry deltas agree through growth, removal, recycled headers, negative coordinates and empty regions"
    );
}
