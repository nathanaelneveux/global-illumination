//! Opt-in native geometry-upload comparisons. Both variants use the same rays;
//! the reference forces the former directory rebuild and full metadata uploads.

use super::*;
use crate::{
    naadf::{linear_index, volume_len},
    world::{AIR, PAGE_SIZE},
};

pub(super) fn compare_probes(a: &[SurfaceProbe], b: &[SurfaceProbe]) {
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.stats().radiance, b.stats().radiance);
        assert_eq!(a.stats().second_moment, b.stats().second_moment);
        assert_eq!(a.stats().direct, b.stats().direct);
        assert_eq!(a.stats().custom_direct, b.stats().custom_direct);
        assert_eq!(a.stats().samples, b.stats().samples);
        assert_eq!(a.stats().sequence, b.stats().sequence);
        for face in 0..6 {
            assert_eq!(a.display[face].sky, b.display[face].sky);
        }
    }
}

#[test]
#[ignore = "native sparse upload correctness through buffer growth and recycling"]
fn sparse_geometry_uploads_survive_growth_and_residency_changes() {
    let mut candidate = super::tests::native_tracer();
    let mut reference = GpuTracer::new(candidate.context.clone()).unwrap();
    reference.full_geometry_uploads_for_benchmark = true;
    reference.packed.revisit_unchanged_for_benchmark = true;
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
