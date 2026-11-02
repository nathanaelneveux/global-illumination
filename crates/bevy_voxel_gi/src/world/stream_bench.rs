//! Scene builds while streaming a declared cube of pages: one terrain page
//! per column, uniform stone below it and air above (a host that declares a
//! box, not only surface pages), in the scene task's 768-page batches.
use super::*;
use crate::input::PageSample;

const COLUMNS: i32 = 48;
const HEIGHT: i32 = 12;
const BATCH: usize = 768;

fn sample(page: IVec3) -> PageSample {
    let surface = 5 + (page.x * 7 + page.z * 3).rem_euclid(3) - 1;
    if page.y < surface {
        PageSample::coarse(PAGE_EDGE, vec![1])
    } else if page.y > surface {
        PageSample::air()
    } else {
        let mut cells = vec![AIR; volume_len(PAGE_SIZE)];
        for z in 0..PAGE_EDGE {
            for x in 0..PAGE_EDGE {
                let h = 8 + ((x * 3 + z * 5 + (page.x * 11 + page.z * 13) as u32) % 17);
                for y in 0..h {
                    cells[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] = 1 + (y % 3) as u16;
                }
            }
        }
        PageSample::full(cells)
    }
}

/// Streams the cube in scene-task batches: `old` takes 768 pages a build,
/// each expanded densely; otherwise uniform pages stay one value and only
/// dense pages count against the 768 (up to 8x as many sampled).
fn stream(old: bool) {
    let mut pages: Vec<IVec3> = (0..COLUMNS)
        .flat_map(|z| (0..COLUMNS).flat_map(move |x| (0..HEIGHT).map(move |y| IVec3::new(x, y, z))))
        .collect();
    let focus = IVec3::new(COLUMNS / 2, 5, COLUMNS / 2);
    pages.sort_unstable_by_key(|p| (*p - focus).as_i64vec3().length_squared());
    let mut scene = TransportScene::default();
    let (mut sample_ms, mut build_ms, mut changes_ms, mut batches) = (0.0, 0.0, 0.0, 0);
    let mut next_page = 0;
    while next_page < pages.len() {
        let started = Instant::now();
        let mut changed = Vec::new();
        let mut dense = 0;
        while next_page < pages.len() && changed.len() < BATCH * if old { 1 } else { 8 } {
            let page = pages[next_page];
            let sample = sample(page);
            let stride = sample.stride().trailing_zeros() as u8;
            let voxels = if old {
                PageVoxels::Dense(sample.into_dense())
            } else {
                sample.into_voxels()
            };
            if matches!(voxels, PageVoxels::Dense(_)) {
                if dense == BATCH {
                    break;
                }
                dense += 1;
            }
            changed.push((page, voxels, stride));
            next_page += 1;
        }
        let sampled = Instant::now();
        let next = Arc::new(TransportScene::with_page_contents(&scene, changed, &[]));
        let built = Instant::now();
        let _ = crate::geometry::SceneChanges::with_materials(&scene, &next, 64, None);
        let done = Instant::now();
        sample_ms += (sampled - started).as_secs_f64() * 1000.0;
        build_ms += (built - sampled).as_secs_f64() * 1000.0;
        changes_ms += (done - built).as_secs_f64() * 1000.0;
        batches += 1;
        scene = Arc::unwrap_or_clone(next);
    }
    eprintln!(
        "{}: {} pages in {batches} builds; total ms: sample {:.0}, regions+top {:.0}, changes {:.0}, all {:.0}",
        if old {
            "768 pages, dense"
        } else {
            "768 dense, uniform free"
        },
        pages.len(),
        sample_ms,
        build_ms,
        changes_ms,
        sample_ms + build_ms + changes_ms
    );
}

#[test]
#[ignore = "scene build benchmark; run with --release --nocapture"]
fn bench_streaming_a_declared_cube() {
    stream(true);
    stream(false);
}
