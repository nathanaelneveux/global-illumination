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
        PageSample::uniform(crate::GiVoxel::solid(1))
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
        PageSample::from_raw(1, cells)
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

/// One build's parts, in milliseconds.
fn time_build(
    scene: &TransportScene,
    changed: Vec<(IVec3, PageVoxels, u8)>,
    packed: &mut crate::gpu_scene::PackedScene,
) -> TransportScene {
    let started = Instant::now();
    let regions = scene.regions.clone();
    let cloned = started.elapsed();
    drop(regions);
    let started = Instant::now();
    let next = TransportScene::with_page_contents(scene, changed, &[]);
    let built = started.elapsed();
    let (fields, blocked) = next.top_level.fields_and_blocked();
    let started = Instant::now();
    let (top, quality) = if fields.is_empty() {
        (std::time::Duration::ZERO, f64::NAN)
    } else {
        let synchronous = next.top_level.synchronous();
        let top = started.elapsed();
        let quality = TopLevelAadf::mean_box_volume(fields, blocked)
            / TopLevelAadf::mean_box_volume(&synchronous, blocked);
        (top, quality)
    };
    next.top_level.assert_valid_fixed_point(&next.regions);
    let started = Instant::now();
    let _ = crate::geometry::SceneChanges::with_materials(scene, &next, 64, None);
    let changes = started.elapsed();
    let started = Instant::now();
    let _ = packed.update(&next);
    let pack = started.elapsed();
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
    eprintln!(
        "  build {:.1} (old synchronous top level {:.1}; mean box volume {:.3}x its own; region map clone {:.3}), changes {:.1}, pack {:.1}; shared field: {}",
        ms(built),
        ms(top),
        quality,
        ms(cloned),
        ms(changes),
        ms(pack),
        next.top_level.shares_field(&scene.top_level)
    );
    next
}

/// A far world (the declared cube at `columns`²), then single-page edits:
/// one inside solid ground (no topology change), one that fills a sky region,
/// and one page loaded past the edge (the bounds grow).
fn edit_far_world(columns: i32) {
    let mut pages: Vec<IVec3> = (0..columns)
        .flat_map(|z| (0..columns).flat_map(move |x| (0..HEIGHT).map(move |y| IVec3::new(x, y, z))))
        .collect();
    pages.sort_unstable_by_key(|p| (p.z, p.y, p.x));
    let mut scene = TransportScene::default();
    let started = Instant::now();
    for batch in pages.chunks(BATCH * 8) {
        let changed = batch
            .iter()
            .map(|&page| {
                let sample = sample(page);
                let stride = sample.stride().trailing_zeros() as u8;
                (page, sample.into_voxels(), stride)
            })
            .collect();
        scene = TransportScene::with_page_contents(&scene, changed, &[]);
    }
    eprintln!(
        "{columns}² columns: {} pages, {} regions, top level {:?}, loaded in {:.0} ms",
        pages.len(),
        scene.regions.len(),
        scene.top_level.size,
        started.elapsed().as_secs_f64() * 1000.0
    );
    let mut packed = crate::gpu_scene::PackedScene::default();
    let _ = packed.update(&scene);
    let dense = |page: IVec3, fill: u16| {
        let mut cells = vec![AIR; volume_len(PAGE_SIZE)];
        for (i, cell) in cells.iter_mut().enumerate() {
            *cell = if i % 7 == 0 { fill } else { AIR };
        }
        (page, PageVoxels::Dense(cells), 0)
    };
    let middle = columns / 2;
    eprintln!(" underground edit:");
    let uniform = |page: IVec3, material: u16| (page, PageVoxels::Uniform(material), 0);
    scene = time_build(
        &scene,
        vec![uniform(IVec3::new(middle, 1, middle), 2)],
        &mut packed,
    );
    eprintln!(" sky region filled:");
    scene = time_build(
        &scene,
        vec![dense(IVec3::new(middle, HEIGHT - 1, middle), 1)],
        &mut packed,
    );
    eprintln!(" page loaded past the edge:");
    let _ = time_build(
        &scene,
        vec![uniform(IVec3::new(columns, 0, middle), 1)],
        &mut packed,
    );
}

#[test]
#[ignore = "scene build benchmark; run with --release --nocapture"]
fn bench_edits_in_a_far_world() {
    for columns in [48, 96, 128] {
        edit_far_world(columns);
    }
}
