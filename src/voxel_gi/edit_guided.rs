//! Test-only edit-directed history diagnostics. These deterministic queries
//! classify history; their values must never enter the Monte Carlo estimator.
//! Fresh/reference transport still runs through the unchanged native GPU kernel.
use super::*;
use crate::voxel_gi::{geometry::GeometryHistory, geometry::SceneChanges, sky_cache::SkyCache};
use bevy::platform::collections::{HashMap, HashSet};
use serde_json::{Value, json};
use std::time::Instant;

fn bands(value: EnergySpectrum) -> [f32; 5] {
    [
        value.visible.x,
        value.visible.y,
        value.visible.z,
        value.infrared,
        value.ionizing,
    ]
}

#[derive(Default)]
struct Queries {
    segments: u32,
    unknown: u32,
}

impl Queries {
    fn trace(
        &mut self,
        scene: &TransportScene,
        origin: Vec3,
        direction: Vec3,
        reach: f32,
        policy: UnloadedPolicy,
    ) -> TraceResult {
        self.segments += 1;
        let result = scene.trace_ray_with_policy(origin, direction, reach, policy);
        if result == TraceResult::Unknown {
            self.unknown += 1;
        }
        result
    }

    /// Same analytic illumination convention as the production path tracer,
    /// with explicit segment accounting and unknown propagation for the audit.
    fn direct(
        &mut self,
        scene: &TransportScene,
        sources: &TransportSources,
        hit: RayHit,
    ) -> EnergySpectrum {
        let origin = hit.position + hit.normal * (2.0 * RAY_EPSILON);
        let mut result = EnergySpectrum::ZERO;
        for light in &sources.directional {
            let direction = light.direction.normalize_or_zero();
            let cosine = hit.normal.dot(direction).max(0.0);
            if cosine > 0.0
                && self.trace(
                    scene,
                    origin,
                    direction,
                    MAX_PATH_DISTANCE,
                    UnloadedPolicy::Conservative,
                ) == TraceResult::Escaped
            {
                result += light.radiance * cosine;
            }
        }
        for light in &sources.points {
            let delta = light.position - hit.position;
            let d2 = delta.length_squared();
            if d2 <= f32::EPSILON || d2 > light.range * light.range {
                continue;
            }
            let direction = delta / d2.sqrt();
            let cosine = hit.normal.dot(direction).max(0.0);
            if cosine > 0.0
                && self.trace(
                    scene,
                    origin,
                    direction,
                    d2.sqrt() - 4.0 * RAY_EPSILON,
                    UnloadedPolicy::Conservative,
                ) == TraceResult::Escaped
            {
                result += light.intensity * (cosine / d2.max(1.0));
            }
        }
        result
    }

    /// One aimed segment, plus analytic shadow queries at its first hit. No
    /// random light proposal, MIS/PDF change, or recursive diffuse bounce can
    /// masquerade as a temporal change in this diagnostic.
    fn incoming(
        &mut self,
        scene: &TransportScene,
        sources: &TransportSources,
        materials: &VoxelMaterialTransportTable,
        origin: Vec3,
        direction: Vec3,
    ) -> EnergySpectrum {
        match self.trace(
            scene,
            origin,
            direction,
            TransportSettings::default().reach,
            UnloadedPolicy::BoundedTransport,
        ) {
            TraceResult::Hit(hit) => {
                let material = materials.get(hit.material);
                material.emission + material.diffuse * self.direct(scene, sources, hit)
            }
            TraceResult::Escaped => {
                sources.environment * direction.y.mul_add(0.35, 0.65).clamp(0.15, 1.0)
            }
            TraceResult::Unknown => EnergySpectrum::ZERO,
        }
    }
}

/// Optical voxel edits, including the former position of a removed voxel.
/// Unknown/air residency alone is not tested by these frozen edit fixtures.
fn edit_targets(before: &TransportScene, after: &TransportScene) -> Vec<Vec3> {
    let regions: HashSet<_> = before
        .regions
        .keys()
        .chain(after.regions.keys())
        .copied()
        .collect();
    let mut changed = Vec::new();
    for region in regions {
        if before
            .regions
            .get(&region)
            .zip(after.regions.get(&region))
            .is_some_and(|(a, b)| Arc::ptr_eq(a, b))
        {
            continue;
        }
        let origin = region * REGION_EDGE as i32;
        for z in 0..REGION_EDGE as i32 {
            for y in 0..REGION_EDGE as i32 {
                for x in 0..REGION_EDGE as i32 {
                    let voxel = origin + IVec3::new(x, y, z);
                    if before.material_at(voxel).unwrap_or(AIR)
                        != after.material_at(voxel).unwrap_or(AIR)
                    {
                        changed.push(voxel);
                    }
                }
            }
        }
    }
    changed.sort_unstable_by_key(|v| (v.z, v.y, v.x));
    changed
        .into_iter()
        .map(|v| v.as_vec3() + Vec3::splat(0.5))
        .collect()
}

/// Reconstruct the existing shared-corner cache for all stable fixture surfaces,
/// not only the evaluated receivers. In gameplay these values already exist;
/// this study explicitly reports cold population and post-edit refresh work.
pub(crate) fn sun_targets(
    before: &Arc<TransportScene>,
    after: &Arc<TransportScene>,
    sources: &TransportSources,
) -> (Vec<Vec3>, usize, usize, usize) {
    if crate::voxel_gi::sky_cache::sky_lights(sources).is_empty() {
        return (Vec::new(), 0, 0, 0);
    }
    let (old, _) = crate::voxel_gi::world::rebuild_surface_probes(before, &HashMap::new());
    let (new, _) = crate::voxel_gi::world::rebuild_surface_probes(after, &HashMap::new());
    let mut receivers: Vec<_> = old
        .values()
        .filter_map(|a| {
            let b = new.get(&a.voxel)?;
            let mask = a.face_mask & b.face_mask;
            (mask != 0).then(|| SurfaceProbe::new(a.voxel, mask))
        })
        .collect();
    receivers.sort_unstable_by_key(|p| (p.voxel.z, p.voxel.y, p.voxel.x));
    let mut cache = SkyCache::default();
    let pending = cache.prepare(before, sources, &receivers);
    let cold_rays = cache.stats.rays;
    for position in pending {
        cache.store(position, sample_sky_corner(before, &cache.lights, position));
    }
    cache.apply(&mut receivers);
    let old = receivers.clone();
    let pending = cache.prepare(after, sources, &receivers);
    let update_rays = cache.stats.rays;
    for position in pending {
        cache.store(position, sample_sky_corner(after, &cache.lights, position));
    }
    cache.apply(&mut receivers);
    let mut targets = Vec::new();
    for (a, b) in old.iter().zip(&receivers) {
        for (face, normal) in FACE_OFFSETS.iter().enumerate() {
            if a.face_mask & (1 << face) == 0 {
                continue;
            }
            let center = a.voxel.as_vec3() + Vec3::splat(0.5) + normal.as_vec3() * 0.5;
            for corner in 0..4 {
                let x = a.display[face].sky[corner];
                let y = b.display[face].sky[corner];
                // This fully populated fixture query promises complete paired
                // evidence. Never certify a missing/budget-exhausted sample as
                // unchanged sunlight. Runtime partial caches need an explicit
                // unknown fallback instead of this test-only assertion.
                assert!(x != 0 && y != 0, "incomplete fixture sunlight evidence");
                if x != y {
                    let position =
                        crate::voxel_gi::sky::corner_position(a.voxel, face, corner).as_vec3();
                    // Aim just inside the unit face to avoid ambiguous edge hits.
                    targets.push(position.lerp(center, 0.01));
                }
            }
        }
    }
    (targets, cold_rays, update_rays, receivers.len())
}

fn origin_normal(probe: &SurfaceProbe) -> (Vec3, Vec3) {
    assert_eq!(
        probe.face_mask.count_ones(),
        1,
        "one evaluated face per fixture probe"
    );
    let normal = FACE_OFFSETS[probe.face_mask.trailing_zeros() as usize].as_vec3();
    (
        probe.voxel.as_vec3() + Vec3::splat(0.5) + normal * 0.502,
        normal,
    )
}

fn select_targets(probe: &SurfaceProbe, targets: &[Vec3], budget: usize) -> Vec<Vec3> {
    let (origin, normal) = origin_normal(probe);
    let mut ranked: Vec<_> = targets
        .iter()
        .filter_map(|&target| {
            let delta = target - origin;
            let distance = delta.length();
            let cosine = normal.dot(delta / distance);
            (distance > 0.004 && distance <= TransportSettings::default().reach && cosine > 1e-5)
                .then_some((cosine / (distance * distance), target))
        })
        .collect();
    ranked.sort_by(|(a, x), (b, y)| {
        b.total_cmp(a)
            .then_with(|| x.x.total_cmp(&y.x))
            .then_with(|| x.y.total_cmp(&y.y))
            .then_with(|| x.z.total_cmp(&y.z))
    });
    let mut selected = Vec::new();
    for (_, target) in ranked {
        // Distinct sub-face points can straddle a hard visibility boundary even
        // with almost identical directions. Only exact duplicates are redundant.
        if selected.contains(&target) {
            continue;
        }
        selected.push(target);
        if selected.len() == budget {
            break;
        }
    }
    selected
}

fn paired(
    probe: &SurfaceProbe,
    targets: &[Vec3],
    before: &TransportScene,
    after: &TransportScene,
    sources: &TransportSources,
    materials: &VoxelMaterialTransportTable,
) -> Value {
    let (origin, normal) = origin_normal(probe);
    let mut queries = Queries::default();
    let mut pairs = Vec::new();
    for target in targets {
        let direction = (*target - origin).normalize();
        let unknown_before = queries.unknown;
        let a =
            queries.incoming(before, sources, materials, origin, direction) * normal.dot(direction);
        let b =
            queries.incoming(after, sources, materials, origin, direction) * normal.dot(direction);
        pairs.push(
            json!({"old":bands(a),"new":bands(b),"known":queries.unknown==unknown_before,
            "target":target.to_array()}),
        );
    }
    json!({"pairs":pairs,"segments":queries.segments,"unknown":queries.unknown})
}

pub(crate) fn record(
    before: &Arc<TransportScene>,
    after: &Arc<TransportScene>,
    sources: &TransportSources,
    materials: &VoxelMaterialTransportTable,
    probes: &[SurfaceProbe],
) -> Value {
    let start = Instant::now();
    let edits = edit_targets(before, after);
    let change = SceneChanges::between(before, after).transport;
    let has_edit = change.is_some();
    let mut history = GeometryHistory::default();
    if let Some(change) = change {
        history.record(0, 1, change);
    }
    let baseline: Vec<_> = probes
        .iter()
        .map(|p| {
            if has_edit {
                history.history_limit(p.voxel, 0, 1)
            } else {
                u32::MAX
            }
        })
        .collect();
    let (sun, cold, update, surfaces) = if has_edit {
        sun_targets(before, after, sources)
    } else {
        (Vec::new(), 0, 0, 0)
    };
    let combined: Vec<_> = edits.iter().chain(&sun).copied().collect();
    let receivers: Vec<_> = probes
        .iter()
        .map(|p| {
            let geometry = select_targets(p, &edits, 1);
            let targets = select_targets(p, &combined, 4);
            json!({"geometry1":paired(p,&geometry,before,after,sources,materials),
            "combined1":paired(p,&targets[..targets.len().min(1)],before,after,sources,materials),
            "combined4":paired(p,&targets,before,after,sources,materials)})
        })
        .collect();
    json!({"has_edit":has_edit,"baseline_caps":baseline,"edit_targets":edits.len(),"sun_targets":sun.len(),
        "cold_sun_segments":cold,"updated_sun_segments":update,"sun_receiver_voxels":surfaces,
        "receivers":receivers,"record_cpu_ms":start.elapsed().as_secs_f64()*1000.0,
        "notes":"CPU NAADF diagnostic segments and actual shared-corner sunlight changes. Not MC samples. Cold surface/cache reconstruction is fixture setup, not a production implementation."})
}

#[test]
fn directed_targets_respect_the_receiving_hemisphere() {
    let p = SurfaceProbe::new(IVec3::ZERO, 4);
    let targets = [Vec3::new(0.5, 5.0, 0.5), Vec3::new(0.5, -5.0, 0.5)];
    assert_eq!(select_targets(&p, &targets, 1), vec![targets[0]]);
}

#[test]
fn nearby_targets_across_a_shadow_boundary_are_not_collapsed() {
    let p = SurfaceProbe::new(IVec3::ZERO, 4);
    let a = Vec3::new(0.5, 16.0, 0.5);
    let b = a + Vec3::X * 0.01;
    assert_eq!(select_targets(&p, &[a, a, b], 4), vec![a, b]);
}

#[test]
fn removed_voxel_keeps_its_edit_target_and_hidden_emission_has_no_gradient() {
    use crate::voxel_gi::{
        naadf::{linear_index, volume_len},
        world::PAGE_SIZE,
    };
    let mut page = vec![AIR; volume_len(PAGE_SIZE)];
    for z in 0..32 {
        for y in 0..32 {
            page[linear_index(UVec3::new(4, y, z), PAGE_SIZE)] = 2;
        }
    }
    page[linear_index(UVec3::new(2, 8, 8), PAGE_SIZE)] = 4;
    let old = Arc::new(TransportScene::with_page_delta(
        &TransportScene::default(),
        vec![(IVec3::ZERO, page.clone())],
        &[],
    ));
    page[linear_index(UVec3::new(2, 8, 8), PAGE_SIZE)] = AIR;
    let new = Arc::new(TransportScene::with_page_delta(
        &old,
        vec![(IVec3::ZERO, page)],
        &[],
    ));
    let targets = edit_targets(&old, &new);
    assert_eq!(targets, vec![Vec3::new(2.5, 8.5, 8.5)]);
    let mut materials = VoxelMaterialTransportTable::default();
    materials.set(
        4,
        crate::voxel_gi::world::MaterialTransport {
            emission: EnergySpectrum::ONE,
            ..default()
        },
    );
    let p = SurfaceProbe::new(IVec3::new(8, 8, 8), 2);
    let pair = paired(
        &p,
        &targets,
        &old,
        &new,
        &TransportSources::default(),
        &materials,
    );
    assert_eq!(pair["pairs"][0]["old"], pair["pairs"][0]["new"]);
    assert_eq!(pair["segments"], 2);
    assert_eq!(pair["pairs"][0]["known"], true);
    let visible = SurfaceProbe::new(IVec3::new(3, 8, 8), 2);
    let pair = paired(
        &visible,
        &targets,
        &old,
        &new,
        &TransportSources::default(),
        &materials,
    );
    assert_ne!(pair["pairs"][0]["old"], pair["pairs"][0]["new"]);
}

#[test]
fn diagnostic_analytic_illumination_matches_transport_in_every_band() {
    let scene = TransportScene::default();
    let sources = TransportSources {
        directional: vec![DirectionalTransportLight {
            direction: Vec3::Y,
            radiance: EnergySpectrum::ONE,
            custom_visible: Vec3::ZERO,
        }],
        points: vec![PointTransportLight {
            position: Vec3::Y * 3.0,
            intensity: EnergySpectrum::ONE * 4.0,
            range: 10.0,
            custom_visible: Vec3::ZERO,
        }],
        ..default()
    };
    let hit = RayHit {
        voxel: IVec3::ZERO,
        position: Vec3::Y,
        normal: Vec3::Y,
        material: 0,
        distance: 0.0,
    };
    let mut query = Queries::default();
    let result = query.direct(&scene, &sources, hit);
    assert_eq!(bands(result), [2.0; 5]);
    assert_eq!(
        result,
        direct_at_surface(&scene, &sources, hit.position, hit.normal).0
    );
    assert_eq!(query.segments, 2);
    assert_eq!(query.unknown, 0);
}

#[test]
fn missing_interior_space_is_not_evidence_of_unchanged_darkness() {
    use crate::voxel_gi::{naadf::volume_len, world::PAGE_SIZE};
    let scene = TransportScene::with_page_delta(
        &TransportScene::default(),
        vec![
            (IVec3::ZERO, vec![AIR; volume_len(PAGE_SIZE)]),
            (IVec3::X * 2, vec![2; volume_len(PAGE_SIZE)]),
        ],
        &[],
    );
    let receiver = SurfaceProbe::new(IVec3::splat(16), 1);
    let result = paired(
        &receiver,
        &[Vec3::new(70.0, 16.5, 16.5)],
        &scene,
        &scene,
        &TransportSources::default(),
        &VoxelMaterialTransportTable::default(),
    );
    assert_eq!(result["pairs"][0]["known"], false);
    assert_eq!(result["unknown"], 2);
}

#[test]
fn geometry_beyond_a_point_light_does_not_make_its_connection_unknown() {
    use crate::voxel_gi::{
        naadf::{linear_index, volume_len},
        world::PAGE_SIZE,
    };
    let mut page = vec![AIR; volume_len(PAGE_SIZE)];
    for z in 0..32 {
        for x in 0..32 {
            page[linear_index(UVec3::new(x, 24, z), PAGE_SIZE)] = 2;
        }
    }
    let scene =
        TransportScene::with_page_delta(&TransportScene::default(), vec![(IVec3::ZERO, page)], &[]);
    let sources = TransportSources {
        points: vec![PointTransportLight {
            position: Vec3::new(16.5, 12.5, 16.5),
            intensity: EnergySpectrum::ONE * 4.0,
            range: 10.0,
            custom_visible: Vec3::ZERO,
        }],
        ..default()
    };
    let hit = RayHit {
        voxel: IVec3::new(16, 8, 16),
        position: Vec3::new(16.5, 8.5, 16.5),
        normal: Vec3::Y,
        material: 0,
        distance: 0.0,
    };
    let mut query = Queries::default();
    assert_eq!(bands(query.direct(&scene, &sources, hit)), [0.25; 5]);
    assert_eq!(query.segments, 1);
    assert_eq!(query.unknown, 0);
}
