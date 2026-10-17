//! Event-complete observations of a fixed receiver patch, after GI publication.
//! One transport result can be applied per Update. Unlike the broad 250 ms
//! sampler, observing every Update cannot skip accepted batches. Unchanged
//! states are run-length encoded; missing receivers are explicit, not zero.
use super::super::render::{GpuProbePatchQueue, GpuTransportCache};
use super::*;
use serde::Serialize;

const LIMIT: usize = 200_000;

#[derive(Clone, PartialEq, Serialize)]
struct State {
    material: u8,
    face_mask: u8,
    samples: u32,
    sequence: u32,
    measured: [f32; 3],
    second_moment: [f32; 3],
    display: [f32; 3],
    confidence: f32,
    sky: [u32; 4],
    revisions: [u64; 3],
    retention_revision: u64,
    resident: [u32; 2],
    fresh_mask: u8,
    stable_batches: u8,
    /// CPU queued publication or latest completed resident presentation snapshot.
    /// Neither is a claim that the currently rendered frame has this value.
    mirror: Option<[f32; 4]>,
    /// Shader-equivalent connected, confidence-weighted corner reconstruction
    /// from that mirror. None while the center is unpublished (shader prior).
    corners: Option<[[f32; 3]; 4]>,
    corner_inputs: [[f32; 4]; 9],
}

#[derive(Serialize)]
struct Event {
    at_s: f32,
    site: usize,
    state: Option<State>,
    transport_revision: u64,
    generation: u64,
}

#[derive(Default)]
pub(crate) struct TemporalRun {
    enabled: Option<bool>,
    previous: Vec<Option<State>>,
    events: Vec<Event>,
    frames: usize,
    end_s: f32,
    overflow: bool,
    sites: Vec<(IVec3, usize)>,
}

pub(super) fn sites(profile: Profile) -> Vec<(IVec3, usize)> {
    if profile == Profile::LiveHallway {
        // Dense parallel floor rows expose spatial error and page boundaries,
        // not just the original logarithmic near-edit distance anchors.
        return (1..80).flat_map(|x| [-2, 0, 2].map(|z| {
            (crate::structures::hallway::floor_receiver(x) + IVec3::Z * z, 2)
        })).collect();
    }
    if matches!(profile, Profile::Cave | Profile::Structures | Profile::LiveCave) {
        let origin = if crate::cave_scene::enabled() {
            IVec3::ZERO
        } else {
            crate::structures::showcase::CAVE_ORIGIN
        };
        // Keep the west-wall receivers clear of the normal world's staircase.
        // The isolated chamber has no stairs, so checking it alone is insufficient.
        return [8, 20, 50]
            .into_iter()
            .flat_map(|t| {
                [9, 22, 37].into_iter().flat_map(move |y| {
                    [
                        (IVec3::new(4, y, t), 0),
                        (IVec3::new(59, y, t), 1),
                        (IVec3::new(t, y, 4), 4),
                        (IVec3::new(t, y, 59), 5),
                    ]
                    .map(|(v, face)| (origin + v, face))
                })
            })
            .collect();
    }
    [-23, -22, -18, -14, -13]
        .into_iter()
        .flat_map(|z| {
            [-5, -4, 0, 4, 5]
                .into_iter()
                .map(move |x| (IVec3::new(x, 17, z), 3))
        })
        .collect()
}

fn corners(patch: &[[f32; 4]]) -> Option<[[f32; 3]; 4]> {
    let center = Vec4::from_array(patch[4]);
    if center.w == 0.0 {
        return None;
    }
    Some(
        [(0, 1, 3), (2, 1, 5), (6, 7, 3), (8, 7, 5)].map(|(diagonal, a, b)| {
            let a = Vec4::from_array(patch[a]);
            let b = Vec4::from_array(patch[b]);
            let diagonal = if a.w > 0.0 || b.w > 0.0 {
                Vec4::from_array(patch[diagonal])
            } else {
                Vec4::ZERO
            };
            ((center.truncate() * center.w
                + a.truncate() * a.w
                + b.truncate() * b.w
                + diagonal.truncate() * diagonal.w)
                / (center.w + a.w + b.w + diagonal.w).max(0.000001))
            .to_array()
        }),
    )
}

impl TemporalRun {
    /// Pin a normal-gameplay receiver without changing sampling, camera,
    /// scene residency or validation mode. The same recorder follows it even
    /// when it unloads or is replaced; missing state remains explicit.
    pub(crate) fn pinned(voxel: IVec3, face: usize) -> Self {
        Self {
            enabled: Some(true),
            sites: vec![(voxel, face)],
            ..default()
        }
    }

    pub(crate) fn record_pinned(
        &mut self,
        at_s: f32,
        field: &VoxelEnergyField,
        runtime: &super::super::VoxelGiRuntime,
        cache: &GpuTransportCache,
        queue: &GpuProbePatchQueue,
    ) {
        self.record(at_s, Profile::Gameplay, field, runtime, cache, queue);
    }

    pub(super) fn record(
        &mut self,
        at_s: f32,
        profile: Profile,
        field: &VoxelEnergyField,
        runtime: &super::super::VoxelGiRuntime,
        cache: &GpuTransportCache,
        queue: &GpuProbePatchQueue,
    ) {
        if !*self
            .enabled
            .get_or_insert_with(|| std::env::var_os("VOXEL_GI_VALIDATE_TEMPORAL").is_some())
        {
            return;
        }
        self.frames += 1;
        self.end_s = at_s;
        if self.sites.is_empty() {
            self.sites = sites(profile);
        }
        let sites = &self.sites;
        let patch: Vec<_> = sites
            .iter()
            .flat_map(|&(voxel, face)| {
                (-1..=1).flat_map(move |v| {
                    (-1..=1).map(move |u| {
                        let offset = match face / 2 {
                            0 => IVec3::new(0, u, v),
                            1 => IVec3::new(u, 0, v),
                            _ => IVec3::new(u, v, 0),
                        };
                        (voxel + offset, face)
                    })
                })
            })
            .collect();
        let mirror = cache.validation_snapshot(queue, field, &patch);
        let first = self.previous.is_empty();
        self.previous.resize(sites.len(), None);
        for (site, (voxel, face)) in sites.iter().enumerate() {
            let face = *face;
            let patch = &mirror[site * 9..site * 9 + 9];
            let state = field
                .probes
                .get(voxel)
                .filter(|p| p.face_mask & (1 << face) != 0)
                .map(|p| State {
                    material: p.material,
                    face_mask: p.face_mask,
                    samples: p.samples[face],
                    sequence: p.sequence[face],
                    measured: p.radiance[face].visible.to_array(),
                    second_moment: p.second_moment[face].visible.to_array(),
                    display: p.display[face].indirect.to_array(),
                    confidence: p.display[face].confidence,
                    sky: p.display[face].sky,
                    revisions: [p.revision, p.source_revision, p.transport_revision],
                    retention_revision: p.retention_revision,
                    resident: p.resident[face].to_array(),
                    fresh_mask: p.fresh_mask,
                    stable_batches: p.stable_batches[face],
                    mirror: (patch[4][3] != 0.0).then_some(patch[4]),
                    corners: corners(patch),
                    corner_inputs: patch.try_into().expect("nine-candidate corner patch"),
                });
            if first || state != self.previous[site] {
                if self.events.len() < LIMIT {
                    self.events.push(Event {
                        at_s,
                        site,
                        state: state.clone(),
                        transport_revision: field.transport_revision,
                        generation: runtime.generation,
                    });
                } else {
                    self.overflow = true;
                }
                self.previous[site] = state;
            }
        }
    }

    pub fn save(&self, output: &std::path::Path) -> std::io::Result<()> {
        if self.enabled != Some(true) {
            return Ok(());
        }
        if self.overflow {
            return Err(std::io::Error::other(
                "temporal recorder capacity exceeded; sequence is incomplete",
            ));
        }
        let file = std::fs::File::create(output.join("temporal.json"))?;
        serde_json::to_writer(
            std::io::BufWriter::new(file),
            &json!({
                "schema_version": 2, "frames": self.frames, "end_s": self.end_s,
                "sites": self.sites.iter().map(|(v,_)| v.to_array()).collect::<Vec<_>>(),
                "site_faces": self.sites.iter().map(|(_,face)| face).collect::<Vec<_>>(),
                "events": self.events,
                "notes": "Every Update after GI systems. Event-compressed fixed receivers; hold each state until next event. Mirror is CPU queued publication or latest completed GPU-resident presentation snapshot, potentially older than the rendered frame. Corners exclude unpublished-center fallback and material/tonemapping. No convergence claim without independent reference."
            }),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pinned_watch_keeps_missing_replaced_and_address_only_transitions() {
        use super::super::super::world::SurfaceProbe;
        let voxel = IVec3::new(-54, -12, -20);
        let mut run = TemporalRun::pinned(voxel, 2);
        let mut field = VoxelEnergyField::default();
        let runtime = super::super::super::VoxelGiRuntime::default();
        let cache = GpuTransportCache::default();
        let queue = GpuProbePatchQueue::unpublished_test_queue();
        run.record_pinned(0.0, &field, &runtime, &cache, &queue);
        let mut probe = SurfaceProbe::new(voxel, 4);
        probe.samples[2] = 2048;
        probe.sequence[2] = 8192;
        probe.resident[2] = UVec2::new(12, 1);
        field.probes.insert(voxel, probe);
        run.record_pinned(1.0, &field, &runtime, &cache, &queue);
        run.record_pinned(1.1, &field, &runtime, &cache, &queue);
        assert_eq!(run.events.len(), 2, "unchanged observations are compressed");
        field.probes.get_mut(&voxel).unwrap().resident[2].y = 2;
        run.record_pinned(2.0, &field, &runtime, &cache, &queue);
        let probe = field.probes.get_mut(&voxel).unwrap();
        probe.revision += 1;
        probe.samples[2] = 0;
        probe.sequence[2] = 0;
        run.record_pinned(3.0, &field, &runtime, &cache, &queue);
        field.probes.remove(&voxel);
        run.record_pinned(4.0, &field, &runtime, &cache, &queue);
        assert_eq!(run.events.len(), 5);
        assert!(run.events[0].state.is_none());
        assert_eq!(run.events[1].state.as_ref().unwrap().sequence, 8192);
        assert_eq!(run.events[2].state.as_ref().unwrap().resident, [12, 2]);
        assert_eq!(run.events[3].state.as_ref().unwrap().sequence, 0);
        assert!(run.events[4].state.is_none());
    }

    #[test]
    fn live_hallway_sites_cover_contiguous_exposed_floor_across_pages() {
        let layer =
            crate::structures::StructureLayer::new(crate::structures::hallway::instances(), 32);
        let sites = sites(Profile::LiveHallway);
        assert_eq!(sites.len(), 237);
        for (voxel, face) in sites {
            assert_eq!(face, 2);
            assert!(matches!(
                layer.sample(voxel),
                crate::structures::VoxelOp::Solid(_)
            ));
            assert_eq!(
                layer.sample(voxel + IVec3::Y),
                crate::structures::VoxelOp::Air
            );
        }
    }

    #[test]
    fn cave_sites_are_exposed_and_house_sites_keep_the_same_definition() {
        let house = sites(Profile::House);
        assert_eq!(house.len(), 25);
        assert!(house.iter().all(|(v, face)| v.y == 17 && *face == 3));
        let origin = if crate::cave_scene::enabled() {
            IVec3::ZERO
        } else {
            crate::structures::showcase::CAVE_ORIGIN
        };
        let cave = sites(Profile::Cave);
        let structures = crate::structures::showcase::world_structures();
        assert_eq!(cave.len(), 36);
        for (voxel, face) in cave {
            let local = voxel - origin;
            assert!(matches!(
                crate::cave_scene::voxel(local),
                WorldVoxel::Solid(_)
            ));
            assert_eq!(
                crate::cave_scene::voxel(local + FACE_OFFSETS[face]),
                WorldVoxel::Air
            );
            let world = local + crate::structures::showcase::CAVE_ORIGIN;
            assert!(matches!(
                structures.sample(world),
                crate::structures::VoxelOp::Solid(_)
            ));
            assert_eq!(
                structures.sample(world + FACE_OFFSETS[face]),
                crate::structures::VoxelOp::Air
            );
        }
    }
    #[test]
    fn mirrored_corners_preserve_constants_and_disconnect_diagonals() {
        assert_eq!(
            corners(&[[2.0, 3.0, 4.0, 0.125]; 9]),
            Some([[2.0, 3.0, 4.0]; 4])
        );
        let mut patch = [[0.0; 4]; 9];
        assert!(corners(&patch).is_none());
        patch[4] = [1.0; 4];
        patch[0] = [100.0, 100.0, 100.0, 1.0];
        assert_eq!(corners(&patch), Some([[1.0; 3]; 4]));
        patch[1] = [1.0; 4];
        assert_eq!(corners(&patch).unwrap()[0], [34.0; 3]);
    }
}
