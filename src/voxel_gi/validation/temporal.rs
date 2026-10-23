//! Event-complete observations of a fixed receiver patch, after GI publication.
//! One transport result can be applied per Update. Unlike the broad 250 ms
//! sampler, observing every Update cannot skip accepted batches. Unchanged
//! states are run-length encoded; missing receivers are explicit, not zero.
use super::*;
use serde::Serialize;

const LIMIT: usize = 600_000;

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
    ) {
        self.record(at_s, Profile::Gameplay, field, runtime);
    }

    pub(super) fn record(
        &mut self,
        at_s: f32,
        profile: Profile,
        field: &VoxelEnergyField,
        runtime: &super::super::VoxelGiRuntime,
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
        let first = self.previous.is_empty();
        self.previous.resize(sites.len(), None);
        for (site, (voxel, face)) in sites.iter().enumerate() {
            let face = *face;
            let state = field
                .probes
                .get(voxel)
                .filter(|p| p.face_mask & (1 << face) != 0)
                .map(|p| State {
                    material: p.material,
                    face_mask: p.face_mask,
                    samples: p.stats().samples[face],
                    sequence: p.stats().sequence[face],
                    measured: p.stats().radiance[face].visible.to_array(),
                    second_moment: p.stats().second_moment[face].visible.to_array(),
                    display: p.display[face].indirect.to_array(),
                    confidence: p.display[face].confidence,
                    sky: p.display[face].sky,
                    revisions: [p.revision, p.source_revision, p.transport_revision],
                    retention_revision: p.retention_revision,
                    resident: p.resident[face].to_array(),
                    fresh_mask: p.fresh_mask,
                    stable_batches: p.stats().stable_batches[face],
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
                "schema_version": 3, "frames": self.frames, "end_s": self.end_s,
                "sites": self.sites.iter().map(|(v,_)| v.to_array()).collect::<Vec<_>>(),
                "site_faces": self.sites.iter().map(|(_,face)| face).collect::<Vec<_>>(),
                "events": self.events,
                "notes": "Every Update after GI systems. Event-compressed fixed receivers; hold each state until next event. World-cache receivers only; the view tier owns what is displayed, so use captures for presentation. No convergence claim without independent reference."
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
        run.record_pinned(0.0, &field, &runtime);
        let mut probe = SurfaceProbe::new(voxel, 4);
        probe.stats_mut().samples[2] = 2048;
        probe.stats_mut().sequence[2] = 8192;
        probe.resident[2] = UVec2::new(12, 1);
        field.probes.insert(voxel, probe);
        run.record_pinned(1.0, &field, &runtime);
        run.record_pinned(1.1, &field, &runtime);
        assert_eq!(run.events.len(), 2, "unchanged observations are compressed");
        field.probes.get_mut(&voxel).unwrap().resident[2].y = 2;
        run.record_pinned(2.0, &field, &runtime);
        let probe = field.probes.get_mut(&voxel).unwrap();
        probe.revision += 1;
        probe.stats_mut().samples[2] = 0;
        probe.stats_mut().sequence[2] = 0;
        run.record_pinned(3.0, &field, &runtime);
        field.probes.remove(&voxel);
        run.record_pinned(4.0, &field, &runtime);
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
}
