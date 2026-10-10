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
    samples: u32,
    sequence: u32,
    measured: [f32; 3],
    second_moment: [f32; 3],
    display: [f32; 3],
    confidence: f32,
    sky: [u32; 4],
    revisions: [u64; 3],
    fresh_mask: u8,
    stable_batches: u8,
    /// CPU mirror queued for rendering, NOT a GPU completion/readback claim.
    mirror: Option<[f32; 4]>,
    /// Shader-equivalent connected, confidence-weighted corner reconstruction
    /// from that mirror. None while the center is unpublished (shader prior).
    corners: Option<[[f32; 3]; 4]>,
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
pub(super) struct TemporalRun {
    enabled: Option<bool>,
    previous: Vec<Option<State>>,
    events: Vec<Event>,
    frames: usize,
    end_s: f32,
    overflow: bool,
}

fn sites() -> Vec<IVec3> {
    [-23, -22, -18, -14, -13]
        .into_iter()
        .flat_map(|z| {
            [-5, -4, 0, 4, 5]
                .into_iter()
                .map(move |x| IVec3::new(x, 17, z))
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
    pub fn record(
        &mut self,
        at_s: f32,
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
        let sites = sites();
        let patch: Vec<_> = sites
            .iter()
            .flat_map(|&voxel| {
                (-1..=1).flat_map(move |z| (-1..=1).map(move |x| (voxel + IVec3::new(x, 0, z), 3)))
            })
            .collect();
        let mirror = cache.validation_mirror(queue, &patch);
        let first = self.previous.is_empty();
        self.previous.resize(sites.len(), None);
        for (site, voxel) in sites.iter().enumerate() {
            let patch = &mirror[site * 9..site * 9 + 9];
            let state = field
                .probes
                .get(voxel)
                .filter(|p| p.face_mask & 8 != 0)
                .map(|p| State {
                    samples: p.samples[3],
                    sequence: p.sequence[3],
                    measured: p.radiance[3].visible.to_array(),
                    second_moment: p.second_moment[3].visible.to_array(),
                    display: p.display[3].indirect.to_array(),
                    confidence: p.display[3].confidence,
                    sky: p.display[3].sky,
                    revisions: [p.revision, p.source_revision, p.transport_revision],
                    fresh_mask: p.fresh_mask,
                    stable_batches: p.stable_batches[3],
                    mirror: (patch[4][3] != 0.0).then_some(patch[4]),
                    corners: corners(patch),
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
                "schema_version": 1, "frames": self.frames, "end_s": self.end_s,
                "sites": sites().iter().map(|v| v.to_array()).collect::<Vec<_>>(), "face": 3,
                "events": self.events,
                "notes": "Every Update after GI systems. Event-compressed fixed receivers; hold each state until next event. Mirror is CPU render-publication state, not GPU readback; corners exclude unpublished-center fallback and material/tonemapping. No convergence claim without independent reference."
            }),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
