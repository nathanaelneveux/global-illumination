//! Fixed world-space receivers across all four walls, including the wall
//! beyond the near scheduling neighborhood. History is not convergence.
use super::*;

#[derive(Default)]
pub(super) struct CaveObservations {
    last_sample: f32,
    pub samples: Vec<serde_json::Value>,
}

impl CaveObservations {
    pub fn record(&mut self, elapsed: f32, field: &VoxelEnergyField) {
        if elapsed - self.last_sample < 0.25 {
            return;
        }
        self.last_sample = elapsed;
        let origin = if crate::cave_scene::enabled() {
            IVec3::ZERO
        } else {
            crate::structures::showcase::CAVE_ORIGIN
        };
        let mut faces = Vec::new();
        for t in (8..58).step_by(7) {
            for y in [9, 22, 37] {
                for (v, face) in [
                    (IVec3::new(4, y, t), 0),
                    (IVec3::new(59, y, t), 1),
                    (IVec3::new(t, y, 4), 4),
                    (IVec3::new(t, y, 59), 5),
                ] {
                    let voxel = origin + v;
                    let p = field
                        .probes
                        .get(&voxel)
                        .filter(|p| p.face_mask & (1 << face) != 0);
                    faces.push(json!({"voxel": voxel.to_array(), "face": face,
                        "samples": p.map(|p| p.samples[face]), "sequence": p.map(|p| p.sequence[face]),
                        "energy": p.map(|p| channels(p.radiance[face])),
                        "second_moment": p.map(|p| channels(p.second_moment[face])),
                        "display": p.map(|p| p.display[face].indirect.to_array()),
                        "age_ms": p.and_then(|p| p.face_updated_at[face]).map(|at| at.elapsed().as_secs_f32() * 1000.0),
                        "transport_revision": p.map(|p| p.transport_revision)}));
                }
            }
        }
        self.samples.push(json!({"at_s": elapsed, "transport_revision": field.transport_revision, "faces": faces}));
    }
}

fn channels(value: EnergySpectrum) -> [f32; 5] {
    [
        value.visible.x,
        value.visible.y,
        value.visible.z,
        value.infrared,
        value.ionizing,
    ]
}
