//! Day cycle: the canyon under a slowly rotating sun. Records how much
//! history the moving sun leaves and what it costs; the paired captures show
//! flicker. Sample counts are the CPU mirror: run with
//! `config/gi-mirror-full.ron` when they must be current.
use super::*;
use crate::world_environment::SunAngle;

pub(super) struct DayRun {
    /// Degrees per second about the world Z axis.
    rate: f32,
    /// Starting angle in degrees (a fixed-sun reference uses rate 0).
    offset: f32,
    /// Quantize the angle to whole steps (degrees); 0 is continuous.
    step: f32,
    angle: Option<f32>,
    last_sample: f32,
    pub samples: Vec<serde_json::Value>,
}

impl Default for DayRun {
    fn default() -> Self {
        let env = |name: &str, default: f32| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .filter(|value| value.is_finite())
                .unwrap_or(default)
        };
        Self {
            rate: env("VOXEL_GI_VALIDATE_SUN_RATE", 0.3),
            offset: env("VOXEL_GI_VALIDATE_SUN_OFFSET", 0.0),
            step: env("VOXEL_GI_VALIDATE_SUN_STEP", 0.0).max(0.0),
            angle: None,
            last_sample: -1.0,
            samples: Vec::new(),
        }
    }
}

impl DayRun {
    pub fn settings(&self) -> serde_json::Value {
        json!({"rate_deg_per_s": self.rate, "offset_deg": self.offset, "step_deg": self.step,
            "axis": "world Z, from the canyon sun"})
    }

    pub fn tick(
        &mut self,
        elapsed: f32,
        commands: &mut Commands,
        field: &VoxelEnergyField,
        stats: &VoxelGiStats,
        runtime: &super::super::VoxelGiRuntime,
    ) {
        let mut angle = self.offset + self.rate * elapsed;
        if self.step > 0.0 {
            angle = self.offset + ((angle - self.offset) / self.step).floor() * self.step;
        }
        if self.angle != Some(angle) {
            self.angle = Some(angle);
            commands.insert_resource(SunAngle(angle));
        }
        if elapsed - self.last_sample < 1.0 {
            return;
        }
        self.last_sample = elapsed;
        self.samples.push(json!({"at_s": elapsed, "angle_deg": angle,
            "traced_paths": stats.traced_paths, "sky_corners_traced": stats.sky_corners_traced,
            "sky_corner_requests": stats.sky_corner_requests, "sky_rays": stats.sky_rays_scheduled,
            "transport_batches": stats.transport_batches, "sky_only_batches": stats.sky_only_batches,
            "batch_size": runtime.batch_size, "sky_batch_size": runtime.sky_batch_size,
            "gpu_ms_per_frame": stats.gpu_ms_per_frame, "trace_batch_ms": stats.last_trace_batch_ms,
            "source_revision": field.source_revision, "transport_revision": field.transport_revision}));
    }
}
