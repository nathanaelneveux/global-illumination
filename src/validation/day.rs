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

/// `VOXEL_GI_VALIDATE_DAY_TURN=1`: look away from the canyon from 60 s to
/// 120 s while the sun moves, then back. Faces returning to view under moved
/// shadows must cut the bounce history they kept (compare the return
/// captures with a fixed-sun reference at the return angle).
pub(super) fn turn() -> bool {
    static TURN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TURN.get_or_init(|| std::env::var_os("VOXEL_GI_VALIDATE_DAY_TURN").is_some())
}

pub(super) const TURN_CAPTURES: &[(f32, &str)] = &[
    (59.0, "day-before-turn"),
    (120.4, "return-0400ms"),
    (121.0, "return-1000ms"),
    (123.0, "return-3000ms"),
    (130.0, "return-10s"),
    (180.0, "day-180s"),
];

/// Looking away (yaw 180 degrees about the eye) during the turn.
pub(super) fn looking_away(elapsed: f32) -> bool {
    turn() && (60.0..120.0).contains(&elapsed)
}

impl DayRun {
    pub fn settings(&self) -> serde_json::Value {
        json!({"rate_deg_per_s": self.rate, "offset_deg": self.offset, "step_deg": self.step,
            "axis": "world Z, from the canyon sun"})
    }

    pub fn tick(&mut self, elapsed: f32, commands: &mut Commands, stats: &VoxelGiStats) {
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
            "traced_paths": stats.scheduler.traced_paths, "transport_batches": stats.scheduler.transport_batches,
            "gpu_ms_per_frame": stats.gpu_ms_per_frame, "trace_batch_ms": stats.scheduler.last_trace_batch_ms}));
    }
}
