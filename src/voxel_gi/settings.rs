use bevy::prelude::*;
use std::time::Duration;

/// Work limits, not world coverage limits. All loaded pages remain eligible.
/// Insert before `VoxelGiPlugin` or change this resource at runtime.
#[derive(Resource, Clone, Debug)]
pub struct VoxelGiSettings {
    /// Disable for a same-kernel continuous-sampling A/B baseline.
    pub adaptive_updates: bool,
    pub trace_budget_ms: f32,
    pub max_probes_per_dispatch: usize,
    /// Main-thread receiver publication budget (capped at 8 ms). A page is
    /// atomic, so one expensive page may exceed it; at most four pages publish
    /// per update. Zero retains the one-page-per-update comparison.
    pub scene_publish_budget: Duration,
    pub display_upload_interval: Duration,
    /// Presentation-only temporal response. Does not filter simulation energy.
    pub display_response: Duration,
    /// Settled faces still receive verification rays at this cadence. This is
    /// eligibility, not a deadline: loaded-world size and work budget matter.
    pub stable_audit_interval: Duration,
    /// Baseline minimum. Dark indirect RGB (peak below 0.02) requires at least
    /// 256 samples before sparse audits, even if early observed variance is zero.
    pub stable_min_samples: u32,
    /// Per-band relative-error heuristic, not a certified convergence bound.
    pub stable_relative_error: f32,
}

impl Default for VoxelGiSettings {
    fn default() -> Self {
        Self {
            adaptive_updates: true,
            trace_budget_ms: 4.0,
            max_probes_per_dispatch: 8192,
            scene_publish_budget: Duration::from_millis(2),
            display_upload_interval: Duration::from_millis(50),
            display_response: Duration::from_millis(75),
            stable_audit_interval: Duration::from_secs(2),
            stable_min_samples: 64,
            stable_relative_error: 0.12,
        }
    }
}

impl VoxelGiSettings {
    pub(super) fn audit_interval(&self) -> Duration {
        self.stable_audit_interval
            .clamp(Duration::from_millis(100), Duration::from_secs(30))
    }
    pub(super) fn stable_samples(&self) -> u32 {
        self.stable_min_samples.clamp(32, 256)
    }
    pub(super) fn stable_error(&self) -> f32 {
        if self.stable_relative_error.is_finite() {
            self.stable_relative_error.clamp(0.01, 0.5)
        } else {
            0.12
        }
    }
    pub(super) fn display_response_seconds(&self) -> f32 {
        self.display_response.as_secs_f32().clamp(0.01, 0.5)
    }
    pub(super) fn budget_ms(&self) -> f32 {
        if self.trace_budget_ms.is_finite() {
            self.trace_budget_ms.clamp(0.25, 16.0)
        } else {
            4.0
        }
    }
    pub(super) fn batch_limit(&self) -> usize {
        self.max_probes_per_dispatch.clamp(64, 8192)
    }
}
