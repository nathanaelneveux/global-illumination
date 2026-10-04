use bevy::prelude::*;
use std::time::Duration;

/// Work limits, not world coverage limits. All loaded pages remain eligible.
/// Insert before `VoxelGiPlugin` or change this resource at runtime.
#[derive(Resource, Clone, Debug)]
pub struct VoxelGiSettings {
    pub trace_budget_ms: f32,
    pub max_probes_per_dispatch: usize,
    pub display_upload_interval: Duration,
    /// Presentation-only temporal response. Does not filter simulation energy.
    pub display_response: Duration,
}

impl Default for VoxelGiSettings {
    fn default() -> Self {
        Self {
            trace_budget_ms: 4.0,
            max_probes_per_dispatch: 8192,
            display_upload_interval: Duration::from_millis(50),
            display_response: Duration::from_millis(75),
        }
    }
}

impl VoxelGiSettings {
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
