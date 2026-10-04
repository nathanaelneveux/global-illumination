use bevy::prelude::*;
use std::time::Duration;

/// Work limits, not world coverage limits. All loaded pages remain eligible.
/// Insert before `VoxelGiPlugin` or change this resource at runtime.
#[derive(Resource, Clone, Debug)]
pub struct VoxelGiSettings {
    pub trace_budget_ms: f32,
    pub max_probes_per_dispatch: usize,
    pub display_upload_interval: Duration,
}

impl Default for VoxelGiSettings {
    fn default() -> Self {
        Self {
            trace_budget_ms: 4.0,
            max_probes_per_dispatch: 8192,
            display_upload_interval: Duration::from_millis(50),
        }
    }
}

impl VoxelGiSettings {
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
