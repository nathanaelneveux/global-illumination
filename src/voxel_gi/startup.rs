//! Per-application construction policy. The reusable plugin never reads shell
//! variables for these choices; the demo translates its launch flags separately.
use bevy::prelude::*;

pub use super::retention::SummaryMode as RetentionSummary;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransportBackend {
    /// GPU-resident accumulation and asynchronous snapshots; retain CPU fallback.
    #[default]
    Auto,
    /// Explicit alias for persistent GPU measurements with CPU fallback.
    Resident,
    /// Historical CPU-owned accumulation and blocking worker readback control.
    Synchronous,
    /// Experimental fresh-path sharing on top of resident measurements.
    Reuse,
    /// Keep GPU display, but run transport through the independent CPU reference.
    Cpu,
}

impl TransportBackend {
    pub(super) fn resident_accumulation(self) -> bool {
        matches!(self, Self::Auto | Self::Resident | Self::Reuse)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetentionPolicy {
    /// Current measured whole-range curve. The summary controls only how
    /// evicted change events are conservatively represented, not the curve.
    WholeRange(RetentionSummary),
    /// Historical two-distance control retained for reproducible comparisons.
    Legacy,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self::WholeRange(RetentionSummary::default())
    }
}

/// Insert before `VoxelGiPlugin`. These are construction choices, not live
/// switches; runtime quality/work changes belong in `VoxelGiSettings` instead.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VoxelGiStartup {
    pub backend: TransportBackend,
    pub retention: RetentionPolicy,
}

pub(super) fn initialize(app: &mut App) {
    app.init_resource::<VoxelGiStartup>();
    let policy = app.world().resource::<VoxelGiStartup>().retention;
    app.world_mut()
        .resource_mut::<super::VoxelGiRuntime>()
        .spatial_history = match policy {
        RetentionPolicy::WholeRange(summary) => {
            Some(super::retention::SpatialHistory::new(summary))
        }
        RetentionPolicy::Legacy => None,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_is_default_with_explicit_independent_controls() {
        assert!(TransportBackend::default().resident_accumulation());
        assert!(TransportBackend::Reuse.resident_accumulation());
        assert!(TransportBackend::Resident.resident_accumulation());
        assert!(!TransportBackend::Synchronous.resident_accumulation());
        assert!(!TransportBackend::Cpu.resident_accumulation());
    }

    #[test]
    fn independent_apps_keep_independent_startup_policy() {
        let mut default_app = App::new();
        default_app.init_resource::<super::super::VoxelGiRuntime>();
        initialize(&mut default_app);
        let mut control = App::new();
        control
            .init_resource::<super::super::VoxelGiRuntime>()
            .insert_resource(VoxelGiStartup {
                backend: TransportBackend::Cpu,
                retention: RetentionPolicy::Legacy,
            });
        initialize(&mut control);
        assert!(
            default_app
                .world()
                .resource::<super::super::VoxelGiRuntime>()
                .spatial_history
                .is_some()
        );
        assert!(
            control
                .world()
                .resource::<super::super::VoxelGiRuntime>()
                .spatial_history
                .is_none()
        );
        assert_eq!(
            *default_app.world().resource::<VoxelGiStartup>(),
            VoxelGiStartup::default()
        );
        assert_eq!(
            control.world().resource::<VoxelGiStartup>().backend,
            TransportBackend::Cpu
        );
    }
}
