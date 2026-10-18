//! Per-application construction policy. The reusable plugin never reads shell
//! variables for these choices; the demo translates its launch flags separately.
use bevy::prelude::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransportBackend {
    /// GPU-resident accumulation and asynchronous snapshots. GPU errors are fatal.
    #[default]
    Auto,
    /// Explicit alias for persistent GPU measurements; never downgrades to CPU.
    Resident,
    /// Historical CPU-owned accumulation and blocking worker readback control.
    Synchronous,
    /// Keep GPU display, but run transport through the independent CPU reference.
    Cpu,
}

impl TransportBackend {
    pub(super) fn resident_accumulation(self) -> bool {
        matches!(self, Self::Auto | Self::Resident)
    }
}

/// Insert before `VoxelGiPlugin`. These are construction choices, not live
/// switches; runtime quality/work changes belong in `VoxelGiSettings` instead.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VoxelGiStartup {
    pub backend: TransportBackend,
}

pub(super) fn initialize(app: &mut App) {
    app.init_resource::<VoxelGiStartup>();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_is_default_with_explicit_independent_controls() {
        assert!(TransportBackend::default().resident_accumulation());
        assert!(TransportBackend::Resident.resident_accumulation());
        assert!(!TransportBackend::Synchronous.resident_accumulation());
        assert!(!TransportBackend::Cpu.resident_accumulation());
    }

    #[test]
    fn independent_apps_keep_independent_startup_policy() {
        let mut default_app = App::new();
        initialize(&mut default_app);
        let mut control = App::new();
        control.insert_resource(VoxelGiStartup {
            backend: TransportBackend::Cpu,
        });
        initialize(&mut control);
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
