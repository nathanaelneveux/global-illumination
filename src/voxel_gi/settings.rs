use bevy::prelude::*;
use std::time::Duration;

const DEFAULT_SIMULATION_SAMPLING_IMPORTANCE: f32 = 0.0;

/// Quality and work limits. All loaded pages remain eligible.
/// Insert before `VoxelGiPlugin` or change this resource at runtime.
#[derive(Resource, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VoxelGiSettings {
    pub transport: TransportSettings,
    /// New paths per selected face, not a lifetime sample limit. Smaller batches
    /// refresh more receivers sooner but pay more scheduling/transfer overhead.
    pub samples_per_dispatch: u32,
    /// Relative importance of IR/ionizing emission when choosing light samples.
    /// Changes variance, not physical energy, history validity or band delivery.
    /// RGB always has importance one. Defaults to zero: GI serves the display,
    /// and simulation bands are not its responsibility (docs/direction.md);
    /// sources with only nonvisual emission are then found by paths alone. Use
    /// one for equal-band sampling. This is not a fixed ratio of sample counts.
    pub simulation_sampling_importance: f32,
    /// Measured retention constant c (default 1); zero restores the whole-range
    /// distance curve.
    /// Resident only. Each edit, load or sunlight change predicts a luminance
    /// change D at nearby faces from its flux; a face whose MSE-optimal history
    /// c * sigma^2 / D^2 would bind keeps that much, is not used as a cache
    /// source for one batch, and uses the fresh batch's noise if it was dark.
    pub measured_retention: f32,
    /// Resident-only, nonrecursive spatial RGB reconstruction.
    /// Zero is the unchanged corner display; one enables uncertainty-weighted
    /// connected-surface filtering. Never changes transport or simulation data.
    pub spatial_reconstruction: f32,
    /// Display-evidence target: a bounded adaptive footprint gathers neighbours
    /// until this much support. Zero disables reconstruction. This is not
    /// physical history or an accuracy guarantee. Requires spatial strength > 0.
    pub spatial_evidence_target: u32,
    /// Maximum unique candidate neighbors in adaptive reconstruction, excluding
    /// the receiver. Zero retains the legacy radius/work bound. This limits
    /// footprint and cost, not target support, physical history or blend strength.
    pub spatial_max_donors: u32,
    /// Disable for a same-kernel continuous-sampling A/B baseline.
    pub adaptive_updates: bool,
    /// GPU time per transport batch: its size, and how coarsely the GPU
    /// share below is spent.
    pub trace_budget_ms: f32,
    /// The GI budget: the share of GPU time transport batches may use
    /// (0.2 = 20%), independent of frame rate. Batches wait while it is spent;
    /// main-thread work per batch scales with them. Publication keeps its own
    /// per-frame caps.
    pub gpu_time_share: f32,
    pub max_probes_per_dispatch: usize,
    /// Main-thread receiver publication budget (capped at 8 ms). A page is
    /// atomic, so one expensive page may exceed it; at most four pages publish
    /// per update. Zero retains the one-page-per-update comparison.
    pub scene_publish_budget: Duration,
    pub display_upload_interval: Duration,
    /// Presentation-only temporal response. Does not filter simulation energy.
    pub display_response: Duration,
    /// Render-time transition between completed resident RGB targets. Unlike
    /// estimator filtering, this advances on every rendered frame even when a
    /// face waits for transport. Zero is the instantaneous publication control.
    pub display_transition_frames: u32,
    /// Settled faces still receive verification rays at this cadence. This is
    /// eligibility, not a deadline: loaded-world size and work budget matter.
    pub stable_audit_interval: Duration,
    /// Baseline minimum. Dark indirect RGB (peak below 0.02) requires at least
    /// 256 samples before sparse audits, even if early observed variance is zero.
    pub stable_min_samples: u32,
    /// Per-band relative-error heuristic, not a certified convergence bound.
    pub stable_relative_error: f32,
    /// Faces whose CPU statistics mirror the GPU merges per batch (the GPU
    /// state is the authority; GPU-picked faces' mirror lags under heavy
    /// refinement). Bounds the main-thread merge. Validation of display
    /// quality from CPU recorders raises it (config/gi-mirror-full.json).
    pub statistics_mirror_per_batch: u32,
    /// Keep a CPU copy of every probe's statistics (~600 B each): validation
    /// recorders, diagnostics and simulation-style queries read it. Off by
    /// default, since the GPU state is the authority and the renderer reads it
    /// directly; config/gi-mirror-full.json turns it on with the full mirror.
    pub cpu_statistics: bool,
    /// Persistent world cache: receivers (probes, resident slots, sun corners)
    /// for pages within this many voxels of an active camera (page nearest
    /// point), kept until 32 voxels beyond it. Every visible face is
    /// view-owned; the cache supplies secondary-ray termination, offscreen
    /// history and edit-flux measurement, and exchanges RGB history with
    /// visible faces. Zero disables it (cacheless). Geometry stays every
    /// loaded page: rays traverse it regardless.
    pub cache_radius: f32,
    /// Fraction of the transport allowance spent refreshing the cache; the
    /// visible faces get the remainder. Not extra budget.
    pub cache_share: f32,
}

impl Default for VoxelGiSettings {
    fn default() -> Self {
        Self {
            transport: TransportSettings::default(),
            samples_per_dispatch: 16,
            simulation_sampling_importance: DEFAULT_SIMULATION_SAMPLING_IMPORTANCE,
            measured_retention: 1.0,
            spatial_reconstruction: 1.0,
            spatial_evidence_target: 1024,
            spatial_max_donors: 0,
            adaptive_updates: true,
            trace_budget_ms: 4.0,
            gpu_time_share: 0.2,
            max_probes_per_dispatch: 8192,
            scene_publish_budget: Duration::from_millis(2),
            display_upload_interval: Duration::from_millis(50),
            display_response: Duration::from_millis(75),
            display_transition_frames: 8,
            stable_audit_interval: Duration::from_secs(2),
            stable_min_samples: 64,
            stable_relative_error: 0.12,
            statistics_mirror_per_batch: 1024,
            cpu_statistics: false,
            cache_radius: 64.0,
            cache_share: 0.25,
        }
    }
}

impl VoxelGiSettings {
    pub(super) fn sampling_importance(&self) -> super::EnergySpectrum {
        let weight = if self.simulation_sampling_importance.is_finite() {
            self.simulation_sampling_importance.clamp(0.0, 1.0)
        } else {
            DEFAULT_SIMULATION_SAMPLING_IMPORTANCE
        };
        super::EnergySpectrum::new(Vec3::ONE, weight, weight)
    }
    pub(super) fn measured_retention(&self) -> f32 {
        if self.measured_retention.is_finite() {
            self.measured_retention.clamp(0.0, 64.0)
        } else {
            0.0
        }
    }
    pub(super) fn batch_samples(&self) -> u32 {
        self.samples_per_dispatch.clamp(1, 64)
    }
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
    pub(super) fn mirror_per_batch(&self) -> u32 {
        self.statistics_mirror_per_batch.clamp(64, 65536)
    }

    pub(super) fn budget_ms(&self) -> f32 {
        if self.trace_budget_ms.is_finite() {
            self.trace_budget_ms.clamp(0.25, 16.0)
        } else {
            4.0
        }
    }
    /// The world cache residency radius; None: cacheless.
    pub(super) fn cache_radius(&self) -> Option<f32> {
        (self.cache_radius.is_finite() && self.cache_radius > 0.0).then_some(self.cache_radius)
    }
    pub(super) fn world_budget_fraction(&self) -> f32 {
        match self.cache_radius() {
            None => 0.0,
            Some(_) if self.cache_share.is_finite() => self.cache_share.clamp(0.01, 0.99),
            Some(_) => 0.25,
        }
    }
    pub(super) fn gpu_share(&self) -> f32 {
        if self.gpu_time_share.is_finite() {
            self.gpu_time_share.clamp(0.02, 1.0)
        } else {
            0.2
        }
    }
    pub(super) fn batch_limit(&self) -> usize {
        self.max_probes_per_dispatch.clamp(64, 8192)
    }
}

/// The integrand, separate from scheduling and presentation. Changing this
/// restarts measurements lazily, retaining display priors and sample sequences.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TransportSettings {
    /// Maximum length in voxels of each diffuse ray and voxel-emitter connection,
    /// not total path length or Minecraft-style per-block attenuation. Directional
    /// visibility keeps its independent 768-voxel horizon. A shortened diffuse
    /// ray exhausting its reach contributes environment only if the remaining
    /// coarse occupancy frontier is empty, never through possible occluders.
    /// Analytic point lights keep their own source-defined range.
    pub reach: f32,
    /// Explicit diffuse surface hits after the receiver. Limited to the current
    /// validated sampler's three-bounce dimension allocation.
    pub bounces: u32,
}

impl Default for TransportSettings {
    fn default() -> Self {
        Self {
            reach: 768.0,
            bounces: 3,
        }
    }
}

impl TransportSettings {
    pub(super) fn normalized(self) -> Self {
        Self {
            reach: if self.reach.is_finite() {
                self.reach.clamp(1.0, 768.0)
            } else {
                768.0
            },
            bounces: self.bounces.clamp(1, 3),
        }
    }
}

/// Reuse the discrete source/model epoch: reject old in-flight measurements,
/// reset each receiver only when selected, and leave geometry/sky caches intact.
pub(super) fn synchronize_transport_settings(
    settings: Res<VoxelGiSettings>,
    mut runtime: ResMut<super::VoxelGiRuntime>,
    mut field: ResMut<super::VoxelEnergyField>,
    mut stats: ResMut<super::VoxelGiStats>,
) {
    if field.cpu_statistics != settings.cpu_statistics {
        // Attach empty statistics (they fill as faces are traced or mirrored)
        // or free them all.
        field.cpu_statistics = settings.cpu_statistics;
        let keep = settings.cpu_statistics;
        let order = field.order.clone();
        for voxel in order {
            if let Some(probe) = field.probes.get_mut(&voxel) {
                if keep {
                    probe.stats_mut();
                } else {
                    probe.stats = None;
                }
            }
        }
    }
    let transport = settings.transport.normalized();
    if runtime.transport == transport {
        return;
    }
    runtime.transport = transport;
    super::sources::invalidate_model(&mut runtime, &mut field, &mut stats);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_radius_sets_residency_and_one_shared_budget() {
        let cached = VoxelGiSettings::default();
        assert_eq!(cached.cache_radius(), Some(64.0));
        assert_eq!(cached.world_budget_fraction(), 0.25);
        let cacheless: VoxelGiSettings =
            serde_json::from_str(include_str!("../../config/gi-cacheless.json")).unwrap();
        assert_eq!(cacheless.cache_radius(), None);
        assert_eq!(cacheless.world_budget_fraction(), 0.0);
        assert_eq!(cacheless.transport, cached.transport);
        assert_eq!(cacheless.gpu_share(), cached.gpu_share());
        let small: VoxelGiSettings =
            serde_json::from_str(include_str!("../../config/gi-cache-32.json")).unwrap();
        assert_eq!(small.cache_radius(), Some(32.0));
        assert_eq!(small.world_budget_fraction(), 0.25);
    }

    #[test]
    fn preferred_spatial_default_has_explicit_unfiltered_control() {
        let settings = VoxelGiSettings::default();
        let preset: VoxelGiSettings =
            serde_json::from_str(include_str!("../../config/gi-display-evidence-1024.json"))
                .unwrap();
        assert_eq!(settings, preset);
        assert_eq!(settings.spatial_reconstruction, 1.0);
        assert_eq!(settings.spatial_evidence_target, 1024);
        assert_eq!(settings.spatial_max_donors, 0);
        let control: VoxelGiSettings =
            serde_json::from_str(include_str!("../../config/gi-display-unfiltered.json")).unwrap();
        assert_eq!(control.spatial_reconstruction, 0.0);
        assert_eq!(control.spatial_evidence_target, 0);
        assert_eq!(control.display_transition_frames, 0);
        assert_eq!(control.transport, settings.transport);
        assert_eq!(
            control.sampling_importance(),
            settings.sampling_importance()
        );
    }

    #[test]
    fn tuning_limits_are_finite_and_defaults_roundtrip() {
        assert_eq!(
            TransportSettings {
                reach: f32::NAN,
                bounces: 99
            }
            .normalized(),
            TransportSettings::default()
        );
        assert_eq!(
            TransportSettings {
                reach: -2.0,
                bounces: 0
            }
            .normalized(),
            TransportSettings {
                reach: 1.0,
                bounces: 1
            }
        );
        let settings = VoxelGiSettings::default();
        let encoded = serde_json::to_string(&settings).unwrap();
        assert_eq!(
            serde_json::from_str::<VoxelGiSettings>(&encoded).unwrap(),
            settings
        );
        assert_eq!(
            serde_json::from_str::<VoxelGiSettings>("{}").unwrap(),
            settings
        );
    }

    #[test]
    fn visible_only_is_the_default_with_rgb_priority_and_equal_band_controls() {
        let settings = VoxelGiSettings::default();
        let expected = super::super::EnergySpectrum::new(Vec3::ONE, 0.0, 0.0);
        assert_eq!(settings.sampling_importance(), expected);
        assert_eq!(
            serde_json::from_str::<VoxelGiSettings>("{}").unwrap().sampling_importance(),
            expected
        );
        // The previous default (simulation bands at one eighth).
        assert_eq!(
            serde_json::from_str::<VoxelGiSettings>(include_str!("../../config/gi-rgb-priority.json"))
                .unwrap()
                .sampling_importance(),
            super::super::EnergySpectrum::new(Vec3::ONE, 0.125, 0.125)
        );
        let control: VoxelGiSettings =
            serde_json::from_str(include_str!("../../config/gi-equal-band.json")).unwrap();
        assert_eq!(
            control.sampling_importance(),
            super::super::EnergySpectrum::ONE
        );
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                VoxelGiSettings {
                    simulation_sampling_importance: value,
                    ..settings.clone()
                }
                .sampling_importance(),
                expected
            );
        }
        for (input, normalized) in [(-1.0, 0.0), (2.0, 1.0)] {
            assert_eq!(
                VoxelGiSettings {
                    simulation_sampling_importance: input,
                    ..settings.clone()
                }
                .sampling_importance(),
                super::super::EnergySpectrum::new(Vec3::ONE, normalized, normalized)
            );
        }
    }

    #[test]
    fn only_model_changes_restart_measurements_without_rebuilding_geometry_or_sky() {
        let mut app = App::new();
        app.init_resource::<VoxelGiSettings>()
            .init_resource::<super::super::VoxelGiRuntime>()
            .init_resource::<super::super::VoxelEnergyField>()
            .init_resource::<super::super::VoxelGiStats>()
            .add_systems(Update, synchronize_transport_settings);
        app.update();
        app.world_mut()
            .resource_mut::<VoxelGiSettings>()
            .trace_budget_ms = 1.0;
        app.world_mut()
            .resource_mut::<VoxelGiSettings>()
            .display_response = Duration::from_millis(200);
        app.world_mut()
            .resource_mut::<VoxelGiSettings>()
            .samples_per_dispatch = 1;
        app.world_mut()
            .resource_mut::<VoxelGiSettings>()
            .simulation_sampling_importance = 1.0;
        app.update();
        assert_eq!(
            app.world()
                .resource::<super::super::VoxelGiRuntime>()
                .source_revision,
            0
        );
        app.world_mut()
            .resource_mut::<VoxelGiSettings>()
            .transport
            .reach = 16.0;
        app.update();
        let runtime = app.world().resource::<super::super::VoxelGiRuntime>();
        assert_eq!(runtime.source_revision, 1);
        assert_eq!(runtime.model_reset_source_revision, 1);
        assert_eq!(runtime.generation, 0);
        assert_eq!(runtime.sky_revision, 0);
        app.update();
        assert_eq!(
            app.world()
                .resource::<super::super::VoxelGiRuntime>()
                .source_revision,
            1
        );
    }
}
