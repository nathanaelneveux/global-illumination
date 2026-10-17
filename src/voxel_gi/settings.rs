use bevy::prelude::*;
use std::time::Duration;

const DEFAULT_SIMULATION_SAMPLING_IMPORTANCE: f32 = 0.125;

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
    /// RGB always has importance one; nonvisual importance stays nonzero so all
    /// sources retain support. Simulation need not meet the RGB visual threshold.
    /// Defaults to 0.125 on every backend, independently of path reuse; use one
    /// for equal-band sampling. This is not a fixed ratio of sample counts.
    pub simulation_sampling_importance: f32,
    /// Resident cache history ceiling in samples. Long histories keep dark,
    /// heavy-tailed faces quiet; shorter ones forget under-converged cache
    /// inputs sooner. Edits and source changes still cap history separately.
    pub history_limit: u32,
    /// Recent-sample weighting. One is a plain running mean. With k > 1, sample
    /// i of n weighs (i/n)^(k-1): early cache values fade as n^-k instead of
    /// n^-1, at a variance cost of k^2/(2k-1). The fixed point is unchanged.
    pub history_recency: f32,
    /// Resident-only, nonrecursive spatial RGB reconstruction.
    /// Zero is the unchanged corner display; one enables uncertainty-weighted
    /// connected-surface filtering. Never changes transport or simulation data.
    pub spatial_reconstruction: f32,
    /// Display-evidence target. Zero selects the fixed stencil;
    /// positive values choose a bounded adaptive footprint instead. This is not
    /// physical history or an accuracy guarantee. Requires spatial strength > 0.
    pub spatial_evidence_target: u32,
    /// Maximum unique candidate neighbors in adaptive reconstruction, excluding
    /// the receiver. Zero retains the legacy radius/work bound. This limits
    /// footprint and cost, not target support, physical history or blend strength.
    pub spatial_max_donors: u32,
    /// Experimental refinement ranking by projected RGB error in total light.
    /// Does not change stability, coverage, edit urgency, or the fair world sweep.
    pub visible_refinement: bool,
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
}

impl Default for VoxelGiSettings {
    fn default() -> Self {
        Self {
            transport: TransportSettings::default(),
            samples_per_dispatch: 16,
            simulation_sampling_importance: DEFAULT_SIMULATION_SAMPLING_IMPORTANCE,
            history_limit: super::tracer::HISTORY_LIMIT,
            history_recency: 1.0,
            spatial_reconstruction: 1.0,
            spatial_evidence_target: 2048,
            spatial_max_donors: 0,
            visible_refinement: false,
            adaptive_updates: true,
            trace_budget_ms: 4.0,
            max_probes_per_dispatch: 8192,
            scene_publish_budget: Duration::from_millis(2),
            display_upload_interval: Duration::from_millis(50),
            display_response: Duration::from_millis(75),
            display_transition_frames: 8,
            stable_audit_interval: Duration::from_secs(2),
            stable_min_samples: 64,
            stable_relative_error: 0.12,
        }
    }
}

/// Host-supplied conversion from normalized transport to visible shading units.
/// Used only by optional visual ranking, never by physical transport. A renderer
/// with camera exposure or an artistic GI gain should update these together.
#[derive(Resource, Clone, Copy, Debug)]
pub struct VoxelGiVisualWeights {
    pub direct: f32,
    pub indirect: f32,
}

impl Default for VoxelGiVisualWeights {
    fn default() -> Self {
        Self {
            direct: 1.0,
            indirect: 1.0,
        }
    }
}

impl VoxelGiSettings {
    pub(super) fn sampling_importance(&self) -> super::EnergySpectrum {
        let weight = if self.simulation_sampling_importance.is_finite() {
            self.simulation_sampling_importance.clamp(0.01, 1.0)
        } else {
            DEFAULT_SIMULATION_SAMPLING_IMPORTANCE
        };
        super::EnergySpectrum::new(Vec3::ONE, weight, weight)
    }
    pub(super) fn history_limit(&self) -> u32 {
        self.history_limit.clamp(64, super::tracer::HISTORY_LIMIT)
    }
    pub(super) fn history_recency(&self) -> f32 {
        if self.history_recency.is_finite() {
            self.history_recency.clamp(1.0, 4.0)
        } else {
            1.0
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
    fn preferred_spatial_default_has_explicit_unfiltered_and_fixed_controls() {
        let settings = VoxelGiSettings::default();
        let preset: VoxelGiSettings =
            serde_json::from_str(include_str!("../../config/gi-display-evidence-2048.json"))
                .unwrap();
        assert_eq!(settings, preset);
        assert_eq!(settings.spatial_reconstruction, 1.0);
        assert_eq!(settings.spatial_evidence_target, 2048);
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
        let fixed: VoxelGiSettings =
            serde_json::from_str(include_str!("../../config/gi-adaptive-spatial.json")).unwrap();
        assert_eq!(fixed.spatial_reconstruction, 1.0);
        assert_eq!(fixed.spatial_evidence_target, 0);
        let ranking: VoxelGiSettings =
            serde_json::from_str(include_str!("../../config/gi-visible-refinement.json")).unwrap();
        assert!(ranking.visible_refinement);
        assert_eq!(ranking.spatial_reconstruction, 0.0);
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
    fn rgb_priority_is_the_default_with_an_explicit_equal_band_control() {
        let settings = VoxelGiSettings::default();
        let expected = super::super::EnergySpectrum::new(Vec3::ONE, 0.125, 0.125);
        assert_eq!(settings.sampling_importance(), expected);
        for json in ["{}", include_str!("../../config/gi-rgb-priority.json")] {
            assert_eq!(
                serde_json::from_str::<VoxelGiSettings>(json)
                    .unwrap()
                    .sampling_importance(),
                expected
            );
        }
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
        for (input, normalized) in [(0.0, 0.01), (2.0, 1.0)] {
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
