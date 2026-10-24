use bevy::prelude::*;
use std::time::Duration;

/// Quality and work limits. All loaded pages remain eligible.
/// Insert before `VoxelGiPlugin` or change this resource at runtime.
#[derive(Resource, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VoxelGiSettings {
    pub transport: TransportSettings,
    /// Measured retention constant c (default 1); zero restores the whole-range
    /// distance curve.
    /// Resident only. Each edit, load or sunlight change predicts a luminance
    /// change D at nearby faces from its flux; a face whose MSE-optimal history
    /// c * sigma^2 / D^2 would bind keeps that much, is not used as a cache
    /// source for one batch, and uses the fresh batch's noise if it was dark.
    pub measured_retention: f32,
    /// View presentation's nonrecursive spatial RGB reconstruction strength.
    /// Zero is the unfiltered corner display; one enables evidence-weighted
    /// coplanar filtering. Never changes transport data.
    pub spatial_reconstruction: f32,
    /// Relative standard error below which a face's own estimate is shown
    /// without spatial help (noise there is under this fraction of its
    /// brightness). Below it the filter pools coplanar neighbours with
    /// MSE-optimal weights from their measured variance and bias. 0.031 is
    /// the former 1024-sample target for a relative sigma of one.
    pub spatial_noise_threshold: f32,
    /// The GI budget: the share of GPU time transport batches may use
    /// (0.2 = 20%), independent of frame rate. Batches wait while it is spent;
    /// main-thread work per batch scales with them. Publication keeps its own
    /// per-frame caps.
    pub gpu_time_share: f32,
    /// Presentation-only temporal response. Does not filter simulation energy.
    pub display_response: Duration,
    /// Persistent world cache: receivers (probes, resident slots, sun corners)
    /// for pages within this many voxels of an active camera (page nearest
    /// point), kept until 32 voxels beyond it. Every visible face is
    /// view-owned; the cache supplies secondary-ray termination, offscreen
    /// history and edit-flux measurement, and exchanges RGB history with
    /// visible faces. Zero disables it (cacheless). Geometry stays every
    /// loaded page: rays traverse it regardless.
    pub cache_radius: f32,
    /// Visible-face receivers per camera (8,192-131,072). Receivers are
    /// discovered on a grid of at least 4-pixel cells; at high resolutions
    /// this cap sets the cell size instead. More receivers resolve smaller
    /// distant faces at a higher fixed discovery/presentation cost, which
    /// `gpu_time_share` does not cover.
    pub view_receivers: u32,
    /// Fraction of the transport allowance spent refreshing the cache; the
    /// visible faces get the remainder. Not extra budget.
    pub cache_share: f32,
    /// Scheduler work bounds: measured, not host policy.
    #[serde(skip)]
    pub(crate) scheduler: SchedulerSettings,
}

/// The world-cache scheduler's batch and audit bounds.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SchedulerSettings {
    /// New paths per selected face, not a lifetime sample limit. Smaller batches
    /// refresh more receivers sooner but pay more scheduling/transfer overhead.
    pub(crate) samples_per_dispatch: u32,
    /// GPU time per transport batch: its size, and how coarsely the GPU
    /// share (`gpu_time_share`) is spent.
    pub(crate) trace_budget_ms: f32,
    pub(crate) max_probes_per_dispatch: usize,
    /// Main-thread receiver publication budget (capped at 8 ms). A page is
    /// atomic, so one expensive page may exceed it; at most four pages publish
    /// per update. Zero retains the one-page-per-update comparison.
    pub(crate) scene_publish_budget: Duration,
    /// Settled faces still receive verification rays at this cadence. This is
    /// eligibility, not a deadline: loaded-world size and work budget matter.
    pub(crate) stable_audit_interval: Duration,
    /// Baseline minimum. Dark indirect RGB (peak below 0.02) requires at least
    /// 256 samples before sparse audits, even if early observed variance is zero.
    pub(crate) stable_min_samples: u32,
    /// Per-band relative-error heuristic, not a certified convergence bound.
    pub(crate) stable_relative_error: f32,
    /// GPU-picked faces whose progress (counts, settled verdict, presentation
    /// seed) the CPU mirrors per batch. The GPU state is the authority; this
    /// bounds the readback and main-thread merge.
    pub(crate) statistics_mirror_per_batch: u32,
}

/// Starting points for `VoxelGiSettings` that trade cost for quality. They
/// change only performance knobs (GPU share, world-cache radius, visible-face
/// receivers); the look settings stay at their defaults.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VoxelGiQuality {
    Low,
    #[default]
    Default,
    High,
}

impl VoxelGiSettings {
    pub fn quality(quality: VoxelGiQuality) -> Self {
        let (gpu_time_share, cache_radius, view_receivers) = match quality {
            VoxelGiQuality::Low => (0.12, 32.0, 32_768),
            VoxelGiQuality::Default => (0.2, 64.0, 65_536),
            VoxelGiQuality::High => (0.3, 96.0, 131_072),
        };
        Self {
            gpu_time_share,
            cache_radius,
            view_receivers,
            ..default()
        }
    }

    /// Parse RON settings; missing fields default and values are clamped.
    pub fn from_ron(text: &str) -> Result<Self, String> {
        ron::from_str::<Self>(text).map(Self::normalized).map_err(|error| error.to_string())
    }

    /// Pretty RON of every field.
    pub fn to_ron(&self) -> String {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default()).expect("finite GI settings")
    }

    /// The preset these settings' performance knobs match, if any.
    pub fn matching_quality(&self) -> Option<VoxelGiQuality> {
        [VoxelGiQuality::Low, VoxelGiQuality::Default, VoxelGiQuality::High]
            .into_iter()
            .find(|quality| {
                let preset = Self::quality(*quality);
                preset.gpu_time_share == self.gpu_time_share
                    && preset.cache_radius == self.cache_radius
                    && preset.view_receivers == self.view_receivers
            })
    }
}

impl Default for VoxelGiSettings {
    fn default() -> Self {
        Self {
            transport: TransportSettings::default(),
            measured_retention: 1.0,
            spatial_reconstruction: 1.0,
            spatial_noise_threshold: 0.031,
            gpu_time_share: 0.2,
            display_response: Duration::from_millis(75),
            cache_radius: 64.0,
            view_receivers: 65_536,
            cache_share: 0.25,
            scheduler: SchedulerSettings::default(),
        }
    }
}

impl Default for SchedulerSettings {
    fn default() -> Self {
        Self {
            samples_per_dispatch: 16,
            trace_budget_ms: 4.0,
            max_probes_per_dispatch: 8192,
            scene_publish_budget: Duration::from_millis(2),
            stable_audit_interval: Duration::from_secs(2),
            stable_min_samples: 64,
            stable_relative_error: 0.12,
            statistics_mirror_per_batch: 1024,
        }
    }
}

impl VoxelGiSettings {
    /// Clamp every field to its supported range (what the solver would use).
    pub fn normalized(mut self) -> Self {
        self.transport = self.transport.normalized();
        self.scheduler.samples_per_dispatch = self.batch_samples();
        self.scheduler.trace_budget_ms = self.budget_ms();
        self.scheduler.max_probes_per_dispatch = self.batch_limit();
        self.scheduler.stable_min_samples = self.stable_samples();
        self.scheduler.stable_relative_error = self.stable_error();
        self.scheduler.stable_audit_interval = self.audit_interval();
        self.display_response = self
            .display_response
            .clamp(Duration::from_millis(10), Duration::from_millis(500));
        self.scheduler.scene_publish_budget = self.scheduler.scene_publish_budget.min(Duration::from_millis(8));
        self.view_receivers = self.view_receiver_limit();
        self.spatial_noise_threshold = self.noise_threshold();
        self
    }

    pub(super) fn noise_threshold(&self) -> f32 {
        if self.spatial_noise_threshold.is_finite() {
            self.spatial_noise_threshold.clamp(0.005, 0.25)
        } else {
            0.031
        }
    }

    pub(super) fn view_receiver_limit(&self) -> u32 {
        self.view_receivers.clamp(8_192, super::gpu::view::MAX_VIEW_RECEIVERS)
    }

    pub(super) fn measured_retention(&self) -> f32 {
        if self.measured_retention.is_finite() {
            self.measured_retention.clamp(0.0, 64.0)
        } else {
            0.0
        }
    }
    pub(super) fn batch_samples(&self) -> u32 {
        self.scheduler.samples_per_dispatch.clamp(1, 64)
    }
    pub(super) fn audit_interval(&self) -> Duration {
        self.scheduler.stable_audit_interval
            .clamp(Duration::from_millis(100), Duration::from_secs(30))
    }
    pub(super) fn stable_samples(&self) -> u32 {
        self.scheduler.stable_min_samples.clamp(32, 256)
    }
    pub(super) fn stable_error(&self) -> f32 {
        if self.scheduler.stable_relative_error.is_finite() {
            self.scheduler.stable_relative_error.clamp(0.01, 0.5)
        } else {
            0.12
        }
    }
    pub(super) fn display_response_seconds(&self) -> f32 {
        self.display_response.as_secs_f32().clamp(0.01, 0.5)
    }
    pub(super) fn mirror_per_batch(&self) -> u32 {
        self.scheduler.statistics_mirror_per_batch.clamp(64, 65536)
    }

    pub(super) fn budget_ms(&self) -> f32 {
        if self.scheduler.trace_budget_ms.is_finite() {
            self.scheduler.trace_budget_ms.clamp(0.25, 16.0)
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
        self.scheduler.max_probes_per_dispatch.clamp(64, 8192)
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
    pub fn normalized(self) -> Self {
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
/// reset each receiver only when selected, and leave geometry intact.
pub(super) fn synchronize_transport_settings(
    settings: Res<VoxelGiSettings>,
    mut runtime: ResMut<super::VoxelGiRuntime>,
    mut field: ResMut<super::ReceiverField>,
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
    fn ron_import_clamps_and_round_trips() {
        let imported =
            VoxelGiSettings::from_ron("(transport: (reach: 16.0, bounces: 0), view_receivers: 1)")
                .unwrap();
        assert_eq!(imported.transport, TransportSettings { reach: 16.0, bounces: 1 });
        assert!(
            VoxelGiSettings::from_ron("(samples_per_dispatch: 16)").is_err(),
            "scheduler bounds are not host settings"
        );
        assert_eq!(imported.view_receivers, 8_192);
        assert!(VoxelGiSettings::from_ron("broken").is_err());
        let high = VoxelGiSettings::quality(VoxelGiQuality::High);
        assert_eq!(VoxelGiSettings::from_ron(&high.to_ron()).unwrap(), high);
        assert_eq!(VoxelGiSettings::from_ron("()").unwrap(), VoxelGiSettings::default());
    }

    #[test]
    fn quality_presets_change_only_cost_knobs_and_round_trip() {
        for quality in [VoxelGiQuality::Low, VoxelGiQuality::Default, VoxelGiQuality::High] {
            let preset = VoxelGiSettings::quality(quality);
            assert_eq!(preset.matching_quality(), Some(quality));
            assert_eq!(preset.clone().normalized(), preset, "presets are within range");
            let look = VoxelGiSettings { gpu_time_share: 0.2, cache_radius: 64.0, view_receivers: 65_536, ..preset };
            assert_eq!(look, VoxelGiSettings::default());
        }
        assert_eq!(VoxelGiSettings::quality(VoxelGiQuality::Default), VoxelGiSettings::default());
    }

    #[test]
    fn cache_radius_sets_residency_and_one_shared_budget() {
        let cached = VoxelGiSettings::default();
        assert_eq!(cached.cache_radius(), Some(64.0));
        assert_eq!(cached.world_budget_fraction(), 0.25);
        let cacheless = VoxelGiSettings { cache_radius: 0.0, ..default() };
        assert_eq!(cacheless.cache_radius(), None);
        assert_eq!(cacheless.world_budget_fraction(), 0.0);
        assert_eq!(cacheless.gpu_share(), cached.gpu_share());
        let small = VoxelGiSettings { cache_radius: 32.0, ..default() };
        assert_eq!(small.cache_radius(), Some(32.0));
        assert_eq!(small.world_budget_fraction(), 0.25);
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
        let encoded = ron::to_string(&settings).unwrap();
        assert_eq!(
            ron::from_str::<VoxelGiSettings>(&encoded).unwrap(),
            settings
        );
        assert_eq!(
            ron::from_str::<VoxelGiSettings>("()").unwrap(),
            settings
        );
    }

    #[test]
    fn only_model_changes_restart_measurements_without_rebuilding_geometry() {
        let mut app = App::new();
        app.init_resource::<VoxelGiSettings>()
            .init_resource::<super::super::VoxelGiRuntime>()
            .init_resource::<super::super::ReceiverField>()
            .init_resource::<super::super::VoxelGiStats>()
            .add_systems(Update, synchronize_transport_settings);
        app.update();
        app.world_mut()
            .resource_mut::<VoxelGiSettings>()
            .scheduler
            .trace_budget_ms = 1.0;
        app.world_mut()
            .resource_mut::<VoxelGiSettings>()
            .display_response = Duration::from_millis(200);
        app.world_mut()
            .resource_mut::<VoxelGiSettings>()
            .scheduler
            .samples_per_dispatch = 1;
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
        app.update();
        assert_eq!(
            app.world()
                .resource::<super::super::VoxelGiRuntime>()
                .source_revision,
            1
        );
    }
}
