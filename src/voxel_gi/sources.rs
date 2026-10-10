//! Analytic source snapshots shared by CPU and GPU transport.
//! Bevy light adaptation lives here; estimators depend only on these values.
use super::spectrum::EnergySpectrum;
use super::{
    EnergyEmitter, VoxelEnergyField, VoxelGiRuntime, VoxelGiStats, VoxelMaterialTransportTable,
    advance_transport_revision, request_global_refresh, sky_cache, telemetry::GiWorkTimings,
    update_stats_from_counters,
};
use bevy::prelude::*;
use std::cmp::Ordering;
use std::sync::Arc;

/// Background radiance in normalized transport units, multiplied by the
/// tracer's upper/lower-hemisphere gradient. Independent of the raster sky color.
/// Insert before `VoxelGiPlugin`, or change at runtime to relight the field.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct VoxelGiEnvironment(pub EnergySpectrum);

impl Default for VoxelGiEnvironment {
    fn default() -> Self {
        Self(EnergySpectrum::new(Vec3::new(0.16, 0.24, 0.38), 0.08, 0.0))
    }
}

pub(super) enum SourceChange {
    Global,
    Local(Vec<(Vec3, f32)>),
}

#[derive(Default)]
pub(super) struct SourceHistory(super::revision::ChangeHistory<SourceChange>);

impl SourceHistory {
    fn record(&mut self, from: u64, to: u64, change: SourceChange) {
        self.0.record(from, to, change);
    }

    pub fn history_limit(&self, voxel: IVec3, from: u64, to: u64) -> u32 {
        if from == to {
            return u32::MAX;
        }
        let mut limit = 64;
        let complete = self.0.visit_since(from, to, |change| {
            let nearby = match change {
                SourceChange::Global => true,
                SourceChange::Local(volumes) => volumes.iter().any(|(position, radius)| {
                    voxel.as_vec3().distance_squared(*position) <= radius * radius
                }),
            };
            if nearby {
                limit = 4;
            }
        });
        // A missing transition may have changed a nearby source. Never assume
        // it was remote just because the most recent light is elsewhere.
        if complete { limit } else { 4 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionalTransportLight {
    /// Direction from a surface toward the light.
    pub direction: Vec3,
    pub radiance: EnergySpectrum,
    pub custom_visible: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointTransportLight {
    pub position: Vec3,
    pub intensity: EnergySpectrum,
    pub range: f32,
    pub custom_visible: Vec3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransportSources {
    pub environment: EnergySpectrum,
    pub directional: Vec<DirectionalTransportLight>,
    pub points: Vec<PointTransportLight>,
}

pub(super) fn synchronize_transport_sources(
    mut runtime: ResMut<VoxelGiRuntime>,
    mut field: ResMut<VoxelEnergyField>,
    mut stats: ResMut<VoxelGiStats>,
    environment: Res<VoxelGiEnvironment>,
    directional_lights: Query<(&DirectionalLight, &GlobalTransform, Option<&EnergyEmitter>)>,
    point_lights: Query<(&PointLight, &GlobalTransform, Option<&EnergyEmitter>)>,
    energy_emitters: Query<
        (&EnergyEmitter, &GlobalTransform),
        (Without<PointLight>, Without<DirectionalLight>),
    >,
    mut timings: ResMut<GiWorkTimings>,
    mut next: Local<TransportSources>,
) {
    let _timing = timings.main_scope(super::telemetry::MainStage::Sources);
    next.environment = environment.0.bounded(1e12);
    next.directional.clear();
    next.points.clear();
    for (light, transform, extra) in &directional_lights {
        let linear = light.color.to_linear();
        let color = Vec3::new(linear.red, linear.green, linear.blue);
        let mut radiance = EnergySpectrum::new(color * (light.illuminance / 10_000.0), 0.0, 0.0);
        if let Some(extra) = extra {
            radiance += extra.spectrum;
        }
        next.directional.push(DirectionalTransportLight {
            direction: transform.back().as_vec3(),
            radiance: radiance.bounded(1e12),
            custom_visible: extra.map_or(Vec3::ZERO, |extra| extra.spectrum.bounded(1e12).visible),
        });
    }
    for (light, transform, extra) in &point_lights {
        let linear = light.color.to_linear();
        let color = Vec3::new(linear.red, linear.green, linear.blue);
        let mut intensity = EnergySpectrum::new(color * (light.intensity / 1_000.0), 0.0, 0.0);
        if let Some(extra) = extra {
            intensity += extra.spectrum;
        }
        next.points.push(PointTransportLight {
            position: transform.translation(),
            intensity: intensity.bounded(1e12),
            range: extra.map_or(light.range, |emitter| emitter.range.max(light.range)),
            custom_visible: extra.map_or(Vec3::ZERO, |extra| extra.spectrum.bounded(1e12).visible),
        });
    }
    for (emitter, transform) in &energy_emitters {
        next.points.push(PointTransportLight {
            position: transform.translation(),
            intensity: emitter.spectrum.bounded(1e12),
            range: emitter.range,
            custom_visible: emitter.spectrum.bounded(1e12).visible,
        });
    }
    next.points
        .retain(|light| light.position.is_finite() && light.range.is_finite() && light.range > 0.0);
    next.directional
        .retain(|light| light.direction.is_finite() && light.direction.length_squared() > 0.0);
    next.canonicalize();
    if *next == runtime.sources {
        return;
    }
    let change = next.change_from(&runtime.sources);
    if sky_cache::sky_lights(&next) != sky_cache::sky_lights(&runtime.sources) {
        runtime.sky_revision = runtime.sky_revision.wrapping_add(1);
        runtime.sky_remaining = field.order.len();
    }
    std::mem::swap(&mut runtime.sources, &mut *next);
    invalidate_source_change(&mut runtime, &mut field, &mut stats, change);
}

impl TransportSources {
    fn canonicalize(&mut self) {
        self.directional.sort_by(|a, b| {
            compare_vec3(a.direction, b.direction)
                .then_with(|| compare_energy(a.radiance, b.radiance))
                .then_with(|| compare_vec3(a.custom_visible, b.custom_visible))
        });
        self.points.sort_by(compare_point);
    }

    /// Both ends of a moved/removed light matter. Merge the canonical multisets
    /// so unchanged lights do not repeatedly invalidate their neighbors.
    fn change_from(&self, previous: &Self) -> SourceChange {
        if self.directional != previous.directional || self.environment != previous.environment {
            return SourceChange::Global;
        }
        let mut old = previous.points.iter().peekable();
        let mut new = self.points.iter().peekable();
        let mut volumes = Vec::new();
        loop {
            let changed = match (old.peek(), new.peek()) {
                (Some(a), Some(b)) => match compare_point(a, b) {
                    Ordering::Less => old.next(),
                    Ordering::Greater => new.next(),
                    Ordering::Equal => {
                        old.next();
                        new.next();
                        continue;
                    }
                },
                (Some(_), None) => old.next(),
                (None, Some(_)) => new.next(),
                (None, None) => break,
            };
            if let Some(light) = changed {
                // This is a responsiveness heuristic, not an influence cutoff:
                // every loaded receiver still refreshes after a source change.
                volumes.push((light.position, light.range + 16.0));
            }
        }
        SourceChange::Local(volumes)
    }
}

fn compare_energy(a: EnergySpectrum, b: EnergySpectrum) -> Ordering {
    compare_vec3(a.visible, b.visible)
        .then_with(|| a.infrared.total_cmp(&b.infrared))
        .then_with(|| a.ionizing.total_cmp(&b.ionizing))
}

fn compare_point(a: &PointTransportLight, b: &PointTransportLight) -> Ordering {
    compare_vec3(a.position, b.position)
        .then_with(|| a.range.total_cmp(&b.range))
        .then_with(|| compare_energy(a.intensity, b.intensity))
        .then_with(|| compare_vec3(a.custom_visible, b.custom_visible))
}

fn compare_vec3(a: Vec3, b: Vec3) -> std::cmp::Ordering {
    a.x.total_cmp(&b.x)
        .then_with(|| a.y.total_cmp(&b.y))
        .then_with(|| a.z.total_cmp(&b.z))
}

pub(super) fn synchronize_material_table(
    mut runtime: ResMut<VoxelGiRuntime>,
    materials: Res<VoxelMaterialTransportTable>,
    mut field: ResMut<VoxelEnergyField>,
    mut stats: ResMut<VoxelGiStats>,
    mut timings: ResMut<GiWorkTimings>,
) {
    let _timing = timings.main_scope(super::telemetry::MainStage::Sources);
    if !materials.is_changed() || materials.as_ref() == runtime.materials.as_ref() {
        return;
    }
    runtime.materials = Arc::new(materials.clone());
    invalidate_model(&mut runtime, &mut field, &mut stats);
}

/// Replace the global optical/transport model, not the residency of a light.
/// Keep the revision advance and restart watermark inseparable: late jobs from
/// the old model must be rejected even after another analytic source moves.
pub(super) fn invalidate_model(
    runtime: &mut VoxelGiRuntime,
    field: &mut VoxelEnergyField,
    stats: &mut VoxelGiStats,
) {
    invalidate_source_change(runtime, field, stats, SourceChange::Global);
    runtime.model_reset_source_revision = runtime.source_revision;
}

pub(super) fn invalidate_source_change(
    runtime: &mut VoxelGiRuntime,
    field: &mut VoxelEnergyField,
    stats: &mut VoxelGiStats,
    change: SourceChange,
) {
    // A moving source must not discard every in-flight batch every frame.
    // Keep a versioned snapshot; shorten history lazily when each probe runs.
    let from = runtime.source_revision;
    runtime.source_revision = from.wrapping_add(1);
    runtime
        .source_history
        .record(from, runtime.source_revision, change);
    field.source_revision = runtime.source_revision;
    advance_transport_revision(runtime, field);
    request_global_refresh(runtime, field.order.len());
    update_stats_from_counters(runtime, field, stats);
}

impl Default for TransportSources {
    fn default() -> Self {
        Self {
            environment: VoxelGiEnvironment::default().0,
            directional: Vec::new(),
            points: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f32, power: f32) -> PointTransportLight {
        PointTransportLight {
            position: Vec3::new(x, 0.0, 0.0),
            intensity: EnergySpectrum::new(Vec3::splat(power), power * 0.5, power * 0.25),
            custom_visible: Vec3::ZERO,
            range: 8.0,
        }
    }

    #[test]
    fn coincident_sources_have_canonical_order_and_deltas_exclude_unchanged_lights() {
        let mut before = TransportSources {
            points: vec![point(0.0, 1.0), point(0.0, 2.0), point(100.0, 1.0)],
            ..default()
        };
        let mut reordered = before.clone();
        reordered.points.reverse();
        before.canonicalize();
        reordered.canonicalize();
        assert_eq!(before, reordered);
        let SourceChange::Local(volumes) = reordered.change_from(&before) else {
            panic!()
        };
        assert!(volumes.is_empty());
        reordered.points[1] = point(200.0, 2.0);
        reordered.canonicalize();
        let SourceChange::Local(volumes) = reordered.change_from(&before) else {
            panic!()
        };
        assert_eq!(
            volumes,
            [(Vec3::ZERO, 24.0), (Vec3::new(200.0, 0.0, 0.0), 24.0)]
        );
        // Multiplicity matters: removing only one of two identical lights is a change.
        before.points = vec![point(0.0, 1.0); 2];
        reordered.points = vec![point(0.0, 1.0)];
        let SourceChange::Local(volumes) = reordered.change_from(&before) else {
            panic!()
        };
        assert_eq!(volumes.len(), 1);
        let mut sun = TransportSources {
            directional: vec![
                DirectionalTransportLight {
                    direction: Vec3::Y,
                    radiance: point(0.0, 1.0).intensity,
                    custom_visible: Vec3::ZERO,
                },
                DirectionalTransportLight {
                    direction: Vec3::Y,
                    radiance: point(0.0, 2.0).intensity,
                    custom_visible: Vec3::ONE,
                },
            ],
            ..default()
        };
        let mut reverse = sun.clone();
        reverse.directional.reverse();
        sun.canonicalize();
        reverse.canonicalize();
        assert_eq!(sun, reverse);
    }

    #[test]
    fn missed_source_updates_include_all_changes_and_missing_history_is_conservative() {
        let mut history = SourceHistory::default();
        history.record(0, 1, SourceChange::Local(vec![(Vec3::ZERO, 24.0)]));
        history.record(1, 2, SourceChange::Local(vec![(Vec3::X * 200.0, 24.0)]));
        assert_eq!(history.history_limit(IVec3::ZERO, 0, 2), 4);
        assert_eq!(history.history_limit(IVec3::ZERO, 1, 2), 64);
        assert_eq!(history.history_limit(IVec3::Y * 200, 0, 2), 64);
        assert_eq!(history.history_limit(IVec3::ZERO, 2, 2), u32::MAX);
        for from in 2..12 {
            history.record(from, from + 1, SourceChange::Local(vec![]));
        }
        assert_eq!(history.history_limit(IVec3::ZERO, 0, 12), 4);
        history.record(12, 13, SourceChange::Global);
        assert_eq!(history.history_limit(IVec3::Y * 200, 12, 13), 4);
    }

    #[test]
    fn host_environment_and_light_removal_advance_revisions_but_static_frames_do_not() {
        let mut app = App::new();
        app.init_resource::<VoxelGiRuntime>()
            .init_resource::<VoxelEnergyField>()
            .init_resource::<VoxelGiStats>()
            .init_resource::<VoxelGiEnvironment>()
            .init_resource::<GiWorkTimings>()
            .add_systems(Update, synchronize_transport_sources);
        let light = app
            .world_mut()
            .spawn((PointLight::default(), GlobalTransform::IDENTITY))
            .id();
        app.update();
        let revision = app.world().resource::<VoxelGiRuntime>().source_revision;
        app.update();
        assert_eq!(
            app.world().resource::<VoxelGiRuntime>().source_revision,
            revision
        );
        app.world_mut().resource_mut::<VoxelGiEnvironment>().0 = EnergySpectrum::ZERO;
        app.update();
        assert_eq!(
            app.world().resource::<VoxelGiRuntime>().source_revision,
            revision + 1
        );
        assert_eq!(
            app.world().resource::<VoxelGiRuntime>().sources.environment,
            EnergySpectrum::ZERO
        );
        app.world_mut().despawn(light);
        app.update();
        assert_eq!(
            app.world().resource::<VoxelGiRuntime>().source_revision,
            revision + 2
        );
        assert!(
            app.world()
                .resource::<VoxelGiRuntime>()
                .sources
                .points
                .is_empty()
        );
    }
}
