//! Analytic source snapshots shared by CPU and GPU transport.
//! Bevy light adaptation lives here; estimators depend only on these values.
use super::spectrum::EnergySpectrum;
use super::{
    EnergyEmitter, VoxelEnergyField, VoxelGiRuntime, VoxelGiStats, VoxelMaterialTransportTable,
    advance_transport_revision, request_global_refresh, sky_cache, telemetry::GiWorkTimings,
    update_stats_from_counters,
};
use bevy::prelude::*;
use std::sync::Arc;

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
    directional_lights: Query<(&DirectionalLight, &GlobalTransform, Option<&EnergyEmitter>)>,
    point_lights: Query<(&PointLight, &GlobalTransform, Option<&EnergyEmitter>)>,
    energy_emitters: Query<
        (&EnergyEmitter, &GlobalTransform),
        (Without<PointLight>, Without<DirectionalLight>),
    >,
    mut timings: ResMut<GiWorkTimings>,
) {
    let _timing = timings.main_scope();
    let mut next = TransportSources::default();
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
    next.directional
        .sort_by(|a, b| compare_vec3(a.direction, b.direction));
    next.points
        .sort_by(|a, b| compare_vec3(a.position, b.position));
    if next == runtime.sources {
        return;
    }
    let global = next.directional != runtime.sources.directional
        || next.environment != runtime.sources.environment;
    // Old and new influence volumes both matter when a source moves or vanishes.
    runtime.source_influences = runtime
        .sources
        .points
        .iter()
        .chain(next.points.iter())
        .map(|light| (light.position, light.range + 16.0))
        .collect();
    if sky_cache::sky_lights(&next) != sky_cache::sky_lights(&runtime.sources) {
        runtime.sky_revision = runtime.sky_revision.wrapping_add(1);
        runtime.sky_remaining = field.order.len();
    }
    runtime.sources = next;
    invalidate_sources(&mut runtime, &mut field, &mut stats);
    if global {
        runtime.global_source_revision = runtime.source_revision;
    }
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
    let _timing = timings.main_scope();
    if !materials.is_changed() || materials.as_ref() == runtime.materials.as_ref() {
        return;
    }
    runtime.materials = Arc::new(materials.clone());
    invalidate_sources(&mut runtime, &mut field, &mut stats);
    runtime.global_source_revision = runtime.source_revision;
    runtime.reset_source_revision = runtime.source_revision;
}

pub(super) fn invalidate_sources(
    runtime: &mut VoxelGiRuntime,
    field: &mut VoxelEnergyField,
    stats: &mut VoxelGiStats,
) {
    // A moving source must not discard every in-flight batch every frame.
    // Keep a versioned snapshot; shorten history lazily when each probe runs.
    runtime.source_revision = runtime.source_revision.wrapping_add(1);
    field.source_revision = runtime.source_revision;
    advance_transport_revision(runtime, field);
    request_global_refresh(runtime, field.order.len());
    update_stats_from_counters(runtime, field, stats);
}

impl Default for TransportSources {
    fn default() -> Self {
        Self {
            environment: EnergySpectrum::new(Vec3::new(0.16, 0.24, 0.38), 0.08, 0.0),
            directional: Vec::new(),
            points: Vec::new(),
        }
    }
}
