//! Analytic source snapshots shared by CPU and GPU transport.
//! Bevy light adaptation lives here; estimators depend only on these values.
use super::spectrum::EnergySpectrum;
use super::{
    EnergyEmitter, VoxelEnergyField, VoxelGiRuntime, VoxelGiStats, VoxelMaterialTransportTable,
    advance_transport_revision, sky_cache, telemetry::GiWorkTimings,
    update_stats_from_counters,
};
use bevy::ecs::system::SystemParam;
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

/// Detect host input changes before rebuilding/canonicalizing a snapshot.
/// Removal readers must all advance, even if another input already changed.
/// Raster visibility is deliberately not a source switch: offscreen/hidden
/// presentation must not implicitly turn off persistent heat or radiation.
#[derive(SystemParam)]
pub(super) struct SourceInputs<'w, 's> {
    changed: Query<
        'w,
        's,
        (),
        (
            Or<(
                With<PointLight>,
                With<DirectionalLight>,
                With<EnergyEmitter>,
            )>,
            Or<(
                Changed<PointLight>,
                Changed<DirectionalLight>,
                Changed<EnergyEmitter>,
                Changed<GlobalTransform>,
            )>,
        ),
    >,
    removed_points: RemovedComponents<'w, 's, PointLight>,
    removed_directional: RemovedComponents<'w, 's, DirectionalLight>,
    removed_emitters: RemovedComponents<'w, 's, EnergyEmitter>,
    removed_transforms: RemovedComponents<'w, 's, GlobalTransform>,
    initialized: Local<'s, bool>,
}

impl SourceInputs<'_, '_> {
    fn changed(&mut self, environment_changed: bool) -> bool {
        let removed = self.removed_points.read().count()
            + self.removed_directional.read().count()
            + self.removed_emitters.read().count()
            + self.removed_transforms.read().count();
        let first = !*self.initialized;
        *self.initialized = true;
        first || environment_changed || removed != 0 || !self.changed.is_empty()
    }
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
    mut inputs: SourceInputs,
) {
    let _timing = timings.main_scope(super::telemetry::MainStage::Sources);
    if !inputs.changed(environment.is_changed()) {
        return;
    }
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
    // Eliminate empty sources before either backend (and the sunlight cache)
    // sees them. Invisible simulation bands and custom primary visibility
    // still count: a visually black source is not necessarily inactive.
    next.points.retain(|light| {
        light.position.is_finite()
            && light.range.is_finite()
            && light.range > 0.0
            && contributes(light.intensity, light.custom_visible)
    });
    next.directional.retain(|light| {
        light.direction.is_finite()
            && light.direction.length_squared() > 0.0
            && contributes(light.radiance, light.custom_visible)
    });
    next.canonicalize();
    if *next == runtime.sources {
        return;
    }
    let change = next.change_from(&runtime.sources);
    // Corner visibility depends only on quantized sun directions.
    if sky_cache::visibility_key(&sky_cache::sky_lights(&next))
        != sky_cache::visibility_key(&sky_cache::sky_lights(&runtime.sources))
    {
        runtime.sky_revision = runtime.sky_revision.wrapping_add(1);
        runtime.sky_remaining = field.order.len();
    }
    std::mem::swap(&mut runtime.sources, &mut *next);
    invalidate_source_change(&mut runtime, &mut field, &mut stats, change);
}

fn contributes(energy: EnergySpectrum, custom_visible: Vec3) -> bool {
    energy.max_component() > 0.0 || custom_visible.max_element() > 0.0
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
    // A model reset restarts every measurement.
    runtime.spatial_history.record_source(0, &[]);
}

pub(super) fn invalidate_source_change(
    runtime: &mut VoxelGiRuntime,
    field: &mut VoxelEnergyField,
    stats: &mut VoxelGiStats,
    change: SourceChange,
) {
    // A moving source must not discard every in-flight batch every frame.
    // Every face's history is capped on the GPU when this publishes: at most
    // 4 samples near a changed light (or everywhere for a global change), 64
    // elsewhere. Every face becomes due for an audit.
    runtime.source_revision = runtime.source_revision.wrapping_add(1);
    match change {
        SourceChange::Global => runtime.spatial_history.record_source(4, &[]),
        SourceChange::Local(volumes) => runtime.spatial_history.record_source(64, &volumes),
    }
    field.source_revision = runtime.source_revision;
    advance_transport_revision(runtime, field);
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

    fn source_app() -> App {
        let mut app = App::new();
        app.init_resource::<VoxelGiRuntime>()
            .init_resource::<VoxelEnergyField>()
            .init_resource::<VoxelGiStats>()
            .init_resource::<VoxelGiEnvironment>()
            .init_resource::<GiWorkTimings>()
            .add_systems(Update, synchronize_transport_sources);
        app
    }

    #[test]
    fn change_detection_covers_reclassification_removal_and_reactivation() {
        #[derive(Resource, Default)]
        struct Rebuilds(usize);
        let mut app = source_app();
        app.init_resource::<Rebuilds>().add_systems(
            Update,
            |mut inputs: SourceInputs,
             environment: Res<VoxelGiEnvironment>,
             mut count: ResMut<Rebuilds>| {
                count.0 += usize::from(inputs.changed(environment.is_changed()));
            },
        );
        let entity = app
            .world_mut()
            .spawn((
                EnergyEmitter {
                    spectrum: EnergySpectrum::new(Vec3::ZERO, 2.0, 0.0),
                    ..default()
                },
                GlobalTransform::IDENTITY,
            ))
            .id();
        app.update();
        let expected = app.world().resource::<VoxelGiRuntime>().sources.clone();
        for _ in 0..8 {
            app.update();
        }
        assert_eq!(app.world().resource::<Rebuilds>().0, 1);
        assert_eq!(app.world().resource::<VoxelGiRuntime>().sources, expected);

        app.world_mut()
            .entity_mut(entity)
            .insert(PointLight::default());
        app.update();
        assert!(
            app.world().resource::<VoxelGiRuntime>().sources.points[0]
                .intensity
                .visible
                .max_element()
                > 0.0
        );
        app.world_mut().entity_mut(entity).remove::<PointLight>();
        app.update();
        assert_eq!(app.world().resource::<VoxelGiRuntime>().sources, expected);
        app.world_mut()
            .entity_mut(entity)
            .remove::<GlobalTransform>();
        app.update();
        assert!(
            app.world()
                .resource::<VoxelGiRuntime>()
                .sources
                .points
                .is_empty()
        );
        app.world_mut()
            .entity_mut(entity)
            .insert(GlobalTransform::from_translation(Vec3::X));
        app.update();
        assert_eq!(
            app.world().resource::<VoxelGiRuntime>().sources.points[0].position,
            Vec3::X
        );
        app.world_mut().entity_mut(entity).remove::<EnergyEmitter>();
        app.update();
        assert!(
            app.world()
                .resource::<VoxelGiRuntime>()
                .sources
                .points
                .is_empty()
        );
        let rebuilds = app.world().resource::<Rebuilds>().0;
        app.update();
        assert_eq!(
            app.world().resource::<Rebuilds>().0,
            rebuilds,
            "removals are consumed once"
        );
        assert_eq!(rebuilds, 6);
    }

    #[test]
    #[ignore = "paired static analytic source synchronization study; no renderer"]
    fn static_source_sync_cost_study() {
        for count in [100, 10_000] {
            let mut app = source_app();
            for x in 0..count {
                app.world_mut().spawn((
                    PointLight::default(),
                    GlobalTransform::from_translation(Vec3::X * x as f32),
                ));
            }
            app.update();
            let expected = app.world().resource::<VoxelGiRuntime>().sources.clone();
            let mut times: [Vec<f64>; 2] = default();
            for round in 0..24 {
                for mode in [round % 2, 1 - round % 2] {
                    if mode == 0 {
                        // Equal-valued changed input forces the original complete
                        // snapshot work without changing transport revisions.
                        app.world_mut()
                            .resource_mut::<VoxelGiEnvironment>()
                            .set_changed();
                    }
                    let start = std::time::Instant::now();
                    app.update();
                    if round >= 4 {
                        times[mode].push(start.elapsed().as_secs_f64() * 1000.0);
                    }
                    assert_eq!(app.world().resource::<VoxelGiRuntime>().sources, expected);
                }
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            println!(
                "STATIC_SOURCES count={count} full_ms={:.4} gated_ms={:.4} exact=true",
                times[0][10], times[1][10]
            );
        }
    }

    #[test]
    fn empty_sources_are_omitted_but_each_transport_band_can_activate_them() {
        let mut app = App::new();
        app.init_resource::<VoxelGiRuntime>()
            .init_resource::<VoxelEnergyField>()
            .init_resource::<VoxelGiStats>()
            .init_resource::<VoxelGiEnvironment>()
            .init_resource::<GiWorkTimings>()
            .add_systems(Update, synchronize_transport_sources);
        let point = app
            .world_mut()
            .spawn((
                PointLight {
                    intensity: 0.0,
                    ..default()
                },
                GlobalTransform::IDENTITY,
            ))
            .id();
        let sun = app
            .world_mut()
            .spawn((
                DirectionalLight {
                    illuminance: 0.0,
                    ..default()
                },
                GlobalTransform::IDENTITY,
            ))
            .id();
        let emitter = app
            .world_mut()
            .spawn((EnergyEmitter::default(), GlobalTransform::IDENTITY))
            .id();
        app.update();
        let runtime = app.world().resource::<VoxelGiRuntime>();
        assert!(runtime.sources.points.is_empty());
        assert!(runtime.sources.directional.is_empty());
        let revision = runtime.source_revision;

        app.world_mut().entity_mut(point).insert(EnergyEmitter {
            spectrum: EnergySpectrum::new(Vec3::ZERO, 2.0, 0.0),
            ..default()
        });
        app.world_mut().entity_mut(sun).insert(EnergyEmitter {
            spectrum: EnergySpectrum::new(Vec3::ZERO, 0.0, 3.0),
            ..default()
        });
        app.world_mut().entity_mut(emitter).insert(EnergyEmitter {
            spectrum: EnergySpectrum::new(Vec3::ONE, 0.0, 0.0),
            ..default()
        });
        app.update();
        let runtime = app.world().resource::<VoxelGiRuntime>();
        assert_eq!(runtime.source_revision, revision + 1);
        assert_eq!(runtime.sources.points.len(), 2);
        assert!(
            runtime
                .sources
                .points
                .iter()
                .any(|p| p.intensity.infrared == 2.0)
        );
        assert!(
            runtime
                .sources
                .points
                .iter()
                .any(|p| p.custom_visible == Vec3::ONE)
        );
        assert_eq!(runtime.sources.directional[0].radiance.ionizing, 3.0);
        assert!(contributes(EnergySpectrum::ZERO, Vec3::ONE));

        for entity in [point, sun, emitter] {
            app.world_mut().entity_mut(entity).remove::<EnergyEmitter>();
        }
        app.update();
        let runtime = app.world().resource::<VoxelGiRuntime>();
        assert_eq!(runtime.source_revision, revision + 2);
        assert!(runtime.sources.points.is_empty());
        assert!(runtime.sources.directional.is_empty());
    }

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
    fn source_changes_cap_every_face_on_the_gpu() {
        use crate::voxel_gi::retention::source_limit;
        let mut runtime = VoxelGiRuntime::default();
        let mut field = VoxelEnergyField::default();
        let mut stats = VoxelGiStats::default();
        invalidate_source_change(
            &mut runtime,
            &mut field,
            &mut stats,
            SourceChange::Local(vec![(Vec3::ZERO, 24.0)]),
        );
        let events = runtime.spatial_history.take_events();
        assert_eq!(source_limit(&events, IVec3::ZERO), 4);
        assert_eq!(source_limit(&events, IVec3::Y * 200), 64);
        invalidate_source_change(&mut runtime, &mut field, &mut stats, SourceChange::Global);
        assert_eq!(source_limit(&runtime.spatial_history.take_events(), IVec3::Y * 200), 4);
        invalidate_model(&mut runtime, &mut field, &mut stats);
        assert_eq!(source_limit(&runtime.spatial_history.take_events(), IVec3::Y * 200), 0);
        assert_eq!(runtime.source_revision, 3);
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
