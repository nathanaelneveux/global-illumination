//! Opt-in end-to-end workload. It modifies only the ephemeral running world.
//! `VOXEL_GI_VALIDATE=1 cargo run` captures static A/B images, then continuously
//! edits terrain, animates a source, streams pages and writes a JSON report.
use super::render::VoxelTransportMaterial;
use super::{EnergyEmitter, EnergySpectrum, VoxelEnergyField, VoxelGiDirty, VoxelGiStats};
use super::{MaterialTransport, VoxelMaterialTransportTable};
use crate::{player::Player, terrain::PrototypeWorld};
use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy_voxel_world::prelude::*;
use bevy_voxel_world::rendering::VoxelWorldMaterialHandle;
use std::time::Instant;

#[derive(Default)]
pub(super) struct ValidationRun {
    started: Option<Instant>,
    frames: Vec<(u32, f32)>,
    captures: u32,
    edits: u32,
    moves: u32,
    mode_changes: u32,
    light: Option<Entity>,
    center: Vec3,
    removed: Option<(IVec3, WorldVoxel<u8>)>,
    emissive: Option<(IVec3, WorldVoxel<u8>, MaterialTransport)>,
    emitter_started: bool,
    emitter_captured: bool,
    max_pages: usize,
    max_near_age_ms: f32,
    done: bool,
}

pub(super) fn run_validation(
    mut run: Local<ValidationRun>,
    time: Res<Time<Real>>,
    stats: Res<VoxelGiStats>,
    field: Res<VoxelEnergyField>,
    mut dirty: ResMut<VoxelGiDirty>,
    mut world: VoxelWorld<PrototypeWorld>,
    mut player: Single<&mut Transform, With<Player>>,
    mut lights: Query<&mut Transform, Without<Player>>,
    mut commands: Commands,
    handle: Option<Res<VoxelWorldMaterialHandle<VoxelTransportMaterial>>>,
    mut materials: ResMut<Assets<VoxelTransportMaterial>>,
    mut transport_materials: ResMut<VoxelMaterialTransportTable>,
) {
    if std::env::var_os("VOXEL_GI_VALIDATE").is_none() || run.done {
        return;
    }
    let elapsed = run
        .started
        .get_or_insert_with(Instant::now)
        .elapsed()
        .as_secs_f32();
    if elapsed < 1.0 {
        run.center = player.translation;
    }
    let phase = if elapsed < 25.0 {
        0
    } else if elapsed < 45.0 {
        1
    } else {
        2
    };
    if elapsed > 10.0 {
        run.frames.push((phase, time.delta_secs() * 1000.0));
    }
    run.max_pages = run.max_pages.max(stats.loaded_pages);
    if (15.0..24.0).contains(&elapsed) {
        let step = ((elapsed - 15.0) / 2.0) as u32;
        if step == run.captures && step < 4 {
            let name = ["corners", "flat", "disabled", "restored"][step as usize];
            if let Err(error) = std::fs::create_dir_all("target/gi-validation") {
                warn!(%error);
                return;
            }
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(format!("target/gi-validation/{name}.png")));
            run.captures += 1;
        }
        // Capture before changing mode; give the render extraction a half-second.
        if (elapsed - 15.0) % 2.0 > 0.5
            && let Some(handle) = handle.as_ref()
            && let Some(mut material) = materials.get_mut(&handle.handle)
        {
            match step {
                0 => material.extension.settings.reconstruction = 0,
                1 => material.extension.settings.enabled = 0,
                _ => {
                    material.extension.settings.enabled = 1;
                    material.extension.settings.reconstruction = 1;
                }
            }
        }
    }
    if (22.0..24.0).contains(&elapsed)
        && let Some(handle) = handle.as_ref()
        && let Some(mut material) = materials.get_mut(&handle.handle)
    {
        material.extension.settings.debug_irradiance = 1;
        material.extension.settings.reconstruction = u32::from(elapsed < 23.0);
        if (22.5..23.0).contains(&elapsed) && run.captures == 4 {
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk("target/gi-validation/irradiance-corners.png"));
            run.captures = 5;
        }
        if elapsed >= 23.5 && run.captures == 5 {
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk("target/gi-validation/irradiance-flat.png"));
            run.captures = 6;
        }
    }
    if elapsed >= 24.0
        && run.captures == 6
        && let Some(handle) = handle.as_ref()
        && let Some(mut material) = materials.get_mut(&handle.handle)
    {
        material.extension.settings.debug_irradiance = 0;
        material.extension.settings.reconstruction = 1;
        run.captures = 7;
    }
    if (25.0..65.0).contains(&elapsed) {
        if std::env::var_os("VOXEL_GI_VALIDATE_TOGGLES").is_some()
            && let Some(handle) = handle.as_ref()
            && let Some(mut material) = materials.get_mut(&handle.handle)
        {
            let step = ((elapsed - 25.0) * 8.0) as u32;
            if step >= run.mode_changes {
                // Exercise the same uniforms as F4-F7 while geometry/cache
                // generations are changing, independently of OS key injection.
                material.extension.settings.enabled = step & 1;
                material.extension.settings.reconstruction = (step >> 1) & 1;
                material.extension.settings.debug_irradiance = (step >> 2) & 1;
                material.extension.settings.voxel_sky = (step >> 3) & 1;
                run.mode_changes = step + 1;
            }
        }
        if !run.emitter_started {
            let position = IVec3::new(0, 12, -18);
            run.emissive = Some((
                position,
                world.get_voxel(position),
                transport_materials.get(250),
            ));
            transport_materials.set(
                250,
                MaterialTransport {
                    emission: EnergySpectrum::new(Vec3::new(12.0, 3.0, 0.3), 4.0, 1.0),
                    ..default()
                },
            );
            world.set_voxel(position, WorldVoxel::Solid(250));
            dirty.mark_voxel(position);
            run.emitter_started = true;
        }
        if elapsed > 40.0 && !run.emitter_captured {
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk("target/gi-validation/emissive-room.png"));
            run.emitter_captured = true;
        }
        let center = run.center;
        let entity = *run.light.get_or_insert_with(|| {
            commands
                .spawn((
                    PointLight {
                        color: Color::srgb(1.0, 0.2, 0.08),
                        intensity: 3000.0,
                        range: 24.0,
                        ..default()
                    },
                    EnergyEmitter {
                        spectrum: EnergySpectrum::new(Vec3::ZERO, 12.0, 3.0),
                        range: 24.0,
                    },
                    Transform::from_translation(center + Vec3::Y * 4.0),
                ))
                .id()
        });
        if let Ok(mut transform) = lights.get_mut(entity) {
            transform.translation = center
                + Vec3::new(
                    (elapsed * 1.2).cos() * 5.0,
                    4.0,
                    (elapsed * 1.2).sin() * 5.0,
                );
        }
        if elapsed < 45.0 && ((elapsed - 25.0) * 4.0) as u32 > run.edits {
            if let Some((voxel, material)) = run.removed.take() {
                world.set_voxel(voxel, material);
                dirty.mark_voxel(voxel);
            } else {
                let eye = player.translation;
                let target = field
                    .probes
                    .keys()
                    .copied()
                    .filter(|voxel| {
                        let distance = voxel.as_vec3().distance_squared(eye);
                        (9.0..144.0).contains(&distance)
                    })
                    .min_by_key(|voxel| {
                        (((voxel.x as i64 * 73_856_093) ^ (voxel.z as i64 * 19_349_663))
                            + run.edits as i64 * 83_492_791)
                            .unsigned_abs()
                    });
                if let Some(voxel) = target {
                    run.removed = Some((voxel, world.get_voxel(voxel)));
                    world.set_voxel(voxel, WorldVoxel::Air);
                    dirty.mark_voxel(voxel);
                }
            }
            run.edits += 1;
        }
        if elapsed >= 45.0 && ((elapsed - 45.0) / 4.0) as u32 >= run.moves {
            player.translation += Vec3::new(24.0, 2.0, 0.0);
            run.moves += 1;
        }
        let eye = player.translation;
        for voxel in field.order.iter().step_by(211) {
            if voxel.as_vec3().distance_squared(eye) > 32.0 * 32.0 {
                continue;
            }
            if let Some(at) = field.probes[voxel].updated_at {
                run.max_near_age_ms = run.max_near_age_ms.max(at.elapsed().as_secs_f32() * 1000.0);
            }
        }
    }
    if elapsed >= 65.0 {
        if run.mode_changes > 0
            && let Some(handle) = handle.as_ref()
            && let Some(mut material) = materials.get_mut(&handle.handle)
        {
            material.extension.settings.enabled = 1;
            material.extension.settings.reconstruction = 1;
            material.extension.settings.debug_irradiance = 0;
            material.extension.settings.voxel_sky = 0;
        }
        if let Some((position, voxel, transport)) = run.emissive.take() {
            world.set_voxel(position, voxel);
            dirty.mark_voxel(position);
            transport_materials.set(250, transport);
        }
        if let Some(entity) = run.light.take() {
            commands.entity(entity).despawn();
        }
        if let Some((voxel, material)) = run.removed.take() {
            world.set_voxel(voxel, material);
            dirty.mark_voxel(voxel);
        }
    }
    if elapsed >= 72.0 {
        let phases: Vec<_> = (0..3).map(|phase| {
            let values: Vec<_> = run.frames.iter().filter(|(p, _)| *p == phase).map(|(_, ms)| *ms).collect();
            format!("{{\"phase\":{phase},\"frames\":{},\"p50_ms\":{:.3},\"p95_ms\":{:.3},\"p99_ms\":{:.3}}}", values.len(), percentile(&values, 0.50), percentile(&values, 0.95), percentile(&values, 0.99))
        }).collect();
        let report = format!(
            "{{\"backend\":\"{}\",\"phases\":[{}],\"edits\":{},\"moves\":{},\"mode_changes\":{},\"max_loaded_pages\":{},\"max_near_sample_age_ms\":{:.3},\"coverage\":{:.5},\"mean_history\":{:.2}}}\n",
            stats.backend,
            phases.join(","),
            run.edits,
            run.moves,
            run.mode_changes,
            run.max_pages,
            run.max_near_age_ms,
            stats.interactive_coverage,
            stats.mean_history_samples
        );
        match std::fs::write("target/gi-validation/report.json", &report) {
            Ok(()) => info!(%report, "GI validation workload complete"),
            Err(error) => error!(%error, "could not save GI validation report"),
        }
        run.done = true;
        if std::env::var_os("VOXEL_GI_VALIDATE_EXIT").is_some() {
            commands.write_message(AppExit::Success);
        }
    }
}

fn percentile(values: &[f32], fraction: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    sorted[((sorted.len() - 1) as f32 * fraction).ceil() as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn percentile_keeps_tail_stalls_visible() {
        assert_eq!(percentile(&[16.0, 16.0, 16.0, 100.0], 0.99), 100.0);
        assert_eq!(percentile(&[], 0.95), 0.0);
    }
}
