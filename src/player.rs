use std::time::Duration;

use avian3d::prelude::*;
use bevy::pbr::{DistanceFog, FogFalloff};
use bevy::prelude::*;
use bevy::window::{CursorOptions, PrimaryWindow};
use bevy_ahoy::input::AccumulatedInput;
use bevy_ahoy::prelude::*;
use bevy_enhanced_input::prelude::*;
use bevy_voxel_world::prelude::{VoxelWorld, VoxelWorldCamera, WorldVoxel};

use crate::terrain::{PrototypeWorld, TERRAIN_VIEW_DISTANCE, TerrainMaterial};
use crate::voxel_gi::VoxelGiDirty;
use crate::world_environment::sky_color;

const FOG_START: f32 = 96.0;
const FOG_END: f32 = TERRAIN_VIEW_DISTANCE;
const PLAYER_RADIUS: f32 = 0.3;
const PLAYER_HEIGHT: f32 = 1.8;
const PLAYER_CROUCH_HEIGHT: f32 = 1.5;
const PLAYER_STANDING_VIEW_HEIGHT: f32 = 1.62;
const PLAYER_CROUCH_VIEW_HEIGHT: f32 = 1.27;
const PLAYER_WALK_SPEED: f32 = 4.32;
const PLAYER_SPRINT_SPEED: f32 = PLAYER_WALK_SPEED * 1.5;
const PLAYER_CROUCH_SPEED_SCALE: f32 = 0.3;
const PLAYER_JUMP_HEIGHT: f32 = 1.0;
const PLAYER_GRAVITY: f32 = 32.0;
const PLAYER_STEP_SIZE: f32 = 0.55;
const PLAYER_MAX_SPEED: f32 = 80.0;
const PLAYER_AIR_ACCELERATION_HZ: f32 = 8.0;
const PLAYER_MAX_AIR_WISH_SPEED: f32 = 0.35;

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_input_context::<PlayerInput>()
            .add_message::<PlaceEmissiveBlock>()
            .add_systems(Startup, spawn_player)
            .add_systems(
                FixedPostUpdate,
                (update_player_movement_speed, filter_held_jump)
                    .chain()
                    .before(AhoySystems::MoveCharacters),
            )
            .add_systems(Update, reset_player_to_world_spawn);
        if !crate::headless_validation() {
            app.add_systems(
                PreUpdate,
                sync_player_input.before(EnhancedInputSystems::Prepare),
            )
            .add_systems(Update, edit_voxels);
        }
    }
}

#[derive(Component)]
pub(crate) struct Player;

#[derive(Component, Default)]
struct PlayerInput;

#[derive(Component)]
struct PlayerCamera;

#[derive(Component, Default)]
struct HeldJumpFilter {
    was_grounded: bool,
}

#[derive(InputAction)]
#[action_output(bool)]
struct Sprint;

#[derive(InputAction)]
#[action_output(bool)]
struct BreakVoxel;

#[derive(InputAction)]
#[action_output(bool)]
struct PlaceVoxel;

#[derive(InputAction)]
#[action_output(bool)]
struct PlaceEmissiveVoxel;

/// The HUD queues the same placement operation as the keyboard shortcut.
#[derive(Message)]
pub(crate) struct PlaceEmissiveBlock;

fn spawn_player(mut commands: Commands, world: Res<PrototypeWorld>) {
    let player = commands
        .spawn((
            Name::new("Player"),
            Player,
            PlayerInput,
            HeldJumpFilter::default(),
            player_controller(),
            Collider::cylinder(PLAYER_RADIUS, PLAYER_HEIGHT),
            Transform::from_translation(world.player_spawn_position()),
            actions!(PlayerInput[
                (
                    Action::<Movement>::new(),
                    DeadZone::default(),
                    Bindings::spawn((
                        Cardinal::wasd_keys(),
                        Cardinal::arrows(),
                        Cardinal::dpad(),
                        Axial::left_stick(),
                    )),
                ),
                (
                    Action::<Jump>::new(),
                    bindings![KeyCode::Space, GamepadButton::North],
                ),
                (
                    Action::<Crouch>::new(),
                    bindings![
                        KeyCode::ControlLeft,
                        KeyCode::ControlRight,
                        GamepadButton::LeftThumb,
                    ],
                ),
                (
                    Action::<Sprint>::new(),
                    bindings![
                        KeyCode::ShiftLeft,
                        KeyCode::ShiftRight,
                        GamepadButton::West,
                    ],
                ),
                (
                    Action::<BreakVoxel>::new(),
                    bindings![MouseButton::Left, GamepadButton::South],
                ),
                (
                    Action::<PlaceVoxel>::new(),
                    bindings![MouseButton::Right, GamepadButton::East],
                ),
                (
                    Action::<PlaceEmissiveVoxel>::new(),
                    bindings![KeyCode::KeyE],
                ),
                (
                    Action::<RotateCamera>::new(),
                    Bindings::spawn((
                        Spawn((Binding::mouse_motion(), Scale::splat(0.08))),
                        Axial::right_stick().with(Scale::splat(2.5)),
                    )),
                ),
            ]),
        ))
        .id();

    if crate::cave_scene::enabled() {
        commands.entity(player).insert(bevy_ahoy::CharacterLook {
            yaw: -std::f32::consts::FRAC_PI_4,
            ..default()
        });
    }

    // Automated GI comparisons own the camera path. Background mouse motion
    // must not rotate it and change the streamed/rendered workload between runs.
    #[cfg(debug_assertions)]
    if std::env::var_os("VOXEL_GI_VALIDATE").is_some() {
        commands
            .entity(player)
            .insert(ContextActivity::<PlayerInput>::INACTIVE);
    }

    commands.spawn((
        Name::new("PlayerCamera"),
        PlayerCamera,
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: if crate::cave_scene::enabled() {
                65.0_f32.to_radians()
            } else {
                std::f32::consts::FRAC_PI_4
            },
            ..default()
        }),
        DistanceFog {
            color: sky_color(),
            falloff: FogFalloff::Linear {
                start: FOG_START,
                end: FOG_END,
            },
            ..default()
        },
        CharacterControllerCameraOf::new(player),
        VoxelWorldCamera::<PrototypeWorld>::default(),
    ));
}

fn player_controller() -> CharacterController {
    CharacterController {
        crouch_height: PLAYER_CROUCH_HEIGHT,
        standing_view_height: PLAYER_STANDING_VIEW_HEIGHT,
        crouch_view_height: PLAYER_CROUCH_VIEW_HEIGHT,
        ground_distance: 0.04,
        min_walk_cos: 45.0_f32.to_radians().cos(),
        stop_speed: PLAYER_WALK_SPEED,
        friction_hz: 12.0,
        acceleration_hz: 16.0,
        air_acceleration_hz: PLAYER_AIR_ACCELERATION_HZ,
        gravity: PLAYER_GRAVITY,
        step_size: PLAYER_STEP_SIZE,
        step_down_detection_distance: PLAYER_STEP_SIZE * 2.0,
        crouch_speed_scale: PLAYER_CROUCH_SPEED_SCALE,
        speed: PLAYER_WALK_SPEED,
        max_speed: PLAYER_MAX_SPEED,
        jump_height: PLAYER_JUMP_HEIGHT,
        max_air_wish_speed: PLAYER_MAX_AIR_WISH_SPEED,
        coyote_time: Duration::from_millis(50),
        jump_input_buffer: Duration::from_millis(50),
        ..default()
    }
}

fn update_player_movement_speed(
    movement: Single<&Action<Movement>>,
    sprint: Single<&Action<Sprint>>,
    crouch: Single<&Action<Crouch>>,
    mut player: Single<(&mut CharacterController, &CharacterControllerState), With<Player>>,
) {
    let movement = movement.into_inner();
    let sprint = sprint.into_inner();
    let crouch = crouch.into_inner();

    let sprinting = **sprint && movement.y > 0.0 && !**crouch && !player.1.crouching;

    player.0.speed = if sprinting {
        PLAYER_SPRINT_SPEED
    } else {
        PLAYER_WALK_SPEED
    };
}

fn filter_held_jump(
    jump: Single<&Action<Jump>>,
    mut player: Single<
        (
            &CharacterControllerState,
            &mut AccumulatedInput,
            &mut HeldJumpFilter,
        ),
        With<Player>,
    >,
) {
    let jump = jump.into_inner();

    let (state, input, filter) = &mut *player;
    let holding_jump = **jump;

    if !holding_jump {
        filter.was_grounded = state.grounded.is_some();
        return;
    }

    if state.grounded.is_none() {
        filter.was_grounded = false;
        return;
    }

    if !filter.was_grounded {
        input.jumped = None;
    }

    filter.was_grounded = true;
}

fn reset_player_to_world_spawn(
    world: Res<PrototypeWorld>,
    mut player: Single<(&mut Transform, Option<&mut LinearVelocity>), With<Player>>,
) {
    if !world.is_changed() {
        return;
    }

    let (transform, velocity) = &mut *player;
    transform.translation = world.player_spawn_position();

    if let Some(velocity) = velocity.as_mut() {
        **velocity = LinearVelocity::ZERO;
    }
}

fn edit_voxels(
    break_events: Single<&ActionEvents, With<Action<BreakVoxel>>>,
    place_events: Single<&ActionEvents, With<Action<PlaceVoxel>>>,
    emissive_events: Single<&ActionEvents, With<Action<PlaceEmissiveVoxel>>>,
    mut hud_requests: MessageReader<PlaceEmissiveBlock>,
    cursor_options: Single<&CursorOptions, With<PrimaryWindow>>,
    camera: Single<&GlobalTransform, (With<PlayerCamera>, Without<Player>)>,
    mut voxel_world: VoxelWorld<PrototypeWorld>,
    mut voxel_gi_dirty: ResMut<VoxelGiDirty>,
) {
    let hud_requested = hud_requests.read().count() > 0;
    #[cfg(debug_assertions)]
    if std::env::var_os("VOXEL_GI_VALIDATE").is_some() {
        return;
    }
    let breaking = !cursor_options.visible && break_events.contains(ActionEvents::START);
    let material = placement_material(
        cursor_options.visible,
        place_events.contains(ActionEvents::START),
        emissive_events.contains(ActionEvents::START),
        hud_requested,
    );

    if !breaking && material.is_none() {
        return;
    }

    let ray = Ray3d::new(camera.translation(), camera.forward());
    let Some(hit) = voxel_world.raycast(ray, &|(_, voxel)| matches!(voxel, WorldVoxel::Solid(_)))
    else {
        return;
    };

    if breaking {
        let edited_position = hit.position.as_ivec3();
        voxel_world.set_voxel(edited_position, WorldVoxel::Air);
        voxel_gi_dirty.mark_voxel(edited_position);
    }

    if let Some(material) = material {
        let Some(normal) = hit.normal else {
            return;
        };
        let placement_position = (hit.position + normal).as_ivec3();

        if matches!(
            voxel_world.get_voxel(placement_position),
            WorldVoxel::Air | WorldVoxel::Unset
        ) {
            let voxel = if crate::cave_scene::enabled() {
                // The cave palette uses 4 for lava; the warm emitter shortcut
                // places a torch instead. Ordinary placement extends its rock.
                WorldVoxel::Solid(if material == TerrainMaterial::Emissive {
                    7
                } else {
                    2
                })
            } else {
                material.voxel()
            };
            voxel_world.set_voxel(placement_position, voxel);
            voxel_gi_dirty.mark_voxel(placement_position);
        }
    }
}

fn placement_material(
    cursor_visible: bool,
    ordinary: bool,
    emissive: bool,
    hud: bool,
) -> Option<TerrainMaterial> {
    if hud || (!cursor_visible && emissive) {
        Some(TerrainMaterial::Emissive)
    } else if !cursor_visible && ordinary {
        Some(TerrainMaterial::Grass)
    } else {
        None
    }
}

// Preserve the aimed-at face while the cursor travels to the HUD button. The
// app/inspector context remains active so Esc can still recapture the cursor.
fn sync_player_input(
    cursor: Single<&CursorOptions, With<PrimaryWindow>>,
    contexts: Query<(Entity, &ContextActivity<PlayerInput>)>,
    mut commands: Commands,
) {
    let active = !(cursor.visible
        || cfg!(debug_assertions) && std::env::var_os("VOXEL_GI_VALIDATE").is_some());
    for (entity, activity) in &contexts {
        if **activity != active {
            commands
                .entity(entity)
                .insert(ContextActivity::<PlayerInput>::new(active));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitting_shortcut_and_hud_share_placement_without_replacing_normal_blocks() {
        assert_eq!(
            placement_material(false, false, true, false),
            Some(TerrainMaterial::Emissive)
        );
        assert_eq!(
            placement_material(true, false, false, true),
            Some(TerrainMaterial::Emissive)
        );
        assert_eq!(
            placement_material(false, true, false, false),
            Some(TerrainMaterial::Grass)
        );
        assert_eq!(placement_material(true, true, true, false), None);
        assert_eq!(
            placement_material(false, true, true, false),
            Some(TerrainMaterial::Emissive)
        );
    }

    #[test]
    fn hud_cursor_pauses_player_input_and_esc_can_resume_it() {
        let mut app = App::new();
        app.add_systems(Update, sync_player_input);
        let window = app
            .world_mut()
            .spawn((
                PrimaryWindow,
                CursorOptions {
                    visible: true,
                    ..default()
                },
            ))
            .id();
        let player = app
            .world_mut()
            .spawn(ContextActivity::<PlayerInput>::ACTIVE)
            .id();
        app.update();
        assert!(
            !**app
                .world()
                .get::<ContextActivity<PlayerInput>>(player)
                .unwrap()
        );
        app.world_mut()
            .get_mut::<CursorOptions>(window)
            .unwrap()
            .visible = false;
        app.update();
        assert!(
            **app
                .world()
                .get::<ContextActivity<PlayerInput>>(player)
                .unwrap()
        );
    }
}
