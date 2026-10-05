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
            .add_systems(Startup, spawn_player)
            .add_systems(
                FixedPostUpdate,
                (update_player_movement_speed, filter_held_jump)
                    .chain()
                    .before(AhoySystems::MoveCharacters),
            )
            .add_systems(Update, (edit_voxels, reset_player_to_world_spawn));
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
                    Action::<RotateCamera>::new(),
                    Bindings::spawn((
                        Spawn((Binding::mouse_motion(), Scale::splat(0.08))),
                        Axial::right_stick().with(Scale::splat(2.5)),
                    )),
                ),
            ]),
        ))
        .id();

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
    cursor_options: Single<&CursorOptions, With<PrimaryWindow>>,
    camera: Single<&GlobalTransform, (With<PlayerCamera>, Without<Player>)>,
    mut voxel_world: VoxelWorld<PrototypeWorld>,
    mut voxel_gi_dirty: ResMut<VoxelGiDirty>,
) {
    let breaking = break_events.contains(ActionEvents::START);
    let placing = place_events.contains(ActionEvents::START);

    if cursor_options.visible || (!breaking && !placing) {
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

    if placing {
        let Some(normal) = hit.normal else {
            return;
        };
        let placement_position = (hit.position + normal).as_ivec3();

        if matches!(
            voxel_world.get_voxel(placement_position),
            WorldVoxel::Air | WorldVoxel::Unset
        ) {
            voxel_world.set_voxel(placement_position, TerrainMaterial::Grass.voxel());
            voxel_gi_dirty.mark_voxel(placement_position);
        }
    }
}
