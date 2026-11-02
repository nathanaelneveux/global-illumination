use std::time::Duration;

use avian3d::prelude::*;
use bevy::pbr::{DistanceFog, FogFalloff};
use bevy::prelude::*;
use bevy::window::{CursorOptions, PrimaryWindow};
use bevy_ahoy::input::AccumulatedInput;
use bevy_ahoy::prelude::*;
use bevy_enhanced_input::prelude::*;
use bevy_voxel_world::prelude::{VoxelWorld, VoxelWorldCamera, WorldVoxel};

use crate::terrain::{PrototypeWorld, TerrainMaterial, terrain_view_distance};
use crate::world_environment::sky_color;
use bevy_voxel_gi::VoxelGiInput;
use bevy_voxel_gi_bvw::BvwVoxelEdits;

const FOG_START: f32 = 96.0;
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
const BOMB_RADIUS: i32 = 5;
/// Creative flight (double-tap Space): voxels per second, and Shift's factor.
const FLY_SPEED: f32 = 20.0;
const FLY_SPRINT_FACTOR: f32 = 3.0;
const DOUBLE_TAP_SECONDS: f32 = 0.3;

/// Creative flight, Minecraft style: double-tap Space to toggle. WASD moves
/// along the view's heading, Space rises, Ctrl descends, Shift is faster.
/// Gravity is off while flying; collision stays on.
#[derive(Component, Default)]
struct Flight {
    flying: bool,
    last_space: Option<f32>,
}

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_input_context::<PlayerInput>()
            .add_message::<PlaceEmissiveBlock>()
            .add_message::<BombAtAim>()
            .add_systems(Startup, spawn_player)
            .add_systems(
                FixedPostUpdate,
                (update_player_movement_speed, filter_held_jump)
                    .chain()
                    .before(AhoySystems::MoveCharacters),
            )
            .add_systems(Update, reset_player_to_world_spawn)
            .add_systems(FixedPostUpdate, fly.after(AhoySystems::MoveCharacters));
        if !crate::headless_validation() {
            app.add_systems(
                PreUpdate,
                sync_player_input.before(EnhancedInputSystems::Prepare),
            )
            .add_systems(Update, (edit_voxels, toggle_flight));
            // Validation captures stay free of HUD overlays.
            if std::env::var_os("VOXEL_GI_VALIDATE").is_none() {
                app.add_systems(Startup, spawn_crosshair)
                    .add_systems(Update, show_crosshair_while_aiming);
            }
        }
    }
}

#[derive(Component)]
pub(crate) struct Player;

#[derive(Component, Default)]
struct PlayerInput;

#[derive(Component)]
struct PlayerCamera;

/// Minecraft-style aiming cross at the screen centre.
#[derive(Component)]
struct Crosshair;

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

#[derive(InputAction)]
#[action_output(bool)]
struct DetonateBomb;

/// The HUD requests the same one-shot blast as B, without changing click modes.
#[derive(Message)]
pub(crate) struct BombAtAim;

const CROSSHAIR_LENGTH: f32 = 18.0;
const CROSSHAIR_THICKNESS: f32 = 2.0;

/// Two bars with a dark outline so the cross reads on snow, sky and shadow.
fn spawn_crosshair(mut commands: Commands) {
    let bar = |width: f32, height: f32| {
        (
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px((CROSSHAIR_LENGTH - width) / 2.0),
                top: Val::Px((CROSSHAIR_LENGTH - height) / 2.0),
                width: Val::Px(width),
                height: Val::Px(height),
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.9)),
            BorderColor::all(Color::srgba(0.0, 0.0, 0.0, 0.6)),
        )
    };
    commands.spawn((
        Crosshair,
        Node {
            position_type: PositionType::Absolute,
            left: Val::Percent(50.0),
            top: Val::Percent(50.0),
            width: Val::Px(CROSSHAIR_LENGTH),
            height: Val::Px(CROSSHAIR_LENGTH),
            margin: UiRect {
                left: Val::Px(-CROSSHAIR_LENGTH / 2.0),
                top: Val::Px(-CROSSHAIR_LENGTH / 2.0),
                ..default()
            },
            ..default()
        },
        children![
            bar(CROSSHAIR_LENGTH, CROSSHAIR_THICKNESS + 2.0),
            bar(CROSSHAIR_THICKNESS + 2.0, CROSSHAIR_LENGTH),
        ],
    ));
}

/// Hidden while the cursor is free for the HUD, as when aiming is paused.
fn show_crosshair_while_aiming(
    cursor: Single<&CursorOptions, With<PrimaryWindow>>,
    mut crosshair: Query<&mut Visibility, With<Crosshair>>,
) {
    let wanted = if cursor.visible {
        Visibility::Hidden
    } else {
        Visibility::Inherited
    };
    for mut visibility in &mut crosshair {
        visibility.set_if_neq(wanted);
    }
}

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
            Flight::default(),
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
                    Action::<DetonateBomb>::new(),
                    bindings![KeyCode::KeyB],
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
    if crate::scene_presets::ScenePreset::current() == crate::scene_presets::ScenePreset::Canyon {
        commands.entity(player).insert(bevy_ahoy::CharacterLook {
            yaw: std::f32::consts::PI,
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
            fov: if crate::scene_presets::ScenePreset::current()
                != crate::scene_presets::ScenePreset::Terrain
            {
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
                end: terrain_view_distance(),
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

fn toggle_flight(
    keyboard: Res<ButtonInput<KeyCode>>,
    cursor: Single<&CursorOptions, With<PrimaryWindow>>,
    time: Res<Time>,
    mut player: Single<
        (
            &mut Flight,
            &mut CharacterController,
            Option<&mut LinearVelocity>,
        ),
        With<Player>,
    >,
) {
    if cursor.visible || !keyboard.just_pressed(KeyCode::Space) {
        return;
    }
    let now = time.elapsed_secs();
    let (flight, controller, velocity) = &mut *player;
    if !flight
        .last_space
        .is_some_and(|last| now - last < DOUBLE_TAP_SECONDS)
    {
        flight.last_space = Some(now);
        return;
    }
    flight.last_space = None;
    flight.flying = !flight.flying;
    controller.gravity = if flight.flying { 0.0 } else { PLAYER_GRAVITY };
    if let Some(velocity) = velocity.as_mut() {
        ***velocity = Vec3::ZERO;
    }
    info!(flying = flight.flying, "creative flight");
}

/// Moves a flying player after the controller's own step, which then sees
/// no momentum to carry (gravity is off).
fn fly(
    time: Res<Time>,
    movement: Single<&Action<Movement>>,
    jump: Single<&Action<Jump>>,
    crouch: Single<&Action<Crouch>>,
    sprint: Single<&Action<Sprint>>,
    mut player: Single<
        (
            &Flight,
            Option<&bevy_ahoy::CharacterLook>,
            &mut Transform,
            Option<&mut LinearVelocity>,
        ),
        With<Player>,
    >,
) {
    let (flight, look, transform, velocity) = &mut *player;
    if !flight.flying {
        return;
    }
    let heading = Quat::from_rotation_y(look.map_or(0.0, |look| look.yaw));
    let movement = **movement.into_inner();
    let mut direction = heading * Vec3::X * movement.x + heading * Vec3::NEG_Z * movement.y;
    if **jump.into_inner() {
        direction += Vec3::Y;
    }
    if **crouch.into_inner() {
        direction -= Vec3::Y;
    }
    let speed = FLY_SPEED
        * if **sprint.into_inner() {
            FLY_SPRINT_FACTOR
        } else {
            1.0
        };
    transform.translation += direction.normalize_or_zero() * speed * time.delta_secs();
    if let Some(velocity) = velocity.as_mut() {
        ***velocity = Vec3::ZERO;
    }
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
    bomb_events: Single<&ActionEvents, With<Action<DetonateBomb>>>,
    mut bomb_requests: MessageReader<BombAtAim>,
    mut hud_requests: MessageReader<PlaceEmissiveBlock>,
    cursor_options: Single<&CursorOptions, With<PrimaryWindow>>,
    camera: Single<&GlobalTransform, (With<PlayerCamera>, Without<Player>)>,
    mut voxel_world: VoxelWorld<PrototypeWorld>,
    mut voxel_gi_dirty: ResMut<VoxelGiInput>,
) {
    let hud_requested = hud_requests.read().count() > 0;
    let hud_bomb = bomb_requests.read().count() > 0;
    #[cfg(debug_assertions)]
    if std::env::var_os("VOXEL_GI_VALIDATE").is_some() {
        return;
    }
    let bombing =
        hud_bomb || (!cursor_options.visible && bomb_events.contains(ActionEvents::START));
    let breaking = !cursor_options.visible && break_events.contains(ActionEvents::START);
    let material = placement_material(
        cursor_options.visible,
        place_events.contains(ActionEvents::START),
        emissive_events.contains(ActionEvents::START),
        hud_requested,
    );

    if !breaking && !bombing && material.is_none() {
        return;
    }

    let ray = Ray3d::new(camera.translation(), camera.forward());
    let Some(hit) = voxel_world.raycast(ray, &|(_, voxel)| matches!(voxel, WorldVoxel::Solid(_)))
    else {
        return;
    };

    if breaking || bombing {
        let radius = if bombing { BOMB_RADIUS } else { 0 };
        for position in destruction_positions(hit.position.as_ivec3(), radius) {
            // Do not manufacture air edits in unloaded chunks or rewrite air.
            // Every actual removal uses the ordinary tracked edit gateway so
            // meshing, collision and GI observe the same blast.
            if matches!(voxel_world.get_voxel(position), WorldVoxel::Solid(_)) {
                voxel_gi_dirty.set_voxel(&mut voxel_world, position, WorldVoxel::Air);
            }
        }
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
            voxel_gi_dirty.set_voxel(&mut voxel_world, placement_position, voxel);
        }
    }
}

/// Integer voxel centers inside a sphere, including its radius boundary.
/// Radius zero is the ordinary single-block tool. Iteration needs no allocation.
fn destruction_positions(center: IVec3, radius: i32) -> impl Iterator<Item = IVec3> {
    (-radius..=radius).flat_map(move |x| {
        (-radius..=radius).flat_map(move |y| {
            (-radius..=radius).filter_map(move |z| {
                let offset = IVec3::new(x, y, z);
                (offset.length_squared() <= radius * radius).then_some(center + offset)
            })
        })
    })
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
    fn bomb_removes_a_radius_five_sphere_around_the_clicked_voxel() {
        let center = IVec3::new(-32, 17, 64);
        let targets: std::collections::HashSet<_> =
            destruction_positions(center, BOMB_RADIUS).collect();
        assert_eq!(targets.len(), 515);
        assert!(targets.contains(&center));
        for offset in [IVec3::X, IVec3::Y, IVec3::Z] {
            assert!(targets.contains(&(center + offset * 5)));
            assert!(targets.contains(&(center - offset * 5)));
            assert!(!targets.contains(&(center + offset * 6)));
        }
        assert!(!targets.contains(&(center + IVec3::splat(5))));
        for &position in &targets {
            let offset = position - center;
            assert!(offset.length_squared() <= 25);
            assert!(targets.contains(&(center - offset)));
        }
        assert_eq!(
            destruction_positions(center, 0).collect::<Vec<_>>(),
            [center]
        );
    }

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
