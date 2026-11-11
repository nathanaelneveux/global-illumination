//! Headless checks that the Bevy 0.20 / Avian 0.8 port still behaves like a
//! character controller: it lands on a floor, stands still there, and walks
//! when its movement action fires. Added with the vendored port; not upstream.

use avian3d::prelude::*;
use bevy::{prelude::*, time::TimeUpdateStrategy};
use bevy_ahoy::prelude::*;
use bevy_enhanced_input::prelude::*;
use core::time::Duration;

#[derive(Component)]
struct Player;

const HEIGHT: f32 = 1.8;
const STEP: Duration = Duration::from_micros(15_625); // 64 Hz, the fixed timestep.

fn app() -> App {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        TransformPlugin,
        bevy::input::InputPlugin,
        PhysicsPlugins::default(),
        EnhancedInputPlugin,
        AhoyPlugins::default(),
    ))
    .add_input_context::<Player>()
    .insert_resource(TimeUpdateStrategy::ManualDuration(STEP));
    // `App::run` would do this; spawn observers need the resources plugins
    // insert while finishing.
    app.finish();
    app.cleanup();
    // A 40 x 40 floor whose top face is y = 0.
    app.world_mut().spawn((
        RigidBody::Static,
        Collider::cuboid(40.0, 1.0, 40.0),
        Transform::from_xyz(0.0, -0.5, 0.0),
    ));
    app
}

fn run(app: &mut App, seconds: f32) {
    for _ in 0..(seconds / STEP.as_secs_f32()).round() as u32 {
        app.update();
    }
}

/// Highest the controller's centre may float: it stands within
/// `ground_distance` plus the move-and-slide skin of the floor, by design.
fn max_standing_y() -> f32 {
    let config = CharacterController::default();
    HEIGHT / 2.0 + config.ground_distance + config.move_and_slide.skin_width
}

fn player_translation(app: &mut App) -> Vec3 {
    let world = app.world_mut();
    world
        .query_filtered::<&Transform, With<Player>>()
        .single(world)
        .unwrap()
        .translation
}

#[test]
fn falls_onto_the_floor_and_stands_on_it() {
    let mut app = app();
    app.world_mut().spawn((
        Player,
        CharacterController::default(),
        Collider::cylinder(0.4, HEIGHT),
        Transform::from_xyz(0.0, 3.0, 0.0),
    ));
    run(&mut app, 2.0);
    let landed = player_translation(&mut app);
    run(&mut app, 1.0);
    let rested = player_translation(&mut app);

    eprintln!("landed at {landed}, rested at {rested}");
    assert!(
        (HEIGHT / 2.0..=max_standing_y()).contains(&landed.y),
        "landed at {landed}, expected y in {}..={}",
        HEIGHT / 2.0,
        max_standing_y()
    );
    assert!(
        landed.distance(rested) < 1e-3,
        "drifted from {landed} to {rested} while standing"
    );
}

#[test]
fn walks_while_movement_fires() {
    let mut app = app();
    app.world_mut().spawn((
        Player,
        CharacterController::default(),
        Collider::cylinder(0.4, HEIGHT),
        Transform::from_xyz(0.0, HEIGHT / 2.0, 0.0),
        related!(Actions<Player>[(
            Action::<Movement>::new(),
            ActionMock::new(TriggerState::Fired, Vec2::Y, MockSpan::Manual),
        )]),
    ));
    run(&mut app, 0.5);
    let start = player_translation(&mut app);
    run(&mut app, 1.0);
    let end = player_translation(&mut app);

    let walked = (end - start).xz().length();
    eprintln!("walked {walked} m in a second ({start} -> {end})");
    assert!(
        walked > 2.0,
        "walked {walked} m in a second ({start} -> {end})"
    );
    assert!(
        (HEIGHT / 2.0..=max_standing_y()).contains(&end.y),
        "left the floor while walking: {end}"
    );
}
