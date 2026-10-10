//! Opt-in whole-schedule wall times, including deferred work and task waits.
//! GI body timers alone cannot attribute a streaming stall to terrain or physics.
use bevy::app::{MainScheduleOrder, RunFixedMainLoop, SpawnScene};
use bevy::ecs::schedule::ScheduleLabel;
use bevy::prelude::*;
use std::time::Instant;

pub(super) const STAGES: [&str; 7] = [
    "first",
    "pre_update",
    "fixed_loop",
    "update",
    "spawn_scene",
    "post_update",
    "last",
];

#[derive(Resource, Default)]
pub(in crate::voxel_gi) struct HostTiming {
    /// Last fully completed Main iteration; current Update must not overwrite it.
    pub completed_ms: [f32; 7],
    current_ms: [f32; 7],
    boundary: Option<Instant>,
}

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct Boundary(usize);

pub(super) fn install(app: &mut App) {
    app.init_resource::<HostTiming>();
    {
        let mut order = app.world_mut().resource_mut::<MainScheduleOrder>();
        order.insert_before(First, Boundary(0));
        order.insert_before(PreUpdate, Boundary(1));
        order.insert_before(RunFixedMainLoop, Boundary(2));
        order.insert_before(Update, Boundary(3));
        order.insert_before(SpawnScene, Boundary(4));
        order.insert_before(PostUpdate, Boundary(5));
        order.insert_before(Last, Boundary(6));
        order.insert_after(Last, Boundary(7));
    }
    for index in 0..=7 {
        // Main already serializes schedules. Exclusive tiny boundary systems
        // need no executor task dispatch and do not constrain systems within them.
        app.add_systems(Boundary(index), move |world: &mut World| {
            let now = Instant::now();
            let mut timing = world.resource_mut::<HostTiming>();
            if index != 0 {
                timing.current_ms[index - 1] =
                    now.duration_since(timing.boundary.unwrap()).as_secs_f32() * 1000.0;
            }
            timing.boundary = Some(now);
            if index == 7 {
                timing.completed_ms = timing.current_ms;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_iteration_survives_the_next_update() {
        let mut app = App::new();
        install(&mut app);
        app.update();
        let previous = app.world().resource::<HostTiming>().completed_ms;
        assert!(previous.iter().all(|ms| *ms > 0.0));
        app.add_systems(Update, move |timing: Res<HostTiming>| {
            assert_eq!(timing.completed_ms, previous);
        });
        app.update();
    }
}
