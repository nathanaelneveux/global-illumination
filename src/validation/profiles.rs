//! Workload identity and presentation are explicit; toggle stress is never a
//! proxy for the cost or appearance of the normal gameplay mode.
use voxel_gi::{GiDebugView, VoxelGiDisplay};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Profile {
    #[default]
    Gameplay,
    Streaming,
    Motion,
    House,
    HouseEdits,
    Response,
    Edits,
    Tunnel,
    Sources,
    Cave,
    Canyon,
    DayCycle,
    Structures,
    Hallway,
    LiveCave,
    LiveHallway,
    Compare,
    Stress,
    Lod,
    Overlook,
}

/// `overlook` repeats a playtest that left far-band faces shimmering: walk
/// 100 voxels from spawn and back, rise `VOXEL_GI_VALIDATE_OVERLOOK_HEIGHT`
/// (default 40) above it, and hold still looking out. With
/// `VOXEL_GI_VALIDATE_OVERLOOK_DIRECT=1` it rises without the walk (the
/// control that did not shimmer).
pub(super) fn overlook_height() -> f32 {
    std::env::var("VOXEL_GI_VALIDATE_OVERLOOK_HEIGHT")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(40.0)
}

/// `VOXEL_GI_VALIDATE_OVERLOOK_TURN=1`: hold facing back toward spawn.
pub(super) fn overlook_turn() -> bool {
    std::env::var_os("VOXEL_GI_VALIDATE_OVERLOOK_TURN").is_some()
}

pub(super) fn overlook_direct() -> bool {
    std::env::var_os("VOXEL_GI_VALIDATE_OVERLOOK_DIRECT").is_some()
}

/// One capture on rising, a 24-frame burst while holding still, and one late.
static OVERLOOK_CAPTURES: std::sync::LazyLock<Vec<(f32, &'static str)>> =
    std::sync::LazyLock::new(|| {
        let mut captures = vec![(34.0, "risen"), (40.0, "hold-40s")];
        for frame in 0..24 {
            captures.push((
                50.0,
                &*Box::leak(format!("hold-{frame:02}").into_boxed_str()),
            ));
        }
        captures.extend([(60.0, "hold-60s"), (68.0, "hold-late")]);
        captures
    });

/// `motion` walk mode (VOXEL_GI_VALIDATE_MOTION_WALK=<voxels per second>).
pub(super) fn motion_walk() -> Option<f32> {
    static WALK: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *WALK.get_or_init(|| {
        std::env::var("VOXEL_GI_VALIDATE_MOTION_WALK")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
    })
}

/// `streaming` turns 180 degrees at 15 s and stands still, capturing 24
/// consecutive frames from the turn (VOXEL_GI_VALIDATE_TURN_BURST=1).
pub(super) fn turn_burst() -> bool {
    static TURN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TURN.get_or_init(|| std::env::var_os("VOXEL_GI_VALIDATE_TURN_BURST").is_some())
}

static TURN_CAPTURES: std::sync::LazyLock<Vec<(f32, &'static str)>> =
    std::sync::LazyLock::new(|| {
        let mut captures = vec![(14.5, "before-turn")];
        for frame in 0..24 {
            captures.push((
                15.0,
                &*Box::leak(format!("turn-{frame:02}").into_boxed_str()),
            ));
        }
        captures.extend([(16.0, "turn-1s"), (18.0, "turn-3s")]);
        captures
    });

/// `edits` places an emitter instead of rock at 19 s and captures 16
/// consecutive frames from the edit (VOXEL_GI_VALIDATE_PLACE_EMITTER=1).
pub(super) fn place_emitter() -> bool {
    static PLACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PLACE.get_or_init(|| std::env::var_os("VOXEL_GI_VALIDATE_PLACE_EMITTER").is_some())
}

static PLACE_EMITTER_CAPTURES: std::sync::LazyLock<Vec<(f32, &'static str)>> =
    std::sync::LazyLock::new(|| {
        let mut captures = vec![(14.5, "room-settled")];
        for frame in 0..16 {
            captures.push((
                19.0,
                &*Box::leak(format!("emitter-place-{frame:02}").into_boxed_str()),
            ));
        }
        captures.push((21.0, "emitter-placed-later"));
        captures
    });

/// Two bursts of consecutive frames while walking (one capture per frame).
static WALK_CAPTURES: std::sync::LazyLock<Vec<(f32, &'static str)>> =
    std::sync::LazyLock::new(|| {
        let mut captures = vec![(14.4, "before")];
        for (at, burst) in [(20.0, "a"), (26.0, "b")] {
            for frame in 0..16 {
                captures.push((
                    at,
                    &*Box::leak(format!("walk-{burst}-{frame:02}").into_boxed_str()),
                ));
            }
        }
        captures
    });

impl Profile {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "gameplay" => Ok(Self::Gameplay),
            "streaming" => Ok(Self::Streaming),
            "motion" => Ok(Self::Motion),
            "house" => Ok(Self::House),
            "house-edits" => Ok(Self::HouseEdits),
            "response" => Ok(Self::Response),
            "edits" => Ok(Self::Edits),
            "tunnel" => Ok(Self::Tunnel),
            "sources" => Ok(Self::Sources),
            "cave" => Ok(Self::Cave),
            "canyon" => Ok(Self::Canyon),
            "day-cycle" => Ok(Self::DayCycle),
            "structures" => Ok(Self::Structures),
            "hallway" => Ok(Self::Hallway),
            "live-cave" => Ok(Self::LiveCave),
            "live-hallway" => Ok(Self::LiveHallway),
            "compare" => Ok(Self::Compare),
            "stress" => Ok(Self::Stress),
            "lod" => Ok(Self::Lod),
            "overlook" => Ok(Self::Overlook),
            _ => Err(format!(
                "unknown validation profile {value:?}; choose gameplay, streaming, motion, house, house-edits, response, edits, tunnel, sources, cave, canyon, day-cycle, structures, hallway, live-cave, live-hallway, compare, stress, lod, or overlook"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Gameplay => "gameplay",
            Self::Streaming => "streaming",
            Self::Motion => "motion",
            Self::House => "house",
            Self::HouseEdits => "house-edits",
            Self::Response => "response",
            Self::Edits => "edits",
            Self::Tunnel => "tunnel",
            Self::Sources => "sources",
            Self::Cave => "cave",
            Self::Canyon => "canyon",
            Self::DayCycle => "day-cycle",
            Self::Structures => "structures",
            Self::Hallway => "hallway",
            Self::LiveCave => "live-cave",
            Self::LiveHallway => "live-hallway",
            Self::Compare => "compare",
            Self::Stress => "stress",
            Self::Lod => "lod",
            Self::Overlook => "overlook",
        }
    }

    pub fn duration(self) -> f32 {
        if self == Self::Cave && super::skylight::mode().is_some() {
            return 46.0;
        }
        if self == Self::Cave && std::env::var_os("VOXEL_GI_VALIDATE_CAVE_LONG").is_some() {
            return 182.0;
        }
        match self {
            Self::Compare => 29.0,
            Self::Lod => 40.0,
            Self::Overlook => 70.0,
            Self::Streaming => 35.0,
            Self::Motion => 32.0,
            Self::House => 90.0,
            Self::LiveCave | Self::LiveHallway => 182.0,
            Self::DayCycle => 183.0,
            Self::Cave | Self::Canyon | Self::Structures | Self::Hallway => 62.0,
            Self::Tunnel => 60.0,
            Self::HouseEdits => 180.0,
            Self::Response | Self::Edits => 30.0,
            _ => 72.0,
        }
    }

    pub fn edits(self) -> bool {
        matches!(self, Self::Gameplay | Self::Stress)
    }

    pub fn sources(self) -> bool {
        matches!(self, Self::Sources | Self::Stress)
    }

    pub fn captures(self) -> &'static [(f32, &'static str)] {
        if self == Self::Overlook {
            return &OVERLOOK_CAPTURES;
        }
        if self == Self::Tunnel {
            return &[
                (14.5, "entrance"),
                (15.05, "depth01-0050ms"),
                (15.4, "depth01-0400ms"),
                (16.7, "depth01-1700ms"),
                (17.05, "depth02-0050ms"),
                (17.4, "depth02-0400ms"),
                (18.7, "depth02-1700ms"),
                (21.05, "depth04-0050ms"),
                (21.4, "depth04-0400ms"),
                (22.7, "depth04-1700ms"),
                (29.05, "depth08-0050ms"),
                (29.4, "depth08-0400ms"),
                (30.7, "depth08-1700ms"),
                (37.05, "depth12-0050ms"),
                (37.4, "depth12-0400ms"),
                (38.7, "depth12-1700ms"),
                (53.05, "depth20-0050ms"),
                (53.4, "depth20-0400ms"),
                (54.7, "depth20-1700ms"),
                (59.0, "tunnel-later"),
            ];
        }
        if self == Self::Streaming && turn_burst() {
            return &TURN_CAPTURES;
        }
        if self == Self::Motion {
            if motion_walk().is_some() {
                return &WALK_CAPTURES;
            }
            return &[
                (14.4, "before"),
                (16.4, "base-1"),
                (17.4, "turned-1"),
                (18.4, "base-2"),
                (19.4, "turned-2"),
                (30.4, "returned"),
            ];
        }
        if super::handoff::enabled(self) {
            return super::handoff::CAPTURES;
        }
        if matches!(self, Self::LiveCave | Self::LiveHallway) {
            return &[
                (15., "live-15s"),
                (30., "live-30s"),
                (60., "live-60s"),
                (120., "live-120s"),
                (180., "live-180s"),
            ];
        }
        if self == Self::Hallway {
            return &[
                (15., "hallway-15s"),
                (30., "hallway-30s"),
                (60., "hallway-60s"),
            ];
        }
        if self == Self::DayCycle && super::day::turn() {
            return super::day::TURN_CAPTURES;
        }
        if self == Self::DayCycle {
            // Pairs one second apart: the sun moves a fraction of a degree,
            // so their difference is mostly flicker, not lighting change.
            return &[
                (30., "day-030s"),
                (31., "day-031s"),
                (60., "day-060s"),
                (61., "day-061s"),
                (90., "day-090s"),
                (91., "day-091s"),
                (120., "day-120s"),
                (121., "day-121s"),
                (150., "day-150s"),
                (151., "day-151s"),
                (180., "day-180s"),
                (181., "day-181s"),
            ];
        }
        if self == Self::Lod {
            return &[
                (10.0, "lod-10s"),
                (20.0, "lod-20s"),
                (30.0, "lod-30s"),
                (39.0, "lod-39s"),
            ];
        }
        if self == Self::Canyon {
            return &[
                (15.0, "canyon-15s"),
                (30.0, "canyon-30s"),
                (60.0, "canyon-60s"),
            ];
        }
        if self == Self::Structures {
            &[
                (10.0, "house-and-path"),
                (22.0, "entrance"),
                (45.0, "cave-21s"),
                (60.0, "cave-36s"),
            ]
        } else if self == Self::Cave {
            if super::skylight::mode().is_some() {
                if super::skylight::sun().is_some() {
                    return super::skylight::SUN_CAPTURES;
                }
                return super::skylight::CAPTURES;
            }
            if std::env::var_os("VOXEL_GI_VALIDATE_CAVE_LONG").is_some() {
                return &[
                    (15.0, "cave-15s"),
                    (30.0, "cave-30s"),
                    (60.0, "cave-60s"),
                    (120.0, "cave-120s"),
                    (180.0, "cave-180s"),
                ];
            }
            &[(15.0, "cave-15s"), (30.0, "cave-30s"), (60.0, "cave-60s")]
        } else if self == Self::HouseEdits {
            &[
                (89.0, "pre-edit"),
                (90.4, "skylight-0400ms"),
                (100.0, "skylight-10s"),
                (119.0, "skylight-29s"),
                (120.4, "west-door-0400ms"),
                (149.0, "west-door-29s"),
                (150.4, "east-door-turn-0400ms"),
                (160.0, "east-door-turn-10s"),
                (179.0, "east-door-turn-29s"),
            ]
        } else if self == Self::House {
            &[
                (29.0, "first-corner"),
                (30.4, "turned-0400ms"),
                (40.0, "turned-10s"),
                (49.0, "second-corner"),
                (50.4, "return-0400ms"),
                (69.0, "first-corner-returned"),
                (89.0, "second-corner-returned"),
            ]
        } else if self == Self::Edits && place_emitter() {
            &PLACE_EMITTER_CAPTURES
        } else if self == Self::Edits {
            &[
                (14.5, "room-settled"),
                (15.05, "dig-0050ms"),
                (15.15, "dig-0150ms"),
                (15.4, "dig-0400ms"),
                (17.05, "restore-0050ms"),
                (17.15, "restore-0150ms"),
                (17.4, "restore-0400ms"),
                (19.05, "place-0050ms"),
                (19.15, "place-0150ms"),
                (19.4, "place-0400ms"),
                (21.05, "remove-0050ms"),
                (21.15, "remove-0150ms"),
                (21.4, "remove-0400ms"),
                (24.5, "surface-settled"),
                (25.05, "surface-dig-0050ms"),
                (25.15, "surface-dig-0150ms"),
                (25.4, "surface-dig-0400ms"),
                (29.0, "surface-recovered"),
            ]
        } else if self == Self::Response {
            &[
                (14.5, "room-settled"),
                (15.1, "dig-0100ms"),
                (15.3, "dig-0300ms"),
                (15.6, "dig-0600ms"),
                (16.0, "dig-1000ms"),
                (18.0, "cavity-settled"),
                (20.2, "emitter-add-0200ms"),
                (21.0, "emitter-add-1000ms"),
                (24.5, "emitter-settled"),
                (25.1, "emitter-remove-0100ms"),
                (25.3, "emitter-remove-0300ms"),
                (25.6, "emitter-remove-0600ms"),
                (26.0, "emitter-remove-1000ms"),
                (29.0, "room-recovered"),
            ]
        } else if self == Self::Streaming {
            &[(14.0, "settled"), (29.0, "streaming"), (34.0, "recovered")]
        } else if self == Self::Compare {
            &[
                (15.0, "preferred-corners"),
                (17.5, "pixel-sun"),
                (19.5, "flat-voxel"),
                (21.5, "irradiance-corners"),
                (23.5, "irradiance-flat"),
                (25.5, "indirect-disabled"),
                (27.5, "preferred-restored"),
            ]
        } else {
            &[
                (15.0, "preferred"),
                (24.0, "settled"),
                (44.0, "active"),
                (64.0, "streaming"),
                (71.0, "recovered"),
            ]
        }
    }

    pub fn phase(self, elapsed: f32) -> &'static str {
        if self == Self::Motion {
            return if elapsed < 15.0 {
                "settling"
            } else {
                "small-turns"
            };
        }
        if self == Self::Lod {
            return if elapsed < 10.0 { "startup" } else { "flight" };
        }
        if self == Self::Overlook {
            return match elapsed {
                t if t < 10.0 => "startup",
                t if t < 30.0 => "walk",
                _ => "hold",
            };
        }
        if self == Self::Structures {
            if elapsed < 12.0 {
                "approach"
            } else if elapsed < 24.0 {
                "entrance"
            } else {
                "cave"
            }
        } else if matches!(
            self,
            Self::Cave
                | Self::Canyon
                | Self::DayCycle
                | Self::Hallway
                | Self::LiveCave
                | Self::LiveHallway
        ) {
            if elapsed < 15.0 {
                "startup"
            } else {
                "settling"
            }
        } else if self == Self::HouseEdits && elapsed >= 90.0 {
            match elapsed {
                t if t < 120.0 => "skylight",
                t if t < 150.0 => "west-door",
                _ => "east-door-turn",
            }
        } else if matches!(self, Self::House | Self::HouseEdits) {
            match elapsed {
                t if t < 30.0 => "first-corner",
                t if t < 50.0 => "second-corner",
                t if t < 70.0 => "first-corner-returned",
                _ => "second-corner-returned",
            }
        } else if self == Self::Edits {
            match elapsed {
                t if t < 15.0 => "settling",
                t if t < 23.0 => "underground-edits",
                _ => "surface-edit",
            }
        } else if self == Self::Response {
            match elapsed {
                t if t < 15.0 => "settling",
                t if t < 20.0 => "cavity",
                t if t < 25.0 => "emitter-added",
                _ => "emitter-removed",
            }
        } else if self == Self::Streaming {
            match elapsed {
                t if t < 10.0 => "startup",
                t if t < 15.0 => "settled",
                t if t < 30.0 => "streaming",
                _ => "recovery",
            }
        } else if self == Self::Compare {
            "comparison"
        } else if elapsed < 10.0 {
            "startup"
        } else if elapsed < 25.0 {
            "settled"
        } else if elapsed < 45.0 {
            if self == Self::Sources {
                "sources"
            } else {
                "editing"
            }
        } else if elapsed < 65.0 {
            "streaming"
        } else {
            "recovery"
        }
    }

    pub fn configure(self, elapsed: f32, settings: &mut VoxelGiDisplay) {
        *settings = VoxelGiDisplay::default();
        if self == Self::Stress && (25.0..65.0).contains(&elapsed) {
            let step = ((elapsed - 25.0) * 8.0) as u32;
            settings.enabled = step & 1 != 0;
            settings.corner_reconstruction = (step >> 1) & 1 != 0;
            settings.debug_view = GiDebugView::from_index((step >> 2) & 1);
            settings.voxel_sun = (step >> 3) & 1 != 0;
        } else if self == Self::Compare {
            match elapsed as u32 {
                16..=17 => settings.voxel_sun = false,
                18..=19 => settings.corner_reconstruction = false,
                20..=21 => settings.debug_view = GiDebugView::Irradiance,
                22..=23 => {
                    settings.debug_view = GiDebugView::Irradiance;
                    settings.corner_reconstruction = false;
                }
                24..=25 => settings.enabled = false,
                _ => {}
            }
        }
    }
}

// Shader diagnostic modes are a material contract: 0 appearance, 1 combined
// irradiance, 2 coverage, 3 indirect, 4 direct sunlight, 5 exact corner sunlight.
// Mode 6 reports missing support and invalid exposed-face identity separately.
// Capture-only controls; ordinary gameplay/performance retains mode zero.
pub(super) fn motion_diagnostic(
    coverage: bool,
    component: Option<&str>,
) -> Result<u32, &'static str> {
    if coverage && component.is_some() {
        return Err("Choose VIEW_COVERAGE or LIGHT_COMPONENT, not both.");
    }
    match component {
        None => Ok(if coverage { 2 } else { 0 }),
        Some("combined") => Ok(1),
        Some("indirect") => Ok(3),
        Some("sun") => Ok(4),
        Some("sun-reference") => Ok(5),
        Some("face-validity") => Ok(6),
        _ => Err(
            "VOXEL_GI_VALIDATE_LIGHT_COMPONENT must be combined, indirect, sun, sun-reference or face-validity.",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motion_diagnostics_are_explicit_and_exclusive() {
        for (component, expected) in [
            (None, 0),
            (Some("combined"), 1),
            (Some("indirect"), 3),
            (Some("sun"), 4),
            (Some("sun-reference"), 5),
            (Some("face-validity"), 6),
        ] {
            assert_eq!(motion_diagnostic(false, component), Ok(expected));
        }
        assert_eq!(motion_diagnostic(true, None), Ok(2));
        assert!(motion_diagnostic(true, Some("sun")).is_err());
        assert!(motion_diagnostic(false, Some("unknown")).is_err());
    }

    #[test]
    fn gameplay_and_sources_never_switch_away_from_preferred_look() {
        for profile in [
            Profile::Gameplay,
            Profile::Streaming,
            Profile::Motion,
            Profile::House,
            Profile::HouseEdits,
            Profile::Response,
            Profile::Edits,
            Profile::Tunnel,
            Profile::Sources,
            Profile::Cave,
            Profile::Structures,
            Profile::Hallway,
        ] {
            for tick in 0..720 {
                let mut settings = VoxelGiDisplay::default();
                profile.configure(tick as f32 / 10.0, &mut settings);
                assert_eq!(settings, VoxelGiDisplay::default());
            }
        }
        assert!(!Profile::Gameplay.sources());
        assert!(!Profile::Sources.edits());
        assert!(Profile::parse("typo").is_err());
    }

    #[test]
    fn comparison_labels_match_explicit_modes_and_restore_defaults() {
        let expected = [
            (true, true, true, 0),
            (true, true, false, 0),
            (true, false, true, 0),
            (true, true, true, 1),
            (true, false, true, 1),
            (false, true, true, 0),
            (true, true, true, 0),
        ];
        for (&(at, _), expected) in Profile::Compare.captures().iter().zip(expected) {
            let mut settings = VoxelGiDisplay::default();
            Profile::Compare.configure(at, &mut settings);
            assert_eq!(
                (
                    settings.enabled,
                    settings.corner_reconstruction,
                    settings.voxel_sun,
                    settings.debug_view.index()
                ),
                expected
            );
        }
        let mut settings = VoxelGiDisplay::default();
        Profile::Stress.configure(71.0, &mut settings);
        assert!(settings.voxel_sun);
    }
}
