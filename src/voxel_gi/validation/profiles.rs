//! Workload identity and presentation are explicit; toggle stress is never a
//! proxy for the cost or appearance of the normal gameplay mode.
use super::super::render::GpuTransportSettings;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Profile {
    #[default]
    Gameplay,
    Streaming,
    House,
    HouseEdits,
    Response,
    Edits,
    Sources,
    Cave,
    Compare,
    Stress,
}

impl Profile {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "gameplay" => Ok(Self::Gameplay),
            "streaming" => Ok(Self::Streaming),
            "house" => Ok(Self::House),
            "house-edits" => Ok(Self::HouseEdits),
            "response" => Ok(Self::Response),
            "edits" => Ok(Self::Edits),
            "sources" => Ok(Self::Sources),
            "cave" => Ok(Self::Cave),
            "compare" => Ok(Self::Compare),
            "stress" => Ok(Self::Stress),
            _ => Err(format!(
                "unknown validation profile {value:?}; choose gameplay, streaming, house, house-edits, response, edits, sources, cave, compare, or stress"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Gameplay => "gameplay",
            Self::Streaming => "streaming",
            Self::House => "house",
            Self::HouseEdits => "house-edits",
            Self::Response => "response",
            Self::Edits => "edits",
            Self::Sources => "sources",
            Self::Cave => "cave",
            Self::Compare => "compare",
            Self::Stress => "stress",
        }
    }

    pub fn duration(self) -> f32 {
        match self {
            Self::Compare => 29.0,
            Self::Streaming => 35.0,
            Self::House => 90.0,
            Self::Cave => 62.0,
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
        if self == Self::Cave {
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
        if self == Self::Cave {
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

    pub fn configure(self, elapsed: f32, settings: &mut GpuTransportSettings) {
        settings.reset_display();
        if self == Self::Stress && (25.0..65.0).contains(&elapsed) {
            let step = ((elapsed - 25.0) * 8.0) as u32;
            settings.enabled = step & 1;
            settings.reconstruction = (step >> 1) & 1;
            settings.debug_irradiance = (step >> 2) & 1;
            settings.voxel_sky = (step >> 3) & 1;
        } else if self == Self::Compare {
            match elapsed as u32 {
                16..=17 => settings.voxel_sky = 0,
                18..=19 => settings.reconstruction = 0,
                20..=21 => settings.debug_irradiance = 1,
                22..=23 => {
                    settings.debug_irradiance = 1;
                    settings.reconstruction = 0;
                }
                24..=25 => settings.enabled = 0,
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gameplay_and_sources_never_switch_away_from_preferred_look() {
        for profile in [
            Profile::Gameplay,
            Profile::Streaming,
            Profile::House,
            Profile::HouseEdits,
            Profile::Response,
            Profile::Edits,
            Profile::Sources,
            Profile::Cave,
        ] {
            for tick in 0..720 {
                let mut settings = GpuTransportSettings {
                    hash_mask: 31,
                    probe_capacity: 20,
                    ..Default::default()
                };
                profile.configure(tick as f32 / 10.0, &mut settings);
                assert_eq!(
                    (
                        settings.enabled,
                        settings.reconstruction,
                        settings.voxel_sky,
                        settings.debug_irradiance
                    ),
                    (1, 1, 1, 0)
                );
                assert_eq!((settings.hash_mask, settings.probe_capacity), (31, 20));
            }
        }
        assert!(!Profile::Gameplay.sources());
        assert!(!Profile::Sources.edits());
        assert!(Profile::parse("typo").is_err());
    }

    #[test]
    fn comparison_labels_match_explicit_modes_and_restore_defaults() {
        let expected = [
            (1, 1, 1, 0),
            (1, 1, 0, 0),
            (1, 0, 1, 0),
            (1, 1, 1, 1),
            (1, 0, 1, 1),
            (0, 1, 1, 0),
            (1, 1, 1, 0),
        ];
        for (&(at, _), expected) in Profile::Compare.captures().iter().zip(expected) {
            let mut settings = GpuTransportSettings::default();
            Profile::Compare.configure(at, &mut settings);
            assert_eq!(
                (
                    settings.enabled,
                    settings.reconstruction,
                    settings.voxel_sky,
                    settings.debug_irradiance
                ),
                expected
            );
        }
        let mut settings = GpuTransportSettings::default();
        Profile::Stress.configure(71.0, &mut settings);
        assert_eq!(settings.voxel_sky, 1);
    }
}
