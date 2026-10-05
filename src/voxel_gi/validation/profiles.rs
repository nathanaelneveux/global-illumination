//! Workload identity and presentation are explicit; toggle stress is never a
//! proxy for the cost or appearance of the normal gameplay mode.
use super::super::render::GpuTransportSettings;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Profile {
    #[default]
    Gameplay,
    Streaming,
    Sources,
    Compare,
    Stress,
}

impl Profile {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "gameplay" => Ok(Self::Gameplay),
            "streaming" => Ok(Self::Streaming),
            "sources" => Ok(Self::Sources),
            "compare" => Ok(Self::Compare),
            "stress" => Ok(Self::Stress),
            _ => Err(format!(
                "unknown validation profile {value:?}; choose gameplay, streaming, sources, compare, or stress"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Gameplay => "gameplay",
            Self::Streaming => "streaming",
            Self::Sources => "sources",
            Self::Compare => "compare",
            Self::Stress => "stress",
        }
    }

    pub fn duration(self) -> f32 {
        match self {
            Self::Compare => 29.0,
            Self::Streaming => 35.0,
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
        if self == Self::Streaming {
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
        if self == Self::Streaming {
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
        for profile in [Profile::Gameplay, Profile::Streaming, Profile::Sources] {
            for tick in 0..720 {
                let mut settings = GpuTransportSettings {
                    hash_mask: 31,
                    probe_count: 20,
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
                assert_eq!((settings.hash_mask, settings.probe_count), (31, 20));
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
