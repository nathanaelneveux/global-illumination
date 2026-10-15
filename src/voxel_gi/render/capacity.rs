//! Capacity policy for the single-buffer display directory. Reserve growth
//! headroom only when it fits; never omit occupied receivers to fit a device.
use super::GPU_PROBE_BYTES;
use bevy::prelude::Resource;
use std::fmt;

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DisplayBufferLimits {
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
}

impl Default for DisplayBufferLimits {
    fn default() -> Self {
        Self::from(&wgpu::Limits::default())
    }
}

impl From<&wgpu::Limits> for DisplayBufferLimits {
    fn from(limits: &wgpu::Limits) -> Self {
        Self {
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DisplayCapacityError {
    ArithmeticOverflow,
    Residency {
        requested: usize,
        supported: usize,
    },
    Buffer {
        name: &'static str,
        bytes: u64,
        limit: u64,
    },
    HashAddressing,
}

impl fmt::Display for DisplayCapacityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArithmeticOverflow => write!(f, "display capacity arithmetic overflow"),
            Self::Residency {
                requested,
                supported,
            } => write!(
                f,
                "display needs {requested} receivers; device supports at most {supported}"
            ),
            Self::Buffer { name, bytes, limit } => {
                write!(f, "display {name} needs {bytes} bytes; limit is {limit}")
            }
            Self::HashAddressing => write!(f, "display hash exceeds shader addressing capacity"),
        }
    }
}

impl std::error::Error for DisplayCapacityError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DisplayCapacity {
    pub probes: usize,
    pub hash: usize,
}

impl DisplayBufferLimits {
    fn byte_limit(self) -> u64 {
        self.max_buffer_size
            .min(self.max_storage_buffer_binding_size)
            .min(isize::MAX as u64)
    }

    pub fn check_bytes(self, name: &'static str, bytes: usize) -> Result<(), DisplayCapacityError> {
        let bytes = u64::try_from(bytes).map_err(|_| DisplayCapacityError::ArithmeticOverflow)?;
        let limit = self.byte_limit();
        if bytes > limit {
            return Err(DisplayCapacityError::Buffer { name, bytes, limit });
        }
        Ok(())
    }

    pub fn check_hash(self, entries: usize) -> Result<(), DisplayCapacityError> {
        if entries > u32::MAX as usize {
            return Err(DisplayCapacityError::HashAddressing);
        }
        let bytes = entries
            .checked_mul(16)
            .ok_or(DisplayCapacityError::ArithmeticOverflow)?;
        self.check_bytes("hash", bytes)
    }

    pub fn grow_hash(self, entries: usize) -> Result<usize, DisplayCapacityError> {
        let next = entries
            .checked_mul(2)
            .ok_or(DisplayCapacityError::ArithmeticOverflow)?;
        self.check_hash(next)?;
        Ok(next)
    }

    /// Maximum normal directory occupancy is 50%. Hash collision retries may
    /// require more space and are independently checked before each allocation.
    pub fn plan(self, resident_count: usize) -> Result<DisplayCapacity, DisplayCapacityError> {
        let minimum = resident_count.max(1);
        let desired = resident_count
            .checked_add((resident_count / 2).max(16_384))
            .ok_or(DisplayCapacityError::ArithmeticOverflow)?;
        let hash_slots = (self.byte_limit() / 16).min(u32::MAX as u64) as usize;
        let hash_ceiling = if hash_slots == 0 {
            0
        } else {
            1usize << (usize::BITS - 1 - hash_slots.leading_zeros())
        };
        let supported =
            (self.byte_limit() / GPU_PROBE_BYTES as u64).min((u32::MAX - 1) as u64) as usize;
        let supported = supported.min(hash_ceiling / 2);
        if minimum > supported {
            return Err(DisplayCapacityError::Residency {
                requested: minimum,
                supported,
            });
        }
        let probes = desired.min(supported);
        let hash = probes
            .checked_mul(2)
            .and_then(usize::checked_next_power_of_two)
            .ok_or(DisplayCapacityError::ArithmeticOverflow)?;
        let bytes = probes
            .checked_mul(GPU_PROBE_BYTES)
            .ok_or(DisplayCapacityError::ArithmeticOverflow)?;
        self.check_bytes("probes", bytes)?;
        self.check_hash(hash)?;
        Ok(DisplayCapacity { probes, hash })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headroom_is_clamped_without_dropping_residents_at_either_device_limit() {
        for limits in [
            DisplayBufferLimits {
                max_buffer_size: 2080,
                max_storage_buffer_binding_size: 4096,
            },
            DisplayBufferLimits {
                max_buffer_size: 4096,
                max_storage_buffer_binding_size: 2080,
            },
        ] {
            assert_eq!(
                limits.plan(10).unwrap(),
                DisplayCapacity {
                    probes: 10,
                    hash: 32
                }
            );
            assert_eq!(limits.plan(1).unwrap().probes, 10);
            assert_eq!(
                limits.plan(11),
                Err(DisplayCapacityError::Residency {
                    requested: 11,
                    supported: 10
                })
            );
            assert!(limits.check_bytes("probes", 2080).is_ok());
            assert!(limits.check_bytes("probes", 2081).is_err());
            assert_eq!(limits.grow_hash(64).unwrap(), 128);
            assert!(limits.grow_hash(128).is_err());
        }
    }

    #[test]
    fn default_headroom_and_large_residency_fit_the_binding_not_just_the_buffer() {
        let limits = DisplayBufferLimits {
            max_buffer_size: 256 << 20,
            max_storage_buffer_binding_size: 128 << 20,
        };
        assert_eq!(limits.plan(100).unwrap().probes, 16_484);
        let plan = limits.plan(500_000).unwrap();
        assert_eq!(plan.probes, (128 << 20) / GPU_PROBE_BYTES);
        assert!(plan.hash >= 2 * plan.probes);
        assert!(limits.plan(plan.probes + 1).is_err());
    }

    #[test]
    fn arithmetic_shader_and_tiny_device_boundaries_are_rejected_without_allocation() {
        let limits = DisplayBufferLimits {
            max_buffer_size: u64::MAX,
            max_storage_buffer_binding_size: u64::MAX,
        };
        assert_eq!(
            limits.plan(usize::MAX),
            Err(DisplayCapacityError::ArithmeticOverflow)
        );
        assert_eq!(
            limits.grow_hash(usize::MAX),
            Err(DisplayCapacityError::ArithmeticOverflow)
        );
        assert_eq!(
            limits.check_hash(u32::MAX as usize + 1),
            Err(DisplayCapacityError::HashAddressing)
        );
        let tiny = DisplayBufferLimits {
            max_buffer_size: 207,
            max_storage_buffer_binding_size: 1024,
        };
        assert!(tiny.plan(0).is_err());
        assert!(tiny.plan(1).is_err());
    }
}
