//! Edit/streaming change plans for retention. Each change is binned to nearby
//! receiver pages with a flux bound; distance and flux control retained
//! statistical weight, never visibility or whether a receiver must refresh.
use bevy::math::I64Vec3;
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;

use super::world::{AIR, PAGE_EDGE, REGION_EDGE, TransportScene};

/// Exact edits of unknown flux (or under the curve rule) requeue receivers
/// within this radius urgently; edits of known flux wake only the faces their
/// predicted change would cut (publication::retention_binds). A scheduling
/// hint: counts are untouched. Probes whose own faces or material change cap
/// their CPU-side sample estimate to `NEAR_HISTORY`.
pub(super) const RELIGHT_RADIUS: i32 = 48;
/// Player edits of known flux wake (make urgent on the GPU) the faces within
/// this radius whose history the change binds; unknown flux, or the curve
/// rule, wakes every face within RELIGHT_RADIUS.
pub(super) const MEASURED_RELIGHT_RADIUS: i32 = 64;
pub(super) const NEAR_HISTORY: u32 = 16;
pub(super) const UNTRACKED_HISTORY: u32 = 64;

/// What a change can do to existing receivers' light, for measured retention.
/// `flux` is a luminance flux bound in normalized transport units: a receiver
/// at distance d sees a change of at most flux / (pi d^2). Unknown falls back
/// to the distance curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum ChangeKind {
    Unknown,
    /// Voxel content changed. The scene worker adds emission; with `measure`,
    /// the GPU adds the light leaving the surfaces around the edit, from its
    /// own statistics (`measure_edits`), before retention applies the change.
    Edit {
        voxels: u32,
        measure: bool,
    },
    /// A region loaded. Only terrain above a receiver can replace sky it saw
    /// (rays into unloaded space are excluded, not black), so the flux is per
    /// unit of height above the receiver, up to the region's top solid voxel.
    Load {
        top: i32,
    },
    /// A region unloaded: its samples now become unknown, not contradicted.
    Unload,
    /// Shared-corner sunlight changed by this luminance flux.
    Light,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ChangeWeight {
    pub kind: ChangeKind,
    pub flux: f32,
}

impl ChangeWeight {
    pub const UNKNOWN: Self = Self {
        kind: ChangeKind::Unknown,
        flux: 0.0,
    };

    /// Predicted luminance change at a receiver, or None to fall back. The
    /// GPU's `apply_events` evaluates this; the CPU statement serves tests.
    #[cfg(test)]
    pub fn change_at(self, distance_squared: i64, receiver_y: i32) -> Option<f32> {
        let area = match self.kind {
            ChangeKind::Unknown => return None,
            ChangeKind::Unload => return Some(0.0),
            // An air-only load (no top voxel) replaces no sky.
            ChangeKind::Load { top } => {
                (i64::from(top) + 1 - i64::from(receiver_y)).clamp(0, i64::from(REGION_EDGE)) as f32
            }
            ChangeKind::Edit { .. } | ChangeKind::Light => 1.0,
        };
        Some(self.flux * area / (std::f32::consts::PI * (distance_squared.max(1) as f32)))
    }
}

/// One worker-side comparison supplies scene publication and spatial history
/// retention. A voxel's optical identity is scene data, regardless of whether
/// that material reflects, emits, or occludes energy.
pub(super) struct SceneChanges {
    pub changed: bool,
    pub transport: Option<GeometryChange>,
}

fn emission_luminance(
    materials: Option<&super::world::VoxelMaterialTransportTable>,
    material: u16,
) -> f32 {
    if material == AIR {
        return 0.0;
    }
    materials.map_or(0.0, |table| {
        let e = table.get(material as u8).emission;
        0.2126 * e.x + 0.7152 * e.y + 0.0722 * e.z
    })
}

impl SceneChanges {
    #[cfg(test)]
    pub fn between(previous: &TransportScene, next: &TransportScene) -> Self {
        Self::with_materials(previous, next, RELIGHT_RADIUS, None)
    }

    /// Bin changes within `radius` of each receiver page and weight each for
    /// measured retention: emission deltas need the material table.
    pub fn with_materials(
        previous: &TransportScene,
        next: &TransportScene,
        radius: i32,
        materials: Option<&super::world::VoxelMaterialTransportTable>,
    ) -> Self {
        let changed = next.changed_regions(previous);
        if changed.is_empty() {
            return Self {
                changed: false,
                transport: None,
            };
        }
        let mut occupied_min = IVec3::MAX;
        let mut occupied_max = IVec3::MIN;
        for (&coordinate, region) in &previous.regions {
            if region.any_solid {
                occupied_min = occupied_min.min(coordinate);
                occupied_max = occupied_max.max(coordinate);
            }
        }
        let mut result = Self {
            changed: false,
            transport: None,
        };
        let mut bounds = Vec::new();
        let mut weights = HashMap::new();
        for coordinate in &changed {
            let before = previous.regions.get(coordinate);
            let after = next.regions.get(coordinate);
            result.changed = true;
            let origin = *coordinate * REGION_EDGE as i32;
            let entire = Bounds {
                min: origin,
                max: origin + IVec3::splat(REGION_EDGE as i32 - 1),
            };
            let mut tight = Bounds {
                min: IVec3::MAX,
                max: IVec3::MIN,
            };
            let mut changed_voxels = 0u32;
            let mut emission_flux = 0.0f32;
            let mut record = |a: u16, b: u16, extent: Bounds| {
                if a == b {
                    return;
                }
                tight.min = tight.min.min(extent.min);
                tight.max = tight.max.max(extent.max);
                let volume = (extent.max - extent.min + IVec3::ONE)
                    .element_product()
                    .max(1) as u32;
                changed_voxels = changed_voxels.saturating_add(volume);
                // Six faces bound the emitting area a voxel can expose.
                emission_flux +=
                    (emission_luminance(materials, a) - emission_luminance(materials, b)).abs()
                        * 6.0
                        * volume as f32;
            };
            let uniform_before = before.map_or(Some(AIR), |r| r.uniform_material());
            let uniform_after = after.map_or(Some(AIR), |r| r.uniform_material());
            if let (Some(a), Some(b)) = (uniform_before, uniform_after) {
                record(a, b, entire);
            } else {
                for z in 0..REGION_EDGE {
                    for y in 0..REGION_EDGE {
                        for x in 0..REGION_EDGE {
                            let local = UVec3::new(x, y, z);
                            record(
                                before.map_or(AIR, |r| r.material(local)),
                                after.map_or(AIR, |r| r.material(local)),
                                Bounds::point(origin + local.as_ivec3()),
                            );
                        }
                    }
                }
            }
            // Residency changes knowledge of air as well as solids. Unknown
            // holes within the previous occupied bounds affect existing paths;
            // outside air-only bookkeeping does not change their integrand.
            let residency = before.is_none() || after.is_none();
            let transport = before.is_some_and(|r| r.any_solid)
                || after.is_some_and(|r| r.any_solid)
                || (coordinate.cmpge(occupied_min).all() && coordinate.cmple(occupied_max).all());
            if transport && (residency || tight.min.cmple(tight.max).all()) {
                let change = if residency { entire } else { tight };
                let kind = if before.is_none() {
                    // Top solid voxel of the loaded region (air-only loads
                    // outside occupied bounds were filtered above).
                    let top = after.map_or(i32::MIN, |region| {
                        (0..REGION_EDGE)
                            .rev()
                            .find(|&y| {
                                (0..REGION_EDGE).any(|z| {
                                    (0..REGION_EDGE)
                                        .any(|x| region.material(UVec3::new(x, y, z)) != AIR)
                                })
                            })
                            .map_or(i32::MIN, |y| origin.y + y as i32)
                    });
                    ChangeKind::Load { top }
                } else if after.is_none() {
                    ChangeKind::Unload
                } else {
                    ChangeKind::Edit {
                        voxels: changed_voxels,
                        measure: false,
                    }
                };
                let flux = if matches!(kind, ChangeKind::Edit { .. }) {
                    emission_flux
                } else {
                    0.0
                };
                weights.insert(change, ChangeWeight { kind, flux });
                bounds.push(change);
            }
        }
        if !bounds.is_empty() {
            let mut change = GeometryChange::from_bounds(previous, &bounds, radius);
            change.weights = weights;
            result.transport = Some(change);
        }
        result
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Bounds {
    pub min: IVec3,
    pub max: IVec3,
}

impl Bounds {
    pub fn distance_squared(self, other: Self) -> i64 {
        let separation = (self.min.as_i64vec3() - other.max.as_i64vec3())
            .max(other.min.as_i64vec3() - self.max.as_i64vec3())
            .max(I64Vec3::ZERO)
            // Beyond the transport horizon no supported retention curve needs
            // an exact distance. Bound arithmetic for disconnected worlds.
            .min(I64Vec3::splat(4096));
        separation.length_squared()
    }

    pub fn point(voxel: IVec3) -> Self {
        Self {
            min: voxel,
            max: voxel,
        }
    }

    pub fn page(page: IVec3) -> Self {
        let min = page * PAGE_EDGE as i32;
        Self {
            min,
            max: min + IVec3::splat(PAGE_EDGE as i32 - 1),
        }
    }
}

/// Worker-built bounds and their minimum separation from the receiver page.
/// Carrying the distance avoids rescanning bounds when publication folds an
/// old event into the bounded retention summary on the main thread.
pub(super) struct PageChanges {
    pub bounds: Vec<Bounds>,
    pub minimum_distance_squared: i64,
}

impl PageChanges {
    #[cfg(test)]
    pub fn new(page: IVec3, bounds: Vec<Bounds>) -> Self {
        let minimum_distance_squared = bounds
            .iter()
            .map(|b| b.distance_squared(Bounds::page(page)))
            .min()
            .unwrap_or(i64::MAX);
        Self {
            bounds,
            minimum_distance_squared,
        }
    }
}

#[derive(Default)]
pub(super) struct GeometryChange {
    /// Existing receiver pages only. New receivers have no history to retain.
    /// Each bin contains just changes within the configured influence radius.
    pub nearby: HashMap<IVec3, PageChanges>,
    /// Measured-retention weight of each change bounds. Missing entries are
    /// unknown and fall back to the distance curve.
    pub weights: HashMap<Bounds, ChangeWeight>,
}

impl GeometryChange {
    pub fn from_bounds(previous: &TransportScene, changes: &[Bounds], radius: i32) -> Self {
        if changes.is_empty() || previous.regions.is_empty() {
            return Self::default();
        }
        // Choose the smaller broad phase. Local edits should not enumerate the
        // resident world, but a huge brush/disconnected world must not enumerate
        // its empty bounding box either. Eight region lookups establish page
        // occupancy; use a conservative cost allowance for those hash probes.
        // distance_squared deliberately saturates at 4096 per axis, so only
        // use geometric range pruning below that exact-distance horizon.
        let local_cells = changes.iter().fold(0_u64, |total, &change| {
            let (min, max) = expanded_page_range(change, radius);
            let size = max.as_i64vec3() - min.as_i64vec3() + I64Vec3::ONE;
            total.saturating_add(
                size.to_array()
                    .into_iter()
                    .fold(1_u64, |a, b| a.saturating_mul(b.max(0) as u64)),
            )
        });
        if (0..4096).contains(&radius)
            && local_cells.saturating_mul(8)
                < (previous.regions.len() as u64).saturating_mul(changes.len() as u64)
        {
            Self::from_local_bounds(previous, changes, radius)
        } else {
            Self::from_resident_bounds(previous, changes, radius)
        }
    }

    fn from_local_bounds(previous: &TransportScene, changes: &[Bounds], radius: i32) -> Self {
        let mut plan = Self::default();
        let mut occupancy = HashMap::new();
        let radius_squared = i64::from(radius).pow(2);
        for &change in changes {
            let (min, max) = expanded_page_range(change, radius);
            for z in min.z..=max.z {
                for y in min.y..=max.y {
                    for x in min.x..=max.x {
                        let page = IVec3::new(x, y, z);
                        let distance = change.distance_squared(Bounds::page(page));
                        if distance > radius_squared {
                            continue;
                        }
                        // Overlapping changes share this negative/positive cache;
                        // repeated empty-space lookups can dominate sunlight deltas.
                        let occupied = *occupancy.entry(page).or_insert_with(|| {
                            (0..8).any(|i| {
                                let offset = IVec3::new(i & 1, (i >> 1) & 1, (i >> 2) & 1);
                                previous
                                    .regions
                                    .get(&(page * 2 + offset))
                                    .is_some_and(|r| r.any_solid)
                            })
                        });
                        if occupied {
                            let entry = plan.nearby.entry(page).or_insert_with(|| PageChanges {
                                bounds: Vec::new(),
                                minimum_distance_squared: distance,
                            });
                            entry.bounds.push(change);
                            entry.minimum_distance_squared =
                                entry.minimum_distance_squared.min(distance);
                        }
                    }
                }
            }
        }
        plan
    }

    fn from_resident_bounds(previous: &TransportScene, changes: &[Bounds], radius: i32) -> Self {
        let pages: HashSet<_> = previous
            .regions
            .iter()
            .filter(|(_, r)| r.any_solid)
            .map(|(r, _)| r.div_euclid(IVec3::splat(2)))
            .collect();
        let mut plan = Self::default();
        let radius_squared = i64::from(radius).pow(2);
        for page in pages {
            let bounds = Bounds::page(page);
            let mut minimum_distance_squared = i64::MAX;
            let nearby: Vec<_> = changes
                .iter()
                .copied()
                .filter(|change| {
                    let distance = change.distance_squared(bounds);
                    if distance > radius_squared {
                        return false;
                    }
                    minimum_distance_squared = minimum_distance_squared.min(distance);
                    true
                })
                .collect();
            if !nearby.is_empty() {
                plan.nearby.insert(
                    page,
                    PageChanges {
                        bounds: nearby,
                        minimum_distance_squared,
                    },
                );
            }
        }
        plan
    }
}

/// Inclusive page span in the supported i32 voxel domain. Widen before padding
/// to avoid overflowing near the coordinate limits; exact sphere tests follow.
fn expanded_page_range(change: Bounds, radius: i32) -> (IVec3, IVec3) {
    let edge = I64Vec3::splat(PAGE_EDGE as i64);
    let padding = I64Vec3::splat(i64::from(radius).abs());
    let lower = I64Vec3::splat(i64::from(i32::MIN)).div_euclid(edge);
    let upper = I64Vec3::splat(i64::from(i32::MAX)).div_euclid(edge);
    let min = (change.min.as_i64vec3() - padding)
        .div_euclid(edge)
        .clamp(lower, upper);
    let max = (change.max.as_i64vec3() + padding)
        .div_euclid(edge)
        .clamp(lower, upper);
    (min.as_ivec3(), max.as_ivec3())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel_gi::{naadf::volume_len, world::PAGE_SIZE};

    fn assert_same_bins(a: &GeometryChange, b: &GeometryChange) {
        assert_eq!(a.nearby.len(), b.nearby.len());
        for (page, expected) in &a.nearby {
            let actual = &b.nearby[page];
            assert_eq!(actual.bounds, expected.bounds, "page {page}");
            assert_eq!(
                actual.minimum_distance_squared, expected.minimum_distance_squared,
                "page {page}"
            );
        }
    }

    #[test]
    fn local_binning_matches_resident_scan_with_gaps_edges_and_overlaps() {
        let mut scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            (-8..8)
                .flat_map(|x| {
                    (-2..2).map(move |z| {
                        (
                            IVec3::new(x, -1, z),
                            vec![if (x + z) % 3 == 0 { AIR } else { 1 }; volume_len(PAGE_SIZE)],
                        )
                    })
                })
                .collect(),
            &[],
        );
        // A partially resident page must still be included if one other region
        // is occupied. Empty/missing pages must never gain retention entries.
        scene.regions.remove(&IVec3::new(0, -2, 0));
        let changes = [
            Bounds::point(IVec3::new(-32, -1, 0)),
            Bounds::point(IVec3::new(0, 0, 0)),
            Bounds::page(IVec3::new(1, 0, 1)),
            Bounds::point(IVec3::new(10000, 0, 0)),
        ];
        for radius in [0, 1, 16, 48, super::super::retention::INFLUENCE_RADIUS] {
            for end in 1..=changes.len() {
                let full = GeometryChange::from_resident_bounds(&scene, &changes[..end], radius);
                assert_same_bins(
                    &full,
                    &GeometryChange::from_local_bounds(&scene, &changes[..end], radius),
                );
                assert_same_bins(
                    &full,
                    &GeometryChange::from_bounds(&scene, &changes[..end], radius),
                );
            }
        }
        for radius in [4096, i32::MAX] {
            assert_same_bins(
                &GeometryChange::from_resident_bounds(&scene, &changes, radius),
                &GeometryChange::from_bounds(&scene, &changes, radius),
            );
        }
        let huge = [Bounds {
            min: IVec3::splat(-1_000_000),
            max: IVec3::splat(1_000_000),
        }];
        assert_same_bins(
            &GeometryChange::from_resident_bounds(&scene, &huge, 192),
            &GeometryChange::from_bounds(&scene, &huge, 192),
        );
        for point in [IVec3::MIN, IVec3::MAX] {
            let (min, max) = expanded_page_range(Bounds::point(point), 100);
            assert!(min.cmple(max).all());
            assert!(min.cmpge(IVec3::splat(i32::MIN / PAGE_EDGE as i32)).all());
            assert!(max.cmple(IVec3::splat(i32::MAX / PAGE_EDGE as i32)).all());
        }
    }

}
