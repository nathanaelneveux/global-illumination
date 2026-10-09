//! Immutable world-generation stamps, independent of storage, meshing and GI.
//!
//! Templates use local integer coordinates; instances own world placement and
//! stable precedence. Explicit air carves terrain, while Keep is transparent.
//! The host overlays player edits after generation; stamps are never reapplied
//! to live chunks. This experiment currently supports translation, not rotation.
pub mod showcase;

use bevy::math::{IVec3, UVec3};
use std::{collections::HashMap, sync::Arc};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VoxelOp {
    #[default]
    Keep,
    Air,
    Solid(u8),
}

/// Half-open bounds. Upper edges belong to the next cell/region.
#[derive(Clone, Copy, Debug)]
pub struct Bounds {
    pub min: IVec3,
    pub max: IVec3,
}

impl Bounds {
    fn contains(self, p: IVec3) -> bool {
        p.cmpge(self.min).all() && p.cmplt(self.max).all()
    }

    fn intersects(self, other: Self) -> bool {
        self.min.cmplt(other.max).all() && other.min.cmplt(self.max).all()
    }

    fn buckets(self, edge: i32) -> impl Iterator<Item = IVec3> {
        let min = self.min.div_euclid(IVec3::splat(edge));
        let max = (self.max - IVec3::ONE).div_euclid(IVec3::splat(edge));
        (min.z..=max.z).flat_map(move |z| {
            (min.y..=max.y).flat_map(move |y| (min.x..=max.x).map(move |x| IVec3::new(x, y, z)))
        })
    }
}

/// Baked, bounded voxel operations. Procedural authors run once, not during
/// individual chunk/voxel lookups. Material IDs are resolved by the host palette.
pub struct StructureTemplate {
    size: UVec3,
    voxels: Box<[VoxelOp]>,
}

impl StructureTemplate {
    pub fn from_fn(size: UVec3, mut sample: impl FnMut(IVec3) -> VoxelOp) -> Arc<Self> {
        assert!(size.cmpgt(UVec3::ZERO).all() && size.cmple(UVec3::splat(i32::MAX as u32)).all());
        let volume = (size.x as usize)
            .checked_mul(size.y as usize)
            .and_then(|n| n.checked_mul(size.z as usize))
            .expect("structure volume must fit address space");
        let mut voxels = Vec::with_capacity(volume);
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    voxels.push(sample(UVec3::new(x, y, z).as_ivec3()));
                }
            }
        }
        Arc::new(Self {
            size,
            voxels: voxels.into_boxed_slice(),
        })
    }

    fn sample(&self, p: IVec3) -> VoxelOp {
        let p = p.as_uvec3();
        self.voxels[p.x as usize
            + self.size.x as usize * (p.y as usize + self.size.y as usize * p.z as usize)]
    }
}

#[derive(Clone)]
pub struct PlacedStructure {
    /// Unique within a plan. Higher IDs win overlaps, regardless of load order.
    pub id: u64,
    bounds: Bounds,
    template: Arc<StructureTemplate>,
}

impl PlacedStructure {
    pub fn new(id: u64, origin: IVec3, template: Arc<StructureTemplate>) -> Self {
        let size = template.size.as_ivec3();
        let max = IVec3::new(
            origin.x.checked_add(size.x).expect("structure x overflow"),
            origin.y.checked_add(size.y).expect("structure y overflow"),
            origin.z.checked_add(size.z).expect("structure z overflow"),
        );
        Self {
            id,
            bounds: Bounds { min: origin, max },
            template,
        }
    }

    fn sample(&self, p: IVec3) -> VoxelOp {
        if self.bounds.contains(p) {
            self.template.sample(p - self.bounds.min)
        } else {
            VoxelOp::Keep
        }
    }
}

/// A deterministic plan indexed in host-sized spatial buckets. Immutable Arc
/// ownership allows parallel chunk generation without locks or live ECS reads.
pub struct StructureLayer {
    instances: Vec<PlacedStructure>,
    buckets: HashMap<IVec3, Vec<usize>>,
    edge: i32,
}

impl StructureLayer {
    pub fn new(mut instances: Vec<PlacedStructure>, edge: i32) -> Self {
        assert!(edge > 0);
        instances.sort_unstable_by_key(|instance| instance.id);
        assert!(
            instances.windows(2).all(|p| p[0].id != p[1].id),
            "structure IDs must be unique"
        );
        let mut buckets = HashMap::<_, Vec<_>>::new();
        for (index, instance) in instances.iter().enumerate() {
            for bucket in instance.bounds.buckets(edge) {
                buckets.entry(bucket).or_default().push(index);
            }
        }
        Self {
            instances,
            buckets,
            edge,
        }
    }

    pub fn sample(&self, p: IVec3) -> VoxelOp {
        let bucket = p.div_euclid(IVec3::splat(self.edge));
        self.buckets.get(&bucket).map_or(VoxelOp::Keep, |indices| {
            first_override(indices.iter().rev().map(|&i| self.instances[i].sample(p)))
        })
    }

    /// Resolve candidates once per generation job, including the host's mesh
    /// halo. No map lookup, allocation or procedural authoring per voxel.
    pub fn for_bounds(&self, bounds: Bounds) -> StructureBatch {
        assert!(bounds.min.cmplt(bounds.max).all());
        let mut indices = Vec::new();
        for bucket in bounds.buckets(self.edge) {
            if let Some(found) = self.buckets.get(&bucket) {
                indices.extend_from_slice(found);
            }
        }
        indices.sort_unstable();
        indices.dedup();
        StructureBatch {
            instances: indices
                .into_iter()
                .filter(|&i| self.instances[i].bounds.intersects(bounds))
                .map(|i| self.instances[i].clone())
                .collect(),
        }
    }
}

pub struct StructureBatch {
    instances: Vec<PlacedStructure>,
}

impl StructureBatch {
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    pub fn sample(&self, p: IVec3) -> VoxelOp {
        first_override(
            self.instances
                .iter()
                .rev()
                .map(|instance| instance.sample(p)),
        )
    }
}

fn first_override(mut operations: impl Iterator<Item = VoxelOp>) -> VoxelOp {
    operations
        .find(|op| *op != VoxelOp::Keep)
        .unwrap_or(VoxelOp::Keep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translated_templates_and_precedence_are_independent_of_input_order() {
        let solid = StructureTemplate::from_fn(UVec3::splat(4), |_| VoxelOp::Solid(7));
        let cut = StructureTemplate::from_fn(UVec3::splat(2), |p| {
            if p.x == 0 {
                VoxelOp::Air
            } else {
                VoxelOp::Keep
            }
        });
        let layer = StructureLayer::new(
            vec![
                PlacedStructure::new(2, IVec3::splat(-1), cut),
                PlacedStructure::new(1, IVec3::splat(-2), solid.clone()),
                PlacedStructure::new(3, IVec3::splat(30), solid),
            ],
            32,
        );
        assert_eq!(layer.sample(IVec3::splat(-1)), VoxelOp::Air);
        assert_eq!(layer.sample(IVec3::ZERO), VoxelOp::Solid(7));
        assert_eq!(layer.sample(IVec3::splat(2)), VoxelOp::Keep);
        assert_eq!(layer.sample(IVec3::splat(32)), VoxelOp::Solid(7));
        // Adjacent generation jobs must agree throughout the meshing halo.
        for chunk in [IVec3::splat(-1), IVec3::ZERO, IVec3::splat(1)] {
            let bounds = Bounds {
                min: chunk * 32 - IVec3::ONE,
                max: chunk * 32 + IVec3::splat(33),
            };
            let batch = layer.for_bounds(bounds);
            for z in bounds.min.z..bounds.max.z {
                for y in bounds.min.y..bounds.max.y {
                    for x in bounds.min.x..bounds.max.x {
                        let p = IVec3::new(x, y, z);
                        assert_eq!(batch.sample(p), layer.sample(p));
                    }
                }
            }
        }
        assert!(
            layer
                .for_bounds(Bounds {
                    min: IVec3::splat(100),
                    max: IVec3::splat(132)
                })
                .is_empty()
        );
    }

    #[test]
    #[should_panic(expected = "structure IDs must be unique")]
    fn ambiguous_instance_ids_are_rejected() {
        let t = StructureTemplate::from_fn(UVec3::ONE, |_| VoxelOp::Air);
        StructureLayer::new(
            vec![
                PlacedStructure::new(1, IVec3::ZERO, t.clone()),
                PlacedStructure::new(1, IVec3::ONE, t),
            ],
            32,
        );
    }
}
