// Spatial keys are trusted engine data. Match the field's fast Bevy maps rather
// than paying the standard library's adversarial-string hashing cost per cell.
#[cfg(test)]
use bevy::platform::collections::HashMap;
use bevy::platform::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use bevy::prelude::*;

use super::color::FiniteColor;
use super::naadf::{Aadf, build_aadf, linear_index, try_grow, volume_len};
mod delta;
mod regions;
pub(crate) use delta::SceneStamp;
pub use regions::RegionMap;

/// Transport-only sentinel outside the complete `u8` material index range.
pub const AIR: u16 = u8::MAX as u16 + 1;
pub const BRICK_EDGE: u32 = 4;
pub const REGION_EDGE: u32 = 16;
pub const PAGE_EDGE: u32 = 32;
pub const BRICKS_PER_AXIS: u32 = REGION_EDGE / BRICK_EDGE;
pub const REGION_SIZE: UVec3 = UVec3::splat(REGION_EDGE);
pub const BRICK_GRID_SIZE: UVec3 = UVec3::splat(BRICKS_PER_AXIS);
pub const PAGE_SIZE: UVec3 = UVec3::splat(PAGE_EDGE);

/// Distance in voxels from `point` to page `page`'s closed box (zero inside).
pub(crate) fn page_distance(page: IVec3, point: Vec3) -> f32 {
    let minimum = (page * PAGE_EDGE as i32).as_vec3();
    let maximum = minimum + Vec3::splat(PAGE_EDGE as f32);
    point.clamp(minimum, maximum).distance(point)
}

/// The inclusive page range holding every page within `radius` of `point`
/// (`page_distance <= radius`).
pub(crate) fn pages_within(point: Vec3, radius: f32) -> (IVec3, IVec3) {
    pages_within_box(point, point, radius)
}

/// The inclusive page range holding every page whose closed box comes within
/// `radius` of the box from `low` to `high`. A page whose upper face lies
/// exactly at that distance starts a whole page below it.
pub(crate) fn pages_within_box(low: Vec3, high: Vec3, radius: f32) -> (IVec3, IVec3) {
    let edge = PAGE_EDGE as f32;
    let minimum = ((low - radius) / edge).ceil().as_ivec3() - IVec3::ONE;
    (minimum, ((high + radius) / edge).floor().as_ivec3())
}

pub const FACE_OFFSETS: [IVec3; 6] = [
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Y,
    IVec3::NEG_Y,
    IVec3::Z,
    IVec3::NEG_Z,
];

/// A transport material's optics: opaque diffuse RGB response and emission.
/// No transmission, heat capacity or dose conversion is implied; those belong
/// to the host simulation. Write one with `..default()` for the fields you
/// leave out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialTransport {
    /// Fraction of incident energy reflected, clamped to [0, 1] by
    /// `VoxelGiMaterials`.
    pub diffuse: Vec3,
    /// Emitted energy in normalized solver units, not watts or temperature.
    pub emission: Vec3,
}

impl Default for MaterialTransport {
    fn default() -> Self {
        Self {
            diffuse: Vec3::splat(0.45),
            emission: Vec3::ZERO,
        }
    }
}

/// The optics of GI's 256 transport materials (see [`GiVoxel`]).
///
/// [`insert`](Self::insert) gives optics a material, sharing one between
/// identical optics. Hosts whose block IDs already fit `0..=255` can instead
/// [`set`](Self::set) each block's optics at its own ID (the BVW connector
/// does). Unassigned materials keep the default optics. Change this resource
/// at runtime to relight the world.
///
/// [`GiVoxel`]: crate::GiVoxel
#[derive(Resource, Clone, Debug)]
pub struct VoxelGiMaterials {
    entries: [MaterialTransport; 256],
    /// Materials given optics by `insert` or `set`, one bit each.
    assigned: [u64; 4],
}

/// Equal when every material has the same optics, assigned or not: that is
/// all transport sees.
impl PartialEq for VoxelGiMaterials {
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries
    }
}

impl Default for VoxelGiMaterials {
    fn default() -> Self {
        Self {
            entries: [MaterialTransport::default(); 256],
            assigned: [0; 4],
        }
    }
}

impl VoxelGiMaterials {
    /// A material with these optics: an assigned one with identical optics,
    /// else the lowest unassigned material.
    pub fn insert(
        &mut self,
        optics: MaterialTransport,
    ) -> Result<crate::GiVoxel, crate::MaterialsFull> {
        let optics = Self::bounded(optics);
        let existing =
            (0..=255u8).find(|&m| self.is_assigned(m) && self.entries[m as usize] == optics);
        let material = match existing {
            Some(material) => material,
            None => {
                let free = (0..=255u8).find(|&m| !self.is_assigned(m));
                let material = free.ok_or(crate::MaterialsFull)?;
                self.set(material, optics);
                material
            }
        };
        Ok(crate::GiVoxel::solid(material))
    }

    /// The optics of material `material`.
    pub fn get(&self, material: u8) -> MaterialTransport {
        self.entries[material as usize]
    }

    /// Sets material `material`'s optics. Diffuse reflectance is clamped to
    /// [0, 1] and magnitudes are bounded, so bad input can't create energy.
    pub fn set(&mut self, material: u8, transport: MaterialTransport) {
        self.entries[material as usize] = Self::bounded(transport);
        self.assigned[material as usize / 64] |= 1 << (material % 64);
    }

    /// Whether `material` was given optics by `insert` or `set`.
    pub fn is_assigned(&self, material: u8) -> bool {
        self.assigned[material as usize / 64] & (1 << (material % 64)) != 0
    }

    /// Materials given optics so far.
    pub fn len(&self) -> usize {
        self.assigned
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// No material has been given optics.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn bounded(transport: MaterialTransport) -> MaterialTransport {
        // A diffuse reflector cannot create energy. Bound source magnitudes
        // as well so malformed host input cannot poison GPU moments.
        MaterialTransport {
            diffuse: transport.diffuse.bounded(1.0),
            emission: transport.emission.bounded(1.0e12),
        }
    }
}

#[derive(Clone)]
pub struct NaadfRegion {
    pub coordinate: IVec3,
    /// log2 of the host page's stride (`PageSample`): coarse cells are
    /// solid blocks, and the view tier keys their faces by block.
    pub stride_log2: u8,
    storage: RegionStorage,
    pub any_solid: bool,
    /// Geometry-owned material presence, independent of the live optical table.
    /// Source discovery can skip regions with no emitting material in O(4).
    material_mask: [u64; 4],
}

#[derive(Clone)]
enum RegionStorage {
    Empty,
    Uniform(u16),
    /// Materials only. Empty-space bounds are packed into the GPU cells and
    /// computed there from these materials (`brick_voxel_bounds`,
    /// `brick_bounds`); the CPU keeps no per-voxel bounds.
    Dense {
        materials: Box<[u16]>,
        brick_occupied: Box<[bool]>,
    },
    /// Every 4³ brick is one material (coarse LOD pages at stride 4 and up,
    /// or blocky full-resolution content): 128 bytes instead of 8 KB. The
    /// GPU already packs uniform bricks without a payload.
    Bricks(Box<[u16; BRICKS_PER_REGION]>),
}

const BRICKS_PER_REGION: usize = (BRICKS_PER_AXIS * BRICKS_PER_AXIS * BRICKS_PER_AXIS) as usize;

/// The brick (x fastest) holding the voxel at a region's linear `index`.
fn brick_of_index(index: usize) -> usize {
    let index = index as u32;
    let local = UVec3::new(
        index % REGION_EDGE,
        (index / REGION_EDGE) % REGION_EDGE,
        index / (REGION_EDGE * REGION_EDGE),
    );
    linear_index(local / BRICK_EDGE, BRICK_GRID_SIZE)
}

impl NaadfRegion {
    fn matches_materials(&self, incoming: &[u16]) -> bool {
        match &self.storage {
            RegionStorage::Empty => incoming.iter().all(|value| *value == AIR),
            RegionStorage::Uniform(material) => incoming.iter().all(|value| value == material),
            RegionStorage::Dense { materials, .. } => materials.as_ref() == incoming,
            RegionStorage::Bricks(bricks) => incoming
                .iter()
                .enumerate()
                .all(|(index, value)| *value == bricks[brick_of_index(index)]),
        }
    }
    pub(crate) fn uniform_material(&self) -> Option<u16> {
        match self.storage {
            RegionStorage::Empty => Some(AIR),
            RegionStorage::Uniform(material) => Some(material),
            RegionStorage::Dense { .. } | RegionStorage::Bricks(_) => None,
        }
    }

    pub(crate) fn contains_any_material(&self, mask: &[u64; 4]) -> bool {
        self.material_mask.iter().zip(mask).any(|(a, b)| a & b != 0)
    }

    pub fn build(coordinate: IVec3, materials: Vec<u16>) -> Self {
        assert_eq!(materials.len(), volume_len(REGION_SIZE));

        if materials.iter().all(|material| *material == AIR) {
            return Self {
                coordinate,
                stride_log2: 0,
                storage: RegionStorage::Empty,
                any_solid: false,
                material_mask: [0; 4],
            };
        }
        if materials.iter().all(|material| *material == materials[0]) {
            let mut material_mask = [0; 4];
            material_mask[materials[0] as usize / 64] |= 1 << (materials[0] % 64);
            return Self {
                coordinate,
                stride_log2: 0,
                storage: RegionStorage::Uniform(materials[0]),
                any_solid: true,
                material_mask,
            };
        }

        // All bricks uniform: keep one material per brick.
        let mut bricks = [AIR; BRICKS_PER_REGION];
        let mut brick_uniform = [true; BRICKS_PER_REGION];
        let mut seen = [false; BRICKS_PER_REGION];
        for (index, &material) in materials.iter().enumerate() {
            let brick = brick_of_index(index);
            if !seen[brick] {
                seen[brick] = true;
                bricks[brick] = material;
            } else if bricks[brick] != material {
                brick_uniform[brick] = false;
            }
        }
        if brick_uniform.iter().all(|uniform| *uniform) {
            let mut material_mask = [0; 4];
            for &material in bricks.iter().filter(|material| **material != AIR) {
                material_mask[material as usize / 64] |= 1 << (material % 64);
            }
            return Self {
                coordinate,
                stride_log2: 0,
                any_solid: bricks.iter().any(|material| *material != AIR),
                storage: RegionStorage::Bricks(Box::new(bricks)),
                material_mask,
            };
        }

        let mut brick_occupied = vec![false; volume_len(BRICK_GRID_SIZE)];
        let mut material_mask = [0; 4];
        for (index, &material) in materials.iter().enumerate() {
            if material == AIR {
                continue;
            }
            material_mask[material as usize / 64] |= 1 << (material % 64);
            let local = UVec3::new(
                index as u32 % REGION_EDGE,
                (index as u32 / REGION_EDGE) % REGION_EDGE,
                index as u32 / (REGION_EDGE * REGION_EDGE),
            );
            brick_occupied[linear_index(local / BRICK_EDGE, BRICK_GRID_SIZE)] = true;
        }
        let any_solid = brick_occupied.iter().any(|occupied| *occupied);
        Self {
            coordinate,
            stride_log2: 0,
            storage: RegionStorage::Dense {
                materials: materials.into_boxed_slice(),
                brick_occupied: brick_occupied.into_boxed_slice(),
            },
            any_solid,
            material_mask,
        }
    }

    fn rebuild(&self, materials: Vec<u16>) -> Self {
        Self::build(self.coordinate, materials)
    }

    /// A region of one material (`AIR` for empty), without materializing its
    /// voxels: what `build` returns for such contents.
    fn uniform(coordinate: IVec3, material: u16) -> Self {
        let mut material_mask = [0; 4];
        if material != AIR {
            material_mask[material as usize / 64] |= 1 << (material % 64);
        }
        Self {
            coordinate,
            stride_log2: 0,
            storage: if material == AIR {
                RegionStorage::Empty
            } else {
                RegionStorage::Uniform(material)
            },
            any_solid: material != AIR,
            material_mask,
        }
    }

    fn with_stride(mut self, stride_log2: u8) -> Self {
        self.stride_log2 = stride_log2;
        self
    }

    /// Empty-space bounds of the 64 voxels in `brick` (within the brick,
    /// voxel order x fastest); zero for solid voxels and non-dense regions.
    pub(crate) fn brick_voxel_bounds(&self, brick: UVec3) -> Vec<Aadf> {
        let brick_size = UVec3::splat(BRICK_EDGE);
        if let RegionStorage::Bricks(bricks) = &self.storage {
            let solid = bricks[linear_index(brick, BRICK_GRID_SIZE)] != AIR;
            return build_aadf(
                brick_size,
                &[solid; (BRICK_EDGE * BRICK_EDGE * BRICK_EDGE) as usize],
            );
        }
        let RegionStorage::Dense { materials, .. } = &self.storage else {
            return vec![Aadf::default(); volume_len(brick_size)];
        };
        let mut blocked = [false; (BRICK_EDGE * BRICK_EDGE * BRICK_EDGE) as usize];
        for (index, cell) in blocked.iter_mut().enumerate() {
            let local = UVec3::new(index as u32 & 3, (index as u32 >> 2) & 3, index as u32 >> 4);
            *cell = materials[linear_index(brick * BRICK_EDGE + local, REGION_SIZE)] != AIR;
        }
        build_aadf(brick_size, &blocked)
    }

    /// Empty-space bounds of the region's 4^3 bricks (brick order x fastest).
    pub(crate) fn brick_bounds(&self) -> Vec<Aadf> {
        match &self.storage {
            RegionStorage::Dense { brick_occupied, .. } => {
                build_aadf(BRICK_GRID_SIZE, brick_occupied)
            }
            RegionStorage::Bricks(bricks) => {
                let occupied: Vec<bool> = bricks.iter().map(|material| *material != AIR).collect();
                build_aadf(BRICK_GRID_SIZE, &occupied)
            }
            _ => vec![Aadf::default(); volume_len(BRICK_GRID_SIZE)],
        }
    }

    pub fn material(&self, local: UVec3) -> u16 {
        match &self.storage {
            RegionStorage::Empty => AIR,
            RegionStorage::Uniform(material) => *material,
            RegionStorage::Dense { materials, .. } => materials[linear_index(local, REGION_SIZE)],
            RegionStorage::Bricks(bricks) => {
                bricks[linear_index(local / BRICK_EDGE, BRICK_GRID_SIZE)]
            }
        }
    }

    #[cfg(test)]
    pub fn voxel_aadf(&self, local: UVec3) -> Aadf {
        let brick = local / BRICK_EDGE;
        self.brick_voxel_bounds(brick)[linear_index(local % BRICK_EDGE, UVec3::splat(BRICK_EDGE))]
    }

    #[cfg(test)]
    pub fn brick_occupied(&self, brick: UVec3) -> bool {
        match &self.storage {
            RegionStorage::Empty => false,
            RegionStorage::Uniform(_) => true,
            RegionStorage::Dense { brick_occupied, .. } => {
                brick_occupied[linear_index(brick, BRICK_GRID_SIZE)]
            }
            RegionStorage::Bricks(bricks) => bricks[linear_index(brick, BRICK_GRID_SIZE)] != AIR,
        }
    }

    #[cfg(test)]
    pub fn brick_aadf(&self, brick: UVec3) -> Aadf {
        self.brick_bounds()[linear_index(brick, BRICK_GRID_SIZE)]
    }
}

/// The top-level empty-space field is dense over the loaded regions' bounding
/// box (6 bytes a region cell, built on the scene worker; the GPU keeps only
/// per-region extents). 262,144 cells (64^3) switched it off for a 48-chunk
/// world (192 x 192 x ~20 regions), leaving rays to cross empty space one
/// region per step. NAADF keeps chunk-level bounds over its whole world.
const TOP_LEVEL_MAX_CELLS: u64 = 4 << 20;

#[derive(Clone, Default)]
pub struct TopLevelAadf {
    pub minimum: IVec3,
    pub maximum: IVec3,
    pub size: UVec3,
    // Immutable scene snapshots can share this independently of changed voxel
    // data: only region residency and empty/nonempty topology define the field.
    fields: Arc<[Aadf]>,
    // Which cells are barriers (solid or not loaded), kept with the field so
    // an update reads only the regions that changed.
    blocked: Arc<[bool]>,
    // The field this one was updated from in place (same bounds), by
    // address, and the region cells whose bounds differ from it. A consumer
    // holding that field can patch just these cells; its own reference keeps
    // the address from being reused.
    parent: Option<(usize, Arc<[IVec3]>)>,
}

impl TopLevelAadf {
    pub(crate) fn shares_field(&self, other: &Self) -> bool {
        self.minimum == other.minimum
            && self.size == other.size
            && Arc::ptr_eq(&self.fields, &other.fields)
    }

    /// The region cells whose bounds differ from `older`, when this field was
    /// updated in place from it; `None` when they are unrelated.
    pub(crate) fn changed_since(&self, older: &Self) -> Option<&[IVec3]> {
        let (parent, cells) = self.parent.as_ref()?;
        (*parent == older.fields.as_ptr() as *const u8 as usize && !older.fields.is_empty())
            .then_some(&**cells)
    }

    /// A field grown from nothing: what a scene without a predecessor gets.
    #[cfg(test)]
    fn build(regions: &RegionMap) -> Self {
        Self::default().updated(regions, &[])
    }

    /// This field after the regions at `changed` changed residency or
    /// emptiness, at the cost of the change (plus a copy of the field).
    ///
    /// Boxes stay valid throughout: a cell turning blocked cuts the boxes
    /// that held it, and boxes grow only by NAADF's merging rule from
    /// neighbours whose own boxes cover the growth. Growth restarts from the
    /// cells whose neighbourhood changed and stops when no box can grow, as
    /// NAADF refines its chunk bounds over frames. The result is a fixed
    /// point of the same rule a full build reaches, not necessarily the same
    /// one: the order of growth differs.
    fn updated(&self, regions: &RegionMap, changed: &[IVec3]) -> Self {
        let Some((minimum, maximum)) = regions.bounds() else {
            return Self::default();
        };
        let size = (maximum - minimum + IVec3::ONE).as_uvec3();

        // Sparse, disconnected loaded islands must not allocate their entire
        // bounding volume (or overflow the u8 AADF extents). Traversal can safely
        // fall back to region DDA without a top-level distance field.
        if size.max_element() > 255 || size.as_u64vec3().element_product() > TOP_LEVEL_MAX_CELLS {
            return Self {
                minimum,
                maximum,
                size,
                ..default()
            };
        }

        // Missing cells are barriers: the tracer never assumes unloaded world
        // space is empty. Loaded, empty regions are the only top-level cells
        // eligible for skipping.
        let is_blocked = |cell: IVec3| regions.get(&cell).is_none_or(|region| region.any_solid);
        let index = |cell: IVec3| linear_index((cell - minimum).as_uvec3(), size);
        let mut growth = Growth::new(size);
        let (mut fields, mut blocked);
        let in_place = !self.fields.is_empty() && self.minimum == minimum && self.size == size;
        if in_place {
            fields = self.fields.to_vec();
            blocked = self.blocked.to_vec();
        } else {
            // New bounds: keep what this field covers, clipped to them, and
            // start the rest from their regions.
            fields = vec![Aadf::default(); volume_len(size)];
            blocked = vec![true; volume_len(size)];
            let kept = !self.fields.is_empty();
            for z in minimum.z..=maximum.z {
                for y in minimum.y..=maximum.y {
                    for x in minimum.x..=maximum.x {
                        let cell = IVec3::new(x, y, z);
                        let i = index(cell);
                        if kept && cell.cmpge(self.minimum).all() && cell.cmple(self.maximum).all()
                        {
                            let old = linear_index((cell - self.minimum).as_uvec3(), self.size);
                            blocked[i] = self.blocked[old];
                            fields[i] = clip(
                                self.fields[old],
                                (cell - minimum).as_uvec3(),
                                (maximum - cell).as_uvec3(),
                            );
                        } else {
                            blocked[i] = is_blocked(cell);
                            if !blocked[i] {
                                growth.touch(i);
                            }
                        }
                    }
                }
            }
        }

        let mut newly_blocked = Vec::new();
        for &cell in changed {
            if cell.cmplt(minimum).any() || cell.cmpgt(maximum).any() {
                continue;
            }
            let i = index(cell);
            let now = is_blocked(cell);
            if now == blocked[i] {
                continue;
            }
            blocked[i] = now;
            if now {
                fields[i] = Aadf::default();
                newly_blocked.push((cell - minimum).as_uvec3());
            } else {
                growth.touch(i);
            }
        }
        if !newly_blocked.is_empty() {
            cut_boxes(&mut fields, &blocked, size, &newly_blocked, &mut growth);
        }
        growth.run(&mut fields, &blocked);

        let parent = in_place.then(|| {
            let cells: Vec<IVec3> = fields
                .iter()
                .zip(self.fields.iter())
                .enumerate()
                .filter(|(_, (new, old))| new != old)
                .map(|(i, _)| {
                    let i = i as u32;
                    minimum
                        + UVec3::new(i % size.x, (i / size.x) % size.y, i / (size.x * size.y))
                            .as_ivec3()
                })
                .collect();
            (self.fields.as_ptr() as *const u8 as usize, cells.into())
        });
        Self {
            minimum,
            maximum,
            size,
            fields: fields.into(),
            blocked: blocked.into(),
            parent,
        }
    }

    /// The full synchronous NAADF build over the same cells (`build_aadf`),
    /// the field before incremental updates: the reference for box quality.
    #[cfg(test)]
    pub(crate) fn synchronous(&self) -> Vec<Aadf> {
        build_aadf(self.size, &self.blocked)
    }

    /// Panics unless every box holds only empty cells inside the bounds and
    /// no box can grow: a fixed point of the merging rule over `regions`.
    /// A summed-volume table of barriers checks each box in eight reads.
    #[cfg(test)]
    pub(crate) fn assert_valid_fixed_point(&self, regions: &RegionMap) {
        if self.fields.is_empty() {
            return;
        }
        let size = self.size;
        let table_size = size + UVec3::ONE;
        let mut table = vec![0u32; volume_len(table_size)];
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let cell = UVec3::new(x, y, z);
                    let i = linear_index(cell, size);
                    let world = cell.as_ivec3() + self.minimum;
                    assert_eq!(
                        self.blocked[i],
                        regions.get(&world).is_none_or(|region| region.any_solid),
                        "blocked state of {world}"
                    );
                    table[linear_index(cell + UVec3::ONE, table_size)] = u32::from(self.blocked[i]);
                }
            }
        }
        let steps = [
            1,
            table_size.x as usize,
            (table_size.x * table_size.y) as usize,
        ];
        for (axis, step) in steps.into_iter().enumerate() {
            for i in 0..table.len() {
                let coordinate = [
                    i % table_size.x as usize,
                    (i / table_size.x as usize) % table_size.y as usize,
                    i / (table_size.x * table_size.y) as usize,
                ];
                if coordinate[axis] > 0 {
                    table[i] += table[i - step];
                }
            }
        }
        let barriers = |a: UVec3, b: UVec3| {
            let b = b + UVec3::ONE;
            let at = |x: u32, y: u32, z: u32| {
                table[linear_index(UVec3::new(x, y, z), table_size)] as i64
            };
            at(b.x, b.y, b.z) - at(a.x, b.y, b.z) - at(b.x, a.y, b.z) - at(b.x, b.y, a.z)
                + at(a.x, a.y, b.z)
                + at(a.x, b.y, a.z)
                + at(b.x, a.y, a.z)
                - at(a.x, a.y, a.z)
        };
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let cell = UVec3::new(x, y, z);
                    let i = linear_index(cell, size);
                    let world = cell.as_ivec3() + self.minimum;
                    let field = self.fields[i];
                    if self.blocked[i] {
                        assert_eq!(field, Aadf::default(), "blocked {world} has a box");
                        continue;
                    }
                    assert!(
                        cell.cmpge(field.negative()).all(),
                        "{world} box below the bounds"
                    );
                    let (minimum, maximum) = field_box(field, cell);
                    assert!(maximum.cmplt(size).all(), "{world} box above the bounds");
                    assert_eq!(
                        barriers(minimum, maximum),
                        0,
                        "{world}'s box holds a barrier"
                    );
                    let mut grown = field;
                    for direction in 0..6 {
                        let neighbour = cell.as_ivec3() + direction_step(direction);
                        try_grow(
                            &mut grown,
                            neighbour,
                            direction,
                            size,
                            &self.blocked,
                            &self.fields,
                        );
                    }
                    assert_eq!(grown, field, "{world}'s box can still grow");
                }
            }
        }
    }

    /// Mean box volume over empty cells, a proxy for how far rays skip.
    #[cfg(test)]
    pub(crate) fn mean_box_volume(fields: &[Aadf], blocked: &[bool]) -> f64 {
        let (mut sum, mut count) = (0.0, 0usize);
        for (field, blocked) in fields.iter().zip(blocked) {
            if !blocked {
                sum += (field.negative() + field.positive() + UVec3::ONE)
                    .as_dvec3()
                    .element_product();
                count += 1;
            }
        }
        sum / count.max(1) as f64
    }

    #[cfg(test)]
    pub(crate) fn fields_and_blocked(&self) -> (&[Aadf], &[bool]) {
        (&self.fields, &self.blocked)
    }

    pub fn field(&self, coordinate: IVec3) -> Aadf {
        if self.fields.is_empty() {
            return Aadf::default();
        }
        let local = (coordinate - self.minimum).as_uvec3();
        self.fields[linear_index(local, self.size)]
    }
}

/// `field` limited to `below` cells on the negative side and `above` on the
/// positive side of its cell on each axis.
fn clip(field: Aadf, below: UVec3, above: UVec3) -> Aadf {
    let mut bounds = field.bounds;
    for axis in 0..3 {
        bounds[axis * 2] = bounds[axis * 2].min(below[axis].min(255) as u8);
        bounds[axis * 2 + 1] = bounds[axis * 2 + 1].min(above[axis].min(255) as u8);
    }
    Aadf { bounds }
}

/// The box of `field` around `cell`, as inclusive corners.
fn field_box(field: Aadf, cell: UVec3) -> (UVec3, UVec3) {
    (cell - field.negative(), cell + field.positive())
}

/// Cuts every box that holds a newly blocked cell to one that does not:
/// along the one direction that keeps the most volume, or down to its own
/// cell when no single direction clears it. A summed-volume table over the
/// newly blocked cells' bounding box counts them in a box in eight reads.
fn cut_boxes(
    fields: &mut [Aadf],
    blocked: &[bool],
    size: UVec3,
    newly_blocked: &[UVec3],
    growth: &mut Growth,
) {
    let low = newly_blocked.iter().copied().reduce(UVec3::min).unwrap();
    let high = newly_blocked.iter().copied().reduce(UVec3::max).unwrap();
    let extent = high - low + UVec3::ONE;
    // table[(x, y, z)] counts newly blocked cells in [low, low + (x, y, z)).
    let table_size = extent + UVec3::ONE;
    let mut table = vec![0u32; volume_len(table_size)];
    for cell in newly_blocked {
        table[linear_index(*cell - low + UVec3::ONE, table_size)] += 1;
    }
    for axis in 0..3 {
        let step = match axis {
            0 => 1,
            1 => table_size.x as usize,
            _ => (table_size.x * table_size.y) as usize,
        };
        for i in 0..table.len() {
            let coordinate = [
                i % table_size.x as usize,
                (i / table_size.x as usize) % table_size.y as usize,
                i / (table_size.x * table_size.y) as usize,
            ];
            if coordinate[axis] > 0 {
                table[i] += table[i - step];
            }
        }
    }
    let count = |minimum: UVec3, maximum: UVec3| -> u32 {
        // Inclusive box, clamped to the table's cells.
        let a = minimum.max(low);
        let b = maximum.min(high);
        if a.cmpgt(b).any() {
            return 0;
        }
        let (a, b) = (a - low, b - low + UVec3::ONE);
        let at =
            |x: u32, y: u32, z: u32| table[linear_index(UVec3::new(x, y, z), table_size)] as i64;
        (at(b.x, b.y, b.z) - at(a.x, b.y, b.z) - at(b.x, a.y, b.z) - at(b.x, b.y, a.z)
            + at(a.x, a.y, b.z)
            + at(a.x, b.y, a.z)
            + at(b.x, a.y, a.z)
            - at(a.x, a.y, a.z)) as u32
    };
    for z in 0..size.z {
        for y in 0..size.y {
            for x in 0..size.x {
                let cell = UVec3::new(x, y, z);
                let i = linear_index(cell, size);
                if blocked[i] || fields[i] == Aadf::default() {
                    continue;
                }
                let (minimum, maximum) = field_box(fields[i], cell);
                if count(minimum, maximum) == 0 {
                    continue;
                }
                let mut best: Option<(u64, Aadf)> = None;
                for direction in 0..6 {
                    let with = |extent: u8| {
                        let mut field = fields[i];
                        field.bounds[direction] = extent;
                        field
                    };
                    let clear = |extent: u8| {
                        let (minimum, maximum) = field_box(with(extent), cell);
                        count(minimum, maximum) == 0
                    };
                    if !clear(0) {
                        continue;
                    }
                    // The largest clear extent: clear(0) holds, clear(full) does not.
                    let (mut good, mut bad) = (0u8, fields[i].bounds[direction]);
                    while bad - good > 1 {
                        let middle = good + (bad - good) / 2;
                        if clear(middle) {
                            good = middle;
                        } else {
                            bad = middle;
                        }
                    }
                    let field = with(good);
                    let volume = (field.negative() + field.positive() + UVec3::ONE)
                        .as_u64vec3()
                        .element_product();
                    if best.is_none_or(|(most, _)| volume > most) {
                        best = Some((volume, field));
                    }
                }
                fields[i] = best.map_or(Aadf::default(), |(_, field)| field);
                growth.push(i);
            }
        }
    }
}

/// Cells whose boxes may grow, grown in place until none can. A box grows
/// by one cell towards a neighbour whose box covers the slab it adds
/// (`try_grow`); after a cell grows, its neighbours may grow towards it.
struct Growth {
    size: UVec3,
    queue: std::collections::VecDeque<usize>,
    queued: Vec<bool>,
}

/// `try_grow`'s direction `d` (`naadf::NEG_X`..`POS_Z`) as a step.
fn direction_step(direction: usize) -> IVec3 {
    let mut step = IVec3::ZERO;
    step[direction / 2] = if direction.is_multiple_of(2) { -1 } else { 1 };
    step
}

impl Growth {
    fn new(size: UVec3) -> Self {
        Self {
            size,
            queue: Default::default(),
            queued: vec![false; volume_len(size)],
        }
    }

    fn cell(&self, i: usize) -> IVec3 {
        let size = self.size;
        UVec3::new(
            i as u32 % size.x,
            (i as u32 / size.x) % size.y,
            i as u32 / (size.x * size.y),
        )
        .as_ivec3()
    }

    fn push(&mut self, i: usize) {
        if !std::mem::replace(&mut self.queued[i], true) {
            self.queue.push_back(i);
        }
    }

    /// A cell whose own state changed: it and its neighbours may grow.
    fn touch(&mut self, i: usize) {
        self.push(i);
        let cell = self.cell(i);
        for direction in 0..6 {
            let n = cell + direction_step(direction);
            if n.cmpge(IVec3::ZERO).all() && n.cmplt(self.size.as_ivec3()).all() {
                self.push(linear_index(n.as_uvec3(), self.size));
            }
        }
    }

    fn run(&mut self, fields: &mut [Aadf], blocked: &[bool]) {
        while let Some(i) = self.queue.pop_front() {
            self.queued[i] = false;
            if blocked[i] {
                continue;
            }
            let cell = self.cell(i);
            let mut current = fields[i];
            for direction in 0..6 {
                let neighbour = cell + direction_step(direction);
                try_grow(
                    &mut current,
                    neighbour,
                    direction,
                    self.size,
                    blocked,
                    fields,
                );
            }
            if current != fields[i] {
                fields[i] = current;
                self.touch(i);
            }
        }
    }
}

#[derive(Clone, Default)]
pub struct TransportScene {
    pub regions: RegionMap,
    pub top_level: TopLevelAadf,
    journal: delta::SceneJournal,
}

impl TransportScene {
    pub(crate) fn stamp(&self) -> SceneStamp {
        self.journal.stamp.clone()
    }

    pub(crate) fn region_delta_since(&self, stamp: &SceneStamp) -> Option<Vec<IVec3>> {
        self.journal.since(stamp)
    }

    /// Canonical region identity delta. Lineage names the candidates
    /// directly; independent scenes and consumers older than the journal
    /// compare the region maps, which skips the shards they share.
    pub(crate) fn changed_regions(&self, previous: &Self) -> Vec<IVec3> {
        let mut keys = match self.region_delta_since(&previous.stamp()) {
            Some(mut keys) => {
                keys.retain(
                    |key| match (previous.regions.get(key), self.regions.get(key)) {
                        (Some(a), Some(b)) => !Arc::ptr_eq(a, b),
                        (None, None) => false,
                        _ => true,
                    },
                );
                keys
            }
            None => self.regions.changed_keys(&previous.regions),
        };
        keys.sort_unstable_by_key(|p| (p.z, p.y, p.x));
        keys.dedup();
        keys
    }

    #[cfg(test)]
    pub fn with_page_delta(
        previous: &Self,
        changed_pages: Vec<(IVec3, Vec<u16>)>,
        removed_pages: &[IVec3],
    ) -> Self {
        let pages = changed_pages
            .into_iter()
            .map(|(page, materials)| (page, materials, 0))
            .collect();
        Self::with_strided_page_delta(previous, pages, removed_pages)
    }

    /// `with_page_delta` with each page's stride (log2, `PageSample`).
    #[cfg(test)]
    pub fn with_strided_page_delta(
        previous: &Self,
        changed_pages: Vec<(IVec3, Vec<u16>, u8)>,
        removed_pages: &[IVec3],
    ) -> Self {
        let pages = changed_pages
            .into_iter()
            .map(|(page, materials, stride)| (page, PageVoxels::Dense(materials), stride))
            .collect();
        Self::with_page_contents(previous, pages, removed_pages)
    }

    /// `with_strided_page_delta` where a page may be one material throughout
    /// (`PageVoxels::Uniform`), built without expanding it to 32³ voxels.
    pub fn with_page_contents(
        previous: &Self,
        changed_pages: Vec<(IVec3, PageVoxels, u8)>,
        removed_pages: &[IVec3],
    ) -> Self {
        let mut regions = previous.regions.clone();
        let mut touched = HashSet::new();
        for page in removed_pages {
            for offset in page_region_offsets() {
                let key = page * 2 + offset;
                if regions.remove(&key).is_some() {
                    touched.insert(key);
                }
            }
        }

        // Scene construction already runs on Bevy's async compute pool. Keeping
        // this loop serial prevents a rebuild from borrowing every Rayon worker
        // and starving rendering/physics on lower-core-count machines.
        let built: Vec<_> = changed_pages
            .into_iter()
            .flat_map(|(page, materials, stride)| {
                build_page_regions(page, &materials, stride, previous)
            })
            .collect();
        for region in built {
            if previous
                .regions
                .get(&region.coordinate)
                .is_none_or(|old| !Arc::ptr_eq(old, &region))
            {
                touched.insert(region.coordinate);
            }
            regions.insert(region.coordinate, region);
        }

        touched.retain(|key| match (previous.regions.get(key), regions.get(key)) {
            (Some(a), Some(b)) => !Arc::ptr_eq(a, b),
            (None, None) => false,
            _ => true,
        });

        // A voxel/material edit inside a still-occupied region changes rays,
        // but not this level's skip bounds. Reuse only the acceleration field;
        // region identities still drive normal transport/source invalidation.
        let topology_unchanged = touched.iter().all(|key| {
            previous
                .regions
                .get(key)
                .zip(regions.get(key))
                .is_some_and(|(old, region)| old.any_solid == region.any_solid)
        });
        let mut keys: Vec<_> = touched.into_iter().collect();
        keys.sort_unstable_by_key(|p| (p.z, p.y, p.x));
        let top_level = if topology_unchanged {
            previous.top_level.clone()
        } else {
            previous.top_level.updated(&regions, &keys)
        };
        Self {
            regions,
            top_level,
            journal: previous.journal.advance(keys),
        }
    }

    /// Returns `None` outside the loaded scene, `Some(AIR)` for loaded air, or
    /// the material index for a solid voxel.
    pub fn material_at(&self, world_voxel: IVec3) -> Option<u16> {
        let region_coordinate = world_voxel.div_euclid(IVec3::splat(REGION_EDGE as i32));
        let local = world_voxel.rem_euclid(IVec3::splat(REGION_EDGE as i32));
        self.regions
            .get(&region_coordinate)
            .map(|region| region.material(local.as_uvec3()))
    }
}

#[derive(Clone, Debug)]
pub struct SurfaceProbe {
    pub voxel: IVec3,
    pub material: u8,
    pub order_index: usize,
    pub emission: Vec3,
    pub face_mask: u8,
    /// Test-only CPU measurements (`ProbeStatistics`); the GPU state is the
    /// authority and production probes carry none.
    #[cfg(test)]
    pub(crate) stats: Option<Box<ProbeStatistics>>,
    /// Small publication/diagnostic mirror, present without full moments.
    pub(crate) progress: ProbeProgress,
    pub(crate) display: [super::history::FaceDisplay; 6],
    /// Optional resident slot and allocation incarnation. Zero incarnation is
    /// the CPU/display-prior fallback; recycled slots cannot alias old readers.
    pub(crate) resident: [UVec2; 6],
    /// Selection in the full-probe test oracle. Production selection belongs
    /// only to the compact `ProbeWork`, never the persistent receiver.
    #[cfg(test)]
    pub(crate) trace_mask: u8,
    /// Faces refreshed in transport_revision; revision changes are lazy O(1).
    pub(crate) fresh_mask: u8,
    /// Faces whose last GPU merge the CPU rejected (selected before a revision
    /// or model change). The next selection asks the GPU to undo that merge.
    pub(crate) stale_merge: u8,
    /// Faces relit (material or exposure change) since their last accepted
    /// CPU job: that job caps their GPU history to NEAR_HISTORY. The cap is
    /// sent explicitly, never derived from `samples`: the CPU's counts are a
    /// mirror of the GPU's and can lag it (or read 0 for a face never merged).
    pub(crate) relit_mask: u8,
    pub(crate) transport_revision: u64,
    /// Independent spatial evidence consumed by statistical retention.
    /// Sunlight changes can arrive after transport refreshed in the same scene.
    /// Changes only when this probe's geometry/history is invalidated.
    pub revision: u64,
    pub source_revision: u64,
    pub updated_at: Option<Instant>,
}

/// Sequence ordering and HUD progress need no CPU radiance/moment copy.
/// Counts saturate at bootstrap, not at a claimed convergence threshold.
#[derive(Clone, Debug, Default)]
pub(crate) struct ProbeProgress {
    pub sequence: [u32; 6],
    pub bootstrap: [u8; 6],
    pub settled: u8,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FaceMeasurement {
    pub radiance: Vec3,
    pub second_moment: Vec3,
    pub direct: Vec3,
    pub custom_direct: Vec3,
    pub samples: u32,
    pub updated_at: Option<Instant>,
}

/// Test-only CPU copy of a probe's measurements, filled by the native GPU test
/// harness and fixtures. Production keeps none: the GPU state is the authority.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeStatistics {
    /// Unfiltered indirect transport measurements. Presentation history is
    /// separate and never changes these raw moments.
    pub radiance: [Vec3; 6],
    /// Direct incident RGB energy.
    pub direct: [Vec3; 6],
    /// Visible analytic contributions not already rendered by Bevy PBR.
    pub custom_direct: [Vec3; 6],
    pub samples: [u32; 6],
    pub(crate) stable_batches: [u8; 6],
    pub(crate) face_updated_at: [Option<Instant>; 6],
    /// Sampling sequence is independent of history length and invalidation.
    pub sequence: [u32; 6],
    /// Per-channel second raw moment.
    pub second_moment: [Vec3; 6],
}

#[cfg(test)]
impl ProbeStatistics {
    pub(crate) const EMPTY: Self = Self {
        radiance: [Vec3::ZERO; 6],
        direct: [Vec3::ZERO; 6],
        custom_direct: [Vec3::ZERO; 6],
        samples: [0; 6],
        stable_batches: [0; 6],
        face_updated_at: [None; 6],
        sequence: [0; 6],
        second_moment: [Vec3::ZERO; 6],
    };
}

#[cfg(test)]
impl Default for ProbeStatistics {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl SurfaceProbe {
    /// A probe with test-only CPU statistics.
    #[cfg(test)]
    pub(crate) fn new(voxel: IVec3, face_mask: u8) -> Self {
        Self {
            stats: Some(Box::default()),
            ..Self::new_slim(voxel, face_mask)
        }
    }

    /// A probe without CPU statistics: production GPU transport.
    pub(crate) fn new_slim(voxel: IVec3, face_mask: u8) -> Self {
        Self {
            voxel,
            material: 0,
            order_index: 0,
            emission: Vec3::ZERO,
            face_mask,
            #[cfg(test)]
            stats: None,
            progress: default(),
            display: [super::history::FaceDisplay::default(); 6],
            resident: [UVec2::ZERO; 6],
            #[cfg(test)]
            trace_mask: 63,
            fresh_mask: 63,
            stale_merge: 0,
            relit_mask: 0,
            transport_revision: 0,
            revision: 0,
            source_revision: 0,
            updated_at: None,
        }
    }

    /// Test helper: append one raw sample to the CPU statistics.
    #[cfg(test)]
    pub(crate) fn accumulate_sample(&mut self, face: usize, sample: Vec3) {
        let sample = sample.finite_or_zero();
        let count = (self.stats().samples[face] + 1).min(super::tracer::HISTORY_LIMIT);
        let alpha = 1.0 / count as f32;
        let stats = self.stats_mut();
        stats.radiance[face] += (sample - stats.radiance[face]) * alpha;
        stats.second_moment[face] += (sample * sample - stats.second_moment[face]) * alpha;
        stats.samples[face] = count;
        stats.sequence[face] = stats.sequence[face].wrapping_add(1);
    }

    /// CPU statistics, or empty ones when this probe keeps none.
    #[cfg(test)]
    pub(crate) fn stats(&self) -> &ProbeStatistics {
        self.stats.as_deref().unwrap_or(&ProbeStatistics::EMPTY)
    }

    /// CPU statistics, allocated on first write.
    #[cfg(test)]
    pub(crate) fn stats_mut(&mut self) -> &mut ProbeStatistics {
        self.stats.get_or_insert_with(Box::default)
    }

    /// A face's CPU measurement for a job input: radiance, second moment,
    /// direct, custom direct, samples and time. Production probes keep none
    /// (the GPU slot is the authority), so this is empty outside tests.
    pub(crate) fn measured(&self, face: usize) -> FaceMeasurement {
        #[cfg(test)]
        if let Some(stats) = &self.stats {
            return FaceMeasurement {
                radiance: stats.radiance[face],
                second_moment: stats.second_moment[face],
                direct: stats.direct[face],
                custom_direct: stats.custom_direct[face],
                samples: stats.samples[face],
                updated_at: stats.face_updated_at[face],
            };
        }
        let _ = face;
        FaceMeasurement::default()
    }

    pub(crate) fn sequence(&self, face: usize) -> u32 {
        #[cfg(test)]
        if let Some(stats) = &self.stats {
            return stats.sequence[face];
        }
        self.progress.sequence[face]
    }

    pub(crate) fn bootstrap_samples(&self, face: usize) -> u32 {
        #[cfg(test)]
        if let Some(stats) = &self.stats {
            return stats.samples[face].min(super::tracer::BOOTSTRAP_SAMPLES);
        }
        u32::from(self.progress.bootstrap[face])
    }

    pub(crate) fn update_face_mask(&mut self, face_mask: u8) {
        if self.face_mask != face_mask {
            self.request_relight();
        }
        let newly_exposed = face_mask & !self.face_mask;
        self.face_mask = face_mask;
        for face in 0..6 {
            if newly_exposed & (1 << face) == 0 {
                continue;
            }
            self.display[face] = default();
            self.resident[face] = UVec2::ZERO;
            self.progress.bootstrap[face] = 0;
            self.progress.settled &= !(1 << face);
            #[cfg(test)]
            if let Some(stats) = self.stats.as_mut() {
                stats.radiance[face] = Vec3::ZERO;
                stats.second_moment[face] = Vec3::ZERO;
                stats.direct[face] = Vec3::ZERO;
                stats.custom_direct[face] = Vec3::ZERO;
                stats.samples[face] = 0;
                stats.face_updated_at[face] = None;
            }
        }
    }

    /// Changed geometry may affect a path, but does not make every nearby
    /// receiver a new surface. Retain bounded measured history, explicitly
    /// stale, while replacing it with fresh paths. No borrowed/display energy
    /// enters these statistics. Material replacement uses the same rule;
    /// only newly exposed faces start without compatible measurements.
    pub(crate) fn request_relight(&mut self) {
        self.request_refresh();
        self.relit_mask |= self.face_mask;
        let limit = super::geometry::NEAR_HISTORY;
        self.progress.bootstrap = self
            .progress
            .bootstrap
            .map(|count| u32::from(count).min(limit) as u8);
        #[cfg(test)]
        if let Some(stats) = self.stats.as_mut() {
            stats.samples = stats.samples.map(|count| count.min(limit));
        }
    }

    /// A nearby edit: reject in-flight pre-edit results and stop treating the
    /// faces as settled, but keep their sample counts. How much history to
    /// keep is measured retention's decision, and it reads these counts.
    pub(crate) fn request_refresh(&mut self) {
        self.revision = self.revision.wrapping_add(1);
        self.fresh_mask = 0;
        self.progress.settled = 0;
        #[cfg(test)]
        if let Some(stats) = self.stats.as_mut() {
            stats.stable_batches = [0; 6];
        }
    }

    /// Full-probe reference for test fixtures. Production model replacement uses
    /// compact `TraceWork`; keep the same measurement/display separation here.
    #[cfg(test)]
    pub(crate) fn restart_measurements(&mut self) {
        self.fresh_mask = 0;
        let stats = self.stats_mut();
        stats.samples = [0; 6];
        stats.stable_batches = [0; 6];
        stats.face_updated_at = [None; 6];
        stats.radiance = [Vec3::ZERO; 6];
        stats.second_moment = [Vec3::ZERO; 6];
        // Presentation weights belong to the retained display, not the reset
        // transport statistics. Changing only weights would visibly relight
        // unchanged shared corners on every edit.
        // Keep the previous display estimate until replacement samples arrive,
        // but never advertise it as a fresh raw measurement.
        self.updated_at = None;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceProbeDescriptor {
    pub voxel: IVec3,
    pub face_mask: u8,
    pub material: u8,
}

/// Pages whose probe topology can change after the supplied page delta. A
/// voxel edit affects membership in its own page and at the six touching page
/// boundaries; no other page needs to be rescanned.
pub fn affected_probe_pages(changed_pages: &[IVec3], removed_pages: &[IVec3]) -> HashSet<IVec3> {
    let mut affected = HashSet::new();
    for page in changed_pages.iter().chain(removed_pages) {
        affected.insert(*page);
        for offset in FACE_OFFSETS {
            affected.insert(*page + offset);
        }
    }
    affected
}

/// Squared page distance from `page` to the nearest of `foci` (the active
/// cameras' pages), or 0 without one: work near any camera goes first.
pub(crate) fn focus_distance_squared(foci: &[IVec3], page: IVec3) -> i64 {
    foci.iter()
        .map(|focus| (page.as_i64vec3() - focus.as_i64vec3()).length_squared())
        .min()
        .unwrap_or(0)
}

/// Rebuild only the probe descriptors in the requested 32-cubed pages. This
/// is the hot edit path; it avoids cloning and rescanning the complete loaded
/// world merely because one voxel changed.
pub fn scan_surface_probe_pages(
    scene: &TransportScene,
    pages: &HashSet<IVec3>,
    foci: &[IVec3],
) -> Vec<SurfaceProbeDescriptor> {
    let mut found = Vec::new();
    let mut ordered_pages: Vec<_> = pages.iter().copied().collect();
    ordered_pages
        .sort_unstable_by_key(|page| (focus_distance_squared(foci, *page), page.z, page.y, page.x));
    for page in ordered_pages {
        let origin = page * PAGE_EDGE as i32;
        for z in 0..PAGE_EDGE {
            for y in 0..PAGE_EDGE {
                for x in 0..PAGE_EDGE {
                    let voxel = origin + UVec3::new(x, y, z).as_ivec3();
                    let Some(material) =
                        scene.material_at(voxel).filter(|material| *material != AIR)
                    else {
                        continue;
                    };

                    let mut face_mask = 0_u8;
                    for (face, offset) in FACE_OFFSETS.into_iter().enumerate() {
                        if !matches!(scene.material_at(voxel + offset), Some(material) if material != AIR)
                        {
                            face_mask |= 1 << face;
                        }
                    }
                    if face_mask != 0 {
                        found.push(SurfaceProbeDescriptor {
                            voxel,
                            face_mask,
                            material: material as u8,
                        });
                    }
                }
            }
        }
    }
    found
}

#[cfg(test)]
pub fn rebuild_surface_probes(
    scene: &TransportScene,
    previous: &HashMap<IVec3, SurfaceProbe>,
) -> (HashMap<IVec3, SurfaceProbe>, Vec<IVec3>) {
    let pages: HashSet<_> = scene
        .regions
        .keys()
        .map(|region| region.div_euclid(IVec3::splat(2)))
        .collect();
    let descriptors = scan_surface_probe_pages(scene, &pages, &[IVec3::ZERO]);

    let mut probes = HashMap::new();
    for descriptor in descriptors {
        let voxel = descriptor.voxel;
        let face_mask = descriptor.face_mask;
        let mut probe = previous
            .get(&voxel)
            .cloned()
            .unwrap_or_else(|| SurfaceProbe::new(voxel, face_mask));
        probe.face_mask = face_mask;
        // Preserve the last stable value on screen while forcing fresh samples
        // after any geometry change.
        probe.stats_mut().samples = [0; 6];
        probes.insert(voxel, probe);
    }

    let mut order: Vec<_> = probes.keys().copied().collect();
    order.sort_unstable_by_key(|p| (p.z, p.y, p.x));
    (probes, order)
}

/// A changed page's voxels for a scene build.
pub enum PageVoxels {
    /// One material throughout (`AIR` for an empty page).
    Uniform(u16),
    /// `32³` materials, x fastest.
    Dense(Vec<u16>),
}

fn build_page_regions(
    page: IVec3,
    voxels: &PageVoxels,
    stride_log2: u8,
    previous: &TransportScene,
) -> Vec<Arc<NaadfRegion>> {
    let page_materials = match voxels {
        PageVoxels::Dense(materials) => materials,
        &PageVoxels::Uniform(material) => {
            return page_region_offsets()
                .map(|offset| {
                    let coordinate = page * 2 + offset;
                    match previous.regions.get(&coordinate) {
                        Some(region)
                            if region.stride_log2 == stride_log2
                                && region.uniform_material() == Some(material) =>
                        {
                            Arc::clone(region)
                        }
                        _ => Arc::new(
                            NaadfRegion::uniform(coordinate, material).with_stride(stride_log2),
                        ),
                    }
                })
                .collect();
        }
    };
    assert_eq!(page_materials.len(), volume_len(PAGE_SIZE));
    let mut regions = Vec::with_capacity(8);
    for offset in page_region_offsets() {
        let mut materials = vec![AIR; volume_len(REGION_SIZE)];
        let page_start = offset.as_uvec3() * REGION_EDGE;
        for z in 0..REGION_EDGE {
            for y in 0..REGION_EDGE {
                for x in 0..REGION_EDGE {
                    let local = UVec3::new(x, y, z);
                    materials[linear_index(local, REGION_SIZE)] =
                        page_materials[linear_index(page_start + local, PAGE_SIZE)];
                }
            }
        }
        let coordinate = page * 2 + offset;
        if let Some(region) = previous.regions.get(&coordinate)
            && region.stride_log2 == stride_log2
            && region.matches_materials(&materials)
        {
            regions.push(Arc::clone(region));
        } else {
            let region = match previous.regions.get(&coordinate) {
                Some(old) => old.rebuild(materials),
                None => NaadfRegion::build(coordinate, materials),
            };
            regions.push(Arc::new(region.with_stride(stride_log2)));
        }
    }
    regions
}

fn page_region_offsets() -> impl Iterator<Item = IVec3> {
    (0..2).flat_map(|z| (0..2).flat_map(move |y| (0..2).map(move |x| IVec3::new(x, y, z))))
}

#[cfg(test)]
mod construction_tests;

#[cfg(test)]
mod stream_bench;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_ranges_hold_every_page_within_reach() {
        // Points and boxes on page faces, and radii ending on them, included.
        let mut seed = 0x2545_f491_u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        fn coordinate(next: &mut impl FnMut() -> u32) -> f32 {
            let raw = (next() % 257) as f32 - 128.0;
            if next().is_multiple_of(3) {
                (raw / 32.0).round() * 32.0
            } else {
                raw + 0.37
            }
        }
        for _ in 0..200 {
            let a = Vec3::new(
                coordinate(&mut next),
                coordinate(&mut next),
                coordinate(&mut next),
            );
            let b = a + Vec3::new(
                coordinate(&mut next),
                coordinate(&mut next),
                coordinate(&mut next),
            )
            .abs()
                * 0.5;
            let radius = [0.0, 16.0, 32.0, 47.5, 64.0][(next() % 5) as usize];
            let (minimum, maximum) = pages_within_box(a, b, radius);
            let (point_min, point_max) = pages_within(a, radius);
            for z in -8..=8 {
                for y in -8..=8 {
                    for x in -8..=8 {
                        let page = IVec3::new(x, y, z);
                        let low = (page * PAGE_EDGE as i32).as_vec3();
                        let high = low + Vec3::splat(PAGE_EDGE as f32);
                        let gap = (a - high).max(low - b).max(Vec3::ZERO).length();
                        if gap <= radius {
                            assert!(
                                page.cmpge(minimum).all() && page.cmple(maximum).all(),
                                "{a} {b} {radius} {page}"
                            );
                        }
                        if page_distance(page, a) <= radius {
                            assert!(
                                page.cmpge(point_min).all() && page.cmple(point_max).all(),
                                "{a} {radius} {page}"
                            );
                        }
                    }
                }
            }
            // Not wastefully wide: at most one page beyond the reach each way.
            let size = maximum - minimum + IVec3::ONE;
            let reach = ((b - a + 2.0 * radius) / PAGE_EDGE as f32)
                .ceil()
                .as_ivec3()
                + IVec3::splat(2);
            assert!(size.cmple(reach).all(), "{size} > {reach}");
        }
    }

    #[test]
    fn uniform_pages_build_the_regions_dense_expansion_would() {
        for (material, stride) in [(AIR, 5), (7, 5), (7, 2), (AIR, 0)] {
            let page = IVec3::new(1, -2, 3);
            let dense = TransportScene::with_strided_page_delta(
                &TransportScene::default(),
                vec![(page, vec![material; volume_len(PAGE_SIZE)], stride)],
                &[],
            );
            let uniform = TransportScene::with_page_contents(
                &TransportScene::default(),
                vec![(page, PageVoxels::Uniform(material), stride)],
                &[],
            );
            assert_eq!(dense.regions.len(), 8);
            for (key, region) in &dense.regions {
                let other = &uniform.regions[key];
                assert_eq!(region.uniform_material(), other.uniform_material());
                assert_eq!(region.stride_log2, other.stride_log2);
                assert_eq!(region.any_solid, other.any_solid);
                assert_eq!(region.material_mask, other.material_mask);
            }
            // Resending it reuses every region; a different material does not.
            let again = TransportScene::with_page_contents(
                &uniform,
                vec![(page, PageVoxels::Uniform(material), stride)],
                &[],
            );
            assert!(
                uniform
                    .regions
                    .iter()
                    .all(|(key, r)| Arc::ptr_eq(r, &again.regions[key]))
            );
            let other = TransportScene::with_page_contents(
                &uniform,
                vec![(
                    page,
                    PageVoxels::Uniform(if material == AIR { 7 } else { AIR }),
                    stride,
                )],
                &[],
            );
            assert!(
                other
                    .regions
                    .iter()
                    .all(|(key, r)| !Arc::ptr_eq(r, &uniform.regions[key]))
            );
        }
    }

    #[test]
    fn unchanged_page_notifications_reuse_acceleration_regions() {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        page[linear_index(UVec3::new(3, 4, 5), PAGE_SIZE)] = 1;
        let scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page.clone())],
            &[],
        );
        let next = TransportScene::with_page_delta(&scene, vec![(IVec3::ZERO, page)], &[]);
        for (key, region) in &scene.regions {
            assert!(Arc::ptr_eq(region, &next.regions[key]));
        }
    }

    #[test]
    fn material_updates_cannot_introduce_nan_or_amplifying_reflectance() {
        let mut table = VoxelGiMaterials::default();
        table.set(
            7,
            MaterialTransport {
                diffuse: Vec3::splat(5.0),
                emission: Vec3::splat(f32::INFINITY),
            },
        );
        let material = table.get(7);
        assert_eq!(material.diffuse, Vec3::ONE);
        assert_eq!(material.emission, Vec3::ZERO);
    }

    #[test]
    fn page_delta_preserves_other_pages_and_removes_all_child_regions() {
        let solid_page = vec![1; volume_len(PAGE_SIZE)];
        let scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, solid_page.clone()), (IVec3::X, solid_page)],
            &[],
        );
        assert_eq!(scene.regions.len(), 16);

        let scene = TransportScene::with_page_delta(&scene, vec![], &[IVec3::ZERO]);
        assert_eq!(scene.regions.len(), 8);
        assert!(scene.regions.keys().all(|coordinate| coordinate.x >= 2));
    }

    #[test]
    fn probes_exist_only_on_exposed_solid_voxels() {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        let voxel = UVec3::new(4, 5, 6);
        page[linear_index(voxel, PAGE_SIZE)] = 1;
        let scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page)],
            &[],
        );
        let (probes, _) = rebuild_surface_probes(&scene, &HashMap::new());

        let probe = probes.get(&voxel.as_ivec3()).unwrap();
        assert_eq!(probe.face_mask, 0b11_1111);
    }
}
