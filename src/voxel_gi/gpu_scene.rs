//! Persistent sparse GPU geometry. Uniform/empty regions have no dense payload;
//! ordinary edits replace only changed 16³ regions. No camera-relative volume.
use bevy::platform::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use super::naadf::Aadf;
use super::sampling::pcg;
use super::world::{AIR, NaadfRegion, TransportScene};
use bevy::prelude::*;

const BRICK_WORDS: usize = 32; // Two 16-bit voxel cells per word.
const TABLE_WORDS: usize = 64;

#[derive(Default)]
pub(crate) struct PackedScene {
    #[cfg(test)]
    pub(super) revisit_unchanged_for_benchmark: bool,
    regions: HashMap<IVec3, RegionAllocation>,
    free_headers: Vec<usize>,
    free_tables: Vec<usize>,
    free_bricks: Vec<usize>,
    brick_lookup: HashMap<[u32; BRICK_WORDS], usize>,
    bricks: Vec<Option<SharedBrick>>,
    free_brick_ids: Vec<usize>,
    pub directory: Vec<IVec4>,
    pub headers: Vec<UVec4>,
    pub cells: Vec<u32>,
}

struct RegionAllocation {
    source: Arc<NaadfRegion>,
    header: usize,
    payload: Option<usize>,
    bricks: Vec<usize>,
}

struct SharedBrick {
    data: [u32; BRICK_WORDS],
    offset: usize,
    references: usize,
}

/// Element ranges, not byte offsets. A new GPU allocation must receive the
/// complete corresponding array; these deltas only patch an existing buffer.
#[derive(Default)]
pub(crate) struct SceneDelta {
    pub directory_changed: bool,
    pub headers: Vec<Range<usize>>,
    pub cells: Vec<Range<usize>>,
}

impl PackedScene {
    pub fn update(&mut self, scene: &TransportScene) -> SceneDelta {
        let removed: Vec<_> = self
            .regions
            .keys()
            .filter(|key| !scene.regions.contains_key(*key))
            .copied()
            .collect();
        let mut delta = SceneDelta {
            directory_changed: !removed.is_empty() || self.directory.is_empty(),
            ..default()
        };
        for key in removed {
            let allocation = self.regions.remove(&key).unwrap();
            self.free_headers.push(allocation.header);
            self.release_bricks(&allocation.bricks);
            if let Some(offset) = allocation.payload {
                self.free_tables.push(offset);
            }
        }
        let mut dirty = Vec::new();
        // Stable ordering makes new allocations and regression tests reproducible.
        let mut ordered: Vec<_> = scene.regions.iter().collect();
        ordered.sort_unstable_by_key(|(key, _)| (key.z, key.y, key.x));
        for (&key, source) in ordered {
            let top = scene.top_level.field(key);
            let negative = pack_bytes(top.negative());
            let positive = pack_bytes(top.positive());
            #[cfg(not(test))]
            let reuse_enabled = true;
            #[cfg(test)]
            let reuse_enabled = !self.revisit_unchanged_for_benchmark;
            let reusable = if reuse_enabled {
                self.regions
                    .get(&key)
                    .filter(|a| Arc::ptr_eq(&a.source, source))
            } else {
                None
            };
            if let Some(allocation) = reusable {
                // Immutable voxel storage and its allocations remain valid.
                // Neighbouring residency/occupancy changes can still alter this
                // region's top-level skip bounds, independently of its voxels.
                let header = &mut self.headers[allocation.header];
                if header.z != negative || header.w != positive {
                    header.z = negative;
                    header.w = positive;
                    delta.headers.push(allocation.header..allocation.header + 1);
                }
                continue;
            }
            let mut allocation = self.regions.remove(&key).unwrap_or_else(|| {
                delta.directory_changed = true;
                let header = self.free_headers.pop().unwrap_or_else(|| {
                    self.headers.push(UVec4::ZERO);
                    self.headers.len() - 1
                });
                RegionAllocation {
                    source: Arc::clone(source),
                    header,
                    payload: None,
                    bricks: Vec::new(),
                }
            });
            let previous_header = self.headers[allocation.header];
            let changed =
                !Arc::ptr_eq(&allocation.source, source) || self.headers[allocation.header].x == 0;
            if let Some(material) = source.uniform_material() {
                self.release_bricks(&allocation.bricks);
                allocation.bricks.clear();
                if let Some(offset) = allocation.payload.take() {
                    self.free_tables.push(offset);
                }
                self.headers[allocation.header] = UVec4::new(
                    if material == AIR { 1 } else { 2 },
                    u32::from(material),
                    negative,
                    positive,
                );
            } else {
                let new_payload = allocation.payload.is_none();
                let offset = *allocation.payload.get_or_insert_with(|| {
                    self.free_tables.pop().unwrap_or_else(|| {
                        let offset = self.cells.len();
                        self.cells.resize(offset + TABLE_WORDS, 0);
                        offset
                    })
                });
                self.headers[allocation.header] = UVec4::new(3, offset as u32, negative, positive);
                if changed || new_payload {
                    self.release_bricks(&allocation.bricks);
                    allocation.bricks.clear();
                    let mut table = [0; TABLE_WORDS];
                    for (index, entry) in table.iter_mut().enumerate() {
                        let brick = UVec3::new(
                            index as u32 & 3,
                            (index as u32 >> 2) & 3,
                            index as u32 >> 4,
                        );
                        let first = source.material(brick * 4);
                        let mut uniform = true;
                        let mut data = [0; BRICK_WORDS];
                        for i in 0..64 {
                            let local = brick * 4
                                + UVec3::new(i as u32 & 3, (i as u32 >> 2) & 3, i as u32 >> 4);
                            let material = source.material(local);
                            uniform &= material == first;
                            // NAADF's tagged 16-bit cell: solid+material OR empty+extents.
                            let cell = if material == AIR {
                                pack_extents(source.voxel_aadf(local)) << 1
                            } else {
                                (u32::from(material) << 1) | 1
                            };
                            data[i / 2] |= cell << ((i & 1) * 16);
                        }
                        *entry = if uniform && first == AIR {
                            pack_extents(source.brick_aadf(brick)) << 2
                        } else if uniform {
                            (u32::from(first) << 2) | 1
                        } else {
                            let id = self.intern_brick(data, &mut dirty);
                            allocation.bricks.push(id);
                            let pointer = self.bricks[id].as_ref().unwrap().offset;
                            ((pointer as u32) << 2) | 2
                        };
                    }
                    self.cells[offset..offset + TABLE_WORDS].copy_from_slice(&table);
                    dirty.push(offset..offset + TABLE_WORDS);
                }
            }
            if self.headers[allocation.header] != previous_header {
                delta.headers.push(allocation.header..allocation.header + 1);
            }
            allocation.source = Arc::clone(source);
            self.regions.insert(key, allocation);
        }
        if delta.directory_changed {
            self.rebuild_directory();
        }
        delta.cells = merge_ranges(dirty);
        delta.headers = merge_ranges(delta.headers);
        delta
    }

    fn intern_brick(&mut self, data: [u32; BRICK_WORDS], dirty: &mut Vec<Range<usize>>) -> usize {
        // Hash collisions are resolved by full array equality, never hash alone.
        if let Some(&id) = self.brick_lookup.get(&data) {
            self.bricks[id].as_mut().unwrap().references += 1;
            return id;
        }
        let offset = self.free_bricks.pop().unwrap_or_else(|| {
            let offset = self.cells.len();
            self.cells.resize(offset + BRICK_WORDS, 0);
            offset
        });
        self.cells[offset..offset + BRICK_WORDS].copy_from_slice(&data);
        dirty.push(offset..offset + BRICK_WORDS);
        let id = self.free_brick_ids.pop().unwrap_or_else(|| {
            self.bricks.push(None);
            self.bricks.len() - 1
        });
        self.bricks[id] = Some(SharedBrick {
            data,
            offset,
            references: 1,
        });
        self.brick_lookup.insert(data, id);
        id
    }

    fn release_bricks(&mut self, ids: &[usize]) {
        for &id in ids {
            let brick = self.bricks[id].as_mut().unwrap();
            brick.references -= 1;
            if brick.references == 0 {
                self.brick_lookup.remove(&brick.data);
                self.free_bricks.push(brick.offset);
                self.free_brick_ids.push(id);
                self.bricks[id] = None;
            }
        }
    }

    pub(super) fn rebuild_directory(&mut self) {
        let mut capacity = (self.regions.len() * 2).next_power_of_two().max(1);
        loop {
            self.directory = vec![IVec4::new(0, 0, 0, -1); capacity];
            let mut fits = true;
            for (&key, allocation) in &self.regions {
                let mut slot = hash_region(key) as usize & (capacity - 1);
                let mut distance = 0;
                while self.directory[slot].w != -1 {
                    slot = (slot + 1) & (capacity - 1);
                    distance += 1;
                }
                if distance >= 32 {
                    fits = false;
                    break;
                }
                self.directory[slot] = key.extend(allocation.header as i32);
            }
            if fits {
                break;
            }
            capacity *= 2;
        }
    }
}

fn merge_ranges(mut ranges: Vec<Range<usize>>) -> Vec<Range<usize>> {
    ranges.sort_unstable_by_key(|range| range.start);
    let mut merged: Vec<Range<usize>> = Vec::new();
    for range in ranges {
        if let Some(last) = merged.last_mut()
            && range.start <= last.end
        {
            last.end = last.end.max(range.end);
        } else {
            merged.push(range);
        }
    }
    merged
}

fn pack_bytes(value: UVec3) -> u32 {
    value.x | (value.y << 8) | (value.z << 16)
}

fn pack_extents(field: Aadf) -> u32 {
    field.bounds.iter().enumerate().fold(0, |bits, (i, value)| {
        assert!(*value < 4);
        bits | (u32::from(*value) << (i * 2))
    })
}

pub(crate) fn hash_region(key: IVec3) -> u32 {
    pcg(pcg(pcg(key.x as u32) ^ key.y as u32) ^ key.z as u32)
}

#[cfg(test)]
mod tests {
    use super::super::naadf::{linear_index, volume_len};
    use super::super::world::PAGE_SIZE;
    use super::*;

    fn decoded_material(packed: &PackedScene, voxel: IVec3) -> u16 {
        let key = voxel.div_euclid(IVec3::splat(16));
        let allocation = &packed.regions[&key];
        let header = packed.headers[allocation.header];
        if header.x == 1 {
            return AIR;
        }
        if header.x == 2 {
            return header.y as u16;
        }
        let local = voxel.rem_euclid(IVec3::splat(16)).as_uvec3();
        let brick = local / 4;
        let entry = packed.cells[header.y as usize + linear_index(brick, UVec3::splat(4))];
        match entry & 3 {
            0 => AIR,
            1 => (entry >> 2) as u16,
            2 => {
                let index = linear_index(local % 4, UVec3::splat(4));
                let word = packed.cells[(entry >> 2) as usize + index / 2];
                let cell = (word >> ((index & 1) * 16)) & 65535;
                if cell & 1 == 0 {
                    AIR
                } else {
                    (cell >> 1) as u16
                }
            }
            _ => panic!("invalid brick state"),
        }
    }

    #[test]
    fn repeated_bricks_share_storage_and_edits_are_copy_on_write() {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        for z in 0..32 {
            for x in 0..32 {
                page[linear_index(UVec3::new(x, 4, z), PAGE_SIZE)] = 255;
            }
        }
        let scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page.clone())],
            &[],
        );
        let mut packed = PackedScene::default();
        packed.update(&scene);
        assert_eq!(packed.brick_lookup.len(), 1);
        assert_eq!(
            packed.cells.len() * 4,
            1152,
            "four region tables and one shared mixed brick replace 66,560 bytes"
        );
        assert_eq!(
            packed
                .bricks
                .iter()
                .flatten()
                .map(|b| b.references)
                .sum::<usize>(),
            64
        );
        page[linear_index(UVec3::new(4, 4, 4), PAGE_SIZE)] = AIR;
        let edited =
            TransportScene::with_page_delta(&scene, vec![(IVec3::ZERO, page.clone())], &[]);
        let dirty = packed.update(&edited);
        assert!(dirty.cells.iter().map(|r| r.len()).sum::<usize>() <= TABLE_WORDS + BRICK_WORDS);
        assert!(!dirty.directory_changed);
        assert!(dirty.headers.is_empty());
        assert_eq!(decoded_material(&packed, IVec3::new(4, 4, 4)), AIR);
        assert_eq!(
            decoded_material(&packed, IVec3::new(20, 4, 20)),
            255,
            "editing one reference must not modify another"
        );
        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    assert_eq!(
                        decoded_material(&packed, IVec3::new(x, y, z)),
                        page[linear_index(UVec3::new(x as u32, y as u32, z as u32), PAGE_SIZE)]
                    );
                }
            }
        }
        let empty = TransportScene::with_page_delta(&edited, vec![], &[IVec3::ZERO]);
        packed.update(&empty);
        assert!(packed.brick_lookup.is_empty());
        assert!(packed.bricks.iter().all(Option::is_none));
        let high_water = packed.cells.len();
        packed.update(&edited);
        assert_eq!(
            packed.cells.len(),
            high_water,
            "streaming reuses freed slots"
        );
    }

    #[test]
    fn mixed_region_of_uniform_bricks_has_no_voxel_payload() {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    page[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] =
                        if x < 8 { 0 } else { 255 };
                }
            }
        }
        let scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page)],
            &[],
        );
        let mut packed = PackedScene::default();
        packed.update(&scene);
        assert_eq!(packed.cells.len(), TABLE_WORDS);
        assert!(packed.brick_lookup.is_empty());
        assert_eq!(decoded_material(&packed, IVec3::new(3, 3, 3)), 0);
        assert_eq!(decoded_material(&packed, IVec3::new(9, 3, 3)), 255);
    }

    #[test]
    fn edits_patch_only_changed_payload_and_uniform_regions_use_no_dense_memory() {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        page[linear_index(UVec3::new(4, 4, 4), PAGE_SIZE)] = 255;
        let scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page.clone())],
            &[],
        );
        let mut packed = PackedScene::default();
        let dirty = packed.update(&scene);
        assert_eq!(dirty.cells.len(), 1);
        assert!(dirty.directory_changed);
        assert_eq!(packed.cells.len(), TABLE_WORDS + BRICK_WORDS);
        assert_eq!((packed.cells[TABLE_WORDS] & 65535) >> 1, 255);
        let unchanged = packed.update(&scene);
        assert!(!unchanged.directory_changed);
        assert!(unchanged.cells.is_empty() && unchanged.headers.is_empty());
        let empty = TransportScene::with_page_delta(
            &scene,
            vec![(IVec3::ZERO, vec![AIR; volume_len(PAGE_SIZE)])],
            &[],
        );
        packed.update(&empty);
        let next = TransportScene::with_page_delta(&empty, vec![(IVec3::ZERO, page)], &[]);
        assert_eq!(packed.update(&next).cells.len(), 1);
        assert_eq!(
            packed.cells.len(),
            TABLE_WORDS + BRICK_WORDS,
            "freed payload must be reused"
        );
    }

    #[test]
    fn upload_deltas_reconstruct_every_live_table_through_residency_changes() {
        let mut packed = PackedScene::default();
        let mut scene = TransportScene::default();
        let mut gpu_headers = Vec::new();
        let mut gpu_cells = Vec::new();
        let mut gpu_directory = Vec::new();
        let mut header_capacity = 16;
        let mut cell_capacity = 64;
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        let point = linear_index(UVec3::new(4, 4, 4), PAGE_SIZE);
        page[point] = 1;
        for stage in 0..7 {
            let (changed, removed) = match stage {
                0 => (vec![(IVec3::ZERO, page.clone())], vec![]),
                1 => {
                    page[point] = 255;
                    (vec![(IVec3::ZERO, page.clone())], vec![])
                }
                2 => (
                    (1..10)
                        .map(|x| (IVec3::new(x, 0, 0), page.clone()))
                        .collect(),
                    vec![],
                ),
                3 => (vec![], (1..10).map(|x| IVec3::new(x, 0, 0)).collect()),
                4 => (
                    vec![(IVec3::new(-1, 0, 0), vec![2; volume_len(PAGE_SIZE)])],
                    vec![IVec3::ZERO],
                ),
                5 => (
                    vec![(IVec3::new(-1, 0, 0), vec![AIR; volume_len(PAGE_SIZE)])],
                    vec![],
                ),
                _ => (vec![], vec![IVec3::new(-1, 0, 0)]),
            };
            scene = TransportScene::with_page_delta(&scene, changed, &removed);
            let delta = packed.update(&scene);
            if delta.directory_changed {
                gpu_directory.clone_from(&packed.directory);
            }
            if packed.headers.len() > header_capacity {
                header_capacity = packed.headers.len().next_power_of_two();
                gpu_headers.clone_from(&packed.headers);
            } else {
                gpu_headers.resize(packed.headers.len(), UVec4::ZERO);
                for range in delta.headers {
                    gpu_headers[range.clone()].copy_from_slice(&packed.headers[range]);
                }
            }
            if packed.cells.len() > cell_capacity {
                cell_capacity = packed.cells.len().next_power_of_two();
                gpu_cells.clone_from(&packed.cells);
            } else {
                gpu_cells.resize(packed.cells.len(), 0);
                for range in delta.cells {
                    gpu_cells[range.clone()].copy_from_slice(&packed.cells[range]);
                }
            }
            assert_eq!(
                gpu_directory, packed.directory,
                "directory at stage {stage}"
            );
            assert_eq!(gpu_headers, packed.headers, "headers at stage {stage}");
            assert_eq!(gpu_cells, packed.cells, "cells at stage {stage}");
            for (&key, allocation) in &packed.regions {
                let field = scene.top_level.field(key);
                assert_eq!(
                    gpu_headers[allocation.header].z,
                    pack_bytes(field.negative())
                );
                assert_eq!(
                    gpu_headers[allocation.header].w,
                    pack_bytes(field.positive())
                );
                let mut slot = hash_region(key) as usize & (gpu_directory.len() - 1);
                let mut found = false;
                for _ in 0..32 {
                    let entry = gpu_directory[slot];
                    if entry.w == -1 {
                        break;
                    }
                    if entry.xyz() == key {
                        assert_eq!(entry.w as usize, allocation.header);
                        found = true;
                        break;
                    }
                    slot = (slot + 1) & (gpu_directory.len() - 1);
                }
                assert!(found, "missing live region at stage {stage}");
            }
        }
    }

    #[test]
    fn disconnected_world_does_not_allocate_its_bounding_box() {
        let scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![
                (IVec3::ZERO, vec![AIR; volume_len(PAGE_SIZE)]),
                (IVec3::splat(10_000), vec![AIR; volume_len(PAGE_SIZE)]),
            ],
            &[],
        );
        let mut packed = PackedScene::default();
        packed.update(&scene);
        assert!(packed.cells.is_empty());
        assert_eq!(packed.headers.len(), 16);
    }
}
