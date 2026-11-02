//! Persistent sparse GPU geometry. Uniform/empty regions have no dense payload;
//! ordinary edits replace only changed 16³ regions. No camera-relative volume.
use bevy::platform::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use super::naadf::Aadf;
use super::sampling::pcg;
use super::world::{AIR, NaadfRegion, SceneStamp, TopLevelAadf, TransportScene};
use bevy::prelude::*;

const BRICK_WORDS: usize = 32; // Two 16-bit voxel cells per word.
const TABLE_WORDS: usize = 64;

#[derive(Default)]
pub(crate) struct PackedScene {
    #[cfg(test)]
    pub(super) revisit_unchanged_for_benchmark: bool,
    #[cfg(test)]
    full_discovery_for_benchmark: bool,
    stamp: Option<SceneStamp>,
    top_level: Option<TopLevelAadf>,
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
    /// The directory was reallocated (first build or a capacity change):
    /// upload it whole. Otherwise only `directory` slot ranges changed.
    pub directory_rebuilt: bool,
    pub directory: Vec<Range<usize>>,
    pub headers: Vec<Range<usize>>,
    pub cells: Vec<Range<usize>>,
}

#[cfg(test)]
impl SceneDelta {
    pub(crate) fn directory_changed(&self) -> bool {
        self.directory_rebuilt || !self.directory.is_empty()
    }
}

const EMPTY_SLOT: IVec4 = IVec4::new(0, 0, 0, -1);
/// The shader's `lookup_region` probes at most this many slots.
const MAX_PROBES: usize = 32;

impl PackedScene {
    pub fn update(&mut self, scene: &TransportScene) -> SceneDelta {
        #[cfg(not(test))]
        let reuse_enabled = true;
        #[cfg(test)]
        let reuse_enabled = !self.revisit_unchanged_for_benchmark;
        #[cfg(not(test))]
        let use_lineage = true;
        #[cfg(test)]
        let use_lineage = !self.full_discovery_for_benchmark;
        let mut changed = (reuse_enabled && use_lineage)
            .then(|| {
                self.stamp
                    .as_ref()
                    .and_then(|stamp| scene.region_delta_since(stamp))
            })
            .flatten()
            .unwrap_or_else(|| {
                scene
                    .regions
                    .keys()
                    .chain(
                        self.regions
                            .keys()
                            .filter(|key| !scene.regions.contains_key(*key)),
                    )
                    .copied()
                    .collect()
            });
        // Only the delta needs stable allocation ordering. Lagging consumers
        // without complete lineage take the same full discovery fallback.
        changed.sort_unstable_by_key(|p| (p.z, p.y, p.x));
        changed.dedup();
        let removed: Vec<_> = changed
            .iter()
            .filter(|key| !scene.regions.contains_key(*key))
            .copied()
            .collect();
        let mut delta = SceneDelta::default();
        // The directory is maintained in place: a full rebuild (and upload)
        // only for the first build or a capacity change. Streaming used to
        // rebuild and re-upload the whole table on every region change.
        let mut rebuild = self.directory.is_empty();
        let mut inserted = Vec::new();
        for key in removed {
            if !rebuild {
                self.directory_remove(key, &mut delta.directory);
            }
            let Some(allocation) = self.regions.remove(&key) else {
                continue;
            };
            self.free_headers.push(allocation.header);
            self.release_bricks(&allocation.bricks);
            if let Some(offset) = allocation.payload {
                self.free_tables.push(offset);
            }
        }
        let mut dirty = Vec::new();
        for key in changed {
            let Some(source) = scene.regions.get(&key) else {
                continue;
            };
            let top = scene.top_level.field(key);
            let negative = pack_bytes(top.negative());
            let positive = pack_bytes(top.positive()) | stride_bits(source);
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
            let existing = self.regions.remove(&key);
            if existing.is_none() {
                inserted.push(key);
            }
            let mut allocation = existing.unwrap_or_else(|| {
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
                    let brick_bounds = source.brick_bounds();
                    for (index, entry) in table.iter_mut().enumerate() {
                        let brick = UVec3::new(
                            index as u32 & 3,
                            (index as u32 >> 2) & 3,
                            index as u32 >> 4,
                        );
                        let first = source.material(brick * 4);
                        let mut voxel_bounds = None;
                        let mut uniform = true;
                        let mut data = [0; BRICK_WORDS];
                        for i in 0..64 {
                            let local = brick * 4
                                + UVec3::new(i as u32 & 3, (i as u32 >> 2) & 3, i as u32 >> 4);
                            let material = source.material(local);
                            uniform &= material == first;
                            // NAADF's tagged 16-bit cell: solid+material OR empty+extents.
                            let cell = if material == AIR {
                                let bounds = voxel_bounds
                                    .get_or_insert_with(|| source.brick_voxel_bounds(brick));
                                pack_extents(bounds[i]) << 1
                            } else {
                                (u32::from(material) << 1) | 1
                            };
                            data[i / 2] |= cell << ((i & 1) * 16);
                        }
                        *entry = if uniform && first == AIR {
                            pack_extents(brick_bounds[index]) << 2
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
        // Region content and acceleration dependencies are different deltas.
        // A remote residency/occupancy change may alter an unchanged region's
        // skip bounds. Only reuse the headers when the shared field is identical.
        if self
            .top_level
            .as_ref()
            .is_none_or(|old| !old.shares_field(&scene.top_level))
        {
            for (&key, allocation) in &self.regions {
                let top = scene.top_level.field(key);
                let header = &mut self.headers[allocation.header];
                let positive = pack_bytes(top.positive()) | stride_bits(&allocation.source);
                let negative = pack_bytes(top.negative());
                if header.z != negative || header.w != positive {
                    header.z = negative;
                    header.w = positive;
                    delta.headers.push(allocation.header..allocation.header + 1);
                }
            }
        }
        let capacity = (self.regions.len() * 2).next_power_of_two().max(1);
        rebuild |= capacity > self.directory.len() || capacity * 8 < self.directory.len();
        if !rebuild {
            for key in inserted {
                let header = self.regions[&key].header;
                if !self.directory_insert(key, header, &mut delta.directory) {
                    rebuild = true;
                    break;
                }
            }
        }
        if rebuild {
            self.rebuild_directory();
            delta.directory.clear();
            delta.directory_rebuilt = true;
        }
        delta.directory = merge_ranges(delta.directory);
        delta.cells = merge_ranges(dirty);
        delta.headers = merge_ranges(delta.headers);
        self.stamp = Some(scene.stamp());
        self.top_level = Some(scene.top_level.clone());
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

    /// Linear-probing insert; false when the probe chain would exceed what the
    /// shader searches (the caller then rebuilds at a larger capacity).
    fn directory_insert(
        &mut self,
        key: IVec3,
        header: usize,
        dirty: &mut Vec<Range<usize>>,
    ) -> bool {
        let mask = self.directory.len() - 1;
        let mut slot = hash_region(key) as usize & mask;
        let mut distance = 0;
        while self.directory[slot].w != -1 {
            slot = (slot + 1) & mask;
            distance += 1;
            if distance >= MAX_PROBES {
                return false;
            }
        }
        self.directory[slot] = key.extend(header as i32);
        dirty.push(slot..slot + 1);
        true
    }

    /// Backward-shift deletion: no tombstones, so lookups still stop at the
    /// first empty slot, and no probe chain gets longer.
    fn directory_remove(&mut self, key: IVec3, dirty: &mut Vec<Range<usize>>) {
        let mask = self.directory.len() - 1;
        let mut slot = hash_region(key) as usize & mask;
        while self.directory[slot].xyz() != key || self.directory[slot].w == -1 {
            if self.directory[slot].w == -1 {
                return;
            }
            slot = (slot + 1) & mask;
        }
        self.directory[slot] = EMPTY_SLOT;
        dirty.push(slot..slot + 1);
        let mut hole = slot;
        let mut next = slot;
        loop {
            next = (next + 1) & mask;
            let entry = self.directory[next];
            if entry.w == -1 {
                return;
            }
            let home = hash_region(entry.xyz()) as usize & mask;
            // The entry may stay when its home lies cyclically in (hole, next].
            let stays = if hole <= next {
                hole < home && home <= next
            } else {
                hole < home || home <= next
            };
            if !stays {
                self.directory[hole] = entry;
                self.directory[next] = EMPTY_SLOT;
                dirty.push(hole..hole + 1);
                dirty.push(next..next + 1);
                hole = next;
            }
        }
    }

    pub(super) fn rebuild_directory(&mut self) {
        let mut capacity = (self.regions.len() * 2).next_power_of_two().max(1);
        loop {
            self.directory = vec![EMPTY_SLOT; capacity];
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

/// Header word w: positive skip bounds in bytes 0-2; bits 28-30 hold the
/// region's stride log2 (`voxel_stride` in voxel_traversal.wgsl).
fn stride_bits(region: &NaadfRegion) -> u32 {
    u32::from(region.stride_log2) << 28
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

    #[test]
    fn lineage_packing_matches_full_discovery_through_skips_eviction_and_topology_changes() {
        let mut scene = TransportScene::default();
        let mut incremental = PackedScene::default();
        let mut control = PackedScene {
            full_discovery_for_benchmark: true,
            ..default()
        };
        for step in 0..48 {
            let page = IVec3::new(step % 4 - 2, 0, -1);
            let mut values = vec![AIR; volume_len(PAGE_SIZE)];
            if step % 3 != 0 {
                values[linear_index(UVec3::splat(8), PAGE_SIZE)] = (step % 7) as u16;
            }
            scene = if step % 7 == 0 {
                TransportScene::with_page_delta(&scene, vec![], &[page])
            } else {
                TransportScene::with_page_delta(&scene, vec![(page, values)], &[])
            };
            // Consumers can miss both a few versions and more than the ledger.
            if step < 5 || step == 17 || step >= 31 {
                let a = incremental.update(&scene);
                let b = control.update(&scene);
                assert_eq!(directory_entries(&incremental), directory_entries(&control));
                assert_eq!(incremental.directory.len(), control.directory.len());
                assert_eq!(incremental.headers, control.headers);
                assert_eq!(incremental.cells, control.cells);
                assert_eq!(a.directory_changed(), b.directory_changed());
                assert_eq!(a.headers, b.headers);
                assert_eq!(a.cells, b.cells);
            }
        }
    }

    #[test]
    fn incremental_directory_matches_lookups_through_streaming_churn() {
        // A camera streaming through a world: regions enter and leave in
        // waves. Every live region must be found within the shader's probe
        // limit, removed ones must miss, and slot patches must reproduce the
        // CPU table without whole-table uploads between capacity changes.
        let mut packed = PackedScene::default();
        let mut scene = TransportScene::default();
        let mut gpu = Vec::new();
        let mut live = bevy::platform::collections::HashSet::<IVec3>::new();
        let mut rebuilds = 0;
        let mut solid = vec![AIR; volume_len(PAGE_SIZE)];
        solid[linear_index(UVec3::splat(3), PAGE_SIZE)] = 7;
        for step in 0..60i32 {
            let enter: Vec<_> = (0..6)
                .map(|i| IVec3::new(step + i, (i * 7 + step) % 5 - 2, i - 3))
                .collect();
            let leave: Vec<_> = live
                .iter()
                .copied()
                .filter(|page: &IVec3| page.x < step - 4 || (page.x + page.y * 3 + step) % 11 == 0)
                .collect();
            let changed: Vec<_> = enter.iter().map(|&page| (page, solid.clone())).collect();
            scene = TransportScene::with_page_delta(&scene, changed, &leave);
            for page in &leave {
                live.remove(page);
            }
            live.extend(enter);
            let delta = packed.update(&scene);
            if delta.directory_rebuilt {
                rebuilds += 1;
                gpu.clone_from(&packed.directory);
            } else {
                for range in delta.directory {
                    gpu[range.clone()].copy_from_slice(&packed.directory[range]);
                }
            }
            assert_eq!(gpu, packed.directory, "patched directory at step {step}");
            let mask = gpu.len() - 1;
            let lookup = |key: IVec3| {
                let mut slot = hash_region(key) as usize & mask;
                for _ in 0..MAX_PROBES {
                    let entry = gpu[slot];
                    if entry.w == -1 {
                        return None;
                    }
                    if entry.xyz() == key {
                        return Some(entry.w as usize);
                    }
                    slot = (slot + 1) & mask;
                }
                None
            };
            for (&key, allocation) in &packed.regions {
                assert_eq!(
                    lookup(key),
                    Some(allocation.header),
                    "region {key} at step {step}"
                );
            }
            for page in &leave {
                for z in 0..2 {
                    for y in 0..2 {
                        for x in 0..2 {
                            assert_eq!(lookup(*page * 2 + IVec3::new(x, y, z)), None);
                        }
                    }
                }
            }
            assert_eq!(
                gpu.iter().filter(|entry| entry.w != -1).count(),
                packed.regions.len()
            );
        }
        assert!(
            rebuilds < 12,
            "{rebuilds} whole-table uploads in 60 streaming steps"
        );
    }

    fn directory_entries(packed: &PackedScene) -> Vec<[i32; 4]> {
        let mut entries: Vec<_> = packed
            .directory
            .iter()
            .filter(|entry| entry.w != -1)
            .map(|entry| entry.to_array())
            .collect();
        entries.sort_unstable();
        entries
    }

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
    fn coarse_page_headers_carry_their_stride() {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        for z in 0..32 {
            for x in 0..32 {
                for y in 0..8 {
                    page[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] = 3;
                }
            }
        }
        let coarse = TransportScene::with_strided_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page.clone(), 3)],
            &[],
        );
        let mut packed = PackedScene::default();
        packed.update(&coarse);
        assert!(!packed.regions.is_empty());
        for allocation in packed.regions.values() {
            let header = packed.headers[allocation.header];
            assert_eq!(header.w >> 28, 3, "stride log2 in header w bits 28-30");
            assert_eq!(
                header.w & 0x00ff_ffff,
                header.w & 0x0fff_ffff,
                "skip bounds unchanged"
            );
        }
        // The same materials at full resolution are a different region: the
        // view keys its faces per voxel again, so it must not be reused.
        let full =
            TransportScene::with_strided_page_delta(&coarse, vec![(IVec3::ZERO, page, 0)], &[]);
        assert_eq!(full.changed_regions(&coarse).len(), coarse.regions.len());
        packed.update(&full);
        for allocation in packed.regions.values() {
            assert_eq!(packed.headers[allocation.header].w >> 28, 0);
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
        assert!(!dirty.directory_changed());
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
        assert!(dirty.directory_changed());
        assert_eq!(packed.cells.len(), TABLE_WORDS + BRICK_WORDS);
        assert_eq!((packed.cells[TABLE_WORDS] & 65535) >> 1, 255);
        let unchanged = packed.update(&scene);
        assert!(!unchanged.directory_changed());
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
            if delta.directory_rebuilt {
                gpu_directory.clone_from(&packed.directory);
            } else {
                for range in delta.directory {
                    gpu_directory[range.clone()].copy_from_slice(&packed.directory[range]);
                }
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
