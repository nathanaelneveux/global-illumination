//! Persistent sparse GPU geometry. Uniform/empty regions have no dense payload;
//! ordinary edits replace only changed 16³ regions. No camera-relative volume.
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use super::naadf::Aadf;
use super::sampling::pcg;
use super::world::{AIR, NaadfRegion, TransportScene};
use bevy::prelude::*;

pub(crate) const REGION_WORDS: usize = 64 + 4096;

#[derive(Default)]
pub(crate) struct PackedScene {
    regions: HashMap<IVec3, RegionAllocation>,
    free_headers: Vec<usize>,
    free_payloads: Vec<usize>,
    pub directory: Vec<IVec4>,
    pub headers: Vec<UVec4>,
    pub cells: Vec<u32>,
}

struct RegionAllocation {
    source: Arc<NaadfRegion>,
    header: usize,
    payload: Option<usize>,
}

impl PackedScene {
    pub fn update(&mut self, scene: &TransportScene) -> Vec<Range<usize>> {
        let removed: Vec<_> = self
            .regions
            .keys()
            .filter(|key| !scene.regions.contains_key(key))
            .copied()
            .collect();
        for key in removed {
            let allocation = self.regions.remove(&key).unwrap();
            self.free_headers.push(allocation.header);
            if let Some(offset) = allocation.payload {
                self.free_payloads.push(offset);
            }
        }
        let mut dirty = Vec::new();
        // Stable ordering makes new allocations and regression tests reproducible.
        let mut ordered: Vec<_> = scene.regions.iter().collect();
        ordered.sort_unstable_by_key(|(key, _)| (key.z, key.y, key.x));
        for (&key, source) in ordered {
            let allocation = self.regions.entry(key).or_insert_with(|| {
                let header = self.free_headers.pop().unwrap_or_else(|| {
                    self.headers.push(UVec4::ZERO);
                    self.headers.len() - 1
                });
                RegionAllocation {
                    source: Arc::clone(source),
                    header,
                    payload: None,
                }
            });
            let changed =
                !Arc::ptr_eq(&allocation.source, source) || self.headers[allocation.header].x == 0;
            let top = scene.top_level.field(key);
            let negative = pack_bytes(top.negative());
            let positive = pack_bytes(top.positive());
            if let Some(material) = source.uniform_material() {
                if let Some(offset) = allocation.payload.take() {
                    self.free_payloads.push(offset);
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
                    self.free_payloads.pop().unwrap_or_else(|| {
                        let offset = self.cells.len();
                        self.cells.resize(offset + REGION_WORDS, 0);
                        offset
                    })
                });
                self.headers[allocation.header] = UVec4::new(3, offset as u32, negative, positive);
                if changed || new_payload {
                    pack_region(source, &mut self.cells[offset..offset + REGION_WORDS]);
                    dirty.push(offset..offset + REGION_WORDS);
                }
            }
            allocation.source = Arc::clone(source);
        }
        self.rebuild_directory();
        dirty
    }

    fn rebuild_directory(&mut self) {
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

fn pack_bytes(value: UVec3) -> u32 {
    value.x | (value.y << 8) | (value.z << 16)
}

fn pack_extents(field: Aadf) -> u32 {
    field.bounds.iter().enumerate().fold(0, |bits, (i, value)| {
        assert!(*value < 4);
        bits | (u32::from(*value) << (i * 2))
    })
}

fn pack_region(region: &NaadfRegion, out: &mut [u32]) {
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                let brick = UVec3::new(x, y, z);
                out[((z * 4 + y) * 4 + x) as usize] = u32::from(region.brick_occupied(brick))
                    | (pack_extents(region.brick_aadf(brick)) << 1);
            }
        }
    }
    for z in 0..16 {
        for y in 0..16 {
            for x in 0..16 {
                let local = UVec3::new(x, y, z);
                out[64 + ((z * 16 + y) * 16 + x) as usize] = u32::from(region.material(local))
                    | (pack_extents(region.voxel_aadf(local)) << 9);
            }
        }
    }
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
        assert_eq!(dirty.len(), 1);
        assert_eq!(packed.cells.len(), REGION_WORDS);
        assert_eq!(packed.cells[64 + (4 * 16 + 4) * 16 + 4] & 511, 255);
        assert!(packed.update(&scene).is_empty());
        let empty = TransportScene::with_page_delta(
            &scene,
            vec![(IVec3::ZERO, vec![AIR; volume_len(PAGE_SIZE)])],
            &[],
        );
        packed.update(&empty);
        let next = TransportScene::with_page_delta(&empty, vec![(IVec3::ZERO, page)], &[]);
        assert_eq!(packed.update(&next).len(), 1);
        assert_eq!(
            packed.cells.len(),
            REGION_WORDS,
            "freed payload must be reused"
        );
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
