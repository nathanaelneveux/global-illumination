//! Worker-owned coplanar adjacency, indexed by stable resident slot. Four edges
//! replace repeated world-position hashes during reconstruction; no radiance,
//! neighborhood lists or history-dependent decisions are cached here.
use super::{HashMap, Key, Slot};
use bevy::prelude::*;

const MISSING: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Node {
    edges: [u32; 4],
    incarnation: u32,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            edges: [MISSING; 4],
            incarnation: 0,
        }
    }
}

/// Twenty bytes per allocated slot, only while adaptive spatial display is on.
/// Slot removal unlinks both endpoints before reuse. Reseeding updates identity
/// without rebuilding geometry. GPU donors still carry incarnation guards.
#[derive(Default)]
pub(super) struct SurfaceTopology {
    nodes: Vec<Node>,
}

impl SurfaceTopology {
    pub fn build(slots: &HashMap<Key, Slot>, high_water: u32) -> Self {
        let mut graph = Self {
            nodes: vec![Node::default(); high_water as usize],
        };
        for slot in slots.values() {
            graph.set_incarnation(slot.index, slot.incarnation);
        }
        for &key in slots.keys() {
            graph.connect(key, slots);
        }
        graph
    }

    pub fn set_incarnation(&mut self, index: u32, incarnation: u32) {
        if self.nodes.len() <= index as usize {
            self.nodes.resize(index as usize + 1, Node::default());
        }
        self.nodes[index as usize].incarnation = incarnation;
    }

    pub fn connect(&mut self, key: Key, slots: &HashMap<Key, Slot>) {
        let index = slots[&key].index;
        let (u, v) = crate::voxel_gi::spatial::tangent_axes(key.1);
        for (edge, offset) in [u, -u, v, -v].into_iter().enumerate() {
            if let Some(other) = slots.get(&(key.0 + offset, key.1)) {
                self.nodes[index as usize].edges[edge] = other.index;
                self.nodes[other.index as usize].edges[edge ^ 1] = index;
            }
        }
    }

    pub fn remove(&mut self, index: u32) {
        for (edge, other) in self.nodes[index as usize].edges.into_iter().enumerate() {
            if other != MISSING {
                self.nodes[other as usize].edges[edge ^ 1] = MISSING;
            }
        }
        self.nodes[index as usize] = Node::default();
    }

    pub fn neighbor(&self, index: u32, edge: usize) -> Option<UVec2> {
        let other = self.nodes[index as usize].edges[edge];
        (other != MISSING).then(|| UVec2::new(other, self.nodes[other as usize].incarnation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel_gi::spatial::evidence::{Gather, IndexedGather};

    fn slot(index: u32) -> Slot {
        Slot {
            index,
            incarnation: index + 1,
            revision: 0,
            model_epoch: 0,
            task_epoch: 0,
            task_index: 0,
        }
    }

    fn compare(graph: &SurfaceTopology, slots: &HashMap<Key, Slot>) {
        let mut hashed = Gather::default();
        let mut indexed = Gather::default();
        let mut compact = IndexedGather::default();
        for (&(voxel, face), slot) in slots {
            for limit in [1, 4, 32, 256, 1024] {
                let center = UVec2::new(slot.index, slot.incarnation);
                hashed.collect(voxel, face, center, limit, |p, f| {
                    slots
                        .get(&(p, f))
                        .map(|s| UVec2::new(s.index, s.incarnation))
                });
                indexed.collect_connected(center, limit, |s, edge, _| graph.neighbor(s.x, edge));
                assert_eq!(
                    hashed.sites, indexed.sites,
                    "{voxel:?}/{face}, limit={limit}"
                );
                compact.collect(center, limit, |s, edge| graph.neighbor(s.x, edge));
                assert_eq!(
                    compact.values().collect::<Vec<_>>(),
                    hashed.sites.iter().map(|s| s.1).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn indexed_walk_matches_hashes_through_growth_holes_reseeding_and_recycling() {
        let mut slots = HashMap::default();
        // All six orientations, negative coordinates, edges and a gap which
        // can be walked around but not jumped. Coplanar normals never mix.
        for face in 0..6 {
            let (u, v) = crate::voxel_gi::spatial::tangent_axes(face);
            for x in -5..6 {
                for y in -5_i32..6 {
                    if x == 0 && y.abs() < 3 {
                        continue;
                    }
                    let index = slots.len() as u32;
                    slots.insert((u * x + v * y, face), slot(index));
                }
            }
        }
        let mut graph = SurfaceTopology::build(&slots, slots.len() as u32);
        compare(&graph, &slots);
        let removed = (IVec3::new(-1, 0, 0), 2);
        let old = slots.remove(&removed).unwrap();
        graph.remove(old.index);
        compare(&graph, &slots);
        // Reuse the same index in the gap, not at its old world position.
        let replacement = (IVec3::ZERO, 2);
        let mut new = slot(old.index);
        new.incarnation += 10_000;
        slots.insert(replacement, new);
        graph.set_incarnation(new.index, new.incarnation);
        graph.connect(replacement, &slots);
        // Reseed an existing face without changing topology.
        let existing = slots.get_mut(&(IVec3::new(1, 0, 0), 2)).unwrap();
        existing.incarnation += 20_000;
        graph.set_incarnation(existing.index, existing.incarnation);
        compare(&graph, &slots);
        assert_eq!(std::mem::size_of::<Node>(), 20);
    }

    #[test]
    #[ignore = "isolated CPU ordered-neighborhood benchmark, no GPU or rendering"]
    fn indexed_neighborhood_benchmark() {
        use std::{hint::black_box, time::Instant};
        let mut slots = HashMap::default();
        for z in 0..96 {
            for x in 0..96 {
                slots.insert((IVec3::new(x, 0, z), 2), slot(slots.len() as u32));
            }
        }
        let started = Instant::now();
        let graph = SurfaceTopology::build(&slots, slots.len() as u32);
        eprintln!(
            "topology build_ms={:.3} slots={} bytes={}",
            started.elapsed().as_secs_f64() * 1000.,
            slots.len(),
            graph.nodes.capacity() * std::mem::size_of::<Node>()
        );
        let mut gather = Gather::default();
        let mut compact = IndexedGather::default();
        for count in [16, 128, 512] {
            let limit = crate::voxel_gi::spatial::evidence::candidate_limit(2048, count, 0);
            let mut times = [Vec::new(), Vec::new(), Vec::new()];
            for iteration in 0..9 {
                for mode in [iteration % 3, (iteration + 1) % 3, (iteration + 2) % 3] {
                    let start = Instant::now();
                    for (&(voxel, face), slot) in &slots {
                        let center = UVec2::new(slot.index, slot.incarnation);
                        if mode == 0 {
                            gather.collect(voxel, face, center, limit, |p, f| {
                                slots
                                    .get(&(p, f))
                                    .map(|s| UVec2::new(s.index, s.incarnation))
                            });
                        } else if mode == 1 {
                            gather.collect_connected(center, limit, |s, edge, _| {
                                graph.neighbor(s.x, edge)
                            });
                        } else {
                            compact.collect(center, limit, |s, edge| graph.neighbor(s.x, edge));
                            black_box(compact.values().last());
                        }
                        black_box(&gather.sites);
                    }
                    times[mode].push(start.elapsed().as_secs_f64() * 1000.);
                }
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            eprintln!(
                "history={count} limit={limit} hash_p50_ms={:.3} indexed_p50_ms={:.3} compact_p50_ms={:.3}",
                times[0][4], times[1][4], times[2][4]
            );
        }
    }
}
