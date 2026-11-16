//! The scene's region map, persistent across builds: regions grouped into
//! shards of `SHARD_EDGE`³ that scene snapshots share. A build copies the
//! shard directory (one entry per shard) and only the shards it changes, so
//! its cost follows the change, not the loaded world.
use super::NaadfRegion;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use std::sync::Arc;

/// Regions per shard side: 8³ regions is a 128-voxel cube (4³ pages).
const SHARD_EDGE: i32 = 8;

type Bounds = Option<(IVec3, IVec3)>;

fn grow(bounds: Bounds, region: IVec3) -> Bounds {
    Some(bounds.map_or((region, region), |(low, high)| {
        (low.min(region), high.max(region))
    }))
}

fn union(a: Bounds, b: Bounds) -> Bounds {
    match (a, b) {
        (Some((a0, a1)), Some((b0, b1))) => Some((a0.min(b0), a1.max(b1))),
        (a, None) => a,
        (None, b) => b,
    }
}

#[derive(Clone, Default)]
struct Shard {
    regions: HashMap<IVec3, Arc<NaadfRegion>>,
    /// The bounding boxes of `regions` and of those holding a solid voxel.
    bounds: Bounds,
    solid: Bounds,
}

impl Shard {
    fn insert(&mut self, region: IVec3, value: Arc<NaadfRegion>) -> Option<Arc<NaadfRegion>> {
        let solid = value.any_solid;
        let old = self.regions.insert(region, value);
        if old.as_ref().is_some_and(|old| old.any_solid && !solid) {
            // A solid region emptied: the solid box may shrink.
            self.recompute();
        } else {
            self.bounds = grow(self.bounds, region);
            if solid {
                self.solid = grow(self.solid, region);
            }
        }
        old
    }

    fn remove(&mut self, region: &IVec3) -> Option<Arc<NaadfRegion>> {
        let old = self.regions.remove(region);
        if old.is_some() {
            self.recompute();
        }
        old
    }

    fn recompute(&mut self) {
        self.bounds = None;
        self.solid = None;
        for (region, value) in &self.regions {
            self.bounds = grow(self.bounds, *region);
            if value.any_solid {
                self.solid = grow(self.solid, *region);
            }
        }
    }
}

#[derive(Clone, Default)]
pub struct RegionMap {
    shards: HashMap<IVec3, Arc<Shard>>,
    len: usize,
}

fn shard_of(region: IVec3) -> IVec3 {
    region.div_euclid(IVec3::splat(SHARD_EDGE))
}

impl RegionMap {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get(&self, region: &IVec3) -> Option<&Arc<NaadfRegion>> {
        self.shards.get(&shard_of(*region))?.regions.get(region)
    }

    pub fn contains_key(&self, region: &IVec3) -> bool {
        self.get(region).is_some()
    }

    /// Copies the region's shard first if another snapshot shares it.
    pub fn insert(&mut self, region: IVec3, value: Arc<NaadfRegion>) -> Option<Arc<NaadfRegion>> {
        let shard = Arc::make_mut(self.shards.entry(shard_of(region)).or_default());
        let old = shard.insert(region, value);
        if old.is_none() {
            self.len += 1;
        }
        old
    }

    pub fn remove(&mut self, region: &IVec3) -> Option<Arc<NaadfRegion>> {
        let key = shard_of(*region);
        let shard = self.shards.get_mut(&key)?;
        if !shard.regions.contains_key(region) {
            return None;
        }
        let shard = Arc::make_mut(shard);
        let old = shard.remove(region);
        if shard.regions.is_empty() {
            self.shards.remove(&key);
        }
        self.len -= 1;
        old
    }

    pub fn iter(&self) -> impl Iterator<Item = (&IVec3, &Arc<NaadfRegion>)> {
        self.shards.values().flat_map(|shard| shard.regions.iter())
    }

    pub fn keys(&self) -> impl Iterator<Item = &IVec3> {
        self.iter().map(|(key, _)| key)
    }

    pub fn values(&self) -> impl Iterator<Item = &Arc<NaadfRegion>> {
        self.iter().map(|(_, value)| value)
    }

    /// The loaded regions' bounding box (inclusive): one entry per shard.
    pub fn bounds(&self) -> Option<(IVec3, IVec3)> {
        self.shards
            .values()
            .map(|shard| shard.bounds)
            .fold(None, union)
    }

    /// The bounding box of the regions holding a solid voxel, the same way.
    pub fn solid_bounds(&self) -> Option<(IVec3, IVec3)> {
        self.shards
            .values()
            .map(|shard| shard.solid)
            .fold(None, union)
    }
}

impl std::ops::Index<&IVec3> for RegionMap {
    type Output = Arc<NaadfRegion>;

    /// Panics if the region is not loaded, as `HashMap` does.
    fn index(&self, region: &IVec3) -> &Self::Output {
        self.get(region).expect("region is not loaded")
    }
}

impl<'a> IntoIterator for &'a RegionMap {
    type Item = (&'a IVec3, &'a Arc<NaadfRegion>);
    type IntoIter = Box<dyn Iterator<Item = Self::Item> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.iter())
    }
}

impl FromIterator<(IVec3, Arc<NaadfRegion>)> for RegionMap {
    fn from_iter<T: IntoIterator<Item = (IVec3, Arc<NaadfRegion>)>>(iter: T) -> Self {
        let mut map = Self::default();
        for (key, value) in iter {
            map.insert(key, value);
        }
        map
    }
}

impl<const N: usize> From<[(IVec3, Arc<NaadfRegion>); N]> for RegionMap {
    fn from(entries: [(IVec3, Arc<NaadfRegion>); N]) -> Self {
        entries.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_share_untouched_shards_and_bounds_are_exact() {
        let region = |c: IVec3| Arc::new(NaadfRegion::build(c, vec![super::super::AIR; 4096]));
        let mut map: RegionMap = (-20..20)
            .flat_map(|x| (0..3).map(move |y| IVec3::new(x, y, x / 3)))
            .map(|c| (c, region(c)))
            .collect();
        assert_eq!(map.len(), 120);
        assert_eq!(
            map.bounds(),
            Some((IVec3::new(-20, 0, -6), IVec3::new(19, 2, 6)))
        );
        let snapshot = map.clone();
        map.insert(IVec3::new(100, -9, 0), region(IVec3::new(100, -9, 0)));
        assert!(map.remove(&IVec3::new(-20, 0, -6)).is_some());
        assert!(map.remove(&IVec3::new(-20, 0, -6)).is_none());
        assert_eq!(map.len(), 120);
        assert_eq!(snapshot.len(), 120);
        assert!(snapshot.contains_key(&IVec3::new(-20, 0, -6)));
        assert!(!snapshot.contains_key(&IVec3::new(100, -9, 0)));
        assert_eq!(
            map.bounds(),
            Some((IVec3::new(-20, -9, -6), IVec3::new(100, 2, 6)))
        );
        // Untouched shards are the same allocation in both snapshots.
        let shared = map
            .shards
            .iter()
            .filter(|(key, shard)| {
                snapshot
                    .shards
                    .get(*key)
                    .is_some_and(|s| Arc::ptr_eq(s, shard))
            })
            .count();
        assert_eq!(shared, map.shards.len() - 2);
        assert_eq!(map.iter().count(), 120);
        assert_eq!(map.solid_bounds(), None);
        let mut solid = vec![super::super::AIR; 4096];
        solid[0] = 1;
        for c in [IVec3::new(3, 1, 0), IVec3::new(-5, 2, -1)] {
            map.insert(c, Arc::new(NaadfRegion::build(c, solid.clone())));
        }
        assert_eq!(
            map.solid_bounds(),
            Some((IVec3::new(-5, 1, -1), IVec3::new(3, 2, 0)))
        );
        map.insert(IVec3::new(3, 1, 0), region(IVec3::new(3, 1, 0)));
        assert_eq!(
            map.solid_bounds(),
            Some((IVec3::new(-5, 2, -1), IVec3::new(-5, 2, -1)))
        );
    }
}
