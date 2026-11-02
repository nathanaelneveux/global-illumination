# bevy_voxel_gi_bvw

Connects [`bevy_voxel_world`](https://crates.io/crates/bevy_voxel_world) 0.17
to [`bevy_voxel_gi`](https://docs.rs/bevy_voxel_gi): chunk events become GI page events, chunk
data (including its stored levels of detail) becomes GI's `VoxelSource`, and
`VoxelTransportMaterial` is BVW's texture-array terrain material lit by GI.

```toml
[dependencies]
bevy_voxel_gi = "0.1"
bevy_voxel_gi_bvw = "0.1"
bevy_voxel_world = "0.17"
```

Any `VoxelWorldConfig` with `u8` materials works; a BVW chunk is exactly one
GI page (32³), and chunk spawn, update and despawn become page events. Add
`VoxelGiBvwPlugin` for your `VoxelWorldConfig` and give BVW the material
from `material_template`.

GI receives BVW's `u8` material indices unchanged, so set each block's
optics in `VoxelMaterialTransportTable` at its BVW index. Don't use
`bevy_voxel_gi::MaterialPalette` here: it assigns its own sequential IDs and
merges blocks with identical optics, so its IDs need not match BVW's, and a
block could take another block's emission. GI treats every solid BVW voxel
as opaque.

```rust
use bevy::prelude::*;
use bevy_voxel_gi::{MaterialTransport, VoxelMaterialTransportTable};

const STONE: u8 = 0;
const LAMP: u8 = 3;

fn block_optics(mut table: ResMut<VoxelMaterialTransportTable>) {
    table.set(STONE, MaterialTransport { diffuse: Vec3::splat(0.4), ..default() });
    table.set(
        LAMP,
        MaterialTransport { diffuse: Vec3::splat(0.8), emission: Vec3::new(4.0, 2.4, 1.0) },
    );
}
```

LoD chunks are read at their stored stride; colliders and full resolution
belong to your near LoD bands, as in BVW. Licensed under MIT or Apache-2.0.
