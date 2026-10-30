# voxel_gi_bvw

Connects [`bevy_voxel_world`](https://crates.io/crates/bevy_voxel_world) 0.17
to [`voxel_gi`](https://docs.rs/voxel_gi): chunk events become GI page events, chunk
data (including its stored levels of detail) becomes GI's `VoxelSource`, and
`VoxelTransportMaterial` is BVW's texture-array terrain material lit by GI.

Any `VoxelWorldConfig` with `u8` materials works; a BVW chunk is exactly one
GI page (32³), and chunk spawn, update and despawn become page events. Add
`VoxelGiBvwPlugin` for your `VoxelWorldConfig`, give BVW the material
from `material_template`, and pass your block optics through
`voxel_gi::MaterialPalette` (BVW's `u8` material indices map one to one when
there are at most 256).

LoD chunks are read at their stored stride; colliders and full resolution
belong to your near LoD bands, as in BVW. Licensed under MIT or Apache-2.0.
