# Voxel GI Lab

Persistent, world-space diffuse lighting for the complete loaded voxel world,
forked from `prototype_voxel_template`. Bevy 0.19.1 and ordinary GPU compute;
hardware ray-query support is not required. BVW supplies terrain and meshes, while
GI owns an independent sparse voxel acceleration structure and energy field.

The preferred look is shared-corner direct sunlight, corner-reconstructed
indirect lighting, textures on, and baked AO off. Indirect transport still starts
at exposed voxel-face centers. Neither transport nor virtual voxel corners
depends on greedy-mesh vertices or screen resolution.

## Start here

- [Architecture](docs/architecture.md): data model, estimator, ownership,
  invalidation, presentation, simulation API, and integration limits.
- [Research and decisions](docs/research.md): what we adopted, what differs from
  NAADF/vxray/DDGI/Solari, and the next speed experiments.
- [Validation](docs/validation.md): focused checks, visual workloads, evidence
  requirements, and the shared build cache.
- [October 8 audit](docs/audit.md): findings, fixes, retained boundaries, and
  verification for this cleanup.
- [Current performance experiments](docs/experiments.md): measured winners,
  isolated alternatives, and the limits of each result.
- [Many-emitter research decisions](docs/research-2026-10-09.md): light-tree win,
  cave workload, residency contract, and rejected/deferred correction strategies.
- [Historical experiments](docs/experiments-2026-10.md): previous measurements,
  rejected experiments, and their limitations. This is not the current design.

```sh
cargo run --locked
```

For the opt-in many-emitter cave:

```sh
VOXEL_GI_SCENE=cave cargo run --locked
```

This finite test chamber contains a lava floor with IR, a blue-lit central spire,
green ceiling crystals with ionizing emission, wall torches, and a roof skylight.
It shares geometry/materials with the compute studies; it is not a calibrated
thermal/radiation simulation or a finished art scene. E places a torch and right
click places rock in this palette. Omit the variable to return to normal terrain.

The top-right HUD reports FPS, Coverage, Bootstrap, Settled, and GI CPU/GPU cost.
Coverage means at least one sample; Bootstrap means progress toward eight samples;
Settled means passing a revisitable stability heuristic. None means convergence.
CPU and GPU work overlap, so their displayed costs must not be added.

## Controls

| Input | Action |
| --- | --- |
| WASD / mouse | Move / look |
| Space / Ctrl / Shift | Jump / crouch / sprint |
| Left / right click | Remove / place a voxel |
| E / Add emitting block button | Place a warm emitting voxel at the aimed-at face |
| F3 | Mesh debug overlay |
| F4 | Toggle indirect GI display |
| F5 | Flat faces / virtual-corner interpolation |
| F6 | Irradiance diagnostic view |
| F7 | Cached corner sun / Bevy per-pixel sun comparison |
| Esc | Inspector and cursor release/capture |

Aim before releasing the cursor to use the emitter button. Movement/look pauses
while the cursor is released. Display toggles do not invalidate transport.

## Current behavior

Four new paths per selected face update RGB, infrared, and ionizing channels.
Every loaded exposed surface remains eligible, including offscreen receivers.
Bounded asynchronous work prioritizes nearby changes while reserving a world
sweep for distant surfaces. Finite work budgets mean lighting changes are
progressive, not instantaneous.

Display smoothing is temporal: each face blends new indirect RGB into its own
previous displayed value when a worker result is accepted. The fragment shader
then reconstructs connected coplanar corner values and interpolates across each
unit face. Those are separate temporal and spatial operations. Direct-sun
visibility is measured at shared corners and is not temporally filtered.

The latest display filter deliberately favors stable transitions over minimum
latency. A 75 ms response setting is not a promise to reach final brightness in
75 ms: sample uncertainty and available tracing work can make it much slower.
Increasing raw history is not our default answer to noise.

Terrain configuration is hot-reloaded from
[default.prototype.ron](assets/config/default.prototype.ron). Work limits live in
`VoxelGiSettings`; background transport energy in `VoxelGiEnvironment`; optical
properties in `VoxelMaterialTransportTable`. Source adaptation and texture assets
have explicit owners rather than being encoded into the tracer.

## Scope

This is a tested prototype engine with library-style internal boundaries, not a
published storage-independent crate or an unlimited-world renderer. It supports
opaque, axis-aligned voxels, diffuse bounces, directional/point sources and
emissive voxel materials. The [architecture limits](docs/architecture.md#integration-contract-and-limits)
are part of the design, not promises of implemented planet warps or physical
radiation simulation.

Infrared/ionizing fields are unfiltered transport estimates in normalized units,
not temperature or dose. Gameplay must check freshness and provide absorption,
heat capacity, conduction, cooling and material-state rules. Radiological
penetration and scattering are not modeled by this diffuse opaque solver.

## License

`CC0-1.0 OR MIT OR Apache-2.0`: [CC0](LICENSE-CC0-1.0.txt),
[MIT](LICENSE-MIT.txt), [Apache](LICENSE-Apache-2.0.txt).
