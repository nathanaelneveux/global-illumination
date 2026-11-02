# Asteroids-redux integration experiment

> Historical integration report at `db5c02d`. The failures below belong to that revision. Coarse pages, host-material lighting functions and discovery interest bounds have since landed; large-world scene-build cost remains a current limitation. See [architecture](architecture.md) and [research priorities](research.md#after-01) for current behavior.

Branch `experiment/voxel-gi` in `~/Developer/asteroids-redux` (local, not
pushed), against GI at `db5c02d`. Goal: drop the crates into a second real BVW
game unchanged and list what breaks. `ASTEROIDS_GI=0` runs the original
material for comparison.

## Wiring needed

- Path dependencies on the three crates, and `[patch.crates-io]` pointing
  `bevy_voxel_world` at the game's fork (the connector names crates.io 0.17;
  the fork is also 0.17.0, so one BVW resolves and types unify).
- `VoxelGiPlugin` and `VoxelGiBvwPlugin::<AsteroidWorld>` before the world
  plugin (it reads `VoxelGiMaterialData` to build the material), and
  `impl GiFallbackVoxel for AsteroidWorld {}`.
- The world's material swapped for `bevy_voxel_gi_bvw::material_template`.

It compiled first time with no GI source changes.

## Blind spots found

1. **GI scene size follows host streaming, not GI's needs.** Every loaded
   chunk becomes a GI page. Asteroids keeps ~117k chunks loaded out to 70
   chunks (78k at LOD 8, data sampled at 9³). Result at 2560x1440 headless
   ("above" view): median frame 155 ms vs 11.7 ms without GI, then a wgpu
   submission timeout; the "inside" view lost a GPU snapshot mapping
   (`receiving on an empty channel`) and panicked. The lab never showed this:
   its streaming distance is small. Core needs its own geometry residency
   radius (receivers already have `cache_radius`), and faces outside the GI
   scene need a defined non-GI look.
2. **LOD chunks are not full-resolution pages.** At LOD > 1 BVW stores
   resampled data; `get_voxel_at_world_position` answers only at sample
   points, and the rest falls to `GiFallbackVoxel` (air by default).
3. **`GiFallbackVoxel` cannot be implemented by this game.** Its per-voxel
   `&self` lookup has no chunk context, while asteroids' generator is a
   chunk-scoped closure (`voxel_lookup_delegate(chunk, lod, previous)`) with
   per-chunk caches. The connector could use BVW's own delegate at full
   resolution instead and drop the trait.
4. **Host materials.** The game's material does an impostor handoff dither
   (discard against a fade uniform) and disables its prepass. GI's material
   replaces it, losing the dither; GI's view tier needs a depth/normal
   prepass, so the game's material as written would give GI nothing to
   discover. GI lighting has to be callable from the host's own material
   (stage 4), and the prepass requirement must be explicit.
5. **Ambient fill.** The game lights shadows with `GlobalAmbientLight`
   (brightness 80); GI's material ignores it and uses `VoxelGiEnvironment`
   (an Earth-like sky gradient by default). Space needs a decision on what
   escaped rays and the fill mean.
6. **Faces outside GI's scene are expensive, not ignored.** The view tier
   discovers every visible face from the prepass; a face whose page is not in
   the GI scene stays at zero evidence forever ("uncovered") and keeps paying
   for spatial reconstruction and the material's per-pixel fallback search.
7. **Prepass of every host mesh.** GI's material opts all chunk meshes into
   the depth/normal prepass; asteroids' material opts out. For ~117k chunks
   that draw alone costs ~13 ms.
8. **Timestamp counters.** Bevy's `RenderDiagnosticsPlugin` together with
   GI's timestamp queries lost the Metal device ("Cannot allocate sample
   buffer").

9. **The BVW performance fork changes GI's edit timing.** With the lab
   patched onto `perf/optimization-series-preview` (2026-10-24), the default
   gate showed tunnel black-then-lit 3,404 px (0 on crates.io 0.17), more
   black pixels after a turn (about 7,700 vs 4,400) and a failed walk run;
   static differences were host content (the fork loads more horizon
   terrain). Freshly edited meshes appear before GI's geometry catches up.
   The lab stays on crates.io 0.17 until this is fixed; asteroids-redux
   uses the fork.

## Measurements

M4, `ASTEROIDS_BENCHMARK` (headless, uncapped, 2560x1440, 15-20 s warmup),
"above" view, median wall frame. `VOXEL_GI_BVW_PAGE_RADIUS` and
`VOXEL_GI_NO_PREPASS` were temporary connector hooks
(`docs/voxel-gi-experiment-hooks.patch` on the asteroids branch, not in GI).

| Configuration | Median ms | GI presentation ms | Uncovered |
| --- | ---: | ---: | ---: |
| GI off | 11.7 | | |
| GI off, camera prepass (voxels still opt out) | 12.3 | | |
| GI on, all chunks are pages | 155.6 (then wgpu timeout) | 16.7 | 49k |
| GI on, pages within 8 chunks | 77.9 | 16.7 | 65k |
| GI on, no pages | 80.3 | 16.5 | 65k |
| GI on, no pages, display off | 49.9 | 16.7 | 65k |
| GI on, no pages, no voxel prepass | 26.3 | 0.4 | 0 |
| GI on, no pages, no voxel prepass, display off | 20.5 | 0.4 | 0 |
| GI on, no pages, voxels hidden | 4.6 (off: 3.8) | 0.4 | 0 |

Reading: GI's own compute is ~0.8 ms when it has nothing to do. With the
prepass feeding it faces it cannot light, its presentation passes cost
~16 ms and the material's display path ~30 ms; the voxel prepass draw ~13 ms;
the material's remaining main-pass cost over the asteroid material ~9 ms (it
shades every LOD chunk the asteroid material would have dithered away).
Near-field GI quality on asteroids was not assessed: from the default
capture pose almost nothing is within a 16-chunk GI radius.

## What this asks of GI

- A GI region: geometry residency by distance (like receivers'
  `cache_radius`), independent of how far the host streams.
- A cheap "not mine" path: discovery and the material skip faces outside the
  GI scene and fall back to plain Bevy lighting, with no per-pixel search.
- Prepass only where GI needs it, or discovery that does not require every
  host mesh in the prepass.
- Lighting callable from a host material (stage 4) so hosts keep their own
  material features (dithered fades).
- Connector sampling through BVW's own lookup delegate at full resolution,
  replacing `GiFallbackVoxel`.
