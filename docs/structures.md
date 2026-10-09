# Structures experiment

The normal world contains the original house and a nearby many-emitter cave.
Run `cargo run --locked`; no scene flag is needed. From spawn, walk toward the
house, take the pale stone path to the right, and follow it to the lit entrance
at approximately **(20, 11, -18)**. Descend the passage; inside, the stone steps
turn left along the lava rim. The block stairs descend without a jump; climbing
back up uses the normal jump controls. Lava and crystals currently demonstrate
transport, not heat damage, fluid simulation or radiation dose.

The chamber spans x=44..99, y=-36..3, z=-60..-5. Its authored skylight continues
through terrain to the surface. It is a finite, deliberately legible lighting
fixture embedded in terrain, not a natural-cave generator. The original isolated
`VOXEL_GI_SCENE=cave` scene and compute fixture remain available and unchanged;
their geometry surroundings, source settings and performance are not equivalent
to the normal world's.

## Ownership and composition

`src/structures/mod.rs` has no BVW, renderer, GI or live ECS dependencies. It uses
integer vector coordinates, resolved material IDs and immutable data:

- **StructureTemplate:** a bounded array of local voxel operations, authored
  once and shared by `Arc`. The current authoring adapter bakes Rust procedures;
  a file loader could produce the same data without altering chunk generation.
- **PlacedStructure:** template reference, translated integer origin, half-open
  bounds and a unique stable ID. Higher IDs win overlapping writes; input order,
  chunk loading order and worker scheduling cannot change the result.
- **StructureLayer:** immutable instance plan with a spatial bucket index.
  Direct geometry queries inspect only their bucket. Chunk jobs collect and
  deduplicate intersecting instances once, including the one-voxel mesh halo,
  then use a compact candidate list for voxel lookups.
- **VoxelOp:** `Keep` falls through to the earlier layer/base terrain; `Air`
  explicitly carves; `Solid(material)` writes a block. Air is never treated as
  a missing template cell. Transparent cells preserve earlier overlapping stamps.

`src/structures/showcase.rs` is the host's authoring/placement policy. The house
foundation and shell, cave chamber, skylight shaft, entrance passage, internal
stairs and approach paths are ordinary instances. The chamber reuses
`cave_scene::voxel`; its local material IDs map to dedicated world materials at
authoring time. Normal terrain material 2 and the warm lamp at 4 are untouched.
Optical responses are registered once by `TerrainPlugin`, not by a GI special
case or every time a block is placed.

Generation order is base terrain, deterministic structure overrides, then BVW's
existing edited-voxel overlay. Stamps are not runtime systems that rewrite loaded
chunks. Player edits therefore continue to override generation and persist under
BVW's existing in-session modified-voxel ownership; this does not add disk saves.
Structures apply before bedrock/empty-chunk fast paths, allowing the underground
chamber to carve below the normal bedrock level without changing global terrain.

Both BVW's worker delegate and GI's `PrototypeWorld::generated_voxel` fallback
query the same plan. Otherwise a cave could look empty while GI/collisions saw
solid terrain. The normal plan is baked once per process and shared across
configuration reloads; it does not retain any receiver/light-cache state.
The isolated cave does not allocate the normal structure plan.

## Deliberate boundaries

This is a generation/composition experiment, not the finished Starwrought
structure subsystem. Prior Starwrought structure chats were not retrievable in
this session, so no claim of matching an earlier undocumented contract is made.

Translation is implemented; rotations, mirrors, authored file assets, procedural
placement rules, terrain-conforming foundations, entity/socket spawning, live
structure lifecycle, network identities and persistent edit migration are not.
Do not confuse a placement ID used for stable precedence with a complete game
entity or network ID. Current template constructors accept trusted Rust-authored
data and assert valid dimensions/unique IDs; untrusted asset loading needs bounded
validation before allocation. Dense stamps suit these compact examples; a run or
sparse representation can later replace their internal storage without changing
the sampler's Keep/Air/Solid contract.

Planet-local coordinates belong in the placement adapter, not inside a cave
shader or the GI estimator. Likewise, material-state simulation should edit the
world normally; it must not mutate a shared generation template to turn one
instance's lava or crystal on/off.

## Focused verification

Structure tests cover translated instances, deterministic overlap, transparent
versus carved cells, duplicate IDs, negative coordinates, chunk halo agreement,
the original house/foundation oracle, palette separation, the supported route
and open skylight, and agreement between BVW jobs and direct GI geometry queries
above/below bedrock. Run `cargo test --locked --offline structure`.

`VOXEL_GI_VALIDATE_PROFILE=structures` captures the house/path, entrance and cave
in the normal streamed world. It uses a fixed 65° FOV and teleports the view at
12/24 seconds; it is a geometry/lighting integration check, not a locomotion
test or a matched comparison with the isolated chamber. The route unit checks
support and three air cells of headroom, not all controller/physics behavior.
Use the standard headless validation flags for unattended captures. New visible
geometry and emitter load change the normal gameplay workload: measurements from
before this experiment are not an equal-scene performance baseline.

### October 9 integration record

Baseline `9930e2b` is retained on `experiment/structures-baseline-20261009`.
Core/house migration `4ac7cc5`; normal-world cave placement `417f00b`.
The test run used the exact placement code in `417f00b` plus the subsequently
committed `structures` validation profile; no other algorithm/settings changes.
Six focused tests, strict all-target Clippy and a development build passed.
The tests preceded a semantics-equivalent Clippy cleanup of the road-height
comparison; Clippy/build and the rendered check followed it.

Apple M4 / Metal, locked dev profile, 2560×1440 offscreen, 65° vertical FOV,
preferred shared-corner sunlight / reconstructed indirect, textures on / AO off,
normal ten-chunk spawning distance and default transport budget. The four
captures have valid dimensions/nonblank content; approach, entrance and final
interior were visually inspected. Info-level render logs contain no warnings or
errors. At completion: 715 loaded pages, 798,246 published voxel probes, no
pending scene publication, 100% coverage/bootstrap. These are workload facts,
not proof of convergence. The distant cave wall still shows refinement noise
36 seconds after entering; unlike the isolated scene it shares the work budget
with the much larger surrounding terrain. No GI scheduling change or FPS gain
is claimed by this placement work.

Evidence: `target/structures-final-{unit,clippy,build,run}.log` and
`target/gi-validation/structures-1791554330438148000-13244/`.
