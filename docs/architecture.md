# Architecture

This is the current design. Historical experiments are kept separately so the
implementation can be understood without reconstructing its development history.
The numerical model and the stable display policy are deliberately separate.

## Data flow and ownership

```text
host chunks / edits / sources / material table
                   |
    scene worker: immutable pages, NAADFs, one SceneChanges comparison
                   |
    main world: bounded receiver publication + revision journals
                   |
    scheduler -> compact worker jobs -> GPU compute or CPU reference
                   |                          |
                   +---- validated results ---+
                   |
         authoritative face measurements
              /                 \
    simulation estimates     temporal RGB display history
                                    |
                       generation-owned GPU display cache
                                    |
                   connected virtual-corner reconstruction
```

| Responsibility | Owner / source |
| --- | --- |
| Plugin wiring and runtime state | `src/voxel_gi/mod.rs` |
| BVW events and generator snapshots | `bvw.rs` |
| Scene/receiver representation | `world.rs`, `field.rs` |
| Acceleration and packing | `naadf.rs`, `gpu_scene.rs` |
| Change metadata and retention | `geometry.rs`, `sources.rs`, `revision.rs` |
| Receiver page publication | `publication.rs` |
| Eligibility, priority and dispatch | `scheduling.rs`, `dispatch.rs` |
| Thread-transfer payload and result merge | `work.rs` |
| Source model and Bevy light adaptation | `sources.rs` |
| Sampling, emitters and transport | `sampling.rs`, `emission.rs`, `tracer.rs`, `gpu.rs`, `voxel_trace.wgsl` |
| Shared uncertainty heuristic | `statistics.rs` |
| Direct-sun visibility cache | `sky.rs`, `sky_cache.rs` |
| Temporal display and initialization | `history.rs` |
| Display publication and rendering | `display.rs`, `render.rs`, `voxel_transport.wgsl` |
| Accounting, optional UI, fixtures | `telemetry.rs`, `diagnostics.rs`, `validation/` |

Shader paths are under `assets/shaders/`; Rust paths above are under
`src/voxel_gi/`. `reconstruction.rs` is an independent CPU test oracle for shader
interpolation, not a second production reconstruction path.

## Geometry and traversal

The host exports immutable dense 32³ pages of material IDs. A page contains eight
16³ regions; each region contains 4³ bricks of 4³ voxels. Material IDs are 0–255;
air is the distinct value 256. Regions that are unchanged retain `Arc` identity.
Empty and uniform regions avoid dense voxel storage.
Uniform empty brick fields use exact distances to the volume edges; fully blocked
fields are zero. Only mixed fields require iterative neighbor merging.

Within a changed dense region, occupancy-identical 4³ bricks reuse their complete
voxel bounds. The region's brick-level bounds are reused only when its brick
occupancy map is unchanged. Changed occupancy is fully rebuilt before publication;
material IDs and presence masks update independently even when bounds are reused.
Old snapshots remain immutable. This is exact incremental construction, not a
temporary coarse field or background-refinement queue.

Nested axis-aligned empty-space bounds accelerate rays at the region, brick and
voxel levels. GPU brick tags distinguish empty, uniform and mixed storage. Mixed
bricks use packed 16-bit cells and content-checked, reference-counted deduplication;
editing shared data is copy-on-write. Freed allocations are recycled. The CPU
tracer is an independent representation/reference, not a decoder of GPU cells.
Unchanged region identities retain packed allocations. Their top-level bounds are
still refreshed when neighbours change. GPU upload plans contain changed header
and cell ranges; the directory is rebuilt/uploaded only for residency changes.
A grown buffer receives the full live array before sparse patching resumes.

Top-level bounds include loaded regions, but do not allocate an arbitrarily large
box between disconnected islands: oversized domains fall back to region DDA.
The top-level field depends only on region residency and empty/nonempty occupancy.
Immutable snapshots share its allocation across edits that preserve that topology;
changed voxel regions still trigger normal source and transport invalidation.
Rays normalize direction, offset from surfaces, traverse at most 2,048 steps and
768 voxel units per segment, and distinguish hits, escape and unknown traversal.
These limits and the small-throughput cutoff are approximations, not an exact
infinite-domain light solver.
GPU empty-space box stepping uses vector operations with the scalar reference's
ordered XYZ tie and epsilon policy. This is an execution optimization, not a new
intersection rule, shorter traversal limit or change to the path estimator.

Transport treats a missing region before occupied loaded regions along a ray as
unknown and contributes no energy for that path. Air-only frontiers do not create
an occluder. Display sun rays instead test all loaded geometry across gaps, so
streaming holes do not cast invented dark slabs. Neither policy sees unloaded
geometry. These are explicit query semantics over the same traversal structure,
not two unrelated voxel engines.

`SceneChanges` computes changed material IDs and tight changed-voxel bounds in one
worker comparison. Region load/unload uses full region bounds because residency
changes knowledge of empty space too. Air-only changes outside previous occupied
bounds need bookkeeping but do not invalidate existing indirect transport; holes
inside those bounds remain conservative invalidations. Material IDs are retained
until publication so the live material table can determine whether emission
changed while the worker ran.

## Receiver and transport model

Each exposed solid voxel owns a six-face ambient cube. Only exposed faces are
sampled. Each face stores a mean, second raw moment, direct term, effective history
weight, independent sample sequence, timestamps and freshness/stability state.
The index owns boxed probes so growing a hash table does not relocate all large
receiver records on the game thread. Stable logical keys are voxel coordinates,
not mesh vertices or array indices.

Each selected face receives four paths per normal dispatch. Rotated Halton
dimensions preserve the tested cosine-hemisphere and roulette sequences. Emitter
selection and emitter surface coordinates use independently padded 2D Sobol pairs
with index shuffling and hash-based Owen scrambling. CPU and GPU use the same
integer operations; the second Sobol dimension uses a Pascal-matrix butterfly,
so no direction tables or per-receiver guide are needed. Sequences advance
independently of retained history, so invalidation does not repeat early rays.
The GPU uses four lanes per face; the serial entry remains an opt-in reference.
Dispatch tasks are encoded directly into a reusable 32-byte-per-task upload
array, with explicit little-endian words and initialized padding. This removes
the intermediate typed array without changing the shader ABI. Scratch capacity
tracks the largest dispatch, not the number of loaded receivers; no path history
is added. A test-only generic encoder is the byte-layout oracle.
The [experiment results](experiments.md) explain why Sobol is restricted to
emitter dimensions rather than replacing every path dimension.

Paths evaluate up to three diffuse bounces, analytic visibility, voxel emission,
and the directional environment gradient. Analytic point lights use range and
inverse-square attenuation with a one-unit near-distance clamp. Directional
lights have no attenuation. All analytic sources are currently evaluated; the
voxel light hierarchy does not accelerate the analytic point-light loop.
Emissive-face next-event sampling uses a power CDF below 32 emitting faces and
a receiver-conditioned spatial binary tree for larger sets. Node importance
approximates distance, projected emitter/receiver normals and power, with a
power-mixture fallback and numerical probability floor to preserve support.
Selection and reverse density use the same receiver position/normal for
balance-heuristic MIS with cosine path hits. Actual visibility still traces
against current geometry. Source weights include all five scalar channels:
R, G, B, infrared and ionizing. The tree shares the emitter storage binding,
adds 64 bytes per node and rebuilds with the worker's emitter snapshot; there is
no per-receiver allocation or incremental tree refit. See the
[many-emitter results and tradeoffs](research-2026-10-09.md#spatial-light-hierarchy).

The core stores normalized energy. It does not attach physical watts, temperature
or dose to those values. The environment gradient uses world Y; it must become a
host-defined directional environment before a planet-local-up integration.

Ordinary Bevy direct RGB stays in PBR (or the explicit cached-sun override).
Face-center direct energy is still measured for simulations, and direct lighting
is evaluated at bounce hits. Indirect RGB, custom-emitter direct RGB and voxel
self-emission are added without duplicating ordinary Bevy direct illumination.
“Sky” in the code can mean two different things: deterministic direct sunlight,
or stochastic diffuse environment radiance. The latter belongs to path transport.

## Shared sunlight, separate estimator

A directional visibility query has a known ray direction. The cache interns
integer corner positions; one ray per corner and visible sun projects onto all
receiving normals. The canonical origin is displaced toward the source, not
toward a particular face normal. Grid-plane ties are consistent on CPU and GPU.

`SkyCache` persists across batches and uses spatial buckets to invalidate corners
intersecting changed geometry's shadow sweeps, including distant receivers.
Sun direction/color changes invalidate visibility; camera motion, display toggles,
points and environment changes do not. Retired receiver corners are pruned.
Unknown corners retry; packed zero means unknown and packed one means measured
black. Four RGBE corner colors per face preserve the 208-byte display probe.

Sky jobs share the indirect worker/submission, geometry and scratch buffers.
`ProbeWork::sky_requested` records explicit sunlight demand. Indirect refinement
requests missing corners, but does not repeatedly look up or copy measured sun
values. Geometry and sun changes use the existing urgent/global sky queues;
overlapping requests attach to the same receiver job in linear time. Publication
ignores unrequested sky output rather than interpreting zeroed scratch as dark.
Sky-only receivers carry no transport faces, cannot alter simulation history,
and submit no GPU work when all corners are cached. Deterministic sky changes
bypass the normal display upload interval. Separate eligibility is useful here:
one deterministic visibility measurement and a noisy hemisphere integral do not
have the same convergence or invalidation requirements.

## Revisions and history

Freshness, retained statistical weight, scheduling urgency and displayed values
are independent. A revision can make an estimate stale without clearing its
display, rewinding its sequence, or resetting the world sweep.

| Change | Retained statistical weight on refresh |
| --- | --- |
| Unchanged transport | Accumulate up to 65,536 |
| Geometry edit/load/unload within 48 voxels | At most 16 |
| More distant tracked geometry change | At most 256 |
| Incomplete geometry transition history | At most 64, or a stricter known local cap |
| Nearby moving analytic source / global source change | At most 4 |
| Tracked remote-only analytic changes | At most 64 |
| Incomplete source transition history | At most 4 |
| Discrete voxel-emitter or material-table change | Reset obsolete measurements to zero |
| New face | No measurements; optional display-only seed |

Geometry and sources use the same eight-transition `ChangeHistory` mechanism,
with different policies over their deltas. Every intervening retained transition
is consulted. The journals keep metadata, not scene snapshots or old rays.
Canonical source ordering prevents ECS order changes from causing relights;
point-source deltas include both old and new influence volumes but exclude
unchanged lights. Distance affects responsiveness, never visibility or eligibility.

Material/emitter resets differ from ordinary visibility changes for a reason:
otherwise an extinguished source survives as old energy diluted over subsequent
batches. Both still preserve temporal display state. No rule tests a house,
tunnel, skylight or gameplay material ID inside the estimator.

Scene publication preserves compatible faces, introduces new faces without
inventing measurements, and publishes at most four atomic pages within a nominal
2 ms budget. A single expensive page can exceed that budget. Existing BVW event
coalescing remains 40 ms for edits / 350 ms for streaming, with a one-second
overdue bound. The audit adds no debounce.

## Scheduling and asynchronous publication

Eligibility is per face. Current well-measured faces rest after four stable
batches, low per-band error/drift and settled display. The default minimum is
64 samples; very dark indirect RGB requires 256 so an initial run of zero-valued
paths is not mistaken for proof of darkness. All bands participate in uncertainty.
The relative-error calculation is a heuristic over correlated, bounded samples,
not a confidence interval or certified convergence test.

Settled faces remain eligible for two-second audits. Simulation requests and
changes wake them. Bounded candidate searches combine urgent/bootstrap work,
nearby rotating windows and a reserved one-quarter global sweep. The camera is
a priority hint, never the extent of the lighting field. Empty/settled candidates
do not consume the ray allowance or collapse the scheduler budget.

Only selected measurements cross the worker boundary in `TraceWork`; topology
indices, temporal display and scheduler state stay authoritative on the main
world. Workers own scene/source snapshots and perform preparation, submission,
waiting and decoding asynchronously. The CPU fallback pool is created lazily and
limited to two threads. One outstanding trace batch bounds resource ownership.

Publication rejects incompatible receiver geometry and obsolete discrete-source
resets. Moving-light results can be accepted as stale to avoid perpetual starvation;
their revision metadata still prevents simulations from treating them as current.
Sky additionally requires matching scene and sun generations. Accepted selected
faces update in place, preserving untouched faces and live topology indices.

## Temporal and spatial display

Temporal filtering runs on CPU result merge, once per updated face. It blends
that face's measured indirect RGB with its own previous display. The base
time response is `1 - exp(-elapsed / response_seconds)`. Sample evidence and
visible-band noise further limit the blend; noisy dark estimates are deliberately
slow. Brightness-compressed weights suppress isolated HDR jumps without imposing
a permanent brightness ceiling. Confidence weights evolve with the filtered
value. This is biased presentation, not a modified physical estimator.

Newly exposed faces can borrow low-confidence display from measured, compatible
pre-edit neighbors connected through known air. Synthetic seeds never become
donors in the same publication and never increment measurements or moments.
Placement can inherit the surface it covered. Unknown sun starts dark rather
than switching temporarily to a shadow map. A shader-only prior bridges the
short mesh/cache publication gap; it also never becomes transport evidence.

At render time the fragment shader reconstructs each virtual unit-face corner
from up to four confidence-weighted coplanar faces. Missing edge faces exclude
disconnected diagonals. It then bilinearly interpolates the four values across
the face. Greedy mesh geometry is unchanged. Direct sun interpolates actual
corner measurements; indirect light interpolates reconstructed corners.

## Display-cache lifetime

`display.rs` coordinates snapshots and sparse patches; `render.rs` owns buffer
layouts and uploads. Snapshot packing is incremental, then encoding runs off
thread. The existing cache stays usable during growth. Installation catches up
measurements changed since the snapshot in the same frame.

Each generation owns fresh hash/probe buffer identities and its own shared patch
queue. Extracted older frames retain their generation. Hash and value patches
preflight together and upload under one lock; a failed bounds check retains both
for retry. Topology updates include values on existing newly exposed faces, so
the value-upload timer cannot split the transaction. The shader's address bound
is allocated capacity, not current occupancy.

Buffers and indices retain high-water capacity. GPU geometry is persistent and
patched; ordinary lighting updates do not rebuild geometry or the entire display
hash. Display disabled with F4 is only an A/B presentation switch: simulation
transport intentionally continues.

## Simulation API

Schedule readers after `VoxelGiSystems`. `VoxelEnergyField::estimate(voxel, normal)`
returns raw direct plus indirect energy, variance, minimum effective samples,
oldest contributing face age, and source/transport freshness. The query combines
axis-aligned faces with squared normal weights. It returns `None` for unpublished,
missing, buried or unmeasured required faces. Retained stale measurements can be
returned with freshness false; they are not silently made current by display seeds.

Use `request_refresh(voxel)` for bounded asynchronous priority, then inspect the
returned estimate later. `sample` is a convenience wrapper that discards metadata;
simulation systems should prefer `estimate`. No refresh request promises a
deadline. Variance is a second-moment diagnostic, not a calibrated error bound.

`EnergyEmitter` augments a Bevy directional/point light or acts as a transformed
point source alone. `VoxelMaterialTransportTable` supplies per-band diffuse
response and emission; inputs are sanitized and reflectance is bounded to one.
`VoxelGiEnvironment` makes the background energy host-configurable. Rendering
never writes filtered RGB back into these measurements.

### Planned tick based simulation delivery

Future GPU-resident accumulation should separate visual updates from CPU
simulation snapshots. Simulation may consume the field at a configurable tick
cadence rather than every rendered frame; no fixed tick rate is assumed.
This reduces required readback frequency, not the accuracy required of each
measurement. The existing CPU-owned field and asynchronous query API remain the
implemented production design. An isolated prototype validates resident
accumulation, sparse snapshots, slot incarnation/source epoch checks, task
reordering and buffer growth that preserves measurements. This is not yet a
production world allocator or tick-delivery implementation.

Snapshots must identify their geometry/source revisions and measurement ages,
retain all five bands and sample/variance metadata, and distinguish unknown or
stale results from valid zero energy. A tick can reuse a sufficiently fresh
measurement or request a bounded priority refresh without blocking rendering.
Topology changes must not let a snapshot name a recycled receiver as the old
surface. Readbacks should be bounded and driven by simulation demand, with an
explicit policy for delayed data. Temporal display RGB is never a simulation
input. Integrating heating or dose over tick duration belongs to the host
simulation, as do thresholds and material changes that become transport updates.

The resident prototype is a measured direction to pursue, not ready-to-merge
storage. Integration needs stable receiver slots with generation-checked recycling,
bounded in-flight work, GPU-side display publication independent of CPU snapshots,
and per-face sample timestamps rather than assigning readback time to old data.
CPU task selection still needs compact scheduling/freshness feedback. Initially
retain the existing explicit-path estimator and reference/fallback; do not combine
residency with recursive cache feedback or path guiding.

## Integration contract and limits

This is one axis-aligned unit-voxel world in integer coordinates, with f32 ray
positions. There is no floating origin, world-ID key, deformation metric or
cross-world ray routing yet. BVW-specific storage/generator adaptation is isolated
in `bvw.rs`; the renderer still uses BVW's vertex layout and a world-position hash.
`VoxelGiPlugin` accepts a host-owned texture handle; the host prepares its array,
palette and reloads. `VoxelGiDiagnosticsPlugin` is optional demo UI, not core input.

For an asteroid or planet, first define canonical local coordinates, world identity,
local-to-world transforms, ray distance units, normals and source conversion.
Nonlinear warps need a correct traversal metric or another acceleration structure;
warping only raster vertices is not sufficient. A point sun is supported, but its
directional corner-cache equivalent is not: point direct RGB still uses Bevy PBR,
and transport evaluates the point source at each receiver/bounce.

Not implemented: transmissive/specular GI, non-voxel dynamic occluders, arbitrary
area/spot-light components, media, spectral physics, or physically calibrated
thermal/radiation simulation. Analytic-light cost is linear in light count.
Indirect invalidation remains a conservative global wake-up; there is no per-path
dependency tracking. Monolithic display buffers need device-limit-aware paging
before substantially larger worlds. Current buffer validation prevents stale
generation writes, not unlimited allocation or out-of-memory failure.

The next architecture changes should preserve these boundaries: a replaceable
sampling distribution, optional GPU-resident accumulation with explicit simulation
readback demand, and a host-independent scene-input API. None requires abandoning
the current traversal, revision or presentation contracts.
