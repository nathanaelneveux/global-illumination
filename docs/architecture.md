# Architecture

This is the current design. Historical experiments are kept separately so the
implementation can be understood without reconstructing its development history.
The planned direction (near/far lighting, the simulation split) is in
[direction](direction.md). Every visible face is view-owned: a per-view tier
discovers faces after the depth/normal prepass and traces them with the shared
estimator. A persistent world cache (`cache_radius`, default 64 voxels; 0 is
cacheless) keeps receivers for pages near the cameras, terminates secondary
rays, measures edit flux and exchanges RGB history with visible faces.
Geometry stays every loaded page. Native transport contracts are in
`gpu/view_tests.rs` and `gpu/view/cache_tests.rs`; integration evidence is in
[the runtime checkpoint](history/far-view-runtime.md) and
[the ownership comparison](history/option-comparison-2026-10-23.md).
The numerical model and the presentation policy are deliberately separate.

## Data flow and ownership

```text
host chunks / edits / sources / material table
                   |
    scene worker: immutable pages, NAADFs, shared region-change lineage
                   |
    main world: bounded receiver publication + revision journals
                   |
    scheduler -> compact worker jobs -> resident GPU measurements
         ^                              /                  \
    validated async snapshots <--------+           GPU temporal RGB
         |                                                 |
    CPU diagnostics/progress                     generation-checked rendering
                                                           |
                                          connected virtual-corner reconstruction
```

The runtime uses resident GPU accumulation; the independent CPU tracer remains
a test oracle. Rendering does not wait for CPU delivery of lighting statistics.

| Responsibility | Owner / source |
| --- | --- |
| Plugin wiring and runtime state | `lib.rs` |
| Host input: pages, edits, voxel source | `input.rs`, `pages.rs` |
| BVW events, edits and source (connector crate) | `voxel_gi_bvw/src/lib.rs` |
| Scene/receiver representation | `world.rs`, `field.rs` |
| Acceleration, packing and shared GPU transactions | `naadf.rs`, `gpu_scene.rs`, `gpu/geometry.rs` |
| Change metadata and retention | `world/delta.rs`, `geometry.rs`, `retention.rs`, `sources.rs`, `revision.rs` |
| Receiver page publication | `publication.rs` |
| Eligibility, priority and dispatch | `scheduling.rs`, `dispatch.rs` |
| Construction policy; model/work configuration and revision synchronization | `startup.rs`, `settings.rs` |
| Thread-transfer payload and result merge | `work.rs` |
| Source model and Bevy light adaptation | `sources.rs` |
| Sampling, emitters and transport | `sampling.rs`, `emission.rs`, `tracer.rs`, `gpu.rs`, shared `voxel_path.wgsl`, face consumer `voxel_trace.wgsl` |
| Surface-cache face lookup | `gpu/face_index.rs` |
| View-owned far receivers, scheduling and asynchronous timing | `gpu/view/runtime.rs`, `gpu/view/timing.rs`, `voxel_view_runtime.wgsl` |
| Bidirectional RGB face ownership (two-way cache) | `voxel_view_cache.wgsl`, `gpu/view/cache_tests.rs` |
| Stable GPU ownership, accumulation and bounded snapshots | `gpu/resident.rs`, `gpu/readback.rs`, `voxel_accumulate.wgsl` |
| Shared uncertainty heuristic | `statistics.rs` |
| Direct-sun visibility cache | `sky.rs`, `sky_cache.rs` |
| Temporal display and initialization | `history.rs` |
| Material data and shared buffers | `render.rs` |
| BVW mesh material (connector crate) | `voxel_gi_bvw/src/material.rs`, `voxel_gi_bvw/shaders/voxel_transport.wgsl` |
| Accounting, test fixtures | `telemetry.rs`, `research.rs` |
| HUD and tuning panel (debug crate) | `voxel_gi_debug/src/lib.rs` |
| Validation harness (lab) | `src/validation.rs`, `src/validation/` |

Unqualified paths are in the core crate: Rust under `crates/voxel_gi/src/`,
shaders under `crates/voxel_gi/shaders/` (embedded with `include_str!`). `reconstruction.rs` is an independent CPU test oracle for shader
interpolation, not a second production reconstruction path.

### View-owned lighting

After Bevy's depth/normal prepass and before opaque rendering, each active view
discovers voxel-face identities using a reduced spatial grid (nominal four-pixel
cells, at most 65,536 receiver slots). Physical receiver slots are independent of grid coordinates. Each 2x2-cell tile
indexes a linked list of projected receivers and can borrow unused slots from
anywhere in the view. A preparation dispatch validates old witnesses, retains
surviving slots and completes the free list before discovery allocates from it.
An index of the retained identities prevents a tile from recreating a face that
is already represented elsewhere. Preparation inserts only canonical previous
owners: their keys are unique, so CAS insertion does not need to inspect any
concurrently initialized current receiver. Discovery reads this index after the
preparation dispatch completes. Newly allocated keys are indexed only after
discovery finishes; no workgroup spins on another group's partially published data.

Discovery retains useful coarse coverage and scans full-resolution footprints
for missing planes under the same world-distance gate as presentation. There is
no four-receiver tile quota. Total pool exhaustion is bounded and explicitly
counted, not an alias or an out-of-bounds write. Quantized normals select their
dominant axis; depth is snapped to the voxel plane before the geometry identity
check. Duplicate pixels share one face owner. A ping-pong hash reprojects the exact same face, its raw
mean/second moment, sequence, retained path and any unfinished continuation.
No filtered image is ever treated as a new transport measurement.

Discovery selection is memoized separately from lighting. Each slot retains a
face-local UNORM16x2 point that was actually visible; the voxel centre may be
hidden on a partially occluded or grazing face. Camera motion projects that fixed
world point and checks current depth, normal, voxel identity and geometry. Valid
selections follow their world points across tile boundaries; a bin crossing alone
does not force rediscovery. Lost witnesses trigger immediate discovery. Vacant
centres check for newly exposed terrain. A viewport, overlap or MSAA change
disables reuse. Static tiles additionally fingerprint their depth/normal footprint.
Footprint searches are staggered over four frames, so a new thin plane can wait
three intervening frames. Missing transport geometry forces retries independently
of this cadence. Full scans preserve valid selections instead of reanchoring them
at screen centres. A 64-frame refresh bounds fingerprint-collision lifetime.
World edits and all existing receivers' lighting stages continue every frame.
The two pool/index buffers round to 4 MiB at 1440p, versus the earlier 1-MiB
tile metadata, with another 384 KiB in the published spatial index. Receiver/path
capacity and transport budgets are unchanged. All metadata stays on the GPU.
Use the raster coverage mask, not the selected-owner counter, to measure visible
holes; global capacity and raster/subpixel visibility still bound coverage.

Each owner advances one path at a time, at most three intersections per wave.
Long paths remain pending; their partial energy is not published. Finishing
black paths count; unknown paths advance the sequence without adding a sample.
There is no fixed bounce cutoff. Exact-identity reuse needs no shifted-path
Jacobian; cross-face path reconnection remains a separate, unimplemented idea.

Shared retention packets are copied GPU-to-GPU after change measurement and
consumed by every active view. They preserve means, cap weights through the same
distance/uncertainty rule as near receivers, and discard affected unfinished
attempts. A view absent for more than a frame loses its history rather than
silently missing events. Near residency is driven only by active cameras;
inactive terrain-loading cameras do not allocate lighting receivers.

Display reconstruction borrows unique coplanar owners within a bounded screen
and world radius; it does not alter raw moments. Both reconstruction and raster
upsampling use the spatial index, never physical slot coordinates. Exact faces
are hash-addressed; otherwise raster upsampling searches the four closest bins,
including the containing bin, with the same coplanarity and 16-voxel gates. Donor weights use
actual world-space positions, not reassigned screen-slot coordinates; camera
motion alone cannot change those weights for a fixed receiver/donor set.
This is presentation smoothing,
not NAADF-style spatial path resampling. Sun visibility is evaluated at the
same four canonical voxel corners as the near cache. A face the view does not
own takes each corner from an owned coplanar face sharing that corner (the
corner's visibility is identical there) and traces only corners none share; it
never borrows a farther face's corners, which used to shadow small distant
faces wrongly at particular camera poses. Emission is
looked up per fragment so a subpixel emitter does not disappear.

The material's world-position interpolant uses perspective-correct **centroid**
sampling. With MSAA, a primitive can cover samples without covering the pixel
centre; extrapolating world position there can select an unrelated or buried
voxel and produce a stable dark speck that changes with camera angle. Centroid
keeps interpolation on the covered primitive without sample-rate fragment
shading, extra transport rays or larger receiver storage. This is a raster/face
identity contract, not temporal smoothing. Exact floating-point grid-edge ties
and discovery of faces absent from prepass sample zero remain separate issues.
See the [WGSL interpolation rules](https://www.w3.org/TR/WGSL/#interpolation).

Per-camera state stays on the GPU. The material reads a shared output scratch
buffer overwritten in camera encoder order, with validity cleared before each
camera. Immutable optical-table generations and coherent geometry/cache buffer
pairs are shared with the worker. Only 224-byte asynchronous timestamp/counter
receipts return to the CPU; callbacks are polled without waiting. The view tier
receives 75% of the existing transport time share and the world cache
`cache_share` (25%). Prepass, discovery,
corner sunlight and reconstruction are separately measured presentation costs,
not hidden in the transport quota. Preparation and scanning have separate
timestamps and their sum is the comparable discovery cost. State is bounded by
viewport capacity, not loaded world size.

### Two-way face cache

The persistent world cache (`cache_radius` admission, 32-voxel retirement
hysteresis) exchanges exact-face RGB history with visible view owners. The
exchange does not expand residency or share paths between different faces.

A resident `(voxel, face, material, incarnation)` record is the single raw RGB
authority. Before view tracing, a matching visible owner loads that record.
On first admission only, a better-supported valid view estimate can replace a
cold resident estimate. Their weights are never added: the two histories can
contain the same path prefix. Once admitted, the resident value wins even when
an edit has reduced its weight. After tracing, the view writes updated raw
mean, second moment and effective history weight back to the same record. The next worker
dispatch or newly revealed view therefore resumes the shared history.
An independent unfinished view attempt is discarded when importing a different
authority's prefix: it may duplicate that prefix and owns no reservation there.
This is distinct from preserving already-admitted tails across background work.

The display value is copied separately as a warm-start prior. Reconstructed
neighbour support never enters raw moments or the physical count, and only raw
energy can terminate a secondary ray. Presentation-only copies do not refresh
scheduler ranks or measurement freshness. Pending change estimates continue to
exclude stale raw values from bounce termination until measured.

Both schedulers consume one advancing sample sequence. A view reserves its
dispatch's potential new attempts; an unfinished path keeps its original index
while background work consumes later indices. Trace/store return the unstarted
reservation suffix before the command buffer ends. Keeping unused indices
would introduce path-length-dependent gaps into the low-discrepancy sequence;
restarting unfinished paths would preferentially discard long contributions.
Only complete, known paths advance measured evidence; new-face initialization
may additionally contribute the single pseudo-count described below. The reservation scratch
adds at most 256 KiB per camera, not another persistent face-history allocation.

Load, trace and store are separate passes in one command buffer on the shared
GPU queue: no pass reads bounce data while writing it. Worker maintenance
revokes a small GPU epoch guard before changing geometry, material identity,
retention or slot allocations. Held render descriptors cannot write afterward;
new descriptors become available after maintenance submission. A temporary stale
descriptor preserves its admission provenance so it cannot resurrect pre-edit
history when current bindings arrive. Genuine eviction clears that provenance.
The shared mode uses GPU retention as authority, not CPU checkpoint rollback,
which could overwrite newer view samples. Ordinary nonshared rollback is unchanged.

There is no new lighting readback, CPU merge, or player-thread wait. Existing
asynchronous timing receipts include exchange cost and load/admission counters.
Cacheless faces retain the existing bounded, visibility-dependent view history.
This does not promise persistent lighting for all distant/offscreen surfaces.
Native tests in `gpu/view/cache_tests.rs` cover ownership, retention, generations,
sequence allocation and analytic RGB energy; rendered qualification is recorded
in [the experiment](history/two-way-cache.md).

### Raw topology bootstrap for new view faces

After retention and cache admission, a new face with zero history and an
unstarted path sequence can inherit a **one-weight raw prior**. Matching normals
one voxel away take precedence. If none are supported, old faces on the removed
voxel, faces covered by placement, and faces bordering the new exterior air cell
provide an orientation-weighted estimate. Nonparallel donors get one quarter of
the weight of aligned donors. Lateral connections must be loaded air, not
unknown space. A published raster face can provisionally establish its own
exterior while the transport model catches up. This is approximate incident irradiance, not a
claim of conserving flux or inferring a full directional distribution.

Donors are last-frame raw view records or resident raw records with at least 16
effective samples. Filtered display RGB is never used. The mean and second moment
are averaged, but the new face gets only one unit of effective history regardless
of donor count. Its QMC sequence stays unchanged. The next N known samples reduce
the prior's influence to 1/(N+1), unless ordinary retention removes it sooner.
Neither pure priors nor fewer than 15 fresh samples plus a prior meet the existing
16-weight transport-cache/donor threshold. This bounds immediate seed-to-seed
propagation; it does not make the prior unbiased or its donor support independent.

Initialization is fused into the existing cache-load pass, before tracing, with
no extra buffers, GPU readback or CPU work. The first display uses the same raw
prior. The two-way authority transfers it, never adds duplicate copies. Existing
sampled faces capped to zero are not re-seeded. This applies to view receivers
(including +cache handoff), not an overhaul of world-only slot initialization.
It cannot initialize an undiscovered face or recover a retired donor absent from
both caches. Standalone raw-prefix
sampler tests do not run this view initialization; a seeded history weight is
not an independent-path count for accuracy calibration.

View-first discovery therefore distinguishes raster-visible identity from
transport-ready identity. The material identity's high bit marks a pending face
when the raster prepass and transport exposure lookup disagree. Such faces keep
their raw prior (or an existing exact face's history) and display record, but
are excluded from the ray queue, resident admission and cache writes. In-flight
old-scene paths and resident leases are discarded. Every retained witness is
reclassified against the current scene each frame; once exposure agrees, the
same record becomes traceable without a black restart. A truly occluded face
still loses its witness normally.
This addresses publication skew without blocking meshing, tracing stale
geometry, adding a screen-color fallback, or extending receiver storage.

## Geometry and traversal

Host terrain generation now composes reusable [structure stamps](structures.md)
for the house and nearby cave. These are generation data, not GI-specific volumes:
BVW generation and direct geometry fallback query the same deterministic plan,
and player edits remain above it. The transport core has no structure identity
or shape special cases.

The host exports immutable dense 32³ pages of material IDs. A page contains eight
16³ regions; each region contains 4³ bricks of 4³ voxels. Material IDs are 0–255;
air is the distinct value 256. Regions that are unchanged retain `Arc` identity.
Empty and uniform regions avoid dense voxel storage.
Uniform empty brick fields use exact distances to the volume edges; fully blocked
fields are zero. Only mixed fields require iterative neighbor merging.

### Host input contract

The prototype opts into `VoxelGiDirty::tracked()`. Its player and validation
systems write through `VoxelGiDirty::set_voxel`, forwarding the same value to BVW
and a sparse, last-write-wins GI overlay. All writers must use this gateway or
explicitly report unknown changes through `mark_voxel` / `mark_all_loaded`.
Coarse BVW chunk messages cannot detect an unreported writer in tracked mode.
`VoxelGiDirty::default()` retains conservative observer mode for integrations
that cannot make this complete-tracking guarantee.

Normal scene construction reads one immutable BVW chunk snapshot per dirty page
(or the deterministic generator if unavailable), then applies that page's edit
overlay. Workers hold copy-on-write overlay snapshots: later writes cannot alter
an in-flight input. This removes per-cell live host lookups from tracked edits;
it does not claim one atomic snapshot of the entire host world. Unknown updates
discard the affected overlay and use the general live resampling fallback.
A later unknown notification supersedes earlier exact values, including within
one frame; subsequent tracked writes can supply newer values again.

Overlays survive unload/reload so an older mesh/chunk snapshot cannot erase a
newer edit. Repeated writes coalesce, but acknowledged edits are not automatically
retired: memory grows with distinct edited positions, including unloaded pages.
Retiring them safely requires an authoritative host acknowledgement/version
contract. This adapter does not modify BVW or provide persistence for game saves.

### Incremental geometry and shared change discovery

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
The worker stages deltas privately, then applies every directory/header/cell
copy in one GPU command buffer. Render submissions cannot interleave those
copies. When any buffer grows, the directory mask changes, or traversal bounds
change, **all three allocations detach together**. Unchanged contents are copied
GPU-to-GPU before sparse patches; old extracted materials retain their coherent
buffers and bounds until replacement. Ordinary edits keep buffer identities and
transfer only changed ranges. These are coherent allocation generations, not
immutable scene snapshots: readers of the current generation observe an entire
transaction before or after its submission.

Geometry synchronization and receiver retirement run on the worker independently
of ray allowance, pending snapshots and near-receiver count. A far-only camera
still gets current geometry. Retirement updates both GPU lookup and live metadata
without emitting rays, allocating a readback or advancing sampling batch IDs.
An unchanged far-only world goes idle. Mandatory lifecycle submissions are not
sampling work; they must not be mistaken for zero-cost GI or used as evidence
that the future far tier fits the shared ray budget.

Top-level bounds include loaded regions, but do not allocate an arbitrarily large
box between disconnected islands: oversized domains fall back to region DDA.
The top-level field depends only on region residency and empty/nonempty occupancy.
Immutable snapshots share its allocation across edits that preserve that topology;
changed voxel regions still trigger normal transport invalidation.
Rays normalize direction, offset from surfaces, traverse at most 2,048 steps and
768 voxel units per segment by default, and distinguish hits, escape and unknown traversal.
These finite-domain/depth limits are explicit approximations, not an exact
infinite-domain light solver. Nonzero paths are never discarded solely because
throughput is small: the next radiance can be large.
Only exact-zero termination and compensated Russian roulette are retained; the
old 0.002 cutoff failed the independent estimator audit.
GPU empty-space box stepping uses vector operations with the scalar reference's
ordered XYZ tie and epsilon policy. This is an execution optimization, not a new
intersection rule, shorter traversal limit or change to the path estimator.

Transport treats a missing region before occupied loaded regions along a ray as
unknown and contributes no energy for that path. Air-only frontiers do not create
an occluder. Display sun rays instead test all loaded geometry across gaps, so
streaming holes do not cast invented dark slabs. Neither policy sees unloaded
geometry. These are explicit query semantics over the same traversal structure,
not two unrelated voxel engines.

Scene construction records actual changed region keys once. A bounded journal of
eight transitions lets geometry invalidation, sunlight, emitter dependencies and
GPU packing discover changes even when they skip scene versions. It stores keys
and identity stamps, **not old voxel payloads**. Consumers filter endpoint region
identities; expired, forked or unknown lineage falls back to a complete comparison.
This discovery journal is separate from the receiver history-retention journal
and does not change sample weights or invalidation distance rules.

`SceneChanges` compares material IDs only within the discovered regions and
computes tight changed-voxel bounds in one worker comparison. Region load/unload
uses full region bounds because residency changes knowledge of empty space too.
Air-only changes outside previous occupied
bounds need bookkeeping but do not invalidate existing indirect transport; holes
inside those bounds remain conservative invalidations. Emitting, reflecting and
occluding voxels use this same path. There is no emission classification or
global source reset attached to a geometry delta. The worker's emitter
distribution is keyed by immutable scene/material snapshots, independently of
the analytic-source revision.

The optimization removes repeated whole-world **discovery**, not every world-sized
operation. Scene construction still clones the region map; occupancy/residency
changes can rebuild the top-level field. Packing then refreshes all affected
top-level headers, including unchanged regions whose skip bounds changed. Sunlight
still checks directional dependencies across receiver buckets: a distant blocker
must not be discarded merely because indirect retention has a distance limit.
Receiver influence planning and residency-directory rebuilds remain explicit costs.
History binning chooses between enumerating nearby page coordinates and scanning
resident pages, using estimated work before allocation. Local enumeration checks
actual occupied residency, memoizes overlapping occupancy lookups, and preserves
the exact change bounds and page distances. Large brushes, sparse/disconnected
worlds and saturated distance ranges retain the resident-scan fallback. This is
only a broad-phase optimization, not a shorter lighting or retention radius.

## Receiver and transport model

Each exposed solid voxel owns a six-face ambient cube. Only exposed faces are
sampled. Each face stores a mean, second raw moment, direct term, effective history
weight, independent sample sequence, timestamps and freshness/stability state.
The index owns boxed probes so growing a hash table does not relocate all large
receiver records on the game thread. Its 64 independently growing partitions
also avoid rehashing the entire loaded-world index in one insertion. This trades
a small lookup cost for smaller growth spikes, not a hard latency guarantee.
Scheduling retains its explicit key order independently of partition iteration.
Stable logical keys are voxel coordinates, not mesh vertices or array indices.

Each selected face receives 16 paths per normal dispatch; the tuning range is
1–64, with no default promotion of the slower-to-refresh 32-path or noisier
eight-path alternatives. Rotated Halton
dimensions preserve the tested cosine-hemisphere and roulette sequences. Emitter
selection and emitter surface coordinates use independently padded 2D Sobol pairs
with index shuffling and hash-based Owen scrambling. CPU and GPU use the same
integer operations; the second Sobol dimension uses a Pascal-matrix butterfly,
so no direction tables or per-receiver guide are needed. Sequences advance
independently of retained history, so invalidation does not repeat early rays.
The GPU uses eight lanes per face at 16 or more paths, and four for smaller
batches; the serial entry remains an opt-in reference. Lanes own disjoint paths,
not reused samples. This layout change preserves all five means and moments.
Dispatch tasks are encoded directly into a reusable 32-byte-per-task upload
array, with explicit little-endian words and initialized padding. This removes
the intermediate typed array without changing the shader ABI. Scratch capacity
tracks the largest dispatch, not the number of loaded receivers; no path history
is added. A test-only generic encoder is the byte-layout oracle.
The [experiment results](history/experiments.md) explain why Sobol is restricted to
emitter dimensions rather than replacing every path dimension.

### Surface radiance cache

Each resident face stores its measured incident energy: emitter NEE plus the
continued path. A path that hits a resident face with at least 16 samples ends
there with `emission * MIS + reflectance * (direct at the hit + that face's
cached energy)`. The cached value already contains the face's own NEE and
continuation, so nothing is counted twice. Bounces accumulate through the cache
over successive updates, which gives infinite-bounce transport. Most paths
cost one cosine ray plus shadow rays. `gpu/face_index.rs` keeps a GPU
`(voxel, face) -> slot` table beside the worker's slot map. Uncached hits
(cold start, faces never scheduled, the synchronous and CPU backends) continue
explicitly as described below. Rays that reach unloaded regions carry no
information: they are excluded from a face's sample count, not averaged in as
black. Evidence and limits: [surface cache](surface-cache.md).

Explicit paths evaluate up to three diffuse bounces, analytic visibility, voxel emission,
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
against current geometry. Emitter visibility connects the receiver and light's
outward-offset endpoints, recomputing its direction and length. The physical
unoffset points still define the PDF and geometry factors. Translating the
origin while keeping the old direction and a fixed length trim falsely shadows
grazing light samples; this is regression-covered by an analytic emitting plane.
This follows the endpoint construction in
[PBRT's ray spawning](https://pbr-book.org/4ed/Shapes/Managing_Rounding_Error#SpawnRayTo),
while retaining our voxel-scale epsilon, not PBRT's full interval error bounds.
Source weighting is RGB-only by default. The legacy importance setting can
include infrared/ionizing for research controls, not gameplay simulation.
The tree shares the emitter storage binding,
adds 64 bytes per node and rebuilds with the worker's emitter snapshot; there is
no per-receiver allocation or incremental tree refit. See the
[many-emitter results and tradeoffs](history/research-2026-10-09.md#spatial-light-hierarchy).

The last diffuse hit has no continuing cosine path to balance an ordinary
light-only estimate. Instead it draws one emission connection from a 50/50
light/cosine mixture and evaluates the same balance density for either branch.
This preserves the finite-bounce emission integral while bounding each terminal
connection by twice the emitter's energy, without clipping or adding another
connection ray. Sky escapes and nonemitting hits contribute zero to this terminal
term; it does not extend the environment bounce count. CPU and GPU use the same
unused terminal path dimensions. See the [cave accuracy study](history/cave-history-retention.md)
for the outlier diagnosis, native cost and independent-reference checks.

The core stores normalized energy. It does not attach physical watts, temperature
or dose to those values. The environment gradient uses world Y; it must become a
host-defined directional environment before a planet-local-up integration.

Ordinary Bevy direct RGB stays in PBR (or the explicit cached-sun override).
Face-center direct energy supports edit-flux measurement, and direct lighting
is evaluated at bounce hits. Indirect RGB, custom-emitter direct RGB and voxel
self-emission are added without duplicating ordinary Bevy direct illumination.
“Sky” in the code can mean two different things: deterministic direct sunlight,
or stochastic diffuse environment radiance. The latter belongs to path transport.

## Shared sunlight, separate estimator

A directional visibility query has a known ray direction. The cache interns
integer corner positions; one ray per corner and visibly colored sun (at most
four) records whether that sun is visible. The canonical origin is displaced
toward the source, not toward a particular face normal. Grid-plane ties are
consistent on CPU and GPU.

A corner stores a visibility word, not energy: zero is unknown, otherwise a
known bit plus one lit bit per sun (source order of the visibly colored suns;
CPU and GPU count the same way). The display shader applies each sun's current
direction and transport color (`GpuTransportSettings` sun fields, kept by
`synchronize_sun_display`) with the face's axis normal, so a sun that moves or
changes color relights every frame without a retrace. Four corner words per
face preserve the 208-byte display probe.

`SkyCache` persists across batches and uses spatial buckets to invalidate corners
intersecting changed geometry's shadow sweeps, including distant receivers.
Visibility generations are keyed by a cube-projected grid of sun directions
with ~0.5° cells (`visibility_key`): a sun leaving its cell invalidates
visibility (corners keep showing the previous word until retraced); color
changes, motion within a cell, camera motion, display toggles, points and
environment changes do not. Retired receiver corners are pruned. Unknown
corners retry. A corner whose lit bits flip records a Light change (luminance
at the current sun), which measured retention spreads to nearby faces: an
opening far above a floor reaches the floor, and a moving shadow border caps
the bounce history it changes. A flip's flux is the reflected change (times
the most reflective solid voxel at the corner), as edit flux is measured;
while the sun moves (a visibility step in the last 10 s) flips merge in their
buckets and publish at most every 250 ms, while under a still sun they publish
at once. A gradually moving sun is not a source reset (`SourceChange::Sun`, see
below); the day-cycle validation profile measures it.

Sky requests come in three tiers: explicit requests (new probes), then the
band, then the world sweep. The band holds the voxels touching corners whose
lit bits flipped in the previous visibility generation and their axis
neighbours (shadow borders move about a voxel per 0.5° step), and the voxels
touching corners that changed geometry invalidated (shadows near an edit and
under new terrain). The worker sorts each band nearest the batch focus; bands
accumulate on the main thread, bounded, with the sweep as fallback. The
publication path does not scan an edit's neighbourhood for sky.

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

On the surface-cache branch the default is **measured retention**
(`measured_retention` = 1; see [retention research](retention-research.md)).
Each change event carries a flux bound. A face whose MSE-optimal history
`c sigma^2 / D^2` (with `D = flux / (pi d^2)`) would bind keeps that much. The
face is not read as a cache source for one batch, and uses the fresh batch's
noise if it was dark. Changes of unknown flux keep the distance curve below.
`config/gi-retention-curve.ron` restores the curve everywhere.

Retention is applied **on the GPU, to every resident face in range, when the
change publishes** (`apply_events` in `voxel_accumulate.wgsl`; `retention.rs`
`resolve` is its CPU statement, checked face by face by a native test). Each
event is its deduplicated change bounds with their flux and their union box (an
early out); every face in reach sums each bound's exact change. (A single
aggregate over a whole streaming event, its total flux at the union's
distance, overstated changes enough to double the hallway's median settle
time: measured retention cut history throughout initial streaming.) A
curve cap lands in the face's GPU count at once. A known measured change
accumulates per slot; when the face is next traced, `mark_changed` decides
from its current GPU statistics whether it binds (or the face was dark), and
accumulate applies it. Nothing waits for a face's next CPU selection, so there
are no per-page journals, eviction summaries or worker-side snapshot queries.
The CPU keeps one stamp per receiver page, the newest event there: receivers
not traced since are due for an audit, and publication restarts their
stability. Since every face is covered, measured retention now also replaces
the curve cap formerly applied to a selected probe's unselected faces.

Freshness, retained statistical weight, scheduling urgency and displayed values
are independent. A revision can make an estimate stale without clearing its
display, rewinding its sequence, or resetting the world sweep.

The default geometry/changed-sunlight policy is the selected
[whole-range equation](history/whole-range-history.md): `H(d) = round((d / 12)^4)`, applied
as `min(existing_weight, H(d))`. Distance is receiver-to-change distance. The
65,536 statistical ceiling implies a 192-block history-binning horizon: past
that distance the equation cannot reduce a supported weight. This is **not** a
light or ray cutoff. Shorter ceilings, recency weighting and a change detector
were measured against cache feedback and rejected; see
[surface cache](surface-cache.md). Changed sunlight contributes its own spatial origins.

Legacy geometry control and the unchanged non-geometry policies:

| Change | Retained statistical weight on refresh |
| --- | --- |
| Unchanged transport | Accumulate up to 65,536 |
| Scene edit/load/unload within 48 voxels, including voxel emission | At most 16 |
| More distant tracked geometry change | At most 256 |
| Incomplete geometry transition history | At most 64, or a stricter known local cap |
| Nearby moving analytic source / global source change | At most 4 |
| Tracked remote-only analytic changes | At most 64 |
| Incomplete source transition history | At most 4 |
| Global material-table or transport-model replacement | Restart measurements; retain display |
| Existing receiver changes material ID | At most 16; retain incident mean, moments and display, mark stale |
| New face | No measurements; optional display-only seed |

Geometry and sources use the same eight-transition `ChangeHistory` mechanism,
with different policies over their deltas. Every intervening retained transition
is consulted. The journals keep metadata, not scene snapshots or old rays.
Directional lights that each turn at most 2° and change energy by at most a
tenth per update are a moving sun (`SourceChange::Sun`): sources are replaced
with no history cap or revision change; shadow-border flips carry the change.
Larger jumps (a time skip) remain a global source change.
Canonical source ordering prevents ECS order changes from causing relights;
point-source deltas include both old and new influence volumes but exclude
unchanged lights. Distance affects responsiveness, never visibility or eligibility.
The adapter omits analytic lights with zero contribution in every transport band
and custom visible output before either backend or the sunlight cache sees them.
Infrared-only and ionizing-only sources remain active. Source inputs use ECS
change detection before normalization/canonicalization. Environment, source or
transform changes, component removal and despawn refresh the snapshot; unchanged
inputs skip rebuilding it. The gate still scans source change ticks and may
conservatively rebuild after an unrelated transform removal. Raster visibility
is not a transport-enable switch; explicit multi-band enable semantics remain
an integration decision. Voxel emission uses scene changes, not this light list.

Removing, replacing or covering an emitting voxel uses ordinary spatial history:
old energy decays as new paths arrive, rather than restarting every receiver.
This trades a bounded history tail for stability; the measured local removal
response is documented in the [scene-history audit](history/scene-history-audit.md).
It is not an instantaneous extinction guarantee, especially at distant receivers
retaining large weights or during frequent streaming. All five bands retain the
same statistical weight. Distance is a response heuristic, not proof of weak
lighting influence: a very bright distant source can still matter greatly.

The [distance/history qualification](history/distance-history.md) derives a history cap
from a bounded lighting change and response budget, then compares real canyon
loads, remote cave changes and long-range shadows. It supports retaining remote
compatible history, but rejects distance alone as a general influence bound.
Spatial retention must also survive bounded-journal eviction; retaining
more samples without that metadata is not a complete solution.

Historical design (superseded by GPU application above): the
[sun-aware retention experiment](history/sun-aware-retention.md) separates
spatial evidence, measured sampler quality, stale-light budget and an empirical
history floor. Changed shared-corner sunlight supplies secondary change origins;
calibration remeasures paths to a fixed quality goal when sampling improves.
The conservative offline arm preserves baseline edit responses while retaining
remote evidence. Its [runtime implementation](history/runtime-retention.md) introduced
per-page bounded journals with timestamped
conservative overflow, independent sun-evidence epochs, and immutable page
snapshots for worker-side distance/weight queries. Selection checks page stamps;
caps apply once before either backend merges new samples. Both known transitions
and conservative unknown sky evidence count. This is not a complete dependency
oracle or universal distance bound; the measured remaining limits are explicit.
That infrastructure now applies the whole-range equation above rather than the
older calibration curve; the legacy control and floor summary were removed on
the surface-cache branch. Analytic-source/model policies in the table remain
unchanged.

The default [distance-band overflow summary](history/distance-band-retention.md) replaces
the two fallback floors with revision stamps per distance band and origin. It
reuses worker-computed change-to-page AABB distances and stores no baked sample
caps. Eight recent exact events and per-receiver consumption epochs remain;
band storage is allocated only when a page first evicts an exact event.
The default journal omits an event when the calibrated prior at its minimum
page distance permits unlimited history for every receiver. Such an event
cannot evict useful bounds or allocate a band summary. A separate per-page
latest-change revision preserves existing refresh wakeups without storing its
bounds; wakeup checks do not depend on the last retained journal entry.
This avoids discarding all distance information at eviction, but page extent
and downward distance quantization are still conservative approximations.
The summary does not change the whole-range equation or its empirical quality target.
The page metadata supplies both retention caps and pending-change wakeups, not ray
traversal or radiance storage. An older closer event cannot generally replace a
newer farther one: a receiver may already have consumed the older revision.
Distance-only pruning would lose freshness as well as retention evidence.

`invalidate_model` is the explicit global restart operation. It advances both
the source snapshot and model-reset watermark together, rejecting in-flight
old-model jobs. Ordinary analytic light changes retain bounded history instead.
Receiver material replacement uses ordinary local relighting: incident energy
has the same spatial definition even when the receiver reflects or emits
differently. Revision advancement rejects pre-replacement jobs; self-emission
updates immediately and never becomes incoming simulation energy. New world-only
faces remain unmeasured; new view faces can use the bounded raw topology prior
described above. A new receiver's display-prior clock is initialized explicitly,
independent of its material ID, not as a side effect of resetting measurements.
No rule tests a house, tunnel, skylight or gameplay material ID inside the
estimator. The [completed policy controls](history/scene-history-audit.md#policy-closeout)
explain why the remaining history limits and dark-discovery minimum are retained
as heuristics, not presented as physical requirements.

Scene publication preserves compatible faces, introduces new faces without
claiming measurements (the view's raw prior is explicitly approximate), and publishes at most four atomic pages within a nominal
2 ms budget. A single expensive page can exceed that budget. Worker scans
can include unchanged touching pages. Equal exposed keys, face masks,
material IDs and emission establish identity; those pages keep their allocations
and indices without destructive removal/reinsertion. They still pass through
the common edit-invalidation and refresh path. Existing BVW event
coalescing remains 40 ms for edits / 350 ms for streaming, with a one-second
overdue bound. The audit adds no debounce.

## Scheduling and asynchronous publication

Eligibility is per face. Current well-measured faces rest after four stable
batches, low per-band error/drift and settled display. The default minimum is
64 samples; very dark indirect RGB requires 256 so an initial run of zero-valued
paths is not mistaken for proof of darkness. All bands participate in uncertainty.
The relative-error calculation is a heuristic over correlated, bounded samples,
not a confidence interval or certified convergence test.

Settled faces remain eligible for two-second audits. Changes wake them.

**Bulk work is selected on the GPU** (`select_classify` / `select_threshold` /
`select_emit` in `voxel_accumulate.wgsl`), inside the batch that traces it,
from the GPU's own statistics. The CPU sends only event work: faces an edit,
relight or explicit request made urgent, with their revision caps and
rollbacks; the rest of the face budget is the GPU's. Every live, due face
competes in one class: coverage (fewer than 8 samples, nearest the camera
first), refinement (within two pages of the camera; highest relative error +
0.1 per second of age, +0.25 within one page) or the world sweep (beyond;
oldest first). Quotas are 1/4, 1/2 and 1/4 of the GPU's budget; an empty
class lends its share to refinement, then coverage, then the sweep. One
histogram per class over 256 log-scaled key buckets gives exact per-class
thresholds; ties at a threshold bucket are taken in arbitrary order.

Selection is memory-bound (~800k slots in the hallway on an M4), so it reads
a compact per-slot ranking record, never the 144-byte state: noise per
sample, last merge time, brightness and a mirrored count, written by the
passes that change them. Each batch re-ranks one rotating quarter of the
slots; keys persist, and a picked face's or CPU job's key is cleared until
its quarter comes round. Histograms and pick reservations aggregate per
workgroup (per-face global atomics on the few hot buckets serialized), and
the threshold scan runs in one workgroup. The selection pass costs ~0.6 ms
of GPU time per batch at that size, inside the trace budget.

Stability is tracked per slot on the GPU (`record_stability`, the same rule
as `scheduling::record_face_batch`): stable batches, freshness against the
transport revision, and the last CPU job's batch. A face with a CPU job in
its readback window (four batches) is never picked, so a pick cannot race
that job's merge. Retention events and source changes revoke stability on
the GPU. Every face the GPU ranks is considered each batch, not a rotating
window; the camera is a priority hint, never the extent of the lighting field.

Picks return with the batch's readback (slot, revision tag, merged count,
incarnation); the CPU uses them for evidence donor lists and rollback. Donor
planning follows GPU history: the merged count is after any retention cap, CPU
jobs a cap left below the evidence target are re-listed from their readback,
and faces near a player edit in the batch are planned as if they had no
history (the GPU measures the cap; the CPU only walks the graph). A pick whose slot's
revision tag is older than the probe's revision traced pre-change light: the
CPU's next job for the probe undoes it, unless one is already in flight.

The GPU state is the statistics authority. Production probes keep no CPU
measurements (test fixtures attach `ProbeStatistics`); the core record keeps
topology, display and sun-corner words, revision flags and resident addresses.
A 32-byte `ProbeProgress` retains per-face sequence, bootstrap-saturated count
and the GPU's settled verdict, which supply HUD coverage/bootstrap and reject
older mirror records. The verdict occupies spare bits in the existing readback
count, never the resident count. These bounded, delayed progress observations
are not a convergence measure. A pick marks its face's mirror stale, and
selection emits at most `statistics_mirror_per_batch` (default 1,024) stale
faces' records per batch: progress and the presentation seed that publication
reuses for neighbouring new faces. The per-pick merge was most of the
main-thread GI cost (gameplay Tracy, 59e7346: ~1.4 of 2.9 ms/frame).
The first bounded mirror was shelved because CPU code still fed
the GPU from mirrored statistics. Two such paths were fixed independently
(eager-slot addresses, 8f361af; relight caps derived from mirrored counts, now
explicit). The rest was edit handling: an
edit's flux (`brightest_outgoing`) and which nearby faces it wakes
(`retention_binds`) read the mirror at the moment of the edit, so with a lagging
mirror the GPU under-reacted (response profile, rendered frame 1 s after an
emitter removal: FLIP to settled .32-.39 vs .08). Both are now computed on the
GPU from its own statistics: `measure_edits` takes the brightest outgoing
luminance (material albedo times incident, plus emission) within one voxel of
each edit, and `apply_events` marks the faces within 64 voxels whose history
the change binds (or all within 48 for unknown flux) urgent; selection serves
urgent faces first, up to three quarters of its budget. The CPU no longer reads
statistics at an edit; it refreshes sky near it and relights changed probes.
Earlier CPU windows and quotas are in the
[streaming experiment record](history/research-2026-10-09.md#refinement-under-continuous-terrain-loading).

Only selected measurements cross the worker boundary in `TraceWork`; topology
indices, temporal display and scheduler state stay authoritative on the main
world. Workers own scene/source snapshots and perform preparation, submission,
waiting and decoding asynchronously. The explicit CPU reference pool is created lazily and
limited to two threads. One outstanding trace batch bounds resource ownership.
The GI budget is `gpu_time_share` (default 0.2): the share of wall time
transport batches may occupy the GPU, independent of frame rate. An allowance
accrues with wall time (banking at most two batches); a batch reserves
`trace_budget_ms` at submission and is charged its measured GPU time at
completion, and batches wait while the allowance is spent. Main-thread work
per batch (selection, merge) scales with the batch rate; scene publication,
display topology (2,048 additions per frame) and value uploads keep their own
per-frame caps.

The default dispatch adds 16 paths per selected face within the same adaptive
4-ms transport target. This amortizes receiver scheduling, transfer and merge;
it does not increase bounce depth, reach, retained history or the work target.
Smaller batches remain available for more frequent individual-receiver refresh.
The [paired cave and response evidence](history/streaming-performance.md#repeated-batch-size-result)
supports the default, including its early-coverage/refresh-cadence tradeoff.

Publication rejects incompatible receiver geometry and obsolete global-model
resets. Moving-light results can be accepted as stale to avoid perpetual starvation;
their revision metadata still prevents simulations from treating them as current.
Sky additionally requires matching scene and sun generations. Accepted selected
faces update in place, preserving untouched faces and live topology indices.

## Presentation

Every visible face is view-owned (see [view-owned lighting](#view-owned-lighting));
the world cache never draws. The terrain material reads only the view tier's
output, the shared geometry (for sun visibility of coverage holes) and the
current suns. The world-cache display path (probe directory, display probes,
evidence donor lists, render transitions and their publication) was removed on
2026-10-23 with the hybrid strategy; see
[the ownership comparison](history/option-comparison-2026-10-23.md).

Temporal filtering still runs on resident GPU accumulation once per updated
face, blending measured RGB with the face's previous temporal value (record 6).
The base response is `1 - exp(-elapsed / response_seconds)`; sample evidence
and visible-band noise further limit the blend. This is biased presentation
state, not a modified physical estimator. Through the two-way exchange it is
the "shown" prior a newly revealed view face can import. Newly exposed world
faces are seeded from compatible pre-edit neighbours (`history.rs`); seeds
never become measurements or moments.

View presentation reconstructs each virtual unit-face corner from up to four
coplanar view-owned faces, excluding disconnected diagonals, then bilinearly
interpolates across the face. `spatial_reconstruction` and
`spatial_evidence_target` configure the view tier's nonrecursive coplanar
filter (`config/gi-display-unfiltered.ron` turns it off). Direct sun uses the
same four-corner visibility, from the view tier's corners or, for faces the
reduced grid missed, traced directly.

## Simulation and sources

GI offers no simulation queries: heat and radiation belong to a deterministic
direct-transport field ([direction](direction.md)). The former
`estimate`/`sample`/`request_refresh` API and its tests were removed.

`EnergyEmitter` augments a Bevy directional/point light or acts as a transformed
point source alone. `VoxelMaterialTransportTable` supplies per-band diffuse
response and emission; inputs are sanitized and reflectance is bounded to one.
`VoxelGiEnvironment` makes the background energy host-configurable. Rendering
never writes filtered RGB back into these measurements.

### Resident measurements and asynchronous scheduling feedback

The runtime implements GPU ownership. Stable `(voxel, face)` slots retain legacy five-band means, second
moments, direct energy, effective count, advancing sequence, and filtered display
RGB. Slot incarnations separate allocation identity from scene freshness.
Growth copies measurements; retirement is local to changed regions; recycling
initializes the new surface before its address can be used for rendering.

The CPU selects work; edit and streaming retention is applied by a GPU pass to
every resident face (see Revisions and history). A GPU prepare pass caps its
**own latest** history with source and revision caps and supplies the current
sequence to the ray task; delayed CPU counts cannot resurrect history or repeat
paths. A probe's source cap also covers its unselected resident faces. The
ray estimator is unchanged. A resolve pass accumulates measurements and applies
the existing presentation filter. The material reads that native buffer through
generation-checked addresses, independently of CPU delivery.

At most three owned readbacks can be in flight. The worker polls readiness;
resident runtime has no blocking GPU wait. A pending receiver is not selected
again until its scheduling feedback is delivered. Each snapshot carries its
job identities and submission timestamp, and publication validates geometry and
source revisions. Sky jobs have generation/batch tickets, so overlapping work
cannot publish superseded corners. The synchronous and CPU runtime backends
were removed on the surface-cache branch; the CPU tracer remains a test oracle.

Snapshots accompany transport batches for CPU scheduling/progress and optional
diagnostics. Their legacy layout still contains nonvisual values; these are not
a simulation delivery contract. The former `estimate` API is removed. Heat and
radiation belong to a separate future server-capable simulation, whose updates
can change voxel materials and therefore their visible emission.

`simulation_sampling_importance` is a legacy research control: zero by default,
one for equal-band proposals. RGB perceptual qualification does not establish
nonvisual accuracy. Shared view/cache writes retain only RGB measurements and
clear unused nonvisual statistics. The old world scheduler still has multi-band
stability machinery; removing it requires the separate recalibration described
in [direction](direction.md), not an unmeasured threshold change in the handoff.

The resident record is 144 bytes per allocated face (including a 16-byte
render-transition record), plus the CPU scheduling snapshot and ownership index.
Render probes grow from 208 to 256 bytes
to carry six addresses. These are deliberate memory costs, not a claim that GPU
residency removes CPU memory. Debug temporal diagnostics reconstruct from the
latest completed resident presentation snapshot; the old CPU upload mirror only
contains initialization values in this mode. Screenshots remain end-to-end
display evidence.

### GPU-owned statistics for relit faces

Resident GPU state is authoritative; an existing slot is never re-seeded from
the CPU snapshot. When a probe's revision changes (an edit nearby, a material or
face change, a model reset), the next selection sends a hard history cap
instead, packed with a rollback flag into the input word that seeds new slots.

- **Cap:** `FaceWork::samples`, which already holds exactly this probe's relight
  (16), source and model-reset (0) caps. Not `history_cap`: in measured mode that
  carries the whole-range curve cap meant only for unselected faces. Applying it
  here zeroed relit faces' history and raised `house-edits` edit transients by
  about 45% in earlier attempts.
- **Rollback:** a batch selected before the change may already have merged
  pre-edit light. The CPU rejects such results (`stale_merge`), and the next
  selection asks the GPU to restore that face's pre-merge checkpoint (48 bytes
  per slot, written on every merge). A rollback is never requested for a merge
  the CPU accepted: for a young face, that may be its only data.

Validated on 2026-10-19 against `7bc829d`, interleaved, four `house-edits`
pairs. Edit-window variation of raw values, 120-180 s, was .146 ± .017 vs
.136 ± .007 (display .094 vs .099); quiet periods equal. `response` and
`sources` were equal.

## Integration contract and limits

### Construction configuration

`VoxelGiPlugin` requires a GPU transport device and always accumulates on the
GPU (resident). There is no construction-time backend choice; distance-band
summaries and measured retention (`VoxelGiSettings::measured_retention`) are
always used. Runtime controls remain in `VoxelGiSettings`. GPU initialization,
dispatch and readback failures are fatal at publication, before partial
results are applied.

The demo parses continuous launch variables before installing
the plugin and reports invalid choices at that boundary. Detailed timing is
enabled by the optional validation adapter. Debug-only history-ablation/audit
switches still use process-global state: embedding multiple independently
configured validation worlds is not yet supported. This cleanup does not claim
the prototype has become a standalone host-independent crate.

### Quality controls and model changes

`VoxelGiSettings.transport` is a normalized `TransportSettings` snapshot: diffuse
ray / voxel-emitter reach (1–768 voxels) and explicit diffuse hits (1–3). The upper
depth bound preserves the existing validated sampler dimension allocation. Reach
is per connection, not total path length or per-block light-level propagation.
Directional visibility remains at 768; analytic point lights retain their own
range. For a shortened diffuse segment, reaching the artificial cutoff does not
certify sky: only an empty remaining coarse occupancy frontier admits environment.
Possible occluders terminate conservatively. This avoids both leaking light
through distant roofs and darkening a receiver merely by loading known-empty
pages. The coarse check can reject energy that a longer fine ray would recover.

A per-leg reach is not the same radius for history invalidation. Several legs,
sun/point visibility and the remaining sky frontier can depend on geometry
beyond it. The [transport-horizon study](history/transport-horizon.md) measures this
quality tradeoff separately from a future bounded-dependency contract. Its
canyon fixture is important: the small enclosed cave alone understates useful
daylight travel distances. Current defaults and invalidation rules are unchanged.

On GPU, depth and finite/default horizon policy are specialization constants.
The worker lazily caches at most twelve face pipelines (three depths × two
policies × two lane layouts);
numeric reach stays in the uniform, so 16→32 does not compile another variant.
The default path keeps the original traversal loop without a runtime cutoff
policy argument. First use of a variant can delay that asynchronous batch during
compilation, not block gameplay. Auto-layout face bindings are recreated whenever
the active pipeline changes; unchanged buffer identities alone are insufficient.
Sky has its own pipeline and is independent of these variants.

Source-informed guiding and fresh path reuse were removed on the surface-cache
branch; neither was qualified with cache termination. Their evidence remains in
[hallway outliers](history/hallway-path-outliers.md) and
[path reuse](history/resident-path-reuse.md).

The normalized model is captured on `TraceWork` for CPU/GPU parity. A model change
calls `invalidate_model`: outstanding old-model results are
rejected, new selections reset measured means/moments lazily, and old simulation
estimates report stale until replaced. Sample sequences and display history
survive; geometry, emissive distributions and direct-sun caches need no rebuild.
No synchronous full-field clear or debounce is introduced. Coverage can remain
high during this transition because it is not a current-model convergence metric.

Work knobs (including 1–64 paths per batch) do not change the integrand or reset
measurements. Adaptive stability tolerances are reevaluated as faces run/audit.
Presentation settings only change the displayed result. Temporal response is a
time constant subject to the existing evidence/uncertainty filter, not a promise
to settle in that many milliseconds. Display gain/toggles do not change energy
returned to simulation readers.

`diagnostics.rs` is the opt-in `VoxelGiDebugPlugin` (Bevy UI, feathers widgets):
a HUD, quality presets (`VoxelGiQuality`, performance knobs only), performance
and look sliders, a staged light model, display switches and RON clipboard
import/export. Widgets bind to `VoxelGiSettings` and `VoxelGiDisplay` through
components and observers; a sync system follows changes made elsewhere. The
lab's `VOXEL_GI_SETTINGS` loader (`src/gi_lab.rs`) is opt-in and fails visibly
on invalid RON; the plugin accepts a host-provided settings resource directly.

### Host integration

This is one axis-aligned unit-voxel world in integer coordinates, with f32 ray
positions. There is no floating origin, world-ID key, deformation metric or
cross-world ray routing yet. Core publishes what a lit material binds as
`VoxelGiMaterialData` (shared geometry, traversal parameters, the visible-face
output and the presentation uniform) and touches no material asset. The BVW
connector crate (`voxel_gi_bvw`) owns page forwarding and the mesh
material: BVW's vertex layout, the host's texture array (`material_template`,
`prepare_texture_array`) and a copy of `VoxelGiMaterialData` on change.
`VoxelGiDebugPlugin` is optional HUD/panel, not core input.

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
sampling distribution, demand-driven snapshots from resident accumulation,
and a host-independent scene-input API. None requires abandoning
the current traversal, revision or presentation contracts.
