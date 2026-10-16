# Architecture

This is the current design. Historical experiments are kept separately so the
implementation can be understood without reconstructing its development history.
The numerical model and the stable display policy are deliberately separate.

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
    CPU simulation estimates                     generation-checked rendering
                                                           |
                                          connected virtual-corner reconstruction
```

The synchronous GPU control and independent CPU reference retain CPU-owned
accumulation. The default resident path does not wait for CPU delivery to render.

| Responsibility | Owner / source |
| --- | --- |
| Plugin wiring and runtime state | `src/voxel_gi/mod.rs` |
| BVW events, tracked edits and immutable page input | `bvw.rs` |
| Scene/receiver representation | `world.rs`, `field.rs` |
| Acceleration and packing | `naadf.rs`, `gpu_scene.rs` |
| Change metadata and retention | `world/delta.rs`, `geometry.rs`, `retention.rs`, `sources.rs`, `revision.rs` |
| Receiver page publication | `publication.rs` |
| Eligibility, priority and dispatch | `scheduling.rs`, `dispatch.rs` |
| Construction policy; model/work configuration and revision synchronization | `startup.rs`, `settings.rs` |
| Thread-transfer payload and result merge | `work.rs` |
| Source model and Bevy light adaptation | `sources.rs` |
| Sampling, emitters and transport | `sampling.rs`, `emission.rs`, `guiding.rs`, `tracer.rs`, `gpu.rs`, `voxel_trace.wgsl` |
| Stable GPU ownership, accumulation and bounded snapshots | `gpu/resident.rs`, `gpu/readback.rs`, `voxel_accumulate.wgsl` |
| Shared uncertainty heuristic | `statistics.rs` |
| Direct-sun visibility cache | `sky.rs`, `sky_cache.rs` |
| Temporal display and initialization | `history.rs` |
| Display publication and rendering | `display.rs`, `render.rs`, `voxel_transport.wgsl` |
| Accounting, optional UI, fixtures | `telemetry.rs`, `diagnostics.rs`, `diagnostics/`, `validation/` |

Shader paths are under `assets/shaders/`; Rust paths above are under
`src/voxel_gi/`. `reconstruction.rs` is an independent CPU test oracle for shader
interpolation, not a second production reconstruction path.

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
A grown buffer receives the full live array before sparse patching resumes.

Top-level bounds include loaded regions, but do not allocate an arbitrarily large
box between disconnected islands: oversized domains fall back to region DDA.
The top-level field depends only on region residency and empty/nonempty occupancy.
Immutable snapshots share its allocation across edits that preserve that topology;
changed voxel regions still trigger normal transport invalidation.
Rays normalize direction, offset from surfaces, traverse at most 2,048 steps and
768 voxel units per segment by default, and distinguish hits, escape and unknown traversal.
These finite-domain/depth limits are explicit approximations, not an exact
infinite-domain light solver. Nonzero paths are never discarded solely because
throughput is small: the next radiance can be large, especially with guiding.
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

The last diffuse hit has no continuing cosine path to balance an ordinary
light-only estimate. Instead it draws one emission connection from a 50/50
light/cosine mixture and evaluates the same balance density for either branch.
This preserves the finite-bounce emission integral while bounding each terminal
connection by twice the emitter's energy, without clipping or adding another
connection ray. Sky escapes and nonemitting hits contribute zero to this terminal
term; it does not extend the environment bounce count. CPU and GPU use the same
unused terminal path dimensions. See the [cave accuracy study](cave-history-retention.md)
for the outlier diagnosis, native cost and independent-reference checks.

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

The default geometry/changed-sunlight policy is the selected
[whole-range equation](whole-range-history.md): `H(d) = round((d / 12)^4)`, applied
as `min(existing_weight, H(d))`. Distance is receiver-to-change distance. The
65,536 statistical ceiling implies a 192-block history-binning horizon: past
that distance the equation cannot reduce a supported weight. This is **not** a
light or ray cutoff, and 2,048 remains a visual-reference sample count, not the
statistical ceiling. Changed sunlight contributes its own spatial origins.

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
response is documented in the [scene-history audit](scene-history-audit.md).
It is not an instantaneous extinction guarantee, especially at distant receivers
retaining large weights or during frequent streaming. All five bands retain the
same statistical weight. Distance is a response heuristic, not proof of weak
lighting influence: a very bright distant source can still matter greatly.

The [distance/history qualification](distance-history.md) derives a history cap
from a bounded lighting change and response budget, then compares real canyon
loads, remote cave changes and long-range shadows. It supports retaining remote
compatible history, but rejects distance alone as a general influence bound.
Spatial retention must also survive bounded-journal eviction; retaining
more samples without that metadata is not a complete solution.

The preceding [sun-aware retention experiment](sun-aware-retention.md) separates
spatial evidence, measured sampler quality, stale-light budget and an empirical
history floor. Changed shared-corner sunlight supplies secondary change origins;
calibration remeasures paths to a fixed quality goal when sampling improves.
The conservative offline arm preserves baseline edit responses while retaining
remote evidence. Its [runtime implementation](runtime-retention.md) introduced
per-page bounded journals with timestamped
conservative overflow, independent sun-evidence epochs, and immutable page
snapshots for worker-side distance/weight queries. Selection checks page stamps;
caps apply once before either backend merges new samples. Both known transitions
and conservative unknown sky evidence count. This is not a complete dependency
oracle or universal distance bound; the measured remaining limits are explicit.
That infrastructure now applies the whole-range equation above rather than the
older calibration curve. `VOXEL_GI_RETENTION=baseline` restores the legacy control;
`whole-range` selects today's default explicitly (`calibrated` is a compatibility
alias, not a way to restore the previous curve). Analytic-source/model policies
in the table remain unchanged.

The default [distance-band overflow summary](distance-band-retention.md) replaces
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
updates immediately and never becomes incoming simulation energy. New faces
remain unmeasured. A new receiver's display-prior clock is initialized explicitly,
independent of its material ID, not as a side effect of resetting measurements.
No rule tests a house, tunnel, skylight or gameplay material ID inside the
estimator. The [completed policy controls](scene-history-audit.md#policy-closeout)
explain why the remaining history limits and dark-discovery minimum are retained
as heuristics, not presented as physical requirements.

Scene publication preserves compatible faces, introduces new faces without
inventing measurements, and publishes at most four atomic pages within a nominal
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

Settled faces remain eligible for two-second audits. Simulation requests and
changes wake them. Bounded candidate searches combine urgent/bootstrap work,
nearby rotating windows and a reserved one-quarter global sweep. Under continuous
streaming, another quarter serves initial coverage and half serves refinement;
explicit edits preempt the three non-global slots. Empty queues lend their slots.
Measured receivers rank ahead of unmeasured fallback candidates in refinement's
bounded window, so initial coverage cannot consume both reservations. The camera is
a priority hint, never the extent of the lighting field. Empty/settled candidates
do not consume the ray allowance or collapse the scheduler budget.

The close window covers the camera page and its immediate neighbors; the second
adaptive window covers the disjoint next page ring (Chebyshev page distance two).
If that ring is empty it falls back to global uncertainty ranking. Both windows
remain capped at 1,024 candidates, with the existing near bonus and close-range
fallback cursor. This gives large rooms useful refinement across their far wall
without mixing all surrounding terrain into the close-range sweep. It changes
work allocation, not transport extent, spatial filtering, or the sampling PDF.
Only receiver membership changes inside these two page rings invalidate the
local index; distant streaming does not rebuild it or rewind its cursor.
The reserved global sweep still visits every loaded receiver, including hidden
simulation bands. The [streaming experiment record](research-2026-10-09.md#refinement-under-continuous-terrain-loading)
includes the rejected wider-window and separate-lane alternatives.

Only selected measurements cross the worker boundary in `TraceWork`; topology
indices, temporal display and scheduler state stay authoritative on the main
world. Workers own scene/source snapshots and perform preparation, submission,
waiting and decoding asynchronously. The explicit CPU reference pool is created lazily and
limited to two threads. One outstanding trace batch bounds resource ownership.
The default dispatch adds 16 paths per selected face within the same adaptive
4-ms transport target. This amortizes receiver scheduling, transfer and merge;
it does not increase bounce depth, reach, retained history or the work target.
Smaller batches remain available for more frequent individual-receiver refresh.
The [paired cave and response evidence](streaming-performance.md#repeated-batch-size-result)
supports the default, including its early-coverage/refresh-cadence tradeoff.

Publication rejects incompatible receiver geometry and obsolete global-model
resets. Moving-light results can be accepted as stale to avoid perpetual starvation;
their revision metadata still prevents simulations from treating them as current.
Sky additionally requires matching scene and sun generations. Accepted selected
faces update in place, preserving untouched faces and live topology indices.

## Temporal and spatial display

Temporal filtering runs on resident GPU accumulation (or CPU result merge in the
CPU backend), once per updated face. It blends
that face's measured indirect RGB with its own previous display. The base
time response is `1 - exp(-elapsed / response_seconds)`. Sample evidence and
visible-band noise further limit the blend; noisy dark estimates are deliberately
slow. Brightness-compressed weights suppress isolated HDR jumps without imposing
a permanent brightness ceiling. Confidence weights evolve with the filtered
value. This is biased presentation, not a modified physical estimator.

Resident RGB then uses nonrecursive, connected-coplanar evidence reconstruction,
with the user-selected target 2048 and no extra donor cap. The radius/work bound
still applies. Compatible raw statistics determine donor weights; temporal RGB
supplies display values. Spatial output never feeds transport, moments, temporal
history, scheduling or simulation readback. The CPU reference keeps the original
corner display. See [reconstruction and cost evidence](early-spatial-reconstruction.md).

Resident presentation also retains the currently displayed RGB when a new
target arrives and interpolates toward that target over eight render frames.
This final transition runs in the material, using Bevy's shared frame counter;
it continues while transport works on other receivers. Interrupted transitions
start from their current interpolated value, not an older endpoint. It adds no
readback, per-frame CPU probe loop or recursive spatial feedback. Counter wrap
and pipelined frames preceding a submission are handled explicitly. The
`display_transition_frames` setting is capped at 32; zero restores instantaneous
publication (`config/gi-display-instant.json`). This smooths arrival steps, not
sampling error, refresh starvation or direct-sun updates. Eight frames is about
133 ms at 60 Hz, with queued submission time counting toward that interval.

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
layouts and uploads. Value publication compares aligned words against the
authoritative byte mirror, skips identical records and unions changed byte
envelopes per probe until extraction. A single changed sky face can transfer
16 bytes instead of the whole 256-byte probe; neighboring ranges still coalesce
to bound copy-command cost. This is not removal of the sky readback round trip
or of scheduler/simulation feedback. No second world-sized mirror is retained.
Snapshot packing uses a 1.5-ms slice, checked every 128
keys (including removed keys), with no additional record ceiling. It does not
promise a hard deadline under OS preemption. Encoding and initial
buffer allocation run off thread. Encoding writes one authoritative pair of byte
mirrors directly, without world-sized typed staging arrays and duplicate encoded
payloads. The worker also prepares the CPU-only fallback payloads. Capacity
includes positive growth in already-built, pending receiver pages; future removals
cannot fund earlier additions.

`render/capacity.rs` checks both maximum buffer size and maximum whole-storage
binding size, including hash collision growth and integer/shader addressing.
Optional 50%/16,384-slot headroom is reduced to fit those limits; occupied receivers
are never dropped to fit. The main world preflights before starting a snapshot,
the worker checks before allocating large mirrors, and native preparation checks
the actual device again before either GPU allocation. These limits apply to CPU
transport too, since presentation still binds the same GPU storage. An oversized
replacement is a fatal capacity error, with the receiver count, generation and
device limit in the diagnostic. Invalid upload ranges are likewise fatal before
submission. Neither condition silently leaves the player with stale lighting.
Normal asynchronous generation handoffs still preserve the live generation;
temporary worker ownership is not an error. This is not multi-buffer paging or
protection against arbitrary system-memory exhaustion; the host must keep
residency within its supported device limits.

A replacement remains private while topology and measurements catch up in
batches of 512 keys and a 1.5-ms soft game-thread target. Large journals capture
compact immutable records within that slice and transfer the private generation
to a worker for index/GPU-memory updates. Small tails reconcile on the game thread
so installation can be atomic with the latest state. There is no additional
8,192-key per-frame catch-up cap. The existing cache keeps receiving value updates;
the replacement's topology, values and material become visible atomically only
when its journal is empty. This trades replacement latency for bounded frame
work, not transport starvation or a debounce. Ordinary sparse edits still use
the live generation when capacity permits.

One journal records actual presentation changes and topology membership changes
through snapshot, worker build and private reconciliation. Snapshot capture
cancels already-covered work; later changes re-enqueue fairly. An unflushed live
upload is not a new change. Catch-up reads current membership and values rather
than replaying stale event payloads, including remove/re-add of the same voxel.
Capacity/hash exhaustion restarts from current residency; a scene revision alone
does not. Worker-built lookup maps reserve growth capacity too, avoiding a
main-thread rehash while catching up.

Private buffers remain mapped until catch-up is complete. Each batch acquires
mapped views once and edits private indices directly; no live-transaction
preflight or sparse-upload ledger is needed for an invisible generation. Each bounded patch
updates their initial bytes directly, so installation cannot produce a giant
deferred patch upload. On devices with `MAPPABLE_PRIMARY_BUFFERS` already enabled,
the buffers include `MAP_WRITE`: this avoids wgpu's otherwise implicit full-size
staging copy on unmap. Bevy normally enables this on supported integrated GPUs,
not discrete GPUs. Other devices retain mapped-at-creation staging; CPU-only
fixtures use byte arrays. The measured large-copy improvement is therefore
specific to the supported integrated-device path, not a universal backend claim.

Installation unmaps both buffers before extraction. Descriptor-only assets and
their exact initialized native buffers are registered before Bevy's asset and
material preparation; matching size, usage and label prevent duplicate allocation.
Native tests cover buffer identity, initial/catch-up byte parity, GPU submission,
generation changes and the CPU fallback. See the
[publication measurements](streaming-performance.md#bounded-display-publication)
for latency, throughput and remaining stalls.

Each generation owns fresh hash/probe buffer identities and its own shared patch
queue. Extracted older frames retain their generation. Hash and value patches
preflight together and snapshot under one lock; a failed bounds check retains both
for retry. Driver writes happen after releasing the lock, using immutable paired
scratch bytes; changes arriving meanwhile form the next transaction.
Ordinary value changes use a coalescing FIFO and a 1.5-ms main-thread slice,
checked every 128 records. A nonblocking lock attempt leaves all pending keys
untouched if the renderer is capturing a transaction. Unfinished values retry
on the next frame without another upload-interval delay. This bounds the value
path, not atomic topology publication or OS scheduling; it is not a hard frame
deadline guarantee.
Sparse writes use wgpu's reusable mapped staging chunks instead of allocating
one temporary driver buffer per range. One copy submission publishes both
buffers before drawing. Recall is asynchronous, with no device wait in the main
world; renderer recovery recreates the device-owned pool.
Topology updates include values on existing newly exposed faces, so
the value-upload timer cannot split the transaction. A separate required-value
ledger contains face-mask/material changes, not the ordinary lighting backlog.
Successful topology publication consumes only those values and additions/removals;
unrelated values stay on their bounded FIFO. Topology also uses a nonblocking
mirror lock: contention retains the entire transaction for retry, without allowing
its required values to publish separately. The shader's address bound
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
Check **both** `sources_current` and `transport_current`: voxel emission belongs
to scene transport, not the analytic-source snapshot. Even when both are true,
the estimate can contain bounded pre-change history. Fresh means refreshed, not
unbiased after an edit or converged. An extinguished IR/radiation source can have
a nonzero retained contribution; simulation must choose acceptable age/error and
integrate exposure explicitly, not interpret these flags as a safety guarantee.

Use `request_refresh(voxel)` for bounded asynchronous priority, then inspect the
returned estimate later. `sample` is a convenience wrapper that discards metadata;
simulation systems should prefer `estimate`. No refresh request promises a
deadline. Variance is a second-moment diagnostic, not a calibrated error bound.

`EnergyEmitter` augments a Bevy directional/point light or acts as a transformed
point source alone. `VoxelMaterialTransportTable` supplies per-band diffuse
response and emission; inputs are sanitized and reflectance is bounded to one.
`VoxelGiEnvironment` makes the background energy host-configurable. Rendering
never writes filtered RGB back into these measurements.

### Resident measurements and asynchronous simulation delivery

The default `Auto` / explicit `Resident` backend implements runtime GPU ownership, not only a
fixed-slot prototype. Stable `(voxel, face)` slots own five-band means, second
moments, direct energy, effective count, advancing sequence, and filtered display
RGB. Slot incarnations separate allocation identity from scene freshness.
Growth copies measurements; retirement is local to changed regions; recycling
initializes the new surface before its address can be used for rendering.

The CPU selects work and applies the same retention evidence as the original
backend. A GPU prepare pass caps its **own latest** history and supplies the
current sequence to the ray task; delayed CPU counts cannot resurrect history
or repeat paths. Probe-wide evidence also caps unselected resident faces. The
ray estimator is unchanged. A resolve pass accumulates measurements and applies
the existing presentation filter. The material reads that native buffer through
generation-checked addresses, independently of CPU delivery.

At most three owned readbacks can be in flight. The worker polls readiness;
resident runtime has no blocking GPU wait. A pending receiver is not selected
again until its scheduling feedback is delivered. Each snapshot carries its
job identities and submission timestamp, and publication validates geometry and
source revisions. Sky jobs have generation/batch tickets, so overlapping work
cannot publish superseded corners. The original synchronous backend and CPU
reference remain available as explicit comparison backends.

Snapshots retain all five bands, sample/variance metadata and measurement ages,
and distinguish unknown or stale results from valid zero energy. They currently
accompany transport batches because the CPU scheduler needs their feedback. A
separate demand/tick-based snapshot cadence is future work, not a prerequisite
for rendering. There is no fixed simulation tick rate or delivery deadline.

Simulation is allowed to lag lighting. `estimate` returns the latest completed
unfiltered measurement with age/currentness metadata; it never stalls the player
to obtain the current render-frame value. Acceptable age, exposure integration,
and material-state transitions belong to the future host simulation. Filtered
display RGB never becomes simulation energy.

RGB perceptual quality and simulation accuracy are separate requirements.
`simulation_sampling_importance` changes emitter proposal weights, not emitted
energy: RGB has weight one and IR/ionizing have a configurable positive weight.
The default nonvisual weight is 0.125 on every backend, independently of path
reuse; one restores equal-band proposals. This is a proposal preference, not an
eight-to-one allocation of rays or a change to the five-band transport model.
CPU and GPU MIS use the same weighted power. Nonzero support preserves physical
expectation, while lower simulation importance can increase its sampling noise.
Proposal-cache changes preserve valid lighting history. Display confidence and
smoothing remain RGB-only; the current conservative sleep heuristic still checks
all bands. Demand-specific simulation tolerances/ticks remain future host work.

The resident record is 144 bytes per allocated face (including a 16-byte
render-transition record), plus the CPU scheduling/
simulation snapshot and ownership index. Render probes grow from 208 to 256 bytes
to carry six addresses. These are deliberate memory costs, not a claim that GPU
residency removes CPU memory. Debug temporal diagnostics reconstruct from the
latest completed resident presentation snapshot; the old CPU upload mirror only
contains initialization values in this mode. Screenshots remain end-to-end
display evidence.

## Integration contract and limits

### Construction configuration

Insert `VoxelGiStartup` before adding `VoxelGiPlugin`. It selects
`TransportBackend::Auto` (required resident compute), `Cpu`,
explicit `Resident`, opt-in `Reuse`, or the `Synchronous` comparison, and
`RetentionPolicy::WholeRange(RetentionSummary)` or the legacy comparison.
Defaults preserve whole-range retention with distance-band overflow summaries.
These are per-App construction choices, not live switches. Runtime controls
remain in `VoxelGiSettings`; CPU transport still uses the normal GPU display.
GPU initialization, dispatch and readback failures are fatal at publication,
before partial results are applied. They never switch the running game to CPU
transport. Select `Cpu` explicitly when the independent reference is wanted.

The demo parses backend/retention/continuous launch variables before installing
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
beyond it. The [transport-horizon study](transport-horizon.md) measures this
quality tradeoff separately from a future bounded-dependency contract. Its
canyon fixture is important: the small enclosed cave alone understates useful
daylight travel distances. Current defaults and invalidation rules are unchanged.

On GPU, depth and finite/default horizon policy are specialization constants.
The worker lazily caches at most twenty-four face pipelines (three depths × two
policies × two lane layouts × guided/control sampling);
numeric reach stays in the uniform, so 16→32 does not compile another variant.
The default path keeps the original traversal loop without a runtime cutoff
policy argument. First use of a variant can delay that asynchronous batch during
compilation, not block gameplay. Auto-layout face bindings are recreated whenever
the active pipeline changes; unchanged buffer identities alone are insufficient.
Sky has its own pipeline and is independent of these variants.

Source-informed guiding is a proposal choice, not part of `TransportSettings`.
It mixes cosine sampling with a source-directed spherical lobe and applies the
same PDF to throughput and emitter MIS, including the terminal connection.
The light-tree PMF remains distinct from the source-concentration heuristic.
A GPU preparation pass writes one temporary 16-byte proposal per selected face
after the result/sky payload; it introduces no CPU upload/readback or persistent
world records. Preparation is included in the compute budget. CPU tracing uses
the same proposal equations directly. Switching proposals preserves physical
history; neither old measurements nor a missing guide are replaced by invented
energy. A zero-weight guide is the intentional cosine proposal, not an error
fallback. The [hallway investigation](hallway-path-outliers.md) records the
variance reduction, extra raw-sampling cost, rejected alternatives, and the
independent correctness audit that removed biased path truncation.

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

`diagnostics/tuning.rs` is the optional host UI: staged model Apply, live work and
display sliders, session A/B presets, and explicit JSON import/export. Presets
copy only presentation fields, never GPU addressing/generation state. Opening an
idle panel does not mark settings or material assets changed every frame. The
demo's `VOXEL_GI_SETTINGS` file loader is opt-in and fails visibly on invalid JSON;
the reusable plugin still accepts a host-provided settings resource directly.

### Host integration

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
sampling distribution, demand-driven snapshots from resident accumulation,
and a host-independent scene-input API. None requires abandoning
the current traversal, revision or presentation contracts.
