# Architecture

Voxel GI estimates diffuse RGB lighting for one axis-aligned voxel world.
The host owns storage, meshing, edits, lights and cameras. The core owns an
immutable transport scene, GPU tracing, history and presentation. The
[crate README](../crates/voxel_gi/README.md) describes the public setup API;
[validation](validation.md) describes how to qualify changes.

## Data flow and ownership

```text
host page events, voxel snapshots, edits, materials and lights
    -> scene worker: region geometry and change metadata
    -> main world: geometry generation and bounded receiver publication
    -> GPU worker: geometry uploads, resident accumulation and maintenance
    -> per-camera GPU view tier: discovery, tracing and cache exchange
    -> host material: reconstruction, sunlight and emission

GPU -> bounded asynchronous receipts -> CPU diagnostics and scheduling feedback
```

Raw light accumulation stays on the GPU. CPU diagnostics are delayed mirrors,
not rendering inputs or authoritative simulation measurements. Geometry workers
and GPU readbacks do not block the player's main thread; scene publication has
its own per-frame work bounds. This is bounded scheduling, not a guarantee
that every frame or source call has a fixed maximum cost.

| Responsibility | Owner / source |
| --- | --- |
| Plugin wiring and runtime state | `lib.rs` |
| Host input: pages, edits, voxel source, interest envelope | `input.rs`, `pages.rs` |
| Host block types to transport materials | `palette.rs` (`MaterialPalette`) |
| BVW events, edits and source (connector crate) | `voxel_gi_bvw/src/lib.rs` |
| Scene/receiver representation | `world.rs`, `field.rs` |
| Acceleration, packing and shared GPU transactions | `naadf.rs`, `gpu_scene.rs`, `gpu/geometry.rs` |
| Change metadata and retention | `world/delta.rs`, `geometry.rs`, `retention.rs`, `sources.rs`, `sun_changes.rs` |
| Receiver page publication | `publication.rs` |
| Eligibility, priority and dispatch | `scheduling.rs`, `dispatch.rs` |
| Model/work configuration and revision synchronization | `settings.rs` |
| Thread-transfer payload and result merge | `work.rs` |
| Source model and Bevy light adaptation | `sources.rs` |
| Sample streams, emitters, transport limits and the GPU tracer | `sampling.rs` (hashes), `emission.rs`, `tracer.rs` (limits), `gpu.rs`, shared `voxel_path.wgsl`, face consumer `voxel_trace.wgsl` |
| Surface-cache face lookup | `gpu/face_index.rs` |
| View-owned far receivers, scheduling and asynchronous timing | `gpu/view/runtime.rs`, `gpu/view/timing.rs`, `voxel_view_runtime.wgsl` |
| Bidirectional RGB face ownership (two-way cache) | `voxel_view_cache.wgsl`, `gpu/view/cache_tests.rs` |
| Stable GPU ownership, accumulation and bounded snapshots | `gpu/resident.rs`, `gpu/readback.rs`, `voxel_accumulate.wgsl` |
| Uncertainty heuristic statement (tests) | `statistics.rs` |
| Direct-sun corner words and change footprints | `sky.rs`, `voxel_path.wgsl`, `voxel_trace.wgsl` (`trace_footprints`) |
| Temporal display and initialization | `history.rs` |
| Material data and shared buffers; GI material bindings; stock material | `render.rs` (`VoxelGiBindings`, `VoxelGiStandardMaterial`), `voxel_gi_standard.wgsl` |
| Presentation for host materials (`voxel_gi::presentation`) | `voxel_gi_presentation.wgsl` |
| BVW mesh material (connector crate) | `voxel_gi_bvw/src/material.rs`, `voxel_gi_bvw/shaders/voxel_transport.wgsl` |
| Accounting; test fixtures | `telemetry.rs`; `research.rs` (tests only) |
| HUD and tuning panel (debug crate) | `voxel_gi_debug/src/lib.rs` |
| Validation harness (lab) | `src/validation.rs`, `src/validation/` |

Unqualified paths are in the core crate: Rust under `crates/voxel_gi/src/`,
shaders under `crates/voxel_gi/shaders/` (embedded with `include_str!`). `reconstruction.rs` is an independent CPU test oracle for shader
interpolation, not a second production reconstruction path.

## Host input contract

`VoxelGiInput` declares loaded, changed and unloaded **32³ pages**.
`VoxelSource::sample_page` returns material IDs 0–255 or `AIR` (256), at a
power-of-two stride from 1 to 32. Source calls run on workers; hosts should
provide immutable snapshots. A single build does not make multiple live host
lookups an atomic world snapshot.

`MaterialPalette` maps host keys to deduplicated transport IDs. Its IDs must
be the ones returned by the source. The BVW connector instead passes raw
`u8` material indices through: hosts populate `VoxelMaterialTransportTable`
at those same indices.

Default input tracking conservatively resamples unknown changes. A host that
opts into `VoxelGiInput::tracked()` must forward every edit through
`edit_voxel`, the connector's `BvwVoxelEdits::set_voxel`, or an explicit
`mark_voxel`/`mark_all_loaded` notification. Exact edits form a sparse,
last-write-wins overlay. Unknown notifications discard the affected overlay
and request authoritative resampling; later exact edits can override again.
Overlays survive unload/reload and grow with distinct edited positions.
There is no save-game persistence or host acknowledgement for retiring them.

The BVW connector samples the geometry its chunk mesh uses. A downgraded chunk
may retain finer voxel data; sampling follows its meshing shape and nearest
resampling rule. Authoritative resampling preserves coarse-cell coverage where
BVW returns `Unset`, then applies known live values and edits. Nonstandard
chunk shapes use the full-resolution lookup and optional `GiFallbackVoxel`.

## Geometry and traversal

Each page contains eight 16³ regions. A region contains 4³ bricks, each with
4³ voxels. Uniform pages and regions avoid dense expansion. Mixed bricks use
packed material storage; unchanged occupancy can reuse refined bounds.

Nested axis-aligned empty-space bounds accelerate region, brick and voxel
traversal. The top-level field skips loaded empty regions. Oversized or sparse
disconnected domains use region DDA instead of allocating their full bounding
volume. Missing regions remain distinct from loaded air.

Scene builds are asynchronous. Sampling is bounded by a time budget checked
between pages, a dense-page limit and a total-page limit. Deferred pages return
to the dirty queue. These limits do not bound region construction, copying the
region map, top-level rebuilding or surface scanning. Large loaded worlds can
therefore take seconds to publish edits or LoD transitions. Shortening that
latency is planned work, not a solved property of the sampling budget.

Immutable regions retain `Arc` identity when unchanged. A bounded scene-lineage
journal stores region keys and identity stamps, not old voxel payloads.
Consumers that miss the journal compare endpoint snapshots conservatively.
Geometry packing emits sparse header/cell patches. Directory changes and
allocation growth are submitted transactionally; geometry buffers and their
bounds detach together so older extracted materials retain a coherent generation.

The default path segment reaches at most 768 voxels and uses at most 2,048
traversal steps. These are approximations, not infinite-domain visibility.
Strict transport excludes unknown paths from evidence. Display visibility can
cross unloaded gaps to test loaded geometry. Budgeted far rays count traversal
exhaustion as a miss. None of these policies can see unloaded terrain.

## View-owned lighting

Each active camera supplies depth and normal prepasses. GI runs after those
prepasses and before opaque shading. Rasterized world position and normal
must agree with the declared voxel geometry; arbitrary warped meshes or absent
prepasses do not satisfy this contract.

A reduced grid organizes discovery, but physical slots are independent of
screen cells. Each 2×2-cell tile indexes projected face witnesses and can borrow
free slots from the whole receiver pool. The preparation pass validates retained
witnesses before discovery allocates new ones. Coarse cell-centre pixels are
computed with integers so backend float rounding cannot change discrete picks.
Full-resolution footprint scans recover missing planes over staggered frames.
Pool exhaustion is counted; coverage is not guaranteed for every raster sample.

A receiver identifies a voxel block face and surface plane. Reprojection keeps
its raw mean, second moment, sample sequence and unfinished path. Witnesses are
points actually visible on the face, so a hidden face centre does not discard
a grazing surface. Static depth/normal fingerprints reduce redundant discovery;
world changes still update lighting. Slots, hashes and all path data stay on GPU.

Each owner advances a path in bounded waves of at most three intersections.
An unfinished path keeps its original sequence and publishes no partial sample.
Known black samples count; unknown attempts consume their sequence but not
history weight. Uncached view continuations have no fixed bounce-depth cutoff.
No reconstructed image is fed back as a new independent transport sample.

Raster-visible faces whose geometry has not yet reached the transport scene
keep a pending identity. They can retain a weak topology prior or compatible
history but do not trace through stale geometry or write the resident cache.
They become traceable when exposure matches, without replacing their identity.
This cannot guarantee a lit first frame when neither geometry nor donors exist.

### Far tier (per pixel)

Faces below `far_tier.voxel_pixels` use per-pixel histories and own no view
receiver. The default threshold is a full-resolution voxel projecting under
8 pixels across, using camera distance rather than view depth. Other measures
include projected voxel or LoD-block area. All consumers share the same test.

The far tier classifies cells, traces a dense queue, then exposes the output to
the material. Automatic cell size follows viewport height (3 pixels at 1440p,
clamped to 1–4). Pixels within a cell rotate across frames. History reprojects
bilinearly with geometry rejection; cells below four samples trace every frame,
then one frame in four by default. History defaults to 32 and the traversal
budget to 128 steps per ray. Classify and trace share the same visibility
policy for standalone emitters' direct light.

The material applies a plane-aware 3×3 filter and per-fragment sun visibility.
Receiver/far handoff seeds compatible history rather than adding duplicate
counts. Far-tier indirect light has more motion flicker than receiver lighting;
`config/gi-far-tier-off.ron` is the comparison control.

### Two-way face cache

The world cache admits receiver pages within `cache_radius` (64 voxels by
default), with 32-voxel retirement hysteresis. Inactive loading cameras do not
expand lighting residency. Geometry remains all declared pages.

A resident face record is the raw RGB authority. View owners load it before
tracing and store updated raw moments afterward. At first admission a supported
view estimate can replace cold resident history; weights are never summed,
since the histories may share a path prefix. Already admitted resident history
wins even after retention reduces its weight.

The schedulers share an advancing sequence. View attempts reserve indices;
unfinished attempts retain theirs, and unstarted reservation suffixes are returned.
Maintenance revokes a GPU generation guard before changing geometry, ownership
or retention. Older extracted descriptors cannot overwrite the newer state.
Generation checks protect identity and lifetime, not unlimited memory use.

### Raw topology bootstrap

A genuinely new, unsampled view face can receive a one-weight raw prior from
supported neighbouring or pre-edit faces. Aligned donors take precedence;
other orientations receive less weight, and connectivity must be known air.
Donors need at least 16 effective samples. The new face keeps its own sequence;
fresh samples replace the prior's influence. The prior is an approximation,
not conserved flux or independent path evidence. Previously sampled faces
capped to zero are not repeatedly reseeded.

## Surface radiance cache

A path reaching a resident face with at least 16 supported samples may end
there with emission plus reflected direct and cached incident light. The cache
already includes that face's emitter sampling and continuation, so these terms
must not be counted twice. Updates propagate multiple bounces over time.

World-cache paths otherwise evaluate the configured finite explicit prefix.
Emitter next-event sampling uses a small CDF or a spatial light hierarchy;
selection and reverse PDF use the same receiver for balance MIS. Terminal
emission uses a bounded light/cosine mixture. View continuations finish through
cache hit, escape, roulette, zero throughput, reach or unknown traversal.
Analytic point lights have inverse-square attenuation and a range; their loop
is linear in light count. The environment uses a world-Y gradient.

This is a biased iterative radiance cache. The analytic furnace and native
view/cache tests qualify its energy behavior. A finite three-bounce reference
cannot establish its final multi-bounce answer. See the
[surface-cache qualification record](surface-cache.md).

## Shared sunlight

Direct sunlight is a separate known-direction visibility estimator. A virtual
face corner stores a known bit and one visibility bit per sun (up to four).
The canonical origin is offset toward the light so shared corners agree.
The material applies current sun color and direction when shading the face.
View corners refresh periodically; unsupported corners use bounded material
visibility rays. Centroid interpolation keeps MSAA positions on the primitive.

Edits can change illumination far beyond their local bounds. GPU shadow
footprints create secondary change origins on affected surfaces. Refreshed
corner visibility flips also report reflected-light changes through asynchronous
receipts; the CPU buckets these into light-change events. Offscreen gradual
sunlight changes remain an approximation until receivers are observed again.

## Revisions and history

Freshness, effective history weight, urgency and displayed light are separate.
Invalidation need not erase a mean, rewind a sequence or clear presentation.

GPU retention applies geometry/light events to resident and visible receivers.
For known flux, measured variance and predicted change determine a history cap.
Unknown flux uses `H(d) = round((d / 12)^4)`, capped by existing weight and the
65,536 statistical ceiling. This is a response heuristic, not a light cutoff
or proof that distant edits cannot matter. Page stamps trigger refresh audits;
there are no per-face historical scene payloads or CPU distance-band journals.
[Retention research](retention-research.md) records the derivation and limits.

Analytic-source changes use canonical source snapshots and bounded revisions.
Small directional changes preserve history and rely on sunlight feedback;
large jumps use global source-change policy. Moving point lights include both
old and new influence volumes. Material-table or transport-model replacement
advances a reset watermark; incompatible in-flight results are rejected.
Local geometry and emitter edits use the same spatial change path.

## Scheduling and publication

Bulk world-cache selection uses GPU statistics: coverage, refinement, age and
urgent edits. Faces that meet the stability heuristic sleep but receive periodic
audits. A zero-valued early prefix is not proof of darkness. CPU mirrors supply
bounded progress and diagnostic observations; they are not authoritative counts.

Tracing accrues a wall-time allowance from `gpu_time_share`. The view tier and
world cache divide it using `cache_share`. Discovery, reconstruction and material
sun visibility have separately measured costs outside that tracing allowance.
Receiver publication is page-atomic and bounded per frame. These work controls
must be compared with freshness and visible stability, not throughput alone.

GPU submission, polling and decoding occur on workers. Bounded readbacks carry
progress, scheduler metadata and sun flips; full lighting does not round-trip
through the CPU for rendering. Coherent buffer generations, slot incarnations
and revision checks prevent stale work from changing a reused face.

## Presentation

Only view output is displayed. Connected coplanar reconstruction and temporal
filtering operate separately from raw moments. The spatial filter uses measured
variance and bias (`spatial_noise_threshold`, default 0.031), then reconstructs
virtual face corners. Disconnected diagonals cannot donate across a hard edge.
Filtered donor support is not an increase in independent samples.

The material reads per-camera scratch output in encoder order. Validity headers
are cleared before each camera. Host materials bind `VoxelGiBindings` and call
`voxel_gi::presentation`; stock and BVW materials use forward rendering.
Directional lighting, emission and indirect terms must each be applied once.

## Quality controls and model changes

`VoxelGiSettings` controls tracing, cache radius, receiver capacity, far-tier
policy, retention and reconstruction. Normalization clamps public settings.
Work and display controls do not change the integrand; transport reach and
explicit-depth changes invalidate the model. Shortened reach admits environment
only when the remaining coarse frontier cannot occlude it. Reach is per segment,
not a Minecraft-style radius for the entire light path.

## Integration contract and limits

There is one integer-coordinate, axis-aligned world with f32 ray positions.
There is no world-ID key, floating origin, nonlinear warp or cross-world routing.
A host's prepass, material positions and transport voxels must use the same
coordinates and surface shape. A planet adapter needs a correct ray metric and
normal/source conversion; warping raster vertices alone is insufficient.

Missing devices and GI pipeline construction failures idle the affected work
with warnings; failed view allocation clears that view's output. Failures during
an established transport run (submission or readback) remain fatal. GI-lit raster
materials are compiled by Bevy separately and still require DXC on DX12. This is
not automatic recovery from every driver, allocation or shader error.

No simulation query API is provided. RGB values are normalized rendering units,
not watts, temperature or dose. A future server simulation should edit host
materials normally and notify GI through the same edit interface.

Not implemented: transmissive/specular GI, non-voxel dynamic occluders, arbitrary
area/spot-light components, participating media or spectral transport. Large-world
scene builds and newly remeshed LoD faces remain latency limits. Device capacity
and receiver-pool exhaustion still bound usable scene and view sizes.

Detailed development notes and superseded layouts are preserved in
[architecture history](history/architecture-design-notes-2026-10.md).
