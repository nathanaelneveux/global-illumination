# Runtime retention experiment

Baseline: `986683a`. Branch: `codex/gi-runtime-retention`. Opt in with
`VOXEL_GI_RETENTION=calibrated`; otherwise the existing 16/256, eight-global-event
policy remains the control. No BVW, sampler, presentation or work-budget changes.

The native curve uses the fixed `cave-user-2048` appearance goal, Q=2048,
baseline work anchor 2048, fresh response work Q/6 and stale-error allowance .03.
Caps round upward to integer weights (16/256/316/1986 at distances
48/128/192/256). Beyond 48/sqrt(.03), the heuristic retains history. This is
not a hard light range or a physical upper bound on remote changes.

Geometry bounds are binned on the scene worker. Per-receiver-page journals hold
eight spatial events; older local events fold into independently timestamped
16/256 conservative floors. Unrelated distant streaming never evicts local
evidence. Retired receiver pages release their metadata. Queries consume all
intervening events, including late sunlight evidence, without a main-thread
world/probe scan.

Existing CPU/GPU shared-corner results contribute sunlight-change evidence.
Known equal values (including black) contribute none; either transition direction
counts. Unknown/new samples are conservative evidence, not black or unchanged.
Points coalesce into 16-voxel bucket bounds and bin on the tracing worker. No new
rays are added. Partial cache coverage is not claimed complete: as-yet unqueried
corners can deliver evidence later. Until then geometric evidence and normal
freshness/audit sampling remain active. This latency must be qualified, especially
for distant shadows; the visible directional cache is not a detector for every
point-light, diffuse-sky or nonvisible-band dependency.

Spatial evidence has its own revision. A selected job carries that stamp through
publication, so later evidence still wakes it. Stale scene results cannot publish
sky display values, but their possible changes remain conservative retention
evidence. All faces consume a cap together, including unselected faces; means,
moments, sequences and display histories are preserved. Source/model replacement
continues using its existing policy. No debounce or radiance clamping is added.

This first checkpoint is an implementation, **not a gameplay qualification or
default promotion**. Focused curve/journal, sky-state and partial-publication
checks plus native CPU/GPU parity are the initial verification. Paired composed
cave runs with moving background terrain, event-complete corner observations,
reference error and separate windowed frame-time tails determine acceptance.
