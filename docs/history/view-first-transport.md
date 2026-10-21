# View-first receiver experiment

Baseline: `9e52b7a` on `gi-far-view-transport`, preserved in the managed
`view-first-baseline` worktree. This branch keeps the hybrid default unchanged.

## Question and controls

Visible receiver ownership and persistent bounce-cache residency are separate
decisions. Both view candidates use the same face-identified view frontend,
shared path estimator, geometry, materials, corner sunlight and display target.
Only cache residency and the division of the existing transport allowance differ
between the two view-first candidates:

| Configuration | Visible owner | World cache radius | World / view budget |
| --- | --- | ---: | ---: |
| Unchanged hybrid | Near world + far view | 64 | 75% / 25% |
| `config/gi-view-cached.json` | View everywhere | 32 | 25% / 75% |
| `config/gi-view-cacheless.json` | View everywhere | None | 0% / 100% |

Radii admit whole 32-voxel pages by nearest distance, with the existing
32-voxel retirement hysteresis. They are not exact spherical face cutoffs.
The total transport share remains 20%. Presentation is measured separately.

View-first does not build or upload the world display directory and does not
build world presentation donor lists. Cached mode retains world probes, raw GPU
statistics, scheduling, sky measurements and edit-flux measurement for secondary
ray termination. Cacheless admits no world probes and submits no world sampling.
The zero-face worker still publishes geometry, sources and edit packets; its
resident event processor and binding-sized empty cache buffers are infrastructure,
not a populated bounce cache. Without outgoing-light observations an edit's flux
is explicitly unknown, so retention uses the existing distance curve rather
than falsely treating no observations as a measured zero change.

This is not spatial path reuse, a new sampler or a new calibrated accuracy
threshold. View histories remain visibility-dependent, and reconstruction remains
the existing bounded coplanar view filter. CPU world-display diagnostics do not
measure view-first presentation; use actual captures and view GPU counters.

View-first can spend its allowance on repeated sequential visits to a visible
face (up to 32 three-hit waves per owner per dispatch, at most 65,536 visits
overall). The feedback target is its frame's transport share, capped at 4 ms;
actual asynchronous GPU receipts repay the same allowance. Hybrid retains its
one-visit policy. A native furnace test compares 128 one-visit dispatches with
16 eight-visit and four 32-visit dispatches: receiver sequence, last path, pending
tail and raw moments are bit-identical. There is no display filtering in this
check. This removes an incidental one-path-per-frame limit, not path termination.

## Qualification

Pending. Start with matched cave, house turns/returns and edit-response captures,
with background terrain loading. Compare visible stability, reveal response,
lighting progress, memory and GPU costs before a windowed frame-pacing pair.
Do not promote either candidate based on counts or lower allocation alone.
