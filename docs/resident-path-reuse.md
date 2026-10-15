# Resident accumulation and fresh path reuse — October 15, 2026

Baseline `e3b46be` is preserved as
`codex/gi-baseline-residency-reuse-20261015`. All experiments are local. BVW,
retention, finite path depth/reach and the 75-ms RGB display filter are unchanged.
Sampling coverage/history is not visual convergence.

## Running the qualified variants

On `codex/gi-resident-qualified`, ordinary `cargo run` uses asynchronous resident
accumulation with the independent path sampler and RGB priority (nonvisual
importance 0.125). The historical synchronous backend is `VOXEL_GI_BACKEND=sync`;
add `VOXEL_GI_SETTINGS=config/gi-equal-band.json` to restore its old sampling
priority too. CPU fallback remains available. Reuse is deliberately opt-in
because its cost/benefit varies:

```sh
VOXEL_GI_BACKEND=reuse cargo run
```

`config/gi-rgb-priority.json` still pins the qualified priority explicitly on any
backend. The tuning panel's IR/radiation importance slider is live and preserves
lighting history. Explicit imported settings override the default. No preset
changes physical source strength, history retention, bounce depth, reach or
display smoothing.

## Ownership and simulation

Runtime residency (`fc5757b`, `4d4d22d`, `cf0001e`) owns live five-band means,
moments, direct light, sequence and display RGB in stable GPU face slots. The
renderer reads those slots directly; the worker polls at most three owned
snapshots. CPU simulation gets unfiltered values, revisions and measurement
ages without delaying the next lighting batch. Simulation may lag lighting.
There is no simulation tick policy or delivery-deadline guarantee yet.

The simulation bands need not meet RGB's perceptual threshold or sampling rate.
FLIP gates apply only to rendered RGB. Physical energy/PDF correctness and
freshness remain requirements for IR and radiation; their eventual accuracy
tolerances belong to the host simulation. No intensity, dose or temperature
threshold is inferred from a lighting screenshot.

The GPU record costs 128 bytes per face, in addition to the CPU field/index.
Render probes cost 256 rather than 208 bytes. A readback carries 96 bytes per
selected face; sky jobs add 32 bytes per corner. Readbacks currently accompany
transport because CPU scheduling still consumes the feedback. Residency is not
yet a claim of reduced total memory or demand-only simulation readback.

## Residency evidence

Apple M4 / Metal, 16 GiB, macOS 26.6.2, Rust 1.95, Bevy 0.19.1. Development
profile: application opt-level 1, dependencies 3. Offscreen 2560×1440, shared
corner sun, reconstructed indirect corners, textures on, baked AO off. Normal
cave structure with background terrain loading; 16 paths/batch, 4-ms target,
8192-receiver limit. No simultaneous GPU workload/compiler in these runs.

| Measure | Baseline pair | Resident pair |
| --- | ---: | ---: |
| Traced paths, millions | 129.09 / 125.94 | 187.35 / 178.13 |
| Worker p95, ms | 9.548 / 9.548 | 2.000 / 2.143 |
| Near measurement age p95, ms | 1086 / 1214 | 872 / 844 |
| Far measurement age p95, ms | 24815 / 26954 | 18257 / 18533 |
| Offscreen frame p95, ms | 26.100 / 26.254 | 24.949 / 24.469 |
| Offscreen frame p99, ms | 28.612 / 28.767 | 28.792 / 29.811 |

41–45% more transport and much shorter worker occupancy is a repeatable win.
Worst-frame tails are mixed. **These are not windowed FPS measurements and do
not establish the 60-FPS target.** The attempted windowed control captured black
frames and was rejected; its candidate was not run.

Evidence under ignored `target/gi-validation/`:
`cave-1792071479397255000-15515`, `cave-1792074241232481000-17139`,
`cave-1792073684624772000-16830`, `cave-1792074303788284000-17137`.
Resident native tests cover five-band parity, model restarts, unselected-face
caps, buffer growth, slot reuse, three owned overlapping readbacks and sky tickets.
Edits run `edits-1792073989969017000-16990` completed with 18 valid captures.

An important harness correction is `7372482`: resident rendering advances without
CPU display uploads, so the old upload mirror contained stale seed values. It
cannot measure resident corner stability. Corrected diagnostics use the latest
completed resident display snapshot; screenshots remain end-to-end evidence.
Earlier resident corner traces are invalid, not perfect stability.

## Reconnection architecture

Each cosine path records its first vertex and diffuse suffix. One geometric
partner can reuse the suffix after an explicit visibility check. Both receivers'
area-measure PDFs enter balanced weights. First-hit emitter energy is evaluated
with each receiver's own light-selection MIS; primary next-event estimation
stays local. Rejected/unknown/occluded connections retain the owner's sample.
No clipping, recursive cache termination or stale path history is introduced.

The current pair kernel keeps records in workgroup memory and evaluates each
connection once. Stable residency indices find partners without a per-batch
sort or extra hash table. Partners prefer three-voxel separation outside the
2×2 corner footprint, then local fallback. Sixteen lanes cooperate at 16 spp;
short sample tails participate uniformly in barriers. Zero-energy connections
can skip visibility only when both the suffix and first-hit emission are zero
in all five bands. This optimization was checked against unpruned means/moments.

This is fresh balanced path reuse, **not an implementation of ReSTIR PT**.
The relevant reference is [RTXDI ReSTIR PT](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/RestirPT.md)
and its [compatibility research](https://research.nvidia.com/labs/rtr/publication/junkins2026compatibility/).
NAADF traversal stays shared by tracing and connection visibility; the
[NAADF implementation](https://github.com/cg-tuwien/NAADF) and
[vxray devlog](https://nelari.us/post/voxel-ray-tracing/) remain traversal and
voxel-transport references, not evidence that our reconnection policy is optimal.

## Branch decisions

| Branch after `codex/gi-` | Checkpoint | Outcome |
| --- | --- | --- |
| compatible-path-reuse | `47ae26a` | Full gather reduced noise, but 98.07M paths versus 187–189M resident; too expensive. |
| compatible-path-reuse | `8c259b9` | Reciprocal evaluation reduced duplicate work; superseded by workgroup kernel. |
| paired-path-reuse | `d2282a2` | Workgroup-only sharing; adjacent donors did not establish a whole-frame win. |
| paired-path-reuse | `55a9bf3`, `a0b8788` | Wider partners and 16 cooperative lanes lower reconstruction error/overhead. |
| quad-path-reuse | `8aaf912` | Correct four-proposal balancing, but roughly 83% more GPU work; not promoted. |
| shared-source-path-reuse | `031ec91` | Shared first-light candidates add too much cost for their small extra gain. |
| rotating-path-reuse | `c6dcdfe` | Rotation worsened 2048-path corner p95 to .02022 versus fixed .01739; not promoted. |
| visible-priority-reuse | `33fe8ac` | Explicit spectral sampling importance; independent full-image qualification below. |

## Fixed-image qualification of equal-band pairs

`c8a3087`, run `cave-1792079193680398000-20124`: same frozen audited scene and
2048-face batches, 16 paths per dispatch, four disjoint sequence windows. Two
independent 131072-path references pass the existing qualification: replica RGB
p99 agreement .8163%, max 1.5107%; eight independent cosine checks pass. The
32768-path preliminary reference failed qualification and is not acceptance
evidence.

The **unchanged user-calibrated** thresholds are FLIP mean .0274008632 and worst
32px tile mean .0729644448 (67 pixels/degree). Worst across four windows:

| Sampler / paths | FLIP mean | Worst tile | Outcome |
| --- | ---: | ---: | --- |
| Independent / 1024 | .036689 | .102968 | Fail |
| Independent / 1536 | .030605 | .082796 | Fail |
| Independent / 2048 | .027428 | .073501 | Marginal fail; do not relax the gate |
| Paired / 1024 | .030744 | .082163 | Fail |
| Paired / 1280 | .028133 | .083630 | Fail |
| Paired / 1536 | .025156 | .067725 | Pass, also passes later prefixes |
| Paired / 2048 | .022259 | .059530 | Pass |

Paired 1536 costs 3.686–3.772 seconds of whole-scene GPU work, independent 2048
3.559–3.691 seconds. Better quality at roughly equal work, not a demonstrated
settling speedup. These compute times are not interactive latency. Analysis:
`target/reuse-qualified-images.json`, `target/reuse-qualified-control-images.json`.

## RGB-prioritized sampling

`simulation_sampling_importance` is a sampling-quality control, not a change to
the light's energy. RGB importance stays one; IR/ionizing importance is clamped
to .01–1 so invisible sources retain support. The integrated default is .125;
one remains the explicit equal-band control. The proposal uses the maximum
importance-weighted band. All CPU/GPU reverse PDFs use the same power; the five transported energy values are
unchanged. Changing importance invalidates the proposal cache, not lighting
history. It does not promise eight times fewer simulation samples at .125.

In the cave, lava's IR is 9 versus visible peak 3, and thorium radiation is 15
versus visible peak 6. Equal-band proposals therefore spend more effort on
those sources than RGB alone requires. At .125, independent priority sampling
improves 1024-path corner p95 from .03204 to .02720; adding reuse improves it to
.02179 (baseline 2048 is .02233). Measured GPU work is 410.5 versus 639.1 ms in
the isolated 2935-face study, not yet a whole-image or gameplay qualification.
Source/cache unit tests and native five-band CPU/GPU selection/MIS parity pass
at importance 1, .125 and .01. Full-cave identical-path CPU assertions are not
appropriate: the equal-band control already has multi-bounce floating-point
path divergence. Independent high-sample references remain the accuracy oracle.

Evidence: `target/reuse-visible-priority-control.log`,
`target/visible-priority-parity.log`.

### Qualified full images

Native `33fe8ac`, run `cave-1792080416160127000-21077`, same hardware/settings
as above with nonvisual importance .125. Complete geometry audit and all capture
checks pass. The independent equal-band 2×131072-path reference qualifies (RGB
replica p99 .81634%, maximum 1.50981%; all eight oracle checks pass).

| Candidate paths | Worst-window FLIP mean | Worst-window tile mean | Fixed user gate |
| --- | ---: | ---: | --- |
| 512 | .033839 | .099822 | Fail |
| 768 | .028623 | .081927 | Fail |
| 1024 | .025271 | .076791 | Fail: local patch |
| 1280 | .023826 | .068416 | Pass |
| 1536 | .022336 | .064828 | Pass |
| 2048 | .020444 | .061399 | Pass |
| 4096 | .016680 | .047847 | Pass |

1280 is the first tested sustained pass across all four windows: 37.5% fewer
paths than the earlier accepted 2048 visual target. It takes 2.990 / 3.100 /
3.189 / 3.194 seconds of whole-scene GPU work versus 3.782 / 3.680 / 3.743 /
3.629 at independent 2048. That is 12–21% less work (about 16% mean saving),
not a 37.5% runtime speedup. The current independent 2048 control is marginally
outside the fixed tile gate, as in the prior run; it is not relabeled a pass.
These are frozen-scene compute totals, not gameplay response times.

At those respective sample counts, worst-window face-center band p95 relative
errors are IR 2.03% versus 1.86%, ionizing 9.59% versus 8.01%, normalized by
`max(abs(reference), .02)`. This is the explicit accuracy tradeoff, not a failed
RGB gate or a claimed simulation accuracy guarantee. The physical estimates
remain unfiltered and keep receiving samples. Analyses:
`target/reuse-priority-qualified-images.json` and
`target/reuse-priority-qualified-control.json`.

### Generality and limits

Native `1e60213` cross-scene harness (400 dense floor receivers, four disjoint
windows, independent 32768-path reference) checks equal .125 spectral priority
on both arms. Whole image FLIP is not inferred from these sparse field errors.

| Fixture | Independent / paired corner p95 at 1024 | 16-path GPU median, ms |
| --- | ---: | ---: |
| Open sky | .0002474 / .0001720 | .07829 / .13096 |
| Black emitter room | .003662 / .003242 | .15408 / .18154 |
| Colored skylight room | .2031 / .1409 | .26404 / .33233 |

The skylit room's paired 1024 error is approximately independent 2048's .14091
for about 63% of the work. All tested checkpoints improve accuracy, but the
already-clean open-sky case pays ~67% more per batch for negligible visible
benefit. Reuse is a demonstrated complex-lighting quality/budget winner, **not
a universal throughput win**. Keep independent resident transport available;
do not make reuse the universal default from cave evidence alone.

The .125 importance control is independent of reuse and can be tuned live.
No simulation tolerance/tick-policy change is bundled with it. The paired
kernel and tests are retained; obsolete two-pass scratch/pipelines were removed
at `1e60213`. 255 regular tests and strict all-target Clippy pass; native zero
pruning preserves five-band means/moments after cleanup. Cost of that exact
pruning optimization varies across short runs; no separate speedup is claimed.

## Background-streaming gameplay pair and default decision

Native `be21e2e` with documentation-only dirty changes. Same preferred look,
resolution, hardware and limits as above, no reference-only residency loader.
Resident control uses importance one; candidate uses reuse/.125. These paired
runs qualify the combination; the independent native study separates its parts.
All captures are valid/nonblank, assets present and render logs clean.

| Cave measure | Independent resident | RGB-prioritized reuse |
| --- | ---: | ---: |
| Paths in 62-second workload | 190.07M | 158.78M |
| Near age p95, ms | 854 | 856 |
| Far age p95, ms | 16482 | 21189 |
| Worker p95, ms | 2.198 | 2.620 |
| Offscreen frame p95 / p99, ms | 25.038 / 30.356 | 24.363 / 29.887 |
| 10–30s corner excess variation p95 | .06227 | .04129 |
| 30–60s corner excess variation p95 | .04610 | .02873 |

The cave's corner oscillation falls 34–38%, with continuing fresh samples and
essentially unchanged near-camera age, but distant refresh is slower. In the
90-second house-turn workload, paths are 241.95M versus 206.16M. Corner excess
variation p95 improves .02665→.01987 (30–50s), .01503→.01286 (50–70s), and
.01194→.00983 (70–90s). All 100 house corners / 144 cave corners are complete;
neither run is scored by starving transport. House frame/age tails are mixed.
Excess variation is a compressed solver-space stability measure, **not FLIP**
or proof of convergence. No windowed 60-FPS claim is made.

Evidence directories under `target/gi-validation/`:
`cave-1792081181637146000-21640`, `cave-1792081244172305000-21732`,
`house-1792081306798925000-21763`, `house-1792081397408083000-21837`.
Analyses: `target/final-{resident,reuse}-{cave,house}-temporal.json`.
House final captures were inspected together; both show the correct textured
corner composition, without a presentation-mode switch.

Initial decision at `f35b197`: promote **independent resident accumulation** to
`Auto`, preserving `sync` and CPU controls. Retain RGB-prioritized reuse as a qualified opt-in for
complex lighting, not a claim of universal throughput superiority. Its next
performance opportunity is avoiding work on already-clean regions, not relaxing
visual gates or altering history. This is separate from simulation tick policy.

Default-selection verification: 256 regular tests pass, strict all-target
Clippy and application build pass. The no-backend-override offscreen edit smoke
`edits-1792081570658921000-22054` reports `gpu-resident`, 18 valid captures and
all five scripted edit stages in `response.json`, with clean logs. Its generic
`report.json` edit counters cover another workload and are not the event ledger
for this response fixture. No toggle stress was repeated; native buffer-generation
and slot-lifetime regressions were verified earlier for the residency changes.

## Independent RGB-priority promotion

Following the user's acceptance, promote nonvisual importance 0.125 for all
backends, leaving independent resident transport as the default and reuse
opt-in. Baseline `f35b197` and the earlier experiment branches remain available.
The algorithm, emitted energies, history policy, scheduler and GPU layout are
unchanged. Invalid nonfinite importance now falls back to the same default;
explicit settings at one still select equal-band proposals.

The attribution study above isolates a roughly 15% reduction in 1024-path cave
corner p95 error from priority alone (.03204 to .02720). That is a native
field-error result, not a separately established full-image FLIP threshold or
gameplay speedup. The 1280-path image pass and 12–21% lower GPU work belong to
the **combined priority-plus-reuse** candidate. No new timing gain is claimed
for changing this default. Existing native five-band parity at 1/.125/.01 and
the recorded gameplay/image runs qualify the unchanged proposal implementation;
the promotion reran three settings/preset tests and eight source/cache tests:
all eleven pass. `git diff --check` passes. Native GPU/image studies were not
rerun for this settings-only change.
