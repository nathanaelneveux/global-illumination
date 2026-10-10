# House-turn temporal regression: 2026-10-09

## Conclusion

The recorder reproduces the reported regression. The nearby emissive cave
exposes an over-broad history invalidation rule: discovering emitting geometry
resets **all** receivers' measured history, including unchanged house roof faces.
The shared-corner display faithfully follows a newly noisy estimate; this is not
primarily a corner reconstruction or render-publication regression.

After the 30-second turn, a representative roof receiver `(0,17,-18)` changes:

- Older `ad6ba8e`: 1,100 samples to 260 (256 retained plus four new paths).
- Current normal world: 1,100 samples to 4 (all measured history discarded).
- Current without cave: 1,108 to 260.
- Current cave with reset suppressed: 1,184 to 260.

In the current world this reset occurs at 31.52 seconds with a source revision
3 → 4, while receiver geometry revision remains unchanged. Its filtered red
value falls from approximately 0.0278 to 0.0233, then rises again. At the west
roof receiver `(-5,17,-18)`, red drops about 0.0144 → 0.0087 then returns toward
0.015; the newly measured values are much noisier still.

`publication::apply_scene_task` calls `invalidate_sources` and advances
`reset_source_revision` for any changed emitting material. Selection interprets
that global watermark as “restart every receiver.” This rule already existed
at the older checkpoint (publication refactor `5476aaec`); cave placement
`417f00b` makes it routinely trigger during normal-world streaming. Turning does
not directly clear history: newly resident emissive geometry does. Removing
the cave is a scene control, not a recommended fix.

## Controlled evidence

Apple M4, 10-core GPU, Metal, development profile, Bevy 0.19.1, shared Cargo
intermediates; offscreen 2560×1440, 60 Hz ScheduleRunner. `house` workload,
normal camera-driven terrain streaming, no additional background loader.
Preferred shared-corner sun and reconstructed indirect, textures on, baked AO
off; four paths/batch, three diffuse hits, reach 768, 4 ms transport target,
8,192 receiver cap, 75 ms display response, 50 ms upload interval, adaptive on.
These are quality diagnostics, **not interactive FPS or throughput wins**.

For 30–50 seconds, compressed-RGB excess variation across 100 observed corners:

| Configuration | Median | p95 | Minimum new paths per roof receiver |
| --- | ---: | ---: | ---: |
| Older checkpoint | 0.0423 | 0.0703 | 724 |
| Current, cave present | 0.0963 | 0.1463 | 720 |
| Current, cave omitted | 0.0407 | 0.0691 | 728 |
| Current, cave present, reset suppressed | 0.0418 | 0.0716 | 732 |

The current normal world has 2.28× the median and 2.08× the p95 variation of
the older checkpoint. The same-world reset ablation reduces median variation
by 57%, back to the older range. The measured-estimate median (before display
smoothing) changes from 0.0866 old to 0.3909 current; display and mirror scores
closely agree. This localizes the excursion upstream of render publication.

Each configuration has one run; this is a large, controlled diagnostic effect,
not a repeated estimate of a small speedup. All four runs completed with seven
valid, nonblank target captures, verified assets/dimensions, and no render errors.
Representative old/current captures were inspected. The 25 receivers/corners
are spatially correlated, not 100 independent trials. No independent house
reference was calculated, so these scores do not certify absolute accuracy or
convergence. This establishes the house-turn failure, not every possible noisy
edit/tunnel/cave case.

### Reproduction identities (all artifacts ignored under `target/`)

- Baseline: `6b3094c`, `codex/temporal-baseline`, based on `ad6ba8e` plus the
  identical recorder only. Worktree `temporal-baseline/global illumination`;
  `house-1791600552042363000-64840`, log `temporal-baseline.log`.
- Current: binary `dddf594` (recorder on `00142b9`); at launch `f510095` added
  only the Python analyzer. No algorithm diff. Worktree
  `temporal-regression/global illumination`; `house-1791600415715578000-64326`,
  log `temporal-current.log`.
- Cave-omitted: `ede5983`, verified rebuilt binary, with
  `VOXEL_GI_VALIDATE_HOUSE_ONLY=1`; `house-1791600739114845000-65621`, log
  `temporal-house-only-verified.log`. Source-only build-identity guard was edited
  while this binary ran and was **not** in the executable.
- Reset ablation: `50e9c07`, `codex/temporal-reset-ablation`, cave present;
  `house-1791600901479051000-66250`, log `temporal-reset-ablation.log`.
  Build-root guard passed. The only transport change is suppression of the
  additional emissive-source reset; geometry invalidation remains active.
  Python analyzer/gate edits during this run do not affect the executable.

The ablation deliberately leaves `changed_materials` unused (one compile warning).
**Do not merge it as a fix**: it can retain stale emitter-removal energy. The
working diagnostic branch restores normal production invalidation afterward.

One attempted cave-omitted run, `house-1791600661863101000-65356`, reused the
baseline package binary from the shared Cargo build cache. It was terminated
and discarded before completion. The first filtered test also matched zero
tests; `ede5983`'s initial verification claim was corrected in `07a9ba5`.
After invalidating the application artifact, the exact house equivalence test
ran (one test, passed), the executable recompiled, and the valid replacement
run above explicitly reports `house_only_control: true`. The optional compile-
root guard now catches cross-worktree binaries; it is not a commit checksum.

## Next production change

Separate rejection of obsolete in-flight source work from receiver history
retention. Preserve spatial emitting-change information through scene building;
use the existing distance/history machinery to reset genuinely affected
receivers and retain bounded history elsewhere. Material/model-wide changes
must still invalidate globally. All receivers still refresh—distance must not
become a hard lighting influence cutoff.

Validate that policy against this stability gate **and** emitter on/off response,
including removals, streaming and all transported bands. Simply removing the
global reset or making every display filter slower is not the accepted fix.
No production transport/filter settings were changed by this investigation.

## Final harness verification

Five Python tests passed. The CLI gate (`corners`, 30–50 seconds, p95 ≤ 0.10,
at least 32 new paths at every receiver) returned exit codes **0 / 1 / 0 / 0**
for older / current / cave-omitted / reset-ablated runs respectively. This is
a regression reproduction, not a production fix or an absolute-accuracy pass.
The production-policy diagnostic branch was restored and rebuilt successfully.
An intentional wrong-build-root launch exited with the expected validation error
before gathering evidence (`target/temporal-build-guard.log`); no app processes
were left running. The user's original detached `ad6ba8e` checkout is unchanged.
