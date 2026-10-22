# Two-way view/world RGB cache

Decision, 2026-10-22: **keep the two-way design in the +cache experiment** on
`codex/two-way-face-cache`. It substantially improves reveal/return stability,
preserves the cave's appearance and keeps offscreen emitter response responsive.
Do not replace the hybrid default yet: windowed frame pacing is unqualified.

Run it with:

```sh
VOXEL_GI_SETTINGS=config/gi-view-cached.json cargo run --locked
```

## Design and boundaries

The visible and background schedulers share one authoritative raw RGB record
per resident face. First admission transfers the better-supported valid estimate
without summing overlapping histories; subsequent visits always load the
authority, including its edit-retained count. View completions update that same
record, and newly revealed faces can import it. A separate display prior avoids
cold presentation without turning reconstructed support into physical evidence.

The residency radius remains 32 voxels, with 32-voxel retirement hysteresis;
the total transport share and 25% world / 75% view split are unchanged. This
does not add persistent records for all loaded terrain or implement spatial
path reconnection. IR/radiation is outside GI's gameplay contract.

GPU generation guards reject stale render writes across maintenance. Both
schedulers advance one sample sequence; unfinished admitted paths keep their
reservation. Unstarted reservations are released within the same command buffer.
An independent, pre-admission tail is cancelled when importing another prefix,
since it may duplicate a sample already in that history. There is no lighting
CPU readback/merge or player-thread wait. See the full
[ownership contract](../architecture.md#experimental-two-way-face-cache).

## Qualification

Control: `913ab35`, preserved in the managed `two-way-cache-baseline` worktree.
`c3494f8` adds only the opt-in cache-admission hallway workload to that control;
house, response and cave transport are identical. Candidate implementation:
`a4384a5`, rank-refresh cleanup `c21e4ee`, sequence/tail correction `bd12f0e`,
and independent admission-tail guard `594f37b`.

M4 / Metal, 16 GB, rustc 1.95.0; development opt-level 1, dependencies opt-level
3; 2560×1440 offscreen targets, background terrain loading, shared-corner sun,
corner indirect, textures on, baked AO off, uncapped 1024-evidence display.
No other game, build or image-analysis job ran during measured phases. The
candidate's dirty changes were documentation and formatting of `handoff.rs`
only; those do not change the workload. Native tests/builds ran separately.

All accepted runs have valid varied captures, the intended camera/target and
clean render logs. Images were inspected as well as measured. Official FLIP
1.7 at 67 pixels/degree compares each early capture with its own later capture;
these are **remaining visible change, not accuracy against a converged oracle**.
House uses the upper 40% (ceiling), excluding the outdoor doorway. Other profiles
use full images. No raw sampler threshold or reconstruction target is changed.

| Remaining change, mean FLIP | One-way control | Two-way |
| --- | ---: | ---: |
| House, 400 ms after turn | .03837 | .01032 |
| House, 400 ms after return | .04611 | .00429 |
| Cave, 15 s → own 60 s | .01039 | .01000 |
| Cave, 30 s → own 60 s | .00624 | .00609 |
| Visible cache admission, 400 ms → own later | .01757 | .01266 |
| Visible cache admission, 1 s → own later | .00801 | .00975 |
| Visible cache admission, 2 s → own later | .00568 | .00470 |
| Offscreen emitter add, 1 s (two runs) | .02086 / .02904 | .01143 / .01279 |
| Offscreen emitter remove, 600 ms (two runs) | .02208 / .15636 | .02455 / .01651 |
| Offscreen emitter remove, 1 s (two runs) | .00884 / .07916 | .00634 / .00785 |

House reveal and return improve 3.7× and 10.7× at the captured 400 ms offsets.
These are not continuous settle-time measurements. The final cross-run cave
difference is .00565 mean FLIP; displayed mean luminance differs by 0.034%.
Admission is mixed at individual early offsets, not a universal speedup:
the important contracts are successful transfers (1,045 promotions counted),
no cold clear, and continuous visible history. This is not a claim that the
unchanged hybrid's boundary handoff has been fixed.

Emitter removal at 300 ms was worse in the first pair (.31355 vs .25212), but
better in the repeat (.30710 vs .48468). By 1 s both candidate runs are close to
their own four-second final image. The baseline varies more; do not claim a
precise removal speedup or a repeatable early regression from these pairs.
Final response images differ by mean FLIP .00529/.02554 (off) and .01733/.02472
(on). These are live finite-time comparisons, not certified physical solutions.

The new GPU exchange passes cost median .059 ms house, .093 ms cave, .056 ms
response and .066 ms admission (logged samples; full range .042–.113 ms).
They add at most 256 KiB of per-camera sequence scratch, no second persistent
face history. Whole-process peak MB: house 976→981, cave 966→967, response
937→942 / 941→944, admission 915→930. These allocation peaks vary with streaming
and scheduling; they are not a static-memory accounting of the bridge alone.

| Offscreen app cadence p95 / p99, ms | Control | Two-way |
| --- | ---: | ---: |
| House | 25.36 / 27.09 | 25.24 / 27.78 |
| Cave | 26.28 / 27.49 | 25.97 / 27.04 |
| Admission | 25.46 / 26.92 | 25.36 / 26.80 |
| Response repeat | 25.18 / 25.93 | 24.95 / 25.84 |

No material offscreen cadence regression is apparent; no small performance gain
is claimed. The windowed admission attempt at `594f37b` stopped on its first
capture (`focused=false`, `varied=false`). It is excluded, not an FPS pass.

## Correctness gates and rejected intermediate designs

- Five native ownership tests cover adoption, non-double-counting, raw/display
  separation, material identity, stale generations, retained weights and pending
  change exclusion from bounce lookup, alternating schedulers and long tails.
- A raw analytic furnace reaches RGB `(1.000815, 2.891734, .666205)` versus
  `(1, 2.888889, .666667)`, under 0.1% worst relative error after 172,442 sequence
  indices. It uses no reconstruction. The count also proves that no unstarted
  reservation becomes a sample or a path-length-dependent sequence gap.
- Restarting unfinished view paths whenever background advanced the cursor was
  rejected: it preferentially discarded long paths. Reserving indices fixed
  ownership, but keeping unused reservation suffixes failed the analytic gate
  (~1.7% blue error). Returning that suffix fixed the new experiment's defect;
  this is not a newly discovered defect in the baseline sampler.
- Raw view-batching parity, single/MSAA runtime pipelines, headless material
  generation integration, 244 regular tests, production Clippy `-D warnings`
  and development build passed at `bd12f0e`. At `594f37b`, the extended native
  admission/pending-tail test, shader composition and build passed; unchanged
  analytic/ownership evidence is reused. The guard only cancels an unowned
  first-admission tail, not already-admitted pending work.
- A test harness initially encoded thousands of compute passes into one Metal
  command buffer and lost its test device. The analytic test now submits bounded
  batches like the runtime. Failed runs were not used as rendering evidence.

## Reproduction and local evidence

Use the commands in [validation](../validation.md), selecting `house`, `response`,
`cave` or `hallway` with `VOXEL_GI_VALIDATE_CACHE_HANDOFF=1` for admission.
`tools/view_cache_quality.py` generates paired pictures and JSON from actual
material captures, not CPU world-field diagnostics. Large artifacts stay ignored.

Run directories below are under each worktree's `target/gi-validation/`:

| Profile | Control run | Candidate run / commit |
| --- | --- | --- |
| House | `house-1792675968064839000-14521` (`913ab35`) | `house-1792678688078865000-17146` / `bd12f0e` |
| Cave | `cave-1792679226901199000-18710` | `cave-1792678872456592000-17781` / `bd12f0e` |
| Admission | `hallway-1792676727882402000-15590` | `hallway-1792679083432038000-18208` / `594f37b` |
| Response 1 | `response-1792677177270594000-16273` | `response-1792678808925171000-17593` / `bd12f0e` |
| Response 2 | `response-1792679290399488000-18697` | `response-1792679351662110000-18982` / `594f37b` |

Other controls are `c3494f8`. Summaries/images are in the candidate's
`target/two-way-final-comparison/` and `target/two-way-repeat-comparison/`.
Logs are `target/two-way-{final,repeat}-*.log`; native gates use
`target/two-way-{native,view,material,unit,clippy}-final.log` and
`target/two-way-admission-tail.log`. Earlier preliminary pairs are retained but
are not substituted for the measurements above.
