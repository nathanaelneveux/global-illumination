# Sun-aware, sampling-calibrated retention — October 13, 2026

## Decision and scope

**Keep the combined change signal and the conservative retention candidate. Do
not promote the aggressive reset curve or first-innovation variance guard.**

The sunlight-change signal fixes the distant-shadow counterexample from the
[distance study](distance-history.md). A retention-only constraint preserves
existing local response while retaining the large remote-cave stability win.
This original study is a qualified **offline policy experiment**. Its subsequent
[runtime follow-up](runtime-retention.md) adds a qualified runtime policy with
durable spatial metadata and incremental sunlight-change evidence. Unknown or
late sky evidence remains conservative and explicit; partial cache coverage is
not silently declared complete. The original offline measurements below do not
themselves qualify runtime cost or appearance.

Branch: `codex/gi-sun-aware-retention`. Main policy recording: clean `cb28ac7`.
The [user-calibrated follow-up](#user-calibrated-follow-up) uses the subsequently
accepted 2,048-path appearance; the original measurements below are retained.

## Separate what changed, sampling quality, and the response policy

1. **Spatial evidence:** changed geometry bounds plus positions of stable
   surfaces whose measured shared-corner sunlight changed. Both becoming lit and
   becoming shadowed count. Known black is not unknown. The experiment uses the
   actual `SkyCache` and `sample_sky_corner`; it does not infer a sunlight change
   from an offline reference or from future path samples.
2. **Sampling calibration:** measured paths `Q` needed to meet a fixed quality
   goal on a separate, qualified reference scene. `SamplingCalibration` recomputes
   this from all four complete sample-sweep windows, requiring that higher tested
   prefixes continue to pass. Current raw-face RGB p99/max goals are 5%/10%, and
   the [composed cave sweep](../cave-sampling.md) gives **Q = 3,072**. This is not a
   perceptual score or an IR/radiation quality certificate.
3. **Change prior:** `D(d) = (48 / max(48, d))²`, using the smaller geometry or
   changed-sunlight distance. This is an experimental normalized change estimate,
   **not a physical upper bound**. Extent, source power and secondary visibility
   can invalidate it. There is no universal 384-block cutoff.
4. **Bias budget:** allow stale contribution `epsilon = 3%` after fresh work
   `N = Q / 6`. For `D > epsilon`, the candidate cap is
   `H_bias = N*epsilon/(D-epsilon)`; otherwise retain available history.
5. **Qualification constraint:** the accepted experimental arm, `sun-retention`,
   at the current calibration does not cut history more aggressively than the
   already-qualified baseline.
   Its floor is `Q*(16/3072)` within 48 blocks of changed geometry, otherwise
   `Q*(256/3072)`. Select `max(H_bias, floor)`, capped by available history.

The last constraint applies to all events, not a cave/emitter/skylight exception.
It deliberately isolates **better retention** from the separate experiment of
**faster resets**. When the floor exceeds the bias cap, the requested bias-response
budget is not satisfied; the report counts these conflicts. A larger retained
weight is not a mathematical guarantee against overshoot either. Actual native
transition replays are the acceptance evidence.

No radiance clamping, debounce, new transport rays, or changes to sampling PDFs
are involved. Caps only reduce the effective weight of the existing mean. The
mean, second moment, path sequence and normal fresh sampling remain intact.
Fractional caps are allowed in this offline model; production integer-weight
rounding has not been qualified.

### Adapting to a better sampler

The fixed inputs are the quality and response goals, not an unversioned table of
block distances and history counts. Re-run the held-out quality sweep when the
estimator changes, record its sampler/model identity and report hash, and import
the new `Q`. Both the bias curve and the qualification floor use this work scale.
Then re-run the same transition controls before changing defaults.

For example, if a new sampler reaches the **same** quality gate in 1,536 rather
than 3,072 paths, illustrative caps for a 4,096-weight cache become:

| Known change distance | Current calibration | Hypothetical 2× sample efficiency |
| --- | ---: | ---: |
| Local geometry, within 48 | 16 | 8 |
| Geometry at 96 or 128, no changed sun nearby | 256 | 128 |
| Geometry at 192 | 473 | 236 |
| Geometry at 256 | 2,979 | 1,489 |
| Geometry at 384 or 512, no changed sun nearby | Keep 4,096 | Keep 4,096 |
| Geometry at 512, changed sun surface nearby | 256 | 128 |

Numbers are rounded for display, not the replay. The full-retention boundary also
depends on available history: for the bias prior above it occurs when
`D <= epsilon*(1+N/H_available)`. It is derived, not a fixed radius. Improving a
sampler does **not** shorten the physical reach of a shadow or turn the prior into
a transport bound. Twofold sample efficiency and twofold lower RMS error are
different claims; use measurements, not an assumed square-law rescaling.

The existing single-path second moment is insufficient calibration for a better
correlated/QMC sampler. Its single-path distribution could remain the same while
its accumulated error improves. The hypothetical column is an API sensitivity
check only: we have not invented an improved sampler or measured its performance.

## Paired results

The existing 12 distance fixtures reuse exactly the same native before/after
paths, two live prefixes, both event directions, and references described in the
distance study. Changed-sun metadata is recorded separately. Every stable face
corner in these fully populated queries is checked for known old/new values.
New or removed surfaces remain outside this paired signal.

At each roof height 64/128/256/384/512, the sun-bounce fixture produces 3,969
changed corner/face target points on the nearby floor. All 16 evaluated receivers
therefore see a nearby lighting dependency even when the edited roof is distant.
The canyon slab and remote cave roof produce **zero** changed sunlight targets.
The retained evidence remains selective rather than automatically global.

Worst result across both prefixes/directions, equal fresh transport work:

| Control / metric | Baseline | `sun-retention` |
| --- | ---: | ---: |
| Repeated remote cave: late RGB p99 | 8.785% | **2.179%** |
| Repeated remote cave: late variation p99 | 2.94233 | **.09369** |
| Repeated remote cave: maximum bright excursion | 16.802% | **1.685%** |
| Repeated remote cave: maximum dark excursion | 12.063% | **1.948%** |
| Repeated remote cave: sustained all-face 5% gate | Never within 8,192 paths | Inside from the start |
| Roof at 512, sun bounce: late RGB p99 | 6.229% | 6.229% |
| Roof at 512, sun bounce: sustained all-face 5% gate | 4,912–5,584 paths | Identical |
| Canyon slab: late RGB p99 | 1.885% | 1.813% |
| Canyon slab: sustained all-face 5% gate | 2,080–2,880 paths | Identical |

The cave improvement is **4.03× lower late p99 error and 31.4× less accumulated
late variation**. It does not mean 31× faster convergence, 31× less rendered
flicker, or higher gameplay FPS. This is the sparse 121-receiver remote-streaming
control, not a claim that the complete composed cave's noise is solved.

All sun-bounce replays match baseline, including its existing behavior at height
64. At heights 128–512 the baseline has no new bright overshoot in these prefixes.
The canyon's 26.80%/25.29% early bright/dark excursions also remain unchanged.
Small canyon differences do not establish a meaningful speedup.

Gentler retention does slightly increase late error for diffuse roof occlusion:
at height 256, p99 rises from .193% to .402%; at 384, .203% to .275%; at 512,
.141% to .197%. All evaluated faces remain within the 5% gate throughout, and
variation and new excursions decrease. This is a small measured bias/stability
tradeoff, not bit-for-bit equivalence or proof of zero quality cost everywhere.

The supplementary [legacy controls](history-edit-guided.md) include additions,
removals, covering an emitter, hidden emitters, sunlit skylights and diffuse-only
skylights. **All 33 case/prefix replays of `sun-retention` match baseline exactly**
in every reported metric and trajectory checkpoint. That preserves both baseline
strengths and its known weaknesses; it does not fix the rare-opening noise.
Their older 10%-p95/floored metrics and weaker rare-skylight references must not be
numerically conflated with the new all-face 5% gate. They are reused source
regression controls, not freshly recorded native performance measurements.

## Rejected arms and what they teach us

- **Sun-aware quartic / unconstrained bias cap:** correctly catches the distant
  roof and can reach the 5% gate in 304–1,104 rather than 4,912–5,584 fresh paths.
  But aggressive local weighting introduces about **12.5–12.9% bright overshoot**
  on sunlight restoration. The unconstrained bias curve also makes the remote
  hidden-emitter null control noisier by cutting below its baseline 256 weight.
- **First-innovation variance floor:** using the first differing batch's moments
  to protect the update removes that roof-restoration overshoot. However, emitter
  addition slows from **160 to 1,152–1,168 fresh paths**, and diffuse-skylight
  behavior can worsen. A single early batch is not a reliable universal noise or
  change estimate. Keep the arm reproducible, but do not promote it.

These results separate a successful dependency signal from an unsuccessful reset
policy. They do not justify a sunlight-specific clamping or emitter-specific
escape path. Reducing fresh-sample variance remains the next independent lever.

## Runtime integration contract — not implemented by this study

- Publish changes incrementally from existing shared-corner updates, carrying
  known/unknown state and scene/source revisions. Do not rebuild all surfaces or
  run this fixture recorder in the game. Coalesce bounded spatial change metadata
  off the main thread; consuming events must not block the player.
- Preserve remote-change evidence across unrelated stream revisions. The current
  eight-transition journal can otherwise fall back to cap64 and erase the win.
  Missing history needs a conservative, explicit fallback, not assumed distance.
- Handle partial cache coverage, newly exposed/removed surfaces, several changes
  before a receiver refresh, and event arrival after a transport dispatch. Do not
  mark a receiver fresh before its relevant change evidence has been processed.
  Continue rejecting old-scene jobs; retention must not suppress wakeups.
- Qualify larger changed areas, source strengths and multibounce visibility.
  Changed direct sunlight covers one important dependency, not every remote
  dependency. The current shared-corner cache is visible directional light, not
  a complete point-sun, diffuse-sky, infrared or ionizing change detector.
- Version calibration with the sampler/transport model and retain an explicit
  quality goal per consumer. RGB acceptance does not certify simulation bands.
  Requalify native rounding, streamed cave/canyon visuals and windowed frame-time
  tails before enabling the policy by default.

## Reproduction and provenance

```sh
VOXEL_GI_SUN_DISTANCE_OUTPUT=target/sun-distance-metadata.json \
cargo test --locked --offline record_sun_distance_metadata -- --ignored --nocapture

PYTHONPATH=tools python3 -m unittest distance_history_analysis sun_retention_study -v
PYTHONPATH=tools python3 tools/sun_retention_study.py \
  target/distance-history-tape.json target/sun-distance-metadata.json \
  --quality-report target/cave-sample-sweep-analysis.json \
  --sampler-id 3331e20-production-terminal-mixture \
  --legacy-tape target/history-edit-guided-tape-final.json \
  --output target/sun-retention-qualified-study.json
```

The sampler ID describes the measured calibration, not necessarily the checkout
running the Python replay. NumPy is required. Large tapes/reports stay ignored
under `target/`; their generating studies document reference qualification.

Native distance samples: clean `df3115d`. Sun metadata: clean `98d0513`.
Calibration: composed-cave binary `3331e20`, subsequent analysis documented in
`cave-sampling.md`. Legacy tape: `7154a01` plus the recorded `73d7e83` diff, as
documented in `history-edit-guided.md`. These are different recorded builds of
the same intended estimator, not a claim of one freshly recorded binary.

Hardware/profile: Apple M4, 16 GiB, Metal, macOS 26.6.2, Rust 1.95, Bevy 0.19.1,
registry BVW 0.17; dev opt1/dependencies opt3, shared Cargo build directory.
Reach 768 / three diffuse hits. No viewport, display filter or gameplay timing.
Full fixture sunlight reconstruction is CPU test setup, not incremental runtime
cost: the canyon alone takes about 3.4–3.6 seconds to construct/populate/query.

Verification: 13 focused Python tests passed at `cb28ac7`; all 52 replayed
baseline results match the previous distance report exactly. Six edit-guided
Rust tests and the distance geometry test passed. Strict test Clippy passed at
`45b193e`; later Rust changes only add the completeness assertion. That assertion
was verified by all 12 metadata fixtures in 3.68s on `cb28ac7` plus the exact
diff committed as `5299f1e`; output is byte-identical to the original metadata.
The 13 Python tests were repeated after extending the counterfactual sensitivity
report to include the retention floor. Transition policies/results are unchanged;
the recorded `cb28ac7` report predates that additional sensitivity-only field.
No unnecessary gameplay/stress run for test-only code.

Input SHA-256 values in `sun-retention-qualified-study.json`:

| Input | SHA-256 |
| --- | --- |
| Distance tape | `ed6ad0b71b212feb9bc6ce4f5492eaf08de42291056035c53fa004eabfd0cd40` |
| Sun metadata | `f0a6425058ff23faf2417eb4919bcc78bc9f80a7534b19b09e6572f247811856` |
| Calibration report | `5510a05941c03bac5954d6f83baf042989688cd3890d3b88241ff31c63a0eed5` |
| Legacy tape | `f2ffb02688a23ab1acea7c593e9185cd34694917d877365187c1b76d19e3ed8f` |

## User-calibrated follow-up

The user subsequently accepted the [2,048-path visual target](../cave-sampling.md#user-calibrated-visual-target).
Add `--quality-profile tools/quality_profiles/cave-user-2048.json` to the replay
command and use `--output target/sun-retention-user-2048-study.json`. Omitting
the option preserves the original raw 5%/10% calibration for reproduction.

Changing the goal is **not** evidence of improved sampling. Both measured work
`Q` and baseline work at the selected goal now equal 2,048. The qualified floor
therefore remains **16 nearby / 256 distant**, not 10.7 / 170.7. If a future
sampler meets this same fixed profile in 1,024 paths, its relative floor can then
halve, subject to transition qualification. Image, corner and raw-face gates
must all pass across complete windows and sustained larger prefixes.

The unchanged response fraction `Q/6` gives 341.3 rather than 512 fresh paths;
this retunes the bias curve without inventing a faster estimator. At geometry
distance 256 the candidate cap falls from 2,979 to 1,986. The conservative floor
still protects previously accepted near/distant reset responses.

Clean `7ecb0c7` replay, reusing the exact tapes and native provenance above:

- All 52 baseline results reproduce, all 20 sun-bounce results match baseline,
  and all 33 legacy edit/source controls match baseline.
- Repeated remote-cave p99/variation remain **2.179% / .09369**, versus baseline
  **8.785% / 2.94233**, at equal fresh sampling work.
- Diffuse roof at 256: late p99 **.337%**, versus the earlier candidate's .402%
  and baseline .193%. The bias/stability tradeoff remains; roofs at 384/512 stay
  .275%/.197%. All these faces remain within the original 5% gate.
- Canyon late p99 remains 1.813% versus baseline 1.885%, with identical settling
  and worst excursions. Its final-checkpoint p99 is 1.614% versus 1.482% baseline:
  the last checkpoint is slightly worse despite the pooled late-period gain.

The transition report retains its original 5% settling metric for comparable
measurements. Sparse raw-cache replays do not receive an image-quality pass by
comparing differently normalized percentages with a rendered-view gate. No
candidate gameplay-FPS or rendered temporal qualification is claimed.

Profile SHA-256:
`fb83f1fabef2640f8edcb0fe15c56b65d2379c577a0c3e5a69a67a159026db9a`.
Other input hashes are unchanged. Seventeen focused Python tests pass, including
same-sampler/different-goal versus better-sampler/same-goal behavior. No BVW,
shader, runtime history or gameplay scheduling changes were made.
