# Far per-pixel tier (experiment, 2026-10-25)

Branch `experiment/far-per-pixel` from 4c9e065. Question: does lighting
small far faces per pixel, NAADF style, instead of with view receivers,
solve distance? `direction.md` planned a per-pixel far tier; what was built
(`far-view-runtime.md`) is a reduced grid of exact face identities, so
per-pixel had never been tried.

## What the branch does

- **Ownership.** A face whose block projects under `pixel_tier.face_pixels`
  pixels of area (side squared times its cosine to the view ray, at the
  block centre's view depth) is per-pixel. Discovery, the per-pixel pass
  and the material compute the same test from the same camera header, so
  they agree per pixel. Per-pixel faces own no receiver.
- **Indirect.** Each `cell`x`cell` block of pixels samples one pixel per
  frame (cycling), reprojects a screen-space history by world position
  (bilinear over previous cells; same face orientation, within four blocks
  along the normal and two cells across), and queues itself for a path when
  its history is under 4 or one frame in `rate`. A dense indirect dispatch
  traces the queue with the existing `trace_receiver` estimator under a
  `steps` ray budget where an exhausted ray is a miss (NAADF's rule,
  `PATH_STEPS` / `EXHAUSTED_IS_SKY` pipeline constants on this kernel only).
  History caps at `history` samples.
- **Material.** Per-pixel faces read a 3x3 cell filter (coplanar first,
  parallel planes within four blocks, weighted by sample count and
  distance), and trace one sun ray per light per fragment.
- `config/gi-pixel-off.ron` (`face_pixels: 0`) is the in-build control: it
  skips the pass and leaves discovery as on 4c9e065.

Not done: custom direct RGB (`params.transport.y`) is not added to per-pixel
cells; no edit or sun-change retention (history is at most 32 samples); no
flicker or reference measurement.

## Receipts

M4, paced headless 2560x1440, one run each unless noted, all in one build
(control via `gi-pixel-off.ron`). GPU ms are view-tier receipts (the
per-pixel pass is its own stage); "presentation" excludes it.

The first finding: **with BVW LoD bands, far faces barely get small.** At
45 degrees and 1440p a unit face is under 4 px across beyond about 430
voxels; the lab's bands double the stride as distance doubles from 256
voxels, so block faces stay 4-9 px across. Only grazing faces are small in
area, hence the area test.

Lod flight (200 chunks, strides 1-32, fast flight; two log samples each):

| | Off | 4 px² | 16 px² | 16 px², rate 4, 3-px cells |
| --- | --- | --- | --- | --- |
| Active receivers | 41k, 39k | 65k, 57k | 26k, 21k | 26k, 22k |
| Discovery | 9.0 ms | 3.9 | 3.0 | 1.6 |
| Presentation | 19.5 ms | 21.1 | 8.9 | 6.1 |
| Per-pixel pass | - | 4.1-8.5 | 24-27 | 6.1-9.3 |
| Owned cells | - | 142-226k | 422-431k | 187-192k |
| Uncovered faces | 28.5k | 4.9k | 83 | 465 |
| Frame p50 / p95 | 91 / 272 ms | 86 / 231 | 88 / 236 | 88 / 234 |

At 4 px² the pool stops saturating and admits 57-65k mid-distance faces,
whose reconstruction doubles (12-14 ms). At 16 px² with quarter rate and
3-px cells the view tier totals about 19 ms against 33 ms off. The flight is
host-bound, so frame p50 moves little. History stays short there (mean 6-8
samples, 89% of cells reprojected): the camera moves fast. A 64-step budget
saved only 8%; ray length is not the cost.

Far 48 (bands 1/2/4/8):

| | Off | 16 px², rate 4 | same, 3-px cells |
| --- | --- | --- | --- |
| Frame p50 / p95 / p99 | 21.8 / 27.2 / 29.6 | 24.0 / 27.9 / 29.4 | 22.9 / 26.9 / 28.9 |
| Active receivers | 11.9k | 9.6-9.8k | 9.8k |
| Per-pixel pass | - | 3.4-4.4 ms (10.5k traced) | 2.0-4.1 ms (6.5k traced) |
| Presentation | 3.15 ms | 2.97 | 2.72 |

Here the tier owns few cells and costs more than it saves: about 300 ns per
traced path, against about 70 ns in the flight. Compacting the trace into a
dense queue did not change that, so it is the paths themselves (long grazing
horizon rays), not idle lanes.

Default lab (10 chunks, full resolution), two pairs: frame p50 23.4 / 23.5
both ways, p95/p99 equal; the pass costs 1.2 ms for 5.6k owned cells and
presentation drops 0.3 ms.

Visual (lod flight, 20 s): comparable at a glance; per-pixel shaded slopes
are slightly darker and noisier. Not judged against a reference.

## Reading

The per-pixel tier's cost is bounded by the screen, as intended: in the
200-chunk flight it replaces a saturated receiver pool and cuts view-tier
GPU by about 14 ms. Where tiny faces are few (far-48, the default lab) it
is neutral to 1 ms worse, and with LoD bands it only takes grazing faces.
Open before adoption: per-path cost at the horizon, flicker in motion with
short histories, and the darker shading against a reference.
