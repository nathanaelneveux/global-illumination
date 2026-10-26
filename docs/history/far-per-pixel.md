# Far per-pixel tier (experiment, 2026-10-25)

Branch `experiment/far-per-pixel` from 4c9e065. Question: does lighting
small far faces per pixel, NAADF style, instead of with view receivers,
solve distance? `direction.md` planned a per-pixel far tier; what was built
(`far-view-runtime.md`) is a reduced grid of exact face identities, so
per-pixel had never been tried.

## What the branch does

- **Ownership.** A pixel is per-pixel where a full-resolution voxel at its
  view depth projects under `pixel_tier.voxel_pixels` pixels across: a pure
  distance band (about 1,740 / `voxel_pixels` voxels at 1440p and 45
  degrees), whatever the LoD stride or viewing angle. (The first version,
  receipts below, used the block face's projected area.) Discovery, the
  per-pixel pass and the material compute the same test from the same
  camera header, so they agree per pixel. Per-pixel faces own no receiver.
- **Indirect.** Each `cell`x`cell` block of pixels samples one pixel per
  frame (cycling), reprojects a screen-space history by world position
  (bilinear over previous cells; same face orientation, within four blocks
  along the normal and two cells across), and queues itself for a path when
  its history is under 4 or one frame in `rate`. A dense indirect dispatch
  traces the queue with the existing `trace_receiver` estimator under a
  `steps` ray budget where an exhausted ray is a miss (NAADF's rule,
  `PATH_STEPS` / `BUDGETED_RAYS` pipeline constants on this kernel only).
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

## Update: horizon cost, distance knob, crossover sweep

**Horizon paths.** In far-48 a few thousand paths cost 2-4 ms whether rays
had 32 or 128 steps and 1 or 3 bounces; the classify scan alone was
0.5-0.7 ms. The cost was `geometry_ahead`: a ray entering a region missing
from the directory (the empty air above terrain) walked region by region,
up to 2,048 of them and outside the step budget, to tell unloaded space
from sky. Horizon rays walked most of the 48-chunk world. Under
`BUDGETED_RAYS` the far tier now crosses missing regions as empty space
within its step budget (exhausted is a miss). Far-48 pass: 2.0-4.2 ms ->
1.03-1.16 ms (0.55 ms classify, 0.5 ms for about 6.5k paths). The world
cache and receivers still use the full walk: there it decides "unknown"
near unloaded pages.

**Distance knob.** Ownership is now by full-resolution voxel size
(`voxel_pixels`), above.

**Sweep** (one run each, one build, lod-off twice; "fixed" is discovery +
presentation + per-pixel pass, the GPU cost the transport budget does not
control; trace is budgeted):

| Run | Frame p50 / p95 / p99 ms | Fixed ms | Receivers | Owned cells | Cell history |
| --- | --- | --- | --- | --- | --- |
| far-48 off | 22.4 / 27.3 / 28.8 | 4.25 | 11.6k | 0 | - |
| far-48 at 2 (870 v) | 22.1 / 27.6 / 29.6 | 4.99 | 11.6k | 1.1k | 25 |
| far-48 at 4 (435 v) | 22.2 / 27.6 / 29.6 | 5.02 | 11.3k | 2.0k | 22 |
| far-48 at 6 (290 v) | 22.2 / 26.9 / 28.8 | 5.07 | 10.1k | 9.1k | 22 |
| far-48 at 8 (217 v) | 22.0 / 25.5 / 27.2 | 4.98 | 8.7k | 20.1k | 22 |
| far-48 at 12 (145 v) | 22.0 / 24.4 / 26.1 | 5.35 | 7.9k | 25.2k | 20 |
| lod off | 91.7 / 216 / 887, 90.7 / 205 / 790 | 28.2, 28.4 | 40k | 0 | - |
| lod at 1 | 83.8 / 241 / 793 | 28.9 | 54k | 21k | 3 |
| lod at 2 | 84.0 / 237 / 908 | 31.0 | 61k | 65k | 4 |
| lod at 4 | 86.5 / 218 / 778 | 15.2 | 36k | 150k | 6 |
| lod at 6 | 87.9 / 255 / 886 | 10.3 | 9.9k | 239k | 6 |
| lod at 8 | 85.5 / 227 / 789 | 10.7 | 0 | 287k | 6 |
| lod at 12 | 85.6 / 237 / 775 | 6.7 | 0 | 296k | 6 |
| default off | 23.4 / 25.6 / 27.1 | 3.52 | 11.0k | 0 | - |
| default at 6 | 23.5 / 25.5 / 26.7 | 4.20 | 10.9k | 0.2k | 10 |
| default at 12 | 23.5 / 25.5 / 27.0 | 4.09 | 8.0k | 12.2k | 17 |

Marginal costs: a per-pixel cell (3x3 px) costs about 0.02-0.05 us; a
receiver about 0.15 us in far-48 and about 0.5 us in the crowded lod pool
(reconstruction). Per-face is cheaper only where one face covers more than
roughly 30-90 px, about `voxel_pixels` 6-8 (220-290 voxels at 1440p);
beyond it per-pixel is cheaper, and in the lod flight much cheaper. A low
threshold (2) is worse than off: the freed pool admits more mid-distance
faces, whose reconstruction costs more. The classify scan is a fixed 0.5
ms even when nothing is owned, which is most of the per-pixel cost in
far-48 and the default lab. Frame p50 is within noise everywhere; far-48
tails improved at 8 and 12 (single runs).

Quality: settled captures against off at 8/12 differ by FLIP 0.003
(default lab; off-vs-off 0.0012) and 0.008 (far-48), invisible in crops;
the mid-motion streaming capture differs 0.06-0.07, the same as off-vs-off
(0.063). Per-pixel history is short (6 cells' samples in the flight, about
20 in far-48, against 170-300 for far-48 receivers); flicker is not
measured. Branch default is now `voxel_pixels: 8`.

The tier-off lod control crashed once on the receiver hash overflow
assertion (`timing.rs`, 64 failed linear probes); a rerun was clean. That
path is main's, not this branch's.

## Update: area measures

`pixel_tier.measure` selects the ownership test: `VoxelSide` (above),
`VoxelArea` (a full-resolution voxel's rough covered area, side squared
times the cosine to the view ray, under `voxel_pixels` pixels) or
`BlockArea` (the same for the face's LoD block). All three are computed
identically by discovery, the pass and the material.

| Run | Frame p50 / p95 / p99 ms | Fixed ms | Receivers | Owned cells | Cell history |
| --- | --- | --- | --- | --- | --- |
| default off | 23.4 / 25.6 / 27.1 | 3.52 | 11.0k | 0 | - |
| default side 12 | 23.5 / 25.5 / 27.0 | 4.09 | 8.0k | 12.2k | 17 |
| default voxel area 64 | 23.5 / 25.4 / 26.1 | 4.21 | 4.8k | 28.0k | 6 |
| default voxel area 100 | 23.4 / 25.4 / 26.4 | 4.61 | 3.8k | 36.9k | 6 |
| default block area 64 | 23.5 / 25.4 / 26.7 | 4.34 | 4.8k | 27.9k | 7 |
| far-48 off | 22.4 / 27.3 / 28.8 | 4.25 | 11.6k | 0 | - |
| far-48 side 8 | 22.0 / 25.5 / 27.2 | 4.98 | 8.7k | 20.1k | 22 |
| far-48 voxel area 36 | 21.8 / 25.3 / 27.1 | 5.81 | 6.7k | 25.8k | 11 |
| far-48 voxel area 64 | 22.2 / 24.3 / 26.9 | 6.22 | 4.8k | 40.1k | 12 |
| far-48 block area 36 | 21.8 / 24.9 / 26.6 | 5.59 | 7.8k | 16.3k | 3 |
| far-48 block area 64 | 22.6 / 25.1 / 26.6 | 5.90 | 6.0k | 26.7k | 5 |
| lod off | 91 / 205-216 / 790-887 | 28.3 | 40k | 0 | - |
| lod side 8 / 12 | 85.5 / 227 / 789, 85.6 / 237 / 775 | 10.7, 6.7 | 0 | 287k, 296k | 6 |
| lod voxel area 16 / 36 / 64 | 84.8 / 225 / 767, 86.8 / 246 / 930, 86.1 / 252 / 911 | 10.7, 8.5, 7.8 | 18k, 2k, 0 | 200-290k | 6 |
| lod block area 16 / 36 / 64 | 83.2 / 216 / 791, 87.9 / 256 / 907, 87.7 / 253 / 889 | 16.4, 9.8, 9.8 | 22k, 2.3k, 74 | 191-293k | 6 |

The area tests hand grazing ground to the far tier much nearer the camera
(28k cells and 4.8k receivers in the default lab at 64, against 12k and 8k
at side 12). That costs more GPU than it saves where faces are cheap
(default lab, far-48) and shortens cell history to about 6 samples, since
nearer ground reprojects less reliably. In the lod flight all three tests
land at 7-11 ms against 28 off. Voxel and block area differ little; block
area keeps a few more receivers where strides are large. Far-48 tails are
better with the tier on in every form (p95 24.3-25.5 against 27.3 off; one
run each).

Seam: the coverage map at voxel area 64 shows the expected checkerboard on
mid-distance terraces (tops per-pixel, risers per-face at the same
distance), but the settled image matches tier-off there by eye. Settled
FLIP against off: default lab 0.003-0.004 (off-vs-off 0.0012), far-48
0.007-0.011 (far-48 at side 1, which owns nothing, reads 0.0026). A few
pixels in the map have neither tier's value (no cell sample nearby and no
receiver to fill from). Motion and flicker at the seam are not measured.
