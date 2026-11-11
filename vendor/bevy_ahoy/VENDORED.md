# Vendored bevy_ahoy

This is [bevy_ahoy](https://github.com/janhohenheim/bevy_ahoy) 0.2.0 by
Jan Hohenheim, MIT OR Apache-2.0 (licences alongside), ported to Bevy 0.20.
The lab's player controller uses it, and upstream has no 0.20 release yet.
Delete this copy once upstream publishes one.

The first commit adding this directory is the published 0.2.0, unmodified
(upstream commit 630a457), so `git diff` against it shows the whole port.

## What changed

- Bevy subcrates 0.19 → 0.20, avian3d 0.7 → 0.8, bevy_enhanced_input
  0.26 → 0.27. `Cuboid` now comes from the new `bevy_shape` crate.
- Avian 0.8: `SpeculativeMargin` is now `SpeculativeCcd`. `aabb` takes a
  margin, which is `0.0` here to keep the old bounds. `contains_point`
  takes a plain `Vec3`.
- Bevy 0.20: `Query::iter_many` yields `Result`s; water uses `.matched()`
  to skip non-water colliders as before.
- The `pickup` feature is removed until avian_pickup supports Bevy 0.20.
  Its code stays behind `cfg(feature = "pickup")`, untouched.
- The examples are left out because their assets are not in the package.
  `bevy` became a dev-dependency for the readme doc test.
- New `tests/port.rs`: a headless character lands on a floor, stands still
  and walks at its configured speed.

Clippy's `question_mark` lint flags one upstream block in `kcc.rs`. It is
left as upstream wrote it.

## Building

It is a workspace member, and the lab depends on it by path. Test it with
`cargo test -p bevy_ahoy`. To return to upstream once it supports Bevy 0.20,
point the lab's `bevy_ahoy` dependency back at crates.io and delete this
directory.
