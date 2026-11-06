# Releasing the crates

The published family is `bevy_voxel_gi`, `bevy_voxel_gi_bvw` and
`bevy_voxel_gi_debug`. They share the version in `[workspace.package]`.
The root `prototype_voxel_template` package is the unpublished lab.

The [changelog](../CHANGELOG.md#known-limitations) records the known runtime
limitations. Review them for each release: passing the packaging checks does
not close a finding, and a fix should say so in the changelog (0.1.1 closed the
BVW resampling and small scene-queue edit-priority findings).

## Prepare a version

Update the workspace version and the versions of all three path dependencies
in `[workspace.dependencies]`. Keep the compatibility notes in the core README
and the dated entry in `CHANGELOG.md` consistent with that version. Review the
documented platform requirements and known limitations.

Run the crate checks from [validation](validation.md#choose-the-relevant-check).
Material or shader import changes also need the native pipeline/material tests,
because a Rust build alone cannot validate WGSL imports and GPU bindings.

Package the three crates together so Cargo can resolve their unpublished
workspace dependencies while verifying the archives:

```sh
cargo package --locked -p bevy_voxel_gi -p bevy_voxel_gi_bvw -p bevy_voxel_gi_debug
```

For an uncommitted local preflight, add `--allow-dirty`; final release evidence
should identify the committed tree. Inspect the resulting `.crate` archives in
`target/package/`: normalized dependency names and versions, embedded shaders,
READMEs and licenses must all be present. `--list` checks inventory and
`--no-verify` creates archives; neither confirms that the packaged code builds.

## Publish

After the release commit and tag are approved, publish the core first:

```sh
cargo publish --locked -p bevy_voxel_gi
```

Wait for that version to become available in the crates.io index, then publish
the connector and debug crate:

```sh
cargo publish --locked -p bevy_voxel_gi_bvw
cargo publish --locked -p bevy_voxel_gi_debug
```

Check the three crates.io pages and docs.rs builds. Confirm that installation
examples use the published names and that the shader import path is
`bevy_voxel_gi::presentation`. A package name availability check does not reserve
the name; publication is a separate action from preparation.
