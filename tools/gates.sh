#!/bin/zsh
# Packaging/refactor gate: one labelled set of headless runs for gates.py.
# usage: tools/gates.sh <label> [extra env...]
# Runs (M4 ~6 min plus build): streaming and cave GPU receipts, static pose
# (combined, sun, indirect), two indirect walks, a streaming turn burst,
# tunnel and response. Run directories are listed in target/gates/<label>.txt.
# Do not edit assets/ while this runs: the asset watcher aborts the run.
set -u
label=$1; shift
root=${0:A:h:h}
cd "$root" || exit 1
mkdir -p target/gates
list=target/gates/$label.txt
: > $list
info=RUST_LOG=warn,bevy_voxel_gi=info
run() {
  local name=$1 profile=$2; shift 2
  local before=$(ls -1d target/gi-validation/$profile-* 2>/dev/null | wc -l)
  env RUST_LOG=warn VOXEL_GI_VALIDATE=1 VOXEL_GI_VALIDATE_HEADLESS=1 VOXEL_GI_VALIDATE_BACKGROUND=1 \
    VOXEL_GI_VALIDATE_FRAME_TIMINGS=1 \
    VOXEL_GI_VALIDATE_PROFILE=$profile VOXEL_GI_VALIDATE_EXIT=1 \
    VOXEL_GI_VALIDATE_EXPECT_BUILD_ROOT="$root" "$@" \
    cargo run --locked --offline > target/gates/$label-$name.log 2>&1
  local code=$?
  local dir=$(ls -td target/gi-validation/$profile-* 2>/dev/null | head -1)
  if [[ $code -ne 0 || $(ls -1d target/gi-validation/$profile-* 2>/dev/null | wc -l) -le $before ]]; then
    echo "$name FAILED (exit $code)"; echo "$name - target/gates/$label-$name.log" >> $list; return
  fi
  echo "$name $dir target/gates/$label-$name.log" >> $list
  echo "$name -> $dir"
}
run streaming streaming $info "$@"
run cave cave $info "$@"
run static motion VOXEL_GI_VALIDATE_MOTION_STATIC=1 "$@"
run static-sun motion VOXEL_GI_VALIDATE_MOTION_STATIC=1 VOXEL_GI_VALIDATE_LIGHT_COMPONENT=sun "$@"
run static-indirect motion VOXEL_GI_VALIDATE_MOTION_STATIC=1 VOXEL_GI_VALIDATE_LIGHT_COMPONENT=indirect "$@"
run walk1 motion VOXEL_GI_VALIDATE_MOTION_WALK=4.32 VOXEL_GI_VALIDATE_LIGHT_COMPONENT=indirect "$@"
run walk2 motion VOXEL_GI_VALIDATE_MOTION_WALK=4.32 VOXEL_GI_VALIDATE_LIGHT_COMPONENT=indirect "$@"
run turn streaming VOXEL_GI_VALIDATE_TURN_BURST=1 "$@"
run tunnel tunnel "$@"
run response response "$@"
