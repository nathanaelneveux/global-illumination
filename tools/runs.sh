#!/bin/zsh
# Labelled headless runs for A/B and sweeps, one build, settings by file.
# usage: tools/runs.sh <list> <label>:<profile>[:<settings.ron|->[:ENV=V,ENV=V...]]...
#   settings: a path relative to the repository (config/gi-far-tier-off.ron,
#   target/gates/x.ron), or - for the defaults. Each finished run appends
#   "<label> <run directory> <log>" to <list> (absolute paths), and "done" at
#   the end. Logs are target/gates/run-<label>.log at RUST_LOG=voxel_gi=info,
#   so tools/view_costs.py can read the view-tier receipts.
# In zsh, brace parameters before a colon when building specs ("${v}:"):
# "$v:s..." is read as a history modifier.
# Do not edit assets/ or compiled sources while this runs.
set -u
root=${0:A:h:h}
cd "$root" || exit 1
mkdir -p target/gates
out=$1; shift
for spec in "$@"; do
  label=${spec%%:*}; rest=${spec#*:}
  profile=${rest%%:*}; rest=${rest#${profile}}; rest=${rest#:}
  settings=${rest%%:*}; rest=${rest#${settings}}; rest=${rest#:}
  envs=()
  [[ -n $settings && $settings != - ]] && envs+=(VOXEL_GI_SETTINGS=$root/$settings)
  [[ -n $rest ]] && envs+=(${(s:,:)rest})
  before=$(ls -1d target/gi-validation/$profile-* 2>/dev/null | wc -l)
  env "${envs[@]}" RUST_LOG=warn,voxel_gi=info VOXEL_GI_VALIDATE=1 VOXEL_GI_VALIDATE_HEADLESS=1 \
    VOXEL_GI_VALIDATE_FRAME_TIMINGS=1 VOXEL_GI_VALIDATE_PROFILE=$profile VOXEL_GI_VALIDATE_EXIT=1 \
    VOXEL_GI_VALIDATE_EXPECT_BUILD_ROOT="$root" cargo run --locked --offline > target/gates/run-$label.log 2>&1
  code=$?
  if [[ $code -ne 0 || $(ls -1d target/gi-validation/$profile-* 2>/dev/null | wc -l) -le $before ]]; then
    echo "$label FAILED (exit $code), see target/gates/run-$label.log" >&2
    continue
  fi
  echo "$label $root/$(ls -td target/gi-validation/$profile-* | head -1) $root/target/gates/run-$label.log" >> $out
done
echo done >> $out
