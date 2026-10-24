#!/bin/zsh
# Base-spec validation suite: run on the target machine from the repository root.
#
#   tools/base_spec_suite.sh [label]
#   BASE_SPEC_ONLY=headless-response tools/base_spec_suite.sh check   # one run
#
# Windowed runs measure real frame pacing: keep the game window in front,
# unobscured and focused, and leave the machine alone while they run. Headless
# runs (Bevy offscreen capture) measure lighting and memory. Results land in
# target/base-spec-<label>-<host>-<time>/ with an archive next to it to copy
# back for analysis. Total time is about 25 minutes plus the first build.
set -u
label=${1:-run}
root=$PWD
stamp=$(date +%Y%m%d-%H%M%S)
out="target/base-spec-$label-$(hostname -s)-$stamp"
mkdir -p "$out"
{
  echo "label: $label"
  echo "host: $(hostname -s)"
  echo "commit: $(git rev-parse HEAD) $(git status --porcelain | wc -l | tr -d ' ') dirty files"
  sysctl -n machdep.cpu.brand_string hw.memsize hw.ncpu 2>/dev/null
  sw_vers 2>/dev/null
} > "$out/machine.txt"

cargo build --locked || exit 1

run() {
  local mode=$1 profile=$2 name=$3; shift 3
  [[ -n ${BASE_SPEC_ONLY:-} && $name != ${~BASE_SPEC_ONLY} ]] && return
  local before=$(ls -1d target/gi-validation/$profile-* 2>/dev/null | wc -l)
  echo "== $name ($mode $profile)"
  local headless=()
  [[ $mode == headless ]] && headless=(VOXEL_GI_VALIDATE_HEADLESS=1)
  env RUST_LOG=warn VOXEL_GI_VALIDATE=1 VOXEL_GI_VALIDATE_BACKGROUND=1 \
    VOXEL_GI_VALIDATE_FRAME_TIMINGS=1 \
    VOXEL_GI_VALIDATE_PROFILE=$profile VOXEL_GI_VALIDATE_EXIT=1 \
    VOXEL_GI_VALIDATE_EXPECT_BUILD_ROOT="$root" $headless "$@" \
    cargo run --locked > "$out/$name.log" 2>&1
  local code=$?
  local dir=$(ls -td target/gi-validation/$profile-* 2>/dev/null | head -1)
  if [[ $(ls -1d target/gi-validation/$profile-* 2>/dev/null | wc -l) -gt $before ]]; then
    mv "$dir" "$out/$name"
    echo "   exit $code -> $out/$name"
  else
    echo "   exit $code, no run directory (see $out/$name.log)"
  fi
}

# Frame pacing in a real window (the numbers that matter for play).
run window gameplay window-gameplay
run window day-cycle window-day-cycle
# Lighting response and memory, offscreen.
run headless response headless-response
run headless house-edits headless-house-edits VOXEL_GI_SETTINGS=config/gi-mirror-full.ron
run headless day-cycle headless-day-cycle
# Fixed-sun references for the day-cycle comparison (angles at 60/120/180 s).
for angle in 18 36 54; do
  run headless day-cycle headless-day-ref$angle VOXEL_GI_VALIDATE_SUN_RATE=0 VOXEL_GI_VALIDATE_SUN_OFFSET=$angle
done

tar -czf "$out.tar.gz" -C target "$(basename "$out")"
echo "Results: $out"
echo "Archive to copy back: $out.tar.gz"
