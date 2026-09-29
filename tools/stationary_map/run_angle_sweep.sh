#!/usr/bin/env bash
# R56: gaussian_surface on pfn1_dsoff with merge angle 5, 3, 1 degrees (per-scan snapshots, scans 3..480, min-points 5).
# Only the angle gate is varied (--merge-angle-deg). The offset gate (0.05 m), gap (0.75 m), leaf, validity and combined_fit
# are defaults; combined_fit uses the SAME angle, so every member must stay within the angle of the union plane.
#
#   CLI=<stationary_map_cli>  CACHE_ROOT=<dir with eee01_cache_r53_pfn1_dsoff>  REF=<reference_surfaces.csv>
#   CFG=<config/ntu_viral.yaml>  IMG=<camera_first.png>  OUT=<dir>  [ANGLES="5 3 1"] [CKPT=3:480:1] [MINPTS=5] [JOBS=3]
#   [BOUNDS="lox loy loz hix hiy hiz"]  (default: R55's color_bounds.txt if you pass it as BOUNDS_FILE, else computed from the a5 run)
#   bash tools/stationary_map/run_angle_sweep.sh
set -u
: "${CLI:?}" "${CACHE_ROOT:?}" "${REF:?}" "${CFG:?}" "${IMG:?}" "${OUT:?}"
ANGLES="${ANGLES:-5 3 1}"; CKPT="${CKPT:-3:480:1}"; MINPTS="${MINPTS:-5}"; JOBS="${JOBS:-3}"
FPS="${FPS:-15}"; SCALE="${SCALE:-0.75}"; SUBSET="${SUBSET:-3,10,25,50,100,150,200,300,400,480}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ARM=pfn1_dsoff; FAM=gaussian_surface
CACHE="$CACHE_ROOT/eee01_cache_r53_$ARM"
mkdir -p "$OUT"/{raw,videos,stills,stats,subset,logs}
echo "angle_deg,seconds,exit_code,checkpoints" > "$OUT/timing.csv"

run_one() {   # angle
  ang="$1"; pre="$OUT/raw/${FAM}_${ARM}_ang${ang}"
  t0=$(date +%s)
  "$CLI" --family "$FAM" --input "$CACHE" --patches-out "${pre}_patches.csv" --summary-out "${pre}_summary.csv" \
         --checkpoints "$CKPT" --min-points "$MINPTS" --merge-angle-deg "$ang" > "$OUT/logs/${FAM}_${ARM}_ang${ang}.log" 2>&1
  rc=$?
  echo "${ang},$(( $(date +%s) - t0 )),$rc,$CKPT" >> "$OUT/timing.csv"
}
export -f run_one
export CLI CACHE OUT CKPT MINPTS FAM ARM
for ang in $ANGLES; do echo "$ang"; done | xargs -P "$JOBS" -L 1 bash -c 'run_one $0'

# colour bounds: shared by all three videos so voxel colours are identical across them
if [ -n "${BOUNDS_FILE:-}" ] && [ -s "$BOUNDS_FILE" ]; then BOUNDS="$(cat "$BOUNDS_FILE")"; fi
if [ -z "${BOUNDS:-}" ]; then
  first=$(echo $ANGLES | awk '{print $1}')
  BOUNDS=$(python3 "$HERE/render_evolution_video.py" --patches "$OUT/raw/${FAM}_${ARM}_ang${first}_patches.csv" \
     --summary "$OUT/raw/${FAM}_${ARM}_ang${first}_summary.csv" --reference "$REF" --camera-yaml "$CFG" --image "$IMG" \
     --out /dev/null --print-bounds)
fi
echo "$BOUNDS" > "$OUT/color_bounds.txt"; echo "colour bounds: $BOUNDS"

render_one() {   # angle
  ang="$1"; pre="$OUT/raw/${FAM}_${ARM}_ang${ang}"; name="${ARM}_${FAM}_ang${ang}"
  [ -s "${pre}_patches.csv" ] || { echo "missing ${pre}_patches.csv" >&2; return 1; }
  python3 "$HERE/render_evolution_video.py" --patches "${pre}_patches.csv" --summary "${pre}_summary.csv" \
     --reference "$REF" --camera-yaml "$CFG" --image "$IMG" --label "$ARM / $FAM / angle ${ang} deg" --manifest "$CACHE/manifest.csv" \
     --color-by xyz --surface-color id --bounds $BOUNDS --fps "$FPS" --scale "$SCALE" \
     --stats-out "$OUT/stats/${name}_frames.csv" --stills-at 3,25,100,200,300,400,480 --stills-dir "$OUT/stills/$name" \
     --out "$OUT/videos/${name}.mp4" > "$OUT/logs/render_${name}.log" 2>&1
  echo "render ${name} rc=$?" >> "$OUT/logs/render_summary.log"
}
export -f render_one
export REF CFG IMG HERE FPS SCALE BOUNDS
for ang in $ANGLES; do echo "$ang"; done | xargs -P "$JOBS" -L 1 bash -c 'render_one $0'

for f in "$OUT"/raw/*_ang*_patches.csv; do
  python3 "$HERE/subset_checkpoints.py" --in "$f" --out "$OUT/subset/$(basename "${f%.csv}")_subset.csv" --checkpoints "$SUBSET"
done
for f in "$OUT"/raw/*_ang*_summary.csv; do cp "$f" "$OUT/subset/"; done
echo "done; see $OUT/timing.csv and $OUT/logs/render_summary.log"
