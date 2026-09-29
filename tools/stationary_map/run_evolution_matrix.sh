#!/usr/bin/env bash
# R55: per-scan stationary-map runs + evolution videos. Everything is a variable; nothing is hard-coded.
#
#   CLI=<path to built stationary_map_cli>            CACHE_ROOT=<dir holding eee01_cache_r53_* caches>
#   REF=<reference_surfaces.csv>   CFG=<config/ntu_viral.yaml>   IMG=<camera_first.png (raw frame)>
#   OUT=<output dir>   [CKPT=3:480:1] [ROBUST_CKPT=$CKPT] [MINPTS=5] [JOBS=4] [FPS=15] [SCALE=0.75]
#   bash tools/stationary_map/run_evolution_matrix.sh
#
# Steps: (1) 10 CLI runs (2 caches x 5 families) with a snapshot after EVERY scan in CKPT
#        (2) shared colour bounds from the canonical incremental_pca last frame
#        (3) 10 videos + stats + stills, plus 2 extra colour-mode videos for canonical/gaussian_surface
#        (4) small checkpoint subsets of every patches CSV for the return zip
# Timing of every run is logged to $OUT/timing.csv. Nothing here interprets results.
set -u
: "${CLI:?}" "${CACHE_ROOT:?}" "${REF:?}" "${CFG:?}" "${IMG:?}" "${OUT:?}"
CKPT="${CKPT:-3:480:1}"; ROBUST_CKPT="${ROBUST_CKPT:-$CKPT}"; MINPTS="${MINPTS:-5}"
JOBS="${JOBS:-4}"; FPS="${FPS:-15}"; SCALE="${SCALE:-0.75}"
SUBSET="${SUBSET:-3,10,25,50,100,150,200,300,400,480}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"/{raw,videos,stills,stats,subset,logs}
echo "config,seconds,exit_code,checkpoints" > "$OUT/timing.csv"

ARMS="canonical pfn1_dsoff"
FAMS="incremental_pca mergeable_voxel gaussian_surface robust_voxel robust_mergeable"
cache_dir() { echo "$CACHE_ROOT/eee01_cache_r53_$1"; }

run_one() {   # arm family
  arm="$1"; fam="$2"; ck="$CKPT"
  case "$fam" in robust_*) ck="$ROBUST_CKPT";; esac
  pre="$OUT/raw/${fam}_${arm}_evo"
  t0=$(date +%s)
  "$CLI" --family "$fam" --input "$(cache_dir "$arm")" --patches-out "${pre}_patches.csv" \
         --summary-out "${pre}_summary.csv" --checkpoints "$ck" --min-points "$MINPTS" \
         > "$OUT/logs/${fam}_${arm}_evo.log" 2>&1
  rc=$?
  echo "${arm}/${fam},$(( $(date +%s) - t0 )),$rc,$ck" >> "$OUT/timing.csv"
}
export -f run_one cache_dir
export CLI CACHE_ROOT OUT CKPT ROBUST_CKPT MINPTS

# (1) CLI runs
for arm in $ARMS; do for fam in $FAMS; do echo "$arm $fam"; done; done | xargs -P "$JOBS" -L 1 bash -c 'run_one $0 $1'

# (2) shared colour bounds
BOUNDS=$(python3 "$HERE/render_evolution_video.py" --patches "$OUT/raw/incremental_pca_canonical_evo_patches.csv" \
   --summary "$OUT/raw/incremental_pca_canonical_evo_summary.csv" --reference "$REF" --camera-yaml "$CFG" --image "$IMG" \
   --out /dev/null --print-bounds)
echo "$BOUNDS" > "$OUT/color_bounds.txt"; echo "colour bounds: $BOUNDS"

# (3) videos
render_one() {   # arm family colorby suffix
  arm="$1"; fam="$2"; cb="$3"; suf="$4"; [ "$suf" = "-" ] && suf=""
  pre="$OUT/raw/${fam}_${arm}_evo"; name="${arm}_${fam}${suf}"
  [ -s "${pre}_patches.csv" ] || { echo "missing ${pre}_patches.csv" >&2; return 1; }
  MAN="$(cache_dir "$arm")/manifest.csv"
  python3 "$HERE/render_evolution_video.py" --patches "${pre}_patches.csv" --summary "${pre}_summary.csv" \
     --reference "$REF" --camera-yaml "$CFG" --image "$IMG" --label "$arm / $fam" --manifest "$MAN" \
     --color-by "$cb" --bounds $BOUNDS --fps "$FPS" --scale "$SCALE" \
     --stats-out "$OUT/stats/${name}_frames.csv" --stills-at 3,25,100,200,300,400,480 --stills-dir "$OUT/stills/$name" \
     --out "$OUT/videos/${name}.mp4" > "$OUT/logs/render_${name}.log" 2>&1
  echo "render ${name} rc=$?" >> "$OUT/logs/render_summary.log"
}
export -f render_one
export REF CFG IMG HERE FPS SCALE BOUNDS
{ for arm in $ARMS; do for fam in $FAMS; do echo "$arm $fam xyz -"; done; done
  echo "canonical gaussian_surface y _colory"; echo "canonical gaussian_surface range _colorrange"; } \
  | xargs -P "$JOBS" -L 1 bash -c 'render_one $0 $1 $2 $3'

# (4) subsets for the return zip
for f in "$OUT"/raw/*_evo_patches.csv; do
  python3 "$HERE/subset_checkpoints.py" --in "$f" --out "$OUT/subset/$(basename "${f%.csv}")_subset.csv" --checkpoints "$SUBSET"
done
for f in "$OUT"/raw/*_evo_summary.csv; do cp "$f" "$OUT/subset/"; done
echo "done; see $OUT/timing.csv and $OUT/logs/render_summary.log"
