#!/usr/bin/env bash
# R59: leaf-size, footprint-support, gap-scaling, min-points, merge-criterion, validity, scratch and schedule checks for the
# threshold gaussian_surface merge (no D2). Same mechanics as run_v2_matrix.sh (R57/R58).
# Per-scan snapshots, scans 3..480, --min-points 5. Everything is a variable; nothing is hard-coded except the arm table below.
#
#   CLI=<stationary_map_cli>  CACHE_ROOT=<dir holding eee01_cache_r53_pfn1_dsoff and eee01_cache_r53_canonical>
#   [DEBIAS_CONTEXT unused] REF=<reference_surfaces.csv>  CFG=<config/ntu_viral.yaml>
#   IMG=<camera_first.png>  BOUNDS_FILE=<R55 color_bounds.txt>  OUT=<output dir>
#   [CKPT=3:480:1] [MINPTS=5] [JOBS=4] [SUBSET=480] [SV=0] [VIDEO_ARMS="l10_sup l25_sup"]
#   [FPS=15] [SCALE=0.5] [ONLY="arm1 arm2"]  (ONLY runs just those arms)
#   bash tools/stationary_map/run_leaf_matrix.sh
#
# Per arm: full raw outputs (patches, summary, v2 summary, surfaces) -> sha256 recorded -> checkpoint subsets kept in
# $OUT/subset -> video (video arms only) -> the big raw patches/surfaces files are DELETED unless KEEP_RAW=1.
# Nothing here interprets results.
set -u
DEBIAS_CONTEXT="${DEBIAS_CONTEXT:-unused}"; : "${CLI:?}" "${CACHE_ROOT:?}" "${REF:?}" "${CFG:?}" "${IMG:?}" "${BOUNDS_FILE:?}" "${OUT:?}"
CKPT="${CKPT:-3:480:1}"; MINPTS="${MINPTS:-5}"; JOBS="${JOBS:-4}"; SUBSET="${SUBSET:-480}"; SV="${SV:-0}"
VIDEO_ARMS="${VIDEO_ARMS:-l10_sup l25_sup}"; FPS="${FPS:-15}"; SCALE="${SCALE:-0.5}"; KEEP_RAW="${KEEP_RAW:-0}"
ONLY="${ONLY:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BOUNDS="$(cat "$BOUNDS_FILE")"
mkdir -p "$OUT"/{raw,subset,videos,logs}
echo "arm,seconds,exit_code,checkpoints" > "$OUT/timing.csv"
: > "$OUT/raw_sha256_manifest.txt"

# name;cache;family;extra CLI flags   (flags given later on the line override earlier ones, incl. --checkpoints)
# Baseline options everywhere: threshold merge, angle 5 deg, offset 0.05 m, combined_fit, plain (non-debiased) fit, no pose covariance.
# Gaps are multiples of the leaf: merge_gap = 3*leaf, support_gap = 1*leaf.
ARMS='l25_ctrl;pfn1_dsoff;gaussian_surface;--leaf 0.25 --merge-gap 0.75
l25_sup;pfn1_dsoff;gaussian_surface;--leaf 0.25 --merge-gap 0.75 --support-test --support-gap 0.25
l25_sup_g10;pfn1_dsoff;gaussian_surface;--leaf 0.25 --merge-gap 0.75 --support-test --support-gap 0.10
l15;pfn1_dsoff;gaussian_surface;--leaf 0.15 --merge-gap 0.45
l15_sup;pfn1_dsoff;gaussian_surface;--leaf 0.15 --merge-gap 0.45 --support-test --support-gap 0.15
l10;pfn1_dsoff;gaussian_surface;--leaf 0.10 --merge-gap 0.30
l10_sup;pfn1_dsoff;gaussian_surface;--leaf 0.10 --merge-gap 0.30 --support-test --support-gap 0.10
l10_sup_mp3;pfn1_dsoff;gaussian_surface;--leaf 0.10 --merge-gap 0.30 --support-test --support-gap 0.10 --min-points 3
l10_sup_g05;pfn1_dsoff;gaussian_surface;--leaf 0.10 --merge-gap 0.30 --support-test --support-gap 0.05
l10_sup_g20;pfn1_dsoff;gaussian_surface;--leaf 0.10 --merge-gap 0.30 --support-test --support-gap 0.20
l10_gap20;pfn1_dsoff;gaussian_surface;--leaf 0.10 --merge-gap 0.20 --support-test --support-gap 0.10
l10_gap50;pfn1_dsoff;gaussian_surface;--leaf 0.10 --merge-gap 0.50 --support-test --support-gap 0.10
l25_sup_scratch;pfn1_dsoff;gaussian_surface;--leaf 0.25 --merge-gap 0.75 --support-test --support-gap 0.25 --rebuild scratch
l10_sup_scratch;pfn1_dsoff;gaussian_surface;--leaf 0.10 --merge-gap 0.30 --support-test --support-gap 0.10 --rebuild scratch
l25_sup_sparse;pfn1_dsoff;gaussian_surface;--leaf 0.25 --merge-gap 0.75 --support-test --support-gap 0.25 --checkpoints 10,50,100,200,300,480
l10_sup_sparse;pfn1_dsoff;gaussian_surface;--leaf 0.10 --merge-gap 0.30 --support-test --support-gap 0.10 --checkpoints 10,50,100,200,300,480
l25_pairwise;pfn1_dsoff;gaussian_surface;--leaf 0.25 --merge-gap 0.75 --merge-criterion pairwise
l25_r52valid;pfn1_dsoff;gaussian_surface;--leaf 0.25 --merge-gap 0.75 --plane-eig-max 1e30 --min-secondary-eig 0 --max-planarity 0.10
can_l25_sup;canonical;gaussian_surface;--leaf 0.25 --merge-gap 0.75 --support-test --support-gap 0.25
can_l10_sup;canonical;gaussian_surface;--leaf 0.10 --merge-gap 0.30 --support-test --support-gap 0.10'

do_arm() {   # "name;cache;family;flags"
  IFS=';' read -r name cache family flags <<< "$1"
  flags="${flags//@CTX@/$DEBIAS_CONTEXT}"
  pre="$OUT/raw/${name}"
  cdir="$CACHE_ROOT/eee01_cache_r53_${cache}"
  t0=$(date +%s)
  # shellcheck disable=SC2086
  "$CLI" --family "$family" --input "$cdir" --patches-out "${pre}_patches.csv" --summary-out "${pre}_summary.csv" \
     --v2-summary-out "${pre}_v2summary.csv" --surfaces-out "${pre}_surfaces.csv" \
     --checkpoints "$CKPT" --min-points "$MINPTS" --sensor-var "$SV" $flags > "$OUT/logs/${name}.log" 2>&1
  rc=$?
  echo "${name},$(( $(date +%s) - t0 )),$rc,$CKPT" >> "$OUT/timing.csv"
  [ $rc -eq 0 ] || return 0
  sha256sum "${pre}_patches.csv" "${pre}_surfaces.csv" >> "$OUT/raw_sha256_manifest.txt"
  python3 "$HERE/subset_checkpoints.py" --in "${pre}_patches.csv" --out "$OUT/subset/${name}_patches_subset.csv" --checkpoints "$SUBSET" >> "$OUT/logs/${name}.log" 2>&1
  python3 "$HERE/subset_checkpoints.py" --in "${pre}_surfaces.csv" --out "$OUT/subset/${name}_surfaces_subset.csv" --checkpoints "$SUBSET" >> "$OUT/logs/${name}.log" 2>&1
  cp "${pre}_summary.csv" "${pre}_v2summary.csv" "$OUT/subset/"
  case " $VIDEO_ARMS " in *" $name "*)
    python3 "$HERE/render_evolution_video.py" --patches "${pre}_patches.csv" --summary "${pre}_summary.csv" \
      --reference "$REF" --camera-yaml "$CFG" --image "$IMG" --label "$cache / $family / $name" --manifest "$cdir/manifest.csv" \
      --color-by xyz --surface-color id --bounds $BOUNDS --fps "$FPS" --scale "$SCALE" \
      --out "$OUT/videos/${name}.mp4" > "$OUT/logs/render_${name}.log" 2>&1
    echo "render ${name} rc=$?" >> "$OUT/logs/render_summary.log" ;;
  esac
  if [ "$KEEP_RAW" != "1" ]; then rm -f "${pre}_patches.csv" "${pre}_surfaces.csv"; fi
}
export -f do_arm
export CLI CACHE_ROOT DEBIAS_CONTEXT REF CFG IMG BOUNDS OUT CKPT MINPTS SUBSET SV VIDEO_ARMS FPS SCALE KEEP_RAW HERE

printf '%s\n' "$ARMS" | while IFS= read -r line; do
  n="${line%%;*}"
  if [ -n "$ONLY" ]; then case " $ONLY " in *" $n "*) ;; *) continue;; esac; fi
  printf '%s\0' "$line"
done | xargs -0 -P "$JOBS" -I{} bash -c 'do_arm "$1"' _ {}
echo "done; see $OUT/timing.csv, $OUT/logs, $OUT/raw_sha256_manifest.txt"
