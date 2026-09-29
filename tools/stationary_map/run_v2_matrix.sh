#!/usr/bin/env bash
# R57: covariance-aware merge/split, support and observation tests, scratch rebuild, robust recursion.
# Per-scan snapshots, scans 3..480, --min-points 5. Everything is a variable; nothing is hard-coded except the arm table below.
#
#   CLI=<stationary_map_cli>  CACHE_ROOT=<dir holding eee01_cache_r53_pfn1_dsoff and eee01_cache_r53_canonical>
#   DEBIAS_CONTEXT=<debias_context.csv (the R54 file)>  REF=<reference_surfaces.csv>  CFG=<config/ntu_viral.yaml>
#   IMG=<camera_first.png>  BOUNDS_FILE=<R55 color_bounds.txt>  OUT=<output dir>
#   [CKPT=3:480:1] [MINPTS=5] [JOBS=4] [SUBSET=480] [SV=0] [VIDEO_ARMS="d2i_sup_obs_scratch rm_d2i_rec2_val"]
#   [FPS=15] [SCALE=0.5] [ONLY="arm1 arm2"]  (ONLY runs just those arms)
#   bash tools/stationary_map/run_v2_matrix.sh
#
# Per arm: full raw outputs (patches, summary, v2 summary, surfaces) -> sha256 recorded -> checkpoint subsets kept in
# $OUT/subset -> video (video arms only) -> the big raw patches/surfaces files are DELETED unless KEEP_RAW=1.
# Nothing here interprets results.
set -u
: "${CLI:?}" "${CACHE_ROOT:?}" "${DEBIAS_CONTEXT:?}" "${REF:?}" "${CFG:?}" "${IMG:?}" "${BOUNDS_FILE:?}" "${OUT:?}"
CKPT="${CKPT:-3:480:1}"; MINPTS="${MINPTS:-5}"; JOBS="${JOBS:-4}"; SUBSET="${SUBSET:-480}"; SV="${SV:-0}"
VIDEO_ARMS="${VIDEO_ARMS:-d2i_sup_obs_scratch rm_d2i_rec2_val}"; FPS="${FPS:-15}"; SCALE="${SCALE:-0.5}"; KEEP_RAW="${KEEP_RAW:-0}"
ONLY="${ONLY:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BOUNDS="$(cat "$BOUNDS_FILE")"
mkdir -p "$OUT"/{raw,subset,videos,logs}
echo "arm,seconds,exit_code,checkpoints" > "$OUT/timing.csv"
: > "$OUT/raw_sha256_manifest.txt"

# name;cache;family;extra CLI flags
ARMS='thr5_control;pfn1_dsoff;gaussian_surface;
thr5_scratch;pfn1_dsoff;gaussian_surface;--rebuild scratch
d2i;pfn1_dsoff;gaussian_surface;--merge-test d2 --plane-var-mode information
d2i_scratch;pfn1_dsoff;gaussian_surface;--merge-test d2 --plane-var-mode information --rebuild scratch
d2i_t78;pfn1_dsoff;gaussian_surface;--merge-test d2 --plane-var-mode information --d2-tau 7.8
d2i_t163;pfn1_dsoff;gaussian_surface;--merge-test d2 --plane-var-mode information --d2-tau 16.3
d2i_sup;pfn1_dsoff;gaussian_surface;--merge-test d2 --plane-var-mode information --support-test
d2i_sup_scratch;pfn1_dsoff;gaussian_surface;--merge-test d2 --plane-var-mode information --support-test --rebuild scratch
d2i_sup_obs;pfn1_dsoff;gaussian_surface;--merge-test d2 --plane-var-mode information --support-test --obs-test
d2i_sup_obs_scratch;pfn1_dsoff;gaussian_surface;--merge-test d2 --plane-var-mode information --support-test --obs-test --rebuild scratch
d2ip;pfn1_dsoff;gaussian_surface;--merge-test d2 --plane-var-mode information --d2-pose-cov --debias-context @CTX@
d2ip_scratch;pfn1_dsoff;gaussian_surface;--merge-test d2 --plane-var-mode information --d2-pose-cov --debias-context @CTX@ --rebuild scratch
d2e;pfn1_dsoff;gaussian_surface;--merge-test d2 --plane-var-mode eigengap --d2-pose-cov --debias-context @CTX@
d2e_scratch;pfn1_dsoff;gaussian_surface;--merge-test d2 --plane-var-mode eigengap --d2-pose-cov --debias-context @CTX@ --rebuild scratch
rm_d2i;pfn1_dsoff;robust_mergeable;--merge-test d2 --plane-var-mode information
rm_d2i_rec2;pfn1_dsoff;robust_mergeable;--merge-test d2 --plane-var-mode information --robust-recursion-depth 2
rm_d2i_rec2_val;pfn1_dsoff;robust_mergeable;--merge-test d2 --plane-var-mode information --robust-recursion-depth 2 --robust-validity-test
can_d2i_scratch;canonical;gaussian_surface;--merge-test d2 --plane-var-mode information --rebuild scratch
can_d2i_sup_obs_scratch;canonical;gaussian_surface;--merge-test d2 --plane-var-mode information --support-test --obs-test --rebuild scratch'

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
