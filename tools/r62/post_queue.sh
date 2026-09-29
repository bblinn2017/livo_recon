#!/usr/bin/env bash
# R62 (adapted from R61): after (or while) `dispatch_queue.sh <queue_dir>` drains the arm queue, collect each finished job's diagnostics into
# $RUNS/<arm>/, write arm.json, run reduce_arm.py, and prune heavy raw files (keeping full raw for KEEP_RAW arms).
# This script does NOT run or schedule jobs: all dispatching is done by dispatch_queue.sh (standing rule 5.6a).
#
#   QUEUE_DIR=<job queue dir the dispatcher drained>   RUNS=<output dir>   ARM_TABLE=r62_manifest/arm_table.tsv
#   [MAP=<tsv: job_dir_name<TAB>arm>]   only if the queue's job directory names differ from the arm names
#   [KEEP_RAW="arm1 arm2 ..."]  [KEEP_ALL_RAW=0]  [REDO=0]
# A job counts as finished when its dir holds DONE (exit_code 0) or FAILED (exit_code 1), the dispatcher's own sentinels.
# ATE is parsed from the job's results_lio.txt as the dispatcher does (`ATE=<metres>`).
# Diagnostic files are located with `find` anywhere under the job dir; a file that is not found is listed in arm.json
# under "missing" (the agent must then say where those files land, see the instructions).
set -u
: "${QUEUE_DIR:?}" "${RUNS:?}" "${ARM_TABLE:?}"
KEEP_RAW="${KEEP_RAW:-A_splineless_pca_baseline_60s B_open_loop_5s_windows_configured_P0_fixed_Q_default_calibration_biases C_open_loop_5s_windows_first_scan_head_seeded D_open_loop_5s_windows_tiny_P0_fixed_Q}"
KEEP_ALL_RAW="${KEEP_ALL_RAW:-0}"; REDO="${REDO:-0}"; MAP="${MAP:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
FILES="state_trace.csv state_reference.txt imu_cov_growth.csv lio_gt_matched.csv motion_q_scan.csv joint_knot_lidar_information.csv results_lio.txt gravity_alignment.txt imu_first_scans.csv calibration_p0.txt config.yaml"
mkdir -p "$RUNS"
for jd in "$QUEUE_DIR"/*/; do
  jd="${jd%/}"; name="$(basename "$jd")"
  [ -f "$jd/DONE" ] || [ -f "$jd/FAILED" ] || continue
  arm="$name"
  if [ -n "$MAP" ] && [ -f "$MAP" ]; then a2="$(awk -F'\t' -v n="$name" '$1==n{print $2}' "$MAP")"; [ -n "$a2" ] && arm="$a2"; fi
  D="$RUNS/$arm"
  [ "$REDO" != "1" ] && [ -f "$D/.reduced" ] && continue
  rm -rf "$D"; mkdir -p "$D"
  missing=""
  for f in $FILES; do
    src="$(find "$jd" -name "$f" -type f 2>/dev/null | head -1)"
    if [ -n "$src" ]; then cp "$src" "$D/$f"; else missing="$missing $f"; fi
  done
  rc=0; [ -f "$jd/FAILED" ] && rc=1
  ate="$(grep -oP 'ATE=\K[0-9.]+' "$D/results_lio.txt" 2>/dev/null | head -1)"
  python3 - "$D" "$arm" "$name" "$rc" "$ate" "$ARM_TABLE" "$missing" <<'PY'
import csv, json, sys
D, arm, job, rc, ate, table, missing = sys.argv[1:8]
m = {"arm": arm, "job_dir": job, "exit_code": int(rc), "missing": missing.split()}
with open(table) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["arm"] == arm:
            m.update({"tier": r["tier"], "arch": r["arch"], "kind": r["kind"], "yaml": r["yaml"]})
try:
    m["ate_m"] = float(ate)
except ValueError:
    pass
json.dump(m, open(D + "/arm.json", "w"))
PY
  if [ $rc -eq 0 ]; then
    python3 "$HERE/reduce_arm.py" "$D" >> "$D/reduce.log" 2>&1 || echo "reduce failed" >> "$D/reduce.log"
    python3 "$HERE/r62_extra.py" "$D" >> "$D/reduce.log" 2>&1 || echo "r62_extra failed" >> "$D/reduce.log"
  fi
  ( cd "$D" && sha256sum * 2>/dev/null > raw_sha256.txt )
  if [ "$KEEP_ALL_RAW" != "1" ]; then
    case " $KEEP_RAW " in *" $arm "*) ;; *) rm -f "$D/state_trace.csv" "$D/joint_knot_lidar_information.csv" ;; esac
  fi
  touch "$D/.reduced"
done
echo "post_queue done: $(ls -d "$RUNS"/*/ 2>/dev/null | wc -l) arms in $RUNS; next: collect_all.py --runs $RUNS --arms $ARM_TABLE --out <OUT> && python3 collect_r62.py --runs $RUNS --arms $ARM_TABLE --out <OUT>"
