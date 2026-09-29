#!/usr/bin/env bash
# R61: after (or while) `dispatch_queue.sh <queue_dir>` drains the arm queue, collect each finished job's diagnostics into
# $RUNS/<arm>/, write arm.json, run reduce_arm.py, and prune heavy raw files (keeping full raw for KEEP_RAW arms).
# This script does NOT run or schedule jobs: all dispatching is done by dispatch_queue.sh (standing rule 5.6a).
#
#   QUEUE_DIR=<job queue dir the dispatcher drained>   RUNS=<output dir>   ARM_TABLE=arms_r61/arm_table.tsv
#   [MAP=<tsv: job_dir_name<TAB>arm>]   only if the queue's job directory names differ from the arm names
#   [KEEP_RAW="arm1 arm2 ..."]  [KEEP_ALL_RAW=0]  [REDO=0]
# A job counts as finished when its dir holds DONE (exit_code 0) or FAILED (exit_code 1), the dispatcher's own sentinels.
# ATE is parsed from the job's results_lio.txt as the dispatcher does (`ATE=<metres>`).
# Diagnostic files are located with `find` anywhere under the job dir; a file that is not found is listed in arm.json
# under "missing" (the agent must then say where those files land, see the instructions).
set -u
: "${QUEUE_DIR:?}" "${RUNS:?}" "${ARM_TABLE:?}"
KEEP_RAW="${KEEP_RAW:-base_cpl_meas base_cpl_meas_so p1_cpl_meas_p0_candidate ol_dec_raw_fixed_r5 ol_dec_raw_fixed_r0 p3_cpl_meas_indep_gi p3_cpl_meas_w050_gi B_L0_60s}"
KEEP_ALL_RAW="${KEEP_ALL_RAW:-0}"; REDO="${REDO:-0}"; MAP="${MAP:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
FILES="state_trace.csv state_reference.txt imu_cov_growth.csv lio_gt_matched.csv motion_q_scan.csv joint_knot_lidar_information.csv results_lio.txt"
mkdir -p "$RUNS"
for jd in "$QUEUE_DIR"/*/; do
  jd="${jd%/}"; name="$(basename "$jd")"
  [ -f "$jd/DONE" ] || [ -f "$jd/FAILED" ] || continue
  arm="$name"
  if [ -n "$MAP" ] && [ -f "$MAP" ]; then a2="$(awk -F'\t' -v n="$name" '$1==n{print $2}' "$MAP")"; [ -n "$a2" ] && arm="$a2"; fi
  D="$RUNS/$arm"
  [ "$REDO" != "1" ] && [ -f "$D/.reduced" ] && continue
  rm -rf "$D"; mkdir -p "$D"
  # R61 agent fix (coding agent, 2026-09-29): gen_jobs.py's livo_recon branch (ntu GT mode) does
  # not wire outputs/path or outputs/debug_log_dir into the job dir at all -- it routes them to
  # livo_recon_results/<batch>/<arm> (OUT_DIR, a separate tree from $QUEUE_DIR/<arm>) via a
  # manifest-level override this round added specifically to give these diagnostics a per-job
  # home instead of the shared /tmp default. `find "$jd" ...` alone therefore never finds
  # state_trace.csv/state_reference.txt/imu_cov_growth.csv/lio_gt_matched.csv/motion_q_scan.csv/
  # joint_knot_lidar_information.csv (confirmed missing from every job dir during this round's
  # dispatch) -- only results_lio.txt is (run_job.sh writes/moves it into JOB_DIR directly, see
  # its "results_lio.txt write race" comment). Read OUT_DIR out of the job's own job.env (written
  # by gen_jobs.py, sourced fresh, not exported) and search there too.
  OUT_DIR=""
  if [ -f "$jd/job.env" ]; then
    OUT_DIR="$(grep -m1 '^OUT_DIR=' "$jd/job.env" | cut -d= -f2-)"
    # OUT_DIR in job.env is a CONTAINER path (/root/catkin_ws/...); this script runs on the host,
    # where the same tree is bind-mounted at /home/bblinn/fast_ws/.
    OUT_DIR="${OUT_DIR/#\/root\/catkin_ws/\/home\/bblinn\/fast_ws}"
  fi
  missing=""
  for f in $FILES; do
    src="$(find "$jd" -name "$f" -type f 2>/dev/null | head -1)"
    if [ -z "$src" ] && [ -n "$OUT_DIR" ] && [ -d "$OUT_DIR" ]; then
      src="$(find "$OUT_DIR" -name "$f" -type f 2>/dev/null | head -1)"
    fi
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
  if [ $rc -eq 0 ]; then python3 "$HERE/reduce_arm.py" "$D" >> "$D/reduce.log" 2>&1 || echo "reduce failed" >> "$D/reduce.log"; fi
  ( cd "$D" && sha256sum * 2>/dev/null > raw_sha256.txt )
  if [ "$KEEP_ALL_RAW" != "1" ]; then
    case " $KEEP_RAW " in *" $arm "*) ;; *) rm -f "$D/state_trace.csv" "$D/joint_knot_lidar_information.csv" ;; esac
  fi
  touch "$D/.reduced"
done
echo "post_queue done: $(ls -d "$RUNS"/*/ 2>/dev/null | wc -l) arms in $RUNS; next: collect_all.py --runs $RUNS --arms $ARM_TABLE --out <OUT>"
