#!/usr/bin/env bash
# R64: after `dispatch_queue.sh <queue_dir>` has drained the arm queue, collect each finished job's diagnostics into
# $RUNS/<arm>/, write arm.json, run r64_reduce.py, prune heavy raw files. Does NOT run or schedule jobs (rule 5.6a).
#   QUEUE_DIR=<queue dir the dispatcher drained>  RUNS=<output dir>  ARM_TABLE=r64_manifest/arm_table.tsv  [REDO=0] [KEEP_ALL_RAW=0]
# Finished = job dir holds DONE or FAILED. ATE from results_lio.txt. Files are searched under the job dir AND under OUT_DIR
# (from job.env, /root/catkin_ws mapped to /home/bblinn/fast_ws as in R62/R63); a file not found is listed in arm.json "missing".
# wall_s = mtime(DONE or FAILED) - mtime(job.env). Raw state_trace.csv is kept for arms whose arm_table keep_raw column is 1;
# plane_fit_trace.csv is gzipped and kept for every arm that wrote one.
set -u
: "${QUEUE_DIR:?}" "${RUNS:?}" "${ARM_TABLE:?}"
KEEP_ALL_RAW="${KEEP_ALL_RAW:-0}"; REDO="${REDO:-0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
FILES="state_trace.csv state_reference.txt lio_gt_matched.csv motion_q_scan.csv results_lio.txt gravity_alignment.txt imu_first_scans.csv calibration_p0.txt config.yaml frame_stats.txt plane_fit_reasons.csv plane_fit_trace.csv run.log"
mkdir -p "$RUNS"
KEEP_LIST="$(awk -F'\t' 'NR>1 && $10==1{printf "%s ", $1}' "$ARM_TABLE")"
for jd in "$QUEUE_DIR"/*/; do
  jd="${jd%/}"; name="$(basename "$jd")"
  [ -f "$jd/DONE" ] || [ -f "$jd/FAILED" ] || continue
  arm="$name"; D="$RUNS/$arm"
  [ "$REDO" != "1" ] && [ -f "$D/.reduced" ] && continue
  rm -rf "$D"; mkdir -p "$D"
  OUT_DIR=""
  if [ -f "$jd/job.env" ]; then
    OUT_DIR="$(grep -m1 '^OUT_DIR=' "$jd/job.env" | cut -d= -f2-)"
    OUT_DIR="${OUT_DIR/#\/root\/catkin_ws/\/home\/bblinn\/fast_ws}"
  fi
  missing=""
  for f in $FILES; do
    src="$(find "$jd" -name "$f" -type f 2>/dev/null | head -1)"
    if [ -z "$src" ] && [ -n "$OUT_DIR" ] && [ -d "$OUT_DIR" ]; then src="$(find "$OUT_DIR" -name "$f" -type f 2>/dev/null | head -1)"; fi
    if [ -n "$src" ]; then cp "$src" "$D/$f"; else missing="$missing $f"; fi
  done
  rc=0; [ -f "$jd/FAILED" ] && rc=1
  fin="$jd/DONE"; [ -f "$jd/FAILED" ] && fin="$jd/FAILED"
  wall=""; if [ -f "$jd/job.env" ]; then wall=$(( $(stat -c %Y "$fin") - $(stat -c %Y "$jd/job.env") )); fi
  ate="$(grep -oP 'ATE=\K[0-9.]+' "$D/results_lio.txt" 2>/dev/null | head -1)"
  python3 - "$D" "$arm" "$name" "$rc" "$ate" "$ARM_TABLE" "$missing" "$wall" <<'PY'
import csv, json, sys
D, arm, job, rc, ate, table, missing, wall = sys.argv[1:9]
m = {"arm": arm, "job_dir": job, "exit_code": int(rc), "missing": missing.split()}
if wall != "":
    m["wall_s"] = int(wall)
with open(table) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["arm"] == arm:
            m.update({"tier": r["tier"], "arch": r["arch"], "kind": r["kind"], "yaml": r["yaml"], "duration_s": r["duration_s"]})
try:
    m["ate_m"] = float(ate)
except ValueError:
    pass
json.dump(m, open(D + "/arm.json", "w"))
PY
  if [ $rc -eq 0 ]; then
    python3 "$HERE/r64_reduce.py" "$D" >> "$D/reduce.log" 2>&1 || echo "reduce failed" >> "$D/reduce.log"
  else
    tail -n 40 "$D/run.log" > "$D/run_log_tail.txt" 2>/dev/null
  fi
  ( cd "$D" && sha256sum * 2>/dev/null > raw_sha256.txt )
  # identity controls compare state_trace.csv with the run_id column blanked: store that digest for EVERY arm
  if [ -f "$D/state_trace.csv" ]; then
    python3 - "$D/state_trace.csv" > "$D/state_trace_blank_runid.sha256" <<'PY'
import csv, hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], newline="") as f:
    r = csv.reader(f)
    for i, row in enumerate(r):
        if i > 0:
            row[0] = ""
        h.update((",".join(row) + "\n").encode())
print(h.hexdigest())
PY
  fi
  [ -f "$D/plane_fit_trace.csv" ] && gzip -f "$D/plane_fit_trace.csv"
  rm -f "$D/run.log"
  if [ "$KEEP_ALL_RAW" != "1" ]; then
    case " $KEEP_LIST " in *" $arm "*) [ -f "$D/state_trace.csv" ] && gzip -f "$D/state_trace.csv" ;; *) rm -f "$D/state_trace.csv" ;; esac
  fi
  [ -f "$D/lio_gt_matched.csv" ] && case " $KEEP_LIST " in *" $arm "*) gzip -f "$D/lio_gt_matched.csv" ;; *) rm -f "$D/lio_gt_matched.csv" ;; esac
  [ -f "$D/frame_stats.txt" ] && case " $KEEP_LIST " in *" $arm "*) gzip -f "$D/frame_stats.txt" ;; *) rm -f "$D/frame_stats.txt" ;; esac
  touch "$D/.reduced"
done
echo "post_queue done: $(ls -d "$RUNS"/*/ 2>/dev/null | wc -l) arms in $RUNS"
echo "next: python3 tools/r64/r64_collect.py --runs $RUNS --arms $ARM_TABLE --out <OUT>"
