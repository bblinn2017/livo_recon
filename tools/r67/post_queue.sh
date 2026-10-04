#!/usr/bin/env bash
# R67: after `dispatch_queue.sh <queue_dir> <containers file>` has drained the arm queue, collect each finished job's diagnostics into
# $RUNS/<arm>/, write arm.json, run r67_reduce.py and r67_compress.py, prune heavy raw files. Does NOT run or schedule jobs (rule 5.6a).
#   QUEUE_DIR=<queue dir the dispatcher drained>  RUNS=<output dir>  ARM_TABLE=r67_manifest/arm_table.tsv  [BAG_PARAMS=bag_params.json] [REDO=0] [KEEP_ALL_RAW=0]
# Finished = job dir holds DONE or FAILED. ATE from results_lio.txt. Files are searched under the job dir AND under OUT_DIR
# (from job.env, /root/catkin_ws mapped to /home/bblinn/fast_ws as in R62-R65); a file not found is listed in arm.json "missing".
# wall_s = mtime(DONE or FAILED) - mtime(job.env). The full raw files stay in the queue results directory (untouched) with sha256 in raw_sha256.txt and
# compress_manifest.json; the copies under $RUNS/<arm>/ are compacted (keep arms: stamps, plane_trace, numeric, gt, small_logs; every other arm: stamps, plane_trace, small_logs_min).
# Arm table columns (R67): arm tier arch kind duration_s seq notes keep_raw dup_of.
set -u
: "${QUEUE_DIR:?}" "${RUNS:?}" "${ARM_TABLE:?}"
KEEP_ALL_RAW="${KEEP_ALL_RAW:-1}"; REDO="${REDO:-0}"; BAG_PARAMS="${BAG_PARAMS:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
FILES="imu_cov_growth.csv post_calibration_voxelmap.csv bootstrap_preparation.txt bootstrap_map_seed.txt state_trace.csv state_reference.txt lio_gt_matched.csv motion_q_scan.csv results_lio.txt gravity_alignment.txt imu_first_scans.csv calibration_p0.txt config.yaml frame_stats.txt plane_fit_reasons.csv plane_fit_trace.csv sensor_stamps.csv point_cov_stats.csv surface_map_frames.csv frame_window.csv akf_q.csv coupled_scan_status.csv engagement.txt run.log"
mkdir -p "$RUNS"
KEEP_LIST="$(awk -F'\t' 'NR>1 && $8==1{printf "%s ", $1}' "$ARM_TABLE")"
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
            m.update({"tier": r["tier"], "arch": r["arch"], "kind": r["kind"], "duration_s": r["duration_s"], "seq": r["seq"]})
try:
    m["ate_m"] = float(ate)
except ValueError:
    pass
json.dump(m, open(D + "/arm.json", "w"))
PY
  SEQ="$(python3 -c "import json;print(json.load(open('$D/arm.json')).get('seq','eee_01'))")"
  ONSET=485
  if [ "$SEQ" != "eee_01" ] && [ -n "$BAG_PARAMS" ] && [ -f "$BAG_PARAMS" ]; then
    ONSET="$(python3 -c "import json;print(json.load(open('$BAG_PARAMS'))['$SEQ']['onset_scan'])")"
  fi
  if [ $rc -eq 0 ]; then
    if [ -n "$BAG_PARAMS" ] && [ -f "$BAG_PARAMS" ]; then python3 "$HERE/r67_reduce.py" "$D" --params "$BAG_PARAMS" >> "$D/reduce.log" 2>&1 || echo "reduce failed" >> "$D/reduce.log"
    else python3 "$HERE/r67_reduce.py" "$D" >> "$D/reduce.log" 2>&1 || echo "reduce failed" >> "$D/reduce.log"; fi
    KINDS="stamps,plane_trace,small_logs_min"
    case " $KEEP_LIST " in *" $arm "*) KINDS="stamps,plane_trace,numeric,gt,small_logs" ;; esac
    python3 "$HERE/r67_compress.py" "$D" --kinds "$KINDS" --onset "$ONSET" >> "$D/compress.log" 2>&1 || echo "compress failed" >> "$D/compress.log"
  else
    tail -n 40 "$D/run.log" > "$D/run_log_tail.txt" 2>/dev/null
  fi
  if [ -f "$D/state_trace.csv" ] && [ $rc -eq 0 ]; then
    python3 "$HERE/bootstrap_reduce.py" "$D" >> "$D/bootstrap_reduce.log" 2>&1 || echo "bootstrap_reduce failed" >> "$D/bootstrap_reduce.log"
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
  for big in plane_fit_trace.csv sensor_stamps.csv surface_map_frames.csv frame_window.csv akf_q.csv coupled_scan_status.csv motion_q_scan.csv; do [ -f "$D/$big" ] && rm -f "$D/$big"; done   # compacted above; the originals stay in the queue results directory
  [ -f "$D/point_cov_stats.csv" ] && gzip -f "$D/point_cov_stats.csv"
  rm -f "$D/run.log"
  if [ "$KEEP_ALL_RAW" != "1" ]; then
    case " $KEEP_LIST " in *" $arm "*) [ -f "$D/state_trace.csv" ] && gzip -f "$D/state_trace.csv" ;; *) rm -f "$D/state_trace.csv" ;; esac
  fi
  [ -f "$D/lio_gt_matched.csv" ] && case " $KEEP_LIST " in *" $arm "*) gzip -f "$D/lio_gt_matched.csv" ;; *) rm -f "$D/lio_gt_matched.csv" ;; esac
  [ -f "$D/frame_stats.txt" ] && case " $KEEP_LIST " in *" $arm "*) gzip -f "$D/frame_stats.txt" ;; *) rm -f "$D/frame_stats.txt" ;; esac
  touch "$D/.reduced"
done
echo "post_queue done: $(ls -d "$RUNS"/*/ 2>/dev/null | wc -l) arms in $RUNS"
echo "next: python3 tools/r67/p_collect.py $RUNS --out P_RETURN --arms <(ls $RUNS) --gt ; python3 tools/r67/build_return.py ..."
