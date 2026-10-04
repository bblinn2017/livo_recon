#!/usr/bin/env bash
# R76 (F-448): run the offline voxel-size statistics tool on all nine bags and collect the estimates. NO odometry is run; no bag is run with an estimated value.
#   bash tools/r69/run_r76_voxel_stats.sh <bag search dir> [<bag search dir> ...] --out R76_RETURN/voxel_stats
# Needs python3 with rosbag (the same interpreter that runs bag_stats.py), numpy, scipy, pandas. One --out directory per bag (voxel_stats_log.txt is written per bag).
# Order: (1) selftest, (2) the nine bags, (3) --collect. A failing bag is reported in voxel_stats_run_log.txt and does not stop the others; the exit code is the number of failed steps.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=""; DIRS=()
while [ $# -gt 0 ]; do case "$1" in --out) OUT="$2"; shift 2;; *) DIRS+=("$1"); shift;; esac; done
[ -n "$OUT" ] && [ ${#DIRS[@]} -gt 0 ] || { echo "usage: $0 <bag search dir>... --out <dir>"; exit 2; }
mkdir -p "$OUT"; LOG="$OUT/voxel_stats_run_log.txt"; : > "$LOG"; FAIL=0
echo "$(date -u +%FT%TZ) selftest" | tee -a "$LOG"
python3 "$HERE/r76_voxel_stats.py" --synthetic-selftest >> "$LOG" 2>&1 || { echo "selftest FAILED (see $LOG)"; FAIL=$((FAIL+1)); }
SEQS="eee_01 eee_02 eee_03 nya_01 nya_02 nya_03 sbs_01 sbs_02 sbs_03"
for s in $SEQS; do
  bag="$(find "${DIRS[@]}" -name "$s.bag" 2>/dev/null | head -1)"
  if [ -z "$bag" ]; then echo "$s: bag NOT FOUND under ${DIRS[*]}" | tee -a "$LOG"; FAIL=$((FAIL+1)); continue; fi
  echo "$(date -u +%FT%TZ) $s  $bag" | tee -a "$LOG"
  python3 "$HERE/r76_voxel_stats.py" --bag "$bag" --seq "$s" --out "$OUT/$s" >> "$LOG" 2>&1 || { echo "$s: FAILED (see $LOG)" | tee -a "$LOG"; FAIL=$((FAIL+1)); }
done
echo "$(date -u +%FT%TZ) collect" | tee -a "$LOG"
python3 "$HERE/r76_voxel_stats.py" --collect "$OUT"/*/ --out "$OUT" >> "$LOG" 2>&1 || { echo "collect FAILED (see $LOG)" | tee -a "$LOG"; FAIL=$((FAIL+1)); }
ls -l "$OUT" | tee -a "$LOG"
exit $FAIL
