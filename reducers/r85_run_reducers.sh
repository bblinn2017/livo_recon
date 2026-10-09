#!/usr/bin/env bash
# R85 reducers: the L0 reducer, then the SHT reducer. Both always run (a missing arm in the first does not skip the second); the exit code is 0 only when both exit 0.
# Usage: bash r85_run_reducers.sh [RET] [MANIFEST] [R84_ALL_ARMS_CSV]
#   RET              the return dir (holds phase_summary.csv and compact/<arm>/short_scan_all.csv.gz; outputs are written here)   default R85_RETURN
#   MANIFEST         the manifest dir (arm_table.tsv, l2_reference_values.csv, onset_scans.json)                                   default r85_manifest
#   R84_ALL_ARMS_CSV optional: r84_layer0_vs_layer2_all_arms.csv (edges 0.30-0.70 m joined into r85_l0_unified.csv when its header is recognised, else NOT JOINED)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RET="${1:-R85_RETURN}"
MAN="${2:-r85_manifest}"
R84="${3:-}"
rc=0
if [ -n "$R84" ]; then
  python3 -I "$HERE/r85_l0_reduce.py" --phase-summary "$RET/phase_summary.csv" --manifest "$MAN" --out "$RET" --r84-all-arms "$R84" || rc=1
else
  python3 -I "$HERE/r85_l0_reduce.py" --phase-summary "$RET/phase_summary.csv" --manifest "$MAN" --out "$RET" || rc=1
fi
python3 -I "$HERE/r85_sht_reduce.py" --ret "$RET" --manifest "$MAN" --out "$RET" || rc=1
exit $rc
