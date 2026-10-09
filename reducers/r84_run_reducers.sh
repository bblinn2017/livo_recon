#!/usr/bin/env bash
# R84 reducers, in order; stops at the first failure (a non-zero exit, including any missing arm). Usage: bash r84_run_reducers.sh [RET] [MANIFEST]
#   RET       the return dir (holds phase_summary.csv and, if returned, r84_sample_window_engagement.csv; outputs are written here)   default R84_RETURN
#   MANIFEST  the manifest dir (arm_table.tsv, l2_reference_values.csv)                                                              default r84_manifest
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RET="${1:-R84_RETURN}"
MAN="${2:-r84_manifest}"
python3 -I "$HERE/r84_t3_reduce.py" --phase-summary "$RET/phase_summary.csv" --manifest "$MAN" --out "$RET"
python3 -I "$HERE/r84_cmp_reduce.py" --phase-summary "$RET/phase_summary.csv" --manifest "$MAN" --engagement "$RET/r84_sample_window_engagement.csv" --out "$RET"
