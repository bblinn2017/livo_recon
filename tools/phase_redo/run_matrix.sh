#!/usr/bin/env bash
# R61: run the phase 1/2/3 redo arms with YOUR existing single-run wrapper, reduce each arm on the spot, keep raw files
# only for designated arms. Resumable (an arm with $RUNS/<arm>/.done is skipped).
#
#   RUNNER=<script>   your standard offline run: RUNNER <yaml_after_ntu_viral> <output_dir>  -> runs config/ntu_viral.yaml then the
#                     yaml, offline_bag_path = eee_01, output/debug_log_dir = <output_dir>; exit code = run status
#                     (it must NOT add other overlays; the arm yaml already carries duration, VIO off, eval keys)
#   ARMS_DIR=arms_r61   ARM_TABLE=$ARMS_DIR/arm_table.tsv   RUNS=<runs dir>   JOBS=2
#   [TIERS="B 1 3 2 O"]  [ARCHS="cpl_meas cpl_tail dec_raw dec_spline"]  [MAXPRIO=9]  [ONLY="arm1 arm2"]  [KEEP_RAW="arm1 arm2 ..."]
#   [INCLUDE_OPTIONAL=0]  [KEEP_ALL_RAW=0]
#   [ATE_CMD=<cmd>]   prints the harness ATE (metres) for an arm directory (e.g. parses results_lio.txt); stored as ate_m in arm.json
# Order: priority ascending (1 first). Nothing here interprets results.
set -u
: "${RUNNER:?}" "${RUNS:?}"
ARMS_DIR="${ARMS_DIR:-arms_r61}"; ARM_TABLE="${ARM_TABLE:-$ARMS_DIR/arm_table.tsv}"; JOBS="${JOBS:-2}"
TIERS="${TIERS:-B 1 3 2 O}"; ARCHS="${ARCHS:-cpl_meas cpl_tail dec_raw dec_spline}"; MAXPRIO="${MAXPRIO:-9}"
ONLY="${ONLY:-}"; INCLUDE_OPTIONAL="${INCLUDE_OPTIONAL:-1}"; KEEP_ALL_RAW="${KEEP_ALL_RAW:-0}"
KEEP_RAW="${KEEP_RAW:-base_cpl_meas base_cpl_meas_so p1_cpl_meas_p0_candidate ol_dec_raw_fixed_r5 ol_dec_raw_fixed_r0 p3_cpl_meas_indep_gi p3_cpl_meas_w050_gi B_L0_60s}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$RUNS"
do_arm() {
  arm="$1"; yaml="$2"; tier="$3"; arch="$4"; kind="$5"
  D="$RUNS/$arm"; [ -f "$D/.done" ] && return 0
  rm -rf "$D"; mkdir -p "$D"
  t0=$(date +%s)
  "$RUNNER" "$yaml" "$D" > "$D/run.log" 2>&1
  rc=$?
  wall=$(( $(date +%s) - t0 ))
  ate=""; if [ -n "${ATE_CMD:-}" ]; then ate=$($ATE_CMD "$D" 2>/dev/null | head -1 | awk '{print $1}'); fi
  python3 - "$D" "$arm" "$tier" "$arch" "$kind" "$rc" "$wall" "$yaml" "$ate" <<'PY'
import json, sys
D, arm, tier, arch, kind, rc, wall, yaml, ate = sys.argv[1:10]
m = {"arm": arm, "tier": tier, "arch": arch, "kind": kind, "exit_code": int(rc), "wall_s": int(wall), "yaml": yaml}
try:
    m["ate_m"] = float(ate)
except ValueError:
    pass
json.dump(m, open(D + "/arm.json", "w"))
PY
  if [ $rc -eq 0 ]; then
    python3 "$HERE/reduce_arm.py" "$D" >> "$D/reduce.log" 2>&1 || echo "reduce failed" >> "$D/reduce.log"
  fi
  sha256sum "$D"/*.csv "$D"/*.txt 2>/dev/null > "$D/raw_sha256.txt"
  if [ "$KEEP_ALL_RAW" != "1" ]; then
    case " $KEEP_RAW " in *" $arm "*) ;; *)
      # keep only what the reduction and the audit need; drop the heavy per-iteration files
      find "$D" -maxdepth 1 \( -name 'joint_knot_all_scans.csv' -o -name 'joint_knot_iterations.csv' -o -name 'joint_knot_states.csv' \
        -o -name 'joint_knot_corrections.csv' -o -name 'stationary_iteration.csv' -o -name 'gating_all_scans.csv' -o -name 'initialization_consistency_all_scans.csv' \
        -o -name 'state_trace.csv' \) -delete ;;
    esac
  fi
  touch "$D/.done"
}
export -f do_arm
export RUNNER RUNS HERE KEEP_RAW KEEP_ALL_RAW ATE_CMD
tail -n +2 "$ARM_TABLE" | sort -t$'\t' -k8,8n | while IFS=$'\t' read -r arm tier arch kind dur yaml notes prio optional; do
  [ "$prio" -le "$MAXPRIO" ] || continue
  case " $TIERS " in *" $tier "*) ;; *) continue;; esac
  case " $ARCHS " in *" $arch "*) ;; *) continue;; esac
  [ "$optional" = "1" ] && [ "$INCLUDE_OPTIONAL" != "1" ] && continue
  if [ -n "$ONLY" ]; then case " $ONLY " in *" $arm "*) ;; *) continue;; esac; fi
  printf '%s\t%s\t%s\t%s\t%s\0' "$arm" "$yaml" "$tier" "$arch" "$kind"
done | xargs -0 -P "$JOBS" -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r a y t r k <<< "$1"; do_arm "$a" "$y" "$t" "$r" "$k"' _ {}
echo "done; run collect_all.py --runs $RUNS --arms $ARM_TABLE --out <out>"
