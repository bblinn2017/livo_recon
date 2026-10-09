#!/usr/bin/env bash
# R82 hold check (planning agent's script; selftest in a scratch directory only, planner sandbox, not evidence).
#
#   bash tools/r69/r82_hold_check.sh [--quiet] [--wait SECONDS] [DIR ...]     exit 0 = no hold, 75 = HOLD, 2 = usage
#   bash tools/r69/r82_hold_check.sh --selftest
#
# A hold exists when a file named exactly HOLD.txt is found in any of: each DIR given, $R82_HOLD_DIR, ablations/_job_queue (relative to the current directory), /root/catkin_ws/ablations/_job_queue,
# /home/bblinn/fast_ws/ablations/_job_queue (the container and host spellings of the same directory, H-19), and the current directory.
# HOLD.txt is written by r82_identity_hold.py (identity FAIL after queue r82_id) and, as HOLD_FROM_GATES.txt -> HOLD.txt by the agent, by r82_gates.py --hold-file (a DET or IDENT FAIL after a bag queue).
#
# On a hold the script prints the first 25 lines of the HOLD.txt it found to stderr and exits 75. With --wait it instead polls every SECONDS until the file is gone/renamed and then exits 0.
# It NEVER deletes, moves, edits or creates any file and never touches a process: it only reads. A hold is lifted only by a person renaming HOLD.txt to HOLD.released_<date>.txt on the planner's word;
# a renamed file is ignored here.
set -u
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
find_hold() {
  local d
  for d in "$@" "${R82_HOLD_DIR:-}" "ablations/_job_queue" "/root/catkin_ws/ablations/_job_queue" "/home/bblinn/fast_ws/ablations/_job_queue" "."; do
    [ -n "$d" ] && [ -f "$d/HOLD.txt" ] && { printf '%s\n' "$d/HOLD.txt"; return 0; }
  done
  return 1
}
check_once() {
  local f
  if f=$(find_hold "$@"); then
    [ "$QUIET" = 1 ] || { echo "HOLD in force: $f (r82_hold_check.sh: do not dispatch; cancel nothing)" >&2; head -n 25 "$f" >&2; }
    return 75
  fi
  [ "$QUIET" = 1 ] || echo "no HOLD.txt found; dispatch may proceed" >&2
  return 0
}
selftest() {
  local t rc fails=""
  t=$(mktemp -d) || exit 2
  mkdir -p "$t/q" "$t/other"
  cd "$t" || exit 2   # hermetic: the relative search directories (ablations/_job_queue, .) are inside the scratch directory
  unset R82_HOLD_DIR
  QUIET=1
  check_once "$t/q"; rc=$?; [ $rc = 0 ] || fails="$fails no-hold-should-be-0(got $rc);"
  echo "HOLD: identity FAIL r82_dc_id_mg_eee_01_fq" > "$t/q/HOLD.txt"
  check_once "$t/q"; rc=$?; [ $rc = 75 ] || fails="$fails hold-should-be-75(got $rc);"
  check_once "$t/other"; rc=$?; [ $rc = 0 ] || fails="$fails other-dir-should-be-0(got $rc);"
  QUIET=0; check_once "$t/q" 2> "$t/err"; rc=$?
  [ $rc = 75 ] && grep -q "identity FAIL r82_dc_id_mg_eee_01_fq" "$t/err" || fails="$fails hold-text-not-shown;"
  # the file is untouched and still there
  [ "$(cat "$t/q/HOLD.txt")" = "HOLD: identity FAIL r82_dc_id_mg_eee_01_fq" ] || fails="$fails hold-file-changed;"
  # wait mode: release from a background job after 2 s, the waiter must return 0 and not before
  ( sleep 2; mv "$t/q/HOLD.txt" "$t/q/HOLD.released_test.txt" ) &
  local t0 t1
  t0=$(date +%s); bash "$SELF" --quiet --wait 1 "$t/q"; rc=$?; t1=$(date +%s)
  [ $rc = 0 ] && [ $((t1 - t0)) -ge 1 ] || fails="$fails wait-mode(rc $rc, waited $((t1 - t0)) s);"
  wait
  # a renamed file is ignored; nothing was deleted
  QUIET=1; check_once "$t/q"; rc=$?; [ $rc = 0 ] || fails="$fails released-should-be-0(got $rc);"
  [ -f "$t/q/HOLD.released_test.txt" ] || fails="$fails released-file-lost;"
  cd / && [ -n "$t" ] && rm -r -- "$t"
  if [ -z "$fails" ]; then echo "SELFTEST r82_hold_check PASS (no hold -> 0; HOLD.txt -> 75 with its text shown; other directory -> 0; file never modified; --wait blocks until the file is renamed, then 0; renamed file ignored)"; exit 0; fi
  echo "SELFTEST r82_hold_check FAIL:$fails"; exit 1
}
QUIET=0; WAIT=""; DIRS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --selftest) selftest ;;
    --quiet) QUIET=1 ;;
    --wait) shift; WAIT="${1:-}"; case "$WAIT" in ''|*[!0-9]*) echo "usage: --wait SECONDS" >&2; exit 2 ;; esac ;;
    -h|--help) sed -n 2,16p "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) DIRS+=("$1") ;;
  esac
  shift
done
if [ -n "$WAIT" ]; then
  while true; do
    check_once "${DIRS[@]+"${DIRS[@]}"}" >/dev/null 2>&1 && exit 0
    [ "$QUIET" = 1 ] || echo "HOLD in force; polling every $WAIT s (nothing is dispatched, nothing is cancelled)" >&2
    sleep "$WAIT"
  done
fi
check_once "${DIRS[@]+"${DIRS[@]}"}"
exit $?
