#!/usr/bin/env bash
# r83_rss_watch.sh -- memory watcher for the D4P arms (queue r83_mem). Wraps nothing in the node: it only READS /proc of a job that is already running, and
# kills that job's process tree if its resident memory passes the abort threshold. Not a node change; no node file, config or key is touched.
#
# USAGE
#   r83_rss_watch.sh --arm-dir DIR (--pid PID | --pattern REGEX) [--interval S] [--abort-kb KB] [--wait-start S] [--kill-cmd 'CMD']
#   r83_rss_watch.sh --print-threshold          print host_ram_kb and the default abort threshold (MemTotal / 2) and exit
#   r83_rss_watch.sh --selftest
#
#   --arm-dir DIR     the arm's result folder; the watcher writes DIR/peak_rss.txt and, on an abort, DIR/ABORTED_RSS
#   --pid PID         the job's root process id (the process whose descendants make up the job)
#   --pattern REGEX   instead of a pid: the newest process whose command line matches REGEX (pgrep -f); the watcher, its subshells and its ancestors are excluded. The watcher waits up to
#                     --wait-start seconds (default 300) for the process to appear, so it can be started BEFORE the job.
#   --interval S      poll period in seconds (default 2)
#   --abort-kb KB     abort threshold in kB; default = half of MemTotal from /proc/meminfo (the rule: abort when RSS exceeds HALF of the host RAM). The override exists for the selftest.
#   --kill-cmd CMD    run CMD instead of signalling the tree (for a job inside a container whose processes the watcher cannot signal, e.g. 'docker kill <name>');
#                     the marker is written either way.
#
# WHAT IT MEASURES: the sum of VmRSS (kB, /proc/<pid>/status) over the root process and ALL of its descendants, every --interval seconds. Shared pages (for example
# a shared library mapped by several processes) are counted once per process, so the sum can overstate the true footprint of a multi-process job; for a single node
# process it is its VmRSS. A job inside a container is read from the host's /proc through its host pid (the watcher must run on the host and see that pid).
#
# OUTPUT peak_rss.txt: ONE line, rewritten whenever the peak grows and once more at the end:
#   peak_rss_kb=<int> host_ram_kb=<int> abort_threshold_kb=<int>
# ABORT: when the sum exceeds the threshold the watcher sends SIGTERM to the whole tree (leaves first), waits 5 s, sends SIGKILL to what is left (or runs --kill-cmd),
# writes DIR/ABORTED_RSS (first line: the RSS in kB that triggered it; second line: UTC time) and exits 3. Exit codes: 0 job ended normally, 3 aborted,
# 4 the process never appeared within --wait-start, 2 usage.
# The job's own exit code is not visible to the watcher; the queue sees the job die and goes on with its next arm (what the queue does with a killed job is the host's).
set -u

interval=2; abort_kb=""; wait_start=300; kill_cmd=""; arm_dir=""; pid=""; pattern=""

host_ram_kb() { awk '/^MemTotal:/{print $2}' /proc/meminfo; }

tree_pids() {   # root pid + all descendants, one per line
  ps -e -o pid=,ppid= | awk -v root="$1" '{p[$1]=$2; id[NR]=$1} END{keep[root]=1; ch=1; while(ch){ch=0; for(i in id){c=id[i]; if(!(c in keep) && (p[c] in keep)){keep[c]=1; ch=1}}} for(k in keep) print k}'
}

tree_rss_kb() {
  local total=0 p v
  for p in $(tree_pids "$1"); do
    v=$(awk '/^VmRSS:/{print $2}' "/proc/$p/status" 2>/dev/null)
    [ -n "$v" ] && total=$((total + v))
  done
  echo "$total"
}

write_peak() {   # $1 peak $2 host $3 threshold ; atomic replace
  printf 'peak_rss_kb=%s host_ram_kb=%s abort_threshold_kb=%s\n' "$1" "$2" "$3" > "$arm_dir/peak_rss.txt.tmp" && mv -f "$arm_dir/peak_rss.txt.tmp" "$arm_dir/peak_rss.txt"
}

kill_tree() {
  local root="$1" p pids
  if [ -n "$kill_cmd" ]; then bash -c "$kill_cmd" >/dev/null 2>&1; return; fi
  pids=$(tree_pids "$root" | sort -rn)          # higher pids first: children before parents in the usual case
  for p in $pids; do kill -TERM "$p" 2>/dev/null; done
  sleep 5
  for p in $(tree_pids "$root"); do kill -KILL "$p" 2>/dev/null; done
}

find_by_pattern() {   # newest (highest pid) process whose command line matches $pattern, excluding the watcher, its subshells and its ancestors
  local skip p
  skip=$( { tree_pids "$$"; p=$$; while [ "$p" -gt 1 ] 2>/dev/null; do echo "$p"; p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' '); [ -n "$p" ] || break; done; } | sort -u)
  pgrep -f -- "$pattern" | grep -v -x -F "$skip" | sort -n | tail -n 1
}

watch() {
  [ -n "$arm_dir" ] && [ -d "$arm_dir" ] || { echo "r83_rss_watch: --arm-dir must be an existing directory" >&2; return 2; }
  [ -n "$pid" ] || [ -n "$pattern" ] || { echo "r83_rss_watch: --pid or --pattern required" >&2; return 2; }
  local host thr root t0 now rss peak=0
  host=$(host_ram_kb)
  if [ -n "$abort_kb" ]; then thr="$abort_kb"; else thr=$((host / 2)); fi
  root="$pid"
  if [ -z "$root" ]; then
    t0=$(date +%s)
    while :; do
      root=$(find_by_pattern)
      [ -n "$root" ] && break
      now=$(date +%s); [ $((now - t0)) -ge "$wait_start" ] && { echo "r83_rss_watch: no process matching '$pattern' within ${wait_start}s" >&2; return 4; }
      sleep "$interval"
    done
  fi
  [ -d "/proc/$root" ] || { echo "r83_rss_watch: pid $root not running" >&2; return 4; }
  rm -f "$arm_dir/ABORTED_RSS"
  write_peak 0 "$host" "$thr"
  while [ -d "/proc/$root" ]; do
    rss=$(tree_rss_kb "$root")
    if [ "$rss" -gt "$peak" ]; then peak="$rss"; write_peak "$peak" "$host" "$thr"; fi
    if [ "$rss" -gt "$thr" ]; then
      printf '%s\n%s\n' "$rss" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$arm_dir/ABORTED_RSS"
      echo "r83_rss_watch: ABORTED_RSS rss ${rss} kB > threshold ${thr} kB (pid $root); killing the tree" >&2
      kill_tree "$root"
      write_peak "$peak" "$host" "$thr"
      return 3
    fi
    sleep "$interval"
  done
  write_peak "$peak" "$host" "$thr"
  return 0
}

selftest() {
  local d fails=0 n=0 self rc dummy pidf
  self=$(readlink -f "$0"); d=$(mktemp -d)
  check() { n=$((n + 1)); if [ "$2" = "1" ]; then printf '%-62s pass  %s\n' "$1" "${3:-}"; else printf '%-62s FAIL  %s\n' "$1" "${3:-}"; fails=$((fails + 1)); fi; }
  # dummy job: python allocating STEP_MB every 0.3 s up to TOTAL_MB, then holding for HOLD s; CHILD=1 puts the allocation into a child process
  cat > "$d/dummy.py" <<'PY'
import os, subprocess, sys, time
total, step, hold, child = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == "1"
if child:
    subprocess.Popen([sys.executable, sys.argv[0], str(total), str(step), str(hold), "0"]).wait()
    sys.exit(0)
buf = []
got = 0
while got < total:
    buf.append(bytearray(os.urandom(1)) * (step * 1024 * 1024)); got += step
    for b in buf: b[::4096] = b"x" * len(b[::4096])   # touch every page
    time.sleep(0.3)
time.sleep(hold)
PY
  host=$(host_ram_kb)
  # 1 default threshold is half of MemTotal
  out=$("$self" --print-threshold); check "1 default threshold = MemTotal / 2" "$([ "$out" = "host_ram_kb=$host abort_threshold_kb=$((host / 2))" ] && echo 1 || echo 0)" "$out"
  # 2 normal case: 80 MB, threshold 400000 kB, completes, no marker
  mkdir "$d/a1"; python3 "$d/dummy.py" 80 20 1 0 & pid=$!
  "$self" --arm-dir "$d/a1" --pid "$pid" --interval 0.2 --abort-kb 400000; rc=$?; wait "$pid" 2>/dev/null
  line=$(cat "$d/a1/peak_rss.txt")
  peak=$(echo "$line" | sed -n 's/^peak_rss_kb=\([0-9]*\) .*/\1/p')
  check "2 normal job: exit 0, no marker" "$([ $rc -eq 0 ] && [ ! -e "$d/a1/ABORTED_RSS" ] && echo 1 || echo 0)" "rc $rc"
  check "2 peak recorded at least the 80 MB held" "$([ -n "$peak" ] && [ "$peak" -ge 80000 ] && echo 1 || echo 0)" "$line"
  check "2 one line, three fields, host_ram from /proc/meminfo, threshold = override" "$(echo "$line" | grep -Eq "^peak_rss_kb=[0-9]+ host_ram_kb=$host abort_threshold_kb=400000$" && [ "$(wc -l < "$d/a1/peak_rss.txt")" = 1 ] && echo 1 || echo 0)" "$line"
  # 3 abort case: 600 MB allocation against 150000 kB threshold
  mkdir "$d/a2"; python3 "$d/dummy.py" 600 20 30 0 & pid=$!
  "$self" --arm-dir "$d/a2" --pid "$pid" --interval 0.2 --abort-kb 150000 2>"$d/a2/err.txt"; rc=$?; sleep 0.5
  line=$(cat "$d/a2/peak_rss.txt"); peak=$(echo "$line" | sed -n 's/^peak_rss_kb=\([0-9]*\) .*/\1/p')
  check "3 abort: exit 3, marker ABORTED_RSS written" "$([ $rc -eq 3 ] && [ -s "$d/a2/ABORTED_RSS" ] && echo 1 || echo 0)" "rc $rc, marker first line $(head -1 "$d/a2/ABORTED_RSS" 2>/dev/null)"
  check "3 abort: the job is gone" "$(kill -0 "$pid" 2>/dev/null && echo 0 || echo 1)"
  check "3 abort: peak above the threshold, abort_threshold_kb=150000 in the file" "$([ -n "$peak" ] && [ "$peak" -gt 150000 ] && echo "$line" | grep -q 'abort_threshold_kb=150000$' && echo 1 || echo 0)" "$line"
  wait "$pid" 2>/dev/null
  # 4 the allocation sits in a CHILD of the job root: children are counted and killed
  mkdir "$d/a3"; python3 "$d/dummy.py" 600 20 30 1 & pid=$!
  "$self" --arm-dir "$d/a3" --pid "$pid" --interval 0.2 --abort-kb 150000 2>/dev/null; rc=$?; sleep 0.5
  left=$(pgrep -f "$d/dummy.py" | wc -l)
  check "4 child holds the memory: abort fired, no dummy process left" "$([ $rc -eq 3 ] && [ -s "$d/a3/ABORTED_RSS" ] && [ "$left" -eq 0 ] && echo 1 || echo 0)" "rc $rc, left $left"
  wait "$pid" 2>/dev/null
  # 5 pattern mode, watcher started before the job
  mkdir "$d/a4"; "$self" --arm-dir "$d/a4" --pattern "r83_selftest_marker_$$" --interval 0.2 --abort-kb 400000 --wait-start 20 & wp=$!
  sleep 1; ( exec -a "x" python3 "$d/dummy.py" 40 20 1 0 "r83_selftest_marker_$$" ) & jp=$!
  wait "$wp"; rc=$?; wait "$jp" 2>/dev/null
  check "5 pattern mode: found the job, normal end, peak recorded" "$([ $rc -eq 0 ] && grep -Eq '^peak_rss_kb=[1-9][0-9]* ' "$d/a4/peak_rss.txt" && echo 1 || echo 0)" "rc $rc $(cat "$d/a4/peak_rss.txt" 2>/dev/null)"
  # 6 pattern that never appears
  mkdir "$d/a5"; "$self" --arm-dir "$d/a5" --pattern "no_such_process_$$" --interval 0.2 --wait-start 1 2>/dev/null; rc=$?
  check "6 no matching process: exit 4, no marker, no peak file" "$([ $rc -eq 4 ] && [ ! -e "$d/a5/ABORTED_RSS" ] && [ ! -e "$d/a5/peak_rss.txt" ] && echo 1 || echo 0)" "rc $rc"
  # 7 --kill-cmd is used instead of signals
  mkdir "$d/a6"; python3 "$d/dummy.py" 600 20 30 0 & pid=$!
  "$self" --arm-dir "$d/a6" --pid "$pid" --interval 0.2 --abort-kb 150000 --kill-cmd "kill -KILL $pid" 2>/dev/null; rc=$?; sleep 0.3; wait "$pid" 2>/dev/null
  check "7 --kill-cmd: abort, marker, job gone" "$([ $rc -eq 3 ] && [ -s "$d/a6/ABORTED_RSS" ] && ! kill -0 "$pid" 2>/dev/null && echo 1 || echo 0)" "rc $rc"
  rm -rf "$d"
  echo "$n checks, failures: $fails"
  [ "$fails" -eq 0 ]
}

while [ $# -gt 0 ]; do
  case "$1" in
    --arm-dir) arm_dir="$2"; shift 2 ;;
    --pid) pid="$2"; shift 2 ;;
    --pattern) pattern="$2"; shift 2 ;;
    --interval) interval="$2"; shift 2 ;;
    --abort-kb) abort_kb="$2"; shift 2 ;;
    --wait-start) wait_start="$2"; shift 2 ;;
    --kill-cmd) kill_cmd="$2"; shift 2 ;;
    --print-threshold) h=$(host_ram_kb); echo "host_ram_kb=$h abort_threshold_kb=$((h / 2))"; exit 0 ;;
    --selftest) selftest; exit $? ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "r83_rss_watch: unknown argument $1" >&2; exit 2 ;;
  esac
done
watch
exit $?
