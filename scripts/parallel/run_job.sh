#!/usr/bin/env bash
set -uo pipefail
# Standardized single-job runner, executed inside a container by
# dispatch_queue.sh. Knows nothing about datasets/modes/sweeps -- every
# fact it needs (bag, launch file, cache file, GT info, output dir) comes
# from job.env, written once by gen_jobs.py. This is the one place the
# per-launch safety steps (rosnode cleanup, rosparam delete /) live for
# every job that goes through the queue, so a new sweep can't skip them
# by hand-rolling its own launch script.
#
# usage: run_job.sh PORT JOB_DIR   (JOB_DIR is the CONTAINER-side path)
PORT="$1"; JOB_DIR="$2"

export ROS_MASTER_URI="http://localhost:${PORT}"
export ROS_HOSTNAME=localhost
cd /root/catkin_ws
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
source scripts/common/test_common.sh
# CQ-18 item (4): `set -a` so every job.env var (BAG/LAUNCH_FILE/OUT_DIR/...)
# is exported, not just script-local -- the stall watchdog wraps run_one()
# in a `setsid bash -c` subshell below, which needs these visible without
# this script threading each one through by hand.
set -a
source "$JOB_DIR/job.env"
set +a

# TQ-4/rule 34c: record the livo_recon repo commit this job actually runs
# against (the built binary, not gen_jobs.py's manifest-generation time --
# those can differ), so a per-job scorer can refuse to read a file whose
# column layout it can't trust (see grid-design.md section 12's
# frame_stats.txt column-order caveat). Appended, not overwritten -- keeps
# job.env's other fields intact for anything else that sources it.
_build_commit="$(git -C /root/catkin_ws/src/livo_recon rev-parse HEAD 2>/dev/null || echo unknown)"
echo "BUILD_COMMIT=$_build_commit" >> "$JOB_DIR/job.env"

yes | rosnode cleanup > /dev/null 2>&1 || true
rosparam delete / > /dev/null 2>&1 || true

mkdir -p "$OUT_DIR"
# R62: the three diagnostics below are written by PersistentLogStream into the /tmp default dir. Remove any copy left by a
# previous job on this container so the `-s` copy guards further down can only pick up THIS run's file.
rm -f /tmp/gravity_alignment.txt /tmp/imu_first_scans.csv /tmp/calibration_p0.txt \
      /tmp/state_trace.csv /tmp/state_reference.txt /tmp/imu_cov_growth.csv \
      /tmp/lio_gt_matched.csv /tmp/motion_q_scan.csv \
      /tmp/cq74_eigenspectrum.txt
chmod 0777 "$OUT_DIR"  # written as root; keep host-side (non-root) cleanup/inspection working
# PACKAGE comes from job.env (written by gen_jobs.py's --package flag);
# defaults to fast_livo for every batch generated before this field
# existed. See dispatch_queue.sh's is_container_free() -- it checks BOTH
# fastlivo_mapping and livo_recon_node precisely because two different
# queues (one per package) can be draining onto the same container pool
# concurrently.
PACKAGE="${PACKAGE:-fast_livo}"

if [[ "$PACKAGE" == "fast_livo" ]]; then
  # RUN_ONE_COPY_ROOT/SEQ_NAME only mean anything to fast_livo's two copy
  # paths (direct_write_dir and the legacy Log/result/ fallback, both
  # gated on package=="fast_livo" inside run_one() itself except the
  # legacy fallback, which isn't -- keeping these fast_livo-scoped here
  # too avoids a spurious "no odometry found at .../Log/result/*.txt"
  # warning on every livo_recon job, since livo_recon has no such file at
  # all (its evo scoring is live, not a post-hoc file read -- see below).
  export RUN_ONE_COPY_ROOT="$(dirname "$OUT_DIR")"
  export RUN_ONE_COPY_SEQ_NAME="$(basename "$OUT_DIR")"
  export RUN_ONE_COPY_CONFIG="$JOB_DIR/config.yaml"
  # 2026-08-20: CACHE_FILE being set means this job was GENERATED to run
  # cached (gen_jobs.py wrote a cache path into job.env) -- if that file
  # doesn't exist/is empty, that's a real bug (wrong filename convention,
  # a cache that was never generated, etc), not a signal to quietly run
  # live instead. Confirmed this session: a wrong cache_suffix in a
  # manifest generator pointed CACHE_FILE at a nonexistent file, this
  # check used to just skip the override, and the whole batch silently
  # ran full live GPU tracking (loading the AOT model every job) instead
  # of cheap cached replay -- no error anywhere, just much slower runs and
  # the exact AOT-model-loading contention this dispatcher exists to
  # avoid. Fail loudly instead so a missing/misnamed cache is caught
  # immediately rather than discovered by noticing runs are slow.
  if [[ -n "${CACHE_FILE:-}" ]]; then
    if [[ ! -s "$CACHE_FILE" ]]; then
      echo "[$(basename "$JOB_DIR")] FATAL: CACHE_FILE='$CACHE_FILE' set but missing/empty -- refusing to silently fall back to live tracking" | tee -a "$JOB_DIR/run.log"
      touch "$JOB_DIR/FAILED"
      exit 1
    fi
    export OFFLINE_EXTRA_OVERRIDES=$'livo_vio:\n  track_cache_path: "'"$CACHE_FILE"'"'
  fi
  run_one fast_livo "$LAUNCH_FILE" "$BAG" "$JOB_DIR/run.log"
  unset RUN_ONE_COPY_ROOT RUN_ONE_COPY_SEQ_NAME RUN_ONE_COPY_CONFIG OFFLINE_EXTRA_OVERRIDES

  if [[ -s "$OUT_DIR/odometry_lio.txt" ]]; then
    if [[ "$GT_MODE" == "ntu" ]]; then
      read -r -a LEVER_ARR <<< "$GT_LEVER_ARM"
      python3 "$REPO_ROOT/evo/fastlivo_evo.py" "$BAG" "$GT_TOPIC" "$OUT_DIR" \
        --stage lio --mode interp --gt-lever-arm "${LEVER_ARR[@]}" \
        --out "$OUT_DIR/results_lio.txt" > /dev/null 2>&1
    elif [[ "$GT_MODE" == "robot_gcp" ]]; then
      # site2_robot_* has no body-frame GT -- see gen_jobs.py's gt_fields()
      # comment. Score via GCP-checkpoint projection instead of a direct
      # trajectory comparison (mirrors run_evo.sh's own is_robot_seq() dispatch).
      python3 "$REPO_ROOT/evo/hilti_robot_evo.py" "$OUT_DIR/odometry_lio.txt" "$SEQ" \
        --stage lio --out "$OUT_DIR/results_lio.txt" > /dev/null 2>&1
    else
      python3 "$REPO_ROOT/evo/fastlivo_evo.py" dummy dummy "$OUT_DIR" \
        --stage lio --mode interp --gt-file "$GT_FILE" \
        --out "$OUT_DIR/results_lio.txt" > /dev/null 2>&1
    fi
  fi

  # VIO stage: only present when this job actually ran VIO (common/
  # img_en=1 + common/use_livo_vio=true) -- a LIO-only job never writes
  # odometry_vio.txt at all, so this block is a silent no-op for those,
  # matching run_evo.sh's own "skipped whenever the run has no
  # odometry_vio.txt" convention rather than treating a missing VIO stage
  # as a failure.
  if [[ -s "$OUT_DIR/odometry_vio.txt" ]]; then
    if [[ "$GT_MODE" == "ntu" ]]; then
      read -r -a LEVER_ARR <<< "$GT_LEVER_ARM"
      python3 "$REPO_ROOT/evo/fastlivo_evo.py" "$BAG" "$GT_TOPIC" "$OUT_DIR" \
        --stage vio --mode interp --gt-lever-arm "${LEVER_ARR[@]}" \
        --out "$OUT_DIR/results_vio.txt" > /dev/null 2>&1
    elif [[ "$GT_MODE" == "robot_gcp" ]]; then
      python3 "$REPO_ROOT/evo/hilti_robot_evo.py" "$OUT_DIR/odometry_vio.txt" "$SEQ" \
        --stage vio --out "$OUT_DIR/results_vio.txt" > /dev/null 2>&1
    else
      python3 "$REPO_ROOT/evo/fastlivo_evo.py" dummy dummy "$OUT_DIR" \
        --stage vio --mode interp --gt-file "$GT_FILE" \
        --out "$OUT_DIR/results_vio.txt" > /dev/null 2>&1
    fi
    [[ -s "$OUT_DIR/results_vio.txt" ]] && cp "$OUT_DIR/results_vio.txt" "$JOB_DIR/results_vio.txt"
  fi

  # DONE/FAILED status is still keyed on the LIO stage alone (unchanged
  # from before this VIO-scoring addition) -- every job runs LIO, not
  # every job runs VIO, so gating completion on VIO would wrongly mark
  # LIO-only jobs as failed.
  if [[ -s "$OUT_DIR/results_lio.txt" ]]; then
    cp "$OUT_DIR/results_lio.txt" "$JOB_DIR/results_lio.txt"
    touch "$JOB_DIR/DONE"
    echo "[$(basename "$JOB_DIR")] lio: $(cat "$JOB_DIR/results_lio.txt")"
    [[ -s "$JOB_DIR/results_vio.txt" ]] && echo "[$(basename "$JOB_DIR")] vio: $(cat "$JOB_DIR/results_vio.txt")"
  else
    touch "$JOB_DIR/FAILED"
    echo "[$(basename "$JOB_DIR")] FAILED (no results_lio.txt -- see $JOB_DIR/run.log)"
  fi

else  # livo_recon
  # livo_recon has no fast_livo-style odometry_lio.txt/fastlivo_evo.py
  # post-hoc scoring path -- evo_processing.cpp scores LIVE during the run
  # (subscribing directly to the GT topic/file its own config/<dataset>.yaml
  # evo: block already points at, no GT wiring needed from this script) and
  # prints "[evo/<mode>]  n=...  ATE=...m  ..." lines to console via the
  # Printer (PrintCategory::EVO, throttle_s=0 by default -- see
  # config/<dataset>.yaml's outputs: block -- so every update lands in run.log, not
  # just a sampled subset). This job.env's own overrides (dotted keys ->
  # nested YAML, see gen_jobs.py's dotted_to_nested()) become
  # OFFLINE_EXTRA_OVERRIDES, appended to the SAME overrides_file every
  # livo_recon launch file already loads last -- mirrors summarize_evo()'s
  # "last [evo line in the log wins" convention (test_common.sh) so this
  # job's results_lio.txt stays in the same "one line, contains ATE=" shape
  # every downstream analysis script (grep 'ATE=\K[0-9.]+') already expects
  # for fast_livo jobs.
  job_overrides="$(cat "$JOB_DIR/config.yaml" 2>/dev/null || true)"
  if [[ -n "$job_overrides" && "$job_overrides" != "{}" ]]; then
    export OFFLINE_EXTRA_OVERRIDES="$job_overrides"
  fi
  # BASE_CONFIG_FILE (2026-09-02): only set when gen_jobs.py's manifest had
  # a delete-override (a dotted key with value null) -- points at a
  # per-job materialized copy of the dataset's base config with those keys
  # actually removed, instead of the launch file's own hardcoded path.
  # rosparam can only SET a key, never unset one, so a key the base config
  # sets unconditionally (e.g. imu/ds/ds_leaf_size) stays present and trips
  # a dead-scope refusal no matter what the sparse overrides file says --
  # see gen_jobs.py's split_deletions()/apply_deletions(). Empty for every
  # job that doesn't need this; the launch file's own default then applies,
  # identical to before this existed.
  extra_launch_args=()
  [[ -n "${BASE_CONFIG_FILE:-}" ]] && extra_launch_args+=("base_config_file:=$BASE_CONFIG_FILE")

  # CQ-18 item (4): the stall detector. Opt-in via CELL_BUDGET_SECS (set by
  # the caller -- e.g. TQ-12's dispatch wrapper, once the multiplier is
  # measured on its own first 50 cells; unset/0 here means "no watchdog",
  # preserving every existing caller's behavior unchanged). Runs run_one()
  # in its own session (setsid) so the watchdog can SIGTERM/SIGKILL its
  # whole process group (roslaunch + livo_recon_node) without touching
  # run_job.sh itself -- see feedback memory "setsid before process-group
  # kill" for why setsid is required for this to be safe.
  if [[ "${CELL_BUDGET_SECS:-0}" -gt 0 ]]; then
    export OFFLINE_EXTRA_OVERRIDES
    setsid bash -c '
      source /opt/ros/noetic/setup.bash
      source /root/catkin_ws/devel/setup.bash
      source scripts/common/test_common.sh
      run_one livo_recon "$1" "$2" "$3" "${@:4}"
    ' _ "$LAUNCH_FILE" "$BAG" "$JOB_DIR/run.log" "${extra_launch_args[@]}" &
    RUN_PID=$!
    bash "$(dirname "${BASH_SOURCE[0]}")/stall_watchdog.sh" \
      "$OUT_DIR/frame_stats.txt" "$RUN_PID" "$CELL_BUDGET_SECS" \
      > "$JOB_DIR/stall_watchdog.log" 2>&1 &
    WD_PID=$!
    wait "$RUN_PID" 2>/dev/null
    kill "$WD_PID" 2>/dev/null; wait "$WD_PID" 2>/dev/null
    if grep -q '^STALLED' "$JOB_DIR/stall_watchdog.log" 2>/dev/null; then
      cp "$JOB_DIR/stall_watchdog.log" "$JOB_DIR/STALLED"
      echo "[$(basename "$JOB_DIR")] STALLED: $(cat "$JOB_DIR/STALLED")"
    fi
  else
    run_one livo_recon "$LAUNCH_FILE" "$BAG" "$JOB_DIR/run.log" "${extra_launch_args[@]}"
  fi
  unset OFFLINE_EXTRA_OVERRIDES

  # T3-0d: only non-empty when this job set voxel_map/plane/log_variance_shares_en
  # (run_one() truncates both paths before every launch -- see test_common.sh's
  # stale-log list -- so "-s" here means THIS run wrote it, not a stale leftover).
  [[ -s /tmp/variance_shares.txt ]] && cp /tmp/variance_shares.txt "$OUT_DIR/variance_shares.txt"
  [[ -s /tmp/plane_fit_stats.txt ]] && cp /tmp/plane_fit_stats.txt "$OUT_DIR/plane_fit_stats.txt"
  [[ -s /tmp/corr.csv ]] && cp /tmp/corr.csv "$OUT_DIR/corr.csv"
  [[ -s /tmp/scan.csv ]] && cp /tmp/scan.csv "$OUT_DIR/scan.csv"
  [[ -s /tmp/nll.txt ]] && cp /tmp/nll.txt "$OUT_DIR/nll.txt"
  [[ -s /tmp/frame_stats.txt ]] && cp /tmp/frame_stats.txt "$OUT_DIR/frame_stats.txt"
  # T7-a/G6 (2026-09-01): qhat.csv/vio_corr.csv were added after this
  # copy list was written and got silently dropped -- neither ever
  # reached OUT_DIR despite log_qhat_en/log_consistency_corr_en (VIO
  # side) both being real, working flags. See test_common.sh's stale-log
  # truncation list, which needed the same fix.
  [[ -s /tmp/qhat.csv ]] && cp /tmp/qhat.csv "$OUT_DIR/qhat.csv"
  [[ -s /tmp/vio_corr.csv ]] && cp /tmp/vio_corr.csv "$OUT_DIR/vio_corr.csv"
  # 14b (2026-09-01): corr_scan.csv -- exact per-scan aggregates, the source
  # of truth for corr.csv's level statistics once corr.csv itself is strided.
  [[ -s /tmp/corr_scan.csv ]] && cp /tmp/corr_scan.csv "$OUT_DIR/corr_scan.csv"
  # spline_q.csv (2026-09-01): same gap as qhat.csv above -- added with the
  # spline-adaptive-q branch after this copy list existed, found missing
  # from V-1's own OUT_DIR during that gate's first run.
  [[ -s /tmp/spline_q.csv ]] && cp /tmp/spline_q.csv "$OUT_DIR/spline_q.csv"
  # engagement.txt (2026-09-02): same gap again -- 5c93cc6 added it
  # (debugLogPath("engagement.txt"), default dir /tmp since no job here
  # sets outputs/debug_log_dir), found missing from SM-1's own OUT_DIR
  # during that card's first run.
  [[ -s /tmp/engagement.txt ]] && cp /tmp/engagement.txt "$OUT_DIR/engagement.txt"
  # state_trace.csv/state_reference.txt/imu_cov_growth.csv
  # (R61, 2026-09-29): same /tmp-default gap class as the files above -- the R61 state-trace
  # patch's new PersistentLogStream diagnostics were missing from a livo_recon job's own OUT_DIR
  # (no job in this manifest sets outputs/debug_log_dir, so they default to /tmp) during R61's
  # wave-1 dispatch smoke check. lio_gt_matched.csv/motion_q_scan.csv are pre-existing diagnostics
  # that had the same gap for the same reason (no prior livo_recon batch needed them copied out).
  [[ -s /tmp/state_trace.csv ]] && cp /tmp/state_trace.csv "$OUT_DIR/state_trace.csv"
  [[ -s /tmp/state_reference.txt ]] && cp /tmp/state_reference.txt "$OUT_DIR/state_reference.txt"
  [[ -s /tmp/imu_cov_growth.csv ]] && cp /tmp/imu_cov_growth.csv "$OUT_DIR/imu_cov_growth.csv"
  [[ -s /tmp/lio_gt_matched.csv ]] && cp /tmp/lio_gt_matched.csv "$OUT_DIR/lio_gt_matched.csv"
  [[ -s /tmp/motion_q_scan.csv ]] && cp /tmp/motion_q_scan.csv "$OUT_DIR/motion_q_scan.csv"
  # R62: gravity_alignment.txt, imu_first_scans.csv (new) and calibration_p0.txt (R59, never copied before): same /tmp-default gap class.
  [[ -s /tmp/gravity_alignment.txt ]] && cp /tmp/gravity_alignment.txt "$OUT_DIR/gravity_alignment.txt"
  [[ -s /tmp/imu_first_scans.csv ]] && cp /tmp/imu_first_scans.csv "$OUT_DIR/imu_first_scans.csv"
  [[ -s /tmp/calibration_p0.txt ]] && cp /tmp/calibration_p0.txt "$OUT_DIR/calibration_p0.txt"

  if [[ "${GT_MODE:-}" == "robot_gcp" ]]; then
    # gen_jobs.py disabled live evo and wired outputs.odom.export/outputs.path
    # to OUT_DIR for these jobs instead (evo_processing.cpp has no
    # GCP-projection logic -- see its gt_fields() comment), so score the
    # exported odometry.txt post-hoc, same tool/convention as the fast_livo
    # branch above. livo_recon writes one merged LIO+VIO file, not separate
    # lio/vio stages, so this is reported as the "lio" stage for consistency
    # with find_stage_file()/results_lio.txt readers, not because it's
    # actually LIO-only.
    if [[ -s "$OUT_DIR/odometry.txt" ]]; then
      python3 "$REPO_ROOT/evo/hilti_robot_evo.py" "$OUT_DIR/odometry.txt" "$SEQ" \
        --stage lio --out "$JOB_DIR/results_lio.txt" > /dev/null 2>&1
    fi
    cp "$JOB_DIR/config.yaml" "$OUT_DIR/config.yaml" 2>/dev/null || true
    cp "$JOB_DIR/run.log" "$OUT_DIR/run.log" 2>/dev/null || true
    if [[ -s "$JOB_DIR/results_lio.txt" ]]; then
      cp "$JOB_DIR/results_lio.txt" "$OUT_DIR/results_lio.txt" 2>/dev/null || true
      touch "$JOB_DIR/DONE"
      echo "[$(basename "$JOB_DIR")] $(cat "$JOB_DIR/results_lio.txt")"
    else
      touch "$JOB_DIR/FAILED"
      echo "[$(basename "$JOB_DIR")] FAILED (no odometry.txt exported, or hilti_robot_evo.py scoring failed -- see $JOB_DIR/run.log)"
    fi
  else
    # Register-flagged bug (2026-09-01): "the results_lio.txt write race has
    # no guard and corrupts silently -- blocks 54 runs read at the last
    # digit." run_one() only waits on roslaunch's own PID exiting -- the
    # child node's stdout, re-emitted THROUGH roslaunch's own process
    # (not a direct inherited fd), can still have a few buffered bytes not
    # yet flushed to run.log at that instant, so a grep taken immediately
    # after can catch the last [evo line mid-write (typically missing its
    # trailing digit(s)/"m"). Retry briefly rather than trust the first read:
    # a truncated line won't match ATE_RE's full "...ATE=<num>m" shape.
    ATE_LINE_RE='ATE=[0-9.eE+-]+m'
    last_evo=""
    for _attempt in 1 2 3 4 5; do
      # TQ-12 bug (2026-09-15): grep '\[evo' | tail -1 takes the LAST
      # line containing the substring "[evo", even when that specific
      # line got corrupted by an unrelated WARN log line interleaving
      # mid-write on the same fd (confirmed: 18 exp05_construction_
      # upper_level_2 cells' true final [evo line was garbled this way
      # right at end-of-run, even though an earlier, well-formed [evo
      # line -- and the run's actual output -- were both fine). Retrying
      # doesn't help a genuinely corrupted final line, since the file
      # doesn't change once the process has exited. Grep for a line that
      # actually matches the full ATE pattern instead of grepping for
      # the substring and checking the match after -- this reports the
      # last WELL-FORMED [evo line, tolerating a garbled one after it.
      last_evo="$(grep -E '\[evo.*'"$ATE_LINE_RE" "$JOB_DIR/run.log" 2>/dev/null | tail -1)"
      [[ "$last_evo" =~ $ATE_LINE_RE ]] && break
      sleep 0.3
    done
    # CQ-12: this used to test `[[ -n "$last_evo" ]]` -- true for ANY line
    # containing the substring "[evo", including an error/startup line that
    # never matches ATE_LINE_RE (e.g. "[evo] gt_source=file but gt_file
    # could not be opened:", or a param-registration line printed before a
    # bag that turns out not to exist is even opened). That accepted a
    # cell that produced no usable result as DONE, with only a WARNING no
    # one was watching for -- confirmed live on TQ-4: 333 exp05 cells that
    # crashed on rosbag::BagIOException within the first second, and every
    # gt_mode="file" cell before the evo gt_source/gt_file threading fix,
    # both reached DONE this way. Now requires the line to actually match
    # ATE_LINE_RE, AND the primary output (odometry.txt) to be non-empty --
    # rule 43o: a gate that accepts anything is not a gate.
    # R69B E1 (F-326): the node's real exit status, independent of the
    # DONE/FAILED decision above (which stays log-grep-based, unchanged).
    # roslaunch itself reports this in run.log -- "process has finished
    # cleanly" (clean exit) or "process has died [pid ..., exit code N, ...]"
    # (N is roslaunch's own convention: negative N = killed by signal -N).
    # Written before DONE/FAILED so a reader never sees a sentinel without it.
    {
      died_line="$(grep -m1 'process has died \[pid' "$JOB_DIR/run.log" 2>/dev/null)"
      if [[ "$died_line" =~ exit\ code\ (-?[0-9]+) ]]; then
        code="${BASH_REMATCH[1]}"
        if [[ "$code" == -* ]]; then
          sig="${code#-}"; sig_name="$(kill -l "$sig" 2>/dev/null || echo "UNKNOWN")"
          echo "exit_code=$code signal=SIG$sig_name"
        else
          echo "exit_code=$code signal=none"
        fi
      elif grep -q 'process has finished cleanly' "$JOB_DIR/run.log" 2>/dev/null; then
        echo "exit_code=0 signal=none"
      else
        echo "exit_code=unknown signal=unknown"
      fi
    } > "$OUT_DIR/exit_status.txt" 2>/dev/null
    # R72: a job that exits cleanly (exit_code=0, no signal) but never prints
    # an [evo ATE=...] line is not necessarily a failure -- a short/truncated
    # run (e.g. gen_jobs.py's EF tier: duration_secs covers calibration plus
    # a handful of dump frames, not the full bag) can legitimately end, via
    # the node's own auto_terminate_idle_secs idle watchdog, well before evo
    # reaches its scoring window (onset_scan ~485+). Before this fix,
    # run_job.sh's generic ATE-line requirement misclassified every one of
    # R72's 40 EF arms as FAILED even though all 40 completed exactly as
    # designed with full, correct raw output -- confirmed by exit_status.txt
    # reading exit_code=0 signal=none for all 40, with residual_dump.csv/
    # plane_fit_trace.csv/state_trace.csv all present and complete.
    #
    # R73 fix (direct user request): the ORIGINAL version of this R72 fix
    # gated DONE on `(last_evo matches ATE_LINE_RE OR clean exit)`, an OR --
    # which meant a job that printed at least one matching line and THEN
    # crashed still read as DONE, since the ATE-line disjunct alone satisfies
    # the condition regardless of the actual exit status. This went
    # uncaught in R72 because evo there only ever logged once, at the very
    # end of a run that had genuinely finished, so "last_evo matches" was in
    # practice a reliable proxy for "reached the end normally" -- but R73's
    # long coupled full-bag arms log evo PERIODICALLY throughout the run
    # (hundreds of lines), so a crash occurring any time after the first
    # periodic evaluation was masked as DONE. Confirmed live: all 3 R73
    # `r73_co_rerun_meas_gateon_knots{4_fq,8_fq,8_mq}_full` arms crashed
    # (SIGABRT, "joint-knot information matrix is not positive definite")
    # but were sentinelled DONE because a mid-run [evo/interp] line (e.g.
    # scan 1336 of knots4_fq_full, ATE=3826.75m) satisfied the old OR before
    # the crash happened afterward.
    #
    # The DONE/FAILED decision must depend ONLY on how the process actually
    # exited (clean vs crashed/killed), never on whether evo happened to
    # produce a line -- ATE-line presence is relevant only to what gets
    # WRITTEN into results_lio.txt (a real ATE line vs the NO_ATE
    # placeholder below), not to whether the job is DONE or FAILED.
    _exit_status_text="$(cat "$OUT_DIR/exit_status.txt" 2>/dev/null)"
    _clean_exit=0
    [[ "$_exit_status_text" == "exit_code=0 signal=none" ]] && _clean_exit=1
    if [[ "$_clean_exit" == "1" && -s "$OUT_DIR/odometry.txt" ]]; then
      # CQ-12 (3): never two sentinels at once -- A_III_site2_robot_1_c0
      # carried a stale FAILED from an earlier scoring attempt alongside a
      # later DONE, because nothing ever cleared the first sentinel before
      # writing the second. Clear the other one before committing to this
      # verdict, in both directions, so exactly one always exists.
      rm -f "$JOB_DIR/FAILED"
      # CQ-12 (1), "the process has actually finished" half: run_one()'s
      # own wait is on roslaunch's PID exiting, but 15 TQ-4 cells were
      # found scored 130-814s before their own run.log stopped growing --
      # so that wait alone does not prove the underlying node is done
      # writing. Require run.log's size to be quiescent (unchanged) across
      # one more short sleep before trusting this as final; a run.log
      # still being appended to fails this half and is reported as such.
      _sz1=$(stat -c%s "$JOB_DIR/run.log" 2>/dev/null || echo 0)
      sleep 1.0
      _sz2=$(stat -c%s "$JOB_DIR/run.log" 2>/dev/null || echo 0)
      if [[ "$_sz1" != "$_sz2" ]]; then
        touch "$JOB_DIR/FAILED"
        echo "[$(basename "$JOB_DIR")] FAILED (run.log still growing after the last [evo line -- process has not actually finished, see $JOB_DIR/run.log)"
      else
        # CQ-5: write-then-rename (matches evo/fastlivo_evo.py's/hilti_robot_evo.py's
        # own T0-G write-then-rename convention) instead of a direct `>` redirect --
        # a reader (dispatch_queue.sh polling for DONE, or a hand grep) racing the
        # truncate-then-write window of a plain `>` could observe a zero-length or
        # partially-written results_lio.txt. rename(2) is atomic, so a concurrent
        # reader only ever sees the old (absent) or new (complete) file, never a
        # partial one.
        _tmp_results="$JOB_DIR/results_lio.txt.tmp$$"
        if [[ "$last_evo" =~ $ATE_LINE_RE ]]; then
          echo "$last_evo" > "$_tmp_results"
        else
          # R72: clean exit, no ATE -- see comment above the outer `if`.
          echo "NO_ATE: clean exit ($_exit_status_text), odometry.txt present, no [evo ATE=...m line in run.log -- evo likely never reached its scoring window (a short/truncated-duration run, or evo disabled/misconfigured for this job); see run.log" > "$_tmp_results"
        fi
        mv -f "$_tmp_results" "$JOB_DIR/results_lio.txt"
        touch "$JOB_DIR/DONE"
        if [[ "$last_evo" =~ $ATE_LINE_RE ]]; then
          echo "[$(basename "$JOB_DIR")] $last_evo"
        else
          echo "[$(basename "$JOB_DIR")] DONE ($_exit_status_text, no ATE line -- see results_lio.txt)"
        fi
      fi
    else
      rm -f "$JOB_DIR/DONE"
      touch "$JOB_DIR/FAILED"
      if [[ ! -s "$OUT_DIR/odometry.txt" ]]; then
        echo "[$(basename "$JOB_DIR")] FAILED ($_exit_status_text -- \$OUT_DIR/odometry.txt missing or empty -- see $JOB_DIR/run.log)"
      else
        echo "[$(basename "$JOB_DIR")] FAILED ($_exit_status_text -- the process did not exit cleanly (crashed, was killed, or an unrecognized exit) -- last [evo line, if any: '${last_evo:-<none>}' -- see $JOB_DIR/run.log)"
      fi
    fi
    # 2026-08-28: livo_recon_results/<batch>/<job_id> (OUT_DIR) is the
    # dedicated results tree (see CLAUDE.md/STRUCTURE.md) -- this branch
    # previously only ever wrote to JOB_DIR (the _job_queue/ dispatch
    # bookkeeping dir, not a results location), leaving every non-robot
    # livo_recon job's actual results directory empty. Mirror the
    # robot_gcp branch above: copy config.yaml/run.log/results_lio.txt
    # into OUT_DIR too, regardless of DONE/FAILED, so a failed run's log
    # is still inspectable from the results tree.
    cp "$JOB_DIR/config.yaml" "$OUT_DIR/config.yaml" 2>/dev/null || true
    cp "$JOB_DIR/run.log" "$OUT_DIR/run.log" 2>/dev/null || true
    [[ -s "$JOB_DIR/results_lio.txt" ]] && cp "$JOB_DIR/results_lio.txt" "$OUT_DIR/results_lio.txt" 2>/dev/null || true
  fi
fi
