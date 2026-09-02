#!/usr/bin/env python3
"""
Per-flag engagement check from spline_q.csv, built for the livo_recon
Experiment Register's SM-1 protocol card ("One-shot engagement check,
before any grid ever again" -- see the bug ledger's "half of every
switch-on cell was inert" row, which is exactly the failure mode this
exists to catch before a real grid is dispatched on top of it).

One job = one arm = one (spline.per_iteration.mode, fit.rotation.mode,
fit.imu_term.mode, adaptive_q.enable, voxel_map.plane.weight_floor.mode)
cell, run on a single sequence. This module reads that job's own
spline_q.csv (already retained in full by the sweep listener -- see
sweep_scoring.py's own docstring on why spline_q.csv never needs a
score-then-delete pass) and answers, per job:

  - what per_iteration mode the run ACTUALLY used (the CSV's own
    `per_iteration` column is the effective value, not the requested
    one -- see engagement_effective_vs_requested() below, which exists
    because of a real bug this found: a plain YAML `mode: off` silently
    parses as the boolean False under YAML 1.1's "Norway problem" and
    the C++ side falls back to its string default instead of refusing,
    so the requested arm never actually ran).
  - whether the refine/reintegrate/redeskew counters and magnitudes are
    non-trivial when their mechanism is nominally live.
  - whether the adaptive-Q applied_acc/applied_gyr range moved off its
    nominal value.
  - the job's own odometry.txt md5, so a caller can compare an "off"
    arm's trajectory against a baseline job's (SM-1's own "the off arm
    is byte-identical to the baseline" assertion).

Deliberately NOT a statistics module like sweep_scoring.py -- SM-1 is an
existence proof over ~10-15 short runs, not a population to summarize,
so this reports per-job facts rather than fitting anything.
"""
import csv
import hashlib
import os


# YAML 1.1's bool/null keywords -- see the module docstring. Any config
# value equal to one of these needs to have survived quoted through
# gen_jobs.py (fixed there 2026-09-02); this set is duplicated rather than
# imported so this module has no dependency on the fast_ws-side script.
_YAML11_AMBIGUOUS = {"y", "yes", "n", "no", "true", "false", "on", "off",
                      "null", "~"}


def _read_rows(spline_q_path):
    with open(spline_q_path, newline="") as f:
        return list(csv.DictReader(f))


def engagement_effective_vs_requested(spline_q_path, requested_per_iteration=None):
    """The single most load-bearing check in this module: does the CSV's
    own `per_iteration` column (what the run actually did) match what the
    job was configured to request? A mismatch here means the config value
    never reached the estimator -- silently, since ConfigResolver's mode()
    only refuses a value outside its allowed SET, not a value that arrived
    as the wrong TYPE (a YAML-coerced bool where a string was requested)."""
    rows = _read_rows(spline_q_path)
    if not rows:
        return {"error": "spline_q.csv is empty"}
    effective = rows[0]["per_iteration"]
    out = {"effective_per_iteration": effective}
    if requested_per_iteration is not None:
        out["requested_per_iteration"] = requested_per_iteration
        out["matches_requested"] = (effective == requested_per_iteration)
        if requested_per_iteration in _YAML11_AMBIGUOUS and not out["matches_requested"]:
            out["likely_cause"] = (
                "requested value is a YAML 1.1 boolean/null keyword -- check "
                "whether it reached config.yaml unquoted (gen_jobs.py's "
                "_quote_if_yaml11_ambiguous(), fixed 2026-09-02, should "
                "prevent this for jobs generated after that commit)")
    return out


def engagement_magnitudes(spline_q_path):
    """Per-run engagement magnitudes, all as (count/total, max) so a zero
    max with a non-zero call count is visible as distinct from a mechanism
    that never ran at all -- see the module docstring's note on
    `per_iteration: redeskew` (no refine/reintegrate): its redeskew_calls
    is legitimately non-zero every frame while redeskew_dp_rms is
    legitimately ~0, because nothing updates the control points between
    iterations under that arm, so re-deskewing from an unchanged spline
    fit has zero ITERATION-TO-ITERATION delta by construction. That is not
    the same claim as "redeskew did nothing" -- it did its one real
    re-projection against the spline fit; this column just doesn't measure
    that against the pre-spline baseline (odometry.txt's own md5 does,
    see baseline_comparison() below)."""
    rows = _read_rows(spline_q_path)
    if not rows:
        return {"error": "spline_q.csv is empty"}
    n = len(rows)

    def col_f(name):
        return [float(r[name]) for r in rows if r.get(name) not in (None, "")]

    def col_i(name):
        return [int(float(r[name])) for r in rows if r.get(name) not in (None, "")]

    out = {"n_frames": n}
    for name in ("refine_dcp_max", "refine_dcp_rms", "reint_dp_max",
                 "reint_drot_deg_max", "redeskew_dp_rms"):
        vals = col_f(name)
        if vals:
            out[f"{name}_max"] = max(vals)
    for name in ("refits", "refine_applied", "refine_rejects", "redeskew_calls"):
        vals = col_i(name)
        if vals:
            out[f"{name}_total"] = sum(vals)
            out[f"{name}_frames_nonzero"] = sum(1 for v in vals if v > 0)
    for name in ("applied_acc", "applied_gyr"):
        vals = col_f(name)
        if vals:
            out[f"{name}_range"] = [min(vals), max(vals)]
    ok = col_i("spline_ok")
    if ok:
        out["fit_success_fraction"] = sum(ok) / len(ok)
    return out


def odometry_md5(odometry_path):
    with open(odometry_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def baseline_comparison(job_odometry_path, baseline_odometry_path):
    """SM-1's "the off arm is byte-identical to the baseline" assertion,
    applied literally -- an arm whose flag is nominally OFF (per_iteration:
    off, fit/imu_term/mode: off, adaptive_q/enable: false) should produce a
    trajectory that is EXACTLY the baseline's, not merely close to it."""
    a, b = odometry_md5(job_odometry_path), odometry_md5(baseline_odometry_path)
    return {"job_md5": a, "baseline_md5": b, "identical_to_baseline": a == b}


def check_job(job_dir, baseline_dir=None, requested_per_iteration=None):
    """Run every check above against one job's own output directory
    (spline_q.csv + odometry.txt, both retained by convention)."""
    spline_q_path = os.path.join(job_dir, "spline_q.csv")
    odometry_path = os.path.join(job_dir, "odometry.txt")
    out = {"job_dir": job_dir}
    if os.path.exists(spline_q_path):
        out["engagement"] = engagement_effective_vs_requested(
            spline_q_path, requested_per_iteration)
        out["magnitudes"] = engagement_magnitudes(spline_q_path)
    else:
        out["error"] = f"no spline_q.csv at {spline_q_path}"
    if os.path.exists(odometry_path):
        out["odometry_md5"] = odometry_md5(odometry_path)
        if baseline_dir:
            out["baseline_comparison"] = baseline_comparison(
                odometry_path, os.path.join(baseline_dir, "odometry.txt"))
    return out


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--baseline-dir", default=None)
    ap.add_argument("--requested-per-iteration", default=None)
    args = ap.parse_args()
    result = check_job(args.job_dir, args.baseline_dir, args.requested_per_iteration)
    print(json.dumps(result, indent=2))
