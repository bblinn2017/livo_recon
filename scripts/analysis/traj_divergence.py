#!/usr/bin/env python3
"""Where, and why, does a trajectory leave ground truth?

ATE is one scalar after ONE global Umeyama fit over the whole run, and that is
exactly why it cannot answer "where". A global alignment is dragged by whatever
the trajectory does at the end: a run that is perfect for 90 s and then departs
reports elevated error from t=0, because the fit compromises between the good
part and the bad part. The onset is smeared away by the metric.

This computes three error curves instead, from the same data:

  global   one Umeyama over all matched samples -- what ATE reports, kept so
           the numbers here can be reconciled with results_lio.txt

  prefix   Umeyama over [t0, t], residual read at t. "Given everything up to
           now, how far off am I now?" Grows monotonically-ish and its first
           sustained rise is the DEPARTURE time.

  window   Umeyama over [t-W/2, t+W/2], residual read at t. Local shape
           agreement with the local alignment freedom removed. Stays flat
           through pure accumulated drift and spikes only where the trajectory
           is locally the wrong SHAPE -- which is what a filter failure looks
           like, as opposed to slow bias.

The pairwise mode does the same for two estimates against each other (both
first aligned to GT), which answers "where does the combined method part company
with baseline pca" without either being assumed correct.

Attitude: NTU-VIRAL's GT quaternion is identity everywhere (laser tracker,
position only) and HILTI's sparse control points carry a dummy identity
quaternion. Rotation error against those is the angular magnitude of the
aligned estimate, not an error. This script DETECTS that and refuses, rather
than emitting a number that has been quoted in this register before.

Reuses bench_eval/evaluate.py's loaders, frame corrections and umeyama_no_scale
so the alignment convention is identical to the scorer's.
"""
import argparse, json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _c in [os.environ.get("BENCH_EVAL_DIR"),
           os.path.join(_HERE, "..", "..", "bench_eval"),
           os.path.join(_HERE, "..", "..", "..", "bench_eval"),
           os.path.join(_HERE, "..", "..", "..", "..", "bench_eval")]:
    if _c and os.path.exists(os.path.join(_c, "evaluate.py")):
        sys.path.insert(0, _c)
        break
try:
    from evaluate import (load_tum, LOADERS, apply_frame_correction, resample,
                          umeyama_no_scale)
except ImportError:
    sys.exit("traj_divergence.py: cannot find bench_eval/evaluate.py. Set "
             "BENCH_EVAL_DIR to the directory containing it.")

from scipy.spatial.transform import Rotation as R


# --------------------------------------------------------------- error curves

def _resid_at(idx, src, dst, lo, hi):
    """Umeyama over [lo,hi), residual of sample idx only."""
    if hi - lo < 6:                      # Umeyama consumes 6 DOF
        return np.nan
    Rm, tv = umeyama_no_scale(src[lo:hi], dst[lo:hi])
    return float(np.linalg.norm((Rm @ src[idx]) + tv - dst[idx]))


def prefix_error(t, p_est, p_gt, min_n=20):
    """Aligned on everything up to and including t. First sustained rise is the
    departure time -- unlike global ATE, nothing after t influences it."""
    n = len(t)
    err = np.full(n, np.nan)
    for i in range(min_n, n):
        err[i] = _resid_at(i, p_est, p_gt, 0, i + 1)
    return err


def rolling_prefix_ate(p_est, p_gt, recompute_every=500, tol=1e-9):
    """C-6: 'what ATE would have read if the run had stopped at k', for every
    k=3..N, computed incrementally (rolling sums + one 3x3 SVD per frame)
    rather than by N full re-alignments -- umeyama_no_scale is least squares
    in closed form, so its sufficient statistics are rolling sums. This is
    the P-1 THE FIGURE object: it REPLACES a cumulative RMS of e_glob (one
    whole-run Umeyama, then averaged), which is a different and misleading
    quantity -- see the register's C-6 card for why a rigid transform fit to
    the whole run displaces a stationary start that has no rotational lever
    arm of its own.

    Numerical guard, register-mandated: world-coordinate accumulators
    (Sigma is a difference of two large nearly-equal terms) lose precision
    as the trajectory leaves the origin, so before accumulating, x and y
    are each shifted by their OWN first sample (not a shared one -- the
    register's card says "subtract y_1 from every x_i and y_i", which
    silently assumes x and y already share an origin; est and GT live in
    independent frames until Umeyama aligns them, so a shared shift leaves
    whichever series has the larger unrelated offset unprotected -- verified
    to reproduce the exact same ate_k as an unshifted computation, since a
    common shift within each series translates but does not change the
    optimal rotation, and t is recovered as t' - R@x[0] + y[0], which cancels
    in every ATE_k^2 term below). An exact from-scratch umeyama_no_scale
    recompute every `recompute_every` frames must agree with the running
    incremental value to `tol` -- this is asserted, not just hoped for, and
    the actual observed max drift is returned so a caller can report it.

    Returns (ate_k, meta) -- ate_k[k] is nan for k<2 (need >=3 samples),
    meta['max_drift'] is the largest |incremental - exact| seen at any
    recompute checkpoint.
    """
    n = len(p_gt)
    ate_k = np.full(n, np.nan)
    if n < 3:
        return ate_k, dict(max_drift=float("nan"), n_checkpoints=0)

    x = np.asarray(p_est, dtype=np.float64) - p_est[0]
    y = np.asarray(p_gt, dtype=np.float64) - p_gt[0]

    Sx = np.cumsum(x, axis=0)
    Sy = np.cumsum(y, axis=0)
    Mxy = np.cumsum(np.einsum("ni,nj->nij", x, y), axis=0)   # sum_{i<=k} x_i y_i^T
    Sxx = np.cumsum(np.sum(x * x, axis=1))
    Syy = np.cumsum(np.sum(y * y, axis=1))

    max_drift, n_checkpoints = 0.0, 0
    for k in range(2, n):
        nk = k + 1
        mu_x, mu_y = Sx[k] / nk, Sy[k] / nk
        Sigma = Mxy[k].T / nk - np.outer(mu_y, mu_x)
        U, _, Vt = np.linalg.svd(Sigma)
        d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
        Rm = U @ np.diag([1.0, 1.0, d]) @ Vt
        tvec = mu_y - Rm @ mu_x

        val = (Sxx[k] + Syy[k] + nk * float(tvec @ tvec)
               + 2.0 * float(tvec @ Rm @ Sx[k]) - 2.0 * float(np.trace(Rm @ Mxy[k]))
               - 2.0 * float(tvec @ Sy[k])) / nk
        ate_k[k] = np.sqrt(max(val, 0.0))

        if nk % recompute_every == 0:
            Rm_e, tv_e = umeyama_no_scale(x[:nk], y[:nk])
            aligned = (Rm_e @ x[:nk].T).T + tv_e
            exact = float(np.sqrt(np.mean(np.sum((aligned - y[:nk]) ** 2, axis=1))))
            drift = abs(exact - ate_k[k])
            max_drift = max(max_drift, drift)
            n_checkpoints += 1
            assert drift < tol, (
                f"rolling_prefix_ate: incremental/exact drift {drift:.3e} at "
                f"k={nk} exceeds tol {tol:.0e} -- the rolling sums have "
                f"decorrelated from the exact recompute, do not trust ate_k "
                f"past this point")

    return ate_k, dict(max_drift=float(max_drift), n_checkpoints=n_checkpoints)


def window_error(t, p_est, p_gt, window):
    """Aligned locally in a +-window/2 span. Flat under pure drift, spikes where
    the local SHAPE is wrong."""
    n = len(t)
    err = np.full(n, np.nan)
    half = window / 2.0
    lo_i = np.searchsorted(t, t - half, side="left")
    hi_i = np.searchsorted(t, t + half, side="right")
    for i in range(n):
        err[i] = _resid_at(i, p_est, p_gt, lo_i[i], hi_i[i])
    return err


def detect_onset(t, err, baseline_span=20.0, k=5.0, sustain=1.0, floor=0.02):
    """First time err stays above max(k*baseline_median, floor) for `sustain` s.

    baseline_median is taken over the first `baseline_span` seconds of FINITE
    error, so a run that is broken from frame 0 reports onset at its own start
    rather than never.
    """
    fin = np.isfinite(err)
    if fin.sum() < 10:
        return None, None
    t0 = t[fin][0]
    base_mask = fin & (t <= t0 + baseline_span)
    if base_mask.sum() < 5:
        base_mask = fin
    base = float(np.median(err[base_mask]))
    thr = max(k * base, floor)
    over = fin & (err > thr)
    i = 0
    while i < len(t):
        if over[i]:
            j = i
            while j < len(t) and (over[j] or not fin[j]):
                j += 1
            if t[j - 1] - t[i] >= sustain:
                return float(t[i]), dict(baseline=base, threshold=thr,
                                         sustained_s=float(t[j - 1] - t[i]))
            i = j
        else:
            i += 1
    return None, dict(baseline=base, threshold=thr)


def gt_has_attitude(q, tol=1e-6):
    """True only if the GT quaternion is a valid, non-identity unit quaternion
    everywhere. NTU-VIRAL and HILTI control points both ship a dummy identity,
    and a rotation error against those is the estimate's own rotation
    magnitude, not an error -- refused the same as identity below.

    Some NTU-VIRAL sequences (confirmed: nya_01, nya_02) publish an all-zero
    quaternion on /leica/pose/relative instead of identity -- a recording
    artifact, not real attitude. load_ntu_csv's normalization guard (n==0 ->
    1.0, to avoid a division-by-zero NaN) leaves a zero quaternion as
    (0,0,0,0): still not unit-norm, so it must be refused here too, or
    R.from_quat() raises on a zero-norm quaternion downstream."""
    if np.any(np.abs(np.linalg.norm(q, axis=-1) - 1.0) > tol):
        return False
    return bool(np.max(np.abs(q - np.array([0.0, 0.0, 0.0, 1.0]))) > tol)


def rot_error_deg(q_est_aligned, q_gt):
    rel = R.from_quat(q_gt).inv() * R.from_quat(q_est_aligned)
    return np.degrees(np.linalg.norm(rel.as_rotvec(), axis=1))


# --------------------------------------------------------------- instruments

def join_instruments(t_abs, scan_csv=None, frame_stats=None):
    """Line up the per-scan instrument with the error curve's absolute time, so
    'why' can be read at the same instant as 'where'. Both files are optional;
    missing columns come back as NaN rather than failing."""
    import csv
    out = {}

    def _read(path, tkey, cols):
        if not path or not os.path.exists(path):
            return
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows or tkey not in rows[0]:
            return
        ts = np.array([float(r[tkey]) for r in rows])
        for c in cols:
            if c not in rows[0]:
                continue
            v = np.array([float(r[c]) if r[c] not in ("", "nan") else np.nan for r in rows])
            idx = np.clip(np.searchsorted(ts, t_abs) - 1, 0, len(ts) - 1)
            out[c] = v[idx]

    _read(frame_stats, "t", ["n_residuals", "n_planes", "h_pp_min_eig",
                             "h_rr_min_eig", "sum_weight", "denom_rejected_count"])
    _read(scan_csv, "t", ["trP_pos", "trP_vel", "trP_att", "omega_norm", "acc_norm"])
    return out


# --------------------------------------------------------------- main

def analyse(est_path, gt_path, gt_format, frame_correction, window, max_gap):
    t_e, p_e, q_e = load_tum(est_path)
    t_g, p_g, q_g = LOADERS[gt_format](gt_path)
    p_e, q_e = apply_frame_correction(p_e, q_e, frame_correction)

    # Evaluate ON THE GT TIMESTAMPS -- the estimate is what gets interpolated,
    # never the reference.
    p_i, q_i, valid = resample(t_e, p_e, q_e, t_g, max_gap)
    t = t_g[valid]
    src, dst = p_i[valid], p_g[valid]
    if len(t) < 20:
        return dict(error="too few matched samples", n=int(len(t)))

    Rm, tv = umeyama_no_scale(src, dst)
    aligned = (Rm @ src.T).T + tv
    e_glob = np.linalg.norm(aligned - dst, axis=1)

    e_pre = prefix_error(t, src, dst)
    e_win = window_error(t, src, dst, window)

    res = dict(
        n_samples=int(len(t)), t_start=float(t[0]), t_end=float(t[-1]),
        ate_rmse=float(np.sqrt(np.mean(e_glob ** 2))),
        ate_median=float(np.median(e_glob)),
        window_s=window,
    )
    for name, e in (("prefix", e_pre), ("window", e_win)):
        onset, meta = detect_onset(t, e)
        res[f"{name}_onset_t"] = onset
        res[f"{name}_onset_rel_s"] = None if onset is None else float(onset - t[0])
        res[f"{name}_detail"] = meta
        res[f"{name}_p50"] = float(np.nanmedian(e))
        res[f"{name}_p90"] = float(np.nanpercentile(e, 90))
        res[f"{name}_max"] = float(np.nanmax(e))

    rot_error_series = None
    if gt_has_attitude(q_g[valid]):
        q_al = (R.from_matrix(Rm) * R.from_quat(q_i[valid])).as_quat()
        rd = rot_error_deg(q_al, q_g[valid])
        rot_error_series = rd
        res["rot_ape_deg_rmse"] = float(np.sqrt(np.mean(rd ** 2)))
        res["rot_ape_deg_p90"] = float(np.percentile(rd, 90))
        res["rot_onset_t"], _ = detect_onset(t, rd, floor=0.5)
    else:
        res["rot_ape_deg_rmse"] = None
        res["rot_note"] = ("GT quaternion is identity everywhere -- position-only "
                           "reference. Rotation error NOT reported; it would be the "
                           "estimate's own rotation magnitude.")
    # T0-H1: the rotation-error time series (same t/valid indexing as
    # e_global/e_prefix/e_window -- rd is computed over q_g[valid], t over
    # the same matched samples), so a caller can put a rotation notch next
    # to a position notch on the same x-axis. None (not a column) when GT
    # has no licensed attitude -- see rot_note above.
    return res, dict(t=t, e_global=e_glob, e_prefix=e_pre, e_window=e_win,
                     rot_error_deg=rot_error_series,
                     # ALIGNED, not raw: the two estimates live in their own
                     # arbitrary frames until Umeyama puts them in GT's, and
                     # differencing the raw ones measures the frame offset
                     # rather than any disagreement about the trajectory.
                     aligned=aligned, src=src, dst=dst)


def pairwise(curveA, curveB, window):
    """Where do two estimates part company? Both are put into the GT frame by
    their own Umeyama fit first, so this is like-for-like and neither is
    assumed correct. `frac_a_better_local` is over the WINDOW error, so it
    answers "which has the better local shape", not "which drifted less"."""
    tA, tB = curveA["t"], curveB["t"]
    t, ia, ib = np.intersect1d(tA, tB, return_indices=True)
    if len(t) < 20:
        return dict(error="too few shared GT samples", n=int(len(t)))
    a, b = curveA["aligned"][ia], curveB["aligned"][ib]

    # Separation under each estimate's own GLOBAL fit. Reported, but NOT the
    # onset basis: a global Umeyama on the worse trajectory is dragged by its
    # tail, so this curve is offset from zero even where the two agree exactly
    # -- the same smearing this whole script exists to avoid, one level up.
    d_glob = np.linalg.norm(a - b, axis=1)

    # Local shape disagreement: align A onto B inside a sliding window and read
    # the residual at the centre. Independent of both global frames and of
    # accumulated drift, so its first sustained rise is genuinely "here is
    # where the two methods stop doing the same thing".
    d_loc = window_error(t, a, b, window)
    onset, meta = detect_onset(t, d_loc)

    ea, eb = curveA["e_window"][ia], curveB["e_window"][ib]
    better = np.where(np.isfinite(ea) & np.isfinite(eb), ea < eb, np.nan)
    return dict(n_shared=int(len(t)),
                split_onset_t=onset, split_detail=meta,
                sep_local_p50=float(np.nanmedian(d_loc)),
                sep_local_p90=float(np.nanpercentile(d_loc, 90)),
                sep_local_max=float(np.nanmax(d_loc)),
                sep_global_p50=float(np.median(d_glob)),
                sep_global_max=float(np.max(d_glob)),
                frac_a_better_local=float(np.nanmean(better)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--est", required=True, help="estimate .tum (the run under test)")
    ap.add_argument("--est-b", help="second estimate .tum, e.g. baseline pca")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--gt-format", default="tum", choices=list(LOADERS))
    ap.add_argument("--frame-correction", default="none",
                    choices=["none", "ntu_prism", "hilti_tip"])
    ap.add_argument("--window", type=float, default=10.0,
                    help="sliding-alignment span in seconds (default 10)")
    ap.add_argument("--max-gap", type=float, default=0.5)
    ap.add_argument("--scan-csv", help="scan.csv, to read trP/omega at the onset")
    ap.add_argument("--frame-stats", help="frame_stats.txt, for n_residuals/h_pp_min_eig")
    ap.add_argument("--out-prefix", help="write <prefix>_error.csv and _summary.json")
    a = ap.parse_args()

    outA = analyse(a.est, a.gt, a.gt_format, a.frame_correction, a.window, a.max_gap)
    if isinstance(outA, dict):
        print(json.dumps(outA, indent=2)); return
    resA, curveA = outA
    summary = dict(est=a.est, gt=a.gt, a=resA)

    inst = join_instruments(curveA["t"], a.scan_csv, a.frame_stats)
    if inst and resA.get("window_onset_t") is not None:
        i = int(np.searchsorted(curveA["t"], resA["window_onset_t"]))
        i = min(i, len(curveA["t"]) - 1)
        summary["instrument_at_window_onset"] = {k: float(v[i]) for k, v in inst.items()
                                                 if np.isfinite(v[i])}

    if a.est_b:
        outB = analyse(a.est_b, a.gt, a.gt_format, a.frame_correction, a.window, a.max_gap)
        if not isinstance(outB, dict):
            resB, curveB = outB
            summary["b"] = resB
            summary["pairwise"] = pairwise(curveA, curveB, a.window)

    print(json.dumps(summary, indent=2, default=str))

    if a.out_prefix:
        with open(a.out_prefix + "_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        cols = {k: curveA[k] for k in ("t", "e_global", "e_prefix", "e_window")}
        if curveA.get("rot_error_deg") is not None:
            cols["rot_error_deg"] = curveA["rot_error_deg"]
        cols.update(inst)
        keys = list(cols)
        with open(a.out_prefix + "_error.csv", "w") as f:
            f.write(",".join(keys) + "\n")
            for i in range(len(cols["t"])):
                f.write(",".join(f"{cols[k][i]:.6g}" for k in keys) + "\n")


if __name__ == "__main__":
    main()
