#!/usr/bin/env python3
"""
Per-job statistical extraction from corr.csv / corr_scan.csv, built for the
"while the raw files exist" section of the livo_recon Experiment Register's
run-protocol cards (queue row 2, SP-4a"/4b" -- see also the bug ledger's
"the sweep scorer does not read the columns" row, which is why this module
exists as a standing, importable piece of the repo rather than another
one-off sweep script). A sweep's score-then-delete pass calls this once per
job, before that job's corr.csv is removed; corr_scan.csv, scan.csv and
spline_q.csv are cheap enough to retain outright and don't need this module.

Deliberately reuses this package's own consistency.py (block_bootstrap_ci,
acf) rather than reimplementing them -- that module's docstring and inline
comments carry the statistical reasoning (why the scan is the unit of
replication, why a naive per-correspondence sigma is wrong, etc.); this one
is the quiet, per-job-summary form of the same math, not a second opinion.

Two entry points, matching which raw file each needs:

  score_corr_scan_series(df)  -- corr_scan.csv ALONE. Exact (not strided)
      per-scan NIS is already accumulated there (sum_nis/n_nis_finite), so
      the block bootstrap and the ACF/Ljung-Box both run on the true
      per-scan series. Cheap, and corr_scan.csv is already retained
      unconditionally by the sweep listener -- this can be re-run any time,
      it does not need to happen before a delete.

  score_corr_raw(df)          -- corr.csv itself, one row per candidate
      correspondence, BEFORE outlier gating (matches consistency.py's own
      docstring). Computes the three statistics that cannot be reconstructed
      from any per-scan aggregate: the plane-share percentiles against the
      full S (T3-0d-2), the occ_aniso/occ_cells quintile-mean NIS (T8-0a-2's
      covariate), and the sigma_diag_squared+plane_var_term-vs-floor
      histogram. MUST be called before corr.csv is deleted -- there is no
      way to recover any of the three afterwards. This is exactly the
      failure SP-4a'/4b' hit: the columns existed, but nothing called for
      them before the file was gone.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from consistency import acf, block_bootstrap_ci  # noqa: E402


def _pctl(a, qs):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {q: None for q in qs}
    v = np.atleast_1d(np.nanpercentile(a, qs))
    return {q: float(x) for q, x in zip(qs, v)}


def score_corr_scan_series(corr_scan_df):
    """Items 1-3 of the protocol's extraction list. corr_scan.csv only."""
    d = corr_scan_df.sort_values("scan_id")
    out = {"n_scans": int(len(d))}
    if len(d) == 0:
        return out

    n_fin = d["n_nis_finite"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_scan_nis = d["sum_nis"].to_numpy(dtype=float) / n_fin
    valid = np.isfinite(per_scan_nis) & (n_fin > 0)
    m = per_scan_nis[valid]

    if m.size >= 4:
        lo, hi = block_bootstrap_ci(m)
        out["nis_scan_mean"] = float(np.mean(m))
        out["nis_scan_median"] = float(np.median(m))
        out["nis_scan_mean_bootstrap_lo"] = float(lo)
        out["nis_scan_mean_bootstrap_hi"] = float(hi)

        # lag-1 ACF and Ljung-Box(10) on the scan-mean normalised innovation
        # (the mean of nu/sqrt(S) per scan is the closest available proxy to
        # consistency.py's own panel_whiteness, which uses the per-CORRE-
        # SPONDENCE z averaged per scan -- from corr_scan.csv alone we only
        # have sum_nis, not sum_z, so this uses the NIS series itself; that
        # is what the register's own text asks for ("scan-mean normalised
        # innovation"), and Ljung-Box on NIS vs on z differ only in that NIS
        # is signed-squared -- whiteness in either is evidence against Q).
        a = acf(m, nlags=10)
        K = int(m.size)
        bound = 1.96 / np.sqrt(K)
        out["nis_lag1_acf"] = float(a[1])
        out["nis_lag1_acf_band"] = float(bound)
        lb_stat = K * (K + 2) * float(np.sum([a[k] ** 2 / (K - k) for k in range(1, 11) if K > k]))
        from scipy import stats as _stats
        out["nis_ljung_box_10_stat"] = lb_stat
        out["nis_ljung_box_10_p"] = float(1 - _stats.chi2.cdf(lb_stat, 10))

    n_candidates = int(d["n_candidates"].sum())
    n_accepted = int(d["n_accepted"].sum())
    n_dropped = int(d["n_dropped"].sum())
    out["n_candidates"] = n_candidates
    out["n_accepted"] = n_accepted
    out["n_dropped_by_ablation"] = n_dropped
    out["accept_fraction"] = n_accepted / max(n_candidates - n_dropped, 1)
    per_scan_count = d["n_accepted"].to_numpy(dtype=float)
    if per_scan_count.size:
        out["correspondence_count_per_scan_median"] = float(np.median(per_scan_count))
        out["correspondence_count_per_scan_p10"] = float(np.percentile(per_scan_count, 10))
    return out


def score_corr_raw(corr_df):
    """Items 4-6 of the protocol's extraction list. Needs corr.csv itself --
    call this before it is deleted, there is no reconstructing it after."""
    d = corr_df
    if "dropped_by_ablation" in d.columns:
        live = d["dropped_by_ablation"].to_numpy(dtype=float) == 0
        d = d[live]
    S = d["S"].to_numpy(dtype=float)
    finite_S = np.isfinite(S) & (S > 0)
    d = d[finite_S]
    S = S[finite_S]
    out = {"n_rows_scored": int(len(d))}
    if len(d) == 0:
        return out

    # --- 4. plane-share vs the FULL S (T3-0d-2) ---
    # corr.csv's own S column is ALREADY the full S (1e-3 + sigma_diag_squared
    # + plane_var_term + s_prior_pose, see voxelplane.cpp) -- no reconstruction
    # needed, just divide. This is the per-point ratio's own percentiles, not
    # (as consistency.py's panel_budget prints for a human) the sum of each
    # component's separate p50 -- the two are not the same statistic when the
    # components are correlated, and the register's own wording ("the
    # plane-share p50/p90/p99 against the full S") is the ratio's percentiles.
    plane_parts = [c for c in ("S_plane_tilt", "S_plane_d") if c in d.columns]
    if plane_parts:
        plane_var_term = sum(d[c].to_numpy(dtype=float) for c in plane_parts)
        share = plane_var_term / S
        p = _pctl(share, [50, 90, 99])
        out["plane_share_p50"] = p[50]
        out["plane_share_p90"] = p[90]
        out["plane_share_p99"] = p[99]

    # --- 5. occ_aniso / occ_cells quintile-mean NIS (T8-0a-2's covariate) ---
    nu = d["nu"].to_numpy(dtype=float)
    nis = nu ** 2 / S
    finite_nis = np.isfinite(nis)
    for cov in ("occ_aniso", "occ_cells"):
        if cov not in d.columns:
            continue
        v = d[cov].to_numpy(dtype=float)
        ok = finite_nis & np.isfinite(v)
        if ok.sum() < 50:
            continue
        q = np.nanpercentile(v[ok], [0, 20, 40, 60, 80, 100])
        q = np.unique(q)
        if len(q) < 3:
            continue
        b = np.clip(np.digitize(v[ok], q[1:-1]), 0, len(q) - 2)
        nis_ok = nis[ok]
        means = [float(nis_ok[b == k].mean()) if (b == k).sum() else None
                 for k in range(len(q) - 1)]
        out[f"nis_quintile_mean_by_{cov}"] = means
        finite_means = [m for m in means if m is not None]
        if len(finite_means) >= 2 and min(finite_means) > 0:
            out[f"nis_quintile_ratio_{cov}"] = float(max(finite_means) / min(finite_means))

    # --- 6. histogram of sigma_diag_squared + plane_var_term vs the 1e-3 floor ---
    # gate() itself uses sigma_gate_squared = sigma_diag_squared + plane_var_term
    # (no floor added there); the WEIGHT elsewhere is 1/(1e-3 + sigma_diag_squared
    # + plane_var_term) -- see the bug ledger's "gate and weight use different
    # floors" row, which this histogram is what closes it. sigma_diag_squared is
    # S_sensor (+S_pose when pose_cov is folded into the residual side);
    # plane_var_term is S_plane_tilt+S_plane_d.
    if "S_sensor" in d.columns and plane_parts:
        sigma_diag = d["S_sensor"].to_numpy(dtype=float)
        if "S_pose" in d.columns:
            sigma_diag = sigma_diag + d["S_pose"].to_numpy(dtype=float)
        plane_var_term = sum(d[c].to_numpy(dtype=float) for c in plane_parts)
        total = sigma_diag + plane_var_term
        finite_total = np.isfinite(total)
        t = total[finite_total]
        if t.size:
            floor = 1e-3
            out["sigma_plus_planevar_median"] = float(np.median(t))
            out["sigma_plus_planevar_below_floor_fraction"] = float(np.mean(t < floor))
            edges = [0.0, floor / 100, floor / 10, floor, floor * 10, floor * 100, np.inf]
            counts, _ = np.histogram(t, bins=edges)
            out["sigma_plus_planevar_histogram_edges"] = [
                (e if np.isfinite(e) else "inf") for e in edges[:-1]
            ] + ["inf"]
            out["sigma_plus_planevar_histogram_counts"] = [int(c) for c in counts]

    return out


def score_job_corr(corr_path, corr_scan_path=None):
    """Convenience wrapper: run both halves against a job's own files and
    merge into one dict. Callers that want to defer the corr.csv read (e.g.
    to confirm corr_scan.csv is present first) should call the two functions
    above directly instead."""
    out = {}
    if corr_scan_path and os.path.exists(corr_scan_path) and os.path.getsize(corr_scan_path) > 0:
        out.update(score_corr_scan_series(pd.read_csv(corr_scan_path)))
    if corr_path and os.path.exists(corr_path) and os.path.getsize(corr_path) > 0:
        out.update(score_corr_raw(pd.read_csv(corr_path)))
    return out


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corr", required=True)
    ap.add_argument("--corr-scan", default=None)
    args = ap.parse_args()
    result = score_job_corr(args.corr, args.corr_scan)
    print(json.dumps(result, indent=2))
