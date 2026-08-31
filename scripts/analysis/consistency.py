#!/usr/bin/env python3
"""
Filter-consistency diagnosis for a LIO/LIVO ESIKF, from logs only. No ground truth.

Answers three questions that a Q grid sweep cannot:
  1. Is the filter over- or under-confident?            -> NIS level vs its null
  2. Is the blame on Q (dynamics) or R (measurement)?   -> innovation whiteness + dynamics slope
  3. Does Q have any authority at all?                  -> tr(Q dt) / tr(P-)

Inputs
------
corr.csv   one row per point-to-plane correspondence, BEFORE outlier gating
    scan_id, nu, S, [S_sensor, S_plane_tilt, S_plane_d, S_pose],
    [range, incidence, N, J, aniso], [gated]
scan.csv   one row per LiDAR scan
    scan_id, t, dt, [trP_*], [trQdt_*], [omega_norm, acc_norm]

Only scan_id/nu/S are required in corr.csv; everything else enables an extra panel.
"""
import argparse, sys
import numpy as np
import pandas as pd
from scipy import stats

M_DOF = 1            # point-to-plane residual is scalar
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- statistics
def block_bootstrap_ci(x, stat=np.mean, block=25, n_boot=2000, alpha=0.05, rng=RNG):
    """CI for a statistic of a serially-correlated series, via moving-block bootstrap.

    Scan-level NIS means are autocorrelated (a pose error persists across scans),
    so an i.i.d. bootstrap or a sqrt(2/N) normal interval is badly overconfident.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n < 2 * block:
        block = max(1, n // 4)
    if n < 4:
        return (np.nan, np.nan)
    n_blocks = int(np.ceil(n / block))
    starts_pool = np.arange(0, n - block + 1)
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.choice(starts_pool, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[b] = stat(x[idx])
    return tuple(np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def acf(x, nlags=30):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    denom = np.dot(x, x)
    if denom == 0:
        return np.zeros(nlags + 1)
    return np.array([np.dot(x[: n - k], x[k:]) / denom for k in range(nlags + 1)])


def hac_ols(y, X, lags=10):
    """OLS with Newey-West standard errors (serial correlation in the residuals)."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1)
        u_t = (X * resid[:, None])[L:]
        u_tL = (X * resid[:, None])[:-L]
        G = u_t.T @ u_tL
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv * n / max(n - k, 1)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    return beta, se, t


# ---------------------------------------------------------------- panels
def scan_level_nis(corr, use_ungated_only=False):
    d = corr
    if use_ungated_only and "gated" in d.columns:
        d = d[d["gated"] == 0]
    d = d[np.isfinite(d["nu"]) & np.isfinite(d["S"]) & (d["S"] > 0)]
    nis = d["nu"].to_numpy() ** 2 / d["S"].to_numpy()
    z = d["nu"].to_numpy() / np.sqrt(d["S"].to_numpy())
    g = pd.DataFrame({"scan_id": d["scan_id"].to_numpy(), "nis": nis, "z": z})
    per = g.groupby("scan_id").agg(nis=("nis", "mean"), z=("z", "mean"), n=("nis", "size"))
    return per.sort_index(), nis


def panel_calibration(corr):
    per, nis_all = scan_level_nis(corr)
    m = per["nis"].to_numpy()
    lo, hi = block_bootstrap_ci(m)
    print("\n=== 1. CALIBRATION  (NIS; consistent filter -> E[NIS] = %d) ===" % M_DOF)
    print(f"  correspondences        : {len(nis_all):,}   scans: {len(m):,}")
    print(f"  mean NIS               : {np.mean(nis_all):.3f}  (per-correspondence)")
    print(f"  mean of per-scan NIS   : {np.mean(m):.3f}   95% block-CI [{lo:.3f}, {hi:.3f}]")
    print(f"  median per-scan NIS    : {np.median(m):.3f}")
    naive = np.sqrt(2.0 / len(nis_all))
    print(f"  (naive per-corr sigma  : {naive:.4f} -- do NOT use; correspondences in a")
    print( "   scan share one pose error, so the scan is the unit of replication)")
    if "gated" in corr.columns and corr["gated"].sum() > 0:
        per_ug, _ = scan_level_nis(corr, use_ungated_only=True)
        print(f"  ungated-only mean NIS  : {np.mean(per_ug['nis']):.3f}   "
              f"(gating removes {100*corr['gated'].mean():.1f}% of rows and truncates the")
        print( "   upper tail, biasing NIS DOWN -- always compute on the pre-gate set)")
    return per, (lo, hi)


def panel_whiteness(per):
    z = per["z"].to_numpy()
    a = acf(z, nlags=30)
    K = len(z)
    bound = 1.96 / np.sqrt(K)
    sig = [(k, a[k]) for k in range(1, 11) if abs(a[k]) > bound]
    lb_stat = K * (K + 2) * np.sum([a[k] ** 2 / (K - k) for k in range(1, 11)])
    lb_p = 1 - stats.chi2.cdf(lb_stat, 10)
    print("\n=== 2. WHITENESS  (normalized innovation; consistent filter -> white) ===")
    print(f"  lag-1 ACF              : {a[1]:+.3f}   (95% band +/-{bound:.3f})")
    print(f"  lag-2..5               : " + "  ".join(f"{a[k]:+.3f}" for k in range(2, 6)))
    print(f"  Ljung-Box(10)          : {lb_stat:.1f}   p = {lb_p:.2e}")
    print(f"  significant lags 1-10  : {[k for k, _ in sig] or 'none'}")
    return a[1], lb_p, bound


def panel_dynamics(per, scan):
    if scan is None or not {"omega_norm", "acc_norm"} <= set(scan.columns):
        print("\n=== 3. DYNAMICS SLOPE === (skipped: no omega_norm/acc_norm in scan.csv)")
        return None
    j = per.join(scan.set_index("scan_id")[["omega_norm", "acc_norm"]], how="inner").dropna()
    if len(j) < 50:
        print("\n=== 3. DYNAMICS SLOPE === (skipped: too few matched scans)")
        return None
    w = j["omega_norm"].to_numpy(); a_ = j["acc_norm"].to_numpy()
    Xw = np.column_stack([(w - w.mean()) / (w.std() or 1), (a_ - a_.mean()) / (a_.std() or 1)])
    beta, se, t = hac_ols(j["nis"].to_numpy(), Xw, lags=20)
    rho = np.corrcoef(Xw[:, 0], Xw[:, 1])[0, 1]
    print("\n=== 3. DYNAMICS SLOPE  (corroborating only -- see the caveat) ===")
    print(f"  intercept              : {beta[0]:+.3f}")
    print(f"  d(NIS)/d(omega, 1 sd)  : {beta[1]:+.3f}  (HAC t = {t[1]:+.2f})")
    print(f"  d(NIS)/d(acc,   1 sd)  : {beta[2]:+.3f}  (HAC t = {t[2]:+.2f})")
    if abs(rho) > 0.9:
        print(f"  WARNING corr(omega, acc) = {rho:+.2f} -- the two slopes are not separately")
        print( "  identified; read their sum, not the individual coefficients.")
    print("  CAVEAT (verified on a fixture with correct Q and a 4x-understated R, which")
    print("  still produced a t = -21 slope): a slope does NOT by itself imply Q. Whenever")
    print("  Q's share of S varies with motion, a mis-specified R also produces one. This")
    print("  panel corroborates a Q verdict that whiteness has already reached; it cannot")
    print("  establish one on its own.")
    return beta, t


def panel_budget(corr):
    parts = [c for c in ["S_sensor", "S_plane_tilt", "S_plane_d", "S_pose"] if c in corr.columns]
    if not parts:
        print("\n=== 4. VARIANCE BUDGET === (skipped: no S_* decomposition columns)")
        return None
    d = corr[corr["S"] > 0]
    print("\n=== 4. VARIANCE BUDGET  (share of S; this is T3-0d) ===")
    shares = {}
    for c in parts:
        sh = (d[c] / d["S"]).to_numpy()
        shares[c] = np.percentile(sh, [50, 90, 99])
        print(f"  {c:<14} p50 {shares[c][0]:6.3f}   p90 {shares[c][1]:6.3f}   p99 {shares[c][2]:6.3f}")
    plane = sum(shares[c][0] for c in parts if c.startswith("S_plane"))
    print(f"  -> plane terms at p50  : {plane:.3f}")
    if plane < 0.10:
        print("     AUTHORITY WARNING: any change to plane_var_ can move S by at most ~"
              f"{100*plane:.0f}% even if it changes plane_var_ by orders of magnitude.")
    return shares


def panel_covariates(corr, per):
    cols = [c for c in ["range", "incidence", "N", "J", "aniso"] if c in corr.columns]
    if not cols:
        print("\n=== 5. NIS vs COVARIATES === (skipped: no covariate columns)")
        return
    d = corr[(corr["S"] > 0) & np.isfinite(corr["nu"])].copy()
    d["nis"] = d["nu"] ** 2 / d["S"]
    print("\n=== 5. NIS vs COVARIATES  (structure here = a mis-specified R, by regime) ===")
    for c in cols:
        v = d[c].to_numpy()
        if not np.isfinite(v).any():
            continue
        q = np.nanpercentile(v, [0, 20, 40, 60, 80, 100])
        q = np.unique(q)
        if len(q) < 3:
            continue
        b = np.clip(np.digitize(v, q[1:-1]), 0, len(q) - 2)
        means = [d["nis"].to_numpy()[b == k].mean() if (b == k).sum() else np.nan
                 for k in range(len(q) - 1)]
        rng_ = np.nanmax(means) / max(np.nanmin(means), 1e-12)
        flag = "  <-- STRUCTURED" if rng_ > 1.5 else ""
        print(f"  {c:<10} quintile mean NIS: " +
              "  ".join(f"{m:6.2f}" for m in means) + f"   ratio {rng_:5.2f}{flag}")


def panel_q_authority(scan):
    if scan is None:
        return
    qcols = [c for c in scan.columns if c.startswith("trQdt_")]
    pcols = [c for c in scan.columns if c.startswith("trP_")]
    if not qcols or not pcols:
        print("\n=== 6. Q AUTHORITY === (skipped: no trQdt_*/trP_* columns)")
        return
    print("\n=== 6. Q AUTHORITY  (tr(Q dt) / tr(P-) per block; <1e-3 means Q is inert) ===")
    for qc in qcols:
        pc = "trP_" + qc[len("trQdt_"):]
        if pc not in scan.columns:
            continue
        r = (scan[qc] / scan[pc].replace(0, np.nan)).to_numpy()
        r = r[np.isfinite(r)]
        if r.size == 0:
            continue
        p50, p90 = np.percentile(r, [50, 90])
        flag = "  <-- INERT" if p90 < 1e-3 else ""
        print(f"  {qc[len('trQdt_'):]:<6} p50 {p50:.2e}   p90 {p90:.2e}{flag}")


def verdict(mean_nis, ci, lag1, bound, dyn):
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    over = ci[0] > 1.0
    under = ci[1] < 1.0
    corr_pos = lag1 > bound
    corr_neg = lag1 < -bound
    if not over and not under:
        print("  Calibrated on average (CI spans 1). Look at panel 5: a flat mean can")
        print("  still hide a regime where NIS is 4x and another where it is 0.3x.")
        return
    if over and corr_pos:
        print("  OVER-CONFIDENT and the innovations are POSITIVELY CORRELATED.")
        print("  -> Q IS TOO LOW.  The filter is too stiff, lags the true trajectory, and")
        print("     consecutive innovations share the same un-tracked error. Raise Q.")
    elif over and not corr_pos:
        print("  OVER-CONFIDENT but the innovations are WHITE.")
        print("  -> R IS TOO LOW, not Q.  The dynamics model is keeping up; the")
        print("     measurement variance is understated. In this codebase that is")
        print("     plane_var_ / sensor noise -- i.e. exactly what T1-T3 change.")
    elif under and corr_neg:
        print("  UNDER-CONFIDENT with NEGATIVE lag-1 correlation (over-correction).")
        print("  -> Q IS TOO HIGH.  The filter chases noise. Lower Q.")
    elif under:
        print("  UNDER-CONFIDENT, innovations white.")
        print("  -> R IS TOO HIGH, or Q is too high without visible over-correction.")
        print("     The dynamics slope below is the tie-breaker.")
    if dyn is not None:
        beta, t = dyn
        slope_sig = abs(t[1]) > 2 or abs(t[2]) > 2
        q_verdict = (over and corr_pos) or (under and corr_neg)
        if slope_sig and q_verdict:
            sign = "rises" if (beta[1] + beta[2]) > 0 else "falls"
            print(f"  Corroborated: NIS {sign} significantly with IMU excitation, which is")
            print( "  consistent with the Q verdict whiteness already reached.")
        elif slope_sig:
            print("  NIS depends significantly on IMU excitation, but whiteness did NOT")
            print( "  point at Q. Do not read this as a Q verdict -- a mis-specified R")
            print( "  produces the same slope whenever Q's share of S varies with motion.")
        elif q_verdict:
            print("  Not corroborated: NIS is flat in IMU excitation. Either the motion")
            print( "  range in this bag is too narrow to expose Q, or the Q error is")
            print( "  uniform rather than dynamics-dependent. Re-check on a livelier bag.")
        else:
            print("  NIS is flat in IMU excitation -- no additional evidence either way.")
    print("\n  Caveat that cannot be removed by more data: Q and R are not jointly")
    print("  identifiable from the innovation sequence alone. The separation above")
    print("  rests on structure -- Q's error is time-correlated and motion-dependent,")
    print("  R's is not -- and that structure is weak when the motion is uniform.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr", required=True)
    ap.add_argument("--scan", default=None)
    a = ap.parse_args()
    corr = pd.read_csv(a.corr)
    scan = pd.read_csv(a.scan) if a.scan else None
    print("=" * 72)
    print(f"FILTER CONSISTENCY REPORT   corr={a.corr}" + (f"  scan={a.scan}" if a.scan else ""))
    print("=" * 72)
    per, ci = panel_calibration(corr)
    lag1, lbp, bound = panel_whiteness(per)
    dyn = panel_dynamics(per, scan)
    panel_budget(corr)
    panel_covariates(corr, per)
    panel_q_authority(scan)
    verdict(per["nis"].mean(), ci, lag1, bound, dyn)


if __name__ == "__main__":
    main()
