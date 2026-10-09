#!/usr/bin/env python3
"""r83_cov_consistency v2: does the state covariance track the actual error, and how does that change over time? Python only, no node run.

Reads one arm's state_trace.csv(.gz) (post_lio rows: quaternion qw..qz of R body->world, position, the upper triangle P_i_j of the 18-dim
covariance, layout R 0-2, P 3-5, V 6-8, BG 9-11, BA 12-14, G 15-17) and a ground-truth TUM file (t px py pz qx qy qz qw), and reports how the predicted
sd compares with the error of the aligned trajectory, per axis, with the NEES and its coverage.

  python3 r83_cov_consistency.py --arm <arm> --trace state_trace.csv.gz --gt <gt.txt> [--lever "x y z"] [--onset-scan N]
        [--max-gt-gap 0.15] [--pipeline-scan arm_scan.csv.gz] [--bins 100 --out-bins bins.csv] --out rows.csv

What is new in v2 (v1 was used for the R82A return):
  --max-gt-gap S   a scan is JOINED to the ground truth only if the two GT samples that bracket it are at most S seconds apart (default 0.15 s;
                   the Leica files are 20 Hz, so a normal gap is 0.05 s). v1 interpolated across any gap, however long. Scans that are not joined
                   are left out of the alignment AND of every statistic; their number is reported (n_scans_total, n_joined, n_gap_excluded).
  rms3d_err_mm     root mean square of the 3D position error over the window's joined scans: the same kind of number as the pipeline's ATE (an RMS),
                   so the two can be compared. med_err3d_mm is the median of the 3D error. (v1 only gave per-axis medians.)
  --pipeline-scan  the arm's arm_scan.csv(.gz) (or arm_scan_all.csv.gz filtered to the arm) with the pipeline's own per-scan GT error column
                   gt_err_mm; adds pipe_rms_err_mm and pipe_med_err_mm over the SAME joined scans, so this tool's error and the pipeline's can be
                   read side by side. Scan column: the first of scan, scan_id, scan_index that exists.
  --bins N         also writes --out-bins: the same statistics for consecutive blocks of N scans (scan 1..N, N+1..2N, ...) with the scan range, the time
                   since the first scan, and (if --onset-scan) the bin start relative to the onset. This is the time series of over-confidence.

Alignment: one rigid transform (Umeyama, position only, rotation allowed) of the whole joined trajectory to GT, the pipeline's own alignment. The
errors are therefore ALIGNED errors: a constant offset between the estimate frame and GT does not show up, and a drift that the alignment
spreads over the run shows up smaller than it is. The covariance is the filter's marginal covariance, not a covariance of the aligned error, so
a NEES above its nominal value says "over-confident for this error", not "wrong filter" (HYPOTHESIS-grade reading).
Attitude: only if the GT file carries a real quaternion (Hilti dense GT; NTU Leica files have an identity). The rotation error is formed in the body
frame, e = Log(R_est^T R_align^T R_gt), and the covariance block P_RR is assumed to be a body-frame (right) perturbation; the output reports
the NEES for BOTH conventions (right and left) so the one that is consistent is visible; which one the node uses must be read from the state code.
Windows: S = scans 30 .. onset-1, W = onset .. onset+59, M = rest, taken from --onset-scan; without it, one window 'all'.
Columns: arm, window, n, n_scans_total, n_joined, n_gap_excluded, then for pos: med_abs_err_mm_{x,y,z}, med_sd_mm_{x,y,z}, med_err3d_mm,
rms3d_err_mm, med_sd3d_mm (sqrt of the trace of the position covariance), nees_pos_med, cover99_pos (fraction of scans with NEES <= the
3-dof chi-square 99 percent point, nominal 0.99), and for rot (if available): the same in mrad plus both conventions; plus the 18-dim reductions.
"""
import argparse, sys, os
import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.spatial.transform import Rotation as Rot, Slerp

CH99 = chi2.ppf(0.99, 3)


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mu_d - R @ mu_s


def load_trace(path):
    t = pd.read_csv(path)
    t = t[t["phase"] == "post_lio"].reset_index(drop=True)
    return t


def cov_block(t, i0, i1):
    n = len(t)
    P = np.zeros((n, i1 - i0, i1 - i0))
    for a in range(i0, i1):
        for b in range(a, i1):
            c = t["P_%d_%d" % (a, b)].to_numpy(float)
            P[:, a - i0, b - i0] = c
            P[:, b - i0, a - i0] = c
    return P


def load_pipeline_err(path, scans):
    """gt_err_mm of the pipeline per scan, aligned to `scans` (NaN where absent)."""
    d = pd.read_csv(path)
    col = next((c for c in ("scan", "scan_id", "scan_index") if c in d.columns), None)
    if col is None or "gt_err_mm" not in d.columns:
        raise ValueError("--pipeline-scan needs a scan/scan_id/scan_index column and gt_err_mm")
    s = pd.Series(d["gt_err_mm"].to_numpy(float), index=d[col].to_numpy(int))
    s = s[~s.index.duplicated(keep="first")]
    return s.reindex(scans).to_numpy(float)


def window_row(arm, name, m, ctx):
    """All statistics of one set of scans m (boolean mask over the trace)."""
    e_est, Ppp, has_q, er_r, er_l, Prr, P18, pipe, tot, joined = (ctx[k] for k in ("e_est", "Ppp", "has_q", "er_r", "er_l", "Prr", "P18", "pipe", "tot", "joined"))
    pinv = np.linalg.inv(Ppp[m])
    nees = np.einsum("ni,nij,nj->n", e_est[m], pinv, e_est[m])
    e3 = np.linalg.norm(e_est[m], axis=1)
    r = dict(arm=arm, window=name, n=int(m.sum()), n_scans_total=int(tot), n_joined=int(joined), n_gap_excluded=int(tot - joined))
    for k, ax in enumerate("xyz"):
        r["med_abs_err_mm_" + ax] = 1000 * np.median(np.abs(e_est[m, k]))
        r["med_sd_mm_" + ax] = 1000 * np.median(np.sqrt(Ppp[m][:, k, k]))
    r["med_err3d_mm"] = 1000 * float(np.median(e3))
    r["rms3d_err_mm"] = 1000 * float(np.sqrt(np.mean(e3 ** 2)))
    r["med_sd3d_mm"] = 1000 * float(np.median(np.sqrt(np.trace(Ppp[m], axis1=1, axis2=2))))
    r["nees_pos_med"] = float(np.median(nees))
    r["cover99_pos"] = float(np.mean(nees <= CH99))
    if pipe is not None:
        pm = pipe[m]
        pm = pm[np.isfinite(pm)]
        r["pipe_rms_err_mm"] = float(np.sqrt(np.mean(pm ** 2))) if len(pm) else np.nan
        r["pipe_med_err_mm"] = float(np.median(pm)) if len(pm) else np.nan
        r["n_pipe_scans"] = int(len(pm))
    if has_q:
        prinv = np.linalg.inv(Prr[m])
        for tag, er in (("right", er_r), ("left", er_l)):
            nr = np.einsum("ni,nij,nj->n", er[m], prinv, er[m])
            r["nees_rot_%s_med" % tag] = float(np.median(nr))
            r["cover99_rot_%s" % tag] = float(np.mean(nr <= CH99))
        for k, ax in enumerate("xyz"):
            r["med_abs_rot_mrad_" + ax] = 1000 * np.median(np.abs(er_r[m, k]))
            r["med_sd_rot_mrad_" + ax] = 1000 * np.median(np.sqrt(Prr[m][:, k, k]))
    if P18 is not None:
        Pm = P18[m]
        ev = np.linalg.eigvalsh(Pm)
        r["log10_cond_P18_med"] = float(np.median(np.log10(np.maximum(ev[:, -1], 1e-300) / np.maximum(ev[:, 0], 1e-300))))
        r["min_eig_P18_med"] = float(np.median(ev[:, 0]))
        r["sd_v_mm_s_med"] = 1000 * float(np.median(np.sqrt(np.trace(Pm[:, 6:9, 6:9], axis1=1, axis2=2) / 3)))
        r["sd_bg_mrad_s_med"] = 1000 * float(np.median(np.sqrt(np.trace(Pm[:, 9:12, 9:12], axis1=1, axis2=2) / 3)))
        r["sd_ba_mm_s2_med"] = 1000 * float(np.median(np.sqrt(np.trace(Pm[:, 12:15, 12:15], axis1=1, axis2=2) / 3)))
    return r


def analyse(tr, gt, lever, onset, arm, max_gap=0.15, pipeline_scan=None, bins=0, min_n=5):
    q = tr[["qx", "qy", "qz", "qw"]].to_numpy(float)
    R_est = Rot.from_quat(q)
    p = tr[["px", "py", "pz"]].to_numpy(float)
    ts = tr["t_abs"].to_numpy(float)
    gt = np.loadtxt(gt, comments="#")
    tg = gt[:, 0]
    inside = (ts >= tg[0]) & (ts <= tg[-1])
    # joined: the two GT samples that bracket the scan are at most max_gap apart
    j = np.clip(np.searchsorted(tg, ts), 1, len(tg) - 1)
    gap = tg[j] - tg[j - 1]
    ok = inside & (gap <= max_gap) if max_gap and max_gap > 0 else inside
    if ok.sum() < 20:
        raise ValueError("fewer than 20 joined scans (inside the GT span and not across a GT gap above %.3f s)" % max_gap)
    pg = np.stack([np.interp(ts, tg, gt[:, k]) for k in (1, 2, 3)], 1)
    has_q = np.any(np.abs(gt[:, 4:7]) > 1e-9)
    pe = p + R_est.apply(np.tile(lever, (len(p), 1)))   # the pipeline's convention: estimate + R_est * lever, compared with the GT point
    Ra, ta = umeyama(pe[ok], pg[ok])
    e = (pg - (pe @ Ra.T + ta))   # GT minus aligned estimate, world frame of GT
    e_est = e @ Ra               # position covariance is in the estimate's world frame: rotate the error back with Ra^T
    P = cov_block(tr, 0, 6)
    P18 = cov_block(tr, 0, 18) if "P_17_17" in tr.columns else None
    scan = tr["scan_id"].to_numpy()
    ctx = dict(e_est=e_est, Ppp=P[:, 3:6, 3:6], has_q=has_q, er_r=None, er_l=None, Prr=None, P18=P18, tot=len(tr), joined=int(ok.sum()),
               pipe=load_pipeline_err(pipeline_scan, scan) if pipeline_scan else None)
    if has_q:
        Rg = Slerp(tg, Rot.from_quat(gt[:, 4:8]))(np.clip(ts, tg[0], tg[-1]))
        ctx["er_r"] = (R_est.inv() * Rot.from_matrix(Ra).inv() * Rg).as_rotvec()
        ctx["er_l"] = (Rot.from_matrix(Ra).inv() * Rg * R_est.inv()).as_rotvec()
        ctx["Prr"] = P[:, 0:3, 0:3]
    if onset:
        wins = {"S": (scan >= 30) & (scan < onset), "W": (scan >= onset) & (scan < onset + 60), "M": scan >= onset + 60}
    else:
        wins = {"all": scan >= 1}
    rows = []
    for w, m in wins.items():
        m = m & ok
        if m.sum() >= min_n:
            rows.append(window_row(arm, w, m, ctx))
    brows = []
    if bins and bins > 0:
        s0 = int(scan.min())
        t0 = ts[0]
        for lo in range(s0, int(scan.max()) + 1, bins):
            m = (scan >= lo) & (scan < lo + bins) & ok
            if m.sum() >= min_n:
                r = window_row(arm, "bin", m, ctx)
                r.update(bin_scan_lo=lo, bin_scan_hi=lo + bins - 1, t_since_first_s=float(ts[m][0] - t0), bin_start_minus_onset=(lo - onset) if onset else np.nan)
                brows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(brows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--lever", default="0 0 0")
    ap.add_argument("--onset-scan", type=int, default=0)
    ap.add_argument("--max-gt-gap", type=float, default=0.15)
    ap.add_argument("--pipeline-scan", default=None)
    ap.add_argument("--bins", type=int, default=0)
    ap.add_argument("--out-bins", default=None)
    ap.add_argument("--out", required=True, help="rows CSV; one file per arm is safest (NTU and Hilti rows have different columns)")
    a = ap.parse_args()
    df, bf = analyse(load_trace(a.trace), a.gt, np.array([float(v) for v in a.lever.split()]), a.onset_scan, a.arm, a.max_gt_gap, a.pipeline_scan, a.bins)
    df.to_csv(a.out, index=False)
    if a.bins and a.out_bins:
        bf.to_csv(a.out_bins, index=False)
    print(df.round(3).to_string(index=False))
    return 0 if len(df) else 1


if __name__ == "__main__":
    sys.exit(main())
