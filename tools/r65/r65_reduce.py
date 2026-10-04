#!/usr/bin/env python3
"""R65 per-arm reducer (planning agent's; R64 reducer plus: exact scan times in arm_scan.csv, rotation error components, point_cov_stats.csv merge). Condenses one arm's raw files into arm_scan.csv + arm_summary.json.
NOTHING HERE INTERPRETS RESULTS: it computes the definitions below and writes them down. The coding agent runs it and
returns its outputs; it does not comment on them.

    python3 r65_reduce.py <arm_dir> [--onset-scan 485] [--stat-last 480] [--trans-last 540]

Inputs looked for in <arm_dir> (missing ones are listed in summary["missing"], never an error):
  state_trace.csv, state_reference.txt, lio_gt_matched.csv, frame_stats.txt, motion_q_scan.csv,
  plane_fit_reasons.csv, plane_fit_trace.csv, point_cov_stats.csv, results_lio.txt, arm.json (written by post_queue.sh).

DEFINITIONS (every number in the summary uses exactly these)
  scan            = state_trace.csv scan_id (frame index; scan k ends 0.1 s * k after the first scan).
  stationary      = scans 1..stat_last (default 480). The IMU dynamic-acceleration step is at scan 485 (onset-scan);
                    scans stat_last+1..trans_last (default 481..540) are 'transition'; later scans 'motion'.
                    These are fixed scan numbers for the eee_01 bag, NOT detected per arm.
  error           = against the STATIONARY REFERENCE in state_reference.txt (the state at the first LiDAR frame):
                    e_p = p - p_ref (m), e_v = v - v_ref, e_R = Log(R_ref^T R) (rad). Meaningful only while stationary.
  position rms    = sqrt(mean over scans of ||e_p||^2), in mm.   rotation rms = same with ||e_R||, in mrad.
  e2/P            = sum(e_i^2) / sum(P_ii) over the stationary scans for one axis i. 1 = calibrated, >1 = overconfident.
                    'pos' / 'rot' entries pool the three position / three rotation axes (sum over axes of both sums).
  NEES9           = e^T P^-1 e over the 9 components (R, p, v) with the 9x9 covariance block; expected mean 9.
  excursion       = max over scans 1..10 of ||e_p|| (mm).      settled offset = ||mean e_p over scans 50..stat_last|| (mm).
  wander          = ||mean e_p (scans 400..480) - mean e_p (scans 50..100)|| (mm).
  disturbance     = for scans 380..470: max ||e_p - median e_p(scans 300..380)|| (mm) and the scan where it occurs.
  GT error        = only for arms whose GT join covers >= 300 scans: est body position (p + R*lever) joined to the GT
                    position track (linear interpolation to the scan time, GT gaps < 0.12 s), one rigid alignment
                    (no scale) fitted on ALL joined scans, error = aligned estimate - GT; ATE = rms of ||error||.
                    Also reported per segment (transition, motion) and per 500-scan block. Arms with a shorter join
                    have gt_status='short_join' and NO GT numbers (the 74-scan 60 s alignment is not trusted).
  open loop       = arm kind 'open' (lio.open_loop.mode=propagate_only): errors are grouped by seconds since the window
                    start (tau bins of 1 s); ol_ratio_* = sum e^2 / sum P per axis over tau >= 1 s, all stationary windows.
"""
import argparse, json, math, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import quat_to_R, log_so3, read_reference, umeyama, jsonable, chi2_ppf  # noqa: E402  (copied beside this file)

AXES = ["Rx", "Ry", "Rz", "px", "py", "pz", "vx", "vy", "vz"]
LEVER = np.array([-0.293656, -0.012288, -0.273095])


def pblock(df, n=9):
    """(N, n, n) covariance blocks from P_i_j columns (upper triangle stored)."""
    P = np.zeros((len(df), n, n))
    for i in range(n):
        for j in range(i, n):
            c = "P_%d_%d" % (i, j)
            if c in df.columns:
                v = df[c].to_numpy(float)
                P[:, i, j] = v
                P[:, j, i] = v
    return P


def state_arrays(rows):
    q = rows[["qw", "qx", "qy", "qz"]].to_numpy(float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R, rows[["px", "py", "pz"]].to_numpy(float), rows[["vx", "vy", "vz"]].to_numpy(float)


def errors(rows, Rref, pref, vref):
    R, p, v = state_arrays(rows)
    eR = np.array([log_so3(Rref.T @ R[i]) for i in range(len(R))])
    return np.hstack([eR, p - pref, v - vref]), R, p


def stat_block(E, P, nees_mask=None):
    """Per-axis and pooled e2/P, rms, NEES9 for one phase on the given rows."""
    out = {"n": int(len(E))}
    if len(E) < 3:
        return out
    Pd = np.stack([P[:, i, i] for i in range(9)], axis=1)
    e2, pv = (E ** 2).sum(0), np.maximum(Pd, 1e-300).sum(0)
    out["axes"] = {ax: {"rms_e": float(math.sqrt((E[:, i] ** 2).mean())), "mean_sd": float(np.sqrt(np.maximum(Pd[:, i], 0)).mean()),
                        "e2_over_P": float(e2[i] / pv[i]), "mean_e": float(E[:, i].mean())} for i, ax in enumerate(AXES)}
    for nm, sl in (("rot", slice(0, 3)), ("pos", slice(3, 6)), ("vel", slice(6, 9))):
        out[nm] = {"rms": float(math.sqrt((E[:, sl] ** 2).sum(1).mean())), "e2_over_P": float(e2[sl].sum() / pv[sl].sum()),
                   "mean_sd_norm": float(np.sqrt(np.maximum(Pd[:, sl].sum(1), 0)).mean())}
    nees = np.full(len(E), np.nan)
    for k in range(len(E)):
        try:
            nees[k] = float(E[k] @ np.linalg.solve(P[k], E[k]))
        except np.linalg.LinAlgError:
            pass
    ok = np.isfinite(nees)
    if ok.any():
        lo, hi = chi2_ppf(0.025, 9), chi2_ppf(0.975, 9)
        nn = nees[ok]
        out["nees9"] = {"mean": float(nn.mean()), "median": float(np.median(nn)), "frac_in_95": float(np.mean((nn > lo) & (nn < hi))),
                        "frac_above_95": float(np.mean(nn >= hi)), "n_finite": int(ok.sum())}
    return out, nees


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir")
    ap.add_argument("--onset-scan", type=int, default=485)
    ap.add_argument("--stat-last", type=int, default=480)
    ap.add_argument("--trans-last", type=int, default=540)
    a = ap.parse_args()
    D = a.arm_dir
    meta = json.load(open(os.path.join(D, "arm.json"))) if os.path.exists(os.path.join(D, "arm.json")) else {}
    S = {"arm": meta.get("arm", os.path.basename(os.path.abspath(D))), "meta": meta, "missing": [], "warnings": [],
         "definitions": {"onset_scan": a.onset_scan, "stat_last": a.stat_last, "trans_last": a.trans_last}}

    def path(fn):
        p = os.path.join(D, fn)
        if os.path.exists(p):
            return p
        S["missing"].append(fn)
        return None

    tp, rp = path("state_trace.csv"), path("state_reference.txt")
    if not (tp and rp):
        S["warnings"].append("state_trace.csv or state_reference.txt missing: nothing reduced")
        json.dump(jsonable(S), open(os.path.join(D, "arm_summary.json"), "w"), indent=1)
        return 1
    df = pd.read_csv(tp)
    Rref, pref, vref, P0 = read_reference(rp)
    imu = df[df.phase == "post_imu"].drop_duplicates("scan_id", keep="last").set_index("scan_id")
    lio = df[df.phase == "post_lio"].drop_duplicates("scan_id", keep="last").set_index("scan_id")
    scans = np.array(sorted(set(imu.index) & set(lio.index)))
    if len(scans) == 0:
        S["warnings"].append("no scan has both post_imu and post_lio rows")
        json.dump(jsonable(S), open(os.path.join(D, "arm_summary.json"), "w"), indent=1)
        return 1
    imu, lio = imu.loc[scans], lio.loc[scans]
    Ei, Ri, pi_ = errors(imu, Rref, pref, vref)
    El, Rl, pl_ = errors(lio, Rref, pref, vref)
    Pi, Pl = pblock(imu), pblock(lio)
    t = lio.t_abs.to_numpy(float)
    S["n_scans"], S["first_scan"], S["last_scan"] = int(len(scans)), int(scans[0]), int(scans[-1])
    S["t_first"], S["t_last"] = float(t[0]), float(t[-1])
    stat = scans <= a.stat_last
    seg = np.where(stat, "stationary", np.where(scans <= a.trans_last, "transition", "motion"))
    S["n_stationary_scans"] = int(stat.sum())
    is_open = (meta.get("kind") == "open")
    S["kind"] = meta.get("kind")

    # ---- stationary consistency, post_lio (the reported state) and post_imu (the prior)
    for nm, E, P in (("post_lio", El, Pl), ("post_imu", Ei, Pi)):
        r = stat_block(E[stat], P[stat])
        S.setdefault("stationary", {})[nm] = r[0] if isinstance(r, tuple) else r
        if nm == "post_lio" and isinstance(r, tuple):
            nees_l = np.full(len(scans), np.nan); nees_l[stat] = r[1]
        if nm == "post_imu" and isinstance(r, tuple):
            nees_i = np.full(len(scans), np.nan); nees_i[stat] = r[1]
    nees_l = locals().get("nees_l", np.full(len(scans), np.nan)); nees_i = locals().get("nees_i", np.full(len(scans), np.nan))

    # ---- stationary shape metrics (position, post_lio), mm
    ep = El[:, 3:6] * 1000.0
    ep_norm = np.linalg.norm(ep, axis=1)

    def win(lo, hi):
        return (scans >= lo) & (scans <= hi)
    m = win(1, 10)
    if m.any():
        k = int(np.argmax(np.where(m, ep_norm, -1)))
        S["excursion_mm"], S["excursion_scan"] = float(ep_norm[k]), int(scans[k])
    m50 = win(50, a.stat_last)
    if m50.sum() > 5:
        S["settled_offset_mm"] = float(np.linalg.norm(ep[m50].mean(0)))
        S["settled_offset_vec_mm"] = ep[m50].mean(0).tolist()
    ma, mb = win(50, 100), win(400, 480)
    if ma.sum() > 5 and mb.sum() > 5:
        S["wander_mm"] = float(np.linalg.norm(ep[mb].mean(0) - ep[ma].mean(0)))
    mref, mw = win(300, 380), win(380, 470)
    if mref.sum() > 5 and mw.sum() > 5:
        dev = np.linalg.norm(ep - np.median(ep[mref], axis=0), axis=1)
        k = int(np.argmax(np.where(mw, dev, -1)))
        S["disturbance_mm"], S["disturbance_scan"] = float(dev[k]), int(scans[k])
    S["stationary_pos_error_mm_at_scans"] = {str(s): float(ep_norm[scans == s][0]) for s in (1, 5, 10, 50, 100, 200, 300, 400, 413, 423, 450, 480)
                                              if (scans == s).any()}

    # ---- iterations and residuals (state_trace)
    ci = lio.completed_iterations.to_numpy(float) if "completed_iterations" in lio else np.full(len(scans), np.nan)
    rc = lio.residual_count.to_numpy(float)
    S["iterations"] = {"max": float(np.nanmax(ci)), "frac_at_max": float(np.mean(ci == np.nanmax(ci))), "median": float(np.nanmedian(ci))}
    for nm, mm in (("stationary", stat), ("transition", seg == "transition"), ("motion", seg == "motion")):
        if mm.any() and not is_open:
            S.setdefault("residuals", {})[nm] = {"median": float(np.median(rc[mm])), "min": float(rc[mm].min()), "max": float(rc[mm].max())}

    # ---- frame_stats.txt (per-frame map / update / redundancy columns)
    fs = None
    fp = path("frame_stats.txt")
    if fp:
        try:
            fs = pd.read_csv(fp).drop_duplicates("frame_idx", keep="last").set_index("frame_idx")
        except Exception as ex:
            S["warnings"].append("frame_stats.txt unreadable: %s" % ex)
    fs_cols = {"fs_n_residuals": "n_residuals", "fs_n_voxels_is_plane": "n_voxels_is_plane", "fs_n_voxels": "n_voxels",
               "fs_iters": "iters", "fs_dx_pos_mm": "dx_pos_mm", "fs_dx_rot_deg": "dx_rot_deg",
               "fs_redund_groups": "redund_groups", "fs_redund_n_raw": "redund_n_raw", "fs_redund_info_ratio": "redund_info_ratio",
               "fs_reduced_chi2": "reduced_chi2", "fs_nis": "nis", "fs_naive_info_gain": "naive_info_gain",
               "fs_woodbury_info_gain": "woodbury_info_gain", "fs_denom_rejected": "denom_rejected_count",
               "fs_max_plane_var_trace": "max_plane_var_trace"}
    fsv = {}
    for k, c in fs_cols.items():
        if fs is not None and c in fs.columns:
            fsv[k] = fs[c].reindex(scans).to_numpy(float)
        else:
            fsv[k] = np.full(len(scans), np.nan)
    if fs is not None:
        for nm, mm in (("stationary", stat), ("transition", seg == "transition"), ("motion", seg == "motion")):
            if mm.any():
                S.setdefault("frame_stats", {})[nm] = {k: (float(np.nanmedian(v[mm])) if np.isfinite(v[mm]).any() else None) for k, v in fsv.items()}

    # ---- GT (position only)
    gt_err = np.full(len(scans), np.nan)
    gp = path("lio_gt_matched.csv")
    S["gt_status"] = "no_gt_file"
    if gp:
        gt = pd.read_csv(gp)
        gt_t = gt.t_abs.to_numpy(float); GTP = gt[["gt_px", "gt_py", "gt_pz"]].to_numpy(float)
        est_body = pl_ + np.einsum("nij,j->ni", Rl, LEVER)
        idx, gtp, ok = np.full(len(scans), -1), np.zeros((len(scans), 3)), np.zeros(len(scans), bool)
        for n in range(len(scans)):
            if gt_t[0] <= t[n] <= gt_t[-1]:
                j = int(np.searchsorted(gt_t, t[n])); j1 = min(max(j, 1), len(gt_t) - 1); j0 = j1 - 1
                if gt_t[j1] - gt_t[j0] < 0.12:
                    w = (t[n] - gt_t[j0]) / max(gt_t[j1] - gt_t[j0], 1e-9)
                    gtp[n] = (1 - w) * GTP[j0] + w * GTP[j1]; ok[n] = True
        S["gt_join_scans"] = int(ok.sum())
        S["gt_join_first_last_scan"] = [int(scans[ok][0]), int(scans[ok][-1])] if ok.any() else None
        if ok.sum() >= 300:
            Ral, tal = umeyama(est_body[ok], gtp[ok])
            e = (Ral @ est_body[ok].T).T + tal - gtp[ok]
            gt_err[ok] = np.linalg.norm(e, axis=1) * 1000.0
            S["gt_status"] = "aligned_all_joined_scans"
            S["gt_ate_mm"] = float(math.sqrt(np.mean(gt_err[ok] ** 2)))
            for nm in ("stationary", "transition", "motion"):
                mm = ok & (seg == nm)
                if mm.sum() > 3:
                    S.setdefault("gt_ate_by_segment_mm", {})[nm] = {"rms": float(math.sqrt(np.mean(gt_err[mm] ** 2))), "n": int(mm.sum())}
            blk = (scans[ok] - 1) // 500
            S["gt_rms_by_500_scan_block_mm"] = {str(int(b) * 500 + 1): float(math.sqrt(np.mean(gt_err[ok][blk == b] ** 2))) for b in np.unique(blk)}
            S["gt_max_mm"], S["gt_max_scan"] = float(np.nanmax(gt_err)), int(scans[np.nanargmax(gt_err)])
        else:
            S["gt_status"] = "short_join"

    # ---- open-loop tau table
    if is_open:
        win_id = imu.ol_window.to_numpy(int)
        tau = np.zeros(len(scans))
        for w in np.unique(win_id):
            mm = (win_id == w)
            tau[mm] = t[mm] - (t[mm].min() - 0.1)
        rows, use = [], stat
        for b in sorted(set(np.floor(tau[use]).astype(int))):
            mm = use & (np.floor(tau).astype(int) == b)
            r = {"tau_bin_s": b, "n": int(mm.sum()), "n_windows": int(len(set(win_id[mm])))}
            for i, ax in enumerate(AXES):
                r["rms_e_" + ax] = float(math.sqrt((Ei[mm, i] ** 2).mean()))
                r["sd_" + ax] = float(math.sqrt(np.maximum(Pi[mm, i, i], 0).mean()))
                r["ratio_" + ax] = float((Ei[mm, i] ** 2).sum() / max(Pi[mm, i, i].sum(), 1e-300))
            r["nees9_mean"] = float(np.nanmean(nees_i[mm])) if np.isfinite(nees_i[mm]).any() else None
            rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(D, "arm_ol_tau.csv"), index=False, float_format="%.6g")
        m1 = use & (tau >= 1.0)
        if m1.any():
            S["ol_ratio_tau_ge_1s"] = {ax: float((Ei[m1, i] ** 2).sum() / max(Pi[m1, i, i].sum(), 1e-300)) for i, ax in enumerate(AXES)}
            S["ol_nees9_mean_tau_ge_1s"] = float(np.nanmean(nees_i[m1]))
            S["ol_n_windows"] = int(len(set(win_id[use])))

    # ---- motion_q_scan.csv (Q actually used)
    mq = path("motion_q_scan.csv")
    if mq:
        q = pd.read_csv(mq)
        cols = [c for c in q.columns if c.startswith("mean_q_") or c in ("acc_cap_fraction", "gyr_cap_fraction")]
        for nm, lo_, hi_ in (("scan1", 1, 1), ("stationary", 1, a.stat_last), ("transition", a.stat_last + 1, a.trans_last), ("motion", a.trans_last + 1, 10 ** 9)):
            mm = (q.scan_index >= lo_) & (q.scan_index <= hi_)
            if mm.any():
                S.setdefault("motion_q", {})[nm] = {c: float(q.loc[mm, c].median()) for c in cols}
                S["motion_q"][nm]["max_mean_q_gyr_max_axis"] = float(q.loc[mm, ["mean_q_gyr_x", "mean_q_gyr_y", "mean_q_gyr_z"]].to_numpy().max())
                S["motion_q"][nm]["max_mean_q_acc_max_axis"] = float(q.loc[mm, ["mean_q_acc_x", "mean_q_acc_y", "mean_q_acc_z"]].to_numpy().max())

    # ---- plane_fit_reasons.csv
    pr = os.path.join(D, "plane_fit_reasons.csv")
    if os.path.exists(pr):
        r = pd.read_csv(pr)
        ev = ["accept", "n_lt_3", "solver_fail", "eig12_small", "floor_reject", "non_finite", "threshold_fail", "denom_reject",
              "ceiling_reject", "info_reject", "retire", "subdivide", "disabled"]
        out = {}
        for pth in r.path.unique():
            rp_ = r[r.path == pth]
            for nm, lo_, hi_ in (("frame0", 0, 0), ("frames1_10", 1, 10), ("frames11_480", 11, a.stat_last), ("frames481_on", a.stat_last + 1, 10 ** 9)):
                mm = (rp_.frame >= lo_) & (rp_.frame <= hi_)
                if mm.any():
                    out["%s/%s" % (pth, nm)] = {c: int(rp_.loc[mm, c].sum()) for c in ev}
        S["plane_fit_reasons"] = out
    ptrace = os.path.join(D, "plane_fit_trace.csv")
    if os.path.exists(ptrace):
        tr = pd.read_csv(ptrace)
        S["plane_fit_trace_rows"] = int(len(tr))
        g = tr.groupby(["path", "frame", "reason"]).size().reset_index(name="rows")
        g.to_csv(os.path.join(D, "plane_fit_trace_counts.csv"), index=False)

    # ---- results_lio.txt ATE line (evo, the agent's official number)
    rl = path("results_lio.txt")
    if rl:
        import re
        txt = open(rl, errors="ignore").read()
        mt = re.search(r"ATE=([0-9.]+)m", txt) or re.search(r"ATE=([0-9.]+)", txt)
        mn = re.search(r"n=(\d+)", txt)
        if mt:
            S["evo_ate_m"] = float(mt.group(1))
        if mn:
            S["evo_n"] = int(mn.group(1))

    # ---- point_cov_stats.csv (R65, write-only per-insertion statistics of trace(pos_cov))
    pcv = {"pc_mean_trace_pos_cov": np.full(len(scans), np.nan), "pc_max_trace_pos_cov": np.full(len(scans), np.nan)}
    pcp = path("point_cov_stats.csv")
    if pcp:
        try:
            pc = pd.read_csv(pcp).drop_duplicates("frame", keep="last").set_index("frame")
            pcv["pc_mean_trace_pos_cov"] = pc["mean_trace_pos_cov"].reindex(scans).to_numpy(float)
            pcv["pc_max_trace_pos_cov"] = pc["max_trace_pos_cov"].reindex(scans).to_numpy(float)
            for nm, mm in (("stationary", stat), ("transition", seg == "transition"), ("motion", seg == "motion")):
                if mm.any() and np.isfinite(pcv["pc_mean_trace_pos_cov"][mm]).any():
                    S.setdefault("point_cov_stats", {})[nm] = {"median_mean_trace": float(np.nanmedian(pcv["pc_mean_trace_pos_cov"][mm])),
                                                                "median_max_trace": float(np.nanmedian(pcv["pc_max_trace_pos_cov"][mm]))}
        except Exception as ex:
            S["warnings"].append("point_cov_stats.csv unreadable: %s" % ex)

    # ---- per-scan table
    tab = pd.DataFrame({"scan": scans, "t": ["%.7f" % x for x in t], "seg": seg, "resid": rc, "iters": ci,
                        "ep_x_mm": ep[:, 0], "ep_y_mm": ep[:, 1], "ep_z_mm": ep[:, 2], "ep_norm_mm": ep_norm,
                        "sd_p_mm": np.sqrt(np.maximum(Pl[:, 3, 3] + Pl[:, 4, 4] + Pl[:, 5, 5], 0)) * 1000.0,
                        "eR_norm_mrad": np.linalg.norm(El[:, 0:3], axis=1) * 1000.0,
                        "eR_x_mrad": El[:, 0] * 1000.0, "eR_y_mrad": El[:, 1] * 1000.0, "eR_z_mrad": El[:, 2] * 1000.0,
                        "sd_R_mrad": np.sqrt(np.maximum(Pl[:, 0, 0] + Pl[:, 1, 1] + Pl[:, 2, 2], 0)) * 1000.0,
                        "ev_norm_mm_s": np.linalg.norm(El[:, 6:9], axis=1) * 1000.0,
                        "nees9_lio": nees_l, "nees9_imu": nees_i, "gt_err_mm": gt_err})
    for k, v in fsv.items():
        tab[k] = v
    for k, v in pcv.items():
        tab[k] = v
    tab.to_csv(os.path.join(D, "arm_scan.csv"), index=False, float_format="%.5g")
    json.dump(jsonable(S), open(os.path.join(D, "arm_summary.json"), "w"), indent=1)
    print("reduced %s: %d scans (%d stationary), gt_status=%s" % (S["arm"], len(scans), int(stat.sum()), S["gt_status"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
