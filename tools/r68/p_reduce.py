#!/usr/bin/env python3
"""p_reduce.py -- reduced form of the per-scan full state covariance (planning agent's, R68 design).

Input  (arm directory; all written by the existing build):
    state_trace.csv        2 rows per scan (post_imu = prior, post_lio = posterior), raw state + upper triangle of P (18 dims)
    state_reference.txt    stationary reference R, p, v (constant) -- errors are formed only while the sensor is stationary
    imu_cov_growth.csv     optional: diag(P) start / end of the scan's IMU propagation and diag(Qacc)   (process-noise share)
    lio_gt_matched.csv     optional (--gt): est_p*/gt_p* matched pairs -> whole-run position error vs GT after one rigid alignment
    arm.json               optional (seq -> per-bag windows via --params bag_params.json)
Output (--out, default <arm_dir>):
    p_scan.csv.gz          one row per kept scan (thinned; see --full-until/--step1/--step2), ~45 numeric columns, 4 significant digits
    p_window.json          per-window medians / 90th percentiles / exceedance fractions computed on the FULL-resolution series
Units: sd in mm (position), mrad (rotation), mm/s (velocity), mrad/s (gyro bias), mm/s^2 (accel bias), mm/s^2 (gravity).
Overconfidence: nees* = e' P^-1 e ; expected value = dof (3 per block, 9 for R,p,v). oc9 = nees9 / 9  (> 1 overconfident, < 1 underconfident).
Only numpy + pandas. Numerics-inert (reads files only).
"""
import argparse, gzip, json, math, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import quat_to_R, log_so3, read_reference, umeyama, jsonable, chi2_ppf  # noqa: E402

BLOCKS = {"R": slice(0, 3), "p": slice(3, 6), "v": slice(6, 9), "bg": slice(9, 12), "ba": slice(12, 15), "g": slice(15, 18)}
UNIT = {"R": 1e3, "p": 1e3, "v": 1e3, "bg": 1e3, "ba": 1e3, "g": 1e3}   # rad->mrad, m->mm, m/s->mm/s, rad/s->mrad/s, m/s^2->mm/s^2
SCHEMA = "p_reduce_v1"


def build_P(df, n=18):
    P = np.zeros((len(df), n, n))
    for c in df.columns:
        if c.startswith("P_"):
            _, i, j = c.split("_")
            i, j = int(i), int(j)
            if i < n and j < n:
                v = df[c].to_numpy(float)
                v = np.where(np.isfinite(v), v, 0.0)
                P[:, i, j] = v
                P[:, j, i] = v
    return P


def rot_arrays(rows):
    q = rows[["qw", "qx", "qy", "qz"]].to_numpy(float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def quad(P, e):
    """e' P^-1 e with an eigenvalue floor (P is PSD up to round-off). Returns (value, min_eig, max_eig)."""
    w, V = np.linalg.eigh(0.5 * (P + P.T))
    top = max(w[-1], 1e-300)
    wf = np.maximum(w, 1e-12 * top)
    c = V.T @ e
    return float(np.sum(c * c / wf)), float(w[0]), float(w[-1])


def logdet(P):
    w = np.linalg.eigvalsh(0.5 * (P + P.T))
    top = max(w[-1], 1e-300)
    return float(np.sum(np.log(np.maximum(w, 1e-12 * top))))


def xcorr(C, a, b):
    return float(np.max(np.abs(C[a, b])))


def corr_matrix(P):
    d = np.sqrt(np.maximum(np.diag(P), 1e-300))
    return P / np.outer(d, d)


def principal_z(Pb, e):
    """z of e along the eigen-directions of a 3x3 block, ordered largest -> smallest eigenvalue."""
    w, V = np.linalg.eigh(0.5 * (Pb + Pb.T))
    w = np.maximum(w, 1e-12 * max(w[-1], 1e-300))
    z = (V.T @ e) / np.sqrt(w)
    return z[::-1], np.sqrt(w[::-1])


def load_windows(a, meta):
    stat, trans = a.stat_last, a.trans_last
    if a.params and os.path.exists(a.params):
        pj = json.load(open(a.params)).get(meta.get("seq", "eee_01"))
        if pj and meta.get("seq", "eee_01") != "eee_01":
            stat, trans = int(pj["stat_last"]), int(pj["trans_last"])
    return stat, trans, {"w0_start": (1, 10), "w1_settle": (11, 100), "w2_stationary": (101, stat),
                         "w3_transition": (stat + 1, trans), "w4_motion_early": (trans + 1, trans + 1000),
                         "w5_motion_late": (trans + 1001, 10 ** 9)}


def reduce_arm(arm_dir, a):
    meta = {}
    mp = os.path.join(arm_dir, "arm.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
    stat, trans, windows = load_windows(a, meta)
    S = {"schema": SCHEMA, "arm": meta.get("arm", os.path.basename(os.path.abspath(arm_dir))), "warnings": [],
         "windows_def": windows, "stat_last": stat, "trans_last": trans}
    tp, rp = os.path.join(arm_dir, "state_trace.csv"), os.path.join(arm_dir, "state_reference.txt")
    if not os.path.exists(tp) and os.path.exists(tp + ".gz"):
        tp = tp + ".gz"
    if not (os.path.exists(tp) and os.path.exists(rp)):
        S["warnings"].append("state_trace.csv or state_reference.txt missing")
        return S, None
    df = pd.read_csv(tp)
    Rref, pref, vref, _ = read_reference(rp)
    dim = int(df["dim"].iloc[0]) if "dim" in df.columns else 18
    if dim < 9:
        S["warnings"].append("state dimension %d < 9" % dim)
        return S, None
    S["dim"] = dim
    pre = df[df.phase == "post_imu"].drop_duplicates("scan_id", keep="last").set_index("scan_id")
    post = df[df.phase == "post_lio"].drop_duplicates("scan_id", keep="last").set_index("scan_id")
    scans = np.array(sorted(set(pre.index) & set(post.index)))
    if len(scans) == 0:
        S["warnings"].append("no scan with both post_imu and post_lio rows")
        return S, None
    pre, post = pre.loc[scans], post.loc[scans]
    n = min(dim, 18)
    Ppre, Ppost = build_P(pre, n), build_P(post, n)
    Rpre, Rpost = rot_arrays(pre), rot_arrays(post)
    pp_pre, pp_post = pre[["px", "py", "pz"]].to_numpy(float), post[["px", "py", "pz"]].to_numpy(float)
    vv_pre, vv_post = pre[["vx", "vy", "vz"]].to_numpy(float), post[["vx", "vy", "vz"]].to_numpy(float)
    N = len(scans)

    # optional Q growth share per scan
    qshare = None
    gp = os.path.join(arm_dir, "imu_cov_growth.csv")
    if os.path.exists(gp):
        g = pd.read_csv(gp)
        g = g.drop_duplicates("scan_index", keep="last").set_index("scan_index")
        # imu_cov_growth scan_index is 0-based or 1-based depending on writer; align by row count if lengths match, else by index
        qshare = g
    # optional GT position error (whole run, one rigid alignment, like ATE)
    gt_err = None
    if a.gt:
        gtp = os.path.join(arm_dir, "lio_gt_matched.csv")
        if not os.path.exists(gtp):
            gtp = os.path.join(arm_dir, "lio_gt_matched_compact.csv.gz")
        if os.path.exists(gtp):
            m = pd.read_csv(gtp)
            est = m[["est_px", "est_py", "est_pz"]].to_numpy(float)
            gt = m[["gt_px", "gt_py", "gt_pz"]].to_numpy(float)
            Ra, ta = umeyama(est, gt)
            err = (est @ Ra.T + ta) - gt
            tt = m["t_abs"].to_numpy(float)
            order = np.argsort(tt)
            tt, err = tt[order], err[order]
            tq = post["t_abs"].to_numpy(float)
            idx = np.clip(np.searchsorted(tt, tq), 1, len(tt) - 1)
            idx = np.where(np.abs(tt[idx - 1] - tq) < np.abs(tt[idx] - tq), idx - 1, idx)
            ok = np.abs(tt[idx] - tq) < 0.06
            gt_err = np.where(ok[:, None], err[idx], np.nan)
            gt_R = Ra
        else:
            S["warnings"].append("--gt requested but lio_gt_matched not found")

    chi_lo9, chi_hi9 = chi2_ppf(0.025, 9), chi2_ppf(0.975, 9)
    chi_hi3 = chi2_ppf(0.975, 3)
    cols = {}

    def put(name, i, val):
        if name not in cols:
            cols[name] = np.full(N, np.nan)
        cols[name][i] = val

    for i in range(N):
        s = scans[i]
        A, B = Ppre[i], Ppost[i]
        for nm, sl in BLOCKS.items():
            if sl.stop > n:
                continue
            put("sd_%s_post" % nm, i, math.sqrt(max(np.trace(B[sl, sl]), 0.0)) * UNIT[nm])
            if nm in ("R", "p", "v"):
                put("sd_%s_pre" % nm, i, math.sqrt(max(np.trace(A[sl, sl]), 0.0)) * UNIT[nm])
                put("sdratio_%s" % nm, i, math.sqrt(max(np.trace(B[sl, sl]), 1e-300) / max(np.trace(A[sl, sl]), 1e-300)))
        for nm in ("R", "p"):
            sl = BLOCKS[nm]
            wa = np.linalg.eigvalsh(0.5 * (A[sl, sl] + A[sl, sl].T)); wb = np.linalg.eigvalsh(0.5 * (B[sl, sl] + B[sl, sl].T))
            put("eigsd_%s_min_post" % nm, i, math.sqrt(max(wb[0], 0.0)) * UNIT[nm]); put("eigsd_%s_max_post" % nm, i, math.sqrt(max(wb[-1], 0.0)) * UNIT[nm])
            put("eigsd_%s_max_pre" % nm, i, math.sqrt(max(wa[-1], 0.0)) * UNIT[nm])
        w18 = np.linalg.eigvalsh(0.5 * (B + B.T))
        w9 = np.linalg.eigvalsh(0.5 * (B[:9, :9] + B[:9, :9].T))
        put("mineig_post", i, float(w18[0])); put("cond9_post_log10", i, math.log10(max(w9[-1], 1e-300) / max(w9[0], 1e-300)) if w9[0] > 0 else np.inf)
        put("cond_all_post_log10", i, math.log10(max(w18[-1], 1e-300) / max(w18[0], 1e-300)) if w18[0] > 0 else np.inf)
        C = corr_matrix(B)
        core = np.abs(C[:9, :9] - np.eye(9))
        put("maxcorr9_post", i, float(core.max()))
        pairs = {"Rp": ("R", "p"), "pv": ("p", "v"), "Rbg": ("R", "bg"), "vba": ("v", "ba"), "Rg": ("R", "g"), "bag": ("ba", "g")}
        for k, (x, y) in pairs.items():
            if BLOCKS[x].stop <= n and BLOCKS[y].stop <= n:
                put("corr_%s_post" % k, i, xcorr(C, BLOCKS[x], BLOCKS[y]))
        put("infogain9_nats", i, 0.5 * (logdet(A[:9, :9]) - logdet(B[:9, :9])))
        # update step measured against the prior covariance (closed-loop; needs no reference)
        dR = log_so3(Rpre[i].T @ Rpost[i]); dp = pp_post[i] - pp_pre[i]; dv = vv_post[i] - vv_pre[i]
        d9 = np.concatenate([dR, dp, dv])
        put("jump9_mahal", i, quad(A[:9, :9], d9)[0])
        put("jump_R_mahal", i, quad(A[0:3, 0:3], dR)[0]); put("jump_p_mahal", i, quad(A[3:6, 3:6], dp)[0]); put("jump_v_mahal", i, quad(A[6:9, 6:9], dv)[0])
        put("jump_p_mm", i, float(np.linalg.norm(dp)) * 1e3)
        put("bg_norm_mrad_s", i, float(np.linalg.norm(post.iloc[i][["bgx", "bgy", "bgz"]].to_numpy(float))) * 1e3) if "bgx" in post.columns else None
        put("ba_norm_mm_s2", i, float(np.linalg.norm(post.iloc[i][["bax", "bay", "baz"]].to_numpy(float))) * 1e3) if "bax" in post.columns else None
        put("g_norm", i, float(np.linalg.norm(post.iloc[i][["gx", "gy", "gz"]].to_numpy(float)))) if "gx" in post.columns else None
        # reference errors: stationary only
        if s <= stat:
            eR_b = log_so3(Rref.T @ Rpost[i]); ep_b = pp_post[i] - pref; ev_b = vv_post[i] - vref
            eR_a = log_so3(Rref.T @ Rpre[i]); ep_a = pp_pre[i] - pref; ev_a = vv_pre[i] - vref
            e_post = np.concatenate([eR_b, ep_b, ev_b]); e_pre = np.concatenate([eR_a, ep_a, ev_a])
            put("eR_mrad", i, float(np.linalg.norm(eR_b)) * 1e3); put("ep_mm", i, float(np.linalg.norm(ep_b)) * 1e3); put("ev_mm_s", i, float(np.linalg.norm(ev_b)) * 1e3)
            n9b, _, _ = quad(B[:9, :9], e_post); n9a, _, _ = quad(A[:9, :9], e_pre)
            put("nees9_post", i, n9b); put("nees9_pre", i, n9a); put("oc9_post", i, n9b / 9.0)
            put("nees_R_post", i, quad(B[0:3, 0:3], eR_b)[0]); put("nees_p_post", i, quad(B[3:6, 3:6], ep_b)[0]); put("nees_v_post", i, quad(B[6:9, 6:9], ev_b)[0])
            sd9 = np.sqrt(np.maximum(np.diag(B)[:9], 1e-300))
            put("zmax_abs9_post", i, float(np.max(np.abs(e_post / sd9))))
            for nm, sl, e in (("R", BLOCKS["R"], eR_b), ("p", BLOCKS["p"], ep_b)):
                z, sds = principal_z(B[sl, sl], e)
                for k in range(3):
                    put("z%s%d_post" % (nm, k + 1), i, float(z[k]))
        if gt_err is not None and np.all(np.isfinite(gt_err[i])):
            e3 = gt_err[i]
            put("gtp_err_mm", i, float(np.linalg.norm(e3)) * 1e3)
            Pw = gt_R @ B[3:6, 3:6] @ gt_R.T      # position block rotated with the alignment (both are world-frame)
            put("gtp_nees3_post", i, quad(Pw, e3)[0])
    if qshare is not None:
        # share of the scan's covariance growth caused by process noise: sum diag(Qacc) / sum diag(P_end), per block.
        # Aligned to scans by nearest t_abs (scan_index numbering differs between the two writers).
        g = qshare.reset_index()
        tg = g["t_abs"].to_numpy(float)
        order = np.argsort(tg)
        g, tg = g.iloc[order].reset_index(drop=True), tg[order]
        tq = post["t_abs"].to_numpy(float)
        idx = np.clip(np.searchsorted(tg, tq), 1, len(tg) - 1)
        idx = np.where(np.abs(tg[idx - 1] - tq) < np.abs(tg[idx] - tq), idx - 1, idx)
        okt = np.abs(tg[idx] - tq) < 0.02
        if not okt.any():
            S["warnings"].append("imu_cov_growth.csv t_abs does not match state_trace t_abs; qshare skipped")
        else:
            for nm in ("R", "p", "v"):
                sl = BLOCKS[nm]
                qa = ["Qacc_%d" % k for k in range(sl.start, sl.stop)]
                pe = ["Pend_%d" % k for k in range(sl.start, sl.stop)]
                if all(c in g.columns for c in qa + pe):
                    r = (g[qa].sum(axis=1) / g[pe].sum(axis=1)).to_numpy(float)[idx]
                    cols["qshare_%s" % nm] = np.where(okt, r, np.nan)
                else:
                    S["warnings"].append("imu_cov_growth.csv columns not recognised (expected Qacc_<i>, Pend_<i>)")
                    break

    T = pd.DataFrame({"scan": scans, "t_abs": post["t_abs"].to_numpy(float)})
    for k, v in cols.items():
        T[k] = v
    T["n_residuals"] = post["residual_count"].to_numpy(float) if "residual_count" in post.columns else np.nan
    T["iters"] = post["completed_iterations"].to_numpy(float) if "completed_iterations" in post.columns else np.nan

    # ---- window summaries on the full-resolution series
    W = {}
    summary_cols = [c for c in T.columns if c not in ("scan", "t_abs")]
    p90_cols = {"nees9_post", "nees9_pre", "nees_p_post", "nees_R_post", "nees_v_post", "jump9_mahal", "jump_p_mahal", "jump_R_mahal",
                "jump_v_mahal", "zmax_abs9_post", "gtp_nees3_post"}
    for wn, (lo, hi) in windows.items():
        m = (T.scan >= lo) & (T.scan <= hi)
        x = T[m]
        W[wn] = {"scans": [int(lo), int(min(hi, T.scan.max()))], "n": int(len(x))}
        if len(x) == 0:
            continue
        for c in summary_cols:
            v = x[c].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            W[wn][c] = {"med": float(np.median(v))}
            if c in p90_cols:
                W[wn][c]["p90"] = float(np.percentile(v, 90))
        for c, hi_, dof in (("nees9_post", chi_hi9, 9), ("nees9_pre", chi_hi9, 9), ("jump9_mahal", chi_hi9, 9), ("nees_p_post", chi_hi3, 3), ("nees_R_post", chi_hi3, 3)):
            if c in x.columns:
                v = x[c].to_numpy(float); v = v[np.isfinite(v)]
                if len(v):
                    W[wn][c]["frac_above_chi2_95"] = float(np.mean(v > hi_))
    S["windows"] = W
    S["n_scans"] = int(N)
    S["expected"] = {"nees9": 9, "nees_block": 3, "chi2_95_9dof": chi_hi9, "chi2_95_3dof": chi_hi3,
                     "jump9_mahal_under_consistent_update": "9 - tr(P_pre^-1 P_post)  (< 9; a value far above 9 means the update step is larger than the prior allows)"}

    # ---- thinning
    keep = (T.scan <= a.full_until) | ((T.scan <= a.until2) & (T.scan % a.step1 == 0)) | (T.scan % a.step2 == 0)
    return S, T[keep].reset_index(drop=True)


def write_outputs(S, T, out):
    os.makedirs(out, exist_ok=True)
    json.dump(jsonable(S), open(os.path.join(out, "p_window.json"), "w"), indent=1)
    if T is not None:
        num = T.select_dtypes("number").columns
        with gzip.open(os.path.join(out, "p_scan.csv.gz"), "wt") as f:
            T.to_csv(f, index=False, float_format="%.4g", columns=list(T.columns))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stat-last", type=int, default=480)
    ap.add_argument("--trans-last", type=int, default=540)
    ap.add_argument("--params", default=None, help="bag_params.json from bag_stats.py (per-bag windows)")
    ap.add_argument("--gt", action="store_true", help="add whole-run GT position error / NEES3 from lio_gt_matched")
    ap.add_argument("--full-until", type=int, default=60, help="keep every scan up to this one")
    ap.add_argument("--until2", type=int, default=700)
    ap.add_argument("--step1", type=int, default=5, help="keep every step1-th scan up to --until2")
    ap.add_argument("--step2", type=int, default=20, help="keep every step2-th scan afterwards")
    a = ap.parse_args()
    S, T = reduce_arm(a.arm_dir, a)
    write_outputs(S, T, a.out or a.arm_dir)
    return 0 if T is not None else 1


if __name__ == "__main__":
    sys.exit(main())
