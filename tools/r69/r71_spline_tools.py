#!/usr/bin/env python3
"""R71 spline / coupled-estimator analysis tools (Python only, offline). Closes the F-370 questions.

    python3 r71_spline_tools.py knots --knots joint_knot_all_scans[_compact].csv[.gz] --out OUTDIR [--stationary-end 480]
    python3 r71_spline_tools.py pairs --scan-table arm_scan_all.csv.gz --pairs "tail_arm:iekf_arm,..." --out OUTDIR

knots  : per knot index and scan window: median correction norms (dp, dtheta, dv, dbg, dba) of the LAST iteration of each scan, median absolute knot velocity and
         biases (needs the R71 `after_*` columns), and for the stationary window the Spearman trend of |dv|, |dbg|, |dba| and of |v|, |bg|, |ba| against scan
         (negative = trending toward zero), plus the ratio median(last 80 stationary scans)/median(scans 11-50).
pairs  : coupled arm minus iekf arm on the same bag, per scan window: median |difference| of position error, rotation error, velocity error and of the reported
         sd (from arm_scan_all.csv.gz columns ep_*, eR_*, ev_norm_mm_s, sd_p_mm, sd_R_mrad), and the error-over-sd ratio of each arm in the window.
"""
import argparse, os
import numpy as np
import pandas as pd

WINDOWS = [(1, 10), (11, 50), (51, 150), (151, 300), (301, 480), (485, 600), (601, 1200), (1201, 2400), (2401, 10 ** 6)]


def spearman(x, y):
    if len(x) < 5:
        return np.nan
    rx = pd.Series(x).rank().to_numpy(); ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else np.nan


def run_knots(a):
    os.makedirs(a.out, exist_ok=True)
    d = pd.read_csv(a.knots)
    if "iteration" in d.columns:
        d = d.sort_values(["scan_id", "iteration"]).groupby(["scan_id", "knot_index"], as_index=False).tail(1)
    norms = ["dp_norm", "dtheta_norm", "dv_norm", "dbg_norm", "dba_norm", "adjacent_bg_difference", "adjacent_ba_difference", "after_velocity_distance"]
    if "after_v_x" in d.columns:
        d["abs_v"] = np.sqrt(d.after_v_x ** 2 + d.after_v_y ** 2 + d.after_v_z ** 2)
        d["abs_bg"] = np.sqrt(d.after_bg_x ** 2 + d.after_bg_y ** 2 + d.after_bg_z ** 2)
        d["abs_ba"] = np.sqrt(d.after_ba_x ** 2 + d.after_ba_y ** 2 + d.after_ba_z ** 2)
        norms += ["abs_v", "abs_bg", "abs_ba"]
    rows = []
    for k, g in d.groupby("knot_index"):
        for lo, hi in WINDOWS:
            w = g[(g.scan_id >= lo) & (g.scan_id <= hi)]
            if len(w):
                r = dict(knot_index=int(k), scan_lo=lo, scan_hi=min(hi, int(g.scan_id.max())), n_scans=int(w.scan_id.nunique()))
                for n in norms:
                    if n in w.columns:
                        r["median_" + n] = float(w[n].median())
                rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "knot_windows.csv"), index=False, float_format="%.6g")
    tr = []
    for k, g in d.groupby("knot_index"):
        st = g[(g.scan_id >= 11) & (g.scan_id <= a.stationary_end)]
        if len(st) < 20:
            continue
        early = st[st.scan_id <= 50]; late = st[st.scan_id > a.stationary_end - 80]
        for n in norms:
            if n in st.columns:
                e = float(early[n].median()); l = float(late[n].median())
                tr.append(dict(knot_index=int(k), quantity=n, spearman_vs_scan=spearman(st.scan_id.to_numpy(), st[n].to_numpy()),
                               median_scans_11_50=e, median_last80=l, ratio_last80_over_early=(l / e) if e > 0 else np.nan))
    pd.DataFrame(tr).to_csv(os.path.join(a.out, "knot_trend_stationary.csv"), index=False, float_format="%.6g")
    print("knot rows", len(d), "knots", d.knot_index.nunique(), "absolute-state columns:", "after_v_x" in d.columns)


def run_pairs(a):
    os.makedirs(a.out, exist_ok=True)
    S = pd.read_csv(a.scan_table)
    rows = []
    for pr in a.pairs.split(","):
        ca, ib = pr.split(":")
        A = S[S.arm == ca].set_index("scan"); B = S[S.arm == ib].set_index("scan")
        idx = A.index.intersection(B.index)
        if len(idx) == 0:
            rows.append(dict(coupled=ca, iekf=ib, window="NO_COMMON_SCANS")); continue
        A = A.loc[idx]; B = B.loc[idx]
        for lo, hi in WINDOWS:
            m = (idx >= lo) & (idx <= hi)
            if m.sum() < 3:
                continue
            r = dict(coupled=ca, iekf=ib, scan_lo=lo, scan_hi=int(min(hi, idx.max())), n=int(m.sum()))
            for nm, cols in (("ep_mm", ["ep_x_mm", "ep_y_mm", "ep_z_mm"]), ("eR_mrad", ["eR_x_mrad", "eR_y_mrad", "eR_z_mrad"])):
                dv = (A[cols].to_numpy()[m] - B[cols].to_numpy()[m])
                r["median_abs_diff_" + nm] = float(np.median(np.linalg.norm(dv, axis=1)))
                r["median_coupled_" + nm] = float(np.median(np.linalg.norm(A[cols].to_numpy()[m], axis=1)))
                r["median_iekf_" + nm] = float(np.median(np.linalg.norm(B[cols].to_numpy()[m], axis=1)))
            for col in ("ev_norm_mm_s", "sd_p_mm", "sd_R_mrad", "nees9_lio"):
                if col in A.columns:
                    r["median_coupled_" + col] = float(A[col][m].median()); r["median_iekf_" + col] = float(B[col][m].median())
            r["coupled_err_over_sd_pos"] = float(np.median(np.linalg.norm(A[["ep_x_mm", "ep_y_mm", "ep_z_mm"]].to_numpy()[m], axis=1) / A["sd_p_mm"].to_numpy()[m]))
            r["iekf_err_over_sd_pos"] = float(np.median(np.linalg.norm(B[["ep_x_mm", "ep_y_mm", "ep_z_mm"]].to_numpy()[m], axis=1) / B["sd_p_mm"].to_numpy()[m]))
            rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "coupled_vs_iekf_windows.csv"), index=False, float_format="%.6g")
    print("pair rows", len(rows))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    k = sp.add_parser("knots"); k.add_argument("--knots", required=True); k.add_argument("--out", required=True); k.add_argument("--stationary-end", type=int, default=480)
    p = sp.add_parser("pairs"); p.add_argument("--scan-table", required=True); p.add_argument("--pairs", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args()
    run_knots(a) if a.cmd == "knots" else run_pairs(a)


if __name__ == "__main__":
    main()
