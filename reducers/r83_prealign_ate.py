#!/usr/bin/env python3
"""r83_prealign_ate: ATE of a collapsing arm on the scans BEFORE it collapses, with a fresh alignment on those scans only (the scans before d include the drift up to --diverge-mm).

Why: for an arm that collapses, the pipeline's whole-run alignment is dragged by the collapse, so the ground-truth error it reports on the early scans is large even though
the node had not moved (R82A: sbs_02 edge 0.20 fixed-Q reads 500+ mm GT error from its first joined scan, scan 273, while its own drift from the start reference is 3 mm).
The addendum's original rule (fresh Umeyama on the scans before the collapse) needs the raw positions, which state_trace.csv has.

The collapse scan d is taken from a CONTROL run of the same bag and same Q mode that did not collapse (given with --control-trace): d is the first scan at which the
arm's position differs from the control's by more than --diverge-mm (default 500) in the node's own world frame. Both runs start from the same stationary calibration, so
their frames agree; the tool prints the difference at scan 1 and 30 as a check (it should be a few mm) and stops with an error if it is above 50 mm.

  python3 r83_prealign_ate.py --arm A --trace state_trace.csv.gz --control-trace control_state_trace.csv.gz --gt gt.txt --lever "x y z"
          [--max-gt-gap 0.15] [--diverge-mm 500] [--min-joined 50] --ate-mm X --max-mm Y --out row.csv

Columns: arm, d_scan (blank if the arm never leaves the control by more than --diverge-mm), n_scans_before_d, n_joined_before_d, prealign_ate_mm, prealign_max_mm
(RMS and maximum of the 3D position error after a position-only Umeyama (rotation allowed, lever arm applied as in the pipeline) on the joined scans before d),
unfiltered_ate_mm, unfiltered_max_mm, diff_at_scan1_mm, diff_at_scan30_mm, status ('ok', 'never_diverges', 'too_few_joined_scans_before_d')."""
import argparse, sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot
import r83_cov_consistency as C

def run(arm, tr, ctl, gt, lever, max_gap, thresh, min_joined, ate, mx):
    r = dict(arm=arm, unfiltered_ate_mm=ate, unfiltered_max_mm=mx)
    a = tr.set_index("scan_id"); c = ctl.set_index("scan_id")
    common = a.index.intersection(c.index)
    dp = 1000 * np.linalg.norm(a.loc[common, ["px", "py", "pz"]].to_numpy(float) - c.loc[common, ["px", "py", "pz"]].to_numpy(float), axis=1)
    dd = pd.Series(dp, index=common)
    r["diff_at_scan1_mm"] = float(dd.iloc[0]); r["diff_at_scan30_mm"] = float(dd.loc[30]) if 30 in dd.index else np.nan
    if dd.iloc[0] > 50:
        raise SystemExit("the arm and the control differ by %.1f mm at scan 1: not the same start frame, choose another control" % dd.iloc[0])
    over = dd[dd > thresh]
    if not len(over):
        r.update(d_scan=np.nan, n_scans_before_d=len(tr), n_joined_before_d=np.nan, prealign_ate_mm=np.nan, prealign_max_mm=np.nan, status="never_diverges")
        return r
    d = int(over.index[0]); r["d_scan"] = d
    pre = tr[tr.scan_id < d].reset_index(drop=True)
    r["n_scans_before_d"] = len(pre)
    gta = np.loadtxt(gt, comments="#"); tg = gta[:, 0]
    ts = pre["t_abs"].to_numpy(float)
    inside = (ts >= tg[0]) & (ts <= tg[-1])
    j = np.clip(np.searchsorted(tg, ts), 1, len(tg) - 1)
    ok = inside & ((tg[j] - tg[j - 1]) <= max_gap if max_gap > 0 else True)
    r["n_joined_before_d"] = int(ok.sum())
    if ok.sum() < min_joined:
        r.update(prealign_ate_mm=np.nan, prealign_max_mm=np.nan, status="too_few_joined_scans_before_d")
        return r
    Re = Rot.from_quat(pre[["qx", "qy", "qz", "qw"]].to_numpy(float))
    pe = pre[["px", "py", "pz"]].to_numpy(float) + Re.apply(np.tile(lever, (len(pre), 1)))
    pg = np.stack([np.interp(ts, tg, gta[:, k]) for k in (1, 2, 3)], 1)
    Ra, ta = C.umeyama(pe[ok], pg[ok])
    e = pg[ok] - (pe[ok] @ Ra.T + ta)
    e3 = 1000 * np.linalg.norm(e, axis=1)
    r["prealign_ate_mm"] = float(np.sqrt(np.mean(e3 ** 2))); r["prealign_max_mm"] = float(e3.max()); r["status"] = "ok"
    return r

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--arm", "--trace", "--control-trace", "--gt", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--lever", default="0 0 0"); ap.add_argument("--max-gt-gap", type=float, default=0.15)
    ap.add_argument("--diverge-mm", type=float, default=500.0); ap.add_argument("--min-joined", type=int, default=50)
    ap.add_argument("--ate-mm", type=float, required=True); ap.add_argument("--max-mm", type=float, required=True)
    a = ap.parse_args()
    r = run(a.arm, C.load_trace(a.trace), C.load_trace(a.control_trace), a.gt, np.array([float(v) for v in a.lever.split()]), a.max_gt_gap, a.diverge_mm, a.min_joined, a.ate_mm, a.max_mm)
    pd.DataFrame([r]).to_csv(a.out, index=False)
    print(pd.DataFrame([r]).T.to_string(header=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
