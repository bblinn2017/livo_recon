#!/usr/bin/env python3
"""r83_gt_gaps: where the ground truth has gaps, and what the pipeline's GT error does on the first scans after a gap.

Why: in R82A the single-scan GT error spikes of 340 mm (sbs_02, scan 1012) and 195 mm (sbs_03, scan 2526) both sit on the FIRST GT-joined scan after a run of scans without a GT join.
This lists every GT gap of a bag (consecutive GT samples further apart than --gap-s) with the pipeline's gt_err_mm on the scans around it, and dumps the raw GT samples around
chosen scans so the sample that causes a spike can be seen.

  python3 r83_gt_gaps.py --bag B --gt gt.txt --arm-scan arm_scan.csv[.gz] [--gap-s 0.15] [--raw-scans 1012 --raw-half-window-s 1.0]
                         --out-gaps gaps.csv [--out-raw raw.csv]

arm_scan needs a scan column (scan, scan_id or scan_index), a time column (t_abs or t, absolute seconds as in the GT file) and gt_err_mm.
gaps.csv columns: bag, gap_start_t, gap_end_t, gap_s, n_scans_in_gap, last_scan_before, err_last_before_mm, first_scan_after, err_after_1_mm, err_after_2_mm, err_after_3_mm,
  gt_jump_over_gap_mm (distance between the GT samples at the two ends of the gap), node_move_over_gap_mm: left blank (not available here).
raw.csv columns: bag, scan, scan_t, gt_t, dt_to_scan_s, x, y, z, step_from_previous_sample_mm."""
import argparse, os, sys
import numpy as np
import pandas as pd

def load_scan(path):
    d = pd.read_csv(path)
    sc = next((c for c in ("scan", "scan_id", "scan_index") if c in d.columns), None)
    tc = next((c for c in ("t_abs", "t") if c in d.columns), None)
    if sc is None or tc is None or "gt_err_mm" not in d.columns:
        raise ValueError("arm_scan needs a scan column, a t_abs/t column and gt_err_mm")
    return pd.DataFrame(dict(scan=d[sc].to_numpy(int), t=d[tc].to_numpy(float), err=d["gt_err_mm"].to_numpy(float))).sort_values("scan").reset_index(drop=True)

def gaps(bag, gt, sc, gap_s):
    tg = gt[:, 0]
    idx = np.nonzero(np.diff(tg) > gap_s)[0]
    rows = []
    for i in idx:
        t0, t1 = tg[i], tg[i + 1]
        inside = sc[(sc.t > t0) & (sc.t < t1)]
        after = sc[sc.t >= t1].head(3)
        before = sc[sc.t <= t0].tail(1)
        r = dict(bag=bag, gap_start_t=t0, gap_end_t=t1, gap_s=t1 - t0, n_scans_in_gap=len(inside),
                 last_scan_before=int(before.scan.iloc[0]) if len(before) else np.nan, err_last_before_mm=float(before.err.iloc[0]) if len(before) else np.nan,
                 first_scan_after=int(after.scan.iloc[0]) if len(after) else np.nan,
                 gt_jump_over_gap_mm=1000 * float(np.linalg.norm(gt[i + 1, 1:4] - gt[i, 1:4])))
        for k in range(3):
            r["err_after_%d_mm" % (k + 1)] = float(after.err.iloc[k]) if len(after) > k else np.nan
        rows.append(r)
    return pd.DataFrame(rows)

def raw(bag, gt, sc, scans, half):
    rows = []
    tg = gt[:, 0]
    for n in scans:
        st = sc.t[sc.scan == n]
        if not len(st):
            continue
        st = float(st.iloc[0])
        for i in np.nonzero(np.abs(tg - st) <= half)[0]:
            step = 1000 * float(np.linalg.norm(gt[i, 1:4] - gt[i - 1, 1:4])) if i > 0 else np.nan
            rows.append(dict(bag=bag, scan=n, scan_t=st, gt_t=tg[i], dt_to_scan_s=tg[i] - st, x=gt[i, 1], y=gt[i, 2], z=gt[i, 3], step_from_previous_sample_mm=step))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--bag", "--gt", "--arm-scan", "--out-gaps"):
        ap.add_argument(k, required=True)
    ap.add_argument("--gap-s", type=float, default=0.15); ap.add_argument("--raw-scans", nargs="*", type=int, default=[])
    ap.add_argument("--raw-half-window-s", type=float, default=1.0); ap.add_argument("--out-raw")
    a = ap.parse_args()
    gt = np.loadtxt(a.gt, comments="#"); sc = load_scan(a.arm_scan)
    g = gaps(a.bag, gt, sc, a.gap_s); g.to_csv(a.out_gaps, index=False)
    print("gaps: %d, longest %.2f s" % (len(g), g.gap_s.max() if len(g) else 0))
    if a.raw_scans and a.out_raw:
        raw(a.bag, gt, sc, a.raw_scans, a.raw_half_window_s).to_csv(a.out_raw, index=False)
    return 0

if __name__ == "__main__":
    sys.exit(main())
