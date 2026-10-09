#!/usr/bin/env python3
"""r83_stationary_drift: how far the position drifts while the platform is still (before the movement onset), beside the unfiltered ATE, for arms that collapse.

Why: for arms that collapse, the pipeline's ground-truth error on the scans before the collapse is dragged by the whole-run alignment (R82A: sbs_02 edge 0.20 reads 500+ mm on its first joined scan while the node's own drift is 3 mm), so it is not the pre-collapse error. The node's own position change from its start reference (ep_*_mm in arm_scan.csv: displacement of the estimate from the stationary reference, NOT an error against GT)
is available from scan 1, and while the platform is at rest it IS the drift. This reports that drift per arm, with the unfiltered ATE and maximum beside it.

  python3 r83_stationary_drift.py --arm A --bag B --onset N --arm-scan <arm_scan.csv[.gz]> --ate-mm X --max-mm Y --out row.csv [--series series.csv]

Per arm: stationary scans = 1 .. onset-1. Columns written:
  arm, bag, onset_scan, n_scans, unfiltered_ate_mm, unfiltered_max_mm,
  stationary_ep_rms_mm, stationary_ep_max_mm, scan_of_stationary_ep_max, first_scan_ep_over_100mm_stationary (blank if none),
  first_joined_scan, gt_err_mm_at_first_joined_scan, first_scan_gt_err_over_500 (blank if none), gt_starts_after_onset (1/0),
  collapse_timing: 'stationary' (ep above 100 mm before the onset), 'transition' (first GT error above 500 mm in onset .. onset+59),
                   'later' (after), 'gt_starts_after_collapse' (the first joined scan already has GT error above 500 mm), 'none' (never above 500).
The per-scan series (scans 1 .. onset+60: arm, scan, ep_norm_mm, gt_err_mm) is written to --series (appended) when given.
arm_scan columns used: a scan column (scan, scan_id or scan_index); position displacement as ep_norm_mm, or ep_x_mm, ep_y_mm, ep_z_mm; gt_err_mm."""
import argparse, os, sys
import numpy as np
import pandas as pd

def load(path):
    d = pd.read_csv(path)
    sc = next((c for c in ("scan", "scan_id", "scan_index") if c in d.columns), None)
    if sc is None:
        raise ValueError("no scan column in " + path)
    if "ep_norm_mm" in d.columns:
        ep = d["ep_norm_mm"].to_numpy(float)
    elif all(c in d.columns for c in ("ep_x_mm", "ep_y_mm", "ep_z_mm")):
        ep = np.sqrt(d.ep_x_mm.to_numpy(float) ** 2 + d.ep_y_mm.to_numpy(float) ** 2 + d.ep_z_mm.to_numpy(float) ** 2)
    else:
        raise ValueError("no ep_norm_mm or ep_x/y/z_mm in " + path)
    g = d["gt_err_mm"].to_numpy(float) if "gt_err_mm" in d.columns else np.full(len(d), np.nan)
    return pd.DataFrame(dict(scan=d[sc].to_numpy(int), ep=ep, gterr=g)).sort_values("scan").reset_index(drop=True)

def analyse(arm, bag, onset, df, ate, mx):
    st = df[(df.scan >= 1) & (df.scan < onset)]
    r = dict(arm=arm, bag=bag, onset_scan=onset, n_scans=len(df), unfiltered_ate_mm=ate, unfiltered_max_mm=mx)
    if len(st):
        r["stationary_ep_rms_mm"] = float(np.sqrt(np.mean(st.ep ** 2)))
        r["stationary_ep_max_mm"] = float(st.ep.max())
        r["scan_of_stationary_ep_max"] = int(st.scan[st.ep.idxmax()])
        over = st[st.ep > 100]
        r["first_scan_ep_over_100mm_stationary"] = int(over.scan.iloc[0]) if len(over) else np.nan
    else:
        r.update(stationary_ep_rms_mm=np.nan, stationary_ep_max_mm=np.nan, scan_of_stationary_ep_max=np.nan, first_scan_ep_over_100mm_stationary=np.nan)
    j = df[df.gterr.notna()]
    if len(j):
        r["first_joined_scan"] = int(j.scan.iloc[0]); r["gt_err_mm_at_first_joined_scan"] = float(j.gterr.iloc[0])
        o5 = j[j.gterr > 500]
        r["first_scan_gt_err_over_500"] = int(o5.scan.iloc[0]) if len(o5) else np.nan
    else:
        r["first_joined_scan"] = np.nan; r["gt_err_mm_at_first_joined_scan"] = np.nan; r["first_scan_gt_err_over_500"] = np.nan
    r["gt_starts_after_onset"] = int(len(j) > 0 and j.scan.iloc[0] >= onset)
    d5 = r["first_scan_gt_err_over_500"]
    if not np.isnan(r["first_scan_ep_over_100mm_stationary"]):
        t = "stationary"
    elif np.isnan(d5):
        t = "none"
    elif d5 == r["first_joined_scan"]:
        t = "gt_starts_after_collapse"
    elif d5 < onset + 60:
        t = "transition"
    else:
        t = "later"
    r["collapse_timing"] = t
    return r

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--arm", "--bag", "--arm-scan", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--onset", type=int, required=True)
    ap.add_argument("--ate-mm", type=float, required=True); ap.add_argument("--max-mm", type=float, required=True)
    ap.add_argument("--series")
    a = ap.parse_args()
    df = load(a.arm_scan)
    r = analyse(a.arm, a.bag, a.onset, df, a.ate_mm, a.max_mm)
    pd.DataFrame([r]).to_csv(a.out, index=False)
    if a.series:
        s = df[df.scan <= a.onset + 60].assign(arm=a.arm)[["arm", "scan", "ep", "gterr"]].rename(columns={"ep": "ep_norm_mm", "gterr": "gt_err_mm"})
        s.to_csv(a.series, mode="a", header=not os.path.exists(a.series), index=False)
    print(pd.DataFrame([r]).T.to_string(header=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
