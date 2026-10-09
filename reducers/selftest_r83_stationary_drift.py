#!/usr/bin/env python3
"""Synthetic checks for r83_stationary_drift (an isolated identity check). Four arms: still and healthy, drifting while still, collapse in the
transition with GT from scan 1, collapse before GT starts. Prints 'failures: 0' on success."""
import os, sys, tempfile, numpy as np, pandas as pd
import r83_stationary_drift as S
n, onset = 600, 200
sc = np.arange(1, n + 1)
def arm(ep_stat, gt_start, collapse_at):
    ep = np.concatenate([ep_stat(sc[:onset - 1]), 1000 * np.arange(1, n - onset + 2) * 0.01 + 50])
    gt = np.full(n, np.nan); gt[gt_start - 1:] = 10.0
    if collapse_at: gt[collapse_at - 1:] = 600.0
    return pd.DataFrame(dict(scan=sc, ep_norm_mm=ep, gt_err_mm=gt))
cases = [
    ("healthy", arm(lambda s: 0.01 * s, 1, 0), "none", 1),
    ("drift while still", arm(lambda s: 2.0 * s, 1, 0), "stationary", 1),
    ("collapse in transition", arm(lambda s: 0.01 * s, 1, 230), "transition", 1),
    ("gt starts after collapse", arm(lambda s: 0.01 * s, 250, 250), "gt_starts_after_collapse", 0),
]
fails = 0
for name, df, want, gt_from_one in cases:
    p = os.path.join(tempfile.mkdtemp(), "arm_scan.csv"); df.to_csv(p, index=False)
    r = S.analyse(name, "b", onset, S.load(p), 20.0, 50.0)
    ok = r["collapse_timing"] == want and (r["first_joined_scan"] == 1) == bool(gt_from_one)
    if name == "healthy": ok = ok and abs(r["stationary_ep_max_mm"] - 0.01 * (onset - 1)) < 1e-9 and r["scan_of_stationary_ep_max"] == onset - 1
    if name == "drift while still": ok = ok and r["first_scan_ep_over_100mm_stationary"] == 51
    print("%-26s timing %-26s stationary max %8.2f mm  %s" % (name, r["collapse_timing"], r["stationary_ep_max_mm"], "pass" if ok else "FAIL"))
    fails += (not ok)
print("failures: %d" % fails)
sys.exit(1 if fails else 0)
