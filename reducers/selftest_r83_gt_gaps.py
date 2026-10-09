#!/usr/bin/env python3
"""Synthetic check for r83_gt_gaps: one planted 3 s gap in a 20 Hz GT; a scan table where the first scan after the gap has a big error. Prints 'failures: 0' on success."""
import sys, numpy as np, pandas as pd
import r83_gt_gaps as G
tg = 1.6e9 + 0.05 * np.arange(0, 400); keep = ~((tg > 1.6e9 + 8.0) & (tg < 1.6e9 + 11.0))
gt = np.column_stack([tg[keep], np.cumsum(np.ones(keep.sum()))[:, None] * np.array([[0.01, 0, 0]]), np.zeros((keep.sum(), 4))])
t = 1.6e9 + 0.1 * np.arange(1, 190); err = np.full(len(t), 10.0); err[(t > 1.6e9 + 8.0) & (t < 1.6e9 + 11.0)] = np.nan
first = np.nonzero(t >= tg[keep][np.nonzero(np.diff(tg[keep]) > 0.15)[0][0] + 1])[0][0]; err[first] = 340.0
sc = pd.DataFrame(dict(scan=np.arange(1, 190), t=t, err=err))
g = G.gaps("b", gt, sc, 0.15)
fails = 0
ok = len(g) == 1 and abs(g.gap_s.iloc[0] - 3.05) < 0.06 and g.n_scans_in_gap.iloc[0] >= 28 and g.first_scan_after.iloc[0] == first + 1 and g.err_after_1_mm.iloc[0] == 340.0 and g.err_after_2_mm.iloc[0] == 10.0
print("one gap found, counted and its first scan after reported  %s" % ("pass" if ok else "FAIL")); fails += (not ok)
r = G.raw("b", gt, sc, [first + 1], 0.5)
ok = len(r) >= 9 and (r.dt_to_scan_s.abs() <= 0.5).all()
print("raw samples around a scan limited to the half window       %s  (%d rows)" % ("pass" if ok else "FAIL", len(r))); fails += (not ok)
print("failures: %d" % fails); sys.exit(1 if fails else 0)
