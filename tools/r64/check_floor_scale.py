#!/usr/bin/env python3
"""R64 check C7 (floor_scale arithmetic). Two OPEN-LOOP arms that differ only in one floor scale see the same states, hence the same
dynamic term, so   mean_q(B) - mean_q(A) = (f_B - f_A) * floor   exactly, scan by scan. This prints the implied floor per axis.

python3 check_floor_scale.py --runs r64_runs --a <armA> --fa <floor_scale_A> --b <armB> --fb <floor_scale_B> --which gyro|acc
Prints, per axis x/y/z: mean and max relative spread (max-min)/mean of the implied floor over scans 1..480, and the number of scans used.
Reports numbers only."""
import argparse, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--fa", type=float, required=True); ap.add_argument("--fb", type=float, required=True)
ap.add_argument("--which", choices=["gyro", "acc"], required=True)
a = ap.parse_args()
qa = pd.read_csv(os.path.join(a.runs, a.a, "motion_q_scan.csv")).set_index("scan_index")
qb = pd.read_csv(os.path.join(a.runs, a.b, "motion_q_scan.csv")).set_index("scan_index")
idx = [i for i in qa.index if i in qb.index and i <= 480]
pre = "mean_q_gyr_" if a.which == "gyro" else "mean_q_acc_"
print("arms %s (floor scale %g) vs %s (%g), %s, %d common scans in 1..480" % (a.a, a.fa, a.b, a.fb, a.which, len(idx)))
for ax in "xyz":
    d = (qb.loc[idx, pre + ax] - qa.loc[idx, pre + ax]) / (a.fb - a.fa)
    print("  axis %s: implied floor mean %.6e  min %.6e  max %.6e  relative spread %.3e" % (ax, d.mean(), d.min(), d.max(), (d.max() - d.min()) / abs(d.mean())))
