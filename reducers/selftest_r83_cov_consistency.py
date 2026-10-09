#!/usr/bin/env python3
"""Synthetic checks for r83_cov_consistency v2 (an isolated identity check; no number from it is evidence about the node).
 1 calibrated covariance: NEES near 3, coverage near 0.99; a covariance 9x too small is flagged.
 2 gap handling: a long GT gap is excluded with --max-gt-gap and counted; without it the error over the gap shows in rms3d.
 3 rms3d equals sqrt(mean(|e|^2)) computed independently; bins tile the joined scans exactly and each bin has NEES near 3.
 4 --pipeline-scan columns are filled and agree with the tool's own rms to within 15 percent when the pipeline column is the true error.
 5 r83_cov_collect concatenates NTU-like (no attitude) and Hilti-like (attitude) rows without a schema error.
Prints 'failures: 0' on success."""
import os, sys, tempfile
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
import r83_cov_consistency as C
import r83_cov_collect as K

def make(scale, seed=0, n=600, with_q=True):
    rng = np.random.default_rng(seed)
    t = 1.6e9 + 0.1 * np.arange(1, n + 1)
    p = np.stack([np.cumsum(rng.normal(size=n)) * 0.05, np.cumsum(rng.normal(size=n)) * 0.05, np.zeros(n)], 1)
    R = Rot.from_rotvec(np.stack([np.zeros(n), np.zeros(n), np.cumsum(rng.normal(size=n)) * 0.01], 1))
    sd_p, sd_r = 0.01, 0.003
    ep = rng.normal(size=(n, 3)) * sd_p * scale
    er = rng.normal(size=(n, 3)) * sd_r * scale
    Rest = R * Rot.from_rotvec(er)
    pest = p - ep
    rows = []
    for i in range(n):
        row = dict(run_id="x", scan_id=i + 1, t_abs=t[i], phase="post_lio")
        qq = Rest[i].as_quat()
        row.update(qw=qq[3], qx=qq[0], qy=qq[1], qz=qq[2], px=pest[i, 0], py=pest[i, 1], pz=pest[i, 2])
        for a in range(18):
            for b in range(a, 18):
                v = 0.0
                if a == b:
                    v = sd_r ** 2 if a < 3 else (sd_p ** 2 if a < 6 else 1.0)
                row["P_%d_%d" % (a, b)] = v
        rows.append(row)
    tr = pd.DataFrame(rows)
    # GT at 20 Hz (twice the scan rate) by linear interpolation of the path; quaternion identity unless with_q
    tg = 1.6e9 + 0.05 * np.arange(2, 2 * n + 1)
    pg = np.stack([np.interp(tg, t, p[:, k]) for k in range(3)], 1)
    if with_q:
        qg = Rot.from_rotvec(np.stack([np.zeros(len(tg)), np.zeros(len(tg)), np.interp(tg, t, np.cumsum(rng.normal(size=n)) * 0 + R.as_rotvec()[:, 2])], 1)).as_quat()
    else:
        qg = np.tile([0, 0, 0, 1.0], (len(tg), 1))
    return tr, np.column_stack([tg, pg, qg]), ep

fails = 0
def check(name, ok, msg):
    global fails
    print("%-44s %s  %s" % (name, "pass" if ok else "FAIL", msg))
    fails += (not ok)

d = tempfile.mkdtemp()
def write(tr, gt, tag):
    tp, gp = os.path.join(d, tag + "_tr.csv"), os.path.join(d, tag + "_gt.txt")
    tr.to_csv(tp, index=False); np.savetxt(gp, gt, fmt="%.12f")
    return tp, gp

# 1 calibrated / too small (position-only GT: identity quaternion)
for scale, name in ((1.0, "1 calibrated"), (3.0, "1 covariance 9x too small")):
    tr, gt, ep = make(scale, with_q=False)
    tp, gp = write(tr, gt, "a%d" % int(scale))
    r = C.analyse(C.load_trace(tp), gp, np.zeros(3), 0, name)[0].iloc[0]
    ok = (abs(r.nees_pos_med - 2.37) < 0.8 and r.cover99_pos > 0.95) if scale == 1.0 else (r.nees_pos_med > 10 and r.cover99_pos < 0.5)
    check(name, ok, "nees_pos_med %.2f cover99_pos %.3f rms3d %.2f mm" % (r.nees_pos_med, r.cover99_pos, r.rms3d_err_mm))

# 2 + 3 + 4: a 6 s GT gap, bins, pipeline column
tr, gt, ep = make(1.0, seed=3, with_q=False)
tg = gt[:, 0]
gap = (tg > tr.t_abs[199]) & (tg < tr.t_abs[259])
gt_g = gt[~gap]
tp, gp = write(tr, gt_g, "g")
tp0, gp0 = write(tr, gt, "ng")
rg, bg = C.analyse(C.load_trace(tp), gp, np.zeros(3), 0, "gap", max_gap=0.15, bins=100)
r0, _ = C.analyse(C.load_trace(tp), gp, np.zeros(3), 0, "nogapfix", max_gap=0.0)
a, b = rg.iloc[0], r0.iloc[0]
check("2 gap excluded and counted", a.n_gap_excluded >= 55 and b.n_gap_excluded == 0, "excluded %d (expected about 59), without the option %d" % (a.n_gap_excluded, b.n_gap_excluded))
check("2 gap error shows without the option", b.rms3d_err_mm > a.rms3d_err_mm + 5, "rms3d %.2f mm with --max-gt-gap, %.2f mm without" % (a.rms3d_err_mm, b.rms3d_err_mm))
# independent rms on the scans that are inside no gap, using the true error ep (alignment moves it little)
mask = np.ones(len(tr), bool); mask[199:260] = False
rms_true = 1000 * np.sqrt(np.mean(np.sum(ep[mask] ** 2, axis=1)))
check("3 rms3d equals independent rms", abs(a.rms3d_err_mm - rms_true) / rms_true < 0.15, "tool %.2f mm, independent %.2f mm" % (a.rms3d_err_mm, rms_true))
tot = int(bg.n.sum())
tiles = all(bg.bin_scan_lo.iloc[i + 1] == bg.bin_scan_hi.iloc[i] + 1 for i in range(len(bg) - 1))
check("3 bins tile the joined scans", tot == int(a.n_joined) and tiles and len(bg) == 6, "bins %d, scans in bins %d, joined %d" % (len(bg), tot, a.n_joined))
check("3 each bin NEES near 3", bool(((bg.nees_pos_med > 1.2) & (bg.nees_pos_med < 4.0)).all()), "bin NEES %s" % np.round(bg.nees_pos_med.values, 2))
ps = pd.DataFrame(dict(scan=tr.scan_id.values, gt_err_mm=1000 * np.linalg.norm(ep, axis=1)))
pp = os.path.join(d, "arm_scan.csv"); ps.to_csv(pp, index=False)
rp, _ = C.analyse(C.load_trace(tp), gp, np.zeros(3), 0, "pipe", max_gap=0.15, pipeline_scan=pp)
q = rp.iloc[0]
check("4 pipeline column filled and consistent", np.isfinite(q.pipe_rms_err_mm) and abs(q.pipe_rms_err_mm - q.rms3d_err_mm) / q.rms3d_err_mm < 0.15, "pipe %.2f mm, tool %.2f mm, n %d" % (q.pipe_rms_err_mm, q.rms3d_err_mm, q.n_pipe_scans))

# 5 collect: NTU-like and Hilti-like rows
trq, gtq, _ = make(1.0, seed=5, with_q=True)
tpq, gpq = write(trq, gtq, "q")
dn, _ = C.analyse(C.load_trace(tp0), gp0, np.zeros(3), 100, "ntu_like")
dh, _ = C.analyse(C.load_trace(tpq), gpq, np.zeros(3), 0, "hilti_like")
f1, f2 = os.path.join(d, "r1.csv"), os.path.join(d, "r2.csv")
dn.to_csv(f1, index=False); dh.to_csv(f2, index=False)
out = os.path.join(d, "all.csv")
n = K.collect([f1, f2], out)
back = pd.read_csv(out)
check("5 collect mixed schemas", n == len(dn) + len(dh) and len(back) == n and "nees_rot_right_med" in back.columns and back.nees_rot_right_med.isna().sum() == len(dn), "%d rows, %d columns" % (len(back), back.shape[1]))
print("failures: %d" % fails)
sys.exit(1 if fails else 0)
