#!/usr/bin/env python3
"""Synthetic checks for r83_prealign_ate (an isolated identity check). A trajectory with 2 cm noise; an arm that follows it and then runs away after scan 300.
 - d is found at the scan where the run-away passes 500 mm from the control;
 - the pre-collapse ATE equals the noise level (about 20 mm RMS 3D) although a whole-run alignment would be dragged by the run-away;
 - a healthy arm is reported never_diverges; a collapse before 50 joined scans is reported too_few_joined_scans_before_d.
Prints 'failures: 0' on success."""
import os, sys, tempfile, numpy as np, pandas as pd
import r83_prealign_ate as A
import r83_cov_consistency as C
rng = np.random.default_rng(1)
n = 500
t = 1.6e9 + 0.1 * np.arange(1, n + 1)
p = np.stack([np.cumsum(rng.normal(size=n)) * 0.05, np.cumsum(rng.normal(size=n)) * 0.05, np.zeros(n)], 1)
def trace(pos):
    return pd.DataFrame(dict(scan_id=np.arange(1, n + 1), t_abs=t, phase="post_lio", qw=1.0, qx=0.0, qy=0.0, qz=0.0, px=pos[:, 0], py=pos[:, 1], pz=pos[:, 2]))
noise = rng.normal(size=(n, 3)) * 0.0115       # 3D rms about 20 mm
ctl = trace(p + noise)
def runaway(start):
    pos = p + noise * 1.0
    k = np.arange(n) - start
    pos[:, 0] += np.where(k > 0, 0.6 + 0.02 * k, 0.0)   # a 600 mm jump at the start scan, then 20 mm per scan
    return trace(pos)
tg = 1.6e9 + 0.05 * np.arange(2, 2 * n + 1)
gt = np.column_stack([tg] + [np.interp(tg, t, p[:, k]) for k in range(3)] + [np.zeros(len(tg))] * 3 + [np.ones(len(tg))])
d = tempfile.mkdtemp(); gp = os.path.join(d, "gt.txt"); np.savetxt(gp, gt, fmt="%.12f")
fails = 0
def check(name, ok, msg):
    global fails
    print("%-40s %s  %s" % (name, "pass" if ok else "FAIL", msg)); fails += (not ok)
r = A.run("a", runaway(300), ctl, gp, np.zeros(3), 0.15, 500, 50, 20, 900)
check("collapse found at the right scan", r["d_scan"] == 302, "d_scan %s (the jump starts at scan 302: k > 0)" % r["d_scan"])
check("pre-collapse ATE at the noise level", r["status"] == "ok" and 12 < r["prealign_ate_mm"] < 40, "prealign_ate_mm %.1f, n joined %s" % (r["prealign_ate_mm"], r["n_joined_before_d"]))
# whole-run error for contrast: aligned over the whole run it is far larger
tw = runaway(300)
Re = tw[["px", "py", "pz"]].to_numpy(float)
Ra, ta = C.umeyama(Re, np.stack([np.interp(t, tg, gt[:, k]) for k in (1, 2, 3)], 1))
whole = 1000 * np.sqrt(np.mean(np.sum((np.stack([np.interp(t, tg, gt[:, k]) for k in (1, 2, 3)], 1) - (Re @ Ra.T + ta)) ** 2, axis=1)))
check("whole-run alignment is much worse", whole > 3 * r["prealign_ate_mm"], "whole-run RMS %.1f mm against %.1f mm before the collapse" % (whole, r["prealign_ate_mm"]))
r2 = A.run("b", trace(p + rng.normal(size=(n, 3)) * 0.0115), ctl, gp, np.zeros(3), 0.15, 500, 50, 20, 40)
check("healthy arm never diverges", r2["status"] == "never_diverges", r2["status"])
r3 = A.run("c", runaway(20), ctl, gp, np.zeros(3), 0.15, 500, 50, 20, 900)
check("early collapse flagged", r3["status"] == "too_few_joined_scans_before_d", "%s, joined %s" % (r3["status"], r3["n_joined_before_d"]))
print("failures: %d" % fails)
sys.exit(1 if fails else 0)
