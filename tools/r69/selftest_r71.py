#!/usr/bin/env python3
"""selftest_r71.py -- proves the R71 analysis statistics can FAIL (rule 2.4). Synthetic data only; prints SELFTEST R71 PASS or exits 1.

    python3 tools/r69/selftest_r71.py [--keep DIR]

T1  independent residuals: random-split observed/expected difference-variance ratio near 1 (0.6..1.6, median over axes), ICC near 0.
T2  plane-level common offsets (the residuals of one plane share an error): the by-plane split ratio and the ICC must be clearly larger than in T1.
T3  cross-frame: per-frame LiDAR-only increments drawn from the claimed information: realised/believed near 1; the same increments scaled by 3 give about 3.
T4  the dump reader rejects nothing silently: a dump with a frame of fewer than 12 residuals is skipped and counted, not analysed.
"""
import os, subprocess, sys, tempfile, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "r71_residual_info.py")


def make_dump(rng, offset_sd, n_planes=150, per_plane=8, frames=(2, 3)):
    rows = []
    for f in frames:
        # planes with random unit normals spread over the sphere, points around them
        for k in range(n_planes):
            n = rng.normal(size=3); n /= np.linalg.norm(n)
            centre = rng.normal(size=3) * 8
            off = rng.normal() * offset_sd
            for j in range(per_plane):
                p = centre + rng.normal(size=3) * 0.8
                sig2 = 0.02 ** 2
                r = rng.normal() * np.sqrt(sig2) + off
                hrot = np.cross(p, n)
                rows.append([f, 0, len(rows), len(rows), 0.0, r, sig2, 0.0, 1000 + k, k, 0, 0, *hrot, *n, *(p), *(p)])
    cols = "frame,iter,idx,source_index,t,r,sigma_squared,plane_var_term,plane_key,node_id,tier,vis,hrot_x,hrot_y,hrot_z,hpos_x,hpos_y,hpos_z,wx,wy,wz,bx,by,bz".split(",")
    return pd.DataFrame(rows, columns=cols)


def run_dump(df, d):
    p = os.path.join(d, "dump.csv"); df.to_csv(p, index=False)
    subprocess.check_call([sys.executable, TOOL, "dump", "--dump", p, "--out", d], stdout=subprocess.DEVNULL)
    return pd.read_csv(os.path.join(d, "residual_info_axes.csv")), pd.read_csv(os.path.join(d, "residual_info_frames.csv"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--keep"); a = ap.parse_args()
    fails = []
    with tempfile.TemporaryDirectory() as t:
        d = a.keep or t; os.makedirs(d, exist_ok=True)
        rng = np.random.default_rng(7)
        d1 = os.path.join(d, "t1"); os.makedirs(d1, exist_ok=True)
        ax1, fr1 = run_dump(make_dump(rng, 0.0), d1)
        r1 = ax1[ax1.split == "random_resid"].ratio_obs_over_expected.median()
        p1 = ax1[ax1.split == "random_plane"].ratio_obs_over_expected.median(); icc1 = fr1.icc_plane.median()
        print("T1 independent: random-split ratio %.2f, by-plane ratio %.2f, ICC %.3f" % (r1, p1, icc1))
        if not (0.6 < r1 < 1.6):
            fails.append("T1 random ratio %.2f outside 0.6..1.6" % r1)
        if abs(icc1) > 0.15:
            fails.append("T1 ICC %.3f not near 0" % icc1)
        d2 = os.path.join(d, "t2"); os.makedirs(d2, exist_ok=True)
        ax2, fr2 = run_dump(make_dump(rng, 0.02), d2)
        p2 = ax2[ax2.split == "random_plane"].ratio_obs_over_expected.median(); icc2 = fr2.icc_plane.median()
        print("T2 plane-level offsets: by-plane ratio %.2f, ICC %.3f" % (p2, icc2))
        if not (p2 > 1.5 * p1 and p2 > 1.8):
            fails.append("T2 by-plane ratio %.2f did not rise clearly above T1 %.2f" % (p2, p1))
        if not icc2 > icc1 + 0.2:
            fails.append("T2 ICC %.3f did not rise above T1 %.3f" % (icc2, icc1))
        # T3 cross-frame
        rows = []
        L = np.diag([4e4, 4e4, 1e3, 2e4, 2e4, 3e4]).astype(float)
        for scale in (1.0, 3.0):
            rows = []
            for f in range(2, 202):
                dx = rng.multivariate_normal(np.zeros(6), np.linalg.inv(L)) * scale
                g = L @ dx
                r = dict(frame=f, iter=0, n_resid=1000)
                k = 0
                for i in range(6):
                    for j in range(i, 6):
                        r["HtH_%d%d" % (i, j)] = L[i, j]
                for i in range(6):
                    r["Htz_%d" % i] = g[i]
                rows.append(r)
            p = os.path.join(d, "ud_%d.csv" % int(scale)); pd.DataFrame(rows).to_csv(p, index=False)
            od = os.path.join(d, "c%d" % int(scale)); os.makedirs(od, exist_ok=True)
            subprocess.check_call([sys.executable, TOOL, "cross", "--diag", p, "--out", od], stdout=subprocess.DEVNULL)
            c = pd.read_csv(os.path.join(od, "crossframe_axes.csv"))
            v = c[c.window == "frames_2_480"].realised_over_believed.median()
            print("T3 cross-frame scale %.0f: realised/believed %.2f" % (scale, v))
            if scale == 1.0 and not (0.8 < v < 1.25):
                fails.append("T3 scale 1 ratio %.2f outside 0.8..1.25" % v)
            if scale == 3.0 and not (2.4 < v < 3.8):
                fails.append("T3 scale 3 ratio %.2f outside 2.4..3.8" % v)
        # T4 tiny frame skipped
        small = make_dump(rng, 0.0, n_planes=1, per_plane=5, frames=(9,))
        d4 = os.path.join(d, "t4"); os.makedirs(d4, exist_ok=True)
        p = os.path.join(d4, "dump.csv"); pd.concat([small, make_dump(rng, 0.0, frames=(2,))]).to_csv(p, index=False)
        subprocess.check_call([sys.executable, TOOL, "dump", "--dump", p, "--out", d4], stdout=subprocess.DEVNULL)
        fr = pd.read_csv(os.path.join(d4, "residual_info_frames.csv"))
        print("T4 frames analysed %s (frame 9 has 5 residuals and must be absent)" % sorted(fr.frame.tolist()))
        if 9 in fr.frame.tolist():
            fails.append("T4 frame with 5 residuals was analysed")
    if fails:
        print("SELFTEST R71 FAIL:\n  " + "\n  ".join(fails)); sys.exit(1)
    print("SELFTEST R71 PASS")


if __name__ == "__main__":
    main()
