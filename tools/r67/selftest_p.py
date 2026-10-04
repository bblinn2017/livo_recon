#!/usr/bin/env python3
"""Self-test on synthetic data with KNOWN answers. A consistent filter must give nees9 ~ 9 and jump9 ~ 9 - tr(P_pre^-1 P_post).
Also builds an overconfident arm (errors 3x larger than P) and checks the reduction reports oc9 ~ 9 and the zipped size.
Run: python3 selftest_p.py   (no project data needed)"""
import os, sys, json, shutil, tempfile, subprocess
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import log_so3  # noqa

def exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def to_quat(R):
    w = np.sqrt(max(1 + R[0, 0] + R[1, 1] + R[2, 2], 0)) / 2
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])

def make_arm(d, scale_err, nscan=800, stat_last=480, seed=1):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = 18
    A = rng.normal(size=(n, n)); C = A @ A.T; s = np.sqrt(np.diag(C)); C = C / np.outer(s, s)
    sd = np.array([1e-3] * 3 + [3e-3] * 3 + [5e-3] * 3 + [1e-4] * 3 + [1e-2] * 3 + [1e-2] * 3)
    Ppre = C * np.outer(sd, sd) * 0.8 + np.diag(sd ** 2) * 0.2
    Ppost = 0.5 * Ppre
    Rref = exp_so3(np.array([0.1, -0.2, 0.3])); pref = np.array([1.0, 2.0, 3.0]); vref = np.zeros(3)
    Lp, Ld = np.linalg.cholesky(Ppost), np.linalg.cholesky(Ppre - Ppost)
    hdr = "run_id,scan_id,t_abs,phase,dim,qw,qx,qy,qz,px,py,pz,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz,gx,gy,gz,residual_count,completed_iterations,ol_window,ol_reset"
    pc = ["P_%d_%d" % (i, j) for i in range(n) for j in range(i, n)]
    rows = []
    for k in range(1, nscan + 1):
        e_post = Lp @ rng.normal(size=n) * scale_err
        dlt = Ld @ rng.normal(size=n)
        for phase, e, P in (("post_imu", e_post - dlt, Ppre), ("post_lio", e_post, Ppost)):
            R = Rref @ exp_so3(e[:3]); q = to_quat(R)
            st = np.concatenate([q, pref + e[3:6], vref + e[6:9], 0.01 + e[9:12], 0.02 + e[12:15], np.array([0, 0, 9.81]) + e[15:18]])
            Pu = [P[i, j] for i in range(n) for j in range(i, n)]
            rows.append(["syn", k, 1e9 + 0.1 * k, phase, n] + list(st) + [1000, 3, 0, 0] + Pu)
    pd.DataFrame(rows, columns=hdr.split(",") + pc).to_csv(os.path.join(d, "state_trace.csv"), index=False, float_format="%.12g")
    q = to_quat(Rref)
    with open(os.path.join(d, "state_reference.txt"), "w") as f:
        f.write("schema state_reference_v1\nq_wxyz %s\np %s\nv %s\nP0_dim 18\nP0_row_major %s\n" % (
            " ".join("%.17g" % x for x in q), " ".join("%.17g" % x for x in pref), " ".join("%.17g" % x for x in vref), " ".join("%.17g" % x for x in Ppre.flatten())))
    g = {"scan_index": np.arange(1, nscan + 1), "t_abs": 1e9 + 0.1 * np.arange(1, nscan + 1), "dim": n}
    for tag, vals in (("Pstart", np.diag(Ppre) * 0.9), ("Pend", np.diag(Ppre)), ("Qacc", np.diag(Ppre) * 0.1)):
        for i in range(n):
            g["%s_%d" % (tag, i)] = vals[i]
    pd.DataFrame(g).to_csv(os.path.join(d, "imu_cov_growth.csv"), index=False)
    json.dump({"arm": os.path.basename(d), "seq": "eee_01"}, open(os.path.join(d, "arm.json"), "w"))
    return Ppre, Ppost

def main():
    tmp = tempfile.mkdtemp(prefix="ptest_")
    root = os.path.join(tmp, "root")
    Ppre, Ppost = make_arm(os.path.join(root, "arm_consistent"), 1.0)
    make_arm(os.path.join(root, "arm_overconfident"), 3.0, seed=2)
    out = os.path.join(tmp, "P_RETURN")
    sl = os.path.join(tmp, "scan_arms.txt"); open(sl, "w").write("arm_overconfident\narm_consistent\n")
    r = subprocess.run([sys.executable, os.path.join(HERE, "p_collect.py"), root, "--out", out, "--scan-arms", sl], capture_output=True, text=True)
    print(r.stdout.strip()); print(r.stderr.strip()[-600:])
    wt = pd.read_csv(os.path.join(out, "window_table.csv")).set_index("arm")
    exp_jump = 9 - np.trace(np.linalg.solve(Ppre[:9, :9], Ppost[:9, :9]))
    ok = True
    def chk(name, val, lo, hi):
        nonlocal ok
        good = lo <= val <= hi; ok &= good
        print("%-58s %10.3f  expected in [%.2f, %.2f]  %s" % (name, val, lo, hi, "ok" if good else "FAIL"))
    c, o = wt.loc["arm_consistent"], wt.loc["arm_overconfident"]
    chk("consistent: median nees9_post (stationary)", c["w2_stationary__nees9_post__med"], 7.5, 10.5)
    chk("consistent: median nees9_pre (stationary)", c["w2_stationary__nees9_pre__med"], 7.5, 10.5)
    chk("consistent: median jump9_mahal (expected %.2f)" % exp_jump, c["w2_stationary__jump9_mahal__med"], exp_jump - 1.0, exp_jump + 1.0)
    chk("consistent: frac nees9_post above chi2 95%", c["w2_stationary__nees9_post__frac_above_chi2_95"], 0.0, 0.12)
    chk("overconfident(x3): median oc9_post (expected ~8)", o["w2_stationary__oc9_post__med"], 6.0, 10.0)
    chk("overconfident(x3): median nees_p_post / 3 (expected ~8)", o["w2_stationary__nees_p_post__med"] / 3, 5.5, 10.5)
    chk("consistent: sd_p_post mm (expected %.2f)" % (np.sqrt(np.trace(Ppost[3:6, 3:6])) * 1e3), c["w2_stationary__sd_p_post__med"], 0.95 * np.sqrt(np.trace(Ppost[3:6, 3:6])) * 1e3, 1.05 * np.sqrt(np.trace(Ppost[3:6, 3:6])) * 1e3)
    chk("consistent: sdratio_p (expected 0.707)", c["w2_stationary__sdratio_p__med"], 0.70, 0.72)
    chk("consistent: qshare_p (expected 0.1)", c["w2_stationary__qshare_p__med"], 0.09, 0.11)
    chk("motion window has no reference NEES (NaN or absent)", 0.0 if ("w4_motion_early__nees9_post__med" not in wt.columns or np.isnan(c.get("w4_motion_early__nees9_post__med", np.nan))) else 1.0, 0.0, 0.0)
    chk("motion window still reports jump9_mahal", c["w4_motion_early__jump9_mahal__med"], 0.0, 20.0)
    sc = pd.read_csv(os.path.join(out, "scan", "arm_consistent.csv.gz"))
    print("scan file rows %d, columns %d" % sc.shape)
    z = out + ".zip"
    print("zip size %.3f MB for 2 arms (window table %d columns)" % (os.path.getsize(z) / 1e6, wt.shape[1]))
    print("SELFTEST", "PASS" if ok else "FAIL")
    if "--keep" in sys.argv:
        print("kept", tmp)
    else:
        shutil.rmtree(tmp)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
