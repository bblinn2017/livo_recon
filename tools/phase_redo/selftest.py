#!/usr/bin/env python3
"""Schema/logic self-test on tiny FIXTURES (synthetic inputs with known statistics, generated here).
The fixtures test the reduction code only; nothing they produce is a result. Exit code 0 = all checks pass."""
import json, os, subprocess, sys, tempfile
import numpy as np, pandas as pd
from common import *

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)

def write_trace(D, rows, P0):
    cols = ["run_id", "scan_id", "t_abs", "phase", "dim", "qw", "qx", "qy", "qz", "px", "py", "pz", "vx", "vy", "vz", "bgx", "bgy", "bgz",
            "bax", "bay", "baz", "gx", "gy", "gz", "residual_count", "completed_iterations", "ol_window", "ol_reset"]
    pc = ["P_%d_%d" % (i, j) for i in range(18) for j in range(i, 18)]
    with open(os.path.join(D, "state_trace.csv"), "w") as f:
        f.write(",".join(cols + pc) + "\n")
        for r in rows:
            P = np.zeros((18, 18)); P[:9, :9] = r["P"]
            vals = [r["run"], r["scan"], "%.6f" % r["t"], r["phase"], 18] + list(r["q"]) + list(r["p"]) + list(r["v"]) + [0] * 6 + [0, 0, -9.81] \
                   + [r.get("res", 100), r.get("it", 3), r.get("win", 0), r.get("reset", 0)]
            vals += [P[i, j] for i in range(18) for j in range(i, 18)]
            f.write(",".join(str(x) for x in vals) + "\n")
    with open(os.path.join(D, "state_reference.txt"), "w") as f:
        f.write("schema state_reference_v1\nq_wxyz 1 0 0 0\np 0 0 0\nv 0 0 0\nP0_dim 18\nP0_row_major " + " ".join(str(x) for x in np.eye(18).ravel() * 1e-3) + "\n")

def exp_q(rv):
    th = np.linalg.norm(rv)
    if th < 1e-12: return np.array([1, *(0.5 * rv)])
    return np.array([np.cos(th / 2), *(np.sin(th / 2) * rv / th)])

def closed_loop_fixture(D, n=400, k_true=0.6):
    os.makedirs(D)
    sd_imu = np.array([0.01] * 3 + [0.02] * 3 + [0.05] * 3)
    rows, gtrows = [], []
    e_lio_prev = np.zeros(9)
    Pi = np.diag(sd_imu ** 2)
    Pl = np.diag((sd_imu * np.sqrt(1 - k_true)) ** 2)      # posterior covariance implies gain k_true
    for i in range(n):
        t = 100.0 + 0.1 * i
        e_imu = rng.normal(0, sd_imu)
        e_lio = e_imu * (1 - k_true) * 1.0 + rng.normal(0, 1e-9, 9)   # realised correction exactly k_true of the error
        for ph, e, P in (("post_imu", e_imu, Pi), ("post_lio", e_lio, Pl)):
            rows.append(dict(run="fx", scan=i + 1, t=t, phase=ph, P=P, q=exp_q(e[:3]), p=e[3:6], v=e[6:9]))
        gtrows.append((t, 0.0, 0.0, 0.0))
    # motion: last 100 scans move along +x at 1 m/s; estimate tracks it with independent error
    write_trace(D, rows, None)
    return rows

def main():
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        D = os.path.join(tmp, "cl")
        closed_loop_fixture(D)
        t = 100.0 + 0.1 * np.arange(400)
        pd.DataFrame({"t_abs": t, "est_px": 0, "est_py": 0, "est_pz": 0, "gt_px": 0.0, "gt_py": 0.0, "gt_pz": 0.0,
                      "est_qw": 1, "est_qx": 0, "est_qy": 0, "est_qz": 0, "gt_qw": 1, "gt_qx": 0, "gt_qy": 0, "gt_qz": 0}).to_csv(os.path.join(D, "lio_gt_matched.csv"), index=False)
        json.dump({"arm": "fixture_closed", "kind": "closed"}, open(os.path.join(D, "arm.json"), "w"))
        subprocess.check_call([sys.executable, os.path.join(HERE, "reduce_arm.py"), D])
        S = json.load(open(os.path.join(D, "arm_summary.json")))
        st = S["stationary"]["post_imu"]
        checks = []
        for ax in AXES:
            checks.append(("imu ratio %s ~1" % ax, abs(st[ax]["ratio_e2_over_P"] - 1) < 0.25))
        checks.append(("imu NEES9 ~9", abs(st["nees9"]["mean"] - 9) < 1.5))
        ph3 = S["phase3_stationary"]
        for ax in AXES:
            checks.append(("gain k %s ~0.6" % ax, abs(ph3[ax]["gain_k"] - 0.6) < 0.1))
            checks.append(("k/predicted %s ~1" % ax, abs(ph3[ax]["k_over_predicted"] - 1) < 0.2))
        for n_, c in checks:
            print(("PASS " if c else "FAIL ") + n_)
            ok &= c
        # open loop fixture: windows of 5 s, e = random walk growth, P consistent
        D2 = os.path.join(tmp, "ol"); os.makedirs(D2)
        rows = []
        scan = 0
        for w in range(6):
            e = np.zeros(9)
            for i in range(50):
                scan += 1
                tau = 0.1 * (i + 1)
                sd = np.array([0.001] * 3 + [0.01] * 3 + [0.02] * 3) * np.sqrt(tau)
                e = rng.normal(0, sd)
                rows.append(dict(run="fx", scan=scan, t=100 + 0.1 * (scan - 1), phase="post_imu", P=np.diag(sd ** 2), q=exp_q(e[:3]), p=e[3:6], v=e[6:9], win=w))
                rows.append(dict(run="fx", scan=scan, t=100 + 0.1 * (scan - 1), phase="post_lio", P=np.diag(sd ** 2), q=exp_q(e[:3]), p=e[3:6], v=e[6:9], win=w + (1 if i == 49 else 0), reset=int(i == 49)))
        write_trace(D2, rows, None)
        json.dump({"arm": "fixture_open", "kind": "open"}, open(os.path.join(D2, "arm.json"), "w"))
        subprocess.check_call([sys.executable, os.path.join(HERE, "reduce_arm.py"), D2])
        tau = pd.read_csv(os.path.join(D2, "arm_ol_tau.csv"))
        c = bool(abs(tau["ratio_px"].mean() - 1) < 0.25 and tau.n_windows.min() == 6 and len(tau) >= 4)
        print(("PASS " if c else "FAIL ") + "open-loop tau table: mean ratio ~1 over bins, 6 windows")
        ok &= c
    # dynamic fixture: 300 stationary scans then 300 moving along a circular arc; GT frame = estimator frame rotated + offset
    with tempfile.TemporaryDirectory() as tmp:
        D3 = os.path.join(tmp, "dyn"); os.makedirs(D3)
        lever = np.array([-0.293656, -0.012288, -0.273095])
        th = np.radians(30.0)
        Ral = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]); tal = np.array([5.0, -3.0, 1.0])
        sdi, sdl = 0.05, 0.02
        rows, gtr = [], []
        for i in range(600):
            t = 100 + 0.1 * i
            if i < 300: ptrue = np.zeros(3)
            else:
                s_ = 0.1 * (i - 300); ptrue = np.array([2.0 * np.sin(0.3 * s_), 2.0 * (1 - np.cos(0.3 * s_)), 0.05 * s_])
            for ph, sd in (("post_imu", sdi), ("post_lio", sdl)):
                p = ptrue + rng.normal(0, sd, 3)
                P = np.diag([1e-8] * 3 + [sd ** 2] * 3 + [1e-4] * 3)
                rows.append(dict(run="fx", scan=i + 1, t=t, phase=ph, P=P, q=[1, 0, 0, 0], p=p, v=[0, 0, 0]))
            g = Ral @ (ptrue + lever) + tal
            gtr.append((t, *(Ral @ (ptrue + lever) + tal), *(Ral @ (ptrue + lever) + tal), 1, 0, 0, 0, 1, 0, 0, 0))
        write_trace(D3, rows, None)
        pd.DataFrame(gtr, columns=["t_abs", "est_px", "est_py", "est_pz", "gt_px", "gt_py", "gt_pz", "est_qw", "est_qx", "est_qy", "est_qz", "gt_qw", "gt_qx", "gt_qy", "gt_qz"]).to_csv(os.path.join(D3, "lio_gt_matched.csv"), index=False)
        json.dump({"arm": "fixture_dyn", "kind": "closed"}, open(os.path.join(D3, "arm.json"), "w"))
        subprocess.check_call([sys.executable, os.path.join(HERE, "reduce_arm.py"), D3])
        S = json.load(open(os.path.join(D3, "arm_summary.json")))
        d = S["dynamic_gt_position"]["fit_all"]
        cs = [("t_move found near 130 s", S["t_move"] is not None and abs(S["t_move"] - 130.0) < 1.0),
              ("alignment recovers 30 deg", abs(d["align_rot_deg"] - 30.0) < 1.5),
              ("lio rms err in ~[0.02,0.05]*sqrt3", 0.02 < d["lio"]["rms_e_norm"] < 0.08),
              ("imu NEES_p3 ~3", d["imu"]["nees_p3_mean"] is not None and abs(d["imu"]["nees_p3_mean"] - 3) < 1.5)]
        for n_, c in cs:
            print(("PASS " if c else "FAIL ") + n_); ok &= c
    print("SELFTEST", "OK" if ok else "FAILED")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
