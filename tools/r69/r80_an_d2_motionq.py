#!/usr/bin/env python3
"""R80 D2 (F-437): why eee_02 fails under motion-Q. Host-only analysis; planning agent's script, not run on real data by its author.

    python3 r80_an_d2_motionq.py --mq-dir <eee_02 mq arm dir> --fq-dir <eee_02 fixed-Q arm dir> --sep-mm 5 [10 ...] [--others bag=<mq arm dir> ...] --out <dir>
    python3 r80_an_d2_motionq.py --selftest

Inputs (names as the estimator writes them; each may also be .gz or the *_all.csv.gz return name):
  <mq-dir>/motion_q_scan.csv   scan_index,t_abs,t_rel_s,model,beta,acc_scale,gyro_scale,n_samples,mean_dynamic_acc_norm,mean_dynamic_gyr_norm,mean_q_acc_{x,y,z},
                               mean_q_gyr_{x,y,z},acc_clamped_axes,gyr_clamped_axes,mean_debiased_*_energy_*,acc_cap_fraction,gyr_cap_fraction   (src/diagnostics/motion_q_writer.cpp)
  <fq-dir>/motion_q_scan.csv   optional (the fixed-Q arm writes the same file with its model name; if absent the mq/fq Q ratio is left empty)
  <mq-dir>, <fq-dir>/pos_trace_all.csv.gz   scan_id,t_abs,t_rel_s,px,py,pz,yaw_rad[,qw..qz]   (r74_frame_errors.py)
NOTE the columns acc_scale / gyro_scale are the CONFIGURED motion scale constants (opts_.motion_acc_scale / motion_gyr_scale), the same on every row. The Q actually
applied per scan is the per-scan mean process variance mean_q_acc_{x,y,z} / mean_q_gyr_{x,y,z} (imu_processing.cpp: q_acc_sum / q_samples). This script uses those:
q_acc = mean of the three axes of mean_q_acc, q_gyr likewise (the three axes are also written out).

Outputs in --out:
  d2_per_scan.csv          per scan of the eee_02 mq arm: q_acc, q_gyr (+ axes, caps, dynamic norms), the same for fq if present, q ratios mq/fq, q ratios to the arm's own
                           whole-bag median, pos_diff_mm = || p_mq - p_fq || (raw positions of the two arms of the SAME bag, no alignment), yaw_diff_mrad (wrapped)
  d2_separation.csv        for each --sep-mm threshold: first scan_id with pos_diff_mm >= threshold that stays so for --sep-hold scans (default 1), or NEVER
  d2_window_vs_bags.csv    for each bag (eee_02 first) over scans --win-lo..--win-hi (default 481 540, the approved text): n, and for q_acc / q_gyr the ratio to that bag's own
                           whole-bag median: min, median, mean, max; and the bag's own whole-bag median in absolute units
  d2_report.txt            the above in text, including the full pos_diff curve for the window (one line per scan)
The curve is never summarised away: d2_per_scan.csv holds every scan. scan_index (motion_q_scan) and scan_id (pos_trace) are joined as scan_index = scan_id + --scan-offset
(default 0; the two counters may differ by one, as state_trace's does; check on the first rows of d2_per_scan.csv)."""
import argparse, os, sys
import numpy as np
import pandas as pd

QNAMES = ["motion_q_scan.csv", "motion_q_scan.csv.gz", "motion_q_scan_all.csv.gz"]
PNAMES = ["pos_trace_all.csv.gz", "pos_trace_all.csv", "pos_trace.csv.gz", "pos_trace.csv"]
QCOLS = ["scan_index", "mean_q_acc_x", "mean_q_acc_y", "mean_q_acc_z", "mean_q_gyr_x", "mean_q_gyr_y", "mean_q_gyr_z"]
PCOLS = ["scan_id", "px", "py", "pz", "yaw_rad"]


def find(d, names, required=True):
    for n in names:
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p
    if required:
        raise SystemExit("none of %s in %s" % (names, d))
    return None


def load_q(d, required=True):
    p = find(d, QNAMES, required)
    if p is None:
        return None
    D = pd.read_csv(p)
    miss = [c for c in QCOLS if c not in D.columns]
    if miss:
        raise SystemExit("%s lacks columns %s" % (p, miss))
    D = D.drop_duplicates("scan_index", keep="last").sort_values("scan_index").reset_index(drop=True)
    D["q_acc"] = D[["mean_q_acc_x", "mean_q_acc_y", "mean_q_acc_z"]].mean(axis=1)
    D["q_gyr"] = D[["mean_q_gyr_x", "mean_q_gyr_y", "mean_q_gyr_z"]].mean(axis=1)
    if len(D) == 0:
        raise SystemExit("%s has no rows" % p)
    return D


def load_pos(d):
    p = find(d, PNAMES)
    D = pd.read_csv(p)
    miss = [c for c in PCOLS if c not in D.columns]
    if miss:
        raise SystemExit("%s lacks columns %s" % (p, miss))
    return D.drop_duplicates("scan_id", keep="last").sort_values("scan_id").reset_index(drop=True)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def per_scan(Qm, Qf, Pm, Pf, off):
    T = Qm[["scan_index", "q_acc", "q_gyr", "mean_q_acc_x", "mean_q_acc_y", "mean_q_acc_z", "mean_q_gyr_x", "mean_q_gyr_y", "mean_q_gyr_z"] +
           [c for c in ("model", "n_samples", "mean_dynamic_acc_norm", "mean_dynamic_gyr_norm", "acc_cap_fraction", "gyr_cap_fraction") if c in Qm.columns]].copy()
    T["scan_id"] = T["scan_index"] - off
    T["q_acc_over_own_median"] = T.q_acc / np.median(T.q_acc)
    T["q_gyr_over_own_median"] = T.q_gyr / np.median(T.q_gyr)
    if Qf is not None:
        F = Qf[["scan_index", "q_acc", "q_gyr"]].rename(columns={"q_acc": "fq_q_acc", "q_gyr": "fq_q_gyr"})
        T = T.merge(F, on="scan_index", how="left")
        T["q_acc_mq_over_fq"] = T.q_acc / T.fq_q_acc
        T["q_gyr_mq_over_fq"] = T.q_gyr / T.fq_q_gyr
    M = Pm[PCOLS].rename(columns={c: "mq_" + c for c in PCOLS[1:]})
    Fp = Pf[PCOLS].rename(columns={c: "fq_" + c for c in PCOLS[1:]})
    J = M.merge(Fp, on="scan_id", how="inner")
    J["pos_diff_mm"] = 1000.0 * np.sqrt((J.mq_px - J.fq_px) ** 2 + (J.mq_py - J.fq_py) ** 2 + (J.mq_pz - J.fq_pz) ** 2)
    J["yaw_diff_mrad"] = 1000.0 * wrap(J.mq_yaw_rad - J.fq_yaw_rad)
    T = T.merge(J[["scan_id", "pos_diff_mm", "yaw_diff_mrad"]], on="scan_id", how="outer").sort_values("scan_id").reset_index(drop=True)
    return T


def separation(T, thr, hold):
    d = T.dropna(subset=["pos_diff_mm"]).sort_values("scan_id")
    s, v = d.scan_id.to_numpy(), d.pos_diff_mm.to_numpy()
    for i in range(len(v)):
        if v[i] >= thr and (v[i:i + hold] >= thr).all() and len(v[i:i + hold]) == hold:
            return int(s[i])
    return "NEVER"


def window_stats(name, Q, lo, hi):
    ma, mg = np.median(Q.q_acc), np.median(Q.q_gyr)
    W = Q[(Q.scan_index >= lo) & (Q.scan_index <= hi)]
    r = {"bag": name, "n_scans_in_window": int(len(W)), "bag_median_q_acc": ma, "bag_median_q_gyr": mg}
    for tag, col, med in (("acc", "q_acc", ma), ("gyr", "q_gyr", mg)):
        x = (W[col] / med).to_numpy() if len(W) else np.array([])
        for k, f in (("min", np.min), ("median", np.median), ("mean", np.mean), ("max", np.max)):
            r["ratio_%s_%s" % (tag, k)] = float(f(x)) if len(x) else float("nan")
    return r


def run(a):
    os.makedirs(a.out, exist_ok=True)
    Qm, Qf = load_q(a.mq_dir), load_q(a.fq_dir, required=False)
    Pm, Pf = load_pos(a.mq_dir), load_pos(a.fq_dir)
    T = per_scan(Qm, Qf, Pm, Pf, a.scan_offset)
    T.to_csv(os.path.join(a.out, "d2_per_scan.csv"), index=False, float_format="%.9g")
    rows = [{"threshold_mm": t, "hold_scans": a.sep_hold, "first_scan_id": separation(T, t, a.sep_hold)} for t in a.sep_mm]
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "d2_separation.csv"), index=False)
    W = [window_stats("eee_02", Qm, a.win_lo, a.win_hi)]
    for spec in a.others:
        nm, _, d = spec.partition("=")
        if not d:
            raise SystemExit("--others wants bag=dir, got %r" % spec)
        W.append(window_stats(nm, load_q(d), a.win_lo, a.win_hi))
    pd.DataFrame(W).to_csv(os.path.join(a.out, "d2_window_vs_bags.csv"), index=False, float_format="%.9g")
    L = ["D2 (F-437) eee_02 motion-Q", "motion_q rows: %d (scan_index %d..%d); fq Q file: %s; scan_index = scan_id + %d" %
         (len(Qm), Qm.scan_index.min(), Qm.scan_index.max(), "present" if Qf is not None else "ABSENT", a.scan_offset), ""]
    L += ["separation (|p_mq - p_fq| >= threshold, held %d scans): %s -> %s" % (a.sep_hold, r["threshold_mm"], r["first_scan_id"]) for r in rows]
    L += ["", "window scans %d..%d: Q ratio to each bag's own whole-bag median" % (a.win_lo, a.win_hi), pd.DataFrame(W).to_string(index=False), "",
          "per-scan curve over the window (scan_id, q_acc/own median, q_gyr/own median, pos_diff_mm)"]
    Wn = T[(T.scan_index >= a.win_lo) & (T.scan_index <= a.win_hi)]
    for _, r in Wn.iterrows():
        L.append("%d %.4g %.4g %s" % (r.scan_index, r.q_acc_over_own_median, r.q_gyr_over_own_median,
                                      "nan" if pd.isna(r.pos_diff_mm) else "%.4g" % r.pos_diff_mm))
    open(os.path.join(a.out, "d2_report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    return 0


def fake(d, n, q_fun, pos_fun, model):
    os.makedirs(d, exist_ok=True)
    s = np.arange(1, n + 1)
    Q = pd.DataFrame({"scan_index": s, "t_abs": s * 0.1, "t_rel_s": s * 0.1, "model": model, "beta": 1.0, "acc_scale": 7.0, "gyro_scale": 3.0, "n_samples": 20})
    qa = q_fun(s)
    for c in "xyz":
        Q["mean_q_acc_" + c] = qa * (1.0 if c == "x" else 2.0 if c == "y" else 3.0)
        Q["mean_q_gyr_" + c] = qa * 10
    Q.to_csv(os.path.join(d, "motion_q_scan.csv"), index=False)
    px, py, pz, yaw = pos_fun(s)
    pd.DataFrame({"scan_id": s, "t_abs": s * 0.1, "t_rel_s": s * 0.1, "px": px, "py": py, "pz": pz, "yaw_rad": yaw}).to_csv(
        os.path.join(d, "pos_trace_all.csv.gz"), index=False, compression="gzip")


def selftest():
    import tempfile
    t = tempfile.mkdtemp()
    n = 800
    flat = lambda s: np.full(len(s), 1e-4)
    spike = lambda s: 1e-4 * np.where((s >= 500) & (s <= 520), 40.0, 1.0)   # mq applies 40x Q on 500..520
    z = np.zeros(n)
    base = lambda s: (0.001 * s, np.zeros(len(s)), np.zeros(len(s)), np.zeros(len(s)))
    def shifted(s):                                                       # separates by exactly 20 mm from scan 510 on
        px, py, pz, yw = base(s)
        return px + np.where(s >= 510, 0.020, 0.0), py, pz, yw + np.where(s >= 510, 0.1, 0.0)
    fq, mq, ot = [os.path.join(t, x) for x in ("fq", "mq", "other")]
    fake(fq, n, flat, base, "fixed")
    fake(mq, n, spike, shifted, "motion")
    fake(ot, n, flat, base, "motion")
    out = os.path.join(t, "out")
    a = argparse.Namespace(mq_dir=mq, fq_dir=fq, sep_mm=[10.0, 50.0], sep_hold=3, others=["nya_01=" + ot], out=out, win_lo=481, win_hi=540, scan_offset=0)
    run(a)
    T = pd.read_csv(os.path.join(out, "d2_per_scan.csv"))
    S = pd.read_csv(os.path.join(out, "d2_separation.csv"), dtype=str)
    W = pd.read_csv(os.path.join(out, "d2_window_vs_bags.csv"))
    bad = []
    r = T[T.scan_id == 515].iloc[0]
    if abs(r.pos_diff_mm - 20.0) > 1e-6: bad.append("pos_diff %g" % r.pos_diff_mm)
    if abs(T[T.scan_id == 509].pos_diff_mm.iloc[0]) > 1e-6: bad.append("pos_diff before sep")
    if abs(r.yaw_diff_mrad - 100.0) > 1e-6: bad.append("yaw diff")
    if abs(r.q_acc - 40e-4 * 2.0) > 1e-12: bad.append("q_acc axis mean (got %g)" % r.q_acc)
    if abs(r.q_acc_over_own_median - 40.0) > 1e-9: bad.append("own-median ratio %g" % r.q_acc_over_own_median)
    if abs(r.q_acc_mq_over_fq - 40.0) > 1e-9: bad.append("mq/fq ratio")
    if S.first_scan_id.tolist() != ["510", "NEVER"]: bad.append("separation %s" % S.first_scan_id.tolist())
    e = W[W.bag == "eee_02"].iloc[0]; o = W[W.bag == "nya_01"].iloc[0]
    if abs(e.ratio_acc_max - 40.0) > 1e-9 or abs(e.ratio_acc_min - 1.0) > 1e-9 or e.n_scans_in_window != 60: bad.append("window stats eee_02 %s" % e.to_dict())
    if abs(o.ratio_acc_max - 1.0) > 1e-9: bad.append("other bag ratio")
    # missing pos_trace column and missing fq Q file are handled / refused
    os.remove(os.path.join(fq, "motion_q_scan.csv"))
    a.out = os.path.join(t, "out2"); run(a)
    T2 = pd.read_csv(os.path.join(a.out, "d2_per_scan.csv"))
    if "q_acc_mq_over_fq" in T2.columns: bad.append("fq-absent should have no ratio column")
    pd.read_csv(os.path.join(mq, "pos_trace_all.csv.gz")).drop(columns=["yaw_rad"]).to_csv(os.path.join(mq, "pos_trace_all.csv.gz"), index=False, compression="gzip")
    try:
        run(a); bad.append("missing yaw_rad not refused")
    except SystemExit:
        pass
    print("SELFTEST", "FAIL: " + "; ".join(bad) if bad else "PASS")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mq-dir"); ap.add_argument("--fq-dir")
    ap.add_argument("--sep-mm", type=float, nargs="+", help="separation threshold(s) on |p_mq - p_fq| in mm; NO default (a choice of the analyst)")
    ap.add_argument("--sep-hold", type=int, default=1, help="scans the difference must stay >= threshold (default 1 = first crossing)")
    ap.add_argument("--others", nargs="*", default=[], help="bag=<dir of that bag's motion-Q arm> ...")
    ap.add_argument("--win-lo", type=int, default=481); ap.add_argument("--win-hi", type=int, default=540)
    ap.add_argument("--scan-offset", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.mq_dir and a.fq_dir and a.out and a.sep_mm):
        ap.error("--mq-dir, --fq-dir, --out and --sep-mm are required")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
