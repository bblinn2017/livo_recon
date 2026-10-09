#!/usr/bin/env python3
"""r83_ds_reduce: reduce a residual dump to the downsampling questions (points per voxel, spread inside a voxel, residual distribution, range), family DST.
Python only, no node run. Extended from the draft in r83_ds_test/ (changes listed at the end of this docstring).

  python3 r83_ds_reduce.py --arm <arm> --dump <residual_dump.csv[.gz]> --out <csv> [--require-voxel-layer]
        [--reasons plane_fit_reasons.csv --out-reasons <csv>] [--trace plane_fit_trace.csv --out-trace <csv>]
  python3 r83_ds_reduce.py --selftest

Arm folder files read (DST arms r83_rt_dst_<bag>_<first|off>_<fq|mq>): residual_dump.csv (keys lio/residual_dump_frames, lio/residual_dump_voxel_info true; header, patch
r83_p3: ...,sigma_gate_squared,voxel_point_count,voxel_distinct_frames,voxel_layer; columns are read BY NAME, so the position of voxel_layer does not matter),
plane_fit_reasons.csv (key voxel_map/plane/plane_fit_reasons_en true; written at shutdown by voxelPlaneFitTraceFlush(), src/lio/voxelplane.cpp; columns frame,path,accept,
n_lt_3,solver_fail,eig12_small,floor_reject,non_finite,threshold_fail,denom_reject,ceiling_reject,info_reject,ev10..ev15,retire,subdivide,disabled,ev19..ev23; one row per
frame and path (pca / debiased), counts of fit attempts by outcome over ALL frames of the run, not only the dumped ones), plane_fit_trace.csv (key plane_fit_trace_frames N:
rows only for frames 0..N; columns frame,path,reason,N,F,mean_x,mean_y,mean_z,raw_e0,raw_e1,raw_e2,sens_n_var,used_e0,used_e1,used_e2,e0_final,threshold,was_plane).
`frame` in the dump, the reasons file and the trace is the same index (VoxelMap frame_idx_, src/processing/lio_base.cpp:379 and src/lio/voxelplane.cpp:1383); this
was read from the source, not tested.

OUT 1 (--out, appends; rows of the same arm already in the file are replaced): one row per frame (iteration 0, plane rows row_kind 0 only):
  arm, frame, n_resid, n_planes, absr_mm_q50/q90/q99 (|r| in mm), absz_q50/q90/q99 and z_sd (z = r / sqrt(sigma_squared)), pts_per_plane_q10/q50/q90 (residual rows per plane
  key), planes_ge10 (planes with >= 10 residual points of the frame), insp_sd1_q50, insp_sd2_q50 (median in-plane spreads, m), thick_sd3_q50 (median thickness, m: square roots
  of the eigenvalues of the plane's world-point covariance), range_m_q10/q50/q90 (body-frame range of the residual points, m),
  with voxel_info: vpc_q10/q50/q90 (voxel_point_count over residual rows where it is >= 0), vpc_share_ge50 (share of rows at or above 50, the PCA lock size), n_vpc_unset (rows
  with -1), vdf_q10/q50/q90 (voxel_distinct_frames, same rule);
  with voxel_layer (patch r83_p3): n_resid_layer0 .. n_resid_layer3 (residual rows whose owning node is at that octree layer), n_resid_layer_ge4, n_resid_layer_unset (-1).
OUT 2 (--out-reasons, appended per arm, rows of the arm replaced): arm + every column of plane_fit_reasons.csv for the frames present in the dump (or all frames with
  --reasons-all-frames), both paths.
OUT 3 (--out-trace): per arm, frame, path, reason: n_rows, N_median, F_median, e0_final_median, threshold_median, n_was_plane.
Native units, no ratios. Exit 1 if a requested input is missing or empty.

CHANGES against the draft: (1) --selftest added (the draft had none); (2) required columns checked by name, absent voxel_layer is reported (and is an error with
--require-voxel-layer); (3) voxel_point_count / voxel_distinct_frames quantiles ignore the -1 'unset' value and count it; (4) n_resid_layer_ge4 and n_resid_layer_unset added;
(5) plane_fit_reasons.csv and plane_fit_trace.csv summaries added (the draft did not read them); (6) re-running an arm replaces its rows instead of duplicating them."""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

REQ = ["frame", "iter", "r", "sigma_squared", "plane_key", "wx", "wy", "wz", "bx", "by", "bz"]


def q3(x, pre):
    x = np.asarray(x, float)
    return {pre + "_q10": float(np.quantile(x, .1)), pre + "_q50": float(np.quantile(x, .5)), pre + "_q90": float(np.quantile(x, .9))}


def reduce(df, arm):
    miss = [c for c in REQ if c not in df.columns]
    if miss:
        raise ValueError("dump lacks columns: %s" % miss)
    rows = []
    d0 = df[(df["iter"] == 0)]
    if "row_kind" in d0:
        d0 = d0[d0["row_kind"] == 0]
    for fr, d in d0.groupby("frame"):
        z = d["r"] / np.sqrt(d["sigma_squared"])
        az = np.abs(z)
        cnt = d.groupby("plane_key").size()
        ext = []
        for _, s in d.groupby("plane_key"):
            if len(s) >= 10:
                C = np.cov(s[["wx", "wy", "wz"]].to_numpy().T)
                ev = np.sort(np.linalg.eigvalsh(C))[::-1]
                ext.append((np.sqrt(ev[0]), np.sqrt(ev[1]), np.sqrt(max(ev[2], 0.0))))
        ext = np.array(ext) if ext else np.full((0, 3), np.nan)
        rng = np.linalg.norm(d[["bx", "by", "bz"]].to_numpy(), axis=1)
        a = np.abs(d["r"].to_numpy()) * 1000.0
        row = dict(arm=arm, frame=int(fr), n_resid=len(d), n_planes=len(cnt),
                   absr_mm_q50=np.quantile(a, .5), absr_mm_q90=np.quantile(a, .9), absr_mm_q99=np.quantile(a, .99),
                   absz_q50=np.quantile(az, .5), absz_q90=np.quantile(az, .9), absz_q99=np.quantile(az, .99), z_sd=float(z.std()),
                   pts_per_plane_q10=float(np.quantile(cnt, .1)), pts_per_plane_q50=float(np.quantile(cnt, .5)), pts_per_plane_q90=float(np.quantile(cnt, .9)),
                   planes_ge10=len(ext), insp_sd1_q50=float(np.median(ext[:, 0])) if len(ext) else np.nan,
                   insp_sd2_q50=float(np.median(ext[:, 1])) if len(ext) else np.nan, thick_sd3_q50=float(np.median(ext[:, 2])) if len(ext) else np.nan,
                   range_m_q10=np.quantile(rng, .1), range_m_q50=np.quantile(rng, .5), range_m_q90=np.quantile(rng, .9))
        if "voxel_point_count" in d:
            v = d["voxel_point_count"].to_numpy()
            ok = v[v >= 0]
            if len(ok):
                row.update(q3(ok, "vpc"))
                row["vpc_share_ge50"] = float(np.mean(ok >= 50))
            row["n_vpc_unset"] = int(np.sum(v < 0))
        if "voxel_distinct_frames" in d:
            v = d["voxel_distinct_frames"].to_numpy()
            ok = v[v >= 0]
            if len(ok):
                row.update(q3(ok, "vdf"))
        if "voxel_layer" in d:
            L = d["voxel_layer"].to_numpy()
            for k in (0, 1, 2, 3):
                row["n_resid_layer%d" % k] = int(np.sum(L == k))
            row["n_resid_layer_ge4"] = int(np.sum(L >= 4))
            row["n_resid_layer_unset"] = int(np.sum(L < 0))
        rows.append(row)
    return pd.DataFrame(rows)


def reasons_rows(path, arm, frames):
    r = pd.read_csv(path)
    if frames is not None:
        r = r[r["frame"].isin(frames)]
    r.insert(0, "arm", arm)
    return r


def trace_rows(path, arm):
    t = pd.read_csv(path)
    out = []
    for (fr, pa, re_), s in t.groupby(["frame", "path", "reason"]):
        out.append(dict(arm=arm, frame=fr, path=pa, reason=re_, n_rows=len(s), N_median=float(s["N"].median()), F_median=float(s["F"].median()),
                        e0_final_median=float(s["e0_final"].median()), threshold_median=float(s["threshold"].median()), n_was_plane=int(s["was_plane"].sum())))
    return pd.DataFrame(out)


def write_replacing(df, path, arm):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        old = pd.read_csv(path)
        old = old[old["arm"] != arm]
        df = pd.concat([old, df], ignore_index=True, sort=False)
    df.to_csv(path, index=False)


def selftest():
    fails, n = 0, 0

    def check(name, ok, msg=""):
        nonlocal fails, n
        n += 1
        print("%-56s %s  %s" % (name, "pass" if ok else "FAIL", msg))
        fails += (not ok)
    rng = np.random.default_rng(1)
    rows = []
    for fr in (5, 6):
        for pk in range(40):
            m = 30 if pk < 30 else 6
            c = rng.normal(size=3) * 5
            pts = c + np.stack([rng.normal(size=m) * 0.5, rng.normal(size=m) * 0.4, rng.normal(size=m) * 0.01], 1)
            for i in range(m):
                rows.append(dict(frame=fr, iter=0, r=rng.normal() * 0.01, sigma_squared=1e-4, plane_key=pk, wx=pts[i, 0], wy=pts[i, 1], wz=pts[i, 2], bx=3.0, by=4.0, bz=0.0,
                                 row_kind=0, voxel_point_count=50 if pk < 20 else (-1 if pk == 39 else 10), voxel_distinct_frames=3, voxel_layer=pk % 3 if pk < 39 else -1))
    for i in range(20):   # an iteration 1 row and a line row that must not count
        rows.append(dict(frame=5, iter=1, r=1.0, sigma_squared=1e-4, plane_key=0, wx=0, wy=0, wz=0, bx=0, by=0, bz=0, row_kind=0, voxel_point_count=1, voxel_distinct_frames=1, voxel_layer=0))
        rows.append(dict(frame=5, iter=0, r=1.0, sigma_squared=1e-4, plane_key=0, wx=0, wy=0, wz=0, bx=0, by=0, bz=0, row_kind=2, voxel_point_count=1, voxel_distinct_frames=1, voxel_layer=0))
    df = pd.DataFrame(rows)
    d = tempfile.mkdtemp()
    dump = os.path.join(d, "residual_dump.csv")
    df.to_csv(dump, index=False)
    out = os.path.join(d, "ds.csv")
    r = reduce(pd.read_csv(dump), "a1")
    check("two frames reduced", len(r) == 2, "n_resid %s" % r.n_resid.tolist())
    n_exp = 30 * 30 + 10 * 6
    check("n_resid counts only iteration 0 plane rows", int(r.n_resid.iloc[0]) == n_exp and int(r.n_planes.iloc[0]) == 40, "%d (expected %d)" % (r.n_resid.iloc[0], n_exp))
    check("planes_ge10 = 30 and thickness near 0.01 m", int(r.planes_ge10.iloc[0]) == 30 and abs(r.thick_sd3_q50.iloc[0] - 0.01) < 0.004, "sd3 %.4f" % r.thick_sd3_q50.iloc[0])
    check("range = 5 m", abs(r.range_m_q50.iloc[0] - 5.0) < 1e-9)
    check("voxel_point_count: unset (-1) counted, not in quantiles", int(r.n_vpc_unset.iloc[0]) == 6 and r.vpc_q10.iloc[0] >= 10, "unset %d q10 %.0f" % (r.n_vpc_unset.iloc[0], r.vpc_q10.iloc[0]))
    L = r.iloc[0]
    exp_l = [sum(1 for pk in range(39) if pk % 3 == k for _ in range(30 if pk < 30 else 6)) for k in range(3)]
    check("voxel_layer counts per layer 0..2 and unset", [int(L.n_resid_layer0), int(L.n_resid_layer1), int(L.n_resid_layer2)] == exp_l and int(L.n_resid_layer_unset) == 6 and int(L.n_resid_layer3) == 0, "%s" % exp_l)
    # column absent
    r2 = reduce(pd.read_csv(dump).drop(columns=["voxel_layer"]), "a1")
    check("without voxel_layer the layer columns are absent", "n_resid_layer0" not in r2.columns)
    # header position: voxel_layer first among the three still reads by name
    cols = ["voxel_layer"] + [c for c in df.columns if c != "voxel_layer"]
    r3 = reduce(df[cols], "a1")
    check("column order does not matter", int(r3.n_resid_layer0.iloc[0]) == int(L.n_resid_layer0))
    # reasons and trace
    rs = pd.DataFrame([dict(frame=f, path=p, accept=100 + f, n_lt_3=5, solver_fail=0) for f in (1, 5, 6, 7) for p in ("pca", "debiased")])
    rp = os.path.join(d, "plane_fit_reasons.csv")
    rs.to_csv(rp, index=False)
    rr = reasons_rows(rp, "a1", set(r.frame))
    check("reasons restricted to the dumped frames, both paths", len(rr) == 4 and set(rr.frame) == {5, 6} and rr.arm.eq("a1").all())
    tr = pd.DataFrame([dict(frame=1, path="pca", reason=1, N=10 + i, F=2, mean_x=0, mean_y=0, mean_z=0, raw_e0=1, raw_e1=1, raw_e2=1, sens_n_var=0, used_e0=1, used_e1=1, used_e2=1, e0_final=0.1, threshold=0.2, was_plane=i % 2) for i in range(5)])
    tp = os.path.join(d, "plane_fit_trace.csv")
    tr.to_csv(tp, index=False)
    tt = trace_rows(tp, "a1")
    check("trace summary", len(tt) == 1 and tt.n_rows.iloc[0] == 5 and tt.N_median.iloc[0] == 12 and tt.n_was_plane.iloc[0] == 2)
    # replace rather than duplicate
    write_replacing(r, out, "a1")
    write_replacing(r, out, "a1")
    write_replacing(r.assign(arm="a2"), out, "a2")
    back = pd.read_csv(out)
    check("re-running an arm replaces its rows", len(back) == 4 and (back.arm == "a1").sum() == 2)
    # CLI on the gz file
    import subprocess
    gz = os.path.join(d, "residual_dump_all.csv.gz")
    df.to_csv(gz, index=False)
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--arm", "a3", "--dump", gz, "--out", os.path.join(d, "o3.csv"), "--require-voxel-layer",
                        "--reasons", rp, "--out-reasons", os.path.join(d, "r3.csv"), "--trace", tp, "--out-trace", os.path.join(d, "t3.csv")], capture_output=True, text=True)
    check("CLI on a .gz dump with reasons and trace", p.returncode == 0 and os.path.exists(os.path.join(d, "r3.csv")) and os.path.exists(os.path.join(d, "t3.csv")), (p.stdout + p.stderr).strip()[:120])
    df.drop(columns=["voxel_layer"]).to_csv(os.path.join(d, "nolayer.csv"), index=False)
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--arm", "a5", "--dump", os.path.join(d, "nolayer.csv"), "--out", os.path.join(d, "o5.csv"), "--require-voxel-layer"], capture_output=True, text=True)
    check("--require-voxel-layer fails when the column is absent", p.returncode == 1 and "voxel_layer" in (p.stdout + p.stderr), (p.stdout + p.stderr).strip()[:100])
    print("%d checks, failures: %d" % (n, fails))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm")
    ap.add_argument("--dump")
    ap.add_argument("--out")
    ap.add_argument("--require-voxel-layer", action="store_true")
    ap.add_argument("--reasons")
    ap.add_argument("--out-reasons")
    ap.add_argument("--reasons-all-frames", action="store_true")
    ap.add_argument("--trace")
    ap.add_argument("--out-trace")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.arm and a.dump and a.out):
        ap.error("--arm, --dump, --out are required")
    df = pd.read_csv(a.dump)
    if "voxel_layer" not in df.columns:
        msg = "%s: column voxel_layer absent (patch r83_p3 not applied, or lio/residual_dump_voxel_info false)" % a.arm
        print(("ERROR " if a.require_voxel_layer else "NOTE ") + msg, file=sys.stderr)
        if a.require_voxel_layer:
            return 1
    r = reduce(df, a.arm)
    write_replacing(r, a.out, a.arm)
    print("%s: %d frames reduced" % (a.arm, len(r)))
    rc = 0 if len(r) else 1
    if a.reasons:
        if not (os.path.isfile(a.reasons) and os.path.getsize(a.reasons) > 0 and a.out_reasons):
            print("ERROR %s: plane_fit_reasons.csv missing or empty (or --out-reasons not given)" % a.arm, file=sys.stderr)
            rc = 1
        else:
            rr = reasons_rows(a.reasons, a.arm, None if a.reasons_all_frames else set(int(x) for x in r["frame"]))
            write_replacing(rr, a.out_reasons, a.arm)
            print("%s: %d reasons rows" % (a.arm, len(rr)))
    if a.trace:
        if not (os.path.isfile(a.trace) and os.path.getsize(a.trace) > 0 and a.out_trace):
            print("ERROR %s: plane_fit_trace.csv missing or empty (or --out-trace not given)" % a.arm, file=sys.stderr)
            rc = 1
        else:
            tt = trace_rows(a.trace, a.arm)
            write_replacing(tt, a.out_trace, a.arm)
            print("%s: %d trace summary rows" % (a.arm, len(tt)))
    return rc


if __name__ == "__main__":
    sys.exit(main())
