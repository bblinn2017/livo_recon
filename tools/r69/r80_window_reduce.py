#!/usr/bin/env python3
"""R80 reducer of the takeoff-window dump (F-449a) and the per-message LiDAR log (F-389). Python only; reads the raw logs of ONE arm, writes the compact `*_all` files that r69_collect returns.

    python3 r80_window_reduce.py --dir <arm dir> --out <arm dir> [--window] [--msglog]
    python3 r80_window_reduce.py --selftest

Window (Bryce 2026-10-05: 'state and covariance for pfn1 and pfn3, both Q, at scans 515-530 ... the per-iteration state, the vertical velocity and its variance, and the covariance rows for position and
velocity ... the dumped per-row residuals and weights for ground rows at scans 518-524'). Frame numbers are the map frame index (the value voxel_map->frame_idx_ has DURING the update; frame_stats.txt,
lio_update_diag.csv, residual_dump.csv and iekf_factor_iteration.csv use it). state_trace.csv's scan_id may differ by one; its window is cut one scan wider on each side (WIDE) and r80_gates.py
finds the offset by matching the position.
  window_state_all.csv.gz        state_trace.csv rows, scan_id LO-1 .. HI+1, every column (state, velocity, the 18 x 18 covariance upper triangle P_i_j, both phases)
  window_iter_all.csv.gz         iekf_factor_iteration.csv rows, frames LO .. HI, every column (position and velocity before / after EVERY iteration, step, rank, the 6 x 6 information A0..A35 and b0..b5)
  window_update_diag_all.csv.gz  lio_update_diag.csv rows, frames LO .. HI, every column (per iteration: HtH, Htz, residual-variance decomposition, step)
  window_frame_stats_all.csv.gz  frame_stats.txt rows, frames LO .. HI, every column (rows used, misses by reason: n_miss_coverage, n_miss_mismatch, n_tier0_*, nis, p_pos_vel_fro_pre, S shares)
  window_dump_state_all.csv.gz   residual_dump_state.csv rows, frames LO .. HI (position and rotation vector at the start of each dumped iteration)
  window_row_summary_all.csv     one row per (frame, iter, class) for frames LO .. HI: rows, sum of weights, mean r^2 / sigma^2, rows with r^2 > 9 sigma^2 (the WEIGHT variance), rows with r^2 > 9 sigma_gate^2 (the
                                 variance the gate used: must be 0), sigma_gate / sigma^2 / plane-variance / prior-pose quantiles. class = ground (|world normal z| >= NZ_GROUND) or other.
  window_ground_rows_all.csv.gz  every GROUND row of frames GLO .. GHI (both dumped iterations): frame, iter, idx, source_index, t, r, sigma_squared, plane_var_term, floor_term, sigma_diag_squared, s_prior_pose, sigma_gate_squared,
                                 plane_key, node_id, tier, vis, normal (hpos_*), world point (wx, wy, wz), row_kind, w = 1 / sigma_squared, r2_over_sigma2.
Msg log:
  lidar_msg_log_all.csv.gz       lidar_msg_log.csv, every row, every column, lossless (strings)
A missing input, a missing column or an empty result is a nonzero exit (a requested file is never silently absent)."""
import argparse, os, sys
import numpy as np
import pandas as pd

LO, HI = 515, 530
GLO, GHI = 518, 524
WIDE = 1
NZ_GROUND = 0.8
SIGMA_NUM = 3.0     # lio gate: r^2 <= sigma_num_squared (9) * (...); the 9 is the gate's own value, recorded here only to count the rows that exceed it
GROUND_COLS = ["frame", "iter", "idx", "source_index", "t", "r", "sigma_squared", "plane_var_term", "floor_term", "sigma_diag_squared", "s_prior_pose", "plane_key", "node_id", "tier", "vis",
               "hpos_x", "hpos_y", "hpos_z", "wx", "wy", "wz", "row_kind", "sigma_gate_squared"]
GROUND_OUT = GROUND_COLS + ["w", "r2_over_sigma2"]
SUMMARY_COLS = ["frame", "iter", "class", "n_rows", "sum_weight", "mean_r2_over_sigma2", "n_r2_over_9sigma2", "n_r2_over_9gate", "sigma_gate_p10", "sigma_gate_p50", "sigma_gate_p90", "sigma2_p10", "sigma2_p50", "sigma2_p90", "plane_var_p10", "plane_var_p50", "plane_var_p90",
                "s_prior_pose_p50", "n_nodes"]
MSG_COLS = ["msg_index", "seq", "header_stamp", "scan_time", "gap_header_s", "n_raw", "n_kept", "t_first", "t_last", "off_first_s", "off_last_s", "span_s", "prev_accepted_last_t", "overlap_s", "verdict"]


def _path(src, name):
    p = os.path.join(src, name)
    if os.path.exists(p):
        return p
    if os.path.exists(p + ".gz"):
        return p + ".gz"
    raise SystemExit("%s absent in %s" % (name, src))


def _window(src, name, key, lo, hi, dst, out, need=()):
    p = _path(src, name)
    keep = []
    for ch in pd.read_csv(p, chunksize=50000):
        if key not in ch.columns:
            raise SystemExit("%s has no column %s" % (name, key))
        miss = [c for c in need if c not in ch.columns]
        if miss:
            raise SystemExit("%s lacks columns %s" % (name, miss))
        k = pd.to_numeric(ch[key], errors="coerce")
        keep.append(ch[(k >= lo) & (k <= hi)])
    D = pd.concat(keep, ignore_index=True)
    if len(D) == 0:
        raise SystemExit("%s has no rows with %s in [%d, %d]" % (name, key, lo, hi))
    D.to_csv(os.path.join(out, dst), index=False, float_format="%.12g", compression="gzip")
    return int(len(D))


def _quant(x, q):
    return float(np.quantile(x, q)) if len(x) else float("nan")


def reduce_dump(src, out):
    p = _path(src, "residual_dump.csv")
    need = GROUND_COLS + ["frame"]
    ground, summ = [], []
    acc = {}
    for ch in pd.read_csv(p, chunksize=200000):
        miss = [c for c in GROUND_COLS if c not in ch.columns]
        if miss:
            raise SystemExit("residual_dump.csv lacks columns %s (R80 build needed for floor_term, sigma_diag_squared, s_prior_pose)" % miss)
        ch = ch[(ch["frame"] >= LO) & (ch["frame"] <= HI)]
        if ch.empty:
            continue
        ch = ch.copy()
        ch["w"] = 1.0 / ch["sigma_squared"]
        ch["r2_over_sigma2"] = ch["r"] ** 2 / ch["sigma_squared"]
        ch["class"] = np.where(ch["hpos_z"].abs() >= NZ_GROUND, "ground", "other")
        for (fr, it, cl), g in ch.groupby(["frame", "iter", "class"]):
            acc.setdefault((fr, it, cl), []).append(g)
        gr = ch[(ch["class"] == "ground") & (ch["frame"] >= GLO) & (ch["frame"] <= GHI)]
        if len(gr):
            ground.append(gr[GROUND_OUT])
    if not acc:
        raise SystemExit("residual_dump.csv has no rows for frames %d..%d" % (LO, HI))
    for (fr, it, cl), parts in sorted(acc.items()):
        g = pd.concat(parts)
        s2, pv = g["sigma_squared"].to_numpy(), g["plane_var_term"].to_numpy()
        summ.append(dict(frame=int(fr), iter=int(it), **{"class": cl}, n_rows=int(len(g)), sum_weight=float(g["w"].sum()), mean_r2_over_sigma2=float(g["r2_over_sigma2"].mean()),
                         n_r2_over_9sigma2=int((g["r2_over_sigma2"] > SIGMA_NUM ** 2).sum()), n_r2_over_9gate=int((g["r"] ** 2 > SIGMA_NUM ** 2 * g["sigma_gate_squared"] * (1 + 1e-9)).sum()),
                         sigma_gate_p10=_quant(g["sigma_gate_squared"].to_numpy(), .1), sigma_gate_p50=_quant(g["sigma_gate_squared"].to_numpy(), .5), sigma_gate_p90=_quant(g["sigma_gate_squared"].to_numpy(), .9), sigma2_p10=_quant(s2, .1), sigma2_p50=_quant(s2, .5), sigma2_p90=_quant(s2, .9),
                         plane_var_p10=_quant(pv, .1), plane_var_p50=_quant(pv, .5), plane_var_p90=_quant(pv, .9), s_prior_pose_p50=_quant(g["s_prior_pose"].to_numpy(), .5),
                         n_nodes=int(g["node_id"].nunique())))
    S = pd.DataFrame(summ)[SUMMARY_COLS]
    S.to_csv(os.path.join(out, "window_row_summary_all.csv"), index=False, float_format="%.9g")
    if not ground:
        raise SystemExit("no ground rows (|normal z| >= %.2f) in frames %d..%d of residual_dump.csv" % (NZ_GROUND, GLO, GHI))
    G = pd.concat(ground, ignore_index=True)
    G.to_csv(os.path.join(out, "window_ground_rows_all.csv.gz"), index=False, float_format="%.9g", compression="gzip")
    return {"summary_rows": int(len(S)), "ground_rows": int(len(G))}


def reduce_window(src, out):
    n = {}
    n["window_state_all.csv.gz"] = _window(src, "state_trace.csv", "scan_id", LO - WIDE, HI + WIDE, "window_state_all.csv.gz", out)
    n["window_iter_all.csv.gz"] = _window(src, "iekf_factor_iteration.csv", "scan_id", LO, HI, "window_iter_all.csv.gz", out, need=("v_before_x", "v_after_z"))
    n["window_update_diag_all.csv.gz"] = _window(src, "lio_update_diag.csv", "frame", LO, HI, "window_update_diag_all.csv.gz", out)
    n["window_frame_stats_all.csv.gz"] = _window(src, "frame_stats.txt", "frame_idx", LO, HI, "window_frame_stats_all.csv.gz", out, need=("n_residuals", "n_miss_coverage", "n_miss_mismatch"))
    n["window_dump_state_all.csv.gz"] = _window(src, "residual_dump_state.csv", "frame", LO, HI, "window_dump_state_all.csv.gz", out)
    n.update(reduce_dump(src, out))
    return n


def reduce_msglog(src, out):
    p = _path(src, "lidar_msg_log.csv")
    D = pd.read_csv(p, dtype=str)
    miss = [c for c in MSG_COLS if c not in D.columns]
    if miss:
        raise SystemExit("lidar_msg_log.csv lacks columns %s" % miss)
    if len(D) == 0:
        raise SystemExit("lidar_msg_log.csv has no rows")
    D.to_csv(os.path.join(out, "lidar_msg_log_all.csv.gz"), index=False, compression="gzip")
    return int(len(D))


# ------------------------------------------------------------------------------------------------ selftest
def fake_arm(d, ground_only_in=None, drop=None):
    """Writes a synthetic arm dir: frames 505..540, 3 iterations, 200 rows per (frame, iter), 40 percent ground."""
    rng = np.random.default_rng(3)
    frames = range(505, 541)
    st = pd.DataFrame([dict(run_id="x", scan_id=f, phase=ph, px=f * 0.01, vz=0.1, P_0_0=1.0, P_5_5=2.0) for f in frames for ph in ("post_imu", "post_lio")])
    it = pd.DataFrame([dict(scan_id=f, iter=i, v_before_x=0., v_after_z=0.1, A0=1.0) for f in frames for i in range(3)])
    ud = pd.DataFrame([dict(frame=f, iter=i, n_resid=200, HtH_00=1.0) for f in frames for i in range(3)])
    fs = pd.DataFrame([dict(t=1.0, frame_idx=f, n_residuals=200, n_miss_coverage=3, n_miss_mismatch=4) for f in frames])
    ds = pd.DataFrame([dict(frame=f, iter=i, n_resid=200, px=0.0) for f in frames for i in (0, 2)])
    rows = []
    for f in frames:
        for i in (0, 2):
            for k in range(200):
                nz = 0.95 if k % 5 < 2 else 0.2
                s2 = 0.01 * (1 + k % 7)
                rows.append(dict(frame=f, iter=i, idx=k, source_index=k, t=0.0, r=0.05 * (1 if k % 11 else 8), sigma_squared=s2, plane_var_term=0.001 * (1 + k % 3), plane_key=1, node_id=k % 17, tier=0, vis=0,
                                 hpos_x=0.0, hpos_y=np.sqrt(1 - nz * nz), hpos_z=nz, wx=0., wy=0., wz=0., row_kind=0, floor_term=0.005, sigma_diag_squared=s2 - 0.005 - 0.001, s_prior_pose=0.001, sigma_gate_squared=6 * s2))
    dd = pd.DataFrame(rows)
    ml = pd.DataFrame([dict(zip(MSG_COLS, [m, m, 1.0 + m * .1, 0.9 + m * .1, .1, 20000, 9000, 1.0, 1.1, 0.0, 0.1, 0.1, 1.0, -0.01, 0])) for m in range(50)])
    for name, df in (("state_trace.csv", st), ("iekf_factor_iteration.csv", it), ("lio_update_diag.csv", ud), ("frame_stats.txt", fs), ("residual_dump_state.csv", ds), ("residual_dump.csv", dd), ("lidar_msg_log.csv", ml)):
        if drop and name == drop:
            continue
        df.to_csv(os.path.join(d, name), index=False)


def selftest():
    import tempfile
    fails = []
    d = tempfile.mkdtemp()
    fake_arm(d)
    n = reduce_window(d, d)
    ST = pd.read_csv(os.path.join(d, "window_state_all.csv.gz"))
    IT = pd.read_csv(os.path.join(d, "window_iter_all.csv.gz"))
    UD = pd.read_csv(os.path.join(d, "window_update_diag_all.csv.gz"))
    FS = pd.read_csv(os.path.join(d, "window_frame_stats_all.csv.gz"))
    GR = pd.read_csv(os.path.join(d, "window_ground_rows_all.csv.gz"))
    SM = pd.read_csv(os.path.join(d, "window_row_summary_all.csv"))
    if ST.scan_id.min() != LO - WIDE or ST.scan_id.max() != HI + WIDE or set(ST.phase) != {"post_imu", "post_lio"}:
        fails.append("state window %s..%s" % (ST.scan_id.min(), ST.scan_id.max()))
    if IT.scan_id.min() != LO or IT.scan_id.max() != HI or len(IT) != 3 * (HI - LO + 1):
        fails.append("iter window rows %d" % len(IT))
    if UD.frame.min() != LO or UD.frame.max() != HI or FS.frame_idx.min() != LO or FS.frame_idx.max() != HI:
        fails.append("update-diag / frame-stats window")
    if GR.frame.min() != GLO or GR.frame.max() != GHI or (GR.hpos_z.abs() < NZ_GROUND).any() or set(GR.iter) != {0, 2}:
        fails.append("ground rows: frames %s..%s, iters %s" % (GR.frame.min(), GR.frame.max(), sorted(set(GR.iter))))
    if len(GR) != (GHI - GLO + 1) * 2 * 80:
        fails.append("ground row count %d (want %d)" % (len(GR), (GHI - GLO + 1) * 2 * 80))
    if not np.allclose(GR.w, 1.0 / GR.sigma_squared) or not np.allclose(GR.r2_over_sigma2, GR.r ** 2 / GR.sigma_squared):
        fails.append("ground w / r2_over_sigma2")
    if (SM.n_r2_over_9gate != 0).any() or not (SM.sigma_gate_p50 > SM.sigma2_p50).all():
        fails.append("gate columns")
    if set(SM["class"]) != {"ground", "other"} or SM.frame.min() != LO or SM.frame.max() != HI or SM.n_rows.sum() != (HI - LO + 1) * 2 * 200 or (SM.n_r2_over_9sigma2 <= 0).all():
        fails.append("summary")
    if list(SM.columns) != SUMMARY_COLS or list(GR.columns) != GROUND_OUT:
        fails.append("column order")
    if reduce_msglog(d, d) != 50 or len(pd.read_csv(os.path.join(d, "lidar_msg_log_all.csv.gz"))) != 50:
        fails.append("msg log")
    # failures
    for drop in ("iekf_factor_iteration.csv", "residual_dump.csv", "frame_stats.txt", "lidar_msg_log.csv"):
        e = tempfile.mkdtemp(); fake_arm(e, drop=drop)
        try:
            if drop == "lidar_msg_log.csv":
                reduce_msglog(e, e)
            else:
                reduce_window(e, e)
            fails.append("missing %s did not fail" % drop)
        except SystemExit:
            pass
    e = tempfile.mkdtemp(); fake_arm(e)
    x = pd.read_csv(os.path.join(e, "residual_dump.csv")).drop(columns=["s_prior_pose"]); x.to_csv(os.path.join(e, "residual_dump.csv"), index=False)
    try:
        reduce_window(e, e); fails.append("a dump without s_prior_pose (old build) did not fail")
    except SystemExit:
        pass
    e = tempfile.mkdtemp(); fake_arm(e)
    x = pd.read_csv(os.path.join(e, "residual_dump.csv")); x["hpos_z"] = 0.1; x.to_csv(os.path.join(e, "residual_dump.csv"), index=False)
    try:
        reduce_window(e, e); fails.append("a dump without ground rows did not fail")
    except SystemExit:
        pass
    print("SELFTEST r80_window_reduce", "FAIL " + "; ".join(fails) if fails else
          "PASS (window files cut to frames %d..%d (state %d..%d), ground rows frames %d..%d both iterations only |nz| >= %.1f, summary per frame/iter/class, w and r2/sigma2 derived; missing file, old-build dump, no ground rows and empty msg log each fail)" % (
              LO, HI, LO - WIDE, HI + WIDE, GLO, GHI, NZ_GROUND))
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir"); ap.add_argument("--out"); ap.add_argument("--window", action="store_true"); ap.add_argument("--msglog", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not (a.window or a.msglog):
        sys.exit("nothing to do: give --window and/or --msglog")
    if a.window:
        print("window", reduce_window(a.dir, a.out or a.dir))
    if a.msglog:
        print("msglog", reduce_msglog(a.dir, a.out or a.dir))
