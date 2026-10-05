#!/usr/bin/env python3
"""R79 reducers of the early-frame diagnostics (F-464, F-441, F-440). Python only; replaces r78_snapshot_reduce.py in the score job.

    python3 r79_early_reduce.py --dir <arm dir> --out <arm dir> [--snapshots] [--early]
    python3 r79_early_reduce.py --selftest

--snapshots  every voxel-map snapshot of the arm in ONE file, snapshot_frames_all.csv.gz: the post-calibration snapshot (map_frame 1, the map frame 1 queries; file
             post_calibration_voxelmap.csv) and voxelmap_snapshot_f<N>.csv (map_frame N, the map frame N queries; N >= 2, written by voxel_map/map/snapshot_frames).
             Columns: map_frame + the 28 columns of r78_snapshot_reduce.COLS (node_id, layer, status, point_count, eigenvalues, plane test values, centre, normal, d).
             The residual dump of frame N joins to map_frame N by node_id. The raw snapshots (scatter matrices, every column) stay in the lossless host archive.
--early      the first frames of the state and of the update: state_trace_early_all.csv.gz (every column of state_trace.csv, both phases, scan_id <= EARLY_LAST: state, 18 x 18 covariance
             upper triangle P_i_j, residual count, iterations) and lio_update_diag_early_all.csv.gz (every column of lio_update_diag.csv: HtH upper triangle, Htz, step, per iteration,
             frame <= EARLY_LAST).
A missing input, a missing column or an empty result is a nonzero exit (a requested file is never silently absent)."""
import argparse, glob, os, re, sys
import pandas as pd

COLS = ["node_id", "layer", "status", "retired", "initialized", "is_plane", "full", "point_count", "distinct_frames", "eig0", "eig1", "eig2", "edge_m", "planarity_value", "planarity_pass",
        "second_eig_below_threshold", "eigengap_denom1", "eigengap_denom2", "ratio_e1_over_e0", "ratio_e2_over_e1", "rel_gap_threshold", "center_x", "center_y", "center_z",
        "normal_x", "normal_y", "normal_z", "d"]
OUT_COLS = ["map_frame"] + COLS
EARLY_LAST = 10


def snapshot_files(src):
    """-> sorted [(map_frame, path)]: frame 1 is post_calibration_voxelmap.csv, frame N the voxelmap_snapshot_f<N>.csv."""
    out = []
    p1 = os.path.join(src, "post_calibration_voxelmap.csv")
    if os.path.exists(p1):
        out.append((1, p1))
    for p in glob.glob(os.path.join(src, "voxelmap_snapshot_f*.csv")):
        m = re.search(r"voxelmap_snapshot_f(\d+)\.csv$", p)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def reduce_snapshots(src, out):
    fs = snapshot_files(src)
    if not fs:
        raise SystemExit("no post_calibration_voxelmap.csv / voxelmap_snapshot_f*.csv in %s" % src)
    parts = []
    for fr, p in fs:
        D = pd.read_csv(p, usecols=lambda c: c in COLS)
        miss = [c for c in COLS if c not in D.columns]
        if miss:
            raise SystemExit("%s lacks columns %s" % (os.path.basename(p), miss))
        if len(D) == 0:
            raise SystemExit("%s has no rows" % os.path.basename(p))
        D = D[COLS].copy()
        D.insert(0, "map_frame", fr)
        parts.append(D)
    R = pd.concat(parts, ignore_index=True)
    R.to_csv(os.path.join(out, "snapshot_frames_all.csv.gz"), index=False, float_format="%.9g", compression="gzip")
    return {fr: int((R.map_frame == fr).sum()) for fr, _ in fs}


def reduce_early(src, out):
    n = {}
    for name, key, dst in (("state_trace.csv", "scan_id", "state_trace_early_all.csv.gz"), ("lio_update_diag.csv", "frame", "lio_update_diag_early_all.csv.gz")):
        p = os.path.join(src, name)
        if not os.path.exists(p) and os.path.exists(p + ".gz"):
            p += ".gz"
        if not os.path.exists(p):
            raise SystemExit("%s absent in %s" % (name, src))
        keep = []
        for ch in pd.read_csv(p, chunksize=20000):
            if key not in ch.columns:
                raise SystemExit("%s has no column %s" % (name, key))
            keep.append(ch[pd.to_numeric(ch[key], errors="coerce") <= EARLY_LAST])
        D = pd.concat(keep, ignore_index=True)
        if len(D) == 0:
            raise SystemExit("%s has no rows with %s <= %d" % (name, key, EARLY_LAST))
        D.to_csv(os.path.join(out, dst), index=False, float_format="%.9g", compression="gzip")
        n[dst] = int(len(D))
    return n


def selftest():
    import tempfile
    import numpy as np
    fails = []
    t = tempfile.mkdtemp()
    hdr = COLS + ["snapshot_frame", "scatter_00", "root_key_x"]
    for fr, nn in ((1, 7), (2, 9), (3, 11)):
        D = pd.DataFrame({c: np.arange(nn, dtype=float) for c in hdr})
        D.to_csv(os.path.join(t, "post_calibration_voxelmap.csv" if fr == 1 else "voxelmap_snapshot_f%d.csv" % fr), index=False)
    r = reduce_snapshots(t, t)
    R = pd.read_csv(os.path.join(t, "snapshot_frames_all.csv.gz"))
    if r != {1: 7, 2: 9, 3: 11} or list(R.columns) != OUT_COLS or len(R) != 27 or sorted(R.map_frame.unique()) != [1, 2, 3]:
        fails.append("snapshots: %s columns %s" % (r, list(R.columns)[:3]))
    bad = pd.DataFrame({c: np.arange(4, dtype=float) for c in hdr}).drop(columns=["eig1"])
    bad.to_csv(os.path.join(t, "voxelmap_snapshot_f3.csv"), index=False)
    try:
        reduce_snapshots(t, t); fails.append("a snapshot without a column did not fail")
    except SystemExit:
        pass
    pd.DataFrame({c: [] for c in hdr}).to_csv(os.path.join(t, "voxelmap_snapshot_f3.csv"), index=False)
    try:
        reduce_snapshots(t, t); fails.append("an empty snapshot did not fail")
    except SystemExit:
        pass
    e = tempfile.mkdtemp()
    try:
        reduce_snapshots(e, e); fails.append("no snapshot files did not fail")
    except SystemExit:
        pass
    # early
    st = pd.DataFrame({"run_id": "x", "scan_id": np.repeat(np.arange(0, 30), 2), "phase": ["post_imu", "post_lio"] * 30, "P_0_0": 1.0})
    st.to_csv(os.path.join(t, "state_trace.csv"), index=False)
    ud = pd.DataFrame({"frame": np.repeat(np.arange(1, 30), 3), "iter": list(range(3)) * 29, "HtH_00": 1.0})
    ud.to_csv(os.path.join(t, "lio_update_diag.csv"), index=False)
    n = reduce_early(t, t)
    S, U = pd.read_csv(os.path.join(t, "state_trace_early_all.csv.gz")), pd.read_csv(os.path.join(t, "lio_update_diag_early_all.csv.gz"))
    if n != {"state_trace_early_all.csv.gz": 22, "lio_update_diag_early_all.csv.gz": 30} or S.scan_id.max() != EARLY_LAST or U.frame.max() != EARLY_LAST or set(S.phase) != {"post_imu", "post_lio"}:
        fails.append("early: %s max scan %s max frame %s" % (n, S.scan_id.max(), U.frame.max()))
    os.remove(os.path.join(t, "lio_update_diag.csv"))
    try:
        reduce_early(t, t); fails.append("a missing lio_update_diag.csv did not fail")
    except SystemExit:
        pass
    print("SELFTEST r79_early_reduce", "FAIL " + "; ".join(fails) if fails else
          "PASS (3 snapshots -> one file with map_frame 1..3 and %d columns; a missing column, an empty snapshot and no snapshot each fail; early state keeps scans 0..%d of both phases, early update diag frames 1..%d; a missing input fails)" % (len(OUT_COLS), EARLY_LAST, EARLY_LAST))
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir"); ap.add_argument("--out"); ap.add_argument("--snapshots", action="store_true"); ap.add_argument("--early", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not (a.snapshots or a.early):
        sys.exit("nothing to do: give --snapshots and/or --early")
    if a.snapshots:
        print("snapshots", reduce_snapshots(a.dir, a.out or a.dir))
    if a.early:
        print("early", reduce_early(a.dir, a.out or a.dir))
