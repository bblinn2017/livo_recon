#!/usr/bin/env python3
"""R78 (F-428): reduce post_calibration_voxelmap.csv to one row per node with the columns of make_r78_manifest.SNAP_COLS -> snapshot_nodes_all.csv.gz.
    python3 r78_snapshot_reduce.py --dir <arm dir> --out <arm dir>
The raw snapshot (every column, scatter matrices) is not kept; this keeps node_id, layer, status, point count, eigenvalues, the plane test values, centre, normal, d.
Selftest: python3 r78_snapshot_reduce.py --selftest (synthetic input with the real header strings; shows a missing column and an empty file both fail)."""
import argparse, os, sys
import pandas as pd

COLS = ["node_id", "layer", "status", "retired", "initialized", "is_plane", "full", "point_count", "distinct_frames", "eig0", "eig1", "eig2", "edge_m", "planarity_value", "planarity_pass",
        "second_eig_below_threshold", "eigengap_denom1", "eigengap_denom2", "ratio_e1_over_e0", "ratio_e2_over_e1", "rel_gap_threshold", "center_x", "center_y", "center_z",
        "normal_x", "normal_y", "normal_z", "d"]


def reduce_dir(src, out):
    p = os.path.join(src, "post_calibration_voxelmap.csv")
    D = pd.read_csv(p, usecols=lambda c: c in COLS)
    miss = [c for c in COLS if c not in D.columns]
    if miss:
        raise SystemExit("post_calibration_voxelmap.csv lacks columns %s" % miss)
    if len(D) == 0:
        raise SystemExit("post_calibration_voxelmap.csv has no rows")
    D = D[COLS]
    D.to_csv(os.path.join(out, "snapshot_nodes_all.csv.gz"), index=False, float_format="%.9g", compression="gzip")
    return len(D)


def selftest():
    import tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np
    t = tempfile.mkdtemp(); fails = []
    hdr = COLS + ["snapshot_frame", "scatter_00", "root_key_x"]
    D = pd.DataFrame({c: np.arange(7, dtype=float) for c in hdr}); D.to_csv(os.path.join(t, "post_calibration_voxelmap.csv"), index=False)
    n = reduce_dir(t, t)
    R = pd.read_csv(os.path.join(t, "snapshot_nodes_all.csv.gz"))
    if n != 7 or list(R.columns) != COLS or len(R) != 7:
        fails.append("good input: rows %d columns %s" % (n, list(R.columns)))
    for name, bad in (("missing column", D.drop(columns=["eigengap_denom2"])), ("empty", D.iloc[:0])):
        bad.to_csv(os.path.join(t, "post_calibration_voxelmap.csv"), index=False)
        try:
            reduce_dir(t, t); fails.append(name + " did not fail")
        except SystemExit:
            pass
    print("SELFTEST r78_snapshot_reduce", "FAIL " + "; ".join(fails) if fails else "PASS (7 rows kept, 28 columns; a missing column and an empty file each fail)")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dir"); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    print(reduce_dir(a.dir, a.out or a.dir), "nodes")
