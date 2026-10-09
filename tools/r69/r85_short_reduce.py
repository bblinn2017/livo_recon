#!/usr/bin/env python3
"""R85 (F-526): per-scan residual and voxel-count table of ONE short (60 s) arm. Planner's script; NOT run on real data by the planner.

    python3 r85_short_reduce.py --dir <arm dir> [--out <arm dir>]

Why this exists: the listener's score job runs no reducer for tier-EF (short) arms, so a 60 s arm would otherwise return no per-scan residual or voxel count (those live in
state_trace.csv and frame_stats.txt, which are deleted from the host after the lossless archive). r74_score_job.py (R85 copy, r85_score_job.py) calls this script for every arm
whose arm_table `cell` column is non-empty (the 144 SHT arms and the smoke arms that carry a cell letter).

Inputs (all written by the node for every arm): state_trace.csv (post_lio rows: scan_id, t_abs, residual_count, px, py, pz), state_reference.txt (first-frame reference pose),
frame_stats.txt (frame_idx, n_residuals, n_voxels, n_voxels_is_plane, iters, dx_pos_mm, nis, reduced_chi2; frame_idx = scan_id, as r69_reduce.py reindexes it).
Output: <out>/short_scan_all.csv.gz, one row per scan, columns
  scan, t_s (seconds since the first scan), residual_count (state_trace), fs_n_residuals, fs_n_voxels, fs_n_voxels_is_plane, fs_iters, fs_dx_pos_mm, fs_nis, fs_reduced_chi2,
  pos_err_mm (norm of p_post_lio - p_reference in mm; the first 48 s of every bag are stationary, so this is a drift reading, not an ATE)
Exit code 1 (loud) when a source is missing, a column is absent, or fs_n_residuals is null on every scan. No interpretation; facts only.
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_reference  # noqa: E402

FS = {"fs_n_residuals": "n_residuals", "fs_n_voxels": "n_voxels", "fs_n_voxels_is_plane": "n_voxels_is_plane", "fs_iters": "iters", "fs_dx_pos_mm": "dx_pos_mm",
      "fs_nis": "nis", "fs_reduced_chi2": "reduced_chi2"}
COLS = ["scan", "t_s", "residual_count"] + list(FS) + ["pos_err_mm"]


def reduce_dir(d, out):
    tp, rp, fp = (os.path.join(d, n) for n in ("state_trace.csv", "state_reference.txt", "frame_stats.txt"))
    for p in (tp, rp, fp):
        if not os.path.exists(p):
            print("ERROR r85_short_reduce: %s missing" % p)
            return 1
    df = pd.read_csv(tp)
    for c in ("phase", "scan_id", "t_abs", "residual_count", "px", "py", "pz"):
        if c not in df.columns:
            print("ERROR r85_short_reduce: state_trace.csv has no column %s" % c)
            return 1
    lio = df[df.phase == "post_lio"].drop_duplicates("scan_id", keep="last").sort_values("scan_id").set_index("scan_id")
    if len(lio) == 0:
        print("ERROR r85_short_reduce: no post_lio rows")
        return 1
    fs = pd.read_csv(fp)
    if "frame_idx" not in fs.columns:
        print("ERROR r85_short_reduce: frame_stats.txt has no frame_idx column")
        return 1
    fs = fs.drop_duplicates("frame_idx", keep="last").set_index("frame_idx")
    scans = lio.index.to_numpy()
    _, pref, _, _ = read_reference(rp)
    t = lio.t_abs.to_numpy(float)
    res = pd.DataFrame({"scan": scans, "t_s": np.round(t - t[0], 4), "residual_count": lio.residual_count.to_numpy(float)})
    for k, c in FS.items():
        res[k] = fs[c].reindex(scans).to_numpy(float) if c in fs.columns else np.nan
    res["pos_err_mm"] = np.round(1000.0 * np.linalg.norm(lio[["px", "py", "pz"]].to_numpy(float) - pref, axis=1), 6)
    res = res[COLS]
    if not np.isfinite(res["fs_n_residuals"].to_numpy(float)).any():
        print("ERROR r85_short_reduce: fs_n_residuals is null on every scan (frame_stats.txt columns: %s)" % list(fs.columns))
        return 1
    res.to_csv(os.path.join(out, "short_scan_all.csv.gz"), index=False, compression="gzip")
    print("short_scan_all.csv.gz: %d scans, fs_n_residuals finite on %d" % (len(res), int(np.isfinite(res["fs_n_residuals"].to_numpy(float)).sum())))
    return 0


def selftest():
    import tempfile
    d = tempfile.mkdtemp(prefix="r85_short_selftest_")
    n = 40
    rows = []
    for k in range(1, n + 1):
        for ph in ("post_imu", "post_lio"):
            rows.append(dict(phase=ph, scan_id=k, t_abs=100.0 + 0.1 * k, residual_count=1000 + k, px=0.001 * k, py=0.0, pz=0.0))
    pd.DataFrame(rows).to_csv(os.path.join(d, "state_trace.csv"), index=False)
    pd.DataFrame(dict(frame_idx=list(range(0, n + 1)) + [n], n_residuals=[2000 + i for i in range(n + 1)] + [9], n_voxels=5, n_voxels_is_plane=3, iters=2, dx_pos_mm=0.1, nis=1.0, reduced_chi2=1.0)).to_csv(os.path.join(d, "frame_stats.txt"), index=False)
    open(os.path.join(d, "state_reference.txt"), "w").write("q_wxyz 1 0 0 0\np 0 0 0\nv 0 0 0\nP0_dim 9\nP0_row_major " + " ".join(["0"] * 81) + "\n")
    ok = 0
    assert reduce_dir(d, d) == 0; ok += 1
    t = pd.read_csv(os.path.join(d, "short_scan_all.csv.gz"))
    assert list(t.columns) == COLS and len(t) == n; ok += 1
    assert t.fs_n_residuals.iloc[4] == 2005 and t.residual_count.iloc[4] == 1005, "duplicate frame_idx must keep the last row and scan k must map to frame k"; ok += 1
    assert abs(t.pos_err_mm.iloc[9] - 10.0) < 1e-6 and abs(t.t_s.iloc[9] - 0.9) < 1e-9; ok += 1
    os.remove(os.path.join(d, "frame_stats.txt"))
    assert reduce_dir(d, d) == 1; ok += 1
    pd.DataFrame(dict(frame_idx=[1, 2], other=[1, 2])).to_csv(os.path.join(d, "frame_stats.txt"), index=False)
    assert reduce_dir(d, d) == 1; ok += 1
    print("selftest checks passed: %d\nfailures: 0\nselftest OK (synthetic fixture; planner sandbox, not evidence)" % ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.dir:
        ap.error("--dir is required")
    sys.exit(reduce_dir(a.dir, a.out or a.dir))


if __name__ == "__main__":
    main()
