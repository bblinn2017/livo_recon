#!/usr/bin/env python3
"""R73 (F-416, F-400): per-scan position / yaw trace and per-frame UNALIGNED errors against the stationary reference. Python only; no rerun.

    python3 r73_frame_errors.py --dir <arm dir> --out <out dir> [--arm NAME] [--first 10]

Inputs: state_trace.csv(.gz) (two rows per scan: post_imu, post_lio) and state_reference.txt (the reference pose; eee_01 is stationary for its first ~48 s, so the error of a
scan against it needs no alignment). All values are of the POSTERIOR (post_lio) row.
Writes
  pos_trace_all.csv.gz        scan_id, t_abs, px, py, pz, yaw_rad          (one row per scan; the alignment-free collapse timing of F-400 compares two arms of the SAME bag
                              scan by scan, norm of the position difference; no alignment is involved because both start from the same reference)
  r73_f416_frame_errors.csv   arm, frame(=scan_id), t_abs, ep_x_mm, ep_y_mm, ep_z_mm, ep_norm_mm, eR_x_mrad, eR_y_mrad, eR_z_mrad (= yaw error), yaw_err_mrad, eR_norm_mrad,
                              excursion_10_mm (max ep_norm over scans 1..first, the same on every row of the arm), first_scan, n_scans
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_reference, log_so3  # noqa: E402


def opener(path):
    return path if os.path.exists(path) else (path + ".gz" if os.path.exists(path + ".gz") else None)


def rotations(df):
    q = df[["qw", "qx", "qy", "qz"]].to_numpy(float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def posterior(trace_path):
    df = pd.read_csv(trace_path)
    post = df[df.phase == "post_lio"].drop_duplicates("scan_id", keep="last").sort_values("scan_id")
    return post


def pos_trace(post):
    R = rotations(post)
    return pd.DataFrame({"scan_id": post["scan_id"].astype(int).to_numpy(), "t_abs": post["t_abs"].to_numpy(float), "px": post["px"].to_numpy(float),
                         "py": post["py"].to_numpy(float), "pz": post["pz"].to_numpy(float), "yaw_rad": np.arctan2(R[:, 1, 0], R[:, 0, 0])})


def frame_errors(post, ref_path, arm, first=10):
    Rref, pref, _, _ = read_reference(ref_path)
    R = rotations(post)
    ep = (post[["px", "py", "pz"]].to_numpy(float) - pref) * 1e3
    eR = np.array([log_so3(Rref.T @ R[i]) for i in range(len(R))]) * 1e3
    sid = post["scan_id"].astype(int).to_numpy()
    nrm = np.linalg.norm(ep, axis=1)
    m = (sid >= 1) & (sid <= first)
    exc = float(nrm[m].max()) if m.any() else np.nan
    return pd.DataFrame({"arm": arm, "frame": sid, "t_abs": post["t_abs"].to_numpy(float), "ep_x_mm": ep[:, 0], "ep_y_mm": ep[:, 1], "ep_z_mm": ep[:, 2], "ep_norm_mm": nrm,
                         "eR_x_mrad": eR[:, 0], "eR_y_mrad": eR[:, 1], "eR_z_mrad": eR[:, 2], "yaw_err_mrad": eR[:, 2], "eR_norm_mrad": np.linalg.norm(eR, axis=1),
                         "excursion_10_mm": exc, "first_scan": int(sid[0]), "n_scans": int(len(sid))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--arm", default=None); ap.add_argument("--first", type=int, default=10)
    ap.add_argument("--no-errors", action="store_true", help="write pos_trace_all.csv.gz only (full-bag arms: the error table is r69_reduce's job)")
    a = ap.parse_args()
    arm = a.arm or os.path.basename(os.path.abspath(a.dir))
    tp = opener(os.path.join(a.dir, "state_trace.csv"))
    if not tp:
        print("no state_trace.csv in", a.dir); return 2
    post = posterior(tp)
    if not len(post):
        print("no post_lio rows in", tp); return 2
    os.makedirs(a.out, exist_ok=True)
    pos_trace(post).to_csv(os.path.join(a.out, "pos_trace_all.csv.gz"), index=False, float_format="%.9g", compression="gzip")
    if not a.no_errors:
        rp = os.path.join(a.dir, "state_reference.txt")
        if not os.path.exists(rp):
            print("no state_reference.txt in", a.dir); return 2
        frame_errors(post, rp, arm, a.first).to_csv(os.path.join(a.out, "r73_f416_frame_errors.csv"), index=False, float_format="%.9g")
    return 0


if __name__ == "__main__":
    sys.exit(main())
