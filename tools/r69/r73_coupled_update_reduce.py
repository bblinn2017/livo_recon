#!/usr/bin/env python3
"""R73 (F-415): reduce coupled_update_iter.csv / coupled_update_scan.csv of one coupled arm.

    python3 r73_coupled_update_reduce.py --dir <arm dir> --out <arm dir>

The joint update step is delta = lidar_delta + prior_delta (A delta = Gamma-weighted LiDAR term + prior term, split by the code). For every scan the table keeps the FIRST and the LAST
iteration and reports, per block (position all knots, rotation all knots, tail), how much of the step comes from the LiDAR term and how much from the IMU prior, in the block's own unit,
plus the conditioning of A and Gamma_L. Nothing is interpreted.
Writes r73_coupled_update_arm.csv (one row per scan and iteration in {first,last}: frame, iter, which, delta/lidar/prior norms for pos and rot, prior_share_pos = |prior|/(|prior|+|lidar|),
lidar_share_pos, A_cond, Gamma_L_rank, imu_prior_info_scale, n_residuals, scan stop reason and tail/knot traces from the scan file) and r73_coupled_update_arm.json
(max / median of the shares, the 10 scans with the largest position step, the scans with A_cond above 1e12, the stop-reason counts).
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd


def opener(d, n):
    for c in (n, n + ".gz"):
        p = os.path.join(d, c)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pi, ps = opener(a.dir, "coupled_update_iter.csv"), opener(a.dir, "coupled_update_scan.csv")
    if not pi:
        print("no coupled_update_iter.csv in", a.dir); return 2
    I = pd.read_csv(pi)
    first = I.sort_values(["frame", "iter"]).groupby("frame", as_index=False).head(1).assign(which="first")
    last = I.sort_values(["frame", "iter"]).groupby("frame", as_index=False).tail(1).assign(which="last")
    T = pd.concat([first, last]).sort_values(["frame", "iter"]).drop_duplicates(["frame", "iter"])
    out = pd.DataFrame({"frame": T["frame"].astype(int), "iter": T["iter"].astype(int), "which": T["which"], "n_residuals": T["n_residuals"],
                        "imu_prior_info_scale": T["imu_prior_info_scale"], "A_cond": T["A_cond"], "Gamma_L_rank": T["Gamma_L_rank_rel1e-9"]})
    for blk, nm in (("pos", "pos_allknots_m"), ("rot", "rot_allknots_rad")):
        d, l, p = (T["%s_%s" % (v, nm)].to_numpy(float) for v in ("delta", "lidar_delta", "prior_delta"))
        out["delta_%s" % blk], out["lidar_%s" % blk], out["prior_%s" % blk] = d, l, p
        s = np.abs(l) + np.abs(p)
        out["prior_share_%s" % blk] = np.where(s > 0, np.abs(p) / np.where(s > 0, s, 1), np.nan)
        out["lidar_share_%s" % blk] = np.where(s > 0, np.abs(l) / np.where(s > 0, s, 1), np.nan)
    for c in ("delta_pos_tail_m", "lidar_delta_pos_tail_m", "prior_delta_pos_tail_m", "delta_norm_total", "lidar_delta_norm_total", "prior_delta_norm_total", "A_eig_min", "A_eig_max", "Gamma_L_eig_max"):
        out[c] = T[c].to_numpy(float)
    S = None
    if ps:
        S = pd.read_csv(ps)
        keep = ["frame", "stop", "iterations"] + [c for c in S.columns if c.startswith("tail18_") or c.startswith("knot")]
        out = out.merge(S[keep].drop_duplicates("frame", keep="last"), on="frame", how="left")
    out.to_csv(os.path.join(a.out, "r73_coupled_update_arm.csv"), index=False, float_format="%.9g")
    lastrows = out[out["which"] == "last"]
    js = {"n_scans": int(out["frame"].nunique()), "prior_share_pos_last_median": float(lastrows["prior_share_pos"].median()), "prior_share_pos_last_max": float(lastrows["prior_share_pos"].max()),
          "lidar_share_pos_last_median": float(lastrows["lidar_share_pos"].median()), "largest_position_steps": lastrows.nlargest(10, "delta_pos")[["frame", "delta_pos", "prior_pos", "lidar_pos", "A_cond"]].to_dict("records"),
          "scans_A_cond_gt_1e12": int((lastrows["A_cond"] > 1e12).sum()), "imu_prior_info_scale": float(out["imu_prior_info_scale"].iloc[0]) if len(out) else None}
    if S is not None and "stop" in S:
        js["stop_counts"] = S["stop"].value_counts().to_dict()
    json.dump(js, open(os.path.join(a.out, "r73_coupled_update_arm.json"), "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
