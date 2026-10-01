#!/usr/bin/env python3
"""R72 (F-396): per-frame early-frame table for one EF arm (eee_01, calibration length x ds mode x pfn x Q mode).

    python3 r72_early_frames.py --dir <arm dir> --out <arm dir> [--ground 0.8] [--wall 0.35]

Inputs (all written by the existing diagnostics; none is new C++): residual_dump.csv (iteration 0 only), lio_update_diag.csv (every iteration),
optional plane_fit_trace.csv (counted, not interpreted).
Definitions (documented here, not tuned): ground residual |nz| >= GROUND, wall residual |nz| <= WALL, with n = (hpos_x, hpos_y, hpos_z) the world-frame plane
normal of the residual; the rest are slanted. Yaw = rotation axis 2 (as r71_residual_info.py). Believed yaw sd = sqrt of the (2,2) entry of the pseudo-inverse of
Lambda = sum h h^T / sigma^2 over the frame's iteration-0 residuals (LiDAR-only, prior ignored), in mrad.
Writes r72_early_frames.csv (one row per frame) and r72_early_frames.json (first frame with >=1/10/100 wall residuals, frames where the yaw step changes sign, ...).
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd


def pinv_psd(L, rel=1e-10):
    w, V = np.linalg.eigh(0.5 * (L + L.T))
    keep = w > rel * max(w.max(), 1e-300)
    inv = (V[:, keep] / w[keep]) @ V[:, keep].T
    return inv, int(keep.sum())


def frame_table(dump, diag, ground=0.8, wall=0.35):
    rows = []
    d0 = dump[dump["iter"] == 0]
    frames = sorted(set(d0["frame"].unique()) | set(diag["frame"].unique()))
    for fr in frames:
        row = {"frame": int(fr)}
        g = d0[d0["frame"] == fr]
        n = len(g)
        row["n_resid_dump"] = n
        if n:
            nz = np.abs(g["hpos_z"].to_numpy(float))
            row["n_ground"] = int((nz >= ground).sum()); row["n_wall"] = int((nz <= wall).sum()); row["n_slanted"] = int(n - row["n_ground"] - row["n_wall"])
            H = g[["hrot_x", "hrot_y", "hrot_z", "hpos_x", "hpos_y", "hpos_z"]].to_numpy(float)
            w = 1.0 / g["sigma_squared"].to_numpy(float)
            L = (H * w[:, None]).T @ H
            inv, rank = pinv_psd(L)
            sd = np.sqrt(np.clip(np.diag(inv), 0, None))
            row["info_rank"] = rank
            row["believed_sd_yaw_mrad"] = float(sd[2] * 1e3); row["believed_sd_roll_mrad"] = float(sd[0] * 1e3); row["believed_sd_pitch_mrad"] = float(sd[1] * 1e3)
            wm = nz <= wall
            if wm.sum() >= 3:
                Lw = (H[wm] * w[wm, None]).T @ H[wm]
                wi, wr = pinv_psd(Lw)
                row["wall_only_believed_sd_yaw_mrad"] = float(np.sqrt(max(wi[2, 2], 0)) * 1e3); row["wall_only_info_rank"] = wr
            row["rms_r_mm"] = float(np.sqrt(np.mean(g["r"].to_numpy(float) ** 2)) * 1e3)
        D = diag[diag["frame"] == fr].sort_values("iter")
        row["n_iter"] = int(len(D))
        if len(D):
            dz = D["dtheta_z"].to_numpy(float)
            row["yaw_step_iter0_mrad"] = float(dz[0] * 1e3)
            row["yaw_step_sum_mrad"] = float(dz.sum() * 1e3)
            row["yaw_step_last_mrad"] = float(dz[-1] * 1e3)
            th = D[["dtheta_x", "dtheta_y", "dtheta_z"]].to_numpy(float)
            row["rot_step_norm_iter0_mrad"] = float(np.linalg.norm(th[0]) * 1e3)
            row["rot_step_norm_last_mrad"] = float(np.linalg.norm(th[-1]) * 1e3)
            dt = D[["dt_x", "dt_y", "dt_z"]].to_numpy(float)
            row["pos_step_norm_iter0_mm"] = float(np.linalg.norm(dt[0]) * 1e3)
            row["pos_step_norm_last_mm"] = float(np.linalg.norm(dt[-1]) * 1e3)
            row["yaw_step_sign_changes"] = int((np.diff(np.sign(dz[np.abs(dz) > 0])) != 0).sum()) if len(dz) > 1 else 0
        rows.append(row)
    return pd.DataFrame(rows)


def summary(T):
    out = {}
    for k in (1, 10, 100):
        hit = T[T.get("n_wall", pd.Series(dtype=float)) >= k]
        out["first_frame_wall_ge_%d" % k] = int(hit["frame"].iloc[0]) if len(hit) else None
    if "believed_sd_yaw_mrad" in T:
        out["believed_sd_yaw_mrad_frames_1_5"] = [None if pd.isna(x) else float(x) for x in T["believed_sd_yaw_mrad"].head(5)]
        out["first_frame_believed_sd_yaw_below_1mrad"] = next((int(f) for f, s in zip(T["frame"], T["believed_sd_yaw_mrad"]) if s < 1.0), None)
    if "yaw_step_sum_mrad" in T:
        out["abs_yaw_step_sum_mrad_max"] = float(T["yaw_step_sum_mrad"].abs().max())
        out["abs_yaw_step_sum_mrad_frames_1_5"] = [None if pd.isna(x) else float(abs(x)) for x in T["yaw_step_sum_mrad"].head(5)]
    out["n_frames"] = int(len(T))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ground", type=float, default=0.8); ap.add_argument("--wall", type=float, default=0.35)
    a = ap.parse_args()
    dp, gp = os.path.join(a.dir, "residual_dump.csv"), os.path.join(a.dir, "lio_update_diag.csv")
    if not os.path.exists(dp) or not os.path.exists(gp):
        print("missing residual_dump.csv or lio_update_diag.csv in", a.dir); return 2
    T = frame_table(pd.read_csv(dp), pd.read_csv(gp), a.ground, a.wall)
    T.to_csv(os.path.join(a.out, "r72_early_frames.csv"), index=False)
    S = summary(T)
    S["definitions"] = {"ground_abs_nz_ge": a.ground, "wall_abs_nz_le": a.wall, "yaw_axis": "rotation axis 2", "believed_sd": "LiDAR-only pinv(Lambda), iteration 0"}
    json.dump(S, open(os.path.join(a.out, "r72_early_frames.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
