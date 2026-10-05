#!/usr/bin/env python3
"""R71 per-residual information analysis of `residual_dump.csv` (lio/residual_dump_frames) and cross-frame LiDAR-only increments from
`lio_update_diag.csv`.  Python only, offline, no estimator numerics.  Closes F-367/F-368 follow-ups (F-374, F-375, F-376, F-377 in followups.md).

    python3 r71_residual_info.py dump  --dump residual_dump.csv [--state residual_dump_state.csv] --out OUTDIR
    python3 r71_residual_info.py cross --diag lio_update_diag.csv|.csv.gz --out OUTDIR [--max-frame 500] [--state state_trace.csv|.csv.gz]

dump writes   residual_info_frames.csv        one row per (frame, iteration): believed information, leverage, split tests, ICC, lag correlations
              residual_info_axes.csv          one row per (frame, iteration, split, axis): believed sd, observed split-half scatter, ratio
              residual_info_planes.csv.gz     one row per (frame, plane): n, info share, mean world point and normal, mean standardised residual, leverage
              residual_info_persistence.csv   per plane seen in >= 2 dumped frames: sign persistence of its mean standardised residual (map error)
              residual_info_whatif.csv        believed sd after offline reweightings (plane-normalised, sqrt-plane-normalised)
cross writes  crossframe_axes.csv            per axis: believed single-frame sd, realised scatter of the LiDAR-only increment, mean, lag-1, cumulative drift.
                                             R80 (F-381): with --state, the believed sd also carries the prior term. The increment d = Lambda^+ g is the LiDAR-only correction FROM the propagated
                                             prior, so it contains the prior's own error: Var(d) = Lambda^+ + P_pred (6 x 6, rotation then position, the post_imu covariance of the same scan).
                                             Columns believed_total_sd_median and realised_over_believed_total are then the consistent ratio; without --state they are blank and
                                             realised_over_believed stays what it was (Lambda^+ only: PROVISIONAL, it overstates the ratio by the prior share).
              crossframe_frames.csv.gz       per frame: LiDAR-only increment per axis and its believed sd

Information layout (lio_base.cpp): residual i has Jacobian row h_i = (hrot_x, hrot_y, hrot_z, hpos_x, hpos_y, hpos_z), residual r_i, variance sigma_squared_i;
Lambda = sum h_i h_i^T / sigma_i^2; g = sum h_i r_i / sigma_i^2; LiDAR-only increment = Lambda^+ g.  Axis 2 of the rotation block is taken as yaw
(vertical); the tool prints a check (information about rotation axis 2 contributed by near-horizontal-normal versus near-vertical-normal residuals) so a wrong
axis assumption is visible in the output rather than silent.
"""
import argparse, gzip, os, sys
import numpy as np
import pandas as pd

AX = ["rot_x", "rot_y", "rot_z", "pos_x", "pos_y", "pos_z"]
import zlib
SEED = 12345                              # F-381: one generator per (frame, iteration, split) so a ratio does not move with the frame set
def split_rng(frame, it, name):
    return np.random.default_rng([SEED, int(frame), int(it), zlib.crc32(name.encode())])


def pinv_psd(L, rel=1e-10):
    L = 0.5 * (L + L.T)
    w, V = np.linalg.eigh(L)
    keep = w > rel * max(w.max(), 1e-300)
    return (V[:, keep] / w[keep]) @ V[:, keep].T, int(keep.sum()), w


def split_axes(H, w, r, mask):
    """Half-split solutions. Returns (delta_a - delta_b, covariance expected if independent) or None when a half is rank deficient."""
    A, B = mask, ~mask
    if A.sum() < 6 or B.sum() < 6:
        return None
    La = (H[A] * w[A, None]).T @ H[A]; Lb = (H[B] * w[B, None]).T @ H[B]
    Ia, ra, _ = pinv_psd(La, 1e-9); Ib, rb, _ = pinv_psd(Lb, 1e-9)
    if ra < 6 or rb < 6:
        return None
    da = Ia @ (H[A].T @ (w[A] * r[A])); db = Ib @ (H[B].T @ (w[B] * r[B]))
    return da - db, Ia + Ib


def analyse_frame(g, nplane_min=3, ndraw=200):
    g = g.reset_index(drop=True)
    N = len(g)
    r = g["r"].to_numpy(float); s2 = g["sigma_squared"].to_numpy(float); w = 1.0 / s2
    H = g[["hrot_x", "hrot_y", "hrot_z", "hpos_x", "hpos_y", "hpos_z"]].to_numpy(float)
    Lam = (H * w[:, None]).T @ H
    Li, rank, ev = pinv_psd(Lam)
    lev = w * np.einsum("ij,jk,ik->i", H, Li, H)
    key = g["plane_key"].to_numpy()
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    row = dict(frame=int(g["frame"].iloc[0]), iter=int(g["iter"].iloc[0]), n_resid=N, n_planes=len(u), rank=rank,
               cond=float(ev.max() / ev[ev > 1e-10 * ev.max()].min()), sum_leverage=float(lev.sum()),
               neff_leverage=float(lev.sum() ** 2 / (lev ** 2).sum()), top1pct_leverage_share=float(np.sort(lev)[::-1][:max(1, N // 100)].sum() / lev.sum()),
               res_per_plane_median=float(np.median(cnt)), res_per_plane_max=int(cnt.max()))
    pl = np.bincount(inv, weights=lev)
    row["neff_plane_leverage"] = float(pl.sum() ** 2 / (pl ** 2).sum())
    delta = Li @ (H.T @ (w * r)); rr = r - H @ delta
    row["chi2_pre_per_resid"] = float(np.mean(w * r * r)); row["chi2_post_per_dof"] = float((w * rr * rr).sum() / max(N - rank, 1))
    row["rms_r_mm"] = float(np.sqrt(np.mean(r * r)) * 1e3); row["rms_sigma_mm"] = float(np.sqrt(np.mean(s2)) * 1e3)
    sd = np.sqrt(np.clip(np.diag(Li), 0, None))
    for a in range(6):
        row["believed_sd_" + AX[a]] = float(sd[a] * (1e3))            # mrad for rotation, mm for position
        row["lidar_only_increment_" + AX[a]] = float(delta[a] * 1e3)
    # information believed per residual and its spread over axes
    claim = w[:, None] * H * H
    for a in range(6):
        row["info_share_top10pct_resid_" + AX[a]] = float(np.sort(claim[:, a])[::-1][:max(1, N // 10)].sum() / max(claim[:, a].sum(), 1e-300))
    # axis check: yaw information from near-vertical-normal residuals should be ~0 if axis 2 is world vertical
    nz = np.abs(g["hpos_z"].to_numpy(float)); vert = nz > 0.9
    row["info_rotz_from_near_vertical_normals_share"] = float(claim[vert, 2].sum() / max(claim[:, 2].sum(), 1e-300))
    row["share_residuals_near_vertical_normals"] = float(vert.mean())
    # standardised residual: ICC within planes, lag correlation in source order
    z = rr * np.sqrt(w)
    big = [np.where(inv == k)[0] for k in range(len(u)) if cnt[k] >= nplane_min]
    if len(big) > 1:
        n_g = np.array([len(i) for i in big]); gm = np.array([z[i].mean() for i in big]); grand = z[np.concatenate(big)].mean()
        msb = (n_g * (gm - grand) ** 2).sum() / (len(big) - 1)
        msw = sum(((z[i] - z[i].mean()) ** 2).sum() for i in big) / max(n_g.sum() - len(big), 1)
        n0 = (n_g.sum() - (n_g ** 2).sum() / n_g.sum()) / (len(big) - 1)
        row["icc_plane"] = float((msb - msw) / (msb + (n0 - 1) * msw)) if (msb + (n0 - 1) * msw) > 0 else np.nan
    order = np.argsort(g["source_index"].to_numpy(float), kind="stable"); zs = z[order]
    for lag in (1, 2, 5, 20):
        row["lag%d_corr_source_order" % lag] = float(np.corrcoef(zs[:-lag], zs[lag:])[0, 1]) if N > lag + 3 else np.nan
    # splits: random residuals, random planes, azimuth halves (front/back), range halves, time halves
    bx = g["bx"].to_numpy(float); by = g["by"].to_numpy(float); bz = g["bz"].to_numpy(float)
    rng_m = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    az = np.arctan2(by, bx)
    splits = {"random_resid": None, "random_plane": None,
              "azimuth_front_back": (np.abs(az) < np.pi / 2), "range_near_far": rng_m < np.median(rng_m),
              "time_halves": g["t"].to_numpy(float) < np.median(g["t"].to_numpy(float)),
              "normal_vertical_vs_horizontal": vert}
    axes_rows = []
    for name, m in splits.items():
        diffs = []; covs = []
        if m is None:
            rng_s = split_rng(row["frame"], row.get("iter", 0), name)
            for _ in range(ndraw):
                mk = (rng_s.random(N) < 0.5) if name == "random_resid" else (rng_s.random(len(u)) < 0.5)[inv]
                res = split_axes(H, w, r, mk)
                if res: diffs.append(res[0]); covs.append(res[1])
        else:
            res = split_axes(H, w, r, np.asarray(m, bool))
            if res: diffs.append(res[0]); covs.append(res[1])
        if not diffs:
            continue
        D = np.array(diffs); C = np.array(covs)
        for a in range(6):
            obs = float(np.mean(D[:, a] ** 2)); exp = float(np.mean(C[:, a, a]))
            axes_rows.append(dict(frame=row["frame"], iter=row["iter"], split=name, axis=AX[a], n_draws=len(D), observed_diff_var=obs,
                                  expected_diff_var_if_independent=exp, ratio_obs_over_expected=obs / exp if exp > 0 else np.nan,
                                  believed_over_delivered_info=(exp / obs) if obs > 0 else np.nan))
        # joint statistic over the six axes (dof 6)
        chi = [float(d @ np.linalg.pinv(c, rcond=1e-9) @ d) for d, c in zip(diffs, covs)]
        row["split_chi2_mean_" + name] = float(np.mean(chi))
    # leave-one-plane-out shifts in believed-sd units (the 20 planes with the largest information)
    top = np.argsort(pl)[::-1][:20]; shifts = []
    for k in top:
        keep = inv != k
        L2 = (H[keep] * w[keep, None]).T @ H[keep]; I2, r2, _ = pinv_psd(L2)
        if r2 < 6:
            shifts.append(np.full(6, np.inf)); continue
        d2 = I2 @ (H[keep].T @ (w[keep] * r[keep]))
        shifts.append(np.abs(d2 - delta) / np.maximum(sd, 1e-300))
    S = np.array(shifts)
    for a in range(6):
        row["loo_plane_max_shift_sd_" + AX[a]] = float(np.max(S[:, a])); row["loo_plane_median_shift_sd_" + AX[a]] = float(np.median(S[:, a]))
    # what-if offline reweighting (never applied to the estimator): equal total weight per plane, and sqrt(n) per plane
    whatif = []
    for nm, scale in (("plane_normalised", cnt[inv].astype(float)), ("plane_sqrt_normalised", np.sqrt(cnt[inv].astype(float)))):
        ww = w / scale
        Lw, rw, _ = pinv_psd((H * ww[:, None]).T @ H)
        d = np.sqrt(np.clip(np.diag(Lw), 0, None))
        whatif.append(dict(frame=row["frame"], iter=row["iter"], scheme=nm, rank=rw,
                           **{"believed_sd_" + AX[a]: float(d[a] * 1e3) for a in range(6)},
                           **{"sd_ratio_vs_actual_" + AX[a]: float(d[a] / max(sd[a], 1e-300)) for a in range(6)}))
    # per-plane table
    P = pd.DataFrame(dict(plane_key=key, lev=lev, z=z, nx=g["hpos_x"], ny=g["hpos_y"], nz=g["hpos_z"], wx=g["wx"], wy=g["wy"], wz=g["wz"],
                          bx=bx, by=by, bz=bz, info=w * (H ** 2).sum(1)))
    PG = P.groupby("plane_key").agg(n=("lev", "size"), leverage=("lev", "sum"), info=("info", "sum"), mean_z=("z", "mean"), nx=("nx", "mean"), ny=("ny", "mean"),
                                    nz=("nz", "mean"), wx=("wx", "mean"), wy=("wy", "mean"), wz=("wz", "mean"), bx=("bx", "mean"), by=("by", "mean"),
                                    bz=("bz", "mean")).reset_index()
    PG.insert(0, "iter", row["iter"]); PG.insert(0, "frame", row["frame"])
    return row, axes_rows, whatif, PG


def run_dump(a):
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.dump)
    fr, ax, wf, pl = [], [], [], []
    for (f, it), g in df.groupby(["frame", "iter"]):
        if len(g) < 12:
            continue
        row, axrows, wi, PG = analyse_frame(g)
        fr.append(row); ax += axrows; wf += wi; pl.append(PG)
    pd.DataFrame(fr).to_csv(os.path.join(a.out, "residual_info_frames.csv"), index=False, float_format="%.6g")
    pd.DataFrame(ax).to_csv(os.path.join(a.out, "residual_info_axes.csv"), index=False, float_format="%.6g")
    pd.DataFrame(wf).to_csv(os.path.join(a.out, "residual_info_whatif.csv"), index=False, float_format="%.6g")
    if pl:
        P = pd.concat(pl); P.to_csv(os.path.join(a.out, "residual_info_planes.csv.gz"), index=False, float_format="%.6g", compression="gzip")
        # persistence: planes seen in >= 2 dumped frames, sign of mean standardised residual
        rows = []
        for k, g in P.groupby("plane_key"):
            if g["frame"].nunique() < 2:
                continue
            s = np.sign(g["mean_z"].to_numpy())
            rows.append(dict(plane_key=k, n_frames=int(g["frame"].nunique()), mean_z_mean=float(g["mean_z"].mean()), mean_z_abs=float(g["mean_z"].abs().mean()),
                             same_sign_all=bool((s == s[0]).all()), total_leverage=float(g["leverage"].sum()), nx=float(g["nx"].mean()), ny=float(g["ny"].mean()),
                             nz=float(g["nz"].mean())))
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "residual_info_persistence.csv"), index=False, float_format="%.6g")
    print("frames analysed:", len(fr))


def load_prior_cov6(path, frames):
    """Rotation-then-position 6 x 6 block of the post_imu covariance (the propagated prior of the LiDAR update) per scan_id == frame. Returns {frame: 6x6} and prints the match count."""
    cols = ["P_%d_%d" % (i, j) for i in range(6) for j in range(i, 6)]
    head = pd.read_csv(path, nrows=1)
    need = ["scan_id", "phase"] + cols
    miss = [c for c in need if c not in head.columns]
    if miss:
        sys.exit("--state lacks columns %s" % miss[:8])
    S = pd.read_csv(path, usecols=need)
    S = S[S["phase"] == "post_imu"].drop_duplicates("scan_id", keep="last")
    out = {}
    for _, r in S.iterrows():
        P = np.zeros((6, 6))
        for i in range(6):
            for j in range(i, 6):
                P[i, j] = P[j, i] = r["P_%d_%d" % (i, j)]
        out[int(r["scan_id"])] = P
    print("prior covariance (post_imu, scan_id == frame): %d scans" % len(out), "of which usable for the frames below are counted at the end")
    return out


def run_cross(a):
    os.makedirs(a.out, exist_ok=True)
    prior = load_prior_cov6(a.state, None) if a.state else None
    df = pd.read_csv(a.diag)
    df = df[df["iter"] == df.groupby("frame")["iter"].transform("min")]       # iteration 0 of every frame
    df = df[df["frame"] <= a.max_frame].sort_values("frame")
    rows = []
    for _, r in df.iterrows():
        L = np.zeros((6, 6)); k = 0
        for i in range(6):
            for j in range(i, 6):
                L[i, j] = L[j, i] = r["HtH_%d%d" % (i, j)]
        g = np.array([r["Htz_%d" % i] for i in range(6)])
        Li, rank, ev = pinv_psd(L)
        d = Li @ g; sd = np.sqrt(np.clip(np.diag(Li), 0, None))
        row = dict(frame=int(r["frame"]), rank=rank, n_resid=int(r["n_resid"]))
        Pp = prior.get(int(r["frame"])) if prior is not None else None
        sdt = np.sqrt(np.clip(np.diag(Li + Pp), 0, None)) if Pp is not None else None   # R80 F-381: prior term (Lambda^+ + P_pred)
        for a_ in range(6):
            row["inc_" + AX[a_]] = d[a_] * 1e3; row["sd_" + AX[a_]] = sd[a_] * 1e3
            row["sdtot_" + AX[a_]] = sdt[a_] * 1e3 if sdt is not None else float("nan")
        rows.append(row)
    F = pd.DataFrame(rows)
    if F.empty:
        print('no iteration-0 rows'); return
    F.to_csv(os.path.join(a.out, "crossframe_frames.csv.gz"), index=False, float_format="%.6g", compression="gzip")
    out = []
    for lo, hi, nm in ((2, 10, "frames_2_10"), (11, 50, "frames_11_50"), (51, 480, "frames_51_480"), (2, 480, "frames_2_480")):
        S = F[(F.frame >= lo) & (F.frame <= hi) & (F["rank"] >= 6)]
        if len(S) < 4:
            continue
        for a_ in range(6):
            x = S["inc_" + AX[a_]].to_numpy(); s = S["sd_" + AX[a_]].to_numpy(); n = len(x)
            st = S["sdtot_" + AX[a_]].to_numpy(); have_t = bool(np.isfinite(st).all()) and n > 0
            out.append(dict(window=nm, axis=AX[a_], n_frames=n, believed_single_frame_sd_median=float(np.median(s)), realised_scatter_sd=float(np.std(x, ddof=1)),
                            believed_total_sd_median=float(np.median(st)) if have_t else float("nan"),
                            realised_over_believed_total=float(np.std(x, ddof=1) / np.median(st)) if have_t else float("nan"),
                            realised_over_believed=float(np.std(x, ddof=1) / np.median(s)), mean_increment=float(x.mean()),
                            mean_over_sd_of_mean=float(x.mean() / (np.std(x, ddof=1) / np.sqrt(n))), mean_over_believed_sd_of_mean=float(x.mean() / (np.median(s) / np.sqrt(n))),
                            lag1_autocorr=float(np.corrcoef(x[:-1], x[1:])[0, 1]), cumulative_drift_end=float(x.sum()),
                            believed_sd_of_mean_if_independent=float(np.median(s) / np.sqrt(n)),
                            realised_sd_of_mean_if_independent=float(np.std(x, ddof=1) / np.sqrt(n))))
    pd.DataFrame(out).to_csv(os.path.join(a.out, "crossframe_axes.csv"), index=False, float_format="%.6g")
    print("frames:", len(F), "full-rank frames:", int((F['rank'] >= 6).sum()), "frames with the prior term:", int(np.isfinite(F["sdtot_rot_x"]).sum()))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("dump"); d.add_argument("--dump", required=True); d.add_argument("--state"); d.add_argument("--out", required=True)
    c = sp.add_parser("cross"); c.add_argument("--diag", required=True); c.add_argument("--out", required=True); c.add_argument("--max-frame", type=int, default=500); c.add_argument("--state", default=None, help="state_trace.csv[.gz]: adds the prior term P_pred (post_imu) to the believed sd (R80, F-381)")
    a = ap.parse_args()
    run_dump(a) if a.cmd == "dump" else run_cross(a)


if __name__ == "__main__":
    main()
