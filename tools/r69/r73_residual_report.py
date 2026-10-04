#!/usr/bin/env python3
"""R73 (F-417): HtH / Htz metrics and ALL residual information, with the weight of every residual.

    python3 r73_residual_report.py --dir <arm dir> --out <out dir> [--arm NAME] [--resid-frames 1,2,3,4,5 | all] [--shadow]

Inputs (any of them may be .gz; none is new C++ except the R73 shadow files):
  residual_dump.csv       per-residual rows (frame, iter, idx, source_index, t, r, sigma_squared, plane_var_term, plane_key, node_id, tier, vis, hrot_x..z, hpos_x..z, ...)
  lio_update_diag.csv     per frame and iteration: HtH_ij (upper triangle of the 6x6 over [rot, pos]), Htz_0..5, n_resid, sum_inv_sigma2, ...
  --shadow: the driver is read from residual_dump.csv + shadow_scan.csv (role driver), the SHADOW from shadow_residual_dump.csv + shadow_scan.csv (role shadow);
            shadow_scan.csv carries HtH0_ij / Htz0_i (iteration 0) and HtHlast_ij / Htzlast_i (last iteration) per role.
Definitions (documented, not tuned): weight w = 1/sigma_squared. Class by the world-frame plane normal n = (hpos_x, hpos_y, hpos_z): ground |n_z| >= 0.8, wall |n_z| <= 0.35,
slanted otherwise (the r72_early_frames.py convention). Rotation axis 2 is yaw. Per-residual information I = w h h^T with h = [hrot, hpos]; Htz contribution = w h r.
Writes (columns fixed, every file has the column `arm` and `role`):
  r73_f417_hth_frames.csv   one row per (frame, iter): hth_trace, eig_1..eig_6 (ascending), eig_min, eig_max, cond, info_yaw, info_x, info_y, info_yaw_nowall, info_x_nowall, info_y_nowall,
                            htz_0..5, htz_norm, htz_rot_norm, htz_pos_norm, n_resid, sum_w, yaw_sd_mrad (sqrt of the pinv (2,2) entry), and the same *_nowall from the dump when present
  r73_f417_classes.csv      one row per (frame, iter, cls): n, weight_sum, weight_mean, weight_max, wr2_sum (sum w r^2), hth_trace_share, info_yaw, info_x, info_y, htz_0..5, htz_norm, rms_r_mm
  r73_f417_residuals.csv.gz one row per residual of the selected frames: the dump columns + weight, cls, info_trace, info_rot_trace, info_pos_trace, info_yaw, info_x, info_y, htz_0..5
  r73_f417_consistency.csv  per (frame, iter) present in both: max_rel_err_hth, max_rel_err_htz (error / max |entry| of the logged matrix), htz_sign (+1/-1: which sign
                            of the sum matches the logged Htz), err_sum_w, n_resid_dump, n_resid_logged, and ok (1 when both errors < 1e-3; the logs carry 7 significant digits)
"""
import argparse, os, sys
import numpy as np
import pandas as pd

GROUND, WALL = 0.8, 0.35
HC = ["hrot_x", "hrot_y", "hrot_z", "hpos_x", "hpos_y", "hpos_z"]
IJ = [(i, j) for i in range(6) for j in range(i, 6)]
# per_residual()'s 12 derived columns: rebuildable offline from HC + sigma_squared + r in one per_residual() call,
# so the *_all.csv.gz return copies drop them (weight/cls are kept -- cheap, not part of this list).
PER_RESIDUAL_DERIVED = ["info_trace", "info_rot_trace", "info_pos_trace", "info_yaw", "info_x", "info_y"] + ["htz_%d" % k for k in range(6)]


def classify(nz):
    a = np.abs(nz)
    return np.where(a >= GROUND, "ground", np.where(a <= WALL, "wall", "slanted"))


def mat6(vals):
    M = np.zeros((6, 6))
    for (i, j), v in zip(IJ, vals):
        M[i, j] = M[j, i] = v
    return M


def diag_from_update_diag(path):
    D = pd.read_csv(path)
    out = pd.DataFrame({"frame": D["frame"].astype(int), "iter": D["iter"].astype(int), "n_resid": D["n_resid"], "sum_w": D.get("sum_inv_sigma2", np.nan)})
    for i, j in IJ:
        out["HtH_%d%d" % (i, j)] = D["HtH_%d_%d" % (i, j)] if "HtH_%d_%d" % (i, j) in D else D["HtH_%d%d" % (i, j)]
    for k in range(6):
        out["Htz_%d" % k] = D["Htz_%d" % k]
    return out


def diag_from_shadow_scan(path, role):
    D = pd.read_csv(path)
    D = D[D["role"] == role]
    rows = []
    for _, r in D.iterrows():
        n_it = int(r["n_iter"])
        for tag, it in (("0", 0), ("last", max(n_it - 1, 0))):
            row = {"frame": int(r["frame"]), "iter": it, "n_resid": float(r["n_resid_iter0"]) if tag == "0" else np.nan, "sum_w": np.nan}
            for i, j in IJ:
                row["HtH_%d%d" % (i, j)] = r["HtH%s_%d%d" % (tag, i, j)]
            for k in range(6):
                row["Htz_%d" % k] = r["Htz%s_%d" % (tag, k)]
            rows.append(row)
    out = pd.DataFrame(rows)
    return out.drop_duplicates(["frame", "iter"], keep="last") if len(out) else out


def per_residual(dump):
    d = dump.copy()
    H = d[HC].to_numpy(float)
    w = 1.0 / d["sigma_squared"].to_numpy(float)
    r = d["r"].to_numpy(float)
    d["weight"] = w
    d["cls"] = classify(d["hpos_z"].to_numpy(float))
    d["info_trace"] = w * (H ** 2).sum(1)
    d["info_rot_trace"] = w * (H[:, :3] ** 2).sum(1)
    d["info_pos_trace"] = w * (H[:, 3:] ** 2).sum(1)
    d["info_yaw"] = w * H[:, 2] ** 2
    d["info_x"] = w * H[:, 3] ** 2
    d["info_y"] = w * H[:, 4] ** 2
    for k in range(6):
        d["htz_%d" % k] = w * H[:, k] * r
    return d


def pinv_yaw_sd(M):
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    keep = w > 1e-10 * max(w.max(), 1e-300)
    inv = (V[:, keep] / w[keep]) @ V[:, keep].T
    return float(np.sqrt(max(inv[2, 2], 0.0)) * 1e3)


def hth_table(diag, pr):
    rows = []
    nowall = {}
    if pr is not None and len(pr):
        for (fr, it), g in pr[pr["cls"] != "wall"].groupby(["frame", "iter"]):
            nowall[(int(fr), int(it))] = (g["info_yaw"].sum(), g["info_x"].sum(), g["info_y"].sum())
    for _, r in diag.iterrows():
        M = mat6([r["HtH_%d%d" % ij] for ij in IJ])
        ev = np.linalg.eigvalsh(0.5 * (M + M.T))
        z = np.array([r["Htz_%d" % k] for k in range(6)])
        key = (int(r["frame"]), int(r["iter"]))
        nw = nowall.get(key, (np.nan, np.nan, np.nan))
        row = {"frame": key[0], "iter": key[1], "hth_trace": float(np.trace(M))}
        for k in range(6):
            row["eig_%d" % (k + 1)] = float(ev[k])
        row.update(eig_min=float(ev[0]), eig_max=float(ev[-1]), cond=float(ev[-1] / ev[0]) if ev[0] > 0 else np.inf,
                   info_yaw=float(M[2, 2]), info_x=float(M[3, 3]), info_y=float(M[4, 4]), info_yaw_nowall=float(nw[0]), info_x_nowall=float(nw[1]), info_y_nowall=float(nw[2]))
        for k in range(6):
            row["htz_%d" % k] = float(z[k])
        row.update(htz_norm=float(np.linalg.norm(z)), htz_rot_norm=float(np.linalg.norm(z[:3])), htz_pos_norm=float(np.linalg.norm(z[3:])),
                   n_resid=r["n_resid"], sum_w=r["sum_w"], yaw_sd_mrad=pinv_yaw_sd(M))
        rows.append(row)
    return pd.DataFrame(rows)


def class_table(pr):
    rows = []
    for (fr, it), g in pr.groupby(["frame", "iter"]):
        tot = g["info_trace"].sum()
        for cls in ("ground", "wall", "slanted"):
            c = g[g["cls"] == cls]
            row = {"frame": int(fr), "iter": int(it), "cls": cls, "n": int(len(c)), "weight_sum": float(c["weight"].sum()),
                   "weight_mean": float(c["weight"].mean()) if len(c) else np.nan, "weight_max": float(c["weight"].max()) if len(c) else np.nan,
                   "wr2_sum": float((c["weight"] * c["r"] ** 2).sum()), "hth_trace_share": float(c["info_trace"].sum() / tot) if tot > 0 else np.nan,
                   "info_yaw": float(c["info_yaw"].sum()), "info_x": float(c["info_x"].sum()), "info_y": float(c["info_y"].sum())}
            z = np.array([c["htz_%d" % k].sum() for k in range(6)])
            for k in range(6):
                row["htz_%d" % k] = float(z[k])
            row["htz_norm"] = float(np.linalg.norm(z))
            row["rms_r_mm"] = float(np.sqrt(np.mean(c["r"] ** 2)) * 1e3) if len(c) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def consistency(pr, diag):
    rows = []
    Dg = {(int(r["frame"]), int(r["iter"])): r for _, r in diag.iterrows()}
    for (fr, it), g in pr.groupby(["frame", "iter"]):
        r = Dg.get((int(fr), int(it)))
        if r is None:
            continue
        H = g[HC].to_numpy(float)
        w = g["weight"].to_numpy(float)
        M = (H * w[:, None]).T @ H
        z = (H * (w * g["r"].to_numpy(float))[:, None]).sum(0)
        Ml = mat6([r["HtH_%d%d" % ij] for ij in IJ])
        zl = np.array([r["Htz_%d" % k] for k in range(6)])
        eh = float(np.abs(M - Ml).max() / max(np.abs(Ml).max(), 1e-300))
        ep, em = float(np.abs(z - zl).max() / max(np.abs(zl).max(), 1e-300)), float(np.abs(-z - zl).max() / max(np.abs(zl).max(), 1e-300))
        sign, ez = (1, ep) if ep <= em else (-1, em)
        sw = r["sum_w"]
        rows.append({"frame": int(fr), "iter": int(it), "max_rel_err_hth": eh, "max_rel_err_htz": ez, "htz_sign": sign,
                     "err_sum_w": float(abs(w.sum() - sw) / max(abs(sw), 1e-300)) if pd.notna(sw) else np.nan,
                     "n_resid_dump": int(len(g)), "n_resid_logged": r["n_resid"], "ok": int(eh < 1e-3 and ez < 1e-3)})
    return pd.DataFrame(rows)


def opener(path):
    return path if os.path.exists(path) else (path + ".gz" if os.path.exists(path + ".gz") else None)


def run_one(dump_path, diag, frames, arm, role, hth_dump_frames_only=False):
    dump = pd.read_csv(dump_path)
    if frames != "all":
        dump = dump[dump["frame"].isin(frames)]
    pr = per_residual(dump)
    out = {}
    hth = hth_table(diag[diag["frame"].isin(pr["frame"].unique())] if hth_dump_frames_only else diag, pr) if len(diag) else pd.DataFrame()
    cls = class_table(pr) if len(pr) else pd.DataFrame()
    con = consistency(pr, diag) if len(diag) and len(pr) else pd.DataFrame()
    for name, df in (("hth", hth), ("cls", cls), ("con", con), ("res", pr)):
        if len(df):
            df.insert(0, "role", role); df.insert(0, "arm", arm)
        out[name] = df
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--arm", default=None)
    ap.add_argument("--resid-frames", default="1,2,3,4,5"); ap.add_argument("--shadow", action="store_true")
    ap.add_argument("--dump-frames-only", action="store_true")
    a = ap.parse_args()
    arm = a.arm or os.path.basename(os.path.abspath(a.dir))
    frames = "all" if a.resid_frames == "all" else [int(x) for x in a.resid_frames.split(",") if x]
    parts = {"hth": [], "cls": [], "con": [], "res": []}
    jobs = []
    dp, up, sp = opener(os.path.join(a.dir, "residual_dump.csv")), opener(os.path.join(a.dir, "lio_update_diag.csv")), opener(os.path.join(a.dir, "shadow_scan.csv"))
    if a.shadow:
        if not (dp and sp):
            print("missing residual_dump.csv or shadow_scan.csv in", a.dir); return 2
        jobs.append(("driver", dp, diag_from_shadow_scan(sp, "driver")))
        shd = opener(os.path.join(a.dir, "shadow_residual_dump.csv"))
        if shd:
            jobs.append(("shadow", shd, diag_from_shadow_scan(sp, "shadow")))
    else:
        if not (dp and up):
            print("missing residual_dump.csv or lio_update_diag.csv in", a.dir); return 2
        jobs.append(("driver", dp, diag_from_update_diag(up)))
    for role, path, diag in jobs:
        o = run_one(path, diag, frames, arm, role, hth_dump_frames_only=a.dump_frames_only)
        for k in parts:
            if len(o[k]):
                parts[k].append(o[k])
    os.makedirs(a.out, exist_ok=True)
    names = {"hth": "r73_f417_hth_frames.csv", "cls": "r73_f417_classes.csv", "con": "r73_f417_consistency.csv", "res": "r73_f417_residuals.csv.gz"}
    for k, nm in names.items():
        df = pd.concat(parts[k], ignore_index=True) if parts[k] else pd.DataFrame()
        df.to_csv(os.path.join(a.out, nm), index=False, float_format="%.9g", compression="gzip" if nm.endswith(".gz") else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
