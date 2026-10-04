#!/usr/bin/env python3
"""R65 analysis-compression script (standing rule 5.12b). Runs on the machine that holds the raw diagnostic files of ONE arm and writes
compact files next to them. The full raw files stay on the machine (sha256 in compress_manifest.json); only the compact outputs are returned.
Nothing here interprets results; it thins, aggregates and copies numbers.

    python3 r65_compress.py <arm_dir> [--kinds stamps,numeric,plane_trace,gt]   (default: every kind whose input exists)

SUFFICIENCY (what the planning agent derives from each output, and what is lost)
  stamps       input sensor_stamps.csv (in-run log: kind L/I, stamp, n_raw, n_kept)
               -> sensor_stamps_compact.csv.gz: one row per LiDAR message with index, stamp, dt to previous, n_raw, n_kept, ratio of n_kept to the
                  centred 21-message median, number of image messages in (previous LiDAR stamp, this stamp], signed gap to the nearest image
                  stamp, camera phase; sensor_stamps_summary.json: LiDAR/camera rate and dt statistics, list of short scans
                  (n_kept < 0.9 x centred median), spacing of short scans, mean n_kept by number of images in the interval.
               Derives: whether short scans follow the image/LiDAR timing (F-164, F-173). Lost: nothing (every message row kept in compact form).
  numeric      input frame_stats.txt, surface_map_frames.csv (any header-row CSV whose first column or a column named frame/frame_idx indexes frames)
               -> <name>_compact.csv.gz: every numeric column, all frames <= 700 and 1000..1100 and every 10th frame elsewhere;
                  <name>_blocks.csv: per 100-frame block min/median/max of every numeric column; <name>_windows.csv: every frame in 400-430 and 500-560.
               Derives: Woodbury group statistics against the scan 413-415 and 505-520 departures (F-149, F-166), surface hit tiers and starvation
               (F-170), map growth. Lost: frames outside the windows are thinned to every 10th in the compact file (blocks keep min/median/max).
  plane_trace  input plane_fit_trace.csv(.gz): one row per plane-fit attempt (frame, path, reason, N, F, raw_e0..2, used_e0..2, e0_final, threshold, was_plane)
               -> plane_trace_compact.csv.gz: per (frame, path, reason) row count and the 5/25/50/75/95 % quantiles of raw_e0, raw_e1, raw_e2, used_e0,
                  e0_final/threshold, N, F; plane_trace_window.csv.gz: the raw rows of frames 500-525 (all of them, capped at 200000 rows).
               Derives: what the fits look like per frame, and which attempts fail at the onset (F-165, F-158). Lost: per-attempt rows outside 500-525.
  gt           input lio_gt_matched.csv -> lio_gt_matched_compact.csv.gz: every row, numbers to 7 significant digits.
               Derives: the GT join and a re-alignment of my own. Lost: nothing beyond digits.
Every kind also appends to compress_manifest.json: input file, its sha256, input rows, output files and bytes."""
import argparse, gzip, hashlib, io, json, math, os, sys
import numpy as np
import pandas as pd


def sha256_of(path):
    h = hashlib.sha256()
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find(D, *names):
    for n in names:
        for cand in (n, n + ".gz"):
            p = os.path.join(D, cand)
            if os.path.exists(p):
                return p
    return None


def record(D, kind, src, n_in, outs):
    mp = os.path.join(D, "compress_manifest.json")
    m = json.load(open(mp)) if os.path.exists(mp) else {}
    m[kind] = {"input": os.path.basename(src), "input_sha256": sha256_of(src), "input_rows": int(n_in),
               "outputs": {o: os.path.getsize(os.path.join(D, o)) for o in outs if os.path.exists(os.path.join(D, o))}}
    json.dump(m, open(mp, "w"), indent=1)


def to_gz(df, path, digits=7):
    df.to_csv(path, index=False, float_format="%%.%dg" % digits, compression="gzip")


# ---------------------------------------------------------------- stamps
def do_stamps(D):
    src = find(D, "sensor_stamps.csv")
    if not src:
        return None
    df = pd.read_csv(src)
    L = df[df.kind == "L"].reset_index(drop=True)
    I = df[df.kind == "I"].reset_index(drop=True)
    out = {"n_lidar": int(len(L)), "n_image": int(len(I))}
    if len(L) == 0:
        json.dump(out, open(os.path.join(D, "sensor_stamps_summary.json"), "w"), indent=1)
        record(D, "stamps", src, len(df), ["sensor_stamps_summary.json"])
        return True
    ls = L.stamp.to_numpy(float)
    dt = np.diff(ls, prepend=np.nan)
    med = pd.Series(L.n_kept.to_numpy(float)).rolling(21, center=True, min_periods=5).median().to_numpy()
    ratio = L.n_kept.to_numpy(float) / med
    istamp = np.sort(I.stamp.to_numpy(float)) if len(I) else np.array([])
    n_img_in, gap, phase = np.zeros(len(L)), np.full(len(L), np.nan), np.full(len(L), np.nan)
    if len(istamp) > 1:
        Tc = float(np.median(np.diff(istamp)))
        prev = np.concatenate([[ls[0] - 0.1], ls[:-1]])
        n_img_in = np.searchsorted(istamp, ls, side="right") - np.searchsorted(istamp, prev, side="right")
        j = np.clip(np.searchsorted(istamp, ls), 1, len(istamp) - 1)
        nearest = np.where(np.abs(istamp[j] - ls) < np.abs(istamp[j - 1] - ls), istamp[j], istamp[j - 1])
        gap = nearest - ls
        phase = np.mod(ls - istamp[0], Tc) / Tc
        out["camera_dt_median_s"] = Tc
        out["camera_dt_std_s"] = float(np.std(np.diff(istamp)))
        out["camera_rate_hz"] = 1.0 / Tc
    tab = pd.DataFrame({"idx": np.arange(len(L)), "stamp": ["%.7f" % x for x in ls], "dt_prev_s": dt, "n_raw": L.n_raw.to_numpy(), "n_kept": L.n_kept.to_numpy(),
                        "kept_over_median21": ratio, "n_images_in_interval": n_img_in, "gap_to_nearest_image_s": gap, "camera_phase": phase})
    to_gz(tab, os.path.join(D, "sensor_stamps_compact.csv.gz"), 8)
    short = np.where(ratio < 0.9)[0]
    out.update({"lidar_dt_median_s": float(np.nanmedian(dt)), "lidar_dt_std_s": float(np.nanstd(dt)), "lidar_rate_hz": float(1.0 / np.nanmedian(dt)),
                "n_kept_median": float(np.median(L.n_kept)), "n_kept_min": int(L.n_kept.min()), "n_kept_max": int(L.n_kept.max()),
                "n_short": int(len(short)), "short_indices_first_200": short[:200].tolist(),
                "short_spacing_counts": {str(k): int(v) for k, v in zip(*np.unique(np.diff(short), return_counts=True))} if len(short) > 1 else {}})
    if len(istamp) > 1:
        out["mean_n_kept_by_images_in_interval"] = {str(int(k)): float(np.mean(L.n_kept.to_numpy(float)[n_img_in == k])) for k in np.unique(n_img_in)}
        out["short_fraction_by_images_in_interval"] = {str(int(k)): float(np.mean(ratio[n_img_in == k] < 0.9)) for k in np.unique(n_img_in)}
    json.dump(out, open(os.path.join(D, "sensor_stamps_summary.json"), "w"), indent=1)
    record(D, "stamps", src, len(df), ["sensor_stamps_compact.csv.gz", "sensor_stamps_summary.json"])
    return True


# ---------------------------------------------------------------- numeric
def do_numeric(D, name):
    src = find(D, name)
    if not src:
        return None
    df = pd.read_csv(src)
    fc = next((c for c in ("frame_idx", "frame", "scan", "frame_id") if c in df.columns), df.columns[0])
    df = df.drop_duplicates(fc, keep="last").sort_values(fc).reset_index(drop=True)
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    f = df[fc].to_numpy(float)
    stem = os.path.splitext(name)[0]
    keep = (f <= 700) | ((f >= 1000) & (f <= 1100)) | (np.mod(f, 10) == 0)
    to_gz(df.loc[keep, num], os.path.join(D, stem + "_compact.csv.gz"), 7)
    win = ((f >= 400) & (f <= 430)) | ((f >= 500) & (f <= 560))
    to_gz(df.loc[win, num], os.path.join(D, stem + "_windows.csv.gz"), 7)
    blk = (np.floor(f / 100)).astype(int)
    rows = []
    for b in np.unique(blk):
        m = blk == b
        r = {"block_start_frame": int(b) * 100, "n": int(m.sum())}
        for c in num:
            v = df.loc[m, c].to_numpy(float)
            r[c + "_min"], r[c + "_med"], r[c + "_max"] = float(np.nanmin(v)), float(np.nanmedian(v)), float(np.nanmax(v))
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(D, stem + "_blocks.csv"), index=False, float_format="%.7g")
    record(D, "numeric:" + name, src, len(df), [stem + "_compact.csv.gz", stem + "_windows.csv.gz", stem + "_blocks.csv"])
    return True


# ---------------------------------------------------------------- plane trace
def do_plane_trace(D):
    src = find(D, "plane_fit_trace.csv")
    if not src:
        return None
    df = pd.read_csv(src)
    df["e0_over_thr"] = df["e0_final"] / df["threshold"].replace(0, np.nan)
    cols = [c for c in ("raw_e0", "raw_e1", "raw_e2", "used_e0", "e0_over_thr", "N", "F") if c in df.columns]
    rows = []
    for (fr, pth, rs), g in df.groupby(["frame", "path", "reason"]):
        r = {"frame": int(fr), "path": pth, "reason": int(rs), "rows": int(len(g))}
        for c in cols:
            v = g[c].to_numpy(float)
            v = v[np.isfinite(v)]
            for q in (5, 25, 50, 75, 95):
                r["%s_p%02d" % (c, q)] = float(np.percentile(v, q)) if len(v) else np.nan
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(D, "plane_trace_compact.csv.gz"), index=False, float_format="%.6g", compression="gzip")
    w = df[(df.frame >= 500) & (df.frame <= 525)].head(200000)
    to_gz(w, os.path.join(D, "plane_trace_window.csv.gz"), 6)
    record(D, "plane_trace", src, len(df), ["plane_trace_compact.csv.gz", "plane_trace_window.csv.gz"])
    return True


# ---------------------------------------------------------------- gt
def do_gt(D):
    src = find(D, "lio_gt_matched.csv")
    if not src:
        return None
    df = pd.read_csv(src)
    to_gz(df, os.path.join(D, "lio_gt_matched_compact.csv.gz"), 9 if "t_abs" in df.columns else 7)
    record(D, "gt", src, len(df), ["lio_gt_matched_compact.csv.gz"])
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir"); ap.add_argument("--kinds", default="stamps,numeric,plane_trace,gt")
    a = ap.parse_args()
    D = a.arm_dir
    kinds = a.kinds.split(",")
    done = []
    if "stamps" in kinds and do_stamps(D):
        done.append("stamps")
    if "numeric" in kinds:
        for nm in ("frame_stats.txt", "surface_map_frames.csv"):
            if do_numeric(D, nm):
                done.append("numeric:" + nm)
    if "plane_trace" in kinds and do_plane_trace(D):
        done.append("plane_trace")
    if "gt" in kinds and do_gt(D):
        done.append("gt")
    print("compressed %s: %s" % (os.path.basename(os.path.abspath(D)), ",".join(done) if done else "nothing to do"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
