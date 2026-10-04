#!/usr/bin/env python3
"""R66 items 0a and 0d: bag inventory and per-bag statistics, straight from the bags (no estimator). Needs the ROS1 python API (`rosbag`) and numpy.

  python3 tools/r66/bag_stats.py inventory --search-dirs <dir> [<dir> ...] --out R66_RETURN/bag_stats
      -> bag_inventory.csv: path, sequence name (file stem), bytes, sha256, duration_s, start_time, topics with message counts for the LiDAR, IMU, image and ground-truth topics.
  python3 tools/r66/bag_stats.py stats --bag <file.bag> --seq <name> --out R66_RETURN/bag_stats [--dup-scans 1000]
      -> <out>/<seq>/lidar_scan_table.csv.gz, imu_window_stats.csv, imu_psd.csv, imu_residual_windows.csv, bag_params_entry.json, README.txt
  python3 tools/r66/bag_stats.py merge --out R66_RETURN/bag_stats --seqs eee_01 eee_02 ...   -> <out>/bag_params.json  (seq -> onset_scan, stat_last, trans_last)

DEFINITIONS (every number uses exactly these)
  scan k (1-based)   = k-th message on the LiDAR topic (config topic /os1_cloud_node1/points); its stamp is the message header stamp.
  IMU window of scan k = IMU samples with stamp in (stamp of scan k-1, stamp of scan k].
  dynamic acceleration = | a(t) - a_rest |, a_rest = mean IMU acceleration vector over the first 385 IMU samples (about 1 s) of the bag;
                       smoothed by a running median over 39 samples (about 0.1 s). (The bag is assumed to start at rest; this is checked, not assumed:
                       the rest-window standard deviation is reported.)
  motion onset time  = first time the smoothed dynamic acceleration exceeds 0.25 m/s^2 and stays above it for 39 consecutive samples.
  onset_scan         = first scan whose stamp is >= the onset time. stat_last = onset_scan - 5, trans_last = onset_scan + 55 (for eee_01 these are 485, 480, 540
                       if the criterion reproduces the established IMU onset; the script prints that comparison, it does not force it).
  windows            = rest: scans 50 .. onset-45; pre-onset: onset-44 .. onset-1; onset: onset .. onset+55; motion: onset+56 .. end. A window with fewer
                       than 20 scans is reported empty.
  lidar table        = per scan: n_points (width x height), exact-duplicate fraction (points whose 1e-9 m rounded xyz appeared in any earlier scan, hash based; first
                       --dup-scans scans only), duplicate fraction against the previous scan only, beam-direction (az, el at 0.1 deg bins, from the IMU-frame lidar origin
                       (-0.050, 0, 0.055)) repeat fraction against the previous scan, gap to the nearest image stamp (signed, s), number of images in (previous stamp, this stamp].
  imu_window_stats   = per window: samples, mean and variance per axis of acc and gyro, autocorrelation at lags 1..5 of each axis (after removing the window mean).
  imu_psd            = per window and axis: power spectral density averaged over 256-sample segments (Hann, 50 % overlap), summed into 10 Hz bands up to 190 Hz. Units (m/s^2)^2/Hz, (rad/s)^2/Hz.
  imu_residual_windows = per window and axis: variance of the residual of a local cubic polynomial fit over 11 IMU samples (a spline-free estimate of white-noise variance, F-207 item 5),
                       computed only if scipy is importable; otherwise the README says so.
Reads only; reports what is there, interprets nothing."""
import argparse, csv, glob, gzip, hashlib, json, os, sys
import numpy as np

LIDAR, IMU, IMAGE = "/os1_cloud_node1/points", "/imu/imu", "/left/image_raw"
GT_TOPICS = ["/leica/pose/relative", "/leica/position", "/leica/pose"]
ORIGIN = np.array([-0.050, 0.0, 0.055])


def need_rosbag():
    try:
        import rosbag
        return rosbag
    except Exception as ex:
        print("rosbag import failed:", ex)
        sys.exit(1)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(a):
    rosbag = need_rosbag()
    os.makedirs(a.out, exist_ok=True)
    bags = []
    for d in a.search_dirs:
        bags += glob.glob(os.path.join(d, "**", "*.bag"), recursive=True)
    rows = []
    for b in sorted(set(bags)):
        try:
            bag = rosbag.Bag(b)
            info = bag.get_type_and_topic_info().topics
            r = {"path": b, "seq": os.path.splitext(os.path.basename(b))[0], "bytes": os.path.getsize(b), "sha256": sha256_of(b),
                 "duration_s": bag.get_end_time() - bag.get_start_time(), "start_time": bag.get_start_time()}
            for nm, tp in (("lidar", LIDAR), ("imu", IMU), ("image", IMAGE)):
                r[nm + "_topic_present"] = tp in info
                r[nm + "_messages"] = info[tp].message_count if tp in info else 0
            r["gt_topics_present"] = " ".join("%s:%d" % (t, info[t].message_count) for t in GT_TOPICS if t in info)
            r["all_topics"] = " ".join("%s:%d" % (t, v.message_count) for t, v in sorted(info.items()))
            bag.close()
        except Exception as ex:
            r = {"path": b, "seq": os.path.splitext(os.path.basename(b))[0], "error": str(ex)}
        rows.append(r)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out, "bag_inventory.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("inventory: %d bags" % len(rows))
    return 0


def xyz_from_cloud(m):
    offs = {f.name: f.offset for f in m.fields}
    n = m.width * m.height
    if n == 0 or not all(k in offs for k in "xyz"):
        return np.zeros((0, 3))
    buf = np.frombuffer(m.data, dtype=np.uint8).reshape(n, m.point_step)
    out = np.empty((n, 3), dtype=np.float64)
    for i, k in enumerate("xyz"):
        out[:, i] = np.ascontiguousarray(buf[:, offs[k]:offs[k] + 4]).view("<f4").reshape(n) if not m.is_bigendian else \
            np.ascontiguousarray(buf[:, offs[k]:offs[k] + 4]).view(">f4").reshape(n)
    ok = np.isfinite(out).all(axis=1)
    return out[ok]


def hashes(p):
    q = np.round(p * 1e9).astype(np.int64)
    return q[:, 0] * np.int64(73856093) ^ q[:, 1] * np.int64(19349663) ^ q[:, 2] * np.int64(83492791)


def beam_bins(p, bin_deg=0.1):
    v = p - ORIGIN
    nr = np.linalg.norm(v, axis=1)
    nr[nr == 0] = 1.0
    v = v / nr[:, None]
    az = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    el = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
    return np.round(az / bin_deg).astype(np.int64) * 100003 + np.round(el / bin_deg).astype(np.int64)


def runmed(x, k):
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    # R66 agent fix: numpy.lib.stride_tricks.sliding_window_view requires numpy>=1.20;
    # this host has numpy 1.17.4. as_strided produces the identical 1D sliding-window view.
    n = len(x)
    windows = np.lib.stride_tricks.as_strided(xp, shape=(n, k), strides=(xp.strides[0], xp.strides[0]))
    return np.median(windows, axis=1)


def stats(a):
    rosbag = need_rosbag()
    out = os.path.join(a.out, a.seq)
    os.makedirs(out, exist_ok=True)
    bag = rosbag.Bag(a.bag)
    info = bag.get_type_and_topic_info().topics
    lid, imu_t, imu_a, imu_g, img = [], [], [], [], []
    seen_h = np.zeros(0, dtype=np.int64)
    prev_h = np.zeros(0, dtype=np.int64)
    prev_b = np.zeros(0, dtype=np.int64)
    for tp, m, t in bag.read_messages(topics=[LIDAR, IMU, IMAGE]):
        if tp == IMU:
            imu_t.append(m.header.stamp.to_sec())
            imu_a.append((m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z))
            imu_g.append((m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z))
        elif tp == IMAGE:
            img.append(m.header.stamp.to_sec())
        else:
            k = len(lid) + 1
            row = {"scan": k, "stamp": m.header.stamp.to_sec(), "n_points": int(m.width * m.height), "dup_any": np.nan, "dup_prev": np.nan, "beam_repeat_prev": np.nan}
            if k <= a.dup_scans:
                p = xyz_from_cloud(m)
                if len(p):
                    h = hashes(p)
                    b = beam_bins(p)
                    row["dup_any"] = float(np.isin(h, seen_h).mean()) if len(seen_h) else 0.0
                    row["dup_prev"] = float(np.isin(h, prev_h).mean()) if len(prev_h) else 0.0
                    row["beam_repeat_prev"] = float(np.isin(b, prev_b).mean()) if len(prev_b) else 0.0
                    seen_h = np.unique(np.concatenate([seen_h, h]))
                    prev_h, prev_b = h, b
            lid.append(row)
    bag.close()
    imu_t, imu_a, imu_g, img = np.array(imu_t), np.array(imu_a), np.array(imu_g), np.sort(np.array(img))
    ls = np.array([r["stamp"] for r in lid])
    # image gap and count
    if len(img) > 1:
        prev = np.concatenate([[ls[0] - 0.1], ls[:-1]])
        nimg = np.searchsorted(img, ls, side="right") - np.searchsorted(img, prev, side="right")
        j = np.clip(np.searchsorted(img, ls), 1, len(img) - 1)
        nearest = np.where(np.abs(img[j] - ls) < np.abs(img[j - 1] - ls), img[j], img[j - 1])
        gap = nearest - ls
    else:
        nimg, gap = np.full(len(ls), np.nan), np.full(len(ls), np.nan)
    for r, g_, n_ in zip(lid, gap, nimg):
        r["gap_to_nearest_image_s"], r["n_images_in_interval"] = float(g_), float(n_)
    with gzip.open(os.path.join(out, "lidar_scan_table.csv.gz"), "wt") as f:
        cols = ["scan", "stamp", "n_points", "dup_any", "dup_prev", "beam_repeat_prev", "gap_to_nearest_image_s", "n_images_in_interval"]
        f.write(",".join(cols) + "\n")
        for r in lid:
            f.write(",".join(("%.9f" % r[c]) if c == "stamp" else ("%.6g" % r[c] if isinstance(r[c], float) else str(r[c])) for c in cols) + "\n")
    # rest reference and onset
    n_rest = min(385, len(imu_t))
    a_rest = imu_a[:n_rest].mean(0)
    dyn = np.linalg.norm(imu_a - a_rest, axis=1)
    sm = runmed(dyn, 39)
    above = sm > 0.25
    onset_t = None
    run = 0
    for i, v in enumerate(above):
        run = run + 1 if v else 0
        if run >= 39:
            onset_t = imu_t[i - 38]
            break
    scans = np.arange(1, len(ls) + 1)
    if onset_t is None:
        onset_scan = int(len(ls))
        note = "no onset found: smoothed dynamic acceleration never stayed above 0.25 m/s^2 for 39 samples; onset_scan set to the last scan"
    else:
        onset_scan = int(scans[np.searchsorted(ls, onset_t)]) if np.searchsorted(ls, onset_t) < len(ls) else int(len(ls))
        note = "onset criterion met at t=%.4f s after the first IMU sample" % (onset_t - imu_t[0])
    entry = {a.seq: {"onset_scan": onset_scan, "stat_last": max(onset_scan - 5, 0), "trans_last": onset_scan + 55, "note": note,
                     "rest_acc_std_m_s2": imu_a[:n_rest].std(0).tolist(), "n_scans": int(len(ls)), "n_imu": int(len(imu_t)),
                     "imu_rate_hz": float(1.0 / np.median(np.diff(imu_t))), "lidar_rate_hz": float(1.0 / np.median(np.diff(ls))),
                     "gt_topics": {t: info[t].message_count for t in GT_TOPICS if t in info}}}
    json.dump(entry, open(os.path.join(out, "bag_params_entry.json"), "w"), indent=1)
    if a.seq == "eee_01":
        print("eee_01 check: criterion onset_scan=%d (established IMU onset 485; established stationary last scan 480)" % onset_scan)
    # windows
    os_ = onset_scan
    wins = {"rest": (50, os_ - 45), "pre_onset": (os_ - 44, os_ - 1), "onset": (os_, os_ + 55), "motion": (os_ + 56, len(ls))}
    lo_t = lambda k: ls[k - 2] if k >= 2 else imu_t[0] - 1e-9
    rows, psd_rows, res_rows = [], [], []
    try:
        from scipy.signal import savgol_filter
    except Exception:
        savgol_filter = None
    for wn, (k0, k1) in wins.items():
        k0, k1 = max(k0, 1), min(k1, len(ls))
        if k1 - k0 + 1 < 20:
            continue
        sel = (imu_t > lo_t(k0)) & (imu_t <= ls[k1 - 1])
        A, G = imu_a[sel], imu_g[sel]
        for nm, X in (("acc", A), ("gyro", G)):
            for ax, axn in enumerate("xyz"):
                x = X[:, ax]
                xc = x - x.mean()
                r = {"window": wn, "scans": "%d-%d" % (k0, k1), "samples": int(len(x)), "sensor": nm, "axis": axn, "mean": float(x.mean()), "variance": float(x.var())}
                for lag in range(1, 6):
                    r["acf_lag%d" % lag] = float((xc[:-lag] * xc[lag:]).mean() / max(xc.var(), 1e-300)) if len(xc) > lag + 2 else np.nan
                rows.append(r)
                seg, fs = 256, 1.0 / np.median(np.diff(imu_t[sel]))
                if len(xc) >= seg:
                    win = np.hanning(seg)
                    spec, cnt = np.zeros(seg // 2 + 1), 0
                    for s0 in range(0, len(xc) - seg + 1, seg // 2):
                        f_ = np.abs(np.fft.rfft(xc[s0:s0 + seg] * win)) ** 2 / (fs * (win ** 2).sum())
                        f_[1:-1] *= 2
                        spec += f_
                        cnt += 1
                    spec /= cnt
                    freqs = np.fft.rfftfreq(seg, 1.0 / fs)
                    for b0 in range(0, 190, 10):
                        m_ = (freqs >= b0) & (freqs < b0 + 10)
                        if m_.any():
                            psd_rows.append({"window": wn, "sensor": nm, "axis": axn, "band_hz": "%d-%d" % (b0, b0 + 10), "psd_mean": float(spec[m_].mean()), "segments": cnt})
                if savgol_filter is not None and len(x) > 11:
                    res = x - savgol_filter(x, 11, 3)
                    res_rows.append({"window": wn, "sensor": nm, "axis": axn, "residual_variance": float(res[5:-5].var()), "samples": int(len(res) - 10)})
    for fn, rr in (("imu_window_stats.csv", rows), ("imu_psd.csv", psd_rows), ("imu_residual_windows.csv", res_rows)):
        if rr:
            keys = list(rr[0].keys())
            with open(os.path.join(out, fn), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rr)
    open(os.path.join(out, "README.txt"), "w").write("bag %s seq %s scans %d imu %d images %d; dup statistics for the first %d scans only; scipy %s\n" % (
        a.bag, a.seq, len(ls), len(imu_t), len(img), a.dup_scans, "present" if savgol_filter else "MISSING (imu_residual_windows.csv not written)"))
    print("stats %s: %d scans, onset_scan %d" % (a.seq, len(ls), onset_scan))
    return 0


def merge(a):
    allp = {}
    for s in a.seqs:
        p = os.path.join(a.out, s, "bag_params_entry.json")
        if os.path.exists(p):
            allp.update(json.load(open(p)))
        else:
            print("missing", p)
    json.dump(allp, open(os.path.join(a.out, "bag_params.json"), "w"), indent=1)
    print("bag_params.json:", sorted(allp))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    i = sp.add_parser("inventory"); i.add_argument("--search-dirs", nargs="+", required=True); i.add_argument("--out", required=True)
    s = sp.add_parser("stats"); s.add_argument("--bag", required=True); s.add_argument("--seq", required=True); s.add_argument("--out", required=True); s.add_argument("--dup-scans", type=int, default=1000)
    m = sp.add_parser("merge"); m.add_argument("--out", required=True); m.add_argument("--seqs", nargs="+", required=True)
    a = ap.parse_args()
    return {"inventory": inventory, "stats": stats, "merge": merge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
