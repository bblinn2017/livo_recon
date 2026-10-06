#!/usr/bin/env python3
"""R80D (F-492): reads-only timing analysis of /imu/imu on the nine bags (no estimator run, nothing written to the bags).

    python3 tools/r69/r80d_imu_timing.py --bag-dir <dir with the nine bags> --out imu_timing        # needs ROS (rosbag)
    python3 tools/r69/r80d_imu_timing.py --csv-in <per-message csv.gz> --label eee_01 --out imu_timing   # re-analyse a written per-message file
    python3 tools/r69/r80d_imu_timing.py --selftest

Per bag (arrays: header stamp h_k and bag record time r_k, integer ns): n, fitted rate (least-squares line of h vs index), header interval mean / std / min / max, share of intervals > 1.5 x median,
seq gaps, record-minus-header percentiles (0, 1, 50, 99, 100) in ms, header sawtooth = residual of h from (a) the least-squares line and (b) the lower-envelope line (min, p50, max ms),
and the correlation of header jitter (h interval minus median) with record jitter (r interval minus median).
Files: imu_timing/imu_timing_summary.csv (one row per bag, all nine bags); imu_timing/imu_msgs_<bag>.csv.gz for eee_01 and sbs_01 only (columns msg_index, seq, header_sec, header_nsec, record_sec,
record_nsec, record_minus_header_ns), as R80B did for the lidar. Planning agent's script; selftest on synthetic stamps only.
"""
import argparse, csv, glob, gzip, os, sys
import numpy as np

BAGS = ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]
MSG_BAGS = ("eee_01", "sbs_01")
SUMCOLS = ["bag", "n", "rate_hz_fit", "hdr_dt_mean_ms", "hdr_dt_std_ms", "hdr_dt_min_ms", "hdr_dt_max_ms", "share_dt_gt_1p5_median", "seq_gaps", "rec_minus_hdr_ms_min", "rec_minus_hdr_ms_p01",
           "rec_minus_hdr_ms_p50", "rec_minus_hdr_ms_p99", "rec_minus_hdr_ms_max", "saw_ls_min_ms", "saw_ls_p50_ms", "saw_ls_max_ms", "saw_env_min_ms", "saw_env_p50_ms", "saw_env_max_ms", "jitter_corr_hdr_rec"]


def lower_env(k, x):
    hull = []
    for i in range(len(k)):
        while len(hull) >= 2:
            i1, i2 = hull[-2], hull[-1]
            if (k[i2] - k[i1]) * (x[i] - x[i1]) - (x[i2] - x[i1]) * (k[i] - k[i1]) <= 0: hull.pop()
            else: break
        hull.append(i)
    best = None; ks = float(k.sum()); n = len(k)
    for j in range(len(hull) - 1):
        i1, i2 = hull[j], hull[j + 1]
        b = (x[i2] - x[i1]) / (k[i2] - k[i1]); a = x[i1] - b * k[i1]
        o = n * a + b * ks
        if best is None or o > best[0]: best = (o, a, b)
    return best[1], best[2]


def analyse(label, seq, h_ns, r_ns):
    seq = np.asarray(seq, dtype=np.int64); h = np.asarray(h_ns, dtype=np.int64); r = np.asarray(r_ns, dtype=np.int64)
    n = len(h)
    if n < 100: raise SystemExit("%s: too few messages (%d)" % (label, n))
    k = np.arange(n, dtype=np.float64); hs = (h - h[0]) * 1e-9
    dt = np.diff(hs); med = float(np.median(dt))
    b, a = np.polyfit(k, hs, 1)
    saw_ls = (hs - (a + b * k)) * 1e3
    ae, be = lower_env(k, hs); saw_env = (hs - (ae + be * k)) * 1e3
    rmh = (r - h) * 1e-6
    rd = np.diff((r - r[0]) * 1e-9)
    jc = float(np.corrcoef(dt - med, rd - np.median(rd))[0, 1]) if np.std(dt) > 0 and np.std(rd) > 0 else float("nan")
    q = lambda x, p: float(np.percentile(x, p))
    return dict(bag=label, n=int(n), rate_hz_fit=round(1.0 / b, 6), hdr_dt_mean_ms=round(1e3 * dt.mean(), 6), hdr_dt_std_ms=round(1e3 * dt.std(), 6), hdr_dt_min_ms=round(1e3 * dt.min(), 6),
                hdr_dt_max_ms=round(1e3 * dt.max(), 6), share_dt_gt_1p5_median=round(float((dt > 1.5 * med).mean()), 6), seq_gaps=int((np.diff(seq) != 1).sum()),
                rec_minus_hdr_ms_min=round(float(rmh.min()), 6), rec_minus_hdr_ms_p01=round(q(rmh, 1), 6), rec_minus_hdr_ms_p50=round(q(rmh, 50), 6), rec_minus_hdr_ms_p99=round(q(rmh, 99), 6),
                rec_minus_hdr_ms_max=round(float(rmh.max()), 6), saw_ls_min_ms=round(float(saw_ls.min()), 6), saw_ls_p50_ms=round(q(saw_ls, 50), 6), saw_ls_max_ms=round(float(saw_ls.max()), 6),
                saw_env_min_ms=round(float(saw_env.min()), 6), saw_env_p50_ms=round(q(saw_env, 50), 6), saw_env_max_ms=round(float(saw_env.max()), 6), jitter_corr_hdr_rec=round(jc, 6))


def read_bag(path, topic):
    import rosbag
    seq, h, r = [], [], []
    with rosbag.Bag(path) as b:
        for _, m, t in b.read_messages(topics=[topic]):
            seq.append(m.header.seq); h.append(m.header.stamp.secs * 10**9 + m.header.stamp.nsecs); r.append(t.secs * 10**9 + t.nsecs)
    if not seq: raise SystemExit("zero messages from %s in %s" % (topic, path))
    return seq, h, r


def write_msgs(path, seq, h, r):
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f); w.writerow(["msg_index", "seq", "header_sec", "header_nsec", "record_sec", "record_nsec", "record_minus_header_ns"])
        for i in range(len(seq)):
            w.writerow([i, seq[i], h[i] // 10**9, h[i] % 10**9, r[i] // 10**9, r[i] % 10**9, r[i] - h[i]])


def read_msgs(path):
    seq, h, r = [], [], []
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            seq.append(int(row["seq"])); h.append(int(row["header_sec"]) * 10**9 + int(row["header_nsec"])); r.append(int(row["record_sec"]) * 10**9 + int(row["record_nsec"]))
    return seq, h, r


def selftest():
    n = 4000; k = np.arange(n); true = 1609000000_000000000 + (k * 5_000_000)          # 200 Hz
    rng = np.random.RandomState(1)
    h = true + (rng.randint(0, 3_000_000, n))                                              # 0..3 ms jitter on the header
    rec = h + 1_200_000 + rng.randint(0, 400_000, n)
    seq = 7 + k
    s = analyse("synthetic", seq, h, rec)
    assert abs(s["rate_hz_fit"] - 200.0) < 0.5 and s["seq_gaps"] == 0 and s["n"] == n
    assert 1.2 <= s["rec_minus_hdr_ms_min"] <= 1.3 and s["rec_minus_hdr_ms_max"] < 1.61
    assert abs(s["saw_env_min_ms"]) < 1e-3 and s["saw_env_max_ms"] > 2.0
    seq2 = seq.copy(); seq2[100:] += 3
    assert analyse("g", seq2, h, rec)["seq_gaps"] == 1
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "m.csv.gz"); write_msgs(p, list(seq), [int(x) for x in h], [int(x) for x in rec])
    s2 = analyse("synthetic", *read_msgs(p))
    assert s2 == s, "round trip through the per-message file changes the summary"
    print("selftest: 200 Hz synthetic stamps: rate, record-minus-header range, envelope sawtooth, seq gap, csv round trip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag-dir"); ap.add_argument("--topic", default="/imu/imu"); ap.add_argument("--csv-in"); ap.add_argument("--label"); ap.add_argument("--out", default="imu_timing"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    os.makedirs(a.out, exist_ok=True); rows = []
    if a.csv_in:
        rows.append(analyse(a.label or os.path.basename(a.csv_in), *read_msgs(a.csv_in)))
    else:
        if not a.bag_dir: sys.exit("give --bag-dir or --csv-in")
        for bag in BAGS:
            hits = sorted(glob.glob(os.path.join(a.bag_dir, "**", "*%s*.bag" % bag), recursive=True))
            if len(hits) != 1: sys.exit("%s: expected exactly one bag matching *%s*.bag under %s, found %d %s" % (bag, bag, a.bag_dir, len(hits), hits))
            seq, h, r = read_bag(hits[0], a.topic)
            if bag in MSG_BAGS: write_msgs(os.path.join(a.out, "imu_msgs_%s.csv.gz" % bag), seq, h, r)
            rows.append(analyse(bag, seq, h, r)); print(rows[-1])
    with open(os.path.join(a.out, "imu_timing_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, SUMCOLS); w.writeheader(); w.writerows(rows)
    print("wrote %d rows" % len(rows))


if __name__ == "__main__":
    main()
