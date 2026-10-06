#!/usr/bin/env python3
"""R80C (F-484): fit the clean lidar start-time grid of one bag.

Grid:  scan_time(seq) = t0_s + (seq - seq0) * period_s       (seconds, same clock as header.stamp)
Fit:   the LOWER ENVELOPE of x_k = header_stamp_k + time_offset over k = seq_k - seq0: among all lines a + b k with a + b k <= x_k for EVERY message, the one that
       maximises the sum of the line (it touches the series at two points; no scan is ever placed later than its own header-derived time, so every implied
       publish delay is >= 0). It is the earliest-header assumption: the least delayed message is taken to have zero delay (the constant floor of the delay is not in the data).
Input: --log  lidar_msg_log_all.csv.gz (columns seq, header_stamp)   or   --bag <bag> --topic <topic> (reads header.seq / header.stamp with rosbag; needs ROS).
Output: JSON {bag, n, seq0, t0_s, period_s, time_offset_s, support_k:[...], max_slack_ms, min_slack_ms, p50_slack_ms} (seconds, 9 decimals).
Planning agent's script. NOT run by me on the host; the arithmetic is the lower convex hull of (k, x_k) (numpy only, no scipy).
"""
import argparse, gzip, json, sys
import numpy as np


def lower_envelope_line(k, x):
    """Return (a, b, support_indices) of the line a + b k below all (k, x) maximising sum(a + b k). k strictly increasing floats."""
    n = len(k)
    hull = []
    for i in range(n):   # monotone chain, lower hull
        while len(hull) >= 2:
            i1, i2 = hull[-2], hull[-1]
            if (k[i2] - k[i1]) * (x[i] - x[i1]) - (x[i2] - x[i1]) * (k[i] - k[i1]) <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    best = None
    ksum = float(np.sum(k))
    for j in range(len(hull) - 1):
        i1, i2 = hull[j], hull[j + 1]
        b = (x[i2] - x[i1]) / (k[i2] - k[i1])
        a = x[i1] - b * k[i1]
        obj = n * a + b * ksum
        if best is None or obj > best[0]:
            best = (obj, a, b, (i1, i2))
    if best is None:
        raise SystemExit("fewer than two hull points")
    return best[1], best[2], best[3]


def fit(seq, header, time_offset, label=""):
    seq = np.asarray(seq, dtype=np.int64); header = np.asarray(header, dtype=np.float64)
    if len(seq) < 100:
        raise SystemExit("too few messages: %d" % len(seq))
    if np.any(np.diff(seq) <= 0):
        raise SystemExit("seq is not strictly increasing")
    seq0 = int(seq[0]); k = (seq - seq0).astype(np.float64)
    base = header[0] + time_offset
    x = header + time_offset - base           # seconds relative to the first scan time, keeps float64 precision
    a, b, sup = lower_envelope_line(k, x)
    slack = (x - (a + b * k)) * 1e3
    if slack.min() < -1e-6:
        raise SystemExit("envelope violated: min slack %g ms" % slack.min())
    return {"bag": label, "n": int(len(seq)), "seq0": seq0, "t0_s": round(float(base + a), 9), "period_s": round(float(b), 12), "time_offset_s": time_offset,
            "support_k": [int(k[sup[0]]), int(k[sup[1]])], "min_slack_ms": round(float(slack.min()), 6), "p50_slack_ms": round(float(np.median(slack)), 4),
            "max_slack_ms": round(float(slack.max()), 4), "seq_gaps": int((np.diff(seq) != 1).sum())}


def read_log(path):
    import csv
    op = gzip.open if path.endswith(".gz") else open
    seq, hs = [], []
    with op(path, "rt") as f:
        for r in csv.DictReader(f):
            seq.append(int(r["seq"])); hs.append(float(r["header_stamp"]))
    return seq, hs


def read_bag(path, topic):
    import rosbag
    seq, hs = [], []
    with rosbag.Bag(path) as b:
        for _, m, _ in b.read_messages(topics=[topic]):
            seq.append(m.header.seq); hs.append(m.header.stamp.secs + m.header.stamp.nsecs * 1e-9)
    if not seq:
        raise SystemExit("zero messages read from %s in %s" % (topic, path))
    return seq, hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log"); ap.add_argument("--bag"); ap.add_argument("--topic", default="/os1_cloud_node1/points")
    ap.add_argument("--label", default=""); ap.add_argument("--time-offset", type=float, default=-0.1); ap.add_argument("--out")
    a = ap.parse_args()
    if bool(a.log) == bool(a.bag):
        sys.exit("give exactly one of --log or --bag")
    seq, hs = read_log(a.log) if a.log else read_bag(a.bag, a.topic)
    r = fit(seq, hs, a.time_offset, a.label)
    s = json.dumps(r, indent=1)
    if a.out:
        open(a.out, "w").write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
