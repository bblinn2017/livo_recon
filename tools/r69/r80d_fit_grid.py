#!/usr/bin/env python3
"""R80D (F-484, F-489): the two lidar start-time grids of one bag, fitted straight from the bag's own headers (never from a rounded log).

    python3 tools/r69/r80d_fit_grid.py --bag <bag.bag> --label eee_03 --out grid_check/eee_03.json
    python3 tools/r69/r80d_fit_grid.py --log lidar_msg_log_all.csv.gz --label eee_03        # planner cross-check input only (6-decimal rounded stamps)
    python3 tools/r69/r80d_fit_grid.py --merge grid_check/*.json --compare r80c_grid.json --out grid_check/r80d_grid_bag.json   # the file make_r80d_manifest.py --grid-bag reads
    python3 tools/r69/r80d_fit_grid.py --selftest

Both grids give scan_time(seq) = t0_s + (seq - seq0) * period_s on the header clock, with time_offset (default -0.1 s) already added into t0_s (cbk/lidar/retime_ref_offset_s = that offset):
  "retime"  : the R80C grid, the LOWER ENVELOPE of header_k + offset (r80c_fit_grid.fit, imported unchanged): no scan later than its own header-derived time.
  "restamp" : the NTU VIRAL authors' utils/restamp.py rule: t0 = first header of the bag, period = (last header - first header) / (N - 1), stamp_k = first + k * period (k = message index; here
              k = seq - seq0, identical to the message index when seq_gaps = 0, which is checked and printed). t0_s = first header + offset.
Output JSON: {"bag", "n", "seq0", "seq_gaps", "time_offset_s", "retime": {...fit fields...}, "restamp": {"t0_s","period_s","first_header_s","last_header_s","n","seq0"}}. Seconds, t0 to 9 decimals, period to 12.
For --bag the headers are read as integer nanoseconds, so the restamp grid has no rounding beyond the final 9-decimal print.
Planning agent's script; the self-test uses synthetic headers only (rule 3.1), run by me for the arithmetic only; the host run on real bags is the coding agent's.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r80c_fit_grid as C  # noqa: E402


def restamp_grid(seq, header_ns, time_offset):
    """header_ns: integer nanoseconds (int64). Returns the restamp.py grid."""
    seq = np.asarray(seq, dtype=np.int64); h = np.asarray(header_ns, dtype=np.int64)
    n = len(seq)
    if n < 100:
        raise SystemExit("too few messages: %d" % n)
    period_ns = (int(h[-1]) - int(h[0])) / float(n - 1)
    return {"t0_s": round(int(h[0]) * 1e-9 + time_offset, 9), "period_s": round(period_ns * 1e-9, 12), "first_header_s": round(int(h[0]) * 1e-9, 9), "last_header_s": round(int(h[-1]) * 1e-9, 9),
            "n": int(n), "seq0": int(seq[0])}


def fit_both(seq, header_ns, time_offset, label=""):
    h_s = np.asarray(header_ns, dtype=np.int64) * 1e-9
    r = C.fit(seq, h_s, time_offset, label)
    r["retime"] = {k: r.pop(k) for k in ("t0_s", "period_s", "support_k", "min_slack_ms", "p50_slack_ms", "max_slack_ms")}
    r["restamp"] = restamp_grid(seq, header_ns, time_offset)
    return r


def read_bag_ns(path, topic):
    import rosbag
    seq, ns = [], []
    with rosbag.Bag(path) as b:
        for _, m, _ in b.read_messages(topics=[topic]):
            seq.append(m.header.seq); ns.append(m.header.stamp.secs * 10**9 + m.header.stamp.nsecs)
    if not seq:
        raise SystemExit("zero messages read from %s in %s" % (topic, path))
    return seq, ns


def read_log_ns(path):
    seq, hs = C.read_log(path)
    return seq, [int(round(x * 1e9)) for x in hs]


def merge(files, compare, out):
    """Merge the per-bag outputs into {bag: output} (refuses a duplicate or a missing bag) and print the informational comparison of the retime grid with the shipped r80c_grid.json (no tolerance, no refusal)."""
    BAGS = ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]
    d = {}
    for f in files:
        j = json.load(open(f)); b = j["bag"]
        if b in d: raise SystemExit("duplicate bag %s (%s)" % (b, f))
        d[b] = j
    miss = [b for b in BAGS if b not in d]
    if miss: raise SystemExit("missing bags: %s" % miss)
    if compare:
        sh = json.load(open(compare)); print("bag      seq0(shipped,bag)  d_t0_s        d_period_s     (informational: shipped values came from 6-decimal logs; eee_03 uses the bag fit)")
        for b in BAGS:
            if b in sh:
                print("%-8s %6d %6d  %+.3e  %+.3e" % (b, sh[b]["seq0"], d[b]["seq0"], d[b]["retime"]["t0_s"] - sh[b]["t0_s"], d[b]["retime"]["period_s"] - sh[b]["period_s"]))
            else:
                print("%-8s not in the shipped grid file" % b)
    json.dump({b: d[b] for b in BAGS}, open(out, "w"), indent=1)
    print("wrote %s (%d bags)" % (out, len(d)))


def selftest():
    rng = np.random.RandomState(0)          # synthetic headers: a clean grid plus a 0..35 ms sawtooth delay
    n = 3000; k = np.arange(n); true = 1609000000.0 + 0.0999975 * k
    delay = 0.035 * ((k * 0.37) % 1.0); delay[1500] = 0.0            # one message with zero delay
    ns = np.round((true + delay) * 1e9).astype(np.int64)
    seq = 100 + k
    g = fit_both(seq, ns, -0.1, "synthetic")
    assert g["seq_gaps"] == 0 and g["seq0"] == 100 and g["n"] == n
    # retime grid: touches the series (min slack 0) and never above any header
    assert abs(g["retime"]["min_slack_ms"]) < 1e-6 and g["retime"]["max_slack_ms"] > 5.0
    # restamp grid: exactly first header + k * (last - first) / (n - 1)
    r = g["restamp"]
    assert abs(r["t0_s"] - (ns[0] * 1e-9 - 0.1)) < 1e-9
    assert abs(r["period_s"] - (ns[-1] - ns[0]) * 1e-9 / (n - 1)) < 1e-12
    pred = r["t0_s"] + (seq - r["seq0"]) * r["period_s"]
    assert abs(pred[0] - (ns[0] * 1e-9 - 0.1)) < 1e-6 and abs(pred[-1] - (ns[-1] * 1e-9 - 0.1)) < 1e-6      # the line passes through the first and last header
    # the two differ (the restamp line is not below every header): shows the check can tell them apart
    hdr_minus_restamp = (ns * 1e-9 - 0.1) - pred
    assert hdr_minus_restamp.min() < -1e-3 or hdr_minus_restamp.max() > 1e-3
    # a seq gap is reported
    seq2 = seq.copy(); seq2[2000:] += 1
    assert C.fit(seq2, ns * 1e-9, -0.1, "gap")["seq_gaps"] == 1
    import tempfile
    t = tempfile.mkdtemp(); fs = []
    for b in ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]:
        gg = dict(g); gg["bag"] = b; p = os.path.join(t, b + ".json"); json.dump(gg, open(p, "w")); fs.append(p)
    merge(fs, None, os.path.join(t, "m.json")); assert sorted(json.load(open(os.path.join(t, "m.json")))) == sorted(["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"])
    try:
        merge(fs[:-1], None, os.path.join(t, "m2.json")); raise AssertionError("merge accepted eight bags")
    except SystemExit:
        pass
    print("selftest: retime min slack 0, restamp line passes through first and last header, gap reported")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log"); ap.add_argument("--bag"); ap.add_argument("--topic", default="/os1_cloud_node1/points")
    ap.add_argument("--label", default=""); ap.add_argument("--time-offset", type=float, default=-0.1); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--merge", nargs="+"); ap.add_argument("--compare")
    a = ap.parse_args()
    if a.merge:
        merge(a.merge, a.compare, a.out or "r80d_grid_bag.json"); return
    if a.selftest:
        selftest(); return
    if bool(a.log) == bool(a.bag):
        sys.exit("give exactly one of --log or --bag")
    seq, ns = read_log_ns(a.log) if a.log else read_bag_ns(a.bag, a.topic)
    r = fit_both(seq, ns, a.time_offset, a.label)
    s = json.dumps(r, indent=1)
    if a.out:
        open(a.out, "w").write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
