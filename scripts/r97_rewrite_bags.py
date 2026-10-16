#!/usr/bin/env python3
"""R97: rewrite an NTU VIRAL bag so that the horizontal lidar header stamp is the start of the scan on a clean grid, and the points that overlap the
previous message are removed. The node (R97 onward) reads the scan time from the header and has no time offset, no overlap policy and no retime keys.

What the script does to the lidar topic (default /os1_cloud_node1/points, sensor_msgs/PointCloud2), message by message, in bag order:
  1. header.stamp := grid(seq) = t0 + (seq - seq0) * period, rounded to the nearest nanosecond (exact rational arithmetic; the t0, period and seq0 of a bag are in
     bag_params_r92.json, keys retime_t0_s, retime_period_s, retime_seq0). t0 is already on the scan-time scale (the grid was fitted to header + ref_offset, ref_offset = -0.1),
     so no offset is added. header.seq and header.frame_id are kept.
  2. The per-point time field (default `t`, uint32, nanoseconds relative to the header stamp; the node's default point_time_mode offset_ns_u32) is NOT changed. Its
     absolute time is new_header_ns + t.
  3. Points whose absolute time is strictly earlier than the largest absolute time of the previous message's points are removed (the same boundary as the node's push rule,
     which discards a message whose first point is strictly earlier than the previous message's last point). Order of the remaining points is kept. The "previous
     message" maximum is taken over ALL points of the previous message after its own trimming, including points the node would later filter (blind distance, point_filter_num),
     so this script can remove slightly MORE points than the old retime_trim, which compared against the last point the node had kept. The per-message counts are reported.
  4. height := 1, width := number of points kept, row_step := width * point_step, data cut to match. A message left with no points is written with width 0 (and counted).
Every other message of every other topic (and all of the lidar connection's metadata) is copied byte for byte, in the original order and with the original record time.

Dependencies: python3, numpy and `rosbags` (pip install rosbags; pure Python). Optional: lz4 for --compression lz4.
Usage:
  r97_rewrite_bags.py rewrite --params bag_params_r92.json --seq eee_01 --in eee_01.bag --out eee_01_r97.bag [--lidar-topic T] [--time-field t] [--no-trim] [--report DIR]
  r97_rewrite_bags.py selftest
"""
import argparse
import csv
import json
import os
import sys
import tempfile
from fractions import Fraction

import numpy as np
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

UINT32 = 6          # sensor_msgs/PointField.UINT32
TS = get_typestore(Stores.ROS1_NOETIC)
PC2 = "sensor_msgs/msg/PointCloud2"


def grid_ns(seq, seq0, t0, period):
    """grid(seq) in integer nanoseconds. t0 and period are the exact values of the doubles in the params file (Fraction(float) is exact)."""
    v = Fraction(t0) + (int(seq) - int(seq0)) * Fraction(period)
    return int(round(v * 1_000_000_000))


def split_ns(ns):
    return ns // 1_000_000_000, ns % 1_000_000_000


def time_field(msg, name):
    for f in msg.fields:
        if f.name == name:
            if f.datatype != UINT32 or f.count != 1:
                raise SystemExit("point time field %r must be a single uint32 (offset_ns_u32); got datatype %d count %d" % (name, f.datatype, f.count))
            return int(f.offset)
    raise SystemExit("PointCloud2 has no field named %r (fields: %s)" % (name, [f.name for f in msg.fields]))


def rewrite_message(msg, new_ns, prev_max_ns, tfield, trim):
    """Return (new_msg, n_in, n_out, new_prev_max_ns). msg is modified in place."""
    n = int(msg.width) * int(msg.height)
    step = int(msg.point_step)
    if msg.is_bigendian:
        raise SystemExit("big-endian PointCloud2 is not supported")
    data = np.asarray(msg.data, dtype=np.uint8)
    if data.size != n * step:
        raise SystemExit("data size %d != width*height*point_step = %d (row padding is not supported)" % (data.size, n * step))
    sec, nsec = split_ns(new_ns)
    msg.header.stamp.sec = int(sec)
    msg.header.stamp.nanosec = int(nsec)
    if n == 0:
        return msg, 0, 0, prev_max_ns
    pts = data.reshape(n, step)
    off = time_field(msg, tfield)
    t = np.ascontiguousarray(pts[:, off:off + 4]).view("<u4").reshape(n).astype(np.int64)
    abs_ns = new_ns + t
    if trim and prev_max_ns is not None:
        keep = abs_ns >= prev_max_ns
    else:
        keep = np.ones(n, dtype=bool)
    n_out = int(keep.sum())
    new_prev = prev_max_ns if n_out == 0 else int(abs_ns[keep].max())
    if n_out != n:
        pts = pts[keep]
    msg.height = 1
    msg.width = n_out
    msg.row_step = n_out * step
    msg.data = np.ascontiguousarray(pts).reshape(-1)
    return msg, n, n_out, new_prev


def rewrite(params_path, seq, path_in, path_out, lidar_topic, tfield, trim, report_dir, compression):
    params = json.load(open(params_path))
    if seq not in params:
        raise SystemExit("sequence %r not in %s (have %s)" % (seq, params_path, sorted(params)))
    p = params[seq]
    t0, period, seq0 = float(p["retime_t0_s"]), float(p["retime_period_s"]), int(p["retime_seq0"])
    if not (t0 > 1e8 and 0.09 < period < 0.11 and seq0 >= 0):
        raise SystemExit("implausible grid for %s: t0=%r period=%r seq0=%r" % (seq, t0, period, seq0))
    rows = []
    prev_max = None
    n_msgs = 0
    with Reader(path_in) as r:
        topics = {c.topic for c in r.connections}
        if lidar_topic not in topics:
            raise SystemExit("topic %s not in %s (have %s)" % (lidar_topic, path_in, sorted(topics)))
        if os.path.exists(path_out):
            raise SystemExit("refusing to overwrite %s" % path_out)
        w = Writer(path_out)
        if compression != "none":
            w.set_compression(Writer.CompressionFormat[compression.upper()])
        with w:
            wconn = {}
            for c in r.connections:
                wconn[c.id] = w.add_connection(c.topic, c.msgtype, msgdef=c.msgdef.data, md5sum=c.digest, callerid=c.ext.callerid, latching=c.ext.latching)
            for conn, ts_ns, raw in r.messages():
                if conn.topic != lidar_topic:
                    w.write(wconn[conn.id], ts_ns, raw)
                    continue
                msg = TS.deserialize_ros1(raw, conn.msgtype)
                old_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                new_ns = grid_ns(msg.header.seq, seq0, t0, period)
                msg, n_in, n_out, prev_new = rewrite_message(msg, new_ns, prev_max, tfield, trim)
                rows.append((n_msgs, int(msg.header.seq), old_ns, new_ns, new_ns - old_ns, n_in, n_out, n_in - n_out, "" if prev_max is None else prev_max))
                prev_max = prev_new
                n_msgs += 1
                w.write(wconn[conn.id], ts_ns, TS.serialize_ros1(msg, conn.msgtype))
    summary = summarize(seq, rows, path_in, path_out, trim)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "rewrite_%s.csv" % seq), "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["msg_index", "seq", "old_header_ns", "new_header_ns", "shift_ns", "n_in", "n_out", "n_trimmed", "prev_max_abs_ns"])
            cw.writerows(rows)
        with open(os.path.join(report_dir, "rewrite_%s.json" % seq), "w") as f:
            json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return summary


def summarize(seq, rows, path_in, path_out, trim):
    trimmed = [r[7] for r in rows]
    shifts = [r[4] for r in rows]
    return {
        "seq": seq, "in": path_in, "out": path_out, "trim": bool(trim), "n_lidar_msgs": len(rows),
        "n_msgs_with_trim": int(sum(1 for x in trimmed if x > 0)), "n_points_in": int(sum(r[5] for r in rows)), "n_points_out": int(sum(r[6] for r in rows)),
        "n_points_trimmed": int(sum(trimmed)), "n_msgs_empty_after_trim": int(sum(1 for r in rows if r[5] > 0 and r[6] == 0)),
        "shift_ns_min": int(min(shifts)) if shifts else None, "shift_ns_max": int(max(shifts)) if shifts else None,
    }


# --------------------------------------------------------------------------------------------------------------------------------------------------
def selftest():
    H, PF, T = TS.types["std_msgs/msg/Header"], TS.types["sensor_msgs/msg/PointField"], TS.types["builtin_interfaces/msg/Time"]
    PC = TS.types[PC2]
    S = TS.types["std_msgs/msg/String"]
    d = tempfile.mkdtemp(prefix="r97_selftest_")
    pin, pout, pj = os.path.join(d, "in.bag"), os.path.join(d, "out.bag"), os.path.join(d, "params.json")
    t0, period, seq0 = 1609059013.0668645, 0.099997231748, 119
    json.dump({"fake_01": {"retime_t0_s": t0, "retime_period_s": period, "retime_seq0": seq0}}, open(pj, "w"))
    step, nper = 16, 21
    # message k: 21 points, t = j * 5 ms for j = 0..20 (a 100 ms sweep plus one point at exactly +100 ms), shuffled order in the message (the node sorts; the script must not assume order)
    rng = np.random.RandomState(7)
    ok = True

    def check(c, msg):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + msg)
        ok = ok and bool(c)

    raw_other = []
    with Writer(pin) as w:
        cl = w.add_connection("/lidar", PC2, typestore=TS)
        cs = w.add_connection("/other", "std_msgs/msg/String", typestore=TS, callerid="/x", latching=1)
        rec = 1609059000 * 1_000_000_000
        for k in range(6):
            seq = seq0 + k + (1 if k >= 3 else 0)    # a gap in seq: the grid must follow seq, not the message index
            jitter = int(rng.randint(-30_000_000, 30_000_000))
            hdr_ns = grid_ns(seq, seq0, t0, period) + 100_000_000 + jitter     # an original header on some other scale: must be replaced
            tt = (np.arange(nper) * 5_000_000).astype(np.uint32)
            order = rng.permutation(nper)
            buf = np.zeros((nper, step), dtype=np.uint8)
            for i, j in enumerate(order):
                buf[i, 12:16] = np.frombuffer(np.uint32(tt[j]).tobytes(), dtype=np.uint8)
                buf[i, 0:4] = np.frombuffer(np.float32(j).tobytes(), dtype=np.uint8)   # an x value that identifies the point
            m = PC(header=H(seq=seq, stamp=T(sec=hdr_ns // 10**9, nanosec=hdr_ns % 10**9), frame_id="os1_lidar"), height=3, width=7,
                   fields=[PF(name="x", offset=0, datatype=7, count=1), PF(name="t", offset=12, datatype=6, count=1)], is_bigendian=False,
                   point_step=step, row_step=7 * step, data=buf.reshape(-1), is_dense=False)
            w.write(cl, rec + k * 100_000_000, TS.serialize_ros1(m, PC2))
            s = TS.serialize_ros1(S(data="msg %d" % k), "std_msgs/msg/String")
            raw_other.append((rec + k * 100_000_000 + 1, s))
            w.write(cs, rec + k * 100_000_000 + 1, s)
    summ = rewrite(pj, "fake_01", pin, pout, "/lidar", "t", True, os.path.join(d, "rep"), "none")
    # expected, computed independently: grid headers; point j of message k+1 (time j*5 ms after its header) is kept iff it is not earlier than the previous message's max
    # absolute time, which is prev_header + 100 ms (j = 20). Next header = prev header + ~100 ms (period 99.997 ms): the points at t < prev_header + 100 ms - next_header are removed.
    with Reader(pout) as r:
        outs = [(c, t, raw) for c, t, raw in r.messages()]
    lid = [(c, t, TS.deserialize_ros1(raw, c.msgtype)) for c, t, raw in outs if c.topic == "/lidar"]
    oth = [(t, raw) for c, t, raw in outs if c.topic == "/other"]
    check(len(lid) == 6 and len(oth) == 6, "6 lidar and 6 other messages written")
    check([x[0] for x in oth] == [t for t, _ in raw_other] and [bytes(x[1]) for x in oth] == [bytes(raw) for _, raw in raw_other], "other topic: record times and bytes identical, same order")
    with Reader(pout) as r:
        cs = [c for c in r.connections if c.topic == "/other"][0]
        check(cs.ext.callerid == "/x" and cs.ext.latching == 1, "other topic: connection metadata (callerid, latching) kept")
    prev_max = None
    exp_trim = 0
    for k, (c, t, m) in enumerate(lid):
        seq = seq0 + k + (1 if k >= 3 else 0)
        want_ns = grid_ns(seq, seq0, t0, period)
        got_ns = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
        check(m.header.seq == seq and got_ns == want_ns, "msg %d: seq kept (%d) and header = grid (%d ns)" % (k, seq, want_ns))
        check(m.height == 1 and m.width * m.point_step == m.row_step == len(m.data) and m.header.frame_id == "os1_lidar", "msg %d: height 1, width/row_step/data consistent, frame_id kept" % k)
        pts = np.asarray(m.data).reshape(m.width, 16)
        tt = pts[:, 12:16].copy().view("<u4").reshape(-1).astype(np.int64)
        xs = pts[:, 0:4].copy().view("<f4").reshape(-1)
        absn = got_ns + tt
        if prev_max is None:
            keep_expect = nper
        else:
            keep_expect = sum(1 for j in range(nper) if want_ns + j * 5_000_000 >= prev_max)
        check(m.width == keep_expect, "msg %d: kept %d of %d points (expected %d)" % (k, m.width, nper, keep_expect))
        check(all(int(x) * 5_000_000 == tt[i] for i, x in enumerate(xs)), "msg %d: each kept point keeps its own t (the point identity x matches t)" % k)
        if prev_max is not None:
            check(absn.min() >= prev_max, "msg %d: no kept point is earlier than the previous message's maximum" % k)
        exp_trim += nper - keep_expect
        prev_max = want_ns + 20 * 5_000_000 if m.width else prev_max
    check(summ["n_points_trimmed"] == exp_trim and exp_trim > 0, "summary trimmed count %d equals the independent count %d (and is > 0: the test does trim)" % (summ["n_points_trimmed"], exp_trim))
    # --no-trim
    p2 = os.path.join(d, "out_notrim.bag")
    s2 = rewrite(pj, "fake_01", pin, p2, "/lidar", "t", False, None, "none")
    check(s2["n_points_trimmed"] == 0 and s2["n_points_out"] == 6 * nper, "--no-trim: headers re-stamped, no point removed")
    print("selftest %s (files in %s)" % ("PASS" if ok else "FAIL", d))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("rewrite")
    a.add_argument("--params", required=True)
    a.add_argument("--seq", required=True)
    a.add_argument("--in", dest="inp", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--lidar-topic", default="/os1_cloud_node1/points")
    a.add_argument("--time-field", default="t")
    a.add_argument("--no-trim", action="store_true")
    a.add_argument("--report", default=None)
    a.add_argument("--compression", choices=["none", "bz2", "lz4"], default="none")
    sub.add_parser("selftest")
    x = ap.parse_args()
    if x.cmd == "selftest":
        sys.exit(selftest())
    rewrite(x.params, x.seq, x.inp, x.out, x.lidar_topic, x.time_field, not x.no_trim, x.report, x.compression)


if __name__ == "__main__":
    main()
