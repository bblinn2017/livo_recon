#!/usr/bin/env python3
"""R64 (F-147): what else does the eee_01 bag record around the scan 401-430 disturbance (41-43 s into the bag)?

    python3 tools/r64/bag_probe.py --bag <path to eee_01.bag> --out r64_bag_probe [--t-lo 30 --t-hi 60]

Needs the ROS1 python API (`rosbag`) on the host that has the bag. Writes into --out:
  topics.csv          every topic: type, message count, first/last time (s since bag start), mean rate (Hz)
  imu_window.csv.gz   every message of the IMU topic in [t-lo, t-hi] s: t, ax, ay, az, wx, wy, wz (raw, unfiltered)
  imu_100ms_stats.csv  the same window in 0.1 s bins: mean and std of |a - median|a||, of |w|, and of each axis
  other_topics.csv    for every OTHER numeric-looking topic (std_msgs numeric types, geometry_msgs/Vector3/Twist/Wrench,
                      sensor_msgs/Imu-like, mavros-style RC/ESC/actuator messages if present), the same 0.1 s bin mean/std of each
                      scalar field in the window. Topics with only images/point clouds are listed in topics.csv and skipped.
The script only reads the bag. It does not interpret anything; it reports what is there. If `rosbag` cannot be imported, write
'rosbag import failed: <error>' into out/README.txt and return that."""
import argparse, csv, gzip, math, os, sys

def flat_fields(msg, prefix=""):
    """Yield (name, float) for every numeric scalar reachable in a message (arrays up to 16 long are expanded)."""
    for slot in getattr(msg, "__slots__", []):
        v = getattr(msg, slot)
        name = prefix + slot
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            yield name, float(v)
        elif isinstance(v, (list, tuple)) and 0 < len(v) <= 16 and all(isinstance(x, (int, float)) for x in v):
            for i, x in enumerate(v):
                yield "%s[%d]" % (name, i), float(x)
        elif hasattr(v, "__slots__"):
            if slot in ("header",):
                continue
            for kv in flat_fields(v, name + "."):
                yield kv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True); ap.add_argument("--out", default="r64_bag_probe")
    ap.add_argument("--t-lo", type=float, default=30.0); ap.add_argument("--t-hi", type=float, default=60.0)
    ap.add_argument("--imu-topic", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    try:
        import rosbag
    except Exception as ex:
        open(os.path.join(a.out, "README.txt"), "w").write("rosbag import failed: %s\n" % ex)
        print("rosbag import failed:", ex)
        return 1
    bag = rosbag.Bag(a.bag)
    t0 = bag.get_start_time()
    info = bag.get_type_and_topic_info()
    with open(os.path.join(a.out, "topics.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["topic", "type", "count", "mean_rate_hz"])
        dur = max(bag.get_end_time() - t0, 1e-9)
        for tp, v in sorted(info.topics.items()):
            w.writerow([tp, v.msg_type, v.message_count, v.message_count / dur])
    imu_topic = a.imu_topic
    if imu_topic is None:
        cands = [tp for tp, v in info.topics.items() if v.msg_type == "sensor_msgs/Imu"]
        imu_topic = sorted(cands)[0] if cands else None
    open(os.path.join(a.out, "README.txt"), "w").write("bag %s start %.3f imu_topic %s window %.1f-%.1f s\n" % (a.bag, t0, imu_topic, a.t_lo, a.t_hi))
    import numpy as np
    # IMU window + 0.1 s stats
    if imu_topic:
        rows = []
        for _, m, t in bag.read_messages(topics=[imu_topic], start_time=None):
            ts = t.to_sec() - t0
            if a.t_lo <= ts <= a.t_hi:
                rows.append([ts, m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z,
                             m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
        arr = np.array(rows)
        with gzip.open(os.path.join(a.out, "imu_window.csv.gz"), "wt") as f:
            f.write("t,ax,ay,az,wx,wy,wz\n")
            for r in rows:
                f.write(",".join("%.9g" % x for x in r) + "\n")
        if len(arr):
            an = np.linalg.norm(arr[:, 1:4], axis=1); amed = np.median(an)
            wn = np.linalg.norm(arr[:, 4:7], axis=1)
            bins = np.floor((arr[:, 0] - a.t_lo) / 0.1).astype(int)
            with open(os.path.join(a.out, "imu_100ms_stats.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["t_bin_start", "n", "acc_norm_dev_mean", "acc_norm_dev_std", "gyro_norm_mean", "gyro_norm_std",
                                                "ax_mean", "ay_mean", "az_mean", "ax_std", "ay_std", "az_std", "wx_std", "wy_std", "wz_std"])
                for b in np.unique(bins):
                    mm = bins == b; s = arr[mm]
                    w.writerow([a.t_lo + 0.1 * b, int(mm.sum()), float(np.mean(an[mm] - amed)), float(np.std(an[mm])), float(np.mean(wn[mm])), float(np.std(wn[mm])),
                                *[float(np.mean(s[:, i])) for i in (1, 2, 3)], *[float(np.std(s[:, i])) for i in (1, 2, 3)], *[float(np.std(s[:, i])) for i in (4, 5, 6)]])
    # other topics
    skip_types = ("sensor_msgs/Image", "sensor_msgs/CompressedImage", "sensor_msgs/PointCloud2", "sensor_msgs/PointCloud", "tf2_msgs/TFMessage")
    with open(os.path.join(a.out, "other_topics.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["topic", "field", "t_bin_start", "n", "mean", "std"])
        for tp, v in sorted(info.topics.items()):
            if tp == imu_topic or v.msg_type in skip_types:
                continue
            acc = {}
            for _, m, t in bag.read_messages(topics=[tp]):
                ts = t.to_sec() - t0
                if not (a.t_lo <= ts <= a.t_hi):
                    continue
                b = int(math.floor((ts - a.t_lo) / 0.1))
                for name, val in flat_fields(m):
                    acc.setdefault((name, b), []).append(val)
            for (name, b), vals in sorted(acc.items()):
                arrv = np.array(vals)
                w.writerow([tp, name, a.t_lo + 0.1 * b, len(vals), float(arrv.mean()), float(arrv.std())])
    print("wrote", a.out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
