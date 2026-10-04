#!/usr/bin/env python3
"""R65 (F-164/F-173): message stamps and sizes of the eee_01 LiDAR and camera topics, read straight from the bag (no estimator).

    python3 tools/r65/bag_stamps.py --bag <eee_01.bag> --out R65_RETURN/bag_stamps
Needs the ROS1 python API (`rosbag`). Writes into --out:
  bag_stamps.csv.gz   one row per message on the LiDAR topics and the image topics:
                      topic, seq_in_topic, header_stamp, bag_time, width, height (PointCloud2/Image), n_bytes
  README.txt          bag path, bag start time, topics used; or 'rosbag import failed: <error>' if rosbag cannot be imported.
Topics used (from the R64 bag probe): /os1_cloud_node1/points, /os1_cloud_node2/points, /left/image_raw, /right/image_raw.
Reads only; reports what is there, interprets nothing."""
import argparse, gzip, os, sys

TOPICS = ["/os1_cloud_node1/points", "/os1_cloud_node2/points", "/left/image_raw", "/right/image_raw"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True); ap.add_argument("--out", default="bag_stamps")
    ap.add_argument("--topics", nargs="*", default=TOPICS)
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
    have = set(bag.get_type_and_topic_info().topics.keys())
    use = [t for t in a.topics if t in have]
    seq = {t: 0 for t in use}
    with gzip.open(os.path.join(a.out, "bag_stamps.csv.gz"), "wt") as f:
        f.write("topic,seq_in_topic,header_stamp,bag_time,width,height,n_bytes\n")
        for tp, m, t in bag.read_messages(topics=use):
            data = getattr(m, "data", b"")
            f.write("%s,%d,%.9f,%.9f,%s,%s,%d\n" % (tp, seq[tp], m.header.stamp.to_sec(), t.to_sec(),
                                                    getattr(m, "width", ""), getattr(m, "height", ""), len(data)))
            seq[tp] += 1
    open(os.path.join(a.out, "README.txt"), "w").write("bag %s start %.3f topics used %s missing %s counts %s\n" % (
        a.bag, t0, use, [t for t in a.topics if t not in have], seq))
    print("wrote", a.out, seq)
    return 0

if __name__ == "__main__":
    sys.exit(main())
