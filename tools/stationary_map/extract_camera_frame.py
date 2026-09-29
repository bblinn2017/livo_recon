#!/usr/bin/env python3
"""Extract camera frames from a ROS1 bag (no cv_bridge needed).

  extract_camera_frame.py --bag BAG.bag [--topic /left/image_raw] --out-dir DIR
      [--near-time T] [--near-manifest cache/manifest.csv --near-observation 0]

Saves DIR/camera_first.png (+ .json) for the first message and, if a target time is
given (--near-time, or --near-manifest with --near-observation whose 'timestamp' column
is used), DIR/camera_near_obs.png (+ .json) for the message nearest that time. The JSON
holds stamp, bag time, encoding, width, height, topic. The image is the RAW (distorted)
frame, exactly as published. Uses `rosbag` if importable, else `rosbags`.
Supports encodings mono8, bgr8, rgb8, mono16 (scaled to 8 bit).
"""
import argparse, csv, json, os
import numpy as np


def iter_msgs(bag, topic):
    try:
        import rosbag
        with rosbag.Bag(bag) as b:
            for _, m, t in b.read_messages(topics=[topic]):
                yield m, t.to_sec()
        return
    except ImportError:
        pass
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bag) as r:
        conns = [c for c in r.connections if c.topic == topic]
        for c, t, raw in r.messages(connections=conns):
            yield ts.deserialize_ros1(raw, c.msgtype), t * 1e-9


def decode(m):
    h, w, enc = int(m.height), int(m.width), m.encoding
    buf = np.frombuffer(bytes(m.data), dtype=np.uint8)
    step = int(m.step)
    if enc in ('mono8', '8UC1'):
        return buf.reshape(h, step)[:, :w]
    if enc in ('bgr8', 'rgb8'):
        img = buf.reshape(h, step)[:, :3 * w].reshape(h, w, 3)
        return img[:, :, ::-1] if enc == 'bgr8' else img
    if enc in ('mono16', '16UC1'):
        img = np.frombuffer(bytes(m.data), dtype='>u2' if m.is_bigendian else '<u2').reshape(h, step // 2)[:, :w]
        return (img / 256.0).astype(np.uint8)
    raise SystemExit('unsupported encoding ' + enc)


def save(img, path):
    try:
        from PIL import Image
        Image.fromarray(np.ascontiguousarray(img)).save(path)
    except ImportError:
        import cv2
        cv2.imwrite(path, img[:, :, ::-1] if img.ndim == 3 else img)


def stamp(m, bag_t):
    try:
        return m.header.stamp.to_sec()
    except AttributeError:
        return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--topic', default='/left/image_raw')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--near-time', type=float, default=None)
    ap.add_argument('--near-manifest', default=None)
    ap.add_argument('--near-observation', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    target = a.near_time
    if target is None and a.near_manifest:
        for r in csv.DictReader(open(a.near_manifest)):
            if int(r['observation_id']) == a.near_observation:
                target = float(r['timestamp'])
    first = None
    best = None
    count = 0
    for m, bt in iter_msgs(a.bag, a.topic):
        st = stamp(m, bt)
        count += 1
        if first is None:
            first = (st, bt, m)
            if target is None:
                break
        if target is not None:
            d = abs(st - target)
            if best is None or d < best[0]:
                best = (d, st, bt, m)
            elif st > target + 1.0:
                break

    def dump(tag, st, bt, m):
        img = decode(m)
        save(img, os.path.join(a.out_dir, tag + '.png'))
        json.dump({'topic': a.topic, 'stamp': st, 'bag_time': bt, 'encoding': m.encoding,
                   'width': int(m.width), 'height': int(m.height), 'target_time': target,
                   'note': 'raw distorted frame as published'},
                  open(os.path.join(a.out_dir, tag + '.json'), 'w'), indent=1)
        print(tag, 'stamp=%.6f' % st, m.encoding, m.width, 'x', m.height)

    if first is None:
        raise SystemExit('no messages on ' + a.topic)
    dump('camera_first', *first)
    if best is not None:
        dump('camera_near_obs', best[1], best[2], best[3])
        print('nearest frame is %.4f s from target' % best[0])


if __name__ == '__main__':
    main()
