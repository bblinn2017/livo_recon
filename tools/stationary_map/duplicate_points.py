#!/usr/bin/env python3
"""B.4 over ALL scans: exact-duplicate points and beam-direction repetition.

  duplicate_points.py --cache CACHE_DIR [--out out.csv] [--max-scans N]
      [--lidar-origin x y z] [--bin-deg 0.1]

Per scan k: n, fraction of points whose exact (1e-9 m rounded) coordinates appeared in
ANY earlier scan (hash based), fraction appearing in the immediately previous scan, and
(az,el) bin repetition (vs previous scan, and vs ANY earlier scan). Beam directions are
computed from the LiDAR origin (default: the IMU-frame extrinsic translation
-0.050 0 0.055 for an IMU-frame cache; pass 0 0 0 for a sensor-frame cache).
Output rows: row_type=duplicate_points.
"""
import argparse, csv, os
import numpy as np


def load(cache, r):
    d = np.load(os.path.join(cache, r['file']))
    return d['points'] if 'points' in d.files else d[d.files[0]]


def hashes(p):
    q = np.round(p.astype(np.float64) * 1e9).astype(np.int64)
    h = q[:, 0] * np.int64(73856093) ^ q[:, 1] * np.int64(19349663) ^ q[:, 2] * np.int64(83492791)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--max-scans', type=int, default=None)
    ap.add_argument('--lidar-origin', type=float, nargs=3, default=[-0.050, 0.0, 0.055])
    ap.add_argument('--bin-deg', type=float, default=0.1)
    a = ap.parse_args()
    rows = sorted(csv.DictReader(open(os.path.join(a.cache, 'manifest.csv'))),
                  key=lambda r: int(r['observation_id']))
    if a.max_scans:
        rows = rows[:a.max_scans]
    org = np.array(a.lidar_origin)
    seen_h = np.zeros(0, dtype=np.int64)
    seen_b = np.zeros(0, dtype=np.int64)
    prev_h = np.zeros(0, dtype=np.int64)
    prev_b = np.zeros(0, dtype=np.int64)
    out = []
    for r in rows:
        k = int(r['observation_id'])
        p = load(a.cache, r)
        n = len(p)
        h = hashes(p)
        v = p - org
        nr = np.linalg.norm(v, axis=1)
        nr[nr == 0] = 1.0
        v = v / nr[:, None]
        az = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
        el = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
        b = (np.round(az / a.bin_deg).astype(np.int64) * 100003 + np.round(el / a.bin_deg).astype(np.int64))
        dup_any = float(np.isin(h, seen_h).mean()) if len(seen_h) else 0.0
        dup_prev = float(np.isin(h, prev_h).mean()) if len(prev_h) else 0.0
        ub = np.unique(b)
        rep_prev = float(np.isin(ub, prev_b).mean()) if len(prev_b) else 0.0
        rep_any = float(np.isin(ub, seen_b).mean()) if len(seen_b) else 0.0
        out.append((k, n, dup_any, dup_prev, len(ub), rep_prev, rep_any))
        seen_h = np.union1d(seen_h, h)
        seen_b = np.union1d(seen_b, ub)
        prev_h, prev_b = np.unique(h), ub
        if k % 25 == 0:
            print('scan %d n=%d dup_any=%.4f dup_prev=%.4f beam_rep_prev=%.4f beam_rep_any=%.4f'
                  % (k, n, dup_any, dup_prev, rep_prev, rep_any), flush=True)
    if a.out:
        with open(a.out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['row_type', 'arm', 'metric', 'value', 'detail'])
            for k, n, da, dp, nb, rp, ra in out:
                for m, v in (('dup_frac_any_earlier', da), ('dup_frac_prev', dp),
                             ('beam_bin_repeat_prev', rp), ('beam_bin_repeat_any_earlier', ra)):
                    w.writerow(['duplicate_points', os.path.basename(a.cache.rstrip('/')), m,
                                '%.6f' % v, 'scan=%d n=%d distinct_bins=%d' % (k, n, nb)])
    print('scans', len(out))


if __name__ == '__main__':
    main()
