#!/usr/bin/env python3
"""S-2: how many angle-passing patch pairs flip their merge decision between the
R52 sign-dependent offset |d_a-d_b| and the sign-invariant planeOffset.

Also reports the split of flips that are sign-driven vs range/tilt-driven, and
repeats the count with the merge_gap (centre distance) gate applied.

  offset_sign_analysis.py --geometry GEOM.csv [--checkpoint N] [--merge-angle 5]
      [--merge-offset 0.05] [--merge-gap 0.75] [--out out.csv]
GEOM.csv needs columns: checkpoint, center_x/y/z, normal_x/y/z, d.
Prints results and (optionally) writes a row_type=suspected_item CSV.
"""
import argparse, csv, math
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geometry', required=True)
    ap.add_argument('--checkpoint', type=int, default=None, help='default: last')
    ap.add_argument('--merge-angle', type=float, default=5.0)
    ap.add_argument('--merge-offset', type=float, default=0.05)
    ap.add_argument('--merge-gap', type=float, default=0.75)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.geometry)))
    cp = a.checkpoint if a.checkpoint is not None else max(int(r['checkpoint']) for r in rows)
    rows = [r for r in rows if int(r['checkpoint']) == cp]
    C = np.array([[float(r['center_x']), float(r['center_y']), float(r['center_z'])] for r in rows])
    N = np.array([[float(r['normal_x']), float(r['normal_y']), float(r['normal_z'])] for r in rows])
    D = np.array([float(r['d']) for r in rows])
    n = len(rows)
    print('checkpoint', cp, 'patches', n)
    cell = a.merge_gap * 2.0
    keys = np.floor(C / cell).astype(int)
    buckets = {}
    for i, k in enumerate(map(tuple, keys)):
        buckets.setdefault(k, []).append(i)
    pairs = set()
    for k, idx in buckets.items():
        nb = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nb.extend(buckets.get((k[0] + dx, k[1] + dy, k[2] + dz), []))
        for i in idx:
            for j in nb:
                if i < j:
                    pairs.add((i, j))
    P = np.array(sorted(pairs))
    i, j = P[:, 0], P[:, 1]
    dist = np.linalg.norm(C[j] - C[i], axis=1)
    cosang = np.clip(np.abs(np.sum(N[i] * N[j], axis=1)), -1, 1)
    ang = np.degrees(np.arccos(cosang))
    dc = C[j] - C[i]
    new_off = np.maximum(np.abs(np.sum(N[i] * dc, axis=1)), np.abs(np.sum(N[j] * dc, axis=1)))
    old_off = np.abs(D[i] - D[j])
    # sign-aligned d difference: flip one d when normals point opposite ways
    dot = np.sum(N[i] * N[j], axis=1)
    aligned = np.abs(D[i] - np.where(dot < 0, -D[j], D[j]))
    res = []
    for label, gate in (('all_nearby', np.ones(len(P), bool)), ('within_gap', dist <= a.merge_gap)):
        m = gate & (ang <= a.merge_angle)
        old_m = old_off <= a.merge_offset
        new_m = new_off <= a.merge_offset
        flips = m & (old_m != new_m)
        sd = flips & ((aligned <= a.merge_offset) == new_m)   # aligned-d test agrees with new -> flip is sign-driven
        line = (label, int(gate.sum()), int(m.sum()), int(flips.sum()), int(sd.sum()),
                int(flips.sum() - sd.sum()))
        print('%-10s pairs=%d angle_pass=%d flips=%d (%.1f%%) sign_driven=%d range_or_tilt_driven=%d'
              % (line[0], line[1], line[2], line[3], 100.0 * line[3] / max(1, line[2]), line[4], line[5]))
        res.append(line)
    if a.out:
        with open(a.out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['row_type', 'arm', 'metric', 'value', 'detail'])
            for l in res:
                w.writerow(['suspected_item', 'S-2_' + l[0], 'flips/angle_pass',
                            '%d/%d' % (l[3], l[2]),
                            'pairs=%d sign_driven=%d range_or_tilt=%d checkpoint=%d' % (l[1], l[4], l[5], cp)])


if __name__ == '__main__':
    main()
