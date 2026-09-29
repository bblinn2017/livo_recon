#!/usr/bin/env python3
"""Production camera-window point counts beside harness scan counts (no pass/fail gate).

  window_equivalence.py --production-support post_calibration_voxel_observation_support.csv
      --cache CACHE_DIR [--out out.csv]

Production observations are camera-frame LiDAR windows; harness observations are scans.
This prints production per-window prepared counts (sum of point_count per observation_id)
and CUMULATIVE totals, beside the harness cumulative totals per scan, and for each
production cumulative total the nearest harness scan index (so a matching checkpoint can be
chosen by cumulative points, not by observation index). Output row_type=cache_equivalence.
"""
import argparse, csv, os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--production-support', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    prod = defaultdict(float)
    for r in csv.DictReader(open(a.production_support)):
        prod[int(r['observation_id'])] += float(r['point_count'])
    hm = sorted(csv.DictReader(open(os.path.join(a.cache, 'manifest.csv'))),
                key=lambda r: int(r['observation_id']))
    hc, tot = [], 0
    for r in hm:
        tot += int(r['points'])
        hc.append(tot)
    rows, ptot = [], 0
    for k in sorted(prod):
        ptot += prod[k]
        j = min(range(len(hc)), key=lambda i: abs(hc[i] - ptot))
        hn = int(hm[k]['points']) if k < len(hm) else -1
        print('prod window %d: %d pts cum=%d | harness scan %d: %d pts cum=%d | nearest harness cum: '
              'scan %d (%d)' % (k, prod[k], ptot, k, hn, hc[k] if k < len(hc) else -1, j, hc[j]))
        rows.append(('window_%d' % k, prod[k], ptot, hn, j, hc[j]))
    if a.out:
        with open(a.out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['row_type', 'arm', 'metric', 'value', 'detail'])
            for name, pn, pc, hn, j, hcj in rows:
                w.writerow(['cache_equivalence', os.path.basename(a.cache.rstrip('/')), name, '%d' % pc,
                            'prod_window_points=%d prod_cum=%d harness_same_index_points=%d '
                            'nearest_harness_scan_by_cum=%d harness_cum_there=%d' % (pn, pc, hn, j, hcj)])


if __name__ == '__main__':
    main()
