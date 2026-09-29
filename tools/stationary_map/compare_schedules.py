#!/usr/bin/env python3
"""Does the checkpoint schedule change the result? Merge/split work runs inside snapshot(), so a run
that snapshots every scan can differ from one that snapshots sparsely.

  compare_schedules.py --a sparse_patches.csv --b every_scan_patches.csv --arm canonical --family gaussian_surface \
      [--out rows.csv]
Compares every checkpoint present in BOTH files: same patch ids, max |centre diff|, same surface partition
(partition equality up to relabelling), number of patches whose surface differs. Output rows: row_type=schedule_check.
"""
import argparse, csv, sys
import numpy as np


def groups(path, want_cp=None):
    out, cur, rows = {}, None, []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            cp = int(float(r['checkpoint']))
            if want_cp is not None and cp not in want_cp:
                continue
            if cp != cur:
                if cur is not None: out[cur] = rows
                cur, rows = cp, []
            rows.append((int(r['patch_id']), int(r['surface_id']), float(r['center_x']), float(r['center_y']), float(r['center_z'])))
    if cur is not None: out[cur] = rows
    return out


def partition(rows):
    by = {}
    for pid, sid, *_ in rows: by.setdefault(sid, []).append(pid)
    return {pid: min(m) for m in by.values() for pid in m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    ap.add_argument('--arm', default=''); ap.add_argument('--family', default=''); ap.add_argument('--out')
    a = ap.parse_args()
    ga = groups(a.a)
    gb = groups(a.b, set(ga))
    common = sorted(set(ga) & set(gb))
    out = []
    for cp in common:
        A = {r[0]: r for r in ga[cp]}; B = {r[0]: r for r in gb[cp]}
        same_ids = set(A) == set(B)
        ids = sorted(set(A) & set(B))
        dc = max((np.abs(np.array(A[i][2:]) - np.array(B[i][2:])).max() for i in ids), default=0.0)
        pa, pb = partition(ga[cp]), partition(gb[cp])
        ndiff = sum(1 for i in ids if pa[i] != pb[i])
        out.append((cp, same_ids, dc, ndiff == 0 and same_ids, ndiff, len(A), len(B)))
        print('cp %d  same_patch_ids=%s  max_centre_diff=%.3g  same_partition=%s  patches_with_different_surface=%d  (%d vs %d patches)'
              % (cp, same_ids, dc, ndiff == 0 and same_ids, ndiff, len(A), len(B)))
    if a.out:
        with open(a.out, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['row_type', 'arm', 'family', 'metric', 'value', 'detail'])
            for cp, si, dc, sp, nd, na, nb in out:
                w.writerow(['schedule_check', a.arm, a.family, 'same_partition', int(sp),
                            'checkpoint=%d same_patch_ids=%d max_centre_diff=%.3g patches_different_surface=%d n_a=%d n_b=%d' % (cp, si, dc, nd, na, nb)])
    if not common:
        sys.exit('no common checkpoints')


if __name__ == '__main__':
    main()
