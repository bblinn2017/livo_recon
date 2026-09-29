#!/usr/bin/env python3
"""Independent numpy oracle for the C++ incremental_pca family.

Reads every cached observation (obs_*.npz), bins ALL points by floor(p/leaf),
and fits each cell from the raw points with a two-pass centred covariance (no
running sums), applying the same validity rule as the C++ default (production
VoxelMap parity). Output columns match compare_backends.py's --oracle input.
This is a cross-check of the C++ additive statistics and validity logic, not a
second benchmark.
"""
import argparse
import csv
import glob
import os
import time
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--leaf', type=float, default=.25)
    p.add_argument('--min-points', type=int, default=3)
    p.add_argument('--plane-eig-max', type=float, default=0.0025)  # = C++ CLI default (ntu_viral.yaml plane_threshold); R53 had 0.01 here, a mismatch
    p.add_argument('--min-secondary-eig', type=float, default=1e-8)
    p.add_argument('--max-planarity', type=float, default=1.0)
    a = p.parse_args()
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(a.input, 'obs_*.npz')))
    X = np.concatenate([np.load(f)['xyz'] for f in files]) if files else np.zeros((0, 3))
    keys = np.floor(X / a.leaf).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind='stable')
    bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    n_out = 0
    with open(a.output, 'w', newline='') as g:
        w = csv.writer(g)
        w.writerow(['patch_id', 'cx', 'cy', 'cz', 'nx', 'ny', 'nz', 'd', 'lambda0', 'lambda1', 'lambda2', 'points', 'rank2'])
        for ci in range(len(uniq)):
            x = X[order[bounds[ci]:bounds[ci + 1]]]
            n = len(x)
            if n < 3:
                continue
            c = x.mean(0)
            dx = x - c
            C = dx.T @ dx / n
            v, U = np.linalg.eigh(C)
            planarity = v[0] / max(v.sum(), 1e-15)
            rank2 = (v[1] >= 1e-8) and (v[2] >= 1e-8)
            valid = (n >= a.min_points and v[1] >= a.min_secondary_eig and v[2] >= a.min_secondary_eig
                     and v[0] < a.plane_eig_max and planarity <= a.max_planarity)
            if not valid:
                continue
            nrm = U[:, 0]
            if nrm @ c > 0:
                nrm = -nrm
            w.writerow([n_out, *c, *nrm, float(-nrm @ c), *v, n, int(rank2)])
            n_out += 1
    print('patches', n_out, 'cells', len(uniq), 'seconds', time.time() - t0)


if __name__ == '__main__':
    main()
