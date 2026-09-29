#!/usr/bin/env python3
"""Comparison rows for the stationary-map benchmark, computed from the CLI's
raw patch CSVs (numpy only). One long-format CSV: row_type,arm,family,metric,value.

row_type values
  family_summary      per family: patches, surfaces, rank-1 fraction, credible counts
  identity            per family pair: are the surface partitions / geometries identical?
                      (R52's gaussian_surface was byte-identical to mergeable_voxel)
  composition         credible multi-member surfaces: angle-to-surface-mean and extent spread
  reference           credible patches vs finite-support reference surfaces
  voxelmap_control    VoxelMap post-calibration plane nodes vs the same reference, and (when a
                      matching harness checkpoint is given) side by side with the harness
  oracle_agreement    C++ incremental_pca vs the numpy oracle (stationary_map_benchmark.py)

Example
  compare_backends.py --patches-dir OUT --arm canonical --reference reference_surfaces.csv \
     --voxelmap pfn1_pca=/path/post_calibration_voxelmap.csv --match-points 6912 \
     --oracle oracle_canonical.csv --output R53_comparison_canonical.csv
Patch files are expected at OUT/<family>_<arm>_patches.csv.
"""
import argparse
import csv
import os
import numpy as np
import stationary_map_common as C

FAMILIES = ['incremental_pca', 'mergeable_voxel', 'robust_voxel', 'robust_mergeable', 'gaussian_surface']


def summary_rows(P):
    n = len(P['patch_id'])
    cred = P['rank2']
    sid = P['surface_id']
    out = {'patches': n, 'surfaces': len(np.unique(sid)),
           'rank1_patches': int((~cred).sum()),
           'rank1_fraction': float(1 - cred.mean()) if n else float('nan'),
           'credible_patches': int(cred.sum()),
           'credible_surfaces': len(np.unique(sid[cred])) if cred.any() else 0,
           'total_points_in_patches': float(P['point_count'].sum())}
    if n:
        cnt = np.bincount(np.unique(sid, return_inverse=True)[1])
        out['largest_surface_patches'] = int(cnt.max())
        out['multi_member_surfaces'] = int((cnt > 1).sum())
    return out


def composition_rows(P):
    cred = P['rank2']
    comp = C.surface_composition(P, cred)
    multi = comp['n'] > 1
    out = {'credible_multi_member_surfaces': int(multi.sum()),
           'credible_patches_in_multi_member_surfaces': int(comp['n'][multi].sum()) if multi.any() else 0}
    if multi.any():
        # worst patch-to-surface-mean angle within each multi-member surface
        for q in (50, 90, 99, 100):
            out[f'surface_max_angle_deg_p{q}'] = float(np.percentile(comp['angle_p_max'][multi], q))
            out[f'surface_extent_m_p{q}'] = float(np.percentile(comp['extent'][multi], q))
    return out


def emit(rows, row_type, arm, family, d):
    for k, v in d.items():
        rows.append([row_type, arm, family, k, v])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--patches-dir', required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--families', default=','.join(FAMILIES))
    p.add_argument('--reference')
    p.add_argument('--voxelmap', action='append', default=[], help='LABEL=path/to/post_calibration_voxelmap.csv')
    p.add_argument('--harness-checkpoint', type=float, help='explicit harness checkpoint (observation count) to set beside the VoxelMap snapshot')
    p.add_argument('--match-points', type=float, help='total prepared points in the VoxelMap snapshot (sum of point_count over its observations); per family the checkpoint whose cumulative ingested points (summary CSV total_raw_points_ingested) is closest is used. Production observations are camera-frame windows, harness observations are whole scans, so observation counts cannot be matched; points can.')
    p.add_argument('--oracle')
    p.add_argument('--angle-deg', type=float, default=5.0)
    p.add_argument('--offset', type=float, default=0.05)
    p.add_argument('--support-margin', type=float, default=0.25)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    fams = a.families.split(',')
    R = C.load_reference(a.reference) if a.reference else None
    rows = []
    P = {}
    for f in fams:
        path = os.path.join(a.patches_dir, f'{f}_{a.arm}_patches.csv')
        if not os.path.exists(path):
            rows.append(['family_summary', a.arm, f, 'missing_file', path])
            continue
        P[f] = C.load_patches(path)
        emit(rows, 'family_summary', a.arm, f, summary_rows(P[f]))
        emit(rows, 'composition', a.arm, f, composition_rows(P[f]))
        if R is not None:
            emit(rows, 'reference', a.arm, f, C.reference_metrics(P[f], R, P[f]['rank2'], angle=a.angle_deg, offset=a.offset, margin=a.support_margin))
    # identity between every pair of families
    ks = list(P)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            A, B = P[ks[i]], P[ks[j]]
            same_ids = np.array_equal(A['patch_id'], B['patch_id'])
            same_geom = same_ids and np.allclose(A['center'], B['center'], atol=1e-9) and np.allclose(A['normal'], B['normal'], atol=1e-9)
            same_part = same_ids and C.partition_hash(A['patch_id'], A['surface_id']) == C.partition_hash(B['patch_id'], B['surface_id'])
            emit(rows, 'identity', a.arm, f'{ks[i]}|{ks[j]}',
                 {'same_patch_ids': int(same_ids), 'same_geometry': int(same_geom), 'same_surface_partition': int(same_part)})
    # VoxelMap control
    for spec in a.voxelmap:
        label, path = spec.split('=', 1)
        V = C.load_voxelmap_snapshot(path)
        s = summary_rows(V)
        s['note_surfaces'] = 'VoxelMap has no surface aggregation; each plane node is its own surface'
        emit(rows, 'voxelmap_control', a.arm, label, s)
        if R is not None:
            emit(rows, 'voxelmap_control', a.arm, label, {'ref_' + k: v for k, v in
                 C.reference_metrics(V, R, V['rank2'], angle=a.angle_deg, offset=a.offset, margin=a.support_margin).items()})
        if a.harness_checkpoint is not None or a.match_points is not None:
            for f in fams:
                path_f = os.path.join(a.patches_dir, f'{f}_{a.arm}_patches.csv')
                if not os.path.exists(path_f):
                    continue
                cp = a.harness_checkpoint
                if cp is None:
                    sm = list(csv.DictReader(open(os.path.join(a.patches_dir, f'{f}_{a.arm}_summary.csv'))))
                    best = min(sm, key=lambda r: abs(float(r['total_raw_points_ingested']) - a.match_points))
                    cp = float(best['checkpoint'])
                    emit(rows, 'voxelmap_control', a.arm, f'{f}@cp{cp:g}',
                         {'matched_cumulative_points': float(best['total_raw_points_ingested']),
                          'target_points': a.match_points})
                H = C.load_patches(path_f, checkpoint=cp)
                emit(rows, 'voxelmap_control', a.arm, f'{f}@cp{cp:g}', summary_rows(H))
                if R is not None:
                    emit(rows, 'voxelmap_control', a.arm, f'{f}@cp{cp:g}',
                         {'ref_' + k: v for k, v in C.reference_metrics(H, R, H['rank2'], angle=a.angle_deg, offset=a.offset, margin=a.support_margin).items()})
    # oracle agreement (incremental_pca only)
    if a.oracle and 'incremental_pca' in P:
        H = P['incremental_pca']
        orc = list(csv.DictReader(open(a.oracle, newline='')))
        key = lambda c: tuple(np.round(c, 6))
        hmap = {key(c): i for i, c in enumerate(H['center'])}
        matched, maxeig, maxcen = 0, 0.0, 0.0
        for r in orc:
            c = np.array([float(r['cx']), float(r['cy']), float(r['cz'])])
            i = hmap.get(key(c))
            if i is None:
                continue
            matched += 1
            e = np.array([float(r['lambda0']), float(r['lambda1']), float(r['lambda2'])])
            maxeig = max(maxeig, float(np.abs(e - H['eig'][i]).max()))
            maxcen = max(maxcen, float(np.abs(c - H['center'][i]).max()))
        emit(rows, 'oracle_agreement', a.arm, 'incremental_pca',
             {'oracle_patches': len(orc), 'harness_patches': len(H['patch_id']), 'matched_by_center': matched,
              'max_abs_eig_diff': maxeig, 'max_abs_center_diff': maxcen})
    with open(a.output, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['row_type', 'arm', 'family', 'metric', 'value'])
        w.writerows(rows)


if __name__ == '__main__':
    main()
