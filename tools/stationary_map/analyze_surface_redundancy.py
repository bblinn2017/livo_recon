#!/usr/bin/env python3
"""Independent redundancy analysis of a stationary_map_cli patches CSV (final
checkpoint). numpy only, spatial-hash clustering (no O(n^2) matrices).

Two scopes are always reported: `all` (every emitted patch) and `credible`
(rank-2 patches only). R52 reported only `all`, and its offset test compared
|d_a - d_b| with arbitrary normal signs; both are fixed here. The clustering is
the same pairwise single-linkage test the backend's `pairwise` criterion uses,
so it is an independent re-implementation of THAT rule, not of the
`combined_fit` rule (use compare_backends.py for that comparison).

With --reference the credible patches are also matched to the finite-support
reference surfaces (support precision/recall, fragmentation, under-segmentation).
"""
import argparse
import csv
import numpy as np
import stationary_map_common as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--planes', required=True)
    p.add_argument('--reference')
    p.add_argument('--output', required=True)
    p.add_argument('--angle-deg', type=float, default=5.0)
    p.add_argument('--offset', type=float, default=0.05)
    p.add_argument('--gap', type=float, default=0.75)
    p.add_argument('--support-margin', type=float, default=0.25)
    a = p.parse_args()
    P = C.load_patches(a.planes)
    out = []
    for scope, mask in (('all', np.ones(len(P['patch_id']), bool)), ('credible', P['rank2'])):
        idx = np.where(mask)[0]
        lab = C.cluster_pairwise(P['center'][idx], P['normal'][idx], a.angle_deg, a.offset, a.gap) if len(idx) else np.zeros(0, int)
        counts = np.bincount(np.unique(lab, return_inverse=True)[1]) if len(idx) else np.zeros(0, int)
        out += [(f'{scope}_planes', len(idx)),
                (f'{scope}_candidate_surfaces', len(counts)),
                (f'{scope}_fragmentation_ratio', len(idx) / max(1, len(counts))),
                (f'{scope}_planes_in_redundant_groups', int(counts[counts > 1].sum())),
                (f'{scope}_largest_group', int(counts.max()) if len(counts) else 0)]
    out.append(('rank1_fraction', float(1.0 - P['rank2'].mean()) if len(P['rank2']) else float('nan')))
    if a.reference:
        R = C.load_reference(a.reference)
        for k, v in C.reference_metrics(P, R, P['rank2'], angle=a.angle_deg, offset=a.offset, margin=a.support_margin).items():
            out.append((f'ref_{k}', v))
    with open(a.output, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerows(out)


if __name__ == '__main__':
    main()
