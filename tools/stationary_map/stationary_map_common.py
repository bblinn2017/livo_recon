#!/usr/bin/env python3
"""Shared loaders and metrics for the stationary-map analysis tools (numpy only).

Everything here is sign-invariant in the plane normal: R52 compared |d_a - d_b|,
but d = -n.c depends on the arbitrary eigenvector sign, so parallel coplanar
patches with opposite normals looked ~2*|d| apart. Offsets are computed from
centres and normals only.
"""
import csv
import hashlib
import numpy as np

RANK2_EIG = 1e-8  # same floor as production VoxelMap (eig1, eig2 >= 1e-8)
csv.field_size_limit(1 << 30)  # child_ids lists of large surfaces exceed the 128 KiB default


# ---------------------------------------------------------------- loaders
def load_patches(path, final_only=True, checkpoint=None):
    """Load a stationary_map_cli patches CSV. `checkpoint` selects one exact
    checkpoint; otherwise with final_only the last one is kept. Returns dict
    of numpy arrays (child_ids is not parsed)."""
    rows = []
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        for r in rd:
            r.pop('child_ids', None)
            rows.append(r)
    if not rows:
        return empty_patches()
    if checkpoint is not None:
        rows = [r for r in rows if float(r['checkpoint']) == float(checkpoint)]
        if not rows:
            return empty_patches()
    elif final_only:
        last = max(float(r['checkpoint']) for r in rows)
        rows = [r for r in rows if float(r['checkpoint']) == last]
    col = lambda k: np.array([float(r[k]) for r in rows])
    eig = np.stack([col('eig0'), col('eig1'), col('eig2')], axis=1)
    out = {
        'checkpoint': col('checkpoint')[0],
        'patch_id': col('patch_id').astype(np.int64),
        'surface_id': col('surface_id').astype(np.int64),
        'point_count': col('point_count'),
        'center': np.stack([col('center_x'), col('center_y'), col('center_z')], axis=1),
        'normal': np.stack([col('normal_x'), col('normal_y'), col('normal_z')], axis=1),
        'd': col('d'),
        'eig': eig,
        'planarity': col('planarity'),
        'bb_min': np.stack([col('bb_min_x'), col('bb_min_y'), col('bb_min_z')], axis=1),
        'bb_max': np.stack([col('bb_max_x'), col('bb_max_y'), col('bb_max_z')], axis=1),
    }
    if 'rank2' in rows[0]:
        out['rank2'] = col('rank2') > 0.5
    else:  # R52-format files have no rank2 column
        out['rank2'] = (eig[:, 1] >= RANK2_EIG) & (eig[:, 2] >= RANK2_EIG)
    return out


def empty_patches():
    z3 = np.zeros((0, 3))
    return {'checkpoint': 0.0, 'patch_id': np.zeros(0, np.int64), 'surface_id': np.zeros(0, np.int64),
            'point_count': np.zeros(0), 'center': z3, 'normal': z3, 'd': np.zeros(0), 'eig': z3,
            'planarity': np.zeros(0), 'bb_min': z3, 'bb_max': z3, 'rank2': np.zeros(0, bool)}


def load_voxelmap_snapshot(path, planes_only=True):
    """Load VoxelMap post_calibration_voxelmap.csv as patches-like arrays.
    Only is_plane nodes are kept by default (they are the planes the estimator
    would use). eig columns are the covariance eigenvalues as fitted."""
    rows = list(csv.DictReader(open(path, newline='')))
    if planes_only:
        rows = [r for r in rows if r['is_plane'] in ('1', 'true', 'True')]
    if not rows:
        return empty_patches()
    col = lambda k: np.array([float(r[k]) for r in rows])
    eig = np.stack([col('eig0'), col('eig1'), col('eig2')], axis=1)
    return {
        'checkpoint': 0.0,
        'patch_id': np.arange(len(rows), dtype=np.int64),
        'surface_id': np.arange(len(rows), dtype=np.int64),
        'point_count': col('point_count'),
        'center': np.stack([col('center_x'), col('center_y'), col('center_z')], axis=1),
        'normal': np.stack([col('normal_x'), col('normal_y'), col('normal_z')], axis=1),
        'd': col('d'), 'eig': eig,
        'planarity': eig[:, 0] / np.maximum(eig.sum(axis=1), 1e-15),
        'bb_min': np.stack([col('cell_min_x'), col('cell_min_y'), col('cell_min_z')], axis=1),
        'bb_max': np.stack([col('cell_max_x'), col('cell_max_y'), col('cell_max_z')], axis=1),
        'rank2': (eig[:, 1] >= RANK2_EIG) & (eig[:, 2] >= RANK2_EIG),
    }


def load_reference(path):
    rows = list(csv.DictReader(open(path, newline='')))
    col = lambda k: np.array([float(r[k]) for r in rows])
    n = np.stack([col('nx'), col('ny'), col('nz')], axis=1)
    return {
        'surface_id': col('surface_id').astype(np.int64),
        'center': np.stack([col('cx'), col('cy'), col('cz')], axis=1),
        'normal': n, 'd': col('d'), 'points': col('points'),
        'bb_min': np.stack([col('bb_min_x'), col('bb_min_y'), col('bb_min_z')], axis=1),
        'bb_max': np.stack([col('bb_max_x'), col('bb_max_y'), col('bb_max_z')], axis=1),
        'plane_group': plane_groups(n, col('d')),
    }


def plane_groups(normals, d, decimals=6):
    """Reference surfaces produced from one RANSAC plane share (n, d) exactly
    (they are its disconnected components). Group them; the group is the
    physical plane."""
    keys = np.round(np.concatenate([normals, d[:, None]], axis=1), decimals)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    return inv.reshape(-1)


# ---------------------------------------------------------------- geometry
def angle_deg(na, nb):
    c = np.clip(np.abs(np.sum(na * nb, axis=-1)), 0.0, 1.0)
    return np.degrees(np.arccos(c))


def pair_offset(na, ca, nb, cb):
    """Sign-invariant: max of the two centre-to-plane distances."""
    dc = cb - ca
    return np.maximum(np.abs(np.sum(na * dc, axis=-1)), np.abs(np.sum(nb * dc, axis=-1)))


def partition_hash(patch_id, surface_id):
    """Hash of the partition of patches into surfaces (labels-independent)."""
    groups = {}
    for p, s in zip(patch_id.tolist(), surface_id.tolist()):
        groups.setdefault(s, []).append(p)
    canon = sorted(tuple(sorted(g)) for g in groups.values())
    return hashlib.sha256(repr(canon).encode()).hexdigest()


def cluster_pairwise(centers, normals, angle, offset, gap):
    """Single-linkage connected components under the pairwise test, using a
    spatial hash (no O(n^2) matrices). Returns integer labels."""
    n = len(centers)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n == 0:
        return parent
    keys = np.floor(centers / gap).astype(np.int64)
    buckets = {}
    for i, k in enumerate(map(tuple, keys)):
        buckets.setdefault(k, []).append(i)
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for i in range(n):
        k = keys[i]
        cand = []
        for o in offs:
            cand.extend(buckets.get((k[0] + o[0], k[1] + o[1], k[2] + o[2]), ()))
        cand = np.array([j for j in cand if j > i], dtype=np.int64)
        if len(cand) == 0:
            continue
        ok = angle_deg(normals[i], normals[cand]) <= angle
        ok &= pair_offset(normals[i], centers[i], normals[cand], centers[cand]) <= offset
        ok &= np.linalg.norm(centers[cand] - centers[i], axis=1) <= gap
        for j in cand[ok]:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    return np.array([find(i) for i in range(n)])


# ------------------------------------------------------- surface composition
def surface_composition(P, mask=None):
    """Per-surface statistics over (optionally masked) patches. Returns a
    dict of arrays indexed by unique surface and a summary dict."""
    idx = np.arange(len(P['patch_id'])) if mask is None else np.where(mask)[0]
    sids = P['surface_id'][idx]
    out = {'n': [], 'angle_p_max': [], 'extent': [], 'points': []}
    for s in np.unique(sids):
        m = idx[sids == s]
        nm = P['normal'][m]
        # sign-align to the first normal, then average
        sgn = np.sign(np.sum(nm * nm[0], axis=1))
        sgn[sgn == 0] = 1
        mean_n = (nm * sgn[:, None]).mean(axis=0)
        mean_n /= max(np.linalg.norm(mean_n), 1e-15)
        ang = angle_deg(nm, mean_n[None, :])
        c = P['center'][m]
        out['n'].append(len(m))
        out['angle_p_max'].append(float(ang.max()))
        out['extent'].append(float(np.linalg.norm(c.max(axis=0) - c.min(axis=0))))
        out['points'].append(float(P['point_count'][m].sum()))
    return {k: np.array(v) for k, v in out.items()}


# --------------------------------------------------------- reference matching
def match_to_reference(P, R, mask, angle=5.0, offset=0.05, margin=0.25, chunk=2000):
    """Assign each masked patch to the reference surface that (a) has normal
    within `angle`, (b) lies within `offset` of the patch centre
    (point-to-plane using the reference normal and centre), and (c) whose
    finite support (bounding box grown by `margin`) contains the patch centre.
    Ties go to the smallest offset. Returns int array (len(P)) with -1 where
    unmatched or unmasked."""
    n = len(P['patch_id'])
    assign = -np.ones(n, dtype=np.int64)
    idx = np.where(mask)[0]
    lo = R['bb_min'] - margin
    hi = R['bb_max'] + margin
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        c = P['center'][ii]           # (m,3)
        nn = P['normal'][ii]
        cosang = np.abs(nn @ R['normal'].T)                       # (m,M)
        ang_ok = np.degrees(np.arccos(np.clip(cosang, 0, 1))) <= angle
        dist = np.abs(np.einsum('mk,Mk->mM', c, R['normal']) - np.sum(R['normal'] * R['center'], axis=1)[None, :])
        inside = np.all((c[:, None, :] >= lo[None, :, :]) & (c[:, None, :] <= hi[None, :, :]), axis=2)
        ok = ang_ok & (dist <= offset) & inside
        dist = np.where(ok, dist, np.inf)
        best = np.argmin(dist, axis=1)
        has = np.isfinite(dist[np.arange(len(ii)), best])
        assign[ii[has]] = best[has]
    return assign


def reference_metrics(P, R, mask, **kw):
    """Support precision/recall, fragmentation and under-segmentation of the
    (masked) patches against the reference surfaces. All returned as a flat
    {metric: value} dict."""
    assign = match_to_reference(P, R, mask, **kw)
    m_idx = np.where(mask)[0]
    n_masked = len(m_idx)
    matched = assign[m_idx] >= 0
    out = {'patches_evaluated': n_masked,
           'support_precision': float(matched.mean()) if n_masked else float('nan'),
           'patches_matched': int(matched.sum())}
    ref_hit = np.unique(assign[m_idx][matched])
    out['reference_surfaces'] = len(R['surface_id'])
    out['reference_surfaces_hit'] = int(len(ref_hit))
    out['support_recall_count'] = len(ref_hit) / max(1, len(R['surface_id']))
    out['support_recall_points'] = float(R['points'][ref_hit].sum() / max(1.0, R['points'].sum()))
    if len(ref_hit):
        per_ref = np.array([np.sum(assign[m_idx] == r) for r in ref_hit])
        out['patches_per_hit_reference_p50'] = float(np.percentile(per_ref, 50))
        out['patches_per_hit_reference_p90'] = float(np.percentile(per_ref, 90))
        # over-segmentation: distinct backend surfaces per hit reference surface
        surf = P['surface_id']
        seg = np.array([len(np.unique(surf[m_idx][assign[m_idx] == r])) for r in ref_hit])
        out['backend_surfaces_per_hit_reference_mean'] = float(seg.mean())
        out['backend_surfaces_per_hit_reference_p90'] = float(np.percentile(seg, 90))
    # under-segmentation: backend surfaces whose matched patches span >1 reference plane
    surf = P['surface_id']
    grp = R['plane_group']
    n_surf = 0
    n_multi_plane = 0
    n_multi_component = 0
    for s in np.unique(surf[m_idx][matched]):
        sel = m_idx[matched][surf[m_idx][matched] == s]
        refs = assign[sel]
        n_surf += 1
        if len(np.unique(grp[refs])) > 1:
            n_multi_plane += 1
        elif len(np.unique(refs)) > 1:
            n_multi_component += 1
    out['matched_backend_surfaces'] = n_surf
    out['undersegmented_surfaces_multi_plane'] = n_multi_plane
    out['bridged_surfaces_same_plane_multi_component'] = n_multi_component
    out['undersegmentation_rate'] = n_multi_plane / max(1, n_surf)
    return out
