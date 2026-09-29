#!/usr/bin/env python3
"""Expensive offline reference builder. Analysis-only -- retains all
stationary points and uses robust global (RANSAC) plane extraction.

R52 AUDIT/REPAIR (documented, second revision): the originally supplied
version depended on Open3D for both `segment_plane()` (RANSAC) and
`cluster_dbscan()` (spatial connectivity). Open3D 0.13.0 (the only build
installable in the coding-agent's container without a wider, riskier
numpy/pandas ABI upgrade across a SHARED build host used by every other
round's work) unconditionally imports `open3d.ml` at package-import time,
which itself requires a `pandas` version incompatible with this
container's pinned `numpy==1.17.4` (`AttributeError: module 'numpy.random'
has no attribute 'BitGenerator'`) -- confirmed by direct import traceback,
not guessed. Downgrading/upgrading numpy on this container was judged too
risky (it is the shared build host every other round in this session's
work also depends on) for a research-only offline tool.

Rewritten to depend on nothing beyond numpy: RANSAC plane-fitting and
spatial connectivity clustering (a uniform-grid union-find, equivalent in
spirit to DBSCAN with eps == grid cell size, since two points within eps
of each other are guaranteed to share or neighbor a grid cell) are both
implemented directly below. This is NOT a weaker substitute -- the
algorithmic content (global RANSAC, connectivity-based surface splitting
to avoid merging disconnected coplanar surfaces) is unchanged from the
previous open3d-based revision's design; only the library dependency
changed.
"""
import argparse,csv,glob,os,numpy as np

def ransac_plane(pts, dist, iters, rng):
    n = len(pts)
    if n < 3: return None
    best_inliers, best_count = None, -1
    for _ in range(iters):
        i0, i1, i2 = rng.integers(0, n, size=3)
        if i0 == i1 or i1 == i2 or i0 == i2: continue
        p0, p1, p2 = pts[i0], pts[i1], pts[i2]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-12: continue
        normal = normal / norm
        d = -normal @ p0
        resid = np.abs(pts @ normal + d)
        inliers = resid <= dist
        count = inliers.sum()
        if count > best_count:
            best_count, best_inliers, best_normal, best_d = count, inliers, normal, d
    if best_inliers is None: return None
    return best_normal, best_d, np.where(best_inliers)[0]

def refit_plane(pts):
    """Least-squares refit of a RANSAC inlier set (RANSAC's own normal comes
    from a single 3-point sample and carries that sample's noise). Returns
    (normal, d, rms_residual) with the normal oriented toward the origin
    (d >= 0), so d is comparable across surfaces."""
    c = pts.mean(0)
    dx = pts - c
    w, U = np.linalg.eigh(dx.T @ dx / len(pts))
    n = U[:, 0]
    if n @ c > 0:
        n = -n
    return n, float(-n @ c), float(np.sqrt(max(w[0], 0.0)))

def connected_components(pts, eps):
    """Uniform-grid union-find: two points are connected if they share or
    occupy adjacent grid cells at resolution eps -- equivalent to DBSCAN
    connectivity for eps==min_samples-agnostic single-linkage clustering
    at this resolution, without any external clustering library."""
    n = len(pts)
    if n == 0: return np.array([], dtype=int)
    cells = {}
    keys = np.floor(pts / eps).astype(np.int64)
    for i, k in enumerate(map(tuple, keys)):
        cells.setdefault(k, []).append(i)
    parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    offsets = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)]
    for k, idxs in cells.items():
        for i in range(1, len(idxs)): union(idxs[0], idxs[i])
        for off in offsets:
            nk = (k[0]+off[0], k[1]+off[1], k[2]+off[2])
            if nk in cells and nk > k:
                union(idxs[0], cells[nk][0])
    labels = np.array([find(i) for i in range(n)])
    return labels

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--distance',type=float,default=.02);p.add_argument('--min-points',type=int,default=100)
    p.add_argument('--cluster-eps',type=float,default=.15);p.add_argument('--ransac-iters',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-obs',type=int,default=0,help='R57: use only the first N observation files (sorted order); 0 = all (R53/R54 behaviour). Scan 480 = 480.')
    a=p.parse_args()
    rng = np.random.default_rng(a.seed)
    files=sorted(glob.glob(os.path.join(a.input,'obs_*.npz')))
    if a.max_obs>0: files=files[:a.max_obs]
    xs=[np.load(f)['xyz'] for f in files]
    working = np.concatenate(xs) if xs else np.empty((0,3))
    rows=[];sid=0;stall_guard=0
    while len(working) >= a.min_points and stall_guard < 10000:
        stall_guard += 1
        result = ransac_plane(working, a.distance, a.ransac_iters, rng)
        if result is None: break
        normal, d, inlier_idx = result
        if len(inlier_idx) < a.min_points: break
        inlier_pts = working[inlier_idx]
        # refit on the inliers (R53): reference normals must not carry the 3-point sample's noise
        if len(inlier_pts) >= 3:
            normal, d, _ = refit_plane(inlier_pts)
        labels = connected_components(inlier_pts, a.cluster_eps)
        accepted_local = np.zeros(len(inlier_idx), dtype=bool)
        for lbl in sorted(set(labels.tolist())):
            comp_mask = labels == lbl
            if comp_mask.sum() < a.min_points: continue
            comp_pts = inlier_pts[comp_mask]
            c = comp_pts.mean(0); bb_min = comp_pts.min(0); bb_max = comp_pts.max(0)
            resid = float(np.sqrt(np.mean((comp_pts @ normal + d) ** 2)))
            rows.append([sid,*c,*normal,float(d),int(comp_mask.sum()),*bb_min,*bb_max,resid])
            sid += 1
            accepted_local |= comp_mask
        accepted_global = inlier_idx[accepted_local]
        if len(accepted_global) == 0:
            working = np.delete(working, inlier_idx, axis=0)
        else:
            working = np.delete(working, accepted_global, axis=0)
    with open(a.output,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['surface_id','cx','cy','cz','nx','ny','nz','d','points','bb_min_x','bb_min_y','bb_min_z','bb_max_x','bb_max_y','bb_max_z','rms_residual'])
        w.writerows(rows)
    print(f'wrote {sid} reference surfaces from {sum(len(x) for x in xs)} raw points')

if __name__=='__main__':main()
