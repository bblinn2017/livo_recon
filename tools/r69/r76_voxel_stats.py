#!/usr/bin/env python3
"""r76_voxel_stats: lidar-scan statistics and candidate estimates of voxel edge E / max_layer.

The map is an octree: root voxel of edge E (layer 0), non-planar nodes split into 8
children down to max_layer; node edge at layer L is E/2**L.  A node with >= min_init
points is plane-fit; it is a plane when its smallest population-covariance (ddof=0)
eigenvalue e0 < plane_threshold, and line-like when e1 < plane_threshold as well.

This tool only REPORTS NUMBERS.  It never runs the odometry and never decides anything.
ALL estimator constants (c in {1.5,2,3}, 20 points/voxel, 1 percent plateau, f=10,
75th/10th percentile ranges, layer clip 0..3, the 8-NN spacing) are the planner's
CANDIDATE choices and are NOT validated; every estimate row says so in `note`.
octree_simulation approximates the map's octree on ONE scan (no map accumulation, no
pose motion, root nodes anchored at `origin`); it is not the real map.

Caveats worth knowing when reading the output:
  * Ring elevation / across-ring statistics are measured in the frame of the xyz given;
    after gravity alignment a body tilt of a few degrees slightly smears per-ring
    elevations (medians over azimuth absorb most of it).
  * pfn>1 keeps every pfn-th ORIGINAL point, so along-ring steps and d8 grow with pfn.
  * Medians over scans of discrete quantities (value_max_layer, plane-leaf counts) can
    be fractional.
  * `rosbag` is imported lazily inside read_bag_scans; the rest needs only numpy/scipy/pandas.
"""
import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

NAN = float("nan")

# PointField datatype -> numpy dtype string (without byte order)
_PF_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}

SPACING_COLS = ["seq", "pfn", "scan", "range_lo", "range_hi", "n_points", "d1_median", "d4_median",
                "d8_median", "d8_p10", "d8_p90", "aniso_median", "across_ring_gap_deg",
                "across_ring_gap_m", "along_ring_step_m", "n_scans"]
EDGES_COLS = ["seq", "pfn", "scan", "edge", "layer", "node_edge", "occupied_nodes",
              "pts_per_node_median", "pts_per_node_p10", "pts_per_node_p90", "frac_nodes_fit",
              "frac_fit_plane", "frac_fit_line", "frac_points_in_fit_plane", "n_scans"]
OCTREE_COLS = ["seq", "pfn", "scan", "edge", "max_layer", "frac_points_plane_leaf",
               "frac_points_line_leaf", "frac_points_nonplane_leaf", "frac_points_sparse_leaf",
               "n_plane_leaves", "plane_leaf_edge_median_pointweighted", "plane_leaf_pts_median",
               "n_scans"]
ESTIMATES_COLS = ["seq", "pfn", "scan", "estimator", "param", "value_edge_m", "value_max_layer",
                  "plane_leaf_fraction", "note", "n_scans"]

_UNVAL = "unvalidated candidate; "


# --------------------------------------------------------------------------- parsing
def msg_to_arrays(msg):
    """Parse a PointCloud2-like object (no ROS libs) -> dict x,y,z float64 (finite rows only),
    optional ring (int64) and index (int64, position in the ORIGINAL ordering)."""
    end = ">" if getattr(msg, "is_bigendian", False) else "<"
    names, formats, offsets = [], [], []
    for f in msg.fields:
        if f.name in ("x", "y", "z", "ring") and f.datatype in _PF_DTYPES and getattr(f, "count", 1) == 1:
            names.append(f.name)
            formats.append(end + _PF_DTYPES[f.datatype])
            offsets.append(int(f.offset))
    for need in ("x", "y", "z"):
        if need not in names:
            raise ValueError("PointCloud2 has no usable field '%s'" % need)
    n = int(msg.width) * int(msg.height)
    dt = np.dtype({"names": names, "formats": formats, "offsets": offsets,
                   "itemsize": int(msg.point_step)})
    raw = np.frombuffer(bytes(msg.data), dtype=dt, count=n)
    x = raw["x"].astype(np.float64)
    y = raw["y"].astype(np.float64)
    z = raw["z"].astype(np.float64)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    out = {"x": x[keep], "y": y[keep], "z": z[keep]}
    if "ring" in names:
        out["ring"] = raw["ring"][keep].astype(np.int64)
    out["index"] = np.nonzero(keep)[0].astype(np.int64)
    return out


def gravity_rotation(accel_samples):
    """Rotation mapping the mean specific-force direction to +z (accel at rest points up)."""
    a = np.asarray(accel_samples, dtype=np.float64).reshape(-1, 3) if np.size(accel_samples) else np.zeros((0, 3))
    if a.shape[0] == 0:
        return np.eye(3)
    f = a.mean(axis=0)
    nf = np.linalg.norm(f)
    if not np.isfinite(nf) or nf < 1e-12:
        return np.eye(3)
    f = f / nf
    z = np.array([0.0, 0.0, 1.0])
    c = float(f @ z)
    if c < -1.0 + 1e-12:  # antiparallel: any 180 deg turn about a horizontal axis
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(f, z)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def filter_points(arrays, pfn, min_range, max_range):
    """Index stride on the ORIGINAL ordering first, then range band [min,max]."""
    r = np.sqrt(arrays["x"] ** 2 + arrays["y"] ** 2 + arrays["z"] ** 2)
    idx = arrays["index"] if "index" in arrays else np.arange(len(r))
    m = (idx % int(pfn) == 0) & (r >= min_range) & (r <= max_range)
    return {k: np.asarray(v)[m] for k, v in arrays.items()}


# --------------------------------------------------------------------------- spacing
def knn_features(xyz, k=8):
    """Per point: distances to self + k neighbours (n,k+1) and aniso=sqrt(l2/l1) of the
    (k+1)-point set covariance.  NaN arrays when fewer than k+1 points."""
    n = len(xyz)
    if n < k + 1:
        return {"d": np.full((n, k + 1), np.nan), "aniso": np.full(n, np.nan)}
    d, idx = cKDTree(xyz).query(xyz, k=k + 1, workers=1)
    p = xyz[idx]
    p = p - p.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", p, p) / (k + 1)
    ev = np.linalg.eigvalsh(cov)
    with np.errstate(divide="ignore", invalid="ignore"):
        aniso = np.where(ev[:, 1] > 0, np.sqrt(ev[:, 2] / ev[:, 1]), np.nan)
    return {"d": d, "aniso": aniso}


def _rings_median_elev(xyz, ring, min_pts=20):
    r = np.linalg.norm(xyz, axis=1)
    el = np.arcsin(np.clip(xyz[:, 2] / np.where(r > 0, r, 1.0), -1, 1))
    out = []
    for g in np.unique(ring):
        m = ring == g
        if m.sum() >= min_pts:
            out.append(np.median(el[m]))
    return np.sort(np.array(out))


def across_ring_gap_deg(xyz, ring):
    """Median gap between median elevations of consecutive rings (rings with >=20 points)."""
    if ring is None or len(xyz) == 0:
        return NAN
    e = _rings_median_elev(xyz, ring)
    return float(np.degrees(np.median(np.diff(e)))) if len(e) >= 2 else NAN


def along_ring_steps(xyz, ring):
    """Per point: angular step to the next same-ring point by azimuth (wrapping) times
    range * cos(elevation).  NaN for rings with a single point."""
    n = len(xyz)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    r = np.linalg.norm(xyz, axis=1)
    el = np.arcsin(np.clip(xyz[:, 2] / np.where(r > 0, r, 1.0), -1, 1))
    order = np.lexsort((az, ring))
    rs, azs = ring[order], az[order]
    starts = np.nonzero(np.r_[True, rs[1:] != rs[:-1]])[0]
    ends = np.r_[starts[1:], n] - 1
    nxt = np.empty(n)
    nxt[:-1] = azs[1:] - azs[:-1]
    for s, e in zip(starts, ends):
        nxt[e] = azs[s] + 2 * np.pi - azs[e]  # wrap to the first point of the ring
        if s == e:
            nxt[e] = np.nan
    out[order] = nxt * r[order] * np.cos(el[order])
    return out


def spacing_stats(xyz, ring, range_bins, k=8, min_bin_points=20, _knn=None):
    """One row per range bin [lo,hi); bins with < min_bin_points points give NaN stats."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    r = np.linalg.norm(xyz, axis=1)
    knn = _knn if _knn is not None else knn_features(xyz, k)
    d = knn["d"]
    if ring is not None:
        gap_deg = across_ring_gap_deg(xyz, ring)
        step = along_ring_steps(xyz, ring)
    else:
        gap_deg, step = NAN, np.full(len(xyz), np.nan)
    rows = []
    for lo, hi in zip(range_bins[:-1], range_bins[1:]):
        m = (r >= lo) & (r < hi)
        nb = int(m.sum())
        row = dict(range_lo=float(lo), range_hi=float(hi), n_points=nb, d1_median=NAN, d4_median=NAN,
                   d8_median=NAN, d8_p10=NAN, d8_p90=NAN, aniso_median=NAN,
                   across_ring_gap_deg=gap_deg, across_ring_gap_m=NAN, along_ring_step_m=NAN)
        if nb >= min_bin_points and d.shape[1] > k:
            row["d1_median"] = float(np.median(d[m, 1]))
            row["d4_median"] = float(np.median(d[m, 4]))
            row["d8_median"] = float(np.median(d[m, k]))
            row["d8_p10"] = float(np.percentile(d[m, k], 10))
            row["d8_p90"] = float(np.percentile(d[m, k], 90))
            a = knn["aniso"][m]
            row["aniso_median"] = float(np.median(a[np.isfinite(a)])) if np.isfinite(a).any() else NAN
            if ring is not None:
                row["across_ring_gap_m"] = float(np.radians(gap_deg) * np.median(r[m]))
                s = step[m]
                row["along_ring_step_m"] = float(np.median(s[np.isfinite(s)])) if np.isfinite(s).any() else NAN
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- voxels
def _group_nodes(idx):
    """Group integer node coordinates (n,3) -> (inverse, counts, first_index)."""
    mn = idx.min(axis=0)
    sh = idx - mn
    dims = [int(v) + 1 for v in sh.max(axis=0)]
    if dims[0] * dims[1] * dims[2] < 2 ** 62:
        key = (sh[:, 0] * dims[1] + sh[:, 1]) * dims[2] + sh[:, 2]
        _, first, inv, counts = np.unique(key, return_index=True, return_inverse=True, return_counts=True)
    else:
        _, first, inv, counts = np.unique(idx, axis=0, return_index=True, return_inverse=True,
                                          return_counts=True)
    return inv.reshape(-1), counts, first


def _node_stats(xyz, node_edge, origin, min_init):
    """counts, inverse, eigenvalues (nodes,3 ascending; NaN rows for nodes with < min_init points).
    Moments are accumulated in node-local coordinates so large offsets do not cancel."""
    org = np.asarray(origin, dtype=np.float64)
    idx = np.floor((xyz - org) / node_edge).astype(np.int64)
    inv, counts, first = _group_nodes(idx)
    nn = len(counts)
    loc = xyz - (idx[first] * node_edge + org)[inv]
    s1 = np.stack([np.bincount(inv, loc[:, i], nn) for i in range(3)], axis=1)
    pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    s2 = {p: np.bincount(inv, loc[:, p[0]] * loc[:, p[1]], nn) for p in pairs}
    mean = s1 / counts[:, None]
    cov = np.empty((nn, 3, 3))
    for (i, j) in pairs:
        cov[:, i, j] = cov[:, j, i] = s2[(i, j)] / counts - mean[:, i] * mean[:, j]
    eig = np.full((nn, 3), np.nan)
    fit = counts >= min_init
    if fit.any():
        eig[fit] = np.linalg.eigvalsh(cov[fit])
    return counts, inv, eig


def voxel_layer_stats(xyz, edge, layer, min_init, plane_threshold, origin=(0, 0, 0)):
    """Statistics of the occupied nodes of one layer (edge/2**layer) of an octree grid."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    ne = edge / 2 ** layer
    res = dict(node_edge=ne, occupied_nodes=0, pts_per_node_median=NAN, pts_per_node_p10=NAN,
               pts_per_node_p90=NAN, frac_nodes_fit=NAN, frac_fit_plane=NAN, frac_fit_line=NAN,
               frac_points_in_fit_plane=NAN)
    if len(xyz) == 0:
        return res
    counts, _, eig = _node_stats(xyz, ne, origin, min_init)
    fit = counts >= min_init
    res["occupied_nodes"] = int(len(counts))
    res["pts_per_node_median"] = float(np.median(counts))
    res["pts_per_node_p10"] = float(np.percentile(counts, 10))
    res["pts_per_node_p90"] = float(np.percentile(counts, 90))
    res["frac_nodes_fit"] = float(fit.mean())
    if fit.any():
        plane = fit & (eig[:, 0] < plane_threshold)
        line = plane & (eig[:, 1] < plane_threshold)
        res["frac_fit_plane"] = float(plane.sum() / fit.sum())
        res["frac_fit_line"] = float(line.sum() / fit.sum())
        res["frac_points_in_fit_plane"] = float(counts[plane].sum() / len(xyz))
    else:
        res["frac_points_in_fit_plane"] = 0.0
    return res


def octree_simulation(xyz, edge, max_layer, min_init, plane_threshold, origin=(0, 0, 0)):
    """Layer-by-layer approximation of the map octree on one scan (see module docstring)."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    n = len(xyz)
    res = dict(frac_points_plane_leaf=NAN, frac_points_line_leaf=NAN, frac_points_nonplane_leaf=NAN,
               frac_points_sparse_leaf=NAN, n_plane_leaves=0,
               plane_leaf_edge_median_pointweighted=NAN, plane_leaf_pts_median=NAN)
    if n == 0:
        return res
    active = np.arange(n)
    n_sparse = n_nonplane = n_line = 0
    plane_counts, plane_edges = [], []
    for layer in range(int(max_layer) + 1):
        if active.size == 0:
            break
        ne = edge / 2 ** layer
        counts, inv, eig = _node_stats(xyz[active], ne, origin, min_init)
        fit = counts >= min_init
        n_sparse += int(counts[~fit].sum())
        plane = fit & (eig[:, 0] < plane_threshold)
        line = plane & (eig[:, 1] < plane_threshold)
        n_line += int(counts[line].sum())
        plane_counts.append(counts[plane])
        plane_edges.append(np.full(int(plane.sum()), ne))
        nonplane = fit & ~plane
        if layer < max_layer:
            active = active[nonplane[inv]]
        else:
            n_nonplane += int(counts[nonplane].sum())
    pc = np.concatenate(plane_counts) if plane_counts else np.zeros(0, dtype=np.int64)
    pe = np.concatenate(plane_edges) if plane_edges else np.zeros(0)
    n_plane = int(pc.sum())
    res.update(frac_points_plane_leaf=n_plane / n, frac_points_line_leaf=n_line / n,
               frac_points_nonplane_leaf=n_nonplane / n, frac_points_sparse_leaf=n_sparse / n,
               n_plane_leaves=int(len(pc)))
    if len(pc):
        res["plane_leaf_edge_median_pointweighted"] = float(np.median(np.repeat(pe, pc)))
        res["plane_leaf_pts_median"] = float(np.median(pc))
    return res


# --------------------------------------------------------------------------- estimates
def _d8_at(spacing_df, rng):
    """d8_median of the range bin containing rng (last bin closed on the right)."""
    if not np.isfinite(rng) or len(spacing_df) == 0:
        return NAN
    lo, hi = spacing_df["range_lo"].values, spacing_df["range_hi"].values
    m = (rng >= lo) & (rng < hi)
    if not m.any() and rng == hi[-1]:
        m = np.arange(len(lo)) == len(lo) - 1
    return float(spacing_df["d8_median"].values[m][0]) if m.any() else NAN


def _row(estimator, param, edge, layer, frac, note):
    return dict(estimator=estimator, param=param, value_edge_m=edge, value_max_layer=layer,
                plane_leaf_fraction=frac, note=note)


def estimates(spacing_df, xyz, ring, edges, max_layers, min_init, plane_threshold, range_err,
              existing=(), d8_point=None, octree_cache=None, origin=(0, 0, 0)):
    """Candidate estimates A-F as a list of row dicts (estimator,param,value_edge_m,
    value_max_layer,plane_leaf_fraction,note).  All constants are unvalidated candidates."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    cache = octree_cache if octree_cache is not None else {}

    def octo(e, L):
        key = (float(e), int(L))
        if key not in cache:
            cache[key] = octree_simulation(xyz, e, L, min_init, plane_threshold, origin)
        return cache[key]

    r = np.linalg.norm(xyz, axis=1)
    rows = []
    # A: neighbour spacing
    if d8_point is None:
        d8_point = knn_features(xyz, 8)["d"][:, 8]
    d8p = d8_point[np.isfinite(d8_point)]
    if len(r):
        q = {"p25_range": np.percentile(r, 25), "median_range": np.percentile(r, 50),
             "p75_range": np.percentile(r, 75)}
    else:
        q = {"p25_range": NAN, "median_range": NAN, "p75_range": NAN}
    for c in (1.5, 2.0, 3.0):
        for name in ("median_range", "p25_range", "p75_range"):
            rows.append(_row("neighbour_spacing", "c=%g;at=%s" % (c, name), c * _d8_at(spacing_df, q[name]),
                             NAN, NAN, _UNVAL + "edge=c*d8 of the range bin containing the %s" % name))
        rows.append(_row("neighbour_spacing", "c=%g;at=all_points_median" % c,
                         c * float(np.median(d8p)) if len(d8p) else NAN, NAN, NAN,
                         _UNVAL + "point-weighted median of c*d8 over all points"))
    # B: points per voxel
    grid = []
    e = 0.1
    while e <= 3.0 + 1e-9:
        grid.append(e)
        e *= 1.05
    val_b = NAN
    for e in grid:
        if len(xyz) == 0:
            break
        counts = _group_nodes(np.floor((xyz - np.asarray(origin, dtype=np.float64)) / e).astype(np.int64))[1]
        if np.median(counts) >= 4 * min_init:
            val_b = float(e)
            break
    rows.append(_row("points_per_voxel", "min_median_pts=%d" % (4 * min_init), val_b, NAN, NAN,
                     _UNVAL + "smallest edge on 0.1-3.0 m grid (ratio 1.05) with median points per occupied layer-0 node >= 4*min_init"))
    # C: plane-leaf fraction
    cand = sorted({(float(e), int(L)) for e in edges for L in max_layers})
    fr = {k: octo(*k)["frac_points_plane_leaf"] for k in cand}
    if cand and any(np.isfinite(v) for v in fr.values()):
        best = min(cand, key=lambda k: (-round(fr[k], 12), k[0], k[1]))
        fmax = fr[best]
        rows.append(_row("plane_leaf_fraction", "argmax", best[0], float(best[1]), fmax,
                         _UNVAL + "argmax of frac_points_plane_leaf over given edges x max_layers; ties to smaller edge then layer"))
        within = [k[0] for k in cand if k[1] == best[1] and fr[k] >= fmax - 0.01]
        rows.append(_row("plane_leaf_fraction", "range_within_1pct_of_max", NAN, float(best[1]), fmax,
                         _UNVAL + "edges=%s within 0.01 absolute of the max at max_layer=%d"
                         % (",".join("%g" % v for v in within), best[1])))
    else:
        rows.append(_row("plane_leaf_fraction", "argmax", NAN, NAN, NAN, _UNVAL + "no scan data"))
        rows.append(_row("plane_leaf_fraction", "range_within_1pct_of_max", NAN, NAN, NAN, _UNVAL + "no scan data"))
    # D: noise lower bound
    f = 10.0
    rows.append(_row("noise_lower_bound", "f=%g" % f, math.sqrt(12.0 * f) * range_err, NAN, NAN,
                     _UNVAL + "sqrt(12*f)*range_err: plane patch in-plane variance L^2/12 >= f*range_err^2, f=10"))
    # E: layers from spacing
    e_root = 2.0 * _d8_at(spacing_df, np.percentile(r, 75) if len(r) else NAN)
    e_leaf = 2.0 * _d8_at(spacing_df, np.percentile(r, 10) if len(r) else NAN)
    if np.isfinite(e_root) and np.isfinite(e_leaf) and e_leaf > 0:
        ml = float(np.clip(round(math.log2(e_root / e_leaf)), 0, 3))
    else:
        ml = NAN
    rows.append(_row("layers_from_spacing", "root=2*d8@p75;leaf=2*d8@p10", e_root, ml, NAN,
                     _UNVAL + "root=2*d8(p75 range), leaf=2*d8(p10 range), max_layer=clip(round(log2(root/leaf)),0,3)"))
    # F: existing configurations
    for edge, layer, label in existing:
        rows.append(_row("existing", "%g:%d" % (edge, layer), float(edge), float(layer),
                         octo(edge, layer)["frac_points_plane_leaf"], str(label)))
    return rows


def run_scan(arrays, cfg):
    """One scan, one pfn -> (spacing_df, edges_df, octree_df, estimates_rows)."""
    xyz = np.column_stack([arrays["x"], arrays["y"], arrays["z"]]).astype(np.float64)
    ring = arrays.get("ring")
    existing = list(cfg.get("existing", []))
    edges = list(cfg["edges"])
    mls = list(cfg["max_layers"])
    min_init, thr = cfg["min_init"], cfg["plane_threshold"]
    knn = knn_features(xyz, 8)
    spacing = spacing_stats(xyz, ring, cfg["range_bins"], _knn=knn)
    all_edges = sorted({float(e) for e in edges} | {float(e[0]) for e in existing})
    top = max(mls + [int(e[1]) for e in existing] + [0])
    erows = []
    for e in all_edges:
        for L in range(top + 1):
            d = voxel_layer_stats(xyz, e, L, min_init, thr)
            erows.append(dict(edge=e, layer=L, **d))
    edges_df = pd.DataFrame(erows, columns=["edge", "layer", "node_edge", "occupied_nodes",
                                            "pts_per_node_median", "pts_per_node_p10", "pts_per_node_p90",
                                            "frac_nodes_fit", "frac_fit_plane", "frac_fit_line",
                                            "frac_points_in_fit_plane"])
    cache = {}
    cfgs = sorted({(float(e), int(L)) for e in edges for L in mls} |
                  {(float(e[0]), int(e[1])) for e in existing})
    orows = []
    for e, L in cfgs:
        cache[(e, L)] = octree_simulation(xyz, e, L, min_init, thr)
        orows.append(dict(edge=e, max_layer=L, **cache[(e, L)]))
    octree_df = pd.DataFrame(orows, columns=["edge", "max_layer", "frac_points_plane_leaf",
                                             "frac_points_line_leaf", "frac_points_nonplane_leaf",
                                             "frac_points_sparse_leaf", "n_plane_leaves",
                                             "plane_leaf_edge_median_pointweighted", "plane_leaf_pts_median"])
    est = estimates(spacing, xyz, ring, edges, mls, min_init, thr, cfg["range_err"], existing=existing,
                    d8_point=knn["d"][:, 8], octree_cache=cache)
    return spacing, edges_df, octree_df, est


# --------------------------------------------------------------------------- bag reading
def read_bag_scans(path, lidar_topic, imu_topic, n_scans):
    """-> (list of (t, msg), accel array (m,3)).  First lidar message at or after each of
    n_scans targets t0 + dur*i/n (target 0 = bag start).  Accel = IMU samples in the first 1.0 s.
    rosbag is imported here only; exercised in tests solely through a stub module."""
    import rosbag  # lazy: not available in the analysis sandbox
    bag = rosbag.Bag(path)
    try:
        t0, t1 = bag.get_start_time(), bag.get_end_time()
        targets = [t0 + (t1 - t0) * i / n_scans for i in range(n_scans)]
        scans, j = [], 0
        for _topic, msg, t in bag.read_messages(topics=[lidar_topic]):
            ts = t.to_sec()
            if j < n_scans and ts >= targets[j]:
                scans.append((ts, msg))
                while j < n_scans and targets[j] <= ts:  # sparse topics: one message may serve several targets
                    j += 1
            if j >= n_scans:
                break
        acc = []
        if imu_topic:
            for _topic, msg, t in bag.read_messages(topics=[imu_topic]):
                if t.to_sec() > t0 + 1.0:
                    break
                la = msg.linear_acceleration
                acc.append((la.x, la.y, la.z))
    finally:
        bag.close()
    return scans, np.asarray(acc, dtype=np.float64).reshape(-1, 3)


# --------------------------------------------------------------------------- aggregation / CLI
def _aggregate(frames, keys, vcols, seq, pfn, cols, note_mode=False):
    """frames: per-scan DataFrames (same row keys).  Returns first-scan rows + median rows."""
    first = frames[0].copy()
    first["scan"], first["n_scans"] = "first", 1
    big = pd.concat(frames, ignore_index=True)
    g = big.groupby(keys, sort=False, dropna=False)
    med = g[vcols].median().reset_index()
    if note_mode:
        notes = []
        for _, sub in g:
            txt = sub["note"].iloc[0]
            lists = [re.search(r"edges=([^ ]*) ", s) for s in sub["note"]]
            if all(m_ is not None for m_ in lists):
                sets = [m_.group(1).split(",") for m_ in lists]
                allv = sorted({v for s in sets for v in s}, key=float)
                keep = [v for v in allv if 2 * sum(v in s for s in sets) >= len(sets)]
                txt = _UNVAL + "edges=%s within 0.01 of the max in >=50%% of scans (argmax layer varies per scan)" % ",".join(keep)
            notes.append(txt)
        med["note"] = notes
    med["scan"], med["n_scans"] = "median_over_scans", len(frames)
    out = pd.concat([first, med], ignore_index=True)
    out["seq"], out["pfn"] = seq, pfn
    return out[cols]


def _parse_existing(s):
    out = []
    for item in [t for t in s.split(",") if t.strip()]:
        e, L, label = (item.split(":", 2) + ["", ""])[:3]
        out.append((float(e), int(L), label.strip()))
    return out


def _floats(s):
    return [float(v) for v in s.split(",") if v.strip()]


def process_bag(args):
    scans, acc = read_bag_scans(args.bag, args.lidar_topic, args.imu_topic, args.n_scans)
    if not scans:
        raise SystemExit("no lidar messages on %s" % args.lidar_topic)
    cfg = dict(range_bins=_floats(args.range_bins), edges=_floats(args.edges),
               max_layers=[int(v) for v in args.max_layers.split(",") if v.strip()],
               min_init=args.min_init, plane_threshold=args.plane_threshold,
               range_err=args.range_err, existing=_parse_existing(args.existing))
    rot = gravity_rotation(acc)
    grav = len(acc) > 0
    os.makedirs(args.out, exist_ok=True)
    per = {p: {"spacing": [], "edges": [], "octree": [], "estimates": []} for p in args.pfn}
    log = []
    for i, (t, msg) in enumerate(scans):
        arr = msg_to_arrays(msg)
        n_raw = int(msg.width) * int(msg.height)
        if grav:
            xyz = rot @ np.vstack([arr["x"], arr["y"], arr["z"]])
            arr["x"], arr["y"], arr["z"] = xyz[0], xyz[1], xyz[2]
        kept = []
        for p in args.pfn:
            a = filter_points(arr, p, args.min_range, args.max_range)
            kept.append(len(a["x"]))
            sp, ed, oc, es = run_scan(a, cfg)
            per[p]["spacing"].append(sp)
            per[p]["edges"].append(ed)
            per[p]["octree"].append(oc)
            per[p]["estimates"].append(pd.DataFrame(es))
        log.append("scan=%d t=%.3f n_raw=%d n_kept=%s gravity_aligned=%s ring_field=%s" % (
            i, t, n_raw, ",".join("pfn%d:%d" % (p, k) for p, k in zip(args.pfn, kept)),
            "yes" if grav else "no", "yes" if "ring" in arr else "no"))
    spec = [("spacing", ["range_lo", "range_hi"], SPACING_COLS, False),
            ("edges", ["edge", "layer"], EDGES_COLS, False),
            ("octree", ["edge", "max_layer"], OCTREE_COLS, False),
            ("estimates", ["estimator", "param"], ESTIMATES_COLS, True)]
    for name, keys, cols, note_mode in spec:
        vcols = [c for c in cols if c not in ("seq", "pfn", "scan", "n_scans", "note") and c not in keys]
        parts = [_aggregate(per[p][name], keys, vcols, args.seq, p, cols, note_mode) for p in args.pfn]
        pd.concat(parts, ignore_index=True).to_csv(
            os.path.join(args.out, "voxel_stats_%s_%s.csv" % (args.seq, name)), index=False)
    with open(os.path.join(args.out, "voxel_stats_log.txt"), "w") as fh:
        fh.write("\n".join(["seq=%s %s" % (args.seq, l) for l in log]) + "\n")


def collect(dirs, out):
    frames = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "voxel_stats_*_estimates.csv"))):
            frames.append(pd.read_csv(f, keep_default_na=True))
    if not frames:
        raise SystemExit("no *_estimates.csv found in %s" % dirs)
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "voxel_estimates_all.csv"), index=False)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bag")
    ap.add_argument("--seq")
    ap.add_argument("--lidar-topic", default="/os1_cloud_node1/points")
    ap.add_argument("--imu-topic", default="/imu/imu", help="empty string disables gravity alignment")
    ap.add_argument("--out", default=".")
    ap.add_argument("--n-scans", type=int, default=20)
    ap.add_argument("--pfn", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--min-range", type=float, default=0.5)
    ap.add_argument("--max-range", type=float, default=100.0)
    ap.add_argument("--min-init", type=int, default=5)
    ap.add_argument("--plane-threshold", type=float, default=2.5e-3)
    ap.add_argument("--range-err", type=float, default=0.02)
    ap.add_argument("--edges", default="0.2,0.3,0.425,0.6,0.8,1.0,1.4,2.0")
    ap.add_argument("--max-layers", default="1,2,3")
    ap.add_argument("--range-bins", default="0.5,2,5,10,20,40,100")
    ap.add_argument("--existing", default="0.425:2:ntu_viral.yaml,1.0:2:F-428 experiment")
    ap.add_argument("--synthetic-selftest", action="store_true")
    ap.add_argument("--collect", nargs="+", metavar="DIR")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.synthetic_selftest:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import selftest_r76_voxel_stats as st
        return st.run_all()
    if args.collect:
        collect(args.collect, args.out)
        return 0
    if not (args.bag and args.seq):
        ap.error("--bag and --seq are required (or use --collect / --synthetic-selftest)")
    process_bag(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
