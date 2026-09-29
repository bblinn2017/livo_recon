"""Shared helpers for the R61 phase-redo reduction scripts (numpy + pandas only)."""
import json, math, os
import numpy as np
import pandas as pd

AXES = ["Rx", "Ry", "Rz", "px", "py", "pz", "vx", "vy", "vz"]

def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def log_so3(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-9:
        return 0.5 * v
    if abs(math.pi - th) < 1e-6:
        # near pi: use symmetric part
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0))
        k = int(np.argmax(axis))
        axis = A[:, k] / max(axis[k], 1e-12)
        return th * axis / np.linalg.norm(axis)
    return th / (2.0 * math.sin(th)) * v

def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def chi2_ppf(p, k):
    """Wilson-Hilferty approximation (adequate for the coverage bounds used here)."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(p)
    return k * (1 - 2.0 / (9 * k) + z * math.sqrt(2.0 / (9 * k))) ** 3

def read_reference(path):
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            d[parts[0]] = parts[1:]
    q = np.array([float(x) for x in d["q_wxyz"]])
    p = np.array([float(x) for x in d["p"]])
    v = np.array([float(x) for x in d["v"]])
    n = int(d["P0_dim"][0])
    P0 = np.array([float(x) for x in d["P0_row_major"]]).reshape(n, n)
    return quat_to_R(q), p, v, P0

def cov_from_row(row, cols_idx, n):
    """Symmetric n x n covariance from the upper-triangle columns P_i_j (NaN -> 0)."""
    P = np.zeros((n, n))
    for (i, j), c in cols_idx.items():
        if i < n and j < n:
            v = row[c]
            if v == v:
                P[i, j] = v
                P[j, i] = v
    return P

def load_trace(path):
    df = pd.read_csv(path)
    pcols = {}
    for c in df.columns:
        if c.startswith("P_"):
            _, i, j = c.split("_")
            pcols[(int(i), int(j))] = c
    return df, pcols

def umeyama(src, dst):
    """Rigid (no scale) alignment dst ~ R src + t. Returns R, t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(Y.T @ X / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    t = mu_d - R @ mu_s
    return R, t

def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o

def block_bootstrap_ratio(num, den, t, block=10.0, n=200, seed=1):
    """CI of sum(num)/sum(den) with resampling of `block`-second blocks."""
    if len(num) < 5:
        return (float("nan"), float("nan"))
    idx = np.floor((t - t.min()) / block).astype(int)
    blocks = [np.where(idx == b)[0] for b in np.unique(idx)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(blocks), len(blocks))
        sel = np.concatenate([blocks[k] for k in pick])
        d = den[sel].sum()
        out.append(num[sel].sum() / d if d > 0 else np.nan)
    out = np.array(out)
    return (float(np.nanpercentile(out, 5)), float(np.nanpercentile(out, 95)))
