#!/usr/bin/env python3
"""R74 shared helpers (F-422, F-423). Python + numpy + pandas only.

  F-422  t_rel_s = scan time minus the bag start time, written with 6 decimals; t_abs (when kept) with 6 decimals; every reducer keys on scan_id (frame), never on a time match.
         bag_start_abs is the `bag_start_abs` line of state_reference.txt (17 significant digits, written by the R74 C++).
  F-423  the rotation difference between two orientations is the ANGLE of R_a^T R_b, bounded by pi, never the norm of a difference of rotation vectors (which is not
         periodic: two rotation vectors that differ by 2 pi describe the same orientation).
Raw files keep the 17-digit t_abs the C++ writes (my reading of the F-422 decision: "t_abs kept only where already required, with 6 decimals" applies to the compact Python outputs;
the raw C++ files are the archive and are not reformatted).
"""
import math
import numpy as np
import pandas as pd


def read_bag_start(state_reference_path):
    """bag_start_abs from state_reference.txt, or None when the line is absent (an R73 or older run)."""
    with open(state_reference_path) as f:
        for line in f:
            p = line.split()
            if len(p) == 2 and p[0] == "bag_start_abs":
                return float(p[1])
    return None


def t_rel(t_abs, bag_start):
    """t_rel_s as float64 numpy array: t_abs - bag_start (exact float64 subtraction of two 17-digit values)."""
    return np.asarray(t_abs, dtype=float) - float(bag_start)


def add_t_rel(df, bag_start, src="t_abs", dst="t_rel_s", after=None):
    """Adds t_rel_s to df (inserted right after `after` or after `src`). A frame that already carries t_rel_s (written by the R74 C++) is left alone."""
    if dst in df.columns:
        return df
    if src not in df.columns:
        raise KeyError("no %s column" % src)
    if bag_start is None:
        raise ValueError("bag_start_abs unknown (no bag_start_abs line in state_reference.txt): t_rel_s cannot be derived")
    pos = list(df.columns).index(after or src) + 1
    df = df.copy()
    df.insert(pos, dst, t_rel(df[src].to_numpy(float), bag_start))
    return df


def fmt_time_columns(df, cols=("t_abs", "t_rel_s")):
    """Returns a copy whose time columns are strings with exactly 6 decimals (for to_csv: float_format would round them to 9 significant digits, which is the F-422 defect)."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = ["%.6f" % v if np.isfinite(v) else "nan" for v in out[c].to_numpy(float)]
    return out


def to_csv_t6(df, path, float_format="%.9g", compression=None, time_cols=("t_abs", "t_rel_s")):
    """to_csv with the time columns at 6 decimals and every other float at float_format."""
    fmt_time_columns(df, time_cols).to_csv(path, index=False, float_format=float_format, compression=compression)


def rodrigues(rv):
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3) + _skew(rv)
    k = rv / th
    K = _skew(k)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rot_angle(Ra, Rb):
    """Angle in [0, pi] of R_a^T R_b (F-423). arccos of the clipped half-trace is poor near 0 and pi, so use atan2(|vee|, trace-1)."""
    M = np.asarray(Ra, float).T @ np.asarray(Rb, float)
    v = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    return math.atan2(0.5 * np.linalg.norm(v), 0.5 * (np.trace(M) - 1.0))


def rotvec_angle(rv_a, rv_b):
    """F-423 from two rotation vectors (as the shadow files carry them): the angle of R(rv_a)^T R(rv_b), in rad, bounded by pi."""
    return rot_angle(rodrigues(rv_a), rodrigues(rv_b))
