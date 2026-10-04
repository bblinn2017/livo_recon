#!/usr/bin/env python3
"""Selftest for r76_voxel_stats.  Prints PASS/FAIL per check, exits 1 if any fails.

Provenance of expected values:
  * Geometry-derived (expected value computed from the construction, not from the tool):
    checks 1 (lattice d8 = s*sqrt(2), aniso = 1), 2 (lattice counts; discrete lattice
    variance s^2 (n^2-1)/12; line eigenvalues 0), 3 (construction shares; sphere whole-node
    eigenvalue R^2/3), 4 (ring gap = 33.2/15 deg, along-ring step = 2*pi/cols * r * cos(el),
    estimator D = sqrt(120)*range_err), 5 (round trip), 6 (R f = +z), 7 (index/range sets).
  * Qualitative / structural: check 4 monotonic d8 and p75 > p25, check 8 determinism,
    check 9 file layout (exact column lists, n_scans, scan labels).
  * The real bag reader is exercised ONLY through a stub `rosbag` module injected into
    sys.modules; it has never run on a real bag.
"""
import math
import os
import shutil
import sys
import tempfile
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r76_voxel_stats as T  # noqa: E402

THR = 2.5e-3
FAILS = 0


def ok(name, cond, detail=""):
    global FAILS
    if not cond:
        FAILS += 1
    print("%s  %s%s" % ("PASS" if cond else "FAIL", name, ("  [" + detail + "]") if detail else ""))
    return cond


def lattice(n, s, x0, y0, z):
    g = np.arange(n) * s
    X, Y = np.meshgrid(x0 + g, y0 + g, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, float(z))])


# ------------------------------------------------------------------ synthetic scan builders
def make_room(seed=76, cols=1024, rings=16, sensor=(0.5, -0.3, 0.0), noise=0.02,
              lo=(-12.0, -8.0, -1.5), hi=(12.0, 8.0, 3.0)):
    """Ray-cast a box room from the sensor; index = col*rings + ring (Ouster-like).
    Range noise is along the ray so ring elevations stay exact.  Sensor frame output."""
    rng = np.random.default_rng(seed)
    el_deg = np.linspace(-16.6, 16.6, rings)
    az = 2 * np.pi * np.arange(cols) / cols
    AZ, EL = np.meshgrid(az, np.radians(el_deg), indexing="ij")
    d = np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)], axis=-1).reshape(-1, 3)
    s = np.asarray(sensor)
    t = np.full(len(d), np.inf)
    for ax in range(3):
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = np.where(d[:, ax] > 0, (hi[ax] - s[ax]) / d[:, ax], (lo[ax] - s[ax]) / d[:, ax])
        t = np.minimum(t, np.where(np.isfinite(ta) & (ta > 0), ta, np.inf))
    t = t + rng.normal(0, noise, len(t))
    xyz = d * t[:, None]
    ring = np.tile(np.arange(rings), cols)
    return {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "ring": ring.astype(np.int64),
            "index": np.arange(len(t), dtype=np.int64)}


def pc2_msg(x, y, z, ring, byteorder="<", pad_nan_rows=()):
    """Hand-built PointCloud2-like stub: x,y,z f32, ring u16 @16, t u32 @20, point_step 32."""
    n = len(x)
    dt = np.dtype({"names": ["x", "y", "z", "ring", "t"],
                   "formats": [byteorder + "f4"] * 3 + [byteorder + "u2", byteorder + "u4"],
                   "offsets": [0, 4, 8, 16, 20], "itemsize": 32})
    a = np.zeros(n, dtype=dt)
    a["x"], a["y"], a["z"], a["ring"], a["t"] = x, y, z, ring, np.arange(n) * 1000
    for i in pad_nan_rows:
        a["y"][i] = np.nan
    F = types.SimpleNamespace
    fields = [F(name="x", offset=0, datatype=7, count=1), F(name="y", offset=4, datatype=7, count=1),
              F(name="z", offset=8, datatype=7, count=1), F(name="ring", offset=16, datatype=4, count=1),
              F(name="t", offset=20, datatype=6, count=1)]
    return F(fields=fields, point_step=32, data=a.tobytes(), width=n // 2, height=2,
             is_bigendian=(byteorder == ">"))


# ------------------------------------------------------------------ checks
def check1():
    s = 0.1
    xyz = lattice(101, s, -5.0, -5.0, 0.5)
    sp = T.spacing_stats(xyz, None, [0.0, 20.0])
    r = sp.iloc[0]
    ok("1 lattice d8 median = s*sqrt(2)", abs(r.d8_median / (s * math.sqrt(2)) - 1) < 0.01,
       "d8=%.5f expect %.5f" % (r.d8_median, s * math.sqrt(2)))
    ok("1 lattice d1 = d4 = s", abs(r.d1_median / s - 1) < 0.01 and abs(r.d4_median / s - 1) < 0.01)
    ok("1 lattice aniso = 1 (3x3 block covariance isotropic)", abs(r.aniso_median - 1) < 0.01,
       "aniso=%.4f" % r.aniso_median)
    ok("1 no ring -> ring columns NaN", np.isnan(r.across_ring_gap_deg) and np.isnan(r.along_ring_step_m))


def check2():
    z = 0.37  # off every node boundary at layers 0..2 (0.25 multiples)
    pts = lattice(10, 0.1, 0.05, 0.05, z)  # 10x10 = 100 points in [0.05,0.95]^2
    d0 = T.voxel_layer_stats(pts, 1.0, 0, 5, THR)
    ok("2 root node holds exactly 100 points", d0["occupied_nodes"] == 1 and d0["pts_per_node_median"] == 100)
    ok("2 root node fit and plane (e0=0 < thr)", d0["frac_nodes_fit"] == 1 and d0["frac_fit_plane"] == 1
       and d0["frac_points_in_fit_plane"] == 1)
    ok("2 e1 NOT < thr for 0.9 m lattice", d0["frac_fit_line"] == 0)
    # discrete lattice variance per axis = s^2 (n^2-1)/12 = 0.0825 (continuous 0.9^2/12 = 0.0675 does not apply)
    lo_ = T.voxel_layer_stats(pts, 1.0, 0, 5, 0.08)["frac_fit_line"]
    hi_ = T.voxel_layer_stats(pts, 1.0, 0, 5, 0.085)["frac_fit_line"]
    ok("2 e1 = s^2(n^2-1)/12 = 0.0825 bracketed by thr 0.08/0.085", lo_ == 0 and hi_ == 1)
    d1 = T.voxel_layer_stats(pts, 1.0, 1, 5, THR)
    ok("2 layer 1: 4 nodes x 25 points, planes, not lines (e1=0.02)",
       d1["occupied_nodes"] == 4 and d1["pts_per_node_median"] == 25 and d1["frac_fit_plane"] == 1
       and d1["frac_fit_line"] == 0 and d1["node_edge"] == 0.5)
    line = np.column_stack([0.05 + 0.05 * np.arange(19), np.full(19, 0.31), np.full(19, z)])
    dl = T.voxel_layer_stats(line, 1.0, 0, 5, THR)
    ok("2 1-D line: e0,e1 < thr so frac_fit_line = 1", dl["frac_fit_line"] == 1 and dl["frac_fit_plane"] == 1)
    d4 = T.voxel_layer_stats(pts[:4], 1.0, 0, 5, THR)
    ok("2 4 points < min_init: not fit", d4["frac_nodes_fit"] == 0 and np.isnan(d4["frac_fit_plane"]))


def sphere(n, R, c):
    i = np.arange(n) + 0.5
    zz = 1 - 2 * i / n
    ph = i * math.pi * (3 - math.sqrt(5))
    rr = np.sqrt(1 - zz ** 2)
    return np.column_stack([rr * np.cos(ph), rr * np.sin(ph), zz]) * R + np.asarray(c)


def check3():
    plane = lattice(10, 0.1, 0.05, 0.05, 0.37)
    far = np.array([[5.3, 5.3, 5.3], [5.6, 5.3, 5.3], [5.3, 5.6, 5.3]])  # 3 isolated pts in one node
    xyz = np.vstack([plane, far])
    o = T.octree_simulation(xyz, 1.0, 2, 5, THR)
    ok("3 plane share = 100/103", abs(o["frac_points_plane_leaf"] - 100 / 103) < 1e-12)
    ok("3 sparse share = 3/103", abs(o["frac_points_sparse_leaf"] - 3 / 103) < 1e-12)
    ok("3 nonplane = 0, line = 0 (e1=0.0825)", o["frac_points_nonplane_leaf"] == 0 and o["frac_points_line_leaf"] == 0)
    ok("3 four fractions sum to 1", abs(o["frac_points_plane_leaf"] + o["frac_points_nonplane_leaf"]
                                        + o["frac_points_sparse_leaf"] - 1) < 1e-12)
    ok("3 one plane leaf, edge 1.0, 100 pts", o["n_plane_leaves"] == 1
       and o["plane_leaf_edge_median_pointweighted"] == 1.0 and o["plane_leaf_pts_median"] == 100)
    # sphere R=0.3: whole-sphere covariance eigenvalues all R^2/3 = 0.03 >> thr; small patches are flat
    sp = sphere(6000, 0.3, (0.511, 0.517, 0.523))
    big = T.octree_simulation(sp, 1.0, 0, 5, THR)
    ok("3 sphere in one 1.0 m node is non-planar (e0=R^2/3=0.03)",
       big["frac_points_nonplane_leaf"] == 1.0 and big["frac_points_plane_leaf"] == 0.0)
    small = T.octree_simulation(sp, 1.0, 3, 5, THR)
    ok("3 same sphere, max_layer 3 (0.125 m leaves): >= 90% of points in plane leaves",
       small["frac_points_plane_leaf"] >= 0.9, "plane=%.3f sparse=%.3f nonplane=%.3f" % (
           small["frac_points_plane_leaf"], small["frac_points_sparse_leaf"], small["frac_points_nonplane_leaf"]))
    ok("3 sphere fractions sum to 1", abs(small["frac_points_plane_leaf"] + small["frac_points_nonplane_leaf"]
                                          + small["frac_points_sparse_leaf"] - 1) < 1e-12)


ROOM_BINS = [0.5, 6, 9, 12, 16, 40]
CFG = dict(range_bins=ROOM_BINS, edges=[0.3, 0.425, 0.6, 1.0, 2.0], max_layers=[1, 2, 3], min_init=5,
           plane_threshold=THR, range_err=0.02, existing=[(0.425, 2, "ntu_viral.yaml"), (1.0, 2, "F-428 experiment")])


def check4():
    cols = 1024
    a = make_room(cols=cols)
    sp, ed, oc, est = T.run_scan(a, CFG)
    gap_expect = 33.2 / 15
    g = sp.across_ring_gap_deg.iloc[0]
    ok("4 across_ring_gap_deg ~ 2.2 (5%)", abs(g / 2.2 - 1) < 0.05 and abs(g / gap_expect - 1) < 0.01,
       "gap=%.4f expect %.4f" % (g, gap_expect))
    ok("4 gap identical on every row", sp.across_ring_gap_deg.nunique() == 1)
    big = sp[sp.n_points >= 20]
    d8 = big.d8_median.values
    ok("4 d8 non-decreasing over bins with >=20 points", len(d8) >= 3 and np.all(np.diff(d8) >= 0),
       "d8=%s n=%s" % (np.round(d8, 3).tolist(), big.n_points.tolist()))
    xyz = np.column_stack([a["x"], a["y"], a["z"]])
    r = np.linalg.norm(xyz, axis=1)
    cosel = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2) / r
    step_true = 2 * np.pi / cols * r * cosel  # azimuth columns are exactly 2*pi/cols apart
    good = True
    for _, row in big.iterrows():
        m = (r >= row.range_lo) & (r < row.range_hi)
        good &= abs(row.along_ring_step_m / np.median(step_true[m]) - 1) < 1e-6
        good &= abs(row.across_ring_gap_m / (np.radians(gap_expect) * np.median(r[m])) - 1) < 1e-3
    ok("4 along_ring_step_m (incl. azimuth wrap) and across_ring_gap_m match geometry", bool(good))
    A = {x["param"]: x for x in est if x["estimator"] == "neighbour_spacing"}
    ok("4 estimator A: c=2 at p75 range > at p25 range",
       A["c=2;at=p75_range"]["value_edge_m"] > A["c=2;at=p25_range"]["value_edge_m"],
       "p25=%.3f p75=%.3f" % (A["c=2;at=p25_range"]["value_edge_m"], A["c=2;at=p75_range"]["value_edge_m"]))
    ok("4 estimator A has 12 rows, value_max_layer NaN", len(A) == 12
       and all(np.isnan(v["value_max_layer"]) for v in A.values()))
    D = [x for x in est if x["estimator"] == "noise_lower_bound"][0]
    ok("4 estimator D = sqrt(120)*range_err", abs(D["value_edge_m"] - math.sqrt(120) * 0.02) < 1e-12)
    bad = []
    for x in est:
        e, p = x["estimator"], x["param"]
        edge_nan_ok = p == "range_within_1pct_of_max"
        layer_nan_ok = e in ("neighbour_spacing", "points_per_voxel", "noise_lower_bound", "layers_from_spacing") \
            and e != "layers_from_spacing"
        if not np.isfinite(x["value_edge_m"]) and not edge_nan_ok:
            bad.append((e, p, "edge"))
        if not np.isfinite(x["value_max_layer"]) and not layer_nan_ok:
            bad.append((e, p, "layer"))
        if e in ("plane_leaf_fraction", "existing") and not np.isfinite(x["plane_leaf_fraction"]):
            bad.append((e, p, "frac"))
        if "note" not in x or not x["note"]:
            bad.append((e, p, "note"))
    ok("4 all estimates finite except documented NaN (A/B/D layer, C range edge)", not bad, str(bad))
    C = [x for x in est if x["estimator"] == "plane_leaf_fraction"]
    rng_row = [x for x in C if x["param"] == "range_within_1pct_of_max"][0]
    am = [x for x in C if x["param"] == "argmax"][0]
    ok("4 estimator C: argmax edge listed in its own 1pct range note",
       ("%g" % am["value_edge_m"]) in rng_row["note"].split("edges=")[1].split(" ")[0].split(","))
    F = [x for x in est if x["estimator"] == "existing"]
    ok("4 estimator F: one row per existing config with its note label",
       [x["note"] for x in F] == ["ntu_viral.yaml", "F-428 experiment"] and F[0]["value_edge_m"] == 0.425)
    ok("4 estimator rows say unvalidated", all("unvalidated" in x["note"] for x in est if x["estimator"] != "existing"))
    # B: face-on lattice of spacing 0.1 needs ~ (edge/0.1)^2 >= 20 -> edge about 0.41-0.45 (discrete counts blur it)
    B = [x for x in T.estimates(T.spacing_stats(lattice(101, 0.1, -5, -5, 0.37), None, [0.0, 20.0]),
                                lattice(101, 0.1, -5, -5, 0.37), None, [0.5], [1], 5, THR, 0.02)
         if x["estimator"] == "points_per_voxel"][0]
    ok("4 estimator B on 0.1 m lattice lies in [0.35,0.5]", 0.35 <= B["value_edge_m"] <= 0.5,
       "B=%.3f" % B["value_edge_m"])


def check5():
    rng = np.random.default_rng(5)
    n = 10
    x, y, z = rng.normal(size=(3, n)).astype(np.float32)
    ring = np.arange(n) % 16
    for bo in ("<", ">"):
        m = pc2_msg(x, y, z, ring, byteorder=bo, pad_nan_rows=(2, 7))
        a = T.msg_to_arrays(m)
        keep = np.array([i for i in range(n) if i not in (2, 7)])
        good = (len(a["x"]) == 8 and np.array_equal(a["index"], keep)
                and np.array_equal(a["x"], x[keep].astype(np.float64)) and np.array_equal(a["y"], y[keep].astype(np.float64))
                and np.array_equal(a["z"], z[keep].astype(np.float64)) and np.array_equal(a["ring"], ring[keep])
                and a["x"].dtype == np.float64 and a["ring"].dtype.kind == "i")
        ok("5 msg_to_arrays round trip, NaN rows dropped, padding/uint32 field ignored (%s-endian)"
           % ("little" if bo == "<" else "big"), good)


def check6():
    f = np.array([0.5, 0.2, 9.7])
    R = T.gravity_rotation(np.tile(f, (50, 1)))
    v = R @ (f / np.linalg.norm(f))
    ok("6 tilted accel -> +z", np.allclose(v, [0, 0, 1], atol=1e-12))
    ok("6 R orthonormal, det +1", np.allclose(R @ R.T, np.eye(3), atol=1e-12) and abs(np.linalg.det(R) - 1) < 1e-12)
    ok("6 empty -> identity", np.array_equal(T.gravity_rotation(np.zeros((0, 3))), np.eye(3)))
    Ra = T.gravity_rotation(np.array([[0, 0, -9.8]]))
    ok("6 antiparallel handled", np.allclose(Ra @ [0, 0, -1], [0, 0, 1], atol=1e-12) and abs(np.linalg.det(Ra) - 1) < 1e-12)
    Rn = T.gravity_rotation(np.array([[9.8, 0, 0], [9.8, 0, 0]]))
    ok("6 sideways accel -> +z", np.allclose(Rn @ [1, 0, 0], [0, 0, 1], atol=1e-12))


def check7():
    n = 1000
    x = np.arange(n) * 0.1
    arr = {"x": x, "y": np.zeros(n), "z": np.zeros(n), "index": np.arange(n)}
    lo, hi = 5.05, 49.95
    out = T.filter_points(arr, 3, lo, hi)
    expect = [i for i in range(n) if i % 3 == 0 and lo <= i * 0.1 <= hi]  # i in 51..499, divisible by 3
    ok("7 pfn=3 keeps exactly original indices divisible by 3 inside the band",
       out["index"].tolist() == expect and len(expect) > 100)
    out1 = T.filter_points(arr, 1, lo, hi)
    ok("7 pfn=1 keeps all in band, none outside",
       out1["index"].tolist() == [i for i in range(n) if lo <= i * 0.1 <= hi])
    # stride acts on ORIGINAL index even when earlier points are missing
    arr2 = {k: v[1:] for k, v in arr.items()}
    ok("7 stride uses original index, not position", all(i % 3 == 0 for i in T.filter_points(arr2, 3, 0, 1e3)["index"]))


def check8():
    a = make_room(cols=256)
    r1 = T.run_scan(a, CFG)
    r2 = T.run_scan(a, CFG)
    same = (r1[0].equals(r2[0]) and r1[1].equals(r2[1]) and r1[2].equals(r2[2])
            and pd.DataFrame(r1[3]).equals(pd.DataFrame(r2[3])))
    ok("8 run_scan deterministic (identical tables)", same)


class _Time:
    def __init__(self, t):
        self.t = t

    def to_sec(self):
        return self.t


def install_stub_rosbag(t0=1000.0, dur=30.0):
    """Stub rosbag serving a synthetic 30 s bag: lidar 10 Hz, IMU 100 Hz accel (0.5,0.2,9.7)."""
    lidar = []
    for k in range(int(dur * 10)):
        a = make_room(seed=100 + k, cols=256, sensor=(0.5 + 0.02 * k, -0.3, 0.0))
        lidar.append((t0 + 0.1 * k, pc2_msg(a["x"], a["y"], a["z"], a["ring"])))
    ns = types.SimpleNamespace
    imu = [(t0 + 0.01 * k, ns(linear_acceleration=ns(x=0.5, y=0.2, z=9.7))) for k in range(int(dur * 100))]

    class Bag:
        def __init__(self, path):
            self.path = path

        def get_start_time(self):
            return t0

        def get_end_time(self):
            return t0 + dur

        def read_messages(self, topics=None):
            src = {"/os1_cloud_node1/points": lidar, "/imu/imu": imu}[topics[0]]
            for t, m in src:
                yield topics[0], m, _Time(t)

        def close(self):
            pass

    mod = types.ModuleType("rosbag")
    mod.Bag = Bag
    sys.modules["rosbag"] = mod


def check9():
    ok("9 rosbag not imported by the tool at import time (lazy)", "rosbag" not in sys.modules)
    tmp = tempfile.mkdtemp(prefix="r76st_")
    try:
        install_stub_rosbag()
        scans, acc = T.read_bag_scans("fake.bag", "/os1_cloud_node1/points", "/imu/imu", 20)
        ts = [t for t, _ in scans]
        ok("9 read_bag_scans (stub): 20 scans, first at bag start, strictly increasing, spacing ~1.5 s",
           len(scans) == 20 and ts[0] == 1000.0 and all(np.diff(ts) > 0) and abs(np.median(np.diff(ts)) - 1.5) < 0.11)
        ok("9 read_bag_scans (stub): IMU window = first 1.0 s, mean accel (0.5,0.2,9.7)",
           100 <= len(acc) <= 102 and np.allclose(acc.mean(axis=0), [0.5, 0.2, 9.7]))
        dirs = []
        for seq in ("SEQ_A", "SEQ_B"):
            out = os.path.join(tmp, seq)
            dirs.append(out)
            rc = T.main(["--bag", "fake.bag", "--seq", seq, "--out", out, "--n-scans", "20", "--pfn", "1", "3"])
            ok("9 CLI %s returns 0" % seq, rc == 0)
        out = dirs[0]
        want = {"spacing": T.SPACING_COLS, "edges": T.EDGES_COLS, "octree": T.OCTREE_COLS,
                "estimates": T.ESTIMATES_COLS}
        for name, cols in want.items():
            f = os.path.join(out, "voxel_stats_SEQ_A_%s.csv" % name)
            if not ok("9 file written: %s" % os.path.basename(f), os.path.exists(f)):
                continue
            df = pd.read_csv(f)
            first = df[df.scan == "first"]
            med = df[df.scan == "median_over_scans"]
            ok("9 %s: exact columns in order, seq/pfn/scan first" % name,
               list(df.columns) == cols and cols[:3] == ["seq", "pfn", "scan"])
            ok("9 %s: scan labels, both pfn, n_scans 1 (first) / 20 (median)" % name,
               set(df.scan) == {"first", "median_over_scans"} and set(df.pfn) == {1, 3}
               and (first.n_scans == 1).all() and (med.n_scans == 20).all() and (df.seq == "SEQ_A").all()
               and len(first) == len(med))
        lines = open(os.path.join(out, "voxel_stats_log.txt")).read().strip().split("\n")
        ok("9 voxel_stats_log.txt: 20 lines with gravity=yes, ring=yes, both pfn counts",
           len(lines) == 20 and all("gravity_aligned=yes" in l and "ring_field=yes" in l
                                    and "pfn1:" in l and "pfn3:" in l for l in lines), lines[0])
        sp = pd.read_csv(os.path.join(out, "voxel_stats_SEQ_A_spacing.csv"))
        ok("9 spacing medians finite where bins populated", np.isfinite(
            sp[(sp.scan == "median_over_scans") & (sp.n_points >= 20)].d8_median).all())
        coll = os.path.join(tmp, "collected")
        rc = T.main(["--collect"] + dirs + ["--out", coll])
        allf = os.path.join(coll, "voxel_estimates_all.csv")
        ok("9 --collect writes voxel_estimates_all.csv", rc == 0 and os.path.exists(allf))
        if os.path.exists(allf):
            ca = pd.read_csv(allf)
            ea = pd.read_csv(os.path.join(dirs[0], "voxel_stats_SEQ_A_estimates.csv"))
            ok("9 collected: seq is the first column, both seqs present, row count adds up",
               ca.columns[0] == "seq" and list(ca.columns) == T.ESTIMATES_COLS and set(ca.seq) == {"SEQ_A", "SEQ_B"}
               and len(ca) == 2 * len(ea))
    finally:
        sys.modules.pop("rosbag", None)
        shutil.rmtree(tmp, ignore_errors=True)


def run_all():
    global FAILS
    FAILS = 0
    for fn in (check1, check2, check3, check4, check5, check6, check7, check8, check9):
        try:
            fn()
        except Exception as e:  # a crash is a failure; keep going
            import traceback
            traceback.print_exc()
            ok("%s raised %s" % (fn.__name__, type(e).__name__), False, str(e))
    print("\n%s (%d failed)" % ("ALL PASS" if FAILS == 0 else "FAILURES", FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(run_all())
