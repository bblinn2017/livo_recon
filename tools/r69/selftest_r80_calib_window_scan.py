#!/usr/bin/env python3
"""Selftest for r80_calib_window_scan.  Synthetic clouds only; prints PASS/FAIL per check, exits 1 on any FAIL.

Hand-computed expectations (geometry, not taken from the tool):
  * Two-sheet slab: lattice x,y = 0.05 + 0.1 i (i = 0..9), every (x,y) carried at z = 0.05 +/- a, a = 0.03.
    200 points.  Any voxel holding a slab part has z-sign balanced per (x,y) so cov(z,x)=cov(z,y)=0 and
    eig0 = a^2 = 9e-4 < 2.5e-3 (plane).  Edge 0.5: 4 voxels x 50 points, sqrt(eig0)/E = 0.03/0.5 = 0.06,
    eig1 = var(x) over 5 lattice values (spacing 0.1) = 0.02 > 4e-4 (not degenerate).  Edge 1.0: one voxel,
    200 points, sqrt(eig0)/E = 0.03, eig1 = (100-1)/12 * 0.01 = 0.0825.
  * Ring: 40 evenly spaced points on a horizontal circle of radius R at the centre of the edge-E root
    voxel: eig0 = 0, eig1 = R^2/2 (2e-4 for R=0.02: degenerate; 8e-4 for R=0.04: not, noise var 4e-4).
  * Noise cube: 4000 uniform points in [0,1)^3, edge 1.0, max_layer 1: root var 1/12 and layer-1 var
    0.25^2/3... (edge 0.5 -> 0.5^2/12 = 0.0208) are both > 2.5e-3 so nothing is a plane; 8 layer-1 leaves,
    all 4000 points, no layer-0 leaf.
"""
import os
import shutil
import sys
import tempfile
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r80_calib_window_scan as S  # noqa: E402

NV = 4e-4
THR = 2.5e-3
FAILS = 0


def ok(name, cond, detail=""):
    global FAILS
    if not cond:
        FAILS += 1
    print("%s  %s%s" % ("PASS" if cond else "FAIL", name, ("  [" + detail + "]") if detail else ""))
    return cond


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def slab(a=0.03):
    g = 0.05 + 0.1 * np.arange(10)
    X, Y = np.meshgrid(g, g, indexing="ij")
    x, y = X.ravel(), Y.ravel()
    return np.vstack([np.column_stack([x, y, np.full(100, 0.05 + a)]),
                      np.column_stack([x, y, np.full(100, 0.05 - a)])])


def ring(edge, radius, n=40, shift=(0.0, 0.0, 0.0)):
    c = np.array([edge / 2] * 3) + np.asarray(shift)
    th = 2 * np.pi * np.arange(n) / n
    return np.column_stack([c[0] + radius * np.cos(th), c[1] + radius * np.sin(th), np.full(n, c[2])])


def row(df, edge):
    r = df[np.isclose(df["edge"], edge)]
    assert len(r) == 1
    return r.iloc[0]


def run(xyz, edges, max_layer=2, census=(1, 2, 3, 4), bag="t"):
    return S.scan_window(bag, xyz, edges, max_layer, list(census), 5, THR, NV)


# --------------------------------------------------------------------------- checks
def check1_slab():
    ed, ly = run(slab(), [0.5, 1.0])
    r = row(ed, 0.5)
    ok("1 slab edge 0.5: 200 points, 4 leaves, 4 plane leaves, share 1.0",
       r.n_points == 200 and r.n_leaves == 4 and r.n_plane_leaves == 4 and close(r.share_points_in_plane_leaves, 1.0))
    ok("1 slab edge 0.5: median sqrt(eig0)/E = 0.06, degenerate share 0",
       close(r.median_sqrt_eig0_over_edge, 0.06, 1e-9) and r.share_degenerate_planes == 0.0,
       "%.12g" % r.median_sqrt_eig0_over_edge)
    r = row(ed, 1.0)
    ok("1 slab edge 1.0: 1 leaf, median sqrt(eig0)/E = 0.03, degenerate share 0",
       r.n_leaves == 1 and r.n_plane_leaves == 1 and close(r.median_sqrt_eig0_over_edge, 0.03, 1e-9)
       and r.share_degenerate_planes == 0.0)
    # thicker slab: a = 0.06 -> eig0 = 3.6e-3 > 2.5e-3: not a plane at layer 0; splits (only 2 z-layers remain)
    ed2, _ = run(slab(0.06) + np.array([0, 0, 0.1]), [1.0], max_layer=0)
    r = row(ed2, 1.0)
    ok("1 thick slab (eig0 3.6e-3), max_layer 0: leaf is NOT a plane; share 0; ratio columns empty (NaN, not 0)",
       r.n_leaves == 1 and r.n_plane_leaves == 0 and r.share_points_in_plane_leaves == 0.0
       and np.isnan(r.median_sqrt_eig0_over_edge) and np.isnan(r.share_degenerate_planes))
    # threshold edge: eig0 = 9e-4 with threshold 9e-4 -> not strictly below -> not plane
    ed3, _ = S.scan_window("t", slab(), [1.0], 0, [1], 5, 0.03 ** 2 * 0.999, NV)
    ok("1 strict inequality: eig0 just above threshold is not a plane", row(ed3, 1.0).n_plane_leaves == 0)


def check2_ring_degenerate():
    for e in (0.2, 0.55, 1.0):
        ed, _ = run(ring(e, 0.02), [e])
        r = row(ed, e)
        ok("2 ring R=0.02 at E=%g: 1 leaf, plane, degenerate share 1, sqrt(eig0)/E ~ 0" % e,
           r.n_leaves == 1 and r.n_plane_leaves == 1 and r.share_degenerate_planes == 1.0
           and r.median_sqrt_eig0_over_edge < 1e-6)
        ed, _ = run(ring(e, 0.04), [e])
        ok("2 ring R=0.04 at E=%g: eig1 8e-4 >= 4e-4 -> degenerate share 0" % e,
           row(ed, e).share_degenerate_planes == 0.0)
        # two rings, far apart in different root voxels: share 0.5
        both = np.vstack([ring(e, 0.02), ring(e, 0.04, shift=(3 * e, 0.0, 0.0))])
        ed, _ = run(both, [e])
        r = row(ed, e)
        ok("2 two rings (R 0.02 and 0.04) at E=%g: 2 plane leaves, degenerate share 0.5, plane-point share 1" % e,
           r.n_plane_leaves == 2 and close(r.share_degenerate_planes, 0.5) and close(r.share_points_in_plane_leaves, 1.0))
    # the noise variance is the argument that is compared: R=0.04 becomes degenerate when noise_var = 1e-3
    ed, _ = S.scan_window("t", ring(0.5, 0.04), [0.5], 2, [1], 5, THR, 1e-3)
    ok("2 noise_var is honoured (1e-3 > eig1 8e-4 -> degenerate share 1)", row(ed, 0.5).share_degenerate_planes == 1.0)


def check3_mixed_share():
    e = 1.0
    rng = np.random.default_rng(3)
    sparse = np.array([[7.1, 7.1, 7.1], [7.15, 7.1, 7.1], [7.1, 7.15, 7.1]])  # 3 points < min_init: sparse leaf
    xyz = np.vstack([slab(), sparse])
    ed, _ = run(xyz, [e])
    r = row(ed, e)
    ok("3 slab + 3 isolated points: n_points 203, 2 leaves, plane share 200/203",
       r.n_points == 203 and r.n_leaves == 2 and r.n_plane_leaves == 1 and close(r.share_points_in_plane_leaves, 200 / 203))
    _ = rng


def check4_census():
    cube = np.random.default_rng(4).random((4000, 3))
    ed, ly = run(cube, [1.0], max_layer=1, census=(1,))
    r = row(ed, 1.0)
    ok("4 noise cube, edge 1.0, max_layer 1: no plane leaf, 8 leaves, share 0, ratio NaN",
       r.n_plane_leaves == 0 and r.n_leaves == 8 and r.share_points_in_plane_leaves == 0.0 and np.isnan(r.median_sqrt_eig0_over_edge))
    c = ly[ly.max_layer == 1].set_index("layer")
    ok("4 census: layer 0 holds 0 leaves/0 points; layer 1 holds 8 leaves/4000 points",
       c.loc[0, "n_leaves"] == 0 and c.loc[0, "n_points"] == 0 and c.loc[1, "n_leaves"] == 8 and c.loc[1, "n_points"] == 4000)
    # census over all max layers and edges: every point ends in exactly one leaf
    xyz = np.vstack([slab(), ring(0.5, 0.02, shift=(5, 5, 5)), cube[:500] + 9.0])
    ed, ly = run(xyz, S.EDGES)
    g = ly.groupby(["edge", "max_layer"])["n_points"].sum()
    ok("4 census conserves points for 17 edges x max_layer 1..4", len(g) == 17 * 4 and (g == len(xyz)).all())
    ok("4 census rows: one per (edge, max_layer, layer 0..max_layer)",
       len(ly) == 17 * sum(m + 1 for m in (1, 2, 3, 4)))
    ok("4 edges table: 17 rows, edges 0.20..1.00 step 0.05", len(ed) == 17 and np.allclose(ed["edge"], np.arange(17) * 0.05 + 0.2))
    # slab census at edge 0.5, any max_layer >= 1: all 4 leaves / 200 points at layer 0, nothing deeper
    _, ly = run(slab(), [0.5])
    ok("4 slab census edge 0.5: layer 0 = 4 leaves / 200 points, deeper layers empty, for max_layer 1..4",
       all(ly[(ly.max_layer == m) & (ly.layer == 0)].iloc[0][["n_leaves", "n_points"]].tolist() == [4, 200]
           and (ly[(ly.max_layer == m) & (ly.layer > 0)][["n_leaves", "n_points"]].to_numpy() == 0).all()
           for m in (1, 2, 3, 4)))
    # a ring split down: thin vertical cylinder wall -> non-plane at layer 0, deterministic descent
    th = 2 * np.pi * np.arange(400) / 400
    wall = np.column_stack([0.5 + 0.4 * np.cos(th), 0.5 + 0.4 * np.sin(th), np.tile(np.linspace(0.01, 0.99, 20), 20)])
    _, ly = run(wall, [1.0], census=(1, 2))
    c1 = ly[(ly.max_layer == 1)]
    ok("4 cylinder census: root is not a leaf (layer 0 empty) and max_layer-1 leaves hold all 400 points",
       c1[c1.layer == 0].iloc[0].n_leaves == 0 and c1[c1.layer == 1].iloc[0].n_points == 400)


def check5_determinism():
    rng = np.random.default_rng(5)
    xyz = np.vstack([slab(), rng.random((800, 3)) * 3.0])
    a = run(xyz, S.EDGES)
    b = run(xyz.copy(), S.EDGES)
    ok("5 same input twice: identical tables", a[0].equals(b[0]) and a[1].equals(b[1]))
    p = rng.permutation(len(xyz))
    c = run(xyz[p], S.EDGES)
    cols = [x for x in S.EDGES_COLS if x not in ("median_sqrt_eig0_over_edge",)]
    ok("5 permuted point order: counts/shares identical, medians equal within 1e-9",
       a[0][cols].equals(c[0][cols]) and np.allclose(a[0].median_sqrt_eig0_over_edge, c[0].median_sqrt_eig0_over_edge,
                                                      atol=1e-9, equal_nan=True) and a[1].equals(c[1]))


def check6_fail_loudly():
    def raises(f, text):
        try:
            f()
        except SystemExit as e:
            return text in str(e)
        except Exception:
            return False
        return False
    ok("6 empty cloud -> SystemExit naming the bag", raises(lambda: run(np.zeros((0, 3)), [0.5], bag="eee_01"), "eee_01: calibration window holds no points"))
    ok("6 fewer than min_init points -> SystemExit", raises(lambda: run(slab()[:4], [0.5], bag="nya_02"), "nya_02"))
    bad = slab()
    bad[3, 0] = np.nan
    ok("6 non-finite coordinate -> SystemExit", raises(lambda: run(bad, [0.5]), "non-finite"))
    ok("6 window length cannot be derived (rate 0) -> SystemExit", raises(lambda: S.window_scan_count(100, 0.0, 9.7), "cannot derive"))
    ok("6 window_scan_count(100 samples, 197.5 Hz, 9.71 Hz) = 5", S.window_scan_count(100, 197.5, 9.71) == 5)
    ok("6 window_scan_count exact multiple does not round up (100 samples, 100 Hz, 10 Hz -> 10)", S.window_scan_count(100, 100.0, 10.0) == 10)


def check7_no_selection():
    ed, ly = run(slab(), [0.5, 1.0])
    bad = ("select", "best", "choice", "chosen", "recommend", "optim", "argmax", "argmin", "rank", "score", "pick", "winner")
    names = [c.lower() for c in list(ed.columns) + list(ly.columns)]
    ok("7 edges columns are exactly the requested 8", list(ed.columns) == S.EDGES_COLS and len(S.EDGES_COLS) == 8)
    ok("7 layers columns are the requested 5 plus `edge`", list(ly.columns) == S.LAYERS_COLS
       and [c for c in S.LAYERS_COLS if c != "edge"] == ["bag", "max_layer", "layer", "n_leaves", "n_points"])
    ok("7 no selection-like column in either table", not any(b in n for n in names for b in bad))


# ---- stub rosbag + end-to-end
class _Time:
    def __init__(self, t):
        self.t = t

    def to_sec(self):
        return self.t


def pc2_msg(xyz):
    n = len(xyz)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    raw = np.zeros(n, dtype=dt)
    raw["x"], raw["y"], raw["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    ns = types.SimpleNamespace
    fields = [ns(name=nm, offset=4 * i, datatype=7, count=1) for i, nm in enumerate("xyz")]
    return ns(fields=fields, width=n, height=1, point_step=12, is_bigendian=False, data=raw.tobytes())


def install_stub(n_lidar, t0=1000.0, cloud=None):
    cloud = slab() if cloud is None else cloud
    ns = types.SimpleNamespace
    lidar = [(t0 + 0.1 * k, pc2_msg(cloud + np.array([0, 0, 0.0]))) for k in range(n_lidar)]
    imu = [(t0 + 0.005 * k, ns(linear_acceleration=ns(x=0.0, y=0.0, z=9.81))) for k in range(400)]

    class Bag:
        def __init__(self, path):
            self.path = path

        def get_start_time(self):
            return t0

        def read_messages(self, topics=None):
            src = {"/os1_cloud_node1/points": lidar, "/imu/imu": imu}[topics[0]]
            for t, m in src:
                yield topics[0], m, _Time(t)

        def close(self):
            pass

    mod = types.ModuleType("rosbag")
    mod.Bag = Bag
    sys.modules["rosbag"] = mod


def check8_end_to_end():
    ok("8 rosbag not imported at tool import (lazy)", "rosbag" not in sys.modules)
    tmp = tempfile.mkdtemp(prefix="r80cw_")
    try:
        # slab clouds offset by +1 in all axes so that min_range (1.0) and ds (0.1) pass them untouched:
        # range of every point >= 1.0 ; lattice 0.1 spacing in x,y and +/-0.03 in z lie in distinct 0.1 cells.
        cloud = slab() + np.array([1.0, 1.0, 1.0])
        install_stub(3, cloud=cloud)
        os.makedirs(os.path.join(tmp, "bags"))
        for nm in ("eee_01", "nya_01"):
            open(os.path.join(tmp, "bags", nm + ".bag"), "wb").write(b"x")
        pj = os.path.join(tmp, "bag_params.json")
        import json
        json.dump({"eee_01": dict(imu_rate_hz=200.0, lidar_rate_hz=10.0), "nya_01": dict(imu_rate_hz=200.0, lidar_rate_hz=10.0)},
                  open(pj, "w"))
        out = os.path.join(tmp, "out")
        base = ["--bag-dir", os.path.join(tmp, "bags"), "--bag-params", pj, "--out", out, "--bags", "eee_01", "nya_01",
                "--pfn", "1", "--min-range", "0.0", "--ds-leaf", "0", "--edges", "0.5,1.0", "--calib-samples", "60"]
        # 60 samples / 200 Hz * 10 Hz = 3 scans; gravity (0,0,9.81) -> identity rotation
        rc = S.main(base)
        ed = pd.read_csv(os.path.join(out, "r80_calib_window_edges.csv"))
        ly = pd.read_csv(os.path.join(out, "r80_calib_window_layers.csv"))
        ok("8 end-to-end (stub bag): exit 0, 2 bags x 2 edges rows, 600 accumulated points per bag",
           rc == 0 and len(ed) == 4 and (ed.n_points == 600).all() and list(ed.columns) == S.EDGES_COLS and list(ly.columns) == S.LAYERS_COLS)
        r = ed[(ed.bag == "eee_01") & np.isclose(ed.edge, 0.5)].iloc[0]
        # stacking 3 identical scans triples every point: each voxel 150 points, eig unchanged; hand values as check 1
        ok("8 end-to-end: edge 0.5 -> 4 plane leaves, share 1, median 0.06 (duplicates leave moments unchanged)",
           r.n_plane_leaves == 4 and close(r.share_points_in_plane_leaves, 1.0) and close(r.median_sqrt_eig0_over_edge, 0.06, 1e-6))
        with open(os.path.join(out, "r80_calib_window_provenance.json")) as fh:
            txt = fh.read()
        prov = json.loads(txt)
        ok("8 provenance: exactly one line, tool sha256 matches the tool file, thresholds present, no selection",
           txt.count("\n") == 1 and prov["tool_sha256"] == S.sha256_file(S.__file__)
           and prov["thresholds"]["noise_var"] == 4e-4 and prov["thresholds"]["plane_threshold"] == 2.5e-3
           and prov["thresholds"]["min_init"] == 5 and prov["inputs"]["eee_01"]["window_scans"] == 3
           and prov["selection"].startswith("none"))
        # determinism end-to-end (byte-identical files)
        out2 = os.path.join(tmp, "out2")
        S.main([a if a != out else out2 for a in base])
        same = all(open(os.path.join(out, f), "rb").read() == open(os.path.join(out2, f), "rb").read()
                   for f in ("r80_calib_window_edges.csv", "r80_calib_window_layers.csv"))
        ok("8 end-to-end determinism: the two CSVs are byte-identical on a rerun", same)
        # short window: bag has 3 lidar messages, ask for 5 -> nonzero exit, nothing written for that bag
        out3 = os.path.join(tmp, "out3")
        rc = S.main([a if a != out else out3 for a in base] + ["--window-scans", "5"])
        ok("8 short window (3 messages, 5 needed): exit code 2 (both bags failed), no output files",
           rc == 2 and not os.path.exists(os.path.join(out3, "r80_calib_window_edges.csv")))
        # missing bag is loud, the other bag still processed
        os.remove(os.path.join(tmp, "bags", "nya_01.bag"))
        out4 = os.path.join(tmp, "out4")
        rc = S.main([a if a != out else out4 for a in base])
        ed4 = pd.read_csv(os.path.join(out4, "r80_calib_window_edges.csv"))
        prov4 = json.load(open(os.path.join(out4, "r80_calib_window_provenance.json")))
        ok("8 missing bag: exit 1, remaining bag written, failure recorded in provenance",
           rc == 1 and set(ed4.bag) == {"eee_01"} and "nya_01" in prov4["failed"])
        # zero lidar messages
        open(os.path.join(tmp, "bags", "nya_01.bag"), "wb").write(b"x")
        install_stub(0)
        out5 = os.path.join(tmp, "out5")
        rc = S.main([a if a != out else out5 for a in base])
        ok("8 empty bag (0 lidar messages): fails loudly, no outputs", rc == 2 and not os.path.exists(os.path.join(out5, "r80_calib_window_edges.csv")))
        # all points removed by the range filter -> loud
        install_stub(3, cloud=cloud)
        out6 = os.path.join(tmp, "out6")
        b6 = [("50" if (i > 0 and base[i - 1] == "--min-range") else (out6 if a == out else a)) for i, a in enumerate(base)]
        rc = S.main(b6)
        ok("8 range filter removes every point: fails loudly", rc == 2 and not os.path.exists(os.path.join(out6, "r80_calib_window_edges.csv")))
    finally:
        sys.modules.pop("rosbag", None)
        shutil.rmtree(tmp, ignore_errors=True)


def check9_ds():
    xyz = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [0.15, 0.01, 0.01], [-0.01, 0.0, 0.0]])
    d = S.ds_first(xyz, 0.1)
    ok("9 ds_first keeps the first point per floor(p/leaf) cell, original order",
       d.tolist() == [[0.01, 0.01, 0.01], [0.15, 0.01, 0.01], [-0.01, 0.0, 0.0]])
    ok("9 ds_first with leaf 0 is a no-op", S.ds_first(xyz, 0.0) is xyz)


def run_all():
    global FAILS
    FAILS = 0
    for f in (check1_slab, check2_ring_degenerate, check3_mixed_share, check4_census, check5_determinism,
              check6_fail_loudly, check7_no_selection, check8_end_to_end, check9_ds):
        f()
    print("RESULT: %s (%d failure(s))" % ("PASS" if FAILS == 0 else "FAIL", FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(run_all())
