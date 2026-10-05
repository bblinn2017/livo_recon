#!/usr/bin/env python3
"""r80_calib_window_scan (R80 D6, follow-up F-450; also carries F-448): octree statistics of the
CALIBRATION-WINDOW point cloud only, one row per (bag, root edge).  Host tool; no odometry, no bag run.

    python3 r80_calib_window_scan.py --bag-dir <dir> [<dir> ...] --out <dir> --bag-params <bag_params.json>
    python3 r80_calib_window_scan.py --selftest

Approved design text (D6): "It builds the octree on the accumulated calibration-window scans for each of
the 17 edges (0.20 to 1.00 m step 0.05) and reports, per edge: the share of points in plane leaves (at
least 5 points and eig0 under 2.5e-3 m^2); the median sqrt(eig0) divided by the edge; the share of
degenerate planes (eig1 below the noise variance)."  Plus an extra max-layer census.

THIS TOOL REPORTS NUMBERS ONLY.  It never selects an edge or a max_layer: any rule that turns the table
into one edge has numeric thresholds that are the user's to decide (rule 1.7).  No column of any output
ranks, scores, recommends or picks an edge.

Reused from tools/r69/r76_voxel_stats.py (imported, not modified): msg_to_arrays (PointCloud2 parsing),
gravity_rotation (first-1 s accelerometer mean -> +z), filter_points (index stride on the ORIGINAL
ordering, then range band), _node_stats (per-node count and population-covariance (ddof=0) eigenvalues,
ascending), and the octree convention of octree_simulation (a node with >= min_init points is plane-fit;
plane iff eig0 < plane_threshold; non-plane fit nodes split into 8 children down to max_layer; node edge at
layer L is E/2**L; root nodes anchored at the origin (0,0,0)).

Definitions used in the outputs (see also the assumptions in the module-level constants):
  leaf        a node that is not split further: a plane node (any layer), a sparse node (< min_init
              points), or a fit non-plane node at layer == max_layer.
  plane leaf  fit node with eig0 < plane_threshold.
  edges.csv   built with --max-layer (default 2 = voxel_map/max_layer in config/ntu_viral.yaml).
              share_points_in_plane_leaves = points in plane leaves / n_points (all window points).
              median_sqrt_eig0_over_edge     = median over plane leaves of sqrt(max(eig0,0)) / E, where E is
                                               the ROOT edge of the row (not the leaf's own edge).
              share_degenerate_planes        = plane leaves with eig1 < noise_var / plane leaves.
              Empty plane-leaf set -> the two ratio columns are empty (NaN), never 0.
  layers.csv  for each edge and each max_layer in --census-layers (default 1 2 3 4): the leaves and the points
              that END at each layer 0..max_layer (each point is in exactly one leaf).  The `edge` column is
              an addition to the column list of the request (needed to make rows unambiguous).

Failure policy: an unreadable/short window (fewer lidar messages than the window needs, no usable points,
fewer points than min_init after filtering, missing bag_params entry, missing bag) raises SystemExit with
the bag named; it is never replaced by an empty row.  With several bags a failing bag does not stop the
others, but nothing is written for it and the exit code is the number of failed bags.
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r76_voxel_stats as T  # noqa: E402  (reused, not modified)

BAGS = ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]
EDGES = [round(0.20 + 0.05 * i, 2) for i in range(17)]          # 0.20 .. 1.00 step 0.05 (design D6)

EDGES_COLS = ["bag", "edge", "n_points", "n_leaves", "n_plane_leaves", "share_points_in_plane_leaves",
              "median_sqrt_eig0_over_edge", "share_degenerate_planes"]
LAYERS_COLS = ["bag", "edge", "max_layer", "layer", "n_leaves", "n_points"]

# Defaults and where each comes from.
DEFAULT_MIN_INIT = 5                  # design D6 text; voxel_map/points/min_init: 5 (config/ntu_viral.yaml)
DEFAULT_PLANE_THRESHOLD = 2.5e-3      # design D6 text; voxel_map/plane/plane_threshold (config/ntu_viral.yaml)
DEFAULT_MAX_LAYER = 2                 # voxel_map/max_layer: 2 (config/ntu_viral.yaml)
# Noise variance = (imu/sensor/range_err)^2.  config/ntu_viral.yaml `imu: sensor: range_err: 0.02`;
# src/map/voxelmap.cpp:246 `opts_->weight_sigma_r2 = range_err * range_err` (the gap-floor comment at
# voxelmap.cpp:427 quotes 4e-4 "with range_err 0.02").  0.02**2 = 4e-4.
DEFAULT_NOISE_VAR = 0.02 ** 2
DEFAULT_CALIB_SAMPLES = 100           # calib/stationary/num_samples: 100 (config/ntu_viral.yaml; compiled default 200)
DEFAULT_PFN = 3                       # cbk/lidar/point_filter_num: 3 (config/ntu_viral.yaml)
DEFAULT_MIN_RANGE = 1.0               # cbk/lidar/blind: 1.0 (config/ntu_viral.yaml)
DEFAULT_MAX_RANGE = 100.0             # ASSUMPTION: same as r76_voxel_stats.py default; no project value found
DEFAULT_DS_LEAF = 0.1                 # imu/ds/ds_leaf_size 0.1, imu/ds/mode "first" (config/ntu_viral.yaml; calib_processing.cpp:557-565)


# --------------------------------------------------------------------------- octree
def build_leaves(xyz, edge, max_layer, min_init, plane_threshold, origin=(0.0, 0.0, 0.0)):
    """Octree on xyz -> dict of per-leaf arrays: layer, count, eig0, eig1 (NaN if unfit), plane (bool)."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    out = {k: [] for k in ("layer", "count", "eig0", "eig1", "plane")}
    active = np.arange(len(xyz))
    for layer in range(int(max_layer) + 1):
        if active.size == 0:
            break
        ne = edge / 2 ** layer
        counts, inv, eig = T._node_stats(xyz[active], ne, origin, min_init)
        fit = counts >= min_init
        with np.errstate(invalid="ignore"):
            plane = fit & (eig[:, 0] < plane_threshold)
        terminal = ~fit | plane | (layer == max_layer)
        out["layer"].append(np.full(int(terminal.sum()), layer, dtype=np.int64))
        out["count"].append(counts[terminal])
        out["eig0"].append(eig[terminal, 0])
        out["eig1"].append(eig[terminal, 1])
        out["plane"].append(plane[terminal])
        active = active[~terminal[inv]]
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}


def edge_row(bag, xyz, edge, max_layer, min_init, plane_threshold, noise_var):
    lv = build_leaves(xyz, edge, max_layer, min_init, plane_threshold)
    n = len(xyz)
    pl = lv["plane"].astype(bool)
    n_plane = int(pl.sum())
    row = dict(bag=bag, edge=edge, n_points=n, n_leaves=int(len(lv["count"])), n_plane_leaves=n_plane,
               share_points_in_plane_leaves=float(lv["count"][pl].sum()) / n,
               median_sqrt_eig0_over_edge=float("nan"), share_degenerate_planes=float("nan"))
    if n_plane:
        row["median_sqrt_eig0_over_edge"] = float(np.median(np.sqrt(np.maximum(lv["eig0"][pl], 0.0)))) / edge
        row["share_degenerate_planes"] = float((lv["eig1"][pl] < noise_var).sum()) / n_plane
    return row


def layer_rows(bag, xyz, edge, max_layer, min_init, plane_threshold):
    lv = build_leaves(xyz, edge, max_layer, min_init, plane_threshold)
    rows = []
    for layer in range(int(max_layer) + 1):
        m = lv["layer"] == layer
        rows.append(dict(bag=bag, edge=edge, max_layer=int(max_layer), layer=layer,
                         n_leaves=int(m.sum()), n_points=int(lv["count"][m].sum())))
    return rows


def scan_window(bag, xyz, edges, max_layer, census_layers, min_init, plane_threshold, noise_var):
    """-> (edges_df, layers_df) for one bag's accumulated window cloud.  Fails loudly on a too-small cloud."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(xyz) == 0:
        raise SystemExit("%s: calibration window holds no points" % bag)
    if len(xyz) < min_init:
        raise SystemExit("%s: calibration window holds %d points, fewer than min_init=%d" % (bag, len(xyz), min_init))
    if not np.isfinite(xyz).all():
        raise SystemExit("%s: calibration window holds non-finite coordinates" % bag)
    er = [edge_row(bag, xyz, e, max_layer, min_init, plane_threshold, noise_var) for e in edges]
    lr = []
    for e in edges:
        for ml in census_layers:
            lr.extend(layer_rows(bag, xyz, e, ml, min_init, plane_threshold))
    return pd.DataFrame(er, columns=EDGES_COLS), pd.DataFrame(lr, columns=LAYERS_COLS)


# --------------------------------------------------------------------------- window / bag reading
def window_scan_count(calib_samples, imu_rate_hz, lidar_rate_hz):
    """Scans spanned by `calib_samples` IMU samples: ceil(samples / imu_rate * lidar_rate).  ASSUMPTION: the
    calibration window is the first this-many lidar messages (calibration consumes scans until the IMU
    sample count is reached; calib_processing.cpp:160-190)."""
    if not (imu_rate_hz > 0 and lidar_rate_hz > 0 and calib_samples > 0):
        raise SystemExit("cannot derive the window length: imu_rate_hz=%r lidar_rate_hz=%r samples=%r"
                         % (imu_rate_hz, lidar_rate_hz, calib_samples))
    return int(math.ceil(calib_samples / imu_rate_hz * lidar_rate_hz - 1e-9))


def read_bag_window(path, lidar_topic, imu_topic, k):
    """First k lidar messages of the bag and the accelerometer samples of the first 1.0 s.
    Fewer than k messages is an error.  `rosbag` is imported lazily (not present in the analysis sandbox)."""
    import rosbag
    bag = rosbag.Bag(path)
    try:
        t0 = bag.get_start_time()
        scans = []
        for _topic, msg, t in bag.read_messages(topics=[lidar_topic]):
            scans.append((t.to_sec(), msg))
            if len(scans) >= k:
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
    if len(scans) < k:
        raise SystemExit("%s: only %d lidar messages on %s, window needs %d" % (path, len(scans), lidar_topic, k))
    return scans, np.asarray(acc, dtype=np.float64).reshape(-1, 3)


def ds_first(xyz, leaf):
    """Keep the first point (original order) of every floor(p/leaf) cell (DsMode FIRST, hashing.h:119-125)."""
    if leaf <= 0 or len(xyz) == 0:
        return xyz
    key = np.floor(xyz / leaf).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return xyz[np.sort(first)]


def window_cloud(scans, acc, pfn, min_range, max_range, ds_leaf):
    """Accumulate the window scans into one cloud.  The platform is stationary in the window, so scans are
    simply concatenated in the sensor frame (no pose).  Gravity-aligned like r76 when IMU samples exist."""
    rot = T.gravity_rotation(acc)
    grav = len(acc) > 0
    parts, log = [], []
    for i, (t, msg) in enumerate(scans):
        arr = T.msg_to_arrays(msg)
        if grav:
            v = rot @ np.vstack([arr["x"], arr["y"], arr["z"]])
            arr["x"], arr["y"], arr["z"] = v[0], v[1], v[2]
        a = T.filter_points(arr, pfn, min_range, max_range)
        xyz = ds_first(np.column_stack([a["x"], a["y"], a["z"]]), ds_leaf)
        parts.append(xyz)
        log.append(dict(scan=i, t=t, n_raw=int(msg.width) * int(msg.height), n_used=len(xyz)))
    xyz = np.vstack(parts) if parts else np.zeros((0, 3))
    return xyz, dict(gravity_aligned=grav, scans=log)


# --------------------------------------------------------------------------- CLI
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def _floats(s):
    return [float(v) for v in s.split(",") if v.strip()]


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bag-dir", nargs="+", help="directories searched (find) for <bag>.bag")
    ap.add_argument("--bags", nargs="+", default=BAGS)
    ap.add_argument("--bag-params", default=None, help="r69_inputs/bag_params.json (imu_rate_hz, lidar_rate_hz per bag); "
                    "required unless --window-scans is given")
    ap.add_argument("--window-scans", type=int, default=0, help="override: window = first N lidar messages for every bag "
                    "(0 = derive from --calib-samples and bag_params rates)")
    ap.add_argument("--calib-samples", type=int, default=DEFAULT_CALIB_SAMPLES, help="calib/stationary/num_samples (ntu_viral.yaml: 100)")
    ap.add_argument("--out", default=".")
    ap.add_argument("--lidar-topic", default="/os1_cloud_node1/points")
    ap.add_argument("--imu-topic", default="/imu/imu", help="empty string disables gravity alignment")
    ap.add_argument("--edges", default=",".join("%.2f" % e for e in EDGES))
    ap.add_argument("--max-layer", type=int, default=DEFAULT_MAX_LAYER, help="max_layer for the edges table (ntu_viral.yaml: 2)")
    ap.add_argument("--census-layers", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--min-init", type=int, default=DEFAULT_MIN_INIT)
    ap.add_argument("--plane-threshold", type=float, default=DEFAULT_PLANE_THRESHOLD)
    ap.add_argument("--noise-var", type=float, default=DEFAULT_NOISE_VAR,
                    help="m^2; (imu/sensor/range_err)^2 = 0.02^2 = 4e-4: config/ntu_viral.yaml imu.sensor.range_err, "
                         "src/map/voxelmap.cpp:246 weight_sigma_r2")
    ap.add_argument("--pfn", type=int, default=DEFAULT_PFN, help="cbk/lidar/point_filter_num (ntu_viral.yaml: 3)")
    ap.add_argument("--min-range", type=float, default=DEFAULT_MIN_RANGE, help="cbk/lidar/blind (ntu_viral.yaml: 1.0)")
    ap.add_argument("--max-range", type=float, default=DEFAULT_MAX_RANGE)
    ap.add_argument("--ds-leaf", type=float, default=DEFAULT_DS_LEAF, help="imu/ds/ds_leaf_size, mode first; 0 disables")
    ap.add_argument("--selftest", action="store_true")
    return ap


def find_bag(dirs, name):
    for d in dirs:
        for root, _dn, files in os.walk(d):
            if name + ".bag" in files:
                return os.path.join(root, name + ".bag")
    return None


def process_bag(args, name, params):
    path = find_bag(args.bag_dir, name)
    if path is None:
        raise SystemExit("%s: bag NOT FOUND under %s" % (name, args.bag_dir))
    if args.window_scans > 0:
        k = args.window_scans
    else:
        if name not in params:
            raise SystemExit("%s: no entry in bag_params" % name)
        k = window_scan_count(args.calib_samples, params[name].get("imu_rate_hz", 0), params[name].get("lidar_rate_hz", 0))
    scans, acc = read_bag_window(path, args.lidar_topic, args.imu_topic, k)
    xyz, info = window_cloud(scans, acc, args.pfn, args.min_range, args.max_range, args.ds_leaf)
    ed, ly = scan_window(name, xyz, _floats(args.edges), args.max_layer, args.census_layers, args.min_init,
                         args.plane_threshold, args.noise_var)
    return ed, ly, dict(path=path, bytes=os.path.getsize(path), window_scans=k, n_points=len(xyz), **info)


def write_outputs(out, edges_df, layers_df, prov):
    os.makedirs(out, exist_ok=True)
    edges_df.to_csv(os.path.join(out, "r80_calib_window_edges.csv"), index=False, float_format="%.10g")
    layers_df.to_csv(os.path.join(out, "r80_calib_window_layers.csv"), index=False)
    with open(os.path.join(out, "r80_calib_window_provenance.json"), "w") as fh:
        fh.write(json.dumps(prov, sort_keys=True, separators=(",", ":")) + "\n")


def make_provenance(args, inputs, failed):
    return dict(tool="r80_calib_window_scan.py", tool_sha256=sha256_file(os.path.abspath(__file__)),
                bag_params=(args.bag_params, sha256_file(args.bag_params) if args.bag_params and os.path.exists(args.bag_params) else None),
                inputs=inputs, failed=failed,
                thresholds=dict(min_init=args.min_init, plane_threshold=args.plane_threshold, noise_var=args.noise_var,
                                edges_max_layer=args.max_layer, census_layers=list(args.census_layers),
                                edges=_floats(args.edges), pfn=args.pfn, min_range=args.min_range, max_range=args.max_range,
                                ds_leaf=args.ds_leaf, calib_samples=args.calib_samples, window_scans_override=args.window_scans),
                selection="none: this tool does not select an edge or a max_layer")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest:
        import selftest_r80_calib_window_scan as st
        return st.run_all()
    if not args.bag_dir:
        raise SystemExit("--bag-dir is required")
    params = {}
    if args.window_scans <= 0:
        if not args.bag_params or not os.path.exists(args.bag_params):
            raise SystemExit("--bag-params <bag_params.json> is required (or --window-scans N); not found: %r" % args.bag_params)
        with open(args.bag_params) as fh:
            params = json.load(fh)
    eds, lys, inputs, failed = [], [], {}, {}
    for name in args.bags:
        try:
            ed, ly, info = process_bag(args, name, params)
        except SystemExit as e:
            failed[name] = str(e)
            print("FAILED %s" % e, file=sys.stderr)
            continue
        eds.append(ed)
        lys.append(ly)
        inputs[name] = info
    if eds:
        write_outputs(args.out, pd.concat(eds, ignore_index=True), pd.concat(lys, ignore_index=True),
                      make_provenance(args, inputs, failed))
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
