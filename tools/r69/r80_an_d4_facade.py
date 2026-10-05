#!/usr/bin/env python3
"""R80 D4 (F-440, facade points labelled ground). HOST analysis, no arm.

Approved text (planner): 'Host analysis, no arm. The snapshot scatter gives the RMS residual of a voxel's own points about its plane, which is sqrt(eig0). For the elevated ground-class rows report
that value, the point count, the plane's distance to the sensor and eig1. Existing evidence: R74 showed these planes pass about 0.09 m from the sensor with eig0 near 0 and eig1 about 0.0013 m^2
(single-ring degenerate fits). Proposed: close F-440 on that analysis (the labelling is a function of the normal only; changing it is the F-430 line-mode work).'

    python3 r80_an_d4_facade.py --root <dir holding one sub-directory per arm> --elevated-above <z in metres> [--out <dir>] [--iter first|N] [--sensor-offset-body X Y Z]
    python3 r80_an_d4_facade.py --selftest

Class definitions
  ground     |world normal z| >= NZ_GROUND (0.8), the value of tools/r69/r80_window_reduce.py NZ_GROUND; the normal is the dump's hpos_z (r.normal, world frame; logResidualDump writes hpos = r.normal).
  elevated   world point z (dump column wz) > --elevated-above. NO DEFAULT: the script refuses to run without it (the user decides the height; z is in the map's world frame, whose origin is the
             start pose, so the height is relative to that frame, not to the floor).
Join: dump rows of frame N with the snapshot of map_frame N by node_id (map frame 1 = post_calibration_voxelmap.csv; see r80_an_d3_snapshot_join.py). A row whose node is not in the snapshot of its
frame is counted and left out of the table (reported as 'rows without snapshot node').
Row group: one group per (arm, frame, iteration, node_id) among the elevated ground-class rows. Per group:
  rms_resid_m       sqrt(eig0) of the node's snapshot (RMS distance of the voxel's own points from its plane; eig0 is the smallest scatter eigenvalue per point, as the approved text states; a
                    value below 0 by rounding would be reported as 0, the raw eig0 is in its own column)
  point_count       snapshot point_count
  plane_dist_m      |n . s + d| with n, d the snapshot's plane (n.x + d = 0, d = -n.centre: voxelplane.cpp:683 r = n.dot(p) + plane_.d) and s the sensor position of that frame and iteration
  eig1              snapshot eig1 (m^2)
  eig0_eig1         the raw pair (eig0, eig1) as text. NO threshold and no label is applied: 'single-ring degenerate' is decided by the reader from this pair.
  n_rows, wz_min, wz_mean, wz_max of the group, snapshot normal z, layer, is_plane.
Sensor position per frame: residual_dump_state.csv (columns frame, iter, n_resid, px, py, pz, rvx, rvy, rvz, written by logResidualDump right before the rows of that iteration) holds the state's
position and rotation vector at the start of each dumped iteration: s = (px, py, pz) of the same (frame, iter) as the rows. That is the BODY (state) origin; the LiDAR origin differs by the
body-to-LiDAR extrinsic, which this script does NOT read from the dump. If the user supplies --sensor-offset-body X Y Z (the LiDAR position in the body frame, metres), the sensor is
s = p + Exp(rv) * (X, Y, Z) (rotation vector -> matrix by Rodrigues, the same convention as Log() in the dump). Without it the distance is from the body origin and the output says so.
(state_trace.csv is not used: its scan_id can differ by one from the map frame, see r80_window_reduce.py.)
Summary: per arm (and per frame), count of groups and rows and the quantiles 0, 0.1, 0.5, 0.9, 1 (display levels, no thresholds) of the four values over the groups."""
import argparse, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r80_an_d3_snapshot_join as D3
from r80_window_reduce import NZ_GROUND

Q = (0.0, 0.1, 0.5, 0.9, 1.0)
DUMP_COLS = ["frame", "iter", "node_id", "hpos_z", "wz"]
SNAP_COLS = ["node_id", "point_count", "eig0", "eig1", "normal_x", "normal_y", "normal_z", "d", "layer", "is_plane"]
STATE_COLS = ["frame", "iter", "px", "py", "pz", "rvx", "rvy", "rvz"]


def load_state(d):
    p = D3._first(d, ["residual_dump_state_all.csv.gz", "residual_dump_state.csv", "residual_dump_state.csv.gz"])
    if p is None:
        raise SystemExit("residual_dump_state absent in %s (the sensor position per frame comes from it)" % d)
    S = pd.read_csv(p)
    miss = [c for c in STATE_COLS if c not in S.columns]
    if miss:
        raise SystemExit("%s lacks columns %s" % (p, miss))
    return S[STATE_COLS].drop_duplicates(["frame", "iter"], keep="last")


def rotvec_to_matrices(rv):
    """(n,3) rotation vectors -> (n,3,3) matrices (Rodrigues)."""
    th = np.linalg.norm(rv, axis=1)
    k = np.divide(rv, th[:, None], out=np.zeros_like(rv), where=th[:, None] > 0)
    K = np.zeros((len(rv), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 2], k[:, 1], k[:, 2], -k[:, 0], -k[:, 1], k[:, 0]
    I = np.broadcast_to(np.eye(3), K.shape)
    return I + np.sin(th)[:, None, None] * K + (1 - np.cos(th))[:, None, None] * (K @ K)


def sensor_positions(ST, offset):
    P = ST[["px", "py", "pz"]].to_numpy(float)
    if offset is not None:
        P = P + np.einsum("nij,j->ni", rotvec_to_matrices(ST[["rvx", "rvy", "rvz"]].to_numpy(float)), np.asarray(offset, float))
    return P


def groups_for_arm(D, S, ST, height, offset=None, it="first"):
    """-> (groups DataFrame, info dict). D dump, S snapshots (map_frame + SNAP_COLS), ST dump state."""
    if height is None or not np.isfinite(height):
        raise SystemExit("--elevated-above is required (no default): the user decides the height")
    miss = [c for c in DUMP_COLS if c not in D.columns]
    if miss:
        raise SystemExit("dump lacks columns %s" % miss)
    D = D3.pick_iter(D, it)
    ground = D[D["hpos_z"].abs() >= NZ_GROUND]
    el = ground[ground["wz"] > height]
    info = dict(dump_rows=int(len(D)), ground_rows=int(len(ground)), elevated_ground_rows=int(len(el)), rows_without_snapshot_node=0)
    if not len(el):
        return pd.DataFrame(), info
    Sx = S.rename(columns={"map_frame": "frame"}).drop_duplicates(["frame", "node_id"])
    M = el.merge(Sx[["frame"] + SNAP_COLS], on=["frame", "node_id"], how="left", validate="m:1")
    nomatch = M["eig0"].isna()
    info["rows_without_snapshot_node"] = int(nomatch.sum())
    M = M[~nomatch]
    ST = ST.copy(); ST[["sx", "sy", "sz"]] = sensor_positions(ST, offset)
    M = M.merge(ST[["frame", "iter", "sx", "sy", "sz"]], on=["frame", "iter"], how="left", validate="m:1")
    if M[["sx", "sy", "sz"]].isna().any().any():
        raise SystemExit("dump rows of (frame, iter) pairs without a residual_dump_state row")
    M["plane_dist_m"] = (M.normal_x * M.sx + M.normal_y * M.sy + M.normal_z * M.sz + M.d).abs()
    g = M.groupby(["frame", "iter", "node_id"], sort=True)
    G = g.agg(n_rows=("wz", "size"), wz_min=("wz", "min"), wz_mean=("wz", "mean"), wz_max=("wz", "max"), point_count=("point_count", "first"), eig0=("eig0", "first"), eig1=("eig1", "first"),
              normal_z=("normal_z", "first"), layer=("layer", "first"), is_plane=("is_plane", "first"), plane_dist_m=("plane_dist_m", "first")).reset_index()
    G.insert(3, "rms_resid_m", np.sqrt(np.clip(G["eig0"].to_numpy(float), 0.0, None)))
    G["eig0_eig1"] = ["(%.6g, %.6g)" % (a, b) for a, b in zip(G.eig0, G.eig1)]
    G["sensor_origin"] = "lidar (body + offset)" if offset is not None else "body origin (no extrinsic applied)"
    return G, info


def summarize(G):
    rows = []
    for label, sub in [("all frames", G)] + [("frame %d" % f, G[G.frame == f]) for f in sorted(G.frame.unique())]:
        for name in ("rms_resid_m", "point_count", "plane_dist_m", "eig1"):
            v = sub[name].to_numpy(float)
            r = dict(scope=label, quantity=name, groups=int(len(sub)), rows=int(sub.n_rows.sum()))
            for q in Q:
                r["q%g" % q] = float(np.quantile(v, q))
            rows.append(r)
    return pd.DataFrame(rows)


def run(root, height, out=None, it="first", offset=None):
    if height is None:
        raise SystemExit("--elevated-above is required (no default): the user decides the height")
    arms = D3.find_arms(root)
    if not arms:
        raise SystemExit("no arm directories (name containing _n<samples>_pfn<pfn>) under %s" % root)
    done = 0
    for ns, pfn, name, d in arms:
        inv = D3.inventory(d)
        if not inv["dump"] or not (inv["snap_compact"] or inv["snap_raw_frames"]):
            print("arm %s: dump or snapshots absent, skipped (ask the user)" % name); continue
        D, S, ST = D3.load_dump(d), D3.load_snaps(d, need=SNAP_COLS), load_state(d)
        G, info = groups_for_arm(D, S, ST, height, offset, it)
        print("\n=== arm %s (samples %d, pfn %d): elevated = wz > %g, ground = |normal z| >= %g; dump rows %d, ground %d, elevated ground %d, without snapshot node %d" %
              (name, ns, pfn, height, NZ_GROUND, info["dump_rows"], info["ground_rows"], info["elevated_ground_rows"], info["rows_without_snapshot_node"]))
        if offset is None:
            print("    plane distance is from the BODY origin (no --sensor-offset-body given)")
        if not len(G):
            print("    no elevated ground-class rows"); done += 1; continue
        Sm = summarize(G)
        print(G[["frame", "iter", "node_id", "n_rows", "rms_resid_m", "point_count", "plane_dist_m", "eig1", "eig0_eig1"]].to_string(index=False, float_format=lambda x: "%.5g" % x))
        print(Sm.to_string(index=False, float_format=lambda x: "%.5g" % x))
        if out:
            os.makedirs(out, exist_ok=True)
            G.to_csv(os.path.join(out, "d4_groups_%s.csv" % name), index=False); Sm.to_csv(os.path.join(out, "d4_summary_%s.csv" % name), index=False)
        done += 1
    if not done:
        raise SystemExit("no arm could be analysed")


# ---------------------------------------------------------------------------------------- selftest
def selftest():
    import tempfile, subprocess, io, contextlib
    fails = []
    root = tempfile.mkdtemp(); d = os.path.join(root, "x_n200_pfn3_fq"); os.makedirs(d)
    nodes = {   # node_id: (normal, d, eig0, eig1, point_count)
        1: ((0, 0, 1.0), -2.0, 4e-6, 0.0013, 25),                       # elevated horizontal plane at z=2 -> distance to a sensor at z=2.09 is 0.09
        2: ((0.0, 0.6, 0.8), -1.0, 1e-4, 0.002, 40),                    # |nz| = 0.8 exactly: ground class (>=); plane n.x + d = 0
        3: ((1.0, 0, 0), -3.0, 1e-5, 0.01, 60),                         # facade normal: not ground class
        4: ((0, 0, 1.0), 0.0, 1e-5, 0.01, 30),                          # low ground plane (wz = 0): not elevated
        5: ((0, 0, 1.0), -2.5, -1e-12, 0.0005, 12)}                     # eig0 slightly negative by rounding
    snaps = []
    for f in (1, 2):
        for nid, (n, dd, e0, e1, pc) in nodes.items():
            snaps.append(dict(map_frame=f, node_id=nid, point_count=pc + f, eig0=e0, eig1=e1, normal_x=n[0], normal_y=n[1], normal_z=n[2], d=dd, layer=0, is_plane=1, initialized=1, distinct_frames=1, eig2=0.1))
    pd.DataFrame(snaps).to_csv(os.path.join(d, "snapshot_frames_all.csv.gz"), index=False)
    rows = []
    for f in (1, 2):
        for it in (0, 1):
            for nid, wz, nz, k in ((1, 2.0, 1.0, 6), (2, 1.5, 0.8, 4), (3, 2.2, 0.0, 9), (4, 0.0, 1.0, 5), (5, 1.2, 1.0, 3), (99, 3.0, 1.0, 2)):
                rows += [dict(frame=f, iter=it, node_id=nid, hpos_z=nz, wz=wz + 0.01 * j) for j in range(k)]
    pd.DataFrame(rows).to_csv(os.path.join(d, "residual_dump_all.csv.gz"), index=False)
    st = pd.DataFrame([dict(frame=f, iter=it, n_resid=1, px=0.0, py=0.0, pz=2.09 + 0.0 * f, rvx=0.0, rvy=0.0, rvz=np.pi / 2) for f in (1, 2) for it in (0, 1)])
    st.to_csv(os.path.join(d, "residual_dump_state.csv"), index=False)
    D, S, ST = D3.load_dump(d), D3.load_snaps(d, need=SNAP_COLS), load_state(d)
    G, info = groups_for_arm(D, S, ST, 1.0)
    if sorted(G.node_id.unique()) != [1, 2, 5] or len(G) != 6:
        fails.append("groups %s" % G[["frame", "iter", "node_id"]].values.tolist())
    if info["rows_without_snapshot_node"] != 2 * 2 or info["elevated_ground_rows"] != 2 * (6 + 4 + 3 + 2):
        fails.append("info %s" % info)
    # default iteration = first: iter column all 0
    if set(G.iter) != {0}:
        fails.append("iteration choice %s" % set(G.iter))
    g1 = G[(G.frame == 1) & (G.node_id == 1)].iloc[0]
    for k, v in dict(n_rows=6, point_count=26, eig1=0.0013, rms_resid_m=np.sqrt(4e-6), plane_dist_m=0.09, wz_min=2.0, wz_max=2.05).items():
        if abs(float(g1[k]) - v) > 1e-12:
            fails.append("node 1 %s = %r want %r" % (k, g1[k], v))
    if g1["eig0_eig1"] != "(4e-06, 0.0013)":
        fails.append("raw pair text %r" % g1["eig0_eig1"])
    g2 = G[(G.frame == 2) & (G.node_id == 2)].iloc[0]
    want2 = abs(0.6 * 0 + 0.8 * 2.09 - 1.0)
    if abs(g2.plane_dist_m - want2) > 1e-12 or g2.n_rows != 4:
        fails.append("node 2 (|nz| = 0.8 boundary): dist %r want %r rows %r" % (g2.plane_dist_m, want2, g2.n_rows))
    g5 = G[(G.frame == 1) & (G.node_id == 5)].iloc[0]
    if g5.rms_resid_m != 0.0 or g5.eig0 != -1e-12:
        fails.append("negative eig0 handling")
    # sensor offset: rv = 90 deg about z, offset (1,0,0) -> sensor moves to (0,1,2.09); node 2 plane 0.6 y + 0.8 z - 1 -> 0.6 + 0.8*2.09 - 1
    G2, _ = groups_for_arm(D, S, ST, 1.0, offset=(1.0, 0.0, 0.0))
    got = G2[(G2.frame == 1) & (G2.node_id == 2)].iloc[0].plane_dist_m
    if abs(got - abs(0.6 + 0.8 * 2.09 - 1.0)) > 1e-12 or "lidar" not in G2.sensor_origin.iloc[0]:
        fails.append("sensor offset: %r" % got)
    Sm = summarize(G)
    allr = Sm[(Sm.scope == "all frames") & (Sm.quantity == "point_count")].iloc[0]
    if allr.groups != 6 or allr.q0 != 13 or allr.q1 != 42 or Sm.scope.nunique() != 3:
        fails.append("summary %s" % allr.to_dict())
    # height changes the set: above 1.4 only node 2 (wz up to 1.53) and node 1 remain
    G3, _ = groups_for_arm(D, S, ST, 1.4)
    if sorted(G3.node_id.unique()) != [1, 2]:
        fails.append("height 1.4 -> %s" % sorted(G3.node_id.unique()))
    # refusal without the height
    me = os.path.abspath(__file__)
    r = subprocess.run([sys.executable, me, "--root", root], capture_output=True, text=True)
    if r.returncode == 0 or "elevated-above" not in r.stderr:
        fails.append("ran without --elevated-above (rc %s)" % r.returncode)
    for bad in (None, float("nan")):
        try:
            groups_for_arm(D, S, ST, bad); fails.append("groups_for_arm accepted height %r" % bad)
        except SystemExit:
            pass
    # full run end to end writes files
    out = os.path.join(root, "out")
    with contextlib.redirect_stdout(io.StringIO()) as cap:
        run(root, 1.0, out=out)
    if not os.path.exists(os.path.join(out, "d4_groups_x_n200_pfn3_fq.csv")) or "BODY origin" not in cap.getvalue():
        fails.append("run output")
    # missing state file fails
    os.remove(os.path.join(d, "residual_dump_state.csv"))
    try:
        load_state(d); fails.append("missing dump state did not fail")
    except SystemExit:
        pass
    print("SELFTEST r80_an_d4_facade", "FAIL " + "; ".join(fails) if fails else
          "PASS (elevated ground-class selection incl. the |nz| = 0.8 boundary, facade and low-ground rows excluded, rows without a snapshot node counted; sqrt(eig0), point_count, eig1, raw pair, plane distance 0.09 from the "
          "planted plane; sensor offset via Rodrigues; height changes the set; the script refuses to run without --elevated-above; summary quantiles; missing state file fails)")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--out"); ap.add_argument("--iter", default="first"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--elevated-above", type=float, default=None, help="world z in metres; REQUIRED, no default")
    ap.add_argument("--sensor-offset-body", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.elevated_above is None:
        sys.exit("--elevated-above is required (no default): the user decides the height above which a ground-class row counts as elevated")
    if not a.root:
        sys.exit("--root is required")
    run(a.root, a.elevated_above, a.out, a.iter, a.sensor_offset_body)
