#!/usr/bin/env python3
"""R74 (F-428; my own approved addition): plane-field consistency check and the frame-1 plane table. Python + numpy + pandas only; no rerun.

    python3 r74_plane_check.py --dir <arm dir> --out <out dir> [--frame 1]

Inputs (all written by the R74 C++ when `voxel_map.map.snapshot_post_calibration` is true and `lio.residual_dump_frames` contains the frame):
  post_calibration_voxelmap.csv   one row per octree node of the map as frame 1 queries it (the plane fields + the R74 test-value columns)
  residual_dump.csv               one row per residual (frame, iter, node_id, normal in hpos_*, world point w*, r, sigma_squared, row_kind, ...)
  residual_dump_state.csv         the state of the dump (p = body position in the world frame, rotation vector)
Joins every residual of (frame, iteration 0) to its plane by node_id and CHECKS (each count is reported; the gate in r74_gates.py requires 0 violations):
  P1  the node_id joins exactly one snapshot row, and that row is a plane that is not retired
  P2  point_count >= min points (5)
  P3  0 <= eig0 <= eig1 <= eig2 (1e-12 absolute slack) and, when plane_fit_mode is pca, eig0 < plane_threshold
  P4  the residual's stored normal equals the plane normal (u0 rows), the tangent_y (u1 rows) or the tangent_x (u2 rows), to 1e-6; |normal| = 1 to 1e-9
  P5  r = normal . w + d  (the signed distance of the world point to the plane) to 1e-6 + 1e-9 |r|  (u0 rows only; only exact at iteration 0 of the frame that queries the snapshot map)
  P6  edge_m = voxel_size / 2^layer (voxel_size = the largest edge_m, layer 0)
  P7  the test-value columns agree with the eigenvalues: planarity_value = eig0, second_eig_below_threshold = (eig1 < plane_threshold), ratio_e1_over_e0 = eig1/eig0,
      ratio_e2_over_e1 = eig2/eig1 (1e-9 relative), rel_gap_threshold = 1 + 3 sqrt(2/(N-1))
Writes
  r74_plane_table.csv.gz  one row per plane matched in the frame: node_id, layer, edge_m, point_count, eig0..2, planarity_value, planarity_pass, second_eig_below_threshold,
                          ratio_e1_over_e0, ratio_e2_over_e1, rel_gap_threshold, center_x..z, normal_x..z, offset_to_sensor_m (= normal . p_state + d, p = the dump's body position, the
                          sensor offset from residual_dump_state), n_resid, weight_sum, mean_height_w_z
  r74_plane_check.json    the counts and the largest deviations above, the number of residuals and planes, voxel_size, plane_threshold
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

MIN_POINTS = 5


def load(d, name):
    p = os.path.join(d, name)
    if os.path.exists(p):
        return pd.read_csv(p)
    if os.path.exists(p + ".gz"):
        return pd.read_csv(p + ".gz")
    return None


def check(snap, dump, state, frame=1, iteration=0, min_points=MIN_POINTS):
    out = {"frame": frame, "iteration": iteration}
    res = dump[(dump["frame"] == frame) & (dump["iter"] == iteration)].copy()
    out["n_residuals"] = int(len(res))
    snap = snap.copy()
    dup = int(snap["node_id"].duplicated().sum())
    out["snapshot_nodes"] = int(len(snap)); out["snapshot_duplicate_node_ids"] = dup
    S = snap.drop_duplicates("node_id").set_index("node_id")
    ids = res["node_id"].to_numpy()
    in_snap = np.isin(ids, S.index.to_numpy())
    out["P1_unjoined"] = int((~in_snap).sum())
    res = res[in_snap]
    P = S.loc[res["node_id"].to_numpy()]
    out["P1_not_plane_or_retired"] = int(((P["is_plane"].to_numpy() != 1) | (P["retired"].to_numpy() != 0)).sum())
    out["P2_too_few_points"] = int((P["point_count"].to_numpy() < min_points).sum())
    e = P[["eig0", "eig1", "eig2"]].to_numpy(float)
    bad_order = (e[:, 0] < -1e-12) | (e[:, 0] > e[:, 1] + 1e-12) | (e[:, 1] > e[:, 2] + 1e-12)
    thr = P["plane_threshold"].to_numpy(float)
    pca = (P["plane_fit_mode"].to_numpy() == "pca")
    bad_thr = pca & ~(e[:, 0] < thr)
    out["P3_eigen_order_violations"] = int(bad_order.sum()); out["P3_planarity_threshold_violations"] = int(bad_thr.sum())
    n = res[["hpos_x", "hpos_y", "hpos_z"]].to_numpy(float)
    kind = res["row_kind"].to_numpy(int) if "row_kind" in res.columns else np.zeros(len(res), int)
    pn = P[["normal_x", "normal_y", "normal_z"]].to_numpy(float)
    ty = P[["tangent_y_x", "tangent_y_y", "tangent_y_z"]].to_numpy(float)   # y_normal_ = u1
    tx = P[["tangent_x_x", "tangent_x_y", "tangent_x_z"]].to_numpy(float)   # x_normal_ = u2
    ref = np.where(np.isin(kind, (2, 4))[:, None], ty, np.where((kind == 5)[:, None], tx, pn))
    dn = np.abs(n - ref).max(axis=1) if len(n) else np.zeros(0)
    out["P4_normal_mismatch"] = int((dn > 1e-6).sum()); out["P4_max_normal_dev"] = float(dn.max()) if len(dn) else 0.0
    out["P4_non_unit_normals"] = int((np.abs(np.linalg.norm(n, axis=1) - 1.0) > 1e-6).sum())   # dump precision is 9 significant digits
    u0 = np.isin(kind, (0, 1, 3))
    w = res[["wx", "wy", "wz"]].to_numpy(float)
    dd = P["d"].to_numpy(float)
    rr = (pn * w).sum(axis=1) + dd
    dr = np.abs(res["r"].to_numpy(float) - rr)
    tol = 1e-6 + 1e-9 * np.abs(rr)
    bad_r = u0 & (dr > tol)
    out["P5_r_mismatch"] = int(bad_r.sum()); out["P5_max_r_dev"] = float(dr[u0].max()) if u0.any() else 0.0
    lay = S["layer"].to_numpy(int)
    edge = S["edge_m"].to_numpy(float)
    vs = float(edge[lay == 0].max()) if (lay == 0).any() else float("nan")
    out["voxel_size"] = vs
    out["P6_edge_mismatch"] = int((np.abs(edge - vs / 2.0 ** lay) > 1e-9 * max(1.0, vs)).sum())
    pl = S[S["is_plane"] == 1]
    rel = lambda a, b: np.abs(a - b) / np.maximum(1e-300, np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        r10 = pl["eig1"].to_numpy(float) / pl["eig0"].to_numpy(float)
        r21 = pl["eig2"].to_numpy(float) / pl["eig1"].to_numpy(float)
        rg = 1.0 + 3.0 * np.sqrt(2.0 / (pl["point_count"].to_numpy(float) - 1.0))
    def mism(col, expect, relative=True):
        got = pl[col].to_numpy(float)
        both_nan = np.isnan(got) & np.isnan(expect)
        ok = both_nan | (rel(got, expect) <= 1e-9 if relative else np.abs(got - expect) <= 1e-12)
        return int((~ok).sum())
    out["P7_planarity_value_mismatch"] = mism("planarity_value", pl["eig0"].to_numpy(float), relative=False)
    out["P7_second_eig_flag_mismatch"] = int((pl["second_eig_below_threshold"].to_numpy(int) != (pl["eig1"].to_numpy(float) < pl["plane_threshold"].to_numpy(float)).astype(int)).sum())
    out["P7_ratio_e1_over_e0_mismatch"] = mism("ratio_e1_over_e0", r10)
    out["P7_ratio_e2_over_e1_mismatch"] = mism("ratio_e2_over_e1", r21)
    out["P7_rel_gap_threshold_mismatch"] = mism("rel_gap_threshold", rg)
    # sensor offset and the plane table
    st = None
    if state is not None:
        s_ = state[(state["frame"] == frame) & (state["iter"] == iteration)]
        if len(s_):
            st = s_.iloc[0][["px", "py", "pz"]].to_numpy(float)
    res = res.assign(_w=1.0 / res["sigma_squared"].to_numpy(float), _z=w[:, 2])
    g = res.groupby("node_id").agg(n_resid=("r", "size"), weight_sum=("_w", "sum"), mean_height_w_z=("_z", "mean")).reset_index()
    T = g.merge(S.reset_index(), on="node_id", how="left")
    keep = ["node_id", "layer", "edge_m", "point_count", "eig0", "eig1", "eig2", "planarity_value", "planarity_pass", "second_eig_below_threshold",
            "ratio_e1_over_e0", "ratio_e2_over_e1", "rel_gap_threshold", "center_x", "center_y", "center_z", "normal_x", "normal_y", "normal_z", "n_resid", "weight_sum", "mean_height_w_z"]
    T = T[keep]
    T["offset_to_sensor_m"] = (T[["normal_x", "normal_y", "normal_z"]].to_numpy(float) @ st + S.loc[T["node_id"].to_numpy(), "d"].to_numpy(float)) if st is not None else np.nan
    out["n_planes_matched"] = int(len(T))
    out["sensor_position_found"] = bool(st is not None)
    out["plane_threshold"] = float(S["plane_threshold"].iloc[0]) if len(S) else float("nan")
    viol = sum(v for k, v in out.items() if k.startswith(("P1_", "P2_", "P3_", "P4_non_unit", "P4_normal_mismatch", "P5_r_mismatch", "P6_", "P7_")) and isinstance(v, int))
    out["violations_total"] = int(viol) + dup
    return out, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--frame", type=int, default=1)
    a = ap.parse_args()
    snap, dump, state = load(a.dir, "post_calibration_voxelmap.csv"), load(a.dir, "residual_dump.csv"), load(a.dir, "residual_dump_state.csv")
    if snap is None or dump is None:
        print("need post_calibration_voxelmap.csv and residual_dump.csv in", a.dir); return 2
    if "edge_m" not in snap.columns:
        print("post_calibration_voxelmap.csv has no R74 test-value columns (edge_m ...): not an R74 run"); return 2
    out, T = check(snap, dump, state, a.frame)
    os.makedirs(a.out, exist_ok=True)
    T.to_csv(os.path.join(a.out, "r74_plane_table.csv.gz"), index=False, float_format="%.9g", compression="gzip")
    json.dump(out, open(os.path.join(a.out, "r74_plane_check.json"), "w"), indent=1)
    print(json.dumps(out))
    return 0 if out["violations_total"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
