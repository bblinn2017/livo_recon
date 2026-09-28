#!/usr/bin/env python3
"""R49 Phase 3B: offline structural / equivalent-plane analysis over the
Phase 3A post-bootstrap voxel map dump matrix.

Offline analysis only -- reads post_calibration_voxelmap.csv and
post_calibration_voxel_observation_support.csv for every Phase 3A job and
reports the structural facts the round's instructions ask for. Does not
modify Gamma_L or merge planes; "equivalent-plane group" is a geometric
candidate label, not a claim of physical identity or statistical
independence.

Usage: python3 r49_phase3b_voxel_correlation_analysis.py <csv_root> <out_csv>
  <csv_root>/<job_id>/post_calibration_voxelmap.csv
  <csv_root>/<job_id>/post_calibration_voxel_observation_support.csv
"""
import sys
import csv
import math
import itertools
from collections import defaultdict

STATUS_NAMES = {"0": "OPEN", "1": "PARENT", "2": "CONVERGED", "3": "DISABLED"}


def load_voxelmap(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_support(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def f(row, key):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def node_key(row):
    return (row["root_key_x"], row["root_key_y"], row["root_key_z"], row["layer"], row["node_id"])


def analyze_job(job_id, rows, support_rows):
    """Section 1: root/node/status/layer/plane counts, point/support
    distributions, plane geometry summary."""
    out = []
    n_nodes = len(rows)
    status_counts = defaultdict(int)
    layer_counts = defaultdict(int)
    is_plane_count = 0
    parent_count = 0
    point_counts = []
    for r in rows:
        status_counts[STATUS_NAMES.get(r["status"], r["status"])] += 1
        layer_counts[r["layer"]] += 1
        if r["is_plane"] == "1":
            is_plane_count += 1
        if r["status"] == "1":
            parent_count += 1
        pc = f(r, "point_count")
        if pc is not None:
            point_counts.append(pc)

    out.append({
        "row_type": "map_snapshot_summary", "job_id": job_id,
        "n_nodes": n_nodes, "n_plane": is_plane_count, "n_non_plane": n_nodes - is_plane_count,
        "n_parent_split": parent_count,
        "status_open": status_counts.get("OPEN", 0), "status_parent": status_counts.get("PARENT", 0),
        "status_converged": status_counts.get("CONVERGED", 0), "status_disabled": status_counts.get("DISABLED", 0),
        "point_count_mean": (sum(point_counts) / len(point_counts)) if point_counts else 0,
        "point_count_max": max(point_counts) if point_counts else 0,
        "n_layers_seen": len(layer_counts),
    })

    # Observation-support distribution: how many distinct observation_ids
    # supported each plane node.
    node_obs = defaultdict(set)
    for s in support_rows:
        node_obs[s["node_id"]].add(s["observation_id"])
    plane_ids = {r["node_id"] for r in rows if r["is_plane"] == "1"}
    n_obs_hist = defaultdict(int)
    for nid in plane_ids:
        n_obs_hist[len(node_obs.get(nid, set()))] += 1
    for n_obs, count in sorted(n_obs_hist.items()):
        out.append({
            "row_type": "plane_observation_support_histogram", "job_id": job_id,
            "n_observations_supporting": n_obs, "n_planes": count,
        })

    # Per-plane rows (geometry).
    for r in rows:
        if r["is_plane"] != "1":
            continue
        out.append({
            "row_type": "plane", "job_id": job_id, "node_id": r["node_id"],
            "root_key_x": r["root_key_x"], "root_key_y": r["root_key_y"], "root_key_z": r["root_key_z"],
            "layer": r["layer"],
            "center_x": r["center_x"], "center_y": r["center_y"], "center_z": r["center_z"],
            "normal_x": r["normal_x"], "normal_y": r["normal_y"], "normal_z": r["normal_z"],
            "d": r["d"], "eig0": r["eig0"], "eig1": r["eig1"], "eig2": r["eig2"],
            "roughness": r.get("roughness", ""), "radius": r.get("radius", ""),
            "point_count": r.get("point_count", ""), "distinct_frames": r.get("distinct_frames", ""),
            "n_observations_supporting": len(node_obs.get(r["node_id"], set())),
        })
    return out, {r["node_id"]: r for r in rows if r["is_plane"] == "1"}


def match_nodes_aggregate_sequential(planes_a, planes_b, prep, plane_model):
    """Deterministic identity match: (root_key, layer, node_id). node_id was
    empirically confirmed identical between aggregate/sequential for the
    PCA plane_fit_mode (same row count, same values) -- for debiased mode
    the two runs produce different total node counts, so this reports the
    matched/unmatched split honestly rather than assuming correspondence."""
    rows = []
    ids_a, ids_b = set(planes_a), set(planes_b)
    matched = ids_a & ids_b
    only_a, only_b = ids_a - ids_b, ids_b - ids_a
    for nid in matched:
        ra, rb = planes_a[nid], planes_b[nid]
        dcenter = math.sqrt(sum((f(ra, k) - f(rb, k))**2 for k in ("center_x", "center_y", "center_z")))
        dnormal = math.sqrt(sum((f(ra, k) - f(rb, k))**2 for k in ("normal_x", "normal_y", "normal_z")))
        dd = abs(f(ra, "d") - f(rb, "d"))
        changed = dcenter > 1e-9 or dnormal > 1e-9 or dd > 1e-9
        if changed:
            rows.append({
                "row_type": "aggregate_vs_sequential_changed_plane", "prep": prep, "plane_model": plane_model,
                "node_id": nid, "delta_center_norm": dcenter, "delta_normal_norm": dnormal, "delta_d": dd,
            })
    rows.append({
        "row_type": "aggregate_vs_sequential_match_summary", "prep": prep, "plane_model": plane_model,
        "n_matched": len(matched), "n_only_aggregate": len(only_a), "n_only_sequential": len(only_b),
        "n_changed_among_matched": sum(1 for r in rows if r["row_type"] == "aggregate_vs_sequential_changed_plane"),
    })
    return rows


def equivalent_plane_groups(all_planes, angle_deg_thresh, sep_thresh):
    """Section 2: candidate equivalent-surface groups within ONE job's plane
    population -- unoriented normal angle + signed separation after
    normal-sign alignment. Geometric candidates only, not physical-surface
    claims. O(n^2) pairwise -- fine at this population size (~50-300 planes
    per job)."""
    planes = list(all_planes.values())
    n = len(planes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    def normal(p):
        v = (f(p, "normal_x"), f(p, "normal_y"), f(p, "normal_z"))
        norm = math.sqrt(sum(c * c for c in v)) or 1.0
        return tuple(c / norm for c in v)

    def center(p):
        return (f(p, "center_x"), f(p, "center_y"), f(p, "center_z"))

    for i, j in itertools.combinations(range(n), 2):
        ni, nj = normal(planes[i]), normal(planes[j])
        dot = sum(a * b for a, b in zip(ni, nj))
        angle = math.degrees(math.acos(max(-1.0, min(1.0, abs(dot)))))
        if angle > angle_deg_thresh:
            continue
        sign = 1.0 if dot >= 0 else -1.0
        ci, cj = center(planes[i]), center(planes[j])
        sep_vec = tuple(a - b for a, b in zip(ci, cj))
        signed_sep = abs(sum(c * n for c, n in zip(sep_vec, ni)))
        if signed_sep > sep_thresh:
            continue
        union(i, j)

    groups = defaultdict(list)
    for idx in range(n):
        groups[find(idx)].append(idx)
    return {gid: [planes[i] for i in members] for gid, members in groups.items() if len(members) > 1}


def main():
    csv_root, out_path = sys.argv[1], sys.argv[2]
    import os
    job_ids = sorted(os.listdir(csv_root))
    all_rows = []
    plane_index = {}  # job_id -> {node_id: row}

    for job_id in job_ids:
        vm_path = os.path.join(csv_root, job_id, "post_calibration_voxelmap.csv")
        sup_path = os.path.join(csv_root, job_id, "post_calibration_voxel_observation_support.csv")
        if not os.path.exists(vm_path):
            continue
        rows = load_voxelmap(vm_path)
        support = load_support(sup_path)
        job_rows, planes = analyze_job(job_id, rows, support)
        all_rows.extend(job_rows)
        plane_index[job_id] = planes

        groups = equivalent_plane_groups(planes, angle_deg_thresh=5.0, sep_thresh=0.05)
        for gid, members in groups.items():
            member_ids = [m["node_id"] for m in members]
            centers = [(f(m, "center_x"), f(m, "center_y"), f(m, "center_z")) for m in members]
            cx = [c[0] for c in centers]; cy = [c[1] for c in centers]; cz = [c[2] for c in centers]
            all_rows.append({
                "row_type": "equivalent_plane_group", "job_id": job_id, "group_id": gid,
                "n_members": len(members), "member_node_ids": ";".join(member_ids),
                "center_spread_x": max(cx) - min(cx), "center_spread_y": max(cy) - min(cy),
                "center_spread_z": max(cz) - min(cz),
            })
            for m in members:
                all_rows.append({
                    "row_type": "group_member", "job_id": job_id, "group_id": gid,
                    "node_id": m["node_id"], "layer": m["layer"],
                    "root_key_x": m["root_key_x"], "root_key_y": m["root_key_y"], "root_key_z": m["root_key_z"],
                })

    # Aggregate-vs-sequential matching, per (prep, plane_model) pair.
    for prep in ("canonical", "dsoff"):
        for plane_model in ("pca_combined", "pca_sensoronly", "debiased_combined", "debiased_sensoronly"):
            ja, jb = f"{prep}_aggregate_{plane_model}", f"{prep}_sequential_{plane_model}"
            if ja in plane_index and jb in plane_index:
                all_rows.extend(match_nodes_aggregate_sequential(plane_index[ja], plane_index[jb], prep, plane_model))

    fieldnames = sorted({k for r in all_rows for k in r.keys()})
    with open(out_path, "w", newline="") as f_out:
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"wrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
