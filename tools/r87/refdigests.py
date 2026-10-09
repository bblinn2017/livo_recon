#!/usr/bin/env python3
"""Canonical digests shared by r87_gates.py and the planner's one-off reference builder (copied from tools/r86b/r86b_dumpchecks.py of R86B)."""
import csv, gzip, hashlib
RES_COLS = ["source_index", "r", "sigma_squared", "plane_var_term", "node_id", "tier", "hrot_x", "hrot_y", "hrot_z", "hpos_x", "hpos_y", "hpos_z", "row_kind", "floor_term",
            "sigma_diag_squared", "s_prior_pose", "sigma_gate_squared", "voxel_point_count", "voxel_distinct_frames", "voxel_layer"]


def rd(p):
    return gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, errors="replace")


def resid_digests(path, frame=1):
    """{iteration: (rows, sha256)} of the canonical rows of one frame of residual_dump.csv(.gz), in file order (plane_key excluded: it is a pointer)."""
    h, n = {}, {}
    with rd(path) as f:
        for r in csv.DictReader(f):
            if int(r["frame"]) != frame:
                continue
            it = int(r["iter"])
            h.setdefault(it, hashlib.sha256())
            n[it] = n.get(it, 0) + 1
            h[it].update(("|".join(r[c] for c in RES_COLS) + "\n").encode())
    return {it: (n[it], h[it].hexdigest()) for it in n}


def map_digest(path):
    """(rows, sha256) of a voxel-map snapshot: every column except tangent_*, tokens as written."""
    h = hashlib.sha256()
    with rd(path) as f:
        rows = csv.reader(f)
        hdr = next(rows)
        keep = [i for i, c in enumerate(hdr) if not c.startswith("tangent_")]
        h.update(("|".join(hdr[i] for i in keep) + "\n").encode())
        n = 0
        for r in rows:
            h.update(("|".join(r[i] for i in keep) + "\n").encode())
            n += 1
    return n, h.hexdigest()
