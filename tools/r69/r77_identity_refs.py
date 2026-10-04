#!/usr/bin/env python3
"""R77: reference signatures of the R76 identity arms that the R77 identity arms must reproduce (written by the PLANNING agent from the returned R76 compact files; shipped as r77_manifest/identity_refs.json).

    python3 tools/r69/r77_identity_refs.py --r76-compact <R76_RETURN/compact> --out r77_manifest/identity_refs.json

For each identity arm of make_r77_manifest.IDENT the file holds, from the reference arm's pos_trace_all.csv.gz (scan_id, px, py, pz, yaw_rad as the CSV text) and residual_dump_all.csv.gz (frame 1):
the per-scan values, the row count, the row_kind counts, a sha256 per dump column (plane_key and node_id excluded: pointer-like ids that differ between runs) and, separately, the sha256 of node_id
(F-452: node_id differed at voxel 1.0 in R76 and is a reported INFO row, not a verdict).
"""
import argparse, hashlib, json, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

POS = ["scan_id", "px", "py", "pz", "yaw_rad"]
SKIP = {"plane_key", "node_id"}


def _sha(D, c):
    return hashlib.sha256(D[c].to_csv(index=False, header=False).encode()).hexdigest()


def dump_sigs(D):
    """-> {column: sha256 of the column's CSV text} for every column but plane_key and node_id."""
    return {c: _sha(D, c) for c in D.columns if c not in SKIP}


def node_sig(D):
    return _sha(D, "node_id") if "node_id" in D.columns else None


def build(entries, c76):
    refs = {}
    for aid, mode, v, ref in entries:
        d = os.path.join(c76, ref)
        P = pd.read_csv(os.path.join(d, "pos_trace_all.csv.gz")); D = pd.read_csv(os.path.join(d, "residual_dump_all.csv.gz"))
        refs[aid] = dict(ref_arm=ref, source="R76", pos={c: P[c].tolist() for c in POS}, dump_rows=int(len(D)),
                         row_kind_counts={str(k): int(n) for k, n in D["row_kind"].value_counts().sort_index().items()} if "row_kind" in D.columns else None,
                         dump_col_sha=dump_sigs(D), node_id_sha=node_sig(D))
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r76-compact", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import make_r77_manifest as M
    refs = build(M.IDENT, a.r76_compact)
    json.dump(refs, open(a.out, "w"), indent=0)
    for k, v in refs.items():
        print(k, "<-", v["ref_arm"], "scans", len(v["pos"]["scan_id"]), "dump rows", v["dump_rows"], "row kinds", v["row_kind_counts"])


if __name__ == "__main__":
    main()
