#!/usr/bin/env python3
"""R76: reference signatures of the R74 / R75 arms that the R76 identity arms must reproduce (written by the PLANNING agent from the returned compact files; shipped as r76_manifest/identity_refs.json).

    python3 tools/r69/r76_identity_refs.py --r74-compact <R74_RETURN/compact> --r75-compact <R75_RETURN/compact> --out r76_manifest/identity_refs.json

For each identity arm of make_r76_manifest.IDENT the file holds, from the reference arm's pos_trace_all.csv.gz (scan_id, px, py, pz, yaw_rad as the CSV text) and residual_dump_all.csv.gz
(frame 1): the per-scan values, the row count, the row_kind counts and a sha256 per dump column (the text of that column, plane_key excluded: it is a pointer-like id that differs between runs).
r76_gates.py compares the new identity arm against it and REPORTS the difference (a fact, not a stop).
"""
import argparse, hashlib, json, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

POS = ["scan_id", "px", "py", "pz", "yaw_rad"]
SKIP = {"plane_key"}


def dump_sigs(D):
    """-> {column: sha256 of the column's CSV text} for every column but plane_key."""
    out = {}
    for c in D.columns:
        if c in SKIP:
            continue
        out[c] = hashlib.sha256(D[c].to_csv(index=False, header=False).encode()).hexdigest()
    return out


def build(entries, c74, c75):
    refs = {}
    for aid, mode, v, ref in entries:
        d = os.path.join(c75 if ref.startswith("r75_") else c74, ref)
        P = pd.read_csv(os.path.join(d, "pos_trace_all.csv.gz")); D = pd.read_csv(os.path.join(d, "residual_dump_all.csv.gz"))
        refs[aid] = dict(ref_arm=ref, source="R75" if ref.startswith("r75_") else "R74", pos={c: P[c].tolist() for c in POS}, dump_rows=int(len(D)),
                         row_kind_counts={str(k): int(n) for k, n in D["row_kind"].value_counts().sort_index().items()} if "row_kind" in D.columns else None,
                         dump_col_sha=dump_sigs(D))
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r74-compact", required=True); ap.add_argument("--r75-compact", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import make_r76_manifest as M
    refs = build(M.IDENT, a.r74_compact, a.r75_compact)
    json.dump(refs, open(a.out, "w"), indent=0)
    for k, v in refs.items():
        print(k, "<-", v["ref_arm"], "scans", len(v["pos"]["scan_id"]), "dump rows", v["dump_rows"], "row kinds", v["row_kind_counts"])


if __name__ == "__main__":
    main()
