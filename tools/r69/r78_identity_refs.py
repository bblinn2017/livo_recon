#!/usr/bin/env python3
"""R78: reference signatures that the R78 identity arms and same-batch controls must reproduce (written by the PLANNING agent from the returned R77 compact files).

    python3 tools/r69/r78_identity_refs.py --r77-compact <R77_RETURN/compact> --out-dir r78_manifest

Writes identity_refs.json (the 3 EF identity arms against r77_id_off_v1p0 / r77_id_hardabs_v0p425: per-scan values, frame-1 dump signatures, as r77_identity_refs did)
and identity_refs_fb.csv.gz (columns arm, scan_id, px, py, pz, yaw_rad: the per-scan positions of the 16 R77 control arms the r78_ct_* arms must reproduce)."""
import argparse, json, os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r77_identity_refs as R7
import make_r78_manifest as M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--r77-compact", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    refs = R7.build([(i, m, v, ref) for (i, m, v, ref, _d) in M.IDENT], a.r77_compact)
    json.dump(refs, open(os.path.join(a.out_dir, "identity_refs.json"), "w"), indent=0)
    for k, v in refs.items():
        print(k, "<-", v["ref_arm"], "scans", len(v["pos"]["scan_id"]), "dump rows", v["dump_rows"], "row kinds", v["row_kind_counts"])
    rows = []
    for t in M.arms():
        if t["group"] != "CTL": continue
        old = t["id"].replace("r78_ct_", "r77_fb_", 1)
        P = pd.read_csv(os.path.join(a.r77_compact, old, "pos_trace_all.csv.gz"))[["scan_id", "px", "py", "pz", "yaw_rad"]]
        P.insert(0, "arm", old); rows.append(P); print(t["id"], "<-", old, "scans", len(P))
    pd.concat(rows).to_csv(os.path.join(a.out_dir, "identity_refs_fb.csv.gz"), index=False, compression="gzip")


if __name__ == "__main__":
    main()
