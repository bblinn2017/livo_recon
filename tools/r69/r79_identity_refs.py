#!/usr/bin/env python3
"""R79: reference signatures that the R79 identity / control arms must reproduce (written by the PLANNING agent from the returned R78 compact files).

    python3 tools/r69/r79_identity_refs.py --r78-compact <R78_RETURN/compact> --out-dir r79_manifest

identity_refs.json: the 3 identity arms against r78_id_off_v1p0 / r78_id_hardabs_v0p425 / r78_id_second_off_v1p0 (per-scan values, frame-1 dump signatures, as r78 did).
identity_refs_fb.csv.gz (columns arm, scan_id, px, py, pz, yaw_rad): the per-scan positions of the R78 arms that the eee_01 NB arms (IDREF_NB) and the F-464 DX arms (IDREF_DX) must reproduce
(arm = the R78 arm name, de-duplicated)."""
import argparse, json, os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r77_identity_refs as R7
import make_r79_manifest as M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--r78-compact", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    refs = R7.build([(i, m, v, ref) for (i, m, v, ref, _x) in M.IDENT], a.r78_compact)
    for v in refs.values(): v["source"] = "R78"
    json.dump(refs, open(os.path.join(a.out_dir, "identity_refs.json"), "w"), indent=0)
    for k, v in refs.items():
        print(k, "<-", v["ref_arm"], "scans", len(v["pos"]["scan_id"]), "dump rows", v["dump_rows"], "row kinds", v["row_kind_counts"])
    rows, seen = [], set()
    for tab in (M.IDREF_NB, M.IDREF_DX):
        for new, old in tab.items():
            if old in seen: continue
            seen.add(old)
            P = pd.read_csv(os.path.join(a.r78_compact, old, "pos_trace_all.csv.gz"))[["scan_id", "px", "py", "pz", "yaw_rad"]]
            P.insert(0, "arm", old); rows.append(P); print(new, "<-", old, "scans", len(P))
    pd.concat(rows).to_csv(os.path.join(a.out_dir, "identity_refs_fb.csv.gz"), index=False, compression="gzip")


if __name__ == "__main__":
    main()
