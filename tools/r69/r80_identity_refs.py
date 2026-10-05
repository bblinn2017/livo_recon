#!/usr/bin/env python3
"""R80: reference per-scan positions that the R80 identity arms must reproduce (written by the PLANNING agent from the returned R79 compact files).

    python3 tools/r69/r80_identity_refs.py --r79-compact <R79_RETURN/compact> --out-dir r80_manifest

identity_refs_fb.csv.gz (columns arm, scan_id, px, py, pz, yaw_rad): the R79 arms named in make_r80_manifest.IDREF, one block per R79 arm (de-duplicated: several R80 arms can share one reference):
  the two pfn 3 window arms          against r79_dx_off_v0p425_pfn3_{fq,mq}            (the 60 s R79 arms)
  the nine message-log BASE3 arms    against r79_nb_<bag>_off_v0p425_pfn3_fq          (fixed-Q)
  the nine BASE3 motion-Q arms       against r79_nb_<bag>_off_v0p425_pfn3_mq          (motion-Q)
  the nine ZUPT detect_only arms     against r79_nb_<bag>_off_v0p425_pfn3_fq          (same reference as the message-log arms: detect_only never touches the state)
arm = the R79 arm name. Layout read: <r79-compact>/<r79 arm>/pos_trace_all.csv.gz (the R79 return's compact/ directory). A reference that is not in the return is an error naming every missing arm (nothing is written)."""
import argparse, os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import make_r80_manifest as M

COLS = ["scan_id", "px", "py", "pz", "yaw_rad"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--r79-compact", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    olds = {}
    for new, old in M.IDREF.items():
        olds.setdefault(old, []).append(new)
    missing = [old for old in olds if not os.path.exists(os.path.join(a.r79_compact, old, "pos_trace_all.csv.gz"))]
    if missing:
        sys.exit("reference arms absent from %s (no pos_trace_all.csv.gz): %s" % (a.r79_compact, ", ".join(missing)))
    rows = []
    for old, news in olds.items():
        P = pd.read_csv(os.path.join(a.r79_compact, old, "pos_trace_all.csv.gz"))[COLS].drop_duplicates("scan_id")
        P.insert(0, "arm", old); rows.append(P)
        print("%s <- %s: scans %d (used by %s)" % (old, os.path.join(a.r79_compact, old), len(P), ", ".join(news)))
    pd.concat(rows).to_csv(os.path.join(a.out_dir, "identity_refs_fb.csv.gz"), index=False, compression="gzip")
    print("wrote %d reference arms (%d R80 arms) to %s" % (len(olds), len(M.IDREF), os.path.join(a.out_dir, "identity_refs_fb.csv.gz")))


if __name__ == "__main__":
    main()
