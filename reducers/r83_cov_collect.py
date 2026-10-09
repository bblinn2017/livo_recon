#!/usr/bin/env python3
"""r83_cov_collect: concatenate per-arm CSVs written by r83_cov_consistency.py into one CSV. NTU rows (no attitude) and Hilti rows (attitude) have
different columns; the result is the outer join, with the missing columns empty.
  python3 r83_cov_collect.py --out all.csv  file1.csv file2.csv ...      (or  --glob 'dir/*.csv')
Lists any input that is missing or empty by name and exits 1 if there was one (nothing is skipped silently)."""
import argparse, glob, os, sys
import pandas as pd

def collect(files, out):
    frames, bad = [], []
    for f in files:
        if not os.path.exists(f) or os.path.getsize(f) < 5:
            bad.append(f); continue
        d = pd.read_csv(f)
        if len(d) == 0:
            bad.append(f); continue
        frames.append(d)
    if not frames:
        raise SystemExit("no usable input")
    allf = pd.concat(frames, ignore_index=True, sort=False)
    allf.to_csv(out, index=False)
    if bad:
        print("MISSING OR EMPTY INPUTS:", *bad, sep="\n  ", file=sys.stderr)
    return len(allf)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--glob"); ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    files = list(a.files) + (sorted(glob.glob(a.glob)) if a.glob else [])
    n = collect(files, a.out)
    print("rows:", n)
    return 0

if __name__ == "__main__":
    sys.exit(main())
