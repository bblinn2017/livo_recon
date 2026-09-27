#!/usr/bin/env python3
import argparse
import csv
import math


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def norm3(row, prefix):
    return math.sqrt(sum(float(row[f"{prefix}_{k}"]) ** 2 for k in ("x", "y", "z")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    args = ap.parse_args()
    rows = [r for r in read_rows(args.csv) if int(r["scan_id"]) == 0]
    if not rows:
        raise SystemExit("no scan_id=0 rows")
    for arch in sorted(set(r["architecture"] for r in rows)):
        ar = [r for r in rows if r["architecture"] == arch]
        ar.sort(key=lambda r: (r["phase"], int(r["iteration"])))
        pre = [r for r in ar if r["phase"] == "pre_update"]
        post = sorted([r for r in ar if r["phase"] == "post_iteration"], key=lambda r: int(r["iteration"]))
        print(f"[{arch}] pre={len(pre)} post={len(post)}")
        if pre:
            r = pre[0]
            print("  pre absolute position norm", norm3(r, "p_x"), "absolute rotation-vector norm", norm3(r, "rlog"))
            print("  initial reference position", r["reference_p_x"], r["reference_p_y"], r["reference_p_z"])
            print("  initial reference rotation-vector", r["reference_rlog_x"], r["reference_rlog_y"], r["reference_rlog_z"])
        if post:
            for r in post:
                print(
                    "  iter", r["iteration"],
                    "delta position from initial", norm3(r, "relative_p"),
                    "delta rotation from initial", norm3(r, "relative_rlog"),
                    "delta position from frame prior", norm3(r, "relative_p"),
                    "n_res", r["n_residuals"],
                    "res_hash", r["residual_hash"],
                )


if __name__ == "__main__":
    main()
