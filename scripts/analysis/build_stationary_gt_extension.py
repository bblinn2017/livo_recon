#!/usr/bin/env python3
"""Build a diagnostic-only stationary extension without contaminating ATE.

The native ate_per_frame export supplies a global SE(3) alignment fitted only
from real GT.  This tool applies that fixed alignment to earlier estimator
samples and compares their position with the first real GT position.  Output
rows are explicitly labelled synthetic and must never be used to fit or report
official ATE.
"""

import argparse
import csv
from pathlib import Path


def number(row, name):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing/non-numeric column {name!r}") from exc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-gt-matches", required=True)
    ap.add_argument("--estimate", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--time-column", default="t_abs")
    ap.add_argument("--x-column", default="p_x")
    ap.add_argument("--y-column", default="p_y")
    ap.add_argument("--z-column", default="p_z")
    ap.add_argument("--stationary-end", type=float, required=True,
                    help="absolute timestamp at which the known-stationary interval ends")
    args = ap.parse_args()

    with open(args.real_gt_matches, newline="") as f:
        real = list(csv.DictReader(f))
    if not real:
        raise SystemExit("real-GT match file is empty")
    if any(r.get("gt_source") != "real" or r.get("gt_is_synthetic") != "0" for r in real):
        raise SystemExit("real-GT input contains a non-real row")

    first = min(real, key=lambda r: number(r, "t"))
    first_real_t = number(first, "t")
    gt_ref = [number(first, "gt_x"), number(first, "gt_y"), number(first, "gt_z")]
    R = [[number(first, f"align_r{i}{j}") for j in range(3)] for i in range(3)]
    trans = [number(first, "align_tx"), number(first, "align_ty"), number(first, "align_tz")]

    with open(args.estimate, newline="") as f:
        estimates = list(csv.DictReader(f))
    rows = []
    for e in estimates:
        t = number(e, args.time_column)
        if t >= first_real_t or t > args.stationary_end:
            continue
        p = [number(e, args.x_column), number(e, args.y_column), number(e, args.z_column)]
        aligned = [sum(R[i][j] * p[j] for j in range(3)) + trans[i] for i in range(3)]
        err = [aligned[i] - gt_ref[i] for i in range(3)]
        rows.append([t, "stationary_extension", 1, "real_gt_global", *p, *gt_ref,
                     *aligned, *(1000.0 * x for x in err)])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "gt_source", "gt_is_synthetic", "alignment_source",
                    "est_x", "est_y", "est_z", "gt_x", "gt_y", "gt_z",
                    "aligned_x", "aligned_y", "aligned_z",
                    "err_x_mm", "err_y_mm", "err_z_mm"])
        w.writerows(rows)
    print(f"wrote {len(rows)} diagnostic-only rows to {out}")


if __name__ == "__main__":
    main()
