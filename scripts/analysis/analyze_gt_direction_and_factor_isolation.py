#!/usr/bin/env python3
"""Offline GT-direction and factor-isolation analysis.

Reads the per-iteration decoupled CSV (or an equivalent CSV with the same
fields) and optionally pose-control gn_iteration/factor_isolation rows from the
unified diagnostics CSV. No estimator code is imported or executed.
"""
import argparse
import csv
import math
from collections import defaultdict


def interp_gt(gt, t, max_gap):
    if not gt or t < gt[0][0] or t > gt[-1][0]:
        return None
    lo, hi = 0, len(gt) - 1
    while lo <= hi and gt[lo][0] != t:
        mid = (lo + hi) // 2
        if gt[mid][0] < t:
            lo = mid + 1
        else:
            hi = mid - 1
    if lo < len(gt) and abs(gt[lo][0] - t) < 1e-12:
        return gt[lo][1:]
    i = max(0, lo - 1)
    j = min(len(gt) - 1, lo)
    if i == j or gt[j][0] - gt[i][0] > max_gap:
        return None
    a = (t - gt[i][0]) / (gt[j][0] - gt[i][0])
    return tuple((1 - a) * gt[i][k + 1] + a * gt[j][k + 1] for k in range(3))


def metrics(row, gt):
    p0 = tuple(float(row[f"tail_p_before_{a}"]) for a in "xyz")
    p1 = tuple(float(row[f"tail_p_after_{a}"]) for a in "xyz")
    d = tuple(p1[k] - p0[k] for k in range(3))
    togt = tuple(gt[k] - p0[k] for k in range(3))
    dn = math.sqrt(sum(x*x for x in d))
    gn = math.sqrt(sum(x*x for x in togt))
    after = math.sqrt(sum((p1[k] - gt[k])**2 for k in range(3)))
    before = gn
    out = dict(row)
    out["gt_error_pre"] = before
    out["gt_error_post"] = after
    out["gt_error_reduction"] = before - after
    if dn > 1e-12 and gn > 1e-12:
        par = sum(d[k] * togt[k] for k in range(3)) / gn
        perp2 = max(0.0, dn*dn - par*par)
        out["gt_cosine"] = sum(d[k] * togt[k] for k in range(3)) / (dn * gn)
        out["gt_parallel"] = par
        out["gt_perpendicular"] = math.sqrt(perp2)
    else:
        out["gt_cosine"] = "NA"
        out["gt_parallel"] = "NA"
        out["gt_perpendicular"] = "NA"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="CSV with timestamp,x,y,z")
    ap.add_argument("--iterations", action="append", default=[], help="iteration CSV; may be repeated")
    ap.add_argument("--full-diagnostics", help="unified pose_control_full_diagnostics.csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt-max-gap", type=float, default=0.15)
    args = ap.parse_args()

    with open(args.gt, newline="") as f:
        r = csv.DictReader(f)
        gt = sorted((float(x["timestamp"]), float(x["x"]), float(x["y"]), float(x["z"])) for x in r)

    rows = []
    for path in args.iterations:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))

    if args.full_diagnostics:
        with open(args.full_diagnostics, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("row_type") == "gn_iteration":
                    rows.append(row)

    out = []
    for row in rows:
        tkey = "scan_timestamp" if row.get("scan_timestamp") not in (None, "", "NA") else "t_abs_iter"
        if tkey not in row:
            continue
        try:
            t = float(row[tkey])
        except ValueError:
            continue
        g = interp_gt(gt, t, args.gt_max_gap)
        if g is not None and all(f"tail_p_{side}_{a}" in row for side in ("before", "after") for a in "xyz"):
            out.append(metrics(row, g))

    fields = sorted({k for r in out for k in r})
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} GT-valid iteration rows to {args.output}")


if __name__ == "__main__":
    main()
