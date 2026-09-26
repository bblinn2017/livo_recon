#!/usr/bin/env python3
"""Compare splineless/decoupled/pose-control physical corrections offline.

Inputs are per-iteration CSVs. Files may have either the decoupled writer
schema or the unified pose-control diagnostics schema. GT is timestamp,x,y,z.
Outputs one row per method/scan/iteration plus pairwise direction metrics.
"""
import argparse
import csv
import math
from collections import defaultdict


def load_gt(path):
    with open(path, newline="") as f:
        rows = csv.DictReader(f)
        out = []
        for r in rows:
            out.append((float(r["timestamp"]), float(r["x"]), float(r["y"]), float(r["z"])))
    return sorted(out)


def interp_gt(gt, t, max_gap):
    if not gt or t < gt[0][0] or t > gt[-1][0]:
        return None
    lo, hi = 0, len(gt) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if gt[mid][0] < t:
            lo = mid + 1
        else:
            hi = mid
    if abs(gt[lo][0] - t) <= 1e-12:
        return gt[lo][1:]
    j = lo
    i = j - 1
    if i < 0 or gt[j][0] - gt[i][0] > max_gap:
        return None
    a = (t - gt[i][0]) / (gt[j][0] - gt[i][0])
    return tuple((1-a) * gt[i][k+1] + a * gt[j][k+1] for k in range(3))


def enrich(row, gt, max_gap):
    tkey = "scan_timestamp" if row.get("scan_timestamp") not in (None, "", "NA") else "t_abs_iter"
    if tkey not in row:
        return None
    try:
        t = float(row[tkey])
    except ValueError:
        return None
    for key in ("tail_p_before_x", "tail_p_before_y", "tail_p_before_z",
                "tail_p_after_x", "tail_p_after_y", "tail_p_after_z"):
        if key not in row:
            return None
    g = interp_gt(gt, t, max_gap)
    out = dict(row)
    out["analysis_timestamp"] = t
    if g is None:
        out.update({"gt_valid": "0", "gt_cosine": "NA", "gt_parallel": "NA",
                    "gt_perpendicular": "NA", "gt_error_pre": "NA",
                    "gt_error_post": "NA", "gt_error_reduction": "NA"})
        return out
    p0 = [float(row[f"tail_p_before_{a}"]) for a in "xyz"]
    p1 = [float(row[f"tail_p_after_{a}"]) for a in "xyz"]
    d = [p1[i] - p0[i] for i in range(3)]
    togt = [g[i] - p0[i] for i in range(3)]
    dn = math.sqrt(sum(x*x for x in d))
    gn = math.sqrt(sum(x*x for x in togt))
    e0 = gn
    e1 = math.sqrt(sum((p1[i] - g[i])**2 for i in range(3)))
    out["gt_valid"] = "1"
    out["gt_error_pre"] = e0
    out["gt_error_post"] = e1
    out["gt_error_reduction"] = e0 - e1
    if dn > 1e-12 and gn > 1e-12:
        out["gt_cosine"] = sum(d[i]*togt[i] for i in range(3)) / (dn*gn)
        out["gt_cosine"] = max(-1.0, min(1.0, out["gt_cosine"]))
        gh = [x/gn for x in togt]
        par = sum(d[i]*gh[i] for i in range(3))
        out["gt_parallel"] = par
        out["gt_perpendicular"] = math.sqrt(max(0.0, dn*dn - par*par))
    else:
        out["gt_cosine"] = out["gt_parallel"] = out["gt_perpendicular"] = "NA"
    return out


def cosine(a, b):
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(x*x for x in b))
    if da <= 1e-12 or db <= 1e-12:
        return None
    return sum(a[i]*b[i] for i in range(3))/(da*db)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--method", action="append", nargs=2, metavar=("NAME", "CSV"), required=True,
                    help="method name and per-iteration CSV; repeat for each method")
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt-max-gap", type=float, default=0.15)
    args = ap.parse_args()

    gt = load_gt(args.gt)
    rows = []
    grouped = defaultdict(dict)
    for name, path in args.method:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                e = enrich(r, gt, args.gt_max_gap)
                if e is None:
                    continue
                e["method"] = name
                rows.append(e)
                try:
                    key = (int(e["scan_id"]), int(e["iter"]))
                except (KeyError, ValueError):
                    continue
                grouped[key][name] = e

    pair_rows = []
    names = [n for n, _ in args.method]
    for key, by_method in grouped.items():
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                a, b = by_method.get(names[i]), by_method.get(names[j])
                if not a or not b:
                    continue
                va = [float(a[f"dp_tail_{x}"]) for x in "xyz"]
                vb = [float(b[f"dp_tail_{x}"]) for x in "xyz"]
                c = cosine(va, vb)
                pair_rows.append({
                    "scan_id": key[0], "iter": key[1],
                    "method_a": names[i], "method_b": names[j],
                    "tail_correction_cosine": "NA" if c is None else c,
                    "tail_magnitude_a": math.sqrt(sum(x*x for x in va)),
                    "tail_magnitude_b": math.sqrt(sum(x*x for x in vb)),
                    "gt_cosine_a": a.get("gt_cosine", "NA"),
                    "gt_cosine_b": b.get("gt_cosine", "NA"),
                    "gt_error_reduction_a": a.get("gt_error_reduction", "NA"),
                    "gt_error_reduction_b": b.get("gt_error_reduction", "NA"),
                })

    fields = set()
    for r in rows: fields.update(r)
    for r in pair_rows: fields.update(r)
    fields = sorted(fields)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
        for r in pair_rows:
            w.writerow(r)
    print(f"wrote {len(rows)} method rows and {len(pair_rows)} pairwise rows to {args.output}")


if __name__ == "__main__":
    main()
