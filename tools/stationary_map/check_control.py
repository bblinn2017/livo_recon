#!/usr/bin/env python3
"""Tolerance-based control: does a new patches/summary CSV equal an earlier one up to floating-point noise?
Usage: check_control.py --ref-patches A.csv --new-patches B.csv [--ref-summary S1.csv --new-summary S2.csv] [--out report.md]
Exit 0 = PASS, 1 = FAIL. Reads both files row by row (constant memory). Interprets nothing beyond PASS/FAIL and the maxima."""
import argparse, csv, sys

STRUCT = ["checkpoint", "patch_id", "surface_id", "n_children", "child_ids", "point_count", "rejected_points",
          "reservoir_pending", "rank2"]
EXACT = ["center_x", "center_y", "center_z"]
ABS = {"normal_x": 1e-12, "normal_y": 1e-12, "normal_z": 1e-12, "d": 1e-12, "planarity": 1e-12, "eig0": 1e-15}
REL = {"eig1": 1e-12, "eig2": 1e-12}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-patches", required=True); ap.add_argument("--new-patches", required=True)
    ap.add_argument("--ref-summary"); ap.add_argument("--new-summary"); ap.add_argument("--out")
    a = ap.parse_args()
    fails = []; maxes = {c: 0.0 for c in list(ABS) + list(REL)}; rows = 0; ndiff_rows = 0; struct_bad = 0; center_bad = 0
    tol_bad = {c: 0 for c in maxes}
    with open(a.ref_patches, newline="") as fr, open(a.new_patches, newline="") as fn:
        r, n = csv.DictReader(fr), csv.DictReader(fn)
        if r.fieldnames != n.fieldnames:
            fails.append("column headers differ")
        for x, y in zip(r, n):
            rows += 1; bad = False
            if any(x[c] != y[c] for c in STRUCT): struct_bad += 1; bad = True
            if any(x[c] != y[c] for c in EXACT): center_bad += 1; bad = True
            for c, t in ABS.items():
                dv = abs(float(x[c]) - float(y[c])); maxes[c] = max(maxes[c], dv)
                if dv > t: tol_bad[c] += 1; bad = True
            for c, t in REL.items():
                u, v = float(x[c]), float(y[c]); dv = abs(u - v) / max(abs(u), abs(v), 1e-300); maxes[c] = max(maxes[c], dv)
                if dv > t: tol_bad[c] += 1; bad = True
            ndiff_rows += bad
        if next(r, None) is not None or next(n, None) is not None: fails.append("row counts differ")
    if struct_bad: fails.append(f"{struct_bad} rows differ in structural columns")
    if center_bad: fails.append(f"{center_bad} rows differ in centre columns")
    for c, k in tol_bad.items():
        if k: fails.append(f"{k} rows exceed tolerance in {c}")
    summ = "not requested"
    if a.ref_summary and a.new_summary:
        def load(p):
            out = []
            for row in csv.reader(open(p, newline="")):
                if len(row) > 9: row[9] = ""          # column 10 = cumulative_insert_ms (wall-clock)
                out.append(row)
            return out
        s1, s2 = load(a.ref_summary), load(a.new_summary)
        summ = "identical (col 10 blanked)" if s1 == s2 else "DIFFERENT"
        if s1 != s2: fails.append("summary differs")
    lines = [f"# Control check: {'PASS' if not fails else 'FAIL'}", "",
             f"rows compared: {rows}; rows exceeding any tolerance: {ndiff_rows}", f"summary: {summ}", "",
             "max observed difference per column (abs for normal/d/planarity/eig0, relative for eig1/eig2):"]
    lines += [f"- {c}: {v:.3e}" for c, v in maxes.items()]
    lines += ["", "tolerances: structure and centres exact; normal, d, planarity abs 1e-12; eig0 abs 1e-15; eig1, eig2 rel 1e-12"]
    lines += [f"- FAIL: {f}" for f in fails]
    txt = "\n".join(lines) + "\n"
    print(txt)
    if a.out: open(a.out, "w").write(txt)
    sys.exit(1 if fails else 0)

main()
