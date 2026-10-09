#!/usr/bin/env python3
"""R84: l2_reference_values.csv = the numbers of the EXISTING layer-2 arms that serve as references for Tier 3 cells (and the R83 baseline arms for the CMP comparison).
Planner script, run in the planner sandbox on files the planner holds (NOT on the node). Inputs: r81_sw.csv (R81 return, sweep rows), combined_sweep.csv (R81 + R82A + R82 control rows, bag,q,edge,gt_ate_mm,gt_max_mm,src,collapsed), R83 phase_summary.csv.
    python3 make_r84_ref_values.py --r81-sw r81_sw.csv --combined combined_sweep.csv --r83-phase phase_summary.csv --out l2_reference_values.csv
Columns: arm, source, bag, ovl, fit, lines, edge, q, gt_ate_mm, gt_max_mm, collapsed (gt_max_mm > 500). Values are copied, not recomputed."""
import argparse, csv, re

def tag(e):
    return ("%g" % e).replace(".", "p")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r81-sw", required=True); ap.add_argument("--combined", required=True); ap.add_argument("--r83-phase", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for r in csv.DictReader(open(a.r81_sw)):
        if r["pfn"] != "3":
            continue
        rows.append(dict(arm=r["arm"], source="R81", bag=r["bag"], ovl="rt", fit="pca", lines="on" if r["line"] == "hardabs" else "off", edge=r["edge"], q=r["q"], gt_ate_mm=r["gt_ate_mm"], gt_max_mm=r["gt_max_mm"]))
    for r in csv.DictReader(open(a.combined)):
        if r["src"] != "R82A":
            continue
        rows.append(dict(arm="r82_rt_sw_%s_off_v%s_pfn3_%s" % (r["bag"], tag(float(r["edge"])), r["q"]), source="R82A", bag=r["bag"], ovl="rt", fit="pca", lines="off", edge=r["edge"], q=r["q"], gt_ate_mm=r["gt_ate_mm"], gt_max_mm=r["gt_max_mm"]))
    pat = re.compile(r"^r83_dc_sw_(?P<bag>[a-z]+_\d\d)_off_v(?P<e>[0-9p]+)_pfn3_(?P<q>fq|mq)$")
    pml = re.compile(r"^r83_rt_ml_(?P<bag>[a-z]+_\d\d)_l2_(?P<q>fq|mq)$")
    for r in csv.DictReader(open(a.r83_phase)):
        m = pat.match(r["arm"]); n = pml.match(r["arm"])
        if m:
            rows.append(dict(arm=r["arm"], source="R83", bag=m["bag"], ovl="dc", fit="pca", lines="off", edge=m["e"].replace("p", "."), q=m["q"], gt_ate_mm=r["gt_ate_mm"], gt_max_mm=r["gt_max_mm"]))
        elif n:
            rows.append(dict(arm=r["arm"], source="R83", bag=n["bag"], ovl="rt", fit="pca", lines="off", edge="0.425", q=n["q"], gt_ate_mm=r["gt_ate_mm"], gt_max_mm=r["gt_max_mm"]))
    seen = {}
    for r in rows:
        r["collapsed"] = "" if r["gt_max_mm"] in ("", None) else str(float(r["gt_max_mm"]) > 500)
        seen[r["arm"]] = r
    cols = ["arm", "source", "bag", "ovl", "fit", "lines", "edge", "q", "gt_ate_mm", "gt_max_mm", "collapsed"]
    w = csv.DictWriter(open(a.out, "w", newline=""), cols); w.writeheader(); w.writerows(sorted(seen.values(), key=lambda r: r["arm"]))
    print("rows", len(seen))

main()
