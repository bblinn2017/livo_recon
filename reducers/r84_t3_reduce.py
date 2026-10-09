#!/usr/bin/env python3
"""r84_t3_reduce: Tier 3 (max_layer 0 voxel-edge sweep) against the layer-2 reference arms, host side. Python only (standard library); no node run.

  python3 r84_t3_reduce.py --phase-summary <return>/phase_summary.csv --manifest r84_manifest --out <return dir>
  python3 r84_t3_reduce.py --selftest

Inputs: phase_summary.csv (return; columns used: arm, gt_ate_mm, gt_max_mm), r84_manifest/arm_table.tsv (group T3: arm, bag, mode, fit, lines, edge, l2_ref, l2_source),
r84_manifest/l2_reference_values.csv (arm, gt_ate_mm, gt_max_mm of the existing reference arms).
The layer-2 reference of a T3 arm is the arm named in arm_table `l2_ref`: its numbers come from phase_summary.csv when l2_source == R84 (the L2R arm), else from
l2_reference_values.csv (l2_source R81, R82A, R83). Collapse = gt_max_mm > 500. d_ate_mm = layer-0 gt_ate_mm minus layer-2 gt_ate_mm (mm).

OUT r84_t3_rows.csv, one row per T3 arm (1,296):
  arm, bag, overlap (arm_table mode: dc discard | rt retime_trim), fit (pca|deb), lines (off|on), q (fq|mq, from the arm name), edge (voxel edge, m),
  gt_ate_mm, gt_max_mm, collapsed (layer-0 arm), l2_ref (layer-2 reference arm), l2_source (R81|R82A|R83|R84), l2_gt_ate_mm, l2_gt_max_mm, l2_collapsed,
  d_ate_mm (blank when either arm is missing)
OUT r84_t3_collapse_counts.csv, one row per (overlap, fit, lines, q, edge):
  n_arms, n_collapsed_layer0, n_collapsed_layer2, n_missing (bags where the layer-0 or the layer-2 arm has no value), n_missing_layer0, n_missing_layer2,
  collapsed_by_bag = one token per bag `<bag>:<a><b>`, a = layer 0, b = layer 2, each 1 collapsed, 0 not collapsed, ? missing
OUT r84_t3_paired.csv, one row per (overlap, fit, lines, q, edge); d_ate_mm as above:
  n_pairs, n_layer0_lower (d<0), n_layer0_higher (d>0), n_equal (d=0 at 0.001 mm), median_d_ate_mm   -- over the bags where neither arm collapsed
  n_pairs_all, n_layer0_lower_all, n_layer0_higher_all, n_equal_all, median_d_ate_mm_all   -- the same over every bag with both arms present
OUT r84_t3_best.csv, one row per (bag, overlap, fit, lines, q):
  l0_best_ate_mm / l0_best_edge (lowest gt_ate_mm among the non-collapsed layer-0 edges; ties -> lower edge), l2_best_ate_mm / l2_best_edge (same, layer 2),
  d_best_ate_mm (l0 best minus l2 best), l0_range_lo_edge / l0_range_hi_edge and l2_range_lo_edge / l2_range_hi_edge (lowest and highest edge, 0.30..0.70,
  at which the arm does not collapse; blank when it always collapses or has no value), n_noncollapsed_l0, n_noncollapsed_l2 (edges not collapsed)
OUT r84_t3_missing.txt, one line per arm (layer-0 arm in phase_summary, layer-2 reference in phase_summary or l2_reference_values) without a usable gt_ate_mm/gt_max_mm.
Exit 1 when anything is missing (tables are still written)."""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python3 -I drops the script dir from sys.path
import r84_common as C  # noqa: E402

ROWS_HDR = ["arm", "bag", "overlap", "fit", "lines", "q", "edge", "gt_ate_mm", "gt_max_mm", "collapsed", "l2_ref", "l2_source", "l2_gt_ate_mm", "l2_gt_max_mm",
            "l2_collapsed", "d_ate_mm"]
KEY = ("overlap", "fit", "lines", "q")


def edge_s(e):
    return "%g" % float(e)


def usable(v):
    return v is not None and v["gt_ate_mm"] is not None and v["gt_max_mm"] is not None


def build_rows(arms, phase, refs, missing):
    rows = []
    for a in arms:
        arm = a["arm"]
        v0 = phase.get(arm)
        if not usable(v0):
            missing.append("phase_summary row (layer 0): " + arm if v0 is None else "phase_summary gt_ate_mm/gt_max_mm empty (layer 0): " + arm)
            v0 = None
        ref = a["l2_ref"]
        if a["l2_source"] == "R84":
            v2, where = phase.get(ref), "phase_summary"
        else:
            v2, where = refs.get(ref), "l2_reference_values"
        if not usable(v2):
            missing.append("%s %s (layer 2, reference of %s): %s" % (where, "row" if v2 is None else "gt_ate_mm/gt_max_mm empty", arm, ref))
            v2 = None
        r = dict(arm=arm, bag=a["bag"], overlap=a["mode"], fit=a["fit"], lines=a["lines"], q=C.q_of(arm), edge=float(a["edge"]), l2_ref=ref, l2_source=a["l2_source"],
                 ate=v0 and v0["gt_ate_mm"], gmax=v0 and v0["gt_max_mm"], l2ate=v2 and v2["gt_ate_mm"], l2gmax=v2 and v2["gt_max_mm"])
        r["col"] = C.collapsed_of(r["gmax"])
        r["l2col"] = C.collapsed_of(r["l2gmax"])
        r["d"] = None if r["ate"] is None or r["l2ate"] is None else round(r["ate"] - r["l2ate"], 3)
        rows.append(r)
    return rows


def cells(rows):
    d = {}
    for r in rows:
        d.setdefault(tuple(r[k] for k in KEY) + (r["edge"],), []).append(r)
    return dict(sorted(d.items()))


def tri(token):
    return "?" if token is None else ("1" if token else "0")


def reduce(arms, phase, refs):
    missing = []
    rows = build_rows(arms, phase, refs, missing)
    out_rows = [[r["arm"], r["bag"], r["overlap"], r["fit"], r["lines"], r["q"], edge_s(r["edge"]), C.f3(r["ate"]), C.f3(r["gmax"]), C.bflag(r["col"]), r["l2_ref"],
                 r["l2_source"], C.f3(r["l2ate"]), C.f3(r["l2gmax"]), C.bflag(r["l2col"]), C.f3(r["d"])] for r in rows]
    counts, paired = [], []
    for k, g in cells(rows).items():
        g = sorted(g, key=lambda r: r["bag"])
        m0 = sum(r["ate"] is None for r in g)
        m2 = sum(r["l2ate"] is None for r in g)
        counts.append(list(k[:4]) + [edge_s(k[4]), len(g), sum(r["col"] is True for r in g), sum(r["l2col"] is True for r in g),
                                     sum(r["ate"] is None or r["l2ate"] is None for r in g), m0, m2,
                                     " ".join("%s:%s%s" % (r["bag"], tri(r["col"]), tri(r["l2col"])) for r in g)])
        both = [r for r in g if r["d"] is not None]
        nc = [r for r in both if r["col"] is False and r["l2col"] is False]

        def stats(s):
            ds = [r["d"] for r in s]
            return [len(ds), sum(d < 0 for d in ds), sum(d > 0 for d in ds), sum(d == 0 for d in ds), C.f3(C.median(ds))]
        paired.append(list(k[:4]) + [edge_s(k[4])] + stats(nc) + stats(both))
    best = []
    bykey = {}
    for r in rows:
        bykey.setdefault((r["bag"],) + tuple(r[k] for k in KEY), []).append(r)
    for k, g in sorted(bykey.items()):
        res = {}
        for lay, ate, col in (("l0", "ate", "col"), ("l2", "l2ate", "l2col")):
            ok = sorted([r for r in g if r[col] is False and r[ate] is not None], key=lambda r: (r[ate], r["edge"]))
            es = sorted(r["edge"] for r in ok)
            res[lay] = (ok[0][ate] if ok else None, ok[0]["edge"] if ok else None, es[0] if es else None, es[-1] if es else None, len(es))
        d = None if res["l0"][0] is None or res["l2"][0] is None else round(res["l0"][0] - res["l2"][0], 3)
        e = lambda x: "" if x is None else edge_s(x)
        best.append(list(k) + [C.f3(res["l0"][0]), e(res["l0"][1]), C.f3(res["l2"][0]), e(res["l2"][1]), C.f3(d), e(res["l0"][2]), e(res["l0"][3]), e(res["l2"][2]),
                               e(res["l2"][3]), res["l0"][4], res["l2"][4]])
    return out_rows, counts, paired, best, missing


COUNTS_HDR = list(KEY) + ["edge", "n_arms", "n_collapsed_layer0", "n_collapsed_layer2", "n_missing", "n_missing_layer0", "n_missing_layer2", "collapsed_by_bag"]
_ST = ["n_pairs", "n_layer0_lower", "n_layer0_higher", "n_equal", "median_d_ate_mm"]
PAIRED_HDR = list(KEY) + ["edge"] + _ST + [s + "_all" if s != "n_pairs" else "n_pairs_all" for s in _ST]
BEST_HDR = ["bag"] + list(KEY) + ["l0_best_ate_mm", "l0_best_edge", "l2_best_ate_mm", "l2_best_edge", "d_best_ate_mm", "l0_range_lo_edge", "l0_range_hi_edge",
                                   "l2_range_lo_edge", "l2_range_hi_edge", "n_noncollapsed_l0", "n_noncollapsed_l2"]


def run(phase_path, manifest, out):
    os.makedirs(out, exist_ok=True)
    arms = C.load_arm_table(manifest, "T3")
    if not arms:
        print("no T3 arms in %s/arm_table.tsv" % manifest, file=sys.stderr)
        return 2
    o = reduce(arms, C.load_phase(phase_path), C.load_refs(manifest))
    C.write_csv(os.path.join(out, "r84_t3_rows.csv"), ROWS_HDR, o[0])
    C.write_csv(os.path.join(out, "r84_t3_collapse_counts.csv"), COUNTS_HDR, o[1])
    C.write_csv(os.path.join(out, "r84_t3_paired.csv"), PAIRED_HDR, o[2])
    C.write_csv(os.path.join(out, "r84_t3_best.csv"), BEST_HDR, o[3])
    print("T3 arms: %d, missing entries: %d" % (len(arms), len(o[4])))
    return C.write_missing(os.path.join(out, "r84_t3_missing.txt"), o[4])


# ---------------------------------------------------------------- selftest
def selftest():
    c = C.Checker()
    d = tempfile.mkdtemp()
    man = os.path.join(d, "man")
    os.makedirs(man)
    cols = ["arm", "group", "bag", "mode", "fit", "lines", "edge", "l2_ref", "l2_source"]
    at, refs, ph = [], [], []
    # 2 bags x edges 0.3, 0.35, 0.4 ; rt pca off fq ; edges 0.3,0.35 reference R81 (values file), 0.4 reference R84 (phase row)
    spec = {  # (bag, edge): (l0 ate, l0 max, l2 ate, l2 max)
        ("eee_01", 0.3): (30.0, 100.0, 40.0, 100.0),    # d = -10
        ("eee_01", 0.35): (50.0, 600.0, 45.0, 100.0),   # layer 0 collapsed, d = +5 (all-bags only)
        ("eee_01", 0.4): (20.0, 100.0, 20.0, 100.0),    # equal, R84 ref
        ("nya_01", 0.3): (60.0, 100.0, 55.0, 100.0),    # d = +5
        ("nya_01", 0.35): (None, None, 10.0, 100.0),    # layer-0 arm missing
        ("nya_01", 0.4): (70.0, 100.0, None, None),     # R84 reference arm missing from phase_summary
    }
    for (bag, e), (a0, m0, a2, m2) in spec.items():
        tag = ("%g" % e).replace(".", "p")
        arm = "r84_rt_t3_%s_pca_off_v%s_fq" % (bag, tag)
        if e < 0.4:
            ref, src = "r81_rt_sw_%s_off_v%s_pfn3_fq" % (bag, tag), "R81"
            if a2 is not None:
                refs.append([ref, "R81", bag, "rt", "pca", "off", e, "fq", a2, m2, m2 > 500])
        else:
            ref, src = "r84_rt_l2_%s_pca_off_v%s_fq" % (bag, tag), "R84"
            if a2 is not None:
                ph.append([ref, a2, m2])
        at.append([arm, "T3", bag, "rt", "pca", "off", e, ref, src])
        if a0 is not None:
            ph.append([arm, a0, m0])
    # decoy: a phase row for a reference arm that must NOT be used for an R81 reference, and a values-file row for the R84 one that must NOT be used
    ph.append(["r81_rt_sw_eee_01_off_v0p3_pfn3_fq", 999.0, 10.0])
    refs.append(["r84_rt_l2_eee_01_pca_off_v0p4_fq", "R84", "eee_01", "rt", "pca", "off", 0.4, "fq", 888.0, 10.0, False])
    with open(os.path.join(man, "arm_table.tsv"), "w") as f:
        f.write("\n".join("\t".join(map(str, r)) for r in [cols] + at) + "\n")
    with open(os.path.join(man, "l2_reference_values.csv"), "w") as f:
        f.write("arm,source,bag,ovl,fit,lines,edge,q,gt_ate_mm,gt_max_mm,collapsed\n" + "\n".join(",".join(map(str, r)) for r in refs) + "\n")
    pp = os.path.join(d, "phase_summary.csv")
    with open(pp, "w") as f:
        f.write("arm,gt_ate_mm,gt_max_mm,exit_code\n" + "\n".join("%s,%s,%s,0" % tuple(r) for r in ph) + "\n")
    out = os.path.join(d, "out")
    rc = run(pp, man, out)
    R = {r["arm"]: r for r in C.read_rows(os.path.join(out, "r84_t3_rows.csv"))}
    c("rows header exact", open(os.path.join(out, "r84_t3_rows.csv")).readline().strip() == ",".join(ROWS_HDR))
    c("6 rows written, 2 arms flagged missing, exit 1", len(R) == 6 and rc == 1)
    a = R["r84_rt_t3_eee_01_pca_off_v0p3_fq"]
    c("d_ate sign: layer0 30 minus layer2 40 = -10.000", a["d_ate_mm"] == "-10.000", a["d_ate_mm"])
    c("R81 reference taken from l2_reference_values, not the phase decoy", a["l2_gt_ate_mm"] == "40.000" and a["l2_source"] == "R81")
    a = R["r84_rt_t3_eee_01_pca_off_v0p4_fq"]
    c("R84 reference taken from phase_summary, not the values decoy", a["l2_gt_ate_mm"] == "20.000" and a["d_ate_mm"] == "0.000")
    a = R["r84_rt_t3_eee_01_pca_off_v0p35_fq"]
    c("collapsed = gt_max > 500 (layer 0 yes, layer 2 no)", a["collapsed"] == "True" and a["l2_collapsed"] == "False")
    a = R["r84_rt_t3_nya_01_pca_off_v0p35_fq"]
    c("missing layer-0 arm: blank values, blank d", a["gt_ate_mm"] == "" and a["collapsed"] == "" and a["d_ate_mm"] == "" and a["l2_gt_ate_mm"] == "10.000")
    a = R["r84_rt_t3_nya_01_pca_off_v0p4_fq"]
    c("missing R84 reference: blank l2, blank d", a["l2_gt_ate_mm"] == "" and a["l2_collapsed"] == "" and a["d_ate_mm"] == "")
    mt = open(os.path.join(out, "r84_t3_missing.txt")).read().splitlines()
    c("missing list names both arms", len(mt) == 2 and any("nya_01_pca_off_v0p35_fq" in x for x in mt) and any("r84_rt_l2_nya_01_pca_off_v0p4_fq" in x for x in mt), str(mt))
    CC = {r["edge"]: r for r in C.read_rows(os.path.join(out, "r84_t3_collapse_counts.csv"))}
    c("collapse counts edge 0.35", CC["0.35"]["n_arms"] == "2" and CC["0.35"]["n_collapsed_layer0"] == "1" and CC["0.35"]["n_collapsed_layer2"] == "0"
      and CC["0.35"]["n_missing"] == "1" and CC["0.35"]["n_missing_layer0"] == "1", str(CC["0.35"]))
    c("per-bag flag string", CC["0.35"]["collapsed_by_bag"] == "eee_01:10 nya_01:?0" and CC["0.4"]["collapsed_by_bag"] == "eee_01:00 nya_01:0?", CC["0.35"]["collapsed_by_bag"])
    P = {r["edge"]: r for r in C.read_rows(os.path.join(out, "r84_t3_paired.csv"))}
    c("paired 0.3: two pairs, one lower, one higher, median -2.5", (P["0.3"]["n_pairs"], P["0.3"]["n_layer0_lower"], P["0.3"]["n_layer0_higher"], P["0.3"]["median_d_ate_mm"])
      == ("2", "1", "1", "-2.500"), str(P["0.3"]))
    c("paired 0.35: non-collapsed set empty, all-bags set has 1 higher", P["0.35"]["n_pairs"] == "0" and P["0.35"]["median_d_ate_mm"] == ""
      and P["0.35"]["n_pairs_all"] == "1" and P["0.35"]["n_layer0_higher_all"] == "1" and P["0.35"]["median_d_ate_mm_all"] == "5.000", str(P["0.35"]))
    c("paired 0.4: one equal pair", P["0.4"]["n_equal"] == "1" and P["0.4"]["n_pairs"] == "1")
    B = {r["bag"]: r for r in C.read_rows(os.path.join(out, "r84_t3_best.csv"))}
    e = B["eee_01"]
    c("best eee_01 layer 0 skips collapsed 0.35: best 20 at 0.4, range 0.3..0.4", (e["l0_best_ate_mm"], e["l0_best_edge"], e["l0_range_lo_edge"], e["l0_range_hi_edge"])
      == ("20.000", "0.4", "0.3", "0.4"), str(e))
    c("best eee_01 layer 2 best 20 at 0.4 and difference 0", (e["l2_best_ate_mm"], e["l2_best_edge"], e["d_best_ate_mm"]) == ("20.000", "0.4", "0.000"))
    n = B["nya_01"]
    c("best nya_01 layer 0 over present arms; layer 2 best 10 at 0.35", (n["l0_best_ate_mm"], n["l0_best_edge"], n["l2_best_ate_mm"], n["l2_best_edge"]) == ("60.000", "0.3", "10.000", "0.35"))
    # all collapsed -> blank
    arms = [dict(arm="x_fq", bag="b", mode="rt", fit="pca", lines="off", edge="0.3", l2_ref="r", l2_source="R81")]
    o = reduce(arms, {"x_fq": dict(gt_ate_mm=5.0, gt_max_mm=900.0)}, {"r": dict(gt_ate_mm=6.0, gt_max_mm=900.0)})
    c("always collapsed: best and range blank", o[3][0][6] == "" and o[3][0][11] == "" and o[3][0][14] == 0 and not o[4])
    return c.done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-summary")
    ap.add_argument("--manifest")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.phase_summary and a.manifest and a.out):
        ap.error("--phase-summary, --manifest and --out are required")
    return run(a.phase_summary, a.manifest, a.out)


if __name__ == "__main__":
    sys.exit(main())
