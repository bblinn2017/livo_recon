#!/usr/bin/env python3
"""r85_l0_reduce: whole-bag layer-0 arms (group L0, voxel edges 0.20 and 0.25 m, max_layer 0) against their layer-2 reference arms, host side. Python only (standard library); no node run.

  python3 r85_l0_reduce.py --phase-summary <return>/phase_summary.csv --manifest r85_manifest --out <return dir> [--r84-all-arms r84_layer0_vs_layer2_all_arms.csv]
  python3 r85_l0_reduce.py --selftest

Inputs: phase_summary.csv (return; columns used: arm, gt_ate_mm, gt_max_mm), r85_manifest/arm_table.tsv (group L0: arm, bag, mode, fit, lines, edge, l2_ref, l2_source),
r85_manifest/l2_reference_values.csv (arm, gt_ate_mm, gt_max_mm of the existing reference arms).
The layer-2 reference of an L0 arm is the arm named in arm_table `l2_ref`: its numbers come from phase_summary.csv when l2_source == R85 (the twin L2R arm of the return),
else from l2_reference_values.csv (l2_source R81, R82A, R83). Collapse = gt_max_mm > 500. d_ate_mm = layer-0 gt_ate_mm minus layer-2 gt_ate_mm (mm).
Native units (mm, counts of cells); no ratios, no percentages, no noise bars or confidence intervals.

OUT r85_l0_rows.csv, one row per L0 arm (288):
  arm, bag, overlap (arm_table mode: dc discard | rt retime_trim), fit (pca|deb), lines (off|on), q (fq|mq, from the arm name), edge (voxel edge, m),
  gt_ate_mm, gt_max_mm, collapsed (layer-0 arm), l2_ref (layer-2 reference arm), l2_source (R81|R82A|R83|R85), l2_gt_ate_mm, l2_gt_max_mm, l2_collapsed,
  d_ate_mm (blank when either arm is missing)
OUT r85_l0_collapse_counts.csv, one row per (overlap, fit, lines, q, edge):
  n_arms, n_collapsed_layer0, n_collapsed_layer2, n_missing (bags where the layer-0 or the layer-2 arm has no value), n_missing_layer0, n_missing_layer2,
  collapsed_by_bag = one token per bag `<bag>:<a><b>`, a = layer 0, b = layer 2, each 1 collapsed, 0 not collapsed, ? missing
OUT r85_l0_paired.csv, one row per (overlap, fit, lines, q, edge); d_ate_mm as above:
  n_pairs, n_layer0_lower (d<0), n_layer0_higher (d>0), n_equal (d=0 at 0.001 mm), median_d_ate_mm   -- over the bags where neither arm collapsed
  n_pairs_all, n_layer0_lower_all, n_layer0_higher_all, n_equal_all, median_d_ate_mm_all   -- the same over every bag with both arms present
OUT r85_l0_unified.csv, one row per (arm type, edge, layer); arm type = overlap (rt|dc) x fit (pca|deb) x lines (off|on) x Q (fq|mq); layer 0 = the L0 arms, layer 2 = their references:
  arm_type, overlap, fit, lines, q, edge, layer, n_cells (bags in the table), n_with_value, n_missing, n_collapsed (gt_max_mm > 500), n_noncollapsed,
  median_gt_ate_noncollapsed_mm (median over the non-collapsed cells with a value; blank when none), median_gt_ate_all_mm (median over every cell with a value),
  worst_gt_ate_mm (largest gt_ate_mm of any cell with a value), worst_bag, source (R85 = this return and the reference values of the manifest; R84_all_arms = joined rows).
  With --r84-all-arms <csv> the same rows for the edges 0.30 to 0.70 m are added from that file (R84 values, not re-run), so one table spans 0.20 to 0.70 m. The file is accepted only if
  its header lets this script identify the bag, overlap, fit, lines, Q and edge of each cell (by columns or by a parseable R84 arm name) and the gt_ate_mm and gt_max_mm values (layer 0
  and/or layer 2 columns, or a layer column); otherwise one line `NOT JOINED: <reason>` is printed and written to r85_l0_notes.txt and the run continues without the join.
OUT r85_l0_missing.txt, one line per arm (layer-0 arm in phase_summary, layer-2 reference in phase_summary or l2_reference_values) without a usable gt_ate_mm/gt_max_mm.
Exit 1 when anything is missing (tables are still written). r85_l0_notes.txt (join result) does not change the exit code."""
import argparse
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python3 -I drops the script dir from sys.path
import r85_common as C  # noqa: E402

ROWS_HDR = ["arm", "bag", "overlap", "fit", "lines", "q", "edge", "gt_ate_mm", "gt_max_mm", "collapsed", "l2_ref", "l2_source", "l2_gt_ate_mm", "l2_gt_max_mm",
            "l2_collapsed", "d_ate_mm"]
KEY = ("overlap", "fit", "lines", "q")
UNIFIED_HDR = ["arm_type", "overlap", "fit", "lines", "q", "edge", "layer", "n_cells", "n_with_value", "n_missing", "n_collapsed", "n_noncollapsed",
               "median_gt_ate_noncollapsed_mm", "median_gt_ate_all_mm", "worst_gt_ate_mm", "worst_bag", "source"]
EDGE_LO, EDGE_HI = 0.30, 0.70   # edges taken from the R84 file (0.20 and 0.25 come from the R85 arms)


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
        if a["l2_source"] == "R85":
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


def arm_type(o, f, l, q):
    return "%s_%s_lines-%s_%s" % (o, f, l, q)


def unified_rows(entries):
    """entries: dicts overlap, fit, lines, q, edge, layer, bag, ate, gmax, source (ate / gmax None when missing) -> rows of r85_l0_unified.csv, sorted."""
    g = {}
    for e in entries:
        g.setdefault((e["overlap"], e["fit"], e["lines"], e["q"], e["edge"], e["layer"]), []).append(e)
    out = []
    for k, es in sorted(g.items()):
        es = sorted(es, key=lambda e: e["bag"])
        ok = [e for e in es if e["ate"] is not None and e["gmax"] is not None]
        nc = [e for e in ok if not C.collapsed_of(e["gmax"])]
        worst = max(ok, key=lambda e: (e["ate"], e["bag"])) if ok else None
        srcs = sorted({e["source"] for e in es})
        out.append([arm_type(*k[:4]), k[0], k[1], k[2], k[3], C.edge_s(k[4]), k[5], len(es), len(ok), len(es) - len(ok), len(ok) - len(nc), len(nc),
                    C.f3(C.median([e["ate"] for e in nc])), C.f3(C.median([e["ate"] for e in ok])), C.f3(worst["ate"] if worst else None), worst["bag"] if worst else "", "+".join(srcs)])
    return out


def entries_of(rows):
    out = []
    for r in rows:
        for layer, ate, gmax in ((0, r["ate"], r["gmax"]), (2, r["l2ate"], r["l2gmax"])):
            ok = ate is not None and gmax is not None
            out.append(dict(overlap=r["overlap"], fit=r["fit"], lines=r["lines"], q=r["q"], edge=r["edge"], layer=layer, bag=r["bag"],
                            ate=ate if ok else None, gmax=gmax if ok else None, source="R85"))
    return out


def reduce(arms, phase, refs):
    missing = []
    rows = build_rows(arms, phase, refs, missing)
    out_rows = [[r["arm"], r["bag"], r["overlap"], r["fit"], r["lines"], r["q"], C.edge_s(r["edge"]), C.f3(r["ate"]), C.f3(r["gmax"]), C.bflag(r["col"]), r["l2_ref"],
                 r["l2_source"], C.f3(r["l2ate"]), C.f3(r["l2gmax"]), C.bflag(r["l2col"]), C.f3(r["d"])] for r in rows]
    counts, paired = [], []
    for k, g in cells(rows).items():
        g = sorted(g, key=lambda r: r["bag"])
        m0 = sum(r["ate"] is None for r in g)
        m2 = sum(r["l2ate"] is None for r in g)
        counts.append(list(k[:4]) + [C.edge_s(k[4]), len(g), sum(r["col"] is True for r in g), sum(r["l2col"] is True for r in g),
                                     sum(r["ate"] is None or r["l2ate"] is None for r in g), m0, m2,
                                     " ".join("%s:%s%s" % (r["bag"], tri(r["col"]), tri(r["l2col"])) for r in g)])
        both = [r for r in g if r["d"] is not None]
        nc = [r for r in both if r["col"] is False and r["l2col"] is False]

        def stats(s):
            ds = [r["d"] for r in s]
            return [len(ds), sum(d < 0 for d in ds), sum(d > 0 for d in ds), sum(d == 0 for d in ds), C.f3(C.median(ds))]
        paired.append(list(k[:4]) + [C.edge_s(k[4])] + stats(nc) + stats(both))
    return out_rows, counts, paired, rows, missing


COUNTS_HDR = list(KEY) + ["edge", "n_arms", "n_collapsed_layer0", "n_collapsed_layer2", "n_missing", "n_missing_layer0", "n_missing_layer2", "collapsed_by_bag"]
_ST = ["n_pairs", "n_layer0_lower", "n_layer0_higher", "n_equal", "median_d_ate_mm"]
PAIRED_HDR = list(KEY) + ["edge"] + _ST + [s + "_all" if s != "n_pairs" else "n_pairs_all" for s in _ST]


# ---------------------------------------------------------------- the R84 all-arms file (header unknown to the author: identified by alias, else NOT JOINED)
ALIAS = {
    "bag": ["bag", "seq"], "overlap": ["overlap", "ovl", "overlap_mode", "mode"], "fit": ["fit", "plane_fit", "fit_mode"], "lines": ["lines", "line", "line_residual"],
    "q": ["q", "qmode", "q_mode"], "edge": ["edge", "edge_m", "voxel_edge", "voxel_size", "voxel_size_m"], "layer": ["layer", "max_layer"], "arm": ["arm", "l0_arm", "arm_l0"],
    "ate0": ["gt_ate_mm", "l0_gt_ate_mm", "layer0_gt_ate_mm", "l0_ate_mm", "layer0_ate_mm", "ate_l0_mm", "ate_layer0_mm"],
    "max0": ["gt_max_mm", "l0_gt_max_mm", "layer0_gt_max_mm", "l0_max_mm", "layer0_max_mm", "max_l0_mm", "max_layer0_mm"],
    "ate2": ["l2_gt_ate_mm", "layer2_gt_ate_mm", "l2_ate_mm", "layer2_ate_mm", "ate_l2_mm", "ate_layer2_mm"],
    "max2": ["l2_gt_max_mm", "layer2_gt_max_mm", "l2_max_mm", "layer2_max_mm", "max_l2_mm", "max_layer2_mm"],
}
ARM_RE = re.compile(r"^r\d+[a-z]?_(?P<overlap>rt|dc)_[a-z0-9]+_(?P<bag>[a-z]+_\d\d)_(?P<fit>pca|deb)_(?P<lines>on|off)_v(?P<edge>[0-9p]+)_(?P<q>fq|mq)$")
NORM = {"overlap": {"rt": "rt", "retime_trim": "rt", "dc": "dc", "discard": "dc"}, "fit": {"pca": "pca", "deb": "deb", "debiased": "deb"},
        "lines": {"on": "on", "1": "on", "true": "on", "hardabs": "on", "off": "off", "0": "off", "false": "off"}, "q": {"fq": "fq", "mq": "mq"}}


def read_r84_all_arms(path):
    """-> (entries or None, note). entries as in unified_rows (source R84_all_arms), edges EDGE_LO..EDGE_HI only. None and a reason when the file cannot be identified."""
    if not os.path.exists(path):
        return None, "file not found: %s" % path
    try:
        rows = C.read_rows(path)
    except Exception as e:   # unreadable / not a CSV
        return None, "cannot read %s as csv (%s)" % (path, str(e)[:80])
    if not rows:
        return None, "%s has no data rows" % path
    hdr = [h for h in rows[0].keys() if h]
    low = {h.strip().lower(): h for h in hdr}
    col = {k: next((low[a] for a in al if a in low), None) for k, al in ALIAS.items()}
    need = ["bag", "overlap", "fit", "lines", "q", "edge"]   # each: a column, else the parsed arm name
    can_parse = col["arm"] is not None and all(ARM_RE.match(r.get(col["arm"], "") or "") for r in rows[:50])
    miss = [k for k in need if col[k] is None and not can_parse]
    if miss:
        return None, "cannot identify %s in the header (%s)" % (", ".join(miss), ",".join(hdr))
    long_fmt = col["layer"] is not None
    if long_fmt:
        if col["ate0"] is None or col["max0"] is None:
            return None, "a layer column is present but gt_ate_mm and gt_max_mm columns are not (%s)" % ",".join(hdr)
        value_cols = None
    else:
        have0, have2 = col["ate0"] and col["max0"], col["ate2"] and col["max2"]
        if not (have0 or have2):
            return None, "no gt_ate_mm / gt_max_mm column pair for layer 0 or layer 2 (%s)" % ",".join(hdr)
        value_cols = ([(0, col["ate0"], col["max0"])] if have0 else []) + ([(2, col["ate2"], col["max2"])] if have2 else [])
    out, seen, skipped_edge, skipped_layer, bad, dups = [], set(), 0, 0, 0, 0
    for r in rows:
        d = {}
        if can_parse:
            m = ARM_RE.match(r.get(col["arm"], "") or "")
            d = m.groupdict() if m else {}
            if d:
                d["edge"] = d["edge"].replace("p", ".")
        for k in need:
            if col[k] is not None and (r.get(col[k]) not in (None, "")):
                d[k] = r[col[k]].strip()
        if any(k not in d for k in need):
            bad += 1
            continue
        try:
            for k in ("overlap", "fit", "lines", "q"):
                d[k] = NORM[k][d[k].lower()]
            edge = float(d["edge"])
        except (KeyError, ValueError):
            bad += 1
            continue
        if not (EDGE_LO - 1e-9 <= edge <= EDGE_HI + 1e-9):
            skipped_edge += 1
            continue
        if long_fmt:
            lay = C.num(r.get(col["layer"]))
            if lay not in (0.0, 2.0):
                skipped_layer += 1
                continue
            vals = [(int(lay), col["ate0"], col["max0"])]
        else:
            vals = value_cols
        for lay, ca, cm in vals:
            key = (d["overlap"], d["fit"], d["lines"], d["q"], edge, lay, d["bag"])
            if key in seen:
                dups += 1
                continue
            seen.add(key)
            ate, gmax = C.num(r.get(ca)), C.num(r.get(cm))
            ok = ate is not None and gmax is not None
            out.append(dict(overlap=d["overlap"], fit=d["fit"], lines=d["lines"], q=d["q"], edge=edge, layer=lay, bag=d["bag"],
                            ate=ate if ok else None, gmax=gmax if ok else None, source="R84_all_arms"))
    if not out:
        return None, "no row of %s could be used (%d with an unrecognised descriptor value, %d outside %.2f-%.2f m, %d with a layer other than 0 or 2)" % (path, bad, skipped_edge, EDGE_LO, EDGE_HI, skipped_layer)
    ncell = len({(e["overlap"], e["fit"], e["lines"], e["q"], e["edge"], e["bag"]) for e in out})
    return out, "%d cells (%d layer values) from %s at edges %s m; %d rows outside %.2f-%.2f m skipped, %d unusable rows, %d duplicate keys ignored, layers present: %s" % (
        ncell, len(out), os.path.basename(path), ",".join(C.edge_s(x) for x in sorted({e["edge"] for e in out})), skipped_edge, EDGE_LO, EDGE_HI, bad, dups,
        ",".join(str(x) for x in sorted({e["layer"] for e in out})))


def run(phase_path, manifest, out, r84_all=None):
    os.makedirs(out, exist_ok=True)
    arms = C.load_arm_table(manifest, "L0")
    if not arms:
        print("no L0 arms in %s/arm_table.tsv" % manifest, file=sys.stderr)
        return 2
    o = reduce(arms, C.load_phase(phase_path), C.load_refs(manifest))
    notes = []
    ent = entries_of(o[3])
    if r84_all:
        e84, note = read_r84_all_arms(r84_all)
        if e84 is None:
            line = "NOT JOINED: " + note
        else:
            ent += e84
            line = "JOINED: " + note
        print(line)
        notes.append(line)
    else:
        notes.append("--r84-all-arms not given: r85_l0_unified.csv holds the edges 0.20 and 0.25 m only")
    C.write_csv(os.path.join(out, "r85_l0_rows.csv"), ROWS_HDR, o[0])
    C.write_csv(os.path.join(out, "r85_l0_collapse_counts.csv"), COUNTS_HDR, o[1])
    C.write_csv(os.path.join(out, "r85_l0_paired.csv"), PAIRED_HDR, o[2])
    C.write_csv(os.path.join(out, "r85_l0_unified.csv"), UNIFIED_HDR, unified_rows(ent))
    with open(os.path.join(out, "r85_l0_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("L0 arms: %d, missing entries: %d" % (len(arms), len(o[4])))
    return C.write_missing(os.path.join(out, "r85_l0_missing.txt"), o[4])


# ---------------------------------------------------------------- selftest (SYNTHETIC fixture built here, 2 bags; not data from any run)
def selftest():
    c = C.Checker()
    d = tempfile.mkdtemp()
    man = os.path.join(d, "man")
    os.makedirs(man)
    cols = ["arm", "group", "bag", "mode", "fit", "lines", "edge", "l2_ref", "l2_source"]
    at, refs, ph = [], [], []
    # 2 bags x edges 0.2, 0.22 (synthetic third edge), 0.25 ; rt pca off fq ; edges 0.2, 0.22 reference R81 (values file), 0.25 reference R85 (phase row)
    spec = {  # (bag, edge): (l0 ate, l0 max, l2 ate, l2 max)
        ("eee_01", 0.2): (30.0, 100.0, 40.0, 100.0),    # d = -10
        ("eee_01", 0.22): (50.0, 600.0, 45.0, 100.0),   # layer 0 collapsed, d = +5 (all-bags only)
        ("eee_01", 0.25): (20.0, 100.0, 20.0, 100.0),   # equal, R85 ref
        ("nya_01", 0.2): (60.0, 100.0, 55.0, 100.0),    # d = +5
        ("nya_01", 0.22): (None, None, 10.0, 100.0),    # layer-0 arm missing
        ("nya_01", 0.25): (70.0, 100.0, None, None),    # R85 reference arm missing from phase_summary
    }
    for (bag, e), (a0, m0, a2, m2) in spec.items():
        tag = ("%g" % e).replace(".", "p")
        arm = "r85_rt_l0_%s_pca_off_v%s_fq" % (bag, tag)
        if e < 0.25:
            ref, src = "r81_rt_sw_%s_off_v%s_pfn3_fq" % (bag, tag), "R81"
            if a2 is not None:
                refs.append([ref, "R81", bag, "rt", "pca", "off", e, "fq", a2, m2, m2 > 500])
        else:
            ref, src = "r85_rt_l2_%s_pca_off_v%s_fq" % (bag, tag), "R85"
            if a2 is not None:
                ph.append([ref, a2, m2])
        at.append([arm, "L0", bag, "rt", "pca", "off", e, ref, src])
        if a0 is not None:
            ph.append([arm, a0, m0])
    # decoys: a phase row for a reference arm that must NOT be used for an R81 reference, a values-file row for the R85 one that must NOT be used; an L2R row in the arm table that is not an L0 arm
    ph.append(["r81_rt_sw_eee_01_off_v0p2_pfn3_fq", 999.0, 10.0])
    refs.append(["r85_rt_l2_eee_01_pca_off_v0p25_fq", "R85", "eee_01", "rt", "pca", "off", 0.25, "fq", 888.0, 10.0, False])
    at.append(["r85_rt_l2_eee_01_pca_off_v0p25_fq", "L2R", "eee_01", "rt", "pca", "off", 0.25, "", "R85"])
    with open(os.path.join(man, "arm_table.tsv"), "w") as f:
        f.write("\n".join("\t".join(map(str, r)) for r in [cols] + at) + "\n")
    with open(os.path.join(man, "l2_reference_values.csv"), "w") as f:
        f.write("arm,source,bag,ovl,fit,lines,edge,q,gt_ate_mm,gt_max_mm,collapsed\n" + "\n".join(",".join(map(str, r)) for r in refs) + "\n")
    pp = os.path.join(d, "phase_summary.csv")
    with open(pp, "w") as f:
        f.write("arm,gt_ate_mm,gt_max_mm,exit_code\n" + "\n".join("%s,%s,%s,0" % tuple(r) for r in ph) + "\n")
    out = os.path.join(d, "out")
    rc = run(pp, man, out)
    R = {r["arm"]: r for r in C.read_rows(os.path.join(out, "r85_l0_rows.csv"))}
    c("rows header exact", open(os.path.join(out, "r85_l0_rows.csv")).readline().strip() == ",".join(ROWS_HDR))
    c("6 rows written (the L2R table row is not an L0 arm), 2 entries missing, exit 1", len(R) == 6 and rc == 1)
    a = R["r85_rt_l0_eee_01_pca_off_v0p2_fq"]
    c("d_ate sign: layer0 30 minus layer2 40 = -10.000", a["d_ate_mm"] == "-10.000", a["d_ate_mm"])
    c("R81 reference taken from l2_reference_values, not the phase decoy", a["l2_gt_ate_mm"] == "40.000" and a["l2_source"] == "R81")
    a = R["r85_rt_l0_eee_01_pca_off_v0p25_fq"]
    c("R85 reference taken from phase_summary, not the values decoy", a["l2_gt_ate_mm"] == "20.000" and a["d_ate_mm"] == "0.000")
    a = R["r85_rt_l0_eee_01_pca_off_v0p22_fq"]
    c("collapsed = gt_max > 500 (layer 0 yes, layer 2 no)", a["collapsed"] == "True" and a["l2_collapsed"] == "False")
    a = R["r85_rt_l0_nya_01_pca_off_v0p22_fq"]
    c("missing layer-0 arm: blank values, blank d", a["gt_ate_mm"] == "" and a["collapsed"] == "" and a["d_ate_mm"] == "" and a["l2_gt_ate_mm"] == "10.000")
    a = R["r85_rt_l0_nya_01_pca_off_v0p25_fq"]
    c("missing R85 reference: blank l2, blank d", a["l2_gt_ate_mm"] == "" and a["l2_collapsed"] == "" and a["d_ate_mm"] == "")
    mt = open(os.path.join(out, "r85_l0_missing.txt")).read().splitlines()
    c("missing list names both arms", len(mt) == 2 and any("nya_01_pca_off_v0p22_fq" in x for x in mt) and any("r85_rt_l2_nya_01_pca_off_v0p25_fq" in x for x in mt), str(mt))
    CC = {r["edge"]: r for r in C.read_rows(os.path.join(out, "r85_l0_collapse_counts.csv"))}
    c("collapse counts edge 0.22", CC["0.22"]["n_arms"] == "2" and CC["0.22"]["n_collapsed_layer0"] == "1" and CC["0.22"]["n_collapsed_layer2"] == "0"
      and CC["0.22"]["n_missing"] == "1" and CC["0.22"]["n_missing_layer0"] == "1", str(CC["0.22"]))
    c("per-bag flag string", CC["0.22"]["collapsed_by_bag"] == "eee_01:10 nya_01:?0" and CC["0.25"]["collapsed_by_bag"] == "eee_01:00 nya_01:0?", CC["0.22"]["collapsed_by_bag"])
    P = {r["edge"]: r for r in C.read_rows(os.path.join(out, "r85_l0_paired.csv"))}
    c("paired 0.2: two pairs, one lower, one higher, median -2.5", (P["0.2"]["n_pairs"], P["0.2"]["n_layer0_lower"], P["0.2"]["n_layer0_higher"], P["0.2"]["median_d_ate_mm"])
      == ("2", "1", "1", "-2.500"), str(P["0.2"]))
    c("paired 0.22: non-collapsed set empty, all-bags set has 1 higher", P["0.22"]["n_pairs"] == "0" and P["0.22"]["median_d_ate_mm"] == ""
      and P["0.22"]["n_pairs_all"] == "1" and P["0.22"]["n_layer0_higher_all"] == "1" and P["0.22"]["median_d_ate_mm_all"] == "5.000", str(P["0.22"]))
    c("paired 0.25: one equal pair", P["0.25"]["n_equal"] == "1" and P["0.25"]["n_pairs"] == "1")
    # unified table (no R84 file)
    U = {(r["edge"], r["layer"]): r for r in C.read_rows(os.path.join(out, "r85_l0_unified.csv"))}
    c("unified header exact", open(os.path.join(out, "r85_l0_unified.csv")).readline().strip() == ",".join(UNIFIED_HDR))
    u = U[("0.2", "0")]
    c("unified 0.2 layer 0: both bags non-collapsed, median of 30 and 60 = 45.000, worst 60 at nya_01", (u["arm_type"], u["n_cells"], u["n_with_value"], u["n_collapsed"], u["median_gt_ate_noncollapsed_mm"], u["median_gt_ate_all_mm"], u["worst_gt_ate_mm"], u["worst_bag"], u["source"])
      == ("rt_pca_lines-off_fq", "2", "2", "0", "45.000", "45.000", "60.000", "nya_01", "R85"), str(u))
    u = U[("0.22", "0")]
    c("unified 0.22 layer 0: one value (collapsed 50), one missing: non-collapsed median blank, all-median 50", (u["n_with_value"], u["n_missing"], u["n_collapsed"], u["n_noncollapsed"], u["median_gt_ate_noncollapsed_mm"], u["median_gt_ate_all_mm"], u["worst_gt_ate_mm"])
      == ("1", "1", "1", "0", "", "50.000", "50.000"), str(u))
    u = U[("0.2", "2")]
    c("unified 0.2 layer 2: values 40 and 55 from the values file, median 47.500", (u["median_gt_ate_noncollapsed_mm"], u["worst_gt_ate_mm"], u["worst_bag"]) == ("47.500", "55.000", "nya_01"), str(u))
    c("unified 0.25 layer 2: the R85 reference of nya_01 is missing, eee_01 value 20 (phase row)", (U[("0.25", "2")]["n_with_value"], U[("0.25", "2")]["median_gt_ate_all_mm"]) == ("1", "20.000"))
    c("unified: 3 edges x 2 layers, no R84 rows without the option", len(U) == 6 and not any("R84" in r["source"] for r in U.values()))
    # R84 all-arms file: wide format with the R84 T3 row columns
    wide = os.path.join(d, "r84_wide.csv")
    with open(wide, "w") as f:
        f.write("arm,bag,overlap,fit,lines,q,edge,gt_ate_mm,gt_max_mm,collapsed,l2_ref,l2_source,l2_gt_ate_mm,l2_gt_max_mm,l2_collapsed,d_ate_mm\n")
        f.write("r84_rt_t3_eee_01_pca_off_v0p3_fq,eee_01,rt,pca,off,fq,0.3,10,100,False,x,R81,12,100,False,-2\n")
        f.write("r84_rt_t3_nya_01_pca_off_v0p3_fq,nya_01,rt,pca,off,fq,0.3,20,700,True,x,R81,30,100,False,-10\n")
        f.write("r84_rt_t3_eee_01_pca_off_v0p25_fq,eee_01,rt,pca,off,fq,0.25,1,1,False,x,R81,1,1,False,0\n")   # 0.25 is an R85 edge: skipped
        f.write("r84_rt_t3_eee_01_pca_off_v0p7_fq,eee_01,rt,pca,off,fq,0.7,5,100,False,x,R81,6,100,False,-1\n")
    out2 = os.path.join(d, "out2")
    run(pp, man, out2, wide)
    k = {(r["edge"], r["layer"]): r for r in C.read_rows(os.path.join(out2, "r85_l0_unified.csv"))}
    c("R84 join (wide): edges 0.3 and 0.7 added at layers 0 and 2, 0.25 not duplicated", set(k) == set(U) | {("0.3", "0"), ("0.3", "2"), ("0.7", "0"), ("0.7", "2")} and k[("0.25", "0")]["source"] == "R85", str(sorted(k)))
    u = k[("0.3", "0")]
    c("R84 join: 0.3 layer 0: values 10 and 20 (20 collapsed): non-collapsed median 10, all-median 15, one collapsed, worst 20 at nya_01, source R84_all_arms",
      (u["n_cells"], u["n_collapsed"], u["median_gt_ate_noncollapsed_mm"], u["median_gt_ate_all_mm"], u["worst_gt_ate_mm"], u["worst_bag"], u["source"]) == ("2", "1", "10.000", "15.000", "20.000", "nya_01", "R84_all_arms"), str(u))
    c("R84 join: 0.3 layer 2: 12 and 30, median 21", (k[("0.3", "2")]["median_gt_ate_all_mm"], k[("0.3", "2")]["n_collapsed"]) == ("21.000", "0"))
    c("notes file says JOINED", open(os.path.join(out2, "r85_l0_notes.txt")).read().startswith("JOINED:"))
    # R84 file in long format (layer column), descriptors only in the arm name
    lng = os.path.join(d, "r84_long.csv")
    with open(lng, "w") as f:
        f.write("arm,layer,gt_ate_mm,gt_max_mm\n")
        f.write("r84_dc_t3_eee_01_deb_on_v0p5_mq,0,8,50\nr84_dc_t3_eee_01_deb_on_v0p5_mq,2,9,50\nr84_dc_t3_eee_01_deb_on_v0p5_mq,3,1,1\n")
    out3 = os.path.join(d, "out3")
    run(pp, man, out3, lng)
    k3 = {(r["arm_type"], r["edge"], r["layer"]): r for r in C.read_rows(os.path.join(out3, "r85_l0_unified.csv"))}
    c("R84 join (long, arm-name descriptors): dc_deb_lines-on_mq at 0.5 layers 0 and 2; layer 3 ignored", k3[("dc_deb_lines-on_mq", "0.5", "0")]["median_gt_ate_all_mm"] == "8.000" and k3[("dc_deb_lines-on_mq", "0.5", "2")]["median_gt_ate_all_mm"] == "9.000" and ("dc_deb_lines-on_mq", "0.5", "3") not in k3)
    # unidentifiable header: NOT JOINED, run continues, unified identical to the no-file result
    bad = os.path.join(d, "r84_bad.csv")
    with open(bad, "w") as f:
        f.write("foo,bar,baz\n1,2,3\n")
    out4 = os.path.join(d, "out4")
    run(pp, man, out4, bad)
    c("unidentifiable header: NOT JOINED line in the notes, unified table unchanged", open(os.path.join(out4, "r85_l0_notes.txt")).read().startswith("NOT JOINED:")
      and open(os.path.join(out4, "r85_l0_unified.csv")).read() == open(os.path.join(out, "r85_l0_unified.csv")).read())
    out5 = os.path.join(d, "out5")
    run(pp, man, out5, os.path.join(d, "does_not_exist.csv"))
    c("missing file: NOT JOINED, run continues", open(os.path.join(out5, "r85_l0_notes.txt")).read().startswith("NOT JOINED: file not found"))
    # all collapsed -> blank non-collapsed median
    arms = [dict(arm="x_fq", bag="b", mode="rt", fit="pca", lines="off", edge="0.2", l2_ref="r", l2_source="R81")]
    o = reduce(arms, {"x_fq": dict(gt_ate_mm=5.0, gt_max_mm=900.0)}, {"r": dict(gt_ate_mm=6.0, gt_max_mm=900.0)})
    ur = unified_rows(entries_of(o[3]))
    c("always collapsed: non-collapsed median blank, all-median present, counts", ur[0][12] == "" and ur[0][13] == "5.000" and ur[0][10] == 1 and ur[0][11] == 0 and not o[4])
    return c.done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-summary")
    ap.add_argument("--manifest")
    ap.add_argument("--out")
    ap.add_argument("--r84-all-arms", default=None, help="r84_layer0_vs_layer2_all_arms.csv; joined only if its header can be identified, else NOT JOINED and the run continues")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.phase_summary and a.manifest and a.out):
        ap.error("--phase-summary, --manifest and --out are required")
    return run(a.phase_summary, a.manifest, a.out, a.r84_all_arms)


if __name__ == "__main__":
    sys.exit(main())
