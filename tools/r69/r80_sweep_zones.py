#!/usr/bin/env python3
"""r80_sweep_zones.py (planning agent, R80, design D1 / F-463 / F-466; host-side scoring, not built or run on real data by me).

Approved design text (D1, verbatim): "Arms. Edges 0.20 to 1.00 m in 0.05 steps (17 sizes), both Q, lines off and hard absolute, pfn 3 and pfn 1, all nine bags, whole bag,
max_layer 2. Scoring. ATE per arm. Per bag and per cell (Q x line x pfn), the zone is every edge within 5 mm of that cell's best non-collapsed ATE. Collapsed arms are
listed and not ranked. Refinement: the refine step (finer edges inside the zones) is a second round after the zones are read. Return: SUMMARIES ONLY."

Usage: python3 r80_sweep_zones.py --return-dir <R80_RETURN> --out <dir> [--tol-mm 5.0] [--ref-return-dir <dir>] [--ref-arm-template ...]
Inputs (from the return directory; real names read from r69_collect.py flat() and r78_analysis.py / r79_load.py):
  phase_summary.csv   one row per arm; columns used: arm, gt_ate_mm (ATE after one global rigid alignment, whole bag), gt_max_mm (largest sampled per-scan error vs ground truth),
                      excursion_mm (max ||e_p|| over scans 1..10), st_lio_nees9_mean (stationary NEES9, 9 = consistent), settled_offset_mm, onset_scan_used (the bag's onset scan).
  compact/<arm>/pos_trace_all.csv.gz   scan_id, px, py, pz, yaw_rad (...)  -- used for (a) yaw movement and (b) the departure of the collapsed rule.
Arm id: r80_sweep id  r80_sw_<bag>_<line>_v<tag>_pfn<p>_<q>  (line off|hardabs; tag = edge with '.' -> 'p'; p 1|3; q fq|mq). Cell = (bag, q, line, pfn).

DEFINITIONS (all from the project; nothing invented here)
 ATE      = gt_ate_mm of phase_summary.csv (pipeline rms position error against ground truth after one global rigid alignment, over the scans that join ground truth).
 Excursion = excursion_mm (largest distance of the estimated position from scan 1 over scans 1 to 10, mm).
 Yaw      = YAW MOVEMENT (r78_analysis.yawmove): max |wrapped(yaw(s) - yaw(1))| over scans 1..10, mrad, from pos_trace_all.csv.gz. phase_summary.csv has NO yaw error column
            and no yaw-vs-ground-truth number exists in it; per-frame yaw error vs ground truth (F-439) is a separate tool. Column name here: yaw_move_mrad.
 NEES     = st_lio_nees9_mean (r78_analysis.metrics).
 COLLAPSED (project standing definition, rule 4.15 c, claude/r76-return-audit.md Definitions; restated claude/r77-return-audit.md; implemented in claude/scripts/r78_analysis.py
            metrics() and r79_tables.py coll()): "the estimated position departs by 500 mm or more from the healthy reference run (0.425 m, every 3rd point, lines off, same
            process noise) between scans 485 and 700 (alignment-free, start offset removed) AND the largest ground-truth error exceeds 500 mm."
            Implementation: departure as r79_load.departure(): over the common scans in [onset, onset+215] (eee_01: 485..700), max norm of
            (arm pos - arm pos at first common scan) - (ref pos - ref pos at first common scan), mm. collapsed = departure >= 500 AND gt_max_mm > 500.
            Reference arm = the same bag, same Q, 0.425 m, pfn 3, lines off arm; its name comes from --ref-arm-template (default the R79 whole-bag name).
            Short-circuit: gt_max_mm <= 500 => not collapsed whatever the departure (the rule is an AND). gt_max_mm > 500 and the departure cannot be computed (trace of the
            arm or of the reference absent) => collapsed = UNDETERMINED: listed, not ranked, counted in n_undetermined, never treated as healthy.
 ZONE     = every edge whose ATE <= best non-collapsed ATE of the cell + tol (inclusive; tol CLI default 5.0 mm, the user's value). Ties for best: the smaller edge is best_edge.
Missing arms (no row, or no gt_ate_mm) are reported MISSING per cell and in r80_sweep_all.csv; a cell with fewer than 17 edges is marked incomplete and scored over the rest.
Outputs: r80_sweep_zones.csv, r80_sweep_all.csv, r80_sweep_collapsed.txt, r80_sweep_ate_by_edge.csv (wide, 17 edge columns), r80_sweep_missing.txt, r80_sweep_provenance.txt."""
import argparse, os, re, sys
import numpy as np
import pandas as pd

BAGS = ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]
LINES = ["off", "hardabs"]
PFNS = [1, 3]
QS = ["fq", "mq"]
EDGES_CM = list(range(20, 101, 5))          # hundredths of a metre: 0.20 .. 1.00 step 0.05 (17 sizes)
COLLAPSE_DEP_MM = 500.0                      # project rule 4.15 c
COLLAPSE_GT_MM = 500.0                       # project rule 4.15 c
DEP_WINDOW_SCANS = 215                       # r79_load / r79_tables: onset .. onset+215 (eee_01 485..700)
DEFAULT_REF = "r79_nb_{bag}_off_v0p425_pfn3_{q}"
ARM_RE = re.compile(r"^r80_sw_(?P<bag>[a-z]{3}_\d\d)_(?P<line>off|hardabs)_v(?P<tag>\d+p\d+)_pfn(?P<pfn>[13])_(?P<q>fq|mq)$")


def edge_tag(cm):
    s = "%d.%02d" % (cm // 100, cm % 100)
    s = s.rstrip("0")
    if s.endswith("."):
        s += "0"
    return s.replace(".", "p")


def arm_id(bag, line, cm, pfn, q):
    return "r80_sw_%s_%s_v%s_pfn%d_%s" % (bag, line, edge_tag(cm), pfn, q)


def expected_arms():
    return [(arm_id(b, l, cm, p, q), b, q, l, p, cm) for b in BAGS for q in QS for l in LINES for p in PFNS for cm in EDGES_CM]


def read_pos(root, arm):
    fp = os.path.join(root, "compact", arm, "pos_trace_all.csv.gz")
    if not os.path.exists(fp):
        return None
    return pd.read_csv(fp).drop_duplicates("scan_id").set_index("scan_id").sort_index()


def yaw_move_mrad(T):
    if T is None or "yaw_rad" not in T:
        return np.nan
    y = T.loc[(T.index >= 1) & (T.index <= 10), "yaw_rad"].to_numpy()
    if len(y) == 0:
        return np.nan
    return float(np.abs((y - y[0] + np.pi) % (2 * np.pi) - np.pi).max() * 1e3)


def departure_mm(A, B, lo, hi):
    """r79_load.departure(): alignment-free, start offset removed at the first common scan of the window. NaN when there is no common scan."""
    if A is None or B is None:
        return np.nan
    c = [s for s in A.index.intersection(B.index) if lo <= s <= hi]
    if not c:
        return np.nan
    x = A[["px", "py", "pz"]]; y = B[["px", "py", "pz"]]
    d = (x.loc[c].to_numpy() - x.loc[c[0]].to_numpy()) - (y.loc[c].to_numpy() - y.loc[c[0]].to_numpy())
    return float(np.linalg.norm(d, axis=1).max() * 1e3)


def collapse_status(gt_max, dep):
    """returns 'no', 'yes' or 'undetermined' (project rule 4.15 c: departure >= 500 AND gt_max > 500)."""
    if gt_max is None or not np.isfinite(gt_max):
        return "undetermined"
    if not gt_max > COLLAPSE_GT_MM:
        return "no"
    if dep is None or not np.isfinite(dep):
        return "undetermined"
    return "yes" if dep >= COLLAPSE_DEP_MM else "no"


def num(r, k):
    v = r.get(k, np.nan) if r is not None else np.nan
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    return v


def score(P, ret_dir, ref_dir, ref_tpl, tol_mm):
    """P: phase_summary DataFrame indexed by arm (unique). Returns (all_df, zones_df, extra dict)."""
    rows = []
    ref_cache = {}
    for aid, bag, q, line, pfn, cm in expected_arms():
        r = P.loc[aid] if aid in P.index else None
        ate = num(r, "gt_ate_mm")
        row = dict(arm=aid, bag=bag, q=q, line=line, pfn=pfn, edge_m=cm / 100.0, ate_mm=ate, gt_max_mm=num(r, "gt_max_mm"), excursion_mm=num(r, "excursion_mm"),
                   yaw_move_mrad=np.nan, nees9=num(r, "st_lio_nees9_mean"), settled_offset_mm=num(r, "settled_offset_mm"), departure_mm=np.nan,
                   status="", collapsed="", in_zone="")
        if r is None:
            row["status"] = "MISSING_NO_ROW"
        elif not np.isfinite(ate):
            row["status"] = "MISSING_NO_ATE"
        else:
            T = read_pos(ret_dir, aid)
            row["yaw_move_mrad"] = yaw_move_mrad(T)
            dep = np.nan
            if np.isfinite(row["gt_max_mm"]) and row["gt_max_mm"] > COLLAPSE_GT_MM and T is not None:
                ra = ref_tpl.format(bag=bag, q=q)
                if ra not in ref_cache:
                    ref_cache[ra] = read_pos(ref_dir, ra)
                onset = num(r, "onset_scan_used")
                onset = int(onset) if np.isfinite(onset) else 485
                dep = departure_mm(T, ref_cache[ra], onset, onset + DEP_WINDOW_SCANS)
            row["departure_mm"] = dep
            cs = collapse_status(row["gt_max_mm"], dep)
            row["collapsed"] = cs
            row["status"] = {"no": "OK", "yes": "COLLAPSED", "undetermined": "COLLAPSE_UNDETERMINED"}[cs]
        rows.append(row)
    A = pd.DataFrame(rows)
    zrows = []
    inzone = {}
    for (bag, q, line, pfn), G in A.groupby(["bag", "q", "line", "pfn"], sort=False):
        G = G.sort_values("edge_m")
        ok = G[G.status == "OK"]
        miss = G[G.status.str.startswith("MISSING")]
        coll = G[G.status == "COLLAPSED"]
        und = G[G.status == "COLLAPSE_UNDETERMINED"]
        z = dict(bag=bag, q=q, line=line, pfn=pfn, best_edge="", best_ate_mm="", zone_edges="", zone_width_edges=0, n_collapsed=len(coll), n_undetermined=len(und),
                 n_present=len(G) - len(miss), n_expected=len(G), n_missing=len(miss), missing_edges=";".join("%g" % e for e in miss.edge_m),
                 incomplete=int(len(miss) > 0), zone_contiguous="")
        if len(ok):
            # deterministic: smallest ATE, ties to the smaller edge (G is sorted by edge, idxmin returns the first minimum)
            best = ok.loc[ok.ate_mm.idxmin()]
            lim = best.ate_mm + tol_mm
            zed = ok[ok.ate_mm <= lim + 1e-9]
            z.update(best_edge="%g" % best.edge_m, best_ate_mm="%.3f" % best.ate_mm, zone_edges=";".join("%g" % e for e in zed.edge_m), zone_width_edges=len(zed))
            cms = [int(round(e * 100)) for e in zed.edge_m]
            z["zone_contiguous"] = int(all(b - a_ == 5 for a_, b in zip(cms, cms[1:])) if len(cms) > 1 else 1)
            for aid in zed.arm:
                inzone[aid] = True
        zrows.append(z)
    A["in_zone"] = [("yes" if inzone.get(a) else ("no" if s == "OK" else "n/a")) for a, s in zip(A.arm, A.status)]
    Z = pd.DataFrame(zrows)
    return A, Z


def wide(A):
    cols = ["%.2f" % (cm / 100.0) for cm in EDGES_CM]
    out = []
    for (bag, q, line, pfn), G in A.groupby(["bag", "q", "line", "pfn"], sort=False):
        r = dict(bag=bag, q=q, line=line, pfn=pfn)
        for cm in EDGES_CM:
            g = G[np.isclose(G.edge_m, cm / 100.0)].iloc[0]
            if g.status.startswith("MISSING"):
                v = "MISSING"
            else:
                v = "%.3f" % g.ate_mm + {"OK": "", "COLLAPSED": "*", "COLLAPSE_UNDETERMINED": "?"}[g.status]
            r["%.2f" % (cm / 100.0)] = v
        out.append(r)
    return pd.DataFrame(out, columns=["bag", "q", "line", "pfn"] + cols)


def write_all(A, Z, out, tol_mm, args_txt, unexpected):
    os.makedirs(out, exist_ok=True)
    Z.to_csv(os.path.join(out, "r80_sweep_zones.csv"), index=False)
    A.to_csv(os.path.join(out, "r80_sweep_all.csv"), index=False, float_format="%.6g")
    wide(A).to_csv(os.path.join(out, "r80_sweep_ate_by_edge.csv"), index=False)
    C = A[A.status == "COLLAPSED"]; U = A[A.status == "COLLAPSE_UNDETERMINED"]
    L = ["COLLAPSED arms (project rule 4.15 c: departure >= %g mm from the reference run AND gt_max_mm > %g); listed, not ranked: %d" % (COLLAPSE_DEP_MM, COLLAPSE_GT_MM, len(C))]
    L += ["%s  ate_mm=%.3f gt_max_mm=%.3f departure_mm=%.3f" % (r.arm, r.ate_mm, r.gt_max_mm, r.departure_mm) for r in C.itertuples()]
    L += ["", "COLLAPSE UNDETERMINED (gt_max_mm > %g but the departure could not be computed, or gt_max_mm absent); not ranked, NOT assumed healthy: %d" % (COLLAPSE_GT_MM, len(U))]
    L += ["%s  ate_mm=%.3f gt_max_mm=%s" % (r.arm, r.ate_mm, r.gt_max_mm) for r in U.itertuples()]
    open(os.path.join(out, "r80_sweep_collapsed.txt"), "w").write("\n".join(L) + "\n")
    M = A[A.status.str.startswith("MISSING")]
    L = ["MISSING arms (expected %d, missing %d); cells with any missing arm are marked incomplete in r80_sweep_zones.csv" % (len(A), len(M))]
    for (bag, q, line, pfn), G in M.groupby(["bag", "q", "line", "pfn"], sort=False):
        L.append("MISSING %s %s %s pfn%d: %d of %d edges: %s" % (bag, q, line, pfn, len(G), len(EDGES_CM), " ".join("%g(%s)" % (e, s[8:]) for e, s in zip(G.edge_m, G.status))))
    if unexpected:
        L += ["", "UNEXPECTED r80_sw_ rows in phase_summary.csv (not in the D1 grid; not scored): %d" % len(unexpected)] + unexpected
    open(os.path.join(out, "r80_sweep_missing.txt"), "w").write("\n".join(L) + "\n")
    prov = ["r80_sweep_zones.py provenance", args_txt, "zone tolerance tol_mm = %g mm (the user's value; CLI default 5.0; inclusive: ATE <= best + tol)" % tol_mm,
            "collapsed rule: project rule 4.15 c as in the module docstring (departure >= %g mm AND gt_max_mm > %g mm; window onset..onset+%d)" % (COLLAPSE_DEP_MM, COLLAPSE_GT_MM, DEP_WINDOW_SCANS),
            "tie rule: smallest ATE, ties to the smaller edge", "expected arms %d (9 bags x 2 Q x 2 lines x 2 pfn x 17 edges)" % len(A)]
    open(os.path.join(out, "r80_sweep_provenance.txt"), "w").write("\n".join(prov) + "\n")


def run(ret_dir, out, tol_mm=5.0, ref_dir=None, ref_tpl=DEFAULT_REF):
    ps = os.path.join(ret_dir, "phase_summary.csv")
    if not os.path.exists(ps):
        raise SystemExit("phase_summary.csv not found in %s" % ret_dir)
    P = pd.read_csv(ps)
    for c in ("arm", "gt_ate_mm"):
        if c not in P.columns:
            raise SystemExit("phase_summary.csv lacks required column %s" % c)
    dup = P.arm[P.arm.duplicated()].unique()
    if len(dup):
        raise SystemExit("duplicate arm rows in phase_summary.csv: %s" % ", ".join(map(str, dup[:10])))
    P = P.set_index("arm")
    grid = {a[0] for a in expected_arms()}
    unexpected = sorted(a for a in P.index if str(a).startswith("r80_sw_") and a not in grid)
    A, Z = score(P, ret_dir, ref_dir or ret_dir, ref_tpl, tol_mm)
    write_all(A, Z, out, tol_mm, "return_dir=%s ref_return_dir=%s ref_arm_template=%s" % (ret_dir, ref_dir or ret_dir, ref_tpl), unexpected)
    return A, Z


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--return-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tol-mm", type=float, default=5.0, help="zone tolerance in mm (the user's value, default 5.0)")
    ap.add_argument("--ref-return-dir", default=None, help="return dir holding compact/<ref arm>/pos_trace_all.csv.gz (default: --return-dir)")
    ap.add_argument("--ref-arm-template", default=DEFAULT_REF)
    a = ap.parse_args()
    A, Z = run(a.return_dir, a.out, a.tol_mm, a.ref_return_dir, a.ref_arm_template)
    print("arms expected %d, missing %d, collapsed %d, undetermined %d; cells %d, incomplete %d" % (
        len(A), A.status.str.startswith("MISSING").sum(), (A.status == "COLLAPSED").sum(), (A.status == "COLLAPSE_UNDETERMINED").sum(), len(Z), Z.incomplete.sum()))


if __name__ == "__main__":
    main()
