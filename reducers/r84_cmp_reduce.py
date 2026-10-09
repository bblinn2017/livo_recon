#!/usr/bin/env python3
"""r84_cmp_reduce: F-520 comparison tier (216 arms: sample window {0,1,2,5,10,20} ms x time_based_process_noise {var_acc, none} x 18 cells), host side.
Python only (standard library); no node run.

  python3 r84_cmp_reduce.py --phase-summary <return>/phase_summary.csv --manifest r84_manifest [--engagement <return>/r84_sample_window_engagement.csv] --out <return dir>
  python3 r84_cmp_reduce.py --selftest

Inputs: phase_summary.csv (arm, gt_ate_mm, gt_max_mm); r84_manifest/arm_table.tsv (group CMP: arm, bag, sw_ms, tbpn, same_batch_control, l2_ref);
r84_manifest/l2_reference_values.csv (the R83 reference arm r83_rt_ml_<bag>_l2_<q> named in l2_ref); optional r84_sample_window_engagement.csv
(columns arm, sw_ms, tbpn, sample_window_s, nodes, point_evaluations, clamped_points, mean_trace_m2, max_trace_m2, inert_marker, not_engaged_marker).

OUT r84_cmp_rows.csv, one row per CMP arm:
  arm, bag, q (fq|mq, from the arm name), sw_ms (sample window, ms), tbpn (imu/undistort/time_based_process_noise: var_acc|none), gt_ate_mm, gt_max_mm,
  d_ate_vs_w0_vacc_mm (this arm's gt_ate_mm minus the gt_ate_mm of the same-batch control arm = the w0 / var_acc arm of the same bag and q, from phase_summary; mm),
  d_ate_vs_r83_mm (this arm minus the R83 arm in l2_ref, whose gt_ate_mm comes from l2_reference_values.csv; mm),
  point_evaluations, clamped_points, mean_trace_m2, max_trace_m2 (from the engagement file; blank where the arm has no row or the file is absent)
  A difference is blank when either arm has no gt_ate_mm. The w0 / var_acc arm compared with itself is 0.000.
OUT r84_cmp_summary.csv, one row per (sw_ms, tbpn, q) with q = fq, mq and all (both q):
  n_cells (arms with a gt_ate_mm; at most 9 per q, 18 for all), n_with_d (arms with a d_ate_vs_w0_vacc_mm), n_lower (d < 0), n_higher (d > 0), n_equal (d = 0 at 0.001 mm),
  median_d_ate_vs_w0_vacc_mm (mm, over the n_with_d arms), median_gt_ate_mm (these arms), median_w0_vacc_gt_ate_mm (their same-batch controls, same cells),
  median_d_ate_vs_r83_mm (mm), n_collapsed (arms with gt_max_mm > 500)
OUT r84_cmp_missing.txt: CMP arms (or their same-batch control / R83 reference) without a value; exit 1 when non-empty (tables are still written).
OUT r84_cmp_gaps.txt: engagement gaps, never a failure: the engagement file absent, an arm with sw_ms > 0 without an engagement row, and arms whose
  inert_marker / not_engaged_marker is non-empty (listed by name). Exit code is not affected."""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python3 -I drops the script dir from sys.path
import r84_common as C  # noqa: E402

ROWS_HDR = ["arm", "bag", "q", "sw_ms", "tbpn", "gt_ate_mm", "gt_max_mm", "d_ate_vs_w0_vacc_mm", "d_ate_vs_r83_mm", "point_evaluations", "clamped_points",
            "mean_trace_m2", "max_trace_m2"]
SUM_HDR = ["sw_ms", "tbpn", "q", "n_cells", "n_with_d", "n_lower", "n_higher", "n_equal", "median_d_ate_vs_w0_vacc_mm", "median_gt_ate_mm", "median_w0_vacc_gt_ate_mm",
           "median_d_ate_vs_r83_mm", "n_collapsed"]
ENG_COLS = ["point_evaluations", "clamped_points", "mean_trace_m2", "max_trace_m2"]


def load_engagement(path, gaps):
    if not path or not os.path.exists(path):
        gaps.append("engagement file absent: %s (columns point_evaluations, clamped_points, mean_trace_m2, max_trace_m2 left blank)" % (path or "<not given>"))
        return None
    d = {}
    for r in C.read_rows(path):
        d.setdefault(r["arm"], r)
    return d


def reduce(arms, phase, refs, eng, gaps):
    missing, rows = [], []
    for a in arms:
        arm = a["arm"]
        v = phase.get(arm)
        if v is None or v["gt_ate_mm"] is None:
            missing.append("phase_summary row (gt_ate_mm): " + arm)
            v = None
        ctl = a["same_batch_control"]
        vc = phase.get(ctl)
        if vc is None or vc["gt_ate_mm"] is None:
            missing.append("phase_summary row (same-batch control of %s): %s" % (arm, ctl))
            vc = None
        r83 = refs.get(a["l2_ref"])
        if r83 is None or r83["gt_ate_mm"] is None:
            missing.append("l2_reference_values row (R83 reference of %s): %s" % (arm, a["l2_ref"]))
            r83 = None
        ate = v and v["gt_ate_mm"]
        d0 = None if ate is None or vc is None else round(ate - vc["gt_ate_mm"], 3)
        d83 = None if ate is None or r83 is None else round(ate - r83["gt_ate_mm"], 3)
        rows.append(dict(arm=arm, bag=a["bag"], q=C.q_of(arm), sw=int(float(a["sw_ms"])), tbpn=a["tbpn"], ate=ate, gmax=v and v["gt_max_mm"], d0=d0, d83=d83,
                         ctl_ate=vc and vc["gt_ate_mm"], e=(eng or {}).get(arm)))
        if eng is not None:
            e = eng.get(arm)
            if e is None and rows[-1]["sw"] > 0:
                gaps.append("no engagement row for sw_ms>0 arm: " + arm)
            elif e is not None:
                for m in ("inert_marker", "not_engaged_marker"):
                    if (e.get(m) or "").strip() not in ("", "0", "False", "false"):
                        gaps.append("%s set: %s" % (m, arm))
    out = []
    for r in rows:
        e = r["e"] or {}
        out.append([r["arm"], r["bag"], r["q"], r["sw"], r["tbpn"], C.f3(r["ate"]), C.f3(r["gmax"]), C.f3(r["d0"]), C.f3(r["d83"])] + [(e.get(c) or "") for c in ENG_COLS])
    summ = []
    keys = sorted({(r["sw"], r["tbpn"]) for r in rows})
    for sw, tb in keys:
        for q in C.QS + ["all"]:
            g = [r for r in rows if r["sw"] == sw and r["tbpn"] == tb and (q == "all" or r["q"] == q) and r["ate"] is not None]
            gd = [r for r in g if r["d0"] is not None]
            ds = [r["d0"] for r in gd]
            summ.append([sw, tb, q, len(g), len(gd), sum(d < 0 for d in ds), sum(d > 0 for d in ds), sum(d == 0 for d in ds), C.f3(C.median(ds)),
                         C.f3(C.median([r["ate"] for r in gd])), C.f3(C.median([r["ctl_ate"] for r in gd])), C.f3(C.median([r["d83"] for r in g if r["d83"] is not None])),
                         sum(r["gmax"] is not None and r["gmax"] > C.COLLAPSE_GT_MM for r in g)])
    return out, summ, sorted(set(missing), key=missing.index)


def run(phase_path, manifest, eng_path, out):
    os.makedirs(out, exist_ok=True)
    arms = C.load_arm_table(manifest, "CMP")
    if not arms:
        print("no CMP arms in %s/arm_table.tsv" % manifest, file=sys.stderr)
        return 2
    gaps = []
    eng = load_engagement(eng_path, gaps)
    rows, summ, missing = reduce(arms, C.load_phase(phase_path), C.load_refs(manifest), eng, gaps)
    C.write_csv(os.path.join(out, "r84_cmp_rows.csv"), ROWS_HDR, rows)
    C.write_csv(os.path.join(out, "r84_cmp_summary.csv"), SUM_HDR, summ)
    gp = os.path.join(out, "r84_cmp_gaps.txt")
    if gaps:
        open(gp, "w").write("\n".join(gaps) + "\n")
        print("engagement gaps (%d), listed in %s" % (len(gaps), gp), file=sys.stderr)
    elif os.path.exists(gp):
        os.remove(gp)
    print("CMP arms: %d, missing entries: %d, engagement gaps: %d" % (len(arms), len(missing), len(gaps)))
    return C.write_missing(os.path.join(out, "r84_cmp_missing.txt"), missing)


# ---------------------------------------------------------------- selftest
def selftest():
    c = C.Checker()
    d = tempfile.mkdtemp()
    man = os.path.join(d, "man")
    os.makedirs(man)
    cols = ["arm", "group", "bag", "sw_ms", "tbpn", "same_batch_control", "l2_ref"]
    at, ph, rf = [], [], []
    ates = {  # (bag, q, sw, tb) -> (ate, max)
        ("eee_01", "fq", 0, "var_acc"): (30.0, 100.0), ("eee_01", "fq", 5, "var_acc"): (25.0, 100.0), ("eee_01", "fq", 5, "none"): (35.0, 100.0),
        ("eee_01", "mq", 0, "var_acc"): (40.0, 100.0), ("eee_01", "mq", 5, "var_acc"): (40.0, 100.0), ("eee_01", "mq", 5, "none"): (50.0, 900.0),
        ("nya_01", "fq", 0, "var_acc"): (20.0, 100.0), ("nya_01", "fq", 5, "var_acc"): (22.0, 100.0), ("nya_01", "fq", 5, "none"): None,   # missing arm
        ("nya_01", "mq", 0, "var_acc"): (60.0, 100.0), ("nya_01", "mq", 5, "var_acc"): (55.0, 100.0), ("nya_01", "mq", 5, "none"): (58.0, 100.0),
    }
    r83 = {("eee_01", "fq"): 33.0, ("eee_01", "mq"): 45.0, ("nya_01", "fq"): 21.0, ("nya_01", "mq"): 50.0}
    for (bag, q, sw, tb), v in ates.items():
        arm = "r84_rt_cmp_%s_w%dms_%s_%s" % (bag, sw, "vacc" if tb == "var_acc" else "none", q)
        ctl = "r84_rt_cmp_%s_w0ms_vacc_%s" % (bag, q)
        at.append([arm, "CMP", bag, sw, tb, ctl, "r83_rt_ml_%s_l2_%s" % (bag, q)])
        if v:
            ph.append([arm, v[0], v[1]])
    for (bag, q), v in r83.items():
        rf.append(["r83_rt_ml_%s_l2_%s" % (bag, q), "R83", bag, "rt", "pca", "off", 0.425, q, v, 100.0, False])
    ph.append(["r83_rt_ml_eee_01_l2_fq", 999.0, 10.0])   # decoy: the R83 reference must come from the values file
    open(os.path.join(man, "arm_table.tsv"), "w").write("\n".join("\t".join(map(str, r)) for r in [cols] + at) + "\n")
    open(os.path.join(man, "l2_reference_values.csv"), "w").write("arm,source,bag,ovl,fit,lines,edge,q,gt_ate_mm,gt_max_mm,collapsed\n" + "\n".join(",".join(map(str, r)) for r in rf) + "\n")
    pp = os.path.join(d, "phase_summary.csv")
    open(pp, "w").write("arm,gt_ate_mm,gt_max_mm\n" + "\n".join("%s,%s,%s" % tuple(r) for r in ph) + "\n")
    ep = os.path.join(d, "eng.csv")
    open(ep, "w").write("arm,sw_ms,tbpn,sample_window_s,nodes,point_evaluations,clamped_points,mean_trace_m2,max_trace_m2,inert_marker,not_engaged_marker\n"
                        "r84_rt_cmp_eee_01_w5ms_vacc_fq,5,var_acc,0.005,5,1000,7,1.5e-6,3.0e-6,,\n"
                        "r84_rt_cmp_eee_01_w5ms_none_mq,5,none,0.005,5,900,0,1.0e-6,2.0e-6,1,\n")
    out = os.path.join(d, "out")
    rc = run(pp, man, ep, out)
    hdr = open(os.path.join(out, "r84_cmp_rows.csv")).readline().strip()
    c("rows header exact", hdr == ",".join(ROWS_HDR))
    R = {r["arm"]: r for r in C.read_rows(os.path.join(out, "r84_cmp_rows.csv"))}
    c("12 rows, one arm missing, exit 1", len(R) == 12 and rc == 1)
    a = R["r84_rt_cmp_eee_01_w5ms_vacc_fq"]
    c("d vs w0 vacc: 25 minus 30 = -5.000", a["d_ate_vs_w0_vacc_mm"] == "-5.000", a["d_ate_vs_w0_vacc_mm"])
    c("d vs R83 from values file: 25 minus 33 = -8.000 (phase decoy ignored)", a["d_ate_vs_r83_mm"] == "-8.000", a["d_ate_vs_r83_mm"])
    c("w0 vacc arm vs itself is 0.000", R["r84_rt_cmp_eee_01_w0ms_vacc_fq"]["d_ate_vs_w0_vacc_mm"] == "0.000")
    c("engagement columns joined", (a["point_evaluations"], a["clamped_points"], a["mean_trace_m2"], a["max_trace_m2"]) == ("1000", "7", "1.5e-6", "3.0e-6"))
    c("arm without engagement row has blank engagement columns", R["r84_rt_cmp_nya_01_w5ms_vacc_mq"]["point_evaluations"] == "")
    m = R["r84_rt_cmp_nya_01_w5ms_none_fq"]
    c("missing arm: blank values and differences", m["gt_ate_mm"] == "" and m["d_ate_vs_w0_vacc_mm"] == "" and m["d_ate_vs_r83_mm"] == "")
    c("missing list names it", open(os.path.join(out, "r84_cmp_missing.txt")).read().strip() == "phase_summary row (gt_ate_mm): r84_rt_cmp_nya_01_w5ms_none_fq")
    S = {(r["sw_ms"], r["tbpn"], r["q"]): r for r in C.read_rows(os.path.join(out, "r84_cmp_summary.csv"))}
    s = S[("5", "var_acc", "all")]
    c("summary 5/var_acc/all: 4 cells, d = -5, 0, +2, -5 -> 2 lower 1 higher 1 equal, median -2.5", (s["n_cells"], s["n_lower"], s["n_higher"], s["n_equal"], s["median_d_ate_vs_w0_vacc_mm"])
      == ("4", "2", "1", "1", "-2.500"), str(s))
    s = S[("5", "none", "fq")]
    c("summary 5/none/fq: the missing arm is not counted", s["n_cells"] == "1" and s["n_with_d"] == "1" and s["n_higher"] == "1", str(s))
    c("summary n_collapsed counts gt_max > 500", S[("5", "none", "mq")]["n_collapsed"] == "1" and S[("5", "none", "all")]["n_collapsed"] == "1")
    c("summary baseline column is the control median", S[("5", "var_acc", "fq")]["median_w0_vacc_gt_ate_mm"] == "25.000", S[("5", "var_acc", "fq")]["median_w0_vacc_gt_ate_mm"])
    g = open(os.path.join(out, "r84_cmp_gaps.txt")).read()
    c("gaps list inert marker and sw>0 arms without a row", "inert_marker set: r84_rt_cmp_eee_01_w5ms_none_mq" in g and "no engagement row for sw_ms>0 arm: r84_rt_cmp_nya_01_w5ms_vacc_fq" in g)
    c("w0 arm without engagement row is not a gap", "w0ms" not in g)
    out2 = os.path.join(d, "out2")
    rc2 = run(pp, man, os.path.join(d, "absent.csv"), out2)
    R2 = C.read_rows(os.path.join(out2, "r84_cmp_rows.csv"))
    c("engagement file absent: columns blank, gap listed, rows still written", len(R2) == 12 and all(r["point_evaluations"] == "" for r in R2)
      and "engagement file absent" in open(os.path.join(out2, "r84_cmp_gaps.txt")).read() and rc2 == 1)
    return c.done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-summary")
    ap.add_argument("--manifest")
    ap.add_argument("--engagement")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.phase_summary and a.manifest and a.out):
        ap.error("--phase-summary, --manifest and --out are required")
    return run(a.phase_summary, a.manifest, a.engagement, a.out)


if __name__ == "__main__":
    sys.exit(main())
