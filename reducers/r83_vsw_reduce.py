#!/usr/bin/env python3
"""r83_vsw_reduce: family VSW (voxel-edge sweep in discard-overlap mode, dc), host side. Python only; no node run.

Arms (234): r83_dc_sw_<bag>_off_v<tag>_pfn3_<fq|mq>, nine bags x Q x 13 edges 0.20, 0.25, ... 0.80 (tag = str(edge) with '.' -> 'p', e.g. v0p25). dc = overlap_policy
discard; lines off, point filter 3, ds first. Control: the R82 arm r82_dc_ds_<bag>_first_<q> (root voxel 0.425, not on the 0.05 grid), read from the R82 phase_summary.
Comparison table: the rt sweep rows (R81 + Tier A + R82 control) of combined_sweep.csv (columns bag,q,edge,gt_ate_mm,gt_max_mm,src,collapsed).

  python3 r83_vsw_reduce.py --phase R83_phase_summary.csv R82_phase_summary.csv --runs <R83 compact> [<R82 compact>] --rt-table combined_sweep.csv
                            --out-rows r83_vsw_rows.csv --out-range r83_vsw_range.csv --out-vs-rt r83_vsw_vs_rt.csv [--onsets onset_scans.json]
  python3 r83_vsw_reduce.py --selftest

OUT 1 r83_vsw_rows.csv, one row per (bag, Q, edge) of the 13 sweep edges plus one control row per (bag, Q) at edge 0.425 (arm_kind = dc_r82_control):
  bag, q, edge, arm_kind, arm, ate_mm, gt_max_mm, diverged (gt_max_mm > 500), collapse (yes/no/undetermined), departure_mm, n_scans, n_gt_joined, collapse_ref
OUT 2 r83_vsw_range.csv, one row per (bag, Q): the working range = [lowest, highest] sweep edge whose collapse flag is 'no' (an undetermined flag counts as NOT
  non-collapsing here):
  bag, q, n_edges_found (of 13), range_lo_edge, range_hi_edge, n_collapsed_inside_range, n_undetermined, n_diverged,
  range_gtmax_lo_edge, range_gtmax_hi_edge (the same range with 'non-collapsing' read as gt_max_mm <= 500, which needs no trace), n_diverged_inside_gtmax_range,
  ctl_ate_mm, ctl_gt_max_mm, ctl_collapse, ctl_n_scans (the R82 dc control at 0.425, same row)
  The range is the span, not a claim that every edge inside it is non-collapsing: n_collapsed_inside_range counts the ones that are.
OUT 3 r83_vsw_vs_rt.csv, one row per (bag, Q, edge) for every dc row (13 edges plus the 0.425 control) with the rt row of the same bag, Q and edge:
  bag, q, edge, dc_arm, dc_ate_mm, dc_gt_max_mm, dc_collapse, dc_n_scans, rt_ate_mm, rt_gt_max_mm, rt_src, rt_collapsed, ate_diff_dc_minus_rt_mm,
  gt_max_diff_dc_minus_rt_mm   (blank where the rt table has no row at that edge; per cell, never pooled over bags)
No arm is filtered, no ratios; missing arms named in <out-rows>_missing.txt (exit 1). Collapse flag: see r83_common (rule 4.15 c; reference = the dc control arm)."""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

import r83_common as C

EDGES = [round(0.20 + 0.05 * i, 2) for i in range(13)]
CTL_EDGE = 0.425


def tag(e):
    return str(round(e, 3)).replace(".", "p")


def reduce(ctx, rt, bags=C.BAGS):
    rows = []
    for bag in bags:
        for q in C.QS:
            ctl = "r82_dc_ds_%s_first_%s" % (bag, q)
            for e in EDGES + [CTL_EDGE]:
                is_ctl = e == CTL_EDGE
                arm = ctl if is_ctl else "r83_dc_sw_%s_off_v%s_pfn3_%s" % (bag, tag(e), q)
                f = ctx.arm_fields(arm, ctl, bag)
                if not f:
                    continue
                if is_ctl:   # the control is its own reference: departure from itself is 0 and says nothing
                    gm = f["gt_max_mm"]
                    f["collapse"] = "no" if gm <= C.COLLAPSE_GT_MM else "undetermined"
                    f["departure_mm"] = np.nan
                rows.append(dict(bag=bag, q=q, edge=e, arm_kind="dc_r82_control" if is_ctl else "dc_r83_sweep", arm=arm, ate_mm=f["ate_mm"], gt_max_mm=f["gt_max_mm"],
                                 diverged=f["diverged"], collapse=f["collapse"], departure_mm=f["departure_mm"], n_scans=f["n_scans"], n_gt_joined=f["n_gt_joined"],
                                 collapse_ref="self (control)" if is_ctl else ctl))
    R = pd.DataFrame(rows)
    # working range
    rr = []
    for bag in bags:
        for q in C.QS:
            s = R[(R.bag == bag) & (R.q == q) & (R.arm_kind == "dc_r83_sweep")] if len(R) else R
            c = R[(R.bag == bag) & (R.q == q) & (R.arm_kind == "dc_r82_control")] if len(R) else R
            row = dict(bag=bag, q=q, n_edges_found=len(s))
            ok = s[s.collapse == "no"]
            row.update(range_lo_edge=ok.edge.min() if len(ok) else np.nan, range_hi_edge=ok.edge.max() if len(ok) else np.nan)
            if len(ok):
                inside = s[(s.edge >= ok.edge.min()) & (s.edge <= ok.edge.max())]
                row["n_collapsed_inside_range"] = int((inside.collapse != "no").sum())
            row["n_undetermined"] = int((s.collapse == "undetermined").sum()) if len(s) else 0
            row["n_diverged"] = int(s.diverged.sum()) if len(s) else 0
            g = s[~s.diverged.astype(bool)] if len(s) else s
            row.update(range_gtmax_lo_edge=g.edge.min() if len(g) else np.nan, range_gtmax_hi_edge=g.edge.max() if len(g) else np.nan)
            if len(g):
                inside = s[(s.edge >= g.edge.min()) & (s.edge <= g.edge.max())]
                row["n_diverged_inside_gtmax_range"] = int(inside.diverged.sum())
            if len(c):
                c = c.iloc[0]
                row.update(ctl_ate_mm=c.ate_mm, ctl_gt_max_mm=c.gt_max_mm, ctl_collapse=c.collapse, ctl_n_scans=c.n_scans)
            rr.append(row)
    RR = pd.DataFrame(rr)
    # side by side with rt
    rt = rt.copy()
    rt["edge3"] = rt.edge.round(3)
    rt = rt.drop_duplicates(["bag", "q", "edge3"])
    X = R.copy()
    X["edge3"] = X.edge.round(3)
    X = X.merge(rt[["bag", "q", "edge3", "gt_ate_mm", "gt_max_mm", "src", "collapsed"]].rename(columns=dict(gt_ate_mm="rt_ate_mm", gt_max_mm="rt_gt_max_mm", src="rt_src", collapsed="rt_collapsed")),
                on=["bag", "q", "edge3"], how="left")
    X["ate_diff_dc_minus_rt_mm"] = (X.ate_mm - X.rt_ate_mm).round(3)
    X["gt_max_diff_dc_minus_rt_mm"] = (X.gt_max_mm - X.rt_gt_max_mm).round(3)
    X = X.rename(columns=dict(arm="dc_arm", ate_mm="dc_ate_mm", gt_max_mm="dc_gt_max_mm", collapse="dc_collapse", n_scans="dc_n_scans"))
    X["rt_ate_mm"] = X.rt_ate_mm.round(3)
    X["rt_gt_max_mm"] = X.rt_gt_max_mm.round(3)
    X = X[["bag", "q", "edge", "dc_arm", "dc_ate_mm", "dc_gt_max_mm", "dc_collapse", "dc_n_scans", "rt_ate_mm", "rt_gt_max_mm", "rt_src", "rt_collapsed",
           "ate_diff_dc_minus_rt_mm", "gt_max_diff_dc_minus_rt_mm"]]
    nrt = int(X.rt_ate_mm.isna().sum()) if len(X) else 0
    for _, r in X[X.rt_ate_mm.isna()].iterrows():
        ctx.miss("rt table row: %s %s %.3f" % (r.bag, r.q, r.edge))
    return R, RR, X


def selftest(rt_path=None, r82_phase=None):
    c = C.Checker()
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(11)
    real = rt_path and os.path.exists(rt_path)
    if real:
        rt = pd.read_csv(rt_path)
        print("selftest uses the REAL rt table %s (%d rows) with synthetic dc arms (rt + noise)" % (rt_path, len(rt)))
    else:
        rows = []
        for b in C.BAGS:
            for q in C.QS:
                for e in [round(0.2 + 0.05 * i, 2) for i in range(19)] + [0.425]:
                    gm = float(rng.uniform(80, 200) + (900 if e <= 0.25 and b == "sbs_02" else 0))
                    rows.append(dict(bag=b, q=q, edge=e, gt_ate_mm=float(rng.uniform(20, 60)), gt_max_mm=gm, src="R81", collapsed=gm > 500))
        rt = pd.DataFrame(rows)
        print("selftest uses a SYNTHETIC rt table (%d rows)" % len(rt))
    rtp = os.path.join(d, "rt.csv")
    rt.to_csv(rtp, index=False)
    # synthetic dc arms: rt + noise, on the 13 edges
    ph, noise = [], {}
    for _, r in rt.iterrows():
        e = round(float(r.edge), 3)
        if e in EDGES:
            a = "r83_dc_sw_%s_off_v%s_pfn3_%s" % (r.bag, tag(e), r.q)
            na, ng = float(rng.normal(0, 3)), float(rng.normal(0, 10))
            gm = max(float(r.gt_max_mm) + ng, 1.0)
            ph.append(dict(arm=a, gt_ate_mm=float(r.gt_ate_mm) + na, gt_max_mm=gm, n_scans=3000))
            noise[(r.bag, r.q, e)] = (float(r.gt_ate_mm) + na, gm)
    # dc controls: from the real R82 phase when present
    if r82_phase and os.path.exists(r82_phase):
        p82 = pd.read_csv(r82_phase, low_memory=False)
        p82 = p82[p82.arm.str.startswith("r82_dc_ds_") & p82.arm.str.contains("_first_")][["arm", "gt_ate_mm", "gt_max_mm", "n_scans"]]
        print("selftest uses the REAL r82_dc_ds_*_first_* control rows (%d) of %s" % (len(p82), r82_phase))
    else:
        p82 = pd.DataFrame([dict(arm="r82_dc_ds_%s_first_%s" % (b, q), gt_ate_mm=40.0, gt_max_mm=150.0, n_scans=3000) for b in C.BAGS for q in C.QS])
    ph = pd.concat([pd.DataFrame(ph), p82], ignore_index=True)
    # make one known collapse pattern on eee_01 fq: edges 0.20, 0.25 collapsed (gt_max 1200, with traces), 0.45 collapsed inside the range (traces), 0.8 not
    ref = "r82_dc_ds_eee_01_first_fq"
    C.write_arm_trace(d, ref, C.synth_trace(seed=3))
    for e in (0.2, 0.25, 0.45):
        a = "r83_dc_sw_eee_01_off_v%s_pfn3_fq" % tag(e)
        ph.loc[ph.arm == a, "gt_max_mm"] = 1200.0
        C.write_arm_trace(d, a, C.synth_trace(seed=3, jump_at=520, jump_mm=900.0))
    pc = os.path.join(d, "ph.csv")
    ph.to_csv(pc, index=False)
    ctx = C.Ctx([pc], [d])
    R, RR, X = reduce(ctx, pd.read_csv(rtp))
    c("rows = 9*2*13 sweep + 18 controls", len(R) == 9 * 2 * 13 + 18, str(len(R)))
    c("range table 18 rows", len(RR) == 18)
    r = RR[(RR.bag == "eee_01") & (RR.q == "fq")].iloc[0]
    # edges flagged yes: 0.2, 0.25, 0.45; others: 'no' if gt_max <= 500, else (real rt rows with gt_max>500 and no trace) undetermined
    s = R[(R.bag == "eee_01") & (R.q == "fq") & (R.arm_kind == "dc_r83_sweep")].set_index("edge")
    ok_edges = sorted(s[s.collapse == "no"].index)
    c("collapse yes on the three planted edges with traces", all(s.loc[e, "collapse"] == "yes" for e in (0.2, 0.25, 0.45)), str(s.collapse.to_dict()))
    c("range lo/hi equal the lowest/highest 'no' edge", r.range_lo_edge == ok_edges[0] and r.range_hi_edge == ok_edges[-1], "%s %s" % (r.range_lo_edge, r.range_hi_edge))
    inside = s[(s.index >= ok_edges[0]) & (s.index <= ok_edges[-1])]
    c("n_collapsed_inside_range counted (0.45 planted)", r.n_collapsed_inside_range == int((inside.collapse != "no").sum()) and r.n_collapsed_inside_range >= 1, str(r.n_collapsed_inside_range))
    c("control in the same range row", np.isfinite(r.ctl_ate_mm) and r.ctl_n_scans > 0)
    # diff per cell equals the noise exactly
    x = X[(X.bag == "sbs_01") & (X.q == "mq") & (X.edge == 0.5)].iloc[0]
    ex = noise[("sbs_01", "mq", 0.5)]
    rt_row = rt[(rt.bag == "sbs_01") & (rt.q == "mq") & (rt.edge.round(3) == 0.5)].iloc[0]
    c("dc minus rt equals the injected noise (3-decimal inputs)", abs(x.ate_diff_dc_minus_rt_mm - round(ex[0] - rt_row.gt_ate_mm, 3)) < 2e-3, "%.3f" % x.ate_diff_dc_minus_rt_mm)
    c("rt columns carried (src, collapsed)", isinstance(x.rt_src, str))
    c("no pooling: every vs_rt row has one bag", X.bag.isin(C.BAGS).all() and len(X) == len(R))
    c("diverged rows are kept (none filtered)", int(R.diverged.sum()) == int(((R.gt_max_mm > 500)).sum()) and len(R) == 9 * 2 * 13 + 18)
    gl = RR[(RR.bag == "eee_01") & (RR.q == "fq")].iloc[0]
    g = s[s.gt_max_mm <= 500]
    c("gt_max-basis range needs no trace", gl.range_gtmax_lo_edge == g.index.min() and gl.range_gtmax_hi_edge == g.index.max())
    c("control row flagged by gt_max only", set(R[R.arm_kind == "dc_r82_control"].collapse) <= {"no", "undetermined"})
    c("no missing inputs in the complete synthetic set", not [m for m in ctx.missing if m.startswith("phase_summary")], str(ctx.missing[:3]))
    # missing arm
    ph3 = ph[ph.arm != "r83_dc_sw_nya_03_off_v0p55_pfn3_mq"]
    p3 = os.path.join(d, "ph3.csv")
    ph3.to_csv(p3, index=False)
    ctx3 = C.Ctx([p3], [d])
    R3, RR3, X3 = reduce(ctx3, pd.read_csv(rtp))
    c("missing dc arm named, row absent", len(R3) == len(R) - 1 and "phase_summary row: r83_dc_sw_nya_03_off_v0p55_pfn3_mq" in ctx3.missing)
    c("n_edges_found reports 12 for that cell", RR3[(RR3.bag == "nya_03") & (RR3.q == "mq")].iloc[0].n_edges_found == 12)
    import subprocess
    o = [os.path.join(d, n) for n in ("o_rows.csv", "o_range.csv", "o_vs.csv")]
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--phase", pc, "--runs", d, "--rt-table", rtp, "--out-rows", o[0], "--out-range", o[1], "--out-vs-rt", o[2]], capture_output=True, text=True)
    c("CLI writes the three files", p.returncode == 0 and all(os.path.exists(x) for x in o) and len(pd.read_csv(o[2])) == len(R), (p.stdout + p.stderr).strip()[:100])
    return c.done()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--rt-table", help="combined_sweep.csv (bag,q,edge,gt_ate_mm,gt_max_mm,src,collapsed)")
    ap.add_argument("--out-rows")
    ap.add_argument("--out-range")
    ap.add_argument("--out-vs-rt")
    ap.add_argument("--selftest-rt-table", default="/tmp/claude-0/r81/combined_sweep.csv")
    ap.add_argument("--selftest-r82-phase", default="/tmp/claude-0/r82a/x/R82A_RETURN_2026-10-06/phase_summary.csv")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.selftest_rt_table, a.selftest_r82_phase)
    if not (a.phase and a.rt_table and a.out_rows and a.out_range and a.out_vs_rt):
        ap.error("--phase, --rt-table, --out-rows, --out-range, --out-vs-rt are required")
    ctx = C.Ctx(a.phase, a.runs, a.onsets)
    R, RR, X = reduce(ctx, pd.read_csv(a.rt_table))
    R.to_csv(a.out_rows, index=False)
    RR.to_csv(a.out_range, index=False)
    X.to_csv(a.out_vs_rt, index=False)
    print("rows %d, range rows %d, vs_rt rows %d" % (len(R), len(RR), len(X)))
    return ctx.write_missing(a.out_rows)


if __name__ == "__main__":
    sys.exit(main())
