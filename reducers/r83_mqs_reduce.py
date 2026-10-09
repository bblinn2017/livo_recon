#!/usr/bin/env python3
"""r83_mqs_reduce: family MQS (motion-Q gyro x acc scale grid), host side. Python only; no node run.

Arms (108): r83_rt_mqs_<bag>_g<10|3|1|0p3>_a<0p3|0p15|0p1>_mq, nine bags x gyro_scale {10, 3, 1, 0.3} x acc_scale {0.3, 0.15, 0.1}, motion-Q only (no Q column).
The cell (g 10, a 0.3) is the in-row baseline of every row of its bag (it equals the R82 mq control and is run anyway).

  python3 r83_mqs_reduce.py --phase R83_phase_summary.csv [R82_phase_summary.csv] --runs <R83 runs dir with <arm>/motion_q_scan.csv[.gz] and pos_trace_all.csv.gz> [<R82 compact>]
                            --out-grid r83_mqs_grid.csv --out-wide r83_mqs_grid_wide.csv --out-scan r83_mqs_scan_summary.csv [--onsets onset_scans.json]
  python3 r83_mqs_reduce.py --selftest

OUT 1 r83_mqs_grid.csv, one row per (bag, g, a) = 108 rows:
  bag, gyro_scale, acc_scale, arm, ate_mm, gt_max_mm, diverged (gt_max_mm > 500), collapse (yes/no/undetermined), departure_mm, n_scans, n_gt_joined,
  base_arm, base_ate_mm, base_gt_max_mm, base_n_scans (the (g 10, a 0.3) cell of the bag), ate_diff_cell_minus_base_mm, gt_max_diff_cell_minus_base_mm,
  r82_ctl_ate_mm, r82_ctl_gt_max_mm (the R82 arm r82_rt_ds_<bag>_first_mq, if its row is in a given phase_summary), collapse_ref (that R82 arm)
OUT 2 r83_mqs_grid_wide.csv, per bag and gyro_scale one row, columns ate_mm_a0p3, ate_mm_a0p15, ate_mm_a0p1, gt_max_mm_a..., collapse_a... (the same numbers laid out as
  a grid for reading; nothing new).
OUT 3 r83_mqs_scan_summary.csv, one row per arm from the arm's motion_q_scan.csv(.gz): the column set of R82A's r83mq_scan_summary.csv
  (arm, scans, q_acc_q10, q_acc_q50, q_acc_q90, q_gyr_q10, q_gyr_q50, q_gyr_q90, acc_cap_fraction_mean, gyr_cap_fraction_mean, and the same ten with suffix _post_onset
  for scans with scan_index >= the bag's onset scan). q_acc = mean of mean_q_acc_x, _y, _z of the scan row, q_gyr likewise (same definition as R82A, R82_ADDENDUM item 4);
  quantiles over scans (linear interpolation); cap fractions = mean over scans of acc_cap_fraction / gyr_cap_fraction (share of axes clamped at the cap). Appended:
  acc_scale_file, gyro_scale_file (first row of the file), scales_match_name (1 if both equal the scale in the arm name to 1e-6), n_scans_post_onset.
  An arm without a motion_q_scan file gets an empty row and is named in the missing list.
No arm is filtered out, no pooling over bags, no ratios. Missing inputs: <out-grid>_missing.txt, exit 1."""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

import r83_common as C

GS = [("10", 10.0), ("3", 3.0), ("1", 1.0), ("0p3", 0.3)]
AS = [("0p3", 0.3), ("0p15", 0.15), ("0p1", 0.1)]
BASE = ("10", "0p3")
QCOLS = ["q_acc_q10", "q_acc_q50", "q_acc_q90", "q_gyr_q10", "q_gyr_q50", "q_gyr_q90", "acc_cap_fraction_mean", "gyr_cap_fraction_mean"]


def arm_name(bag, g, a):
    return "r83_rt_mqs_%s_g%s_a%s_mq" % (bag, g, a)


def grid(ctx, bags=C.BAGS):
    rows = []
    for bag in bags:
        ctl = "r82_rt_ds_%s_first_mq" % bag
        base = arm_name(bag, *BASE)
        fb = ctx.arm_fields(base, ctl, bag)
        for g, gv in GS:
            for a, av in AS:
                arm = arm_name(bag, g, a)
                f = ctx.arm_fields(arm, ctl, bag)
                if not f:
                    continue
                r = dict(bag=bag, gyro_scale=gv, acc_scale=av, arm=arm, **f)
                r.update(base_arm=base)
                if fb:
                    r.update(base_ate_mm=fb["ate_mm"], base_gt_max_mm=fb["gt_max_mm"], base_n_scans=fb["n_scans"],
                             ate_diff_cell_minus_base_mm=C.f3(f["ate_mm"] - fb["ate_mm"]), gt_max_diff_cell_minus_base_mm=C.f3(f["gt_max_mm"] - fb["gt_max_mm"]))
                if ctx.phase.has(ctl):
                    r.update(r82_ctl_ate_mm=C.f3(ctx.phase.get(ctl, "gt_ate_mm")), r82_ctl_gt_max_mm=C.f3(ctx.phase.get(ctl, "gt_max_mm")))
                r["collapse_ref"] = ctl
                rows.append(r)
    return pd.DataFrame(rows)


def wide(G):
    out = []
    if not len(G):
        return pd.DataFrame()
    for (bag, g), s in G.groupby(["bag", "gyro_scale"], sort=False):
        r = dict(bag=bag, gyro_scale=g)
        for a, av in AS:
            x = s[s.acc_scale == av]
            if len(x):
                x = x.iloc[0]
                r.update({"ate_mm_a" + a: x.ate_mm, "gt_max_mm_a" + a: x.gt_max_mm, "collapse_a" + a: x.collapse, "ate_diff_mm_a" + a: x.get("ate_diff_cell_minus_base_mm", np.nan)})
        out.append(r)
    return pd.DataFrame(out)


def scan_summary(ctx, bags=C.BAGS):
    rows = []
    for bag in bags:
        on = ctx.onsets[bag]
        for g, gv in GS:
            for a, av in AS:
                arm = arm_name(bag, g, a)
                p = C.find_file(ctx.roots, arm, "motion_q_scan.csv") or C.find_file(ctx.roots, arm, "motion_q_scan_all.csv.gz")
                if p is None:
                    ctx.miss("motion_q_scan.csv[.gz]: " + arm)
                    rows.append(dict(arm=arm))
                    continue
                d = pd.read_csv(p)
                if "arm" in d.columns:
                    d = d[d["arm"] == arm]
                if len(d) == 0:
                    ctx.miss("motion_q_scan.csv empty: " + arm)
                    rows.append(dict(arm=arm))
                    continue
                d = d.assign(q_acc=d[["mean_q_acc_x", "mean_q_acc_y", "mean_q_acc_z"]].mean(axis=1), q_gyr=d[["mean_q_gyr_x", "mean_q_gyr_y", "mean_q_gyr_z"]].mean(axis=1))

                def stats(s, suf):
                    o = {}
                    for nm, col in (("q_acc", "q_acc"), ("q_gyr", "q_gyr")):
                        qs = np.quantile(s[col], [0.1, 0.5, 0.9]) if len(s) else [np.nan] * 3
                        for k, v in zip(("q10", "q50", "q90"), qs):
                            o["%s_%s%s" % (nm, k, suf)] = float(v)
                    o["acc_cap_fraction_mean" + suf] = float(s["acc_cap_fraction"].mean()) if len(s) else np.nan
                    o["gyr_cap_fraction_mean" + suf] = float(s["gyr_cap_fraction"].mean()) if len(s) else np.nan
                    return o
                r = dict(arm=arm, scans=len(d))
                r.update(stats(d, ""))
                post = d[d.scan_index >= on]
                r.update(stats(post, "_post_onset"))
                fa, fg = float(d["acc_scale"].iloc[0]), float(d["gyro_scale"].iloc[0])
                r.update(acc_scale_file=fa, gyro_scale_file=fg, scales_match_name=int(abs(fa - av) < 1e-6 and abs(fg - gv) < 1e-6), n_scans_post_onset=len(post))
                rows.append(r)
    return pd.DataFrame(rows)


def selftest():
    c = C.Checker()
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(2)
    ph = []
    for i, bag in enumerate(C.BAGS):
        ph.append(dict(arm="r82_rt_ds_%s_first_mq" % bag, gt_ate_mm=33.0, gt_max_mm=100.0, n_scans=900))
        for g, gv in GS:
            for a, av in AS:
                arm = arm_name(bag, g, a)
                ate = 33.0 + (0.0 if (g, a) == BASE else gv * 0.7 - av * 5)
                ph.append(dict(arm=arm, gt_ate_mm=ate, gt_max_mm=100.0 + gv, n_scans=900))
                n = 900
                m = pd.DataFrame(dict(scan_index=np.arange(1, n + 1), acc_scale=av, gyro_scale=gv))
                for ax in "xyz":
                    m["mean_q_acc_" + ax] = np.linspace(0.01, 0.05, n) * (1 + 0 * rng.normal())
                    m["mean_q_gyr_" + ax] = np.linspace(0.1, 2.0, n)
                m["acc_cap_fraction"] = 0.0
                m["gyr_cap_fraction"] = np.where(m.scan_index > 800, 1.0 / 3, 0.0)
                if bag == "eee_01" and (g, a) == ("1", "0p15"):
                    m["gyro_scale"] = 3.0   # planted mismatch
                if bag == "nya_02" and (g, a) == ("3", "0p1"):
                    continue   # planted missing scan file
                dd = os.path.join(d, arm)
                os.makedirs(dd, exist_ok=True)
                (m.to_csv(os.path.join(dd, "motion_q_scan.csv"), index=False) if (g, a) != ("0p3", "0p1") else m.to_csv(os.path.join(dd, "motion_q_scan.csv.gz"), index=False))
    # a diverged cell with a trace
    arm = arm_name("sbs_03", "0p3", "0p1")
    ph = pd.DataFrame(ph)
    ph.loc[ph.arm == arm, "gt_max_mm"] = 2000.0
    C.write_arm_trace(d, arm, C.synth_trace(seed=9, jump_at=400, jump_mm=1000.0))
    C.write_arm_trace(d, "r82_rt_ds_sbs_03_first_mq", C.synth_trace(seed=9))
    pc = os.path.join(d, "ph.csv")
    ph.to_csv(pc, index=False)
    ctx = C.Ctx([pc], [d])
    G = grid(ctx)
    c("108 grid rows", len(G) == 108, str(len(G)))
    r = G[(G.bag == "eee_02") & (G.gyro_scale == 3.0) & (G.acc_scale == 0.15)].iloc[0]
    c("diff cell minus base in mm", abs(r.ate_diff_cell_minus_base_mm - (3 * 0.7 - 0.15 * 5)) < 1e-9, "%.3f" % r.ate_diff_cell_minus_base_mm)
    c("baseline in the row, its n", r.base_ate_mm == 33.0 and r.base_n_scans == 900 and r.n_scans == 900)
    b = G[(G.gyro_scale == 10.0) & (G.acc_scale == 0.3)]
    c("baseline cell has difference 0 in all 9 bags", len(b) == 9 and (b.ate_diff_cell_minus_base_mm == 0).all())
    r = G[(G.bag == "sbs_03") & (G.gyro_scale == 0.3) & (G.acc_scale == 0.1)].iloc[0]
    c("diverged cell kept, collapse yes via trace", r.collapse == "yes" and r.diverged and r.gt_max_mm == 2000.0, r.collapse)
    W = wide(G)
    c("wide grid 36 rows with 3 acc columns", len(W) == 36 and {"ate_mm_a0p3", "ate_mm_a0p15", "ate_mm_a0p1"} <= set(W.columns))
    S = scan_summary(ctx)
    c("scan summary 108 rows", len(S) == 108)
    s = S[S.arm == arm_name("eee_03", "10", "0p3")].iloc[0]
    exp = np.quantile(np.linspace(0.01, 0.05, 900), [0.1, 0.5, 0.9])
    c("q_acc quantiles equal an independent numpy value", np.allclose([s.q_acc_q10, s.q_acc_q50, s.q_acc_q90], exp), "%.5f %.5f %.5f" % (s.q_acc_q10, s.q_acc_q50, s.q_acc_q90))
    c("gyr cap fraction mean (100 of 900 scans at 1/3)", abs(s.gyr_cap_fraction_mean - (100 / 3) / 900) < 1e-9, "%.6f" % s.gyr_cap_fraction_mean)
    on = C.ONSETS["eee_03"]
    c("post-onset scans counted and quantiles over them", s.n_scans_post_onset == 900 - on + 1 and abs(s.q_acc_q50_post_onset - np.median(np.linspace(0.01, 0.05, 900)[on - 1:])) < 1e-9, str(s.n_scans_post_onset))
    c("scale mismatch flagged", S[S.arm == arm_name("eee_01", "1", "0p15")].iloc[0].scales_match_name == 0 and S[S.arm == arm_name("eee_01", "3", "0p3")].iloc[0].scales_match_name == 1)
    c("gz motion_q_scan read", S[S.arm == arm_name("eee_01", "0p3", "0p1")].iloc[0].scans == 900)
    c("missing scan file named and row empty", any("nya_02_g3_a0p1" in m for m in ctx.missing) and np.isnan(S[S.arm == arm_name("nya_02", "3", "0p1")].iloc[0].get("scans", np.nan)), str(ctx.missing))
    c("no ratio / pooled columns", not any("ratio" in k or "pooled" in k for k in list(G.columns) + list(S.columns)))
    import subprocess
    o = [os.path.join(d, n) for n in ("o_grid.csv", "o_wide.csv", "o_scan.csv")]
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--phase", pc, "--runs", d, "--out-grid", o[0], "--out-wide", o[1], "--out-scan", o[2]], capture_output=True, text=True)
    c("CLI writes the three files, exit 1 for the planted missing scan file", p.returncode == 1 and all(os.path.exists(x) for x in o) and os.path.exists(os.path.join(d, "o_grid_missing.txt")), (p.stdout + p.stderr).strip()[:100])
    return c.done()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--out-grid")
    ap.add_argument("--out-wide")
    ap.add_argument("--out-scan")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.phase and a.out_grid and a.out_wide and a.out_scan):
        ap.error("--phase, --out-grid, --out-wide, --out-scan are required")
    ctx = C.Ctx(a.phase, a.runs, a.onsets)
    G = grid(ctx)
    G.to_csv(a.out_grid, index=False)
    wide(G).to_csv(a.out_wide, index=False)
    S = scan_summary(ctx)
    S.to_csv(a.out_scan, index=False)
    print("grid rows %d, scan summary rows %d" % (len(G), len(S)))
    return ctx.write_missing(a.out_grid)


if __name__ == "__main__":
    sys.exit(main())
