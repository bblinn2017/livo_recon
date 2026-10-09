#!/usr/bin/env python3
"""r83_mxl_reduce: family MXL (max_layer 1 against 2), host side, reads archived run folders, writes one small CSV. Python only; no node run.

Arms (36): r83_rt_ml_<bag>_l<1|2>_<fq|mq>, nine bags x max_layer {1, 2} x Q. Baseline = max_layer 2 (the node's default), in the same row.

  python3 r83_mxl_reduce.py --phase R83_phase_summary.csv [R82_phase_summary.csv] --runs <R83 runs/compact dir> [<R82 compact dir>] --out r83_mxl.csv
                            [--onsets r82_manifest/onset_scans.json]
  python3 r83_mxl_reduce.py --selftest

One row per (bag, Q), 18 rows, columns:
  bag, q,
  ate_mm_l1, gt_max_mm_l1, collapse_l1, diverged_l1, n_scans_l1, n_gt_joined_l1        max_layer 1 arm (ATE after one global rigid alignment, largest per-scan GT error, mm)
  ate_mm_l2, gt_max_mm_l2, collapse_l2, diverged_l2, n_scans_l2, n_gt_joined_l2        max_layer 2 arm = the baseline of this row
  ate_diff_l1_minus_l2_mm, gt_max_diff_l1_minus_l2_mm                                  differences in mm (negative = max_layer 1 lower)
  n_paired_scans, pos_diff_median_mm, pos_diff_max_mm   per-scan distance between the two arms' own estimated positions (pos_trace_all.csv.gz, scan_id join,
                                                        NOT aligned to ground truth; both runs start from the same stationary calibration so their frames agree; mm)
  ate_mm_r82_ctl, gt_max_mm_r82_ctl                     the R82 arm r82_rt_ds_<bag>_first_<q> (same settings, max_layer 2) when its row is in a given phase_summary; blank if not
  collapse_ref                                          the arm used as the healthy reference of the collapse flag (the R82 control)
collapse = yes / no / undetermined (see r83_common docstring: rule 4.15 c; undetermined when gt_max_mm > 500 and a trace is missing). diverged = gt_max_mm > 500.
No arm is filtered out, no pooling over bags, no ratios. Missing arms or traces are named in <out>_missing.txt (exit 1)."""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

import r83_common as C


def pos_diff(ctx, a1, a2):
    t1, t2 = ctx.trace(a1), ctx.trace(a2)
    if t1 is None:
        ctx.miss("pos_trace_all.csv.gz: " + a1)
    if t2 is None:
        ctx.miss("pos_trace_all.csv.gz: " + a2)
    if t1 is None or t2 is None:
        return 0, np.nan, np.nan
    a = t1.drop_duplicates("scan_id").set_index("scan_id")[["px", "py", "pz"]]
    b = t2.drop_duplicates("scan_id").set_index("scan_id")[["px", "py", "pz"]]
    ix = a.index.intersection(b.index)
    if len(ix) == 0:
        return 0, np.nan, np.nan
    d = 1000.0 * np.linalg.norm(a.loc[ix].to_numpy(float) - b.loc[ix].to_numpy(float), axis=1)
    return len(ix), C.f3(np.median(d)), C.f3(np.max(d))


def reduce(ctx, bags=C.BAGS):
    rows = []
    for bag in bags:
        for q in C.QS:
            a1, a2 = "r83_rt_ml_%s_l1_%s" % (bag, q), "r83_rt_ml_%s_l2_%s" % (bag, q)
            ctl = "r82_rt_ds_%s_first_%s" % (bag, q)
            r = dict(bag=bag, q=q)
            f1, f2 = ctx.arm_fields(a1, ctl, bag, "_l1"), ctx.arm_fields(a2, ctl, bag, "_l2")
            r.update(f1)
            r.update(f2)
            if f1 and f2:
                r["ate_diff_l1_minus_l2_mm"] = C.f3(f1["ate_mm_l1"] - f2["ate_mm_l2"])
                r["gt_max_diff_l1_minus_l2_mm"] = C.f3(f1["gt_max_mm_l1"] - f2["gt_max_mm_l2"])
                n, med, mx = pos_diff(ctx, a1, a2)
                r.update(n_paired_scans=n, pos_diff_median_mm=med, pos_diff_max_mm=mx)
            if ctx.phase.has(ctl):
                r["ate_mm_r82_ctl"] = C.f3(ctx.phase.get(ctl, "gt_ate_mm"))
                r["gt_max_mm_r82_ctl"] = C.f3(ctx.phase.get(ctl, "gt_max_mm"))
            r["collapse_ref"] = ctl
            rows.append(r)
    return pd.DataFrame(rows)


def selftest():
    c = C.Checker()
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(5)
    ph = []
    for i, bag in enumerate(C.BAGS):
        for q in C.QS:
            base = C.synth_trace(seed=i * 2 + (q == "mq"))
            ctl = "r82_rt_ds_%s_first_%s" % (bag, q)
            ph.append(dict(arm=ctl, gt_ate_mm=30.0 + i, gt_max_mm=120.0, n_scans=800))
            C.write_arm_trace(d, ctl, base)
            for L in (1, 2):
                a = "r83_rt_ml_%s_l%d_%s" % (bag, L, q)
                tr = base.copy()
                if L == 1:
                    tr["px"] = tr["px"] + 0.003 * (tr.scan_id / 800.0)   # grows to 3 mm at the last scan
                ph.append(dict(arm=a, gt_ate_mm=30.0 + i + (2.5 if L == 1 else 0.0), gt_max_mm=120.0, n_scans=800))
                C.write_arm_trace(d, a, tr)
    # one diverged l1 arm (eee_02 fq): 900 mm jump at scan 500; one diverged l1 arm without trace (sbs_01 mq)
    ph = pd.DataFrame(ph)
    a = "r83_rt_ml_eee_02_l1_fq"
    tr = C.synth_trace(seed=2)
    tr["px"] += 0.003 * (tr.scan_id / 800.0) + np.where(tr.scan_id > 500, 0.9, 0.0)
    C.write_arm_trace(d, a, tr)
    ph.loc[ph.arm == a, ["gt_ate_mm", "gt_max_mm"]] = [400.0, 1500.0]
    a2 = "r83_rt_ml_sbs_01_l1_mq"
    ph.loc[ph.arm == a2, ["gt_ate_mm", "gt_max_mm"]] = [380.0, 1100.0]
    os.remove(os.path.join(d, "compact", a2, "pos_trace_all.csv.gz"))
    pc = os.path.join(d, "phase_summary.csv")
    ph.to_csv(pc, index=False)
    ctx = C.Ctx([pc], [d])
    out = reduce(ctx)
    c("18 rows (9 bags x 2 Q)", len(out) == 18, str(len(out)))
    r = out[(out.bag == "eee_01") & (out.q == "fq")].iloc[0]
    c("difference l1 minus l2 is 2.5 mm", abs(r.ate_diff_l1_minus_l2_mm - 2.5) < 1e-9, "%.3f" % r.ate_diff_l1_minus_l2_mm)
    c("baseline l2 in the same row, n scans both", r.ate_mm_l2 == 30.0 and r.n_scans_l1 == 800 and r.n_scans_l2 == 800)
    c("paired position diff median ~1.5 mm, max 3 mm", abs(r.pos_diff_max_mm - 3.0) < 0.01 and abs(r.pos_diff_median_mm - 1.5) < 0.05, "med %.3f max %.3f" % (r.pos_diff_median_mm, r.pos_diff_max_mm))
    c("paired scans = 800", r.n_paired_scans == 800)
    r = out[(out.bag == "eee_02") & (out.q == "fq")].iloc[0]
    c("diverged arm stays in the table, collapse yes", r.collapse_l1 == "yes" and bool(r.diverged_l1) and r.gt_max_mm_l1 == 1500.0, "dep %.1f" % r.departure_mm_l1)
    r = out[(out.bag == "sbs_01") & (out.q == "mq")].iloc[0]
    c("diverged arm without trace is undetermined", r.collapse_l1 == "undetermined")
    c("missing trace named", any("sbs_01_l1_mq" in m for m in ctx.missing), str(ctx.missing))
    c("R82 control columns filled", out.ate_mm_r82_ctl.notna().all())
    c("no pooled/ratio columns", not any(("ratio" in k) or ("mean_over" in k) for k in out.columns))
    c("three decimals", all(abs(v * 1000 - round(v * 1000)) < 1e-6 for v in out.ate_mm_l1.dropna()))
    # missing arm entirely
    ph2 = ph[ph.arm != "r83_rt_ml_nya_01_l2_mq"]
    p2 = os.path.join(d, "ph2.csv")
    ph2.to_csv(p2, index=False)
    ctx2 = C.Ctx([p2], [d])
    o2 = reduce(ctx2)
    c("missing arm row kept with l1 only, named missing", len(o2) == 18 and "phase_summary row: r83_rt_ml_nya_01_l2_mq" in ctx2.missing and np.isnan(o2[(o2.bag == "nya_01") & (o2.q == "mq")].iloc[0].get("ate_diff_l1_minus_l2_mm", np.nan)))
    import subprocess
    o = os.path.join(d, "o_mxl.csv")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--phase", pc, "--runs", d, "--out", o], capture_output=True, text=True)
    c("CLI writes the table, exit 1 (planted missing trace)", p.returncode == 1 and len(pd.read_csv(o)) == 18, (p.stdout + p.stderr).strip()[:100])
    return c.done()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.phase and a.out):
        ap.error("--phase and --out are required")
    ctx = C.Ctx(a.phase, a.runs, a.onsets)
    out = reduce(ctx)
    out.to_csv(a.out, index=False)
    print("%s: %d rows" % (a.out, len(out)))
    return ctx.write_missing(a.out)


if __name__ == "__main__":
    sys.exit(main())
