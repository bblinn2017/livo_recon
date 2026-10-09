#!/usr/bin/env python3
"""r83_pcl_reduce: family D4P (PCA classify-at-fit with larger point counts / always_update), host side. Python only; no node run.

Arms (24): r83_rt_d4_<bag>_pc<n100|n200|a50|a200>_<fq|mq>, bags eee_01, nya_02, sbs_03.
  n100 = convergence_mode normal, voxel_map/points/max 100;  n200 = normal, 200;  a50 = always_update, 50;  a200 = always_update, 200.
Control (in every row): the R82 arm r82_rt_d4_<bag>_pcls_<q> (normal, max 50), read from the R82 phase_summary.

  python3 r83_pcl_reduce.py --phase R83_phase_summary.csv R82_phase_summary.csv --runs <R83 run dir with <arm>/ folders> [<R82 compact>] --out r83_d4p.csv
        [--arm-scan-all arm_scan_all.csv.gz] [--onsets onset_scans.json]
  python3 r83_pcl_reduce.py --selftest

Per-arm inputs read from <runs>/<arm>/: arm_scan.csv(.gz) (column fs_n_residuals, fs_n_voxels, fs_n_voxels_is_plane; or filtered from --arm-scan-all), peak_rss.txt (written by
r83_rss_watch.sh, ONE line `peak_rss_kb=<int> host_ram_kb=<int> abort_threshold_kb=<int>`), the marker file ABORTED_RSS (written by the watcher when it killed the job; the
file holds the RSS in kB at the kill), and optionally residual_dump.csv(.gz) with voxel_point_count (only if the run wrote a dump; the D4P arms do not ask for one).
Columns of r83_d4p.csv, one row per arm (24 rows) with the control in the same row:
  bag, q, variant, convergence_mode, max_points, arm, status ('ok', 'ABORTED_RSS', 'no_phase_row'),
  ate_mm, gt_max_mm, diverged (gt_max_mm > 500), collapse (yes/no/undetermined), n_scans, n_gt_joined, wall_s,
  ctl_arm, ctl_ate_mm, ctl_gt_max_mm, ctl_n_scans, ctl_wall_s, ate_diff_arm_minus_ctl_mm, gt_max_diff_arm_minus_ctl_mm, wall_s_diff_arm_minus_ctl,
  resid_per_scan_median, resid_per_scan_q10, resid_per_scan_q90, n_scans_with_resid (arm_scan rows with a residual count), voxels_per_scan_median, plane_voxels_per_scan_median,
  ctl_resid_per_scan_median, resid_median_diff_arm_minus_ctl (counts, when the control's arm_scan is available),
  vpc_q10, vpc_q50, vpc_q90, vpc_n (voxel_point_count over the iteration-0 rows of the dump, blank if no dump),
  peak_rss_kb, host_ram_kb, abort_threshold_kb, abort_threshold_minus_peak_kb, ctl_peak_rss_kb (if the control folder has a peak_rss.txt, blank otherwise),
  aborted_rss_kb (content of the ABORTED_RSS marker)
A second line printed at the end: the number of ABORTED_RSS markers found under the run roots for these 24 arms (also in <out>_aborted.txt with the arm names).
No arm is filtered out, no pooling over bags, no ratios. Missing inputs (phase row, arm_scan, peak_rss.txt) are named in <out>_missing.txt and the exit code is 1."""
import argparse
import gzip
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd

import r83_common as C

BAGS_D4P = ["eee_01", "nya_02", "sbs_03"]
VARIANTS = {"n100": ("normal", 100), "n200": ("normal", 200), "a50": ("always_update", 50), "a200": ("always_update", 200)}
RSS_RE = re.compile(r"peak_rss_kb=(\d+)\s+host_ram_kb=(\d+)\s+abort_threshold_kb=(\d+)")


def read_rss(path):
    last = None
    with open(path) as f:
        for ln in f:
            m = RSS_RE.search(ln)
            if m:
                last = tuple(int(x) for x in m.groups())
    return last


def load_scan(ctx, arm, all_df):
    p = C.find_file(ctx.roots, arm, "arm_scan.csv.gz")
    if p:
        return pd.read_csv(p)
    if all_df is not None:
        d = all_df[all_df["arm"] == arm]
        if len(d):
            return d
    return None


def resid_stats(d):
    if d is None or "fs_n_residuals" not in d.columns:
        return {}
    r = d["fs_n_residuals"].dropna()
    r = r[r > 0] if (r > 0).any() else r
    o = {}
    if len(r):
        o.update(resid_per_scan_median=float(np.median(r)), resid_per_scan_q10=float(np.quantile(r, .1)), resid_per_scan_q90=float(np.quantile(r, .9)), n_scans_with_resid=int(len(r)))
    for col, key in (("fs_n_voxels", "voxels_per_scan_median"), ("fs_n_voxels_is_plane", "plane_voxels_per_scan_median")):
        if col in d.columns and d[col].notna().any():
            o[key] = float(d[col].dropna().median())
    return o


def vpc_stats(ctx, arm):
    p = C.find_file(ctx.roots, arm, "residual_dump.csv") or C.find_file(ctx.roots, arm, "residual_dump_all.csv.gz")
    if not p:
        return {}
    hdr = pd.read_csv(p, nrows=0).columns
    if "voxel_point_count" not in hdr:
        return {}
    d = pd.read_csv(p, usecols=[c for c in ("iter", "voxel_point_count") if c in hdr])
    if "iter" in d:
        d = d[d["iter"] == 0]
    v = d["voxel_point_count"]
    v = v[v >= 0]
    if not len(v):
        return {}
    return dict(vpc_q10=float(np.quantile(v, .1)), vpc_q50=float(np.quantile(v, .5)), vpc_q90=float(np.quantile(v, .9)), vpc_n=int(len(v)))


def reduce(ctx, all_df=None, bags=BAGS_D4P):
    rows, aborted = [], []
    for bag in bags:
        for q in C.QS:
            ctl = "r82_rt_d4_%s_pcls_%s" % (bag, q)
            ref = "r82_rt_ds_%s_first_%s" % (bag, q)   # healthy reference of the collapse flag: the plain 0.425 m arm of the same bag and Q
            cscan = load_scan(ctx, ctl, all_df)
            cr = resid_stats(cscan)
            cp = C.find_file(ctx.roots, ctl, "peak_rss.txt")
            crss = read_rss(cp) if cp else None
            for v, (mode, mx) in VARIANTS.items():
                arm = "r83_rt_d4_%s_pc%s_%s" % (bag, v, q)
                r = dict(bag=bag, q=q, variant=v, convergence_mode=mode, max_points=mx, arm=arm)
                ab = C.find_file(ctx.roots, arm, "ABORTED_RSS")
                f = {} if (ab and not ctx.phase.has(arm)) else ctx.arm_fields(arm, ref, bag)   # an aborted arm may have no phase row: that is expected, not a missing input
                r["status"] = "ABORTED_RSS" if ab else ("ok" if f else "no_phase_row")
                if ab:
                    aborted.append(arm)
                    try:
                        r["aborted_rss_kb"] = int(re.findall(r"\d+", open(ab).read() or "")[0])
                    except (IndexError, OSError):
                        pass
                r.update(f)
                r["wall_s"] = C.f3(ctx.phase.get(arm, "wall_s"))
                r.update(ctl_arm=ctl)
                if ctx.phase.has(ctl):
                    r.update(ctl_ate_mm=C.f3(ctx.phase.get(ctl, "gt_ate_mm")), ctl_gt_max_mm=C.f3(ctx.phase.get(ctl, "gt_max_mm")), ctl_n_scans=ctx.phase.get(ctl, "n_scans"),
                             ctl_wall_s=C.f3(ctx.phase.get(ctl, "wall_s")))
                    if f:
                        r["ate_diff_arm_minus_ctl_mm"] = C.f3(f["ate_mm"] - r["ctl_ate_mm"])
                        r["gt_max_diff_arm_minus_ctl_mm"] = C.f3(f["gt_max_mm"] - r["ctl_gt_max_mm"])
                        if np.isfinite(r["wall_s"]) and np.isfinite(r["ctl_wall_s"]):
                            r["wall_s_diff_arm_minus_ctl"] = C.f3(r["wall_s"] - r["ctl_wall_s"])
                else:
                    ctx.miss("phase_summary row (control): " + ctl)
                sc = load_scan(ctx, arm, all_df)
                if sc is None:
                    if not ab:
                        ctx.miss("arm_scan.csv[.gz]: " + arm)
                st = resid_stats(sc)
                r.update(st)
                if cr and "resid_per_scan_median" in cr:
                    r["ctl_resid_per_scan_median"] = cr["resid_per_scan_median"]
                    if "resid_per_scan_median" in st:
                        r["resid_median_diff_arm_minus_ctl"] = st["resid_per_scan_median"] - cr["resid_per_scan_median"]
                r.update(vpc_stats(ctx, arm))
                pp = C.find_file(ctx.roots, arm, "peak_rss.txt")
                rss = read_rss(pp) if pp else None
                if rss:
                    r.update(peak_rss_kb=rss[0], host_ram_kb=rss[1], abort_threshold_kb=rss[2], abort_threshold_minus_peak_kb=rss[2] - rss[0])
                else:
                    ctx.miss("peak_rss.txt: " + arm)
                if crss:
                    r["ctl_peak_rss_kb"] = crss[0]
                rows.append(r)
    return pd.DataFrame(rows), aborted


def selftest():
    c = C.Checker()
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(4)
    ph = []

    def mkscan(arm, resid, n=500, gz=True):
        dd = os.path.join(d, arm)
        os.makedirs(dd, exist_ok=True)
        s = pd.DataFrame(dict(scan=np.arange(1, n + 1), fs_n_residuals=rng.poisson(resid, n), fs_n_voxels=5000 + np.arange(n), fs_n_voxels_is_plane=400))
        s.loc[:4, "fs_n_residuals"] = 0   # leading zero rows must not drag the median
        s.to_csv(os.path.join(dd, "arm_scan.csv.gz"), index=False)
        return dd
    for bag in BAGS_D4P:
        for q in C.QS:
            ctl = "r82_rt_d4_%s_pcls_%s" % (bag, q)
            ph.append(dict(arm=ctl, gt_ate_mm=33.0, gt_max_mm=100.0, n_scans=500, wall_s=120.0))
            mkscan(ctl, 1000)
            for v, (mode, mx) in VARIANTS.items():
                arm = "r83_rt_d4_%s_pc%s_%s" % (bag, v, q)
                ph.append(dict(arm=arm, gt_ate_mm=34.0 + mx / 100.0, gt_max_mm=101.0, n_scans=500, wall_s=130.0 + mx / 10.0))
                dd = mkscan(arm, 1000 + 3 * mx)
                with open(os.path.join(dd, "peak_rss.txt"), "w") as f:
                    f.write("peak_rss_kb=%d host_ram_kb=8000000 abort_threshold_kb=4000000\n" % (1000000 + 1000 * mx))
    # planted: aborted arm (no phase row, marker, rss line), arm with a dump, arm lacking peak_rss
    ph = pd.DataFrame(ph)
    ab = "r83_rt_d4_sbs_03_pca200_mq"
    ph = ph[ph.arm != ab]
    open(os.path.join(d, ab, "ABORTED_RSS"), "w").write("4100000\n")
    open(os.path.join(d, ab, "peak_rss.txt"), "w").write("peak_rss_kb=4100000 host_ram_kb=8000000 abort_threshold_kb=4000000\n")
    os.remove(os.path.join(d, "r83_rt_d4_eee_01_pcn100_fq", "peak_rss.txt"))
    dumparm = "r83_rt_d4_nya_02_pcn200_fq"
    pd.DataFrame(dict(iter=[0] * 100 + [1] * 10, voxel_point_count=list(range(100)) + [999] * 10)).to_csv(os.path.join(d, dumparm, "residual_dump.csv"), index=False)
    pc = os.path.join(d, "ph.csv")
    ph.to_csv(pc, index=False)
    ctx = C.Ctx([pc], [d])
    R, aborted = reduce(ctx)
    c("24 rows", len(R) == 24, str(len(R)))
    r = R[(R.bag == "eee_01") & (R.q == "mq") & (R.variant == "n200")].iloc[0]
    c("difference arm minus control in mm", abs(r.ate_diff_arm_minus_ctl_mm - 3.0) < 1e-9 and r.ctl_ate_mm == 33.0 and r.n_scans == 500 and r.ctl_n_scans == 500, "%.3f" % r.ate_diff_arm_minus_ctl_mm)
    c("residual median near 1600 (leading zero rows excluded)", abs(r.resid_per_scan_median - 1600) < 25 and abs(r.resid_median_diff_arm_minus_ctl - 600) < 40, "%.1f" % r.resid_per_scan_median)
    c("peak rss parsed with threshold difference", r.peak_rss_kb == 1200000 and r.abort_threshold_minus_peak_kb == 2800000, str(r.peak_rss_kb))
    c("wall time difference", abs(r.wall_s_diff_arm_minus_ctl - 30.0) < 1e-9)
    x = R[R.arm == ab].iloc[0]
    c("aborted arm kept with status and marker kb, no phase row", x.status == "ABORTED_RSS" and x.aborted_rss_kb == 4100000 and x.peak_rss_kb == 4100000, x.status)
    c("aborted count and name", aborted == [ab], str(aborted))
    c("aborted arm without phase row is not listed as missing", not any(ab in m and m.startswith("phase_summary") for m in ctx.missing))
    c("missing peak_rss named", "peak_rss.txt: r83_rt_d4_eee_01_pcn100_fq" in ctx.missing, str(ctx.missing))
    y = R[R.arm == dumparm].iloc[0]
    c("voxel_point_count quantiles from the dump (iteration 0 only)", y.vpc_n == 100 and abs(y.vpc_q50 - 49.5) < 1e-9, "n %s q50 %s" % (y.vpc_n, y.vpc_q50))
    c("no dump gives blank vpc", np.isnan(R[R.arm == "r83_rt_d4_eee_01_pcn200_fq"].iloc[0].get("vpc_q50", np.nan)))
    c("no ratio columns", not any("ratio" in k or "pct" in k or "percent" in k for k in R.columns))
    import subprocess
    o = os.path.join(d, "o_d4p.csv")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--phase", pc, "--runs", d, "--out", o], capture_output=True, text=True)
    c("CLI writes the table and the aborted list, exit 1 (peak_rss missing)", p.returncode == 1 and len(pd.read_csv(o)) == 24 and "ABORTED_RSS markers: 1" in open(os.path.join(d, "o_d4p_aborted.txt")).read(), (p.stdout + p.stderr).strip()[:100])
    return c.done()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--arm-scan-all", default=None, help="arm_scan_all.csv.gz (used for an arm with no arm_scan.csv[.gz] folder)")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.phase and a.out):
        ap.error("--phase and --out are required")
    ctx = C.Ctx(a.phase, a.runs, a.onsets)
    all_df = pd.read_csv(a.arm_scan_all, usecols=lambda c: c in ("arm", "scan", "fs_n_residuals", "fs_n_voxels", "fs_n_voxels_is_plane")) if a.arm_scan_all else None
    R, aborted = reduce(ctx, all_df)
    R.to_csv(a.out, index=False)
    ap_path = os.path.splitext(a.out)[0] + "_aborted.txt"
    with open(ap_path, "w") as f:
        f.write("ABORTED_RSS markers: %d\n" % len(aborted))
        f.writelines(x + "\n" for x in aborted)
    print("%s: %d rows; ABORTED_RSS markers: %d" % (a.out, len(R), len(aborted)))
    return ctx.write_missing(a.out)


if __name__ == "__main__":
    sys.exit(main())
