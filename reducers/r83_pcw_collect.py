#!/usr/bin/env python3
"""r83_pcw_collect: host reducer of family D1B (point-covariance window, patch r83_p4). Python only; no node run.

Arms (72): r83_rt_pc_<bag>_w<5|10|20|50>ms_<fq|mq>, nine bags x window {0.005, 0.010, 0.020, 0.050} s x Q (key lio/point_cov/window_s).
For EVERY arm it
  (1) runs r83_cov_consistency.py v2 (unchanged, through its Python API: analyse() with max_gt_gap 0.15, pipeline_scan = that arm's arm_scan, bins 100) on the arm's
      state_trace with the NTU lever -0.293656 -0.012288 -0.273095 and the onset scan of the bag (onset_scans.json, key onset_scan_used), writes ONE rows file and ONE
      bins file per arm (<out-dir>/rows/<arm>.csv, <out-dir>/bins/<arm>.csv), then r83_cov_collect.collect() into r83cov_d1b_rows.csv and r83cov_d1b_bins.csv;
  (2) summarises the arm's point_cov_window.csv (written per scan by the node when window_s > 0): median over scans of every numeric column
      (window_s, n_steps, sd_p_med_m, sd_p_p95_m, sd_p_max_m, sd_r_med_rad, sd_r_p95_rad, sd_r_max_rad) as med_<column>, and n = number of rows (scans);
      plus window_s_matches_name (1 if the median window_s equals the window in the arm name to 1e-9);
  (3) parses the timer `imu/window_cov_profile` (name, total, mean): FIRST from the profiler export file in the arm folder (default name profiler.txt; the node writes
      it at the end of the run when outputs.profiler.export_path points there; format `name<spaces>mean ms  (N frames)`, mean over the frames in which the timer ran;
      total = mean x N), ELSE from the per-frame lines `  imu/window_cov_profile: <x> ms` of run.log (printed only when outputs.printer.profiler.enable is true; the
      printer is throttled, so these are SAMPLES: timer_n counts the lines found, timer_total_ms is their sum, and timer_source says 'run_log_samples').
      timer_source is 'none' (and the arm is listed) when neither file holds the timer.

  python3 r83_pcw_collect.py --runs <runs dir with <arm>/ folders> --gt-template '/path/to/leica/{bag}.txt' --onsets onset_scans.json --out-dir <dir>
        [--lever "-0.293656 -0.012288 -0.273095"] [--bags eee_01 ...] [--arm-scan-all arm_scan_all.csv.gz] [--profiler-name profiler.txt]
  python3 r83_pcw_collect.py --selftest

Writes <out-dir>/r83cov_d1b_rows.csv, r83cov_d1b_bins.csv (columns of r83_cov_consistency v2: arm, window S/W/M/all, n, rms3d_err_mm, pipe_rms_err_mm, nees_pos_med,
cover99_pos, ... one row per window; bins: 100-scan blocks), r83_pcw_window_summary.csv (one row per arm: bag, q, window_s_name, n, med_*, window_s_matches_name,
timer_source, timer_name, timer_total_ms, timer_mean_ms, timer_n), and r83_pcw_missing.txt. Missing or failed inputs are listed BY NAME (state_trace, ground-truth file,
arm_scan, point_cov_window.csv, timer, and the analyse() error text); nothing is skipped silently; exit code 1 if any item is listed.
No ratios, no pooling over bags: every row is one arm (and one window of it)."""
import argparse
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd

import r83_common as C
import r83_cov_consistency as CC
import r83_cov_collect as CK

NTU_LEVER = "-0.293656 -0.012288 -0.273095"
WINDOWS_MS = [5, 10, 20, 50]
TIMER = "imu/window_cov_profile"
WCOLS = ["window_s", "n_steps", "sd_p_med_m", "sd_p_p95_m", "sd_p_max_m", "sd_r_med_rad", "sd_r_p95_rad", "sd_r_max_rad"]
ARM_RE = re.compile(r"^r83_rt_pc_(?P<bag>[a-z]{3}_\d\d)_w(?P<w>\d+)ms_(?P<q>fq|mq)$")


def arm_list(bags):
    return ["r83_rt_pc_%s_w%dms_%s" % (b, w, q) for b in bags for w in WINDOWS_MS for q in C.QS]


def parse_timer(folder, profiler_name):
    """(source, mean_ms, total_ms, n) of TIMER from the arm folder, or ('none', nan, nan, 0)."""
    p = os.path.join(folder, profiler_name)
    if os.path.isfile(p):
        rx = re.compile(r"^\s*" + re.escape(TIMER) + r"\s+([0-9.eE+-]+)\s*ms\s*\(\s*(\d+)\s*frames?\s*\)")
        for ln in open(p, errors="replace"):
            m = rx.search(ln)
            if m:
                mean, n = float(m.group(1)), int(m.group(2))
                return "profiler_export", mean, mean * n, n
    for lg in ("run.log", "run.log.gz"):
        p = os.path.join(folder, lg)
        if os.path.isfile(p):
            import gzip
            op = gzip.open if lg.endswith(".gz") else open
            rx = re.compile(re.escape(TIMER) + r":\s*([0-9.eE+-]+)\s*ms")
            v = []
            with op(p, "rt", errors="replace") as f:
                for ln in f:
                    m = rx.search(ln)
                    if m:
                        v.append(float(m.group(1)))
            if v:
                return "run_log_samples", float(np.mean(v)), float(np.sum(v)), len(v)
    return "none", np.nan, np.nan, 0


def run_arm(arm, a, ctx_missing, all_df):
    m = ARM_RE.match(arm)
    bag, q, w = m.group("bag"), m.group("q"), int(m.group("w"))
    folder = None
    for r in a.runs:
        for sub in (os.path.join(r, arm), os.path.join(r, "compact", arm)):
            if os.path.isdir(sub):
                folder = sub
                break
        if folder:
            break
    summ = dict(arm=arm, bag=bag, q=q, window_s_name=w / 1000.0)
    if folder is None:
        ctx_missing.append("arm folder: " + arm)
        return None, None, summ
    roots = [os.path.dirname(folder)]
    # point_cov_window.csv
    pw = C.find_file(roots, arm, "point_cov_window.csv") or os.path.join(folder, "point_cov_window.csv")
    if os.path.isfile(pw) and os.path.getsize(pw) > 0:
        d = pd.read_csv(pw)
        summ["n"] = len(d)
        for c in WCOLS:
            if c in d.columns:
                summ["med_" + c] = float(d[c].median())
        if "window_s" in d.columns:
            summ["window_s_matches_name"] = int(abs(float(d["window_s"].median()) - w / 1000.0) < 1e-9)
        else:
            ctx_missing.append("point_cov_window.csv has no window_s column: " + arm)
    else:
        ctx_missing.append("point_cov_window.csv: " + arm)
    src, mean, tot, n = parse_timer(folder, a.profiler_name)
    summ.update(timer_source=src, timer_name=TIMER, timer_total_ms=tot, timer_mean_ms=mean, timer_n=n)
    if src == "none":
        ctx_missing.append("timer %s in %s or run.log: %s" % (TIMER, a.profiler_name, arm))
    # covariance consistency
    tp = C.find_file(roots, arm, "state_trace.csv.gz")
    if tp is None:
        ctx_missing.append("state_trace.csv[.gz]: " + arm)
        return None, None, summ
    gt = a.gt_template.format(bag=bag)
    if not os.path.isfile(gt):
        ctx_missing.append("ground-truth file %s (for %s)" % (gt, arm))
        return None, None, summ
    ps = C.find_file(roots, arm, "arm_scan.csv.gz")
    tmp = None
    if ps is None and all_df is not None and (all_df["arm"] == arm).any():
        tmp = os.path.join(a.out_dir, "_tmp_arm_scan_%s.csv" % arm)
        all_df[all_df["arm"] == arm].to_csv(tmp, index=False)
        ps = tmp
    if ps is None:
        ctx_missing.append("arm_scan.csv[.gz] (pipeline gt_err_mm): " + arm)
        return None, None, summ
    try:
        rows, bins = CC.analyse(CC.load_trace(tp), gt, np.array([float(v) for v in a.lever.split()]), a.onsets_d.get(bag, 0), arm, 0.15, ps, 100)
    except Exception as e:   # named, never silent
        ctx_missing.append("r83_cov_consistency failed for %s: %s: %s" % (arm, type(e).__name__, e))
        return None, None, summ
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
    return rows, bins, summ


def collect(a):
    os.makedirs(os.path.join(a.out_dir, "rows"), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "bins"), exist_ok=True)
    a.onsets_d = C.load_onsets(a.onsets)
    all_df = pd.read_csv(a.arm_scan_all, usecols=lambda c: c in ("arm", "scan", "scan_id", "gt_err_mm")) if a.arm_scan_all else None
    arms = arm_list(a.bags) if not a.arms_file else [l.strip() for l in open(a.arms_file) if l.strip()]
    missing, summ, rf, bf = [], [], [], []
    for arm in arms:
        if not ARM_RE.match(arm):
            missing.append("arm name not a D1B arm name: " + arm)
            continue
        rows, bins, s = run_arm(arm, a, missing, all_df)
        summ.append(s)
        if rows is not None and len(rows):
            p = os.path.join(a.out_dir, "rows", arm + ".csv")
            rows.to_csv(p, index=False)
            rf.append(p)
            if bins is not None and len(bins):
                p2 = os.path.join(a.out_dir, "bins", arm + ".csv")
                bins.to_csv(p2, index=False)
                bf.append(p2)
        elif rows is not None:
            missing.append("r83_cov_consistency gave no window with >= 5 joined scans: " + arm)
    nr = nb = 0
    if rf:
        nr = CK.collect(rf, os.path.join(a.out_dir, "r83cov_d1b_rows.csv"))
    else:
        missing.append("no rows file at all: r83cov_d1b_rows.csv not written")
    if bf:
        nb = CK.collect(bf, os.path.join(a.out_dir, "r83cov_d1b_bins.csv"))
    else:
        missing.append("no bins file at all: r83cov_d1b_bins.csv not written")
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(a.out_dir, "r83_pcw_window_summary.csv"), index=False)
    mp = os.path.join(a.out_dir, "r83_pcw_missing.txt")
    if missing:
        open(mp, "w").write("\n".join(missing) + "\n")
        print("MISSING OR FAILED (%d), listed in %s:" % (len(missing), mp), *missing[:40], sep="\n  ", file=sys.stderr)
    elif os.path.exists(mp):
        os.remove(mp)
    print("arms %d, rows file %d rows, bins file %d rows, summary %d rows, missing/failed items %d" % (len(arms), nr, nb, len(S), len(missing)))
    return 1 if missing else 0, S, missing


# ------------------------------------------------------------------ selftest
def make_trace(scale, seed, n=700):
    rng = np.random.default_rng(seed)
    t = 1.6e9 + 0.1 * np.arange(1, n + 1)
    p = np.stack([np.cumsum(rng.normal(size=n)) * 0.05, np.cumsum(rng.normal(size=n)) * 0.05, np.zeros(n)], 1)
    sd_p, sd_r = 0.01, 0.003
    ep = rng.normal(size=(n, 3)) * sd_p * scale
    pest = p - ep
    cols = dict(run_id="x", scan_id=np.arange(1, n + 1), t_abs=t, phase="post_lio", qw=1.0, qx=0.0, qy=0.0, qz=0.0, px=pest[:, 0], py=pest[:, 1], pz=pest[:, 2])
    for i in range(18):
        for j in range(i, 18):
            cols["P_%d_%d" % (i, j)] = (sd_r ** 2 if i < 3 else (sd_p ** 2 if i < 6 else 1.0)) if i == j else 0.0
    tg = 1.6e9 + 0.05 * np.arange(2, 2 * n + 1)
    pg = np.stack([np.interp(tg, t, p[:, k]) for k in range(3)], 1)
    gt = np.column_stack([tg, pg, np.tile([0, 0, 0, 1.0], (len(tg), 1))])
    return pd.DataFrame(cols), gt, ep


def selftest():
    c = C.Checker()
    d = tempfile.mkdtemp()
    runs, out = os.path.join(d, "runs"), os.path.join(d, "out")
    bags = ["eee_01", "sbs_02"]
    onsets = os.path.join(d, "onset_scans.json")
    import json
    json.dump({"onset_scan_used": C.ONSETS}, open(onsets, "w"))
    for b in bags:
        tr, gt, _ = make_trace(1.0, 1)
        np.savetxt(os.path.join(d, b + ".txt"), gt, fmt="%.12f")
    rng = np.random.default_rng(0)
    for k, arm in enumerate(arm_list(bags)):
        m = ARM_RE.match(arm)
        w = int(m.group("w")) / 1000.0
        af = os.path.join(runs, arm)
        os.makedirs(af)
        tr, gt, ep = make_trace(1.0 + 0.1 * k, k)
        tr.to_csv(os.path.join(af, "state_trace.csv.gz"), index=False)
        pd.DataFrame(dict(scan=np.arange(1, 701), gt_err_mm=1000 * np.linalg.norm(ep, axis=1))).to_csv(os.path.join(af, "arm_scan.csv.gz"), index=False)
        n = 700
        pd.DataFrame(dict(scan_index=np.arange(1, n + 1), t_abs=tr.t_abs, t_rel_s=0.1 * np.arange(1, n + 1), window_s=w, n_steps=40, sd_p_med_m=0.001 * (1 + w * 10) + 0 * np.arange(n),
                          sd_p_p95_m=0.002, sd_p_max_m=0.003, sd_r_med_rad=1e-4, sd_r_p95_rad=2e-4, sd_r_max_rad=3e-4)).to_csv(os.path.join(af, "point_cov_window.csv"), index=False)
        if k % 2 == 0:
            open(os.path.join(af, "profiler.txt"), "w").write("%-40s%.3f ms  (%d frames)\n%-40s%.3f ms  (%d frames)\n" % ("imu/propagate", 1.5, 399, TIMER, 0.25 + k * 0.01, 380))
        else:
            open(os.path.join(af, "run.log"), "w").write("[profiler]\n  imu: 3.0 ms\n  %s: 0.2 ms\n  total: 9.0 ms\n[profiler]\n  %s: 0.4 ms\n" % (TIMER, TIMER))
    # planted failures: no state trace; no ground truth for a bag handled below; no point_cov_window; no timer anywhere
    a_nost = "r83_rt_pc_eee_01_w5ms_fq"
    os.remove(os.path.join(runs, a_nost, "state_trace.csv.gz"))
    a_nopcw = "r83_rt_pc_eee_01_w10ms_fq"
    os.remove(os.path.join(runs, a_nopcw, "point_cov_window.csv"))
    a_notimer = "r83_rt_pc_eee_01_w20ms_mq"
    for fn in ("profiler.txt", "run.log"):
        if os.path.exists(os.path.join(runs, a_notimer, fn)):
            os.remove(os.path.join(runs, a_notimer, fn))
    ns = argparse.Namespace(runs=[runs], gt_template=os.path.join(d, "{bag}.txt"), onsets=onsets, out_dir=out, lever="0 0 0", bags=bags, arm_scan_all=None, profiler_name="profiler.txt", arms_file=None)
    rc, S, missing = collect(ns)
    rows = pd.read_csv(os.path.join(out, "r83cov_d1b_rows.csv"))
    bins = pd.read_csv(os.path.join(out, "r83cov_d1b_bins.csv"))
    arms_done = set(rows.arm)
    c("16 arms in the summary", len(S) == 16)
    c("rows file holds 15 arms (one has no state trace)", len(arms_done) == 15 and a_nost not in arms_done, str(len(arms_done)))
    c("rows have the three windows S/W/M for eee_01 and pipeline columns", set(rows[rows.arm == "r83_rt_pc_eee_01_w50ms_fq"].window) == {"S", "W", "M"} and rows.pipe_rms_err_mm.notna().all())
    c("one rows file and one bins file per arm", len(os.listdir(os.path.join(out, "rows"))) == 15 and len(os.listdir(os.path.join(out, "bins"))) == 15)
    c("bins file has bin columns", {"bin_scan_lo", "bin_scan_hi", "bin_start_minus_onset"} <= set(bins.columns) and len(bins) > 15)
    c("onset used: eee_01 bin_start_minus_onset = lo - 485", (bins[bins.arm.str.contains("eee_01")].bin_start_minus_onset == bins[bins.arm.str.contains("eee_01")].bin_scan_lo - 485).all())
    c("missing state_trace listed by name", "state_trace.csv[.gz]: " + a_nost in missing)
    c("missing point_cov_window listed by name", "point_cov_window.csv: " + a_nopcw in missing)
    c("missing timer listed by name", any(a_notimer in m and "timer" in m for m in missing))
    c("exit code 1 when anything is listed", rc == 1 and os.path.exists(os.path.join(out, "r83_pcw_missing.txt")))
    s = S[S.arm == "r83_rt_pc_sbs_02_w20ms_fq"].iloc[0]
    c("window summary median and n", s.n == 700 and abs(s.med_sd_p_med_m - 0.001 * (1 + 0.02 * 10)) < 1e-12 and s.window_s_matches_name == 1, "n %s med %.6f" % (s.n, s.med_sd_p_med_m))
    k = list(S.arm).index("r83_rt_pc_sbs_02_w20ms_fq")
    c("timer from the profiler export: mean, N, total = mean x N", s.timer_source == ("profiler_export" if k % 2 == 0 else "run_log_samples"), "%s %s" % (s.timer_source, k))
    e = S[S.arm == "r83_rt_pc_eee_01_w50ms_fq"].iloc[0]   # k = 6 -> export
    c("export timer values", e.timer_source == "profiler_export" and e.timer_n == 380 and abs(e.timer_mean_ms - 0.31) < 1e-9 and abs(e.timer_total_ms - 0.31 * 380) < 1e-6, "%s %s %s" % (e.timer_source, e.timer_n, e.timer_mean_ms))
    o = S[S.arm == "r83_rt_pc_eee_01_w10ms_mq"].iloc[0]    # k = 3 -> run.log
    c("run.log timer: sample mean 0.3, n 2, total 0.6", o.timer_source == "run_log_samples" and o.timer_n == 2 and abs(o.timer_mean_ms - 0.3) < 1e-9 and abs(o.timer_total_ms - 0.6) < 1e-9)
    c("default NTU lever string", NTU_LEVER == "-0.293656 -0.012288 -0.273095")
    # a bag without a GT file: named, other arms unaffected
    ns2 = argparse.Namespace(**{**vars(ns), "gt_template": os.path.join(d, "{bag}.txt").replace(d, d + "/nonexistent"), "out_dir": os.path.join(d, "out2"), "bags": ["eee_01"]})
    rc2, S2, miss2 = collect(ns2)
    c("no GT file: every arm with a trace named, exit 1, no rows file written", rc2 == 1 and sum("ground-truth file" in m for m in miss2) == 7 and any("no rows file at all" in m for m in miss2))
    # too-short GT span: analyse() error is named
    short = os.path.join(d, "short.txt")
    np.savetxt(short, np.loadtxt(os.path.join(d, "eee_01.txt"))[:10], fmt="%.12f")
    ns3 = argparse.Namespace(**{**vars(ns), "gt_template": short, "out_dir": os.path.join(d, "out3"), "bags": ["eee_01"]})
    rc3, S3, miss3 = collect(ns3)
    c("analyse error text named per arm", any("r83_cov_consistency failed for r83_rt_pc_eee_01_w5ms_mq" in m and "fewer than 20 joined scans" in m for m in miss3))
    import subprocess
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--runs", runs, "--gt-template", os.path.join(d, "{bag}.txt"), "--onsets", onsets, "--out-dir", os.path.join(d, "out4"), "--lever", "0 0 0",
                        "--bags", "eee_01"], capture_output=True, text=True)
    c("CLI runs and writes the collected files", os.path.exists(os.path.join(d, "out4", "r83cov_d1b_rows.csv")) and os.path.exists(os.path.join(d, "out4", "r83_pcw_window_summary.csv")), (p.stdout + p.stderr).strip()[-100:])
    return c.done()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--gt-template", help="path with {bag}, e.g. /data/ntu/{bag}/leica.txt (TUM: t px py pz qx qy qz qw)")
    ap.add_argument("--onsets", default=None)
    ap.add_argument("--out-dir")
    ap.add_argument("--lever", default=NTU_LEVER)
    ap.add_argument("--bags", nargs="*", default=C.BAGS)
    ap.add_argument("--arms-file", default=None, help="one arm name per line instead of the 72 generated names")
    ap.add_argument("--arm-scan-all", default=None)
    ap.add_argument("--profiler-name", default="profiler.txt")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.runs and a.gt_template and a.out_dir):
        ap.error("--runs, --gt-template and --out-dir are required")
    os.makedirs(a.out_dir, exist_ok=True)
    return collect(a)[0]


if __name__ == "__main__":
    sys.exit(main())
