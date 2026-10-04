#!/usr/bin/env python3
"""R67: gather every reduced arm into compact return tables. Nothing here interprets results.

python3 r67_collect.py --runs <runs_dir> --arms r67_manifest/arm_table.tsv --out <out_dir> [--level 0] [--params bag_params.json]
Writes into <out_dir>:
  phase_summary.csv            one row per arm (flat reducer metrics) plus the alignment-free deviation-from-baseline columns (below)
  arm_scan_all.csv.gz          every arm's per-scan table THINNED by --level (levels below). Column dev_from_baseline_mm added. Exact scan times in column t (4 decimals).
  arm_ol_tau_all.csv.gz, plane_fit_reasons_all.csv.gz, plane_fit_trace_counts_all.csv.gz, completeness.csv, summaries_all.json (every arm_summary.json in one file)
  compact/<arm>/...            the r67_compress.py outputs of each keep arm
  compress_manifests.json, sensor_stamps_summaries.json, identity_controls.txt, duplicates.tsv (configurations not written twice: which earlier arm they equal)
THINNING LEVELS of arm_scan_all.csv.gz (scan numbers; "onset window" = onset-20 .. onset+115 scans of that sequence, eee_01: 470..600; a keep arm is one
with keep_raw=1 in the arm table). Every level keeps scans 1..12 of every arm and the whole onset window of every arm at full resolution.
  level 0: outside the window every 25th scan;  keep arms complete (every scan).
  level 1: every 25th scan;  keep arms every 5th scan.
  level 2: every 50th scan;  keep arms every 10th scan.
  level 3: every 100th scan; keep arms every 25th scan; onset window onset-20 .. onset+70.
SUFFICIENCY: settled offset, wander, excursion, stationary NEES, ATE (evo and alignment), first-deviation scans and onset traces are computed from the FULL per-scan table
by r67_reduce.py before thinning and are in phase_summary.csv; the thinned table is for plotting drift, for my own re-derivations inside the onset window, and for
checks. Not recoverable from a thinned table: per-scan values outside the window for non-keep arms (kept on the host with sha256 in raw_sha256.txt).
DEVIATION FROM BASELINE (alignment-free): dev_from_baseline_mm[scan] = || ep(arm, scan) - ep(baseline, scan) || in mm (ep = displacement from the arm's own first-scan
reference). Baseline = 00_id_baseline_full (eee_01 full), 00_id_baseline_60s (60 s), 00_id_baseline_48s (48 s), E_<seq>_baseline_full (other sequences).
Summary columns: dev_max_mm, dev_max_scan, dev_first_scan_above_<X>mm (X = 25,50,100,200,500; scans after the onset scan), dev_at_scan_<S>, dev_rms_scans_541_on_mm."""
import argparse, glob, gzip, json, os, shutil
import numpy as np
import pandas as pd

AX = ["Rx", "Ry", "Rz", "px", "py", "pz", "vx", "vy", "vz"]
BASE = {"full": "00_id_baseline_full", "60": "00_id_baseline_60s", "48": "00_id_baseline_48s"}
R64_DIGESTS = {"00_id_baseline_48s": "197d7f00e23f45160a1a7efbbc9e86dc85bfae069140d1eb3790f3dc75500341",
               "00_id_baseline_60s": "a096c8b9e581f3c47bfa184c2909aea450ff74700a77a766ab93fded3391c8ad",
               "00_id_baseline_full": "2d3eebfb2a84ae0fe8e03222232ea814bc323e02027de395681b54f32b4f6857"}
LEVELS = {0: (25, None, 0), 1: (25, 5, 0), 2: (50, 10, 0), 3: (100, 25, 45)}   # (step, keep-arm step (None = every scan), onset-window trim after onset+70)

def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d

def flat(S):
    m = S.get("meta", {})
    r = {"arm": S.get("arm"), "kind": S.get("kind")}
    for k in ("tier", "duration_s", "exit_code", "wall_s", "ate_m"):
        r[k] = m.get(k)
    for k in ("n_scans", "n_stationary_scans", "evo_ate_m", "evo_n", "gt_status", "gt_join_scans", "gt_ate_mm", "gt_max_mm", "gt_max_scan",
              "excursion_mm", "excursion_scan", "settled_offset_mm", "wander_mm", "disturbance_mm", "disturbance_scan",
              "ol_n_windows", "ol_nees9_mean_tau_ge_1s"):
        r[k] = S.get(k)
    for ph in ("post_lio", "post_imu"):
        b = g(S, "stationary", ph) or {}
        for grp in ("pos", "rot", "vel"):
            if grp in b:
                r["st_%s_%s_rms" % (ph[5:], grp)] = b[grp]["rms"]
                r["st_%s_%s_e2_over_P" % (ph[5:], grp)] = b[grp]["e2_over_P"]
                r["st_%s_%s_mean_sd_norm" % (ph[5:], grp)] = b[grp].get("mean_sd_norm")
        if "nees9" in b:
            r["st_%s_nees9_mean" % ph[5:]] = b["nees9"]["mean"]
            r["st_%s_nees9_frac_in95" % ph[5:]] = b["nees9"]["frac_in_95"]
        for ax in AX:
            if "axes" in b:
                r["st_%s_e2_over_P_%s" % (ph[5:], ax)] = b["axes"][ax]["e2_over_P"]
    for nm in ("transition", "motion", "stationary"):
        r["gt_ate_%s_mm" % nm] = g(S, "gt_ate_by_segment_mm", nm, "rms")
    for k, v in (S.get("gt_rms_by_500_scan_block_mm") or {}).items():
        r["gt_block_%s_mm" % k] = v
    for k, v in (S.get("ol_ratio_tau_ge_1s") or {}).items():
        r["ol_e2_over_P_%s" % k] = v
    it = S.get("iterations") or {}
    r["iters_max"], r["iters_frac_at_max"], r["iters_median"] = it.get("max"), it.get("frac_at_max"), it.get("median")
    for nm in ("stationary", "transition", "motion"):
        r["resid_median_%s" % nm] = g(S, "residuals", nm, "median")
        for k, v in (g(S, "frame_stats", nm) or {}).items():
            r["%s_%s" % (k, nm)] = v
        for k, v in (g(S, "point_cov_stats", nm) or {}).items():
            r["pcs_%s_%s" % (k, nm)] = v
    for nm in ("scan1", "stationary", "motion"):
        for k, v in (g(S, "motion_q", nm) or {}).items():
            r["mq_%s_%s" % (nm, k)] = v
    for k, v in (S.get("plane_fit_reasons") or {}).items():
        for c, n in v.items():
            r["pfr_%s_%s" % (k.replace("/", "_"), c)] = n
    r["plane_fit_trace_rows"] = S.get("plane_fit_trace_rows")
    for k in ("coupled_status", "akf_q", "frame_window", "r67_engagement_from_run_log"):
        for kk, vv in (S.get(k) or {}).items():
            r["%s_%s" % (k, kk)] = (vv if not isinstance(vv, list) else " | ".join(vv))
    r["seq"] = S.get("seq")
    r["onset_scan_used"] = g(S, "definitions", "onset_scan")
    r["stat_last_used"] = g(S, "definitions", "stat_last")
    r["n_missing"] = len(S.get("missing", []))
    r["n_warnings"] = len(S.get("warnings", []))
    return r

def thin(df, keep_arm, level, onset):
    step, kstep, trim = LEVELS[level]
    st = (kstep if keep_arm else step)
    sc = df.scan
    win_hi = onset + (115 - trim)
    m = (sc <= 12) | ((sc >= onset - 20) & (sc <= win_hi))
    if st is None:
        return df
    return df[m | (sc % st == 0)]

def dev_table(df, base):
    """alignment-free deviation of df's ep from base's ep on common scans."""
    a = df.set_index("scan")[["ep_x_mm", "ep_y_mm", "ep_z_mm"]]
    b = base.set_index("scan")[["ep_x_mm", "ep_y_mm", "ep_z_mm"]]
    idx = a.index.intersection(b.index)
    d = pd.Series(np.nan, index=a.index)
    d.loc[idx] = np.linalg.norm(a.loc[idx].to_numpy() - b.loc[idx].to_numpy(), axis=1)
    return d

def dev_summary(d):
    r = {}
    if d.notna().sum() == 0:
        return r
    r["dev_max_mm"], r["dev_max_scan"] = float(d.max()), int(d.idxmax())
    late = d[d.index > 481].dropna()
    for x in (25, 50, 100, 200, 500):
        ex = late[late > x]
        r["dev_first_scan_above_%dmm" % x] = int(ex.index[0]) if len(ex) else None
    for s in (500, 520, 540, 600, 800, 1000, 1500, 2000, 3000, 3900):
        r["dev_at_scan_%d" % s] = float(d.loc[s]) if s in d.index and np.isfinite(d.loc[s]) else None
    m = d[d.index >= 541].dropna()
    if len(m):
        r["dev_rms_scans_541_on_mm"] = float(np.sqrt((m ** 2).mean()))
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--arms", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--level", type=int, default=0, choices=sorted(LEVELS)); ap.add_argument("--params", default=None)
    a = ap.parse_args()
    pj = json.load(open(a.params)) if a.params and os.path.exists(a.params) else {}
    os.makedirs(os.path.join(a.out, "summaries"), exist_ok=True)
    plan = pd.read_csv(a.arms, sep="\t")
    plan[plan.kind == "dup"][["arm", "dup_of", "notes"]].to_csv(os.path.join(a.out, "duplicates.tsv"), sep="\t", index=False)
    plan = plan[plan.kind != "dup"].reset_index(drop=True)
    summaries_all = {}
    rows, comp, scans, ols, reasons, tcounts = [], [], [], [], [], []
    sha, manifests, stamps = {}, {}, {}
    base_tabs = {}
    def load_base(nm):
        fp = os.path.join(a.runs, nm, "arm_scan.csv")
        return pd.read_csv(fp, dtype={"t": str}) if os.path.exists(fp) else None
    for key, nm in BASE.items():
        base_tabs[key] = load_base(nm)
    for sq in sorted(set(plan.seq) - {"eee_01"}):
        base_tabs["seq:" + sq] = load_base("50_E_%s_baseline_full" % sq)
    for _, p in plan.iterrows():
        D = os.path.join(a.runs, p.arm)
        sj = os.path.join(D, "arm_summary.json")
        meta = json.load(open(os.path.join(D, "arm.json"))) if os.path.exists(os.path.join(D, "arm.json")) else {}
        shp = os.path.join(D, "state_trace_blank_runid.sha256")
        sha[p.arm] = open(shp).read().strip() if os.path.exists(shp) else ""
        comp.append({"arm": p.arm, "tier": p.tier, "kind": p.kind, "duration_s": p.duration_s, "has_dir": os.path.isdir(D), "has_summary": os.path.exists(sj),
                     "exit_code": meta.get("exit_code"), "wall_s": meta.get("wall_s"), "missing": " ".join(meta.get("missing", [])),
                     "state_trace_sha256_run_id_blanked": sha[p.arm]})
        cm = os.path.join(D, "compress_manifest.json")
        if os.path.exists(cm):
            manifests[p.arm] = json.load(open(cm))
        # compact outputs of r65_compress.py
        cdir = os.path.join(a.out, "compact", p.arm)
        for pat in ("*_compact.csv.gz", "*_windows.csv.gz", "*_blocks.csv", "*_all.csv.gz", "plane_trace_window.csv.gz", "sensor_stamps_summary.json", "compress_manifest.json"):
            for fp in glob.glob(os.path.join(D, pat)):
                os.makedirs(cdir, exist_ok=True)
                shutil.copy(fp, cdir)
        ssj = os.path.join(D, "sensor_stamps_summary.json")
        if os.path.exists(ssj):
            stamps[p.arm] = json.load(open(ssj))
        if not os.path.exists(sj):
            continue
        S = json.load(open(sj))
        row = flat(S)
        summaries_all[p.arm] = S
        full = str(p.duration_s) == "full"
        keep_arm = int(p.keep_raw) == 1
        onset = int((pj.get(p.seq) or {}).get("onset_scan", 485)) if p.seq != "eee_01" else 485
        for fn, acc in (("arm_scan.csv", scans), ("arm_ol_tau.csv", ols), ("plane_fit_reasons.csv", reasons), ("plane_fit_trace_counts.csv", tcounts)):
            fp = os.path.join(D, fn)
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp, dtype={"t": str})   # keep the scan time as the string the reducer wrote
            if fn == "arm_scan.csv":
                key = ("seq:" + p.seq) if p.seq != "eee_01" else ("full" if full else str(p.duration_s))
                bt = base_tabs.get(key)
                if bt is not None and p.kind != "open" and p.tier != "RF" and (p.tier != "SMK" or not full):
                    d = dev_table(df, bt)
                    df["dev_from_baseline_mm"] = df.scan.map(d)
                    row.update(dev_summary(d))
                df = thin(df, keep_arm, a.level, onset) if full else df
            df.insert(0, "arm", p.arm)
            acc.append(df)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "phase_summary.csv"), index=False)
    pd.DataFrame(comp).to_csv(os.path.join(a.out, "completeness.csv"), index=False)
    for nm, acc in (("arm_scan_all", scans), ("arm_ol_tau_all", ols), ("plane_fit_reasons_all", reasons), ("plane_fit_trace_counts_all", tcounts)):
        if acc:
            big = pd.concat(acc)
            big.to_csv(os.path.join(a.out, nm + ".csv.gz"), index=False, float_format="%.5g", compression="gzip")   # t is a pre-formatted string column (4 decimals)
    json.dump(summaries_all, open(os.path.join(a.out, "summaries_all.json"), "w"), default=lambda o: None)
    json.dump(manifests, open(os.path.join(a.out, "compress_manifests.json"), "w"), indent=1)
    json.dump(stamps, open(os.path.join(a.out, "sensor_stamps_summaries.json"), "w"), indent=1)
    groups = [
        ("00_id_baseline_full", ["00_id_baseline_full_rerun"], "determinism control: identical full-bag rerun"),
        ("00_id_baseline_60s", ["10_smoke_fixedq_grid_corner_60s"], "EXPECTED DIFFERENT (control that a changed Q changes the trace): fixed Q at the grid corner"),
        ("00_id_baseline_60s", ["00_id_new_keys_at_defaults_60s"], "R67 keys at defaults (bias_model random_walk, akf_q off) and write-only frame_window_log are inert (60 s)"),
        ("10_smoke_spline_60s", ["10_smoke_spline_explicit_defaults_60s"], "spline/head_pin_update=on_ok spelled out is inert (60 s)"),
        ("10_smoke_coupled_tail_minimal_60s", ["10_smoke_coupled_tail_explicit_defaults_60s"], "coupled tail: on_solver_error throw and step_scale 1.0 spelled out are inert (60 s)"),
        ("10_smoke_coupled_measurement_minimal_60s", ["10_smoke_coupled_measurement_explicit_defaults_60s"], "coupled measurement: on_solver_error throw and step_scale 1.0 spelled out are inert (60 s)"),
        ("10_smoke_coupled_tail_minimal_60s", ["10_smoke_coupled_tail_skip_scan_60s"], "coupled tail: skip_scan is inert when nothing throws (equal = it never fired)"),
        ("10_smoke_coupled_measurement_minimal_60s", ["10_smoke_coupled_measurement_skip_scan_60s"], "coupled measurement: skip_scan inert when nothing throws"),
    ]
    for sq in sorted(set(plan.seq) - {"eee_01"}):
        groups.append(("50_E_%s_baseline_full" % sq, ["50_E_%s_baseline_rerun_full" % sq], "determinism control on %s" % sq))
    with open(os.path.join(a.out, "identity_controls.txt"), "w") as f:
        f.write("sha256 of state_trace.csv with the run_id column blanked (empty = file missing)\n")
        for ref, others, why in groups:
            f.write("%s | %s\n" % (why, ref))
            f.write("  %s  %s\n" % (sha.get(ref, "") or "(missing)", ref))
            for o in others:
                f.write("  %s  %s  -> %s\n" % (sha.get(o, "") or "(missing)", o, "IDENTICAL" if sha.get(ref) and sha.get(ref) == sha.get(o) else "DIFFERENT or missing"))
        f.write("\nR64 digests (from the R64 return) against this batch's baselines\n")
        for arm, dg in R64_DIGESTS.items():
            f.write("  %s  R64 %s  R67 %s  -> %s\n" % (arm, dg[:16], (sha.get(arm, "") or "(missing)")[:16], "IDENTICAL" if sha.get(arm) == dg else "DIFFERENT or missing"))
    print("collected %d/%d arms" % (len(rows), len(plan)))

if __name__ == "__main__":
    main()
