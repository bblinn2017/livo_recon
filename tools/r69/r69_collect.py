#!/usr/bin/env python3
"""R69: gather every reduced arm into compact return tables. Nothing here interprets results.

python3 r69_collect.py --runs <runs_dir> --arms r72_manifest/arm_table.tsv --out <out_dir> [--level 0] [--params bag_params.json]
Writes into <out_dir>:
  phase_summary.csv            one row per arm (flat reducer metrics) plus the alignment-free deviation-from-baseline columns (below)
  arm_scan_all.csv.gz          every arm's per-scan table THINNED by --level (levels below). Column dev_from_baseline_mm added. Exact scan times in column t (4 decimals).
  arm_ol_tau_all.csv.gz, plane_fit_reasons_all.csv.gz, plane_fit_trace_counts_all.csv.gz, completeness.csv, summaries_all.json (every arm_summary.json in one file)
  compact/<arm>/...            the r69_compress.py outputs of each keep arm
  compress_manifests.json, sensor_stamps_summaries.json, identity_controls.txt, duplicates.tsv (configurations not written twice: which earlier arm they equal)
THINNING LEVELS of arm_scan_all.csv.gz (scan numbers; "onset window" = onset-20 .. onset+115 scans of that sequence, eee_01: 470..600; a keep arm is one
with keep_raw=1 in the arm table). Every level keeps scans 1..12 of every arm and the whole onset window of every arm at full resolution.
  level -1: no thinning (R80, F-381: every scan of every arm; its size is the arm count x scans x row bytes and must be stated before use, rule 5.15).
  level 0: outside the window every 25th scan;  keep arms complete (every scan).
  level 1: every 25th scan;  keep arms every 5th scan.
  level 2: every 50th scan;  keep arms every 10th scan.
  level 3: every 100th scan; keep arms every 25th scan; onset window onset-20 .. onset+70.
SUFFICIENCY: settled offset, wander, excursion, stationary NEES, ATE (evo and alignment), first-deviation scans and onset traces are computed from the FULL per-scan table
by r69_reduce.py before thinning and are in phase_summary.csv; the thinned table is for plotting drift, for my own re-derivations inside the onset window, and for
checks. Not recoverable from a thinned table: per-scan values outside the window for non-keep arms (kept on the host with sha256 in raw_sha256.txt).
DEVIATION FROM BASELINE (alignment-free): dev_from_baseline_mm[scan] = || ep(arm, scan) - ep(baseline, scan) || in mm (ep = displacement from the arm's own first-scan
reference). Baseline = 00_id_baseline_full (eee_01 full), 00_id_baseline_60s (60 s), 00_id_baseline_48s (48 s), E_<seq>_baseline_full (other sequences).
Summary columns: dev_max_mm, dev_max_scan, dev_first_scan_above_<X>mm (X = 25,50,100,200,500; scans after the onset scan), dev_at_scan_<S>, dev_rms_scans_541_on_mm."""
import argparse, glob, gzip, json, os, shutil
import numpy as np
import pandas as pd

AX = ["Rx", "Ry", "Rz", "px", "py", "pz", "vx", "vy", "vz"]
REF60 = "r74_own_id_baseline_60s"
BASE = {"full": "r80_mg_eee_01_fq", "60": "r80_dp_pfn3_fq", "48": "r74_own_id_baseline_48s"}   # R74: the fq baseline of the same batch (F-418) / the 0.425 identity arm; no 48 s arm exists   # R72: names of the in-round identity arms (48 s does not exist in R72)
R73_DIGEST_60S = "6cd2d754"      # prefix of the R73 identity reference digest of eee_01 60 s fixed-Q (r73_00_id_baseline_60s; claude/followups.md F-419 R73 RESULT)
DROP_FROM_RETURN = {"r73_f417_residuals.csv.gz",
                     # Return-trim (Bryce, R74): not named in required_outputs.json; stay in each arm's host archive only.
                     # Redundant for R75's LN/DET arms (already excluded via F1_GROUPS/F1_KEEP below) but kept global for consistency.
                     "lio_update_diag_compact.csv.gz", "residual_geometry_compact.csv.gz",
                     "frame_stats_compact.csv.gz", "frame_stats_blocks.csv", "frame_stats_windows.csv.gz",
                     "lio_gt_matched_compact.csv.gz", "crossframe_frames.csv.gz", "crossframe_axes.csv",
                     "live_point_cov_compact.csv.gz", "motion_q_scan_all.csv.gz", "frame_window_all.csv.gz"}   # R74: duplicates residual_dump_all.csv.gz (same rows, weight and class); stays in the host bundle and archive, not in the return (return size, rule 5.12a)
KEEP_IF_ARM_HAS = {"r73_f417_residuals.csv.gz": "_n200_", "motion_q_scan_all.csv.gz": "r80_b3_eee_02_mq"}   # R80 (F-437 / D2): the R80 BASE3 eee_02 motion-Q arm replaces the R79 arm name; R79 (F-437): the motion-Q trace of that ONE arm is returned (substring of the arm name; the file is every row of motion_q_scan.csv); R74 addendum 1 (Bryce: return it for calibration samples N=200 only): a dropped file is still returned for an arm whose name contains the value
F1_GROUPS = {"DP", "MG", "VX", "LN", "SM", "DET", "FB", "IDN", "SMK", "GAP", "SWP", "STA", "SPL", "CTL", "IF3", "YAW", "ML1", "ROOT", "SNP", "REF", "NB", "QV", "DX", "EC",
             "B3", "B1", "ZD", "GP", "NG", "PC", "DB", "TL", "EC3", "SC", "SW", "SMK"}   # R80: BASE3 / BASE1, D7 detect-only (ZD), F-433 (GP), F-476 (NG), D8 (PC), F-469 (DB), TAIL-1 (TL), D3 (EC3), D5 (SC), D1 sweep (SW), smoke (SMK; already listed)   # R79: F-463 (NB), F-401 / F-418 (QV), F-464 (DX), F-441 / F-440 (EC); R78: its groups (Stage A, split, controls, information-matched thinning, yaw, max layer 1, root 1.4, snapshot, refuse); R77: smoke (SMK), gap-floor (GAP) and voxel-sweep (SWP) groups as well;   # R76: FB (the full-bag grid) and IDN (identity arms) as well;   # R74 addendum 5: these groups' other per-frame compact files stay in the host archive; only frame-1 files are returned
F1_KEEP = {"residual_dump_all.csv.gz", "r73_f417_residuals.csv.gz", "r74_plane_table.csv.gz",
           "r74_plane_check.json", "r73_f417_hth_frames.csv", "r73_f417_classes.csv",
           "r73_f417_consistency.csv", "pos_trace_all.csv.gz", "compress_manifest.json",
           "sensor_stamps_summary.json", "line_rows_per_scan_all.csv.gz",   # R77 (F-451)
           "snapshot_nodes_all.csv.gz",   # R78 (F-428): every node of the post-calibration snapshot
           "snapshot_frames_all.csv.gz", "state_trace_early_all.csv.gz", "lio_update_diag_early_all.csv.gz", "residual_dump_state_all.csv.gz",   # R79 (F-464, F-441, F-440)
           "window_state_all.csv.gz", "window_iter_all.csv.gz", "window_update_diag_all.csv.gz", "window_frame_stats_all.csv.gz", "window_dump_state_all.csv.gz", "window_row_summary_all.csv",
           "window_ground_rows_all.csv.gz", "lidar_msg_log_all.csv.gz",   # R80 (F-449a, F-389)
           "zupt_all.csv.gz",       # R80 (F-467): gzip copy of zupt.csv, group ZD only (see GROUP_ONLY_FILES)
           "q_adapt_all.csv.gz",    # R80 (F-476): gzip copy of q_adapt.csv, group NG only
           "step_cap.csv.gz",       # R80 (F-403): written by step_cap_reduce, group SC only
           "motion_q_scan_all.csv.gz"}   # R79 (F-437): returned only for the arm named in KEEP_IF_ARM_HAS (the DROP_FROM_RETURN rule filters every other arm first)
GROUP_ONLY_FILES = {"zupt_all.csv.gz": {"ZD"}, "q_adapt_all.csv.gz": {"NG"}, "step_cap.csv.gz": {"SC"}}   # R80: a returned file that belongs to one group only (other groups never produce these, this is a second guard)
# R80 (D1, Bryce 2026-10-05): the voxel-size sweep (group SW, 1224 arms) returns SUMMARIES ONLY (phase_summary.csv rows, arm_scan_all, summaries_all.json: all built for every arm below,
# independent of this rule); per-arm compact files stay in the host archive. ONE exception: pos_trace_all.csv.gz of an SW arm whose gt_max_mm is above SW_TRACE_GT_MM, because the project
# collapse rule (rule 4.15 c; tools/r69/r80_sweep_zones.py) is 'departure >= 500 mm AND gt_max_mm > 500' and its departure needs the trace. gt_max_mm <= 500 can never be collapsed, so no trace is needed.
SW_GROUPS = {"SW"}
SW_TRACE_FILE = "pos_trace_all.csv.gz"
SW_TRACE_GT_MM = 500.0   # mm; the gt_max_mm limb of the collapse rule (same constant as COLLAPSE_GT_MM in r80_sweep_zones.py)
LEVELS = {-1: (None, None, 0), 0: (25, None, 0), 1: (25, 5, 0), 2: (50, 10, 0), 3: (100, 25, 45)}   # R80 (F-381): level -1 keeps every scan of every arm (no thinning)   # (step, keep-arm step (None = every scan), onset-window trim after onset+70)

def sw_wants_trace(gt_max_mm):
    """True when an SW arm's pos_trace_all.csv.gz must be returned. gt_max_mm is S['gt_max_mm'] of the arm's arm_summary.json (the same value as the phase_summary.csv column).
    Strictly above SW_TRACE_GT_MM -> True. Missing / NaN / unparseable (gt not joined, undetermined) -> True: r80_sweep_zones.py lists such arms as UNDETERMINED and never treats them as healthy."""
    try:
        v = float(gt_max_mm)
    except (TypeError, ValueError):
        return True
    return v != v or v > SW_TRACE_GT_MM


def return_file(base, arm, group, gt_max_mm=None):
    """Single decision: is compact file `base` of `arm` (arm-table group `group`) copied into the return? gt_max_mm only matters for SW arms."""
    if base in DROP_FROM_RETURN and KEEP_IF_ARM_HAS.get(base, "\0") not in arm:
        return False
    if group in SW_GROUPS:
        return base == SW_TRACE_FILE and sw_wants_trace(gt_max_mm)
    if group in F1_GROUPS and base not in F1_KEEP:
        return False
    if base in GROUP_ONLY_FILES and group not in GROUP_ONLY_FILES[base]:
        return False
    return True


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
    for k in ("coupled_status", "akf_q", "frame_window", "r69_engagement_from_run_log"):
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
        base_tabs["seq:" + sq] = load_base("r80_mg_%s_fq" % sq)   # R80: the whole-bag message-log arm (lines off, 0.425, pfn 3, fixed-Q) of the bag; R79: the same-batch lines-off 0.425 m fixed-Q control of the bag (R74: r74_qb_<bag>_fq)
    for _, p in plan.iterrows():
        D = os.path.join(a.runs, p.arm)
        sj = os.path.join(D, "arm_summary.json")
        meta = json.load(open(os.path.join(D, "arm.json"))) if os.path.exists(os.path.join(D, "arm.json")) else {}
        shp = os.path.join(D, "state_trace_blank_runid.sha256")
        sha[p.arm] = open(shp).read().strip() if os.path.exists(shp) else ""
        comp.append({"arm": p.arm, "tier": p.tier, "kind": p.kind, "duration_s": p.duration_s, "has_dir": os.path.isdir(D), "has_summary": os.path.exists(sj),
                     "exit_code": meta.get("exit_code"), "wall_s": meta.get("wall_s"), "missing": " ".join(meta.get("missing", [])), "requires_missing": " ".join(meta.get("requires_missing", [])), "arm": p.arm,
                     "state_trace_sha256_run_id_blanked": sha[p.arm]})
        cm = os.path.join(D, "compress_manifest.json")
        if os.path.exists(cm):
            manifests[p.arm] = json.load(open(cm))
        # R80: the arm summary is read BEFORE the compact copy because the SW rule needs its gt_max_mm (field S['gt_max_mm'], written by r69_reduce.py)
        S = json.load(open(sj)) if os.path.exists(sj) else None
        gt_max = S.get("gt_max_mm") if S else None
        cdir = os.path.join(a.out, "compact", p.arm)
        for pat in ("*_compact.csv.gz", "*_windows.csv.gz", "*_blocks.csv", "*_all.csv.gz", "plane_trace_window.csv.gz", "sensor_stamps_summary.json", "compress_manifest.json",
                    "residual_info_*", "crossframe_*", "knot_windows.csv", "knot_trend_stationary.csv",
                    "r72_*", "r73_*", "r74_*", "step_cap.csv.gz", "shadow_*_all.csv.gz", "scan_log_all.csv.gz", "residual_dump_*all.csv.gz", "coupled_update_*_all.csv.gz", "pos_trace_all.csv.gz"):   # R71-R73 analysis outputs
            for fp in glob.glob(os.path.join(D, pat)):
                if not return_file(os.path.basename(fp), p.arm, str(p.get("group", "")), gt_max):
                    continue
                os.makedirs(cdir, exist_ok=True)
                shutil.copy(fp, cdir)
        ssj = os.path.join(D, "sensor_stamps_summary.json")
        if os.path.exists(ssj):
            stamps[p.arm] = json.load(open(ssj))
        if S is None:
            continue
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
            if fn == "arm_scan.csv" and str(p.get("group", "")) in SW_GROUPS:
                continue   # R80 (D1): SW arms return summaries only; their per-scan table stays in the host archive (phase_summary.csv row and summaries_all.json entry are still written; dev_* columns were computed above)
            df.insert(0, "arm", p.arm)
            acc.append(df)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "phase_summary.csv"), index=False)
    pd.DataFrame(comp).to_csv(os.path.join(a.out, "completeness.csv"), index=False)
    for nm, acc in (("arm_scan_all", scans),):   # R73 (F-409): the open-loop / plane-fit aggregate tables of R69 are retired (nothing requests them)
        if acc:
            big = pd.concat(acc)
            big.to_csv(os.path.join(a.out, nm + ".csv.gz"), index=False, float_format="%.5g", compression="gzip")   # t is a pre-formatted string column (4 decimals)
    json.dump(summaries_all, open(os.path.join(a.out, "summaries_all.json"), "w"), default=lambda o: None)
    json.dump(manifests, open(os.path.join(a.out, "compress_manifests.json"), "w"), indent=1)
    json.dump(stamps, open(os.path.join(a.out, "sensor_stamps_summaries.json"), "w"), indent=1)
    with open(os.path.join(a_out := a.out, "identity_controls.txt"), "w") as f:
        f.write("sha256 of state_trace.csv with the run_id column blanked (empty = file missing)\n")
        if REF60 in set(plan.arm):   # R75 has no identity arm
            f.write("R74 identity arm (eee_01 60 s, fixed-Q, no new option on; F-419 extension changes round-off only, so the verdict is the 0.005 mm metric gate in r74_gates.txt; digest equality with R73 is reported as a fact)\n")
            d0 = sha.get(REF60, "")
            f.write("  %s  %s  R73 reference digest starts %s -> %s\n" % (d0 or "(missing)", REF60, R73_DIGEST_60S, "IDENTICAL to R73" if d0.startswith(R73_DIGEST_60S) else "DIFFERENT or missing"))
        for pre, rep in (("r74_own_det_", "r74_vox_"), ("r75_own_det_", "r75_line_"), ("r76_own_det_id_", "r76_id_"), ("r76_own_det_fb_", "r76_fb_"), ("r77_own_det_id_", "r77_id_"), ("r77_own_det_fb_", "r77_fb_"), ("r78_own_det_id_", "r78_id_"), ("r78_own_det_fb_", "r78_ct_"), ("r79_own_det_id_", "r79_id_"), ("r79_own_det_nb_", "r79_nb_"), ("r80_own_det_dp_", "r80_dp_")):   # R75: the control's original is the r75_line_ arm of the same name
            for x in [x for x in sha if x.startswith(pre)]:
                orig = x.replace(pre, rep)
                f.write("determinism: %s  %s  vs %s  %s  -> %s\n" % (sha.get(x, "") or "(missing)", x, sha.get(orig, "") or "(missing)", orig, "IDENTICAL" if sha.get(x) and sha.get(x) == sha.get(orig) else "DIFFERENT or missing"))
    print("collected %d/%d arms" % (len(rows), len(plan)))

if __name__ == "__main__":
    main()
