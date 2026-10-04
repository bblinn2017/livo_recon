#!/usr/bin/env python3
"""R65: gather every reduced arm into compact return tables. Nothing here interprets results.

python3 r65_collect.py --runs <runs_dir> --arms r65_manifest/arm_table.tsv --out <out_dir>
Writes into <out_dir>:
  phase_summary.csv            one row per arm (flat reducer metrics) plus the alignment-free deviation-from-baseline columns (below)
  arm_scan_all.csv.gz          every arm's per-scan table; full-bag arms thinned to every scan <= 700 and every 10th scan after; column
                               dev_from_baseline_mm added (see below). Exact scan times are in column t (7 decimals).
  arm_ol_tau_all.csv.gz, plane_fit_reasons_all.csv.gz, plane_fit_trace_counts_all.csv.gz, completeness.csv, summaries/<arm>.json
  compact/<arm>/...            the r65_compress.py outputs of each arm (sensor_stamps, frame_stats, surface_map_frames, plane_trace, gt)
  compress_manifests.json      per arm: raw input file, its sha256, rows, output sizes
  sensor_stamps_summaries.json the stamp-table summaries of the arms that wrote one
  identity_controls.txt        pairwise sha256 comparisons named below, and the comparison with the R64 digests
DEVIATION FROM BASELINE (alignment-free): dev_from_baseline_mm[scan] = || ep(arm, scan) - ep(baseline, scan) || in mm, where ep is the
reducer's displacement from the arm's own first-scan reference (arm_scan.csv ep_x/y/z_mm) and baseline is A_baseline_full_rep1 for full-bag arms,
A_baseline_60s_rep1 for 60 s arms, A_baseline_48s_rep1 for 48 s arms (closed loop only). Summary columns: dev_max_mm, dev_max_scan,
dev_first_scan_above_<X>mm (first scan after 481 whose deviation exceeds X = 25, 50, 100, 200, 500; blank if never), dev_at_scan_<S> (S in
500 520 540 600 800 1000 1500 2000 3000 3900), dev_rms_scans_541_on_mm."""
import argparse, glob, gzip, json, os, shutil
import numpy as np
import pandas as pd

AX = ["Rx", "Ry", "Rz", "px", "py", "pz", "vx", "vy", "vz"]
BASE = {"full": "A_baseline_full_rep1", "60": "A_baseline_60s_rep1", "48": "A_baseline_48s_rep1"}
R64_DIGESTS = {"A_baseline_48s_rep1": "197d7f00e23f45160a1a7efbbc9e86dc85bfae069140d1eb3790f3dc75500341",
               "A_baseline_60s_rep1": "a096c8b9e581f3c47bfa184c2909aea450ff74700a77a766ab93fded3391c8ad",
               "A_baseline_full_rep1": "2d3eebfb2a84ae0fe8e03222232ea814bc323e02027de395681b54f32b4f6857"}

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
    r["n_missing"] = len(S.get("missing", []))
    r["n_warnings"] = len(S.get("warnings", []))
    return r

def thin(df, full):
    if not full:
        return df
    return df[(df.scan <= 700) | (df.scan % 10 == 0)]

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
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "summaries"), exist_ok=True)
    plan = pd.read_csv(a.arms, sep="\t")
    rows, comp, scans, ols, reasons, tcounts = [], [], [], [], [], []
    sha, manifests, stamps = {}, {}, {}
    base_tabs = {}
    for key, nm in BASE.items():
        fp = os.path.join(a.runs, nm, "arm_scan.csv")
        base_tabs[key] = pd.read_csv(fp) if os.path.exists(fp) else None
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
        for pat in ("*_compact.csv.gz", "*_windows.csv.gz", "*_blocks.csv", "plane_trace_window.csv.gz", "sensor_stamps_summary.json", "compress_manifest.json"):
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
        shutil.copy(sj, os.path.join(a.out, "summaries", p.arm + ".json"))
        full = str(p.duration_s) == "full"
        for fn, acc in (("arm_scan.csv", scans), ("arm_ol_tau.csv", ols), ("plane_fit_reasons.csv", reasons), ("plane_fit_trace_counts.csv", tcounts)):
            fp = os.path.join(D, fn)
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp)
            if fn == "arm_scan.csv":
                key = "full" if full else str(p.duration_s)
                bt = base_tabs.get(key)
                if bt is not None and p.kind != "open" and p.tier != "RF":
                    d = dev_table(df, bt)
                    df["dev_from_baseline_mm"] = df.scan.map(d)
                    row.update(dev_summary(d))
                df = thin(df, full)
            df.insert(0, "arm", p.arm)
            acc.append(df)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "phase_summary.csv"), index=False)
    pd.DataFrame(comp).to_csv(os.path.join(a.out, "completeness.csv"), index=False)
    for nm, acc in (("arm_scan_all", scans), ("arm_ol_tau_all", ols), ("plane_fit_reasons_all", reasons), ("plane_fit_trace_counts_all", tcounts)):
        if acc:
            pd.concat(acc).to_csv(os.path.join(a.out, nm + ".csv.gz"), index=False, float_format="%.5g", compression="gzip")
    json.dump(manifests, open(os.path.join(a.out, "compress_manifests.json"), "w"), indent=1)
    json.dump(stamps, open(os.path.join(a.out, "sensor_stamps_summaries.json"), "w"), indent=1)
    groups = [
        ("A_baseline_60s_rep1", ["A_new_keys_spelled_out_at_defaults_60s", "A_diagnostics_on_60s", "A_perturb_std0_seed7_60s"], "R65 keys at defaults and write-only diagnostics are inert (60 s)"),
        ("A_baseline_full_rep1", ["A_baseline_full_rep2", "A_baseline_full_diagnostics_on"], "full-bag repeat and write-only diagnostics are inert"),
        ("D_debiased_full_eigengap_60s", ["D_debiased_full_eigengap_explicit_keys_60s"], "debiased keys spelled out at defaults (60 s)"),
        ("D_debiased_full_eigengap_full", ["D_debiased_full_eigengap_explicit_keys_full"], "debiased keys spelled out at defaults (full bag; both carry reasons, trace 540 only on the first)"),
        ("A_baseline_60s_rep1", ["A_baseline_60s_perturb_seed1"], "EXPECTED DIFFERENT: perturbation seed 1 changes the run"),
        ("A_baseline_full_rep1", ["A_baseline_full_perturb_seed1", "A_baseline_full_perturb_seed2"], "EXPECTED DIFFERENT: perturbation seeds change the run"),
    ]
    with open(os.path.join(a.out, "identity_controls.txt"), "w") as f:
        f.write("sha256 of state_trace.csv with the run_id column blanked (empty = file missing)\n")
        for ref, others, why in groups:
            f.write("%s | %s\n" % (why, ref))
            f.write("  %s  %s\n" % (sha.get(ref, "") or "(missing)", ref))
            for o in others:
                f.write("  %s  %s  -> %s\n" % (sha.get(o, "") or "(missing)", o, "IDENTICAL" if sha.get(ref) and sha.get(ref) == sha.get(o) else "DIFFERENT or missing"))
        f.write("\nR64 digests (from the R64 return) against this batch's baselines\n")
        for arm, dg in R64_DIGESTS.items():
            f.write("  %s  R64 %s  R65 %s  -> %s\n" % (arm, dg[:16], (sha.get(arm, "") or "(missing)")[:16], "IDENTICAL" if sha.get(arm) == dg else "DIFFERENT or missing"))
    print("collected %d/%d arms" % (len(rows), len(plan)))

if __name__ == "__main__":
    main()
