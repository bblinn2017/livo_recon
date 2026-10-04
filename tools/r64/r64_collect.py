#!/usr/bin/env python3
"""R64: gather every reduced arm into compact return tables. Nothing here interprets results.

python3 r64_collect.py --runs <runs_dir> --arms r64_manifest/arm_table.tsv --out <out_dir>
Writes into <out_dir>: phase_summary.csv (one row per arm, flat), arm_scan_all.csv.gz (all arms; full-bag arms thinned: every scan
<= 620 and 380..470, every 5th scan otherwise), arm_ol_tau_all.csv.gz, plane_fit_reasons_all.csv.gz, plane_fit_trace_counts_all.csv.gz,
completeness.csv (planned vs done, exit code, wall_s, missing files, sha256 of state_trace.csv with run_id blanked), summaries/<arm>.json,
identity_controls.txt (pairwise sha256 comparisons named in the instructions)."""
import argparse, glob, gzip, json, os, shutil
import pandas as pd

AX = ["Rx", "Ry", "Rz", "px", "py", "pz", "vx", "vy", "vz"]

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
    r["iters_max"], r["iters_frac_at_max"] = it.get("max"), it.get("frac_at_max")
    for nm in ("stationary", "transition", "motion"):
        r["resid_median_%s" % nm] = g(S, "residuals", nm, "median")
        for k, v in (g(S, "frame_stats", nm) or {}).items():
            r["%s_%s" % (k, nm)] = v
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
    keep = (df.scan <= 620) | ((df.scan >= 380) & (df.scan <= 470)) | (df.scan % 5 == 0)
    return df[keep]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--arms", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "summaries"), exist_ok=True)
    plan = pd.read_csv(a.arms, sep="\t")
    rows, comp, scans, ols, reasons, tcounts = [], [], [], [], [], []
    sha = {}
    for _, p in plan.iterrows():
        D = os.path.join(a.runs, p.arm)
        sj = os.path.join(D, "arm_summary.json")
        meta = json.load(open(os.path.join(D, "arm.json"))) if os.path.exists(os.path.join(D, "arm.json")) else {}
        shp = os.path.join(D, "state_trace_blank_runid.sha256")
        sha[p.arm] = open(shp).read().strip() if os.path.exists(shp) else ""
        comp.append({"arm": p.arm, "tier": p.tier, "kind": p.kind, "duration_s": p.duration_s, "has_dir": os.path.isdir(D), "has_summary": os.path.exists(sj),
                     "exit_code": meta.get("exit_code"), "wall_s": meta.get("wall_s"), "missing": " ".join(meta.get("missing", [])),
                     "state_trace_sha256_run_id_blanked": sha[p.arm]})
        if not os.path.exists(sj):
            continue
        S = json.load(open(sj))
        rows.append(flat(S))
        shutil.copy(sj, os.path.join(a.out, "summaries", p.arm + ".json"))
        for fn, acc, full in (("arm_scan.csv", scans, str(p.duration_s) == "full"), ("arm_ol_tau.csv", ols, False),
                              ("plane_fit_reasons.csv", reasons, False), ("plane_fit_trace_counts.csv", tcounts, False)):
            fp = os.path.join(D, fn)
            if os.path.exists(fp):
                df = pd.read_csv(fp)
                if fn == "arm_scan.csv":
                    df = thin(df, full)
                df.insert(0, "arm", p.arm)
                acc.append(df)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "phase_summary.csv"), index=False)
    pd.DataFrame(comp).to_csv(os.path.join(a.out, "completeness.csv"), index=False)
    for nm, acc in (("arm_scan_all", scans), ("arm_ol_tau_all", ols), ("plane_fit_reasons_all", reasons), ("plane_fit_trace_counts_all", tcounts)):
        if acc:
            pd.concat(acc).to_csv(os.path.join(a.out, nm + ".csv.gz"), index=False, float_format="%.5g", compression="gzip")
    pairs = [("A_baseline_60s_rep1", "A_baseline_60s_new_keys_spelled_out_at_defaults", "new keys at defaults are inert (pca)"),
             ("A_baseline_60s_rep1", "A_baseline_60s_diagnostics_on", "plane-fit diagnostics are write-only (pca)"),
             ("P_debiased_full_eigengap_60s", "P_debiased_full_eigengap_explicit_keys_diagnostics_off_60s", "debias_mode=full / hysteresis=1 / diagnostics off equals defaults with diagnostics on (debiased)"),
             ("A_baseline_60s_rep1", "A_baseline_60s_rep2", "repeat 1 vs 2 (same config)"), ("A_baseline_60s_rep1", "A_baseline_60s_rep3", "repeat 1 vs 3"),
             ("A_baseline_48s_rep1", "A_baseline_48s_rep2", "48 s repeat 1 vs 2"), ("A_baseline_48s_rep1", "A_baseline_48s_rep3", "48 s repeat 1 vs 3"),
             ("A_baseline_full_rep1", "A_baseline_full_rep2", "full-bag repeat 1 vs 2"), ("A_baseline_full_rep1", "A_baseline_full_rep3", "full-bag repeat 1 vs 3")]
    with open(os.path.join(a.out, "identity_controls.txt"), "w") as f:
        f.write("sha256 of state_trace.csv with the run_id column blanked (empty = file missing)\n")
        for x, y, why in pairs:
            f.write("%s | %s | %s\n  %s\n  %s\n  -> %s\n" % (x, y, why, sha.get(x, ""), sha.get(y, ""),
                    "IDENTICAL" if sha.get(x) and sha.get(x) == sha.get(y) else "DIFFERENT or missing"))
    print("collected %d/%d arms" % (len(rows), len(plan)))

if __name__ == "__main__":
    main()
