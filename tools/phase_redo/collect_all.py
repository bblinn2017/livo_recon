#!/usr/bin/env python3
"""R61: gather every reduced arm into compact return tables.

python3 collect_all.py --runs <runs_dir> --arms arm_table.tsv --out <out_dir>
Writes: phase_summary.csv (one row per arm, flattened key metrics), arm_scan_all.csv.gz (every arm's arm_scan.csv with an
`arm` column), arm_ol_tau_all.csv.gz, completeness.csv (planned vs done/exit code/wall seconds), summaries.tar.gz-ready
folder summaries/<arm>.json. Nothing here interprets results."""
import argparse, glob, gzip, json, os, shutil, sys
import pandas as pd

AX = ["Rx", "Ry", "Rz", "px", "py", "pz", "vx", "vy", "vz"]

def flat(S):
    r = {"arm": S.get("arm")}
    m = S.get("meta", {})
    for k in ("tier", "arch", "kind", "exit_code", "wall_s", "ate_m"):
        r[k] = m.get(k)
    for k in ("n_scans", "n_stationary_scans", "t_move", "gt_join_rate"):
        r[k] = S.get(k)
    for ph in ("post_imu", "post_lio"):
        b = S.get("stationary", {}).get(ph, {})
        for ax in AX:
            if ax in b:
                r["st_%s_ratio_%s" % (ph[5:], ax)] = b[ax]["ratio_e2_over_P"]
                r["st_%s_rms_%s" % (ph[5:], ax)] = b[ax]["rms_e"]
        if "nees9" in b:
            r["st_%s_nees9_mean" % ph[5:]] = b["nees9"]["mean"]
            r["st_%s_nees9_frac_in95" % ph[5:]] = b["nees9"]["frac_in_95"]
    for ax, c in (S.get("phase3_stationary") or {}).items():
        r["p3_k_" + ax] = c["gain_k"]
        r["p3_k_over_pred_" + ax] = c["k_over_predicted"]
        r["p3_err2_ratio_" + ax] = c["realised_err2_ratio"]
        r["p3_pred_P_ratio_" + ax] = c["predicted_P_ratio"]
    for vn, d in (S.get("dynamic_gt_position") or {}).items():
        for ph in ("imu", "lio"):
            for ax in ("along", "lateral", "vertical"):
                r["dyn_%s_%s_%s_ratio" % (vn[4:], ph, ax)] = d[ph][ax]["ratio_e2_over_P"]
                r["dyn_%s_%s_%s_rms" % (vn[4:], ph, ax)] = d[ph][ax]["rms_e"]
            r["dyn_%s_%s_nees3" % (vn[4:], ph)] = d[ph]["nees_p3_mean"]
        for ax, c in d.get("correction", {}).items():
            r["dyn_%s_k_over_pred_%s" % (vn[4:], ax)] = c["k_over_predicted"]
            r["dyn_%s_err2_ratio_%s" % (vn[4:], ax)] = c["realised_err2_ratio"]
    r["n_warnings"] = len(S.get("warnings", []))
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "summaries"), exist_ok=True)
    plan = pd.read_csv(a.arms, sep="\t")
    rows, comp, scans, ols = [], [], [], []
    for _, p in plan.iterrows():
        D = os.path.join(a.runs, p.arm)
        sj = os.path.join(D, "arm_summary.json")
        meta = json.load(open(os.path.join(D, "arm.json"))) if os.path.exists(os.path.join(D, "arm.json")) else {}
        comp.append({"arm": p.arm, "tier": p.tier, "optional": p.optional, "has_dir": os.path.isdir(D), "has_summary": os.path.exists(sj),
                     "exit_code": meta.get("exit_code"), "wall_s": meta.get("wall_s"), "note": meta.get("note", "")})
        if not os.path.exists(sj):
            continue
        S = json.load(open(sj))
        S.setdefault("meta", {}).update({k: meta.get(k, S["meta"].get(k)) for k in meta})
        S["meta"].setdefault("tier", p.tier); S["meta"].setdefault("arch", p.arch); S["meta"].setdefault("kind", p.kind)
        rows.append(flat(S))
        shutil.copy(sj, os.path.join(a.out, "summaries", p.arm + ".json"))
        sc = os.path.join(D, "arm_scan.csv")
        if os.path.exists(sc):
            d = pd.read_csv(sc); d.insert(0, "arm", p.arm); scans.append(d)
        ot = os.path.join(D, "arm_ol_tau.csv")
        if os.path.exists(ot):
            d = pd.read_csv(ot); d.insert(0, "arm", p.arm); ols.append(d)
    pd.DataFrame(comp).to_csv(os.path.join(a.out, "completeness.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "phase_summary.csv"), index=False)
    if scans:
        pd.concat(scans).to_csv(os.path.join(a.out, "arm_scan_all.csv.gz"), index=False, float_format="%.5g", compression="gzip")
    if ols:
        pd.concat(ols).to_csv(os.path.join(a.out, "arm_ol_tau_all.csv.gz"), index=False, float_format="%.5g", compression="gzip")
    print("collected %d/%d arms" % (len(rows), len(plan)))

if __name__ == "__main__":
    sys.exit(main())
