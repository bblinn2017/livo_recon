#!/usr/bin/env python3
"""R62: flatten every arm's r62_extra.json into r62_extra_summary.csv and copy them to <out>/r62_extra/<arm>.json.
python3 collect_r62.py --runs <runs_dir> --arms r62_manifest/arm_table.tsv --out <out_dir>"""
import argparse, json, os, shutil
import pandas as pd
AX = ["Rx", "Ry", "Rz", "px", "py", "pz", "vx", "vy", "vz"]
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--arms", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "r62_extra"), exist_ok=True)
    rows = []
    for arm in pd.read_csv(a.arms, sep="\t").arm:
        for fn in ("r62_extra.json",):
            p = os.path.join(a.runs, arm, fn)
            if not os.path.exists(p):
                continue
            S = json.load(open(p)); shutil.copy(p, os.path.join(a.out, "r62_extra", arm + ".json"))
            r = {"arm": arm, "run_id_match": (S.get("identity") or {}).get("match")}
            f1 = S.get("frame1") or {}
            for ph in ("post_imu", "post_lio"):
                for ax in AX:
                    c = (f1.get(ph) or {}).get(ax)
                    if c:
                        r["f1_%s_e_%s" % (ph[5:], ax)] = c["e"]; r["f1_%s_sd_%s" % (ph[5:], ax)] = c["sd"]; r["f1_%s_z_%s" % (ph[5:], ax)] = c["z"]
                for k in ("speed_error_norm", "position_error_norm", "rotation_error_norm_rad"):
                    if k in (f1.get(ph) or {}):
                        r["f1_%s_%s" % (ph[5:], k)] = f1[ph][k]
            for w, v in (S.get("settled_rot") or {}).items():
                if isinstance(v, dict):
                    for i, ax in enumerate(("Rx", "Ry", "Rz")):
                        r["rot_%s_mean_e_mrad_%s" % (w, ax)] = v["mean_e_R_mrad"][i]; r["rot_%s_rms_e_mrad_%s" % (w, ax)] = v["rms_e_R_mrad"][i]; r["rot_%s_mean_sd_mrad_%s" % (w, ax)] = v["mean_sd_R_mrad"][i]
            for nm, v in (S.get("biases") or {}).items():
                for i, ax in enumerate("xyz"):
                    r["%s_scan1_%s" % (nm, ax)] = v["scan1"][i]; r["%s_change_%s" % (nm, ax)] = v["change"][i]
            r["n_warnings"] = len(S.get("warnings", []))
            rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "r62_extra_summary.csv"), index=False, float_format="%.10g")
    print("r62_extra_summary.csv: %d arms" % len(rows))
if __name__ == "__main__":
    main()
