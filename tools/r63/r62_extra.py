#!/usr/bin/env python3
"""R62: extra per-arm analyses that reduce_arm.py does not do. Writes <arm_dir>/r62_extra.json. Interprets nothing.

python3 r62_extra.py <arm_dir>

Sections (each is skipped with a warning if its input file is missing):
  identity      state_trace.csv run_id column vs arm name (job_id) -- must match.
  frame1        first scan: error vs the first-frame reference, predicted sd, z = e/sd, for post_imu and post_lio, all 9 blocks;
                position/velocity also split into world-vertical (z) and horizontal. Body-frame right perturbation e_R = Log(R_ref^T R).
  settled_rot   e_R (mrad) and sd at post_lio averaged over stationary windows (0-2, 2-5, 5-10 s after first scan and the last 10 s of the
                stationary span): does R settle to something other than the first-frame reference?
  biases        state bg, ba, gravity at scan 1, and drift to the end of the stationary span.
  prop_vs_qacc  imu_cov_growth.csv: median over scans of diag(P_end)-diag(P_start) and diag(Qacc) per state index, stationary vs later.
                (Only diagonals are logged, so P_end = Phi P_start Phi^T + Qacc cannot be checked entry by entry; the difference of
                diagonals includes the Phi-coupling terms. Reported as-is.)
  first_scans   imu_first_scans.csv verbatim (rows are few) plus head-sample summary.
  gravity_align gravity_alignment.txt scalar lines parsed to numbers where possible.
"""
import json, math, os, sys
import numpy as np
import pandas as pd
from common import *

def parse_txt(path):
    d = {}
    with open(path) as f:
        lines = f.read().splitlines()
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 2:
            try:
                vals = [float(x) for x in parts[1:]]
                d[parts[0]] = vals if len(vals) > 1 else vals[0]
            except ValueError:
                d[parts[0]] = " ".join(parts[1:])
    return d

def main():
    D = sys.argv[1]
    arm = os.path.basename(os.path.abspath(D))
    warn = []
    S = {"arm": arm, "warnings": warn}
    tr, ref = os.path.join(D, "state_trace.csv"), os.path.join(D, "state_reference.txt")
    if os.path.exists(tr) and os.path.exists(ref):
        df, pcols = load_trace(tr)
        Rref, pref, vref, P0 = read_reference(ref)
        rid = sorted(set(df.run_id.astype(str)))
        S["identity"] = {"run_ids_in_trace": rid, "arm": arm, "match": rid == [arm]}
        imu = df[df.phase == "post_imu"].drop_duplicates("scan_id", keep="last").set_index("scan_id")
        lio = df[df.phase == "post_lio"].drop_duplicates("scan_id", keep="last").set_index("scan_id")
        scans = sorted(set(imu.index) & set(lio.index))
        def st(row):
            return quat_to_R(np.array([row.qw, row.qx, row.qy, row.qz])), np.array([row.px, row.py, row.pz]), np.array([row.vx, row.vy, row.vz])
        def err(row):
            R, p, v = st(row)
            return np.concatenate([log_so3(Rref.T @ R), p - pref, v - vref]), cov_from_row(row, pcols, 9)
        if scans:
            f1 = {}
            for ph, tab in (("post_imu", imu), ("post_lio", lio)):
                e, P = err(tab.loc[scans[0]])
                sd = np.sqrt(np.maximum(np.diag(P), 0))
                f1[ph] = {ax: {"e": float(e[i]), "sd": float(sd[i]), "z": float(e[i] / sd[i]) if sd[i] > 0 else None} for i, ax in enumerate(AXES)}
                f1[ph]["speed_error_norm"] = float(np.linalg.norm(e[6:9]))
                f1[ph]["position_error_norm"] = float(np.linalg.norm(e[3:6]))
                f1[ph]["rotation_error_norm_rad"] = float(np.linalg.norm(e[0:3]))
                f1[ph]["horizontal_p_error"] = float(np.linalg.norm(e[3:5])); f1[ph]["vertical_p_error"] = float(e[5])
                f1[ph]["horizontal_v_error"] = float(np.linalg.norm(e[6:8])); f1[ph]["vertical_v_error"] = float(e[8])
                f1[ph]["horizontal_p_sd_rss"] = float(math.sqrt(sd[3] ** 2 + sd[4] ** 2)); f1[ph]["vertical_p_sd"] = float(sd[5])
                f1[ph]["horizontal_v_sd_rss"] = float(math.sqrt(sd[6] ** 2 + sd[7] ** 2)); f1[ph]["vertical_v_sd"] = float(sd[8])
            f1["P0_sd_first_frame_reference"] = [float(math.sqrt(max(P0[i, i], 0))) for i in range(9)]
            f1["scan_id"] = int(scans[0]); f1["t_abs"] = float(imu.loc[scans[0]].t_abs)
            S["frame1"] = f1
            t = np.array([imu.loc[s].t_abs for s in scans]); t0 = t[0]
            E = np.array([err(lio.loc[s])[0] for s in scans]); SD = np.array([np.sqrt(np.maximum(np.diag(err(lio.loc[s])[1]), 0)) for s in scans])
            # stationary span: from arm_summary.json if present, else first 20 s
            t_move = None
            sj = os.path.join(D, "arm_summary.json")
            if os.path.exists(sj):
                t_move = json.load(open(sj)).get("t_move")
            t_end = (t_move - 2.0) if t_move else t0 + 20.0
            wins = {"0-2s": (0, 2), "2-5s": (2, 5), "5-10s": (5, 10)}
            out = {}
            for nm, (a, b) in wins.items():
                m = (t - t0 >= a) & (t - t0 < b) & (t < t_end)
                if m.any():
                    out[nm] = {"n": int(m.sum()), "mean_e_R_mrad": (1e3 * E[m, 0:3].mean(0)).tolist(), "rms_e_R_mrad": (1e3 * np.sqrt((E[m, 0:3] ** 2).mean(0))).tolist(),
                               "mean_sd_R_mrad": (1e3 * SD[m, 0:3].mean(0)).tolist(), "mean_e_p_m": E[m, 3:6].mean(0).tolist(), "mean_e_v_mps": E[m, 6:9].mean(0).tolist()}
            m = (t >= t_end - 10.0) & (t < t_end)
            if m.any():
                out["last10s_of_stationary"] = {"n": int(m.sum()), "mean_e_R_mrad": (1e3 * E[m, 0:3].mean(0)).tolist(), "rms_e_R_mrad": (1e3 * np.sqrt((E[m, 0:3] ** 2).mean(0))).tolist(),
                                                "mean_sd_R_mrad": (1e3 * SD[m, 0:3].mean(0)).tolist(), "mean_e_p_m": E[m, 3:6].mean(0).tolist(), "mean_e_v_mps": E[m, 6:9].mean(0).tolist()}
            out["stationary_end_t_rel"] = float(t_end - t0)
            S["settled_rot"] = out
            B = {}
            first, last = lio.loc[scans[0]], lio.loc[scans[max(0, int(np.searchsorted(t, t_end)) - 1)]]
            for nm, cols in (("bg", ["bgx", "bgy", "bgz"]), ("ba", ["bax", "bay", "baz"]), ("gravity", ["gx", "gy", "gz"])):
                B[nm] = {"scan1": [float(first[c]) for c in cols], "end_of_stationary": [float(last[c]) for c in cols]}
                B[nm]["change"] = [b - a_ for a_, b in zip(B[nm]["scan1"], B[nm]["end_of_stationary"])]
            S["biases"] = B
        else:
            warn.append("no scan with post_imu and post_lio")
    else:
        warn.append("state_trace.csv / state_reference.txt missing: identity, frame1, settled_rot, biases skipped")

    cg = os.path.join(D, "imu_cov_growth.csv")
    if os.path.exists(cg):
        g = pd.read_csv(cg)
        n = int(g["dim"].iloc[0])
        sm = json.load(open(os.path.join(D, "arm_summary.json"))) if os.path.exists(os.path.join(D, "arm_summary.json")) else {}
        tm = sm.get("t_move")
        out = {}
        for nm, mask in (("stationary", g.t_abs < (tm - 2.0) if tm else g.t_abs < g.t_abs.iloc[0] + 20), ("later", g.t_abs >= (tm if tm else 1e30))):
            if mask.any():
                out[nm] = {"n": int(mask.sum()),
                           "median_dPdiag": [float(np.median(g.loc[mask, "Pend_%d" % i] - g.loc[mask, "Pstart_%d" % i])) for i in range(n)],
                           "median_Qacc_diag": [float(np.median(g.loc[mask, "Qacc_%d" % i])) for i in range(n)]}
        S["prop_vs_qacc"] = out
    else:
        warn.append("imu_cov_growth.csv missing")

    fs = os.path.join(D, "imu_first_scans.csv")
    if os.path.exists(fs):
        f = pd.read_csv(fs)
        S["first_scans"] = {"rows": f.to_dict(orient="records")}
    else:
        warn.append("imu_first_scans.csv missing (expected only when eval/imu_first_scans_en was on and the patch is built)")
    ga = os.path.join(D, "gravity_alignment.txt")
    if os.path.exists(ga):
        S["gravity_align"] = parse_txt(ga)
    else:
        warn.append("gravity_alignment.txt missing")
    json.dump(jsonable(S), open(os.path.join(D, "r62_extra.json"), "w"), indent=1)
    print("r62_extra %s: sections %s" % (arm, [k for k in S if k not in ("arm", "warnings")]))
    return 0

if __name__ == "__main__":
    sys.exit(main())
