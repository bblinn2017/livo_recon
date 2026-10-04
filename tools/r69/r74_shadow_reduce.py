#!/usr/bin/env python3
"""R74 (F-414, F-422, F-423): reduce the shadow-update logs.

    python3 r74_shadow_reduce.py arm   --dir <arm dir> --out <arm dir>          one shadow arm -> r74_shadow_arm.csv (+ r74_shadow_arm.json: digest and counter checks)
    python3 r74_shadow_reduce.py pairs --runs <runs dir> --pairs r74_manifest/shadow_pairs.json --out <return dir>
                                                                                  -> r74_shadow_pairs.csv, r74_shadow_pairs.json, r74_freeze_check.csv

Definitions (fixed BEFORE the data exist, F-414):
  step(role, frame)   = post position - start position of the scan, mm (propagation AND update; both roles start from the same state, the driver's previous posterior)
  S                   = max over the window frames of || step_shadow - step_driver ||                       (what the other process-noise model would have done at the same state)
  D_real              = max over the window frames of || step_controlA - step_controlB ||                   (what the two REAL runs did, from their pos_trace_all.csv.gz at the frame times)
  ratio               = S / D_real
  verdict             ratio >= 0.5: update_fails_on_healthy_state; ratio < 0.25: needs_accumulated_history; otherwise partial.   No other threshold is used, none is tuned afterwards.
  determinism repeat  every numeric column of shadow_scan_all.csv.gz of the repeat arm equals the original's (max |difference|, reported)
  freeze validity     a freeze arm is valid only if its frame-1 map digest equals that of its plain control (scan_log_all.csv.gz) and map_frozen_scans counts the live frames
A shadow arm is INVALID (verdict invalid_map_changed) if the digest after a shadow pass differs from the digest before it in any window scan.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r74_time as RT  # noqa: E402


def rd(d, *names):
    for n in names:
        p = os.path.join(d, n)
        if os.path.exists(p):
            return pd.read_csv(p, dtype={"map_digest_before_pass": str, "map_digest_after_pass": str, "map_digest_exact": str, "map_digest_rounded_1e-3": str})
    return None


def steps(S):
    drv, sh = S[S["role"] == "driver"].set_index("frame"), S[S["role"] == "shadow"].set_index("frame")
    fr = sorted(set(drv.index) & set(sh.index))
    T = pd.DataFrame(index=fr)
    p = lambda D, a, b: (D.loc[fr, [b + "_x", b + "_y", b + "_z"]].to_numpy(float) - D.loc[fr, [a + "_x", a + "_y", a + "_z"]].to_numpy(float))
    sd, ss = p(drv, "start_pos", "post_pos") * 1e3, p(sh, "start_pos", "post_pos") * 1e3
    ud, us = p(drv, "prop_pos", "post_pos") * 1e3, p(sh, "prop_pos", "post_pos") * 1e3
    rd_, rs = p(drv, "start_rv", "post_rv") * 1e3, p(sh, "start_rv", "post_rv") * 1e3
    T["frame"] = fr
    if "t_rel_s" in drv.columns:
        T["t_rel_s"] = drv.loc[fr, "t_rel_s"].to_numpy(float)
    T["step_pos_driver_mm"] = np.linalg.norm(sd, axis=1); T["step_pos_shadow_mm"] = np.linalg.norm(ss, axis=1); T["step_diff_mm"] = np.linalg.norm(ss - sd, axis=1)
    T["update_pos_driver_mm"] = np.linalg.norm(ud, axis=1); T["update_pos_shadow_mm"] = np.linalg.norm(us, axis=1); T["update_diff_mm"] = np.linalg.norm(us - ud, axis=1)
    # F-423: the rotation difference is the ANGLE of R_driver_post^T R_shadow_post (both roles start from the same state, so this is the difference of the two updates),
    # bounded by pi; the R73 norm of a difference of rotation vectors was not periodic (a 2 pi artefact of 6,281 mrad in eee02_e).
    post_rv_d = drv.loc[fr, ["post_rv_x", "post_rv_y", "post_rv_z"]].to_numpy(float)
    post_rv_s = sh.loc[fr, ["post_rv_x", "post_rv_y", "post_rv_z"]].to_numpy(float)
    T["step_rot_diff_mrad"] = np.array([RT.rotvec_angle(a_, b_) for a_, b_ in zip(post_rv_d, post_rv_s)]) * 1e3
    for c in ("prior_trace_rot", "prior_trace_pos", "post_trace_rot", "post_trace_pos", "n_iter", "n_resid_iter0", "reduced_chi2_iter0"):
        T[c + "_driver"] = drv.loc[fr, c].to_numpy(float); T[c + "_shadow"] = sh.loc[fr, c].to_numpy(float)
    for c in ("q_acc_x", "q_gyr_x", "q_gyr_z"):
        T[c + "_driver"] = drv.loc[fr, c].to_numpy(float); T[c + "_shadow"] = sh.loc[fr, c].to_numpy(float)
    return T.reset_index(drop=True)


def checks(S):
    sh = S[S["role"] == "shadow"]
    changed = int((sh["map_digest_before_pass"] != sh["map_digest_after_pass"]).sum()) if len(sh) else None
    return {"n_shadow_rows": int(len(sh)), "n_driver_rows": int((S["role"] == "driver").sum()), "map_digest_changed_scans": changed,
            "frames_first_last": [int(S["frame"].min()), int(S["frame"].max())] if len(S) else None}


def arm_main(a):
    S = rd(a.dir, "shadow_scan.csv", "shadow_scan_all.csv.gz")
    if S is None:
        print("no shadow_scan.csv in", a.dir); return 2
    T = steps(S)
    arm = os.path.basename(os.path.abspath(a.dir))
    T.insert(0, "arm", arm)
    RT.to_csv_t6(T, os.path.join(a.out, "r74_shadow_arm.csv"))
    json.dump(dict(arm=arm, **checks(S)), open(os.path.join(a.out, "r74_shadow_arm.json"), "w"), indent=1)
    return 0


def pos_at(P, frames):
    """F-422: positions of a plain control at the shadow arm's frames, matched by scan_id (frame == scan_id), never by a time match. ok = the frame exists in the control."""
    P = P.drop_duplicates("scan_id").set_index("scan_id")
    fr = np.asarray(frames, int)
    ok = np.isin(fr, P.index.to_numpy())
    pos = np.full((len(fr), 3), np.nan)
    if ok.any():
        pos[ok] = P.loc[fr[ok], ["px", "py", "pz"]].to_numpy(float)
    return pos, ok


def pairs_main(a):
    pairs = json.load(open(a.pairs))
    rows, frz = [], []
    for pr in pairs:
        d = os.path.join(a.runs, pr["shadow_arm"])
        S = rd(d, "shadow_scan_all.csv.gz", "shadow_scan.csv")
        if S is None:
            rows.append(dict(pair=pr["pair"], verdict="MISSING", note="no shadow_scan in %s" % d)); continue
        T = steps(S)
        lo, hi = pr["window"]
        W = S[(S["role"] == "driver") & (S["frame"] >= lo) & (S["frame"] <= hi)]
        row = dict(pair=pr["pair"], shadow_arm=pr["shadow_arm"], n_window_frames=int(len(W)), n_frames_with_both_roles=int(len(T)))
        ck = checks(S)
        row["map_digest_changed_scans"] = ck["map_digest_changed_scans"]
        if "repeat_of" in pr:
            o = rd(os.path.join(a.runs, pr["repeat_of"]), "shadow_scan_all.csv.gz", "shadow_scan.csv")
            if o is None or len(o) != len(S):
                row.update(verdict="MISSING", note="original or row count mismatch")
            else:
                num = [c for c in S.columns if S[c].dtype.kind in "fi" and c in o.columns]
                diff = float(np.nanmax(np.abs(S[num].to_numpy(float) - o[num].to_numpy(float)))) if num else np.nan
                row.update(repeat_identical=int(diff == 0.0), repeat_max_abs_diff=diff, verdict="REPEAT_IDENTICAL" if diff == 0.0 else "REPEAT_DIFFERS")
            rows.append(row); continue
        row["S_mm"] = float(T["step_diff_mm"].max()) if len(T) else np.nan
        row["S_frame"] = int(T.loc[T["step_diff_mm"].idxmax(), "frame"]) if len(T) else None
        ca = rd(os.path.join(a.runs, pr["driver_control"]), "pos_trace_all.csv.gz")
        cb = rd(os.path.join(a.runs, pr["shadow_config_control"]), "pos_trace_all.csv.gz")
        if ca is None or cb is None or len(T) < 3:
            row.update(verdict="MISSING", note="control pos_trace_all.csv.gz or shadow rows missing"); rows.append(row); continue
        fr_ = T["frame"].to_numpy(int)
        pa, oka = pos_at(ca, fr_); pb, okb = pos_at(cb, fr_)
        ok = oka & okb
        row["n_frames_matched_by_scan_id"] = int(ok.sum())
        stepA, stepB = np.diff(pa, axis=0) * 1e3, np.diff(pb, axis=0) * 1e3
        okd = ok[1:] & ok[:-1]
        dd = np.linalg.norm(stepB - stepA, axis=1)
        dd = np.where(okd, dd, np.nan)
        row["D_real_mm"] = float(np.nanmax(dd)) if okd.any() else np.nan
        row["D_frame"] = int(T["frame"].to_numpy()[1:][int(np.nanargmax(dd))]) if okd.any() else None
        # the driver of the shadow arm must follow its plain control (the shadow has no effect on the driver)
        sd = S[S["role"] == "driver"].set_index("frame").loc[T["frame"], ["post_pos_x", "post_pos_y", "post_pos_z"]].to_numpy(float)
        row["driver_vs_control_max_mm"] = float(np.nanmax(np.where(oka[:, None], np.abs(sd - pa), np.nan)) * 1e3)
        ratio = row["S_mm"] / row["D_real_mm"] if row.get("D_real_mm") and row["D_real_mm"] > 0 else np.nan
        row["ratio"] = float(ratio)
        if ck["map_digest_changed_scans"]:
            row["verdict"] = "invalid_map_changed"
        elif not np.isfinite(ratio):
            row["verdict"] = "undefined"
        elif ratio >= 0.5:
            row["verdict"] = "update_fails_on_healthy_state"
        elif ratio < 0.25:
            row["verdict"] = "needs_accumulated_history"
        else:
            row["verdict"] = "partial"
        row["S_rot_mrad"] = float(T["step_rot_diff_mrad"].max())
        row["prior_trace_pos_ratio_shadow_over_driver_mean"] = float((T["prior_trace_pos_shadow"] / T["prior_trace_pos_driver"]).mean())
        row["post_trace_pos_ratio_shadow_over_driver_mean"] = float((T["post_trace_pos_shadow"] / T["post_trace_pos_driver"]).mean())
        rows.append(row)
    R = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    R.to_csv(os.path.join(a.out, "r74_shadow_pairs.csv"), index=False, float_format="%.9g")  # no time column in this table
    json.dump(json.loads(R.to_json(orient="records")), open(os.path.join(a.out, "r74_shadow_pairs.json"), "w"), indent=1)
    # freeze validity
    for fz in sorted(x for x in os.listdir(a.runs) if x.startswith("r74_fz_")):
        ctl = fz.replace("r74_fz_", "r74_sh_").replace("_fq", "_ctl_fq").replace("_g5a1", "_ctl_g5a1").replace("eee02_mq", "eee02_ctl_mq")
        F, C = rd(os.path.join(a.runs, fz), "scan_log_all.csv.gz"), rd(os.path.join(a.runs, ctl), "scan_log_all.csv.gz")
        r = dict(freeze_arm=fz, control_arm=ctl)
        if F is None or C is None or not len(F) or not len(C):
            r["valid"] = "MISSING"
        else:
            f1, c1 = F.sort_values("frame").iloc[0], C.sort_values("frame").iloc[0]
            r.update(frame1_digest_freeze=f1["map_digest_exact"], frame1_digest_control=c1["map_digest_exact"], frame1_equal=int(f1["map_digest_exact"] == c1["map_digest_exact"]),
                     distinct_digests_after_frame1_freeze=int(F[F["frame"] > f1["frame"]]["map_digest_exact"].nunique()), frozen_scans_last=int(F["map_frozen_scans"].iloc[-1]),
                     n_frames=int(len(F)), distinct_digests_control=int(C["map_digest_exact"].nunique()))
            r["valid"] = "VALID" if r["frame1_equal"] and r["distinct_digests_after_frame1_freeze"] <= 1 and r["frozen_scans_last"] > 0 else "INVALID"
        frz.append(r)
    pd.DataFrame(frz).to_csv(os.path.join(a.out, "r74_freeze_check.csv"), index=False)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a1 = sp.add_parser("arm"); a1.add_argument("--dir", required=True); a1.add_argument("--out", required=True)
    a2 = sp.add_parser("pairs"); a2.add_argument("--runs", required=True); a2.add_argument("--pairs", required=True); a2.add_argument("--out", required=True)
    a = ap.parse_args()
    return arm_main(a) if a.cmd == "arm" else pairs_main(a)


if __name__ == "__main__":
    sys.exit(main())
