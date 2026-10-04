#!/usr/bin/env python3
"""R61: reduce one arm's raw diagnostics to a compact per-scan table and a summary json.

python3 reduce_arm.py <arm_dir> [--lever -0.293656 -0.012288 -0.273095] [--move-thresh 0.05] [--margin 2.0]

Reads (missing files are tolerated and reported): state_trace.csv, state_reference.txt, lio_gt_matched.csv,
imu_cov_growth.csv, motion_q_scan.csv, joint_knot_lidar_information.csv, arm.json (written by post_queue.sh).
Writes: arm_scan.csv, arm_ol_tau.csv (open-loop arms), arm_summary.json.

Conventions (see state_trace.h): R is body->world, errors are right/body perturbations e_R = Log(R_ref^T R), e_p = p - p_ref,
e_v = v - v_ref, all against the STATIONARY reference (valid only while stationary). Against GT (position only) errors are
formed in the GT frame after a rigid alignment fitted on a stated segment. Nothing here interprets results.
"""
import argparse, json, math, os, sys
import numpy as np
import pandas as pd
from common import *

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir")
    ap.add_argument("--lever", nargs=3, type=float, default=[-0.293656, -0.012288, -0.273095])
    ap.add_argument("--move-thresh", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=2.0)
    a = ap.parse_args()
    D = a.arm_dir
    warn = []
    meta = {}
    if os.path.exists(os.path.join(D, "arm.json")):
        meta = json.load(open(os.path.join(D, "arm.json")))
    S = {"arm": meta.get("arm", os.path.basename(os.path.abspath(D))), "meta": meta, "warnings": warn}

    tr_path = os.path.join(D, "state_trace.csv")
    ref_path = os.path.join(D, "state_reference.txt")
    if not (os.path.exists(tr_path) and os.path.exists(ref_path)):
        warn.append("state_trace.csv or state_reference.txt missing: nothing to reduce")
        json.dump(jsonable(S), open(os.path.join(D, "arm_summary.json"), "w"), indent=1)
        return 1
    df, pcols = load_trace(tr_path)
    Rref, pref, vref, P0 = read_reference(ref_path)
    lever = np.array(a.lever)

    imu = df[df.phase == "post_imu"].drop_duplicates("scan_id", keep="last").set_index("scan_id")
    lio = df[df.phase == "post_lio"].drop_duplicates("scan_id", keep="last").set_index("scan_id")
    scans = sorted(set(imu.index) & set(lio.index))
    if not scans:
        warn.append("no scan has both post_imu and post_lio rows")
        json.dump(jsonable(S), open(os.path.join(D, "arm_summary.json"), "w"), indent=1)
        return 1

    def state_of(row):
        R = quat_to_R(np.array([row.qw, row.qx, row.qy, row.qz]))
        return R, np.array([row.px, row.py, row.pz]), np.array([row.vx, row.vy, row.vz])

    def P_of(row, n):
        return cov_from_row(row, pcols, n)

    recs = []
    for sc in scans:
        r = {"scan": sc, "t": imu.loc[sc].t_abs, "ol_window": int(imu.loc[sc].ol_window), "ol_reset_after": int(lio.loc[sc].ol_reset),
             "resid": int(lio.loc[sc].residual_count), "iters": int(lio.loc[sc].completed_iterations)}
        for ph, tab in (("imu", imu), ("lio", lio)):
            row = tab.loc[sc]
            R, p, v = state_of(row)
            P = P_of(row, 9)
            e = np.concatenate([log_so3(Rref.T @ R), p - pref, v - vref])
            r["_R_" + ph], r["_p_" + ph], r["_v_" + ph], r["_P_" + ph], r["_e_" + ph] = R, p, v, P, e
            r["_g_" + ph] = np.array([row.gx, row.gy, row.gz])
            for i, ax in enumerate(AXES):
                r["e_%s_%s" % (ph, ax)] = e[i]
                r["sd_%s_%s" % (ph, ax)] = math.sqrt(max(P[i, i], 0.0))
            try:
                r["nees_%s" % ph] = float(e @ np.linalg.solve(P, e))
            except np.linalg.LinAlgError:
                r["nees_%s" % ph] = float("nan")
        recs.append(r)
    t = np.array([r["t"] for r in recs])
    dt0 = float(np.median(np.diff(t))) if len(t) > 1 else 0.1

    # ---------- GT: matching, movement onset ----------
    gt = None
    gpath = os.path.join(D, "lio_gt_matched.csv")
    if os.path.exists(gpath):
        gt = pd.read_csv(gpath)
    t_move, src_move = None, None
    if gt is not None and len(gt) > 10:
        g = gt[["gt_px", "gt_py", "gt_pz"]].to_numpy()
        base = np.median(g[: max(5, int(0.05 * len(g)))], axis=0)
        d = np.linalg.norm(g - base, axis=1)
        k = np.where(d > a.move_thresh)[0]
        if len(k):
            t_move, src_move = float(gt.t_abs.iloc[k[0]]), "gt_displacement>%.3f" % a.move_thresh
    if t_move is None:
        pl = np.array([np.linalg.norm(r["_p_lio"] - pref) for r in recs])
        k = np.where(pl > 0.10)[0]
        if len(k):
            t_move, src_move = float(t[k[0]]), "estimate_displacement>0.10 (no GT onset)"
            warn.append("movement onset taken from the estimate, not GT")
    S["t_move"], S["t_move_source"] = t_move, src_move
    S["t_first"], S["t_last"], S["n_scans"], S["dt_scan_median"] = float(t[0]), float(t[-1]), len(recs), dt0
    stat_mask = np.array([(t_move is None) or (r["t"] < t_move - a.margin) for r in recs])
    S["n_stationary_scans"] = int(stat_mask.sum())

    # ---------- stationary consistency (per phase) ----------
    def stat_block(ph, mask):
        E = np.array([r["_e_" + ph] for r, m in zip(recs, mask) if m])
        Pd = np.array([np.diag(r["_P_" + ph]) for r, m in zip(recs, mask) if m])
        nees = np.array([r["nees_" + ph] for r, m in zip(recs, mask) if m])
        tt = t[mask]
        out = {"n": int(len(E))}
        if len(E) < 3:
            return out
        for i, ax in enumerate(AXES):
            e, pv = E[:, i], np.maximum(Pd[:, i], 1e-300)
            z = e / np.sqrt(pv)
            lo, hi = block_bootstrap_ratio(e ** 2, pv, tt)
            out[ax] = {"rms_e": float(np.sqrt(np.mean(e ** 2))), "mean_sd": float(np.mean(np.sqrt(pv))),
                       "ratio_e2_over_P": float(np.sum(e ** 2) / np.sum(pv)), "ratio_ci90": [lo, hi],
                       "mean_z2": float(np.mean(z ** 2)), "cov1": float(np.mean(np.abs(z) < 1)), "cov2": float(np.mean(np.abs(z) < 2)),
                       "mean_e": float(np.mean(e)), "P_ii_first": float(pv[0]), "P_ii_last": float(pv[-1])}
        ok = np.isfinite(nees)
        if ok.any():
            lo95, hi95 = chi2_ppf(0.025, 9), chi2_ppf(0.975, 9)
            nn = nees[ok]
            out["nees9"] = {"mean": float(nn.mean()), "median": float(np.median(nn)), "frac_in_95": float(np.mean((nn > lo95) & (nn < hi95))),
                            "frac_above": float(np.mean(nn >= hi95)), "lag1": float(np.corrcoef(nn[:-1], nn[1:])[0, 1]) if len(nn) > 3 else None,
                            "dof": 9, "n_finite": int(ok.sum())}
        # 10 s block series of ratio_e2_over_P per axis
        blk = np.floor((tt - tt.min()) / 10.0).astype(int)
        out["blocks10s"] = [{"t0": float(tt.min() + 10 * b), **{ax: float(np.sum(E[blk == b, i] ** 2) / np.sum(np.maximum(Pd[blk == b, i], 1e-300)))
                            for i, ax in enumerate(AXES)}} for b in np.unique(blk)]
        return out

    S["stationary"] = {"post_imu": stat_block("imu", stat_mask), "post_lio": stat_block("lio", stat_mask)}
    S["first_frame_check"] = {"P0_diag_9": [float(P0[i, i]) for i in range(min(9, len(P0)))],
                              "first_post_imu_e": recs[0]["_e_imu"].tolist(),
                              "first_post_imu_sd": [recs[0]["sd_imu_" + ax] for ax in AXES]}

    # ---------- open-loop: drift ensemble by time since window start ----------
    is_open = meta.get("kind") == "open" or (df.ol_window.max() > 0)
    if meta.get("kind") == "open" or "propagate_only" in json.dumps(meta):
        is_open = True
    S["open_loop"] = bool(is_open)
    if is_open:
        rows = []
        wins = {}
        for r, m in zip(recs, stat_mask):
            if m:
                wins.setdefault(r["ol_window"], []).append(r)
        for w, lst in wins.items():
            t0 = lst[0]["t"] - dt0
            for r in lst:
                r["tau"] = r["t"] - t0
        bins = {}
        for w, lst in wins.items():
            for r in lst:
                bins.setdefault(int(math.floor(r["tau"])), []).append(r)
        for b in sorted(bins):
            lst = bins[b]
            E = np.array([r["_e_imu"] for r in lst]); Pd = np.array([np.diag(r["_P_imu"]) for r in lst])
            nees = np.array([r["nees_imu"] for r in lst])
            row = {"tau_bin_s": b, "n_samples": len(lst), "n_windows": len({r["ol_window"] for r in lst})}
            for i, ax in enumerate(AXES):
                row["rms_e_" + ax] = float(np.sqrt(np.mean(E[:, i] ** 2)))
                row["sd_" + ax] = float(np.sqrt(np.mean(Pd[:, i])))
                row["ratio_" + ax] = float(np.sum(E[:, i] ** 2) / max(np.sum(Pd[:, i]), 1e-300))
            row["nees9_mean"] = float(np.nanmean(nees))
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(D, "arm_ol_tau.csv"), index=False)
        S["open_loop_windows"] = {int(w): len(l) for w, l in wins.items()}

    # ---------- phase-3 correction analysis (closed loop, stationary) ----------
    if not is_open:
        corr = {}
        idx = [i for i, m in enumerate(stat_mask) if m]
        if len(idx) >= 5:
            Ei = np.array([recs[i]["_e_imu"] for i in idx]); El = np.array([recs[i]["_e_lio"] for i in idx])
            Pi = np.array([np.diag(recs[i]["_P_imu"]) for i in idx]); Pl = np.array([np.diag(recs[i]["_P_lio"]) for i in idx])
            tt = t[idx]
            for i, ax in enumerate(AXES):
                c = El[:, i] - Ei[:, i]
                den = np.sum(Ei[:, i] ** 2)
                k = float(-np.sum(c * Ei[:, i]) / den) if den > 0 else float("nan")
                g = float(np.mean(1.0 - Pl[:, i] / np.maximum(Pi[:, i], 1e-300)))
                rho = float(np.sum(El[:, i] ** 2) / den) if den > 0 else float("nan")
                r_pred = float(np.mean(Pl[:, i] / np.maximum(Pi[:, i], 1e-300)))
                lo, hi = block_bootstrap_ratio(El[:, i] ** 2, np.maximum(Ei[:, i] ** 2, 1e-300), tt)
                corr[ax] = {"gain_k": k, "predicted_gain": g, "k_over_predicted": (k / g if g > 0 else float("nan")),
                            "realised_err2_ratio": rho, "realised_ci90": [lo, hi], "predicted_P_ratio": r_pred,
                            "rms_e_imu": float(np.sqrt(np.mean(Ei[:, i] ** 2))), "rms_e_lio": float(np.sqrt(np.mean(El[:, i] ** 2))),
                            "frac_scans_error_sign_flips": float(np.mean(np.sign(El[:, i]) != np.sign(Ei[:, i])))}
        S["phase3_stationary"] = corr

    # ---------- dynamic, GT position only ----------
    dyn = {}
    if gt is not None and t_move is not None:
        gt_t = gt.t_abs.to_numpy()
        joined = []
        GTP = gt[["gt_px", "gt_py", "gt_pz"]].to_numpy()
        for r in recs:
            # GT rows are on their own 20 Hz grid (not the scan grid): linearly interpolate GT to the scan time,
            # only inside the GT span and with a neighbouring sample within 0.06 s.
            if gt_t[0] <= r["t"] <= gt_t[-1]:
                j = int(np.searchsorted(gt_t, r["t"]))
                j1 = min(max(j, 1), len(gt_t) - 1); j0 = j1 - 1
                if gt_t[j1] - gt_t[j0] < 0.12:
                    w = (r["t"] - gt_t[j0]) / max(gt_t[j1] - gt_t[j0], 1e-9)
                    r["_gtp"] = (1 - w) * GTP[j0] + w * GTP[j1]
                    joined.append((r, j0))
        S["gt_join_rate"] = len(joined) / max(len(recs), 1)
        dj = [(r, j) for r, j in joined if r["t"] >= t_move]
        if len(dj) >= 10:
            gtp = np.array([r["_gtp"] for r, _ in dj])
            pl = np.array([r["_p_lio"] + r["_R_lio"] @ lever for r, _ in dj])
            pi = np.array([r["_p_imu"] + r["_R_imu"] @ lever for r, _ in dj])
            tt = np.array([r["t"] for r, _ in dj])
            # direction of travel from GT (1 s baseline) and vertical from the estimator's gravity state
            k1 = max(1, int(round(1.0 / dt0)))
            vel = np.zeros_like(gtp)
            for i in range(len(gtp)):
                a0, b0 = max(0, i - k1), min(len(gtp) - 1, i + k1)
                vel[i] = (gtp[b0] - gtp[a0]) / max(tt[b0] - tt[a0], 1e-6)
            variants = {"fit_first_half": np.arange(len(dj)) < len(dj) // 2, "fit_all": np.ones(len(dj), bool)}
            for vn, fit in variants.items():
                ev_ = np.linalg.eigvalsh(np.cov(gtp[fit].T))
                if ev_[1] < 1e-3 * max(ev_[2], 1e-12):
                    warn.append("%s: fitted GT segment is near-collinear (eig ratio %.2g); alignment rotation about the travel axis is unobservable, treat lateral/vertical splits with care" % (vn, ev_[1] / max(ev_[2], 1e-12)))
                Ral, tal = umeyama(pl[fit], gtp[fit])
                ev = ~fit if vn == "fit_first_half" else fit
                out = {"align_rot_deg": float(np.degrees(np.linalg.norm(log_so3(Ral)))), "n_fit": int(fit.sum()), "n_eval": int(ev.sum())}
                axes_e = {"imu": [], "lio": []}
                U = []
                for n_, (r, _) in enumerate(dj):
                    sp = np.linalg.norm(vel[n_])
                    along = vel[n_] / sp if sp > 0.05 else np.array([1.0, 0, 0])
                    up = Ral @ (-r["_g_lio"] / np.linalg.norm(r["_g_lio"]))
                    lat = np.cross(up, along); lat /= max(np.linalg.norm(lat), 1e-9)
                    up2 = np.cross(along, lat)
                    U.append(np.stack([along, lat, up2]))
                U = np.array(U)
                res = {}
                for ph, ptab in (("imu", pi), ("lio", pl)):
                    e_gt = (Ral @ ptab.T).T + tal - gtp
                    sd, ee, nees = [], [], []
                    for n_, (r, _) in enumerate(dj):
                        R = r["_R_" + ph]
                        Pfull = cov_from_row((imu if ph == "imu" else lio).loc[r["scan"]], pcols, 6)
                        A = np.hstack([-R @ skew(lever), np.eye(3)])   # d(est_pos_corrected) / d[dtheta_body, dp]
                        Pc = Ral @ (A @ Pfull @ A.T) @ Ral.T
                        Pu = U[n_] @ Pc @ U[n_].T
                        eu = U[n_] @ e_gt[n_]
                        sd.append(np.sqrt(np.maximum(np.diag(Pu), 0)))
                        ee.append(eu)
                        try:
                            nees.append(float(e_gt[n_] @ np.linalg.solve(Pc, e_gt[n_])))
                        except np.linalg.LinAlgError:
                            nees.append(float("nan"))
                    res[ph] = {"e": np.array(ee), "sd": np.array(sd), "nees": np.array(nees), "e_gt": e_gt}
                for ph in ("imu", "lio"):
                    E_, S_ = res[ph]["e"][ev], res[ph]["sd"][ev]
                    o = {}
                    for i, ax in enumerate(["along", "lateral", "vertical"]):
                        o[ax] = {"rms_e": float(np.sqrt(np.mean(E_[:, i] ** 2))), "mean_sd": float(np.mean(S_[:, i])),
                                 "ratio_e2_over_P": float(np.sum(E_[:, i] ** 2) / max(np.sum(S_[:, i] ** 2), 1e-300)),
                                 "cov2": float(np.mean(np.abs(E_[:, i]) < 2 * np.maximum(S_[:, i], 1e-12)))}
                    nn = res[ph]["nees"][ev]
                    nn = nn[np.isfinite(nn)]
                    o["nees_p3_mean"] = float(nn.mean()) if len(nn) else None
                    o["nees_p3_frac_above_95"] = float(np.mean(nn > chi2_ppf(0.975, 3))) if len(nn) else None
                    o["rms_e_norm"] = float(np.sqrt(np.mean(np.sum(res[ph]["e_gt"][ev] ** 2, axis=1))))
                    out[ph] = o
                # correction gain per direction (imu -> lio)
                gain = {}
                Ei_, El_ = res["imu"]["e"][ev], res["lio"]["e"][ev]
                Si_, Sl_ = res["imu"]["sd"][ev], res["lio"]["sd"][ev]
                for i, ax in enumerate(["along", "lateral", "vertical"]):
                    c = El_[:, i] - Ei_[:, i]
                    den = np.sum(Ei_[:, i] ** 2)
                    k = float(-np.sum(c * Ei_[:, i]) / den) if den > 0 else float("nan")
                    g = float(np.mean(1 - Sl_[:, i] ** 2 / np.maximum(Si_[:, i] ** 2, 1e-300)))
                    gain[ax] = {"gain_k": k, "predicted_gain": g, "k_over_predicted": k / g if g > 0 else float("nan"),
                                "realised_err2_ratio": float(np.sum(El_[:, i] ** 2) / den) if den > 0 else float("nan"),
                                "predicted_P_ratio": float(np.mean(Sl_[:, i] ** 2 / np.maximum(Si_[:, i] ** 2, 1e-300)))}
                out["correction"] = gain
                dyn[vn] = out
                if vn == "fit_first_half":
                    for n_, (r, _) in enumerate(dj):
                        r["dyn_ok"] = 1
                        for ph in ("imu", "lio"):
                            for i, ax in enumerate(["along", "lateral", "vertical"]):
                                r["gt_e_%s_%s" % (ph, ax)] = res[ph]["e"][n_, i]
                                r["gt_sd_%s_%s" % (ph, ax)] = res[ph]["sd"][n_, i]
                            r["gt_nees_%s" % ph] = res[ph]["nees"][n_]
        else:
            warn.append("fewer than 10 GT-matched dynamic scans: dynamic analysis skipped")
    elif gt is None:
        warn.append("lio_gt_matched.csv missing: no GT analysis")
    S["dynamic_gt_position"] = dyn
    S["rotation_velocity_dynamic"] = "no GT available (Leica is position-only): not assessed"

    # ---------- other sources ----------
    def seg_median(dfx, tcol="t_abs"):
        if dfx is None or tcol not in dfx.columns or t_move is None:
            return {}
        num = dfx.select_dtypes(include=[np.number]).drop(columns=[c for c in ("scan_id", "scan_index", "iteration", tcol) if c in dfx.columns], errors="ignore")
        out = {}
        for nm, m in (("stationary", dfx[tcol] < t_move - a.margin), ("dynamic", dfx[tcol] >= t_move)):
            if m.any():
                out[nm] = {c: float(num.loc[m, c].median()) for c in num.columns}
        return out
    for nm, fn in (("motion_q_scan", "motion_q_scan.csv"), ("imu_cov_growth", "imu_cov_growth.csv"),
                   ("lidar_information", "joint_knot_lidar_information.csv")):
        p = os.path.join(D, fn)
        if os.path.exists(p):
            try:
                S[nm] = seg_median(pd.read_csv(p))
            except Exception as ex:
                warn.append("%s: %s" % (fn, ex))
        else:
            S[nm] = "missing"

    # ---------- write ----------
    rows = []
    for r in recs:
        rows.append({k: v for k, v in r.items() if not k.startswith("_")})
    out = pd.DataFrame(rows)
    out.insert(2, "seg", ["stationary" if m else ("dynamic" if (t_move is not None and r["t"] >= t_move) else "transition")
                          for r, m in zip(recs, stat_mask)])
    out.to_csv(os.path.join(D, "arm_scan.csv"), index=False, float_format="%.6g")
    json.dump(jsonable(S), open(os.path.join(D, "arm_summary.json"), "w"), indent=1)
    print("reduced %s: %d scans, %d stationary, t_move=%s" % (S["arm"], len(recs), int(stat_mask.sum()), t_move))
    return 0

if __name__ == "__main__":
    sys.exit(main())
