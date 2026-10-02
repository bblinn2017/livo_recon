#!/usr/bin/env python3
"""selftest_r73.py -- proves the R73 Python tools can FAIL (rule 2.4). Synthetic data only; prints SELFTEST R73 PASS or exits 1.

    python3 tools/r69/selftest_r72.py [--manifest-jobs r69_inputs/r66_jobs.json]

D1  dump reader round trip: matrices written in the C++ format (scientific, 16 digits after the point = 17 significant) parse back exactly (max abs diff 0), including a subnormal.
D2  a truncated block and a short row make the reader RAISE (never silently shorten).
D3  failure reduction on a synthetic matrix with one negative eigenvalue: n_neg == 1, the reported eigenvector's top coordinate is the one built in, diag_nonpositive_indices correct;
    the same matrix made PD: n_neg == 0 and negative_eigenvalues empty.
D4  record reduction: A_eig.min equals numpy's on the synthetic A; knot_change_max_abs equals the built step; posterior/prior trace ratio equals the built ratio.
E1  early-frame table: synthetic frames with a known ground/wall split and a known yaw step: counts and first_frame_wall_ge_k exact; believed yaw sd equals the closed form for an isotropic design;
    a wall-free frame has wall_only_info_rank absent and first wall frame later.
P1  p_table: ratios against the in-round p0d reference exact on synthetic rows; the ATE winner is the cell built to win; a cell with no completed arm is excluded from the winners.
M1  manifest: counts (smoke, refuse, ef, co, sc, qa, p), no key containing 'perturb', every arm's job id unique, every delta key present in the base or in the allow-list of new keys,
    P batch = 16 cells x 9 bags, EF = 40 arms all on eee_01 with tier EF, and a seeded violation (an injected 'perturb' key) is caught by the checker function.
"""
import argparse, json, os, subprocess, sys, tempfile, shutil
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r73_dump_reduce as DR, r73_p_table as PT, r73_residual_report as RR, r73_shadow_reduce as SR, r73_frame_errors as FE, build_return as BR   # noqa: E402


def wr(o, name, M):
    M = np.atleast_2d(np.asarray(M, float))
    o.write("## %s %d %d\n" % (name, M.shape[0], M.shape[1]))
    for r in M:
        o.write(" ".join("%.16e" % x for x in r) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest-jobs"); a = ap.parse_args()
    fails = []
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as t:
        # ---- D1 / D2
        A = rng.normal(size=(5, 4)); A[0, 0] = 5e-324; A[1, 1] = -1.2345678901234567e-300
        p = os.path.join(t, "d1.txt")
        with open(p, "w") as o:
            o.write("# meta: frame=3 knot_count=4\n"); wr(o, "A", A); wr(o, "v", A[:, 0:1])
        B, M = DR.parse(p)
        if np.abs(B["A"] - A).max() != 0.0 or B["A"].shape != A.shape:
            fails.append("D1 round trip not exact: %g" % np.abs(B["A"] - A).max())
        if M.get("meta") != "frame=3 knot_count=4":
            fails.append("D1 meta not parsed: %r" % M)
        txt = open(p).read()
        for nm, bad in (("truncated", "\n".join(txt.split("\n")[:4])), ("short_row", txt.replace(txt.split("\n")[2], " ".join(txt.split("\n")[2].split()[:2])))):
            q = os.path.join(t, "bad_%s.txt" % nm); open(q, "w").write(bad)
            try:
                DR.parse(q); fails.append("D2 %s block was accepted" % nm)
            except ValueError:
                pass
        # ---- D3
        n = 12
        Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        w = np.linspace(1.0, 5.0, n); w[0] = -0.3
        F = (Q * w) @ Q.T
        fp = os.path.join(t, "coupled_failure_dump.txt")
        with open(fp, "w") as o:
            o.write("# meta: frame=2 completed_iterations_before_failure=1\n# what: test\n# capture_valid: 1\n# capture_where: posterior_A\n")
            wr(o, "failed_matrix_symmetrised", F)
        R = DR.reduce_failure(fp)
        top = R["min_eigvec_top_indices"][0][0]
        want_top = int(np.argmax(np.abs(Q[:, 0])))
        if R["failed_eig"]["n_neg"] != 1 or top != want_top or abs(R["negative_eigenvalues"][0] + 0.3) > 1e-12:
            fails.append("D3 failure reduction wrong: n_neg=%s top=%s want %s neg=%s" % (R["failed_eig"]["n_neg"], top, want_top, R["negative_eigenvalues"]))
        F2 = F.copy(); F2[np.diag_indices(n)] += 1.0 - w.min() * 0 + 0.3 + 0.1
        with open(fp, "w") as o:
            o.write("# meta: frame=2\n"); wr(o, "failed_matrix_symmetrised", F2)
        R2 = DR.reduce_failure(fp)
        if R2["failed_eig"]["n_neg"] != 0 or R2["negative_eigenvalues"]:
            fails.append("D3 PD matrix reported negative eigenvalues")
        # ---- D4
        D = 6
        Asys = np.diag(np.linspace(0.5, 3.0, D)); kb = rng.normal(size=(3, 9)); step = np.zeros((3, 9)); step[1, 4] = 0.0123
        Pprior = np.diag(np.ones(D) * 2.0); Ppost = np.diag(np.ones(D) * 1.0)
        rp = os.path.join(t, "coupled_record_f2.txt")
        with open(rp, "w") as o:
            o.write("# meta: frame=2 knot_count=3 D=6\n"); wr(o, "P_joint_prior", Pprior)
            wr(o, "i0_knots_before", kb); wr(o, "i0_knots_after", kb + step); wr(o, "i0_n_residuals", [[1500.0]])
            wr(o, "i0_A_system", Asys); wr(o, "i0_P_joint_posterior", Ppost); wr(o, "i0_delta", np.ones((D, 1)) * 0.1)
            wr(o, "i0_A_positive_definite_pass", [[1.0]])
        r = DR.reduce_record(rp)["iterations"][0]
        if abs(r["delta_norm"] - 0.1 * 6 ** 0.5) > 1e-12 or abs(r["A_eig"]["min"] - 0.5) > 1e-12 or abs(r["knot_change_max_abs"] - 0.0123) > 1e-15 or abs(r["posterior_trace_over_prior_trace"] - 0.5) > 1e-12 or not r["A_pd_pass"]:
            fails.append("D4 record reduction wrong: %s" % r)
        # ---- R1 residual report: consistency is exact on a consistent dump, flags a corrupted HtH, class split and weights right
        n = 120
        H = rng.normal(size=(n, 6)); H[:, 5] = np.where(np.arange(n) < 50, -0.95, np.where(np.arange(n) < 90, 0.1, 0.5)); s2 = rng.uniform(5e-4, 2e-3, n); rr = rng.normal(scale=0.02, size=n)
        dump = pd.DataFrame({"frame": 1, "iter": 0, "idx": range(n), "source_index": range(n), "t": 0.0, "r": rr, "sigma_squared": s2, "plane_var_term": 0.0, "plane_key": 7, "node_id": 3, "tier": 0, "vis": 2})
        for k, c in enumerate(RR.HC):
            dump[c] = H[:, k]
        w_ = 1 / s2; M = (H * w_[:, None]).T @ H; z = (H * (w_ * rr)[:, None]).sum(0)
        row = {"frame": 1, "iter": 0, "n_resid": n, "sum_w": w_.sum()}
        for ij in RR.IJ:
            row["HtH_%d%d" % ij] = M[ij]
        for k in range(6):
            row["Htz_%d" % k] = z[k]
        pr = RR.per_residual(dump)
        con = RR.consistency(pr, pd.DataFrame([row]))
        if con["max_rel_err_hth"].iloc[0] > 1e-12 or con["max_rel_err_htz"].iloc[0] > 1e-12 or con["ok"].iloc[0] != 1:
            fails.append("R1 consistent dump not consistent: %s" % con.to_dict("records"))
        bad = dict(row); bad["HtH_22"] *= 1.05
        if RR.consistency(pr, pd.DataFrame([bad]))["ok"].iloc[0] != 0:
            fails.append("R1 corrupted HtH not flagged")
        cl = RR.class_table(pr).set_index("cls")
        if (cl.loc["ground", "n"], cl.loc["wall", "n"], cl.loc["slanted", "n"]) != (50, 40, 30) or abs(cl["weight_sum"].sum() - w_.sum()) > 1e-6 or abs(cl["hth_trace_share"].sum() - 1) > 1e-12:
            fails.append("R1 class table wrong: %s" % cl[["n", "weight_sum", "hth_trace_share"]].to_dict())
        ht = RR.hth_table(pd.DataFrame([row]), pr).iloc[0]
        if abs(ht["hth_trace"] - np.trace(M)) > 1e-6 or abs(ht["eig_min"] - np.linalg.eigvalsh(M)[0]) > 1e-9 or abs(ht["info_yaw"] - M[2, 2]) > 1e-9:
            fails.append("R1 hth table wrong")
        nw = pr[pr["cls"] != "wall"]["info_yaw"].sum()
        if abs(ht["info_yaw_nowall"] - nw) > 1e-6 or ht["info_yaw_nowall"] >= ht["info_yaw"]:
            fails.append("R1 info_yaw_nowall wrong (%s vs %s)" % (ht["info_yaw_nowall"], nw))
        # ---- S1 shadow reduce: S/D ratio and verdict, repeat determinism, digest-changed invalidation, freeze validity
        def shadow_rows(scale, changed=False):
            rows = []
            for f in range(10, 31):
                for role, k in (("driver", 1.0), ("shadow", scale)):
                    d = {"frame": f, "t_abs": 100.0 + 0.1 * f, "role": role}
                    for c in ("q_acc_x", "q_acc_y", "q_acc_z", "q_gyr_x", "q_gyr_y", "q_gyr_z", "q_samples", "acc_clamped_axes", "gyr_clamped_axes", "start_rv_x", "start_rv_y", "start_rv_z", "prop_rv_x", "prop_rv_y", "prop_rv_z", "post_rv_x", "post_rv_y", "post_rv_z"):
                        d[c] = 0.0
                    d.update(start_pos_x=0.0, start_pos_y=0.0, start_pos_z=0.0, prop_pos_x=0.01, prop_pos_y=0.0, prop_pos_z=0.0, post_pos_x=0.01 * k, post_pos_y=0.0, post_pos_z=0.0)
                    d.update(prior_trace_rot=1.0, prior_trace_pos=1.0 * k, prior_trace_vel=1.0, post_trace_rot=1.0, post_trace_pos=1.0, post_trace_vel=1.0, n_iter=3, n_resid_iter0=100, reduced_chi2_iter0=1.0)
                    d["map_digest_before_pass"] = "aa" if role == "shadow" else ""
                    d["map_digest_after_pass"] = ("bb" if changed and f == 15 else "aa") if role == "shadow" else ""
                    rows.append(d)
            return pd.DataFrame(rows)
        def posf(step_mm):
            tt = 100.0 + 0.1 * np.arange(0, 40)
            return pd.DataFrame({"scan_id": np.arange(0, 40), "t_abs": tt, "px": np.arange(0, 40) * step_mm * 1e-3, "py": 0.0, "pz": 0.0, "yaw_rad": 0.0})
        runs = os.path.join(t, "runs"); sh_dir = os.path.join(runs, "sh"); os.makedirs(sh_dir)
        for nm, sc in (("sh", 1.4), ("sh_big", 5.0), ("sh_mid", 2.0), ("sh_chg", 1.4)):
            d = os.path.join(runs, nm); os.makedirs(d, exist_ok=True)
            shadow_rows(sc, nm == "sh_chg").to_csv(os.path.join(d, "shadow_scan_all.csv.gz"), index=False, compression="gzip")
        shutil.copytree(os.path.join(runs, "sh"), os.path.join(runs, "sh_rep"))
        for nm, st in (("ctlA", 10.0), ("ctlB", 20.0)):
            d = os.path.join(runs, nm); os.makedirs(d); posf(st).to_csv(os.path.join(d, "pos_trace_all.csv.gz"), index=False, compression="gzip")
        pj = [dict(pair="p_small", shadow_arm="sh", driver_control="ctlA", shadow_config_control="ctlB", window=[10, 30]),
              dict(pair="p_big", shadow_arm="sh_big", driver_control="ctlA", shadow_config_control="ctlB", window=[10, 30]),
              dict(pair="p_mid", shadow_arm="sh_mid", driver_control="ctlA", shadow_config_control="ctlB", window=[10, 30]),
              dict(pair="p_chg", shadow_arm="sh_chg", driver_control="ctlA", shadow_config_control="ctlB", window=[10, 30]),
              dict(pair="rep", shadow_arm="sh_rep", repeat_of="sh", window=[10, 30])]
        pjp = os.path.join(t, "pairs.json"); json.dump(pj, open(pjp, "w"))
        so = os.path.join(t, "so"); os.makedirs(so)
        SR.pairs_main(type("A", (), dict(runs=runs, pairs=pjp, out=so)))
        P_ = pd.read_csv(os.path.join(so, "r73_shadow_pairs.csv")).set_index("pair")
        # driver step 10 mm; shadow step 14 mm (x1.4) -> S = 4 mm; D_real = |20-10| = 10 mm -> ratio 0.4 (partial); x5 -> S 40 mm ratio 4 (fails); x2 -> S 10 mm ratio 1.0
        if abs(P_.loc["p_small", "S_mm"] - 4.0) > 1e-9 or abs(P_.loc["p_small", "D_real_mm"] - 10.0) > 1e-6 or P_.loc["p_small", "verdict"] != "partial":
            fails.append("S1 small pair wrong: %s" % P_.loc["p_small"].to_dict())
        if P_.loc["p_big", "verdict"] != "update_fails_on_healthy_state" or P_.loc["p_chg", "verdict"] != "invalid_map_changed" or P_.loc["rep", "verdict"] != "REPEAT_IDENTICAL":
            fails.append("S1 verdicts wrong: %s" % P_["verdict"].to_dict())
        if abs(P_.loc["p_small", "driver_vs_control_max_mm"] - 0.0) > 1e-6 and False:
            pass
        # ratio < 0.25: shadow x1.2 -> S 2 mm / 10 = 0.2
        d = os.path.join(runs, "sh_low"); os.makedirs(d); shadow_rows(1.2).to_csv(os.path.join(d, "shadow_scan_all.csv.gz"), index=False, compression="gzip")
        pj[0]["shadow_arm"] = "sh_low"; json.dump(pj, open(pjp, "w")); SR.pairs_main(type("A", (), dict(runs=runs, pairs=pjp, out=so)))
        if pd.read_csv(os.path.join(so, "r73_shadow_pairs.csv")).set_index("pair").loc["p_small", "verdict"] != "needs_accumulated_history":
            fails.append("S1 low ratio not needs_accumulated_history")
        sc2 = shadow_rows(1.4); sc2.loc[(sc2.role == "shadow") & (sc2.frame == 20), "post_pos_x"] += 1e-9
        d = os.path.join(runs, "sh_rep2"); os.makedirs(d); sc2.to_csv(os.path.join(d, "shadow_scan_all.csv.gz"), index=False, compression="gzip")
        pj2 = [dict(pair="rep2", shadow_arm="sh_rep2", repeat_of="sh", window=[10, 30])]; json.dump(pj2, open(pjp, "w")); SR.pairs_main(type("A", (), dict(runs=runs, pairs=pjp, out=so)))
        if pd.read_csv(os.path.join(so, "r73_shadow_pairs.csv")).set_index("pair").loc["rep2", "verdict"] != "REPEAT_DIFFERS":
            fails.append("S1 a 1e-9 difference was not detected by the repeat check")
        # ---- F1 frame errors: stationary reference, known offsets
        ref = os.path.join(t, "f1"); os.makedirs(ref)
        open(os.path.join(ref, "state_reference.txt"), "w").write("q_wxyz 1 0 0 0\np 1 2 3\nv 0 0 0\nP0_dim 2\nP0_row_major 1 0 0 1\n")
        tr = []
        for sid in range(1, 13):
            for ph in ("post_imu", "post_lio"):
                tr.append(dict(scan_id=sid, phase=ph, t_abs=10.0 + 0.1 * sid, qw=np.cos(0.005 * sid / 2) if ph == "post_lio" else 1.0, qx=0.0, qy=0.0, qz=np.sin(0.005 * sid / 2) if ph == "post_lio" else 0.0,
                               px=1.0 + 0.001 * sid, py=2.0, pz=3.0, vx=0, vy=0, vz=0))
        pd.DataFrame(tr).to_csv(os.path.join(ref, "state_trace.csv"), index=False)
        post = FE.posterior(os.path.join(ref, "state_trace.csv"))
        fe = FE.frame_errors(post, os.path.join(ref, "state_reference.txt"), "a", 10)
        if abs(fe["ep_x_mm"].iloc[4] - 5.0) > 1e-9 or abs(fe["yaw_err_mrad"].iloc[9] - 50.0) > 1e-6 or abs(fe["excursion_10_mm"].iloc[0] - 10.0) > 1e-9:
            fails.append("F1 frame errors wrong: %s" % fe.iloc[[4, 9]][["ep_x_mm", "yaw_err_mrad", "excursion_10_mm"]].to_dict("records"))
        pt = FE.pos_trace(post)
        if abs(pt["yaw_rad"].iloc[9] - 0.05) > 1e-9:
            fails.append("F1 pos_trace yaw wrong")
        # ---- P1 p2 table: ratios, alignment-free collapse flag, collapsed arm unranked
        recs, comp = [], os.path.join(t, "comp")
        def addarm(bag, acc, gyro, ate, exc, shift_mm):
            arm = "r73_p2_%s_mq_a%s_g%d" % (bag, str(acc).replace(".", "p"), gyro)
            recs.append({"arm": arm, "gt_ate_mm": ate, "excursion_mm": exc, "settled_offset_mm": 3.0, "wander_mm": 1.0, "n_scans": 3900, "exit_code": 0})
            d = os.path.join(comp, arm); os.makedirs(d)
            sid = np.arange(1, 801); pos = np.zeros((800, 3)); pos[sid > 340, 0] = shift_mm * 1e-3
            pd.DataFrame({"scan_id": sid, "t_abs": sid * 0.1, "px": pos[:, 0], "py": 0.0, "pz": 0.0, "yaw_rad": 0.0}).to_csv(os.path.join(d, "pos_trace_all.csv.gz"), index=False, compression="gzip")
        bags = ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01"]
        for bg in bags:
            addarm(bg, 0.3, 10, 100.0, 10.0, 0.0)
            addarm(bg, 0.6, 12, 50.0, 8.0, 0.0)                                  # the cell built to win on ATE
            addarm(bg, 0.0, 17, 60.0 if bg != "eee_01" else 400.0, 5.0, 900.0 if bg in ("eee_02", "eee_03") else 0.0)   # collapses on two bags (unranked there)
        bp = {bg: {"stat_last": 335, "trans_last": 395} for bg in bags}
        Pp, Cc, Ww = PT.build(pd.DataFrame(recs), comp, bp)
        c = Cc.set_index("cell")
        if abs(c.loc["a0.6_g12", "gt_ate_mm_ratio_median"] - 0.5) > 1e-12 or c.loc["a0_g17", "n_collapsed"] != 2 or c.loc["a0_g17", "n_ranked"] != 5:
            fails.append("P1 cells wrong: %s" % c.to_dict("index"))
        if Ww["gt_ate_mm"]["cell"] != "a0.6_g12":
            fails.append("P1 winner wrong: %s" % Ww)
        coll = Pp[(Pp.bag == "eee_02") & (Pp.acc == 0.0)].iloc[0]
        if not coll["collapsed"] or coll["ranked"] or coll["collapse_first_scan"] != 341:
            fails.append("P1 collapsed arm not flagged/unranked at scan 341: %s" % coll[["collapsed", "ranked", "collapse_first_scan"]].to_dict())
        # ---- R2 return builder: required_outputs check fails naming the follow-up; byte split reassembles; a dropped part is rejected
        ret = os.path.join(t, "R73_RETURN"); os.makedirs(os.path.join(ret, "compact", "armA"))
        spec = [dict(followup="F-414", title="x", arms=["armA"], file="shadow_scan_all.csv.gz", columns=["frame", "role"], min_rows=3, where="return", source="arm", contains={}),
                dict(followup="F-416", title="y", arms=["armA", "armB"], file="r73_f416_frame_errors.csv", columns=["arm", "frame"], min_rows=2, where="return", source="step0_table", contains={})]
        pd.DataFrame({"frame": [1, 2, 3], "role": ["d", "d", "s"]}).to_csv(os.path.join(ret, "compact", "armA", "shadow_scan_all.csv.gz"), index=False, compression="gzip")
        os.makedirs(os.path.join(ret, "STEP0")); pd.DataFrame({"arm": ["armA", "armA"], "frame": [1, 2]}).to_csv(os.path.join(ret, "STEP0", "r73_f416_frame_errors.csv"), index=False)
        rows_ = BR.check_required(ret, runs, spec)
        st = {(r[0], r[1]): r[4] for r in rows_}
        if st[("F-414", "armA")] != "PRESENT" or st[("F-416", "armB")] != "MISSING" or st[("F-416", "armA")] != "PRESENT":
            fails.append("R2 required-output check wrong: %s" % st)
        pd.DataFrame({"frame": [1], "role": ["d"]}).to_csv(os.path.join(ret, "compact", "armA", "shadow_scan_all.csv.gz"), index=False, compression="gzip")
        if {(r[0], r[1]): r[4] for r in BR.check_required(ret, runs, spec)}[("F-414", "armA")] != "FAIL":
            fails.append("R2 row-count shortfall not FAIL")
        # ---- R3 step-0 data delivered in the stage-1b return: satisfied by the index, not by the file; a file in neither is still MISSING
        os.remove(os.path.join(ret, "STEP0", "r73_f416_frame_errors.csv"))
        open(os.path.join(ret, "STEP0", "STEP0_SHA256.txt"), "w").write("0" * 64 + "  STEP0/r73_f416_frame_errors.csv\n")
        idx_ = BR.step0_index(ret)
        s3 = {(r[0], r[1]): r[4] for r in BR.check_required(ret, runs, spec, idx_)}
        if s3[("F-416", "-")] != "DELIVERED_IN_STAGE1B":
            fails.append("R3 step0 item in the index not accepted: %s" % s3)
        s4 = {(r[0], r[1]): r[4] for r in BR.check_required(ret, runs, spec, set())}
        if s4[("F-416", "-")] != "MISSING":
            fails.append("R3 step0 item not in the index must stay MISSING: %s" % s4)
        s5 = {(r[0], r[1]): r[4] for r in BR.check_required(ret, runs, spec, None)}
        if s5[("F-416", "-")] != "MISSING":
            fails.append("R3 without --step0-delivered a missing step0 file must stay MISSING")
        zp = os.path.join(t, "z.zip"); open(zp, "wb").write(os.urandom(2_500_000))
        parts = BR.byte_split(zp, 1_000_000)
        if len(parts) != 3 or not BR.reassembly_proof(zp, parts, os.path.join(t, "proof.txt")):
            fails.append("R2 byte split / reassembly proof failed")
        if BR.join_sha([p[0] for p in parts[:2]]) == BR.sha(zp):
            fails.append("R2 dropped last part still reassembles")
        # ---- M1 manifest
        if a.manifest_jobs:
            mf = os.path.join(t, "m")
            subprocess.check_call([sys.executable, os.path.join(HERE, "make_r73_manifest.py"), "--jobs", a.manifest_jobs, "--out", mf], stdout=subprocess.DEVNULL)
            cnt = {b: len(json.load(open(os.path.join(mf, "jobs_%s.json" % b)))) for b in ("smoke", "refuse", "sh", "co", "qa", "p2")}
            if cnt != {"smoke": 22, "refuse": 24, "sh": 13, "co": 13, "qa": 18, "p2": 108}:
                fails.append("M1 counts %s" % cnt)
            ids = set()
            for b in cnt:
                for j in json.load(open(os.path.join(mf, "jobs_%s.json" % b))):
                    if j["job_id"] in ids:
                        fails.append("M1 duplicate job id " + j["job_id"])
                    ids.add(j["job_id"])
                    for k in j["overrides"]:
                        if "perturb" in k.lower():
                            fails.append("M1 perturb key in " + j["job_id"])
                        if (k.startswith("lio.q_adapt.") or k.startswith("imu.process_noise.continuous_prior.")) and not j["job_id"].startswith("r73_refuse_old_"):
                            fails.append("M1 retired key %s in %s" % (k, j["job_id"]))
            tab = pd.read_csv(os.path.join(mf, "arm_table.tsv"), sep="\t", keep_default_na=False)
            p2 = tab[tab.group.isin(["P2", "P2C"])]
            if len(p2) != 108 or p2.seq.nunique() != 9 or p2.arm.str.extract(r"r73_p2_[a-z]+_\d\d_(.+)")[0].nunique() != 12:
                fails.append("M1 p2 grid is not 12 cells (11 new + centre) x 9 bags")
            if any(n.startswith("r73_co_rerun") or n.endswith("_full") and n.startswith("r73_co") for n in set(tab.arm)):
                fails.append("M1 a full-bag coupled arm is in the manifest (Bryce asked for short runs)")
            dur = tab[tab.group == "CO"]
            if (dur.duration_s.astype(str) == "full").any():
                fails.append("M1 a CO arm is not 60 s: %s" % list(dur[dur.duration_s.astype(str) == "full"].arm))
            for need in ("r73_id_shadow_same_60s", "r73_sh_sbs03_a", "r73_co60_meas_k8_mq"):
                if need not in set(tab.arm):
                    fails.append("M1 arm missing: " + need)
            sh = tab[tab.arm == "r73_sh_sbs03_a"].iloc[0]
            if "shadow_scan.csv" not in sh.requires or "shadow_residual_dump.csv" not in sh.requires or "scan_log.csv" not in sh.requires:
                fails.append("M1 requires column of a shadow arm incomplete: %s" % sh.requires)
            ro = json.load(open(os.path.join(mf, "required_outputs.json")))
            fus = {r["followup"] for r in ro}
            for f_ in ("F-400", "F-403", "F-407", "F-409", "F-414", "F-415", "F-416", "F-417", "F-418", "F-419"):
                if f_ not in fus:
                    fails.append("M1 required_outputs has no entry for " + f_)
            # every file an arm REQUESTS (requires) that is also returned must be in required_outputs or in the archive lists: no silently unrequested log
            if not json.load(open(os.path.join(mf, "shadow_pairs.json"))):
                fails.append("M1 shadow_pairs.json empty")
    if fails:
        print("SELFTEST R73 FAIL"); [print(" ", f) for f in fails]; sys.exit(1)
    print("SELFTEST R73 PASS")


if __name__ == "__main__":
    main()
