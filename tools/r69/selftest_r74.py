#!/usr/bin/env python3
"""selftest_r74.py -- proves the R74 Python tools can FAIL (rule 2.4). Synthetic data only; prints SELFTEST R74 PASS or exits 1.

    python3 tools/r69/selftest_r74.py [--manifest-jobs r69_inputs/r66_jobs.json] [--tree <repo root>]

T1  r74_time: the angle of R_a^T R_b (atan2) is exact for a known rotation, bounded by pi, and two rotation vectors that differ by 2 pi give ~0 (F-423); fmt_time_columns writes exactly
    6 decimals and keeps two times 0.04 s apart distinct at t_abs ~ 1.6e9 (F-422); add_t_rel needs bag_start_abs and leaves an existing t_rel_s alone; the R73 formula (norm of a rotation-vector
    difference) is shown to give 2 pi on the same input (the control that the new formula differs from the old).
T2  r74_frame_errors: pos_trace_all columns scan_id,t_abs,t_rel_s,px,py,pz,yaw_rad; t_rel_s = t_abs - bag_start_abs from state_reference.txt; a trace without the bag_start line still works when it
    carries t_rel_s and raises when it carries neither.
T3  r74_shadow_reduce arm: step_rot_diff_mrad is ~0 for post rotation vectors that differ by 2 pi and equals the built angle otherwise; the t_rel_s column is carried.
T4  r74_switch_reduce: part2 on a switch arm that follows control A before the onset and control B from it: dA_before max 0, dB_from_onset max 0, dA_from_onset > 0; part1 summary
    max_before / max_from_onset equal the built step differences.
T5  r74_plane_check: a consistent synthetic snapshot + dump gives violations_total 0; each seeded fault (wrong normal, wrong d, eigenvalue order, wrong edge, wrong ratio column, unjoined node,
    a retired plane, a u1 row carrying the u0 normal) is counted by the check it belongs to.
T6  r74_score_job: step_cap_reduce writes next_converged = 1 / 0 / -1 correctly and recomputes the applied columns for an old log; the f417 opt-in: with f417=0 the residual report is NOT run
    and no *_all residual file is written, with f417=1 it is run; the plane check runs only when plane_check=1.
T7  host script scripts/parallel/run_job.sh: the DONE gate is the exact R73 line (clean exit AND non-empty odometry.txt) and the OR form (`ATE_LINE_RE ... OR`) as code is absent (F-408).
T8  manifest: counts (smoke 3, refuse 8, ef 320, f410 1, sw 4, qa 126; 462 arms, 127 full-bag), every arm has an `asked`, group sizes, no 'perturb' key, unique job ids, every R74 key is
    consumed somewhere in the C++ source (slash form found), required_outputs has every follow-up, f417 flags exact, the three own arms exist.
T9  build_return.check_required: seeded missing / short / absent-column files are named with their follow-up; the F-426 generated files are looked up at the return root.
"""
import argparse, json, os, subprocess, sys, tempfile, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r74_time as RT, r74_frame_errors as FE, r74_shadow_reduce as SR, r74_switch_reduce as SW, r74_plane_check as PC, r74_score_job as SJ, build_return as BR   # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest-jobs"); ap.add_argument("--tree", default=os.path.abspath(os.path.join(HERE, "..", "..")))
    a = ap.parse_args()
    fails, ran = [], []
    with tempfile.TemporaryDirectory() as t:
        # ---------------- T1
        ang = 0.7
        Rz = RT.rodrigues([0, 0, ang])
        if abs(RT.rot_angle(np.eye(3), Rz) - ang) > 1e-12:
            fails.append("T1 rot_angle of a known rotation wrong")
        if abs(RT.rotvec_angle([0.0, 0.0, 3.0], [0.0, 0.0, 3.0 + 2 * np.pi])) > 1e-9:
            fails.append("T1 two rotation vectors 2 pi apart must give ~0: %g" % RT.rotvec_angle([0.0, 0.0, 3.0], [0.0, 0.0, 3.0 + 2 * np.pi]))
        if abs(RT.rotvec_angle([0, 0, 3.0], [0, 0, 3.2]) - 0.2) > 1e-9 and abs(RT.rotvec_angle([0, 0, 3.0], [0, 0, 3.2]) - (2 * np.pi - 0.2)) > 1e-9 and abs(RT.rotvec_angle([0, 0, 3.0], [0, 0, 3.2]) - 3.083) > 1e-3:
            fails.append("T1 rotvec_angle 3.0 vs 3.2 wrong: %g" % RT.rotvec_angle([0, 0, 3.0], [0, 0, 3.2]))
        old = np.linalg.norm(np.array([0, 0, 3.0 + 2 * np.pi]) - np.array([0, 0, 3.0]))
        if abs(old - 2 * np.pi) > 1e-12 or RT.rotvec_angle([0, 0, 3.0], [0, 0, 3.0 + 2 * np.pi]) > 1e-6:
            fails.append("T1 the control (R73 norm formula) must give 2 pi where the angle gives ~0")
        for x in np.linspace(0, 20, 41):
            if RT.rotvec_angle([0, 0, 0.0], [0.3 * x, 0.1 * x, 0.2 * x]) > np.pi + 1e-12:
                fails.append("T1 angle above pi"); break
        df = pd.DataFrame({"t_abs": [1.609056017e9, 1.609056017e9 + 0.04], "v": [1.0, 2.0]})
        f6 = RT.fmt_time_columns(RT.add_t_rel(df, 1.609056000e9))
        if list(f6["t_abs"]) != ["1609056017.000000", "1609056017.040000"] or f6["t_rel_s"].iloc[1] != "17.040000":
            fails.append("T1 time columns not 6 decimals: %s" % f6.to_dict("list"))
        try:
            RT.add_t_rel(df, None); fails.append("T1 add_t_rel without bag_start must raise")
        except ValueError:
            pass
        d2 = df.assign(t_rel_s=[9.0, 9.0])
        if list(RT.add_t_rel(d2, 1.0)["t_rel_s"]) != [9.0, 9.0]:
            fails.append("T1 an existing t_rel_s must be left alone")
        # ---------------- T2
        ref = os.path.join(t, "f1"); os.makedirs(ref)
        bag0 = 1609056000.123456
        open(os.path.join(ref, "state_reference.txt"), "w").write("q_wxyz 1 0 0 0\np 1 2 3\nv 0 0 0\nbag_start_abs %.17g\nP0_dim 2\nP0_row_major 1 0 0 1\n" % bag0)
        tr = []
        for sid in range(1, 13):
            for ph in ("post_imu", "post_lio"):
                tr.append(dict(scan_id=sid, phase=ph, t_abs=bag0 + 0.1 * sid, qw=np.cos(0.005 * sid / 2) if ph == "post_lio" else 1.0, qx=0.0, qy=0.0, qz=np.sin(0.005 * sid / 2) if ph == "post_lio" else 0.0,
                               px=1.0 + 0.001 * sid, py=2.0, pz=3.0, vx=0, vy=0, vz=0))
        pd.DataFrame(tr).to_csv(os.path.join(ref, "state_trace.csv"), index=False, float_format="%.17g")
        subprocess.check_call([sys.executable, os.path.join(HERE, "r74_frame_errors.py"), "--dir", ref, "--out", ref, "--no-errors"], stdout=subprocess.DEVNULL)
        pt = pd.read_csv(os.path.join(ref, "pos_trace_all.csv.gz"), dtype={"t_abs": str, "t_rel_s": str})
        if list(pt.columns) != ["scan_id", "t_abs", "t_rel_s", "px", "py", "pz", "yaw_rad"]:
            fails.append("T2 pos_trace columns %s" % list(pt.columns))
        else:
            if abs(float(pt["t_rel_s"].iloc[2]) - 0.3) > 1.5e-6 or len(pt["t_abs"].iloc[0].split(".")[1]) != 6:
                fails.append("T2 t_rel_s / t_abs format wrong: %s" % pt.iloc[2].to_dict())
            if pt["t_rel_s"].nunique() != len(pt):
                fails.append("T2 t_rel_s not distinct per scan")
        # ---------------- T3
        def shadow_rows(rv_shift):
            rows = []
            for f in range(10, 16):
                for role in ("driver", "shadow"):
                    d = {"frame": f, "t_abs": 100.0 + 0.1 * f, "t_rel_s": 0.1 * f, "role": role}
                    for c in ("q_acc_x", "q_acc_y", "q_acc_z", "q_gyr_x", "q_gyr_y", "q_gyr_z", "q_samples", "acc_clamped_axes", "gyr_clamped_axes", "start_rv_x", "start_rv_y", "start_rv_z", "prop_rv_x", "prop_rv_y", "prop_rv_z"):
                        d[c] = 0.0
                    d.update(post_rv_x=0.0, post_rv_y=0.0, post_rv_z=3.0 + (rv_shift if role == "shadow" else 0.0))
                    d.update(start_pos_x=0.0, start_pos_y=0.0, start_pos_z=0.0, prop_pos_x=0.01, prop_pos_y=0.0, prop_pos_z=0.0, post_pos_x=0.01, post_pos_y=0.0, post_pos_z=0.0)
                    d.update(prior_trace_rot=1.0, prior_trace_pos=1.0, prior_trace_vel=1.0, post_trace_rot=1.0, post_trace_pos=1.0, post_trace_vel=1.0, n_iter=3, n_resid_iter0=100, reduced_chi2_iter0=1.0)
                    d["map_digest_before_pass"] = "aa" if role == "shadow" else ""; d["map_digest_after_pass"] = "aa" if role == "shadow" else ""
                    rows.append(d)
            return pd.DataFrame(rows)
        T3a, T3b = SR.steps(shadow_rows(2 * np.pi)), SR.steps(shadow_rows(0.05))
        if T3a["step_rot_diff_mrad"].max() > 1e-3 or abs(T3b["step_rot_diff_mrad"].iloc[0] - 50.0) > 1e-6 or "t_rel_s" not in T3a.columns:
            fails.append("T3 rotation difference wrong: wrap %g built %g (want ~0 and 50) t_rel_s %s" % (T3a["step_rot_diff_mrad"].max(), T3b["step_rot_diff_mrad"].iloc[0], "t_rel_s" in T3a.columns))
        # ---------------- T4
        fr = np.arange(320, 431)
        def pos(stepmm):
            return np.cumsum(stepmm) * 1e-3
        stepA, stepB = np.full(len(fr), 10.0), np.full(len(fr), 14.0)
        onset = 340
        stepS = np.where(fr < onset, stepA, stepB)
        def mk(arm, st):
            d = os.path.join(t, "runs", arm); os.makedirs(d, exist_ok=True)
            pd.DataFrame({"scan_id": fr, "t_abs": fr * 0.1, "t_rel_s": fr * 0.1, "px": pos(st), "py": 0.0, "pz": 0.0, "yaw_rad": 0.0}).to_csv(os.path.join(d, "pos_trace_all.csv.gz"), index=False, compression="gzip")
        ctl = os.path.join(t, "ctl")
        for nm, st in (("ctlA", stepA), ("ctlB", stepB)):
            d = os.path.join(ctl, nm); os.makedirs(d); pd.DataFrame({"scan_id": fr, "t_abs": fr * 0.1, "px": pos(st), "py": 0.0, "pz": 0.0, "yaw_rad": 0.0}).to_csv(os.path.join(d, "pos_trace_all.csv.gz"), index=False, compression="gzip")
        mk("swarm", stepS)
        sw_pairs = os.path.join(t, "sw.json"); json.dump([dict(pair="a", kind="switch", switch_arm="swarm", control_before="ctlA", control_after="ctlB", shadow_arm_r73="shA", window=[330, 420], onset=onset)], open(sw_pairs, "w"))
        so = os.path.join(t, "so"); os.makedirs(so)
        SW.part2(type("A", (), dict(runs=os.path.join(t, "runs"), controls=ctl, pairs=sw_pairs, out=so)))
        p2 = pd.read_csv(os.path.join(so, "r74_f426_part2_summary.csv")).iloc[0]
        if p2["dA_before_max_mm"] > 1e-6 or p2["dB_from_onset_max_mm"] > 1e-6 or abs(p2["dA_from_onset_max_mm"] - 4.0) > 1e-6 or abs(p2["dB_before_max_mm"] - 4.0) > 1e-6:
            fails.append("T4 part2 wrong: %s" % p2.to_dict())
        shd = os.path.join(t, "shd", "shA"); os.makedirs(shd)
        rows = []
        for f in range(330, 421):
            for role in ("driver", "shadow"):
                k = (4.0 if (role == "shadow" and f >= onset) else 0.0)
                rows.append(dict(frame=f, role=role, start_pos_x=0.0, start_pos_y=0.0, start_pos_z=0.0, post_pos_x=(10.0 + k) * 1e-3, post_pos_y=0.0, post_pos_z=0.0))
        pd.DataFrame(rows).to_csv(os.path.join(shd, "shadow_scan_all.csv.gz"), index=False, compression="gzip")
        SW.part1(type("A", (), dict(shadow_dirs=os.path.join(t, "shd"), pairs=sw_pairs, out=so)))
        p1 = pd.read_csv(os.path.join(so, "r74_f426_part1_summary.csv")).iloc[0]
        if p1["max_before_mm"] > 1e-9 or abs(p1["max_from_onset_mm"] - 4.0) > 1e-9 or p1["n_before"] != 10:
            fails.append("T4 part1 wrong: %s" % p1.to_dict())
        # a missing control is MISSING, never a number
        json.dump([dict(pair="a", kind="switch", switch_arm="swarm", control_before="nope", control_after="ctlB", shadow_arm_r73="shA", window=[330, 420], onset=onset)], open(sw_pairs, "w"))
        SW.part2(type("A", (), dict(runs=os.path.join(t, "runs"), controls=ctl, pairs=sw_pairs, out=so)))
        if "MISSING" not in str(pd.read_csv(os.path.join(so, "r74_f426_part2_summary.csv")).iloc[0].get("note")):
            fails.append("T4 a missing control must be reported MISSING")
        # ---------------- T5 plane check
        rng = np.random.default_rng(5)
        n_nodes = 12
        nodes = []
        for i in range(n_nodes):
            nrm = rng.normal(size=3); nrm /= np.linalg.norm(nrm)
            tx = np.cross(nrm, [0, 0, 1.0]); tx /= np.linalg.norm(tx); ty = np.cross(nrm, tx)
            lay = i % 3; e0 = rng.uniform(1e-5, 1e-3); e1 = e0 * rng.uniform(5, 50); e2 = e1 * rng.uniform(2, 20); N = int(rng.integers(8, 60))
            nodes.append(dict(node_id=100 + i, is_plane=1, retired=0, point_count=N, eig0=e0, eig1=e1, eig2=e2, plane_threshold=2.5e-3, plane_fit_mode="pca", layer=lay, edge_m=1.0 / 2 ** lay,
                              normal_x=nrm[0], normal_y=nrm[1], normal_z=nrm[2], tangent_y_x=ty[0], tangent_y_y=ty[1], tangent_y_z=ty[2], tangent_x_x=tx[0], tangent_x_y=tx[1], tangent_x_z=tx[2],
                              d=rng.normal(), planarity_value=e0, planarity_pass=1, second_eig_below_threshold=int(e1 < 2.5e-3), ratio_e1_over_e0=e1 / e0, ratio_e2_over_e1=e2 / e1,
                              rel_gap_threshold=1 + 3 * np.sqrt(2 / (N - 1)), center_x=0.0, center_y=0.0, center_z=0.0))
        snap = pd.DataFrame(nodes)
        snap.loc[0, "layer"] = 0; snap.loc[0, "edge_m"] = 1.0
        dump_rows = []
        for k in range(60):
            nd = nodes[k % n_nodes]
            w_ = rng.normal(size=3)
            kind = [0, 0, 0, 2, 5][k % 5]
            nv = {0: [nd["normal_x"], nd["normal_y"], nd["normal_z"]], 2: [nd["tangent_y_x"], nd["tangent_y_y"], nd["tangent_y_z"]], 5: [nd["tangent_x_x"], nd["tangent_x_y"], nd["tangent_x_z"]]}[kind]
            r_ = (np.array([nd["normal_x"], nd["normal_y"], nd["normal_z"]]) @ w_ + nd["d"]) if kind == 0 else 0.123
            dump_rows.append(dict(frame=1, iter=0, node_id=nd["node_id"], hpos_x=nv[0], hpos_y=nv[1], hpos_z=nv[2], wx=w_[0], wy=w_[1], wz=w_[2], r=r_, sigma_squared=1e-3, row_kind=kind))
        dump = pd.DataFrame(dump_rows)
        state = pd.DataFrame([dict(frame=1, iter=0, px=0.1, py=0.2, pz=0.3)])
        o, T_ = PC.check(snap, dump, state)
        if o["violations_total"] != 0 or o["n_planes_matched"] != n_nodes:
            fails.append("T5 consistent data gave violations: %s" % {k: v for k, v in o.items() if k.startswith(("P", "viol"))})
        def u1_with_u0(d):
            m = d.merge(snap[["node_id", "normal_x", "normal_y", "normal_z"]], on="node_id", how="left")
            sel = (d["row_kind"] == 2).to_numpy()
            for c, nc in (("hpos_x", "normal_x"), ("hpos_y", "normal_y"), ("hpos_z", "normal_z")):
                d.loc[sel, c] = m.loc[sel, nc].to_numpy()
        def fault(name, mut_snap=None, mut_dump=None):
            s_, d_ = snap.copy(), dump.copy()
            if mut_snap:
                mut_snap(s_)
            if mut_dump:
                mut_dump(d_)
            oo, _ = PC.check(s_, d_, state)
            return oo
        cases = [("P4 wrong normal", lambda s: s.__setitem__("normal_x", s["normal_x"] + 0.01), None, "P4_normal_mismatch"),
                 ("P5 wrong d", lambda s: s.__setitem__("d", s["d"] + 0.01), None, "P5_r_mismatch"),
                 ("P3 eigen order", lambda s: s.__setitem__("eig0", s["eig1"] * 2), None, "P3_eigen_order_violations"),
                 ("P6 wrong edge", lambda s: s.__setitem__("edge_m", s["edge_m"].where(s.index != 1, 0.6)), None, "P6_edge_mismatch"),
                 ("P7 wrong ratio", lambda s: s.__setitem__("ratio_e1_over_e0", s["ratio_e1_over_e0"] * 1.01), None, "P7_ratio_e1_over_e0_mismatch"),
                 ("P1 unjoined", None, lambda d: d.__setitem__("node_id", d["node_id"].where(d.index != 0, 999)), "P1_unjoined"),
                 ("P1 retired", lambda s: s.__setitem__("retired", 1), None, "P1_not_plane_or_retired"),
                 ("P2 few points", lambda s: s.__setitem__("point_count", 3), None, "P2_too_few_points"),
                 ("P4 u1 row with u0 normal", None, u1_with_u0, "P4_normal_mismatch")]
        for nm, ms, md, key in cases:
            oo = fault(nm, ms, md)
            if not (oo.get(key, 0) > 0) or oo["violations_total"] <= 0:
                fails.append("T5 seeded fault '%s' not counted in %s (violations_total=%s)" % (nm, key, oo["violations_total"]))
        # ---------------- T6 score job pieces
        sd = os.path.join(t, "sc"); os.makedirs(sd)
        pd.DataFrame({"frame": [1, 1, 1, 2, 2], "iter": [0, 1, 2, 0, 1], "rot_raw": [5e-2, 2e-2, 5e-4, 3e-2, 4e-4], "pos_raw": 0.01, "scale": [0.2, 0.5, 1.0, 0.3, 1.0], "bound": [1, 1, 0, 1, 0],
                      "cap_rot": 0.01, "cap_pos": 0.01}).to_csv(os.path.join(sd, "step_cap.csv"), index=False)
        SJ.step_cap_reduce(sd)
        sc = pd.read_csv(os.path.join(sd, "step_cap.csv.gz"))
        if list(sc["next_converged"]) != [0, 1, -1, 1, -1] or abs(sc["rot_applied"].iloc[0] - 1e-2) > 1e-12 or list(sc.columns) != ["frame", "iter", "rot_raw", "pos_raw", "rot_applied", "pos_applied", "scale", "bound", "next_converged"]:
            fails.append("T6 step_cap reduce wrong: %s" % sc.to_dict("list"))
        calls = []
        orig_run = SJ.run
        import r73_residual_report as RR_
        orig_pr = RR_.per_residual
        RR_.per_residual = lambda df: pd.DataFrame({"a": [1]})
        SJ.run = lambda cmd, log: (calls.append(os.path.basename(cmd[1])), 0)[1]
        try:
            ad = os.path.join(t, "arm"); os.makedirs(ad)
            for nm in ("residual_dump.csv", "lio_update_diag.csv", "post_calibration_voxelmap.csv", "state_trace.csv"):
                open(os.path.join(ad, nm), "w").write("a\n")
            calls.clear(); SJ.r74_analyses(ad, HERE, {"f417": "0", "plane_check": "0", "group": "SW"}, 0)
            if any(c in ("r73_residual_report.py", "r74_plane_check.py") for c in calls) or os.path.exists(os.path.join(ad, "residual_dump_all.csv.gz")):
                fails.append("T6 f417=0 / plane_check=0 arm ran a report: %s" % calls)
            calls.clear(); SJ.r74_analyses(ad, HERE, {"f417": "1", "plane_check": "1", "group": "VX"}, 0)
            if "r73_residual_report.py" not in calls or "r74_plane_check.py" not in calls:
                fails.append("T6 f417=1 / plane_check=1 arm did not run its reports: %s" % calls)
            if "r74_frame_errors.py" not in calls:
                fails.append("T6 pos_trace not produced from state_trace.csv: %s" % calls)
        except Exception as e:
            fails.append("T6 r74_analyses raised on fake files: %r" % e)
        finally:
            SJ.run = orig_run; RR_.per_residual = orig_pr
        # ---------------- T7 host script
        rj = os.path.join(a.tree, "scripts", "parallel", "run_job.sh")
        if os.path.exists(rj):
            txt = open(rj).read()
            code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
            gate = '[[ "$_exit_status_text" == "exit_code=0 signal=none" ]] && _clean_exit=1'
            if gate not in code or 'if [[ "$_clean_exit" == "1" && -s "$OUT_DIR/odometry.txt" ]]; then' not in code:
                fails.append("T7 run_job.sh does not carry the R73 DONE gate lines")
            if re.search(r"ATE_LINE_RE[^\n]*\|\|", code) and "_clean_exit" not in code:
                fails.append("T7 run_job.sh DONE gate is the OR form")
            ran.append("T7 run_job.sh gate present")
        else:
            fails.append("T7 scripts/parallel/run_job.sh absent from the tree")
        # ---------------- T8 manifest
        sys.path.insert(0, HERE)
        import make_r74_manifest as M
        arms_ = M.arms({})
        cnt = {}
        for r in arms_:
            cnt[r["batch"]] = cnt.get(r["batch"], 0) + 1
        if cnt != {"smoke": 3, "refuse": 8, "ef": 320, "f410": 1, "sw": 4, "qa": 126} or len(arms_) != 462:
            fails.append("T8 counts %s total %d" % (cnt, len(arms_)))
        ids = [r["id"] for r in arms_]
        if len(set(ids)) != len(ids):
            fails.append("T8 duplicate arm ids")
        if any(not r["asked"] for r in arms_):
            fails.append("T8 an arm has no `asked` text: %s" % [r["id"] for r in arms_ if not r["asked"]][:3])
        grp = pd.Series([r["group"] for r in arms_]).value_counts().to_dict()
        if grp != {"LN": 240, "QA": 108, "VX": 80, "QB": 18, "REF": 8, "SW": 4, "ID": 1, "SM": 1, "DET": 1, "CO": 1}:
            fails.append("T8 group sizes %s" % grp)
        own = [r["id"] for r in arms_ if r["asked"].startswith("OWN")]
        if sorted(own) != ["r74_own_det_v1p0_ml2_n1600_dsoff_pfn3_fq", "r74_own_id_baseline_60s", "r74_own_smoke_v1p0_60s"]:
            fails.append("T8 own arms %s" % own)
        if mf := a.manifest_jobs:
            out = os.path.join(t, "m")
            subprocess.check_call([sys.executable, os.path.join(HERE, "make_r74_manifest.py"), "--jobs", mf, "--out", out], stdout=subprocess.DEVNULL)
            tab = pd.read_csv(os.path.join(out, "arm_table.tsv"), sep="\t", keep_default_na=False)
            proof = tab[tab.group == "PROOF"]
            if len(proof) != 1 or proof.arm.iloc[0] != "r74_00_id_broken_clone" or proof.batch.iloc[0] != "r74_proof":
                fails.append("T8 the listener-proof row is missing or wrong")
            tab = tab[tab.group != "PROOF"]
            if len(tab) != 462 or (tab.duration_s.astype(str) == "full").sum() != 127:
                fails.append("T8 table: %d arms, %d full-bag (want 462, 127)" % (len(tab), (tab.duration_s.astype(str) == "full").sum()))
            for fn, nq in (("listener_spec_smoke.json", [3, 8]), ("listener_spec_smoke_norefuse.json", [3]), ("listener_spec_broken.json", [1])):
                sj = json.load(open(os.path.join(out, fn)))
                if [q["expected_jobs"] for q in sj["queues"]] != nq or any(fn not in " ".join(sg["cmd"]) for sg in sj["stages"]):
                    fails.append("T8 %s: queues %s (want %s) or stage command does not name itself" % (fn, [q["expected_jobs"] for q in sj["queues"]], nq))
            if set(tab[tab.f417.astype(str) == "1"].group) != {"VX", "LN", "SM", "DET"} or (tab[tab.group.isin(["SW", "QA", "QB", "CO", "ID", "REF"])].f417.astype(str) == "1").any():
                fails.append("T8 f417 flags wrong")
            if set(tab[tab.plane_check.astype(str) == "1"].group) != {"VX", "SM", "DET"}:
                fails.append("T8 plane_check flags wrong")
            ef = tab[tab.group.isin(["VX", "LN", "DET", "SM"]) & (tab.tier == "EF")]
            if (ef.seq != "eee_01").any():
                fails.append("T8 an EF arm is not on eee_01")
            if not ef[ef.arm.str.contains("n3850")].duration_s.astype(int).eq(28).all():
                fails.append("T8 n3850 duration is not 28 s by the R72 formula")
            for b in ("smoke", "refuse", "ef", "f410", "sw", "qa"):
                for j in json.load(open(os.path.join(out, "jobs_%s.json" % b))):
                    for k in j["overrides"]:
                        if "perturb" in k.lower():
                            fails.append("T8 perturb key in " + j["job_id"])
            ro = json.load(open(os.path.join(out, "required_outputs.json")))
            for f_ in ("F-407", "F-409", "F-410", "F-418", "F-419", "F-425", "F-426", "F-428", "F-430"):
                if f_ not in {r["followup"] for r in ro}:
                    fails.append("T8 required_outputs has no entry for " + f_)
            sp = json.load(open(os.path.join(out, "switch_pairs.json")))
            if [p["pair"] for p in sp] != ["a", "b", "d", "e"] or [p["onset"] for p in sp] != [340, 340, 481, 481]:
                fails.append("T8 switch pairs wrong")
            # every R74 key is consumed in the C++ (slash form present in src)
            src = ""
            for dp, _, fn in os.walk(os.path.join(a.tree, "src")):
                for f in fn:
                    if f.endswith((".cpp", ".h")):
                        src += open(os.path.join(dp, f), errors="replace").read()
            for dp, _, fn in os.walk(os.path.join(a.tree, "include")):
                for f in fn:
                    if f.endswith(".h"):
                        src += open(os.path.join(dp, f), errors="replace").read()
            keys = set()
            for b in ("ef", "sw", "f410", "refuse"):
                for j in json.load(open(os.path.join(out, "jobs_%s.json" % b))):
                    keys |= {k for k in j["overrides"] if k.startswith(("voxel_map.plane.line_residual", "imu.process_noise.switch", "voxel_map.map.snapshot", "voxel_map.map.voxel_size", "voxel_map.map.max_layer",
                                                                         "lio.residual_dump_frames", "lio.first_frame_dump_frames", "lio.update_diag_en", "estimator.coupled.update_log_en"))}
            for k in sorted(keys):
                if k.replace(".", "/") not in src:
                    fails.append("T8 override key not found in the C++ source (slash form): " + k)
        # ---------------- T9 required-output check
        ret = os.path.join(t, "R74_RETURN"); os.makedirs(os.path.join(ret, "compact", "armA"))
        spec = [dict(followup="F-428", title="x", arms=["armA"], file="residual_dump_all.csv.gz", columns=["frame", "row_kind"], min_rows=3, where="return", source="arm", contains={}),
                dict(followup="F-426", title="y", arms=[], file="r74_f426_part2_summary.csv", columns=["pair", "onset"], min_rows=1, where="return", source="generated", contains={}),
                dict(followup="F-430", title="z", arms=["armB"], file="pos_trace_all.csv.gz", columns=["scan_id", "t_rel_s"], min_rows=1, where="return", source="arm", contains={})]
        pd.DataFrame({"frame": [1, 1, 1], "row_kind": [0, 1, 2]}).to_csv(os.path.join(ret, "compact", "armA", "residual_dump_all.csv.gz"), index=False, compression="gzip")
        pd.DataFrame({"pair": ["a"], "onset": [340]}).to_csv(os.path.join(ret, "r74_f426_part2_summary.csv"), index=False)
        st = {(r[0], r[1]): r[4] for r in BR.check_required(ret, t, spec)}
        if st[("F-428", "armA")] != "PRESENT" or st[("F-426", "-")] != "PRESENT" or st[("F-430", "armB")] != "MISSING":
            fails.append("T9 required-output check wrong: %s" % st)
        pd.DataFrame({"frame": [1, 1, 1], "kind": [0, 1, 2]}).to_csv(os.path.join(ret, "compact", "armA", "residual_dump_all.csv.gz"), index=False, compression="gzip")
        if {(r[0], r[1]): r[4] for r in BR.check_required(ret, t, spec)}[("F-428", "armA")] != "FAIL":
            fails.append("T9 absent column must FAIL")
    if fails:
        print("SELFTEST R74 FAIL"); [print(" ", f) for f in fails]; sys.exit(1)
    print("SELFTEST R74 PASS" + ("  (%s)" % "; ".join(ran) if ran else ""))


if __name__ == "__main__":
    main()
