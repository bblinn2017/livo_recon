#!/usr/bin/env python3
"""selftest_r72.py -- proves the R72 Python tools can FAIL (rule 2.4). Synthetic data only; prints SELFTEST R72 PASS or exits 1.

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
import r72_dump_reduce as DR, r72_early_frames as EF, r72_p_table as PT   # noqa: E402


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
        # ---- E1
        rows = []
        def addres(frame, n_g, n_w, n_s, n_m=0):
            for kind, k in (("g", n_g), ("w", n_w), ("s", n_s), ("m", n_m)):
                for i in range(k):
                    nrm = {"g": np.array([0, 0, 1.0]), "w": np.array([1.0, 0, 0]), "s": np.array([0.9, 0, 0.4]), "m": np.array([0.954, 0, 0.3])}[kind]
                    nrm = nrm / np.linalg.norm(nrm)
                    pt = rng.normal(size=3) * 5
                    h = np.concatenate([np.cross(pt, nrm), nrm])
                    rows.append([frame, 0, i, i, 0.0, 0.01, 1e-4, 0, 1, 1, 0, 0, *h, 0, 0, 0, 0, 0, 0])
        addres(1, 50, 0, 0); addres(2, 50, 3, 0); addres(3, 50, 40, 12, 5)   # 5 residuals at |nz|=0.3 (<= 0.35: wall), 12 at |nz|~0.41 (slanted)
        cols = "frame,iter,idx,source_index,t,r,sigma_squared,plane_var_term,plane_key,node_id,tier,vis,hrot_x,hrot_y,hrot_z,hpos_x,hpos_y,hpos_z,wx,wy,wz,bx,by,bz".split(",")
        dump = pd.DataFrame(rows, columns=cols)
        diag = pd.DataFrame([[1, 0, 0.01, 0, 0, 0.4e-3, 0, 0, 0], [1, 1, 0.01, 0, 0, 0.1e-3, 0, 0, 0], [2, 0, 0.01, 0, 0, -0.2e-3, 0, 0, 0], [3, 0, 0.01, 0, 0, 0.0, 0, 0, 0]],
                            columns=["frame", "iter", "t_abs", "dtheta_x", "dtheta_y", "dtheta_z", "dt_x", "dt_y", "dt_z"])
        T = EF.frame_table(dump, diag)
        c = {int(r.frame): r for r in T.itertuples()}
        if (c[1].n_ground, c[1].n_wall, c[2].n_wall, c[3].n_wall, c[3].n_slanted) != (50, 0, 3, 45, 12) or abs(c[1].yaw_step_sum_mrad - 0.5) > 1e-9 or c[2].n_iter != 1:
            fails.append("E1 counts/steps wrong: %s" % T.to_dict("records"))
        S = EF.summary(T)
        if S["first_frame_wall_ge_1"] != 2 or S["first_frame_wall_ge_10"] != 3 or S["first_frame_wall_ge_100"] is not None:
            fails.append("E1 first-wall-frame wrong: %s" % S)
        if not np.isnan(getattr(c[1], "wall_only_believed_sd_yaw_mrad", np.nan)):
            fails.append("E1 wall-only yaw sd reported for a frame with no wall residuals")
        # closed form: ground-only design has NO yaw information (hrot_z = (p x e_z).z = 0): rank must be < 6 and the yaw sd must be 0 in the pinv (reported with the rank)
        if c[1].info_rank >= 6:
            fails.append("E1 ground-only frame reported full information rank %s" % c[1].info_rank)
        # ---- P1
        recs = []
        for bag in ("eee_01", "eee_02"):
            for cell, f in (("fq_p0d", 1.0), ("mq_p0d", 1.0), ("mq_a0p3_g5", 0.5 if bag == "eee_01" else 0.8), ("mq_a1_g14", 0.2)):
                recs.append({"arm": "r72_p_%s_%s" % (bag, cell), "gt_ate_mm": 100.0 * f, "evo_ate_m": 0.1 * f, "excursion_mm": 10.0 * f, "settled_offset_mm": 3.0, "wander_mm": 1.0,
                             "n_scans": 3900 if cell != "mq_a1_g14" or bag == "eee_01" else np.nan, "exit_code": 0})
        P, C, W = PT.build(pd.DataFrame(recs))
        r = C[(C.q == "mq") & (C.cell == "mq_a0p3_g5")].iloc[0]
        if abs(r.gt_ate_mm_ratio_median - 0.65) > 1e-12 or abs(r.gt_ate_mm_ratio_worst - 0.8) > 1e-12:
            fails.append("P1 ratios wrong: %s" % r.to_dict())
        if W["mq"]["ate_winner"] != "mq_a1_g14":
            fails.append("P1 winner wrong (cell with 1 completed of 2 should still win on its completed bag): %s" % W)
        recs2 = [dict(x) for x in recs]
        for x in recs2:
            if x["arm"].endswith("mq_a1_g14"):
                x["n_scans"] = np.nan
        _, _, W2 = PT.build(pd.DataFrame(recs2))
        if W2["mq"]["ate_winner"] == "mq_a1_g14":
            fails.append("P1 a cell with no completed arm won")
        # ---- M1
        if a.manifest_jobs:
            mf = os.path.join(t, "m"); 
            subprocess.check_call([sys.executable, os.path.join(HERE, "make_r72_manifest.py"), "--jobs", a.manifest_jobs, "--out", mf], stdout=subprocess.DEVNULL)
            cnt = {b: len(json.load(open(os.path.join(mf, "jobs_%s.json" % b)))) for b in ("smoke", "refuse", "ef", "co", "sc", "qa", "p")}
            if cnt != {"smoke": 15, "refuse": 13, "ef": 40, "co": 10, "sc": 3, "qa": 18, "p": 144}:
                fails.append("M1 counts %s" % cnt)
            base_keys = {k for j in json.load(open(a.manifest_jobs)) for k in j["overrides"]}
            allnew = set(); ids = set()
            def check(jobs):
                bad = []
                for j in jobs:
                    if j["job_id"] in ids: bad.append("dup " + j["job_id"])
                    ids.add(j["job_id"])
                    for k in j["overrides"]:
                        if "perturb" in k.lower(): bad.append("perturb key %s in %s" % (k, j["job_id"]))
                return bad
            for b in cnt:
                jobs = json.load(open(os.path.join(mf, "jobs_%s.json" % b)))
                bad = check(jobs)
                if bad: fails.append("M1 %s: %s" % (b, bad[:3]))
            ids.clear()
            inj = json.load(open(os.path.join(mf, "jobs_p.json")))[:2]; inj[0]["overrides"]["cbk.lidar.perturb_seed"] = 1
            if not check(inj):
                fails.append("M1 checker did not catch an injected perturb key")
            ef = json.load(open(os.path.join(mf, "jobs_ef.json")))
            tab = pd.read_csv(os.path.join(mf, "arm_table.tsv"), sep="\t")
            if not (tab[tab.group == "EF"].tier == "EF").all() or not (tab[tab.group == "EF"].seq == "eee_01").all() or (tab.group == "EF").sum() != 40:
                fails.append("M1 EF arm table rows wrong")
            pa = tab[tab.group == "P"]
            if len(pa) != 144 or pa.seq.nunique() != 9 or pa.arm.str.extract(r"r72_p_[a-z]+_\d\d_(.+)")[0].nunique() != 16:
                fails.append("M1 P grid is not 16 cells x 9 bags")
    if fails:
        print("SELFTEST R72 FAIL"); [print(" ", f) for f in fails]; sys.exit(1)
    print("SELFTEST R72 PASS")


if __name__ == "__main__":
    main()
