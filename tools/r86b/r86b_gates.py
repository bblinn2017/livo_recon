#!/usr/bin/env python3
"""r86b_gates.py -- R86B gates (planner's script, selftest on a synthetic folder only; not evidence, rule 1.1). Reads a collected return folder; writes r86b_gates.txt,
r86b_controls.csv, r86b_identity.csv, r86b_dump_crosschecks.csv (checks in r86b_dumpchecks.py), r86b_stream_engagement.csv, r86b_dry_run_stream.csv, r86b_trim_first_frames.csv. Verdicts: PASS, FAIL, MISSING (never a pass), INFO. Facts only; no interpretation.
Usage: python3 tools/r86b/r86b_gates.py --ret-dir R86B_RETURN --manifest r86b_manifest [--out R86B_RETURN]
       python3 tools/r86b/r86b_gates.py --selftest
"""
import argparse, csv, gzip, io, json, os, re, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r86b_dumpchecks as DC

TOL_MM = 0.005
IDENT_TOL_MM = 0.0005   # positions are written with 9 significant digits; R86 and R86A agreed to 0.000 mm
PFN = {"m3r3": (3, 3), "m1r1": (1, 1), "m1r3": (1, 3), "m3r1": (3, 1)}
REFS = {"m3r3": "A", "m1r1": "B"}


def aid(tag):
    return "r86b_rt_sp_eee_01_v0p25_" + tag


def rd_gz(p):
    return gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, errors="replace")


def read_csv_gz(p):
    with rd_gz(p) as f:
        return list(csv.DictReader(f))


def engagement_blocks(path):
    b, cur = {}, None
    for ln in open(path, errors="replace"):
        if ln.startswith("### "):
            cur = ln[4:].strip(); b[cur] = []
        elif cur:
            b[cur].append(ln.rstrip("\n"))
    return b


def run(ret, man, out):
    R = []   # (gate, verdict, text)
    P = lambda g, t: R.append((g, "PASS", t)); F = lambda g, t: R.append((g, "FAIL", t)); M = lambda g, t: R.append((g, "MISSING", t)); I = lambda g, t: R.append((g, "INFO", t))
    arms = {t: aid(t) for t in PFN}
    # ARMS4 + REQUIRED
    req = json.load(open(os.path.join(man, "required_outputs.json")))
    miss = []
    for r in req:
        if r["where"] == "return" and r["source"] == "arm":
            ps = [os.path.join(ret, "compact", a, r["file"]) for a in r["arms"]]
        elif r["where"] == "return":
            ps = [os.path.join(ret, r["file"])] if r["file"] not in ("r86b_gates.txt", "r86b_controls.csv", "r86b_identity.csv", "r86b_dump_crosschecks.csv", "r86b_stream_engagement.csv", "r86b_trim_first_frames.csv", "r86b_dry_run_stream.csv") else []
        else:
            ps = [os.path.join(ret, "archive", a, r["file"]) for a in r["arms"]]
        miss += [p for p in ps if not os.path.isfile(p) or os.path.getsize(p) == 0]
    (F if miss else P)("REQUIRED", "%d required path(s) missing or empty%s" % (len(miss), ": " + "; ".join(os.path.relpath(m, ret) for m in miss[:20]) if miss else ""))
    # EXIT
    bad = []
    for t, a in arms.items():
        p = os.path.join(ret, "archive", a, "exit_status.txt.gz")
        if not os.path.isfile(p): bad.append(a + " (absent)"); continue
        s = rd_gz(p).read().split()
        # exit_status.txt's actual format is "exit_code=<N> signal=<name>" (same bug/fix as
        # tools/r86/r86b_gates.py's EXIT row this round's predecessor had -- confirmed R86 and
        # R86B both: all arms read "exit_code=0 signal=none"), not a bare "0".
        code = s[0].split("=", 1)[1] if s and s[0].startswith("exit_code=") else (s[0] if s else None)
        if code != "0": bad.append("%s (%s)" % (a, s[:1]))
    (F if bad else P)("EXIT", "exit status 0 for all four arms" if not bad else "non-zero or absent: " + ", ".join(bad))
    # STREAM engagement
    eng = os.path.join(ret, "engagement_all.txt")
    rows = []
    if not os.path.isfile(eng):
        M("STREAM", "engagement_all.txt absent")
    else:
        blocks = engagement_blocks(eng)
        ok = True; msgs = []
        for t, a in arms.items():
            txt = "\n".join(blocks.get(a, []))
            m = re.search(r"lio/iekf_point_stream=(\w+) dry_run_point_filter_num=(\d+) frames=(\d+) frames_second=(\d+) frames_fallback=(\d+) mean_points_map=([\d.eE+-]+) mean_points_iekf=([\d.eE+-]+)", txt)
            mp, rp = PFN[t]
            if not txt:
                ok = False; msgs.append(a + ": no engagement block"); continue
            if mp == rp:
                if m: ok = False; msgs.append(a + ": stream line present although stream is primary")
                rows.append(dict(arm=a, stream="primary", frames="", frames_second="", frames_fallback="", mean_points_map="", mean_points_iekf="")); continue
            if not m: ok = False; msgs.append(a + ": no iekf_point_stream engagement line"); continue
            st, dn, fr, fs, fb, pm, pi = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), float(m.group(6)), float(m.group(7))
            rows.append(dict(arm=a, stream=st, frames=fr, frames_second=fs, frames_fallback=fb, mean_points_map=pm, mean_points_iekf=pi))
            if st != "second" or dn != rp or fr <= 0 or fs != fr or fb != 0 or "<-- INERT" in txt or "<-- PARTLY" in txt:   # the per-flag markers; the bare words are in the footer of every engagement block
                ok = False; msgs.append("%s: stream=%s dry_run=%d frames=%d second=%d fallback=%d" % (a, st, dn, fr, fs, fb))
            elif (mp < rp and not pi < pm) or (mp > rp and not pi > pm):
                ok = False; msgs.append("%s: direction wrong (map %.1f, iekf %.1f points)" % (a, pm, pi))
        (P if ok else F)("STREAM", "second stream used in every frame of m1r3 and m3r1, fallbacks 0, direction as designed; primary arms have no stream line" if ok else "; ".join(msgs))
    with open(os.path.join(out, "r86b_stream_engagement.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "stream", "frames", "frames_second", "frames_fallback", "mean_points_map", "mean_points_iekf"]); w.writeheader(); w.writerows(rows)
    # CONTROLS against the shipped R85 cells
    crow = []
    for t, c in REFS.items():
        a = arms[t]
        pn = os.path.join(ret, "compact", a, "pos_trace_all.csv.gz"); pr = os.path.join(man, "ref_r85_%s_pos_trace_all.csv.gz" % c)
        sn = os.path.join(ret, "compact", a, "short_scan_all.csv.gz"); sr = os.path.join(man, "ref_r85_%s_short_scan_all.csv.gz" % c)
        if not (os.path.isfile(pn) and os.path.isfile(sn)):
            M("CONTROL_" + t, a + ": compact files absent"); continue
        N = {int(float(r["scan_id"])): r for r in read_csv_gz(pn)}; Rf = {int(float(r["scan_id"])): r for r in read_csv_gz(pr)}
        com = sorted(set(N) & set(Rf)); mx = 0.0
        for s in com:
            d = max(abs(float(N[s][k]) - float(Rf[s][k])) for k in ("px", "py", "pz")) * 1000.0
            mx = max(mx, d)
        SN = {int(float(r["scan"])): r for r in read_csv_gz(sn)}; SR = {int(float(r["scan"])): r for r in read_csv_gz(sr)}
        cs = sorted(set(SN) & set(SR)); ne = sum(1 for s in cs if any(SN[s][k] != SR[s][k] for k in ("fs_n_residuals", "fs_n_voxels", "fs_n_voxels_is_plane")))
        crow.append(dict(arm=a, ref_arm="r85_rt_sht_eee_01_v0p25_" + c, n_common_scans=len(com), pos_max_abs_diff_mm=round(mx, 6), n_scans_count_columns_differ=ne, n_common_short_rows=len(cs)))
        ok = len(com) >= 500 and mx <= TOL_MM and ne == 0 and len(cs) >= 500
        (P if ok else F)("CONTROL_" + t, "%s vs R85 cell %s: %d common scans, max position difference %.6f mm (tolerance %.3f), residual / voxel counts differ in %d of %d scans" % (a, c, len(com), mx, TOL_MM, ne, len(cs)))
    with open(os.path.join(out, "r86b_controls.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "ref_arm", "n_common_scans", "pos_max_abs_diff_mm", "n_scans_count_columns_differ", "n_common_short_rows"]); w.writeheader(); w.writerows(crow)
    # IDENT_R86A: every R86B dump is write-only, so all four arms must reproduce the R86A traces (written to r86b_identity.csv)
    irow = []
    for t in PFN:
        a = arms[t]
        pn = os.path.join(ret, "compact", a, "pos_trace_all.csv.gz"); pr = os.path.join(man, "ref_r86a_%s_pos_trace_all.csv.gz" % t)
        sn = os.path.join(ret, "compact", a, "short_scan_all.csv.gz"); sr = os.path.join(man, "ref_r86a_%s_short_scan_all.csv.gz" % t)
        if not (os.path.isfile(pn) and os.path.isfile(sn)):
            M("IDENT_R86A_" + t, a + ": compact files absent"); continue
        N = {int(float(r["scan_id"])): r for r in read_csv_gz(pn)}; Rf = {int(float(r["scan_id"])): r for r in read_csv_gz(pr)}
        com = sorted(set(N) & set(Rf)); mx = 0.0
        for sc in com:
            mx = max(mx, max(abs(float(N[sc][k]) - float(Rf[sc][k])) for k in ("px", "py", "pz")) * 1000.0)
        SN = {int(float(r["scan"])): r for r in read_csv_gz(sn)}; SR = {int(float(r["scan"])): r for r in read_csv_gz(sr)}
        cs = sorted(set(SN) & set(SR)); ne = sum(1 for sc in cs if any(SN[sc][k] != SR[sc][k] for k in ("fs_n_residuals", "fs_n_voxels", "fs_n_voxels_is_plane")))
        irow.append(dict(arm=a, ref="R86A " + t, n_common_scans=len(com), pos_max_abs_diff_mm=round(mx, 6), n_scans_count_columns_differ=ne, n_common_short_rows=len(cs)))
        ok = len(com) >= 500 and mx <= IDENT_TOL_MM and ne == 0 and len(cs) >= 500
        (P if ok else F)("IDENT_R86A_" + t, "%s vs R86A: %d common scans, max position difference %.6f mm (tolerance %.4f), residual / voxel counts differ in %d of %d scans" % (a, len(com), mx, IDENT_TOL_MM, ne, len(cs)))
    with open(os.path.join(out, "r86b_identity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "ref", "n_common_scans", "pos_max_abs_diff_mm", "n_scans_count_columns_differ", "n_common_short_rows"]); w.writeheader(); w.writerows(irow)
    # the dump checks (PREIMU, CHAIN, ITERS, COVBUDGET, DIGEST, MAPLINK, MAPSAME, SNAPF2); each compares two independent dumps of one quantity, or a dump with R86A
    verd, xrows = DC.run_checks(ret, man, arms)
    for g, v, t in verd:
        R.append((g, v, t))
    with open(os.path.join(out, "r86b_dump_crosschecks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "arm", "quantity", "value_a", "value_b", "abs_diff", "tol", "verdict"]); w.writeheader(); w.writerows(xrows)
    # DUMPS present with content (rows only; the analysis is the planner's)
    dm = []
    for a in arms.values():
        for fn in ("residual_dump.csv.gz", "post_calibration_voxelmap.csv.gz", "voxelmap_snapshot_f2.csv.gz", "pose_control_first_frame_solve_matrices.txt.gz",
                   "pose_control_first_frame.csv.gz", "pose_control_first_frame_state_chain.csv.gz", "pose_control_first_frame_covariance_budget.csv.gz"):
            p = os.path.join(ret, "archive", a, fn)
            if not os.path.isfile(p): dm.append(a + "/" + fn + " absent"); continue
            with rd_gz(p) as f:
                n = sum(1 for _ in f) - 1
            if n < 1: dm.append("%s/%s has %d data rows" % (a, fn, n))
        p = os.path.join(ret, "archive", a, "residual_dump.csv.gz")
        if os.path.isfile(p):
            hd = rd_gz(p).readline()
            if "voxel_point_count" not in hd or "voxel_distinct_frames" not in hd: dm.append(a + ": residual_dump header lacks the voxel columns")
    (F if dm else P)("DUMPS", "residual dump (with the voxel columns), post-calibration and f2 snapshots, frame-1 solve matrices, state chain and covariance budget hold data rows in all four arms" if not dm else "; ".join(dm))
    # SMALLSCANS: the F-532 failure signature. No scan of any arm may have a residual set under half the arm's median (R86: 40 of 595 scans of the split runs did).
    sm = []; ss = {}
    for t, a in arms.items():
        p = os.path.join(ret, "compact", a, "short_scan_all.csv.gz")
        if not os.path.isfile(p): sm.append(a + ": short_scan_all absent"); continue
        rr = [float(r["fs_n_residuals"]) for r in read_csv_gz(p) if r.get("fs_n_residuals") not in (None, "", "nan")]
        if not rr: sm.append(a + ": no residual counts"); continue
        srt = sorted(rr); med = srt[len(srt) // 2]
        low = sum(1 for x in rr if x < 0.5 * med); ss[a] = (len(rr), med, low)
        if low > 0: sm.append("%s: %d of %d scans under half the median %.0f" % (a, low, len(rr), med))
    (F if sm else P)("SMALLSCANS", "no scan under half the arm's median residual count in any arm (%s)" % "; ".join("%s %d scans median %.0f" % (k[-4:], v[0], v[1]) for k, v in ss.items()) if not sm else "; ".join(sm))
    # DRYSTREAM: [dry_run_stream] counters (new in R86B) from dry_run_stream_lines.txt; split arms must show trimming engaged and no rejected message
    dl = os.path.join(ret, "dry_run_stream_lines.txt"); drow = []
    if not os.path.isfile(dl):
        M("DRYSTREAM", "dry_run_stream_lines.txt absent")
    else:
        bl = engagement_blocks(dl); ok = True; msgs = []
        for t, a in arms.items():
            mp, rp = PFN[t]
            lines = [x for x in bl.get(a, []) if "[dry_run_stream]" in x]
            m = None
            for x in lines:
                m = re.search(r"msgs=(\d+) trimmed_msgs=(\d+) trimmed_points=(\d+) pushed_msgs=(\d+) rejected_msgs=(\d+)", x) or m
            if mp == rp:
                if lines: ok = False; msgs.append(a + ": [dry_run_stream] line although no second stream is configured")
                continue
            if not m: ok = False; msgs.append(a + ": no [dry_run_stream] line"); continue
            n, tm, tp, pu, rj = [int(m.group(i)) for i in range(1, 6)]
            drow.append(dict(arm=a, msgs=n, trimmed_msgs=tm, trimmed_points=tp, pushed_msgs=pu, rejected_msgs=rj))
            if rj != 0 or tm <= 0 or pu <= 0 or n < 500:
                ok = False; msgs.append("%s: msgs=%d trimmed_msgs=%d pushed=%d rejected=%d (need rejected 0, trimmed > 0, msgs >= 500)" % (a, n, tm, pu, rj))
        (P if ok else F)("DRYSTREAM", "second stream: trimming engaged and 0 rejected messages in both split arms; none in the primary arms" if ok else "; ".join(msgs))
    with open(os.path.join(out, "r86b_dry_run_stream.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "msgs", "trimmed_msgs", "trimmed_points", "pushed_msgs", "rejected_msgs"]); w.writeheader(); w.writerows(drow)
    # TRIM: first 15 messages
    trows, tot = [], {}
    for t, a in arms.items():
        p = os.path.join(ret, "archive", a, "lidar_msg_log.csv.gz")
        if not os.path.isfile(p): continue
        rr = read_csv_gz(p)[:15]
        tot[a] = sum(int(float(r.get("n_trimmed", 0) or 0)) for r in rr)
        for r in rr:
            trows.append(dict(arm=a, msg_index=r.get("msg_index"), seq=r.get("seq"), n_kept=r.get("n_kept"), verdict=r.get("verdict"), n_trimmed=r.get("n_trimmed")))
    with open(os.path.join(out, "r86b_trim_first_frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "msg_index", "seq", "n_kept", "verdict", "n_trimmed"]); w.writeheader(); w.writerows(trows)
    (I if tot else M)("TRIM", "points trimmed by retime_trim in the first 15 messages (primary stream only; the second stream is pushed before the trim, SUSPECTED): " + (", ".join("%s %d" % (k, v) for k, v in tot.items()) if tot else "lidar_msg_log absent"))
    npass = sum(1 for r in R if r[1] == "PASS"); nf = sum(1 for r in R if r[1] == "FAIL"); nm = sum(1 for r in R if r[1] == "MISSING")
    with open(os.path.join(out, "r86b_gates.txt"), "w") as f:
        for g, v, t in R:
            f.write("%-8s %-14s %s\n" % (v, g, t))
        f.write("PASS %d  FAIL %d  MISSING %d\n" % (npass, nf, nm))
    print(open(os.path.join(out, "r86b_gates.txt")).read())
    return nf, nm


N_IT = 3   # iterations of the synthetic frame 1
HDR_RES = ("frame,iter,idx,source_index,t,r,sigma_squared,plane_var_term,plane_key,node_id,tier,vis,hrot_x,hrot_y,hrot_z,hpos_x,hpos_y,hpos_z,wx,wy,wz,bx,by,bz,row_kind,floor_term,"
           "sigma_diag_squared,s_prior_pose,sigma_gate_squared,voxel_point_count,voxel_distinct_frames,voxel_layer")   # copied from logResidualDump (lio_base.cpp)
HDR_CHAIN = ("architecture,phase,scan_id,iteration,t_abs,t_rel,x0_px,x0_py,x0_pz,ximu_px,ximu_py,ximu_pz,xlio_px,xlio_py,xlio_pz,x0_r_x,x0_r_y,x0_r_z,ximu_r_x,ximu_r_y,ximu_r_z,"
             "xlio_r_x,xlio_r_y,xlio_r_z,x0_vx,x0_vy,x0_vz,ximu_vx,ximu_vy,ximu_vz,xlio_vx,xlio_vy,xlio_vz")   # leading columns of logFirstFrameStateChain (lio_iekf.cpp); the rest are not read
HDR_BUDGET = ("architecture,phase,scan_id,iteration,t_abs,t_rel,dim,p_before_trace,p_after_imu_trace,p_after_lio_trace,delta_p_trace_imu,delta_p_trace_lio")   # leading columns of logFirstFrameCovarianceBudget
ST_COLS = (["run_id", "scan_id", "t_abs", "t_rel_s", "phase", "dim", "qw", "qx", "qy", "qz", "px", "py", "pz", "vx", "vy", "vz", "bgx", "bgy", "bgz", "bax", "bay", "baz", "gx", "gy", "gz",
            "residual_count", "completed_iterations", "ol_window", "ol_reset"] + ["P_%d_%d" % (i, j) for i in range(18) for j in range(i, 18)])   # writeStateTraceRow (state_trace.cpp)


def build_fixture(t, mut=()):
    """A synthetic, mutually consistent return folder; `mut` names one deliberate corruption. Returns (ret, man). Formats follow the C++ writers and the R86A files; the four pose_control files
    are written from the source, never from a real file (UNVERIFIED)."""
    import shutil
    here = os.path.dirname(os.path.abspath(__file__)); man0 = os.path.join(here, "..", "..", "r86b_manifest")
    ret = os.path.join(t, "ret"); man = os.path.join(t, "man"); os.makedirs(ret); shutil.copytree(man0, man)
    refs = {"note": "synthetic", "resid": {}, "map": {}}
    blocks, dl = [], []
    q = {"pre": (-0.0437, -0.0098, 0.99899, -0.0001), "imu": (-0.04372, -0.00985, 0.998985, -0.00011), "lio": (-0.04375, -0.00968, 0.998995, 0.00006)}
    pos = {"pre": (1.0e-4, -2.0e-4, 3.0e-5), "imu": (1.2e-4, -1.9e-4, 2.0e-5)}
    aft = [(1.3e-4, -1.7e-4, 2.0e-5), (1.31e-4, -1.69e-4, 1.9e-5), (1.311e-4, -1.689e-4, 1.8e-5)]
    posl = aft[-1]
    dg = {"pre": 1e-6, "imu": 2e-6, "lio": 5e-7}
    def stline(scan, ph, qq, pp, dgv, rc, it):
        d = dict(run_id="x", scan_id=scan, t_abs="1609059013.40685", t_rel_s="0.1", phase=ph, dim=18, qw=qq[0], qx=qq[1], qy=qq[2], qz=qq[3], px=repr(pp[0]), py=repr(pp[1]), pz=repr(pp[2]),
                 vx=0, vy=0, vz=0, bgx=0, bgy=0, bgz=0, bax=0, bay=0, baz=0, gx=0, gy=0, gz=-9.81, residual_count=rc, completed_iterations=it, ol_window=0, ol_reset=0)
        for i in range(18):
            for j in range(i, 18):
                d["P_%d_%d" % (i, j)] = repr(dgv * (1 + i)) if i == j else "0"
        return ",".join(str(d[c]) for c in ST_COLS)
    for tag, (mp, rp) in PFN.items():
        a = aid(tag); ad = os.path.join(ret, "archive", a); cd = os.path.join(ret, "compact", a); os.makedirs(ad); os.makedirs(cd)
        W = lambda fn, txt: gzip.open(os.path.join(ad, fn), "wt").write(txt) if False else _wgz(os.path.join(ad, fn), txt)
        # compact tables: the shipped R86A traces (IDENT) and short scan tables
        for fn in ("pos_trace_all.csv.gz", "short_scan_all.csv.gz"):
            shutil.copy(os.path.join(man, "ref_r86a_%s_%s" % (tag, fn)), os.path.join(cd, fn))
        if "ident" in mut and tag == "m3r3":
            rows = read_csv_gz(os.path.join(cd, "pos_trace_all.csv.gz")); rows[10]["px"] = repr(float(rows[10]["px"]) + 1e-5)
            _wgz(os.path.join(cd, "pos_trace_all.csv.gz"), _csv_text(rows))
        s1 = [r for r in read_csv_gz(os.path.join(cd, "short_scan_all.csv.gz")) if r["scan"] == "1"][0]
        nvox, npl = int(float(s1["fs_n_voxels"])), int(float(s1["fs_n_voxels_is_plane"]))
        # state_trace
        sl = [",".join(ST_COLS)]
        if "nopre" not in mut: sl.append(stline(1, "pre_imu", q["pre"], pos["pre"], dg["pre"], 0, 0))
        sl.append(stline(1, "post_imu", q["imu"], pos["imu"], dg["imu"], 0, 0))
        sl.append(stline(1, "post_lio", q["lio"], posl, dg["lio"], 6, N_IT))
        sl.append(stline(2, "post_imu", q["lio"], posl, dg["lio"], 0, 0))
        W("state_trace.csv.gz", "\n".join(sl) + "\n")
        # residual dump (frame 1, every iteration), its state file, the solve matrices
        rl, sr, sm = [HDR_RES], ["frame,iter,n_resid,px,py,pz,rvx,rvy,rvz"], []
        for k in range(N_IT):
            n = 5 + k
            if "dropiter" in mut and k == N_IT - 1 and tag == "m3r3": continue
            for r in range(n):
                nid = 100 + r
                if "badnode" in mut and k == 0 and r == 0 and tag == "m3r3": nid = 999
                rl.append("1,%d,%d,%d,0.01,%s,0.0007,1e-05,12345,%d,0,1,0,0,1,0,0,1,1,1,1,0,0,0,0,0.0007,0.0007,0,0.0007,%d,1,0" % (k, r, 7 + r, repr(0.01 * (r + 1) / (k + 1)), nid, 3 + r))
            ap = aft[k]
            if "state_mismatch" in mut and k == 1 and tag == "m3r3": ap = (ap[0] + 1e-7, ap[1], ap[2])
            sr.append("1,%d,%d,%s,%s,%s,0,-3.05,0" % (k, n, repr(ap[0]), repr(ap[1]), repr(ap[2])))
            b = pos["imu"] if k == 0 else aft[k - 1]
            sm.append("=== iekf_solve_snapshot ===\narchitecture=iekf\nframe_idx=1\nscan_id=0\niteration=%d\nt_abs=1609059013.4\nn_points=3512\nn_residuals=%d\nstate_before_pos= %s %s %s\n"
                      "state_after_pos=  %s %s %s\ndelta_pos=0 0 0\nmatrix prior_cov rows=18 cols=18\nresidual 0 t=0.02 r=0.013\n=== end_iekf_solve_snapshot ===" % (k, n, repr(b[0]), repr(b[1]), repr(b[2]), repr(aft[k][0]), repr(aft[k][1]), repr(aft[k][2])))
        if "digest" in mut and tag == "m3r3": rl[2] = rl[2].replace("0.0007,1e-05", "0.0008,1e-05", 1)
        rtxt = "\n".join(rl) + "\n"
        W("residual_dump.csv.gz", rtxt); W("residual_dump_state.csv.gz", "\n".join(sr) + "\n"); W("pose_control_first_frame_solve_matrices.txt.gz", "\n".join(sm) + "\n")
        pth = os.path.join(ad, "residual_dump.csv.gz")
        d = DC.resid_digests(pth)
        if "digest" in mut and tag == "m3r3":   # the reference is the uncorrupted file
            rl2 = rtxt.replace("0.0008,1e-05", "0.0007,1e-05", 1); _wgz(os.path.join(t, "tmp.csv.gz"), rl2); d = DC.resid_digests(os.path.join(t, "tmp.csv.gz"))
        refs["resid"][tag] = {"0": list(d[0]), "last_iter": N_IT - 1, "last": list(d.get(N_IT - 1, d[0]))}
        # maps: post-calibration (nodes 100..139, planes 100..129) and f2 (nvox rows, npl planes)
        ml = ["snapshot_frame,node_id,layer,is_plane,point_count,tangent_x_x,tangent_y_z"]
        for n in range(100, 140):
            ml.append("0,%d,0,%d,%d,0.5,0.25" % (n, 1 if n < 130 else 0, 3 + (n - 100)))
        mtxt = "\n".join(ml) + "\n"
        refs["map"][tag] = None
        if "mapchange" in mut and tag == "m3r3": ml[3] = ml[3].replace("0,102,0,1,5", "0,102,0,1,6")
        if "tangent" in mut and tag == "m3r3": ml[4] = ml[4].replace("0.5,0.25", "0.5000001,0.25")
        W("post_calibration_voxelmap.csv.gz", "\n".join(ml) + "\n")
        _wgz(os.path.join(t, "ref.csv.gz"), mtxt); refs["map"][tag] = list(DC.map_digest(os.path.join(t, "ref.csv.gz")))
        n_snap = nvox - (1 if "snapshort" in mut and tag == "m3r3" else 0)
        W("voxelmap_snapshot_f2.csv.gz", "snapshot_frame,node_id,layer,is_plane,point_count\n" + "".join("2,%d,0,%d,4\n" % (n, 1 if n < npl else 0) for n in range(n_snap)))
        # chain and covariance budget
        if "nochain" not in mut:
            def rv(qq): return DC.rotvec(*qq)
            def crow(phase, it, x0p, ximp, xlip, rr):
                x0r, xir, xlr = rv(q["pre"]), rv(q["imu"]), rr
                return ",".join(["iekf", phase, "0", str(it), "1609059013.4", "0.1"] + [repr(v) for v in list(x0p) + list(ximp) + list(xlip) + list(x0r) + list(xir) + list(xlr) + [0.0] * 9])
            ch = [HDR_CHAIN]
            x0 = pos["pre"] if "chainpos" not in mut or tag != "m3r3" else (pos["pre"][0] + 1e-6,) + pos["pre"][1:]
            ch.append(crow("pre_update", -1, x0, pos["imu"], pos["imu"], rv(q["imu"])))
            for k in range(N_IT):
                ch.append(crow("post_iteration", k, x0, pos["imu"], aft[k], rv(q["lio"])))
            W("pose_control_first_frame_state_chain.csv.gz", "\n".join(ch) + "\n")
        bc = [HDR_BUDGET] + ["iekf,%s,0,%d,1609059013.4,0.1,18,1,1,1,0,0" % (ph, it) for ph, it in [("pre_update", -1)] + [("post_iteration", k) for k in range(N_IT)] + [("post_covariance", N_IT - 1)]]
        W("pose_control_first_frame_covariance_budget.csv.gz", "\n".join(bc) + "\n")
        def mat(dgv): return "\n".join(" ".join(("%g" % (dgv * (1 + i)) if i == j else "0") for j in range(18)) for i in range(18))
        pb = dg["pre"] * (1.01 if "covchange" in mut and tag == "m3r3" else 1.0)
        txt = ""
        for ph, it in [("pre_update", -1)] + [("post_iteration", k) for k in range(N_IT)] + [("post_covariance", N_IT - 1)]:
            txt += "=== covariance_budget iekf %s iter=%d t_abs=1609059013.4 ===\nP_before_IMU\n%s\nP_after_IMU\n%s\nP_after_LIO\n%s\ndelta_P_IMU\n0\ndelta_P_LIO\n0\n" % (ph, it, mat(pb), mat(dg["imu"]), mat(dg["lio"]))
        W("pose_control_first_frame_covariance_budget.txt.gz", txt)
        W("pose_control_first_frame.csv.gz", "architecture,phase,scan_id,iteration,t_abs,t_rel,n_points,n_residuals,map_points,active_voxels,point_hash,residual_hash,p_x,p_y,p_z\n" +
          "".join("iekf,%s,0,%d,1609059013.4,0.1,3512,%d,1,1,0,0,0,0,0\n" % (ph, it, 5 + max(it, 0)) for ph, it in [("pre_update", -1)] + [("post_iteration", k) for k in range(N_IT)]))   # leading columns of logIekfFirstFrame
        # files of the older gates
        W("exit_status.txt.gz", "exit_code=0 signal=none\n")
        W("lidar_msg_log.csv.gz", "msg_index,seq,n_kept,verdict,n_trimmed\n" + "".join("%d,%d,4000,0,%d\n" % (i, 119 + i, 2 if i > 6 else 0) for i in range(20)))
        footer = "[engagement] a flag marked INERT above did not test anything; the cell is invalid, not null.\n"   # as in every real engagement block
        if mp == rp: blocks.append("### %s\n--- engagement.txt\n[engagement]\n%s" % (a, footer))
        else:
            pm, pi = (12000.0, 4000.0) if mp < rp else (4000.0, 12000.0)
            blocks.append("### %s\n--- engagement.txt\n  lio/iekf_point_stream=second dry_run_point_filter_num=%d frames=595 frames_second=595 frames_fallback=0 mean_points_map=%s mean_points_iekf=%s\n%s" % (a, rp, pm, pi, footer))
            dl.append("### %s\n[ INFO] [1.0]: [dry_run_stream] msgs=600 trimmed_msgs=58 trimmed_points=300 pushed_msgs=600 rejected_msgs=0\n" % a)
    open(os.path.join(ret, "engagement_all.txt"), "w").write("\n".join(blocks))
    open(os.path.join(ret, "dry_run_stream_lines.txt"), "w").write("\n".join(dl))
    json.dump(refs, open(os.path.join(man, "ref_r86a_digests.json"), "w"))
    return ret, man


def _wgz(path, txt):
    with gzip.open(path, "wt", newline="") as f:
        f.write(txt)


def _csv_text(rows):
    o = io.StringIO(); w = csv.DictWriter(o, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows); return o.getvalue()


def selftest():
    ck = []
    def gates_of(mut):
        t = tempfile.mkdtemp(prefix="r86b_gates_"); ret, man = build_fixture(t, mut)
        run(ret, man, ret)
        return {ln.split()[1]: ln.split()[0] for ln in open(os.path.join(ret, "r86b_gates.txt")) if ln[:4] in ("PASS", "FAIL", "MISS", "INFO")}, ret
    g, ret = gates_of(())
    for name in ("EXIT", "STREAM", "DUMPS", "SMALLSCANS", "DRYSTREAM", "CONTROL_m3r3", "CONTROL_m1r1", "PREIMU", "CHAIN", "ITERS", "COVBUDGET", "DIGEST", "MAPLINK", "MAPSAME", "SNAPF2") + tuple("IDENT_R86A_" + x for x in PFN):
        ck.append(("clean fixture: %s PASS (got %s)" % (name, g.get(name)), g.get(name) == "PASS"))
    ck.append(("crosscheck csv has rows", len(list(csv.DictReader(open(os.path.join(ret, "r86b_dump_crosschecks.csv"))))) > 200))
    # each deliberate corruption must flip exactly the gate it targets
    for mut, gate, verdict in (("nopre", "PREIMU", "FAIL"), ("chainpos", "CHAIN", "FAIL"), ("state_mismatch", "ITERS", "FAIL"), ("dropiter", "ITERS", "FAIL"), ("covchange", "COVBUDGET", "FAIL"),
                               ("digest", "DIGEST", "FAIL"), ("badnode", "MAPLINK", "FAIL"), ("mapchange", "MAPSAME", "FAIL"), ("snapshort", "SNAPF2", "FAIL"), ("ident", "IDENT_R86A_m3r3", "FAIL"),
                               ("nochain", "CHAIN", "MISSING")):
        gg, _ = gates_of((mut,))
        ck.append(("%s -> %s %s (got %s)" % (mut, gate, verdict, gg.get(gate)), gg.get(gate) == verdict))
    gg, _ = gates_of(("tangent",))
    ck.append(("a change in tangent_* only still passes MAPSAME (got %s)" % gg.get("MAPSAME"), gg.get("MAPSAME") == "PASS"))
    # the older gates: the R86 signature and a wrong stream direction
    t = tempfile.mkdtemp(prefix="r86b_gates_"); ret, man = build_fixture(t)
    p3 = os.path.join(ret, "compact", aid("m1r3"), "short_scan_all.csv.gz")
    rows = read_csv_gz(p3); rows[4]["fs_n_residuals"] = "53"; _wgz(p3, _csv_text(rows))
    dl = os.path.join(ret, "dry_run_stream_lines.txt"); open(dl, "w").write(open(dl).read().replace("rejected_msgs=0", "rejected_msgs=40", 1))
    run(ret, man, ret); g2 = open(os.path.join(ret, "r86b_gates.txt")).read()
    ck.append(("a scan with 53 residuals fails SMALLSCANS", "FAIL     SMALLSCANS" in g2))
    ck.append(("rejected messages fail DRYSTREAM", "FAIL     DRYSTREAM" in g2))
    eg = os.path.join(ret, "engagement_all.txt")
    open(eg, "w").write(open(eg).read().replace("mean_points_map=4000.0 mean_points_iekf=12000.0", "mean_points_map=12000.0 mean_points_iekf=4000.0")); run(ret, man, ret)
    ck.append(("wrong direction fails STREAM", "FAIL     STREAM" in open(os.path.join(ret, "r86b_gates.txt")).read()))
    open(eg, "w").write(open(eg).read().replace("frames_fallback=0", "frames_fallback=0 <-- INERT: x", 1)); run(ret, man, ret)
    ck.append(("a per-flag INERT marker fails STREAM", "FAIL     STREAM" in open(os.path.join(ret, "r86b_gates.txt")).read()))
    # the exit status in its real format, and a non-zero one
    ex = os.path.join(ret, "archive", aid("m3r1"), "exit_status.txt.gz"); _wgz(ex, "exit_code=1 signal=none\n"); run(ret, man, ret)
    ck.append(("exit_code=1 fails EXIT", "FAIL     EXIT" in open(os.path.join(ret, "r86b_gates.txt")).read()))
    ck.append(("rotvec equals the node's Log on real R86A numbers", max(abs(x - y) for x, y in zip(DC.rotvec(-0.043702467586554113, -0.0098436788785481854, 0.99899608829459474, -0.00010901871474733691),
                                                                                              (0.030092920077094246, -3.0540115959992891, 0.00033327900170954458))) < 1e-12))
    bad = [n for n, ok in ck if not ok]
    print("SELFTEST r86b_gates", ("FAIL " + "; ".join(bad)) if bad else "PASS", "checks: %d" % len(ck))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ret-dir"); ap.add_argument("--manifest", default="r86b_manifest"); ap.add_argument("--out", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    nf, nm = run(a.ret_dir, a.manifest, a.out or a.ret_dir)
    sys.exit(1 if nf else 0)
