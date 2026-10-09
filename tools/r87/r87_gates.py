#!/usr/bin/env python3
"""r87_gates.py -- R87 gates (planner's script, selftest on a synthetic folder only; not evidence, rule 1.1). Reads a collected return folder; writes r87_gates.txt, r87_identity.csv,
r87_admission.csv, r87_stream_engagement.csv. Verdicts: PASS, FAIL, MISSING (never a pass), INFO. Facts only; no interpretation.
Usage: python3 tools/r87/r87_gates.py --ret-dir R87_RETURN --manifest r87_manifest [--out R87_RETURN]
       python3 tools/r87/r87_gates.py --selftest
"""
import argparse, csv, gzip, hashlib, io, json, os, re, shutil, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refdigests as RD

IDENT_TOL_MM = 0.0005
MIN_POINTS = 5
PFN = {"m3r3": (3, 3), "m1r1": (1, 1), "m1r3": (1, 3), "m3r1": (3, 1)}
MAPREF = {"m3r3": "mpf3", "m3r1": "mpf3", "m1r1": "mpf1", "m1r3": "mpf1"}
SETS = ["c31", "n55"]
C = "r87_rt_sp_eee_01_v0p25_"


def aid(tag, s):
    return "%s%s_%s" % (C, tag, s)


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


def pre_imu_hash(path):
    with rd_gz(path) as f:
        rows = list(csv.reader(f))
    h = rows[0]; ip, isc = h.index("phase"), h.index("scan_id")
    pre = [r for r in rows[1:] if r[ip] == "pre_imu" and r[isc] == "1"]
    ph = [r[ip] for r in rows[1:] if r[isc] == "1"]
    return (hashlib.sha256(",".join(pre[0][1:]).encode()).hexdigest() if len(pre) == 1 else None), ph


def map_planes(path):
    """{node_id: tuple(tokens without tangent_* and snapshot_frame)} of the plane voxels, and {node_id: point_count} of them."""
    out, pc = {}, {}
    with rd_gz(path) as f:
        r = csv.reader(f); hdr = next(r)
        keep = [i for i, c in enumerate(hdr) if not c.startswith("tangent_") and c != "snapshot_frame"]
        iid, ipl, ipc = hdr.index("node_id"), hdr.index("is_plane"), hdr.index("point_count")
        for row in r:
            if row[ipl] == "1":
                out[row[iid]] = tuple(row[i] for i in keep); pc[row[iid]] = int(float(row[ipc]))
    return out, pc


def all_point_counts(path):
    with rd_gz(path) as f:
        r = csv.reader(f); hdr = next(r); iid, ipc = hdr.index("node_id"), hdr.index("point_count")
        return {row[iid]: int(float(row[ipc])) for row in r}


def run(ret, man, out):
    R = []
    P = lambda g, t: R.append((g, "PASS", t)); F = lambda g, t: R.append((g, "FAIL", t)); M = lambda g, t: R.append((g, "MISSING", t)); I = lambda g, t: R.append((g, "INFO", t))
    arms = {(t, s): aid(t, s) for s in SETS for t in PFN}
    ref = json.load(open(os.path.join(man, "ref_r86b_digests.json")))["arms"]
    req = json.load(open(os.path.join(man, "required_outputs.json")))
    gen = ("r87_gates.txt", "r87_identity.csv", "r87_admission.csv", "r87_stream_engagement.csv")
    miss = []
    for r in req:
        if r["where"] == "return" and r["source"] == "arm":
            ps = [os.path.join(ret, "compact", a, r["file"]) for a in r["arms"]]
        elif r["where"] == "return":
            ps = [os.path.join(ret, r["file"])] if r["file"] not in gen else []
        else:
            ps = [os.path.join(ret, "archive", a, r["file"]) for a in r["arms"]]
        miss += [p for p in ps if not os.path.isfile(p) or os.path.getsize(p) == 0]
    (F if miss else P)("REQUIRED", "%d required path(s) missing or empty%s" % (len(miss), ": " + "; ".join(os.path.relpath(m, ret) for m in miss[:20]) if miss else ""))
    bad = []
    for a in arms.values():
        p = os.path.join(ret, "archive", a, "exit_status.txt.gz")
        if not os.path.isfile(p): bad.append(a + " (absent)"); continue
        s = rd_gz(p).read().split()
        code = s[0].split("=", 1)[1] if s and s[0].startswith("exit_code=") else (s[0] if s else None)
        if code != "0": bad.append("%s (%s)" % (a, s[:1]))
    (F if bad else P)("EXIT", "exit status 0 for all 8 arms" if not bad else "non-zero or absent: " + ", ".join(bad))
    # IDENT_R86B: the c31 arms reproduce the R86B arm of the same tag (position, counts, process-noise file, every-iteration residual rows, map without tangent_*)
    irow = []
    for t in PFN:
        a = arms[(t, "c31")]
        pn = os.path.join(ret, "compact", a, "pos_trace_all.csv.gz"); pr = os.path.join(man, "ref_r86b_%s_pos_trace_all.csv.gz" % t)
        sn = os.path.join(ret, "compact", a, "short_scan_all.csv.gz"); sr = os.path.join(man, "ref_r86b_%s_short_scan_all.csv.gz" % t)
        cg = os.path.join(ret, "archive", a, "imu_cov_growth.csv.gz"); rdp = os.path.join(ret, "archive", a, "residual_dump.csv.gz"); mp = os.path.join(ret, "archive", a, "post_calibration_voxelmap.csv.gz")
        if not all(os.path.isfile(x) for x in (pn, sn, cg, rdp, mp)):
            M("IDENT_R86B_" + t, a + ": files absent"); continue
        N = {int(float(r["scan_id"])): r for r in read_csv_gz(pn)}; Rf = {int(float(r["scan_id"])): r for r in read_csv_gz(pr)}
        com = sorted(set(N) & set(Rf)); mx = 0.0
        for sc in com:
            mx = max(mx, max(abs(float(N[sc][k]) - float(Rf[sc][k])) for k in ("px", "py", "pz")) * 1000.0)
        SN = {int(float(r["scan"])): r for r in read_csv_gz(sn)}; SR = {int(float(r["scan"])): r for r in read_csv_gz(sr)}
        cs = sorted(set(SN) & set(SR)); ne = sum(1 for sc in cs if any(SN[sc][k] != SR[sc][k] for k in ("fs_n_residuals", "fs_n_voxels", "fs_n_voxels_is_plane")))
        shaok = hashlib.sha256(gzip.open(cg, "rb").read()).hexdigest() == ref[t]["imu_cov_growth_sha256"]
        d = RD.resid_digests(rdp); want = {int(k): tuple(v) for k, v in ref[t]["resid"].items()}
        residok = d == want
        mapok = list(RD.map_digest(mp)) == ref[t]["map"]
        irow.append(dict(arm=a, ref="R86B " + t, n_common_scans=len(com), pos_max_abs_diff_mm=round(mx, 6), n_scans_count_columns_differ=ne, imu_cov_growth_sha256_equal=int(shaok),
                         residual_rows_all_iterations_equal=int(residok), map_equal=int(mapok)))
        ok = len(com) >= 500 and mx <= IDENT_TOL_MM and ne == 0 and len(cs) >= 500 and shaok and residok and mapok
        (P if ok else F)("IDENT_R86B_" + t, "%s vs R86B: %d common scans, max position difference %.6f mm (tolerance %.4f), counts differ in %d of %d scans, imu_cov_growth %s, frame-1 residual rows (all iterations) %s, map (no tangent_*) %s"
                         % (a, len(com), mx, IDENT_TOL_MM, ne, len(cs), "equal" if shaok else "DIFFERENT", "equal" if residok else "DIFFERENT", "equal" if mapok else "DIFFERENT"))
    with open(os.path.join(out, "r87_identity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "ref", "n_common_scans", "pos_max_abs_diff_mm", "n_scans_count_columns_differ", "imu_cov_growth_sha256_equal", "residual_rows_all_iterations_equal", "map_equal"]); w.writeheader(); w.writerows(irow)
    # MAP_ADMISSION: the n55 post-calibration map is the R86B map restricted to voxels with at least 5 raw points (same fit values), and nothing else
    arow = []
    for t in PFN:
        a = arms[(t, "n55")]
        mp = os.path.join(ret, "archive", a, "post_calibration_voxelmap.csv.gz"); rp = os.path.join(man, "ref_r86b_map_%s.csv.gz" % MAPREF[t])
        if not os.path.isfile(mp): M("MAP_ADMISSION_" + t, a + ": map absent"); continue
        new, newpc = map_planes(mp); old, oldpc = map_planes(rp)
        n_below = sum(1 for k, c in newpc.items() if c < MIN_POINTS)
        extra = [k for k in new if k not in old]
        changed = [k for k in new if k in old and new[k] != old[k]]
        expect = {k for k, c in oldpc.items() if c >= MIN_POINTS}
        missing = sorted(expect - set(new))
        n3 = sum(1 for c in oldpc.values() if c == 3)
        arow.append(dict(arm=a, ref_plane_voxels=len(old), ref_plane_voxels_with_3_points=n3, ref_plane_voxels_with_5_or_more_points=len(expect), new_plane_voxels=len(new),
                         new_plane_voxels_below_5_points=n_below, new_plane_voxels_not_in_ref=len(extra), new_plane_voxels_with_changed_values=len(changed), ref_voxels_with_5_or_more_not_plane_in_new=len(missing)))
        ok = n_below == 0 and not extra and not changed and not missing
        (P if ok else F)("MAP_ADMISSION_" + t, "%s: plane voxels %d (R86B %d; R86B with 5 or more points %d); below 5 points %d; not in R86B %d; values changed %d; R86B 5-or-more voxels missing %d"
                         % (a, len(new), len(old), len(expect), n_below, len(extra), len(changed), len(missing)))
    with open(os.path.join(out, "r87_admission.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "ref_plane_voxels", "ref_plane_voxels_with_3_points", "ref_plane_voxels_with_5_or_more_points", "new_plane_voxels", "new_plane_voxels_below_5_points",
                                          "new_plane_voxels_not_in_ref", "new_plane_voxels_with_changed_values", "ref_voxels_with_5_or_more_not_plane_in_new"]); w.writeheader(); w.writerows(arow)
    # ENGAGED + MINPOINTS: the rule reached the LIO residuals of frame 1
    bad_e, bad_m, info = [], [], []
    for t in PFN:
        a = arms[(t, "n55")]; p = os.path.join(ret, "archive", a, "residual_dump.csv.gz")
        if not os.path.isfile(p): bad_e.append(a + ": residual_dump absent"); continue
        rows = [r for r in read_csv_gz(p) if r["frame"] == "1"]
        cnt0 = sum(1 for r in rows if r["iter"] == "0"); old0 = ref[t]["resid"]["0"][0]
        low = sum(1 for r in rows if int(float(r["voxel_point_count"])) < MIN_POINTS)
        info.append("%s iteration-0 rows %d (R86B %d)" % (t, cnt0, old0))
        if not cnt0 < old0: bad_e.append("%s: iteration-0 rows %d not below R86B %d" % (a, cnt0, old0))
        if low: bad_m.append("%s: %d residual rows from voxels with fewer than %d points" % (a, low, MIN_POINTS))
    (F if bad_e else P)("ENGAGED", "iteration-0 residual rows of frame 1 below R86B in every n55 arm (" + "; ".join(info) + ")" if not bad_e else "; ".join(bad_e))
    (F if bad_m else P)("MINPOINTS", "no frame-1 residual row (any iteration) comes from a voxel with fewer than %d points in the four n55 arms" % MIN_POINTS if not bad_m else "; ".join(bad_m))
    # PREIMU_SAME: the admission rule acts on the map, not on the state before propagation
    bad = []
    for t in PFN:
        a = arms[(t, "n55")]; p = os.path.join(ret, "archive", a, "state_trace.csv.gz")
        if not os.path.isfile(p): bad.append(a + ": state_trace absent"); continue
        h, ph = pre_imu_hash(p)
        if h != ref[t]["pre_imu_row_sha256"]: bad.append("%s: pre_imu row differs from R86B (or absent)" % a)
        if ph != ["pre_imu", "post_imu", "post_lio"]: bad.append("%s: scan-1 phases %s" % (a, ph))
    (F if bad else P)("PREIMU_SAME", "scan-1 pre_imu row equals R86B's in the four n55 arms; phases pre_imu, post_imu, post_lio" if not bad else "; ".join(bad))
    # SMALLSCANS: stationary window (scans 1..475) must have no scan under half the arm's median; lower scans later (motion) are listed as INFO
    sm = []; late = []
    for (t, s), a in arms.items():
        p = os.path.join(ret, "compact", a, "short_scan_all.csv.gz")
        if not os.path.isfile(p): sm.append(a + ": short_scan_all absent"); continue
        rr = [(int(float(r["scan"])), float(r["fs_n_residuals"])) for r in read_csv_gz(p) if r.get("fs_n_residuals") not in (None, "", "nan")]
        if not rr: sm.append(a + ": no residual counts"); continue
        med = sorted(x for _, x in rr)[len(rr) // 2]
        for sc, x in rr:
            if x < 0.5 * med:
                (sm if sc <= 475 else late).append("%s scan %d (%d vs median %.0f)" % (a, sc, x, med))
    (F if sm else P)("SMALLSCANS", "no scan up to 475 under half the arm's median residual count in any of the 8 arms" if not sm else "; ".join(sm))
    if late: I("SMALLSCANS_LATE", "scans after 475 under half the median: " + "; ".join(late))
    # STREAM
    eng = os.path.join(ret, "engagement_all.txt"); rows = []
    if not os.path.isfile(eng):
        M("STREAM", "engagement_all.txt absent")
    else:
        blocks = engagement_blocks(eng); ok = True; msgs = []
        for (t, s), a in arms.items():
            txt = "\n".join(blocks.get(a, [])); mp, rp = PFN[t]
            m = re.search(r"lio/iekf_point_stream=(\w+) dry_run_point_filter_num=(\d+) frames=(\d+) frames_second=(\d+) frames_fallback=(\d+) mean_points_map=([\d.eE+-]+) mean_points_iekf=([\d.eE+-]+)", txt)
            if not txt: ok = False; msgs.append(a + ": no engagement block"); continue
            if mp == rp:
                if m: ok = False; msgs.append(a + ": stream line present although stream is primary")
                rows.append(dict(arm=a, stream="primary", frames="", frames_second="", frames_fallback="", mean_points_map="", mean_points_iekf="")); continue
            if not m: ok = False; msgs.append(a + ": no iekf_point_stream engagement line"); continue
            st, dn, fr, fs, fb, pm, pi = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), float(m.group(6)), float(m.group(7))
            rows.append(dict(arm=a, stream=st, frames=fr, frames_second=fs, frames_fallback=fb, mean_points_map=pm, mean_points_iekf=pi))
            if st != "second" or dn != rp or fr <= 0 or fs != fr or fb != 0 or "<-- INERT" in txt or "<-- PARTLY" in txt:
                ok = False; msgs.append("%s: stream=%s dry_run=%d frames=%d second=%d fallback=%d" % (a, st, dn, fr, fs, fb))
            elif (mp < rp and not pi < pm) or (mp > rp and not pi > pm):
                ok = False; msgs.append("%s: direction wrong (map %.1f, iekf %.1f points)" % (a, pm, pi))
        (P if ok else F)("STREAM", "second stream used in every frame of the m1r3 and m3r1 arms (4 of 8), fallbacks 0, direction as designed; primary arms have no stream line" if ok else "; ".join(msgs))
    with open(os.path.join(out, "r87_stream_engagement.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "stream", "frames", "frames_second", "frames_fallback", "mean_points_map", "mean_points_iekf"]); w.writeheader(); w.writerows(rows)
    # DUMPS (n55 arms): chain, budget, first-frame csv, state dump, solve matrices, f2 snapshot hold rows; residual dump iterations 0..completed_iterations-1
    dm = []
    for t in PFN:
        a = arms[(t, "n55")]
        for fn in ("pose_control_first_frame.csv.gz", "pose_control_first_frame_state_chain.csv.gz", "pose_control_first_frame_covariance_budget.csv.gz", "residual_dump_state.csv.gz",
                   "voxelmap_snapshot_f2.csv.gz", "pose_control_first_frame_solve_matrices.txt.gz"):
            p = os.path.join(ret, "archive", a, fn)
            if not os.path.isfile(p): dm.append(a + "/" + fn + " absent"); continue
            if sum(1 for _ in rd_gz(p)) - 1 < 1: dm.append(a + "/" + fn + " has no data rows")
        p = os.path.join(ret, "archive", a, "residual_dump.csv.gz")
        if os.path.isfile(p):
            its = sorted({int(float(r["iter"])) for r in read_csv_gz(p)})
            st = [r for r in read_csv_gz(os.path.join(ret, "archive", a, "state_trace.csv.gz")) if r["phase"] == "post_lio" and r["scan_id"] == "1"]
            n = int(float(st[0]["completed_iterations"])) if st else -1
            if its != list(range(n)) or n < 1: dm.append("%s: residual_dump iterations %s, completed_iterations %d" % (a, its, n))
    (F if dm else P)("DUMPS", "frame-1 chain, budget, first-frame csv, state dump, solve matrices and f2 snapshot hold rows in the four n55 arms; residual dump has iterations 0..completed_iterations-1" if not dm else "; ".join(dm))
    npass = sum(1 for r in R if r[1] == "PASS"); nf = sum(1 for r in R if r[1] == "FAIL"); nm = sum(1 for r in R if r[1] == "MISSING")
    with open(os.path.join(out, "r87_gates.txt"), "w") as f:
        for g, v, t in R:
            f.write("%-8s %-18s %s\n" % (v, g, t))
        f.write("PASS %d  FAIL %d  MISSING %d\n" % (npass, nf, nm))
    print(open(os.path.join(out, "r87_gates.txt")).read())
    return nf, nm


# ------------------------------------------------------------------ selftest fixture (synthetic; not evidence)
NSC = 600
MAP_HDR = ["snapshot_frame", "node_id", "is_plane", "eig0", "eig1", "point_count", "tangent_x_x", "tangent_y_y"]
RES_HDR = ["frame", "iter", "idx", "source_index", "t", "r", "sigma_squared", "plane_var_term", "plane_key", "node_id", "tier", "vis", "hrot_x", "hrot_y", "hrot_z", "hpos_x", "hpos_y", "hpos_z",
           "wx", "wy", "wz", "bx", "by", "bz", "row_kind", "floor_term", "sigma_diag_squared", "s_prior_pose", "sigma_gate_squared", "voxel_point_count", "voxel_distinct_frames", "voxel_layer"]


def _wgz(path, txt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(txt.encode())


def _csv(rows, hdr):
    b = io.StringIO(); w = csv.writer(b, lineterminator="\n"); w.writerow(hdr); w.writerows(rows); return b.getvalue()


def _resid(n_per_iter, vpc, iters=5):
    rows = []
    for it in range(iters):
        for i in range(n_per_iter):
            r = {c: "0" for c in RES_HDR}; r.update(frame="1", iter=str(it), idx=str(i), source_index=str(i), r="0.00%d" % (i % 7), node_id=str(i), voxel_point_count=str(vpc(i)))
            rows.append([r[c] for c in RES_HDR])
    return _csv(rows, RES_HDR)


def _ref_map(nvox):
    # voxel k: plane; point_count 3 for k % 3 == 0, else 6 + k % 5
    return [[0, k, 1, "1e-14", "1e-14", 3 if k % 3 == 0 else 6 + k % 5, "0.1", "0.2"] for k in range(nvox)]


def build_fixture(t, mut=()):
    shutil.rmtree(t, ignore_errors=True); os.makedirs(t)
    man = os.path.join(t, "man"); ret = os.path.join(t, "ret"); os.makedirs(man); os.makedirs(ret)
    ref = {"note": "synthetic", "arms": {}}
    ph = ["scan_id", "t_abs", "t_rel_s", "px", "py", "pz", "yaw_rad", "qw", "qx", "qy", "qz"]
    sh = ["scan", "t_s", "residual_count", "fs_n_residuals", "fs_n_voxels", "fs_n_voxels_is_plane", "fs_iters", "fs_dx_pos_mm", "fs_nis", "fs_reduced_chi2", "pos_err_mm"]
    STH = ["run_id", "scan_id", "t_abs", "t_rel_s", "phase", "dim", "px", "py", "pz", "completed_iterations"]
    pre_row = ["x", "1", "1.6e9", "0.1", "pre_imu", "18", "0", "0", "0", "0"]
    refmaps = {"mpf3": _ref_map(30), "mpf1": _ref_map(40)}
    for k, rows in refmaps.items():
        _wgz(os.path.join(man, "ref_r86b_map_%s.csv.gz" % k), _csv(rows, MAP_HDR))
    for tag in PFN:
        pos = [[s, 1.6e9 + s * 0.1, s * 0.1, 0.0, 0.0, 0.0, 0.0, 1, 0, 0, 0] for s in range(1, NSC + 1)]
        sr = [[s, s * 0.1, 2000, 2000, 500, 400, 5, 0, 0, 0, 0] for s in range(1, NSC + 1)]
        _wgz(os.path.join(man, "ref_r86b_%s_pos_trace_all.csv.gz" % tag), _csv(pos, ph)); _wgz(os.path.join(man, "ref_r86b_%s_short_scan_all.csv.gz" % tag), _csv(sr, sh))
        cov = _csv([[s, 0, 0, 18, 1e-6 * (1 + tag.__hash__() % 3)] for s in range(1, NSC + 1)], ["scan_index", "t_abs", "t_rel_s", "dim", "Qacc_6"])
        n_ref = len(refmaps[MAPREF[tag]])
        rtxt = _resid(8, lambda i: 3 if i % 4 == 0 else 7)
        # reference digests built through the shared functions
        tmpr = os.path.join(t, "tmp_resid.csv.gz"); _wgz(tmpr, rtxt)
        tmpm = os.path.join(t, "tmp_map.csv.gz"); _wgz(tmpm, _csv(refmaps[MAPREF[tag]], MAP_HDR))
        ref["arms"][tag] = {"imu_cov_growth_sha256": hashlib.sha256(cov.encode()).hexdigest(), "pre_imu_row_sha256": hashlib.sha256(",".join(pre_row[1:]).encode()).hexdigest(),
                            "resid": {str(k): list(v) for k, v in sorted(RD.resid_digests(tmpr).items())}, "map": list(RD.map_digest(tmpm))}
        for s in SETS:
            a = aid(tag, s); mp, rp = PFN[tag]
            pp = [list(r) for r in pos]; ss = [list(r) for r in sr]
            if "shift" in mut and s == "c31" and tag == "m3r3": pp[100][3] += 0.0005
            if "smallscan" in mut and s == "n55" and tag == "m1r1": ss[50][3] = 100
            if "latesmall" in mut and s == "n55" and tag == "m1r1": ss[520][3] = 100
            _wgz(os.path.join(ret, "compact", a, "pos_trace_all.csv.gz"), _csv(pp, ph)); _wgz(os.path.join(ret, "compact", a, "short_scan_all.csv.gz"), _csv(ss, sh))
            _wgz(os.path.join(ret, "archive", a, "imu_cov_growth.csv.gz"), cov if not ("cov" in mut and s == "c31" and tag == "m3r3") else cov + "extra\n")
            _wgz(os.path.join(ret, "archive", a, "exit_status.txt.gz"), "exit_code=%d signal=none\n" % (1 if ("exit" in mut and s == "n55" and tag == "m3r3") else 0))
            _wgz(os.path.join(ret, "archive", a, "engagement_imu.txt.gz"), "x\n")
            if s == "c31":
                rt = rtxt
                if "residchg" in mut and tag == "m1r1": rt = rtxt.replace("0.003", "0.009", 1)
                _wgz(os.path.join(ret, "archive", a, "residual_dump.csv.gz"), rt)
                mrows = _ref_map(n_ref)
                if "mapchg" in mut and tag == "m1r3": mrows[3][3] = "2e-14"
                _wgz(os.path.join(ret, "archive", a, "post_calibration_voxelmap.csv.gz"), _csv(mrows, MAP_HDR))
            else:
                # fix on: only voxels with 5 or more points are planes; the others are written as non-plane; residual rows only from those voxels
                mrows = []
                for r in _ref_map(n_ref):
                    r = list(r)
                    if r[5] < 5: r[2] = 0
                    mrows.append(r)
                if "belowmin" in mut and tag == "m3r1": mrows[0][2] = 1
                if "extra" in mut and tag == "m1r1": mrows.append([0, 999, 1, "1e-14", "1e-14", 7, "0.1", "0.2"])
                if "valchg" in mut and tag == "m1r3": mrows[1][3] = "5e-14"
                if "dropped" in mut and tag == "m3r3": mrows[1][2] = 0
                _wgz(os.path.join(ret, "archive", a, "post_calibration_voxelmap.csv.gz"), _csv(mrows, MAP_HDR))
                nres = 8 if "inert" in mut and tag == "m3r3" else 6
                its = 5 if not ("iters" in mut and tag == "m3r3") else 4
                rt = _resid(nres, lambda i: 7, iters=5)
                if "iters" in mut and tag == "m3r3": rt = _resid(nres, lambda i: 7, iters=4)
                if "lowrow" in mut and tag == "m1r1": rt = rt.replace(",7,0,0", ",3,0,0", 1)
                _wgz(os.path.join(ret, "archive", a, "residual_dump.csv.gz"), rt)
                srows = [pre_row if not ("pre" in mut and tag == "m3r3") else ["x", "1", "1.6e9", "0.1", "pre_imu", "18", "9", "0", "0", "0"],
                         ["x", "1", "1.6e9", "0.1", "post_imu", "18", "0", "0", "0", "0"], ["x", "1", "1.6e9", "0.1", "post_lio", "18", "0", "0", "0", "5"]]
                _wgz(os.path.join(ret, "archive", a, "state_trace.csv.gz"), _csv(srows, STH))
                for fn in ("pose_control_first_frame.csv", "pose_control_first_frame_state_chain.csv", "pose_control_first_frame_covariance_budget.csv", "residual_dump_state.csv", "voxelmap_snapshot_f2.csv"):
                    _wgz(os.path.join(ret, "archive", a, fn + ".gz"), "h\n1\n" if not ("nochain" in mut and fn.endswith("state_chain.csv") and tag == "m3r3") else "h\n")
                _wgz(os.path.join(ret, "archive", a, "pose_control_first_frame_solve_matrices.txt.gz"), "h\nrow\n")
    json.dump(ref, open(os.path.join(man, "ref_r86b_digests.json"), "w"))
    blk = []
    for s in SETS:
        for tag in PFN:
            a = aid(tag, s); mp, rp = PFN[tag]; blk.append("### " + a); blk.append("[engagement] synthetic")
            if mp != rp:
                fs = 595 if not ("stream" in mut and tag == "m1r3" and s == "n55") else 590
                blk.append("lio/iekf_point_stream=second dry_run_point_filter_num=%d frames=595 frames_second=%d frames_fallback=0 mean_points_map=%s mean_points_iekf=%s" % (rp, fs, "6000" if mp < rp else "2000", "2000" if mp < rp else "6000"))
    open(os.path.join(ret, "engagement_all.txt"), "w").write("\n".join(blk) + "\n")
    reqs = [dict(file="pos_trace_all.csv.gz", where="return", source="arm", arms=[aid(tg, s) for s in SETS for tg in PFN])]
    if "reqmiss" in mut: reqs.append(dict(file="state_reference.txt", where="return", source="arm", arms=[aid("m3r3", "c31")]))
    json.dump(reqs, open(os.path.join(man, "required_outputs.json"), "w"))
    return ret, man


def selftest():
    t = tempfile.mkdtemp(prefix="r87_gates_"); fails = []
    def go(mut):
        ret, man = build_fixture(os.path.join(t, "fx"), mut)
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            run(ret, man, ret)
        v = {}
        for ln in open(os.path.join(ret, "r87_gates.txt")):
            p = ln.split(None, 2)
            if len(p) >= 2 and p[0] in ("PASS", "FAIL", "MISSING", "INFO"): v[p[1]] = p[0]
        return v
    base = go(())
    for g, x in base.items():
        if x not in ("PASS", "INFO"): fails.append("clean fixture: %s is %s" % (g, x))
    want = [("shift", "IDENT_R86B_m3r3"), ("cov", "IDENT_R86B_m3r3"), ("residchg", "IDENT_R86B_m1r1"), ("mapchg", "IDENT_R86B_m1r3"), ("belowmin", "MAP_ADMISSION_m3r1"), ("extra", "MAP_ADMISSION_m1r1"),
            ("valchg", "MAP_ADMISSION_m1r3"), ("dropped", "MAP_ADMISSION_m3r3"), ("inert", "ENGAGED"), ("lowrow", "MINPOINTS"), ("pre", "PREIMU_SAME"), ("exit", "EXIT"), ("smallscan", "SMALLSCANS"),
            ("stream", "STREAM"), ("iters", "DUMPS"), ("nochain", "DUMPS"), ("reqmiss", "REQUIRED")]
    for m, g in want:
        v = go((m,))
        if v.get(g) != "FAIL": fails.append("mutation %s did not turn %s (got %s)" % (m, g, v.get(g)))
        if base.get(g) != "PASS": fails.append("clean state of %s is not PASS" % g)
    v = go(("latesmall",))
    if v.get("SMALLSCANS") != "PASS" or v.get("SMALLSCANS_LATE") != "INFO": fails.append("a late small scan must be INFO only: %s" % {k: x for k, x in v.items() if k.startswith("SMALL")})
    shutil.rmtree(t, ignore_errors=True)
    print("SELFTEST r87_gates", ("FAIL " + "; ".join(fails)) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ret-dir"); ap.add_argument("--manifest", default="r87_manifest"); ap.add_argument("--out", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    nf, nm = run(a.ret_dir, a.manifest, a.out or a.ret_dir)
    sys.exit(1 if nf or nm else 0)
