#!/usr/bin/env python3
"""r86c_gates.py -- R86C gates (planner's script, selftest on a synthetic folder only; not evidence, rule 1.1). Reads a collected return folder; writes r86c_gates.txt, r86c_identity.csv,
r86c_acc_engagement.csv, r86c_stream_engagement.csv. Verdicts: PASS, FAIL, MISSING (never a pass), INFO. Facts only; no interpretation.
Usage: python3 tools/r86c/r86c_gates.py --ret-dir R86C_RETURN --manifest r86c_manifest [--out R86C_RETURN]
       python3 tools/r86c/r86c_gates.py --selftest
"""
import argparse, csv, gzip, hashlib, io, json, os, re, shutil, statistics, sys, tempfile

IDENT_TOL_MM = 0.0005
MONO_REL = 1e-9
PFN = {"m3r3": (3, 3), "m1r1": (1, 1), "m1r3": (1, 3), "m3r1": (3, 1)}
LEVELS = [("a300", 0.3), ("a100", 0.1), ("a030", 0.03), ("a000", 0.0)]
C = "r86c_rt_sp_eee_01_v0p25_"


def aid(tag, lvl):
    return "%s%s_%s" % (C, tag, lvl)


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


def qv_by_scan(path):
    out = {}
    for r in read_csv_gz(path):
        out[int(float(r["scan_index"]))] = sum(float(r["Qacc_%d" % i]) for i in (6, 7, 8)) / 3.0
    return out


def pre_imu_hash(path):
    with rd_gz(path) as f:
        rows = list(csv.reader(f))
    h = rows[0]; ip, isc = h.index("phase"), h.index("scan_id")
    pre = [r for r in rows[1:] if r[ip] == "pre_imu" and r[isc] == "1"]
    ph = [r[ip] for r in rows[1:] if r[isc] == "1"]
    return (hashlib.sha256(",".join(pre[0][1:]).encode()).hexdigest() if len(pre) == 1 else None), ph


def run(ret, man, out):
    R = []
    P = lambda g, t: R.append((g, "PASS", t)); F = lambda g, t: R.append((g, "FAIL", t)); M = lambda g, t: R.append((g, "MISSING", t)); I = lambda g, t: R.append((g, "INFO", t))
    arms = {(t, l): aid(t, l) for l, _ in LEVELS for t in PFN}
    ref = json.load(open(os.path.join(man, "ref_r86b_digests.json")))["arms"]
    req = json.load(open(os.path.join(man, "required_outputs.json")))
    gen_by_gates = ("r86c_gates.txt", "r86c_identity.csv", "r86c_acc_engagement.csv", "r86c_stream_engagement.csv")
    miss = []
    for r in req:
        if r["where"] == "return" and r["source"] == "arm":
            ps = [os.path.join(ret, "compact", a, r["file"]) for a in r["arms"]]
        elif r["where"] == "return":
            ps = [os.path.join(ret, r["file"])] if r["file"] not in gen_by_gates else []
        else:
            ps = [os.path.join(ret, "archive", a, r["file"]) for a in r["arms"]]
        miss += [p for p in ps if not os.path.isfile(p) or os.path.getsize(p) == 0]
    (F if miss else P)("REQUIRED", "%d required path(s) missing or empty%s" % (len(miss), ": " + "; ".join(os.path.relpath(m, ret) for m in miss[:20]) if miss else ""))
    # EXIT
    bad = []
    for a in arms.values():
        p = os.path.join(ret, "archive", a, "exit_status.txt.gz")
        if not os.path.isfile(p): bad.append(a + " (absent)"); continue
        s = rd_gz(p).read().split()
        code = s[0].split("=", 1)[1] if s and s[0].startswith("exit_code=") else (s[0] if s else None)
        if code != "0": bad.append("%s (%s)" % (a, s[:1]))
    (F if bad else P)("EXIT", "exit status 0 for all 16 arms" if not bad else "non-zero or absent: " + ", ".join(bad))
    # IDENT_R86B: the a300 arms reproduce the R86B arm of the same tag
    irow = []
    for t in PFN:
        a = arms[(t, "a300")]
        pn = os.path.join(ret, "compact", a, "pos_trace_all.csv.gz"); pr = os.path.join(man, "ref_r86b_%s_pos_trace_all.csv.gz" % t)
        sn = os.path.join(ret, "compact", a, "short_scan_all.csv.gz"); sr = os.path.join(man, "ref_r86b_%s_short_scan_all.csv.gz" % t)
        cg = os.path.join(ret, "archive", a, "imu_cov_growth.csv.gz")
        if not (os.path.isfile(pn) and os.path.isfile(sn) and os.path.isfile(cg)):
            M("IDENT_R86B_" + t, a + ": files absent"); continue
        N = {int(float(r["scan_id"])): r for r in read_csv_gz(pn)}; Rf = {int(float(r["scan_id"])): r for r in read_csv_gz(pr)}
        com = sorted(set(N) & set(Rf)); mx = 0.0
        for sc in com:
            mx = max(mx, max(abs(float(N[sc][k]) - float(Rf[sc][k])) for k in ("px", "py", "pz")) * 1000.0)
        SN = {int(float(r["scan"])): r for r in read_csv_gz(sn)}; SR = {int(float(r["scan"])): r for r in read_csv_gz(sr)}
        cs = sorted(set(SN) & set(SR)); ne = sum(1 for sc in cs if any(SN[sc][k] != SR[sc][k] for k in ("fs_n_residuals", "fs_n_voxels", "fs_n_voxels_is_plane")))
        sha = hashlib.sha256(gzip.open(cg, "rb").read()).hexdigest()
        shaok = sha == ref[t]["imu_cov_growth_sha256"]
        irow.append(dict(arm=a, ref="R86B " + t, n_common_scans=len(com), pos_max_abs_diff_mm=round(mx, 6), n_scans_count_columns_differ=ne, n_common_short_rows=len(cs), imu_cov_growth_sha256_equal=int(shaok)))
        ok = len(com) >= 500 and mx <= IDENT_TOL_MM and ne == 0 and len(cs) >= 500 and shaok
        (P if ok else F)("IDENT_R86B_" + t, "%s vs R86B: %d common scans, max position difference %.6f mm (tolerance %.4f), counts differ in %d of %d scans, imu_cov_growth sha256 %s" % (a, len(com), mx, IDENT_TOL_MM, ne, len(cs), "equal" if shaok else "DIFFERENT"))
    with open(os.path.join(out, "r86c_identity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "ref", "n_common_scans", "pos_max_abs_diff_mm", "n_scans_count_columns_differ", "n_common_short_rows", "imu_cov_growth_sha256_equal"]); w.writeheader(); w.writerows(irow)
    # PREIMU_SAME: acc_scale acts on the propagation, so the pre-IMU row of scan 1 must equal R86B's in every non-control arm; the three rows pre_imu, post_imu, post_lio must exist
    bad = []
    for (t, l), a in arms.items():
        if l == "a300": continue
        p = os.path.join(ret, "archive", a, "state_trace.csv.gz")
        if not os.path.isfile(p): bad.append(a + ": state_trace absent"); continue
        h, ph = pre_imu_hash(p)
        if h != ref[t]["pre_imu_row_sha256"]: bad.append("%s: pre_imu row differs from R86B (or absent)" % a)
        if ph != ["pre_imu", "post_imu", "post_lio"]: bad.append("%s: scan-1 phases %s" % (a, ph))
    (F if bad else P)("PREIMU_SAME", "scan-1 pre_imu row equals R86B's in all 12 non-control arms; phases pre_imu, post_imu, post_lio" if not bad else "; ".join(bad))
    # ACC_ENGAGED: Q_acc of the velocity block per scan, per tag, over the four levels (the IMU data are the same, so the order must hold scan by scan)
    erow = []; bad = []; infos = []
    for t in PFN:
        qs = {}
        ok_files = True
        for l, v in LEVELS:
            p = os.path.join(ret, "archive", arms[(t, l)], "imu_cov_growth.csv.gz")
            if not os.path.isfile(p): ok_files = False; break
            qs[l] = qv_by_scan(p)
        if not ok_files: bad.append(t + ": imu_cov_growth absent"); continue
        scans = sorted(qs["a300"])
        if any(sorted(qs[l]) != scans for l, _ in LEVELS): bad.append(t + ": scan sets differ between levels"); continue
        order = [l for l, _ in LEVELS]
        viol = 0
        for sc in scans:
            for hi, lo in zip(order[:-1], order[1:]):
                if qs[lo][sc] > qs[hi][sc] * (1.0 + MONO_REL): viol += 1
        engaged = sum(1 for sc in scans if qs["a300"][sc] > qs["a000"][sc] * 1.001)
        if viol: bad.append("%s: Q_acc not non-increasing from 0.3 to 0.0 in %d (scan, level pair) cases" % (t, viol))
        if engaged == 0: bad.append("%s: Q_acc(0.3) never exceeds Q_acc(0.0): the key had no effect" % t)
        mid = [sc for sc in scans if 11 <= sc <= 475]
        q0 = {sc: qs["a000"][sc] for sc in scans}
        for l, v in LEVELS:
            med = statistics.median(qs[l][sc] for sc in mid) if mid else float("nan")
            # quadratic law check: Q(level) - Q(0.0) = (level/0.3)^2 (Q(0.3) - Q(0.0)); worst relative deviation, scans where the 0.3 excess is above 1e-3 of Q(0.0)
            dev = 0.0
            for sc in scans:
                ex = qs["a300"][sc] - q0[sc]
                if ex > 1e-3 * q0[sc]:
                    dev = max(dev, abs((qs[l][sc] - q0[sc]) - (v / 0.3) ** 2 * ex) / ex)
            erow.append(dict(tag=t, level=v, arm=arms[(t, l)], n_scans=len(scans), median_Qv_scans_11_475=med, n_scans_excess_above_floor=engaged, max_rel_dev_from_quadratic_law=dev))
    with open(os.path.join(out, "r86c_acc_engagement.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "level", "arm", "n_scans", "median_Qv_scans_11_475", "n_scans_excess_above_floor", "max_rel_dev_from_quadratic_law"]); w.writeheader(); w.writerows(erow)
    (F if bad else P)("ACC_ENGAGED", "Q_acc (velocity block) is non-increasing from acc_scale 0.3 to 0.0 in every scan of every tag, and Q_acc(0.3) exceeds Q_acc(0.0) in some scans (table r86c_acc_engagement.csv)" if not bad else "; ".join(bad))
    if erow:
        I("ACC_LAW", "worst relative deviation from the quadratic law Q(level)-Q(0.0) = (level/0.3)^2 (Q(0.3)-Q(0.0)): %.3g (information only; a prediction of the planner, not a pass criterion)" % max(r["max_rel_dev_from_quadratic_law"] for r in erow))
    # SMALLSCANS
    sm = []; ss = {}
    for (t, l), a in arms.items():
        p = os.path.join(ret, "compact", a, "short_scan_all.csv.gz")
        if not os.path.isfile(p): sm.append(a + ": short_scan_all absent"); continue
        rr = [float(r["fs_n_residuals"]) for r in read_csv_gz(p) if r.get("fs_n_residuals") not in (None, "", "nan")]
        if not rr: sm.append(a + ": no residual counts"); continue
        med = sorted(rr)[len(rr) // 2]; low = sum(1 for x in rr if x < 0.5 * med); ss[a] = (len(rr), med)
        if low: sm.append("%s: %d of %d scans under half the median %.0f" % (a, low, len(rr), med))
    (F if sm else P)("SMALLSCANS", "no scan under half the arm's median residual count in any of the 16 arms" if not sm else "; ".join(sm))
    # STREAM
    eng = os.path.join(ret, "engagement_all.txt"); rows = []
    if not os.path.isfile(eng):
        M("STREAM", "engagement_all.txt absent")
    else:
        blocks = engagement_blocks(eng); ok = True; msgs = []
        for (t, l), a in arms.items():
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
        (P if ok else F)("STREAM", "second stream used in every frame of the m1r3 and m3r1 arms (8 of 16), fallbacks 0, direction as designed; primary arms have no stream line" if ok else "; ".join(msgs))
    with open(os.path.join(out, "r86c_stream_engagement.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "stream", "frames", "frames_second", "frames_fallback", "mean_points_map", "mean_points_iekf"]); w.writeheader(); w.writerows(rows)
    # DUMPS: chain / budget / first-frame files hold rows in the 12 non-control arms; the residual dump of the a000 arms holds contiguous iterations 0..n-1 equal in count to completed_iterations
    dm = []
    for (t, l), a in arms.items():
        if l == "a300": continue
        for fn in ("pose_control_first_frame.csv.gz", "pose_control_first_frame_state_chain.csv.gz", "pose_control_first_frame_covariance_budget.csv.gz", "residual_dump_state.csv.gz"):
            p = os.path.join(ret, "archive", a, fn)
            if not os.path.isfile(p): dm.append(a + "/" + fn + " absent"); continue
            if sum(1 for _ in rd_gz(p)) - 1 < 1: dm.append(a + "/" + fn + " has no data rows")
        if l == "a000":
            p = os.path.join(ret, "archive", a, "residual_dump.csv.gz")
            if not os.path.isfile(p): dm.append(a + "/residual_dump.csv.gz absent"); continue
            its = sorted({int(float(r["iter"])) for r in read_csv_gz(p)})
            st = [r for r in read_csv_gz(os.path.join(ret, "archive", a, "state_trace.csv.gz")) if r["phase"] == "post_lio" and r["scan_id"] == "1"]
            n = int(float(st[0]["completed_iterations"])) if st else -1
            if its != list(range(n)) or n < 1: dm.append("%s: residual_dump iterations %s, completed_iterations %d" % (a, its, n))
    (F if dm else P)("DUMPS", "frame-1 chain, budget, first-frame csv and residual_dump_state hold rows in the 12 non-control arms; residual_dump of the four a000 arms has iterations 0..completed_iterations-1" if not dm else "; ".join(dm))
    npass = sum(1 for r in R if r[1] == "PASS"); nf = sum(1 for r in R if r[1] == "FAIL"); nm = sum(1 for r in R if r[1] == "MISSING")
    with open(os.path.join(out, "r86c_gates.txt"), "w") as f:
        for g, v, t in R:
            f.write("%-8s %-14s %s\n" % (v, g, t))
        f.write("PASS %d  FAIL %d  MISSING %d\n" % (npass, nf, nm))
    print(open(os.path.join(out, "r86c_gates.txt")).read())
    return nf, nm


# ------------------------------------------------------------------ selftest fixture (synthetic; not evidence)
NSC = 600
ST_HDR = ["run_id", "scan_id", "t_abs", "t_rel_s", "phase", "dim", "px", "py", "pz", "completed_iterations"]


def _wgz(path, txt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(txt.encode())


def _csv(rows, hdr):
    b = io.StringIO(); w = csv.writer(b, lineterminator="\n"); w.writerow(hdr); w.writerows(rows); return b.getvalue()


def _qrow(sc, base, scale):
    # floor + (scale/0.3)^2 * excess, excess pattern depends on the scan
    ex = 0.0 if sc < 5 else 1e-6 * (1 + (sc % 7))
    q = base + (scale / 0.3) ** 2 * ex
    return [sc, 1.6e9 + sc * 0.1, sc * 0.1, 18] + [0.0] * 18 + [0.0] * 18 + [0, 0, 0, 0, 0, 0, q, q, q] + [0.0] * 9


def build_fixture(t, mut=()):
    shutil.rmtree(t, ignore_errors=True); os.makedirs(t)
    man = os.path.join(t, "man"); ret = os.path.join(t, "ret"); os.makedirs(man); os.makedirs(ret)
    ref = {"note": "synthetic", "arms": {}}
    qh = ["scan_index", "t_abs", "t_rel_s", "dim"] + ["Pstart_%d" % i for i in range(18)] + ["Pend_%d" % i for i in range(18)] + ["Qacc_%d" % i for i in range(18)]
    req = []
    pre = ["1.6e9", "0", "pre_imu", "18", "0.0", "0.0", "0.0", "0"]
    for tag in PFN:
        pos = [[s, 1.6e9 + s * 0.1, s * 0.1, 0.0, 0.0, 0.0, 0.0, 1, 0, 0, 0] for s in range(1, NSC + 1)]
        ph = ["scan_id", "t_abs", "t_rel_s", "px", "py", "pz", "yaw_rad", "qw", "qx", "qy", "qz"]
        sh = ["scan", "t_s", "residual_count", "fs_n_residuals", "fs_n_voxels", "fs_n_voxels_is_plane", "fs_iters", "fs_dx_pos_mm", "fs_nis", "fs_reduced_chi2", "pos_err_mm"]
        sr = [[s, s * 0.1, 2000, 2000, 500, 400, 5, 0, 0, 0, 0] for s in range(1, NSC + 1)]
        _wgz(os.path.join(man, "ref_r86b_%s_pos_trace_all.csv.gz" % tag), _csv(pos, ph)); _wgz(os.path.join(man, "ref_r86b_%s_short_scan_all.csv.gz" % tag), _csv(sr, sh))
        base = 5e-7 * (1 + 0.1 * list(PFN).index(tag))
        ctl = _csv([_qrow(s, base, 0.3) for s in range(1, NSC + 1)], qh)
        pre_row = ["x", "1", "1.6e9", "0.1", "pre_imu", "18", "0", "0", "0", "0"]
        ref["arms"][tag] = {"imu_cov_growth_sha256": hashlib.sha256(ctl.encode()).hexdigest(), "pre_imu_row_sha256": hashlib.sha256(",".join(pre_row[1:]).encode()).hexdigest()}
        for l, v in LEVELS:
            a = aid(tag, l); mp, rp = PFN[tag]
            pp = [list(r) for r in pos]; ss = [list(r) for r in sr]
            if "shift" in mut and l == "a300" and tag == "m3r3": pp[100][3] += 0.0005
            if "smallscan" in mut and l == "a100" and tag == "m1r1": ss[50][3] = 100
            _wgz(os.path.join(ret, "compact", a, "pos_trace_all.csv.gz"), _csv(pp, ph)); _wgz(os.path.join(ret, "compact", a, "short_scan_all.csv.gz"), _csv(ss, sh))
            q = _csv([_qrow(s, base, v) for s in range(1, NSC + 1)], qh)
            if "badq" in mut and l == "a030" and tag == "m1r3":
                q = _csv([_qrow(s, base, 0.6) for s in range(1, NSC + 1)], qh)          # larger Q at a lower level
            if "inert" in mut and l == "a000" and tag == "m3r1":
                q = _csv([_qrow(s, base, 0.3) for s in range(1, NSC + 1)], qh)          # key had no effect: 0.0 equals 0.3
                # (then a030 > a000 would also break the order; keep the other levels equal too)
            if "cov" in mut and l == "a300" and tag == "m3r3":
                q = q.replace("\n2,", "\n2,", 1) + "extra\n"
            _wgz(os.path.join(ret, "archive", a, "imu_cov_growth.csv.gz"), q)
            _wgz(os.path.join(ret, "archive", a, "exit_status.txt.gz"), "exit_code=%d signal=none\n" % (1 if ("exit" in mut and l == "a000" and tag == "m3r3") else 0))
            _wgz(os.path.join(ret, "archive", a, "engagement_imu.txt.gz"), "x\n")
            if l != "a300":
                srows = [pre_row if not ("pre" in mut and l == "a100" and tag == "m3r3") else ["x", "1", "1.6e9", "0.1", "pre_imu", "18", "9", "0", "0", "0"]]
                srows += [["x", "1", "1.6e9", "0.1", "post_imu", "18", "0", "0", "0", "0"], ["x", "1", "1.6e9", "0.1", "post_lio", "18", "0", "0", "0", "5"]]
                if "phases" in mut and l == "a030" and tag == "m3r3": srows = srows[1:]
                _wgz(os.path.join(ret, "archive", a, "state_trace.csv.gz"), _csv(srows, ST_HDR))
                for fn in ("pose_control_first_frame.csv", "pose_control_first_frame_state_chain.csv", "pose_control_first_frame_covariance_budget.csv", "residual_dump_state.csv"):
                    _wgz(os.path.join(ret, "archive", a, fn + ".gz"), "h\n1\n" if not ("nochain" in mut and fn.startswith("pose_control_first_frame_state") and l == "a100" and tag == "m3r3") else "h\n")
                _wgz(os.path.join(ret, "archive", a, "pose_control_first_frame_covariance_budget.txt.gz"), "txt\n")
            if l == "a000":
                its = range(5) if not ("iters" in mut and tag == "m3r3") else [0, 1, 2, 4]
                _wgz(os.path.join(ret, "archive", a, "residual_dump.csv.gz"), _csv([[i, 0.0] for i in its], ["iter", "r"]))
        # sanity: ctl is exactly what the a300 arm wrote
    for tag in PFN:
        pass
    json.dump(ref, open(os.path.join(man, "ref_r86b_digests.json"), "w"))
    # engagement
    blk = []
    for (tag, l), a in [((tg, lv), aid(tg, lv)) for lv, _ in LEVELS for tg in PFN]:
        mp, rp = PFN[tag]; blk.append("### " + a); blk.append("[engagement] synthetic")
        if mp != rp:
            fs = 595 if not ("stream" in mut and tag == "m1r3" and l == "a100") else 590
            blk.append("lio/iekf_point_stream=second dry_run_point_filter_num=%d frames=595 frames_second=%d frames_fallback=0 mean_points_map=%s mean_points_iekf=%s" % (rp, fs, "6000" if mp < rp else "2000", "2000" if mp < rp else "6000"))
    open(os.path.join(ret, "engagement_all.txt"), "w").write("\n".join(blk) + "\n")
    # required (only to exercise REQUIRED)
    reqs = [dict(file="pos_trace_all.csv.gz", where="return", source="arm", arms=[aid(tg, lv) for lv, _ in LEVELS for tg in PFN])]
    if "reqmiss" in mut: reqs.append(dict(file="state_reference.txt", where="return", source="arm", arms=[aid("m3r3", "a300")]))
    json.dump(reqs, open(os.path.join(man, "required_outputs.json"), "w"))
    return ret, man


def selftest():
    t = tempfile.mkdtemp(prefix="r86c_gates_")
    fails = []
    def go(mut):
        ret, man = build_fixture(os.path.join(t, "fx"), mut)
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            nf, nm = run(ret, man, ret)
        v = {}
        for ln in open(os.path.join(ret, "r86c_gates.txt")):
            p = ln.split(None, 2)
            if len(p) >= 2 and p[0] in ("PASS", "FAIL", "MISSING", "INFO"): v[p[1]] = p[0]
        return v
    base = go(())
    for g, x in base.items():
        if x not in ("PASS", "INFO"): fails.append("clean fixture: %s is %s" % (g, x))
    want = {"IDENT_R86B_m3r3": "shift", "ACC_ENGAGED": "badq", "PREIMU_SAME": "pre", "EXIT": "exit", "SMALLSCANS": "smallscan", "STREAM": "stream", "DUMPS": "iters", "REQUIRED": "reqmiss"}
    for g, m in want.items():
        v = go((m,))
        if v.get(g) != "FAIL": fails.append("mutation %s did not turn %s (got %s)" % (m, g, v.get(g)))
        if base.get(g) not in ("PASS",): fails.append("clean state of %s is not PASS" % g)
    v = go(("inert",))
    if v.get("ACC_ENGAGED") != "FAIL": fails.append("inert key not caught")
    v = go(("phases",))
    if v.get("PREIMU_SAME") != "FAIL": fails.append("missing pre_imu phase not caught")
    v = go(("nochain",))
    if v.get("DUMPS") != "FAIL": fails.append("empty chain not caught")
    shutil.rmtree(t, ignore_errors=True)
    print("SELFTEST r86c_gates", ("FAIL " + "; ".join(fails)) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ret-dir"); ap.add_argument("--manifest", default="r86c_manifest"); ap.add_argument("--out", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    nf, nm = run(a.ret_dir, a.manifest, a.out or a.ret_dir)
    sys.exit(1 if nf or nm else 0)
