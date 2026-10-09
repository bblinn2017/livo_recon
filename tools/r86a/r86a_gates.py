#!/usr/bin/env python3
"""r86a_gates.py -- R86A gates (planner's script, selftest on a synthetic folder only; not evidence, rule 1.1). Reads a collected return folder; writes r86a_gates.txt,
r86a_controls.csv, r86a_stream_engagement.csv, r86a_trim_first_frames.csv. Verdicts: PASS, FAIL, MISSING (never a pass), INFO. Facts only; no interpretation.
Usage: python3 tools/r86a/r86a_gates.py --ret-dir R86A_RETURN --manifest r86a_manifest [--out R86A_RETURN]
       python3 tools/r86a/r86a_gates.py --selftest
"""
import argparse, csv, gzip, io, json, os, re, sys, tempfile

TOL_MM = 0.005
PFN = {"m3r3": (3, 3), "m1r1": (1, 1), "m1r3": (1, 3), "m3r1": (3, 1)}
REFS = {"m3r3": "A", "m1r1": "B"}


def aid(tag):
    return "r86a_rt_sp_eee_01_v0p25_" + tag


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
            ps = [os.path.join(ret, r["file"])] if r["file"] not in ("r86a_gates.txt", "r86a_controls.csv", "r86a_stream_engagement.csv", "r86a_trim_first_frames.csv", "r86a_dry_run_stream.csv") else []
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
        # tools/r86/r86a_gates.py's EXIT row this round's predecessor had -- confirmed R86 and
        # R86A both: all arms read "exit_code=0 signal=none"), not a bare "0".
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
            # bare "INERT" also matches the always-present disclaimer footer every engagement.txt
            # ends with, not just a real per-flag "<-- INERT: ..." marker -- same bug/fix as
            # tools/r86/r86_gates.py's STREAM row this round's predecessor had.
            if st != "second" or dn != rp or fr <= 0 or fs != fr or fb != 0 or "<-- INERT" in txt or "<-- PARTLY" in txt:
                ok = False; msgs.append("%s: stream=%s dry_run=%d frames=%d second=%d fallback=%d" % (a, st, dn, fr, fs, fb))
            elif (mp < rp and not pi < pm) or (mp > rp and not pi > pm):
                ok = False; msgs.append("%s: direction wrong (map %.1f, iekf %.1f points)" % (a, pm, pi))
        (P if ok else F)("STREAM", "second stream used in every frame of m1r3 and m3r1, fallbacks 0, direction as designed; primary arms have no stream line" if ok else "; ".join(msgs))
    with open(os.path.join(out, "r86a_stream_engagement.csv"), "w", newline="") as f:
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
    with open(os.path.join(out, "r86a_controls.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "ref_arm", "n_common_scans", "pos_max_abs_diff_mm", "n_scans_count_columns_differ", "n_common_short_rows"]); w.writeheader(); w.writerows(crow)
    # DUMPS present with content (rows only; the analysis is the planner's)
    dm = []
    for a in arms.values():
        for fn in ("residual_dump.csv.gz", "post_calibration_voxelmap.csv.gz", "pose_control_first_frame_solve_matrices.txt.gz"):
            p = os.path.join(ret, "archive", a, fn)
            if not os.path.isfile(p): dm.append(a + "/" + fn + " absent"); continue
            with rd_gz(p) as f:
                n = sum(1 for _ in f) - 1
            if n < 1: dm.append("%s/%s has %d data rows" % (a, fn, n))
        p = os.path.join(ret, "archive", a, "residual_dump.csv.gz")
        if os.path.isfile(p):
            hd = rd_gz(p).readline()
            if "voxel_point_count" not in hd or "voxel_distinct_frames" not in hd: dm.append(a + ": residual_dump header lacks the voxel columns")
    (F if dm else P)("DUMPS", "residual dump (with the voxel columns), post-calibration snapshot and frame-1 solve matrices hold data rows in all four arms" if not dm else "; ".join(dm))
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
    # DRYSTREAM: [dry_run_stream] counters (new in R86A) from dry_run_stream_lines.txt; split arms must show trimming engaged and no rejected message
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
    with open(os.path.join(out, "r86a_dry_run_stream.csv"), "w", newline="") as f:
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
    with open(os.path.join(out, "r86a_trim_first_frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "msg_index", "seq", "n_kept", "verdict", "n_trimmed"]); w.writeheader(); w.writerows(trows)
    (I if tot else M)("TRIM", "points trimmed by retime_trim in the first 15 messages (primary stream only; the second stream is pushed before the trim, SUSPECTED): " + (", ".join("%s %d" % (k, v) for k, v in tot.items()) if tot else "lidar_msg_log absent"))
    npass = sum(1 for r in R if r[1] == "PASS"); nf = sum(1 for r in R if r[1] == "FAIL"); nm = sum(1 for r in R if r[1] == "MISSING")
    with open(os.path.join(out, "r86a_gates.txt"), "w") as f:
        for g, v, t in R:
            f.write("%-8s %-14s %s\n" % (v, g, t))
        f.write("PASS %d  FAIL %d  MISSING %d\n" % (npass, nf, nm))
    print(open(os.path.join(out, "r86a_gates.txt")).read())
    return nf, nm


def selftest():
    import shutil
    here = os.path.dirname(os.path.abspath(__file__)); man = os.path.join(here, "..", "..", "r86a_manifest")
    t = tempfile.mkdtemp(prefix="r86a_gates_"); ret = os.path.join(t, "ret"); os.makedirs(ret)
    ck = []
    req = json.load(open(os.path.join(man, "required_outputs.json"))) if os.path.isfile(os.path.join(man, "required_outputs.json")) else []
    blocks = []
    for tag, (mp, rp) in PFN.items():
        a = aid(tag); ad = os.path.join(ret, "archive", a); cd = os.path.join(ret, "compact", a); os.makedirs(ad); os.makedirs(cd)
        for fn in ("residual_dump.csv.gz", "post_calibration_voxelmap.csv.gz", "pose_control_first_frame_solve_matrices.txt.gz"):
            with gzip.open(os.path.join(ad, fn), "wt") as f:
                f.write("frame,voxel_point_count,voxel_distinct_frames\n1,3,1\n" if "residual" in fn else "a,b\n1,2\n")
        with gzip.open(os.path.join(ad, "exit_status.txt.gz"), "wt") as f: f.write("0\n")
        with gzip.open(os.path.join(ad, "lidar_msg_log.csv.gz"), "wt") as f: f.write("msg_index,seq,n_kept,verdict,n_trimmed\n" + "".join("%d,%d,4000,0,%d\n" % (i, 119 + i, 2 if i > 6 else 0) for i in range(20)))
        if tag in REFS:
            c = REFS[tag]
            for fn in ("pos_trace_all.csv.gz", "short_scan_all.csv.gz"):
                shutil.copy(os.path.join(man, "ref_r85_%s_%s" % (c, fn)), os.path.join(cd, fn))
        else:
            for fn in ("pos_trace_all.csv.gz", "short_scan_all.csv.gz"):
                shutil.copy(os.path.join(man, "ref_r85_A_" + fn), os.path.join(cd, fn))
        if mp == rp: blocks.append("### %s\n--- engagement.txt\n[engagement]\n" % a)
        else:
            pm, pi = (12000.0, 4000.0) if mp < rp else (4000.0, 12000.0)
            blocks.append("### %s\n--- engagement.txt\n  lio/iekf_point_stream=second dry_run_point_filter_num=%d frames=595 frames_second=595 frames_fallback=0 mean_points_map=%s mean_points_iekf=%s\n" % (a, rp, pm, pi))
    open(os.path.join(ret, "engagement_all.txt"), "w").write("\n".join(blocks))
    dl = []
    for tag, (mp, rp) in PFN.items():
        if mp != rp: dl.append("### %s\n[ INFO] [1.0]: [dry_run_stream] msgs=600 trimmed_msgs=58 trimmed_points=300 pushed_msgs=600 rejected_msgs=0\n" % aid(tag))
    open(os.path.join(ret, "dry_run_stream_lines.txt"), "w").write("\n".join(dl))
    nf, nm = run(ret, man, ret) if req else (0, 0)
    g = open(os.path.join(ret, "r86a_gates.txt")).read()
    ck.append(("CONTROL m3r3 passes on the shipped R85 A trace", "PASS     CONTROL_m3r3" in g))
    ck.append(("STREAM pass", "PASS     STREAM" in g))
    ck.append(("DUMPS pass", "PASS     DUMPS" in g))
    ck.append(("SMALLSCANS pass on normal tables", "PASS     SMALLSCANS" in g))
    ck.append(("DRYSTREAM pass on clean counters", "PASS     DRYSTREAM" in g))
    # the R86 signature: a table with a tiny residual set, and rejected messages, must FAIL
    p3 = os.path.join(ret, "compact", aid("m1r3"), "short_scan_all.csv.gz")
    rows = read_csv_gz(p3); rows[4]["fs_n_residuals"] = "53"
    with gzip.open(p3, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    open(os.path.join(ret, "dry_run_stream_lines.txt"), "w").write(open(os.path.join(ret, "dry_run_stream_lines.txt")).read().replace("rejected_msgs=0", "rejected_msgs=40", 1))
    run(ret, man, ret); g2 = open(os.path.join(ret, "r86a_gates.txt")).read()
    ck.append(("a scan with 53 residuals fails SMALLSCANS", "FAIL     SMALLSCANS" in g2))
    ck.append(("rejected messages fail DRYSTREAM", "FAIL     DRYSTREAM" in g2))
    # break the direction of m3r1 -> STREAM must FAIL
    txt = open(os.path.join(ret, "engagement_all.txt")).read().replace("mean_points_map=4000.0 mean_points_iekf=12000.0", "mean_points_map=12000.0 mean_points_iekf=4000.0")
    open(os.path.join(ret, "engagement_all.txt"), "w").write(txt); run(ret, man, ret)
    ck.append(("wrong direction fails STREAM", "FAIL     STREAM" in open(os.path.join(ret, "r86a_gates.txt")).read()))
    bad = [n for n, ok in ck if not ok]
    print("SELFTEST r86a_gates", ("FAIL " + "; ".join(bad)) if bad else "PASS", "checks: %d" % len(ck))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ret-dir"); ap.add_argument("--manifest", default="r86a_manifest"); ap.add_argument("--out", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    nf, nm = run(a.ret_dir, a.manifest, a.out or a.ret_dir)
    sys.exit(1 if nf else 0)
