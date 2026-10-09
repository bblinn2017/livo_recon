#!/usr/bin/env python3
"""r88_gates.py -- R88 gates (planner's script, selftest on a synthetic folder only; not evidence, rule 1.1). Reads a collected return folder; writes r88_gates.txt, r88_binning.csv, r88_stream_engagement.csv.
Verdicts: PASS, FAIL, MISSING (never a pass), INFO. Facts only; no interpretation.
Usage: python3 tools/r88/r88_gates.py --ret-dir R88_RETURN --manifest r88_manifest [--out R88_RETURN]
       python3 tools/r88/r88_gates.py --selftest
"""
import argparse, csv, gzip, hashlib, io, json, os, re, shutil, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MIN_BINS = 5
PFN = {"m3r3": (3, 3), "m1r1": (1, 1), "m1r3": (1, 3), "m3r1": (3, 1)}
MAPREF = {"m3r3": "mpf3", "m3r1": "mpf3", "m1r1": "mpf1", "m1r3": "mpf1"}
VARIANTS = {"ac": ("average", "count"), "au": ("average", "uniform"), "sc": ("shrunk", "count"), "su": ("shrunk", "uniform")}
CORNERS = ("ac", "su")
C = "r88_rt_sp_eee_01_v0p25_"


def aid(tag, v):
    return "%s%s_%s" % (C, tag, v)


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


def map_table(path):
    """{(root_key_x, root_key_y, root_key_z, layer): (is_plane, point_count)} of every voxel row."""
    out = {}
    with rd_gz(path) as f:
        r = csv.reader(f); hdr = next(r)
        ix = [hdr.index(c) for c in ("root_key_x", "root_key_y", "root_key_z", "layer")]
        ipl, ipc = hdr.index("is_plane"), hdr.index("point_count")
        for row in r:
            out[tuple(row[i] for i in ix)] = (row[ipl] == "1", int(float(row[ipc])))
    return out


ENG = re.compile(r"voxel_map/backend = binned: bin_size_fraction = ([\d.eE+-]+) cov_mode = (\w+) weight_mode = (\w+) binned_points=(\d+) binned_new_bins=(\d+) binned_fits=(\d+)")


def run(ret, man, out):
    R = []
    P = lambda g, t: R.append((g, "PASS", t)); F = lambda g, t: R.append((g, "FAIL", t)); M = lambda g, t: R.append((g, "MISSING", t)); I = lambda g, t: R.append((g, "INFO", t))
    arms = {(t, v): aid(t, v) for v in VARIANTS for t in PFN}
    ref = json.load(open(os.path.join(man, "ref_r86b_digests.json")))["arms"]
    r87 = json.load(open(os.path.join(man, "ref_r87_n55_iter0.json")))["n55_iter0_rows"]
    req = json.load(open(os.path.join(man, "required_outputs.json")))
    gen = ("r88_gates.txt", "r88_binning.csv", "r88_stream_engagement.csv")
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
    (F if bad else P)("EXIT", "exit status 0 for all 16 arms" if not bad else "non-zero or absent: " + ", ".join(bad))
    # BINNED_ENGAGED: the binned branch ran with the modes the arm names (engagement block of every arm)
    eng = os.path.join(ret, "engagement_all.txt"); brow = []
    if not os.path.isfile(eng):
        M("BINNED_ENGAGED", "engagement_all.txt absent")
        blocks = {}
    else:
        blocks = engagement_blocks(eng); msgs = []
        for (t, v), a in arms.items():
            txt = "\n".join(blocks.get(a, [])); m = ENG.search(txt)
            if not txt: msgs.append(a + ": no engagement block"); continue
            if not m: msgs.append(a + ": no binned engagement line"); continue
            cm, wm = VARIANTS[v]; fr, c, w, bp, bb, bf = float(m.group(1)), m.group(2), m.group(3), int(m.group(4)), int(m.group(5)), int(m.group(6))
            brow.append(dict(arm=a, cov_mode=c, weight_mode=w, binned_points=bp, binned_new_bins=bb, binned_fits=bf))
            if (c, w) != (cm, wm): msgs.append("%s: modes %s/%s, expected %s/%s" % (a, c, w, cm, wm))
            elif fr != 0.2: msgs.append("%s: bin_size_fraction %s" % (a, m.group(1)))
            elif bp <= 0 or bb <= 0 or bf <= 0 or not bb < bp: msgs.append("%s: points %d, bins opened %d, fits %d (bins must be fewer than points)" % (a, bp, bb, bf))
            elif "<-- INERT" in txt: msgs.append(a + ": INERT flag")
        (F if msgs else P)("BINNED_ENGAGED", "binned branch ran in all 16 arms with the modes the arm names; bins opened fewer than points; fits made" if not msgs else "; ".join(msgs))
    with open(os.path.join(out, "r88_binning.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "cov_mode", "weight_mode", "binned_points", "binned_new_bins", "binned_fits"]); w.writeheader(); w.writerows(brow)
    # MAP_BINS per tag: the post-calibration map of the four settings against the R86B (raw point) map
    for t in PFN:
        paths = {v: os.path.join(ret, "archive", arms[(t, v)], "post_calibration_voxelmap.csv.gz") for v in VARIANTS}
        if not all(os.path.isfile(p) for p in paths.values()): M("MAP_BINS_" + t, "map absent for " + ", ".join(v for v, p in paths.items() if not os.path.isfile(p))); continue
        rawm = map_table(os.path.join(man, "ref_r86b_map_%s.csv.gz" % MAPREF[t]))
        tabs = {v: map_table(p) for v, p in paths.items()}; msgs = []; info = []
        base = tabs["ac"]
        for v, tb in tabs.items():
            if set(tb) != set(rawm): msgs.append("%s: %d voxels, R86B has %d (%d differ)" % (v, len(tb), len(rawm), len(set(tb) ^ set(rawm)))); continue
            gt = sum(1 for k, (pl, c) in tb.items() if c > rawm[k][1]); lt = sum(1 for k, (pl, c) in tb.items() if c < rawm[k][1])
            lowp = sum(1 for k, (pl, c) in tb.items() if pl and c < MIN_BINS)
            nplane = sum(1 for k, (pl, c) in tb.items() if pl)
            info.append("%s plane voxels %d" % (v, nplane))
            if gt: msgs.append("%s: %d voxels with more bins than R86B raw points" % (v, gt))
            if lt == 0: msgs.append("%s: no voxel has fewer bins than raw points (binning inert)" % v)
            if lowp: msgs.append("%s: %d plane voxels with fewer than %d bins" % (v, lowp, MIN_BINS))
            if {k: c for k, (pl, c) in tb.items()} != {k: c for k, (pl, c) in base.items()}: msgs.append("%s: bin counts differ from setting ac" % v)
        nlt = sum(1 for k, (pl, c) in base.items() if k in rawm and c < rawm[k][1])
        (F if msgs else P)("MAP_BINS_" + t, ("%d voxels, bins never above and in %d voxels below the R86B raw point count; every plane voxel has %d or more bins; bin counts equal across the four settings; " % (len(base), nlt, MIN_BINS) + "; ".join(info)) if not msgs else "; ".join(msgs))
    # MINBINS + ITER0 (settings ac and su: residual dump)
    bad_m, info = [], []
    for t in PFN:
        for v in CORNERS:
            a = arms[(t, v)]; p = os.path.join(ret, "archive", a, "residual_dump.csv.gz")
            if not os.path.isfile(p): bad_m.append(a + ": residual_dump absent"); continue
            rows = [r for r in read_csv_gz(p) if r["frame"] == "1"]
            cnt0 = sum(1 for r in rows if r["iter"] == "0"); low = sum(1 for r in rows if int(float(r["voxel_point_count"])) < MIN_BINS)
            info.append("%s %s iteration-0 rows %d (R86B %d, R87 n55 %d)" % (t, v, cnt0, ref[t]["resid"]["0"][0], r87[t]))
            if low: bad_m.append("%s: %d residual rows from voxels with fewer than %d bins" % (a, low, MIN_BINS))
    (F if bad_m else P)("MINBINS", "no frame-1 residual row (any iteration) comes from a voxel with fewer than %d bins in the eight ac and su arms" % MIN_BINS if not bad_m else "; ".join(bad_m))
    I("ITER0_ROWS", "; ".join(info))
    # PREIMU_SAME (ac and su)
    bad = []
    for t in PFN:
        for v in CORNERS:
            a = arms[(t, v)]; p = os.path.join(ret, "archive", a, "state_trace.csv.gz")
            if not os.path.isfile(p): bad.append(a + ": state_trace absent"); continue
            h, ph = pre_imu_hash(p)
            if h != ref[t]["pre_imu_row_sha256"]: bad.append("%s: pre_imu row differs from R86B (or absent)" % a)
            if ph != ["pre_imu", "post_imu", "post_lio"]: bad.append("%s: scan-1 phases %s" % (a, ph))
    (F if bad else P)("PREIMU_SAME", "scan-1 pre_imu row equals R86B's in the eight ac and su arms; phases pre_imu, post_imu, post_lio" if not bad else "; ".join(bad))
    # SMALLSCANS: fails only on a scan with no residuals; the counts under half the median are listed (fewer admitted voxels make small scans by design)
    zero = []; lst = []
    for (t, v), a in arms.items():
        p = os.path.join(ret, "compact", a, "short_scan_all.csv.gz")
        if not os.path.isfile(p): zero.append(a + ": short_scan_all absent"); continue
        rr = [(int(float(r["scan"])), float(r["fs_n_residuals"])) for r in read_csv_gz(p) if r.get("fs_n_residuals") not in (None, "", "nan")]
        if not rr: zero.append(a + ": no residual counts"); continue
        med = sorted(x for _, x in rr)[len(rr) // 2]
        zero += ["%s scan %d has no residuals" % (a, sc) for sc, x in rr if x <= 0]
        lst.append("%s up to 475: %d, after 475: %d (median %.0f)" % (a, sum(1 for sc, x in rr if sc <= 475 and x < 0.5 * med), sum(1 for sc, x in rr if sc > 475 and x < 0.5 * med), med))
    (F if zero else P)("SMALLSCANS", "no scan without residuals in any of the 16 arms" if not zero else "; ".join(zero))
    I("SMALLSCANS_COUNTS", "scans under half the arm's median residual count: " + "; ".join(lst))
    # STREAM
    srows = []
    if not os.path.isfile(eng):
        M("STREAM", "engagement_all.txt absent")
    else:
        ok = True; msgs = []
        for (t, v), a in arms.items():
            txt = "\n".join(blocks.get(a, [])); mp, rp = PFN[t]
            m = re.search(r"lio/iekf_point_stream=(\w+) dry_run_point_filter_num=(\d+) frames=(\d+) frames_second=(\d+) frames_fallback=(\d+) mean_points_map=([\d.eE+-]+) mean_points_iekf=([\d.eE+-]+)", txt)
            if not txt: ok = False; msgs.append(a + ": no engagement block"); continue
            if mp == rp:
                if m: ok = False; msgs.append(a + ": stream line present although stream is primary")
                srows.append(dict(arm=a, stream="primary", frames="", frames_second="", frames_fallback="", mean_points_map="", mean_points_iekf="")); continue
            if not m: ok = False; msgs.append(a + ": no iekf_point_stream engagement line"); continue
            st, dn, fr, fs, fb, pm, pi = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), float(m.group(6)), float(m.group(7))
            srows.append(dict(arm=a, stream=st, frames=fr, frames_second=fs, frames_fallback=fb, mean_points_map=pm, mean_points_iekf=pi))
            if st != "second" or dn != rp or fr <= 0 or fs != fr or fb != 0 or "<-- INERT" in txt or "<-- PARTLY" in txt:
                ok = False; msgs.append("%s: stream=%s dry_run=%d frames=%d second=%d fallback=%d" % (a, st, dn, fr, fs, fb))
            elif (mp < rp and not pi < pm) or (mp > rp and not pi > pm):
                ok = False; msgs.append("%s: direction wrong (map %.1f, iekf %.1f points)" % (a, pm, pi))
        (P if ok else F)("STREAM", "second stream used in every frame of the m1r3 and m3r1 arms (8 of 16), fallbacks 0, direction as designed; primary arms have no stream line" if ok else "; ".join(msgs))
    with open(os.path.join(out, "r88_stream_engagement.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "stream", "frames", "frames_second", "frames_fallback", "mean_points_map", "mean_points_iekf"]); w.writeheader(); w.writerows(srows)
    # DUMPS (ac and su): residual dump iterations 0..completed_iterations-1
    dm = []
    for t in PFN:
        for v in CORNERS:
            a = arms[(t, v)]; p = os.path.join(ret, "archive", a, "residual_dump.csv.gz"); sp = os.path.join(ret, "archive", a, "state_trace.csv.gz")
            if not (os.path.isfile(p) and os.path.isfile(sp)): dm.append(a + ": residual_dump or state_trace absent"); continue
            its = sorted({int(float(r["iter"])) for r in read_csv_gz(p)})
            st = [r for r in read_csv_gz(sp) if r["phase"] == "post_lio" and r["scan_id"] == "1"]
            n = int(float(st[0]["completed_iterations"])) if st else -1
            if its != list(range(n)) or n < 1: dm.append("%s: residual_dump iterations %s, completed_iterations %d" % (a, its, n))
    (F if dm else P)("DUMPS", "residual dump has iterations 0..completed_iterations-1 in the eight ac and su arms" if not dm else "; ".join(dm))
    npass = sum(1 for r in R if r[1] == "PASS"); nf = sum(1 for r in R if r[1] == "FAIL"); nm = sum(1 for r in R if r[1] == "MISSING")
    with open(os.path.join(out, "r88_gates.txt"), "w") as f:
        for g, v, t in R:
            f.write("%-8s %-18s %s\n" % (v, g, t))
        f.write("PASS %d  FAIL %d  MISSING %d\n" % (npass, nf, nm))
    print(open(os.path.join(out, "r88_gates.txt")).read())
    return nf, nm


# ------------------------------------------------------------------ selftest fixture (synthetic; not evidence)
NSC = 600
MAP_HDR = ["snapshot_frame", "node_id", "root_key_x", "root_key_y", "root_key_z", "layer", "is_plane", "point_count"]
RES_HDR = ["frame", "iter", "idx", "node_id", "r", "voxel_point_count"]


def _wgz(path, txt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(txt.encode())


def _csv(rows, hdr):
    b = io.StringIO(); w = csv.writer(b, lineterminator="\n"); w.writerow(hdr); w.writerows(rows); return b.getvalue()


def _resid(n, iters=5, vpc=7):
    return _csv([[1, it, i, i, "0.001", vpc] for it in range(iters) for i in range(n)], RES_HDR)


def _raw_map(nvox):
    # voxel k: raw point_count 3 for k % 3 == 0, else 7 + k % 5; plane when raw >= 3 (the R86B map flags the 3-point voxels as planes)
    return [[0, k, k, 0, 0, 0, 1, 3 if k % 3 == 0 else 7 + k % 5] for k in range(nvox)]


def _bins_of(raw):
    return 1 if raw == 3 else raw - 1     # bins <= raw, fewer in every voxel


def build_fixture(t, mut=()):
    shutil.rmtree(t, ignore_errors=True); os.makedirs(t)
    man = os.path.join(t, "man"); ret = os.path.join(t, "ret"); os.makedirs(man); os.makedirs(ret)
    sh = ["scan", "t_s", "residual_count", "fs_n_residuals", "fs_n_voxels", "fs_n_voxels_is_plane", "fs_iters", "fs_dx_pos_mm", "fs_nis", "fs_reduced_chi2", "pos_err_mm"]
    STH = ["run_id", "scan_id", "t_abs", "t_rel_s", "phase", "dim", "px", "py", "pz", "completed_iterations"]
    pre_row = ["x", "1", "1.6e9", "0.1", "pre_imu", "18", "0", "0", "0", "0"]
    refmaps = {"mpf3": _raw_map(30), "mpf1": _raw_map(40)}
    for k, rows in refmaps.items():
        _wgz(os.path.join(man, "ref_r86b_map_%s.csv.gz" % k), _csv(rows, MAP_HDR))
    ref = {"arms": {tag: {"pre_imu_row_sha256": hashlib.sha256(",".join(pre_row[1:]).encode()).hexdigest(), "resid": {"0": [3000, "x"]}} for tag in PFN}}
    json.dump(ref, open(os.path.join(man, "ref_r86b_digests.json"), "w"))
    json.dump({"n55_iter0_rows": {tag: 1500 for tag in PFN}}, open(os.path.join(man, "ref_r87_n55_iter0.json"), "w"))
    for tag in PFN:
        n_ref = len(refmaps[MAPREF[tag]])
        for v in VARIANTS:
            a = aid(tag, v)
            ss = [[s, s * 0.1, 2000, 2000, 500, 400, 5, 0, 0, 0, 0] for s in range(1, NSC + 1)]
            if "zeroscan" in mut and tag == "m1r1" and v == "au": ss[50][3] = 0
            _wgz(os.path.join(ret, "compact", a, "short_scan_all.csv.gz"), _csv(ss, sh))
            _wgz(os.path.join(ret, "archive", a, "exit_status.txt.gz"), "exit_code=%d signal=none\n" % (1 if ("exit" in mut and tag == "m3r3" and v == "sc") else 0))
            _wgz(os.path.join(ret, "archive", a, "engagement_imu.txt.gz"), "x\n")
            mrows = []
            for r in _raw_map(n_ref):
                r = list(r); raw = r[7]; b = _bins_of(raw)
                if "equalbins" in mut and tag == "m3r1": b = raw
                r[7] = b; r[6] = 1 if b >= 5 else 0
                mrows.append(r)
            if "binsgtraw" in mut and tag == "m1r1" and v == "ac": mrows[4][7] += 5
            if "planelow" in mut and tag == "m3r3" and v == "ac": mrows[0][6] = 1
            if "varmapdiff" in mut and tag == "m1r3" and v == "su": mrows[2][7] -= 1
            if "keyset" in mut and tag == "m3r1" and v == "sc": mrows.append([0, 999, 999, 0, 0, 0, 0, 2])
            _wgz(os.path.join(ret, "archive", a, "post_calibration_voxelmap.csv.gz"), _csv(mrows, MAP_HDR))
            if v in CORNERS:
                rt = _resid(6, 5 if not ("iters" in mut and tag == "m3r3" and v == "ac") else 4)
                if "lowrow" in mut and tag == "m1r1" and v == "su": rt = rt.replace(",0.001,7", ",0.001,3", 1)
                _wgz(os.path.join(ret, "archive", a, "residual_dump.csv.gz"), rt)
                srows = [pre_row if not ("pre" in mut and tag == "m3r3" and v == "su") else ["x", "1", "1.6e9", "0.1", "pre_imu", "18", "9", "0", "0", "0"],
                         ["x", "1", "1.6e9", "0.1", "post_imu", "18", "0", "0", "0", "0"], ["x", "1", "1.6e9", "0.1", "post_lio", "18", "0", "0", "0", "5"]]
                _wgz(os.path.join(ret, "archive", a, "state_trace.csv.gz"), _csv(srows, STH))
    blk = []
    for tag in PFN:
        for v in VARIANTS:
            a = aid(tag, v); mp, rp = PFN[tag]; cm, wm = VARIANTS[v]
            blk.append("### " + a); blk.append("[engagement] synthetic")
            bp, bb, bf = 90000, 30000, 4000
            if "bbeqbp" in mut and tag == "m1r1" and v == "ac": bb = bp
            if "engmode" in mut and tag == "m3r3" and v == "au": wm = "count"
            line = "  voxel_map/backend = binned: bin_size_fraction = 0.2 cov_mode = %s weight_mode = %s binned_points=%d binned_new_bins=%d binned_fits=%d" % (cm, wm, bp, bb, bf)
            if "inert" in mut and tag == "m1r3" and v == "sc": line += "   <-- INERT: the binned map was selected and no fit was made from bins"
            if "nobinned" in mut and tag == "m3r1" and v == "au": line = ""
            if line: blk.append(line)
            if mp != rp:
                fs = 595 if not ("stream" in mut and tag == "m1r3" and v == "ac") else 590
                blk.append("lio/iekf_point_stream=second dry_run_point_filter_num=%d frames=595 frames_second=%d frames_fallback=0 mean_points_map=%s mean_points_iekf=%s" % (rp, fs, "6000" if mp < rp else "2000", "2000" if mp < rp else "6000"))
    open(os.path.join(ret, "engagement_all.txt"), "w").write("\n".join(blk) + "\n")
    reqs = [dict(file="short_scan_all.csv.gz", where="return", source="arm", arms=[aid(tg, v) for v in VARIANTS for tg in PFN])]
    if "reqmiss" in mut: reqs.append(dict(file="state_reference.txt", where="return", source="arm", arms=[aid("m3r3", "ac")]))
    json.dump(reqs, open(os.path.join(man, "required_outputs.json"), "w"))
    return ret, man


def selftest():
    t = tempfile.mkdtemp(prefix="r88_gates_"); fails = []
    def go(mut):
        ret, man = build_fixture(os.path.join(t, "fx"), mut)
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            run(ret, man, ret)
        v = {}
        for ln in open(os.path.join(ret, "r88_gates.txt")):
            p = ln.split(None, 2)
            if len(p) >= 2 and p[0] in ("PASS", "FAIL", "MISSING", "INFO"): v[p[1]] = p[0]
        return v
    base = go(())
    for g, x in base.items():
        if x not in ("PASS", "INFO"): fails.append("clean fixture: %s is %s" % (g, x))
    want = [("engmode", "BINNED_ENGAGED"), ("inert", "BINNED_ENGAGED"), ("bbeqbp", "BINNED_ENGAGED"), ("nobinned", "BINNED_ENGAGED"), ("binsgtraw", "MAP_BINS_m1r1"), ("equalbins", "MAP_BINS_m3r1"),
            ("planelow", "MAP_BINS_m3r3"), ("varmapdiff", "MAP_BINS_m1r3"), ("keyset", "MAP_BINS_m3r1"), ("lowrow", "MINBINS"), ("pre", "PREIMU_SAME"), ("exit", "EXIT"), ("zeroscan", "SMALLSCANS"),
            ("stream", "STREAM"), ("iters", "DUMPS"), ("reqmiss", "REQUIRED")]
    for m, g in want:
        v = go((m,))
        if v.get(g) != "FAIL": fails.append("mutation %s did not turn %s (got %s)" % (m, g, v.get(g)))
        if base.get(g) != "PASS": fails.append("clean state of %s is not PASS" % g)
    shutil.rmtree(t, ignore_errors=True)
    print("SELFTEST r88_gates", ("FAIL " + "; ".join(fails)) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ret-dir"); ap.add_argument("--manifest", default="r88_manifest"); ap.add_argument("--out", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    nf, nm = run(a.ret_dir, a.manifest, a.out or a.ret_dir)
    sys.exit(1 if nf or nm else 0)
