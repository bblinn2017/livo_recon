#!/usr/bin/env python3
"""R83: job files, arm_table.tsv, listener specs, queue order, required_outputs.json, size ledger and compliance table.

    python3 tools/r69/make_r83_manifest.py --jobs r69_inputs/r66_jobs.json --grid-retime r80c_grid.json --grid-bag r83_manifest/grid_bag.json \
        --onsets r83_manifest/onset_scans.json --out r83_manifest --results-root /root/catkin_ws/livo_recon_results
    python3 -I tools/r69/make_r83_manifest.py --selftest

Planning agent's script. NOT run on the node by the planner; the selftest runs in the planner sandbox on a synthetic job list (generator output only, not evidence).
Successor of make_r82_manifest.py (with the odometry-export fix of make_r82_manifest_r83_odomfix.py). Reuses make_r81_manifest (Ctx, rt_keys, stub jobs) and through it make_r80_manifest.

Arm families (Bryce's words are in each arm's `asked` field and in COMPLIANCE_TABLE.md):
  DST  downsampling test: eee_01, eee_03, sbs_03 x imu.ds.mode {first, off} x Q, rt, pfn 3, lines off, root 0.425, max_layer 2; run to onset + 70 scans (12)
  MXL  whole-bag max_layer 1 against 2, nine bags x Q, rt, pfn 3, ds first, root 0.425; nothing reused (36)
  VSW  voxel-edge sweep in the other overlap mode (dc), 13 edges 0.20 .. 0.80 step 0.05, nine bags x Q, pfn 3, ds first, lines off (234)
  D1B  point-covariance window lio/point_cov/window_s {5, 10, 20, 50} ms, nine bags x Q, rt, pfn 3, ds first, time_cov off (72)
  D4P  PCA classify-at-fit with larger point counts and always_update on eee_01, nya_02, sbs_03 x Q x 4 variants, queue r83_mem, one at a time, peak RSS (24)
  MQS  motion-Q scales gyro {10, 3, 1, 0.3} x acc {0.3, 0.15, 0.1}, nine bags, motion Q only, rt, pfn 3, ds first (108)
  OWN  identity, window identity, determinism, smoke and refuse arms (planner rule arms; the count is asserted below)
dc = overlap_policy discard (default, no extra keys); rt = retime_trim (set per arm, never through the config default). Hilti: one smoke arm, dc.
"""
import argparse, copy, csv, json, math, os, shutil, sys, tempfile
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.environ.get("R83_TREE_TOOLS", ""), os.environ.get("R82_TREE_TOOLS", ""), "/tmp/claude-0/r83/tree/tools/r69"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
import make_r81_manifest as R81  # noqa: E402

R80, R9, M = R81.R80, R81.R9, R81.M
BAGS = list(R80.BAGS)
Q_NAME = R80.Q_NAME
OWN = R80.OWN_R80
GAP = {R9.GAP: True}
DIST_KEY = dict(R80.DIST_KEY)
REASONS = {"voxel_map.plane.plane_fit_reasons_en": True, "voxel_map.plane.plane_fit_trace_frames": 10}
K_CLASSIFY = "voxel_map.plane.line_residual.classify_at_fit"
K_MAXL = "voxel_map.map.max_layer"
K_PTS_MAX, K_CONV = "voxel_map.points.max", "voxel_map.plane.convergence_mode"
K_WIN = "lio.point_cov.window_s"
K_PROF = "outputs.profiler.export_path"       # FrameProfiler::setExportPath; the file is rewritten every measure group and at shutdown, and the timers accumulate whether or not outputs.printer.profiler.enable is set
K_TC_MODE = "lio.time_cov.mode"
K_ACC, K_GYR = "imu.process_noise.motion.acc_scale", "imu.process_noise.motion.gyro_scale"
K_DUMP, K_DUMP_VI = "lio.residual_dump_frames", "lio.residual_dump_voxel_info"
K_ODOM = "outputs.odom.export"
DST_BAGS = ("eee_01", "eee_03", "sbs_03")
D4P_BAGS = ("eee_01", "nya_02", "sbs_03")
WINDOWS_MS = (5, 10, 20, 50)
EDGES_VSW = [round(0.20 + 0.05 * i, 2) for i in range(13)]                 # 0.20 .. 0.80
GYRO = (10.0, 3.0, 1.0, 0.3)
ACC = (0.3, 0.15, 0.1)
D4P_VARIANTS = (("n100", "normal", 100), ("a50", "always_update", 50), ("n200", "normal", 200), ("a200", "always_update", 200))   # run order: lighter first
DST_MARGIN_S = 3          # planner's choice: 30 scans of margin on top of onset + 70 scans
DST_STAT_LO, DST_STAT_GAP = 30, 40   # stationary frames: first at scan 30, last at onset - 40; the rule is > 10 and <= onset - 20
RSS_ABORT_FRAC = 0.5
EXPECT = {"DST": 12, "MXL": 36, "VSW": 234, "D1B": 72, "D4P": 24, "MQS": 108}
OWN_EXPECT = 20           # 4 identity + 9 determinism + 4 smoke + 3 refuse (the SPEC says "about 16"; the list in SPEC.md adds to 20)
OWN_GROUP = "OWN"
ALIGNFREE_GROUPS = ("D1B", "MXL", "MQS", "DST")
POS_COLS, MSG_COLS_C, N_SCANS = R81.POS_COLS, R81.MSG_COLS_C, R81.N_SCANS
HILTI_GT = "<<FILL: dense ground-truth file of this sequence>>"
QUEUE_MEM = "r83_mem"
PROF_PLACEHOLDER = "<arm path>/profiler.txt"   # replaced per arm in build() by outputs.path + "/profiler.txt"

BRYCE_GO = "Bryce (2026-10-07): 'I agree with the voxel sweep set, and we can go ahead and build the zip'"
ASK = {
    "DST": "Bryce (2026-10-06/07, design 1 in claude/r83-designs-approved.md): downsampling test approved, 'I would like to include the downsampling test, yes' (mq included): bags eee_01, eee_03, sbs_03 x imu.ds.mode {first, off} x Q {fixed, motion}, rt, pfn 3, lines off, run to onset + 70 scans, dump 5 stationary frames and 4 around the onset, lio/residual_dump_voxel_info, plane_fit_reasons_en; " + BRYCE_GO,
    "MXL": "Bryce (2026-10-06/07, design 2 in claude/r83-designs-approved.md and decisions of 2026-10-07): whole-bag max_layer 1 against 2, 'full 36 arms (no reuse)': nine bags x max_layer {1, 2} x Q, rt, pfn 3, ds first, root 0.425 ('runs are far faster than bag time'); " + BRYCE_GO,
    "VSW": "Bryce (2026-10-06/07): 'run all of the arms that we ran in the other mode, which would be from 0.2 to 0.8, I believe, with 0.05 spacing', then 'I agree with the voxel sweep set'; Tier A ran retime_trim, so the other mode is dc (PLANNER READING, see ASSUMPTION-READING): dc, pfn 3, lines off, ds first, nine bags x Q x 13 edges",
    "D1B": "Bryce (2026-10-07, design 4): D1 option B approved, 'construct it': patch r83_p4, key lio/point_cov/window_s; windows 0.005, 0.010, 0.020, 0.050 s, nine bags, root 0.425, time_cov off ('72 arms if fq and mq'); " + BRYCE_GO,
    "D4P": "Bryce (2026-10-07, decisions file, Q6): 'APPROVED 2026-10-07: planner draft as stated': eee_01, nya_02, sbs_03; fq and mq; max points 100 / 200 with normal convergence, 50 / 200 with always_update; 24 arms; one at a time; record peak RSS; abort above half the host RAM; " + BRYCE_GO,
    "MQS": "Bryce (2026-10-07): 'give the motion cue gyro and acceleration scales an arm family in the handoff' (PLANNER READING, see ASSUMPTION-READING: gyro_scale {10, 3, 1, 0.3} crossed with acc_scale {0.3, 0.15, 0.1}, nine bags, motion Q only, caps untouched)",
}
ASK_MODES = "overlap mode set per arm (dc = discard, rt = retime_trim), never through the config default"


# ---------------------------------------------------------------------------------------------------------------------------------- helpers
def queue_of(a):
    if a["fam_kind"] in ("id", "smoke", "refuse") and a.get("bag") != "hilti":
        return "r83_id"
    if a.get("bag") == "hilti":
        return "r83_hilti"
    if a["group"] == "D4P":
        return QUEUE_MEM
    return "r83_%s" % a["bag"]


def derive(C, src, aid, ovl, group, fam, extra, asked, purpose, pop=(), dur=None, kind="fb", ref=None, msglog=0, flag="", **flags):
    """One arm from a source arm dict (an R80 arm or an R80.mk arm): same base job and overrides, plus extra and, for rt, the retime keys."""
    b = copy.deepcopy(src)
    b["id"] = aid
    d = dict(b["delta"])
    for k in pop:
        d.pop(k, None)
    d.update(extra)
    if ovl == "rt":
        d.update(R81.rt_keys(C.grid(b["bag"])))
    b["delta"] = d
    b.update(kind=kind, ref=ref, group=group, fam=fam, ovl=ovl, src_arm=src["id"], msglog=msglog, win=0, f417=0, dump=0, early=0, flag=flag, fam_kind=kind)
    b["tier"] = "EF" if dur else "FB"
    b["dur"] = dur if dur else None
    b.update(flags)
    b["asked"] = asked + "; same arm as %s (%s, point filter %s, %s Q) with overlap mode %s%s" % (
        src["id"], "first %s s" % b["dur"] if b["dur"] else "whole bag", b["pfn"], Q_NAME.get(b["q"], b["q"]), ovl, (", extra keys " + json.dumps(extra, sort_keys=True)) if extra else "")
    b["purpose"] = purpose
    b["batch"] = queue_of(b)
    b.setdefault("control", ""); b.setdefault("r82_control", ""); b.setdefault("dump_frames", "")
    return b


def newsrc(C, bag, q, mode, v=0.425, pfn=3, extra=None, aid="tmp"):
    return R80.mk(aid, bag, mode, v, q, "SRC", "src", "src", pfn=pfn, extra=dict(extra or {}), batch="x")


def tg(x):
    return M.tag(x)


def wtag(ms):
    return "w%dms" % ms


def stag(x):
    return ("%g" % x).replace(".", "p")


def dst_duration_s(onset, margin=DST_MARGIN_S):
    """ceil((onset + 70) * 0.1 + margin) seconds, in exact arithmetic."""
    return int(math.ceil(Fraction(onset + 70, 10) + margin))


def dst_frames(onset):
    """5 stationary frames evenly spaced from scan DST_STAT_LO to onset - DST_STAT_GAP, then onset - 1, onset, onset + 1, onset + 30. Frame index = scan index (assumption, see REPORT)."""
    lo, hi = DST_STAT_LO, onset - DST_STAT_GAP
    stat = [lo + int(round(i * (hi - lo) / 4.0)) for i in range(5)]
    return stat + [onset - 1, onset, onset + 1, onset + 30]


# ---------------------------------------------------------------------------------------------------------------------------------- arm builders
def arms(C):
    S, A = C.src, []
    # ---- DST (12)
    for bag in DST_BAGS:
        onset = C.onsets[bag]
        dur = dst_duration_s(onset)
        fr = dst_frames(onset)
        assert all(10 < f <= dur * 10 for f in fr) and all(f <= onset - 20 for f in fr[:5]) and len(set(fr)) == 9, (bag, fr, dur)
        for q in ("fq", "mq"):
            for ds in ("first", "off"):
                ex = {K_MAXL: 2, K_DUMP: ",".join(str(f) for f in fr), K_DUMP_VI: True}
                ex.update(REASONS)
                if ds == "off":
                    ex.update({"imu.ds.mode": "off", "imu.ds.ds_leaf_size": None})
                a = derive(C, C.base3(bag, q), "r83_rt_dst_%s_%s_%s" % (bag, ds, q), "rt", "DST", "DST", ex, ASK["DST"] + "; " + ASK_MODES,
                           "R83 downsampling test: ds %s, %s %s, first %d s (onset scan %d + 70 scans + %d s margin), dump frames %s" % (ds, bag, q, dur, onset, DST_MARGIN_S, ex[K_DUMP]), dur=dur)
                a["dump_frames"] = ex[K_DUMP]
                a["control"] = "" if ds == "first" else "r83_rt_dst_%s_first_%s" % (bag, q)
                A.append(a)
    # ---- MXL (36)
    for bag in BAGS:
        for q in ("fq", "mq"):
            for ml in (1, 2):
                a = derive(C, C.base3(bag, q), "r83_rt_ml_%s_l%d_%s" % (bag, ml, q), "rt", "MXL", "ML", {K_MAXL: ml}, ASK["MXL"] + "; " + ASK_MODES,
                           "R83 max_layer %d, %s %s, whole bag" % (ml, bag, q), msglog=1 if ml == 2 else 0)
                a["control"] = "r83_rt_ml_%s_l2_%s" % (bag, q) if ml == 1 else ""
                a["r82_control"] = "r82_rt_ds_%s_first_%s" % (bag, q) if ml == 2 else ""
                A.append(a)
    # ---- VSW (234): dc, 13 edges
    for bag in BAGS:
        for q in ("fq", "mq"):
            for e in EDGES_VSW:
                src = S["r80_sw_%s_off_v%s_pfn3_%s" % (bag, tg(e), q)]
                a = derive(C, src, "r83_dc_sw_%s_off_v%s_pfn3_%s" % (bag, tg(e), q), "dc", "VSW", "SW", {}, ASK["VSW"] + "; mode dc = overlap_policy discard (no extra keys)",
                           "R83 voxel sweep (other mode, dc): %s %s edge %s m" % (bag, q, e))
                a["r82_control"] = "r82_dc_ds_%s_first_%s" % (bag, q)
                A.append(a)
    # ---- D1B (72)
    for bag in BAGS:
        for q in ("fq", "mq"):
            for ms in WINDOWS_MS:
                a = derive(C, C.base3(bag, q), "r83_rt_pc_%s_%s_%s" % (bag, wtag(ms), q), "rt", "D1B", "PC", {K_WIN: ms / 1000.0, K_TC_MODE: "off", K_PROF: PROF_PLACEHOLDER}, ASK["D1B"] + "; " + ASK_MODES,
                           "R83 D1 option B: point-covariance window %d ms, %s %s" % (ms, bag, q))
                a["control"] = "r83_rt_ml_%s_l2_%s" % (bag, q)
                a["r82_control"] = "r82_rt_ds_%s_first_%s" % (bag, q)
                A.append(a)
    # ---- D4P (24), queue r83_mem
    line = dict(DIST_KEY, **GAP, **REASONS)
    for vname, conv, mx in D4P_VARIANTS:
        for bag in D4P_BAGS:
            for q in ("fq", "mq"):
                ex = dict(line, **{K_CLASSIFY: True, K_CONV: conv, K_PTS_MAX: mx})
                fr = None
                if bag == "eee_01":        # OWN: points per voxel on eee_01 only, the same 9 frames as the DST arms of that bag
                    fr = ",".join(str(f) for f in dst_frames(C.onsets[bag]))
                    ex.update({K_DUMP: fr, K_DUMP_VI: True})
                src = newsrc(C, bag, q, "hardabs", extra=ex)               # the same call as the R82 pcls arm (r82_rt_d4_<bag>_pcls_<q>) plus the two new keys
                a = derive(C, src, "r83_rt_d4_%s_pc%s_%s" % (bag, vname, q), "rt", "D4P", "LC", {}, ASK["D4P"] + "; " + ASK_MODES,
                           "R83 PCA classify-at-fit, convergence %s, voxel_map/points/max %d, %s %s (queue %s, one at a time, peak RSS)" % (conv, mx, bag, q, QUEUE_MEM))
                a["r82_control"] = "r82_rt_d4_%s_pcls_%s" % (bag, q)
                if fr:
                    a["dump_frames"] = fr
                    a["own_dump"] = True
                A.append(a)
    # ---- MQS (108)
    for bag in BAGS:
        for g in GYRO:
            for ac in ACC:
                a = derive(C, C.base3(bag, "mq"), "r83_rt_mqs_%s_g%s_a%s_mq" % (bag, stag(g), stag(ac)), "rt", "MQS", "MQ", {K_GYR: g, K_ACC: ac, "imu.process_noise.model": "isotropic"}, ASK["MQS"] + "; " + ASK_MODES,
                           "R83 motion-Q scales: gyro_scale %g, acc_scale %g, %s" % (g, ac, bag))
                a["control"] = "r83_rt_ml_%s_l2_mq" % bag
                a["r82_control"] = "r82_rt_ds_%s_first_mq" % bag
                A.append(a)
    A += own_arms(C, A)
    for a in A:
        a["same_batch_control"] = a.get("control", "")
        if a["group"] in ("D1B", "MXL", "MQS"):       # alignfree pairing: the R82 control of the same bag and Q
            a["control"] = a["r82_control"] if a["r82_control"] else "r82_rt_ds_%s_first_%s" % (a["bag"], a["q"])
        a["mode_line"] = a["mode"]
        a["alignfree"] = 1 if a["group"] in ALIGNFREE_GROUPS else 0
        a["concurrency"] = 1 if a["batch"] == QUEUE_MEM else ""
        a["rss_abort_frac"] = RSS_ABORT_FRAC if a["batch"] == QUEUE_MEM else ""
    return A


def _hilti(C, aid, bag, q, ds, mq, group, fam, asked, purpose, dur=None, kind="fb", extra=None):
    d = {"evo.enable": True, "vio.enable": False, "evo.gt_source": "file", "evo.gt_file": HILTI_GT, "evo.dense_gt_metrics": True, "eval.state_trace_en": True,
         K_ODOM: True,   # run_job.sh gates DONE on a non-empty odometry.txt, written only with this key (every R82 Hilti job lacked it): kept on every Hilti job, selftest asserts it
         "diagnostics.frame_window_log": True, "imu.ds.mode": ds, "cbk.lidar.point_filter_num": 3}
    if ds == "off":
        d["imu.ds.ds_leaf_size"] = None
    d["imu.process_noise.model"] = "fixed" if q == "fq" else mq.get("imu.process_noise.model", "isotropic")
    if q == "mq":
        d.update(mq)
    # no point-time keys: since patch r83_p2 the Hilti config itself sets cbk/lidar/point_time_field timestamp and point_time_mode absolute_s_f64
    d.update(extra or {})
    return dict(id=aid, base="HILTI_TEMPLATE", group=group, tier="EF" if dur else "FB", batch="r83_hilti", dur=dur, mode=None, voxel=None, kind=kind, ref=None, delta=d, f417=0,
                asked=asked + "; overlap mode dc (no Hilti retime grid exists)", purpose=purpose, layer=None, gap=False, stream="primary", info3=False, pfn=3, q=q, pe=False, dump=0, bag="hilti",
                early=0, snap5=False, zupt=False, nis=False, win=0, msglog=0, fam=fam, ovl="dc", src_arm="(Hilti template)", flag="", fam_kind=kind, hbag=bag, seq_hilti=None,
                control="", r82_control="", dump_frames="")


def _smoke(C, aid, q, extra, why, mode="off", bag="eee_01"):
    a = R80.mk(aid, bag, mode, 0.425, q, OWN_GROUP, OWN + "rule 3.6 smoke arm for the new key: %s; 60 s, eee_01, 0.425 m, pfn 3, %s Q, overlap_policy default (dc)" % (why, Q_NAME[q]),
               "OWN smoke: %s" % why, pfn=3, extra=dict(extra), dur=R80.DP_DUR, batch="r83_id", tier="EF", kind="smoke")
    a.update(fam="SMK", ovl="dc", src_arm="(new key, no R80 source arm; R80 smoke pattern)", flag="", fam_kind="smoke", batch="r83_id", ref=None, control="", r82_control="", dump_frames="")
    return a


def own_arms(C, A):
    S, out = C.src, []
    idinfo = (("dc", "fq", "r83_dc_id_eee_01_fq", "r82_dc_ds_eee_01_first_fq", {}), ("rt", "fq", "r83_rt_id_eee_01_fq", "r82_rt_ds_eee_01_first_fq", {}),
              ("rt", "mq", "r83_rt_id_eee_01_mq", "r82_rt_ds_eee_01_first_mq", {}), ("rt", "fq", "r83_rt_id_pc0_eee_01_fq", "r82_rt_ds_eee_01_first_fq", {K_WIN: 0.0}))
    for ovl, q, aid, ref, ex in idinfo:
        a = derive(C, C.base3("eee_01", q), aid, ovl, OWN_GROUP, "ID", ex, OWN + "identity arm: whole bag eee_01, %s, %s Q, every new R83 key at its default%s; the state must equal R82 %s (state_trace and pos_trace within 0.005 mm); the keys removed in p1 (lio/zupt/*) were not set by it"
                   % (ovl, Q_NAME[q], " except lio/point_cov/window_s set explicitly to 0" if ex else "", ref), "OWN identity: %s %s vs %s" % (ovl, q, ref), kind="id", ref=ref, msglog=1)
        a["r82_control"] = ref
        out.append(a)
    for bag in BAGS:
        o = next(x for x in A if x["id"] == "r83_rt_ml_%s_l2_fq" % bag)
        d = copy.deepcopy(o)
        d.update(id="r83_own_det_x_%s" % bag, kind="det_fb", fam_kind="det_fb", ref=o["id"], group=OWN_GROUP, fam="DET", msglog=0, flag="", control="", r82_control="")
        d["asked"] = OWN + "rule 3.7 determinism control (one repeat per bag queue, rule 1.16): an exact whole-bag repeat of %s (same batch); its state-trace digest must equal that arm's" % o["id"]
        d["purpose"] = "OWN determinism control: repeat of %s" % o["id"]
        d["batch"] = queue_of(d)
        out.append(d)
    sm = lambda aid, q, extra, why, mode="off": _smoke(C, aid, q, extra, why, mode=mode)
    line = dict(DIST_KEY, **GAP, **REASONS)
    out.append(sm("r83_dc_sm_pcw_60s", "fq", {K_WIN: 0.02}, "lio/point_cov/window_s = 0.02 (D1B): point_cov_window.csv returned with rows whose sd_p_med_m is non-zero; the engagement block and the imu/window_cov_profile timer line are present"))
    out.append(sm("r83_dc_sm_pcl_60s", "fq", dict(line, **{K_CLASSIFY: True, K_CONV: "always_update", K_PTS_MAX: 200}), "voxel_map/plane/convergence_mode = always_update with voxel_map/points/max 200 (D4P): the parameter dump shows both, peak RSS is readable", mode="hardabs"))
    out.append(sm("r83_dc_sm_dst_60s", "fq", dict(REASONS, **{K_DUMP: "400,401,402", K_DUMP_VI: True}), "residual dump with lio/residual_dump_voxel_info (DST, patch r83_p3): residual_dump.csv has a voxel_layer column and non-empty values (0 to max_layer), plane_fit_reasons.csv written"))
    mq = {k: v for k, v in C.J[C.base3("eee_01", "mq")["base"]]["overrides"].items() if k.startswith("imu.process_noise.")}
    h = _hilti(C, "r83_dc_sm_hl_exp04_60s", "exp04", "fq", "first", mq, OWN_GROUP, "SMK", OWN + "rule 3.6 smoke arm for patch r83_p2 (the Hilti config sets cbk/lidar/point_time_field timestamp and point_time_mode absolute_s_f64 by itself, so no point-time key is set by this job) and the odometry export fix (outputs.odom.export true), Hilti exp04, 60 s, fixed Q (Bryce: no Hilti motion Q until the estimator numbers return): the engagement line shows the f64 read, odometry.txt non-empty, the [evo] line has ARE and ATE_first",
               "OWN smoke: Hilti exp04, point time from the config default (p2) and the odometry export", dur=60, kind="smoke")
    out.append(h)
    R = S["r80_refuse_rows_distance_smooth"]
    refs = (("zupt_removed", {"lio.zupt.mode": "apply"}, ["lio/zupt/mode"]),
            ("window_s_neg", {K_WIN: -0.01}, ["lio/point_cov/window_s"]),
            ("window_s_big", {K_WIN: 0.2}, ["lio/point_cov/window_s"]))
    for suf, delta, needles in refs:
        r = copy.deepcopy(R)
        r.update(id="r83_dc_refuse_%s" % suf, delta=dict(delta), needles=needles, group=OWN_GROUP, fam="REF", ovl="dc", src_arm="", flag="", fam_kind="refuse", batch="r83_id", kind="refuse", ref=None, control="", r82_control="", dump_frames="")
        r["asked"] = OWN + "rule 3.4 refusal: MUST FAIL at startup, text contains %s; its exit_status.txt must be non-zero" % " and ".join(repr(n) for n in needles)
        r["purpose"] = "MUST FAIL: %s" % suf
        out.append(r)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- counts
def group_counts(ALL):
    cnt = {}
    for a in ALL:
        cnt[a["group"]] = cnt.get(a["group"], 0) + 1
    return cnt


def check_counts(ALL, die=sys.exit):
    cnt = group_counts(ALL)
    rows, bad = [], False
    for g in list(EXPECT) + [OWN_GROUP]:
        have, want = cnt.get(g, 0), (EXPECT.get(g) if g != OWN_GROUP else OWN_EXPECT)
        ok = have == want
        bad |= not ok
        rows.append("%-5s spec %-6s real %-6d %s" % (g, want, have, "" if ok else "MISMATCH"))
    extra = set(cnt) - set(EXPECT) - {OWN_GROUP}
    if extra:
        bad = True
        rows.append("UNKNOWN GROUPS: %s" % sorted(extra))
    rows.append("total spec %d + OWN %d = %d, real %d" % (sum(EXPECT.values()), OWN_EXPECT, sum(EXPECT.values()) + OWN_EXPECT, len(ALL)))
    table = "\n".join(rows)
    if bad:
        die("COUNT MISMATCH against the SPEC arm-group table. Real counts follow.\n" + table)
    return table


# ---------------------------------------------------------------------------------------------------------------------------------- required outputs and ledger
def req_of(o, aid):
    r = list(R80.req_of(o))
    if o.get(K_WIN, 0) and float(o[K_WIN]) > 0:
        r.append("point_cov_window.csv")
    if o.get("voxel_map.plane.plane_fit_reasons_en"):
        r.append("plane_fit_reasons.csv")
    if "_mqs_" in aid:
        r.append("motion_q_scan.csv")
    if o.get(K_CONV) is not None:
        r.append("peak_rss.txt")
    return r


def required_outputs(table):
    by = {t["arm"]: t for t in table}
    sel = lambda f: [t["arm"] for t in table if f(t)]
    nonref = sel(lambda t: t["arm_kind"] != "refuse")
    nonsw = sel(lambda t: t["arm_kind"] != "refuse" and t["group"] != "VSW")
    whole = lambda t: t["duration_s"] == "full"
    ntu = sel(lambda t: t["arm_kind"] != "refuse" and t["bag"] != "hilti")
    R = []

    def add(fu, title, arms_, file, cols, min_rows, per=None, where="return", source="arm", eb=None, maxb=400000, **extra):
        e = eb if eb is not None else (int(sum(per(by[a]) for a in arms_)) if (where == "return" and arms_ and per) else 0)
        R.append(dict(followup=fu, title=title, arms=arms_, file=file, columns=cols, min_rows=min_rows, where=where, source=source, contains={}, expected_bytes=int(e), max_bytes_per_file=maxb, **extra))
    n = lambda t: R81.nscan(t) if t["bag"] != "hilti" else 4000
    add("F-484", "per-scan position, yaw and quaternion of every whole-bag arm except the VSW sweep arms (those return summaries only, plus a trace when gt_max_mm > 500; a ceiling is ledgered)",
        [a for a in nonsw if whole(by[a])], "pos_trace_all.csv.gz", POS_COLS, 1500, per=lambda t: 56 * n(t), maxb=330000)
    add("F-484", "per-scan position, yaw and quaternion of every short arm", [a for a in nonsw if not whole(by[a])], "pos_trace_all.csv.gz", POS_COLS, 5, per=lambda t: 56 * n(t), maxb=330000)
    mg = sel(lambda t: t["msglog"] == "1")
    add("F-484", "per-message LiDAR log (17 columns incl. retime_shift_s, n_trimmed) of the same-batch controls (max_layer 2 arms) and the identity arms", mg, "lidar_msg_log_all.csv.gz", MSG_COLS_C, 1500, per=lambda t: 100 * n(t), maxb=900000)
    add("F-484", "collector by-product per arm (line_rows_per_scan.csv), ledgered", nonsw, "line_rows_per_scan_all.csv.gz", [], 1, per=lambda t: 25000 if whole(t) else 10000, maxb=60000)
    add("F-452", "per-scan table of every non-sweep NTU whole-bag arm (arm_scan.csv; thinned by the collector --level to fit the ledger); sweep, Hilti and short arms return phase_summary rows only", [], "arm_scan_all.csv.gz", [], 5,
        eb=min(60000000, sum(120 * n(by[a]) for a in ntu if by[a]["group"] != "VSW" and whole(by[a]))), source="generated", maxb=400000000)
    add("F-452", "phase_summary row of every arm (metrics, onset scan, collapse flags)", [], "phase_summary.csv", ["arm"], len(nonref), eb=4000 * len(nonref), source="generated", maxb=20000000)
    add("F-409", "engagement of every dispatched arm concatenated (includes every new key's engagement block)", [], "engagement_all.txt", [], 10, eb=2500 * len(table), source="generated", maxb=8000000)
    add("F-409", "per-arm completeness (arm_table 'requires' column)", [], "completeness.csv", ["arm", "requires_missing"], 10, eb=1200 * len(table), source="generated", maxb=3000000)
    add("F-326", "exit_status.txt of every dispatched arm: refuse arms non-zero, all others zero (a D4P arm aborted by the memory watcher is non-zero and carries ABORTED_RSS)", sel(lambda t: True), "exit_status.txt", [], 1, where="archive")
    add("F-407", "engagement_imu.txt present for every dispatched arm that is not a refuse arm", nonref, "engagement_imu.txt", [], 1, where="archive")
    add("F-496", "state_reference.txt of every NTU arm except the VSW sweep arms (first-frame reference of Bryce's metric rule)", [a for a in nonsw if by[a]["bag"] != "hilti"], "state_reference.txt", [], 1, per=lambda t: 400, maxb=4000)
    af = sel(lambda t: t["group"] in ALIGNFREE_GROUPS)
    add("F-496", "alignment-free stationary drift, excursion, settled offset and departure of the D1B, MXL, MQS and DST arms ONLY (%d arms; the R83 alignfree tool compares each with the same-bag control named in arm_table column control / r82_control); the R81 rows of the R82 file (532) are NOT carried" % len(af),
        af, "alignfree_summary.csv", [], len(af), eb=1500 * len(af), source="generated", maxb=8000000)
    add("F-452", "gates of R83: identity, determinism, grid identity per mode, key engagement, refusals, exit status, completeness", [], "r83_gates.txt", [], 1, eb=1500000, source="generated", maxb=4000000)
    ids = sel(lambda t: t["arm_kind"] == "id")
    add("F-452", "identity comparison of the four identity arms against their R82 references", [], "r83_identity_vs_old.csv", ["arm", "ref_arm", "n_common_scans", "pos_max_abs_diff_mm", "yaw_max_abs_diff_mrad"], len(ids), eb=3000, source="generated", maxb=40000)
    add("F-452", "determinism controls and their originals", [], "identity_controls.txt", [], 1, eb=40000, source="generated", maxb=150000)
    # new groups
    d4dump = sel(lambda t: t["group"] == "D4P" and t["bag"] == "eee_01")
    dst = sel(lambda t: t["group"] == "DST" or t["arm"] == "r83_dc_sm_dst_60s") + d4dump
    add("R83-DST", "residual dump of the DST arms and of the 8 eee_01 D4P arms (9 frames each, iteration 0) with voxel_point_count, voxel_distinct_frames and voxel_layer (gzip copy made by the score job)", dst, "residual_dump_all.csv.gz",
        ["frame", "voxel_point_count", "voxel_distinct_frames", "voxel_layer"], 100, per=lambda t: 300000, maxb=800000)
    add("R83-DST", "plane-fit rejection reasons of the DST and D4P arms and the two smoke arms that set plane_fit_reasons_en (host archive)", sel(lambda t: t["group"] in ("DST", "D4P") or t["arm"] in ("r83_dc_sm_dst_60s", "r83_dc_sm_pcl_60s")), "plane_fit_reasons.csv", [], 1, where="archive")
    add("R83-DST", "one row per frame of r83_ds_reduce.py over the DST arms and the 8 eee_01 D4P arms (points per voxel, spread, residual distribution, range, layer census)", [], "r83_ds_rows.csv", ["arm", "frame"], len(dst) - 1, eb=20000 * len(dst), source="generated", maxb=1000000)
    pcw = sel(lambda t: t["group"] == "D1B" or t["arm"] == "r83_dc_sm_pcw_60s")
    add("R83-D1B", "point_cov_window.csv per scan (scan_index, t_abs, window_s, n_steps, sd_p_med_m ... sd_r_max_rad) of every D1B arm and the window smoke arm (host archive, lossless; the listener raw.delete list names it so the score job archives it)", pcw, "point_cov_window.csv",
        ["scan_index", "window_s", "n_steps", "sd_p_med_m"], 100, where="archive")
    add("R83-D1B", "profiler.txt per D1B arm: the FrameProfiler export (one line per timer with average ms and frame count, includes imu/window_cov_profile), written to the arm's own folder by outputs.profiler.export_path; host archive, lossless. NOTE FOR THE GATES: outputs.printer.profiler.enable is false, so run.log has no [profiler] lines; read profiler.txt first", sel(lambda t: t["group"] == "D1B"), "profiler.txt", [], 1, where="archive")
    cov = sel(lambda t: t["group"] == "D1B" or (t["group"] == "MXL" and "_l2_" in t["arm"]))
    add("R83-D1B", "covariance-consistency rows from state_trace.csv (host script r83_cov_consistency.py per arm, joined by r83_cov_collect.py) of the D1B arms and their same-batch controls (the %d max_layer 2 arms; each arm has one row per window S, W, M)" % (len(cov) - 72), [], "r83_cov_rows_all.csv", ["arm", "window"], len(cov), eb=20000 * len(cov), source="generated", maxb=3000000)
    add("R83-D1B", "the imu/window_cov_profile timer line, one per D1B arm (profile of the window pass), collected from each arm's profiler.txt (run.log only as a fallback)", [], "r83_window_timer_all.txt", [], len(pcw) - 1, eb=300 * len(pcw), source="generated", maxb=100000)
    d4p = sel(lambda t: t["group"] == "D4P")
    add("R83-D4P", "peak resident set size per D4P arm, written by the memory watcher into the arm's job directory (name peak_rss.txt exactly); host archive", d4p, "peak_rss.txt", [], 1, where="archive")
    add("R83-D4P", "ABORTED_RSS marker (job killed by the memory watcher above half of the host RAM, then the next arm continues): present ONLY on an aborted arm; every aborted arm must be reported", [], "ABORTED_RSS", [], 0, where="archive", conditional=True)
    add("R83-D4P", "line / plane classification counters of the D4P arms and the always_update smoke arm", [], "r83_d4_counters.csv", [], len(d4p), eb=60000, source="generated", maxb=400000)
    add("F-469", "plane-fit trace counts of the DST and D4P arms and the two smoke arms (the collector builds it from plane_fit_trace.csv)", [], "plane_fit_trace_counts_all.csv", [], 5, eb=40000, source="generated", maxb=200000)
    mqs = sel(lambda t: t["group"] == "MQS")
    add("R83-MQS", "motion_q_scan.csv per scan of every MQS arm (columns acc_scale, gyro_scale, acc_cap_fraction, gyr_cap_fraction); host archive, lossless", mqs, "motion_q_scan.csv",
        ["scan_index", "acc_scale", "gyro_scale", "acc_cap_fraction", "gyr_cap_fraction"], 100, where="archive")
    hil = sel(lambda t: t["bag"] == "hilti" and t["arm_kind"] != "refuse")
    add("F-506", "Hilti metrics of the r83_hilti smoke arm from the [evo] line: ATE, RTE, ROE, ARE and ATE_first, ARE_first (tools/r69/r83_hilti_reduce.py)", [], "r83_hilti_metrics.csv", ["arm", "ate_m", "are_deg", "ate_first_m", "are_first_deg"], len(hil), eb=3000 * len(hil), source="generated", maxb=1000000)
    add("F-504", "point-time read report of the Hilti smoke arm (the engagement block of the point_time_* keys)", [], "r83_point_time_engagement.csv", [], len(hil), eb=1500 * len(hil), source="generated", maxb=500000)
    return R


MEASURED = R81.MEASURED


def ledger(RO, table):
    out = []
    for e in RO:
        if e["where"] != "return" or not e["expected_bytes"]:
            continue
        mult = 1.5 if e["file"] in MEASURED else 3.0
        out.append({"pattern": e["file"], "max_bytes_per_file": e["max_bytes_per_file"], "max_total_bytes": int(e["expected_bytes"] * mult),
                    "basis": ("measured on the R80 / R81 returns (per-arm size taken from the R82 ledger), allowance 1.5x" if e["file"] in MEASURED else "estimate (new files unmeasured), allowance 3x"), "note": e["followup"] + " (%d arms)" % len(e["arms"])})
    res, at = [], {}
    for x in out:
        if x["pattern"] in at:
            y = res[at[x["pattern"]]]
            y["max_bytes_per_file"] = max(y["max_bytes_per_file"], x["max_bytes_per_file"]); y["max_total_bytes"] += x["max_total_bytes"]; y["note"] += " + " + x["note"]
        else:
            at[x["pattern"]] = len(res); res.append(dict(x))
    out = res
    nvsw = sum(1 for t in table if t["group"] == "VSW")
    for x in out:
        if x["pattern"] == "pos_trace_all.csv.gz":
            x["max_total_bytes"] += nvsw * 56 * 4300
            x["basis"] += "; plus a ceiling for ALL %d VSW arms whose trace is returned when gt_max_mm > 500 (R82 ceilinged 150 of its 162)" % nvsw
    nn = len(table)
    for pat, mx, tot in (("summaries_all.json", 40000000, 16000 * nn), ("stop_reasons.csv", 2000000, 400 * nn), ("listener_*.log", 5000000, 40000000), ("R83_tree.tar.gz", 80000000, 80000000), ("agent_edits.patch", 2000000, 2000000),
                         ("raw_sha256.txt", 20000, 20000), ("compress_manifests.json", 100000, 100000), ("*.md", 1000000, 4000000), ("*.json", 2000000, 20000000), ("*.jsonl", 2000000, 10000000), ("*.txt", 1000000, 10000000),
                         ("*.tsv", 1000000, 3000000), ("*.csv", 3000000, 10000000), ("*.sha256", 100000, 100000), ("imu_timing/*", 4000000, 40000000)):
        out.append({"pattern": pat, "max_bytes_per_file": mx, "max_total_bytes": int(tot), "basis": "generic ceiling (R82 ledger scaled to the arm count); not a measurement", "note": "non-arm file"})
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- build
COLS = ["arm", "tier", "arch", "kind", "duration_s", "seq", "notes", "keep_raw", "dup_of", "r66_id", "group", "batch", "sampling", "requires", "f417", "plane_check", "asked", "early", "pfn", "pe", "dump", "bag", "win", "msglog",
        "mode", "fam", "src_arm", "arm_kind", "ref", "needles", "flag",
        "control", "r82_control", "alignfree", "concurrency", "rss_abort_frac", "dump_frames", "same_batch_control"]          # the last seven are new in R83 (appended; the R82 column set and order are unchanged)
RAW_ADD = ["point_cov_window.csv", "peak_rss.txt", "plane_fit_reasons.csv", "profiler.txt"]      # archived losslessly by the score job (it archives the raw.delete names), so they are kept in the host archive


def build(a, J, grid_retime, grid_bag, onsets, out_dir, results_root, min_free_gb=150.0, workers=6, quiet=False, onsets_src=None, grid_bag_src=None):
    for bag in BAGS:
        if bag not in grid_bag or bag not in grid_retime:
            sys.exit("grid files have no entry for %s" % bag)
        if grid_bag[bag]["seq_gaps"] != 0 or grid_retime[bag]["seq_gaps"] != 0:
            sys.exit("%s: seq_gaps != 0 in a grid file; stop" % bag)
    for bag in BAGS:
        if bag not in onsets:
            sys.exit("no onset scan for %s" % bag)
    os.makedirs(out_dir, exist_ok=True)
    C = R81.Ctx(J, grid_retime, grid_bag, onsets)
    ALL = arms(C)
    counts_txt = check_counts(ALL)
    out, table, seen = {}, [], set()
    prio = {"MXL": 0, "OWN": 0, "DST": 1, "D1B": 2, "MQS": 3, "D4P": 4, "VSW": 5}
    d4o = {v[0]: i for i, v in enumerate(D4P_VARIANTS)}
    key = lambda i: (ALL[i]["batch"] != "r83_id", prio.get(ALL[i]["group"], 9), d4o.get(ALL[i]["id"].split("_pc")[-1].rsplit("_", 1)[0], 0) if ALL[i]["group"] == "D4P" else 0, i)
    order = sorted(range(len(ALL)), key=key)
    hilti_base = {"job_id": "HILTI_TEMPLATE", "dataset": "slam_2022_handheld", "seq": "<<FILL>>", "overrides": {}}
    for i in order:
        r = ALL[i]
        assert r["id"] not in seen, "duplicate arm id " + r["id"]
        seen.add(r["id"])
        assert r.get("asked"), "arm without an asked line: " + r["id"]
        if r["base"] == "HILTI_TEMPLATE":
            j = copy.deepcopy(hilti_base)
            j["seq"] = r.get("hbag", "exp04")
            j["hilti_placeholders"] = ["evo.gt_file", "bag path through the node's own job generator (the planner holds neither)"]
        else:
            if r["base"] not in J:
                sys.exit("base job %s not in r66_jobs.json" % r["base"])
            j = copy.deepcopy(J[r["base"]])
        j["job_id"] = r["id"]
        o = j["overrides"]
        for k, v in r["delta"].items():
            o[k] = v
        if o.get("estimator.mode") == "decoupled":
            o["estimator.mode"] = "iekf"
        o.pop("spline.mode", None)
        for k in o:
            if "perturb" in k.lower():
                sys.exit("perturbation key forbidden: " + k)
        o["eval.state_trace_run_id"] = r["id"]
        path = "%s/%s/%s" % (results_root, r["batch"], r["id"])
        o["outputs.path"] = path; o["outputs.debug_log_dir"] = path
        if o.get(K_PROF) == PROF_PLACEHOLDER:
            o[K_PROF] = path + "/profiler.txt"
        if r["dur"]:
            j["duration_secs"] = r["dur"]
        out.setdefault(r["batch"], []).append(j)
        sampling = "pfn%s_ds%s" % (o.get("cbk.lidar.point_filter_num", "def"), o.get("imu.ds.mode", "def"))
        req = [] if r["kind"] == "refuse" else req_of(o, r["id"])
        table.append(dict(arm=r["id"], tier=r["tier"], arch="iekf_raw", kind="closed", duration_s=str(j.get("duration_secs", "full")), seq=j["seq"], notes=r["purpose"], keep_raw="0", dup_of="",
                          r66_id=r["base"], group=r["group"], batch=r["batch"], sampling=sampling, requires=" ".join(dict.fromkeys(req)), f417=str(r["f417"]), plane_check="0",
                          asked=r["asked"], early=str(r["early"]), pfn=str(r["pfn"]), pe="0", dump=str(r["dump"]), bag=r["bag"], win=str(r["win"]), msglog=str(r["msglog"]),
                          mode=r["ovl"], fam=r["fam"], src_arm=r["src_arm"], arm_kind=r["kind"], ref=r.get("ref") or "", needles="|".join(r.get("needles") or []), flag=r.get("flag", ""),
                          control=r.get("control", ""), r82_control=r.get("r82_control", ""), alignfree=str(r["alignfree"]), concurrency=str(r["concurrency"]), rss_abort_frac=str(r["rss_abort_frac"]),
                          dump_frames=r.get("dump_frames", ""), same_batch_control=r["same_batch_control"]))
    qlist = sorted(out)
    for b in qlist:
        json.dump(out[b], open(os.path.join(out_dir, "jobs_%s.json" % b), "w"), indent=1)
    with open(os.path.join(out_dir, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, COLS, delimiter="\t"); w.writeheader(); w.writerows(table)
    qd = lambda b: dict({"dir": "ablations/_job_queue/%s" % b, "results_root": "%s/%s" % (results_root, b), "expected_jobs": len(out[b])}, **({"concurrency": 1, "serial": True} if b == QUEUE_MEM else {}))
    specs = [("id", ["r83_id"])] + [(b, ["r83_" + b]) for b in BAGS] + [("hilti", ["r83_hilti"]), ("mem", [QUEUE_MEM])]
    order_lines = []
    for nm, qs in specs:
        qs = [q for q in qs if q in out]
        sp = M.listener_spec("R83-%s" % nm, [qd(q) for q in qs], results_root, min_free_gb, 1 if nm == "mem" else workers)
        fn = "listener_spec_%s.json" % nm
        for blk in sp["stages"] + sp["verify"]:
            blk["cmd"] = [c.replace("%s", fn).replace("r74_manifest", "r83_manifest") for c in blk["cmd"]]
        sp["raw"]["delete"] = sp["raw"]["delete"] + ["iekf_factor_iteration.csv", "lidar_msg_log.csv", "post_calibration_voxel_observation_support.csv", "line_rows_per_scan.csv", "time_cov.csv"] + RAW_ADD
        json.dump(sp, open(os.path.join(out_dir, fn), "w"), indent=1)
        order_lines.append("%s\t%s\t%d arms\t%s" % (fn, ",".join(qs), sum(len(out[q]) for q in qs), "concurrency 1 (dispatch ONE job at a time; memory watcher aborts a job above half of the host RAM)" if nm == "mem" else "default"))
    open(os.path.join(out_dir, "queue_order.tsv"), "w").write("# listener spec\tqueues\tarms\tdispatch concurrency   (id FIRST and alone; then the per-bag specs, hilti and mem: a preference, not a dependency, except id: rule 1.16 and the HOLD rule; mem is dispatched serially)\n" + "\n".join(order_lines) + "\n")
    if onsets_src:
        shutil.copyfile(onsets_src, os.path.join(out_dir, "onset_scans.json"))
    else:
        json.dump({"onset_scan_used": onsets, "source": "phase_summary.csv of the R81 BASE arms r81_rt_mg_<bag>_fq (onset_scan_used)"}, open(os.path.join(out_dir, "onset_scans.json"), "w"), indent=1)
    if grid_bag_src:
        shutil.copyfile(grid_bag_src, os.path.join(out_dir, "grid_bag.json"))
    else:
        json.dump(grid_bag, open(os.path.join(out_dir, "grid_bag.json"), "w"), indent=1)
    RO = required_outputs(table)
    json.dump(RO, open(os.path.join(out_dir, "required_outputs.json"), "w"), indent=1)
    LG = ledger(RO, table)
    json.dump(LG, open(os.path.join(out_dir, "r83_file_size_ledger.json"), "w"), indent=1)
    write_compliance(out_dir, ALL, table)
    byid = {r["id"]: r for r in ALL}
    tot_exp = sum(e["expected_bytes"] for e in RO if e["where"] == "return")
    tot_led = sum(x["max_total_bytes"] for x in LG if x["note"] != "non-arm file")
    plan, plan_led = R81.part_plan(tot_exp), R81.part_plan(tot_led)
    info = dict(counts=counts_txt, total=len(table), queues={b: len(out[b]) for b in qlist}, expected_return_bytes=tot_exp, ledger_return_bytes=tot_led, plan=plan, plan_ledger=plan_led,
                groups=group_counts(ALL))
    if not quiet:
        print(counts_txt)
        print("arms %d; queues %d: %s; listener specs %d" % (len(table), len(qlist), ", ".join("%s=%d" % (b, len(out[b])) for b in qlist), len(specs)))
        print("expected return %.1f MB; ledger ceiling %.1f MB; exceeds 30 MB: %s; parts at 28 MiB: %d / %d" % (tot_exp / 1e6, tot_led / 1e6, plan["exceeds_30MB"], plan["parts_split_28M"], plan_led["parts_split_28M"]))
    verify(out, table, ALL, byid)
    return info


def verify(out, table, ALL, byid):
    ids = [t["arm"] for t in table]
    assert len(ids) == len(set(ids)), "arm ids not unique"
    for b, lst in out.items():
        assert len({j["job_id"] for j in lst}) == len(lst), b
        for j in lst:
            if j.get("dataset") == "slam_2022_handheld":
                assert j["overrides"].get(K_ODOM) is True, "Hilti job without outputs.odom.export: " + j["job_id"]
    for t in table:
        assert t["asked"].strip(), t["arm"]
        assert t["mode"] in ("dc", "rt"), t["arm"]
        r = byid[t["arm"]]
        if t["mode"] == "rt":
            assert r["delta"].get("cbk.lidar.overlap_policy") == "retime_trim", t["arm"]
        else:
            assert "cbk.lidar.overlap_policy" not in r["delta"], t["arm"]
        assert not any("perturb" in k.lower() for k in r["delta"]), t["arm"]
        if t["arm"] != "r83_dc_refuse_zupt_removed":
            assert not any("zupt" in k.lower() for k in r["delta"]), "ZUPT key in " + t["arm"]
    assert any(t["batch"] == "r83_id" for t in table), "no r83_id queue"
    assert {t["batch"] for t in table if t["group"] == "D4P"} == {QUEUE_MEM}
    assert all(t["group"] == "D4P" for t in table if t["batch"] == QUEUE_MEM)


def write_compliance(out_dir, ALL, table):
    with open(os.path.join(out_dir, "COMPLIANCE_TABLE.md"), "w") as f:
        f.write("# R83 compliance table (rule 1.10 item 2, 1.12)\n\nMarks: MATCHES = literally what Bryce said; OWN = the planner's rule arm (needs his sight); ASSUMPTION = a value or choice he did not give; ASSUMPTION-READING = the family itself is the planner's reading of his message (both quoted). The reason is in the last column.\n\n"
                "| arm | Bryce's words | mode | source arm | mark | note |\n|---|---|---|---|---|---|\n")
        for t in table:
            r = next(x for x in ALL if x["id"] == t["arm"])
            g, mark, note = r["group"], "MATCHES", ""
            words = "OWN rule (identity / determinism / smoke / refuse); not asked for in Bryce's words" if g == OWN_GROUP else ASK[g]
            if g == OWN_GROUP:
                mark = "OWN"
            elif g == "DST":
                note = "inside the approved design the planner chose: margin %d s on top of onset + 70 scans (duration %s s), stationary dump frames evenly spaced from scan %d to onset - %d, around-onset frames onset -1, 0, +1, +30; frame index taken equal to scan index" % (DST_MARGIN_S, t["duration_s"], DST_STAT_LO, DST_STAT_GAP)
            elif g == "MXL":
                note = "no R82 arm reused (decision of 2026-10-07 overrides the earlier 'may be reused')"
            elif g == "VSW":
                mark, note = "ASSUMPTION-READING", "'the other mode' read as dc (overlap_policy discard) because Tier A ran rt; ds first as Tier A; 13 edges 0.20..0.80 step 0.05; 0.425 is off the grid, so no 0.425 arm (controls are the R82 dc first arms)"
            elif g == "D1B":
                note = "outputs/profiler/export_path set to the arm's own folder (profiler.txt) so imu/window_cov_profile is archived: a measurement key, not asked for in words; 72 needs both Q (Bryce wrote '72 arms if fq and mq'); lio/point_cov/mode left at its default; time_cov explicitly off"
            elif g == "D4P" and r.get("own_dump"):
                mark, note = "OWN", "eee_01 D4P arm with lio/residual_dump_frames (the 9 DST frames of eee_01) and lio/residual_dump_voxel_info added for points per voxel: Bryce did not ask for the dump in words (he asked for memory care on D4P); settings otherwise as the D4P family"
            elif g == "D4P":
                note = "settings = the R82 pcls arm (hard absolute lines, distance rows, gap floor, classify_at_fit, reasons) plus the two memory keys; the R82 pcls arm (normal, max 50) is the cross-batch reference, not re-run"
            elif g == "MQS":
                mark, note = "ASSUMPTION-READING", "'arm family' read as gyro_scale x acc_scale crossed (4 x 3 = 12 per bag, 108); Bryce to correct if he meant gyro only (36 arms); caps stay at the base job's 1e6; the (10, 0.3) cell equals the R82 mq control and is run anyway"
            f.write("| %s | %s | %s | %s | %s | %s |\n" % (t["arm"], words.replace("|", "/"), t["mode"], t["src_arm"] or "(none)", mark, note))


# ---------------------------------------------------------------------------------------------------------------------------------- selftest
def selftest():
    J = R81.stub_jobs(); g, gb = R81.stub_grids()
    onsets = {b: 300 + 10 * i for i, b in enumerate(BAGS)}
    onsets.update({"eee_01": 485, "eee_03": 310, "sbs_03": 340})
    d = tempfile.mkdtemp(prefix="r83_selftest_")
    info = build(None, J, g, gb, onsets, d, "/root/catkin_ws/livo_recon_results", quiet=True)
    print(info["counts"])
    n_ok = [0]

    def ok(c, m):
        assert c, m
        n_ok[0] += 1
    tab = list(csv.DictReader(open(os.path.join(d, "arm_table.tsv")), delimiter="\t"))
    ok(info["total"] == sum(EXPECT.values()) + OWN_EXPECT == len(tab) == 506, "total 506")
    ok(list(tab[0].keys())[:31] == R81_R82_COLS and list(tab[0].keys())[31:] == ["control", "r82_control", "alignfree", "concurrency", "rss_abort_frac", "dump_frames", "same_batch_control"], "arm_table columns: R82 set first, new ones at the end")
    for gname, want in dict(EXPECT, OWN=OWN_EXPECT).items():
        ok(sum(1 for t in tab if t["group"] == gname) == want, "group count " + gname)
    ok(len({t["arm"] for t in tab}) == len(tab), "names unique")
    jobs, qjobs = {}, {}
    for fn in os.listdir(d):
        if fn.startswith("jobs_"):
            lst = json.load(open(os.path.join(d, fn)))
            qjobs[fn[5:-5]] = lst
            for j in lst:
                jobs[j["job_id"]] = j["overrides"]
    ok(len(jobs) == 506, "jobs total")
    # queue membership counts
    qc = {q: len(v) for q, v in qjobs.items()}
    ok(qc["r83_id"] == 10 and qc["r83_hilti"] == 1 and qc[QUEUE_MEM] == 24, "id 10, hilti 1, mem 24")
    for b in BAGS:
        ok(qc["r83_" + b] == (55 if b in DST_BAGS else 51), "queue size " + b)
    ok(sum(qc.values()) == 506 and len(qc) == 12, "12 queues")
    ok(all(t["batch"] == QUEUE_MEM for t in tab if t["group"] == "D4P"), "D4P in mem")
    for fn in ("listener_spec_id.json", "listener_spec_hilti.json", "listener_spec_mem.json") + tuple("listener_spec_%s.json" % b for b in BAGS):
        sp = json.load(open(os.path.join(d, fn)))
        ok(sum(x["expected_jobs"] for x in sp["queues"]) == len(qjobs[sp["queues"][0]["dir"].split("/")[-1]]), "expected_jobs " + fn)
        ok("point_cov_window.csv" in sp["raw"]["delete"] and "peak_rss.txt" in sp["raw"]["delete"] and "time_cov.csv" in sp["raw"]["delete"], "raw list " + fn)
    mem = json.load(open(os.path.join(d, "listener_spec_mem.json")))
    ok(mem["queues"][0]["concurrency"] == 1 and mem["queues"][0]["serial"] is True and mem["limits"]["workers"] == 1, "mem serial flag")
    q = open(os.path.join(d, "queue_order.tsv")).read().splitlines()
    ok(q[1].startswith("listener_spec_id.json") and q[-1].startswith("listener_spec_mem.json") and "concurrency 1" in q[-1], "queue_order")
    # no ZUPT, all new keys
    ok(not any("zupt" in k for n_, o in jobs.items() if n_ != "r83_dc_refuse_zupt_removed" for k in o), "no ZUPT keys")
    ok("lio.zupt.mode" in jobs["r83_dc_refuse_zupt_removed"], "zupt refuse arm")
    ok(all(v.get(K_ODOM) is True for k, v in jobs.items() if k.startswith("r83_dc_sm_hl")), "Hilti outputs.odom.export")
    h = jobs["r83_dc_sm_hl_exp04_60s"]
    ok(h["evo.dense_gt_metrics"] is True and not any(k.startswith("cbk.lidar.point_time") for k in h) and h["imu.process_noise.model"] == "fixed", "Hilti smoke keys")
    ok([j for j in qjobs["r83_hilti"]][0]["duration_secs"] == 60 and qjobs["r83_hilti"][0]["dataset"] == "slam_2022_handheld", "Hilti job")
    # VSW
    vsw = sorted({round(jobs[k]["voxel_map.map.voxel_size"], 2) for k in jobs if k.startswith("r83_dc_sw_")})
    ok(vsw == [round(0.2 + 0.05 * i, 2) for i in range(13)] and len(vsw) == 13 and vsw[0] == 0.2 and vsw[-1] == 0.8, "VSW edges exactly 13, 0.20..0.80")
    ok(all("cbk.lidar.overlap_policy" not in jobs[k] and jobs[k]["cbk.lidar.point_filter_num"] == 3 and jobs[k]["imu.ds.mode"] == "first" for k in jobs if k.startswith("r83_dc_sw_")), "VSW dc, pfn 3, ds first")
    ok("r83_dc_sw_sbs_03_off_v0p25_pfn3_mq" in jobs and "r83_dc_sw_sbs_03_off_v0p425_pfn3_mq" not in jobs, "VSW names")
    # DST
    for bag in DST_BAGS:
        for ds in ("first", "off"):
            for qq in ("fq", "mq"):
                o = jobs["r83_rt_dst_%s_%s_%s" % (bag, ds, qq)]
                dur = next(j for j in qjobs["r83_" + bag] if j["job_id"] == "r83_rt_dst_%s_%s_%s" % (bag, ds, qq))["duration_secs"]
                fr = [int(x) for x in o[K_DUMP].split(",")]
                ok(dur == dst_duration_s(onsets[bag]) and len(fr) == 9 and max(fr) <= dur * 10 and min(fr) > 10, "DST duration and frames " + bag)
                ok(all(f <= onsets[bag] - 20 for f in fr[:5]) and fr[5:] == [onsets[bag] - 1, onsets[bag], onsets[bag] + 1, onsets[bag] + 30], "DST frames " + bag)
                ok(o[K_DUMP_VI] is True and o["voxel_map.plane.plane_fit_reasons_en"] is True and o["voxel_map.plane.plane_fit_trace_frames"] == 10 and o[K_MAXL] == 2 and o["cbk.lidar.overlap_policy"] == "retime_trim", "DST keys")
                ok((o["imu.ds.mode"], "imu.ds.ds_leaf_size" in o and o["imu.ds.ds_leaf_size"]) == (ds, None if ds == "off" else 0.1), "DST ds " + ds)
    ok([dst_duration_s(485), dst_duration_s(310), dst_duration_s(340)] == [59, 41, 44], "DST durations of the R82 onsets")
    # MXL, D1B, MQS, D4P
    ok(all(jobs["r83_rt_ml_%s_l%d_%s" % (b, ml, qq)][K_MAXL] == ml for b in BAGS for ml in (1, 2) for qq in ("fq", "mq")), "MXL max_layer")
    ok(all(jobs["r83_rt_pc_%s_w%dms_%s" % (b, ms, qq)][K_WIN] == ms / 1000.0 and jobs["r83_rt_pc_%s_w%dms_%s" % (b, ms, qq)][K_TC_MODE] == "off" and "lio.point_cov.mode" not in jobs["r83_rt_pc_%s_w%dms_%s" % (b, ms, qq)]
           for b in BAGS for ms in WINDOWS_MS for qq in ("fq", "mq")), "D1B window keys")
    ok(sorted({jobs[k][K_WIN] for k in jobs if k.startswith("r83_rt_pc_")}) == [0.005, 0.01, 0.02, 0.05], "D1B values")
    mqs = [k for k in jobs if k.startswith("r83_rt_mqs_")]
    ok(len(mqs) == 108 and {jobs[k][K_GYR] for k in mqs} == set(GYRO) and {jobs[k][K_ACC] for k in mqs} == set(ACC), "MQS values")
    ok(all(jobs[k]["imu.process_noise.model"] == "isotropic" for k in mqs) and not any("max_dynamic_variance" in kk for k in {kk for k in mqs for kk in jobs[k]} - {kk for kk in jobs["r83_rt_ml_eee_01_l2_mq"]}), "MQS: caps are not overridden by the arm (the base mq job carries them)")
    ok("r83_rt_mqs_eee_01_g0p3_a0p15_mq" in jobs and "r83_rt_mqs_eee_01_g10_a0p3_mq" in jobs, "MQS names")
    for vname, conv, mx in D4P_VARIANTS:
        for b in D4P_BAGS:
            o = jobs["r83_rt_d4_%s_pc%s_fq" % (b, vname)]
            ok(o[K_CONV] == conv and o[K_PTS_MAX] == mx and o[K_CLASSIFY] is True and o[R9.GAP] is True and o["voxel_map.plane.line_residual.enabled"] is True and o["voxel_map.plane.line_residual.rows"] == "distance", "D4P keys")
    ok(sorted({(jobs[k][K_CONV], jobs[k][K_PTS_MAX]) for k in jobs if "_d4_" in k}) == [("always_update", 50), ("always_update", 200), ("normal", 100), ("normal", 200)], "D4P four variants")
    by = {t["arm"]: t for t in tab}
    dk = [k for k in jobs if K_DUMP in jobs[k] and "_d4_" in k]
    ok(len(dk) == 8 and all("_d4_eee_01_" in k and jobs[k][K_DUMP_VI] is True for k in dk) and sum(1 for t in tab if t["group"] == "D4P" and t["dump_frames"]) == 8, "D4P dump keys on exactly the 8 eee_01 arms")
    ok(jobs["r83_rt_d4_eee_01_pca50_fq"][K_DUMP] == jobs["r83_rt_dst_eee_01_first_fq"][K_DUMP], "D4P dump frames equal the DST frames of eee_01")
    pk = [k for k in jobs if K_PROF in jobs[k]]
    ok(len(pk) == 72 and all(k.startswith("r83_rt_pc_") and jobs[k][K_PROF] == jobs[k]["outputs.path"] + "/profiler.txt" for k in pk), "profiler export path on the 72 D1B arms only")
    ok(all(by[k]["control"] == "r82_rt_ds_%s_first_%s" % (by[k]["bag"], k[-2:]) for k in by if by[k]["group"] in ("D1B", "MXL")), "D1B and MXL control = R82 rt first arm")
    ok(all(by[k]["control"] == "r82_rt_ds_%s_first_mq" % by[k]["bag"] for k in by if by[k]["group"] == "MQS"), "MQS control = R82 rt first mq arm")
    ok(all(by[k]["control"] == "r83_rt_dst_%s_first_%s" % (by[k]["bag"], k[-2:]) for k in by if by[k]["group"] == "DST" and "_off_" in k), "DST off control = DST first, same bag and Q")
    ok(all("profiler.txt" in json.load(open(os.path.join(d, "listener_spec_eee_01.json")))["raw"]["delete"] for _ in [0]), "profiler.txt in raw.delete")
    ok(by["r83_rt_d4_eee_01_pcn200_fq"]["concurrency"] == "1" and by["r83_rt_d4_eee_01_pcn200_fq"]["rss_abort_frac"] == "0.5", "D4P concurrency column")
    ok("peak_rss.txt" in by["r83_rt_d4_eee_01_pca200_mq"]["requires"].split() and "plane_fit_reasons.csv" in by["r83_rt_d4_eee_01_pca200_mq"]["requires"].split(), "D4P requires")
    ok("motion_q_scan.csv" in by["r83_rt_mqs_nya_01_g3_a0p1_mq"]["requires"].split() and "point_cov_window.csv" in by["r83_rt_pc_nya_01_w20ms_fq"]["requires"].split(), "MQS/D1B requires")
    ok("residual_dump.csv" in by["r83_rt_dst_eee_01_off_mq"]["requires"].split() and "plane_fit_reasons.csv" in by["r83_rt_dst_eee_01_off_mq"]["requires"].split(), "DST requires")
    # OWN
    ok(by["r83_dc_id_eee_01_fq"]["ref"] == "r82_dc_ds_eee_01_first_fq" and by["r83_rt_id_eee_01_mq"]["ref"] == "r82_rt_ds_eee_01_first_mq" and by["r83_rt_id_pc0_eee_01_fq"]["ref"] == "r82_rt_ds_eee_01_first_fq", "identity refs")
    ok(jobs["r83_rt_id_pc0_eee_01_fq"][K_WIN] == 0.0 and K_WIN not in jobs["r83_rt_id_eee_01_fq"] and by["r83_dc_id_eee_01_fq"]["batch"] == "r83_id", "pc0 identity")
    ok(by["r83_own_det_x_nya_02"]["ref"] == "r83_rt_ml_nya_02_l2_fq" and by["r83_own_det_x_nya_02"]["batch"] == "r83_nya_02", "determinism repeat")
    idk = {k: v for k, v in jobs["r83_dc_id_eee_01_fq"].items() if k not in ("outputs.path", "outputs.debug_log_dir", "eval.state_trace_run_id")}
    ok(not any(k.startswith(("cbk.lidar.overlap", "lio.point_cov.window", K_DUMP, K_CONV, K_PTS_MAX)) for k in idk), "identity arm carries no new key")
    ok(jobs["r83_dc_refuse_window_s_neg"][K_WIN] == -0.01 and jobs["r83_dc_refuse_window_s_big"][K_WIN] == 0.2 and by["r83_dc_refuse_window_s_big"]["arm_kind"] == "refuse", "refuse arms")
    ok(not any("conflict" in k for k in jobs), "no point_cov/time_cov conflict arm (the patch does not refuse that combination)")
    # required outputs and ledger
    RO = json.load(open(os.path.join(d, "required_outputs.json")))
    af = [e for e in RO if e["file"] == "alignfree_summary.csv"][0]
    ok(len(af["arms"]) == 228 and af["min_rows"] == 228 and all(by[x]["group"] in ALIGNFREE_GROUPS for x in af["arms"]), "alignfree rows only for D1B, MXL, MQS, DST (228)")
    ok(all(by[x]["alignfree"] == "1" for x in af["arms"]) and sum(1 for t in tab if t["alignfree"] == "1") == 228, "alignfree column")
    ok(not any(e["min_rows"] == 532 for e in RO), "no inherited 532 rows")
    ok([e for e in RO if e["file"] == "r83_cov_rows_all.csv"][0]["min_rows"] == 90, "cov rows: 72 D1B + 18 max_layer 2 controls")
    ok({e["file"] for e in RO} >= {"point_cov_window.csv", "peak_rss.txt", "ABORTED_RSS", "motion_q_scan.csv", "residual_dump_all.csv.gz", "plane_fit_reasons.csv", "r83_cov_rows_all.csv"}, "new required files")
    LG = json.load(open(os.path.join(d, "r83_file_size_ledger.json")))
    ok(any(x["pattern"] == "pos_trace_all.csv.gz" for x in LG) and all(x["max_total_bytes"] > 0 for x in LG), "ledger")
    for fn in ("onset_scans.json", "grid_bag.json", "COMPLIANCE_TABLE.md", "required_outputs.json", "r83_file_size_ledger.json", "queue_order.tsv", "arm_table.tsv"):
        ok(os.path.exists(os.path.join(d, fn)), fn)
    comp = open(os.path.join(d, "COMPLIANCE_TABLE.md")).read().splitlines()
    ok(sum(1 for l in comp if l.startswith("| r83_")) == 506 and all(l.rstrip().split("|")[-3].strip() in ("MATCHES", "OWN", "ASSUMPTION", "ASSUMPTION-READING") for l in comp if l.startswith("| r83_")), "every arm has a mark")
    ok(all("ASSUMPTION-READING" in l for l in comp if l.startswith("| r83_dc_sw_") or l.startswith("| r83_rt_mqs_")) and "run all of the arms that we ran in the other mode" in "\n".join(comp), "VSW and MQS marked ASSUMPTION-READING with Bryce's words")
    # guards
    C = R81.Ctx(J, g, gb, onsets); ALL = arms(C)
    try:
        check_counts([a for a in ALL if a["group"] != "D1B"], die=lambda m: (_ for _ in ()).throw(RuntimeError(m)))
        raise SystemExit("selftest FAILED: a short arm list was not refused")
    except RuntimeError as e:
        ok("COUNT MISMATCH" in str(e), "count mismatch refused")
    J2 = R81.stub_jobs(); J2["00_id_baseline_full"]["overrides"]["x.perturb_y"] = 1
    try:
        build(None, J2, g, gb, onsets, tempfile.mkdtemp(), "/r", quiet=True)
        raise SystemExit("selftest FAILED: perturbation key not refused")
    except SystemExit as e:
        ok("perturbation" in str(e), "perturbation refused")
    d2 = tempfile.mkdtemp(prefix="r83_selftest2_")
    build(None, J, g, gb, onsets, d2, "/root/catkin_ws/livo_recon_results", quiet=True)
    ok(all(open(os.path.join(d, f), "rb").read() == open(os.path.join(d2, f), "rb").read() for f in sorted(os.listdir(d))), "two builds identical")
    print("selftest checks passed: %d" % n_ok[0])
    print("failures: 0")
    print("selftest OK (synthetic job list and grids; planner sandbox, generator output only, not evidence)")


R81_R82_COLS = ["arm", "tier", "arch", "kind", "duration_s", "seq", "notes", "keep_raw", "dup_of", "r66_id", "group", "batch", "sampling", "requires", "f417", "plane_check", "asked", "early", "pfn", "pe", "dump", "bag", "win", "msglog",
                "mode", "fam", "src_arm", "arm_kind", "ref", "needles", "flag"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--jobs"); ap.add_argument("--grid-retime"); ap.add_argument("--grid-bag"); ap.add_argument("--phase-summary"); ap.add_argument("--onsets")
    ap.add_argument("--out", default="r83_manifest"); ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--min-free-gb", type=float, default=150.0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.jobs and a.grid_retime and a.grid_bag):
        ap.error("--jobs, --grid-retime and --grid-bag are required")
    J = {j["job_id"]: j for j in json.load(open(a.jobs))}
    onsets_src = None
    if a.phase_summary:
        onsets = {}
        for row in csv.DictReader(open(a.phase_summary)):
            for pre in ("r81_rt_mg_", "r80_mg_"):
                if row["arm"].startswith(pre) and row["arm"].endswith("_fq"):
                    onsets.setdefault(row["arm"][len(pre):-3], int(float(row["onset_scan_used"])))
    elif a.onsets:
        onsets = json.load(open(a.onsets))["onset_scan_used"]
        onsets_src = a.onsets
    else:
        ap.error("--phase-summary or --onsets is required")
    for b in DST_BAGS:
        print("onset %s: %s -> DST duration %s s, dump frames %s" % (b, onsets.get(b), dst_duration_s(onsets[b]), ",".join(str(f) for f in dst_frames(onsets[b]))))
    build(a, J, json.load(open(a.grid_retime)), json.load(open(a.grid_bag)), onsets, a.out, a.results_root, a.min_free_gb, a.workers, onsets_src=onsets_src, grid_bag_src=a.grid_bag)


if __name__ == "__main__":
    main()
