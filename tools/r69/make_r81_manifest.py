#!/usr/bin/env python3
"""R81: job files, arm_table.tsv, listener specs, queue order, required_outputs.json, size ledger and compliance table.

    python3 tools/r69/make_r81_manifest.py --jobs r69_inputs/r66_jobs.json --grid-retime r80c_grid.json --grid-bag r81_manifest/grid_bag.json \
        --phase-summary <R80 phase_summary.csv>  (or --onsets r81_manifest/onset_scans.json) --out r81_manifest --results-root /root/catkin_ws/livo_recon_results
    python3 tools/r69/make_r81_manifest.py --selftest

Planning agent's script (Agent B1); NOT run on the node by the planner. It runs in the planner sandbox (generator output only, not evidence).
Reuses make_r80_manifest (as R80: the R80 arm list, R80.mk, req_of, ledger helpers, the R9 / M helper chain) from the same directory (the R81 tree's tools/r69). The R80D helpers it needs
(make_r80d_manifest.make / src_arms pattern and the retime grid keys) are copied into this file, so nothing of R80D is imported.

Every arm is built from its R80 source arm (same bag, Q, pfn, voxel, lines) plus the new delta; rt arms (overlap_policy retime_trim) additionally get the retime grid keys exactly as R80D `make()`.
Overlap modes: dc = overlap_policy discard (Bryce's "none", the default, no extra keys), rt = retime_trim.
Arm id: r81_<mode>_<family>_<rest>. A rerun of R80 arm r80_X is r81_rt_X. OWN arms (identity, determinism, smoke, refuse) are r81_dc_id_* / r81_rt_id_* / r81_own_*.
Queues: r81_id (identity, smoke, refuse; runs first), r81_<bag> (whole-bag arms), r81_<bag>_s (short arms). One listener spec per bag covers both its queues (rule 1.16); listener_spec_id.json covers r81_id.
Internal arm keys: "mode" keeps the R80 meaning (line mode off|hardabs); the overlap mode is "ovl" (written to the arm_table column `mode`).
"""
import argparse, copy, csv, json, math, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.environ.get("R81_TREE_TOOLS", ""), "/home/claude/work/R81/tree/tools/r69"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
import make_r80_manifest as R80  # noqa: E402

R9, M = R80.R9, R80.M
BAGS = list(R80.BAGS)
N_SCANS = R9.N_SCANS
LN2 = 0.6931471805599453
REF_OFFSET = -0.1
MSG_COLS_C = R80.MSG_COLS + ["retime_shift_s", "n_trimmed"]
POS_COLS = R80.POS_COLS
EDGES_SWT = [round(0.25 + 0.05 * i, 2) for i in range(8)]            # 0.25 .. 0.60
RR_GROUPS = ("GP", "B3", "B1", "MG", "ZD", "NG", "PC", "DB", "DP", "EC3", "SC")
RRX_GROUPS = ("DP", "EC3", "SC")                                      # flag RRX: omitted from the planner's first count, Bryce may drop them
BASE_GROUPS = ("B3", "B1", "MG")                                      # lidar_msg_log_all.csv.gz is returned for these and the identity arms
N479_BAGS = ("eee_02", "sbs_03", "nya_01", "sbs_01")
N468_BAGS = ("eee_01", "nya_01")
N474_NS = (200, 400, 800, 1600, 3850)
VMIN = (4.0e-4, 4.5e-4)
VMIN_485 = 4.0e-4
S2 = (2.0e-4, 4.0e-4)
F2B_FRAMES = ",".join(str(i) for i in range(400, 411))
FRAMES20 = ",".join(str(i) for i in range(1, 21))
EXPECT = {"SWT": 576, "RR": 119, "N478": 36, "N467": 36, "N470": 96, "N485B": 8, "N469": 4, "N468": 32, "N485C": 32, "N474": 720, "N479": 24}   # SPEC arm-group table
Q_NAME = R80.Q_NAME
OWN = R80.OWN_R80

ASK_MODES = ("Bryce (2026-10-05): 'run every new R81 task in both modes' (discard = his 'none', and retime_trim); 'retime_trim is the standard fix for the dropped lidar data', the policy is set per arm, never through the config default")
ASK = {
    "SWT": "Bryce (2026-10-05): D1 voxel sweep rerun under retime_trim, 'all 8 configs (Q 2 x line 2 x pfn 2) x 9 bags x edges 0.25 to 0.60 m step 0.05 (8 edges)'; 'The 0.05 edge step is kept, 0.425 m is not run. Confirmed.'; 'D1 refine round: replaced by the 576-arm subset'",
    "RR": "Bryce (2026-10-05): 'rerun all completed R80 groups under retime_trim against their R80 controls' (the R80 dc arm of the same name is the control; TAIL-1 shelved by him earlier)",
    "N478": "Bryce (2026-10-05): 'For F478, I suppose we can try your proposed value of 0.1.'; 'For F478. I say you can use 2.' (clip ln 2); same 18 arms as F-476 (nine bags, both Q, whole bag, 0.425 m, pfn 3, lines off), controls BASE3; both overlap modes",
    "N467": "Bryce (2026-10-05): 'For F467, I think we can try 0.25 and also try 0.2 to fill in the gap.' (lio/zupt/acc_thresh, detect_only, the nine D7 arms: fixed-Q, 0.425 m, lines off, pfn 3, whole bag); both overlap modes",
    "N470": "Bryce (2026-10-05): 'For F470, I do what all three treatments.' (ps, pp, pb; eight other bags x 2 Q, whole bag, 0.425 m, lines off, pfn 3; eee_01 done in R80); both overlap modes",
    "N485B": "Bryce (2026-10-05): 'For F485, I think, yes, we can do part one plus 2A first, and then you can have the four 2B dump arms.' (eee_01, 60 s, fq/mq x pfn 1/3, ground-row dump of frames 400-410 with voxel point count and distinct frames); both overlap modes",
    "N469": "Bryce (2026-10-05): F-469 'agreed as proposed' / 'F-469: keep': pca control arms with plane_fit_reasons_en and plane_fit_trace_frames 10 on BASE3 eee_01, fixed-Q and motion-Q; both overlap modes",
    "N468": "Bryce (2026-10-05): 'F-468 clamp: run both v_min = 4e-4 m^2 and 4.5e-4 m^2'; arm set confirmed (16 per overlap mode: eee_01 and nya_01 x fq/mq x pfn 1/3, whole bag)",
    "N485C": "Bryce (2026-10-05): F-485 Part 3 in R81, 'planner chooses s2'; then 's2 = 2e-4 and 4e-4 m^2, with v_min fixed at 4e-4 ... this is fine i accept' (no data-derived value); v_min 4e-4 m^2 fixed; eee_01 and nya_01 x fq/mq x pfn 1/3; both overlap modes",
    "N474": "Bryce (2026-10-05): F-474 'in R81', the early-bag grid on all nine bags: calibration samples {200,400,800,1600,3850} x ds {first, off} x pfn {1,3} x Q {fixed, motion}; short arms, ceil(ns/197.5+8) s; both overlap modes",
    "N479": "Bryce (2026-10-05): F-479 step 2 'in R81': process-noise switch arms fq->mq and mq->fq on eee_02, sbs_03, nya_01, sbs_01, whole bag, plus plain discard-mode controls; both overlap modes (the rt controls are the RR B3 / MG arms)",
}
WORDS = {k: v for k, v in ASK.items()}


# ---------------------------------------------------------------------------------------------------------------------------------- grids and arm builders
class Ctx:
    def __init__(self, J, grid_retime, grid_bag, onsets):
        self.J, self.gr, self.gb, self.onsets = J, grid_retime, grid_bag, onsets
        self.src = {a["id"]: a for a in R80.arms()}

    def grid(self, bag):
        """R80D rule: eee_03 from the bag fit, the eight R80C bags exactly as shipped in r80c_grid.json."""
        if bag == "eee_03":
            g = self.gb["eee_03"]
            return dict(g["retime"], seq0=g["seq0"])
        return self.gr[bag]

    def base3(self, bag, q):
        return self.src["r80_mg_%s_fq" % bag] if q == "fq" else self.src["r80_b3_%s_mq" % bag]

    def base1(self, bag, q):
        return self.src["r80_b1_%s_%s" % (bag, q)]

    def basep(self, bag, q, pfn):
        return self.base3(bag, q) if pfn == 3 else self.base1(bag, q)


def rt_keys(g):
    return {"cbk.lidar.overlap_policy": "retime_trim", "cbk.lidar.retime_t0_s": g["t0_s"], "cbk.lidar.retime_period_s": g["period_s"], "cbk.lidar.retime_seq0": g["seq0"],
            "cbk.lidar.retime_ref_offset_s": REF_OFFSET, "data_queues.log_lidar_msgs": True}


def queue_of(a):
    if a["fam_kind"] in ("id", "smoke", "refuse"):
        return "r81_id"
    return "r81_%s_s" % a["bag"] if a["dur"] else "r81_%s" % a["bag"]


def derive(C, src, aid, ovl, group, fam, extra, asked, purpose, pop=(), dur=None, kind="fb", ref=None, msglog=0, flag="", **flags):
    """One arm from its R80 source arm: same base job, same overrides, plus `extra` (and the retime keys when ovl == rt)."""
    b = copy.deepcopy(src)
    b["id"] = aid
    d = dict(b["delta"])
    for k in pop:
        d.pop(k, None)
    d.update(extra)
    if ovl == "rt":
        d.update(rt_keys(C.grid(b["bag"])))
    b["delta"] = d
    b.update(kind=kind, ref=ref, group=group, fam=fam, ovl=ovl, src_arm=src["id"], msglog=msglog, win=0, f417=0, dump=0, early=0, flag=flag, fam_kind=kind)
    b["tier"] = "EF" if dur else "FB"
    if dur:
        b["dur"] = dur
    else:
        b["dur"] = None
    b.update(flags)
    b["asked"] = asked + "; same arm as %s (%s, 0.425 m, lines off, point filter %s, %s Q) with overlap mode %s%s" % (
        src["id"], "first %s s" % b["dur"] if b["dur"] else "whole bag", b["pfn"], Q_NAME.get(b["q"], b["q"]), ovl, (", extra keys " + json.dumps(extra, sort_keys=True)) if extra else "")
    b["purpose"] = purpose
    b["batch"] = queue_of(b)
    return b


def both(C, src, mk_id, group, fam, extra, ask_key, purpose, **kw):
    """The same arm in both overlap modes (dc first)."""
    out = []
    for ovl in ("dc", "rt"):
        out.append(derive(C, src, mk_id(ovl), ovl, group, fam, extra, ASK[ask_key] + "; " + ASK_MODES.split(": ", 1)[1], "%s [%s]" % (purpose, ovl), **kw))
    return out


def arms(C):
    S = C.src
    A = []
    # ---- SWT: D1 sweep rerun, rt only, eight configs x nine bags x eight edges
    for bag in BAGS:
        for q in ("fq", "mq"):
            for line in R80.LINES:
                for pfn in R80.PFNS:
                    for e in EDGES_SWT:
                        src = S["r80_sw_%s_%s_v%s_pfn%d_%s" % (bag, line, M.tag(e), pfn, q)]
                        A.append(derive(C, src, "r81_rt_" + src["id"][len("r80_"):], "rt", "SWT", "SW", {}, ASK["SWT"], "R81 SWT: D1 rerun, %s %s %s pfn %d edge %s, retime_trim" % (bag, q, line, pfn, e)))
    # ---- RR: completed R80 groups under rt (the R80 dc arm is the control)
    for fam in RR_GROUPS:
        for src in [a for a in R80.arms() if a["group"] == fam]:
            A.append(derive(C, src, "r81_rt_" + src["id"][len("r80_"):], "rt", "RR", fam, {}, ASK["RR"], "R81 RR: rerun of %s under retime_trim" % src["id"],
                            dur=src["dur"], msglog=1 if fam in BASE_GROUPS else 0, flag="RRX" if fam in RRX_GROUPS else ""))
    # ---- N478: smooth NIS gain
    nis = M.GAIN("nis_smooth", nis_slope=0.1, nis_clip=LN2, decay=0.95, min_gain=0.25)
    for bag in BAGS:
        for q in ("fq", "mq"):
            A += both(C, C.base3(bag, q), lambda o, bag=bag, q=q: "r81_%s_nis_%s_%s" % (o, bag, q), "N478", "NIS", nis, "N478", "R81 F-478: smooth NIS gain, %s %s" % (bag, q))
    # ---- N467: ZUPT detect_only at acc_thresh 0.25 and 0.20
    for bag in BAGS:
        for a_, tg in ((0.25, "a025"), (0.20, "a020")):
            A += both(C, S["r80_zd_%s_fq" % bag], lambda o, bag=bag, tg=tg: "r81_%s_zt_%s_%s" % (o, bag, tg), "N467", "ZT", M.ZUPT("detect_only", a=a_), "N467",
                      "R81 F-467: ZUPT detect_only acc_thresh %s, %s fq" % (a_, bag), kind="fb")
    # ---- N470: pose covariance treatments on the eight other bags
    for bag in BAGS:
        if bag == "eee_01":
            continue
        for k in ("ps", "pp", "pb"):
            for q in ("fq", "mq"):
                A += both(C, C.base3(bag, q), lambda o, bag=bag, k=k, q=q: "r81_%s_pcx_%s_%s_%s" % (o, bag, k, q), "N470", "PCX", dict(R80.PC[k]), "N470", "R81 F-470: %s, %s %s" % (k, bag, q))
    # ---- N485B: ground-row dump frames 400..410 with the two new columns (source: the R80 DP window arms of the same pfn and Q)
    for pfn in R80.PFNS:
        for q in ("fq", "mq"):
            A += both(C, S["r80_dp_pfn%d_%s" % (pfn, q)], lambda o, pfn=pfn, q=q: "r81_%s_f2b_pfn%d_%s" % (o, pfn, q), "N485B", "F2B",
                      {"lio.residual_dump_frames": F2B_FRAMES, "lio.residual_dump_voxel_info": True}, "N485B", "R81 F-485 2B: ground-row dump frames 400-410 with voxel point_count and distinct_frames, pfn %d %s" % (pfn, q), dur=60)
    # ---- N469: pca control arms (the R80 DB arm without plane_fit_mode debiased)
    for q in ("fq", "mq"):
        A += both(C, S["r80_db_%s" % q], lambda o, q=q: "r81_%s_pca_%s" % (o, q), "N469", "PCA", {}, "N469", "R81 F-469: pca control with fit-reason log and trace frames 10, eee_01 %s" % q, pop=("voxel_map.plane.plane_fit_mode",))
    # ---- N468: plane_var_min clamp only; N485C: v_min 4e-4 and plane_var_scan_s2
    for bag in N468_BAGS:
        for q in ("fq", "mq"):
            for pfn in R80.PFNS:
                for v in VMIN:
                    tg = "vm%03d" % round(v * 1e6)
                    A += both(C, C.basep(bag, q, pfn), lambda o, bag=bag, q=q, pfn=pfn, tg=tg: "r81_%s_vmin_%s_%s_pfn%d_%s" % (o, bag, tg, pfn, q), "N468", "VMIN",
                              {"voxel_map.plane.plane_var_min": v}, "N468", "R81 F-468: plane_var_min %g, %s pfn %d %s" % (v, bag, pfn, q))
                for s2 in S2:
                    tg = "s%03d" % round(s2 * 1e6)
                    A += both(C, C.basep(bag, q, pfn), lambda o, bag=bag, q=q, pfn=pfn, tg=tg: "r81_%s_vs2_%s_%s_pfn%d_%s" % (o, bag, tg, pfn, q), "N485C", "VS2",
                              {"voxel_map.plane.plane_var_min": VMIN_485, "voxel_map.plane.plane_var_scan_s2": s2}, "N485C", "R81 F-485 Part 3: plane_var_min %g + scan_s2 %g, %s pfn %d %s" % (VMIN_485, s2, bag, pfn, q),
                              flag="")
    # ---- N474: early-bag grid, short arms
    for bag in BAGS:
        for ns in N474_NS:
            dur = int(math.ceil(ns / 197.5 + 8.0))
            for ds in ("first", "off"):
                for pfn in R80.PFNS:
                    for q in ("fq", "mq"):
                        ex = {"calib.stationary.num_samples": ns, "imu.ds.mode": ds, "lio.first_frame_dump_frames": 5, "lio.residual_dump_frames": FRAMES20, "lio.update_diag_en": True}
                        if ds == "off":
                            ex["imu.ds.ds_leaf_size"] = None                    # as the R72 / R74 EF cells (key removed by a null override)
                        A += both(C, C.basep(bag, q, pfn), lambda o, bag=bag, ns=ns, ds=ds, pfn=pfn, q=q: "r81_%s_ef_%s_n%d_ds%s_pfn%d_%s" % (o, bag, ns, ds, pfn, q), "N474", "EF", ex, "N474",
                                  "R81 F-474: early grid %s n%d ds %s pfn %d %s, %d s, dump frames 1-20" % (bag, ns, ds, pfn, q, dur), dur=dur, early=1, dump=20)
    # ---- N479: process-noise switch arms and plain discard controls
    for bag in N479_BAGS:
        onset = C.onsets[bag]
        mqj = C.J[C.base3(bag, "mq")["base"]]["overrides"]
        acc, gyr = mqj["imu.process_noise.motion.acc_scale"], mqj["imu.process_noise.motion.gyro_scale"]
        A += both(C, C.base3(bag, "fq"), lambda o, bag=bag: "r81_%s_swx_%s_fq2mq" % (o, bag), "N479", "SWX", M.switch(onset, "isotropic", acc, gyr), "N479",
                  "R81 F-479: %s fixed-Q until scan %d, motion-Q (acc %s, gyro %s) from the onset scan %d" % (bag, onset - 1, acc, gyr, onset))
        A += both(C, C.base3(bag, "mq"), lambda o, bag=bag: "r81_%s_swx_%s_mq2fq" % (o, bag), "N479", "SWX", M.switch(onset, "fixed"), "N479",
                  "R81 F-479: %s motion-Q until scan %d, fixed-Q from the onset scan %d" % (bag, onset - 1, onset))
        for q in ("fq", "mq"):
            A.append(derive(C, C.base3(bag, q), "r81_dc_swc_%s_%s" % (bag, q), "dc", "N479", "SWC", {}, ASK["N479"] + "; plain DISCARD-mode control of the switch arms (the rt controls are the RR B3 / MG arms of the bag)",
                            "R81 F-479 control: plain %s %s, discard" % (bag, q)))
    # ---- OWN: identity, determinism, smoke, refuse
    A += own_arms(C, A)
    for a in A:
        a["mode_line"] = a["mode"]
    return A


def own_arms(C, A):
    S = C.src
    out = []
    ID1, ID2 = "r81_dc_id_mg_eee_01_fq", "r81_rt_id_mg_eee_01_fq"
    a = derive(C, S["r80_mg_eee_01_fq"], ID1, "dc", "OWN", "ID", {}, OWN + "identity arm (1): every new key at its default, overlap_policy default, so it must be bit-identical to R80 r80_mg_eee_01_fq (state_trace and pos_trace digest within 0.005 mm)",
               "OWN identity: dc, every new key at its default, vs r80_mg_eee_01_fq", kind="id", ref="r80_mg_eee_01_fq", msglog=1)
    a["asked"] = OWN + "identity arm (1): r80_mg_eee_01_fq unchanged (same base job and overrides), every new R81 key at its default; its pos_trace and state_trace must equal R80 r80_mg_eee_01_fq within 0.005 mm"
    out.append(a)
    a = derive(C, S["r80_mg_eee_01_fq"], ID2, "rt", "OWN", "ID", {}, OWN + "identity arm (2)", "OWN identity: rt, vs the R80C retime_trim arm of the same src", kind="id", ref="r80c_rtt_mg_eee_01_fq", msglog=1)
    a["asked"] = OWN + "identity arm (2): r80_mg_eee_01_fq plus the R80D retime_trim keys (eee_01 grid from r80c_grid.json, retime_ref_offset_s -0.1 = the default, so identical to R80C); must equal R80C arm r80c_rtt_mg_eee_01_fq (state_trace and pos_trace within 0.005 mm)"
    out.append(a)
    # determinism: one repeat per bag queue of that bag's r81_rt_mg_<bag>_fq
    by = {x["id"]: x for x in A}
    for bag in BAGS:
        o = by["r81_rt_mg_%s_fq" % bag]
        d = copy.deepcopy(o)
        d.update(id="r81_own_det_x_%s" % bag, kind="det_fb", fam_kind="det_fb", ref=o["id"], group="OWN", fam="DET", msglog=0, flag="", src_arm=o["src_arm"])
        d["asked"] = OWN + "rule 3.7 determinism control (one repeat per bag queue, rule 1.16): an exact whole-bag repeat of %s; its state-trace digest must equal that arm's" % o["id"]
        d["purpose"] = "OWN determinism control: repeat of %s" % o["id"]
        out.append(d)
    # smoke arms, 60 s each (rule 3.6)
    sm = lambda aid, q, extra, why, **kw: _smoke(C, aid, q, extra, why, **kw)
    out.append(sm("r81_dc_sm_nis_60s", "mq", M.GAIN("nis_smooth", nis_slope=0.1, nis_clip=LN2, decay=0.95, min_gain=0.25), "imu/process_noise/gain/mode = nis_smooth (F-478): engagement line, q_adapt.csv non-empty and the gain moving"))
    out.append(sm("r81_dc_sm_vmin_60s", "fq", {"voxel_map.plane.plane_var_min": VMIN[0]}, "voxel_map/plane/plane_var_min (F-468): engagement line with a non-zero clamp count"))
    out.append(sm("r81_dc_sm_vs2_60s", "fq", {"voxel_map.plane.plane_var_min": VMIN_485, "voxel_map.plane.plane_var_scan_s2": S2[0]}, "voxel_map/plane/plane_var_scan_s2 with plane_var_min (F-485 Part 3): engagement line, scan term applied"))
    out.append(sm("r81_dc_sm_f2b_60s", "fq", {"lio.residual_dump_frames": "400,401,402", "lio.residual_dump_voxel_info": True}, "lio/residual_dump_voxel_info (F-485 2B columns): residual_dump.csv has voxel_point_count and voxel_distinct_frames, non-zero"))
    # refuse arms: invalid key combinations (rule 3.4), each must die at startup naming the key
    R = S["r80_refuse_rows_distance_smooth"]
    for suf, delta, needles in (
            ("nis_smooth_no_clip", dict(M.GAIN("nis_smooth", nis_slope=0.1, decay=0.95)), ["nis_clip > 0"]),
            ("nis_smooth_bad_slope", dict(M.GAIN("nis_smooth", nis_slope=-0.1, nis_clip=LN2, decay=0.95)), ["nis_slope > 0"]),
            ("nis_clip_without_mode", {"imu.process_noise.gain.nis_clip": LN2}, ["imu/process_noise/gain/nis_clip"]),
            ("plane_var_min_negative", {"voxel_map.plane.plane_var_min": -1.0e-4}, ["voxel_map/plane/plane_var_min"]),
            ("plane_var_scan_s2_negative", {"voxel_map.plane.plane_var_scan_s2": -1.0e-4}, ["voxel_map/plane/plane_var_scan_s2"])):
        r = copy.deepcopy(R)
        r.update(id="r81_dc_refuse_%s" % suf, delta=dict(delta), needles=needles, group="OWN", fam="REF", ovl="dc", src_arm="", flag="", fam_kind="refuse", batch="r81_id", kind="refuse", ref=None)
        r["asked"] = OWN + "rule 3.4 refusal: MUST FAIL at startup, text contains %s; its exit_status.txt must be non-zero. (The wording of the plane-variance and nis_clip-without-mode messages is Agent A's: the needle is the key name)" % " and ".join(repr(n) for n in needles)
        r["purpose"] = "MUST FAIL: %s" % suf
        out.append(r)
    return out


def _smoke(C, aid, q, extra, why, bag="eee_01"):
    a = R80.mk(aid, bag, "off", 0.425, q, "OWN", OWN + "rule 3.6 smoke arm for the new key: %s; 60 s, eee_01, 0.425 m, lines off, pfn 3, %s Q, overlap_policy default (dc)" % (why, Q_NAME[q]),
               "OWN smoke: %s" % why, pfn=3, extra=dict(extra), dur=R80.DP_DUR, batch="r81_id", tier="EF", kind="smoke")
    a.update(fam="SMK", ovl="dc", src_arm="(new key, no R80 source arm; R80 smoke pattern)", flag="", fam_kind="smoke", batch="r81_id", ref=None)
    return a


# ---------------------------------------------------------------------------------------------------------------------------------- count table
def group_counts(ALL):
    cnt = {}
    for a in ALL:
        cnt[a["group"]] = cnt.get(a["group"], 0) + 1
    return cnt


def check_counts(ALL, die=sys.exit):
    cnt = group_counts(ALL)
    rows, bad = [], False
    for g in list(EXPECT) + ["OWN"]:
        have, want = cnt.get(g, 0), EXPECT.get(g)
        ok = want is None or have == want
        bad |= not ok
        rows.append("%-6s spec %-6s real %-6d %s" % (g, "to count" if want is None else want, have, "" if ok else "MISMATCH"))
    extra = set(cnt) - set(EXPECT) - {"OWN"}
    if extra:
        bad = True
        rows.append("UNKNOWN GROUPS: %s" % sorted(extra))
    nown = cnt.get("OWN", 0)
    tot_spec = sum(EXPECT.values())
    rows.append("total  spec %d + OWN  real %d (non-OWN %d + OWN %d)" % (tot_spec, len(ALL), len(ALL) - nown, nown))
    table = "\n".join(rows)
    if bad:
        die("COUNT MISMATCH against the SPEC arm-group table. Real counts follow.\n" + table)
    return table


# ---------------------------------------------------------------------------------------------------------------------------------- required outputs and ledger
def nscan(t):
    n = N_SCANS[t["bag"]]
    d = t["duration_s"]
    return n if d == "full" else min(n, int(float(d) * 10) + 10)


def required_outputs(table):
    by = {t["arm"]: t for t in table}
    sel = lambda f: [t["arm"] for t in table if f(t)]
    nonref = sel(lambda t: t["arm_kind"] != "refuse")
    nonsw = sel(lambda t: t["arm_kind"] != "refuse" and t["group"] != "SWT")
    whole = lambda t: t["duration_s"] == "full"
    R = []

    def add(fu, title, arms_, file, cols, min_rows, per=None, where="return", source="arm", eb=None, maxb=400000):
        e = eb if eb is not None else (int(sum(per(by[a]) for a in arms_)) if (where == "return" and arms_ and per) else 0)
        R.append(dict(followup=fu, title=title, arms=arms_, file=file, columns=cols, min_rows=min_rows, where=where, source=source, contains={}, expected_bytes=int(e), max_bytes_per_file=maxb))
    # per-arm files
    add("F-484", "per-scan position, yaw and quaternion, every whole-bag arm except the sweep arms (SWT returns summaries only; a trace only when gt_max_mm > 500, a ceiling is ledgered)",
        [a for a in nonsw if whole(by[a])], "pos_trace_all.csv.gz", POS_COLS, 1500, per=lambda t: 56 * nscan(t), maxb=330000)
    add("F-484", "per-scan position, yaw and quaternion, every short arm", [a for a in nonsw if not whole(by[a])], "pos_trace_all.csv.gz", POS_COLS, 5, per=lambda t: 56 * nscan(t), maxb=330000)
    mg = sel(lambda t: t["msglog"] == "1")
    add("F-484", "per-message LiDAR log (17 columns incl. retime_shift_s, n_trimmed) of the BASE arms (B3, B1, MG) and the identity arms; the other arms log it in the archive", mg, "lidar_msg_log_all.csv.gz", MSG_COLS_C, 1500,
        per=lambda t: 100 * nscan(t), maxb=900000)
    add("F-484", "collector by-product per arm (line_rows_per_scan.csv), ledgered", nonsw, "line_rows_per_scan_all.csv.gz", [], 1, per=lambda t: 25000 if whole(t) else 10000, maxb=60000)
    add("F-452", "per-scan table of every non-sweep arm except the 720 N474 short arms (arm_scan.csv rows; R80 measured 116 bytes per row compressed; thinned by the collector --level to fit the ledger); the sweep arms and the N474 arms return phase_summary rows only (planner decision on B1 question 1)", [], "arm_scan_all.csv.gz", [], 5,
        eb=min(60000000, sum(120 * nscan(by[a]) for a in nonsw if by[a]["group"] != "N474")), source="generated", maxb=400000000)   # unthinned about 168 MB; the agent picks the r69_collect --level that brings it to at most 60 MB
    add("F-452", "phase_summary row of every arm (metrics, onset scan, collapse flags)", [], "phase_summary.csv", ["arm"], len(nonref), eb=4000 * len(nonref), source="generated", maxb=20000000)
    add("F-409", "engagement of every dispatched arm concatenated (includes the new key engagement lines)", [], "engagement_all.txt", [], 10, eb=1500 * len(table), source="generated", maxb=4000000)
    add("F-409", "per-arm completeness (arm_table 'requires' column)", [], "completeness.csv", ["arm", "requires_missing"], 10, eb=1200 * len(table), source="generated", maxb=3000000)
    add("F-326", "exit_status.txt of every dispatched arm: refuse arms non-zero, all others zero", sel(lambda t: True), "exit_status.txt", [], 1, where="archive")
    add("F-407", "engagement_imu.txt present for every dispatched arm that is not a refuse arm", nonref, "engagement_imu.txt", [], 1, where="archive")
    add("F-496", "state_reference.txt (the calibration state before IMU propagation: q_wxyz, p, v) of every whole-bag and short arm except the sweep arms (tiny, returned in compact/<arm>/ because the pos_trace holds only post-LIO rows); r81_alignfree.py and r81_early_grid_reduce.py use it as the first-frame reference (Bryce's metric rule); without it they fall back to the scan-1 posterior and flag every row",
        nonsw, "state_reference.txt", [], 1, per=lambda t: 400, maxb=4000)
    add("F-496", "alignment-free stationary drift, excursion, settled offset and departure per arm in each run's own first-frame coordinates (R81 version of r80d_alignfree.py)", [], "alignfree_summary.csv", [], len(nonref),
        eb=1500 * len(nonref), source="generated", maxb=8000000)
    # gates
    add("F-452", "gates of R81: identity, determinism, grid identities per mode, engagement per new key, refusals, exit status, completeness", [], "r81_gates.txt", [], 1, eb=1500000, source="generated", maxb=4000000)
    add("F-452", "identity comparison of the two identity arms against their references", [], "r81_identity_vs_old.csv", ["arm", "ref_arm", "n_common_scans", "pos_max_abs_diff_mm", "yaw_max_abs_diff_mrad"], 2, eb=3000, source="generated", maxb=40000)
    add("F-452", "determinism controls and their originals", [], "identity_controls.txt", [], 1, eb=40000, source="generated", maxb=150000)
    # group files
    add("F-467", "zupt.csv per frame of the detect_only arms (N467 acc 0.25 / 0.20 in both modes and the RR ZD arms); gzip copy of zupt.csv (R80 measured 83 kB per whole-bag arm)", sel(lambda t: t["fam"] in ("ZT", "ZD")), "zupt_all.csv.gz",
        ["frame", "gyro_mean", "acc_mean", "run", "stationary", "applied", "vnorm_before"], 100, per=lambda t: 125000, maxb=700000)
    add("F-478", "q_adapt.csv per scan of the nis_smooth arms (N478) and the RR NG arms; gzip copy (R80 measured 163 kB per arm)", sel(lambda t: t["fam"] in ("NIS", "NG")), "q_adapt_all.csv.gz", [], 100, per=lambda t: 245000, maxb=500000)
    add("F-403", "step_cap.csv.gz of the two RR SC arms (R80 measured 56 kB)", sel(lambda t: t["fam"] == "SC"), "step_cap.csv.gz",
        ["frame", "iter", "rot_raw", "pos_raw", "rot_applied", "pos_applied", "scale", "bound", "next_converged"], 20, per=lambda t: 85000, maxb=150000)
    n474 = sel(lambda t: t["group"] == "N474")
    add("F-474", "early-frame summary per arm and frame 1..20: position and yaw error against the first-frame calibration state, yaw step per iteration, converged, wall and ground residual counts, first frame with 1 / 10 / 100 wall residuals, "
        "believed yaw sd, excursion over scans 1-10 (reducer on the archived dumps; the raw dumps stay in the host archive)", [], "early_grid_summary.csv",
        ["arm", "frame", "pos_err_mm", "yaw_err_mrad", "yaw_step", "converged", "n_wall", "n_ground", "yaw_sd"], len(n474), eb=6000 * len(n474), source="generated", maxb=12000000)
    f2b = sel(lambda t: t["group"] == "N485B")
    add("F-485", "ground-row dump summary of frames 400-410 per arm with voxel_point_count and voxel_distinct_frames (the raw dump stays in the archive)", [], "f485_2b_summary.csv",
        ["arm", "frame", "voxel_point_count", "voxel_distinct_frames"], len(f2b), eb=40000 * len(f2b), source="generated", maxb=2000000)
    add("F-479", "switch pairs: per switch arm the scan at which it leaves control A and control B, collapse flag (reducer; alignment-free)", [], "r81_switch_pairs.csv", [], 8, eb=30000, source="generated", maxb=300000)
    add("F-479", "switch frames: per-scan departures of the switch arms from both controls", [], "r81_switch_frames.csv.gz", [], 100, eb=2500000, source="generated", maxb=6000000)
    add("F-480", "floor height per bag (analysis record, estimator unchanged)", [], "r81_floor_heights.csv", [], 9, eb=3000, source="generated", maxb=50000)
    for f, tt in (("r80_calib_window_edges.csv", "D6 (F-450 / F-448): host tool tools/r69/r80_calib_window_scan.py run UNCHANGED on the nine calibration-window clouds"), ("r80_calib_window_layers.csv", "D6: max-layer census"),
                  ("r80_calib_window_provenance.json", "D6: inputs, thresholds used, tool sha256")):
        add("F-450", tt, [], f, [], 1, eb={"r80_calib_window_edges.csv": 15000, "r80_calib_window_layers.csv": 60000}.get(f, 8000), source="generated", maxb=200000)
    add("F-469", "plane-fit trace counts of the debiased arms and the pca arms (tools/r69/plane_fit_trace_counts.py, run by the collector, not a manual step)", [], "plane_fit_trace_counts_all.csv", [], 5, eb=12000, source="generated", maxb=100000)
    return R


MEASURED = {"pos_trace_all.csv.gz", "line_rows_per_scan_all.csv.gz", "zupt_all.csv.gz", "q_adapt_all.csv.gz", "step_cap.csv.gz", "arm_scan_all.csv.gz"}   # sizes measured on the R80 return (allowance 1.5x), the rest 3x


def ledger(RO, table):
    n = len(table)
    out = []
    for e in RO:
        if e["where"] != "return" or not e["expected_bytes"]:
            continue
        mult = 1.5 if e["file"] in MEASURED else 3.0
        out.append({"pattern": e["file"], "max_bytes_per_file": e["max_bytes_per_file"], "max_total_bytes": int(e["expected_bytes"] * mult),
                    "basis": ("measured on the R80 return (pos_trace 56 bytes per scan, arm_scan 116 bytes per row, zupt 83 kB, q_adapt 163 kB, step_cap 56 kB, line_rows 15-24 kB per arm); allowance 1.5x" if e["file"] in MEASURED else
                              "estimate (message log measured 42 bytes per message against 100 here; new files unmeasured); allowance 3x"), "note": e["followup"] + " (%d arms)" % len(e["arms"])})
    res, at = [], {}                                     # one entry per pattern (pos_trace has a whole-bag and a short-arm requirement)
    for x in out:
        if x["pattern"] in at:
            y = res[at[x["pattern"]]]
            y["max_bytes_per_file"] = max(y["max_bytes_per_file"], x["max_bytes_per_file"]); y["max_total_bytes"] += x["max_total_bytes"]; y["note"] += " + " + x["note"]
        else:
            at[x["pattern"]] = len(res); res.append(dict(x))
    out = res
    for x in out:
        if x["pattern"] == "pos_trace_all.csv.gz":
            x["max_total_bytes"] += 150 * 56 * 4300      # sweep arms with gt_max_mm > 500 return their trace (collapse rule): ceiling 150 of the 576 arms at the longest bag
            x["basis"] += "; plus a ceiling for up to 150 SWT arms whose trace is returned because gt_max_mm > 500"
    for pat, mx, tot in (("summaries_all.json", 40000000, 16000 * n), ("stop_reasons.csv", 2000000, 400 * n), ("listener_*.log", 5000000, 40000000), ("R81_tree.tar.gz", 80000000, 80000000), ("agent_edits.patch", 2000000, 2000000),
                         ("raw_sha256.txt", 20000, 20000), ("compress_manifests.json", 100000, 100000), ("*.md", 1000000, 4000000), ("*.json", 2000000, 20000000), ("*.jsonl", 2000000, 10000000), ("*.txt", 1000000, 10000000),
                         ("*.tsv", 1000000, 3000000), ("*.csv", 3000000, 10000000), ("*.sha256", 100000, 100000), ("imu_timing/*", 4000000, 40000000)):
        out.append({"pattern": pat, "max_bytes_per_file": mx, "max_total_bytes": int(tot), "basis": "generic ceiling (R80 / R80D return sizes scaled to the arm count, or a round figure); not a measurement", "note": "non-arm file"})
    return out


def part_plan(total, threshold=30e6, part=28 * 1024 * 1024):
    return dict(total_bytes=int(total), exceeds_30MB=total > threshold, parts_split_28M=int(math.ceil(total / part)) if total > threshold else 1)


# ---------------------------------------------------------------------------------------------------------------------------------- build
def build(a, J, grid_retime, grid_bag, onsets, out_dir, results_root, min_free_gb=150.0, workers=6, quiet=False):
    for bag in BAGS:
        if bag not in grid_bag:
            sys.exit("grid-bag has no entry for %s" % bag)
        if grid_bag[bag]["seq_gaps"] != 0 or grid_retime[bag]["seq_gaps"] != 0:
            sys.exit("%s: seq_gaps != 0 in a grid file; stop (the retime grid assumes a gap-free seq)" % bag)
    for bag in N479_BAGS:
        if bag not in onsets:
            sys.exit("no onset scan for %s" % bag)
    os.makedirs(out_dir, exist_ok=True)
    C = Ctx(J, grid_retime, grid_bag, onsets)
    ALL = arms(C)
    counts_txt = check_counts(ALL)
    out, table, seen = {}, [], set()
    prio = {("RR", "MG"): 0, ("OWN", "DET"): 1, ("RR", "B3"): 2, ("RR", "B1"): 3, ("N478", None): 4, ("N467", None): 5, ("N470", None): 6, ("N479", None): 7, ("N468", None): 8, ("N485C", None): 9, ("N469", None): 10,
            ("N485B", None): 11, ("RR", None): 12, ("N474", None): 13, ("SWT", None): 14, ("OWN", None): 0}
    order = sorted(range(len(ALL)), key=lambda i: (ALL[i]["batch"] != "r81_id", prio.get((ALL[i]["group"], ALL[i]["fam"]), prio.get((ALL[i]["group"], None), 12)), i))
    for i in order:
        r = ALL[i]
        if r["base"] not in J:
            sys.exit("base job %s not in r66_jobs.json" % r["base"])
        assert r["id"] not in seen, "duplicate arm id " + r["id"]
        seen.add(r["id"])
        assert r.get("asked"), "arm without an asked line: " + r["id"]
        j = copy.deepcopy(J[r["base"]]); j["job_id"] = r["id"]
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
        if r["dur"]:
            j["duration_secs"] = r["dur"]
        out.setdefault(r["batch"], []).append(j)
        sampling = "pfn%s_ds%s" % (o.get("cbk.lidar.point_filter_num", "def"), o.get("imu.ds.mode", "def"))
        table.append(dict(arm=r["id"], tier=r["tier"], arch="iekf_raw", kind="closed", duration_s=str(j.get("duration_secs", "full")), seq=j["seq"], notes=r["purpose"], keep_raw="0", dup_of="",
                          r66_id=r["base"], group=r["group"], batch=r["batch"], sampling=sampling, requires=" ".join([] if r["kind"] == "refuse" else R80.req_of(o)), f417=str(r["f417"]), plane_check="0",
                          asked=r["asked"], early=str(r["early"]), pfn=str(r["pfn"]), pe="0", dump=str(r["dump"]), bag=r["bag"], win=str(r["win"]), msglog=str(r["msglog"]),
                          mode=r["ovl"], fam=r["fam"], src_arm=r["src_arm"], arm_kind=r["kind"], ref=r.get("ref") or "", needles="|".join(r.get("needles") or []), flag=r.get("flag", "")))
    qlist = sorted(out)
    for b in qlist:
        json.dump(out[b], open(os.path.join(out_dir, "jobs_%s.json" % b), "w"), indent=1)
    cols = ["arm", "tier", "arch", "kind", "duration_s", "seq", "notes", "keep_raw", "dup_of", "r66_id", "group", "batch", "sampling", "requires", "f417", "plane_check", "asked", "early", "pfn", "pe", "dump", "bag", "win", "msglog",
            "mode", "fam", "src_arm", "arm_kind", "ref", "needles", "flag"]
    with open(os.path.join(out_dir, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t"); w.writeheader(); w.writerows(table)
    qd = lambda b: {"dir": "ablations/_job_queue/%s" % b, "results_root": "%s/%s" % (results_root, b), "expected_jobs": len(out[b])}
    specs = [("id", ["r81_id"])] + [(b, ["r81_" + b, "r81_%s_s" % b]) for b in BAGS]
    order_lines = []
    for nm, qs in specs:
        qs = [q for q in qs if q in out]
        sp = M.listener_spec("R81-%s" % nm, [qd(q) for q in qs], results_root, min_free_gb, workers)
        fn = "listener_spec_%s.json" % nm
        for blk in sp["stages"] + sp["verify"]:
            blk["cmd"] = [c.replace("%s", fn).replace("r74_manifest", "r81_manifest") for c in blk["cmd"]]
        sp["raw"]["delete"] = sp["raw"]["delete"] + ["iekf_factor_iteration.csv", "lidar_msg_log.csv", "post_calibration_voxel_observation_support.csv", "line_rows_per_scan.csv"]
        json.dump(sp, open(os.path.join(out_dir, fn), "w"), indent=1)
        order_lines.append("%s\t%s\t%d arms" % (fn, ",".join(qs), sum(len(out[q]) for q in qs)))
    open(os.path.join(out_dir, "queue_order.tsv"), "w").write("# listener spec\tqueues\tarms   (id FIRST and alone, then the per-bag specs: a preference, not a dependency, except id: rule 1.16 and the HOLD rule)\n" + "\n".join(order_lines) + "\n")
    json.dump({"onset_scan_used": onsets, "source": "phase_summary.csv of the R80 BASE arms r80_mg_<bag>_fq (onset_scan_used)"}, open(os.path.join(out_dir, "onset_scans.json"), "w"), indent=1)
    RO = required_outputs(table)
    json.dump(RO, open(os.path.join(out_dir, "required_outputs.json"), "w"), indent=1)
    LG = ledger(RO, table)
    json.dump(LG, open(os.path.join(out_dir, "r81_file_size_ledger.json"), "w"), indent=1)
    write_compliance(out_dir, ALL, table)
    byid = {r["id"]: r for r in ALL}
    tot_exp = sum(e["expected_bytes"] for e in RO if e["where"] == "return")
    tot_led = sum(x["max_total_bytes"] for x in LG if x["note"] != "non-arm file")
    plan = part_plan(tot_exp)
    plan_led = part_plan(tot_led)
    info = dict(counts=counts_txt, total=len(table), queues={b: len(out[b]) for b in qlist}, expected_return_bytes=tot_exp, ledger_return_bytes=tot_led, plan=plan, plan_ledger=plan_led, per_file=sorted(
        ((e["file"], e["expected_bytes"]) for e in RO if e["where"] == "return" and e["expected_bytes"]), key=lambda x: -x[1])[:8])
    if not quiet:
        print(counts_txt)
        print("arms %d; queues %d: %s; listener specs %d" % (len(table), len(qlist), ", ".join("%s=%d" % (b, len(out[b])) for b in qlist), len(specs)))
        print("expected return (sum of expected_bytes, return entries) %.1f MB; ledger ceiling (1.5x/3x) %.1f MB; exceeds 30 MB: %s; parts at 28 MiB: %d (expected) / %d (ledger)" % (
            tot_exp / 1e6, tot_led / 1e6, plan["exceeds_30MB"], plan["parts_split_28M"], plan_led["parts_split_28M"]))
        print("largest files: " + ", ".join("%s %.1f MB" % (f, b / 1e6) for f, b in info["per_file"]))
    verify(out, table, ALL, byid)
    return info


def verify(out, table, ALL, byid):
    ids = [t["arm"] for t in table]
    assert len(ids) == len(set(ids)), "arm ids not unique"
    for b, lst in out.items():
        assert len({j["job_id"] for j in lst}) == len(lst), b
    for t in table:
        assert t["asked"].strip(), t["arm"]
        assert t["mode"] in ("dc", "rt"), t["arm"]
        r = byid[t["arm"]]
        if t["mode"] == "rt":
            assert r["delta"].get("cbk.lidar.overlap_policy") == "retime_trim", t["arm"]
        else:
            assert "cbk.lidar.overlap_policy" not in r["delta"], t["arm"]
        assert not any("perturb" in k.lower() for k in r["delta"]), t["arm"]
    assert [t["batch"] for t in table if t["batch"] == "r81_id"], "no r81_id queue"


def write_compliance(out_dir, ALL, table):
    own_words = "OWN rule (identity / determinism / smoke / refuse); not asked for in Bryce's words"
    with open(os.path.join(out_dir, "COMPLIANCE_TABLE.md"), "w") as f:
        f.write("# R81 compliance table (rule 1.10 item 2, 1.12)\n\nMarks: MATCHES = literally what Bryce said; OWN = the planner's rule arm (needs his sight); ASSUMPTION = a value or choice he did not give (named in the last column).\n"
                "The RRX flag marks the 12 reruns (DP 4, EC3 6, SC 2) the planner's first count omitted: they are completed R80 groups, Bryce may drop them.\n\n| arm | Bryce's words | mode | source R80 arm | mark | note |\n|---|---|---|---|---|---|\n")
        for t, r in zip(table, [next(x for x in ALL if x["id"] == t["arm"]) for t in table]):
            g = r["group"]
            mark, note = "MATCHES", ""
            if g == "OWN":
                words, mark = own_words, "OWN"
            else:
                words = WORDS[g]
            if g == "RR" and r.get("flag") == "RRX":
                note = "RRX flag"
            if g == "N485C":
                mark, note = "MATCHES", "s2 values 2e-4 and 4e-4 m^2: planner's proposal, accepted by Bryce (2026-10-05)"
            if g == "N479" and r["fam"] == "SWX":
                mark, note = "ASSUMPTION", "isotropic motion scales taken from the bag's motion-Q base job (0.3 / 10.0), onset from phase_summary; R74 used 1.0 / 5.0 for sbs_03"
            if g == "N474":
                note = "duration ceil(ns/197.5+8) s; dump frames 1-20 (planner's reading of the F-396 keys); ds off = null ds_leaf_size as R74"
            if g == "N485B":
                note = "source = R80 DP arm (keeps log_debug_en, update_diag_en, residual_dump_last_iter); frames 400-410"
            f.write("| %s | %s | %s | %s | %s | %s |\n" % (t["arm"], words.replace("|", "/"), t["mode"], t["src_arm"] or "(none)", mark, note))


# ---------------------------------------------------------------------------------------------------------------------------------- selftest
def stub_jobs():
    J = {}
    base = {"evo.enable": True, "vio.enable": False, "estimator.mode": "decoupled", "spline.mode": "raw_imu", "cbk.lidar.point_filter_num": 3, "imu.ds.mode": "first", "imu.ds.ds_leaf_size": 0.1, "imu.process_noise.model": "fixed"}
    mq = dict(base, **{"imu.process_noise.model": "isotropic", "imu.process_noise.motion.acc_scale": 0.3, "imu.process_noise.motion.gyro_scale": 10.0})
    J["00_id_baseline_full"] = {"job_id": "00_id_baseline_full", "seq": "eee_01", "overrides": dict(base)}
    J["00_id_motionq_full"] = {"job_id": "00_id_motionq_full", "seq": "eee_01", "overrides": dict(mq)}
    J["00_id_baseline_60s"] = {"job_id": "00_id_baseline_60s", "seq": "eee_01", "overrides": dict(base), "duration_secs": 60}
    for b in BAGS:
        J["50_E_%s_baseline_full" % b] = {"job_id": "50_E_%s_baseline_full" % b, "seq": b, "overrides": dict(base)}
        J["50_E_%s_motionq_full" % b] = {"job_id": "50_E_%s_motionq_full" % b, "seq": b, "overrides": dict(mq)}
    return J


def stub_grids():
    g = {b: {"bag": b, "n": N_SCANS[b], "seq0": 100 + i, "t0_s": 1.6e9 + i, "period_s": 0.0999972, "seq_gaps": 0} for i, b in enumerate(BAGS)}
    gb = {b: {"bag": b, "seq0": 200 + i, "seq_gaps": 0, "retime": {"t0_s": 1.7e9 + i, "period_s": 0.0999966}, "restamp": {}} for i, b in enumerate(BAGS)}
    return g, gb


def selftest():
    J = stub_jobs(); g, gb = stub_grids()
    onsets = {"eee_02": 481, "sbs_03": 340, "nya_01": 583, "sbs_01": 415}
    d = tempfile.mkdtemp(prefix="r81_selftest_")
    info = build(None, J, g, gb, onsets, d, "/root/catkin_ws/livo_recon_results", quiet=True)
    print(info["counts"])
    assert info["total"] == sum(EXPECT.values()) + group_counts_own(d)
    # files
    for fn in ["arm_table.tsv", "queue_order.tsv", "required_outputs.json", "r81_file_size_ledger.json", "COMPLIANCE_TABLE.md", "listener_spec_id.json"] + ["listener_spec_%s.json" % b for b in BAGS]:
        assert os.path.exists(os.path.join(d, fn)), fn
    q = open(os.path.join(d, "queue_order.tsv")).read().splitlines()
    assert q[1].startswith("listener_spec_id.json"), "id must be first"
    for b in BAGS:
        sp = json.load(open(os.path.join(d, "listener_spec_%s.json" % b)))
        assert [x["dir"] for x in sp["queues"]] == ["ablations/_job_queue/r81_%s" % b, "ablations/_job_queue/r81_%s_s" % b], b
        for x in sp["queues"]:
            assert x["expected_jobs"] == len(json.load(open(os.path.join(d, "jobs_%s.json" % x["dir"].split("/")[-1])))), x
    tab = list(csv.DictReader(open(os.path.join(d, "arm_table.tsv")), delimiter="\t"))
    assert all(t["asked"] for t in tab)
    by = {t["arm"]: t for t in tab}
    # key checks on single arms
    jobs = {}
    for fn in os.listdir(d):
        if fn.startswith("jobs_"):
            for j in json.load(open(os.path.join(d, fn))):
                jobs[j["job_id"]] = j["overrides"]
    o = jobs["r81_rt_nis_eee_01_fq"]
    assert o["imu.process_noise.gain.mode"] == "nis_smooth" and o["imu.process_noise.gain.nis_clip"] == LN2 and o["cbk.lidar.overlap_policy"] == "retime_trim" and o["cbk.lidar.retime_ref_offset_s"] == -0.1
    o = jobs["r81_dc_nis_eee_01_fq"]
    assert "cbk.lidar.overlap_policy" not in o and o["imu.process_noise.gain.decay"] == 0.95
    assert jobs["r81_dc_id_mg_eee_01_fq"] == {**jobs["r81_dc_id_mg_eee_01_fq"]}                       # exists
    idk = {k: v for k, v in jobs["r81_dc_id_mg_eee_01_fq"].items() if k not in ("outputs.path", "outputs.debug_log_dir", "eval.state_trace_run_id")}
    assert not any(k.startswith(("cbk.lidar.overlap", "voxel_map.plane.plane_var", "lio.residual_dump_voxel")) for k in idk), "identity arm carries a new key"
    assert jobs["r81_dc_vs2_nya_01_s400_pfn1_mq"]["voxel_map.plane.plane_var_scan_s2"] == 4.0e-4
    assert jobs["r81_rt_f2b_pfn1_mq"]["lio.residual_dump_voxel_info"] is True and jobs["r81_rt_f2b_pfn1_mq"]["lio.residual_dump_frames"].startswith("400,401")
    assert "voxel_map.plane.plane_fit_mode" not in jobs["r81_dc_pca_fq"] or jobs["r81_dc_pca_fq"]["voxel_map.plane.plane_fit_mode"] == "pca"
    assert jobs["r81_dc_swx_sbs_03_fq2mq"]["imu.process_noise.switch.scan"] == 340 and jobs["r81_rt_swx_sbs_03_mq2fq"]["imu.process_noise.switch.model"] == "fixed"
    assert by["r81_dc_ef_eee_01_n3850_dsoff_pfn1_fq"]["duration_s"] == "28" and by["r81_dc_ef_eee_01_n200_dsfirst_pfn3_mq"]["duration_s"] == "10"
    assert by["r81_dc_id_mg_eee_01_fq"]["batch"] == "r81_id" and by["r81_dc_refuse_nis_smooth_no_clip"]["arm_kind"] == "refuse"
    # a count mismatch must be refused loudly, with the real counts printed
    C = Ctx(J, g, gb, onsets); ALL = arms(C)
    try:
        check_counts([a for a in ALL if a["group"] != "SWT"], die=lambda m: (_ for _ in ()).throw(RuntimeError(m)))
        raise SystemExit("selftest FAILED: a short arm list was not refused")
    except RuntimeError as e:
        assert "COUNT MISMATCH" in str(e) and "real" in str(e)
    # the perturbation guard must fire
    J2 = stub_jobs(); J2["00_id_baseline_full"]["overrides"]["x.perturb_y"] = 1
    try:
        build(None, J2, g, gb, onsets, tempfile.mkdtemp(), "/r", quiet=True)
        raise SystemExit("selftest FAILED: perturbation key not refused")
    except SystemExit as e:
        assert "perturbation" in str(e)
    print("selftest OK (synthetic job list and grids; planner sandbox, generator output only, not evidence)")


def group_counts_own(d):
    return sum(1 for t in csv.DictReader(open(os.path.join(d, "arm_table.tsv")), delimiter="\t") if t["group"] == "OWN")


def read_onsets(path):
    r = {}
    for row in csv.DictReader(open(path)):
        if row["arm"].startswith("r80_mg_") and row["arm"].endswith("_fq"):
            r[row["arm"][len("r80_mg_"):-len("_fq")]] = int(float(row["onset_scan_used"]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--jobs"); ap.add_argument("--grid-retime"); ap.add_argument("--grid-bag"); ap.add_argument("--phase-summary"); ap.add_argument("--onsets")
    ap.add_argument("--out", default="r81_manifest"); ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--min-free-gb", type=float, default=150.0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.jobs and a.grid_retime and a.grid_bag):
        ap.error("--jobs, --grid-retime and --grid-bag are required")
    J = {j["job_id"]: j for j in json.load(open(a.jobs))}
    if a.phase_summary:
        ons = read_onsets(a.phase_summary)
        onsets = {b: ons[b] for b in N479_BAGS if b in ons}
    elif a.onsets:
        onsets = json.load(open(a.onsets))["onset_scan_used"]
    else:
        ap.error("--phase-summary or --onsets is required (the onset scans of the N479 bags)")
    r73 = {"eee_02": 481, "sbs_03": 340}
    for b, v in r73.items():
        print("onset %s: %s (R73/R74 value %d: %s)" % (b, onsets.get(b), v, "same" if onsets.get(b) == v else "DIFFERS, the phase_summary value is used"))
    for b in ("nya_01", "sbs_01"):
        print("onset %s: %s (phase_summary onset_scan_used of r80_mg_%s_fq)" % (b, onsets.get(b), b))
    build(a, J, json.load(open(a.grid_retime)), json.load(open(a.grid_bag)), onsets, a.out, a.results_root, a.min_free_gb, a.workers)


if __name__ == "__main__":
    main()
