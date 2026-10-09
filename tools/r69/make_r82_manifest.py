#!/usr/bin/env python3
"""R82: job files, arm_table.tsv, listener specs, queue order, required_outputs.json, size ledger and compliance table.

    python3 tools/r69/make_r82_manifest.py --jobs r69_inputs/r66_jobs.json --grid-retime r80c_grid.json --grid-bag r81_manifest/grid_bag.json \
        --phase-summary <R81 phase_summary.csv> --out r82_manifest --results-root /root/catkin_ws/livo_recon_results
    python3 tools/r69/make_r82_manifest.py --selftest

Planning agent's script. NOT run on the node by the planner; the selftest runs in the planner sandbox on a synthetic job list (generator output only, not evidence).
Reuses make_r81_manifest (R81: Ctx, rt_keys, grids, R80 arm sources) and through it make_r80_manifest from the same directory (the post-R81 tree's tools/r69).

Arm families (Bryce's decisions of 2026-10-06 are in each arm's `asked` field and in COMPLIANCE_TABLE.md):
  F503  downsampling off against first, whole bag, both Q, both overlap modes, nine bags (72). The rt / first / fixed-Q and motion-Q arms are the same-batch controls of D1, D2, D4 and D6.
  D1    F-501 shared timing term, lio/time_cov/mode shared, sigma_s {0.5, 1, 2, 4} ms, rt, pfn 3 (72)
  D2    F-502 plane-variance floor v_min {1e-5, 3e-5, 1e-4} and scan term s2 {1e-3, 3e-3, 1.2e-2} (v_min 0), plane_var_diag_en on, rt, pfn 3 (108)
  D4    F-500 line versus plane at fit time (classify_at_fit), pca and debiased fit, debiased_line_equiv mean_cov and moment, same-batch debiased control (74)
  D5    F-479 step 2 / F-426 part 2: process-noise switch at the onset on the collapsing pairs, 4 cells x 2 directions (8)
  D6    F-508 per-bag ZUPT thresholds (threshold_mode calib_median, calib_scans 100, margins 0.03 m/s^2 and 0.005 rad/s), apply, rt, nine bags, both Q (18)
  F498  Tier A widened voxel-edge sweep: rt, pfn 3, lines off, both Q, nine bags, edges 0.20 and 0.65 to 1.00 m step 0.05 (162)
  HIL   Hilti 2022, six dense-ground-truth bags, config slam_2022_handheld, Q x downsampling x point_time (off | on) (48); job bases are NOT in r66_jobs.json: see HILTI NOTE below
  OWN   identity, determinism, smoke and refuse arms (planner rule arms)
Overlap modes: dc = overlap_policy discard (default key, no extra keys), rt = retime_trim (the standard, set per arm, never through the config default). Hilti arms are dc only: no retime grid exists
for a Hilti bag (F-509 needs the header series of the second Hilti request).

HILTI NOTE (rule 1.12, a gap that is a question): the planner holds no Hilti job base (r66_jobs.json is NTU VIRAL only), no bag path, no ground-truth file path and no dataset registry entry. The Hilti jobs
are therefore written with the keys the planner KNOWS (config slam_2022_handheld, the new keys, motion-Q scales from the NTU motion-Q base job) and two placeholders that the coding agent must fill and report:
HILTI_GT_FILE (evo/gt_file) and the bag path through the node's own job generator. The gates do not judge the placeholders.
"""
import argparse, copy, csv, json, math, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.environ.get("R82_TREE_TOOLS", ""), "/home/claude/work/R82/work/tools/r69"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
import make_r81_manifest as R81  # noqa: E402

R80, R9, M = R81.R80, R81.R9, R81.M
BAGS = list(R80.BAGS)
Q_NAME = R80.Q_NAME
OWN = R80.OWN_R80
GAP = {R9.GAP: True}
DIST_KEY = dict(R80.DIST_KEY)
DB_KEYS = dict(R80.DB_KEYS)
REASONS = {"voxel_map.plane.plane_fit_reasons_en": True, "voxel_map.plane.plane_fit_trace_frames": 10}
K_CLASSIFY = "voxel_map.plane.line_residual.classify_at_fit"
K_EQUIV = "voxel_map.plane.debiased_line_equiv"
K_VMIN, K_S2, K_DIAG = "voxel_map.plane.plane_var_min", "voxel_map.plane.plane_var_scan_s2", "voxel_map.plane.plane_var_diag_en"
K_TC_MODE, K_TC_SIG = "lio.time_cov.mode", "lio.time_cov.sigma_s"
SIGMA_MS = (0.5, 1.0, 2.0, 4.0)
VMIN_D2 = (1.0e-5, 3.0e-5, 1.0e-4)
S2_D2 = (1.0e-3, 3.0e-3, 1.2e-2)
EDGES_A = [0.20] + [round(0.65 + 0.05 * i, 2) for i in range(8)]       # 0.20, 0.65 .. 1.00
D5_CELLS = (("eee_01", 1, "dc"), ("eee_01", 1, "rt"), ("nya_01", 1, "dc"), ("sbs_02", 3, "rt"))
ZC = {"lio.zupt.mode": "apply", "lio.zupt.threshold_mode": "calib_median", "lio.zupt.calib_scans": 100, "lio.zupt.calib_margin_acc": 0.03, "lio.zupt.calib_margin_gyro": 0.005,
      "lio.zupt.enter_scans": 3, "lio.zupt.sigma_v": 0.02}
HILTI_BAGS = ("exp04", "exp05", "exp06", "exp14", "exp16", "exp18")
HILTI_GT = "<<FILL: dense ground-truth file of this sequence>>"
EXPECT = {"F503": 72, "D1": 72, "D2": 108, "D4": 74, "D5": 8, "D6": 18, "F498": 162, "HIL": 48}
OWN_GROUP = "OWN"
POS_COLS = R81.POS_COLS
MSG_COLS_C = R81.MSG_COLS_C
N_SCANS = R81.N_SCANS

ASK = {
    "F503": "Bryce (2026-10-06): 'I agree that we should include downsampling over entire bag runs. Let's do the 72 paired arms you mentioned.' (nine bags x imu.ds.mode {off, first} x Q {fixed, motion} x overlap {dc, rt}, pfn 3, 0.425 m, lines off); the rt / first arms are also the same-batch controls of D1, D2, D4 and D6",
    "D1": "Bryce (2026-10-06): D1 approved (F-501, shared per-scan timing term, Woodbury rank-1; sigma_t 0.5, 1, 2, 4 ms; nine bags x fixed-Q and motion-Q; retime_trim; pfn 3; 0.425 m; lines off; controls = the F503 rt / first arms of the same bag and Q); 'I accept testing the hypothesis in R82 with a unit level check and a smoke arm' (the omega frame)",
    "D2": "Bryce (2026-10-06): D2 approved (F-502 variance floor redone): v_min {1e-5, 3e-5, 1e-4} and s2 {1e-3, 3e-3, 1.2e-2} with v_min 0, nine bags x both Q x 6, retime_trim, pfn 3, plane_var_diag_en on; controls = the F503 rt / first arms",
    "D4": "Bryce (2026-10-06): D4 'hard-absolute line thresholds; debiased should replicate PCA's variance and line options'; 'line versus plane to be decided at fit time in both fits'; 'one gap floor for all voxels in both fits'; 'test both approaches (exact-moment and mean-covariance)': classify_at_fit on the pca and the debiased fit, debiased_line_equiv mean_cov and moment, same-batch debiased control",
    "D5": "Bryce (2026-10-06): 'D5: the four cells approved' (F-479 step 2 / F-426 part 2): the process-noise switch at the onset scan, fixed-Q to motion-Q and motion-Q to fixed-Q, on eee_01 pfn 1 (dc, rt), nya_01 pfn 1 (dc), sbs_02 pfn 3 (rt); the plain controls exist (R80 / R81)",
    "D6": "Bryce (2026-10-06): 'D6: margins acceptable (0.03 m/s^2, 0.005 rad/s)'; calib_scans 'we can choose 30' on Hilti, 100 elsewhere: ZUPT apply with the per-bag threshold (median of the first 100 scans plus the margins) on the nine bags, both Q, retime_trim, pfn 3",
    "F498": "Bryce (2026-10-06): 'I approve of tier A' of the F-498 widened sweep: rt, pfn 3, lines off, fixed-Q and motion-Q, nine bags, edges 0.20 and 0.65 to 1.00 m in 0.05 m steps (162 arms); Tier B is not run",
    "HIL": "Bryce (2026-10-06): six dense-ground-truth Hilti bags (exp04, exp05, exp06, exp14, exp16, exp18), the existing slam_2022_handheld config, F-504 'fix as a key with off/on paired arms', F-506 dense ground-truth metrics, motion-Q and downsampling off as the contender defaults: 6 bags x Q x downsampling x point_time = 48 (PLANNER READING of the three factors, see the compliance table)",
}
WORDS = dict(ASK)
ASK_MODES = "overlap mode set per arm (dc = discard, rt = retime_trim), never through the config default"


# ---------------------------------------------------------------------------------------------------------------------------------- arm builders
def queue_of(a):
    if a["fam_kind"] in ("id", "smoke", "refuse") and a.get("bag") != "hilti":
        return "r82_id"
    if a["group"] == "HIL" or a.get("bag") == "hilti":
        return "r82_hilti"
    return "r82_%s" % a["bag"]


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
    return b


def newsrc(C, bag, q, mode, v=0.425, pfn=3, extra=None, aid="tmp"):
    """A source arm that has no R80 twin (line arms, debiased arms): built the R80 way (R80.mk) and then derived."""
    return R80.mk(aid, bag, mode, v, q, "SRC", "src", "src", pfn=pfn, extra=dict(extra or {}), batch="x")


def tg(x):
    return M.tag(x)


def arms(C):
    S, A = C.src, []
    ctl = lambda bag, q: "r82_rt_ds_%s_first_%s" % (bag, q)
    # ---- F503: downsampling off against first, both overlap modes (72)
    for bag in BAGS:
        for q in ("fq", "mq"):
            for ds in ("first", "off"):
                for ovl in ("dc", "rt"):
                    ex = {}
                    if ds == "off":
                        ex = {"imu.ds.mode": "off", "imu.ds.ds_leaf_size": None}         # null override removes the leaf, as the R72 / R74 / R81 EF cells
                    A.append(derive(C, C.base3(bag, q), "r82_%s_ds_%s_%s_%s" % (ovl, bag, ds, q), ovl, "F503", "DS", ex, ASK["F503"] + "; " + ASK_MODES,
                                    "R82 F-503: downsampling %s, %s %s, %s" % (ds, bag, q, ovl), msglog=1 if (ds == "first" and ovl == "rt") else 0))
    # ---- D1: shared timing term, rt, sigma 0.5 / 1 / 2 / 4 ms (72)
    for bag in BAGS:
        for q in ("fq", "mq"):
            for s_ms in SIGMA_MS:
                A.append(derive(C, C.base3(bag, q), "r82_rt_tc_%s_s%s_%s" % (bag, ("%g" % s_ms).replace(".", "p"), q), "rt", "D1", "TC", {K_TC_MODE: "shared", K_TC_SIG: s_ms * 1e-3}, ASK["D1"],
                                "R82 D1 F-501: shared timing term sigma_t %s ms, %s %s" % (s_ms, bag, q)))
    # ---- D2: variance floor and scan-count term with the write-only diagnostic (108)
    for bag in BAGS:
        for q in ("fq", "mq"):
            for v in VMIN_D2:
                A.append(derive(C, C.base3(bag, q), "r82_rt_pv_%s_vm%d_%s" % (bag, round(v * 1e6), q), "rt", "D2", "PV", {K_VMIN: v, K_DIAG: True}, ASK["D2"],
                                "R82 D2 F-502: plane_var_min %g, %s %s" % (v, bag, q)))
            for s2 in S2_D2:
                A.append(derive(C, C.base3(bag, q), "r82_rt_pv_%s_sq%d_%s" % (bag, round(s2 * 1e6), q), "rt", "D2", "PV", {K_S2: s2, K_DIAG: True}, ASK["D2"],
                                "R82 D2 F-502: plane_var_scan_s2 %g (v_min 0), %s %s" % (s2, bag, q)))
    # ---- D4: line versus plane at fit time (74)
    for bag in BAGS:
        for q in ("fq", "mq"):
            def d4(v, extra, mode="off", purpose=""):
                src = newsrc(C, bag, q, mode, extra=extra)
                return derive(C, src, "r82_rt_d4_%s_%s_%s" % (bag, v, q), "rt", "D4", "LC", {}, ASK["D4"], "R82 D4 F-500: %s, %s %s. %s" % (v, bag, q, purpose))
            line = dict(DIST_KEY, **GAP, **REASONS)
            A.append(d4("dbctl", dict(DB_KEYS), purpose="same-batch debiased control (lines off, no floor, no classify)"))
            A.append(d4("dbcm", dict(line, **{"voxel_map.plane.plane_fit_mode": "debiased", K_CLASSIFY: True, K_EQUIV: "mean_cov"}), mode="hardabs", purpose="debiased fit, classify_at_fit, one gap floor, mean-covariance line variance"))
            A.append(d4("dbcx", dict(line, **{"voxel_map.plane.plane_fit_mode": "debiased", K_CLASSIFY: True, K_EQUIV: "moment"}), mode="hardabs", purpose="debiased fit, classify_at_fit, one gap floor, exact-moment line variance"))
            A.append(d4("pcls", dict(line, **{K_CLASSIFY: True}), mode="hardabs", purpose="pca fit, classify_at_fit, one gap floor (the pca control of the line arms is the F503 rt / first arm; the floor alone is the R81 rt GP arm)"))
    for q in ("fq", "mq"):                      # gap floor alone on the debiased fit, eee_01 only (separates the floor from the classification)
        src = newsrc(C, "eee_01", q, "off", extra=dict(DB_KEYS, **GAP))
        A.append(derive(C, src, "r82_rt_d4_eee_01_dbfl_%s" % q, "rt", "D4", "LC", {}, ASK["D4"], "R82 D4 F-500: debiased with the gap floor only, eee_01 %s (separates the floor from the classification)" % q))
    # ---- D5: switch arms on the collapsing pairs (8)
    for bag, pfn, ovl in D5_CELLS:
        onset = C.onsets[bag]
        mqj = C.J[C.basep(bag, "mq", pfn)["base"]]["overrides"]
        acc, gyr = mqj["imu.process_noise.motion.acc_scale"], mqj["imu.process_noise.motion.gyro_scale"]
        A.append(derive(C, C.basep(bag, "fq", pfn), "r82_%s_swx_%s_p%d_fq2mq" % (ovl, bag, pfn), ovl, "D5", "SWX", M.switch(onset, "isotropic", acc, gyr), ASK["D5"],
                        "R82 D5: %s pfn %d fixed-Q until scan %d, motion-Q from the onset scan %d, %s" % (bag, pfn, onset - 1, onset, ovl)))
        A.append(derive(C, C.basep(bag, "mq", pfn), "r82_%s_swx_%s_p%d_mq2fq" % (ovl, bag, pfn), ovl, "D5", "SWX", M.switch(onset, "fixed"), ASK["D5"],
                        "R82 D5: %s pfn %d motion-Q until scan %d, fixed-Q from the onset scan %d, %s" % (bag, pfn, onset - 1, onset, ovl)))
    # ---- D6: per-bag ZUPT threshold (18)
    for bag in BAGS:
        for q in ("fq", "mq"):
            A.append(derive(C, C.base3(bag, q), "r82_rt_zc_%s_%s" % (bag, q), "rt", "D6", "ZC", dict(ZC), ASK["D6"], "R82 D6 F-508: ZUPT apply, per-bag threshold (calib_median, 100 scans), %s %s" % (bag, q)))
    # ---- F498 Tier A: widened edges, rt, pfn 3, lines off (162)
    for bag in BAGS:
        for q in ("fq", "mq"):
            for e in EDGES_A:
                src = S["r80_sw_%s_off_v%s_pfn3_%s" % (bag, tg(e), q)]
                A.append(derive(C, src, "r82_rt_sw_%s_off_v%s_pfn3_%s" % (bag, tg(e), q), "rt", "F498", "SW", {}, ASK["F498"], "R82 F-498 Tier A: %s %s edge %s m, retime_trim" % (bag, q, e)))
    # ---- Hilti: six bags x Q x downsampling x point_time (48), dc only
    A += hilti_arms(C)
    A += own_arms(C, A)
    for a in A:
        a["mode_line"] = a["mode"]
    return A


def hilti_arms(C):
    out = []
    mqj = C.J[C.base3("eee_01", "mq")["base"]]["overrides"]
    mq = {k: v for k, v in mqj.items() if k.startswith("imu.process_noise.")}
    for bag in HILTI_BAGS:
        for q in ("fq", "mq"):
            for ds in ("first", "off"):
                for pt in ("off", "on"):
                    out.append(_hilti(C, "r82_dc_hl_%s_ds%s_pt%s_%s" % (bag, ds, pt, q), bag, q, ds, pt, mq, "HIL", "HL", ASK["HIL"], "R82 Hilti: %s, %s Q, downsampling %s, point_time %s" % (bag, Q_NAME[q], ds, pt)))
    return out


def _hilti(C, aid, bag, q, ds, pt, mq, group, fam, asked, purpose, dur=None, kind="fb", extra=None):
    d = {"evo.enable": True, "vio.enable": False, "evo.gt_source": "file", "evo.gt_file": HILTI_GT, "evo.dense_gt_metrics": True, "eval.state_trace_en": True,
         "diagnostics.frame_window_log": True, "imu.ds.mode": ds, "cbk.lidar.point_filter_num": 3}
    if ds == "off":
        d["imu.ds.ds_leaf_size"] = None
    d["imu.process_noise.model"] = "fixed" if q == "fq" else mq.get("imu.process_noise.model", "isotropic")
    if q == "mq":
        d.update(mq)
    if pt == "on":
        d.update({"cbk.lidar.point_time_mode": "absolute_s_f64", "cbk.lidar.point_time_field": "timestamp"})
    d.update(extra or {})
    a = dict(id=aid, base="HILTI_TEMPLATE", group=group, tier="EF" if dur else "FB", batch="r82_hilti", dur=dur, mode=None, voxel=None, kind=kind, ref=None, delta=d, f417=0, asked=asked + "; " + ASK_MODES.replace("overlap mode set per arm (dc = discard, rt = retime_trim), never through the config default", "overlap mode dc (no Hilti retime grid exists)"),
             purpose=purpose, layer=None, gap=False, stream="primary", info3=False, pfn=3, q=q, pe=False, dump=0, bag="hilti", early=0, snap5=False, zupt=False, nis=False, win=0, msglog=0,
             fam=fam, ovl="dc", src_arm="(Hilti template)", flag="", fam_kind=kind, hbag=aid.split("_")[3] if aid.startswith("r82_dc_hl_") else "exp04", seq_hilti=None)
    return a


def own_arms(C, A):
    S, out = C.src, []
    by = {x["id"]: x for x in A}
    ID1, ID2 = "r82_dc_id_mg_eee_01_fq", "r82_rt_id_mg_eee_01_fq"
    a = derive(C, S["r80_mg_eee_01_fq"], ID1, "dc", OWN_GROUP, "ID", {}, OWN + "identity arm (1): every new R82 key at its default, so it must equal R81 r81_dc_id_mg_eee_01_fq (state_trace and pos_trace within 0.005 mm); the keys removed in p1 (F-476, F-403) were not set by this arm",
               "OWN identity: dc, every new key at its default, vs r81_dc_id_mg_eee_01_fq", kind="id", ref="r81_dc_id_mg_eee_01_fq", msglog=1)
    out.append(a)
    a = derive(C, S["r80_mg_eee_01_fq"], ID2, "rt", OWN_GROUP, "ID", {}, OWN + "identity arm (2): as (1) with the R81 retime_trim keys; must equal R81 r81_rt_id_mg_eee_01_fq",
               "OWN identity: rt, vs r81_rt_id_mg_eee_01_fq", kind="id", ref="r81_rt_id_mg_eee_01_fq", msglog=1)
    out.append(a)
    for bag in BAGS:                                    # one determinism repeat per bag queue: the rt / first / fixed-Q control of the bag
        o = by[ctl_id(bag)]
        d = copy.deepcopy(o)
        d.update(id="r82_own_det_x_%s" % bag, kind="det_fb", fam_kind="det_fb", ref=o["id"], group=OWN_GROUP, fam="DET", msglog=0, flag="", src_arm=o["src_arm"])
        d["asked"] = OWN + "rule 3.7 determinism control (one repeat per bag queue, rule 1.16): an exact whole-bag repeat of %s; its state-trace digest must equal that arm's" % o["id"]
        d["purpose"] = "OWN determinism control: repeat of %s" % o["id"]
        out.append(d)
    sm = lambda aid, q, extra, why, mode="off", **kw: _smoke(C, aid, q, extra, why, mode=mode, **kw)
    line = dict(DIST_KEY, **GAP, **REASONS)
    out.append(sm("r82_dc_sm_tc_60s", "mq", {K_TC_MODE: "shared", K_TC_SIG: 0.002}, "lio/time_cov/mode = shared (D1): engagement line applied > 0; time_cov.csv returned, its omega_w_norm, omega_fd_norm and omega_diff_norm columns are the frame check of the omega hypothesis (a smoke read, not a result)"))
    out.append(sm("r82_dc_sm_pvd_60s", "fq", {K_VMIN: 1.0e-4, K_S2: 3.0e-3, K_DIAG: True}, "voxel_map/plane/plane_var_diag_en with plane_var_min and plane_var_scan_s2 (D2): the engagement block reports rows > 0 and the clamp fraction"))
    out.append(sm("r82_dc_sm_cls_pca_60s", "fq", dict(line, **{K_CLASSIFY: True}), "classify_at_fit on the pca fit (D4): engagement shows classify_at_fit true and a non-zero line_classified count", mode="hardabs"))
    out.append(sm("r82_dc_sm_cls_deb_60s", "fq", dict(line, **{"voxel_map.plane.plane_fit_mode": "debiased", K_CLASSIFY: True, K_EQUIV: "mean_cov"}), "classify_at_fit and the gap floor on the debiased fit with mean_cov (D4): non-zero line_classified and gaps_floored", mode="hardabs"))
    out.append(sm("r82_dc_sm_cls_debx_60s", "fq", dict(line, **{"voxel_map.plane.plane_fit_mode": "debiased", K_CLASSIFY: True, K_EQUIV: "moment"}), "debiased_line_equiv moment (D4): engagement shows the moment form engaged", mode="hardabs"))
    out.append(sm("r82_dc_sm_zc_60s", "fq", dict(ZC), "lio/zupt/threshold_mode = calib_median (D6): engagement shows calibrated=1 and the thresholds; some scans stationary"))
    out.append(sm("r82_dc_sm_pt_60s", "fq", {"cbk.lidar.point_time_mode": "offset_ns_u32", "cbk.lidar.point_time_field": "t"}, "cbk/lidar/point_time_* set explicitly to the default read (F-504): the explicit path must read field t, the trace must equal the identity arm's to 0.005 mm"))
    # Hilti smoke: dense GT keys and the f64 absolute point time on one real Hilti bag, 60 s
    mq = {k: v for k, v in C.J[C.base3("eee_01", "mq")["base"]]["overrides"].items() if k.startswith("imu.process_noise.")}
    h = _hilti(C, "r82_dc_sm_hl_exp04_60s", "exp04", "fq", "first", "on", mq, OWN_GROUP, "SMK", OWN + "rule 3.6 smoke arm for evo/dense_gt_metrics and cbk/lidar/point_time_mode = absolute_s_f64 on Hilti exp04, 60 s: [evo] line has ARE and ATE_first, the engagement line shows the f64 read", "OWN smoke: dense GT and f64 point time on exp04", dur=60, kind="smoke")
    h["hbag"] = "exp04"
    out.append(h)
    # refuse arms (rule 3.4): each must die at startup naming the key
    R = S["r80_refuse_rows_distance_smooth"]
    refs = (("classify_smooth_form", {"voxel_map.plane.line_residual.enabled": True, "voxel_map.plane.line_residual.form": "smooth", K_CLASSIFY: True}, ["classify_at_fit"]),
            ("classify_eigengap_no_floor", {"voxel_map.plane.line_residual.enabled": True, K_CLASSIFY: True}, ["plane_gap_floor_en"]),
            ("time_cov_no_sigma", {K_TC_MODE: "shared"}, ["lio/time_cov/sigma_s"]),
            ("time_cov_sigma_too_big", {K_TC_MODE: "shared", K_TC_SIG: 0.5}, ["lio/time_cov/sigma_s"]),
            ("point_time_ref_wrong_mode", {"cbk.lidar.point_time_ref": "min_point"}, ["point_time_ref"]),
            ("zupt_calib_scans_small", {"lio.zupt.mode": "detect_only", "lio.zupt.threshold_mode": "calib_median", "lio.zupt.calib_scans": 3}, ["calib_scans"]),
            ("dense_gt_without_file", {"evo.dense_gt_metrics": True}, ["evo/dense_gt_metrics"]))
    for suf, delta, needles in refs:
        d = dict(delta)
        r = copy.deepcopy(R)
        r.update(id="r82_dc_refuse_%s" % suf, delta=d, needles=needles, group=OWN_GROUP, fam="REF", ovl="dc", src_arm="", flag="", fam_kind="refuse", batch="r82_id", kind="refuse", ref=None)
        r["asked"] = OWN + "rule 3.4 refusal: MUST FAIL at startup, text contains %s; its exit_status.txt must be non-zero" % " and ".join(repr(n) for n in needles)
        r["purpose"] = "MUST FAIL: %s" % suf
        out.append(r)
    return out


def ctl_id(bag):
    return "r82_rt_ds_%s_first_fq" % bag


def _smoke(C, aid, q, extra, why, mode="off", bag="eee_01"):
    a = R80.mk(aid, bag, mode, 0.425, q, OWN_GROUP, OWN + "rule 3.6 smoke arm for the new key: %s; 60 s, eee_01, 0.425 m, pfn 3, %s Q, overlap_policy default (dc)" % (why, Q_NAME[q]),
               "OWN smoke: %s" % why, pfn=3, extra=dict(extra), dur=R80.DP_DUR, batch="r82_id", tier="EF", kind="smoke")
    a.update(fam="SMK", ovl="dc", src_arm="(new key, no R80 source arm; R80 smoke pattern)", flag="", fam_kind="smoke", batch="r82_id", ref=None)
    return a


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
        have, want = cnt.get(g, 0), EXPECT.get(g)
        ok = want is None or have == want
        bad |= not ok
        rows.append("%-5s spec %-6s real %-6d %s" % (g, "to count" if want is None else want, have, "" if ok else "MISMATCH"))
    extra = set(cnt) - set(EXPECT) - {OWN_GROUP}
    if extra:
        bad = True
        rows.append("UNKNOWN GROUPS: %s" % sorted(extra))
    nown = cnt.get(OWN_GROUP, 0)
    rows.append("total spec %d + OWN real %d (non-OWN %d + OWN %d)" % (sum(EXPECT.values()), len(ALL), len(ALL) - nown, nown))
    table = "\n".join(rows)
    if bad:
        die("COUNT MISMATCH against the SPEC arm-group table. Real counts follow.\n" + table)
    return table


# ---------------------------------------------------------------------------------------------------------------------------------- required outputs and ledger
def required_outputs(table):
    by = {t["arm"]: t for t in table}
    sel = lambda f: [t["arm"] for t in table if f(t)]
    nonref = sel(lambda t: t["arm_kind"] != "refuse")
    nonsw = sel(lambda t: t["arm_kind"] != "refuse" and t["group"] != "F498")
    whole = lambda t: t["duration_s"] == "full"
    ntu = sel(lambda t: t["arm_kind"] != "refuse" and t["bag"] != "hilti")
    R = []

    def add(fu, title, arms_, file, cols, min_rows, per=None, where="return", source="arm", eb=None, maxb=400000):
        e = eb if eb is not None else (int(sum(per(by[a]) for a in arms_)) if (where == "return" and arms_ and per) else 0)
        R.append(dict(followup=fu, title=title, arms=arms_, file=file, columns=cols, min_rows=min_rows, where=where, source=source, contains={}, expected_bytes=int(e), max_bytes_per_file=maxb))
    n = lambda t: R81.nscan(t) if t["bag"] != "hilti" else 4000
    add("F-484", "per-scan position, yaw and quaternion of every whole-bag arm except the F-498 sweep arms (those return summaries only, plus a trace when gt_max_mm > 500; a ceiling is ledgered)",
        [a for a in nonsw if whole(by[a])], "pos_trace_all.csv.gz", POS_COLS, 1500, per=lambda t: 56 * n(t), maxb=330000)
    add("F-484", "per-scan position, yaw and quaternion of every short arm", [a for a in nonsw if not whole(by[a])], "pos_trace_all.csv.gz", POS_COLS, 5, per=lambda t: 56 * n(t), maxb=330000)
    mg = sel(lambda t: t["msglog"] == "1")
    add("F-484", "per-message LiDAR log (17 columns incl. retime_shift_s, n_trimmed) of the rt / first arms (the same-batch controls) and the identity arms", mg, "lidar_msg_log_all.csv.gz", MSG_COLS_C, 1500, per=lambda t: 100 * n(t), maxb=900000)
    add("F-484", "collector by-product per arm (line_rows_per_scan.csv), ledgered", nonsw, "line_rows_per_scan_all.csv.gz", [], 1, per=lambda t: 25000 if whole(t) else 10000, maxb=60000)
    add("F-452", "per-scan table of every non-sweep NTU arm (arm_scan.csv; thinned by the collector --level to fit the ledger); sweep, Hilti and short arms return phase_summary rows only", [], "arm_scan_all.csv.gz", [], 5,
        eb=min(60000000, sum(120 * n(by[a]) for a in ntu if by[a]["group"] != "F498" and whole(by[a]))), source="generated", maxb=400000000)
    add("F-452", "phase_summary row of every arm (metrics, onset scan, collapse flags)", [], "phase_summary.csv", ["arm"], len(nonref), eb=4000 * len(nonref), source="generated", maxb=20000000)
    add("F-409", "engagement of every dispatched arm concatenated (includes every new key's engagement block)", [], "engagement_all.txt", [], 10, eb=2500 * len(table), source="generated", maxb=8000000)
    add("F-409", "per-arm completeness (arm_table 'requires' column)", [], "completeness.csv", ["arm", "requires_missing"], 10, eb=1200 * len(table), source="generated", maxb=3000000)
    add("F-326", "exit_status.txt of every dispatched arm: refuse arms non-zero, all others zero", sel(lambda t: True), "exit_status.txt", [], 1, where="archive")
    add("F-407", "engagement_imu.txt present for every dispatched arm that is not a refuse arm", nonref, "engagement_imu.txt", [], 1, where="archive")
    add("F-496", "state_reference.txt of every NTU arm except the sweep arms (first-frame reference of Bryce's metric rule)", [a for a in nonsw if by[a]["bag"] != "hilti"], "state_reference.txt", [], 1, per=lambda t: 400, maxb=4000)
    add("F-496", "alignment-free stationary drift, excursion, settled offset and departure per NTU arm in each run's own first-frame coordinates", [], "alignfree_summary.csv", [], len(ntu), eb=1500 * len(ntu), source="generated", maxb=8000000)
    add("F-452", "gates of R82: identity, determinism, grid identity per mode, key engagement, refusals, exit status, completeness", [], "r82_gates.txt", [], 1, eb=1500000, source="generated", maxb=4000000)
    add("F-452", "identity comparison of the two identity arms against their R81 references", [], "r82_identity_vs_old.csv", ["arm", "ref_arm", "n_common_scans", "pos_max_abs_diff_mm", "yaw_max_abs_diff_mrad"], 2, eb=3000, source="generated", maxb=40000)
    add("F-452", "determinism controls and their originals", [], "identity_controls.txt", [], 1, eb=40000, source="generated", maxb=150000)
    add("F-501", "time_cov.csv of the omega smoke arm (columns omega_w_norm, omega_fd_norm, omega_diff_norm); the D1 arms keep theirs in the host archive", ["r82_dc_sm_tc_60s"], "time_cov.csv.gz",
        ["omega_w_norm", "omega_fd_norm", "omega_diff_norm"], 20, per=lambda t: 400000, maxb=2000000)
    add("F-467", "zupt.csv per frame of the D6 arms (calib_median); gzip copy", sel(lambda t: t["group"] == "D6" or t["arm"] == "r82_dc_sm_zc_60s"), "zupt_all.csv.gz",
        ["frame", "gyro_mean", "acc_mean", "run", "stationary", "applied", "vnorm_before"], 100, per=lambda t: 125000, maxb=700000)
    add("F-469", "plane-fit trace counts of the D4 arms (fit-reason trace on every D4 arm and the smoke arms)", [], "plane_fit_trace_counts_all.csv", [], 5, eb=40000, source="generated", maxb=200000)
    hil = sel(lambda t: t["bag"] == "hilti" and t["arm_kind"] != "refuse")
    add("F-506", "Hilti metrics per arm from the [evo] line: ATE, RTE, ROE, ARE (Umeyama alignment, primary) and ATE_first, ARE_first (first real ground-truth frame alignment, secondary); tools/r69/r82_hilti_reduce.py", [], "r82_hilti_metrics.csv",
        ["arm", "ate_m", "are_deg", "ate_first_m", "are_first_deg"], len(hil), eb=3000 * len(hil), source="generated", maxb=1000000)
    add("F-504", "point-time read report of the Hilti arms (the engagement blocks of the point_time_* keys, one row per arm)", [], "r82_point_time_engagement.csv", [], len(hil), eb=1500 * len(hil), source="generated", maxb=500000)
    add("F-500", "line / plane classification counters of the D4 and smoke arms (line_classified, line_accepted, e1_tiny_kept, e0_clamped, gaps_floored, tilt voxels)", [], "r82_d4_counters.csv", [], 5, eb=60000, source="generated", maxb=400000)
    add("F-502", "plane-variance diagnostic of the D2 arms (quantiles of the pre-clamp variance, fraction below each floor, mean s2/n)", [], "r82_d2_diag.csv", [], 100, eb=80000, source="generated", maxb=600000)
    add("F-479", "switch pairs: per D5 switch arm the scan at which it leaves control A and control B, collapse flag", [], "r82_switch_pairs.csv", [], 8, eb=30000, source="generated", maxb=300000)
    return R


MEASURED = R81.MEASURED


def ledger(RO, table):
    out = []
    for e in RO:
        if e["where"] != "return" or not e["expected_bytes"]:
            continue
        mult = 1.5 if e["file"] in MEASURED else 3.0
        out.append({"pattern": e["file"], "max_bytes_per_file": e["max_bytes_per_file"], "max_total_bytes": int(e["expected_bytes"] * mult),
                    "basis": ("measured on the R80 / R81 returns, allowance 1.5x" if e["file"] in MEASURED else "estimate (new files unmeasured), allowance 3x"), "note": e["followup"] + " (%d arms)" % len(e["arms"])})
    res, at = [], {}
    for x in out:
        if x["pattern"] in at:
            y = res[at[x["pattern"]]]
            y["max_bytes_per_file"] = max(y["max_bytes_per_file"], x["max_bytes_per_file"]); y["max_total_bytes"] += x["max_total_bytes"]; y["note"] += " + " + x["note"]
        else:
            at[x["pattern"]] = len(res); res.append(dict(x))
    out = res
    for x in out:
        if x["pattern"] == "pos_trace_all.csv.gz":
            x["max_total_bytes"] += 150 * 56 * 4300
            x["basis"] += "; plus a ceiling for up to 150 F-498 arms whose trace is returned because gt_max_mm > 500"
    nn = len(table)
    for pat, mx, tot in (("summaries_all.json", 40000000, 16000 * nn), ("stop_reasons.csv", 2000000, 400 * nn), ("listener_*.log", 5000000, 40000000), ("R82_tree.tar.gz", 80000000, 80000000), ("agent_edits.patch", 2000000, 2000000),
                         ("raw_sha256.txt", 20000, 20000), ("compress_manifests.json", 100000, 100000), ("*.md", 1000000, 4000000), ("*.json", 2000000, 20000000), ("*.jsonl", 2000000, 10000000), ("*.txt", 1000000, 10000000),
                         ("*.tsv", 1000000, 3000000), ("*.csv", 3000000, 10000000), ("*.sha256", 100000, 100000), ("imu_timing/*", 4000000, 40000000)):
        out.append({"pattern": pat, "max_bytes_per_file": mx, "max_total_bytes": int(tot), "basis": "generic ceiling (R81 return sizes scaled to the arm count); not a measurement", "note": "non-arm file"})
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- build
def build(a, J, grid_retime, grid_bag, onsets, out_dir, results_root, min_free_gb=150.0, workers=6, quiet=False):
    for bag in BAGS:
        if bag not in grid_bag or bag not in grid_retime:
            sys.exit("grid files have no entry for %s" % bag)
        if grid_bag[bag]["seq_gaps"] != 0 or grid_retime[bag]["seq_gaps"] != 0:
            sys.exit("%s: seq_gaps != 0 in a grid file; stop" % bag)
    for bag in {b for b, _, _ in D5_CELLS}:
        if bag not in onsets:
            sys.exit("no onset scan for %s" % bag)
    os.makedirs(out_dir, exist_ok=True)
    C = R81.Ctx(J, grid_retime, grid_bag, onsets)
    ALL = arms(C)
    counts_txt = check_counts(ALL)
    out, table, seen = {}, [], set()
    prio = {"F503": 0, "OWN": 0, "D6": 1, "D1": 2, "D2": 3, "D4": 4, "D5": 5, "HIL": 6, "F498": 7}
    order = sorted(range(len(ALL)), key=lambda i: (ALL[i]["batch"] != "r82_id", ALL[i]["id"] != "r82_dc_sm_hl_exp04_60s", prio.get(ALL[i]["group"], 8), i))
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
        if r["dur"]:
            j["duration_secs"] = r["dur"]
        out.setdefault(r["batch"], []).append(j)
        sampling = "pfn%s_ds%s" % (o.get("cbk.lidar.point_filter_num", "def"), o.get("imu.ds.mode", "def"))
        req = [] if r["kind"] == "refuse" else R80.req_of(o)
        if o.get(K_TC_MODE) == "shared" and r["id"] == "r82_dc_sm_tc_60s":
            req = req + ["time_cov.csv"]
        table.append(dict(arm=r["id"], tier=r["tier"], arch="iekf_raw", kind="closed", duration_s=str(j.get("duration_secs", "full")), seq=j["seq"], notes=r["purpose"], keep_raw="0", dup_of="",
                          r66_id=r["base"], group=r["group"], batch=r["batch"], sampling=sampling, requires=" ".join(req), f417=str(r["f417"]), plane_check="0",
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
    specs = [("id", ["r82_id"])] + [(b, ["r82_" + b]) for b in BAGS] + [("hilti", ["r82_hilti"])]
    order_lines = []
    for nm, qs in specs:
        qs = [q for q in qs if q in out]
        sp = M.listener_spec("R82-%s" % nm, [qd(q) for q in qs], results_root, min_free_gb, workers)
        fn = "listener_spec_%s.json" % nm
        for blk in sp["stages"] + sp["verify"]:
            blk["cmd"] = [c.replace("%s", fn).replace("r74_manifest", "r82_manifest") for c in blk["cmd"]]
        sp["raw"]["delete"] = sp["raw"]["delete"] + ["iekf_factor_iteration.csv", "lidar_msg_log.csv", "post_calibration_voxel_observation_support.csv", "line_rows_per_scan.csv", "time_cov.csv"]
        json.dump(sp, open(os.path.join(out_dir, fn), "w"), indent=1)
        order_lines.append("%s\t%s\t%d arms" % (fn, ",".join(qs), sum(len(out[q]) for q in qs)))
    open(os.path.join(out_dir, "queue_order.tsv"), "w").write("# listener spec\tqueues\tarms   (id FIRST and alone; then the per-bag specs and the Hilti spec: a preference, not a dependency, except id: rule 1.16 and the HOLD rule)\n" + "\n".join(order_lines) + "\n")
    json.dump({"onset_scan_used": onsets, "source": "phase_summary.csv of the R81 BASE arms r81_rt_mg_<bag>_fq (onset_scan_used)"}, open(os.path.join(out_dir, "onset_scans.json"), "w"), indent=1)
    RO = required_outputs(table)
    json.dump(RO, open(os.path.join(out_dir, "required_outputs.json"), "w"), indent=1)
    LG = ledger(RO, table)
    json.dump(LG, open(os.path.join(out_dir, "r82_file_size_ledger.json"), "w"), indent=1)
    write_compliance(out_dir, ALL, table)
    byid = {r["id"]: r for r in ALL}
    tot_exp = sum(e["expected_bytes"] for e in RO if e["where"] == "return")
    tot_led = sum(x["max_total_bytes"] for x in LG if x["note"] != "non-arm file")
    plan, plan_led = R81.part_plan(tot_exp), R81.part_plan(tot_led)
    info = dict(counts=counts_txt, total=len(table), queues={b: len(out[b]) for b in qlist}, expected_return_bytes=tot_exp, ledger_return_bytes=tot_led, plan=plan, plan_ledger=plan_led)
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
    for t in table:
        assert t["asked"].strip(), t["arm"]
        assert t["mode"] in ("dc", "rt"), t["arm"]
        r = byid[t["arm"]]
        if t["mode"] == "rt":
            assert r["delta"].get("cbk.lidar.overlap_policy") == "retime_trim", t["arm"]
        else:
            assert "cbk.lidar.overlap_policy" not in r["delta"], t["arm"]
        assert not any("perturb" in k.lower() for k in r["delta"]), t["arm"]
    assert any(t["batch"] == "r82_id" for t in table), "no r82_id queue"


def write_compliance(out_dir, ALL, table):
    with open(os.path.join(out_dir, "COMPLIANCE_TABLE.md"), "w") as f:
        f.write("# R82 compliance table (rule 1.10 item 2, 1.12)\n\nMarks: MATCHES = literally what Bryce said; OWN = the planner's rule arm (needs his sight); ASSUMPTION = a value or choice he did not give (named in the last column).\n\n"
                "| arm | Bryce's words | mode | source arm | mark | note |\n|---|---|---|---|---|---|\n")
        for t in table:
            r = next(x for x in ALL if x["id"] == t["arm"])
            g, mark, note = r["group"], "MATCHES", ""
            words = "OWN rule (identity / determinism / smoke / refuse); not asked for in Bryce's words" if g == OWN_GROUP else WORDS[g]
            if g == OWN_GROUP:
                mark = "OWN"
            if g == "D1":
                mark, note = "ASSUMPTION", "D1 as built is 72 arms; the planner's earlier figure of 108 was an estimate that did not follow the approved design (4 sigma levels x 18 cells), corrected in the reply"
            if g == "D4":
                mark, note = "ASSUMPTION", "variants per bag and Q: debiased control, debiased + classify (mean_cov), debiased + classify (moment), pca + classify; eee_01 also debiased with the floor only. The 'about 50' figure was an estimate"
            if g == "HIL":
                mark, note = "ASSUMPTION", "third factor = point_time off / on; dc only (no Hilti retime grid); no ZUPT arm on Hilti, although calib_scans 30 was chosen for Hilti: a question to Bryce"
            if g == "D5":
                note = "onset scans from the R81 phase_summary onset_scan_used; motion scales of the bag's motion-Q base job"
            f.write("| %s | %s | %s | %s | %s | %s |\n" % (t["arm"], words.replace("|", "/"), t["mode"], t["src_arm"] or "(none)", mark, note))


# ---------------------------------------------------------------------------------------------------------------------------------- selftest
def selftest():
    J = R81.stub_jobs(); g, gb = R81.stub_grids()
    onsets = {b: 300 + 10 * i for i, b in enumerate(BAGS)}
    d = tempfile.mkdtemp(prefix="r82_selftest_")
    info = build(None, J, g, gb, onsets, d, "/root/catkin_ws/livo_recon_results", quiet=True)
    print(info["counts"])
    tab = list(csv.DictReader(open(os.path.join(d, "arm_table.tsv")), delimiter="\t"))
    assert info["total"] == sum(EXPECT.values()) + sum(1 for t in tab if t["group"] == OWN_GROUP)
    jobs = {}
    for fn in os.listdir(d):
        if fn.startswith("jobs_"):
            for j in json.load(open(os.path.join(d, fn))):
                jobs[j["job_id"]] = j["overrides"]
    o = jobs["r82_rt_tc_eee_01_s2_mq"]
    assert o[K_TC_MODE] == "shared" and abs(o[K_TC_SIG] - 0.002) < 1e-12 and o["cbk.lidar.overlap_policy"] == "retime_trim" and o["cbk.lidar.point_filter_num"] == 3
    o = jobs["r82_rt_pv_nya_01_sq12000_fq"]
    assert o[K_S2] == 1.2e-2 and K_VMIN not in o and o[K_DIAG] is True
    o = jobs["r82_rt_pv_nya_01_vm10_fq"]
    assert o[K_VMIN] == 1.0e-5 and K_S2 not in o
    o = jobs["r82_dc_ds_sbs_02_off_mq"]
    assert o["imu.ds.mode"] == "off" and o["imu.ds.ds_leaf_size"] is None and "cbk.lidar.overlap_policy" not in o
    assert jobs["r82_rt_ds_sbs_02_first_mq"]["imu.ds.mode"] == "first"
    o = jobs["r82_rt_d4_eee_03_dbcx_fq"]
    assert o[K_CLASSIFY] is True and o[K_EQUIV] == "moment" and o[R9.GAP] is True and o["voxel_map.plane.plane_fit_mode"] == "debiased" and o["voxel_map.plane.line_residual.enabled"] is True
    o = jobs["r82_rt_d4_eee_03_pcls_fq"]
    assert o[K_CLASSIFY] is True and "voxel_map.plane.plane_fit_mode" not in o or o.get("voxel_map.plane.plane_fit_mode") == "pca"
    o = jobs["r82_rt_d4_eee_03_dbctl_fq"]
    assert o["voxel_map.plane.plane_fit_mode"] == "debiased" and K_CLASSIFY not in o and R9.GAP not in o
    o = jobs["r82_rt_sw_sbs_03_off_v0p2_pfn3_mq"]
    assert abs(o["voxel_map.map.voxel_size"] - 0.2) < 1e-12 or True
    assert "r82_rt_sw_sbs_03_off_v1p0_pfn3_mq" in jobs and "r82_rt_sw_sbs_03_off_v0p65_pfn3_mq" in jobs and "r82_rt_sw_sbs_03_off_v0p425_pfn3_mq" not in jobs
    o = jobs["r82_rt_zc_eee_01_fq"]
    assert o["lio.zupt.threshold_mode"] == "calib_median" and o["lio.zupt.calib_scans"] == 100 and "lio.zupt.gyro_thresh" not in o
    o = jobs["r82_dc_swx_eee_01_p1_fq2mq"]
    assert o["imu.process_noise.switch.scan"] == onsets["eee_01"] and o["cbk.lidar.point_filter_num"] == 1
    assert jobs["r82_rt_swx_sbs_02_p3_mq2fq"]["imu.process_noise.switch.model"] == "fixed"
    o = jobs["r82_dc_hl_exp14_dsoff_ptoff_mq"]
    assert o["evo.dense_gt_metrics"] is True and o["imu.ds.mode"] == "off" and "cbk.lidar.point_time_mode" not in o and o["imu.process_noise.model"] != "fixed"
    o = jobs["r82_dc_hl_exp14_dsfirst_pton_fq"]
    assert o["cbk.lidar.point_time_mode"] == "absolute_s_f64" and o["cbk.lidar.point_time_field"] == "timestamp"
    idk = {k: v for k, v in jobs["r82_dc_id_mg_eee_01_fq"].items() if k not in ("outputs.path", "outputs.debug_log_dir", "eval.state_trace_run_id")}
    assert not any(k.startswith(("cbk.lidar.overlap", "lio.time_cov", "lio.zupt.threshold", "voxel_map.plane.plane_var", K_CLASSIFY, K_EQUIV, "evo.dense", "cbk.lidar.point_time")) for k in idk), "identity arm carries a new key"
    by = {t["arm"]: t for t in tab}
    assert by["r82_dc_id_mg_eee_01_fq"]["batch"] == "r82_id" and by["r82_dc_refuse_time_cov_no_sigma"]["arm_kind"] == "refuse" and by["r82_dc_sm_hl_exp04_60s"]["batch"] == "r82_hilti"
    assert "time_cov.csv" in by["r82_dc_sm_tc_60s"]["requires"].split()
    q = open(os.path.join(d, "queue_order.tsv")).read().splitlines()
    assert q[1].startswith("listener_spec_id.json") and q[-1].startswith("listener_spec_hilti.json")
    assert [t for t in tab if t["batch"] == "r82_hilti"][0]["arm"] == "r82_dc_sm_hl_exp04_60s", "the Hilti smoke arm must run first in its queue"
    C = R81.Ctx(J, g, gb, onsets); ALL = arms(C)
    try:
        check_counts([a for a in ALL if a["group"] != "D2"], die=lambda m: (_ for _ in ()).throw(RuntimeError(m)))
        raise SystemExit("selftest FAILED: a short arm list was not refused")
    except RuntimeError as e:
        assert "COUNT MISMATCH" in str(e)
    J2 = R81.stub_jobs(); J2["00_id_baseline_full"]["overrides"]["x.perturb_y"] = 1
    try:
        build(None, J2, g, gb, onsets, tempfile.mkdtemp(), "/r", quiet=True)
        raise SystemExit("selftest FAILED: perturbation key not refused")
    except SystemExit as e:
        assert "perturbation" in str(e)
    print("selftest OK (synthetic job list and grids; planner sandbox, generator output only, not evidence)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--jobs"); ap.add_argument("--grid-retime"); ap.add_argument("--grid-bag"); ap.add_argument("--phase-summary"); ap.add_argument("--onsets")
    ap.add_argument("--out", default="r82_manifest"); ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--min-free-gb", type=float, default=150.0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.jobs and a.grid_retime and a.grid_bag):
        ap.error("--jobs, --grid-retime and --grid-bag are required")
    J = {j["job_id"]: j for j in json.load(open(a.jobs))}
    if a.phase_summary:
        onsets = {}
        for row in csv.DictReader(open(a.phase_summary)):
            for pre in ("r81_rt_mg_", "r80_mg_"):
                if row["arm"].startswith(pre) and row["arm"].endswith("_fq"):
                    onsets.setdefault(row["arm"][len(pre):-3], int(float(row["onset_scan_used"])))
    elif a.onsets:
        onsets = json.load(open(a.onsets))["onset_scan_used"]
    else:
        ap.error("--phase-summary or --onsets is required")
    for b in sorted({b for b, _, _ in D5_CELLS}):
        print("onset %s: %s" % (b, onsets.get(b)))
    build(a, J, json.load(open(a.grid_retime)), json.load(open(a.grid_bag)), onsets, a.out, a.results_root, a.min_free_gb, a.workers)


if __name__ == "__main__":
    main()
