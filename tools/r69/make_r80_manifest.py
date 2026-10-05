#!/usr/bin/env python3
"""R80: build the job files, arm_table.tsv, listener specs, required_outputs.json and the size ledger from the R66 job list (r66_jobs.json).

    python3 tools/r69/make_r80_manifest.py --jobs r69_inputs/r66_jobs.json --out r80_manifest --results-root /root/catkin_ws/livo_recon_results

Planning agent's script; NOT run against the real host by me. Reuses the R79 helpers (make_r79_manifest as R9): the same R66 base jobs, the same overrides.

Bryce (2026-10-05) asked for two things, in his words (each arm's `asked` field and COMPLIANCE_TABLE.md repeat them):
  F-449(a) 'State and covariance for pfn1 and pfn3, both Q, at scans 515-530, on eee_01 at 0.425 m with lines off. It would need the per-iteration state, the vertical velocity and its variance, and the
           covariance rows for position and velocity. I would add the dumped per-row residuals and weights for ground rows at scans 518-524.'  -> draft it
           = 4 arms (point filter 1 and 3, fixed-Q and motion-Q), eee_01, root voxel 0.425, lines off, first 60 s, with
             lio.log_debug_en        iekf_factor_iteration.csv: position and velocity before / after EVERY iteration, step, 6 x 6 information   (existing log, no new C++)
             lio.update_diag_en      lio_update_diag.csv: HtH, Htz, residual-variance decomposition, step per iteration                           (existing log)
             lio.residual_dump_frames 515..530 + lio.residual_dump_last_iter: per-row residual, sigma^2 (+ its parts), plane-variance term, normal, world point, iteration 0 and the last iteration
             state_trace.csv (always on): state, velocity and the full 18 x 18 covariance of both phases for every scan
             frame_stats.txt (always on): rows used and the misses by reason per frame
  F-389    'confirm by measuring header time to last point received as the first point could still be delayed by some amount. I'd like to be entirely certain about why we're missing scans before we change
           our code here.'  -> data_queues/log_lidar_msgs (new, write-only, default false): lidar_msg_log.csv, one row per message, header-to-first-point, header-to-last-point, previous accepted last point and
           pushLidar's real verdict. On the 4 dump arms and on nine whole-bag arms (one per bag, fixed-Q, pfn 3, 0.425, lines off = the R79 F-463 lines-off 0.425 fixed-Q arms; they are also the identity arms).
Batches (queues):  r80_id  1 arm  = the pfn3 fixed-Q dump arm (the smoke arm of the new keys AND the identity arm against r79_dx_off_v0p425_pfn3_fq)
                   r80_dp  4 arms = pfn3 motion-Q (identity against r79_dx_off_v0p425_pfn3_mq), pfn1 fixed-Q, pfn1 motion-Q, the determinism repeat of pfn1 fixed-Q
                   r80_mg  9 arms = whole bags with the message log (identity against r79_nb_<bag>_off_v0p425_pfn3_fq)
"""
import argparse, copy, csv, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_r79_manifest as R9  # noqa: E402

M = R9.M
BAGS = R9.BAGS
QS = R9.QS
N_SCANS = R9.N_SCANS
LO, HI, GLO, GHI = 515, 530, 518, 524
DP_DUR = 60
WIN_FRAMES = ",".join(str(i) for i in range(LO, HI + 1))
DUMP_KEYS = {"lio.log_debug_en": True, "lio.update_diag_en": True, "lio.residual_dump_frames": WIN_FRAMES, "lio.residual_dump_last_iter": True, "data_queues.log_lidar_msgs": True}
MSG_KEY = {"data_queues.log_lidar_msgs": True}
MQ_TRACE_ARM = "r80_b3_eee_02_mq"   # F-437 / D2: the one arm whose motion_q_scan.csv is returned (r69_collect.KEEP_IF_ARM_HAS)
SMOKE_ARM = "r80_dp_pfn3_fq"          # first arm of the first queue
DET_ARM, DET_ORIG = "r80_own_det_dp_pfn1_fq", "r80_dp_pfn1_fq"
IDREF = {"r80_dp_pfn3_fq": "r79_dx_off_v0p425_pfn3_fq", "r80_dp_pfn3_mq": "r79_dx_off_v0p425_pfn3_mq"}
for _b in BAGS:
    IDREF["r80_mg_%s_fq" % _b] = "r79_nb_%s_off_v0p425_pfn3_fq" % _b   # BASE3 fixed-Q with the message log
    IDREF["r80_b3_%s_mq" % _b] = "r79_nb_%s_off_v0p425_pfn3_mq" % _b   # BASE3 motion-Q
    IDREF["r80_zd_%s_fq" % _b] = "r79_nb_%s_off_v0p425_pfn3_fq" % _b   # D7 detect_only never touches the state
DX_LAST_EXCLUDED = R9.DX_LAST_EXCLUDED
POS_COLS = R9.POS_COLS
WIN_FILES = ["window_state_all.csv.gz", "window_iter_all.csv.gz", "window_update_diag_all.csv.gz", "window_frame_stats_all.csv.gz", "window_dump_state_all.csv.gz",
             "window_row_summary_all.csv", "window_ground_rows_all.csv.gz"]
EXPECTED = {"window_state_all.csv.gz": 40000, "window_iter_all.csv.gz": 30000, "window_update_diag_all.csv.gz": 15000, "window_frame_stats_all.csv.gz": 15000, "window_dump_state_all.csv.gz": 3000,
            "window_row_summary_all.csv": 6000, "window_ground_rows_all.csv.gz": 1500000, "lidar_msg_log_all.csv.gz": 60000, "pos_trace_all.csv.gz": 56}
MAXFILE = {"window_state_all.csv.gz": 150000, "window_iter_all.csv.gz": 150000, "window_update_diag_all.csv.gz": 100000, "window_frame_stats_all.csv.gz": 100000, "window_dump_state_all.csv.gz": 30000,
           "window_row_summary_all.csv": 60000, "window_ground_rows_all.csv.gz": 9000000, "lidar_msg_log_all.csv.gz": 900000, "pos_trace_all.csv.gz": 330000}
STATE_COLS = ["run_id", "scan_id", "t_abs", "phase", "px", "py", "pz", "vx", "vy", "vz", "P_0_0"]
ITER_COLS = ["scan_id", "iter", "p_before_x", "p_after_x", "v_before_z", "v_after_z", "dtheta_norm", "A0", "b0"]
UPD_COLS = ["frame", "iter", "n_resid", "sum_inv_sigma2", "mean_sigma2", "mean_plane_var_term", "HtH_00", "Htz_0", "dt_x"]
FS_COLS = ["frame_idx", "n_points_after_pfn", "n_residuals", "n_miss_coverage", "n_miss_mismatch", "n_tier0_miss_coverage", "n_tier0_miss_mismatch", "nis", "p_pos_vel_fro_pre"]
DSTATE_COLS = ["frame", "iter", "n_resid", "px", "py", "pz", "rvx", "rvy", "rvz"]
SUM_COLS = ["frame", "iter", "class", "n_rows", "sum_weight", "mean_r2_over_sigma2", "n_r2_over_9sigma2", "n_r2_over_9gate", "sigma_gate_p50", "sigma2_p50", "plane_var_p50", "s_prior_pose_p50"]
GROUND_COLS = ["frame", "iter", "idx", "r", "sigma_squared", "plane_var_term", "floor_term", "sigma_diag_squared", "s_prior_pose", "sigma_gate_squared", "node_id", "hpos_z", "w", "r2_over_sigma2"]
MSG_COLS = ["msg_index", "seq", "header_stamp", "scan_time", "gap_header_s", "n_raw", "n_kept", "t_first", "t_last", "off_first_s", "off_last_s", "span_s", "prev_accepted_last_t", "overlap_s", "verdict"]

ASK_DP = ("Bryce (2026-10-05): 'I would like to see the dumps from this proposal: The dump I would propose. State and covariance for pfn1 and pfn3, both Q, at scans 515-530, on eee_01 at 0.425 m with lines off. "
          "It would need the per-iteration state, the vertical velocity and its variance, and the covariance rows for position and velocity. I would add the dumped per-row residuals and weights for ground rows "
          "at scans 518-524. Do you want me to draft it as part of the next handoff?' -> 'draft it'")
ASK_MSG = ("Bryce (2026-10-05): 'confirm by measuring header time to last point received as the first point could still be delayed by some amount. I'd like to be entirely certain about why we're missing "
           "scans before we change our code here.'")


EDGES = [round(0.20 + 0.05 * i, 2) for i in range(17)]          # D1: 0.20 .. 1.00 m step 0.05
LINES = ("off", "hardabs")
PFNS = (3, 1)
DIST_KEY = {"voxel_map.plane.line_residual.rows": "distance"}      # R80 F-453 (Bryce 2026-10-05): one distance row per hard line voxel
GAP_KEY = {R9.GAP: True}                                           # F-433 gap floor (R77 flag)
NG_KEYS = M.GAIN("nis_gate", nis_hi=1.3, inflate=1.25, decay=0.95, nis_lo=0.75, deflate=0.8, min_gain=0.25)   # F-476, all six accepted by Bryce 2026-10-05
SC_KEYS = {"lio.ekf.step_cap.mode": "hard", "lio.ekf.step_cap.rotation_rad": 0.01, "lio.ekf.step_cap.position_m": 0.01}   # D5: 10 mrad and 10 mm
PC = {"ps": {"voxel_map.residual.pose_cov_in_sigma": True}, "pp": {"lio.point_cov.mode": "propagated"}, "pb": {"voxel_map.residual.pose_cov_in_sigma": True, "lio.point_cov.mode": "propagated"}}   # D8
D3_N = (250, 300, 350)                                             # D3 (Bryce 2026-10-05: keep 250/300/350, pfn 1 and 3)
TL_ITERS = (5, 10, 20)                                             # TAIL-1 (5 = the default, the control)
TL_COUPLED = {"estimator.mode": "coupled", "estimator.coupled.residual_evaluation_time": "tail", "estimator.coupled.update_log_en": True}
DB_KEYS = {"voxel_map.plane.plane_fit_mode": "debiased", "voxel_map.plane.plane_fit_reasons_en": True, "voxel_map.plane.plane_fit_trace_frames": 10}
Q_NAME = {"fq": "fixed", "mq": "motion"}
F1_DEFAULT = dict(win=0, msglog=0, f417=0, dump=0, early=0)

ASK_D = "Bryce (2026-10-05): approved designs D1-D8 (text in claude/r80-designs-D1-D8-original.md)"


def mk(aid, bag, mode, v, q, group, asked, purpose, pfn=3, extra=None, dur=None, batch="x", tier="FB", kind="fb", ref=None, **flags):
    a = R9.nbarm(aid, bag, mode, v, q, group, asked, purpose, pfn=pfn, extra=extra, dump=False, dur=dur, tier=tier, batch=batch)
    a.update(F1_DEFAULT); a.update(kind=kind, ref=ref, group=group); a.update(flags)
    return a


def merged(*ds):
    o = {}
    for x in ds:
        o.update(x)
    return o


def arms():
    A = []
    # ---- A: F-449a window dumps (4) + determinism repeat; F-389 message log on the nine BASE3 fixed-Q arms
    for pfn in (3, 1):
        for q in QS:
            aid = "r80_dp_pfn%d_%s" % (pfn, q)
            a = R9.nbarm(aid, "eee_01", "off", 0.425, q, "DP",
                         "%s: eee_01, root voxel 0.425, lines off, point filter %d, %s Q, first %d s; log_debug_en (per-iteration state with velocity), update_diag_en, residual dump of frames %d to %d (iteration 0 and last), "
                         "message log" % (ASK_DP, pfn, Q_NAME[q], DP_DUR, LO, HI),
                         "R80 F-449(a): window dump, pfn %d, %s, 60 s" % (pfn, q), pfn=pfn, extra=dict(DUMP_KEYS), dump=False, dur=DP_DUR, tier="EF", batch="id" if aid == SMOKE_ARM else "dp")
            a.update(win=1, msglog=1, group="DP", f417=0, dump=0, early=0)
            if aid in IDREF:
                a.update(kind="id", ref=IDREF[aid])
            A.append(a)
    d = R9.nbarm(DET_ARM, "eee_01", "off", 0.425, "fq", "DET", "Own rule 3.7 determinism control: an exact repeat of %s; its state-trace digest must equal that arm's" % DET_ORIG,
                 "OWN determinism control: repeat of %s" % DET_ORIG, pfn=1, extra=dict(DUMP_KEYS), dump=False, dur=DP_DUR, tier="EF", batch="dp")
    d.update(kind="det_fb", ref=DET_ORIG, win=0, msglog=0, group="DET", f417=0, dump=0, early=0)
    A.append(d)
    # ---- own smoke arms for the new keys (rule 3.6), in the id queue, 60 s each
    A.append(mk("r80_sm_dist_60s", "eee_01", "hardabs", 0.425, "fq", "SMK", "OWN rule 3.6 smoke arm for voxel_map/plane/line_residual/rows = distance (F-453): hard absolute, plane_equivalent, 60 s; the engagement line must state rows distance, line_rows_per_scan.csv must have a non-zero n_line_dist column and the dump no kind 1 or 2 rows",
                "R80 smoke: rows = distance", pfn=3, extra=dict(DIST_KEY), dur=DP_DUR, batch="id", tier="EF", kind="smoke"))
    A.append(mk("r80_sm_nis_60s", "eee_01", "off", 0.425, "mq", "SMK", "OWN rule 3.6 smoke arm for the two-sided NIS gate (F-476): nis_hi 1.3, nis_lo 0.75, deflate 0.8, min_gain 0.25, 60 s; the engagement line and q_adapt.csv must exist",
                "R80 smoke: two-sided NIS gate", pfn=3, extra=dict(NG_KEYS), dur=DP_DUR, batch="id", tier="EF", kind="smoke"))
    A.append(mk("r80_sm_tl_60s", "eee_01", "off", 0.425, "mq", "SMK", "OWN rule 3.6 smoke arm for the TAIL-1 arms: coupled tail, motion-Q, lio/ekf/max_iterations 10, 60 s; the agent reports the wall time per scan from it",
                "R80 smoke: coupled tail, 10 iterations", pfn=3, extra=merged(TL_COUPLED, {"lio.ekf.max_iterations": 10}), dur=DP_DUR, batch="id", tier="EF", kind="smoke"))
    # ---- refuse arms (rule 3.4; each is also the deliberately refused job of the F-326 check: exit_status.txt must be non-zero)
    for suf, delta, needles in (("rows_distance_smooth", merged({"voxel_map.plane.line_residual.enabled": True, "voxel_map.plane.line_residual.form": "smooth"}, DIST_KEY), ["rows = distance needs form = hard"]),
                                ("rows_distance_lines_off", dict(DIST_KEY), ["voxel_map/plane/line_residual/rows"]),
                                ("nis_lo_without_deflate", M.GAIN("nis_gate", nis_lo=0.75), ["nis_lo > 0 needs deflate < 1"])):
        a = dict(id="r80_refuse_%s" % suf, base="00_id_baseline_60s", group="REF", tier="REF", batch="ref", dur=5, mode=None, voxel=None, kind="refuse", ref=None, needles=needles, delta=dict(delta), f417=0,
                 asked=OWN_R80 + "rule 3.4 refusal (also the refused job of F-326): MUST FAIL at startup, text contains %s; its exit_status.txt must be non-zero" % " and ".join(repr(n) for n in needles),
                 purpose="MUST FAIL: %s" % suf, layer=None, gap=False, stream="primary", info3=False, pfn=None, q=None, pe=False, dump=0, bag="eee_01", early=0, snap5=False, zupt=False, nis=False)
        a.update(win=0, msglog=0)
        A.append(a)
    # ---- per-bag queues x_<bag>: BASE3 (fq with message log = F-389, mq), BASE1, D7, F-433, F-476
    for bag in BAGS:
        qn = "x_" + bag
        a = R9.nbarm("r80_mg_%s_fq" % bag, bag, "off", 0.425, "fq", "MG",
                     "%s: %s whole bag, root voxel 0.425, lines off, point filter 3, fixed-Q; data_queues/log_lidar_msgs on (BASE3: identity arm of r79_nb_%s_off_v0p425_pfn3_fq; control of F-433 pfn 3 and F-476)" % (ASK_MSG, bag, bag),
                     "R80 F-389 / BASE3: message log, %s whole bag" % bag, pfn=3, extra=dict(MSG_KEY), dump=False, tier="FB", batch=qn)
        a.update(kind="id", ref=IDREF[a["id"]], win=0, msglog=1, group="MG", f417=0, dump=0, early=0)
        A.append(a)
        dt = R9.nbarm("r80_own_det_x_%s" % bag, bag, "off", 0.425, "fq", "DET", "Own rule 3.7 determinism control (one repeat per bag queue, Bryce's rule 1.16 split): an exact whole-bag repeat of r80_mg_%s_fq; its state-trace digest must equal that arm's" % bag,
                      "OWN determinism control: repeat of r80_mg_%s_fq" % bag, pfn=3, extra=dict(MSG_KEY), dump=False, tier="FB", batch=qn)
        dt.update(kind="det_fb", ref="r80_mg_%s_fq" % bag, win=0, msglog=1, group="DET", f417=0, dump=0, early=0)
        A.append(dt)
        aid = "r80_b3_%s_mq" % bag
        extra_note = " Also F-437 / D2: its motion_q_scan.csv is returned." if aid == MQ_TRACE_ARM else ""
        A.append(mk(aid, bag, "off", 0.425, "mq", "B3", ASK_D + "; BASE3 same-batch control (identity arm of r79_nb_%s_off_v0p425_pfn3_mq; control of F-433 pfn 3 and F-476). D2 (F-437): eee_02 motion-Q whole bag.%s" % (bag, extra_note),
                    "R80 BASE3: %s, 0.425, pfn 3, motion-Q" % bag, pfn=3, batch=qn, kind="id", ref=IDREF[aid]))
        for q in QS:
            A.append(mk("r80_b1_%s_%s" % (bag, q), bag, "off", 0.425, q, "B1", ASK_D + "; BASE1: the pfn 1 same-batch control (no earlier arm exists) of F-433 pfn 1 and of D1's pfn 1 cells at 0.425 m: %s, %s Q" % (bag, Q_NAME[q]),
                        "R80 BASE1: %s, 0.425, pfn 1, %s" % (bag, q), pfn=1, batch=qn))
        aid = "r80_zd_%s_fq" % bag
        A.append(mk(aid, bag, "off", 0.425, "fq", "ZD", ASK_D + "; D7 F-467: lio/zupt/mode = detect_only (never touches the state), fixed-Q, 0.425 m, lines off, pfn 3, whole bag; identity arm of r79_nb_%s_off_v0p425_pfn3_fq; zupt.csv returned per frame" % bag,
                    "R80 D7 F-467: ZUPT detect_only, %s" % bag, pfn=3, extra=M.ZUPT("detect_only"), batch=qn, kind="id", ref=IDREF[aid]))
        for pfn in PFNS:
            for q in QS:
                A.append(mk("r80_gp_%s_pfn%d_%s" % (bag, pfn, q), bag, "off", 0.425, q, "GP", ASK_D + "; F-433 (Bryce: 'let's do the 36 arms as you described', 'with controls'): plane_gap_floor_en on, 0.425 m, lines off, whole bag: %s, pfn %d, %s Q; controls = BASE3 (pfn 3) and BASE1 (pfn 1)" % (bag, pfn, Q_NAME[q]),
                            "R80 F-433: gap floor on, %s, pfn %d, %s" % (bag, pfn, q), pfn=pfn, extra=dict(GAP_KEY), batch=qn))
        for q in QS:
            A.append(mk("r80_ng_%s_%s" % (bag, q), bag, "off", 0.425, q, "NG", ASK_D + "; F-476 (Bryce 2026-10-05: nis_lo 0.75, deflate 0.8, min_gain 0.25; nis_hi 1.3, inflate 1.25, decay 0.95; 'nine bags x fixed-Q and motion-Q, gate on and off'): %s, %s Q, whole bag, 0.425 m, pfn 3, lines off; gate-off control = BASE3" % (bag, Q_NAME[q]),
                        "R80 F-476: two-sided NIS gate, %s, %s" % (bag, q), pfn=3, extra=dict(NG_KEYS), batch=qn))
    # ---- e1 queue: eee_01 only: D8 (6), F-469 (2), TAIL-1 (3)
    for q in QS:
        for k in ("ps", "pp", "pb"):
            A.append(mk("r80_pc_%s_%s" % (k, q), "eee_01", "off", 0.425, q, "PC", ASK_D + "; D8 F-470: eee_01 whole bag, 0.425 m, lines off, pfn 3, %s Q, cell %s (ps = pose_cov_in_sigma true; pp = lio/point_cov/mode propagated; pb = both); controls = BASE3 eee_01" % (Q_NAME[q], k),
                        "R80 D8 F-470: %s, %s" % (k, q), pfn=3, extra=dict(PC[k]), batch="e1"))
        A.append(mk("r80_db_%s" % q, "eee_01", "off", 0.425, q, "DB", "Bryce (2026-10-05): F-469 'agreed as proposed': plane_fit_mode debiased against pca (BASE3 eee_01), plane_fit_reasons_en and plane_fit_trace_frames 10, eee_01 whole bag, 0.425 m, lines off, pfn 3, %s Q" % Q_NAME[q],
                    "R80 F-469: debiased fit, %s" % q, pfn=3, extra=dict(DB_KEYS), batch="e1"))
    for it in TL_ITERS:
        A.append(mk("r80_tl_it%d" % it, "eee_01", "off", 0.425, "mq", "TL", "Bryce (2026-10-05): TAIL-1 approved ('we can do 10/20 iterations sure'): coupled tail, motion-Q, eee_01 whole bag, 0.425 m, pfn 3, lines off, lio/ekf/max_iterations %d%s; coupled_update_iter.csv logged" % (it, " (the default: the control)" if it == 5 else ""),
                    "R80 TAIL-1: coupled tail, max_iterations %d" % it, pfn=3, extra=merged(TL_COUPLED, {"lio.ekf.max_iterations": it}), batch="e1"))
    # ---- ef queue: D3 (6) and D5 (2), first 60 s or the R72 formula
    for ns in D3_N:
        dur = int(math.ceil(ns / 197.5 + 8.0))
        for pfn in PFNS:
            A.append(mk("r80_ec_n%d_pfn%d_fq" % (ns, pfn), "eee_01", "off", 0.425, "fq", "EC3", ASK_D + "; D3 F-465/F-441: calibration samples %d, point filter %d, eee_01, fixed-Q, 0.425 m, lines off, %d s by the R72 formula ceil(ns/197.5+8); residual dump of frames 1 to 5 and a map snapshot at each of frames 1 to 5 (post-calibration included)" % (ns, pfn, dur),
                        "R80 D3: samples %d, pfn %d" % (ns, pfn), pfn=pfn, extra={"calib.stationary.num_samples": ns, **R9.F5DUMP, **R9.SNAP5}, dur=dur, tier="EF", batch="ef", early=1, dump=5, f417=1))
    for q in QS:
        A.append(mk("r80_sc_%s" % q, "eee_01", "off", 0.425, q, "SC", ASK_D + "; D5 F-403: lio/ekf/step_cap hard, 10 mrad and 10 mm, eee_01 first 60 s, 0.425 m, lines off, pfn 3, %s Q; control = r80_dp_pfn3_%s (no cap, same 60 s)" % (Q_NAME[q], q),
                    "R80 D5 F-403: step cap, %s, 60 s" % q, pfn=3, extra=dict(SC_KEYS), dur=DP_DUR, tier="EF", batch="ef"))
    # ---- D1 sweep: one queue per (bag, q, line, pfn) cell, 17 edges each (1224 arms)
    for bag in BAGS:
        for q in QS:
            for line in LINES:
                for pfn in PFNS:
                    qn = "sw_%s_%s_%s_p%d" % (bag, q, line, pfn)
                    for e in EDGES:
                        aid = "r80_sw_%s_%s_v%s_pfn%d_%s" % (bag, line, M.tag(e), pfn, q)
                        A.append(mk(aid, bag, line, e, q, "SW", ASK_D + "; D1 F-463/F-466: root voxel %s m, max_layer 2, %s, %s Q, pfn %d, %s whole bag%s" % (e, "lines off" if line == "off" else "hard absolute with rows = distance and plane_equivalent variance",
                                    Q_NAME[q], pfn, bag, "" if line == "off" else ""), "R80 D1 sweep: %s %s %s pfn %d edge %s" % (bag, q, line, pfn, e),
                                    pfn=pfn, extra=(dict(DIST_KEY) if line == "hardabs" else None), batch=qn))
    dsw = mk("r80_own_det_sw_eee_03_hardabs_v0p35_pfn3_mq", "eee_03", "hardabs", 0.35, "mq", "DET", "Own rule 3.7 determinism control (D1: 'plus one determinism repeat'): an exact whole-bag repeat of r80_sw_eee_03_hardabs_v0p35_pfn3_mq (hard absolute, rows = distance); its state-trace digest must equal that arm's",
             "OWN determinism control: repeat of r80_sw_eee_03_hardabs_v0p35_pfn3_mq", pfn=3, extra=dict(DIST_KEY), batch="sw_eee_03_mq_hardabs_p3", kind="det_fb", ref="r80_sw_eee_03_hardabs_v0p35_pfn3_mq")
    A.append(dsw)
    return A


def sw_cells():
    return ["sw_%s_%s_%s_p%d" % (b, q, l, p) for b in BAGS for q in QS for l in LINES for p in PFNS]


def req_of(o):
    r = M.requires_of(o, "x")
    if o.get("lio.log_debug_en"):
        r = r + ["iekf_factor_iteration.csv"]
    if o.get("data_queues.log_lidar_msgs"):
        r = r + ["lidar_msg_log.csv"]
    if o.get("lio.line_rows_log_en"):
        r = r + ["line_rows_per_scan.csv"]
    if o.get("voxel_map.map.snapshot_frames"):
        r = r + ["voxelmap_snapshot_f%d.csv" % n for n in (2, 3, 4, 5)]
    return r


OWN_R80 = R9.OWN
EXP2 = {"zupt_all.csv.gz": 300000, "q_adapt_all.csv.gz": 100000, "step_cap.csv.gz": 30000, "snapshot_frames_all.csv.gz": 0}
MAX2 = {"zupt_all.csv.gz": 700000, "q_adapt_all.csv.gz": 500000, "step_cap.csv.gz": 150000}
EARLY_FILES = (("snapshot_frames_all.csv.gz", R9.SNAP_COLS, 300, lambda t: 40000 * 5 if t["pfn"] == "3" else 60000 * 5),
               ("residual_dump_all.csv.gz", R9.DUMP_COLS, 500, lambda t: 420000 * (3 if t["pfn"] == "1" else 1)),
               ("r73_f417_hth_frames.csv", ["frame", "iter", "hth_trace", "eig_min", "cond", "info_yaw"], 5, lambda t: 4000),
               ("residual_dump_state_all.csv.gz", [], 5, lambda t: 3000),
               ("state_trace_early_all.csv.gz", R9.STATE_EARLY_COLS, 10, lambda t: 15000),
               ("lio_update_diag_early_all.csv.gz", R9.UPD_EARLY_COLS, 5, lambda t: 8000))


def required_outputs(table):
    by = {t["arm"]: t for t in table}
    sel = lambda f: [t["arm"] for t in table if f(t)]
    win, msg, allarm = sel(lambda t: t["win"] == "1"), sel(lambda t: t["msglog"] == "1"), sel(lambda t: t["group"] != "REF")
    notsw = sel(lambda t: t["group"] not in ("REF", "SW"))
    R = []

    def add(fu, title, arms_, file, cols, min_rows, where="return", source="arm", expected_per=None, maxb=None):
        eb = int(sum(expected_per(a) for a in arms_)) if (where == "return" and arms_ and expected_per) else (int(EXPECTED.get(file, EXP2.get(file, 50000)) * max(1, len(arms_))) if where == "return" else 0)
        R.append(dict(followup=fu, title=title, arms=arms_, file=file, columns=cols, min_rows=min_rows, where=where, source=source, contains={}, expected_bytes=eb,
                      max_bytes_per_file=maxb or MAXFILE.get(file, MAX2.get(file, 400000))))
    w = lambda f: (lambda a: EXPECTED[f])
    add("F-449a", "state, velocity and 18 x 18 covariance (upper triangle P_i_j) of both phases, scans %d to %d" % (LO - 1, HI + 1), win, "window_state_all.csv.gz", STATE_COLS, 20, expected_per=w("window_state_all.csv.gz"))
    add("F-449a", "per-iteration position and velocity before / after every iteration, step, rank, 6 x 6 information, frames %d to %d" % (LO, HI), win, "window_iter_all.csv.gz", ITER_COLS, 16, expected_per=w("window_iter_all.csv.gz"))
    add("F-449a", "per-iteration HtH, Htz, residual-variance decomposition and step, frames %d to %d" % (LO, HI), win, "window_update_diag_all.csv.gz", UPD_COLS, 16, expected_per=w("window_update_diag_all.csv.gz"))
    add("F-449a", "per-frame rows used and misses by reason (coverage, mismatch, tier 0), NIS, position-velocity covariance norm, frames %d to %d" % (LO, HI), win, "window_frame_stats_all.csv.gz", FS_COLS, 16, expected_per=w("window_frame_stats_all.csv.gz"))
    add("F-449a", "state at the start of each dumped iteration, frames %d to %d" % (LO, HI), win, "window_dump_state_all.csv.gz", DSTATE_COLS, 16, expected_per=w("window_dump_state_all.csv.gz"))
    add("F-449a", "per (frame, iteration, ground / other): rows, summed weight, mean r2 / sigma2, rows over the gate, quantiles, frames %d to %d" % (LO, HI), win, "window_row_summary_all.csv", SUM_COLS, 60, expected_per=w("window_row_summary_all.csv"))
    add("F-449a", "every ground row (|world normal z| >= 0.8) of frames %d to %d" % (GLO, GHI), win, "window_ground_rows_all.csv.gz", GROUND_COLS, 500, expected_per=lambda a: 1500000 if by[a]["pfn"] == "1" else 600000)
    add("F-389", "per-message LiDAR log, every row", msg, "lidar_msg_log_all.csv.gz", MSG_COLS, 100, expected_per=lambda a: 100 * (N_SCANS[by[a]["bag"]] if by[a]["duration_s"] == "full" else 620))
    add("F-463", "per-scan position, yaw and quaternion of every NON-sweep arm (identity vs R79, divergence of the window arms; the sweep arms return summaries only, D1: raw data stays in the host archive)", notsw, "pos_trace_all.csv.gz", POS_COLS, 100,
        expected_per=lambda a: 56 * (N_SCANS[by[a]["bag"]] if by[a]["duration_s"] == "full" else 620))
    add("F-437", "motion-Q trace (motion_q_scan.csv, every row) of the eee_02 lines-off 0.425 m motion-Q BASE3 arm (D2)", [MQ_TRACE_ARM], "motion_q_scan_all.csv.gz", ["scan_index", "t_rel_s", "model", "acc_scale", "gyro_scale", "mean_q_acc_x"], 2000, expected_per=lambda a: 160000, maxb=400000)
    add("F-467", "zupt.csv per frame of the nine detect_only arms (gyro mean, acceleration mean, run length, stationary flag) (D7); reducer: gzip copy of zupt.csv", sel(lambda t: t["group"] == "ZD"), "zupt_all.csv.gz", ["frame", "gyro_mean", "acc_mean", "run", "stationary", "applied", "vnorm_before"], 100, expected_per=lambda a: 300000)
    add("F-476", "q_adapt.csv per scan of the 18 two-sided NIS gate arms (gain, NIS, inflate / deflate / hold); reducer: gzip copy of q_adapt.csv", sel(lambda t: t["group"] == "NG"), "q_adapt_all.csv.gz", [], 100, expected_per=lambda a: 100000)
    add("F-403", "step_cap.csv.gz (frame, iter, rot_raw, pos_raw, rot_applied, pos_applied, scale, bound, next_converged) of the two D5 arms", sel(lambda t: t["group"] == "SC"), "step_cap.csv.gz", ["frame", "iter", "rot_raw", "pos_raw", "rot_applied", "pos_applied", "scale", "bound", "next_converged"], 20, expected_per=lambda a: 30000)
    for f, cols, mr, ep in EARLY_FILES:
        ar = sel(lambda t: t["group"] == "EC3")
        add("F-465", "D3 early-frame file %s (frames 1 to 5)" % f, ar, f, cols, mr, expected_per=lambda a, ep=ep: ep(by[a]))
    add("F-463", "per-scan position, yaw and quaternion of the early D3 / D5 arms", sel(lambda t: t["group"] in ("EC3", "SC")), "pos_trace_all.csv.gz", POS_COLS, 5, expected_per=lambda a: 56 * 150)
    add("F-452", "gates: identity vs R79 / R80 controls, determinism digest, effective config of every group, engagement, window files, message log, frame mapping, refusals, exit_status, sweep completeness", [], "r80_gates.txt", [], 1, source="generated")
    add("F-452", "identity comparison of every identity arm against its R79 counterpart", [], "r80_identity_vs_old.csv", ["arm", "ref_arm", "n_common_scans", "pos_max_abs_diff_mm", "yaw_max_abs_diff_mrad", "dump_equal", "node_id_equal", "digest_equal_if_known"], 10, source="generated")
    add("F-452", "determinism control and its original", [], "identity_controls.txt", [], 1, source="generated")
    for f, t in (("r80_calib_window_edges.csv", "D6 F-450 / F-448: per bag and edge, share of points in plane leaves, median sqrt(eig0)/edge, share of degenerate planes (host tool tools/r69/r80_calib_window_scan.py, run by the agent on the nine bags; no selection column)"),
                 ("r80_calib_window_layers.csv", "D6: max-layer census per bag, edge and max_layer"), ("r80_calib_window_provenance.json", "D6: inputs, thresholds used, tool sha256")):
        add("F-450", t, [], f, [], 1, source="generated")
    add("F-409", "engagement of every dispatched arm concatenated", [], "engagement_all.txt", [], 10, source="generated")
    add("F-409", "per-arm completeness (arm_table 'requires' column)", [], "completeness.csv", ["arm", "requires_missing"], 10, source="generated")
    add("F-326", "exit_status.txt of every dispatched arm: the real node exit status, refuse arms non-zero, others zero", allarm, "exit_status.txt", [], 1, where="archive")
    add("F-407", "engagement_imu.txt present for every dispatched arm that is not a refuse arm", allarm, "engagement_imu.txt", [], 1, where="archive")
    add("F-469", "plane-fit trace counts of the two debiased arms and the BASE3 eee_01 arms (tools/r69/plane_fit_trace_counts.py, run on the archive by the agent)", [], "plane_fit_trace_counts_all.csv", [], 5, source="generated")
    add("F-415", "TAIL-1 coupled_update_iter.csv of the three coupled arms stays in the host archive (size); the reducer summaries are in phase_summary.csv", sel(lambda t: t["group"] == "TL"), "coupled_update_iter.csv", [], 1, where="archive")
    return R


def ledger(RO):
    out = []
    for e in RO:
        if e["where"] != "return" or not e["expected_bytes"] or e["source"] == "generated":
            continue
        new = e["file"].startswith("window_") or e["file"] in ("lidar_msg_log_all.csv.gz", "zupt_all.csv.gz", "q_adapt_all.csv.gz", "step_cap.csv.gz", "snapshot_frames_all.csv.gz", "state_trace_early_all.csv.gz",
                                                               "lio_update_diag_early_all.csv.gz", "residual_dump_state_all.csv.gz") or (e["file"] == "residual_dump_all.csv.gz" and e["min_rows"] >= 500)
        out.append({"pattern": e["file"], "max_bytes_per_file": e["max_bytes_per_file"], "max_total_bytes": int(e["expected_bytes"] * (3 if new else 1.5)),
                    "basis": ("no measurement of the real file yet; estimate from the R78/R79 rows; the agent measures one arm of each with size_ledger.py measure; allowance 3x" if new else
                              "measured on the R79 return: pos_trace 56 bytes per scan; allowance 1.5x"), "note": e["followup"] + " (%d arms)" % len(e["arms"])})
    for x in out:
        if x["pattern"] == "pos_trace_all.csv.gz":
            x["max_total_bytes"] += 100 * 56 * 4300   # D1: sweep arms with gt_max_mm > 500 return their trace (collapse rule); count unknown, ceiling 100 arms of the longest bag
            x["basis"] += "; plus a ceiling for up to 100 sweep arms whose trace is returned because gt_max_mm > 500 (the count is unknown in advance)"
    for pat, mx, tot in (("r80_gates.txt", 900000, 900000), ("identity_controls.txt", 150000, 150000), ("r80_identity_vs_old.csv", 40000, 40000), ("engagement_all.txt", 2000000, 2000000),
                         ("completeness.csv", 600000, 600000), ("phase_summary.csv", 8000000, 8000000), ("arm_scan_all.csv.gz", 70000000, 70000000), ("raw_sha256.txt", 20000, 20000),
                         ("R80_tree.tar.gz", 80000000, 80000000), ("agent_edits.patch", 2000000, 2000000), ("listener_*.log", 5000000, 40000000), ("*.md", 1000000, 3000000),
                         ("*.json", 2000000, 20000000), ("*.jsonl", 2000000, 10000000), ("*.txt", 1000000, 10000000), ("*.tsv", 1000000, 3000000), ("*.csv", 1000000, 5000000), ("*.sha256", 100000, 100000)):
        out.append({"pattern": pat, "max_bytes_per_file": mx, "max_total_bytes": tot, "basis": "generic ceiling (R74-R80 ledger values or a round figure); not a measurement", "note": "non-arm file"})
    res, at = [], {}
    for x in out:
        if x["pattern"] in at:
            y = res[at[x["pattern"]]]
            y["max_bytes_per_file"] = max(y["max_bytes_per_file"], x["max_bytes_per_file"]); y["max_total_bytes"] += x["max_total_bytes"]; y["note"] += " + " + x["note"]
        else:
            at[x["pattern"]] = len(res); res.append(dict(x))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", default="r80_manifest")
    ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--min-free-gb", type=float, default=150.0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    J = {j["job_id"]: j for j in json.load(open(a.jobs))}
    os.makedirs(a.out, exist_ok=True)
    out = {}; table = []; seen = set()
    for r in arms():
        if r["base"] not in J:
            sys.exit("base job %s not in r66_jobs.json" % r["base"])
        assert r["id"] not in seen, r["id"]; seen.add(r["id"])
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
        path = "%s/r80_%s/%s" % (a.results_root, r["batch"], r["id"])
        o["outputs.path"] = path; o["outputs.debug_log_dir"] = path
        if r["dur"]:
            j["duration_secs"] = r["dur"]
        out.setdefault(r["batch"], []).append(j)
        sampling = "pfn%s_ds%s" % (o.get("cbk.lidar.point_filter_num", "def"), o.get("imu.ds.mode", "def"))
        table.append(dict(arm=r["id"], tier=r["tier"], arch="iekf_raw", kind="closed", duration_s=str(j.get("duration_secs", "full")), seq=j["seq"], notes=r["purpose"], keep_raw="0", dup_of="",
                          r66_id=r["base"], group=r["group"], batch="r80_" + r["batch"], sampling=sampling, requires=" ".join([] if r["batch"] == "ref" else req_of(o)), f417=str(r["f417"]), plane_check="0",
                          asked=r["asked"], early=str(r["early"]), pfn=str(r["pfn"]), pe="1" if r["pe"] else "0", dump=str(r["dump"]), bag=r["bag"], win=str(r["win"]), msglog=str(r["msglog"])))
    for b, lst in out.items():
        json.dump(lst, open(os.path.join(a.out, "jobs_%s.json" % b), "w"), indent=1)
    cols = ["arm", "tier", "arch", "kind", "duration_s", "seq", "notes", "keep_raw", "dup_of", "r66_id", "group", "batch", "sampling", "requires", "f417", "plane_check", "asked", "early", "pfn", "pe", "dump", "bag", "win", "msglog"]
    with open(os.path.join(a.out, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t"); w.writeheader(); w.writerows(table)
    q = lambda b, n: {"dir": "ablations/_job_queue/r80_%s" % b, "results_root": "%s/r80_%s" % (a.results_root, b), "expected_jobs": n}
    # independent listener specs (rule 1.16): each spec stands alone; a failed spec stops nothing else
    specs = [("id", ["id", "ref"]), ("dp", ["dp", "ef"]), ("e1", ["e1"])] + [("x_" + b, ["x_" + b]) for b in BAGS] + [("sw_" + b, [c for c in sw_cells() if c.startswith("sw_%s_" % b)]) for b in BAGS]
    order = []
    for nm, bs in specs:
        sp = M.listener_spec("R80-%s" % nm, [q(b, len(out[b])) for b in bs], a.results_root, a.min_free_gb, a.workers)
        fn = "listener_spec_%s.json" % nm
        for blk in sp["stages"] + sp["verify"]:
            blk["cmd"] = [c.replace("%s", fn).replace("r74_manifest", "r80_manifest") for c in blk["cmd"]]
        sp["raw"]["delete"] = sp["raw"]["delete"] + ["iekf_factor_iteration.csv", "lidar_msg_log.csv", "post_calibration_voxel_observation_support.csv", "line_rows_per_scan.csv"]
        json.dump(sp, open(os.path.join(a.out, fn), "w"), indent=1)
        order.append("%s\t%s\t%d arms" % (fn, ",".join(bs) if len(bs) < 4 else "%d queues" % len(bs), sum(len(out[b]) for b in bs)))
    open(os.path.join(a.out, "queue_order.tsv"), "w").write("# listener spec\tqueues\tarms   (a preference, not a dependency: rule 1.16)\n" + "\n".join(order) + "\n")
    RO = required_outputs(table)
    json.dump(RO, open(os.path.join(a.out, "required_outputs.json"), "w"), indent=1)
    json.dump(ledger(RO), open(os.path.join(a.out, "r80_file_size_ledger.json"), "w"), indent=1)
    cnt = {}
    for t in table:
        cnt[t["group"]] = cnt.get(t["group"], 0) + 1
    print("arms %d: %s; queues %d; listener specs %d" % (len(table), ", ".join("%s %d" % kv for kv in sorted(cnt.items())), len(out), len(specs)))


if __name__ == "__main__":
    main()
