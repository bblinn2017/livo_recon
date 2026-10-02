#!/usr/bin/env python3
"""R73: build jobs_<batch>.json, arm_table.tsv, scan_arms_priority.txt, coupled_pairs.txt, shadow_pairs.txt, required_outputs.json and the listener specs from the R66 job
list (r66_jobs.json).

    python3 tools/r69/make_r73_manifest.py --jobs r69_inputs/r66_jobs.json --out r73_manifest --results-root /root/catkin_ws/livo_recon_results

Planning agent's script; NOT run against the real host by me (it is run in the selftest on the shipped r66_jobs.json only: counts and key checks).
Every arm is an R66 job (same overrides) plus the delta written below. estimator.mode "decoupled" -> "iekf"; spline.mode removed. No perturbation key anywhere (rule 3.11).
Key names are the R73 names (F-404): imu.process_noise.gain.* (was lio.q_adapt.*), imu.process_noise.disturbance.* (was imu.process_noise.continuous_prior.*).

Batches (queues)
  smoke    identity gates (R73 code inert at defaults; the shadow and the logs must not change the driver) and one 60 s smoke per new flag (rule 3.6)
  refuse   jobs that MUST fail at startup (new dependency refusals, the retired key names)
  sh       F-414: shadow-update pairs on sbs_03 and eee_02 (70 s), a determinism repeat, plain controls and the map-freeze controls
  co       F-415: 8 coupled 60 s arms with the update log, IMU-prior information-scale arms (60 s), iekf controls (60 s). No full-bag coupled arm (Bryce asked for short runs).
  qa       F-401 / F-418: the 18 motion-Q alternative arms on eee_01 (released)
  p2       F-400 Stage 2: motion-Q, derived P0, gyro_scale {7,10,12,14,17} x acc_scale {0,0.3,0.6}, 11 new cells + the centre control, 9 bags (108 arms)
The requires column of arm_table.tsv names the files an arm must have produced (derived from its overrides); r73_score_job.py fails the arm's stage if one is absent,
and required_outputs.json (follow-up -> arm group -> file -> columns -> minimum rows -> return/archive) drives the return checklist (build_return.py).
"""
import argparse, collections, copy, csv, json, math, os, sys

BAGS = ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]
DIAG = {"lio.update_diag_en": True, "lio.live_point_cov_en": True, "lio.first_frame_dump_frames": 5, "lio.residual_geometry_en": True}
MQ = {"imu.process_noise.model": "isotropic", "imu.process_noise.motion.beta": 0.9, "imu.process_noise.motion.acc_scale": 0.3, "imu.process_noise.motion.gyro_scale": 10.0,
      "imu.process_noise.motion.acc_max_dynamic_variance": 1000000.0, "imu.process_noise.motion.gyro_max_dynamic_variance": 1000000.0}
CO_FULL = {"estimator.coupled.minimal_diagnostics": False}
CO_REC = {"estimator.coupled.failure_dump": True, "estimator.coupled.record_frames": 5}
CO_LOG = {"estimator.coupled.update_log_en": True}
STEPCAP = lambda rot=1e-2, pos=1e-2: {"lio.ekf.step_cap.mode": "hard", "lio.ekf.step_cap.rotation_rad": rot, "lio.ekf.step_cap.position_m": pos}
ZUPT = lambda mode, g=0.02, a=0.15, n=3, s=0.02: dict({"lio.zupt.mode": mode, "lio.zupt.gyro_thresh": g, "lio.zupt.acc_thresh": a, "lio.zupt.enter_scans": n},
                                                      **({"lio.zupt.sigma_v": s} if mode == "apply" else {}))
GAIN = lambda mode, **kw: dict({"imu.process_noise.gain.mode": mode}, **{"imu.process_noise.gain." + k: v for k, v in kw.items()})
WNOA = lambda qa, qg: {"imu.process_noise.disturbance.mode": "wnoa", "imu.process_noise.disturbance.qc_acc": qa, "imu.process_noise.disturbance.qc_gyr": qg}
FREEZE = {"voxel_map.map.freeze_after_calibration": True}
SCANLOG = {"lio.scan_log_en": True}
SH_DUR = 70
# dump frames (map frame_idx), inclusive ranges
SBS_WIN, SBS_DUMP = (330, 420), (370, 380)
EEE_WIN, EEE_DUMP = (470, 540), (488, 498)


def mqs(acc, gyro):
    return {"imu.process_noise.motion.acc_scale": float(acc), "imu.process_noise.motion.gyro_scale": float(gyro)}


def rng(lo, hi):
    return ",".join(str(i) for i in range(lo, hi + 1))


def shadow(win, model, acc=None, gyr=None):
    o = {"lio.shadow.mode": "on", "lio.shadow.window_start": win[0], "lio.shadow.window_end": win[1], "lio.shadow.process_noise_model": model}
    if model == "isotropic":
        o.update({"lio.shadow.motion_beta": 0.9, "lio.shadow.motion_acc_scale": float(acc), "lio.shadow.motion_gyr_scale": float(gyr),
                  "lio.shadow.motion_acc_cap": 1.0e6, "lio.shadow.motion_gyr_cap": 1.0e6})
    return o


def base_of(bag, q):
    """R66 base job: eee_01 identity jobs, other bags the R67 50_E jobs. q in fq|mq."""
    if bag == "eee_01":
        return "00_id_baseline_full" if q == "fq" else "00_id_motionq_full"
    return "50_E_%s_%s_full" % (bag, "baseline" if q == "fq" else "motionq")


# files an arm must produce, derived from its overrides (arm_table.tsv column "requires"; r73_score_job.py fails the arm's stage when one is absent)
def requires_of(o, batch):
    if batch == "refuse":
        return []
    r = ["engagement_imu.txt"]
    if o.get("lio.scan_log_en"):
        r.append("scan_log.csv")
    if o.get("lio.shadow.mode") == "on":
        r += ["shadow_scan.csv", "shadow_cov.csv", "engagement_imu_shadow.txt"]
    if o.get("lio.residual_dump_frames"):
        r += ["residual_dump.csv", "residual_dump_state.csv"]
        if o.get("lio.shadow.mode") == "on":
            r += ["shadow_residual_dump.csv", "shadow_residual_dump_state.csv"]
    if o.get("lio.update_diag_en"):
        r.append("lio_update_diag.csv")
    if o.get("estimator.coupled.update_log_en"):
        r += ["coupled_update_iter.csv", "coupled_update_scan.csv"]
    if o.get("estimator.coupled.record_frames"):
        r += ["coupled_record_f%d.txt" % i for i in range(1, int(o["estimator.coupled.record_frames"]) + 1)]
    if o.get("imu.process_noise.gain.mode") not in (None, "off"):
        r.append("q_adapt.csv")
    if o.get("lio.zupt.mode") not in (None, "off"):
        r.append("zupt.csv")
    if o.get("lio.ekf.step_cap.mode") == "hard":
        r.append("step_cap.csv")
    return r


def arms(J):
    A = []

    def add(id_, base, delta=None, batch="main", group="", tier="", purpose="", keep=0, dur=None):
        A.append(dict(id=id_, base=base, delta=delta or {}, batch=batch, group=group, tier=tier, purpose=purpose, keep=keep, dur=dur))

    def d(*ds):
        o = {}
        for x in ds:
            o.update(x)
        return o

    B60, BF, BM = "00_id_baseline_60s", "00_id_baseline_full", "00_id_motionq_full"
    # ---- smoke: identity gates. The three reference arms get NEW digests (F-419 symmetrisation changes the arithmetic); every other identity arm must equal its R73 reference digest.
    for n in (B60, BF, BM):
        add("r73_" + n, n, batch="smoke", group="ID", tier="ID", keep=1,
            purpose="R73 REFERENCE: ATE / excursion / settled must equal the R72 values within 0.005 mm (F-419 decision); its digest becomes the R73 identity reference")
    IDP = lambda id_, delta, why: add(id_, B60, delta, batch="smoke", group="ID", tier="ID", keep=1, purpose="IDENTITY: " + why + "; digest equals r73_00_id_baseline_60s")
    IDP("r73_id_stepcap_wide_60s", STEPCAP(1e6, 1e6), "step cap with caps 1e6 never binds, engagement NOT ENGAGED (bound=0) and iterations > 0")
    IDP("r73_id_zupt_detect_60s", ZUPT("detect_only"), "ZUPT detect_only runs the detector and zupt.csv but must not touch the state")
    IDP("r73_id_nisgate_dead_60s", GAIN("nis_gate", nis_hi=1e9), "nis_gate with nis_hi 1e9 cannot inflate (new key names, F-404)")
    IDP("r73_id_diag_on_60s", DIAG, "write-only logs on")
    IDP("r73_id_scanlog_60s", SCANLOG, "scan_log.csv (map statistics, digests) is write-only; it must not change the driver")
    IDP("r73_id_dump_lastiter_60s", {"lio.residual_dump_frames": "1,2,3", "lio.residual_dump_last_iter": True}, "residual dump with the last-iteration option is write-only")
    IDP("r73_id_shadow_same_60s", d(SCANLOG, shadow((20, 50), "same"), {"lio.residual_dump_frames": rng(30, 32)}),
        "the shadow update (same process noise, window 20-50) runs against the driver's map and must leave the DRIVER byte-identical (this is the proof that the shadow pass has no side effect on the driver, F-414)")
    # ---- smoke: one 60 s smoke per new flag (rule 3.6)
    SM = lambda id_, base, delta, why, dur=60: add(id_, base, delta, batch="smoke", group="SM", tier="SMOKE", dur=dur, keep=1, purpose=why)
    SM("r73_smoke_stepcap_fq_60s", B60, d(DIAG, STEPCAP()), "smoke of the step cap, fixed-Q: step_cap.csv written, bound > 0")
    SM("r73_smoke_zupt_apply_60s", BM, d(DIAG, ZUPT("apply")), "smoke of ZUPT apply under motion-Q: applied > 0")
    SM("r73_smoke_innov_scale_60s", BM, d(DIAG, GAIN("innovation_scale")), "smoke of innovation_scale (key imu.process_noise.gain.mode): q_adapt.csv written, engagement counts agree")
    SM("r73_smoke_nis_gate_60s", BM, d(DIAG, GAIN("nis_gate")), "smoke of nis_gate")
    SM("r73_smoke_local_poly_60s", BM, {"imu.process_noise.motion.energy_source": "local_poly"}, "smoke of the local-polynomial energy source: engagement_imu.txt scans_used > 0")
    SM("r73_smoke_wnoa_60s", BM, WNOA(0.01, 0.0001), "smoke of the WNOA disturbance (key imu.process_noise.disturbance.*): engagement_imu.txt wnoa steps > 0")
    SM("r73_smoke_shadow_iso_60s", B60, d(SCANLOG, shadow((20, 50), "isotropic", 0.3, 10.0), {"lio.residual_dump_frames": rng(30, 32), "lio.residual_dump_last_iter": True}),
       "smoke of the shadow update (fixed-Q driver, isotropic motion-Q shadow): shadow_scan.csv has a driver and a shadow row for every frame 20-50, shadow_cov.csv, shadow residual dump for frames 30-32, map digests before = after for every shadow pass")
    SM("r73_smoke_shadow_fixed_mq_60s", BM, d(SCANLOG, shadow((20, 50), "fixed")), "smoke of the shadow update the other way round (motion-Q driver, fixed-Q shadow)")
    SM("r73_smoke_freeze_60s", B60, d(SCANLOG, FREEZE), "smoke of voxel_map/map/freeze_after_calibration: scan_log.csv map_frozen_scans counts every live frame, the map digest does not change after frame 1")
    for t in ("tail", "measurement"):
        SM("r73_smoke_co_%s_rec_60s" % t, "10_smoke_coupled_%s_fulldiag_60s" % t, d(CO_FULL, CO_REC, CO_LOG),
           "smoke of the coupled record AND the coupled update log: coupled_record_f1..f5.txt, coupled_update_iter.csv, coupled_update_scan.csv; no failure_dump file")
    SM("r73_smoke_co_ipscale_60s", "10_smoke_coupled_measurement_fulldiag_60s", d(CO_FULL, CO_LOG, {"estimator.coupled.imu_prior_info_scale": 0.1}),
       "smoke of estimator/coupled/imu_prior_info_scale 0.1: the engagement line states it and coupled_update_iter.csv has imu_prior_info_scale 0.1 in every row")
    # ---- refuse (every one MUST die at startup)
    R = lambda id_, base, delta, purpose: add(id_, base, delta, batch="refuse", group="REF", tier="REF", dur=5, purpose=purpose)
    R("r73_refuse_stepcap_suboption", B60, {"lio.ekf.step_cap.rotation_rad": 0.01}, "MUST FAIL: step_cap/rotation_rad outside mode=hard")
    R("r73_refuse_stepcap_coupled", "10_smoke_coupled_tail_fulldiag_60s", STEPCAP(), "MUST FAIL: step cap under estimator/mode=coupled")
    R("r73_refuse_stepcap_zero", B60, STEPCAP(0.0, 0.01), "MUST FAIL: step cap with a zero cap")
    R("r73_refuse_gain_suboption", B60, {"imu.process_noise.gain.forgetting": 0.5}, "MUST FAIL: gain/forgetting outside mode=innovation_scale")
    R("r73_refuse_gain_inert_forgetting", BM, GAIN("innovation_scale", forgetting=1.0), "MUST FAIL: forgetting 1 is inert")
    R("r73_refuse_gain_inert_bounds", BM, GAIN("nis_gate", min_gain=1.0, max_gain=1.0), "MUST FAIL: gain bounds 1..1 are inert")
    R("r73_refuse_gain_coupled", "10_smoke_coupled_tail_fulldiag_60s", GAIN("nis_gate"), "MUST FAIL: gain keys under coupled are unclaimed")
    R("r73_refuse_zupt_suboption", B60, {"lio.zupt.sigma_v": 0.02}, "MUST FAIL: zupt/sigma_v outside mode=apply")
    R("r73_refuse_zupt_threshold", B60, ZUPT("detect_only", g=-1.0), "MUST FAIL: negative gyro threshold")
    R("r73_refuse_local_poly_fixed", B60, {"imu.process_noise.motion.energy_source": "local_poly"}, "MUST FAIL: local_poly under the fixed model")
    R("r73_refuse_local_poly_order_instant", BM, {"imu.process_noise.motion.local_poly.order": 2}, "MUST FAIL: local_poly/order under energy_source=instant")
    R("r73_refuse_wnoa_zero", B60, WNOA(0.0, 0.0), "MUST FAIL: wnoa with both densities 0 is inert")
    R("r73_refuse_wnoa_suboption", B60, {"imu.process_noise.disturbance.qc_acc": 0.01}, "MUST FAIL: qc_acc outside mode=wnoa")
    R("r73_refuse_old_qadapt_name", BM, {"lio.q_adapt.mode": "nis_gate"}, "MUST FAIL (F-404): the retired key lio/q_adapt/mode; the message must say 'renamed' and name imu/process_noise/gain/mode")
    R("r73_refuse_old_cprior_name", BM, {"imu.process_noise.continuous_prior.mode": "wnoa", "imu.process_noise.continuous_prior.qc_acc": 0.01, "imu.process_noise.continuous_prior.qc_gyr": 0.0001},
      "MUST FAIL (F-404): the retired key imu/process_noise/continuous_prior/mode; message names imu/process_noise/disturbance/mode")
    R("r73_refuse_shadow_window0", B60, {"lio.shadow.mode": "on", "lio.shadow.window_start": 0, "lio.shadow.window_end": 10}, "MUST FAIL: shadow window_start must be >= 1")
    R("r73_refuse_shadow_window_order", B60, {"lio.shadow.mode": "on", "lio.shadow.window_start": 20, "lio.shadow.window_end": 10}, "MUST FAIL: shadow window_end < window_start")
    R("r73_refuse_shadow_suboption", B60, {"lio.shadow.window_start": 10}, "MUST FAIL: shadow sub-key with lio/shadow/mode=off")
    R("r73_refuse_shadow_motion_sub", B60, {"lio.shadow.mode": "on", "lio.shadow.window_start": 10, "lio.shadow.window_end": 20, "lio.shadow.motion_beta": 0.5},
      "MUST FAIL: lio/shadow/motion_beta with process_noise_model=same")
    R("r73_refuse_shadow_coupled", "10_smoke_coupled_tail_fulldiag_60s", {"lio.shadow.mode": "on", "lio.shadow.window_start": 10, "lio.shadow.window_end": 20}, "MUST FAIL: shadow under estimator/mode=coupled")
    R("r73_refuse_scanlog_coupled", "10_smoke_coupled_tail_fulldiag_60s", SCANLOG, "MUST FAIL: scan_log under estimator/mode=coupled")
    R("r73_refuse_ipscale_iekf", B60, {"estimator.coupled.imu_prior_info_scale": 0.5}, "MUST FAIL: estimator/coupled/imu_prior_info_scale outside estimator/mode=coupled")
    R("r73_refuse_ipscale_zero", "10_smoke_coupled_measurement_fulldiag_60s", {"estimator.coupled.imu_prior_info_scale": 0.0}, "MUST FAIL: imu_prior_info_scale must be > 0")
    R("r73_refuse_dump_last_iter_alone", B60, {"lio.residual_dump_last_iter": True}, "MUST FAIL: residual_dump_last_iter without lio/residual_dump_frames")
    # ---- sh: F-414 shadow pairs (70 s), controls, determinism repeat, map-freeze controls
    for bag, win, dmp, tag in (("sbs_03", SBS_WIN, SBS_DUMP, "sbs03"), ("eee_02", EEE_WIN, EEE_DUMP, "eee02")):
        dump = {"lio.residual_dump_frames": rng(*dmp), "lio.residual_dump_last_iter": True}
        mq2 = mqs(1.0, 5.0) if bag == "sbs_03" else {}
        qn = "g5a1" if bag == "sbs_03" else "mq"
        bfq, bmq = base_of(bag, "fq"), base_of(bag, "mq")
        SH = lambda id_, base, delta, why, grp="SH": add(id_, base, d(SCANLOG, dump if grp == "SH" else {}, delta), batch="sh", group=grp, tier=grp, dur=SH_DUR, keep=1, purpose=why)
        SH("r73_sh_%s_ctl_fq" % tag, bfq, {}, "F-414 plain control, %s fixed-Q, 70 s, scan log and residual dump of frames %d-%d, no shadow" % (bag, dmp[0], dmp[1]))
        SH("r73_sh_%s_ctl_%s" % (tag, qn), bmq, mq2, "F-414 plain control, %s motion-Q %s, 70 s, no shadow" % (bag, qn))
        if bag == "sbs_03":
            SH("r73_sh_sbs03_a", bfq, shadow(win, "isotropic", 1.0, 5.0), "F-414 pair (a): driver fixed-Q baseline, SHADOW motion-Q gyro 5 acc 1, window %d-%d" % win)
            SH("r73_sh_sbs03_b", bmq, d(mq2, shadow(win, "fixed")), "F-414 pair (b): driver motion-Q gyro 5 acc 1, SHADOW fixed-Q, window %d-%d" % win)
            SH("r73_sh_sbs03_a_rep", bfq, shadow(win, "isotropic", 1.0, 5.0), "F-414 determinism repeat of pair (a): every column of shadow_scan.csv and shadow_cov.csv must reproduce bit for bit")
        else:
            SH("r73_sh_eee02_d", bfq, shadow(win, "isotropic", 0.3, 10.0), "F-414 pair (d): driver fixed-Q, SHADOW motion-Q (0.3, 10), window %d-%d" % win)
            SH("r73_sh_eee02_e", bmq, shadow(win, "fixed"), "F-414 pair (e): driver motion-Q (0.3, 10), SHADOW fixed-Q, window %d-%d" % win)
        SH("r73_fz_%s_fq" % tag, bfq, FREEZE, "F-414 secondary control: map frozen after calibration, %s fixed-Q (valid only if its frame-1 map digest equals the plain control's)" % bag, "FZ")
        SH("r73_fz_%s_%s" % (tag, qn), bmq, d(mq2, FREEZE), "F-414 secondary control: map frozen after calibration, %s motion-Q %s" % (bag, qn), "FZ")
    # ---- co: F-415 / F-410
    # Bryce (2026-10-01): the coupled arms run SHORT (60 s). The three full-bag F-410 reruns that were here were NOT asked for and are removed (addendum 5, rule 1.12).
    for t, tn in (("tail", "tail"), ("measurement", "meas")):
        for k in (4, 8):
            for q, qn in (("fixedq", "fq"), ("motionq", "mq")):
                n = "60_D3_coupled_%s_gateon_knots%d_iters5_%s_full" % (t, k, q)
                add("r73_co60_%s_k%d_%s" % (tn, k, qn), n, d(CO_FULL, CO_REC, CO_LOG), batch="co", group="CO", tier="CO", dur=60, keep=1,
                    purpose="F-415: coupled %s, gate on, %d knots, %s, first 60 s, update log on, record frames 1-5, failure_dump on" % (t, k, qn))
    for q in ("fq", "mq"):
        add("r73_co_control_iekf_%s_60s" % q, base_of("eee_01", q), d(DIAG, SCANLOG), batch="co", group="CO", tier="CO", dur=60, keep=1,
            purpose="F-415 control: splineless iekf, same Q, first-frame matrices frames 1-5, scan log on")
    for sc in (0.01, 0.1, 1.0):
        add("r73_co_ips_%s_meas_k4_mq_60s" % str(sc).replace(".", "p"), "60_D3_coupled_measurement_gateon_knots4_iters5_motionq_full", d(CO_FULL, CO_REC, CO_LOG, {"estimator.coupled.imu_prior_info_scale": sc}),
            batch="co", group="CO", tier="CO", dur=60, keep=1,
            purpose="F-415: IMU-prior information scale %s (joint prior P^-1 times %s), measurement mode, 4 knots, motion-Q, 60 s%s" % (sc, sc, " (CONTROL: explicit 1.0, must equal r73_co60_meas_k4_mq)" if sc == 1.0 else ""))
    # ---- qa: F-401 / F-418 (released; PROVISIONAL parameters kept; each arm is reported as its own arm, rule 4.15a)
    MQB = BM
    QA = lambda id_, base, delta, p: add(id_, base, d(DIAG, delta), batch="qa", group="QA", tier="QA", keep=1, purpose="F-401 / F-418: " + p)
    for f_ in (0.8, 0.95):
        QA("r73_qa_innov_f%s_mq" % str(f_).replace(".", "p"), MQB, GAIN("innovation_scale", forgetting=f_), "innovation_scale on motion-Q, forgetting %s" % f_)
    QA("r73_qa_innov_fq", BF, GAIN("innovation_scale"), "innovation_scale on fixed-Q (does it recover the motion-Q behaviour)")
    for hi in (1.3, 1.6):
        QA("r73_qa_nis_hi%s_mq" % str(hi).replace(".", "p"), MQB, GAIN("nis_gate", nis_hi=hi), "nis_gate on motion-Q, nis_hi %s" % hi)
    QA("r73_qa_nis_white_mq", MQB, GAIN("nis_gate", whiteness_max=0.3), "nis_gate with whiteness gate 0.3 on motion-Q")
    QA("r73_qa_nis_fq", BF, GAIN("nis_gate"), "nis_gate on fixed-Q")
    for o_ in (1, 2):
        QA("r73_qa_localpoly_o%d_mq" % o_, MQB, {"imu.process_noise.motion.energy_source": "local_poly", "imu.process_noise.motion.local_poly.order": o_},
           "local polynomial energy source order %d (replacement of the instantaneous energy; same beta/scale/caps)" % o_)
    for qa_, qg_ in ((0.001, 1e-5), (0.01, 1e-4), (0.1, 1e-3)):
        QA("r73_qa_wnoa_a%s_fq" % str(qa_).replace(".", "p"), BF, WNOA(qa_, qg_), "WNOA disturbance added to fixed-Q: qc_acc %s qc_gyr %s" % (qa_, qg_))
    QA("r73_qa_wnoa_mq", MQB, WNOA(0.01, 1e-4), "WNOA added to motion-Q (addition rather than replacement)")
    for tag, g_, a_, s_ in (("loose", 0.05, 0.3, 0.05), ("mid", 0.02, 0.15, 0.02), ("tight", 0.01, 0.08, 0.01)):
        QA("r73_qa_zupt_%s_mq" % tag, MQB, ZUPT("apply", g_, a_, 3, s_), "ZUPT apply on motion-Q, thresholds gyro %s rad/s acc %s m/s2, sigma_v %s m/s" % (g_, a_, s_))
    QA("r73_qa_zupt_mid_fq", BF, ZUPT("apply"), "ZUPT apply on fixed-Q, mid thresholds")
    QA("r73_qa_zupt_detect_mq", MQB, ZUPT("detect_only"), "ZUPT detect_only on motion-Q full bag: the detector statistics that fix the thresholds (control, byte-identical to motion-Q)")
    # ---- p2: F-400 Stage 2, motion-Q, derived P0. 11 new cells x 9 bags + the centre cell (0.3, 10) as an in-round control on every bag.
    DONE_S1 = {(0.0, 10), (0.3, 10), (0.0, 14), (0.3, 14)}      # already run in R72 Stage 1 (the centre (0.3, 10) is re-run as the control)
    cells = [(a_, g_) for g_ in (7, 10, 12, 14, 17) for a_ in (0.0, 0.3, 0.6) if (a_, g_) not in DONE_S1]
    assert len(cells) == 11, len(cells)
    for bag in BAGS:
        add("r73_p2_%s_mq_a0p3_g10" % bag, base_of(bag, "mq"), mqs(0.3, 10), batch="p2", group="P2C", tier="P2", keep=0,
            purpose="F-400 Stage 2 CENTRE CONTROL on %s: motion-Q acc 0.3, gyro 10, derived P0 (in-round reference for the ratios and for the alignment-free transition displacement)" % bag)
        for a_, g_ in cells:
            add("r73_p2_%s_mq_a%s_g%d" % (bag, str(a_).replace(".", "p"), g_), base_of(bag, "mq"), mqs(a_, g_), batch="p2", group="P2", tier="P2", keep=0,
                purpose="F-400 Stage 2, %s, motion-Q acc_scale %s gyro_scale %d, derived P0" % (bag, a_, g_))
    return A


def listener_spec(rnd, queues, out_root, min_free_gb, workers):
    return {
        "spec_version": 1, "round": rnd, "queues": queues,
        "trigger": {"sentinels": ["DONE", "FAILED"], "settle_seconds": 15, "settle_files": ["state_trace.csv"], "quiet_files": ["run.log"]},
        "on_failed_job": "skip",
        "path_maps": [["/root/catkin_ws", "/home/bblinn/fast_ws"]],
        "stages": [{
            "name": "reduce",
            # AGENT FIX (R69): these four paths are relative to src/livo_recon/ (where
            # tools/r69, r73_manifest, r69_inputs actually live), but sweep_listener.py's
            # cwd=root for the stage subprocess is the SAME root used to resolve
            # queues[].dir (ablations/_job_queue/..., which only exists under fast_ws,
            # since gen_jobs.py's REPO_ROOT is hardcoded to /root/catkin_ws == fast_ws).
            # One --root can't be both fast_ws (for queue/out_dir resolution, which also
            # needs path_maps below to translate job.env's OUT_DIR) and src/livo_recon
            # (for these tool/data paths) at the same time -- confirmed live: running with
            # --root fast_ws found all 10 smoke/refuse jobs correctly (raw_seen events with
            # real byte counts) but every reduce stage failed exit code 2, "can't open file
            # '/home/bblinn/fast_ws/tools/r69/r69_score_job.py'" (it only exists under
            # src/livo_recon/tools/r69/). Prefixing these four with src/livo_recon/ lets
            # --root stay fast_ws (the root queue/out_dir resolution needs) while the stage
            # command still finds its own tools and data. {spec_dir} is unaffected (already
            # an absolute path built from the spec file's own location, not relative to root).
            "cmd": ["python3", "src/livo_recon/tools/r69/r73_score_job.py", "--job-dir", "{job_dir}", "--out-dir", "{out_dir}", "--stage-out", "{stage_out}",
                    "--job-id", "{job_id}", "--spec", "{spec_dir}/%s", "--arm-table", "src/livo_recon/r73_manifest/arm_table.tsv",
                    "--params", "src/livo_recon/r69_inputs/bag_params.json", "--tools-dir", "src/livo_recon/tools/r69"],
            "timeout_s": 1800, "nice": 10,
            "outputs": [
                {"path": "summary_stage.json", "kind": "json", "min_bytes": 100, "merge_into_summary": True,
                 "required_fields": ["schema_version", "arm", "seq", "n_scans", "window_check_ok", "p_reduce_ok", "stage_ok", "archive_files"]},
                {"path": "arm_bundle.zip", "kind": "zip", "min_bytes": 2000},
                {"path": "archive/archive_index.json", "kind": "json", "min_bytes": 10},
                {"path": "archive/archive.sha256", "kind": "sha_manifest", "min_bytes": 10},
            ],
            # R69C: the lossless gzip archives are persisted into listener_out/reduce/archive/ (R69B: they were not, so they died with tmp/)
            "persist": ["archive"]}],
        "verify": [{"name": "archive_roundtrip",
                    "cmd": ["python3", "src/livo_recon/tools/r69/verify_archive.py", "--job-dir", "{job_dir}", "--out-dir", "{out_dir}",
                            "--archive-dir", "{stage_out_of:reduce}/archive", "--spec", "{spec_dir}/%s"],
                    "timeout_s": 1800, "nice": 10}],
        "summary": {"schema_version": 3, "required_fields": ["schema_version", "arm", "n_scans", "window_check_ok", "p_reduce_ok", "listener"]},
        "raw": {"delete": ["state_trace.csv", "lio_gt_matched.csv", "imu_cov_growth.csv", "frame_stats.txt", "plane_fit_trace.csv", "sensor_stamps.csv",
                           "point_cov_stats.csv", "surface_map_frames.csv", "frame_window.csv", "akf_q.csv", "coupled_scan_status.csv", "motion_q_scan.csv",
                           "post_calibration_voxelmap.csv", "lio_update_diag.csv", "live_point_cov.csv", "pose_control_first_frame_solve_matrices.txt",
                           "residual_geometry.csv", "info_scale.csv", "redundancy_renorm.csv",
                           "residual_dump.csv", "residual_dump_state.csv", "joint_knot_all_scans.csv", "joint_knot_scan_summary.csv",
                           "step_cap.csv", "q_adapt.csv", "zupt.csv", "coupled_failure_dump.txt", "engagement_imu.txt",
                           "scan_log.csv", "shadow_scan.csv", "shadow_cov.csv", "shadow_residual_dump.csv", "shadow_residual_dump_state.csv",
                           "coupled_update_iter.csv", "coupled_update_scan.csv", "engagement_imu_shadow.txt"] + ["coupled_record_f%d.txt" % i for i in range(1, 6)],   # R73: all are in the lossless archive (BIG list of r73_score_job.py)
                "sha256": "twice"},
        "limits": {"workers": workers, "max_failures": 10, "max_attempts": 2, "min_free_gb": min_free_gb, "poll_interval_s": 10},
        "exit": {"when": "all_expected_handled"},
    }


def required_outputs(table):
    """Follow-up -> arms -> file -> columns -> minimum rows -> where (return | archive) -> source. build_return.py checks every entry and names the follow-up in a failure.
    'return' files are looked up in compact/<arm>/ (or at the return root for generated files); 'archive' files must be listed in <runs>/<arm>/raw_sha256.txt (lossless host archive)."""
    arm_ids = lambda grp=None, pre=None: [t["arm"] for t in table if (grp is None or t["group"] in grp) and (pre is None or t["arm"].startswith(pre))]
    sh_arms = [a for a in arm_ids({"SH"}) if "_ctl_" not in a]
    R = []

    def add(fu, title, arms, file, cols, min_rows, where="return", source="arm", contains=None):
        R.append(dict(followup=fu, title=title, arms=arms, file=file, columns=cols, min_rows=min_rows, where=where, source=source, contains=contains or {}))
    # F-414 shadow update
    add("F-414", "shadow scan table (driver and shadow rows per window scan: state steps, Q, covariances' traces, HtH/Htz iter 0 and last, digests)", sh_arms, "shadow_scan_all.csv.gz",
        ["frame", "role", "start_pos_x", "post_pos_x", "prior_trace_pos", "post_trace_pos", "HtH0_00", "Htz0_0", "map_digest_before_pass", "map_digest_after_pass"], 50)
    add("F-414", "shadow and driver full prior and posterior covariance per window scan", sh_arms, "shadow_cov_all.csv.gz", ["frame", "role", "kind", "dim", "row_major_values"], 100)
    add("F-414", "per-residual dump of the driver, every dumped frame, with weight and class", arm_ids({"SH"}), "residual_dump_all.csv.gz",
        ["frame", "iter", "r", "sigma_squared", "weight", "cls", "plane_key", "node_id", "tier", "hrot_z", "hpos_x"], 1000)
    add("F-414", "per-residual dump of the shadow pass, every dumped frame, with weight and class", sh_arms, "shadow_residual_dump_all.csv.gz",
        ["frame", "iter", "r", "sigma_squared", "weight", "cls", "plane_key", "node_id", "tier", "hrot_z", "hpos_x"], 1000)
    add("F-414", "scan log (map statistics, digests, per-scan update summary) of every sh/fz arm", arm_ids({"SH", "FZ"}), "scan_log_all.csv.gz", ["frame", "map_digest_exact", "map_frozen_scans", "n_nodes", "n_resid_iter0"], 300)
    add("F-414", "shadow reducer output per arm", sh_arms, "r73_shadow_arm.csv", ["frame", "step_pos_driver_mm", "step_pos_shadow_mm", "step_diff_mm"], 50)
    add("F-414", "shadow decision table (S per pair, real divergence, ratio, verdict, determinism and map-digest checks)", [], "r73_shadow_pairs.csv", ["pair", "S_mm", "D_real_mm", "ratio", "verdict"], 4, source="generated")
    add("F-414", "engagement of the shadow (counters, digest-changed count)", sh_arms, "engagement_imu_shadow.txt", [], 1, where="archive")
    # F-415 / F-410 coupled
    co_arms = arm_ids({"CO"}, "r73_co") + []
    co_upd = [a for a in co_arms if "control" not in a]
    add("F-415", "coupled update log per iteration (step norms per block with units, A/Gamma_L conditioning)", co_upd, "coupled_update_iter_all.csv.gz", ["frame", "iter", "imu_prior_info_scale", "A_cond"], 100)
    add("F-415", "coupled update log per scan (tail and per-knot prior/posterior traces)", co_upd, "coupled_update_scan_all.csv.gz", ["frame", "stop", "iterations"], 100)
    add("F-415", "coupled update reducer output (per arm)", co_upd, "r73_coupled_update_arm.csv", ["frame", "prior_share_pos", "lidar_share_pos"], 100)
    add("F-415", "coupled raw first-frame matrices (record frames 1-5) of the 60 s arms", arm_ids({"CO"}, "r73_co60_"), "coupled_record_f1.txt", [], 1, where="archive")
    # F-400 stage 2
    add("F-400", "per-scan posterior position and yaw of every Stage-2 arm (alignment-free collapse timing needs it)", arm_ids({"P2", "P2C"}), "pos_trace_all.csv.gz", ["scan_id", "t_abs", "px", "py", "pz", "yaw_rad"], 500)
    add("F-400", "Stage-2 cell table, per-arm table and selection (collapsed arms excluded)", [], "r73_p2_cells.csv", ["cell", "n_bags"], 10, source="generated")
    # F-401 / F-418
    add("F-418", "the 18 qa arms: each arm's own row in phase_summary.csv and the per-arm file below", arm_ids({"QA"}), "pos_trace_all.csv.gz", ["scan_id", "t_abs", "px", "py", "pz", "yaw_rad"], 500)
    # F-403 / F-406: the step-cap file of the stage-1 R72 arms is retrieved in step 0
    add("F-403", "step_cap.csv of r72_sc_fq_60s, r72_sc_fq_full, r72_sc_mq_full (R72 archive, step 0)", ["r72_sc_fq_60s", "r72_sc_fq_full", "r72_sc_mq_full"], "step_cap.csv.gz", ["frame", "iter", "rot_raw", "pos_raw", "scale", "bound"], 100, source="step0")
    add("F-403", "step_cap.csv of the R73 step-cap smoke arm", ["r73_smoke_stepcap_fq_60s"], "step_cap.csv", [], 1, where="archive")
    # F-416 / F-417 step 0
    ef = ["r72_ef_n%d_ds%s_pfn%d_%s" % (n, ds, p, q) for n in (200, 400, 800, 1600, 3850) for ds in ("first", "off") for p in (1, 3) for q in ("fq", "mq")]
    add("F-416", "per-frame unaligned position and yaw error and excursion for the 40 EF arms (state_trace + reference, archive)", ef, "r73_f416_frame_errors.csv", ["frame", "ep_norm_mm", "ep_x_mm", "yaw_err_mrad", "excursion_10_mm"], 8, source="step0_table")
    add("F-417", "HtH / Htz per frame and iteration (6x6, eigenvalues, condition, yaw/x/y information with and without walls)", ef, "r73_f417_hth_frames.csv",
        ["frame", "iter", "hth_trace", "eig_min", "cond", "info_yaw", "info_x", "info_y", "info_yaw_nowall", "htz_norm"], 8, source="step0_table")
    add("F-417", "residual-class table per frame and iteration (ground / wall / slanted: counts, weights, information shares, Htz shares)", ef, "r73_f417_classes.csv",
        ["frame", "iter", "cls", "n", "weight_sum", "weight_mean", "hth_trace_share", "info_yaw", "htz_norm"], 8, source="step0_table")
    add("F-417", "per-residual table, frames 1-5, with weight, class, plane/node/tier, h rows, per-residual information and Htz contributions", ef, "r73_f417_residuals.csv.gz",
        ["frame", "iter", "idx", "r", "sigma_squared", "weight", "cls", "plane_key", "node_id", "tier", "info_trace", "info_yaw", "htz_0"], 50, source="step0_table")
    add("F-417", "consistency: per-residual sums vs the logged lio_update_diag HtH / Htz", ef, "r73_f417_consistency.csv", ["frame", "iter", "max_rel_err_hth", "max_rel_err_htz"], 1, source="step0_table")
    raw4 = ["r72_ef_n1600_ds%s_pfn%d_%s" % (ds, p, q) for ds in ("off",) for p in (1, 3) for q in ("fq", "mq")]
    add("F-417", "RAW frame-1 files of the four named EF arms: residual_dump, lio_update_diag, pose_control_first_frame_solve_matrices (byte-split return)", raw4, "residual_dump.csv.gz", [], 1, source="step0_raw")
    add("F-417", "RAW pose_control_first_frame_solve_matrices of the same four arms", raw4, "pose_control_first_frame_solve_matrices.txt.gz", [], 1, source="step0_raw")
    add("F-417", "residual report (HtH/Htz metrics) of the R73 sh arms' dumped frames", sh_arms, "r73_f417_hth_frames.csv", ["frame", "iter", "hth_trace", "eig_min", "cond", "info_yaw"], 5)
    # F-398 carry: coupled first 60 s of the R72 full-bag arms
    add("F-415", "R72 coupled full-bag arms: raw coupled_record_f1..5 and the first 600 scans of coupled_scan_status / joint_knot_all_scans (R72 archive, step 0)",
        ["r72_co_tail_gateon_knots4_fq", "r72_co_tail_gateon_knots4_mq", "r72_co_tail_gateon_knots8_fq", "r72_co_tail_gateon_knots8_mq", "r72_co_meas_gateon_knots4_fq",
         "r72_co_meas_gateon_knots4_mq", "r72_co_meas_gateon_knots8_fq", "r72_co_meas_gateon_knots8_mq"], "coupled_record_f1.txt.gz", [], 1, source="step0_raw")
    add("F-417", "RAW lio_update_diag of the same four arms (every frame, every iteration)", raw4, "lio_update_diag.csv.gz", [], 1, source="step0_raw")
    add("F-409", "engagement of every dispatched arm (engagement.txt, engagement_imu.txt, engagement_imu_shadow.txt) concatenated; every shadow arm's block must carry the shadow counters", [], "engagement_all.txt", [], 50, source="generated",
        contains={a: ["scans_where_the_map_digest_changed"] for a in sh_arms})
    add("F-409", "per-arm completeness: required files present (the arm_table 'requires' column)", [], "completeness.csv", ["arm", "requires_missing"], 100, source="generated")
    # F-419 identity
    add("F-419", "identity references: R73 baseline digests and the metric comparison at 0.005 mm", [], "identity_controls.txt", [], 1, source="generated")
    add("F-419", "symmetry gate output (grep gate on covMut assignments)", [], "r73_symmetrize_gate.txt", [], 1, source="agent")
    # F-407
    add("F-407", "engagement_imu.txt present for EVERY dispatched arm", [t["arm"] for t in table if t["group"] not in ("REF", "PROOF")], "engagement_imu.txt", [], 1, where="archive")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", default="r73_manifest")
    ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--min-free-gb", type=float, default=150.0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    J = {j["job_id"]: j for j in json.load(open(a.jobs))}
    os.makedirs(a.out, exist_ok=True)
    BATCHES = ["smoke", "refuse", "sh", "co", "qa", "p2"]
    out = {b: [] for b in BATCHES}; table = []; seen = set()
    for r in arms(J):
        if r["base"] not in J:
            sys.exit("base job %s not in r66_jobs.json" % r["base"])
        assert r["id"] not in seen, r["id"]; seen.add(r["id"])
        j = copy.deepcopy(J[r["base"]]); j["job_id"] = r["id"]
        o = j["overrides"]
        for k, v in r["delta"].items():
            if v is None and k == "voxel_map.plane.plane_fit_pose_cov_mode":
                o.pop(k, None)   # override-only key: gen_jobs.py's null-deletion strips only keys the base config sets (R68 agent fix, merged)
            else:
                o[k] = v
        if o.get("estimator.mode") == "decoupled":
            o["estimator.mode"] = "iekf"
        if "calib.bootstrap.point_cov" in o and r["batch"] != "refuse":
            sys.exit("point_cov key forbidden outside the refuse batch")
        o.pop("spline.mode", None)
        for k in o:
            if "perturb" in k.lower():
                sys.exit("perturbation key forbidden: " + k)
            if k.startswith("lio.q_adapt.") or k.startswith("imu.process_noise.continuous_prior."):
                if not r["id"].startswith("r73_refuse_old_"):
                    sys.exit("retired key %s in arm %s (F-404)" % (k, r["id"]))
        o["eval.state_trace_run_id"] = r["id"]
        path = "%s/r73_%s/%s" % (a.results_root, r["batch"], r["id"])
        o["outputs.path"] = path; o["outputs.debug_log_dir"] = path
        if r["dur"]:
            j["duration_secs"] = r["dur"]
        out[r["batch"]].append(j)
        sampling = "pfn%s_ds%s" % (o.get("cbk.lidar.point_filter_num", "def"), o.get("imu.ds.mode", "def"))
        table.append(dict(arm=r["id"], tier=r["tier"], arch="iekf_raw", kind="closed", duration_s=str(j.get("duration_secs", "full")), seq=j["seq"],
                          notes=r["purpose"], keep_raw=str(r["keep"]), dup_of="", r66_id=r["base"], group=r["group"], batch="r73_" + r["batch"],
                          sampling=sampling, requires=" ".join(requires_of(o, r["batch"]))))
    # R69B stage-1b: the broken-clone proof job is a COPY of a finished smoke job (tools/r69/make_broken_clone.py); it needs an arm-table row so that the
    # scorer fails only because its state_reference.txt was deleted, not because the arm is unknown (R69 stage-1 defect F-324(6)).
    table.append(dict(arm="r73_00_id_broken_clone", tier="PROOF", arch="iekf_raw", kind="closed", duration_s="60", seq=next(t["seq"] for t in table if t["arm"] == "r73_00_id_baseline_60s"),
                      notes="listener failure-path proof: copy of r73_00_id_baseline_60s with state_reference.txt deleted (never dispatched)", keep_raw="0", dup_of="",
                      r66_id="00_id_baseline_60s", group="PROOF", batch="r73_proof", sampling=next(t["sampling"] for t in table if t["arm"] == "r73_00_id_baseline_60s"), requires=""))
    for b, lst in out.items():
        json.dump(lst, open(os.path.join(a.out, "jobs_%s.json" % b), "w"), indent=1)
    cols = ["arm", "tier", "arch", "kind", "duration_s", "seq", "notes", "keep_raw", "dup_of", "r66_id", "group", "batch", "sampling", "requires"]
    with open(os.path.join(a.out, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t"); w.writeheader(); w.writerows(table)
    q = lambda b, n: {"dir": "ablations/_job_queue/r73_%s" % b, "results_root": "%s/r73_%s" % (a.results_root, b), "expected_jobs": n}
    specs = {"smoke": ("R73-smoke", ["smoke", "refuse"]), "sh": ("R73-sh", ["sh"]), "co": ("R73-co", ["co"]), "qa": ("R73-qa", ["qa"]), "p2": ("R73-p2", ["p2"])}
    spec_objs = {nm: listener_spec(rn, [q(b, len(out[b])) for b in bs], a.results_root, a.min_free_gb, a.workers) for nm, (rn, bs) in specs.items()}
    spec_objs["broken"] = listener_spec("R73-broken", [{"dir": "ablations/_job_queue/r73_smoke_broken", "results_root": "%s/r73_smoke_broken" % a.results_root, "expected_jobs": 1}],
                                        a.results_root, a.min_free_gb, 1)
    for nm, sp in spec_objs.items():
        fn = "listener_spec_%s.json" % nm
        for sg in sp["stages"]:
            sg["cmd"] = [c.replace("%s", fn) for c in sg["cmd"]]
        for vf in sp["verify"]:
            vf["cmd"] = [c.replace("%s", fn) for c in vf["cmd"]]
        json.dump(sp, open(os.path.join(a.out, fn), "w"), indent=1)
    pairs = ",".join("%s:%s" % (t["arm"], "r73_co_control_iekf_mq_60s" if t["arm"].endswith("_mq") else "r73_co_control_iekf_fq_60s") for t in table if t["arm"].startswith("r73_co60_"))
    open(os.path.join(a.out, "coupled_pairs.txt"), "w").write(pairs + "\n")      # build_return.py: coupled arm : iekf control arm of the same Q (first-frame comparison only)
    # shadow pairs for r73_shadow_reduce.py: shadow arm : the two plain controls (the arm with the driver's config and the arm with the shadow's config), window
    sp = [("r73_sh_sbs03_a", "r73_sh_sbs03_ctl_fq", "r73_sh_sbs03_ctl_g5a1", SBS_WIN), ("r73_sh_sbs03_b", "r73_sh_sbs03_ctl_g5a1", "r73_sh_sbs03_ctl_fq", SBS_WIN),
          ("r73_sh_eee02_d", "r73_sh_eee02_ctl_fq", "r73_sh_eee02_ctl_mq", EEE_WIN), ("r73_sh_eee02_e", "r73_sh_eee02_ctl_mq", "r73_sh_eee02_ctl_fq", EEE_WIN)]
    json.dump([dict(pair=p[0], shadow_arm=p[0], driver_control=p[1], shadow_config_control=p[2], window=list(p[3])) for p in sp] +
              [dict(pair="r73_sh_sbs03_a_rep", shadow_arm="r73_sh_sbs03_a_rep", repeat_of="r73_sh_sbs03_a", window=list(SBS_WIN))], open(os.path.join(a.out, "shadow_pairs.json"), "w"), indent=1)
    json.dump(required_outputs(table), open(os.path.join(a.out, "required_outputs.json"), "w"), indent=1)
    with open(os.path.join(a.out, "scan_arms_priority.txt"), "w") as f:       # return-thinning order: identity, analysis groups first
        order = {"ID": 0, "SH": 1, "FZ": 2, "CO": 3, "QA": 4, "P2C": 5, "P2": 6, "SM": 7}
        for t in sorted((t for t in table if t["group"] in order), key=lambda t: order[t["group"]]):
            f.write(t["arm"] + "\n")
    print("  ".join("%s %d" % (b, len(out[b])) for b in BATCHES), " total %d (arm table also has the PROOF row)" % len(table))


main()
