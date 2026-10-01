#!/usr/bin/env python3
"""R72: build jobs_<batch>.json, arm_table.tsv, scan_arms_priority.txt, coupled_pairs.txt and the listener specs from the R66 job list (r66_jobs.json).

    python3 tools/r69/make_r72_manifest.py --jobs r69_inputs/r66_jobs.json --out r72_manifest --results-root /root/catkin_ws/livo_recon_results

Planning agent's script; NOT run against the real host by me (it is run in the selftest on the shipped r66_jobs.json only: counts and key checks).
Every arm is an R66 job (same overrides) plus the delta written below. estimator.mode "decoupled" -> "iekf"; spline.mode removed. No perturbation key
anywhere (rule 3.11). expected_jobs in the specs = jobs written.

Batches (queues)
  smoke    identity gates (R72 code inert at defaults, step cap with huge caps, ZUPT detect_only, nis_gate that cannot fire) and one 60 s smoke per new flag (rule 3.6)
  refuse   jobs that MUST fail at startup (new dependency refusals)
  ef       F-396 early-frame grid: eee_01, calib samples x ds mode x pfn x Q mode = 40 short arms (duration = calibration time + 8 s)
  co       F-398: 8 coupled arms (full bag, failure_dump on, record_frames 5) + iekf first-frame control arms
  sc       F-403: step cap on the fixed-Q 60 s baseline and motion-Q (inert check)
  qa       F-401: motion-Q alternatives on eee_01 (PROVISIONAL parameters; dispatch only after Bryce confirms, see R72_INSTRUCTIONS section 5)
  p        F-400 Stage 1: 16 arms x 9 bags = 144 full-length arms
"""
import argparse, collections, copy, csv, json, math, os, sys

BAGS = ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]
DIAG = {"lio.update_diag_en": True, "lio.live_point_cov_en": True, "lio.first_frame_dump_frames": 5, "lio.residual_geometry_en": True}
MQ = {"imu.process_noise.model": "isotropic", "imu.process_noise.motion.beta": 0.9, "imu.process_noise.motion.acc_scale": 0.3, "imu.process_noise.motion.gyro_scale": 10.0,
      "imu.process_noise.motion.acc_max_dynamic_variance": 1000000.0, "imu.process_noise.motion.gyro_max_dynamic_variance": 1000000.0}
P0_CONFIGURED = {"calib.p0.mode": "configured", "calib.p0.known_pos_variance": None, "calib.p0.known_vel_variance": None, "calib.p0.tilt_ba_ambiguity_accel_std": None,
                 "calib.p0.bg_model_floor": None, "calib.p0.ba_radial_model_floor": None, "state.cov.rot_yaw": None}   # = 55_F_configuredP0_motionq_control_full
TILT_STD = 0.03     # calib.p0.tilt_ba_ambiguity_accel_std at the base (variance factor f -> std 0.03 sqrt(f); matches 30_B8_p0_tilt_ambiguity_var_x0p1/x10)
CO_FULL = {"estimator.coupled.minimal_diagnostics": False}
CO_REC = {"estimator.coupled.failure_dump": True, "estimator.coupled.record_frames": 5}
STEPCAP = lambda rot=1e-2, pos=1e-2: {"lio.ekf.step_cap.mode": "hard", "lio.ekf.step_cap.rotation_rad": rot, "lio.ekf.step_cap.position_m": pos}
ZUPT = lambda mode, g=0.02, a=0.15, n=3, s=0.02: dict({"lio.zupt.mode": mode, "lio.zupt.gyro_thresh": g, "lio.zupt.acc_thresh": a, "lio.zupt.enter_scans": n},
                                                      **({"lio.zupt.sigma_v": s} if mode == "apply" else {}))
DUMP20 = ",".join(str(i) for i in range(1, 21))


def tilt(f):
    return {"calib.p0.tilt_ba_ambiguity_accel_std": TILT_STD * math.sqrt(f)}


def mqs(acc, gyro):
    return {"imu.process_noise.motion.acc_scale": float(acc), "imu.process_noise.motion.gyro_scale": float(gyro)}


def base_of(bag, q):
    """R66 base job: eee_01 identity jobs, other bags the R67 50_E jobs. q in fq|mq."""
    if bag == "eee_01":
        return "00_id_baseline_full" if q == "fq" else "00_id_motionq_full"
    return "50_E_%s_%s_full" % (bag, "baseline" if q == "fq" else "motionq")


def arms(J):
    A = []

    def add(id_, base, delta=None, batch="main", group="", tier="", purpose="", keep=0, dur=None):
        A.append(dict(id=id_, base=base, delta=delta or {}, batch=batch, group=group, tier=tier, purpose=purpose, keep=keep, dur=dur))

    def d(*ds):
        o = {}
        for x in ds:
            o.update(x)
        return o

    # ---- smoke: identity gates (the R72 code must be inert at defaults) and one smoke per new flag
    for n in ("00_id_baseline_60s", "00_id_baseline_full", "00_id_motionq_full"):
        add("r72_" + n, n, batch="smoke", group="ID", tier="ID", keep=1, purpose="identity: digest must equal the R71 digest of the same job (R72 code inert at defaults)")
    add("r72_id_stepcap_wide_60s", "00_id_baseline_60s", STEPCAP(1e6, 1e6), batch="smoke", group="ID", tier="ID", keep=1,
        purpose="IDENTITY: step cap with caps 1e6 never binds, so the digest must equal r72_00_id_baseline_60s; engagement must say NOT ENGAGED (bound=0) and iterations > 0 (the cap code ran)")
    add("r72_id_zupt_detect_60s", "00_id_baseline_60s", ZUPT("detect_only"), batch="smoke", group="ID", tier="ID", keep=1,
        purpose="IDENTITY: ZUPT detect_only runs detector and zupt.csv but must not touch the state: digest equals r72_00_id_baseline_60s; zupt.csv rows > 0 and stationary > 0 (eee_01 rests for 48 s)")
    add("r72_id_nisgate_dead_60s", "00_id_baseline_60s", {"lio.q_adapt.mode": "nis_gate", "lio.q_adapt.nis_hi": 1e9}, batch="smoke", group="ID", tier="ID", keep=1,
        purpose="IDENTITY: nis_gate with nis_hi 1e9 cannot inflate, gain stays exactly 1: digest equals r72_00_id_baseline_60s; q_adapt.csv rows > 0, engagement NOT ENGAGED")
    add("r72_id_diag_on_60s", "00_id_baseline_60s", DIAG, batch="smoke", group="ID", tier="ID", keep=1, purpose="write-only logs on: digest equals r72_00_id_baseline_60s")
    add("r72_smoke_stepcap_fq_60s", "00_id_baseline_60s", d(DIAG, STEPCAP()), batch="smoke", group="SM", tier="SMOKE", keep=1,
        purpose="smoke of the step cap, fixed-Q: step_cap.csv written, bound > 0 (frame-3 rotation steps are 25-54 mrad against a 10 mrad cap)")
    add("r72_smoke_zupt_apply_60s", "00_id_motionq_full", d(DIAG, ZUPT("apply")), batch="smoke", group="SM", tier="SMOKE", dur=60, keep=1,
        purpose="smoke of ZUPT apply under motion-Q: applied > 0, velocity norm before update recorded")
    add("r72_smoke_innov_scale_60s", "00_id_motionq_full", d(DIAG, {"lio.q_adapt.mode": "innovation_scale"}), batch="smoke", group="SM", tier="SMOKE", dur=60, keep=1,
        purpose="smoke of innovation_scale (defaults forgetting 0.9, bounds 1..100): q_adapt.csv written; engagement_imu.txt q_gain steps_with_gain_ne_1 must agree with the iekf engagement")
    add("r72_smoke_nis_gate_60s", "00_id_motionq_full", d(DIAG, {"lio.q_adapt.mode": "nis_gate"}), batch="smoke", group="SM", tier="SMOKE", dur=60, keep=1,
        purpose="smoke of nis_gate (nis_hi 1.5, inflate 1.25, decay 0.95)")
    add("r72_smoke_local_poly_60s", "00_id_motionq_full", {"imu.process_noise.motion.energy_source": "local_poly"}, batch="smoke", group="SM", tier="SMOKE", dur=60, keep=1,
        purpose="smoke of the local-polynomial energy source under motion-Q: engagement_imu.txt scans_used > 0")
    add("r72_smoke_wnoa_60s", "00_id_motionq_full", {"imu.process_noise.continuous_prior.mode": "wnoa", "imu.process_noise.continuous_prior.qc_acc": 0.01,
                                                      "imu.process_noise.continuous_prior.qc_gyr": 0.0001}, batch="smoke", group="SM", tier="SMOKE", dur=60, keep=1,
        purpose="smoke of the WNOA disturbance (qc_acc 0.01 m2/s5, qc_gyr 1e-4 rad2/s3 PROVISIONAL): engagement_imu.txt wnoa steps > 0")
    for t in ("tail", "measurement"):
        add("r72_smoke_co_%s_rec_60s" % t, "10_smoke_coupled_%s_fulldiag_60s" % t, d(CO_FULL, CO_REC), batch="smoke", group="SM", tier="SMOKE", dur=60, keep=1,
            purpose="smoke of the coupled record: coupled_record_f1..f5.txt written, A_positive_definite_pass present for every iteration; no failure_dump file (no failure)")
    # ---- refuse (every one MUST die at startup)
    R = lambda id_, base, delta, purpose: add(id_, base, delta, batch="refuse", group="REF", tier="REF", dur=5, purpose=purpose)
    R("r72_refuse_stepcap_suboption", "00_id_baseline_60s", {"lio.ekf.step_cap.rotation_rad": 0.01}, "MUST FAIL: step_cap/rotation_rad outside mode=hard")
    R("r72_refuse_stepcap_coupled", "10_smoke_coupled_tail_fulldiag_60s", STEPCAP(), "MUST FAIL: step cap under estimator/mode=coupled")
    R("r72_refuse_stepcap_zero", "00_id_baseline_60s", STEPCAP(0.0, 0.01), "MUST FAIL: step cap with a zero cap")
    R("r72_refuse_qadapt_suboption", "00_id_baseline_60s", {"lio.q_adapt.forgetting": 0.5}, "MUST FAIL: q_adapt/forgetting outside mode=innovation_scale")
    R("r72_refuse_qadapt_inert_forgetting", "00_id_motionq_full", {"lio.q_adapt.mode": "innovation_scale", "lio.q_adapt.forgetting": 1.0}, "MUST FAIL: forgetting 1 is inert")
    R("r72_refuse_qadapt_inert_bounds", "00_id_motionq_full", {"lio.q_adapt.mode": "nis_gate", "lio.q_adapt.min_gain": 1.0, "lio.q_adapt.max_gain": 1.0}, "MUST FAIL: gain bounds 1..1 are inert")
    R("r72_refuse_qadapt_coupled", "10_smoke_coupled_tail_fulldiag_60s", {"lio.q_adapt.mode": "nis_gate"}, "MUST FAIL: q_adapt keys under coupled are unclaimed")
    R("r72_refuse_zupt_suboption", "00_id_baseline_60s", {"lio.zupt.sigma_v": 0.02}, "MUST FAIL: zupt/sigma_v outside mode=apply")
    R("r72_refuse_zupt_threshold", "00_id_baseline_60s", ZUPT("detect_only", g=-1.0), "MUST FAIL: negative gyro threshold")
    R("r72_refuse_local_poly_fixed", "00_id_baseline_60s", {"imu.process_noise.motion.energy_source": "local_poly"}, "MUST FAIL: local_poly under the fixed model")
    R("r72_refuse_local_poly_order_instant", "00_id_motionq_full", {"imu.process_noise.motion.local_poly.order": 2}, "MUST FAIL: local_poly/order under energy_source=instant")
    R("r72_refuse_wnoa_zero", "00_id_baseline_60s", {"imu.process_noise.continuous_prior.mode": "wnoa", "imu.process_noise.continuous_prior.qc_acc": 0.0,
                                                      "imu.process_noise.continuous_prior.qc_gyr": 0.0}, "MUST FAIL: wnoa with both densities 0 is inert")
    R("r72_refuse_wnoa_suboption", "00_id_baseline_60s", {"imu.process_noise.continuous_prior.qc_acc": 0.01}, "MUST FAIL: qc_acc outside mode=wnoa")
    # ---- ef: F-396 (eee_01 only)
    for ns in (200, 400, 800, 1600, 3850):
        for ds in ("first", "off"):
            for pfn in (1, 3):
                for q in ("fq", "mq"):
                    dl = {"calib.stationary.num_samples": ns, "imu.ds.mode": ds, "cbk.lidar.point_filter_num": pfn,
                          "lio.residual_dump_frames": DUMP20, "voxel_map.plane.plane_fit_trace_frames": 20}
                    if ds == "off":
                        dl["imu.ds.ds_leaf_size"] = None
                    dur = int(math.ceil(ns / 197.5 + 8.0))
                    add("r72_ef_n%d_ds%s_pfn%d_%s" % (ns, ds, pfn, q), base_of("eee_01", q), d(DIAG, dl), batch="ef", group="EF", tier="EF", dur=dur, keep=0,
                        purpose="F-396: calibration samples %d, ds %s, pfn %d, %s, %d s (calibration time + 8 s), residual dump frames 1-20, plane_fit_trace on" % (ns, ds, pfn, q, dur))
    # ---- co: F-398
    for t, tn in (("tail", "tail"), ("measurement", "meas")):
        for k in (4, 8):
            for q, qn in (("fixedq", "fq"), ("motionq", "mq")):
                n = "60_D3_coupled_%s_gateon_knots%d_iters5_%s_full" % (t, k, q)
                if n not in J:
                    sys.exit("base job %s not in r66_jobs.json" % n)
                add("r72_co_%s_gateon_knots%d_%s" % (tn, k, qn), n, d(CO_FULL, CO_REC), batch="co", group="CO", tier="CO", keep=1,
                    purpose="F-398: coupled %s, gate on, %d knots, %s, full bag, failure_dump on, record_frames 5 (R71 twin: r71_co_%s_gateon_knots%d_%s; digest and abort scan must reproduce)" % (t, k, qn, tn, k, qn))
    for q in ("fq", "mq"):
        add("r72_co_control_iekf_%s_60s" % q, base_of("eee_01", q), d(DIAG), batch="co", group="CO", tier="CO", dur=60, keep=1,
            purpose="F-398 control: splineless iekf, same Q, first-frame matrices frames 1-5 (lio_update_diag, first-frame dump); common blocks only are comparable with the coupled records")
    # ---- sc: F-403
    add("r72_sc_fq_60s", "00_id_baseline_60s", d(DIAG, STEPCAP(), {"lio.residual_dump_frames": "1,2,3,4,5,10"}), batch="sc", group="SC", tier="SC", keep=1,
        purpose="F-403: step cap rot 1e-2 rad / pos 1e-2 m on the fixed-Q baseline, 60 s (reference 31.002/82.004/65.332 mm full bag; here the early frames)")
    add("r72_sc_fq_full", "00_id_baseline_full", d(DIAG, STEPCAP()), batch="sc", group="SC", tier="SC", keep=1, purpose="F-403: step cap on the fixed-Q baseline, full bag: ATE, excursion, settled vs 31.002/82.004/65.332")
    add("r72_sc_mq_full", "00_id_motionq_full", d(DIAG, STEPCAP()), batch="sc", group="SC", tier="SC", keep=1,
        purpose="F-403: step cap under motion-Q, full bag: expected inert or near-inert (bound count reported); reference 25.819/3.782/2.496")
    # ---- qa: F-401 (PROVISIONAL parameters, Bryce to confirm before dispatch)
    MQB = "00_id_motionq_full"
    QA = lambda id_, base, delta, p: add(id_, base, d(DIAG, delta), batch="qa", group="QA", tier="QA", keep=1, purpose="F-401 (PROVISIONAL parameters): " + p)
    for f_ in (0.8, 0.95):
        QA("r72_qa_innov_f%s_mq" % str(f_).replace(".", "p"), MQB, {"lio.q_adapt.mode": "innovation_scale", "lio.q_adapt.forgetting": f_}, "innovation_scale on motion-Q, forgetting %s" % f_)
    QA("r72_qa_innov_fq", "00_id_baseline_full", {"lio.q_adapt.mode": "innovation_scale"}, "innovation_scale on fixed-Q (does it recover the motion-Q behaviour)")
    for hi in (1.3, 1.6):
        QA("r72_qa_nis_hi%s_mq" % str(hi).replace(".", "p"), MQB, {"lio.q_adapt.mode": "nis_gate", "lio.q_adapt.nis_hi": hi}, "nis_gate on motion-Q, nis_hi %s" % hi)
    QA("r72_qa_nis_white_mq", MQB, {"lio.q_adapt.mode": "nis_gate", "lio.q_adapt.whiteness_max": 0.3}, "nis_gate with whiteness gate 0.3 on motion-Q")
    QA("r72_qa_nis_fq", "00_id_baseline_full", {"lio.q_adapt.mode": "nis_gate"}, "nis_gate on fixed-Q")
    for o_ in (1, 2):
        QA("r72_qa_localpoly_o%d_mq" % o_, MQB, {"imu.process_noise.motion.energy_source": "local_poly", "imu.process_noise.motion.local_poly.order": o_},
           "local polynomial energy source order %d (replacement of the instantaneous energy; same beta/scale/caps)" % o_)
    for qa_, qg_ in ((0.001, 1e-5), (0.01, 1e-4), (0.1, 1e-3)):
        QA("r72_qa_wnoa_a%s_fq" % str(qa_).replace(".", "p"), "00_id_baseline_full", {"imu.process_noise.continuous_prior.mode": "wnoa", "imu.process_noise.continuous_prior.qc_acc": qa_,
           "imu.process_noise.continuous_prior.qc_gyr": qg_}, "WNOA disturbance added to fixed-Q: qc_acc %s qc_gyr %s" % (qa_, qg_))
    QA("r72_qa_wnoa_mq", MQB, {"imu.process_noise.continuous_prior.mode": "wnoa", "imu.process_noise.continuous_prior.qc_acc": 0.01, "imu.process_noise.continuous_prior.qc_gyr": 1e-4},
       "WNOA added to motion-Q (addition rather than replacement)")
    for tag, g_, a_, s_ in (("loose", 0.05, 0.3, 0.05), ("mid", 0.02, 0.15, 0.02), ("tight", 0.01, 0.08, 0.01)):
        QA("r72_qa_zupt_%s_mq" % tag, MQB, ZUPT("apply", g_, a_, 3, s_), "ZUPT apply on motion-Q, thresholds gyro %s rad/s acc %s m/s2, sigma_v %s m/s" % (g_, a_, s_))
    QA("r72_qa_zupt_mid_fq", "00_id_baseline_full", ZUPT("apply"), "ZUPT apply on fixed-Q, mid thresholds")
    QA("r72_qa_zupt_detect_mq", MQB, ZUPT("detect_only"), "ZUPT detect_only on motion-Q full bag: the detector statistics that fix the thresholds (control, byte-identical to motion-Q)")
    # ---- p: F-400 Stage 1, 9 bags x 16 arms
    for bag in BAGS:
        cells = []
        cells.append(("fq_p0d", "fq", {}))
        cells.append(("mq_p0d", "mq", {}))
        cells.append(("fq_p0c", "fq", dict(P0_CONFIGURED)))
        cells.append(("mq_p0c", "mq", dict(P0_CONFIGURED)))
        for acc in (0, 0.3, 1):
            for gyro in (5, 10, 14):
                if (acc, gyro) != (0.3, 10):
                    cells.append(("mq_a%s_g%s" % (str(acc).replace(".", "p"), gyro), "mq", mqs(acc, gyro)))
        for q in ("fq", "mq"):
            for f_ in (0.1, 10):
                cells.append(("%s_tilt%s" % (q, str(f_).replace(".", "p")), q, tilt(f_)))
        assert len(cells) == 16, len(cells)
        for cn, q, dl in cells:
            add("r72_p_%s_%s" % (bag, cn), base_of(bag, q), d(dl), batch="p", group="P", tier="P", keep=0,
                purpose="F-400 Stage 1, %s, cell %s (P0 derived/configured x Q fixed/motion, Q ladder acc {0,0.3,1} x gyro {5,10,14} at P0 derived, tilt/ba variance x0.1 and x10)" % (bag, cn))
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
            # tools/r69, r72_manifest, r69_inputs actually live), but sweep_listener.py's
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
            "cmd": ["python3", "src/livo_recon/tools/r69/r72_score_job.py", "--job-dir", "{job_dir}", "--out-dir", "{out_dir}", "--stage-out", "{stage_out}",
                    "--job-id", "{job_id}", "--spec", "{spec_dir}/%s", "--arm-table", "src/livo_recon/r72_manifest/arm_table.tsv",
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
                           "step_cap.csv", "q_adapt.csv", "zupt.csv", "coupled_failure_dump.txt", "engagement_imu.txt"] + ["coupled_record_f%d.txt" % i for i in range(1, 6)],   # R72: all are in the lossless archive (BIG list of r69_score_job.py)
                "sha256": "twice"},
        "limits": {"workers": workers, "max_failures": 10, "max_attempts": 2, "min_free_gb": min_free_gb, "poll_interval_s": 10},
        "exit": {"when": "all_expected_handled"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", default="r72_manifest")
    ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--min-free-gb", type=float, default=150.0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    J = {j["job_id"]: j for j in json.load(open(a.jobs))}
    os.makedirs(a.out, exist_ok=True)
    BATCHES = ["smoke", "refuse", "ef", "co", "sc", "qa", "p"]
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
        if o.get("estimator.mode") == "decoupled" and r["id"] != "r72_refuse_estimator_decoupled":
            o["estimator.mode"] = "iekf"
        if "calib.bootstrap.point_cov" in o and r["batch"] != "refuse":
            sys.exit("point_cov key forbidden outside the refuse batch")
        o.pop("spline.mode", None)
        if r["id"] == "r72_refuse_spline_mode":
            o["spline.mode"] = "raw_imu"
        if r["id"] == "r72_refuse_estimator_decoupled":
            o["estimator.mode"] = "decoupled"
        for k in o:
            if "perturb" in k.lower():
                sys.exit("perturbation key forbidden: " + k)
        o["eval.state_trace_run_id"] = r["id"]
        path = "%s/r72_%s/%s" % (a.results_root, r["batch"], r["id"])
        o["outputs.path"] = path; o["outputs.debug_log_dir"] = path
        if r["dur"]:
            j["duration_secs"] = r["dur"]
        out[r["batch"]].append(j)
        sampling = "pfn%s_ds%s" % (o.get("cbk.lidar.point_filter_num", "def"), o.get("imu.ds.mode", "def"))
        table.append(dict(arm=r["id"], tier=r["tier"], arch="iekf_raw", kind=("open" if r["group"] == "OL" else "closed"), duration_s=str(j.get("duration_secs", "full")), seq=j["seq"],
                          notes=r["purpose"], keep_raw=str(r["keep"]), dup_of="", r66_id=r["base"], group=r["group"], batch="r72_" + r["batch"],
                          sampling=sampling))
    # R69B stage-1b: the broken-clone proof job is a COPY of a finished smoke job (tools/r69/make_broken_clone.py); it needs an arm-table row so that the
    # scorer fails only because its state_reference.txt was deleted, not because the arm is unknown (R69 stage-1 defect F-324(6)).
    table.append(dict(arm="r72_00_id_broken_clone", tier="PROOF", arch="iekf_raw", kind="closed", duration_s="60", seq=next(t["seq"] for t in table if t["arm"] == "r72_00_id_baseline_60s"),
                      notes="listener failure-path proof: copy of r72_00_id_baseline_60s with state_reference.txt deleted (never dispatched)", keep_raw="0", dup_of="",
                      r66_id="00_id_baseline_60s", group="PROOF", batch="r72_proof", sampling=next(t["sampling"] for t in table if t["arm"] == "r72_00_id_baseline_60s")))
    for b, lst in out.items():
        json.dump(lst, open(os.path.join(a.out, "jobs_%s.json" % b), "w"), indent=1)
    cols = ["arm", "tier", "arch", "kind", "duration_s", "seq", "notes", "keep_raw", "dup_of", "r66_id", "group", "batch", "sampling"]
    with open(os.path.join(a.out, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t"); w.writeheader(); w.writerows(table)
    q = lambda b, n: {"dir": "ablations/_job_queue/r72_%s" % b, "results_root": "%s/r72_%s" % (a.results_root, b), "expected_jobs": n}
    specs = {"smoke": ("R72-smoke", ["smoke", "refuse"]), "ef": ("R72-ef", ["ef"]), "co": ("R72-co", ["co"]), "sc": ("R72-sc", ["sc"]),
             "qa": ("R72-qa", ["qa"]), "p": ("R72-p", ["p"])}
    spec_objs = {nm: listener_spec(rn, [q(b, len(out[b])) for b in bs], a.results_root, a.min_free_gb, a.workers) for nm, (rn, bs) in specs.items()}
    spec_objs["broken"] = listener_spec("R72-broken", [{"dir": "ablations/_job_queue/r72_smoke_broken", "results_root": "%s/r72_smoke_broken" % a.results_root, "expected_jobs": 1}],
                                        a.results_root, a.min_free_gb, 1)
    for nm, sp in spec_objs.items():
        fn = "listener_spec_%s.json" % nm
        for sg in sp["stages"]:
            sg["cmd"] = [c.replace("%s", fn) for c in sg["cmd"]]
        for vf in sp["verify"]:
            vf["cmd"] = [c.replace("%s", fn) for c in vf["cmd"]]
        json.dump(sp, open(os.path.join(a.out, fn), "w"), indent=1)
    pairs = ",".join("%s:%s" % (t["arm"], "r72_co_control_iekf_mq_60s" if t["arm"].endswith("_mq") else "r72_co_control_iekf_fq_60s") for t in table if t["arm"].startswith("r72_co_") and t["batch"] == "r72_co" and "control" not in t["arm"])
    open(os.path.join(a.out, "coupled_pairs.txt"), "w").write(pairs + "\n")      # build_return.py: coupled arm : iekf control arm of the same Q (first-frame comparison only)
    with open(os.path.join(a.out, "scan_arms_priority.txt"), "w") as f:       # return-thinning order: identity, analysis groups first
        order = {"ID": 0, "CO": 1, "EF": 2, "SC": 3, "QA": 4, "P": 5, "SM": 6}
        for t in sorted((t for t in table if t["group"] in order), key=lambda t: order[t["group"]]):
            f.write(t["arm"] + "\n")
    print("  ".join("%s %d" % (b, len(out[b])) for b in BATCHES), " total %d (arm table also has the PROOF row)" % len(table))


main()
