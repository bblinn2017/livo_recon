#!/usr/bin/env python3
"""R70: build jobs_smoke.json, jobs_refuse.json, jobs_main.json, jobs_sf.json, arm_table.tsv and the three listener specs from the R66 job list
(r66_jobs.json, shipped in the package).

    python3 tools/r69/make_r70_manifest.py --jobs r69_inputs/r66_jobs.json --out r70_manifest --results-root /root/catkin_ws/livo_recon_results

Every R70 arm is an R66 job (same overrides) plus the delta written below. Rewrites applied to EVERY job: estimator.mode "decoupled" -> "iekf"
(the old value is refused by the build), spline.mode removed (the key is refused). NO job sets calib.bootstrap.point_cov except the refuse job.
No perturbation key anywhere (rule 3.11). Nothing edits the repo. expected_jobs in the specs = the number of jobs written (job-count landmine).
Every arm of the round is listed with its follow-up(s) in the arm table (`notes`) and in R70_INSTRUCTIONS.md section 5.

Blocks (batch main unless stated)
  smoke   identity gates G1-G5, G8a (new write-only logs are inert), plus one 60 s smoke per new flag (rule 3.6): block information scale, redundancy
          renormalisation, surface 0.425 with the new miss counters, octree 0.25 voxels
  refuse  nine jobs that MUST fail at startup: removed point_cov key, removed spline/mode, renamed estimator mode, surface cell 0.10, octree voxel 0.10,
          block information scale inert / sub-option out of scope / coupled estimator, renorm without Woodbury
  ID      00_id_baseline_full_rerun (determinism, G5)
  MX      the six first-frame matrix arms, 60 s (F-300, F-331: the dumps never reached the R69B return)
  CB      eee_01 combined x global F re-run with the archive kept (F-331; fixed-Q and motion-Q, F = 1, 10, 30)
  BL      block information scale (F-320, F-334, F-336): combined map, rotation x position grid on eee_01 (fixed-Q and motion-Q); sensor-only cells; eee_03 / nya_03
  RD      redundancy ladder (F-317): Woodbury alone, + renormalisation holding the weakest position eigenvalue, + block scale; fixed-Q and motion-Q
  IT      iteration count and early exit (F-168)
  OL      open-loop with P0 for R, p, v near zero (F-73)
  CO      coupled estimator (Phase 4, coupled only; F-83, F-223, F-240): 60 s full-diagnostics smokes in main, full-bag arms in sf (no wall cap)
  SF      (batch sf, dispatched last, no wall cap) surface backend: cell-size ladder at default density, 0.425 default and dense, merge off, motion-Q, octree 0.25
          voxels and dense octree controls (F-340); cell 0.10 is RETIRED (Bryce 2026-10-01) and has no arm
"""
import argparse, collections, copy, csv, json, os, sys

SEQS_OTHER = ["eee_03", "nya_03"]
INFL = lambda f: {"lio.bootstrap_inflate.scans": 100000, "lio.bootstrap_inflate.factor": float(f)}
DIAG = {"lio.update_diag_en": True, "lio.live_point_cov_en": True, "lio.first_frame_dump_frames": 5, "lio.residual_geometry_en": True}
COMB = {"voxel_map.plane.plane_fit_pose_cov_mode": "combined"}
BLK = lambda fr, fp: {"lio.info_scale.mode": "block", "lio.info_scale.rotation": float(fr), "lio.info_scale.position": float(fp)}
WB = lambda rho, renorm: dict({"lio.residual_redundancy.mode": "woodbury", "lio.residual_redundancy.rho": float(rho)},
                              **({"lio.residual_redundancy.renorm": "hold_pos_min_eig"} if renorm else {}))
MQ = {"imu.process_noise.model": "isotropic", "imu.process_noise.motion.beta": 0.9, "imu.process_noise.motion.acc_scale": 0.3, "imu.process_noise.motion.gyro_scale": 10.0,
      "imu.process_noise.motion.acc_max_dynamic_variance": 1000000.0, "imu.process_noise.motion.gyro_max_dynamic_variance": 1000000.0}
P0_RPV_OFF = {"calib.p0.mode": "configured", "state.cov.rot_tilt": 1e-12, "state.cov.rot_yaw": 1e-12, "state.cov.pos": 1e-12, "state.cov.vel": 1e-12,
              "state.cov.bg": 1e-07, "state.cov.ba": 1e-07, "state.cov.gravity": 1e-07}
SRF25 = "70_C5_surface_cell0p25_default_aggregate_independent_full"
SRF_DENSE = "70_C5_surface_cell0p1_dense_aggregate_independent_full"   # base job only (cell size overridden below; 0.10 itself is retired)


def arms(J):
    A = []

    def add(id_, base, delta=None, batch="main", group="", tier="", purpose="", keep=0, dur=None):
        A.append(dict(id=id_, base=base, delta=delta or {}, batch=batch, group=group, tier=tier, purpose=purpose, keep=keep, dur=dur))

    def d(*ds):
        o = {}
        for x in ds:
            o.update(x)
        return o

    # ---- smoke (identity gates; diagnostics OFF except the G8a arm) + one 60 s smoke per new flag
    for n in ("00_id_baseline_48s", "00_id_baseline_60s", "00_id_baseline_full", "00_id_motionq_full"):
        add(n, n, batch="smoke", group="ID", tier="ID", purpose="identity: digest must equal the R67 sensor_model digest (the R70 code is inert at its defaults)", keep=1)
    add("00_id_default_posecov_60s", "00_id_baseline_60s", {"voxel_map.plane.plane_fit_pose_cov_mode": None}, batch="smoke", group="ID", tier="ID",
        purpose="plane_fit_pose_cov_mode key deleted: compiled default sensor_only equals the pinned value; digest equals 00_id_baseline_60s")
    add("00_id_infl_whole_f1_60s", "00_id_baseline_60s", INFL(1.0), batch="smoke", group="ID", tier="ID",
        purpose="whole-run inflation at factor 1.0 is the identity: digest equals 00_id_baseline_60s")
    add("r70_diag_on_baseline_60s", "00_id_baseline_60s", DIAG, batch="smoke", group="ID", tier="ID", keep=1,
        purpose="ALL write-only logs ON (update_diag, live_point_cov, first-frame dump frames 1..5, residual_geometry): digest must equal 00_id_baseline_60s (G8a); also the listener's real-data proof job")
    add("r70_smoke_block_f12_2_60s", "40_C1_pca_eigengap_posecombined_fixedq_full", d(DIAG, BLK(12, 2)), batch="smoke", group="BL", tier="SMOKE", dur=60, keep=1,
        purpose="smoke of lio/info_scale block 12 / 2 (combined map, fixed-Q, 60 s): info_scale.csv written, engagement not INERT (F-320)")
    add("r70_smoke_block_f10_10_60s", "40_C1_pca_eigengap_posecombined_fixedq_full", d(DIAG, BLK(10, 10)), batch="smoke", group="BL", tier="SMOKE", dur=60, keep=1,
        purpose="block scale 10 / 10 equals the global scale F=10 (algebraic identity, G9a): frame-1 HtH/Htz equal those of r70_mx_cb_fq_f10_60s")
    add("r70_smoke_renorm_60s", "00_id_baseline_60s", d(DIAG, WB(0.5, True)), batch="smoke", group="RD", tier="SMOKE", keep=1,
        purpose="smoke of lio/residual_redundancy/renorm hold_pos_min_eig with Woodbury rho 0.5 (60 s): redundancy_renorm.csv written, c > 1 occurs (F-317)")
    add("r70_smoke_surface0p425_60s", SRF25, d(DIAG, {"voxel_map.surface.cell_size": 0.425}), batch="smoke", group="SF", tier="SMOKE", dur=60, keep=1,
        purpose="smoke of the surface miss-reason counters (surface_map_frames.csv new columns) and residual_geometry at cell 0.425, default density (F-340)")
    add("r70_smoke_octree_vox0p25_60s", "00_id_baseline_60s", d(DIAG, {"voxel_map.map.voxel_size": 0.25}), batch="smoke", group="SF", tier="SMOKE", keep=1,
        purpose="smoke of octree voxel_size 0.25 (60 s): the key is accepted and residual_geometry written (F-340 same-scale octree comparator)")
    # ---- refuse (every one MUST die at startup)
    R = lambda id_, base, delta, purpose: add(id_, base, delta, batch="refuse", group="REF", tier="REF", dur=5, purpose=purpose)
    R("r70_refuse_point_cov_zero", "00_id_baseline_60s", {"calib.bootstrap.point_cov": "zero"}, "MUST FAIL at startup: removed key calib/bootstrap/point_cov")
    R("r70_refuse_spline_mode", "00_id_baseline_60s", {}, "MUST FAIL at startup: the removed key spline/mode (even with the value raw_imu)")
    R("r70_refuse_estimator_decoupled", "00_id_baseline_60s", {}, "MUST FAIL at startup: estimator/mode 'decoupled' was renamed 'iekf'")
    R("r70_refuse_surface_cell0p10", SRF25, {"voxel_map.surface.cell_size": 0.10}, "MUST FAIL at startup: surface cell size 0.10 is retired (Bryce 2026-10-01)")
    R("r70_refuse_octree_voxel0p10", "00_id_baseline_60s", {"voxel_map.map.voxel_size": 0.10}, "MUST FAIL at startup: octree voxel size 0.10 is retired")
    R("r70_refuse_info_scale_inert", "00_id_baseline_60s", {"lio.info_scale.mode": "block"}, "MUST FAIL at startup: block mode with rotation == position == 1 is inert")
    R("r70_refuse_info_scale_suboption", "00_id_baseline_60s", {"lio.info_scale.rotation": 5.0}, "MUST FAIL at startup: lio/info_scale/rotation outside mode=block")
    R("r70_refuse_info_scale_coupled", "10_smoke_coupled_tail_fulldiag_60s", BLK(5, 2), "MUST FAIL at startup: block information scale under estimator/mode=coupled")
    R("r70_refuse_renorm_without_woodbury", "00_id_baseline_60s", {"lio.residual_redundancy.renorm": "hold_pos_min_eig"},
      "MUST FAIL at startup: renorm outside residual_redundancy/mode=woodbury")
    # ---- main
    add("00_id_baseline_full_rerun", "00_id_baseline_full", group="ID", tier="ID", keep=1, purpose="determinism control (digest equals 00_id_baseline_full), G5")
    # MX: the six first-frame matrix arms, 60 s (F-331: the matrices never reached the R69B return)
    add("r70_mx_baseline_fq_60s", "00_id_baseline_60s", dict(DIAG), group="MX", tier="MX", keep=1, purpose="first-frame matrices frames 1..5, sensor-only fixed-Q (F-300, F-331)")
    add("r70_mx_baseline_mq_60s", "00_id_motionq_full", dict(DIAG), group="MX", tier="MX", dur=60, keep=1, purpose="first-frame matrices frames 1..5, sensor-only motion-Q")
    add("r70_mx_cb_fq_f1_60s", "40_C1_pca_eigengap_posecombined_fixedq_full", dict(DIAG), group="MX", tier="MX", dur=60, keep=1, purpose="first-frame matrices, combined, fixed-Q, F=1")
    add("r70_mx_cb_fq_f10_60s", "40_C1_pca_eigengap_posecombined_fixedq_full", d(DIAG, INFL(10)), group="MX", tier="MX", dur=60, keep=1, purpose="first-frame matrices, combined, fixed-Q, F=10")
    add("r70_mx_cb_mq_f10_60s", "40_C1_pca_eigengap_posecombined_motionq_full", d(DIAG, INFL(10)), group="MX", tier="MX", dur=60, keep=1, purpose="first-frame matrices, combined, motion-Q, F=10")
    add("r70_mx_sensor_fq_f10_60s", "00_id_baseline_60s", d(DIAG, INFL(10)), group="MX", tier="MX", keep=1, purpose="first-frame matrices, sensor-only, fixed-Q, F=10")
    # CB: archive-kept re-run of the eee_01 combined x F cells (F-331)
    for q, b in (("fq", "40_C1_pca_eigengap_posecombined_fixedq_full"), ("mq", "40_C1_pca_eigengap_posecombined_motionq_full")):
        for f in (1, 10, 30):
            add("r70_cb_eee_01_%s_f%d" % (q, f), b, d(DIAG, INFL(f)) if f != 1 else dict(DIAG), group="CB", tier="CB", keep=1,
                purpose="eee_01 %s combined map, global F=%d, re-run of the R69B cell with the lossless archive kept (F-331); identical config, results must reproduce R69B" % (q, f))
    # BL: block information scale
    for q, b in (("fq", "40_C1_pca_eigengap_posecombined_fixedq_full"), ("mq", "40_C1_pca_eigengap_posecombined_motionq_full")):
        for fr in (3, 6, 12, 25):
            for fp in (1, 2, 4):
                add("r70_bl_eee_01_%s_r%d_p%d" % (q, fr, fp), b, d(DIAG, BLK(fr, fp)), group="BL", tier="BL", keep=(1 if (fr, fp) in ((6, 2), (12, 2)) else 0),
                    purpose="eee_01 %s combined map, block information scale rotation %d / position %d (F-320; F-334 crossings: rotation 6-11, position 1.5-2.3)" % (q, fr, fp))
    for q, b in (("fq", "00_id_baseline_full"), ("mq", "00_id_motionq_full")):
        for fr, fp in ((300, 21), (1000, 21)):
            add("r70_bl_eee_01_%s_sensor_r%d_p%d" % (q, fr, fp), b, d(DIAG, BLK(fr, fp)), group="BL", tier="BL", keep=0,
                purpose="eee_01 %s sensor-only map, block information scale rotation %d / position %d (F-334: sensor-only needs > 1000 for rotation, about 21 for position)" % (q, fr, fp))
    for sq in SEQS_OTHER:
        for q, suf in (("fq", "baseline"), ("mq", "motionq")):
            for fr, fp in ((12, 2), (25, 2)):
                add("r70_bl_%s_%s_r%d_p%d" % (sq, q, fr, fp), "50_E_%s_%s_full" % (sq, suf), d(DIAG, COMB, BLK(fr, fp)), group="BL", tier="BL", keep=(1 if (fr, fp) == (12, 2) else 0),
                    purpose="%s %s combined map, block information scale rotation %d / position %d (F-334 crossings on this bag)" % (sq, suf, fr, fp))
    # RD: redundancy ladder (rule 10d: a ladder, scored on collapse and consistency rather than on ATE)
    for q, b in (("fq", "00_id_baseline_full"), ("mq", "00_id_motionq_full")):
        add("r70_rd_%s_w0p5" % q, b, d(DIAG, WB(0.5, False)), group="RD", tier="RD", keep=1, purpose="rung 1 control: Woodbury rho 0.5 alone, %s (F-317; diagnostics now on so redundancy statistics are logged, F-149/F-166)" % q)
        add("r70_rd_%s_w1p0" % q, b, d(DIAG, WB(1.0, False)), group="RD", tier="RD", keep=0, purpose="rung 1: Woodbury rho 1.0 (textbook), %s" % q)
        for rho in (0.25, 0.5, 1.0):
            add("r70_rd_%s_w%s_renorm" % (q, str(rho).replace(".", "p")), b, d(DIAG, WB(rho, True)), group="RD", tier="RD", keep=(1 if rho == 0.5 else 0),
                purpose="rung 3: Woodbury rho %s + renormalisation holding the weakest position eigenvalue, %s (F-317)" % (rho, q))
    for q, b in (("fq", "40_C1_pca_eigengap_posecombined_fixedq_full"), ("mq", "40_C1_pca_eigengap_posecombined_motionq_full")):
        add("r70_rd_%s_comb_w0p5_renorm_r12_p2" % q, b, d(DIAG, WB(0.5, True), BLK(12, 2)), group="RD", tier="RD", keep=1,
            purpose="rung 4: combined map + block scale 12/2 + Woodbury 0.5 + renormalisation, %s (F-317 + F-320)" % q)
    # IT: iteration count / early exit (F-168)
    for n, p in (("40_C3_fixedq_iters10_exitoff_full", "10 iterations, early exit off"), ("40_C3_fixedq_iters5_exitoff_full", "5 iterations, early exit off (control)"),
                 ("40_C3_fixedq_iters10_norm1em4_full", "10 iterations, step-norm exit 1e-4")):
        add("r70_it_" + n.replace("40_C3_fixedq_", "").replace("_full", ""), n, dict(DIAG), group="IT", tier="IT", keep=1,
            purpose="F-168: %s, fixed-Q, diagnostics on (per-iteration step size dtheta/dt visible in lio_update_diag)" % p)
    # OL: open loop with P0 for R, p, v near zero (F-73)
    for n, p in (("30_B11_openloop5s_baseline_60s", "baseline fixed-Q"), ("30_B11_openloop5s_datasheet_x1_60s", "datasheet Q x1"), ("30_B11_openloop5s_datasheet_x100_60s", "datasheet Q x100")):
        add("r70_ol_p0off_" + n.replace("30_B11_openloop5s_", "").replace("_60s", ""), n, dict(P0_RPV_OFF), group="OL", tier="OL", keep=1,
            purpose="F-73: open-loop 5 s windows, P0 for R, p, v = 1e-12 (configured), biases and gravity 1e-7, %s" % p)
    # CO (main part): coupled estimator, 60 s, full diagnostics (minimal_diagnostics false)
    add("r70_co_meas_fulldiag_60s", "10_smoke_coupled_measurement_fulldiag_60s", {"estimator.coupled.minimal_diagnostics": False}, group="CO", tier="CO", dur=60, keep=1,
        purpose="F-83/F-240: coupled measurement-time, fixed-Q, full coupled diagnostics, 60 s")
    add("r70_co_tail_fulldiag_60s", "10_smoke_coupled_tail_fulldiag_60s", {"estimator.coupled.minimal_diagnostics": False}, group="CO", tier="CO", dur=60, keep=1,
        purpose="F-83/F-240: coupled tail, fixed-Q, full coupled diagnostics, 60 s (comparator)")
    # ---- SF (separate queue, dispatched last, no wall cap)
    add("r70_sf_cell0p25_default", SRF25, dict(DIAG), group="SF", tier="SF", batch="sf", keep=1,
        purpose="surface cell 0.25 default density (digest must equal the R69B arm, G11; aggregate = sequential in R69B, so one arm)")
    for cs in (0.15, 0.20, 0.30, 0.35, 0.50):
        add("r70_sf_cell%s_default" % str(cs).replace(".", "p"), SRF25, d(DIAG, {"voxel_map.surface.cell_size": cs}), group="SF", tier="SF", batch="sf", keep=1,
            purpose="surface cell-size ladder at default density: cell %s (F-340: where does the backend start failing?)" % cs)
    add("r70_sf_cell0p425_default", SRF25, d(DIAG, {"voxel_map.surface.cell_size": 0.425}), group="SF", tier="SF", batch="sf", keep=1,
        purpose="surface cell 0.425 default density (digest must equal the R69B arm, G11)")
    add("r70_sf_cell0p425_dense", SRF_DENSE, d(DIAG, {"voxel_map.surface.cell_size": 0.425}), group="SF", tier="SF", batch="sf", keep=1,
        purpose="surface cell 0.425 dense density PFN1 ds-off (digest must equal the R69B arm, G11)")
    for cs, base, nm in ((0.25, SRF25, "cell0p25_default"), (0.425, SRF25, "cell0p425_default"), (0.425, SRF_DENSE, "cell0p425_dense")):
        add("r70_sf_%s_mergeoff" % nm, base, d(DIAG, {"voxel_map.surface.cell_size": cs, "voxel_map.surface.merge_angle_deg": 0.001}), group="SF", tier="SF", batch="sf", keep=1,
            purpose="surface %s with merging effectively off (merge angle 0.001 deg: no pair passes): isolates the per-cell planes from the merge step (F-340)" % nm)
    for cs in (0.25, 0.425):
        add("r70_sf_cell%s_default_mq" % str(cs).replace(".", "p"), SRF25, d(DIAG, MQ, {"voxel_map.surface.cell_size": cs}), group="SF", tier="SF", batch="sf", keep=1,
            purpose="surface cell %s default density under motion-Q (F-340: is the failure Q-dependent?)" % cs)
    add("r70_sf_octree_vox0p25_fq", "00_id_baseline_full", d(DIAG, {"voxel_map.map.voxel_size": 0.25}), group="SF", tier="SF", batch="sf", keep=1,
        purpose="octree at voxel size 0.25 (same scale as the working surface cell), fixed-Q (F-340 same-scale comparator)")
    for n, q in (("40_C6_pfn1_dsoff_fixedq_full", "fixedq"), ("40_C6_pfn1_dsoff_motionq_full", "motionq")):
        add("r70_sf_pfn1_dsoff_" + q, n, dict(DIAG), group="SF", tier="SF", batch="sf", keep=1,
            purpose="dense PFN1 ds-off pca on the octree (0.425 voxels), %s, residual_geometry on (digest must equal the R69B arm, G11); comparator for the dense surface arms (F-136)" % q)
    # CO (sf part): coupled, full bag, no wall cap (F-223: R67 coupled-measurement arms ended at 1,208-3,661 scans)
    for n, p in (("60_D3_coupled_measurement_gateoff_knots4_iters5_fixedq_full", "measurement-time, knots 4, fixed-Q"),
                 ("60_D3_coupled_measurement_gateoff_knots4_iters5_motionq_full", "measurement-time, knots 4, motion-Q"),
                 ("60_D3_coupled_tail_gateon_knots8_iters5_fixedq_full", "tail, knots 8, fixed-Q (the R67 coupled arm that completed)"),
                 ("60_D3_coupled_tail_gateon_knots6_iters5_motionq_full", "tail, knots 6, motion-Q (diverged in R67)")):
        if n not in J:
            sys.exit("base job %s not in r66_jobs.json" % n)
        add("r70_co_" + n.replace("60_D3_coupled_", "").replace("_full", ""), n, {}, group="CO", tier="CO", batch="sf", keep=1,
            purpose="F-83/F-223/F-240: coupled %s, full bag, no wall cap: report last scan and stop reason" % p)
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
            # tools/r69, r70_manifest, r70_inputs actually live), but sweep_listener.py's
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
            "cmd": ["python3", "src/livo_recon/tools/r69/r69_score_job.py", "--job-dir", "{job_dir}", "--out-dir", "{out_dir}", "--stage-out", "{stage_out}",
                    "--job-id", "{job_id}", "--spec", "{spec_dir}/%s", "--arm-table", "src/livo_recon/r70_manifest/arm_table.tsv",
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
                           "residual_geometry.csv", "info_scale.csv", "redundancy_renorm.csv"],
                "sha256": "twice"},
        "limits": {"workers": workers, "max_failures": 10, "max_attempts": 2, "min_free_gb": min_free_gb, "poll_interval_s": 10},
        "exit": {"when": "all_expected_handled"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", default="r70_manifest")
    ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--min-free-gb", type=float, default=150.0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    J = {j["job_id"]: j for j in json.load(open(a.jobs))}
    os.makedirs(a.out, exist_ok=True)
    out = {"smoke": [], "refuse": [], "main": [], "sf": []}; table = []; seen = set()
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
        if o.get("estimator.mode") == "decoupled" and r["id"] != "r70_refuse_estimator_decoupled":
            o["estimator.mode"] = "iekf"
        if "calib.bootstrap.point_cov" in o and r["batch"] != "refuse":
            sys.exit("point_cov key forbidden outside the refuse batch")
        o.pop("spline.mode", None)
        if r["id"] == "r70_refuse_spline_mode":
            o["spline.mode"] = "raw_imu"
        if r["id"] == "r70_refuse_estimator_decoupled":
            o["estimator.mode"] = "decoupled"
        for k in o:
            if "perturb" in k.lower():
                sys.exit("perturbation key forbidden: " + k)
        o["eval.state_trace_run_id"] = r["id"]
        path = "%s/r70_%s/%s" % (a.results_root, r["batch"], r["id"])
        o["outputs.path"] = path; o["outputs.debug_log_dir"] = path
        if r["dur"]:
            j["duration_secs"] = r["dur"]
        out[r["batch"]].append(j)
        sampling = "pfn%s_ds%s" % (o.get("cbk.lidar.point_filter_num", "def"), o.get("imu.ds.mode", "def"))
        table.append(dict(arm=r["id"], tier=r["tier"], arch="iekf_raw", kind="closed", duration_s=str(j.get("duration_secs", "full")), seq=j["seq"],
                          notes=r["purpose"], keep_raw=str(r["keep"]), dup_of="", r66_id=r["base"], group=r["group"], batch="r70_" + r["batch"],
                          sampling=sampling))
    # R69B stage-1b: the broken-clone proof job is a COPY of a finished smoke job (tools/r69/make_broken_clone.py); it needs an arm-table row so that the
    # scorer fails only because its state_reference.txt was deleted, not because the arm is unknown (R69 stage-1 defect F-324(6)).
    table.append(dict(arm="00_id_broken_clone", tier="PROOF", arch="iekf_raw", kind="closed", duration_s="60", seq=next(t["seq"] for t in table if t["arm"] == "00_id_baseline_60s"),
                      notes="listener failure-path proof: copy of 00_id_baseline_60s with state_reference.txt deleted (never dispatched)", keep_raw="0", dup_of="",
                      r66_id="00_id_baseline_60s", group="PROOF", batch="r70_proof", sampling=next(t["sampling"] for t in table if t["arm"] == "00_id_baseline_60s")))
    for b, lst in out.items():
        json.dump(lst, open(os.path.join(a.out, "jobs_%s.json" % b), "w"), indent=1)
    cols = ["arm", "tier", "arch", "kind", "duration_s", "seq", "notes", "keep_raw", "dup_of", "r66_id", "group", "batch", "sampling"]
    with open(os.path.join(a.out, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t"); w.writeheader(); w.writerows(table)
    q = lambda b, n: {"dir": "ablations/_job_queue/r70_%s" % b, "results_root": "%s/r70_%s" % (a.results_root, b), "expected_jobs": n}
    smoke_spec = listener_spec("R70-smoke", [q("smoke", len(out["smoke"])), q("refuse", len(out["refuse"]))], a.results_root, a.min_free_gb, a.workers)
    main_spec = listener_spec("R70-main", [q("main", len(out["main"])), q("sf", len(out["sf"]))], a.results_root, a.min_free_gb, a.workers)
    broken_spec = listener_spec("R70-broken", [{"dir": "ablations/_job_queue/r70_smoke_broken", "results_root": "%s/r70_smoke_broken" % a.results_root, "expected_jobs": 1}],
                                a.results_root, a.min_free_gb, 1)
    for nm, sp in (("smoke", smoke_spec), ("main", main_spec), ("broken", broken_spec)):
        fn = "listener_spec_%s.json" % nm
        for sg in sp["stages"]:
            sg["cmd"] = [c.replace("%s", fn) for c in sg["cmd"]]
        for vf in sp["verify"]:
            vf["cmd"] = [c.replace("%s", fn) for c in vf["cmd"]]
        json.dump(sp, open(os.path.join(a.out, fn), "w"), indent=1)
    print("smoke %d  refuse %d  main %d  sf %d  total %d (arm table also has the PROOF row)" % (len(out["smoke"]), len(out["refuse"]), len(out["main"]), len(out["sf"]), len(table)))
    print("main by group:", dict(collections.Counter(t["group"] for t in table if t["batch"] == "r70_main")))


main()
