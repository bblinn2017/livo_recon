#!/usr/bin/env python3
"""R71: build jobs_smoke.json, jobs_refuse.json, jobs_main.json, jobs_sf.json, arm_table.tsv, coupled_pairs.txt, scan_arms_priority.txt and the three listener specs
from the R66 job list (r66_jobs.json, shipped in the package).

    python3 tools/r69/make_r71_manifest.py --jobs r69_inputs/r66_jobs.json --out r71_manifest --results-root /root/catkin_ws/livo_recon_results

Every R71 arm is an R66 job (same overrides) plus the delta written below. Rewrites applied to EVERY job: estimator.mode "decoupled" -> "iekf" (the old value is refused
by the build), spline.mode removed (the key is refused). NO job sets calib.bootstrap.point_cov except the refuse job. No perturbation key anywhere (rule 3.11). Nothing
edits the repo. expected_jobs in the specs = the number of jobs written (job-count landmine). Every arm is listed with its follow-up(s) in the arm table (`notes`).

Blocks (batch main unless stated)
  smoke   identity gates G1-G5, G8a, G8c (all write-only logs, including the residual dump, are inert), one 60 s smoke per new flag (rule 3.6), the control that makes the
          dump-frame check C9 able to fail, the coupled full-diagnostics smokes that exercise the new knot columns
  refuse  ten jobs that MUST fail at startup (the nine R70 refusals plus a malformed lio/residual_dump_frames)
  ID      00_id_baseline_full_rerun (determinism, G5)
  MX      three first-frame matrix arms the gates compare (G8b, G9)
  RA      twelve 60 s arms with the per-residual dump at 13 frames (F-374, F-375, F-377): baselines, combined F=1/10, sensor F=10, dense octree, surface, octree 0.25
  CB      global F ladder 1, 3, 10, 30, 100 on the combined map, fixed-Q and motion-Q (F_rot = F_pos, per-axis sd; F-376)
  SO      global F 3, 10, 30 on the sensor-only map, fixed-Q and motion-Q (F-376)
  IT      iteration count and early exit under motion-Q, plain and with global F=10 (F-369)
  SF      (batch sf, dispatched last, no wall cap) surface backend with census columns: cell ladder, 0.25 dense, min_points probes, octree 0.25 and dense octree controls
  CO      (batch sf) coupled estimator, full bag, full coupled diagnostics, gate on: tail and measurement-time, 4 and 8 knots, fixed-Q and motion-Q (F-370)
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


DUMPF = "1,2,3,4,5,10,25,50,100,200,300,400,480"        # residual dump frames (stationary window ends at scan 480; onset at 485)
DUMP = lambda fr=DUMPF: {"lio.residual_dump_frames": fr}
CO_FULL = {"estimator.coupled.minimal_diagnostics": False}
C1FQ, C1MQ = "40_C1_pca_eigengap_posecombined_fixedq_full", "40_C1_pca_eigengap_posecombined_motionq_full"


def arms(J):
    A = []

    def add(id_, base, delta=None, batch="main", group="", tier="", purpose="", keep=0, dur=None):
        A.append(dict(id=id_, base=base, delta=delta or {}, batch=batch, group=group, tier=tier, purpose=purpose, keep=keep, dur=dur))

    def d(*ds):
        o = {}
        for x in ds:
            o.update(x)
        return o

    # ---- smoke: identity gates + one 60 s smoke per new flag (rule 3.6)
    for n in ("00_id_baseline_48s", "00_id_baseline_60s", "00_id_baseline_full", "00_id_motionq_full"):
        add(n, n, batch="smoke", group="ID", tier="ID", purpose="identity: digest must equal the R67 sensor_model digest (the R71 code is inert at its defaults)", keep=1)
    add("00_id_default_posecov_60s", "00_id_baseline_60s", {"voxel_map.plane.plane_fit_pose_cov_mode": None}, batch="smoke", group="ID", tier="ID",
        purpose="plane_fit_pose_cov_mode key deleted: compiled default sensor_only equals the pinned value; digest equals 00_id_baseline_60s")
    add("00_id_infl_whole_f1_60s", "00_id_baseline_60s", INFL(1.0), batch="smoke", group="ID", tier="ID",
        purpose="whole-run inflation at factor 1.0 is the identity: digest equals 00_id_baseline_60s")
    add("r71_diag_on_baseline_60s", "00_id_baseline_60s", DIAG, batch="smoke", group="ID", tier="ID", keep=1,
        purpose="ALL write-only logs ON except the residual dump (update_diag, live_point_cov, first-frame dump frames 1..5, residual_geometry): digest must equal 00_id_baseline_60s (G8a); the listener's real-data proof job")
    add("r71_smoke_dump_on_baseline_60s", "00_id_baseline_60s", d(DIAG, DUMP("1,2,3,5,10,50,100,200,400,480")), batch="smoke", group="ID", tier="ID", keep=1,
        purpose="residual dump ON at frames 1,2,3,5,10,50,100,200,400,480 (F-363, F-375): digest must equal 00_id_baseline_60s (G8c); check C9 rebuilds Lambda from the dump and compares it with HtH of lio_update_diag")
    add("r71_smoke_dump_frame_beyond_60s", "00_id_baseline_60s", d(DIAG, DUMP("1,99999")), batch="smoke", group="ID", tier="ID", keep=0,
        purpose="CONTROL for check C9 (rule 2.4): frame list 1,99999; frame 99999 must be ABSENT from residual_dump.csv and frame 1 present, so the frame check can fail")
    add("r71_smoke_block_f12_2_60s", C1FQ, d(DIAG, BLK(12, 2)), batch="smoke", group="BL", tier="SMOKE", dur=60, keep=1,
        purpose="smoke of lio/info_scale block 12 / 2 (combined map, fixed-Q, 60 s): info_scale.csv written, engagement not INERT (F-320); G9b")
    add("r71_smoke_block_f10_10_60s", C1FQ, d(DIAG, BLK(10, 10)), batch="smoke", group="BL", tier="SMOKE", dur=60, keep=1,
        purpose="block scale 10 / 10 equals the global scale F=10 (algebraic identity, G9a): frame-1 HtH/Htz equal those of r71_mx_cb_fq_f10_60s")
    add("r71_smoke_renorm_60s", "00_id_baseline_60s", d(DIAG, WB(0.5, True)), batch="smoke", group="RD", tier="SMOKE", keep=1,
        purpose="smoke of lio/residual_redundancy/renorm hold_pos_min_eig with Woodbury rho 0.5 (60 s): redundancy_renorm.csv written, engagement not INERT (F-317)")
    add("r71_smoke_surface0p425_60s", SRF25, d(DIAG, {"voxel_map.surface.cell_size": 0.425}), batch="smoke", group="SF", tier="SMOKE", dur=60, keep=1,
        purpose="smoke of the surface census: the six new columns of surface_map_frames.csv (F-363, F-377) and residual_geometry at cell 0.425")
    add("r71_smoke_octree_vox0p25_60s", "00_id_baseline_60s", d(DIAG, {"voxel_map.map.voxel_size": 0.25}), batch="smoke", group="SF", tier="SMOKE", keep=1,
        purpose="smoke of octree voxel_size 0.25 (60 s): the key is accepted and residual_geometry written (F-340 same-scale octree comparator)")
    add("r71_co_tail_fulldiag_60s", "10_smoke_coupled_tail_fulldiag_60s", dict(CO_FULL), batch="smoke", group="CO", tier="SMOKE", dur=60, keep=1,
        purpose="smoke of the new absolute knot-state columns in joint_knot_all_scans.csv (F-370; check C10): coupled tail, fixed-Q, full coupled diagnostics, 60 s")
    add("r71_co_meas_fulldiag_60s", "10_smoke_coupled_measurement_fulldiag_60s", dict(CO_FULL), batch="smoke", group="CO", tier="SMOKE", dur=60, keep=1,
        purpose="coupled measurement-time, fixed-Q, full coupled diagnostics, 60 s (comparator for the tail smoke)")
    # ---- refuse (every one MUST die at startup)
    R = lambda id_, base, delta, purpose: add(id_, base, delta, batch="refuse", group="REF", tier="REF", dur=5, purpose=purpose)
    R("r71_refuse_point_cov_zero", "00_id_baseline_60s", {"calib.bootstrap.point_cov": "zero"}, "MUST FAIL at startup: removed key calib/bootstrap/point_cov")
    R("r71_refuse_spline_mode", "00_id_baseline_60s", {}, "MUST FAIL at startup: the removed key spline/mode (even with the value raw_imu)")
    R("r71_refuse_estimator_decoupled", "00_id_baseline_60s", {}, "MUST FAIL at startup: estimator/mode 'decoupled' was renamed 'iekf'")
    R("r71_refuse_surface_cell0p10", SRF25, {"voxel_map.surface.cell_size": 0.10}, "MUST FAIL at startup: surface cell size 0.10 is retired (Bryce 2026-10-01)")
    R("r71_refuse_octree_voxel0p10", "00_id_baseline_60s", {"voxel_map.map.voxel_size": 0.10}, "MUST FAIL at startup: octree voxel size 0.10 is retired")
    R("r71_refuse_info_scale_inert", "00_id_baseline_60s", {"lio.info_scale.mode": "block"}, "MUST FAIL at startup: block mode with rotation == position == 1 is inert")
    R("r71_refuse_info_scale_suboption", "00_id_baseline_60s", {"lio.info_scale.rotation": 5.0}, "MUST FAIL at startup: lio/info_scale/rotation outside mode=block")
    R("r71_refuse_info_scale_coupled", "10_smoke_coupled_tail_fulldiag_60s", BLK(5, 2), "MUST FAIL at startup: block information scale under estimator/mode=coupled")
    R("r71_refuse_renorm_without_woodbury", "00_id_baseline_60s", {"lio.residual_redundancy.renorm": "hold_pos_min_eig"},
      "MUST FAIL at startup: renorm outside residual_redundancy/mode=woodbury")
    R("r71_refuse_dump_frames_bad", "00_id_baseline_60s", {"lio.residual_dump_frames": "1,two,3"},
      "MUST FAIL at startup: lio/residual_dump_frames with a token that is not a positive integer")
    # ---- main
    add("00_id_baseline_full_rerun", "00_id_baseline_full", group="ID", tier="ID", keep=1, purpose="determinism control (digest equals 00_id_baseline_full), G5")
    # MX: the first-frame matrix arms the gates compare (G8b, G9); the per-residual dumps below supersede the matrices for analysis
    add("r71_mx_baseline_fq_60s", "00_id_baseline_60s", dict(DIAG), group="MX", tier="MX", keep=1, purpose="first-frame matrices frames 1..5, sensor-only fixed-Q (G8b control)")
    add("r71_mx_cb_fq_f1_60s", C1FQ, dict(DIAG), group="MX", tier="MX", dur=60, keep=1, purpose="first-frame matrices, combined, fixed-Q, F=1 (G9 reference)")
    add("r71_mx_cb_fq_f10_60s", C1FQ, d(DIAG, INFL(10)), group="MX", tier="MX", dur=60, keep=1, purpose="first-frame matrices, combined, fixed-Q, F=10 (G9 reference)")
    # RA: per-residual information analysis (F-374, F-375, F-368): 60 s arms with the residual dump at DUMPF and diagnostics on (lio_update_diag for the cross-frame increments)
    RA = lambda id_, base, delta, purpose, keep=0, dur=60: add(id_, base, d(DIAG, DUMP(), delta), group="RA", tier="RA", dur=dur, keep=keep, purpose=purpose)
    RA("r71_ra_baseline_fq_60s", "00_id_baseline_60s", {}, "residual dump, sensor-only default fixed-Q (the reference arm)", keep=1, dur=None)
    RA("r71_ra_baseline_mq_60s", "00_id_motionq_full", {}, "residual dump, sensor-only default motion-Q", keep=1)
    RA("r71_ra_cb_fq_f1_60s", C1FQ, {}, "residual dump, combined map, fixed-Q, F=1", dur=60)
    RA("r71_ra_cb_fq_f10_60s", C1FQ, INFL(10), "residual dump, combined map, fixed-Q, global F=10 (removes the yaw error in R70): compare per-frame information with F=1", keep=1)
    RA("r71_ra_cb_mq_f1_60s", C1MQ, {}, "residual dump, combined map, motion-Q, F=1")
    RA("r71_ra_cb_mq_f10_60s", C1MQ, INFL(10), "residual dump, combined map, motion-Q, global F=10")
    RA("r71_ra_sensor_fq_f10_60s", "00_id_baseline_60s", INFL(10), "residual dump, sensor-only map, fixed-Q, F=10")
    RA("r71_ra_dense_fq_60s", "40_C6_pfn1_dsoff_fixedq_full", {}, "residual dump, dense octree (point_filter_num 1, downsampling off), fixed-Q: back-wall planes with downsampling off (F-377)", keep=1)
    RA("r71_ra_dense_mq_60s", "40_C6_pfn1_dsoff_motionq_full", {}, "residual dump, dense octree, motion-Q")
    RA("r71_ra_surface0p25_fq_60s", SRF25, {"voxel_map.surface.cell_size": 0.25}, "residual dump, surface backend cell 0.25 default density: which planes carry the information (F-377)")
    RA("r71_ra_surface0p25_dense_fq_60s", SRF_DENSE, {"voxel_map.surface.cell_size": 0.25}, "residual dump, surface backend cell 0.25 dense density: are the back-wall planes present when downsampling is off (F-377)", keep=1)
    RA("r71_ra_octree_vox0p25_60s", "00_id_baseline_60s", {"voxel_map.map.voxel_size": 0.25}, "residual dump, octree at voxel 0.25 (same scale as the surface cell)")
    # CB: global F ladder with F_rot = F_pos and per-axis sd (F-376); F = 1, 10, 30 repeat R70 (G11), F = 3 and 100 are new
    for q, b in (("fq", C1FQ), ("mq", C1MQ)):
        for f in (1, 3, 10, 30, 100):
            add("r71_cb_eee_01_%s_f%d" % (q, f), b, d(DIAG, INFL(f)) if f != 1 else dict(DIAG), group="CB", tier="CB", keep=(1 if f in (10, 100) else 0),
                purpose="eee_01 %s combined map, global F=%d (F_rot = F_pos), full bag, diagnostics on; per-axis sd in the reduced tables (F-376)%s" % (q, f, "; repeat of the R70 / R69B cell (digest G11)" if f in (1, 10, 30) else ""))
    for q, b in (("fq", "00_id_baseline_full"), ("mq", "00_id_motionq_full")):
        for f in (3, 10, 30):
            add("r71_so_eee_01_%s_f%d" % (q, f), b, d(DIAG, INFL(f)), group="SO", tier="SO", keep=0,
                purpose="eee_01 %s sensor-only map, global F=%d, full bag (F_rot = F_pos ladder on the default map, F-376)" % (q, f))
    # IT: iteration count and early exit under motion-Q (F-369); fixed-Q was R70
    add("r71_it_mq_iters10_exitoff", "40_C3_motionq_iters10_exitoff_full", dict(DIAG), group="IT", tier="IT", keep=1, purpose="F-369: motion-Q, 10 iterations, early exit off, diagnostics on")
    add("r71_it_mq_iters10_norm1em4", "40_C3_motionq_iters10_norm1em4_full", dict(DIAG), group="IT", tier="IT", keep=1, purpose="F-369: motion-Q, 10 iterations, step-norm exit 1e-4")
    add("r71_it_mq_iters5_exitoff", "40_C3_motionq_iters5_exitoff_full", dict(DIAG), group="IT", tier="IT", keep=0, purpose="F-369: motion-Q, 5 iterations early exit off (control; equals the default count)")
    for n, dl, p in (("exitoff", {"lio.ekf.max_iterations": 10}, "10 iterations, early exit off"),
                     ("norm1em4", {"lio.ekf.max_iterations": 10, "lio.ekf.min_norm_dtheta": 0.0001, "lio.ekf.min_norm_dt": 0.0001}, "10 iterations, step-norm exit 1e-4")):
        add("r71_it_mq_f10_iters10_" + n, C1MQ, d(DIAG, INFL(10), dl), group="IT", tier="IT", keep=0, purpose="F-369: combined map, motion-Q, global F=10, %s" % p)
    # SF (separate queue, dispatched last, no wall cap): surface backend, census columns on every arm
    add("r71_sf_cell0p25_default", SRF25, dict(DIAG), group="SF", tier="SF", batch="sf", keep=1,
        purpose="surface cell 0.25 default density (digest must equal the R69B arm, G11); the new census columns (onset window, per-test failure)")
    for cs in (0.15, 0.20, 0.30, 0.35, 0.50):
        add("r71_sf_cell%s_default" % str(cs).replace(".", "p"), SRF25, d(DIAG, {"voxel_map.surface.cell_size": cs}), group="SF", tier="SF", batch="sf", keep=0,
            purpose="surface cell-size ladder at default density: cell %s (valid share and per-test failure against cell size, F-377)" % cs)
    add("r71_sf_cell0p425_default", SRF25, d(DIAG, {"voxel_map.surface.cell_size": 0.425}), group="SF", tier="SF", batch="sf", keep=1,
        purpose="surface cell 0.425 default density (digest must equal the R69B arm, G11)")
    add("r71_sf_cell0p425_dense", SRF_DENSE, d(DIAG, {"voxel_map.surface.cell_size": 0.425}), group="SF", tier="SF", batch="sf", keep=1,
        purpose="surface cell 0.425 dense density (digest must equal the R69B arm, G11)")
    add("r71_sf_cell0p25_dense", SRF_DENSE, d(DIAG, {"voxel_map.surface.cell_size": 0.25}), group="SF", tier="SF", batch="sf", keep=1,
        purpose="surface cell 0.25 dense density, full bag (new; the dense map at the cell size that works)")
    add("r71_sf_cell0p25_default_mq", SRF25, d(DIAG, MQ, {"voxel_map.surface.cell_size": 0.25}), group="SF", tier="SF", batch="sf", keep=1,
        purpose="surface cell 0.25 default density under motion-Q (F-340: is the failure Q-dependent?)")
    for mp in (3, 8):
        add("r71_sf_cell0p25_minpts%d" % mp, SRF25, d(DIAG, {"voxel_map.surface.cell_size": 0.25, "voxel_map.surface.min_points": mp}), group="SF", tier="SF", batch="sf", keep=0,
            purpose="PROBE (existing key, not a proposed default): surface min_points %d at cell 0.25 (base 5): does the valid share and the onset plane count move (F-371, F-377)" % mp)
    for q, b in (("fq", "00_id_baseline_full"), ("mq", "00_id_motionq_full")):
        add("r71_sf_octree_vox0p25_%s" % q, b, d(DIAG, {"voxel_map.map.voxel_size": 0.25}), group="SF", tier="SF", batch="sf", keep=1,
            purpose="octree at voxel size 0.25 (same scale as the surface cell), %s: valid share and plane_fit_reasons at the same cell size (F-371)" % q)
    for n, q in (("40_C6_pfn1_dsoff_fixedq_full", "fixedq"), ("40_C6_pfn1_dsoff_motionq_full", "motionq")):
        add("r71_sf_pfn1_dsoff_" + q, n, dict(DIAG), group="SF", tier="SF", batch="sf", keep=1,
            purpose="dense PFN1 ds-off pca on the octree (0.425 voxels), %s (digest must equal the R69B arm, G11); comparator for the dense surface arms" % q)
    # CO: coupled estimator, full bag, full coupled diagnostics, gate on (the gate-off 4-knot arms aborted in R70, F-353): tail and measurement-time, 4 and 8 knots, both Q
    for t, tn in (("tail", "tail"), ("measurement", "meas")):
        for k in (4, 8):
            for q, qn in (("fixedq", "fq"), ("motionq", "mq")):
                n = "60_D3_coupled_%s_gateon_knots%d_iters5_%s_full" % (t, k, q)
                if n not in J:
                    sys.exit("base job %s not in r66_jobs.json" % n)
                add("r71_co_%s_gateon_knots%d_%s" % (tn, k, qn), n, dict(CO_FULL), group="CO", tier="CO", batch="sf", keep=1,
                    purpose="F-370: coupled %s, gate on, %d knots, %s, full bag, full coupled diagnostics (absolute knot state per scan); compared with the decoupled iekf baseline of the same Q (coupled_pairs.txt)" % (t, k, q))
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
            # tools/r69, r71_manifest, r71_inputs actually live), but sweep_listener.py's
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
                    "--job-id", "{job_id}", "--spec", "{spec_dir}/%s", "--arm-table", "src/livo_recon/r71_manifest/arm_table.tsv",
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
                           "residual_dump.csv", "residual_dump_state.csv", "joint_knot_all_scans.csv", "joint_knot_scan_summary.csv"],   # R71: all are in the lossless archive (BIG list of r69_score_job.py)
                "sha256": "twice"},
        "limits": {"workers": workers, "max_failures": 10, "max_attempts": 2, "min_free_gb": min_free_gb, "poll_interval_s": 10},
        "exit": {"when": "all_expected_handled"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", default="r71_manifest")
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
        if o.get("estimator.mode") == "decoupled" and r["id"] != "r71_refuse_estimator_decoupled":
            o["estimator.mode"] = "iekf"
        if "calib.bootstrap.point_cov" in o and r["batch"] != "refuse":
            sys.exit("point_cov key forbidden outside the refuse batch")
        o.pop("spline.mode", None)
        if r["id"] == "r71_refuse_spline_mode":
            o["spline.mode"] = "raw_imu"
        if r["id"] == "r71_refuse_estimator_decoupled":
            o["estimator.mode"] = "decoupled"
        for k in o:
            if "perturb" in k.lower():
                sys.exit("perturbation key forbidden: " + k)
        o["eval.state_trace_run_id"] = r["id"]
        path = "%s/r71_%s/%s" % (a.results_root, r["batch"], r["id"])
        o["outputs.path"] = path; o["outputs.debug_log_dir"] = path
        if r["dur"]:
            j["duration_secs"] = r["dur"]
        out[r["batch"]].append(j)
        sampling = "pfn%s_ds%s" % (o.get("cbk.lidar.point_filter_num", "def"), o.get("imu.ds.mode", "def"))
        table.append(dict(arm=r["id"], tier=r["tier"], arch="iekf_raw", kind=("open" if r["group"] == "OL" else "closed"), duration_s=str(j.get("duration_secs", "full")), seq=j["seq"],
                          notes=r["purpose"], keep_raw=str(r["keep"]), dup_of="", r66_id=r["base"], group=r["group"], batch="r71_" + r["batch"],
                          sampling=sampling))
    # R69B stage-1b: the broken-clone proof job is a COPY of a finished smoke job (tools/r69/make_broken_clone.py); it needs an arm-table row so that the
    # scorer fails only because its state_reference.txt was deleted, not because the arm is unknown (R69 stage-1 defect F-324(6)).
    table.append(dict(arm="00_id_broken_clone", tier="PROOF", arch="iekf_raw", kind="closed", duration_s="60", seq=next(t["seq"] for t in table if t["arm"] == "00_id_baseline_60s"),
                      notes="listener failure-path proof: copy of 00_id_baseline_60s with state_reference.txt deleted (never dispatched)", keep_raw="0", dup_of="",
                      r66_id="00_id_baseline_60s", group="PROOF", batch="r71_proof", sampling=next(t["sampling"] for t in table if t["arm"] == "00_id_baseline_60s")))
    for b, lst in out.items():
        json.dump(lst, open(os.path.join(a.out, "jobs_%s.json" % b), "w"), indent=1)
    cols = ["arm", "tier", "arch", "kind", "duration_s", "seq", "notes", "keep_raw", "dup_of", "r66_id", "group", "batch", "sampling"]
    with open(os.path.join(a.out, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t"); w.writeheader(); w.writerows(table)
    q = lambda b, n: {"dir": "ablations/_job_queue/r71_%s" % b, "results_root": "%s/r71_%s" % (a.results_root, b), "expected_jobs": n}
    smoke_spec = listener_spec("R71-smoke", [q("smoke", len(out["smoke"])), q("refuse", len(out["refuse"]))], a.results_root, a.min_free_gb, a.workers)
    main_spec = listener_spec("R71-main", [q("main", len(out["main"])), q("sf", len(out["sf"]))], a.results_root, a.min_free_gb, a.workers)
    broken_spec = listener_spec("R71-broken", [{"dir": "ablations/_job_queue/r71_smoke_broken", "results_root": "%s/r71_smoke_broken" % a.results_root, "expected_jobs": 1}],
                                a.results_root, a.min_free_gb, 1)
    for nm, sp in (("smoke", smoke_spec), ("main", main_spec), ("broken", broken_spec)):
        fn = "listener_spec_%s.json" % nm
        for sg in sp["stages"]:
            sg["cmd"] = [c.replace("%s", fn) for c in sg["cmd"]]
        for vf in sp["verify"]:
            vf["cmd"] = [c.replace("%s", fn) for c in vf["cmd"]]
        json.dump(sp, open(os.path.join(a.out, fn), "w"), indent=1)
    pairs = ",".join("%s:%s" % (t["arm"], "00_id_motionq_full" if t["arm"].endswith("_mq") else "00_id_baseline_full") for t in table if t["arm"].startswith("r71_co_") and t["batch"] == "r71_sf")
    open(os.path.join(a.out, "coupled_pairs.txt"), "w").write(pairs + "\n")      # build_return.py: coupled arm : decoupled iekf arm of the same Q
    with open(os.path.join(a.out, "scan_arms_priority.txt"), "w") as f:       # return-thinning order: identity, analysis groups first
        order = {"ID": 0, "RA": 1, "CB": 2, "CO": 3, "IT": 4, "SO": 5, "SF": 6, "MX": 7}
        for t in sorted((t for t in table if t["group"] in order), key=lambda t: order[t["group"]]):
            f.write(t["arm"] + "\n")
    print("smoke %d  refuse %d  main %d  sf %d  total %d (arm table also has the PROOF row)" % (len(out["smoke"]), len(out["refuse"]), len(out["main"]), len(out["sf"]), len(table)))
    print("main by group:", dict(collections.Counter(t["group"] for t in table if t["batch"] == "r71_main")))


main()
