#!/usr/bin/env python3
"""R74: build jobs_<batch>.json, arm_table.tsv, scan_arms_priority.txt, switch_pairs.json, required_outputs.json and the listener specs from the R66 job list (r66_jobs.json).

    python3 tools/r69/make_r74_manifest.py --jobs r69_inputs/r66_jobs.json --out r74_manifest --results-root /root/catkin_ws/livo_recon_results

Planning agent's script; NOT run against the real host by me (it is run in the selftest on the shipped r66_jobs.json only: counts and key checks).
Every arm is an R66 job (same overrides) plus the delta written below. estimator.mode "decoupled" -> "iekf"; spline.mode removed. No perturbation key anywhere (rule 3.11).
Every arm carries `asked`: Bryce's words (or the register text of the follow-up he authorised) that it implements, or "OWN: <why>" for the three arms I added (approved 2026-10-02) and
"F-427 scope" for the refuse arms (the register text of F-427 names them). INSTRUCTIONS section 1 prints this as the compliance table.

Batches (queues)
  smoke   3 own arms: smoke at voxel 1.0, determinism control, identity arm at voxel 0.425 on the R74 tree
  refuse  8 arms: each removed key (F-427) must die at startup naming it
  ef      F-428 (80) + F-430 (240): eee_01 EF grid (calibration samples x ds x pfn x Q, 40 cells)
  f410    1 full-bag coupled measurement k8 motion-Q arm (lazy prior)
  sw      F-426 part 2: 4 process-noise switch arms (70 s)
  qa      F-418: 12 lever arms x 9 bags = 108, plus a fixed-Q and a motion-Q baseline per bag (18), all full bag
"""
import argparse, collections, copy, csv, json, math, os, sys

BAGS = ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]
DIAG = {"lio.update_diag_en": True, "lio.live_point_cov_en": True, "lio.first_frame_dump_frames": 5, "lio.residual_geometry_en": True}
CO_FULL = {"estimator.coupled.minimal_diagnostics": False}
CO_REC = {"estimator.coupled.failure_dump": True, "estimator.coupled.record_frames": 5}
CO_LOG = {"estimator.coupled.update_log_en": True}
ZUPT = lambda mode, g=0.02, a=0.15, n=3, s=0.02: dict({"lio.zupt.mode": mode, "lio.zupt.gyro_thresh": g, "lio.zupt.acc_thresh": a, "lio.zupt.enter_scans": n},
                                                      **({"lio.zupt.sigma_v": s} if mode == "apply" else {}))
GAIN = lambda mode, **kw: dict({"imu.process_noise.gain.mode": mode}, **{"imu.process_noise.gain." + k: v for k, v in kw.items()})
# F-428 / F-430 frame-1 dump: the frame-1 residual dump, update diagnostics, and (F-428) the post-calibration voxel map with the plane test values
F1DUMP = {"lio.first_frame_dump_frames": 1, "lio.residual_dump_frames": "1", "lio.update_diag_en": True}
SNAP = {"voxel_map.map.snapshot_post_calibration": True}
VOX = lambda v, ml: {"voxel_map.map.voxel_size": float(v), "voxel_map.map.max_layer": int(ml)}
LINE = {"hardabs": {"voxel_map.plane.line_residual.mode": "hard", "voxel_map.plane.line_residual.threshold": "absolute"},
        "hardrel": {"voxel_map.plane.line_residual.mode": "hard", "voxel_map.plane.line_residual.threshold": "relative", "voxel_map.plane.line_residual.rel_sigmas": 3.0},
        "smooth": {"voxel_map.plane.line_residual.mode": "smooth"}}
SW_DUR = 70
SBS_WIN, SBS_ONSET = (330, 420), 340
EEE_WIN, EEE_ONSET = (470, 540), 481
EF_N = (200, 400, 800, 1600, 3850)
REMOVED_KEYS = ["imu.process_noise.motion.energy_source", "imu.process_noise.motion.local_poly.order", "imu.process_noise.disturbance.mode", "imu.process_noise.disturbance.qc_acc",
                "imu.process_noise.disturbance.qc_gyr", "imu.process_noise.continuous_prior.mode", "imu.process_noise.continuous_prior.qc_acc", "imu.process_noise.continuous_prior.qc_gyr"]
REMOVED_VAL = {"imu.process_noise.motion.energy_source": "local_poly", "imu.process_noise.motion.local_poly.order": 2, "imu.process_noise.disturbance.mode": "wnoa",
               "imu.process_noise.disturbance.qc_acc": 0.01, "imu.process_noise.disturbance.qc_gyr": 1e-4, "imu.process_noise.continuous_prior.mode": "wnoa",
               "imu.process_noise.continuous_prior.qc_acc": 0.01, "imu.process_noise.continuous_prior.qc_gyr": 1e-4}


def mqs(acc, gyro):
    return {"imu.process_noise.motion.acc_scale": float(acc), "imu.process_noise.motion.gyro_scale": float(gyro)}


def tag(x):
    return str(x).replace(".", "p")


def base_of(bag, q):
    """R66 base job: eee_01 identity jobs, other bags the R67 50_E jobs. q in fq|mq."""
    if bag == "eee_01":
        return "00_id_baseline_full" if q == "fq" else "00_id_motionq_full"
    return "50_E_%s_%s_full" % (bag, "baseline" if q == "fq" else "motionq")


def switch(onset, model, acc=None, gyr=None):
    o = {"imu.process_noise.switch.scan": onset, "imu.process_noise.switch.model": model}
    if model == "isotropic":
        o.update({"imu.process_noise.switch.motion_beta": 0.9, "imu.process_noise.switch.motion_acc_scale": float(acc), "imu.process_noise.switch.motion_gyro_scale": float(gyr),
                  "imu.process_noise.switch.motion_acc_cap": 1.0e6, "imu.process_noise.switch.motion_gyro_cap": 1.0e6})
    return o


def requires_of(o, batch):
    """Files an arm must produce, derived from its overrides (arm_table.tsv column "requires"; r74_score_job.py fails the arm's stage when one is absent)."""
    if batch == "refuse":
        return []
    r = ["engagement_imu.txt"]
    if o.get("lio.residual_dump_frames"):
        r += ["residual_dump.csv", "residual_dump_state.csv"]
    if o.get("lio.update_diag_en"):
        r.append("lio_update_diag.csv")
    if o.get("voxel_map.map.snapshot_post_calibration"):
        r.append("post_calibration_voxelmap.csv")
    if o.get("estimator.coupled.update_log_en"):
        r += ["coupled_update_iter.csv", "coupled_update_scan.csv"]
    if o.get("estimator.coupled.record_frames"):
        r += ["coupled_record_f%d.txt" % i for i in range(1, int(o["estimator.coupled.record_frames"]) + 1)]
    if o.get("imu.process_noise.gain.mode") not in (None, "off"):
        r.append("q_adapt.csv")
    if o.get("lio.zupt.mode") not in (None, "off"):
        r.append("zupt.csv")
    if o.get("lio.ekf.step_cap.mode") == "hard":
        r += ["step_cap.csv"]
    return r


def arms(J):
    A = []

    def add(id_, base, delta=None, batch="main", group="", tier="", purpose="", keep=0, dur=None, asked="", f417=0, plane_check=0):
        A.append(dict(id=id_, base=base, delta=delta or {}, batch=batch, group=group, tier=tier, purpose=purpose, keep=keep, dur=dur, asked=asked, f417=f417, plane_check=plane_check))

    def d(*ds):
        o = {}
        for x in ds:
            o.update(x)
        return o

    B60, BF, BM = "00_id_baseline_60s", "00_id_baseline_full", "00_id_motionq_full"
    # ---- own arms (each explained to Bryce for approval; approved 2026-10-02)
    add("r74_own_id_baseline_60s", B60, {}, batch="smoke", group="ID", tier="ID", keep=1,
        asked="OWN (approved): identity arm on the R74 tree. eee_01 60 s fixed-Q, every R74 option at its default (voxel 0.425, line mode off, no switch): metrics within 0.005 mm of R72, digest compared with the R73 reference",
        purpose="OWN identity arm: the R74 tree at defaults reproduces the R73 baseline (the line-residual and switch code is off)")
    add("r74_own_smoke_v1p0_60s", B60, d(VOX(1.0, 2), F1DUMP, SNAP), batch="smoke", group="SM", tier="SMOKE", dur=60, keep=1, f417=1, plane_check=1,
        asked="OWN (approved): one 60 s smoke arm at voxel 1.0 before the 80-arm F-428 grid is trusted",
        purpose="OWN smoke: eee_01 60 s fixed-Q at root voxel 1.0 / max_layer 2, frame-1 dump, post-calibration snapshot")

    def ef_cells():
        for ns in EF_N:
            for ds in ("first", "off"):
                for pfn in (1, 3):
                    for q in ("fq", "mq"):
                        dl = {"calib.stationary.num_samples": ns, "imu.ds.mode": ds, "cbk.lidar.point_filter_num": pfn}
                        if ds == "off":
                            dl["imu.ds.ds_leaf_size"] = None
                        yield ns, ds, pfn, q, dl, int(math.ceil(ns / 197.5 + 8.0))
    # ---- F-428: 1 m root voxel, 40 cells x max_layer {2, 3}
    for ml in (2, 3):
        for ns, ds, pfn, q, dl, dur in ef_cells():
            add("r74_vox_v1p0_ml%d_n%d_ds%s_pfn%d_%s" % (ml, ns, ds, pfn, q), base_of("eee_01", q), d(VOX(1.0, ml), F1DUMP, SNAP, dl), batch="ef", group="VX", tier="EF", dur=dur, f417=1, plane_check=1,
                asked="F-428 (Bryce: 'try 1m voxels', the EF grid, max_layer 2 and 3): eee_01 EF cell calibration samples %d, ds %s, pfn %d, %s; root voxel 1.0, max_layer %d; %d s by the R72 formula ceil(ns/197.5+8)" % (ns, ds, pfn, q, ml, dur),
                purpose="F-428: voxel 1.0, max_layer %d, EF cell n%d ds %s pfn %d %s, frame-1 dump, post-calibration snapshot with plane test values" % (ml, ns, ds, pfn, q))
    add("r74_own_det_v1p0_ml2_n1600_dsoff_pfn3_fq", base_of("eee_01", "fq"), d(VOX(1.0, 2), F1DUMP, SNAP, {"calib.stationary.num_samples": 1600, "imu.ds.mode": "off", "cbk.lidar.point_filter_num": 3, "imu.ds.ds_leaf_size": None}),
        batch="smoke", group="DET", tier="EF", dur=int(math.ceil(1600 / 197.5 + 8.0)), f417=1, plane_check=1,
        asked="OWN (approved): determinism control, an exact repeat of r74_vox_v1p0_ml2_n1600_dsoff_pfn3_fq (state-trace digest must equal)",
        purpose="OWN determinism control: repeat of the F-428 cell n1600 ds off pfn 3 fq at voxel 1.0 / max_layer 2")
    # ---- F-430: three line variants x 40 cells x two voxel sizes (0.425 / 1.0, max_layer 2). Constants as accepted by Bryce (threshold = plane_threshold, rel_sigmas 3, smooth floor = sensor noise)
    for var in ("hardabs", "hardrel", "smooth"):
        for v in (0.425, 1.0):
            for ns, ds, pfn, q, dl, dur in ef_cells():
                add("r74_line_%s_v%s_n%d_ds%s_pfn%d_%s" % (var, tag(v), ns, ds, pfn, q), base_of("eee_01", q), d(VOX(v, 2), LINE[var], F1DUMP, dl), batch="ef", group="LN", tier="EF", dur=dur, f417=1,
                    asked="F-430 (Bryce: all-eigenvalue plane/line residual, three variants, the EF grid, both voxel sizes, constants 'Accept all of these'): variant %s, root voxel %s, max_layer 2, EF cell n%d ds %s pfn %d %s" % (var, v, ns, ds, pfn, q),
                    purpose="F-430: line residual variant %s at voxel %s / max_layer 2, EF cell n%d ds %s pfn %d %s, frame-1 dump" % (var, v, ns, ds, pfn, q))
    # ---- F-410: lazy coupled prior past the R72 abort (R72 aborted measurement k8 motion-Q at scan 1130)
    add("r74_f410_meas_k8_mq_full", "60_D3_coupled_measurement_gateon_knots8_iters5_motionq_full", d(CO_FULL, CO_REC, CO_LOG), batch="f410", group="CO", tier="CO", keep=1,
        asked="F-410 (Bryce: unit test plus ONE full-bag arm): coupled measurement, gate on, 8 knots, 5 iterations, motion-Q, full bag; R72 aborted this configuration at scan 1130",
        purpose="F-410: lazy IMU prior; full bag, the arm that aborted at scan 1130 in R72; update log and record frames as in the R73 co arms")
    # ---- F-426 part 2: switch arms (driver runs config A to onset-1 and config B from the onset)
    for bag, win, onset, tg in (("sbs_03", SBS_WIN, SBS_ONSET, "sbs03"), ("eee_02", EEE_WIN, EEE_ONSET, "eee02")):
        bfq, bmq = base_of(bag, "fq"), base_of(bag, "mq")
        mq2 = mqs(1.0, 5.0) if bag == "sbs_03" else {}
        acc, gyr = (1.0, 5.0) if bag == "sbs_03" else (0.3, 10.0)
        first, second = ("a", "b") if bag == "sbs_03" else ("d", "e")
        add("r74_sw_%s_%s" % (tg, first), bfq, switch(onset, "isotropic", acc, gyr), batch="sw", group="SW", tier="SW", dur=SW_DUR, keep=1,
            asked="F-426 part 2 (Bryce authorised; pairs a, b, d, e): %s fixed-Q until scan %d, motion-Q (acc %s, gyro %s) from scan %d; 70 s; window %d-%d" % (bag, onset - 1, acc, gyr, onset, win[0], win[1]),
            purpose="F-426 switch arm %s: %s fixed-Q -> motion-Q at scan %d" % (first, bag, onset))
        add("r74_sw_%s_%s" % (tg, second), bmq, d(mq2, switch(onset, "fixed")), batch="sw", group="SW", tier="SW", dur=SW_DUR, keep=1,
            asked="F-426 part 2 (Bryce authorised; pairs a, b, d, e): %s motion-Q (acc %s, gyro %s) until scan %d, fixed-Q from scan %d; 70 s; window %d-%d" % (bag, acc, gyr, onset - 1, onset, win[0], win[1]),
            purpose="F-426 switch arm %s: %s motion-Q -> fixed-Q at scan %d" % (second, bag, onset))
    # ---- refuse: F-427 removed keys
    for k in REMOVED_KEYS:
        add("r74_refuse_removed_%s" % k.split("process_noise.")[1].replace(".", "_"), BM if "motion" in k else B60, {k: REMOVED_VAL[k]}, batch="refuse", group="REF", tier="REF", dur=5,
            asked="F-427 scope (register text: the refuse arms that name the removed keys are rewritten to test the refusal): MUST FAIL at startup naming %s" % k,
            purpose="MUST FAIL: removed key %s is refused (F-427)" % k)
    # ---- F-418: the R73 qa lever grid minus the two deleted levers (local-polynomial energy source, WNOA), on all 9 bags, full bag, same-batch baselines
    for bag in BAGS:
        keep = 1 if bag == "eee_01" else 0
        BFq, BMq = base_of(bag, "fq"), base_of(bag, "mq")
        for q, b_ in (("fq", BFq), ("mq", BMq)):
            add("r74_qb_%s_%s" % (bag, q), b_, DIAG, batch="qa", group="QB", tier="QB", keep=keep,
                asked="F-418 (Bryce: same lever grid, all 9 bags, full bag, same-batch baselines): plain %s baseline of %s" % (q, bag), purpose="F-418 same-batch baseline, %s %s" % (bag, q))
        Q = lambda id_, base, delta, p: add("r74_qa_%s_%s" % (bag, id_), base, d(DIAG, delta), batch="qa", group="QA", tier="QA", keep=keep,
                                            asked="F-418 (Bryce: same lever grid minus the two deleted args, all 9 bags, full bag): " + p + " on " + bag, purpose="F-418: " + p + ", " + bag)
        for f_ in (0.8, 0.95):
            Q("innov_f%s_mq" % tag(f_), BMq, GAIN("innovation_scale", forgetting=f_), "innovation_scale on motion-Q, forgetting %s" % f_)
        Q("innov_fq", BFq, GAIN("innovation_scale"), "innovation_scale on fixed-Q")
        for hi in (1.3, 1.6):
            Q("nis_hi%s_mq" % tag(hi), BMq, GAIN("nis_gate", nis_hi=hi), "nis_gate on motion-Q, nis_hi %s" % hi)
        Q("nis_white_mq", BMq, GAIN("nis_gate", whiteness_max=0.3), "nis_gate on motion-Q with whiteness gate 0.3")
        Q("nis_fq", BFq, GAIN("nis_gate"), "nis_gate on fixed-Q")
        for nm, g_, a_, s_ in (("loose", 0.05, 0.3, 0.05), ("mid", 0.02, 0.15, 0.02), ("tight", 0.01, 0.08, 0.01)):
            Q("zupt_%s_mq" % nm, BMq, ZUPT("apply", g_, a_, 3, s_), "ZUPT apply on motion-Q, gyro %s rad/s acc %s m/s2 sigma_v %s m/s" % (g_, a_, s_))
        Q("zupt_mid_fq", BFq, ZUPT("apply"), "ZUPT apply on fixed-Q, mid thresholds")
        Q("zupt_detect_mq", BMq, ZUPT("detect_only"), "ZUPT detect_only on motion-Q")
    return A


def listener_spec(rnd, queues, out_root, min_free_gb, workers):
    return {
        "spec_version": 1, "round": rnd, "queues": queues,
        "trigger": {"sentinels": ["DONE", "FAILED"], "settle_seconds": 15, "settle_files": ["state_trace.csv"], "quiet_files": ["run.log"]},
        "on_failed_job": "skip",
        "path_maps": [["/root/catkin_ws", "/home/bblinn/fast_ws"]],
        "stages": [{
            "name": "reduce",
            # paths relative to src/livo_recon/ (see the R73 manifest note: --root stays fast_ws, the stage command finds its tools under src/livo_recon/)
            "cmd": ["python3", "src/livo_recon/tools/r69/r74_score_job.py", "--job-dir", "{job_dir}", "--out-dir", "{out_dir}", "--stage-out", "{stage_out}",
                    "--job-id", "{job_id}", "--spec", "{spec_dir}/%s", "--arm-table", "src/livo_recon/r74_manifest/arm_table.tsv",
                    "--params", "src/livo_recon/r69_inputs/bag_params.json", "--tools-dir", "src/livo_recon/tools/r69"],
            "timeout_s": 1800, "nice": 10,
            "outputs": [
                {"path": "summary_stage.json", "kind": "json", "min_bytes": 100, "merge_into_summary": True,
                 "required_fields": ["schema_version", "arm", "seq", "n_scans", "window_check_ok", "p_reduce_ok", "stage_ok", "archive_files"]},
                {"path": "arm_bundle.zip", "kind": "zip", "min_bytes": 2000},
                {"path": "archive/archive_index.json", "kind": "json", "min_bytes": 10},
                {"path": "archive/archive.sha256", "kind": "sha_manifest", "min_bytes": 10},
            ],
            "persist": ["archive"]}],
        "verify": [{"name": "archive_roundtrip",
                    "cmd": ["python3", "src/livo_recon/tools/r69/verify_archive.py", "--job-dir", "{job_dir}", "--out-dir", "{out_dir}",
                            "--archive-dir", "{stage_out_of:reduce}/archive", "--spec", "{spec_dir}/%s"],
                    "timeout_s": 1800, "nice": 10}],
        "summary": {"schema_version": 4, "required_fields": ["schema_version", "arm", "n_scans", "window_check_ok", "p_reduce_ok", "listener"]},
        "raw": {"delete": ["state_trace.csv", "lio_gt_matched.csv", "imu_cov_growth.csv", "frame_stats.txt", "plane_fit_trace.csv", "sensor_stamps.csv",
                           "point_cov_stats.csv", "surface_map_frames.csv", "frame_window.csv", "akf_q.csv", "coupled_scan_status.csv", "motion_q_scan.csv",
                           "post_calibration_voxelmap.csv", "lio_update_diag.csv", "live_point_cov.csv", "pose_control_first_frame_solve_matrices.txt",
                           "residual_geometry.csv", "info_scale.csv", "redundancy_renorm.csv",
                           "residual_dump.csv", "residual_dump_state.csv", "joint_knot_all_scans.csv", "joint_knot_scan_summary.csv",
                           "step_cap.csv", "q_adapt.csv", "zupt.csv", "coupled_failure_dump.txt", "engagement_imu.txt",
                           "scan_log.csv", "coupled_update_iter.csv", "coupled_update_scan.csv"] + ["coupled_record_f%d.txt" % i for i in range(1, 6)],   # all are in the lossless archive (BIG list of r74_score_job.py)
                "sha256": "twice"},
        "limits": {"workers": workers, "max_failures": 10, "max_attempts": 2, "min_free_gb": min_free_gb, "poll_interval_s": 10},
        "exit": {"when": "all_expected_handled"},
    }


# expected bytes per returned file (rule 5.15; estimates, basis in r74_file_size_ledger.json; the agent reports the measured sizes of one smoke arm and one full arm with size_ledger.py measure)
EXPECTED = {"residual_dump_all.csv.gz": 600000, "r74_plane_table.csv.gz": 300000, "r74_plane_check.json": 2000, "r73_f417_hth_frames.csv": 4000, "pos_trace_all.csv.gz": 104000,
            "r73_f417_residuals.csv.gz": 1300000, "coupled_update_iter_all.csv.gz": 6000000, "coupled_update_scan_all.csv.gz": 200000, "step_cap.csv.gz": 340000}
POS_COLS = ["scan_id", "t_abs", "t_rel_s", "px", "py", "pz", "yaw_rad"]
DUMP_COLS = ["frame", "iter", "r", "sigma_squared", "weight", "cls", "plane_key", "node_id", "tier", "hrot_z", "hpos_x"]


def required_outputs(table):
    """Follow-up -> arms -> file -> columns -> minimum rows -> where (return | archive) -> source. build_return.py checks every entry and names the follow-up in a failure."""
    ids = lambda grp=None, pre=None: [t["arm"] for t in table if (grp is None or t["group"] in grp) and (pre is None or t["arm"].startswith(pre))]
    R = []

    def add(fu, title, arms, file, cols, min_rows, where="return", source="arm", contains=None):
        R.append(dict(followup=fu, title=title, arms=arms, file=file, columns=cols, min_rows=min_rows, where=where, source=source, contains=contains or {},
                      expected_bytes=EXPECTED.get(file, 50000) * max(1, len(arms)) if where == "return" else 0))
    vx, ln = ids({"VX"}), ids({"LN"})
    # F-428
    add("F-428", "per-residual dump of frame 1 with weight and class (all VX arms; f417 on)", vx, "residual_dump_all.csv.gz", DUMP_COLS + ["row_kind"], 100)
    add("F-428", "plane test values table: every frame-1 residual joined to its voxel's test values (P1-P7 checks)", ids({"VX", "SM", "DET"}), "r74_plane_table.csv.gz", ["node_id", "layer", "edge_m", "point_count", "eig0", "eig1", "eig2", "n_resid"], 50)   # R75 (F-434b): the file is one row per matched voxel node, key node_id (R74 asked for frame, iter, plane_key: 82 checklist FAILs)
    add("F-428", "plane check result (violation counts per check P1..P7)", ids({"VX", "SM", "DET"}), "r74_plane_check.json", [], 1)
    add("F-428", "residual report of frame 1 (HtH / Htz metrics) for every VX arm", vx, "r73_f417_hth_frames.csv", ["frame", "iter", "hth_trace", "eig_min", "cond", "info_yaw"], 1)
    add("F-428", "per-scan posterior position and yaw (t_rel_s) of every VX arm", vx, "pos_trace_all.csv.gz", POS_COLS, 20)
    add("F-428", "post-calibration voxel map snapshot with the test-value columns (lossless host archive)", ids({"VX", "SM", "DET"}), "post_calibration_voxelmap.csv", [], 1, where="archive")
    # F-428 / F-430 addendum 1 (Bryce: the F-417 per-residual file for calibration samples N=200 only)
    add("F-428", "F-417 per-residual table of frame 1 (weight, class, information), only the VX arms with calibration samples 200", [a for a in vx if "_n200_" in a], "r73_f417_residuals.csv.gz", ["frame", "iter", "weight", "cls"], 100)
    add("F-430", "F-417 per-residual table of frame 1 (weight, class, information, row_kind rows), only the LN arms with calibration samples 200", [a for a in ln if "_n200_" in a], "r73_f417_residuals.csv.gz", ["frame", "iter", "weight", "cls"], 100)
    # F-430
    add("F-430", "per-residual dump of frame 1 with row_kind (0 plane, 1-2 hard line u0 u1, 3-5 smooth u0 u1 u2)", ln, "residual_dump_all.csv.gz", DUMP_COLS + ["row_kind"], 100)
    add("F-430", "residual report of frame 1 for every LN arm", ln, "r73_f417_hth_frames.csv", ["frame", "iter", "hth_trace", "eig_min", "cond", "info_yaw"], 1)
    add("F-430", "per-scan posterior position and yaw (t_rel_s) of every LN arm", ln, "pos_trace_all.csv.gz", POS_COLS, 20)
    # F-431 / F-432 are the smooth and the u1 rows of F-430 (same arms)
    # F-425
    add("F-425", "no F-417 / per-residual file in arms whose f417 column is 0 (gate in r74_gates.py)", [], "r74_gates.txt", [], 1, source="generated")
    # F-426
    sw = ids({"SW"})
    add("F-426", "per-scan posterior position and yaw (t_rel_s) of the 4 switch arms", sw, "pos_trace_all.csv.gz", POS_COLS, 500)
    add("F-426", "part 1: per-frame shadow-vs-driver step difference over the window with the onset marked (existing R73 shadow files, no new run)", [], "r74_f426_part1_frames.csv", ["pair", "frame", "step_diff_mm", "at_or_after_onset"], 200, source="generated")
    add("F-426", "part 1 summary (max / median before and from the onset)", [], "r74_f426_part1_summary.csv", ["pair", "onset", "max_before_mm", "max_from_onset_mm"], 4, source="generated")
    add("F-426", "part 2: per-frame alignment-free comparison of each switch arm with the two plain controls (dA, dB, dAB)", [], "r74_f426_part2_frames.csv", ["pair", "frame", "dA_mm", "dB_mm", "dAB_mm"], 200, source="generated")
    add("F-426", "part 2 summary", [], "r74_f426_part2_summary.csv", ["pair", "switch_arm", "onset", "dA_before_max_mm", "dB_from_onset_max_mm"], 4, source="generated")
    add("F-426", "engagement of the switch (process_noise_switch line) of the 4 arms", sw, "engagement_imu.txt", [], 1, where="archive")
    # F-418
    qa = ids({"QA", "QB"})
    add("F-418", "per-scan posterior position and yaw (t_rel_s) of every qa / qb arm (alignment-free comparison with the same-batch baseline)", qa, "pos_trace_all.csv.gz", POS_COLS, 500)
    # F-410
    add("F-410", "coupled update log per iteration of the full-bag measurement k8 motion-Q arm", ["r74_f410_meas_k8_mq_full"], "coupled_update_iter_all.csv.gz", ["frame", "iter", "A_cond"], 100)
    add("F-410", "coupled update log per scan", ["r74_f410_meas_k8_mq_full"], "coupled_update_scan_all.csv.gz", ["frame", "stop", "iterations"], 100)
    add("F-410", "per-scan posterior position and yaw", ["r74_f410_meas_k8_mq_full"], "pos_trace_all.csv.gz", POS_COLS, 500)
    add("F-410", "failure dump present only if the arm aborted (listed when written)", [], "r74_gates.txt", [], 1, source="generated")
    # F-406: step_cap.csv.gz of every arm with the step cap on (none in this round's arms: the entry follows the arm table, so a future arm that turns the cap on is checked)
    cap = [t["arm"] for t in table if "step_cap.csv" in t["requires"].split()]
    if cap:
        add("F-406", "step_cap.csv.gz with rot_applied, pos_applied, next_converged", cap, "step_cap.csv.gz", ["frame", "iter", "rot_raw", "pos_raw", "rot_applied", "pos_applied", "scale", "bound", "next_converged"], 1)
    # F-419
    add("F-419", "identity: the 0.425 identity arm and the determinism control", [], "identity_controls.txt", [], 1, source="generated")
    add("F-419", "symmetry gate output (extended gate, exemption table)", [], "r74_symmetrize_gate.txt", [], 1, source="agent")
    # F-409 / F-407
    add("F-409", "engagement of every dispatched arm concatenated", [], "engagement_all.txt", [], 50, source="generated")
    add("F-409", "per-arm completeness: required files present (arm_table 'requires' column)", [], "completeness.csv", ["arm", "requires_missing"], 100, source="generated")
    add("F-407", "engagement_imu.txt present for EVERY dispatched arm", [t["arm"] for t in table if t["group"] not in ("REF", "PROOF")], "engagement_imu.txt", [], 1, where="archive")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", default="r74_manifest")
    ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--min-free-gb", type=float, default=150.0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    J = {j["job_id"]: j for j in json.load(open(a.jobs))}
    os.makedirs(a.out, exist_ok=True)
    BATCHES = ["smoke", "refuse", "ef", "f410", "sw", "qa"]
    out = {b: [] for b in BATCHES}; table = []; seen = set()
    for r in arms(J):
        if r["base"] not in J:
            sys.exit("base job %s not in r66_jobs.json" % r["base"])
        assert r["id"] not in seen, r["id"]; seen.add(r["id"])
        j = copy.deepcopy(J[r["base"]]); j["job_id"] = r["id"]
        o = j["overrides"]
        for k, v in r["delta"].items():
            if v is None and k == "voxel_map.plane.plane_fit_pose_cov_mode":
                o.pop(k, None)
            else:
                o[k] = v   # None for imu.ds.ds_leaf_size: gen_jobs.py's null-deletion strips the key the base config sets (as R72 / R73)
        if o.get("estimator.mode") == "decoupled":
            o["estimator.mode"] = "iekf"
        if "calib.bootstrap.point_cov" in o and r["batch"] != "refuse":
            sys.exit("point_cov key forbidden outside the refuse batch")
        o.pop("spline.mode", None)
        for k in o:
            if "perturb" in k.lower():
                sys.exit("perturbation key forbidden: " + k)
        o["eval.state_trace_run_id"] = r["id"]
        path = "%s/r74_%s/%s" % (a.results_root, r["batch"], r["id"])
        o["outputs.path"] = path; o["outputs.debug_log_dir"] = path
        if r["dur"]:
            j["duration_secs"] = r["dur"]
        out[r["batch"]].append(j)
        sampling = "pfn%s_ds%s" % (o.get("cbk.lidar.point_filter_num", "def"), o.get("imu.ds.mode", "def"))
        table.append(dict(arm=r["id"], tier=r["tier"], arch="iekf_raw", kind="closed", duration_s=str(j.get("duration_secs", "full")), seq=j["seq"],
                          notes=r["purpose"], keep_raw=str(r["keep"]), dup_of="", r66_id=r["base"], group=r["group"], batch="r74_" + r["batch"],
                          sampling=sampling, requires=" ".join(requires_of(o, r["batch"])), f417=str(r["f417"]), plane_check=str(r["plane_check"]), asked=r["asked"]))
    # Listener failure-path proof (rule 5.13 item 6c): a COPY of the finished identity arm with state_reference.txt deleted (tools/r69/make_broken_clone.py). It is NEVER dispatched;
    # it needs an arm-table row so that the scorer fails only because the file is missing, not because the arm is unknown (R69 stage-1 defect F-324(6)).
    base_row = next(t for t in table if t["arm"] == "r74_own_id_baseline_60s")
    table.append(dict(arm="r74_00_id_broken_clone", tier="PROOF", arch="iekf_raw", kind="closed", duration_s="60", seq=base_row["seq"],
                      notes="listener failure-path proof: copy of r74_own_id_baseline_60s with state_reference.txt deleted (never dispatched)", keep_raw="0", dup_of="",
                      r66_id=base_row["r66_id"], group="PROOF", batch="r74_proof", sampling=base_row["sampling"], requires="", f417="0", plane_check="0",
                      asked="LISTENER PROOF (rule 5.13 item 6c; not an experiment arm, never dispatched)"))
    for b, lst in out.items():
        json.dump(lst, open(os.path.join(a.out, "jobs_%s.json" % b), "w"), indent=1)
    cols = ["arm", "tier", "arch", "kind", "duration_s", "seq", "notes", "keep_raw", "dup_of", "r66_id", "group", "batch", "sampling", "requires", "f417", "plane_check", "asked"]
    with open(os.path.join(a.out, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t"); w.writeheader(); w.writerows(table)
    q = lambda b, n: {"dir": "ablations/_job_queue/r74_%s" % b, "results_root": "%s/r74_%s" % (a.results_root, b), "expected_jobs": n}
    specs = {"smoke": ("R74-smoke", ["smoke", "refuse"]), "ef": ("R74-ef", ["ef"]), "f410": ("R74-f410", ["f410"]), "sw": ("R74-sw", ["sw"]), "qa": ("R74-qa", ["qa"])}
    for nm, (rn, bs) in specs.items():
        sp = listener_spec(rn, [q(b, len(out[b])) for b in bs], a.results_root, a.min_free_gb, a.workers)
        fn = "listener_spec_%s.json" % nm
        for sg in sp["stages"]:
            sg["cmd"] = [c.replace("%s", fn) for c in sg["cmd"]]
        for vf in sp["verify"]:
            vf["cmd"] = [c.replace("%s", fn) for c in vf["cmd"]]
        json.dump(sp, open(os.path.join(a.out, fn), "w"), indent=1)
    sp = listener_spec("R74-broken", [{"dir": "ablations/_job_queue/r74_smoke_broken", "results_root": "%s/r74_smoke_broken" % a.results_root, "expected_jobs": 1}],
                       a.results_root, a.min_free_gb, 1)
    for sg in sp["stages"] + sp["verify"]:
        sg["cmd"] = [c.replace("%s", "listener_spec_broken.json") for c in sg["cmd"]]
    json.dump(sp, open(os.path.join(a.out, "listener_spec_broken.json"), "w"), indent=1)
    # the same smoke spec without the refuse queue, for the case that Bryce denies the 8 F-427 refuse arms (expected_jobs then 3, not 11)
    sp = listener_spec("R74-smoke", [q("smoke", len(out["smoke"]))], a.results_root, a.min_free_gb, a.workers)
    for sg in sp["stages"] + sp["verify"]:
        sg["cmd"] = [c.replace("%s", "listener_spec_smoke_norefuse.json") for c in sg["cmd"]]
    json.dump(sp, open(os.path.join(a.out, "listener_spec_smoke_norefuse.json"), "w"), indent=1)
    # F-426 pairs: switch arm, the two R73 plain controls (config before the onset, config after), the R73 shadow arm of the same pair (part 1)
    pairs = [("a", "r74_sw_sbs03_a", "r73_sh_sbs03_ctl_fq", "r73_sh_sbs03_ctl_g5a1", "r73_sh_sbs03_a", SBS_WIN, SBS_ONSET),
             ("b", "r74_sw_sbs03_b", "r73_sh_sbs03_ctl_g5a1", "r73_sh_sbs03_ctl_fq", "r73_sh_sbs03_b", SBS_WIN, SBS_ONSET),
             ("d", "r74_sw_eee02_d", "r73_sh_eee02_ctl_fq", "r73_sh_eee02_ctl_mq", "r73_sh_eee02_d", EEE_WIN, EEE_ONSET),
             ("e", "r74_sw_eee02_e", "r73_sh_eee02_ctl_mq", "r73_sh_eee02_ctl_fq", "r73_sh_eee02_e", EEE_WIN, EEE_ONSET)]
    json.dump([dict(pair=p[0], kind="switch", switch_arm=p[1], control_before=p[2], control_after=p[3], shadow_arm_r73=p[4], window=list(p[5]), onset=p[6]) for p in pairs],
              open(os.path.join(a.out, "switch_pairs.json"), "w"), indent=1)
    json.dump(required_outputs(table), open(os.path.join(a.out, "required_outputs.json"), "w"), indent=1)
    with open(os.path.join(a.out, "scan_arms_priority.txt"), "w") as f:
        order = {"ID": 0, "DET": 1, "SW": 2, "CO": 3, "VX": 4, "LN": 5, "QB": 6, "QA": 7, "SM": 8}
        for t in sorted((t for t in table if t["group"] in order), key=lambda t: order[t["group"]]):
            f.write(t["arm"] + "\n")
    print("  ".join("%s %d" % (b, len(out[b])) for b in BATCHES), " total %d (+1 proof row, never dispatched)" % (len(table) - 1))


if __name__ == "__main__":
    main()
