#!/usr/bin/env python3
"""R65: write jobs.json (for gen_jobs.py) and arm_table.tsv.

    python3 tools/r65/make_manifest.py --repo . --out r65_manifest --batch r65_eee01 --results-root <root> --surface-keys surface_keys.json
Dispatch ONLY through the queue (standing rule 5.6a):
    python3 scripts/parallel/gen_jobs.py --manifest r65_manifest/jobs.json --package livo_recon --batch r65_eee01
    scripts/parallel/dispatch_queue.sh ablations/_job_queue/r65_eee01 scripts/parallel/containers_12_CURRENT.txt
Every job is eee_01. Duration: full bag unless the job_id says 48s / 60s / 10s. No common.* key is set. Nothing edits the repo. Every job_id
is unique (asserted). Prints the job count per family. The arm_table.tsv column `keep_raw` = 1 marks arms whose GT-match and plane-fit-trace
compaction is returned (all arms get the standard per-scan compaction).
Perturbation repeats: keys cbk.lidar.perturb_std_m (1e-4 m) and cbk.lidar.perturb_seed; identical inputs give bit-identical runs, so a
'seedN' arm is the same configuration with a seeded sub-noise input perturbation.
"""
import argparse, json, os, sys
try:
    import yaml
    def _load(p):
        with open(p) as f:
            return yaml.safe_load(f) or {}
except ImportError:
    from ruamel.yaml import YAML
    def _load(p):
        with open(p) as f:
            return YAML(typ="safe").load(f) or {}

def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        key = pre + str(k)
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out

# Splineless L0 pca baseline (baseline-reference.md Part 1). use_bins deliberately unset (dead scope under debiased).
L0 = {
    "estimator.mode": "decoupled", "spline.mode": "raw_imu",
    "voxel_map.plane.plane_fit_mode": "pca", "voxel_map.plane.plane_var_mode": "eigengap",
    "voxel_map.plane.plane_gate_mode": "disc", "voxel_map.plane.plane_fit_pose_cov_mode": "sensor_only",
    "voxel_map.plane.weight_floor.mode": "sensor_range", "voxel_map.residual.pose_cov_in_sigma": False,
    "cbk.lidar.point_filter_num": 3, "imu.ds.mode": "first", "imu.ds.ds_leaf_size": 0.1,
}
COMMON = {"evo.enable": True, "vio.enable": False, "eval.state_trace_en": True, "eval.imu_first_scans_en": True}
PFN1_DSOFF = {"cbk.lidar.point_filter_num": 1, "imu.ds.mode": "off", "imu.ds.ds_leaf_size": None}   # None = delete key
DEBIASED = {"voxel_map.plane.plane_fit_mode": "debiased"}
INFO = {"voxel_map.plane.plane_var_mode": "information", "voxel_map.plane.weight_floor.mode": "roughness"}
INFODIR = {"voxel_map.plane.plane_var_mode": "information_directional", "voxel_map.plane.weight_floor.mode": "roughness"}
P = "voxel_map.plane."
REASONS = {P + "plane_fit_reasons_en": True}
TRACE60 = {P + "plane_fit_reasons_en": True, P + "plane_fit_trace_frames": 12, P + "plane_fit_trace_max_rows": 60000}
TRACE540 = {P + "plane_fit_reasons_en": True, P + "plane_fit_trace_frames": 540, P + "plane_fit_trace_max_rows": 300000}
CAPS_OFF = {"imu.process_noise.motion.acc_max_dynamic_variance": 1.0e6, "imu.process_noise.motion.gyro_max_dynamic_variance": 1.0e6}
STAMPS = {"data_queues.log_sensor_stamps": True}
PCSTATS = {"voxel_map.point_cov_stats_en": True}

def q(model, beta, acc, gyr):
    return {"imu.process_noise.model": model, "imu.process_noise.motion.beta": float(beta),
            "imu.process_noise.motion.acc_scale": float(acc), "imu.process_noise.motion.gyro_scale": float(gyr)}
def fixedq(acc_v, gyr_v):
    return {"imu.process_noise.fixed.acc_variance": float(acc_v), "imu.process_noise.fixed.gyro_variance": float(gyr_v)}
def ol(reset=5.0):
    return {"lio.open_loop.mode": "propagate_only", "lio.open_loop.reset_period_s": float(reset)}
def woodbury(rho):
    return {"lio.residual_redundancy.mode": "off"} if rho is None else \
           {"lio.residual_redundancy.mode": "woodbury", "lio.residual_redundancy.rho": float(rho)}
def hold(n):
    return {"lio.startup_hold.scans": int(n)}
def pert(seed):
    return {"cbk.lidar.perturb_std_m": 1.0e-4, "cbk.lidar.perturb_seed": int(seed)}
def pcm(mode, ins="posterior"):
    return {"lio.point_cov.mode": mode, "lio.point_cov.insertion_source": ins}
def boot(n, f):
    return {"lio.bootstrap_inflate.scans": int(n), "lio.bootstrap_inflate.factor": float(f)}
def tag(x):
    return ("%g" % x).replace(".", "p").replace("-", "m").replace("+", "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="r65_manifest")
    ap.add_argument("--surface-keys", default=None)
    ap.add_argument("--batch", default="r65_eee01")
    ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results",
                    help="root under which the queue writes results; outputs.path/debug_log_dir = <root>/<batch>/<job_id>")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frag = lambda rel: flat(_load(os.path.join(a.repo, rel)))
    cand = frag("config/experiments/p0_q/p0_candidate_fixed_q.yaml")        # calibration-derived P0, fixed Q
    cand.pop("estimator.coupled.gating_state_uncertainty", None)            # refused outside coupled mode
    conf = frag("config/experiments/p0_q/p0_configured_control.yaml")       # configured P0, fixed Q
    conf.pop("estimator.coupled.gating_state_uncertainty", None)
    L0_ONLY_DEAD_UNDER_SURFACE = ("voxel_map.plane.plane_fit_mode", "voxel_map.plane.plane_gate_mode", "voxel_map.plane.weight_floor.mode")
    if not (a.surface_keys and os.path.exists(a.surface_keys)):
        print("ERROR: --surface-keys is required (family SR)", file=sys.stderr); return 2
    sk = json.load(open(a.surface_keys))

    jobs, rows = [], []
    def add(name, family, purpose, *frags, duration=0, keep=False, surface=False, refusal=False):
        o = {}
        o.update(COMMON); o.update(L0)
        for f in frags:
            o.update(f)
        o["eval.state_trace_run_id"] = name
        o["outputs.path"] = "%s/%s/%s" % (a.results_root, a.batch, name)
        o["outputs.debug_log_dir"] = o["outputs.path"]
        o["outputs.odom.export"] = True
        if surface:
            for k in L0_ONLY_DEAD_UNDER_SURFACE:
                o.pop(k, None)
        j = {"job_id": name, "dataset": "ntu_viral", "seq": "eee_01", "overrides": o, "use_cache": False}
        if duration and duration > 0:
            j["duration_secs"] = duration
        jobs.append(j)
        kind = "refusal" if refusal else ("open" if o.get("lio.open_loop.mode") == "propagate_only" else "closed")
        rows.append([name, family, kind, duration if duration else "full", purpose, 1 if keep else 0])

    def surf(cell, rho):
        fr = dict(sk["backend_overrides"]); fr[sk["cell_size_key"]] = cell; fr[sk["bootstrap_key"]] = "sequential"
        fr.update(woodbury(rho))
        return fr

    # ===== A: baselines, identity controls, perturbation distribution (calibration-derived P0, fixed Q, splineless, PFN 3, pca)
    add("A_baseline_48s_rep1", "A", "48 s baseline", cand, duration=48)
    add("A_baseline_60s_rep1", "A", "60 s baseline", cand, duration=60)
    add("A_baseline_full_rep1", "A", "full-bag baseline (reference for every full-bag comparison in this batch)", cand, keep=True)
    add("A_baseline_full_rep2", "A", "identity: repeat of the full-bag baseline; must equal rep1 (deterministic)", cand)
    add("A_new_keys_spelled_out_at_defaults_60s", "A",
        "identity: every R65 key spelled out at its default (point_cov end_pose/posterior, bootstrap_inflate scans 0, perturb std 0 seed 0, stamps off, point_cov_stats off, debias guard not applicable under pca); state trace must equal A_baseline_60s_rep1",
        cand, pcm("end_pose", "posterior"), {"lio.bootstrap_inflate.scans": 0, "cbk.lidar.perturb_std_m": 0.0, "cbk.lidar.perturb_seed": 0,
                                              "data_queues.log_sensor_stamps": False, "voxel_map.point_cov_stats_en": False}, duration=60)
    add("A_diagnostics_on_60s", "A", "identity: stamps + point_cov_stats + plane-fit reasons on (all write-only); must equal A_baseline_60s_rep1",
        cand, STAMPS, PCSTATS, REASONS, duration=60)
    add("A_perturb_std0_seed7_60s", "A", "identity: perturb std 0 with a nonzero seed is off; must equal A_baseline_60s_rep1",
        cand, {"cbk.lidar.perturb_std_m": 0.0, "cbk.lidar.perturb_seed": 7}, duration=60)
    add("A_baseline_full_diagnostics_on", "A", "full-bag baseline with sensor_stamps.csv and point_cov_stats.csv on (write-only); must equal A_baseline_full_rep1; source of the F-164/F-173 stamp table",
        cand, STAMPS, PCSTATS, keep=True)
    for s in range(1, 6):
        add("A_baseline_full_perturb_seed%d" % s, "A", "full-bag baseline with a seeded 0.1 mm input perturbation (distribution of the baseline under equivalent inputs)", cand, pert(s))
    add("A_baseline_60s_perturb_seed1", "A", "60 s baseline with perturbation seed 1 (smoke check that the perturbation changes the run)", cand, pert(1), duration=60)

    # ===== D: debiased family after the F-158 fix, against pca, full bag (plane-fit trace frames 0..540 on the arms marked keep)
    def dd(name, purpose, *frags, diag=REASONS, keep=False, duration=0):
        add(name, "D", purpose, cand, diag, *frags, keep=keep, duration=duration)
    dd("D_pca_eigengap_full", "reference plane fit (pca, eigengap) with plane-fit trace to frame 540", diag=TRACE540, keep=True)
    dd("D_debiased_full_eigengap_full", "debiased (debias_mode full) after F-158: children now receive the stored history", DEBIASED, diag=TRACE540, keep=True)
    dd("D_debiased_lambda0_only_eigengap_full", "debiased, only lambda0 corrected", DEBIASED, {P + "debias_mode": "lambda0_only"}, keep=True)
    dd("D_debiased_none_eigengap_full", "debiased lifecycle, no noise correction", DEBIASED, {P + "debias_mode": "none"}, diag=TRACE540, keep=True)
    dd("D_debiased_full_eigengap_hysteresis2_full", "debiased full, plane revoked only above 2x threshold", DEBIASED, {P + "plane_threshold_hysteresis": 2.0}, keep=True)
    dd("D_debiased_full_eigengap_denominator_guard_off_full", "debiased full with the eigengap-denominator rejection off (key debias_denominator_guard=false)",
       DEBIASED, {P + "debias_denominator_guard": False}, keep=True)
    dd("D_debiased_full_eigengap_pose_combined_full", "debiased full with plane_fit_pose_cov_mode combined", DEBIASED, {P + "plane_fit_pose_cov_mode": "combined"}, keep=True)
    dd("D_pca_eigengap_pose_combined_full", "pca with plane_fit_pose_cov_mode combined (control for the line above)", {P + "plane_fit_pose_cov_mode": "combined"}, keep=True)
    dd("D_pca_eigengap_denominator_floor_on_full", "pca with the existing plane_var_denom_floor_en=true (control for the guard)", {P + "plane_var_denom_floor_en": True}, keep=True)
    for n in (2, 3):
        dd("D_debiased_full_eigengap_min_frames_to_init%d_full" % n, "debiased full, first fit waits for %d frames" % n, DEBIASED, {P + "min_frames_to_init": n})
    dd("D_pca_eigengap_min_frames_to_init2_full", "pca, first fit waits for 2 frames (control)", {P + "min_frames_to_init": 2})
    dd("D_pca_information_full", "pca x information variance (control)", INFO, keep=True)
    dd("D_debiased_full_information_full", "debiased full x information variance", DEBIASED, INFO, keep=True)
    dd("D_debiased_none_information_full", "debiased none x information variance", DEBIASED, INFO, {P + "debias_mode": "none"})
    dd("D_pca_eigengap_floor_eig0_full", "pca with sensor_noise_floor_eig0", {P + "sensor_noise_floor_eig0": True})
    dd("D_debiased_full_eigengap_floor_eig0_full", "debiased full with sensor_noise_floor_eig0", DEBIASED, {P + "sensor_noise_floor_eig0": True})
    add("D_debiased_full_eigengap_explicit_keys_full", "D", "identity: debias_mode=full, hysteresis 1, debias_denominator_guard=true spelled out; state trace must equal D_debiased_full_eigengap_full (which also has trace 540 on: diagnostics are write-only)",
        cand, REASONS, DEBIASED, {P + "debias_mode": "full", P + "plane_threshold_hysteresis": 1.0, P + "debias_denominator_guard": True}, keep=False)
    add("D_debiased_full_eigengap_60s", "D", "60 s smoke arm: debiased full", cand, TRACE60, DEBIASED, duration=60)
    add("D_debiased_full_eigengap_explicit_keys_60s", "D", "60 s smoke identity: explicit keys; must equal D_debiased_full_eigengap_60s", cand, TRACE60, DEBIASED,
        {P + "debias_mode": "full", P + "plane_threshold_hysteresis": 1.0, P + "debias_denominator_guard": True}, duration=60)

    # ===== PC: per-point pose covariance (F-169). gate always sees a pose covariance (prior); residual sigma only with pose_cov_in_sigma; the plane fit only with combined.
    SIG = {"voxel_map.residual.pose_cov_in_sigma": True}
    COMB = {P + "plane_fit_pose_cov_mode": "combined"}
    add("PC_gate_only_end_pose_full", "PC", "control: baseline gate covariance (end pose), statistics on", cand, PCSTATS, pcm("end_pose"), keep=True)
    add("PC_gate_only_propagated_full", "PC", "gate covariance with the propagated t_p-to-scan-end term; fit and sigma unchanged", cand, PCSTATS, pcm("propagated"), keep=True)
    for nm, fr in (("fit_combined", COMB), ("sigma_only", SIG), ("fit_combined_sigma", dict(COMB, **SIG))):
        for m, ins in (("end_pose", "posterior"), ("end_pose", "prior"), ("propagated", "posterior"), ("propagated", "prior")):
            if nm == "sigma_only" and ins == "prior":
                continue   # residual sigma reads the gate covariance, insertion covariance is unused with sensor_only fit
            add("PC_%s_%s_%s_full" % (nm, m, ins), "PC", "pose covariance in %s; point_cov mode %s, insertion source %s" % (nm, m, ins),
                cand, PCSTATS, fr, pcm(m, ins), keep=(m == "propagated"))
    for m in ("end_pose", "propagated"):
        add("PC_debiased_full_combined_%s_full" % m, "PC", "debiased full with combined fit covariance, point_cov mode %s" % m, cand, PCSTATS, REASONS, DEBIASED, COMB, pcm(m))
    add("PC_smoke_gate_propagated_60s", "PC", "60 s smoke: propagated gate covariance with statistics on", cand, PCSTATS, pcm("propagated"), duration=60)
    add("PC_smoke_fit_combined_propagated_prior_60s", "PC", "60 s smoke: combined fit, propagated, insertion prior", cand, PCSTATS, COMB, pcm("propagated", "prior"), duration=60)

    # ===== H: startup hold at full length (F-160) and onset (F-161)
    for n in (5, 7, 10, 15, 20):
        add("H_hold%d_full" % n, "H", "startup hold %d scans, fixed Q, calibration P0, full bag" % n, cand, hold(n), keep=(n == 10))
    for s in range(1, 5):
        add("H_hold10_full_perturb_seed%d" % s, "H", "startup hold 10 with perturbation seed %d" % s, cand, hold(10), pert(s))
    AXB = q("axis_aware", 0.9, 1.0, 10.0)
    add("H_axis_aware_beta09_gyro10_caps_off_no_hold_full", "H", "axis-aware motion Q (beta 0.9, acc 1, gyro 10), caps off, no hold (control)", cand, AXB, CAPS_OFF)
    for n in (10, 20):
        add("H_axis_aware_beta09_gyro10_caps_off_hold%d_full" % n, "H", "axis-aware motion Q with hold %d" % n, cand, AXB, CAPS_OFF, hold(n))
    add("H_configured_P0_baseline_full", "H", "configured (not calibration-derived) P0, full bag", conf)
    add("H_configured_P0_hold10_full", "H", "configured P0 with hold 10, full bag", conf, hold(10))
    add("H_hold10_48s", "H", "48 s smoke: hold 10", cand, hold(10), duration=48)
    for av, gv in ((1e-3, 1e-4), (1e-2, 1e-3), (1e-1, 1e-2)):
        add("O_fixed_acc_%s_gyro_%s_full" % (tag(av), tag(gv)), "O", "onset: small fixed Q, full bag", cand, fixedq(av, gv), keep=True)
        for s in (1, 2):
            add("O_fixed_acc_%s_gyro_%s_full_perturb_seed%d" % (tag(av), tag(gv), s), "O", "onset: same with perturbation seed %d" % s, cand, fixedq(av, gv), pert(s))

    # ===== G: what sets open-loop P (F-162): P0 velocity variance x P0 yaw variance x bias walk, 48 s, 5 s open-loop windows, production fixed Q
    for vv in (1e-6, 1e-4, 1e-2):
        for ry in (1e-4, 1e-2, 1e-1):
            for bw in (1e-7, 1e-5, 1e-3):
                add("G_open_loop_5s_p0_vel_%s_rotyaw_%s_biaswalk_%s_48s" % (tag(vv), tag(ry), tag(bw)), "G",
                    "open-loop P vs error: calib.p0.known_vel_variance %g, state.cov.rot_yaw %g, both bias walk rates %g" % (vv, ry, bw),
                    cand, ol(5.0), {"calib.p0.known_vel_variance": vv, "state.cov.rot_yaw": ry,
                                    "imu.process_noise.bias_random_walk.gyro_variance_rate": bw,
                                    "imu.process_noise.bias_random_walk.accel_variance_rate": bw}, duration=48)

    # ===== IT: iteration cap and early-exit thresholds (F-168)
    for it in (2, 3, 10):
        add("IT_max_iterations_%d_full" % it, "IT", "lio/ekf/max_iterations %d (baseline 5)" % it, cand, {"lio.ekf.max_iterations": it})
        add("IT_max_iterations_%d_60s" % it, "IT", "lio/ekf/max_iterations %d, 60 s" % it, cand, {"lio.ekf.max_iterations": it}, duration=60)
    for nm, fr in (("norm1e-3", {"lio.ekf.min_norm_dtheta": 1e-3, "lio.ekf.min_norm_dt": 1e-3}),
                   ("norm1e-4", {"lio.ekf.min_norm_dtheta": 1e-4, "lio.ekf.min_norm_dt": 1e-4}),
                   ("rel1e-3", {"lio.ekf.min_diff_error": 1e-3})):
        add("IT_max10_%s_full" % nm.replace("-", "m"), "IT", "max_iterations 10 with early exit %s" % nm, cand, {"lio.ekf.max_iterations": 10}, fr)
        add("IT_max10_%s_60s" % nm.replace("-", "m"), "IT", "max_iterations 10 with early exit %s, 60 s" % nm, cand, {"lio.ekf.max_iterations": 10}, fr, duration=60)

    # ===== SR: surface backend, sequential bootstrap, Woodbury rho scan (F-170). dense = PFN1 ds-off; default = PFN3 ds-first.
    RHOS = (None, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
    def rtag(r):
        return "independent" if r is None else "rho%s" % tag(r)
    for r in RHOS:
        add("SR_dense_cell0p25_%s_full" % rtag(r), "SR", "surface cell 0.25, dense points, sequential bootstrap, Woodbury %s" % rtag(r), cand, PFN1_DSOFF, surf(0.25, r), surface=True, keep=True)
        for s in (1, 2, 3):
            add("SR_dense_cell0p25_%s_full_perturb_seed%d" % (rtag(r), s), "SR", "same with perturbation seed %d" % s, cand, PFN1_DSOFF, surf(0.25, r), pert(s), surface=True)
    for r in (None, 0.15, 0.25, 0.35):
        add("SR_dense_cell0p10_%s_full" % rtag(r), "SR", "surface cell 0.10, dense points, sequential bootstrap, Woodbury %s" % rtag(r), cand, PFN1_DSOFF, surf(0.10, r), surface=True, keep=True)
        add("SR_default_density_cell0p25_%s_full" % rtag(r), "SR", "surface cell 0.25, default density (PFN 3, ds first), sequential bootstrap, Woodbury %s" % rtag(r), cand, surf(0.25, r), surface=True, keep=True)
    add("SR_smoke_dense_cell0p25_rho0p25_60s", "SR", "60 s smoke: surface backend loads and runs", cand, PFN1_DSOFF, surf(0.25, 0.25), duration=60, surface=True)

    # ===== WV: voxel-path Woodbury (dense points, information_directional and eigengap) with group statistics at the onset
    for rho in (None, 0.25, 0.5):
        add("WV_dense_pca_information_directional_%s_full" % rtag(rho), "WV", "dense points, information_directional variance, Woodbury %s" % rtag(rho), cand, PFN1_DSOFF, REASONS, INFODIR, woodbury(rho), keep=True)
    for s in (1, 2):
        add("WV_dense_pca_information_directional_rho0p25_full_perturb_seed%d" % s, "WV", "same with perturbation seed %d" % s, cand, PFN1_DSOFF, REASONS, INFODIR, woodbury(0.25), pert(s))
    for rho in (None, 0.25):
        add("WV_dense_pca_eigengap_%s_full" % rtag(rho), "WV", "dense points, eigengap variance, Woodbury %s" % rtag(rho), cand, PFN1_DSOFF, REASONS, woodbury(rho), keep=True)

    # ===== M: motion-Q coarse sweep on the full bag (F-145, F-121). caps: on = defaults 0.5/0.3, off = 1e6
    for model in ("isotropic", "axis_aware"):
        for acc in (0.3, 1.0):
            for gyr in (5, 10, 30, 60):
                for caps, cf in (("caps_on", {}), ("caps_off", CAPS_OFF)):
                    add("M_%s_beta09_acc%s_gyro%d_%s_full" % (model, tag(acc), gyr, caps), "M", "motion Q %s beta 0.9 acc scale %g gyro scale %d %s" % (model, acc, gyr, caps),
                        cand, q(model, 0.9, acc, gyr), cf, keep=(model == "axis_aware" and acc == 1.0 and gyr == 10 and caps == "caps_off"))
    for model in ("isotropic", "axis_aware"):
        for gyr in (10, 30):
            add("M_%s_beta09_acc0_gyro%d_caps_off_full" % (model, gyr), "M", "motion Q %s beta 0.9 acc scale 0 gyro scale %d caps off" % (model, gyr), cand, q(model, 0.9, 0.0, gyr), CAPS_OFF)
    for gyr in (10, 30):
        add("M_axis_aware_beta0_acc1_gyro%d_caps_off_full" % gyr, "M", "motion Q axis-aware beta 0 acc 1 gyro %d caps off" % gyr, cand, q("axis_aware", 0.0, 1.0, gyr), CAPS_OFF)

    # ===== S: accelerometer scale and gravity/bias covariance values, gravity model (F-171, F-122)
    SC = {"calib.stationary.accel_excess_model": "scale"}
    S2 = {"state.gravity_model": "s2"}
    for s in (0, 1, 2):
        add("S_vector3_accel_scale_full" + ("" if s == 0 else "_perturb_seed%d" % s), "S", "vector3 gravity, accelerometer scale (ba stays near 0)", cand, SC, *( [pert(s)] if s else []), keep=(s == 0))
    for s in range(0, 5):
        add("S_s2_accel_bias_full" + ("" if s == 0 else "_perturb_seed%d" % s), "S", "s2 gravity, excess in ba_z", cand, S2, *( [pert(s)] if s else []), keep=(s == 0))
        add("S_s2_accel_scale_full" + ("" if s == 0 else "_perturb_seed%d" % s), "S", "s2 gravity with accelerometer scale", cand, S2, SC, *( [pert(s)] if s else []), keep=(s == 0))
    for cg in (1e-7, 1e-3):
        add("S_vector3_accel_scale_gravity_cov_%s_full" % tag(cg), "S", "state.cov.gravity %g (default 1e-5); may be inert under calibration P0 (identical trace is a fact)" % cg, cand, SC, {"state.cov.gravity": cg})
        add("S_s2_accel_bias_gravity_cov_%s_full" % tag(cg), "S", "s2, state.cov.gravity %g" % cg, cand, S2, {"state.cov.gravity": cg})
        add("S_s2_accel_scale_gravity_cov_%s_full" % tag(cg), "S", "s2 + scale, state.cov.gravity %g" % cg, cand, S2, SC, {"state.cov.gravity": cg})
    for cb in (1e-8, 1e-4):
        add("S_vector3_accel_bias_ba_cov_%s_full" % tag(cb), "S", "vector3, bias model, state.cov.ba %g (default 1e-6); may be inert under calibration P0" % cb, cand, {"state.cov.ba": cb})
        add("S_s2_accel_bias_ba_cov_%s_full" % tag(cb), "S", "s2, bias model, state.cov.ba %g" % cb, cand, S2, {"state.cov.ba": cb})

    # ===== B: bootstrap measurement-noise inflation for the first N LIO scans (F-128, F-135), 48 s (excursion window) plus two full arms
    for n in (2, 4, 8):
        for f in (10.0, 100.0):
            add("B_inflate_first%d_scans_x%d_48s" % (n, f), "B", "sigma_squared x%g for LIO scans 1..%d, 48 s" % (f, n), cand, boot(n, f), duration=48)
            for s in (1, 2, 3):
                add("B_inflate_first%d_scans_x%d_48s_perturb_seed%d" % (n, f, s), "B", "same with perturbation seed %d" % s, cand, boot(n, f), pert(s), duration=48)
    add("B_baseline_48s_perturb_seed%d" % 1, "B", "48 s baseline, perturbation seed 1 (control for the arms above)", cand, pert(1), duration=48)
    add("B_baseline_48s_perturb_seed2", "B", "48 s baseline, perturbation seed 2", cand, pert(2), duration=48)
    add("B_baseline_48s_perturb_seed3", "B", "48 s baseline, perturbation seed 3", cand, pert(3), duration=48)
    add("B_inflate_first4_scans_x10_full", "B", "sigma_squared x10 for LIO scans 1..4, full bag", cand, boot(4, 10.0))
    add("B_inflate_first8_scans_x100_full", "B", "sigma_squared x100 for LIO scans 1..8, full bag", cand, boot(8, 100.0))

    # ===== RF: startup-refusal checks. EXPECTED to fail at startup; the refusal text is the deliverable.
    for nm, fr, why in (
        ("RF_perturb_negative", {"cbk.lidar.perturb_std_m": -1.0e-4}, "must be refused: cbk/lidar/perturb_std_m must be >= 0"),
        ("RF_point_cov_mode_unknown", {"lio.point_cov.mode": "bogus"}, "must be refused: not end_pose/propagated"),
        ("RF_point_cov_insertion_unknown", {"lio.point_cov.insertion_source": "bogus"}, "must be refused: not posterior/prior"),
        ("RF_bootstrap_inflate_factor_below_one", boot(4, 0.5), "must be refused: factor must be >= 1"),
        ("RF_bootstrap_inflate_scans_negative", {"lio.bootstrap_inflate.scans": -1}, "must be refused: scans must be >= 0"),
        ("RF_debias_guard_under_pca", {P + "debias_denominator_guard": False}, "must be refused: nested under plane_fit_mode=debiased")):
        add(nm, "RF", why, cand, fr, duration=10, refusal=True)

    names = [j["job_id"] for j in jobs]
    assert len(set(names)) == len(names), "duplicate job_id: %s" % sorted(n for n in names if names.count(n) > 1)
    json.dump(jobs, open(os.path.join(a.out, "jobs.json"), "w"), indent=1)
    with open(os.path.join(a.out, "arm_table.tsv"), "w") as f:
        f.write("arm\ttier\tarch\tkind\tduration_s\tyaml\tnotes\tpriority\toptional\tkeep_raw\n")
        for nm, fam, kind, dur, purpose, keep in rows:
            f.write("\t".join(str(x) for x in [nm, fam, "dec_raw", kind, dur, "", purpose, 1, 0, keep]) + "\n")
    from collections import Counter
    c = Counter(r[1] for r in rows)
    print("wrote %d jobs to %s/jobs.json; per family: %s" % (len(jobs), a.out, ", ".join("%s=%d" % kv for kv in sorted(c.items()))))
    return 0

if __name__ == "__main__":
    sys.exit(main())
