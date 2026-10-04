#!/usr/bin/env python3
"""R64: write jobs.json (for gen_jobs.py) and arm_table.tsv.

    python3 tools/r64/make_manifest.py --repo . --out r64_manifest --batch r64_eee01 --results-root <root> --surface-keys surface_keys.json
Dispatch ONLY through the queue (standing rule 5.6a):
    python3 scripts/parallel/gen_jobs.py --manifest r64_manifest/jobs.json --package livo_recon --batch r64_eee01
    scripts/parallel/dispatch_queue.sh ablations/_job_queue/r64_eee01 scripts/parallel/containers_12_CURRENT.txt
Every job is eee_01. Durations: 48 s (stationary study), 60 s, 50 s, 10 s (refusal) or full bag (job_id says full). No common.* key
is set. Nothing edits the repo. Every job_id is unique; the script asserts it and prints the job count per family.
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
TRACE = {P + "plane_fit_reasons_en": True, P + "plane_fit_trace_frames": 12, P + "plane_fit_trace_max_rows": 60000}
CAPS_OFF = {"imu.process_noise.motion.acc_max_dynamic_variance": 1.0e6, "imu.process_noise.motion.gyro_max_dynamic_variance": 1.0e6}
# every new R64 key spelled out at its default (identity control). debias_mode / hysteresis are nested under plane_fit_mode=debiased
# and are refused on a pca arm, so they are spelled out only on the debiased identity control below.
IDENT_NEW_KEYS_PCA = {"imu.process_noise.motion.acc_floor_scale": 1.0, "imu.process_noise.motion.gyro_floor_scale": 1.0,
                      "lio.startup_hold.scans": 0, P + "plane_fit_reasons_en": False, P + "plane_fit_trace_frames": 0,
                      P + "plane_fit_trace_max_rows": 60000}

def q(model, beta, acc, gyr):
    return {"imu.process_noise.model": model, "imu.process_noise.motion.beta": float(beta),
            "imu.process_noise.motion.acc_scale": float(acc), "imu.process_noise.motion.gyro_scale": float(gyr)}
def floors(acc_f, gyr_f):
    return {"imu.process_noise.motion.acc_floor_scale": float(acc_f), "imu.process_noise.motion.gyro_floor_scale": float(gyr_f)}
def fixedq(acc_v, gyr_v):
    return {"imu.process_noise.fixed.acc_variance": float(acc_v), "imu.process_noise.fixed.gyro_variance": float(gyr_v)}
def ol(reset=5.0):
    return {"lio.open_loop.mode": "propagate_only", "lio.open_loop.reset_period_s": float(reset)}
def woodbury(rho):
    return {"lio.residual_redundancy.mode": "off"} if rho is None else \
           {"lio.residual_redundancy.mode": "woodbury", "lio.residual_redundancy.rho": float(rho)}
def hold(n):
    return {"lio.startup_hold.scans": int(n)}
def tag(x):
    s = ("%g" % x).replace(".", "p").replace("-", "m").replace("+", "")
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="r64_manifest")
    ap.add_argument("--surface-keys", default=None)
    ap.add_argument("--batch", default="r64_eee01")
    ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results",
                    help="root under which the queue writes results; outputs.path/debug_log_dir = <root>/<batch>/<job_id>")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frag = lambda rel: flat(_load(os.path.join(a.repo, rel)))
    cand = frag("config/experiments/p0_q/p0_candidate_fixed_q.yaml")        # calibration-derived P0, fixed Q
    cand.pop("estimator.coupled.gating_state_uncertainty", None)            # refused outside coupled mode
    conf = frag("config/experiments/p0_q/p0_configured_control.yaml")       # configured P0 (not calibration-derived), fixed Q
    conf.pop("estimator.coupled.gating_state_uncertainty", None)
    L0_ONLY_DEAD_UNDER_SURFACE = ("voxel_map.plane.plane_fit_mode", "voxel_map.plane.plane_gate_mode", "voxel_map.plane.weight_floor.mode")

    jobs, rows = [], []
    def add(name, family, purpose, *frags, duration=60, keep=False, log=True, surface=False, refusal=False):
        o = {}
        o.update(COMMON); o.update(L0)
        for f in frags:
            o.update(f)
        if not log:
            o["eval.imu_first_scans_en"] = False
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

    def dur_tag(d):
        return "full" if not d else "%ds" % d

    # ================= A: baselines, repeats, identity controls (calibration-derived P0, fixed Q, splineless, PFN 3)
    for d in (48, 60, 0):
        for r in (1, 2, 3):
            add("A_baseline_%s_rep%d" % (dur_tag(d), r), "A", "baseline repeat %d: variance between runs of the SAME config (nondeterminism, host)" % r,
                cand, duration=d, keep=(d == 0 and r == 1))
    add("A_baseline_60s_new_keys_spelled_out_at_defaults", "A",
        "identity control: every new R64 key spelled out at its default; state_trace.csv must equal A_baseline_60s_rep1 with run_id blanked", cand, IDENT_NEW_KEYS_PCA)
    add("A_baseline_60s_diagnostics_on", "A",
        "identity control: plane-fit reasons + trace on; state_trace.csv must equal A_baseline_60s_rep1 with run_id blanked (diagnostics are write-only)", cand, TRACE)

    # ================= P: plane fit / debiased investigation (60 s, diagnostics on; then full-bag versions)
    def pl(name, purpose, *frags, duration=60, diag=TRACE, keep=False):
        add(name, "P", purpose, cand, diag, *frags, duration=duration, keep=keep)
    pl("P_pca_eigengap_60s", "reference plane fit (pca, eigengap variance)", keep=True)
    pl("P_debiased_full_eigengap_60s", "debiased, R63 arithmetic (debias_mode full), pose covariance NOT in the fit (sensor_only)", DEBIASED, keep=True)
    pl("P_debiased_full_eigengap_pose_combined_60s", "debiased full with plane_fit_pose_cov_mode=combined: pose covariance IS added to the fit accumulators and (shrunk) subtracted",
       DEBIASED, {P + "plane_fit_pose_cov_mode": "combined"})
    pl("P_pca_eigengap_pose_combined_60s", "pca with plane_fit_pose_cov_mode=combined (control for the line above)", {P + "plane_fit_pose_cov_mode": "combined"})
    pl("P_debiased_lambda0_only_eigengap_60s", "debiased: raw normal, only lambda0 corrected (first order)", DEBIASED, {P + "debias_mode": "lambda0_only"})
    pl("P_debiased_none_eigengap_60s", "debiased lifecycle with NO noise correction (raw scatter): isolates lifecycle from arithmetic", DEBIASED, {P + "debias_mode": "none"})
    for h in (2.0, 4.0):
        pl("P_debiased_full_eigengap_hysteresis%s_60s" % tag(h), "debiased full, an existing plane is revoked only above %gx plane_threshold" % h,
           DEBIASED, {P + "plane_threshold_hysteresis": h})
    pl("P_debiased_lambda0_only_eigengap_hysteresis2_60s", "lambda0_only + hysteresis 2", DEBIASED, {P + "debias_mode": "lambda0_only", P + "plane_threshold_hysteresis": 2.0})
    pl("P_debiased_none_eigengap_hysteresis2_60s", "none + hysteresis 2", DEBIASED, {P + "debias_mode": "none", P + "plane_threshold_hysteresis": 2.0})
    for n in (2, 3):
        pl("P_debiased_full_eigengap_min_frames_to_init%d_60s" % n, "debiased full, first fit waits for %d distinct frames" % n, DEBIASED, {P + "min_frames_to_init": n})
        pl("P_pca_eigengap_min_frames_to_init%d_60s" % n, "pca, first fit waits for %d distinct frames (control)" % n, {P + "min_frames_to_init": n})
    pl("P_pca_information_60s", "pca x information variance (control for the information rows)", INFO)
    pl("P_debiased_full_information_60s", "debiased full x information variance", DEBIASED, INFO)
    pl("P_debiased_lambda0_only_information_60s", "debiased lambda0_only x information variance", DEBIASED, INFO, {P + "debias_mode": "lambda0_only"})
    pl("P_debiased_none_information_60s", "debiased none x information variance", DEBIASED, INFO, {P + "debias_mode": "none"})
    pl("P_pca_eigengap_floor_eig0_60s", "pca with sensor_noise_floor_eig0 (existing key)", {P + "sensor_noise_floor_eig0": True})
    pl("P_debiased_full_eigengap_floor_eig0_60s", "debiased full with sensor_noise_floor_eig0", DEBIASED, {P + "sensor_noise_floor_eig0": True})
    add("P_debiased_full_eigengap_explicit_keys_diagnostics_off_60s", "P",
        "identity control: debias_mode=full, hysteresis=1.0 spelled out, diagnostics off; state_trace.csv must equal P_debiased_full_eigengap_60s with run_id blanked",
        cand, DEBIASED, {P + "debias_mode": "full", P + "plane_threshold_hysteresis": 1.0, P + "plane_fit_reasons_en": False, P + "plane_fit_trace_frames": 0})
    for nm, fr in (("pca_eigengap", {}), ("debiased_full_eigengap", DEBIASED), ("debiased_lambda0_only_eigengap", dict(DEBIASED, **{P + "debias_mode": "lambda0_only"})),
                   ("debiased_none_eigengap", dict(DEBIASED, **{P + "debias_mode": "none"})),
                   ("debiased_full_eigengap_hysteresis2", dict(DEBIASED, **{P + "plane_threshold_hysteresis": 2.0}))):
        add("P_%s_full" % nm, "P", "full bag: dynamic behaviour of the plane-fit variant", cand, REASONS, fr, duration=0, keep=(nm == "debiased_full_eigengap"))

    # ================= Q: process noise. (i) open loop stationary balance, (ii) closed loop stationary, (iii) full bag
    GRID = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0)
    AXQ = q("axis_aware", 0.0, 1.0, 1.0)
    for af in GRID:
        for gf in GRID:
            add("QO_open_loop_5s_axis_aware_acc_floor_%s_gyro_floor_%s_48s" % (tag(af), tag(gf)), "QO",
                "open-loop stationary balance: Q = floor_scale*floor + scale^2*excess, scales 1, caps off, floors acc x%g gyro x%g" % (af, gf),
                cand, ol(5.0), AXQ, CAPS_OFF, floors(af, gf), duration=48)
    add("QO_open_loop_5s_isotropic_floors_1_1_48s", "QO", "open-loop control: isotropic model, floors 1, scales 1, caps off", cand, ol(5.0), q("isotropic", 0.0, 1.0, 1.0), CAPS_OFF, duration=48)
    add("QO_open_loop_10s_axis_aware_floors_1_1_48s", "QO", "open-loop, 10 s windows (longer horizon), floors 1", cand, ol(10.0), AXQ, CAPS_OFF, duration=48)
    add("QO_open_loop_10s_axis_aware_floors_0p25_0p25_48s", "QO", "open-loop, 10 s windows, floors 0.25", cand, ol(10.0), AXQ, CAPS_OFF, floors(0.25, 0.25), duration=48)
    for which in ("gyro", "accel"):
        for rate in (1e-7, 1e-6, 1e-4):
            add("QO_open_loop_5s_axis_aware_floors_1_1_%s_bias_walk_rate_%s_48s" % (which, tag(rate)), "QO",
                "open-loop: %s bias random-walk variance rate %g (base 1e-5), everything else at the floors-1 arm" % (which, rate),
                cand, ol(5.0), AXQ, CAPS_OFF, {"imu.process_noise.bias_random_walk.%s_variance_rate" % which: rate}, duration=48)
    FAC = (1e-4, 1e-3, 1e-2, 1e-1, 0.5)
    FGY = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.3)
    for av in FAC:
        for gv in FGY:
            add("QO_open_loop_5s_fixed_acc_var_%s_gyro_var_%s_48s" % (tag(av), tag(gv)), "QO",
                "open-loop stationary balance of the PRODUCTION model (fixed Q): acc variance %g gyro variance %g (production 0.5 / 0.3)" % (av, gv),
                cand, ol(5.0), fixedq(av, gv), duration=48)
    CF = (0.25, 0.5, 1.0)
    for af in CF:
        for gf in CF:
            add("QC_closed_loop_axis_aware_acc_floor_%s_gyro_floor_%s_48s" % (tag(af), tag(gf)), "QC",
                "closed-loop stationary (48 s): axis_aware beta 0 scales 1 caps off, floors acc x%g gyro x%g" % (af, gf),
                cand, AXQ, CAPS_OFF, floors(af, gf), duration=48)
    for av, gv in ((1e-3, 1e-4), (1e-2, 1e-3), (1e-1, 1e-2)):
        add("QC_closed_loop_fixed_acc_var_%s_gyro_var_%s_48s" % (tag(av), tag(gv)), "QC", "closed-loop stationary (48 s), fixed Q", cand, fixedq(av, gv), duration=48)
        add("QF_fixed_acc_var_%s_gyro_var_%s_full" % (tag(av), tag(gv)), "QF", "full bag, fixed Q at reduced variances", cand, fixedq(av, gv), duration=0)
    for model, beta in (("isotropic", 0.9), ("axis_aware", 0.9)):
        for af in CF:
            for gf in CF:
                add("QF_%s_beta09_acc_scale_1_gyro_scale_10_acc_floor_%s_gyro_floor_%s_caps_off_full" % (model, tag(af), tag(gf)), "QF",
                    "full bag, floors acc x%g gyro x%g, motion scales acc 1 gyro 10, caps off" % (af, gf),
                    cand, q(model, beta, 1.0, 10.0), CAPS_OFF, floors(af, gf), duration=0)
    for acc in (1, 3, 10):
        for gyr in (3, 10, 30, 100):
            add("QF_isotropic_beta09_acc_scale_%d_gyro_scale_%d_caps_off_full" % (acc, gyr), "QF",
                "full bag motion-scale sweep, floors 1, caps off", cand, q("isotropic", 0.9, acc, gyr), CAPS_OFF, duration=0)
    add("QF_isotropic_beta09_acc_scale_1_gyro_scale_10_caps_on_full", "QF", "control for the caps: same as the caps-off arm, default caps 0.5/0.3", cand, q("isotropic", 0.9, 1.0, 10.0), duration=0)

    # ================= S: accelerometer scale (existing key calib.stationary.accel_excess_model) on the full bag with paired baselines
    SC = {"calib.stationary.accel_excess_model": "scale"}
    S2 = {"state.gravity_model": "s2"}
    for r in (1, 2, 3):
        add("S_vector3_accel_excess_as_scale_full_rep%d" % r, "S", "full bag: accel scaled by |g|/|mean stationary acc|, ba stays ~0 (repeat %d)" % r, cand, SC, duration=0)
    add("S_vector3_accel_excess_as_scale_48s", "S", "stationary 48 s with accel scale", cand, SC, duration=48)
    add("S_s2_accel_excess_as_bias_full", "S", "full bag, |g| fixed, excess in ba_z (paired with the next line in the same batch)", cand, S2, duration=0)
    add("S_s2_accel_excess_as_scale_full", "S", "full bag, |g| fixed and accel scaled", cand, S2, SC, duration=0)

    # ================= W: dense-point Woodbury, disturbance window 40-46 s, full bag
    def wd(name, purpose, *frags, duration=0, diag=REASONS):
        add(name, "W", purpose, cand, PFN1_DSOFF, diag, *frags, duration=duration)
    for rho in (None, 0.25, 0.5):
        wd("W_PFN1_ds_off_pca_information_directional_%s_full" % ("independent" if rho is None else "woodbury_rho%s" % tag(rho)),
           "dense points, information_directional variance, full bag", INFODIR, woodbury(rho))
    for rho in (None, 0.25, 0.5, 1.0):
        wd("W_PFN1_ds_off_pca_eigengap_%s_full" % ("independent" if rho is None else "woodbury_rho%s" % tag(rho)),
           "dense points, eigengap variance, full bag", woodbury(rho))
    for rho in (0.25, 0.5):
        for r in (1, 2, 3):
            wd("W_PFN1_ds_off_pca_information_directional_woodbury_rho%s_50s_rep%d" % (tag(rho), r),
               "repeatability of the scan 401-430 departure: 50 s, repeat %d" % r, INFODIR, woodbury(rho), duration=50)
    if a.surface_keys and os.path.exists(a.surface_keys):
        sk = json.load(open(a.surface_keys))
        for rho in (None, 0.25, 0.5):
            fr = dict(sk["backend_overrides"]); fr[sk["cell_size_key"]] = 0.25; fr[sk["bootstrap_key"]] = "sequential"
            add("W_PFN1_ds_off_surface_cell0p25_sequential_%s_full" % ("independent" if rho is None else "woodbury_rho%s" % tag(rho)), "W",
                "surface backend cell 0.25, sequential bootstrap (the one Woodbury cell that beat its baseline at 60 s), full bag",
                cand, PFN1_DSOFF, fr, woodbury(rho), duration=0, surface=True)
    else:
        print("NOTE: --surface-keys not given: the three surface arms in family W are skipped", file=sys.stderr)

    # ================= X: startup excursion (scans 1-10) and iteration count
    for n in (3, 10, 30):
        for r in (1, 2):
            add("X_startup_hold_%d_scans_48s_rep%d" % (n, r), "X", "LiDAR update withheld for the first %d scans (IMU propagation only), then released" % n, cand, hold(n), duration=48)
    add("X_configured_P0_baseline_48s", "X", "reference for the next line: configured (not calibration-derived) P0",
        conf, duration=48)
    add("X_configured_P0_startup_hold_10_scans_48s", "X", "configured P0, hold 10 scans", conf, hold(10), duration=48)
    for it in (2, 10):
        add("X_max_iterations_%d_60s" % it, "X", "lio/ekf/max_iterations %d (baseline 5): does the update converge in 5?" % it, cand, {"lio.ekf.max_iterations": it}, duration=60)
        add("X_max_iterations_%d_full" % it, "X", "lio/ekf/max_iterations %d, full bag" % it, cand, {"lio.ekf.max_iterations": it}, duration=0)

    # ================= R: startup-refusal checks. EXPECTED to fail at startup; the refusal text is the deliverable.
    for nm, fr, why in (
        ("R_refusal_floor_scale_under_fixed_model", floors(0.5, 1.0), "must be refused: floor_scale != 1 has no effect under process_noise/model=fixed"),
        ("R_refusal_debias_mode_under_pca", {P + "debias_mode": "none"}, "must be refused: debias_mode is nested under plane_fit_mode=debiased"),
        ("R_refusal_debias_mode_unknown_value", dict(DEBIASED, **{P + "debias_mode": "bogus"}), "must be refused: not one of full/lambda0_only/none"),
        ("R_refusal_startup_hold_negative", hold(-1), "must be refused: lio/startup_hold/scans must be >= 0")):
        add(nm, "R", why, cand, fr, duration=10, refusal=True)

    names = [j["job_id"] for j in jobs]
    assert len(set(names)) == len(names), "duplicate job_id"
    json.dump(jobs, open(os.path.join(a.out, "jobs.json"), "w"), indent=1)
    with open(os.path.join(a.out, "arm_table.tsv"), "w") as f:
        f.write("arm\ttier\tarch\tkind\tduration_s\tyaml\tnotes\tpriority\toptional\tkeep_raw\n")
        for nm, fam, kind, dur, purpose, keep in rows:
            f.write("\t".join(str(x) for x in [nm, fam, "dec_raw", kind, dur, "", purpose, 1, 0, keep]) + "\n")
    from collections import Counter
    c = Counter(r[1] for r in rows)
    print("wrote %d jobs to %s/jobs.json; per family: %s" % (len(jobs), a.out, ", ".join("%s=%d" % kv for kv in sorted(c.items()))))

if __name__ == "__main__":
    sys.exit(main())
