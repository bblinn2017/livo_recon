#!/usr/bin/env python3
"""R63: write jobs.json (for gen_jobs.py) and arm_table.tsv.

    python3 tools/r63/make_manifest.py --repo . --out r63_manifest [--surface-keys surface_keys.json] [--batch r63_eee01] [--results-root <root>]
Dispatch ONLY through the queue (standing rule 5.6a):
    python3 scripts/parallel/gen_jobs.py --manifest r63_manifest/jobs.json --package livo_recon --batch r63_eee01
    scripts/parallel/dispatch_queue.sh ablations/_job_queue/r63_eee01 scripts/parallel/containers_12_CURRENT.txt
Every job is eee_01, 60 s unless the job_id says full_length. No common.* key is set. Nothing edits the repo.

Surface-backend arms (family M) need key names from the R60 code, which the planning agent does not hold. Write
surface_keys.json from YOUR tree (see R63_INSTRUCTIONS.md section 5) and pass it with --surface-keys; without it
family M is skipped and the script says so.
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

def q(model, beta, acc, gyr):
    return {"imu.process_noise.model": model, "imu.process_noise.motion.beta": float(beta),
            "imu.process_noise.motion.acc_scale": float(acc), "imu.process_noise.motion.gyro_scale": float(gyr)}
def ol(reset=5.0):
    return {"lio.open_loop.mode": "propagate_only", "lio.open_loop.reset_period_s": float(reset)}
def woodbury(rho):
    return {"lio.residual_redundancy.mode": "off"} if rho is None else \
           {"lio.residual_redundancy.mode": "woodbury", "lio.residual_redundancy.rho": float(rho)}
def tag(x):
    return str(x).replace(".", "")

# plane_fit x plane_var blocks (grid-design.md section 2); debiased x information_directional is refused at startup (P7).
VARS = {  # name -> (fit overrides, var mode, weight floor mode)
    "pca_eigengap": ({}, "eigengap", "sensor_range"),
    "pca_information": ({}, "information", "roughness"),
    "pca_information_directional": ({}, "information_directional", "roughness"),
    "debiased_eigengap": (DEBIASED, "eigengap", "sensor_range"),
    "debiased_information": (DEBIASED, "information", "roughness"),
}
def var_frag(name):
    fit, var, wf = VARS[name]
    o = dict(fit); o["voxel_map.plane.plane_var_mode"] = var; o["voxel_map.plane.weight_floor.mode"] = wf
    return o

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="r63_manifest")
    ap.add_argument("--surface-keys", default=None)
    ap.add_argument("--batch", default="r63_eee01")
    ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results",
                    help="root under which the queue writes results; outputs.path/debug_log_dir = <root>/<batch>/<job_id>")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frag = lambda rel: flat(_load(os.path.join(a.repo, rel)))
    cand = frag("config/experiments/p0_q/p0_candidate_fixed_q.yaml")        # calibration-derived P0, fixed Q
    cand.pop("estimator.coupled.gating_state_uncertainty", None)            # refused outside coupled mode

    jobs, rows = [], []
    def add(name, family, purpose, *frags, duration=60, keep=False, log=True):
        o = {}
        o.update(COMMON); o.update(L0)
        for f in frags:
            o.update(f)
        if not log:
            o["eval.imu_first_scans_en"] = False
        o["eval.state_trace_run_id"] = name
        # R62 landmine, folded in from the agent's fix: without these three every job is marked FAILED by run_job.sh.
        o["outputs.path"] = "%s/%s/%s" % (a.results_root, a.batch, name)
        o["outputs.debug_log_dir"] = o["outputs.path"]
        o["outputs.odom.export"] = True
        j = {"job_id": name, "dataset": "ntu_viral", "seq": "eee_01", "overrides": o, "use_cache": False}
        if duration and duration > 0:
            j["duration_secs"] = duration
        jobs.append(j)
        rows.append([name, family, duration if duration else "full", purpose, 1 if keep else 0])

    # ---- A: baselines, re-measured now that the zero head sample is gone (F-118); every old baseline number is superseded
    add("A_splineless_pca_configured_P0_fixed_Q_60s", "A", "reference: configured P0, fixed Q (R62 value with the old head: 0.0576 m seeded)")
    add("A_splineless_pca_configured_P0_fixed_Q_60s_first_scan_log_off", "A",
        "byte-identity control (S6): identical to the line above except eval.imu_first_scans_en=false", log=False)
    add("A_splineless_pca_configured_P0_fixed_Q_full_length", "A", "full bag, configured P0 (old 0.0286 m validated)", duration=0)
    add("A_splineless_pca_calibration_P0_fixed_Q_60s", "A", "reference for G, H, S: calibration-derived P0, fixed Q", cand, keep=True)
    add("A_splineless_pca_calibration_P0_fixed_Q_full_length", "A", "full bag, calibration-derived P0", cand, duration=0, keep=True)

    add("A_splineless_pca_calibration_P0_fixed_Q_60s_gravity_vector3_accel_excess_bias_explicit", "A",
        "identity control: the two new keys spelled out at their defaults; state_trace.csv must match the calibration-P0 baseline above (run_id column blanked)",
        cand, {"state.gravity_model": "vector3", "calib.stationary.accel_excess_model": "bias"})

    # ---- G: closed-loop gyro-scale grid (F-115/F-107): does a larger motion gyro Q help? cal-P0, acc scale 1
    for model in ("isotropic", "axis_aware"):
        for beta in (0.0, 0.9):
            for gyr in (1, 3, 10):
                add("G_closed_loop_calibration_P0_%s_Q_beta%s_gyro_scale%d_60s" % (model, tag(beta), gyr), "G",
                    "motion-dependent Q, gyro scale grid", cand, q(model, beta, 1.0, gyr),
                    keep=(model == "isotropic" and beta == 0.9 and gyr == 3))
    for gyr in (1, 3, 10):
        add("G_closed_loop_calibration_P0_isotropic_Q_beta09_gyro_scale%d_full_length" % gyr, "G",
            "full-length check of the beta 0.9 isotropic arms (60 s ATE is blind to stationary offsets, F-109)",
            cand, q("isotropic", 0.9, 1.0, gyr), duration=0)

    # ---- H: dense-point Gamma_L retest (F-117): PFN 1, downsampling off, cal-P0, plane-fit/variance block x Woodbury
    for vn in VARS:
        for rho in (None, 0.25, 0.5, 1.0):
            add("H_closed_loop_PFN1_ds_off_%s_%s_60s" % (vn, "independent" if rho is None else "woodbury_rho%s" % tag(rho)),
                "H", "dense-point case, plane block x redundancy correction", cand, PFN1_DSOFF, var_frag(vn), woodbury(rho),
                keep=(vn == "pca_eigengap" and rho is None))
    # contrast: does Woodbury do anything at the production density?
    for rho in (0.25, 0.5, 1.0):
        add("H_closed_loop_PFN3_ds_first_pca_eigengap_woodbury_rho%s_60s" % tag(rho), "H",
            "same correction at production density (PFN 3, downsample first)", cand, woodbury(rho))

    # ---- S: gravity state (F-116)
    S2 = {"state.gravity_model": "s2"}
    SC = {"calib.stationary.accel_excess_model": "scale"}
    add("S_closed_loop_vector3_gravity_accel_excess_as_scale_60s", "S", "FAST-LIO2/FAST-LIVO2 style: accel scaled by G/|mean|, ba stays 0", cand, SC)
    add("S_closed_loop_s2_gravity_accel_excess_as_bias_60s", "S", "|g| fixed; excess still absorbed by ba_z", cand, S2, keep=True)
    add("S_closed_loop_s2_gravity_accel_excess_as_scale_60s", "S", "|g| fixed and accel scaled (both upstream conventions)", cand, S2, SC)
    add("S_closed_loop_vector3_gravity_init_var_1e-2_60s", "S", "FAST-LIVO2 gravity init covariance (0.01)", cand, {"state.cov.gravity": 1e-2})
    add("S_closed_loop_s2_gravity_init_var_1e-4_60s", "S", "FAST-LIO2-like gravity init covariance (1e-4 per tangent axis)", cand, S2, {"state.cov.gravity": 1e-4})
    add("S_open_loop_5s_windows_calibration_P0_vector3_gravity_60s", "S", "open loop reference for the two below", cand, ol(5.0))
    add("S_open_loop_5s_windows_calibration_P0_s2_gravity_60s", "S", "open loop, |g| fixed", cand, ol(5.0), S2)
    add("S_open_loop_5s_windows_calibration_P0_vector3_gravity_accel_excess_as_scale_60s", "S", "open loop, accel scale", cand, ol(5.0), SC)
    add("S_closed_loop_s2_gravity_accel_excess_as_bias_full_length", "S", "full bag, s2", cand, S2, duration=0)
    add("S_closed_loop_s2_gravity_accel_excess_as_scale_full_length", "S", "full bag, s2 + accel scale", cand, S2, SC, duration=0)

    # ---- K: calibration window length (F-111)
    for n in (100, 400):
        add("K_closed_loop_calibration_P0_num_samples_%d_60s" % n, "K", "calib/stationary/num_samples sensitivity (default 200; R62 saw 100 used)",
            cand, {"calib.stationary.num_samples": n})

    # ---- M: surface backend (mergeable plane maps) x Woodbury; keys from the agent's tree
    if a.surface_keys and os.path.exists(a.surface_keys):
        sk = json.load(open(a.surface_keys))
        # Agent fix (coding agent, 2026-09-29): add()'s o.update(L0) unconditionally injects
        # voxel_map.plane.{plane_fit_mode,plane_gate_mode,weight_floor.mode} (L0's own pca/disc/
        # sensor_range pins) into EVERY job including family M -- these are override-injected
        # values, not base-config-file defaults, so gen_jobs.py's None-delete mechanism (which
        # only removes a key that exists in the base ntu_viral.yaml file itself) raises a KeyError
        # for them ("does not exist in the base config at that nesting"). The base-file keys
        # (voxel_size/max_layer/points.*/search.*) ARE real base-file defaults and delete cleanly
        # via None as sk["backend_overrides"] already sets them. Popping the three L0-only keys
        # out of the job's overrides dict entirely (after add() appends it) is the fix -- they
        # must be ABSENT, not "deleted from base" (there is no base default to delete).
        L0_ONLY_DEAD_UNDER_SURFACE = ("voxel_map.plane.plane_fit_mode", "voxel_map.plane.plane_gate_mode",
                                       "voxel_map.plane.weight_floor.mode")
        for cell in sk["cell_sizes"]:
            for bs in sk["bootstrap_values"]:
                for rho in (None, 0.25, 0.5, 1.0):
                    fr = dict(sk["backend_overrides"]); fr[sk["cell_size_key"]] = cell; fr[sk["bootstrap_key"]] = bs
                    add("M_closed_loop_PFN1_ds_off_surface_cell%s_bootstrap_%s_%s_60s" % (tag(cell), bs,
                        "independent" if rho is None else "woodbury_rho%s" % tag(rho)), "M",
                        "surface (mergeable plane) backend x redundancy correction", cand, PFN1_DSOFF, fr, woodbury(rho))
                    for k in L0_ONLY_DEAD_UNDER_SURFACE:
                        jobs[-1]["overrides"].pop(k, None)
    else:
        print("NOTE: --surface-keys not given: family M (surface backend) skipped", file=sys.stderr)

    # ---- R: startup-refusal checks. These jobs are EXPECTED to fail at startup; the refusal text is the deliverable.
    for nm, fr, why in (
        ("R_refusal_s2_gravity_with_coupled_estimator", {"estimator.mode": "coupled", "state.gravity_model": "s2"},
         "must be refused: coupled joint knots carry a 3-vector gravity"),
        ("R_refusal_s2_gravity_with_spline_mode_spline", {"spline.mode": "spline", "state.gravity_model": "s2"},
         "must be refused: spline paths read/write gravity as a 3-vector"),
        ("R_refusal_removed_first_scan_head_key", {"imu.first_scan_head": "seed_from_first_sample"},
         "must be refused: the key was removed in R63 (F-118)")):
        add(nm, "R", why, fr, duration=10)
        rows[-1][4] = 0
    # (the refusal jobs use the same COMMON/L0 base; spline.mode / estimator.mode above override the L0 pins deliberately)

    names = [j["job_id"] for j in jobs]
    assert len(set(names)) == len(names), "duplicate job_id"
    json.dump(jobs, open(os.path.join(a.out, "jobs.json"), "w"), indent=1)
    with open(os.path.join(a.out, "arm_table.tsv"), "w") as f:
        f.write("arm\ttier\tarch\tkind\tduration_s\tyaml\tnotes\tpriority\toptional\tkeep_raw\n")
        for (nm, fam, dur, purpose, keep), j in zip(rows, jobs):
            kind = "refusal" if nm.startswith("R_refusal") else ("open" if j["overrides"].get("lio.open_loop.mode") == "propagate_only" else "closed")
            f.write("\t".join(str(x) for x in [nm, fam, "dec_raw", kind, dur, "", purpose, 1, 0, keep]) + "\n")
    print("wrote %d jobs to %s/jobs.json" % (len(jobs), a.out))

if __name__ == "__main__":
    sys.exit(main())
