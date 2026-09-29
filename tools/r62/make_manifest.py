#!/usr/bin/env python3
"""R62: write the dispatch manifest (jobs.json) and arm_table.tsv for the R62 experiments.

Run from the repo root (any container that has PyYAML or ruamel):
    python3 tools/r62/make_manifest.py --repo . --out r62_manifest [--archived-config <path/to/dx1r_l0_eee_01/config.yaml>]
Then dispatch ONLY through the queue (standing rule 5.6a):
    python3 scripts/parallel/gen_jobs.py --manifest r62_manifest/jobs.json --package livo_recon --batch r62_eee01_60s
    scripts/parallel/dispatch_queue.sh <queue dir gen_jobs.py printed> scripts/parallel/containers_12_CURRENT.txt   (exact call: see R62_INSTRUCTIONS.md, step 4)
Every job is eee_01. Duration is 60 s unless the row says full length. job_id is the run's plain-language name.
Nothing here edits the repo; fragments under config/experiments/ are read and flattened into dotted overrides.
"""
import argparse, json, os, sys

try:
    import yaml
    def _load(p):
        with open(p) as f:
            return yaml.safe_load(f) or {}
except ImportError:                                   # ruamel fallback
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

# ---- the splineless (spline/enable false) L0 pca baseline, from baseline-reference.md Part 1 ----
# NOTE (R62 config-validity task): "splineless" is spelled here in the CURRENT nested form (estimator.mode decoupled,
# spline.mode raw_imu). Whether that form equals the archived dx1r_l0 behaviour is exactly what the agent must verify.
L0 = {
    "estimator.mode": "decoupled", "spline.mode": "raw_imu",
    "voxel_map.plane.plane_fit_mode": "pca", "voxel_map.plane.plane_var_mode": "eigengap",
    "voxel_map.plane.plane_gate_mode": "disc", "voxel_map.plane.plane_fit_pose_cov_mode": "sensor_only",
    "voxel_map.plane.weight_floor.mode": "sensor_range", "voxel_map.residual.pose_cov_in_sigma": False,
    "cbk.lidar.point_filter_num": 3, "imu.ds.mode": "first", "imu.ds.ds_leaf_size": 0.1,
}
# use_bins is deliberately NOT set: under plane_fit_mode debiased it is a dead-scope refusal (R61 L0D failure).
COMMON = {"evo.enable": True, "vio.enable": False, "eval.state_trace_en": True}

def q(model, beta, acc, gyr):
    return {"imu.process_noise.model": model, "imu.process_noise.motion.beta": float(beta),
            "imu.process_noise.motion.acc_scale": float(acc), "imu.process_noise.motion.gyro_scale": float(gyr)}
def ol(reset=5.0):
    return {"lio.open_loop.mode": "propagate_only", "lio.open_loop.reset_period_s": float(reset)}
BIAS = {
    "default_calibration_biases": {},
    "accel_bias_not_applied": {"calib.stationary.apply_to_state.accel_bias": False},
    "gyro_bias_not_applied": {"calib.stationary.apply_to_state.gyro_bias": False},
    "both_biases_not_applied": {"calib.stationary.apply_to_state.accel_bias": False,
                                "calib.stationary.apply_to_state.gyro_bias": False},
}
SEED = {"imu.first_scan_head": "seed_from_first_sample"}
PFN1_DSOFF = {"cbk.lidar.point_filter_num": 1, "imu.ds.mode": "off", "imu.ds.ds_leaf_size": None}   # None = delete the key
DEBIASED = {"voxel_map.plane.plane_fit_mode": "debiased"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="r62_manifest")
    ap.add_argument("--archived-config", default=None,
                    help="config.yaml of the archived dx1r_l0_eee_01 run; adds the verbatim-archive jobs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frag = lambda rel: flat(_load(os.path.join(a.repo, rel)))
    cand = frag("config/experiments/p0_q/p0_candidate_fixed_q.yaml")        # calibration-derived P0, fixed Q
    cand_fixedq = dict(cand)
    tight = frag("config/experiments/frame1_covariance/all_tight.yaml")     # P0 blocks made tiny: predicted P growth is then Q only
    # the candidate fragment also sets estimator.coupled.gating_state_uncertainty, which is refused outside coupled mode
    for d in (cand, cand_fixedq):
        d.pop("estimator.coupled.gating_state_uncertainty", None)

    jobs, rows = [], []
    def add(name, family, purpose, *frags, duration=60, log=True):
        o = {}
        o.update(COMMON); o.update(L0)
        for f in frags:
            o.update(f)
        o["eval.state_trace_run_id"] = name
        if log:
            o["eval.imu_first_scans_en"] = True
        j = {"job_id": name, "dataset": "ntu_viral", "seq": "eee_01", "overrides": o, "use_cache": False}
        if duration and duration > 0:
            j["duration_secs"] = duration
        jobs.append(j)
        rows.append([name, family, duration if duration else "full", purpose])

    # ---- A: baselines / controls ----
    add("A_splineless_pca_baseline_60s", "A", "reference for every closed-loop comparison", )
    add("A_splineless_pca_baseline_60s_no_new_logging", "A",
        "byte-identity control: same as the line above but eval.imu_first_scans_en unset; results_lio.txt, state_trace.csv must be identical", log=False)
    add("A_splineless_pca_baseline_full_length", "A", "full bag: is 0.0281 m reproduced here?", duration=0)
    if a.archived_config:
        arch = flat(_load(a.archived_config))
        for nm, dur in (("A_archived_dx1r_l0_config_verbatim_60s", 60), ("A_archived_dx1r_l0_config_verbatim_full_length", 0)):
            o = dict(COMMON); o.update(arch); o["eval.state_trace_run_id"] = nm; o["eval.imu_first_scans_en"] = True
            j = {"job_id": nm, "dataset": "ntu_viral", "seq": "eee_01", "overrides": o, "use_cache": False}
            if dur:
                j["duration_secs"] = dur
            jobs.append(j); rows.append([nm, "A", dur or "full", "archived config flattened to dotted keys, no L0 pins added"])

    # ---- B: bias ablation (calibration biases not applied to the state) ----
    for k, f in BIAS.items():
        if k == "default_calibration_biases":
            continue
        add("B_closed_loop_splineless_%s_60s" % k, "B", "closed loop, configured P0, fixed Q", f)
    for k, f in BIAS.items():
        add("B_open_loop_5s_windows_configured_P0_fixed_Q_%s" % k, "B",
            "open loop (propagation only, 5 s windows), configured P0, fixed Q", ol(5.0), f)

    # ---- C: first-scan head sample ----
    add("C_closed_loop_splineless_first_scan_head_seeded_60s", "C", "removes the zero head-sample defect (scan 1)", SEED)
    add("C_open_loop_5s_windows_first_scan_head_seeded", "C", "as the B open-loop default-biases arm but head seeded", ol(5.0), SEED)
    add("C_open_loop_5s_windows_first_scan_head_seeded_both_biases_not_applied", "C", "head seeded and biases not applied",
        ol(5.0), SEED, BIAS["both_biases_not_applied"])

    # ---- D: open-loop Q with P0 made tiny (P0 removed as a source of predicted sd) ----
    add("D_open_loop_5s_windows_tiny_P0_fixed_Q", "D", "predicted sd from fixed Q alone", ol(5.0), tight)
    add("D_open_loop_full_run_tiny_P0_fixed_Q", "D", "same, no window reset (60 s of propagation)", ol(0.0), tight)
    for model in ("isotropic", "axis_aware"):
        for beta in (0.0, 0.9):
            add("D_open_loop_5s_windows_tiny_P0_%s_Q_beta%s" % (model, str(beta).replace(".", "")), "D",
                "motion-dependent Q, engagement cell", ol(5.0), tight, q(model, beta, 1.0, 1.0))
    for acc in (0.3, 1.0):
        for gyr in (20, 60):
            add("D_open_loop_5s_windows_tiny_P0_isotropic_Q_beta09_acc%s_gyro%d" % (str(acc).replace(".", ""), gyr), "D",
                "R45 grid corner", ol(5.0), tight, q("isotropic", 0.9, acc, gyr))
    add("D_open_loop_5s_windows_tiny_P0_fixed_Q_first_scan_head_seeded", "D", "tiny-P0 fixed-Q with head seeded", ol(5.0), tight, SEED)

    # ---- E: voxel-map type and PFN / downsample ablation (closed loop) ----
    for mp, mf in (("pca_map", {}), ("debiased_map", DEBIASED)):
        for gr, gf in (("PFN3_ds_first", {}), ("PFN1_ds_off", PFN1_DSOFF)):
            for p0n, p0f in (("configured_P0", {}), ("calibration_derived_P0", cand)):
                if mp == "pca_map" and gr == "PFN3_ds_first" and p0n == "configured_P0":
                    continue                                   # = A_splineless_pca_baseline_60s
                add("E_closed_loop_splineless_%s_%s_%s_60s" % (mp, gr, p0n), "E", "map type x point density x P0", mf, gf, p0f)

    # ---- F: closed-loop splineless P0 and Q ----
    add("F_closed_loop_splineless_calibration_derived_P0_fixed_Q_60s", "F", "P0 from calibration, Q fixed", cand)
    add("F_closed_loop_splineless_calibration_derived_P0_isotropic_Q_beta09_60s", "F", "P0 from calibration, isotropic motion Q", cand, q("isotropic", 0.9, 1.0, 1.0))
    add("F_closed_loop_splineless_calibration_derived_P0_axis_aware_Q_beta09_60s", "F", "P0 from calibration, axis-aware motion Q", cand, q("axis_aware", 0.9, 1.0, 1.0))
    add("F_closed_loop_splineless_calibration_derived_P0_isotropic_Q_beta0_60s", "F", "beta 0 control for the above", cand, q("isotropic", 0.0, 1.0, 1.0))
    add("F_closed_loop_splineless_calibration_derived_P0_fixed_Q_first_scan_head_seeded_60s", "F", "calibration P0 + head seeded", cand, SEED)
    add("F_closed_loop_splineless_calibration_derived_P0_fixed_Q_both_biases_not_applied_60s", "F", "calibration P0 + biases not applied", cand, BIAS["both_biases_not_applied"])

    names = [j["job_id"] for j in jobs]
    assert len(set(names)) == len(names), "duplicate job_id"
    json.dump(jobs, open(os.path.join(a.out, "jobs.json"), "w"), indent=1)
    with open(os.path.join(a.out, "arm_table.tsv"), "w") as f:
        # columns kept compatible with the R61 reduction scripts (tier/arch/kind/yaml/notes/priority/optional)
        f.write("arm\ttier\tarch\tkind\tduration_s\tyaml\tnotes\tpriority\toptional\n")
        for (nm, fam, dur, purpose), j in zip(rows, jobs):
            kind = "open" if j["overrides"].get("lio.open_loop.mode") == "propagate_only" else "closed"
            f.write("\t".join(str(x) for x in [nm, fam, "dec_raw", kind, dur, "", purpose, 1, 0]) + "\n")
    print("wrote %d jobs to %s/jobs.json" % (len(jobs), a.out))

if __name__ == "__main__":
    sys.exit(main())
