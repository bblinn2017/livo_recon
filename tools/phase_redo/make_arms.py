#!/usr/bin/env python3
"""R61: generate the arm overlays and arm_table.tsv for the phase 1/2/3 redo.

Usage (repo root):  python3 tools/phase_redo/make_arms.py --repo . --out arms_r61
Each arm gets ONE yaml (arms_r61/<arm>.yaml) holding only the deltas from config/ntu_viral.yaml; load it after
config/ntu_viral.yaml exactly like the earlier experiment fragments. The generator deep-merges the existing repo
fragments (config/experiments/**, coupled_joint_knots*.yaml, spline_refine_overrides.yaml), so their content is not
re-typed here. It never edits the repo.
"""
import argparse, copy, itertools, os, sys
import yaml

ARCHS = ["cpl_meas", "cpl_tail", "dec_raw", "dec_spline"]

def load(repo, rel):
    with open(os.path.join(repo, rel)) as f:
        return yaml.safe_load(f) or {}

def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def setp(d, dotted, value):
    cur = d
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value
    return d

def arch_overlay(repo, arch, test_id):
    if arch in ("cpl_meas", "cpl_tail"):
        o = load(repo, "config/coupled_joint_knots.yaml")
        setp(o, "estimator.coupled.residual_evaluation_time", "measurement" if arch == "cpl_meas" else "tail")
        setp(o, "estimator.coupled.test_id", test_id)
        return o
    o = {"estimator": {"mode": "decoupled"}}
    if arch == "dec_raw":
        setp(o, "spline.mode", "raw_imu")
    else:
        o = merge(o, load(repo, "config/spline_refine_overrides.yaml"))
    return o

def strip_for_arch(o, arch):
    # estimator.coupled is refused outside coupled mode (refuseUnclaimed), so fragments that carry it are stripped.
    if arch.startswith("dec_"):
        est = o.get("estimator", {})
        est.pop("coupled", None)
    return o

def common(duration, run_id):
    o = {"common": {"offline_duration_secs": float(duration)}, "evo": {"enable": True},
         "eval": {"state_trace_en": True, "state_trace_run_id": run_id}, "vio": {"enable": False}}
    return o

FRAG = {
    "p0_candidate": "config/experiments/p0_q/p0_candidate_fixed_q.yaml",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="arms_r61")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []

    def emit(name, tier, arch, kind, duration, factors, notes="", priority=5, optional=False):
        o = common(duration, name)
        o = merge(o, arch_overlay(a.repo, arch, name))
        for f in factors:
            o = merge(o, f)
        o = strip_for_arch(o, arch)
        # keep test_id equal to the arm name for coupled arms (outputs are keyed by it)
        if arch.startswith("cpl_"):
            setp(o, "estimator.coupled.test_id", name)
        path = os.path.join(a.out, name + ".yaml")
        with open(path, "w") as f:
            yaml.safe_dump(o, f, sort_keys=False)
        rows.append([name, tier, arch, kind, duration, path, notes, priority, int(optional)])

    cand = load(a.repo, FRAG["p0_candidate"])
    def cand_f(): return cand

    # ---------------- Tier B: baselines ----------------
    pins = {}
    for path, val in [("voxel_map.plane.plane_var_mode", "eigengap"), ("voxel_map.plane.plane_gate_mode", "disc"),
                      ("voxel_map.plane.plane_fit_pose_cov_mode", "sensor_only"),
                      ("voxel_map.plane.use_bins", False), ("voxel_map.plane.weight_floor.mode", "sensor_range"),
                      ("voxel_map.residual.pose_cov_in_sigma", False), ("cbk.lidar.point_filter_num", 3),
                      ("imu.ds.mode", "first"), ("imu.ds.ds_leaf_size", 0.1)]:
        setp(pins, path, val)
    for nm, fit in [("L0", "pca"), ("L0D", "debiased")]:
        f = merge(pins, {}); setp(f, "voxel_map.plane.plane_fit_mode", fit)
        emit("B_%s_full" % nm, "B", "dec_raw", "baseline_full", -1, [f],
             "recorded baseline reproduction (0.0281 m L0, ~0.0312 m L0D, harness ATE); dec_raw + pins from baseline-reference.md", 1)
        emit("B_%s_60s" % nm, "B", "dec_raw", "baseline", 60, [f], "same config, 60 s", 1)
    # L5: best-effort mapping of the archived spline config into the current nested mode form. NOT a verified pin
    # (weight_floor/mode unknown per baseline-reference.md). Optional: report refused keys and skip if refused.
    for bam in ["none", "exact"]:
        f = merge(pins, {})
        setp(f, "voxel_map.plane.plane_fit_mode", "pca")
        setp(f, "voxel_map.plane.plane_var_mode", "information_directional")
        setp(f, "voxel_map.plane.weight_floor.mode", "roughness")
        setp(f, "spline.boundary_anchor_mode", bam)
        for dur, suf in [(-1, "full"), (60, "60s")]:
            emit("B_L5_%s_%s" % (bam, suf), "B", "dec_spline", "baseline_full" if dur < 0 else "baseline", dur, [f],
                 "OPTIONAL best-effort L5 (weight_floor/mode UNKNOWN, roughness guessed; spline keys mapped to the current nested form). Report refused keys; a refusal is a finding, not a failure to fix.", 3, True)
    # architecture baselines for phases 1-3: unmodified ntu_viral.yaml + architecture overlay, configured P0, fixed Q, independent residuals
    for arch in ARCHS:
        emit("base_%s" % arch, "B", arch, "closed", 60, [], "unmodified ntu_viral.yaml (code default plane_fit_pose_cov_mode=combined) + architecture overlay", 1)
        so = {"voxel_map": {"plane": {"plane_fit_pose_cov_mode": "sensor_only"}}}
        emit("base_%s_so" % arch, "B", arch, "closed", 60, [so], "as base_<arch> with plane_fit_pose_cov_mode=sensor_only (isolates that key vs the recorded L0 pins)", 2)

    # ---------------- Tier 1: Phase 1 (initial covariance), closed loop, all four architectures ----------------
    frames = {}
    fdir = os.path.join(a.repo, "config/experiments/frame1_covariance")
    for fn in sorted(os.listdir(fdir)):
        if fn.endswith(".yaml"):
            frames["f1_" + fn[:-5]] = load(a.repo, "config/experiments/frame1_covariance/" + fn)
    sdir = os.path.join(a.repo, "config/experiments/stationary_initialization_combinations")
    for fn in sorted(os.listdir(sdir)):
        if fn.endswith(".yaml"):
            frames["si_" + fn[:-5]] = load(a.repo, "config/experiments/stationary_initialization_combinations/" + fn)
    zero = merge(cand, {"calib": {"p0": {"tilt_ba_ambiguity_accel_std": 0.0}}})
    p1 = {"p0_candidate": cand, "p0_zero_ambiguity": zero}
    p1.update(frames)
    for arch in ARCHS:
        pr = {"cpl_meas": 2, "cpl_tail": 4, "dec_raw": 4, "dec_spline": 4}[arch]
        for nm, frag in p1.items():
            emit("p1_%s_%s" % (arch, nm), "1", arch, "closed", 60, [frag], "phase 1", pr)

    # ---------------- Tier 2: Phase 2 (motion Q) ----------------
    def qfrag(model, beta, acc, gyr):
        return {"imu": {"process_noise": {"model": model, "motion": {"beta": float(beta), "acc_scale": float(acc), "gyro_scale": float(gyr)}}}}
    grid = list(itertools.product([0.0, 0.3, 0.6, 1.0], [20, 30, 45, 60], [0.0, 0.5, 0.9]))
    for arch in ARCHS:
        pr = {"cpl_meas": 3, "cpl_tail": 6, "dec_raw": 6, "dec_spline": 6}[arch]
        # (the candidate-P0 + fixed-Q control IS arm p1_<arch>_p0_candidate; not duplicated here)
        for model in ["isotropic", "axis_aware"]:
            for beta in [0.0, 0.9]:
                emit("p2_%s_%s_b%s_a1_g1" % (arch, model, str(beta).replace(".", "")), "2", arch, "closed", 60,
                     [cand, qfrag(model, beta, 1.0, 1.0)], "R44 engagement cell", pr)
        for (acc, gyr, beta) in grid:
            emit("p2_%s_iso_b%s_a%s_g%d" % (arch, str(beta).replace(".", ""), str(acc).replace(".", ""), gyr), "2", arch, "closed", 60,
                 [cand, qfrag("isotropic", beta, acc, gyr)], "R45 grid", pr + 1)

    # ---------------- Tier 3: Phase 3 (Gamma_L) ----------------
    def rr(rho): return {"lio": {"residual_redundancy": {"mode": "woodbury", "rho": rho}}}
    for arch in ["cpl_meas", "cpl_tail"]:
        for gate in [False, True]:
            g = {"estimator": {"coupled": {"gating_state_uncertainty": gate}}}
            gs = "gs" if gate else "gi"
            emit("p3_%s_indep_%s" % (arch, gs), "3", arch, "closed", 60, [g], "independent residual covariance", 2)
            for rho in [0.25, 0.5, 1.0]:
                emit("p3_%s_w%03d_%s" % (arch, int(rho * 100), gs), "3", arch, "closed", 60, [g, rr(rho)], "matched-plane Woodbury rho=%s" % rho, 2)
    for arch in ["dec_raw", "dec_spline"]:
        emit("p3_%s_indep" % arch, "3", arch, "closed", 60, [], "independent control (Gamma_L port does not apply to this state layout)", 4)
        for rho in [0.25, 0.5, 1.0]:
            emit("p3_%s_w%03d" % (arch, int(rho * 100)), "3", arch, "closed", 60, [rr(rho)],
                 "lio.residual_redundancy is a shared key: allowed, but NOT the joint-knot Gamma_L port; label accordingly", 6)

    # ---------------- Tier O: open-loop IMU propagation (phases 1/2 accumulated drift) ----------------
    # State error does not depend on Q, only P does, so one arch (dec_raw, cheapest) carries the grid.
    def ol(reset): return {"lio": {"open_loop": {"mode": "propagate_only", "reset_period_s": float(reset)}}}
    for reset in [5.0, 0.0]:
        rs = "r%d" % int(reset)
        emit("ol_dec_raw_cfgP0_fixed_%s" % rs, "O", "dec_raw", "open", 60, [ol(reset)], "configured P0 + fixed Q, open loop", 1)
        emit("ol_dec_raw_fixed_%s" % rs, "O", "dec_raw", "open", 60, [cand, ol(reset)], "candidate P0 + fixed Q, open loop", 1)
        for model in ["isotropic", "axis_aware"]:
            for beta in [0.0, 0.9]:
                emit("ol_dec_raw_%s_b%s_a1_g1_%s" % (model, str(beta).replace(".", ""), rs), "O", "dec_raw", "open", 60,
                     [cand, qfrag(model, beta, 1.0, 1.0), ol(reset)], "R44 cell, open loop", 2)
        for (acc, gyr, beta) in grid:
            emit("ol_dec_raw_iso_b%s_a%s_g%d_%s" % (str(beta).replace(".", ""), str(acc).replace(".", ""), gyr, rs), "O", "dec_raw", "open", 60,
                 [cand, qfrag("isotropic", beta, acc, gyr), ol(reset)], "R45 grid, open loop", 2)
    emit("ol_cpl_meas_fixed_r5_check", "O", "cpl_meas", "open", 60, [cand, ol(5.0)], "identity check: open-loop trace must equal the dec_raw arm (state paths independent of estimator)", 1)

    with open(os.path.join(a.out, "arm_table.tsv"), "w") as f:
        f.write("arm\ttier\tarch\tkind\tduration_s\tyaml\tnotes\tpriority\toptional\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print("wrote %d arms to %s" % (len(rows), a.out))

if __name__ == "__main__":
    main()
