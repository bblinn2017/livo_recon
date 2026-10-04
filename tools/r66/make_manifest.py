#!/usr/bin/env python3
"""R66: write jobs.json (for gen_jobs.py) and arm_table.tsv.

    python3 tools/r66/make_manifest.py --repo . --out r66_manifest --batch r66_all --results-root <root> \
            --surface-keys surface_keys.json [--bags bags.txt]
Dispatch ONLY through the queue (standing rule 5.6a; see R66_INSTRUCTIONS.md section 7):
    python3 scripts/parallel/gen_jobs.py --manifest r66_manifest/jobs.json --package livo_recon --batch r66_all
    scripts/parallel/dispatch_queue.sh ablations/_job_queue/r66_all scripts/parallel/containers_12_CURRENT.txt

bags.txt = the nine other NTU VIRAL sequence names the coding agent found in stage 0d (one seq name per line, no eee_01).
Without --bags block E is not written (and the script says so). No common.* key is ever set. Nothing edits the repo.
Every job_id is unique (asserted). Job ids start with a two-digit stage code so that a name-ordered queue runs the identity
controls first, then smoke, then voxel blocks, then block D, then the dense surface arms last.
Arms whose full override set is identical to an earlier arm are NOT written twice: the later one is recorded in the arm
table column dup_of (the planning agent reads it from the earlier arm). RULE 3.11: there is no perturbation key anywhere.
"""
import argparse, json, os, sys
from collections import Counter, OrderedDict
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


P = "voxel_map.plane."
# splineless L0 pca baseline (baseline-reference.md Part 1)
L0 = {
    "estimator.mode": "decoupled", "spline.mode": "raw_imu",
    "voxel_map.plane.plane_fit_mode": "pca", "voxel_map.plane.plane_var_mode": "eigengap",
    "voxel_map.plane.plane_gate_mode": "disc", "voxel_map.plane.plane_fit_pose_cov_mode": "sensor_only",
    "voxel_map.plane.weight_floor.mode": "sensor_range", "voxel_map.residual.pose_cov_in_sigma": False,
    "cbk.lidar.point_filter_num": 3, "imu.ds.mode": "first", "imu.ds.ds_leaf_size": 0.1,
}
COMMON = {"evo.enable": True, "vio.enable": False, "eval.state_trace_en": True, "eval.imu_first_scans_en": True,
          "diagnostics.frame_window_log": True}
PFN1_DSOFF = {"cbk.lidar.point_filter_num": 1, "imu.ds.mode": "off", "imu.ds.ds_leaf_size": None}   # None = delete key
PFN3_DSOFF = {"imu.ds.mode": "off", "imu.ds.ds_leaf_size": None}
PFN1_DSFIRST = {"cbk.lidar.point_filter_num": 1}
DEBIASED = {P + "plane_fit_mode": "debiased"}
INFO = {P + "plane_var_mode": "information", P + "weight_floor.mode": "roughness"}
COMB = {P + "plane_fit_pose_cov_mode": "combined"}
REASONS = {P + "plane_fit_reasons_en": True}
CAPS_OFF = {"imu.process_noise.motion.acc_max_dynamic_variance": 1.0e6, "imu.process_noise.motion.gyro_max_dynamic_variance": 1.0e6}
STAMPS = {"data_queues.log_sensor_stamps": True}
MINDIAG = {"estimator.coupled.minimal_diagnostics": True}


def q(model, beta, acc, gyr, caps_off=True):
    d = {"imu.process_noise.model": model, "imu.process_noise.motion.beta": float(beta),
         "imu.process_noise.motion.acc_scale": float(acc), "imu.process_noise.motion.gyro_scale": float(gyr)}
    if caps_off:
        d.update(CAPS_OFF)
    return d


def fixedq(acc_v, gyr_v):
    return {"imu.process_noise.fixed.acc_variance": float(acc_v), "imu.process_noise.fixed.gyro_variance": float(gyr_v)}


def ol(reset=5.0):
    return {"lio.open_loop.mode": "propagate_only", "lio.open_loop.reset_period_s": float(reset)}


def woodbury(rho):
    return {"lio.residual_redundancy.mode": "off"} if rho is None else \
        {"lio.residual_redundancy.mode": "woodbury", "lio.residual_redundancy.rho": float(rho)}


def hold(n):
    return {"lio.startup_hold.scans": int(n)}


def boot(n, f):
    return {"lio.bootstrap_inflate.scans": int(n), "lio.bootstrap_inflate.factor": float(f)}


def bmode(m):
    return {"voxel_map.map.bootstrap_mode": m}


def calib(n):
    return {"calib.stationary.num_samples": int(n)}


def pcm(mode, ins="posterior"):
    return {"lio.point_cov.mode": mode, "lio.point_cov.insertion_source": ins}


def tag(x):
    return ("%g" % x).replace(".", "p").replace("-", "m").replace("+", "")


# best R65 motion-Q (R65 audit; isotropic, beta 0.9, acc scale 0.3, gyro scale 10, caps off): the 'motion-Q' of this plan
BEST = q("isotropic", 0.9, 0.3, 10)
AXB30 = q("axis_aware", 0.9, 0.3, 30, caps_off=False)
DS_ACC, DS_GYR = 7.3e-4, 1.4e-6            # VN-100 datasheet per-sample variances at 385 Hz (acc (m/s^2)^2, gyro (rad/s)^2)
GM_GYR_SIGMA = 0.664 * 7.5 * 3.141592653589793 / 180.0 / 3600.0    # 0.664 x (7.5 deg/hr Allan bias instability), rad/s
GM_ACC_SIGMA = 0.664 * 0.04e-3 * 9.80665                             # 0.664 x (0.04 mg Allan bias instability), m/s^2


def spline(mode="spline", hz=None, **extra):
    d = {"spline.mode": mode}
    if hz is not None:
        d["spline.control_points.hz"] = float(hz)
    d.update(extra)
    return d


def coupled(mode, gating=False, knots=4, iters=None, rho=None, minimal=True, **extra):
    d = {"estimator.mode": "coupled", "estimator.coupled.residual_evaluation_time": mode,
         "estimator.coupled.gating_state_uncertainty": bool(gating), "estimator.coupled.knot_count": int(knots),
         "estimator.coupled.test_id": "r66_%s" % mode}
    if minimal:
        d.update(MINDIAG)
    if iters is not None:
        d["lio.ekf.max_iterations"] = int(iters)
    if rho is not None:
        d.update(woodbury(rho))
    d.update(extra)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="r66_manifest")
    ap.add_argument("--surface-keys", required=True)
    ap.add_argument("--batch", default="r66_all")
    ap.add_argument("--bags", default=None, help="file with the nine other NTU VIRAL sequence names (block E)")
    ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frag = lambda rel: flat(_load(os.path.join(a.repo, rel)))
    cand = frag("config/experiments/p0_q/p0_candidate_fixed_q.yaml")
    cand.pop("estimator.coupled.gating_state_uncertainty", None)      # refused outside coupled mode
    conf = frag("config/experiments/p0_q/p0_configured_control.yaml")
    conf.pop("estimator.coupled.gating_state_uncertainty", None)
    sk = json.load(open(a.surface_keys))
    DEAD_UNDER_SURFACE = (P + "plane_fit_mode", P + "plane_gate_mode", P + "weight_floor.mode")
    bags = []
    if a.bags:
        bags = [l.strip() for l in open(a.bags) if l.strip() and not l.startswith("#")]
        # R66 agent fix: only 8 NTU VIRAL sequences other than eee_01 exist on this host (confirmed
        # by bag_stats.py inventory across every container); relaxed from == 9 to <= 9 per explicit
        # user instruction, keeping the not-eee_01 and distinct checks.
        assert len(bags) <= 9 and "eee_01" not in bags and len(set(bags)) == len(bags), "bags.txt must list at most 9 distinct sequences, not eee_01: %s" % bags
    else:
        print("WARNING: --bags not given; block E (9 other bags) is NOT written", file=sys.stderr)

    jobs, rows, sigs = [], [], {}
    aliases = []

    def add(name, stage, block, purpose, *frags, duration=0, keep=False, surface=False, refusal=False, seq="eee_01", base=None, control=False):
        o = {}
        o.update(COMMON)
        o.update(L0)
        for f in (frags if base is None else ((base,) + frags)):
            o.update(f)
        # signature: identical configurations are not run twice within the same sequence and duration
        sig = json.dumps([seq, duration, sorted((k, repr(v)) for k, v in o.items())])
        if sig in sigs and not refusal and not control:
            aliases.append((name, sigs[sig]))
            rows.append([name, block, "dup", duration or "full", seq, "DUPLICATE of %s: %s" % (sigs[sig], purpose), 0, sigs[sig]])
            return sigs[sig]
        jid = "%s_%s" % (stage, name)
        if not control:
            sigs[sig] = jid
        o["eval.state_trace_run_id"] = jid
        o["outputs.path"] = "%s/%s/%s" % (a.results_root, a.batch, jid)
        o["outputs.debug_log_dir"] = o["outputs.path"]
        o["outputs.odom.export"] = True
        if surface:
            for k in DEAD_UNDER_SURFACE:
                o.pop(k, None)
        j = {"job_id": jid, "dataset": "ntu_viral", "seq": seq, "overrides": o, "use_cache": False}
        if duration and duration > 0:
            j["duration_secs"] = duration
        jobs.append(j)
        kind = "refusal" if refusal else ("open" if o.get("lio.open_loop.mode") == "propagate_only" else "closed")
        rows.append([jid, block, kind, duration or "full", seq, purpose, 1 if keep else 0, ""])
        return jid

    # ================= stage 00: identity digests and controls (must be first and must match R64/R65)
    add("id_baseline_48s", "00", "ID", "identity: 48 s baseline; state-trace digest must be 197d7f00...0341", cand, duration=48, keep=True)
    add("id_baseline_60s", "00", "ID", "identity: 60 s baseline; digest must be a096c8b9...c8ad", cand, duration=60, keep=True)
    add("id_baseline_full", "00", "ID", "identity: full-bag baseline; digest must be 2d3eebfb...6857; reference for every full-bag comparison", cand, keep=True)
    add("id_baseline_full_rerun", "00", "ID", "determinism control: identical rerun of the full-bag baseline; every digest must equal id_baseline_full", cand, keep=True, control=True)
    add("id_new_keys_at_defaults_60s", "00", "ID",
        "identity: every R66 key that is in scope for a splineless decoupled run, spelled at its default (imu bias_model random_walk, akf_q/enable false, frame_window_log on); state trace must equal id_baseline_60s",
        cand, {"imu.process_noise.bias_model": "random_walk", "akf_q.enable": False}, duration=60)
    add("id_motionq_full", "00", "ID", "motion-Q control: best R65 motion-Q (isotropic, beta 0.9, acc scale 0.3, gyro scale 10, caps off); the 'motion-Q' of this batch", cand, BEST, keep=True)

    # ================= stage 10: smoke, 60 s, one per new key / mode / family (rule 3.6)
    S = "10"
    add("smoke_gm_gyro100_tau100_60s", S, "SMK", "60 s smoke: Gauss-Markov bias on (gyro sigma x100 datasheet, tau 100 s)", cand,
        {"imu.process_noise.bias_model": "gauss_markov", "imu.process_noise.gauss_markov.gyro.correlation_time_s": 100.0,
         "imu.process_noise.gauss_markov.gyro.sigma": 100 * GM_GYR_SIGMA, "imu.process_noise.gauss_markov.accel.correlation_time_s": 100.0,
         "imu.process_noise.gauss_markov.accel.sigma": 100 * GM_ACC_SIGMA}, duration=60)
    add("smoke_akf_q_60s", S, "SMK", "60 s smoke: AKF-style Q adaptation on the fixed-Q base (forgetting 0.99, warmup 20)", cand,
        {"akf_q.enable": True, "akf_q.forgetting": 0.99}, duration=60)
    add("smoke_spline_60s", S, "SMK", "60 s smoke: decoupled spline", cand, spline("spline"), duration=60)
    add("smoke_spline_refine_60s", S, "SMK", "60 s smoke: decoupled spline+refine", cand, spline("spline+refine"), duration=60)
    add("smoke_spline_explicit_defaults_60s", S, "SMK", "identity pair: decoupled spline with head_pin_update on_ok spelled out; must equal smoke_spline_60s", cand,
        spline("spline", **{"spline.head_pin_update": "on_ok"}), duration=60)
    add("smoke_spline_head_pin_always_60s", S, "SMK", "60 s smoke: spline head_pin_update always", cand, spline("spline", **{"spline.head_pin_update": "always"}), duration=60)
    for m in ("tail", "measurement"):
        add("smoke_coupled_%s_minimal_60s" % m, S, "SMK", "60 s smoke: coupled %s, minimal diagnostics (wall time decides block D order)" % m, cand, coupled(m), duration=60)
        add("smoke_coupled_%s_fulldiag_60s" % m, S, "SMK", "60 s smoke: coupled %s, FULL diagnostics (one short full-diagnostics cell per estimator, C-D11)" % m, cand,
            coupled(m, minimal=False), duration=60)
        add("smoke_coupled_%s_explicit_defaults_60s" % m, S, "SMK", "identity pair: coupled %s with on_solver_error throw and step_scale 1.0 spelled out; must equal smoke_coupled_%s_minimal_60s" % (m, m),
            cand, coupled(m, **{"estimator.coupled.on_solver_error": "throw", "estimator.coupled.step_scale": 1.0}), duration=60)
        add("smoke_coupled_%s_skip_scan_60s" % m, S, "SMK", "60 s smoke: coupled %s with on_solver_error skip_scan" % m, cand,
            coupled(m, **{"estimator.coupled.on_solver_error": "skip_scan"}), duration=60)
        add("smoke_coupled_%s_step0p5_60s" % m, S, "SMK", "60 s smoke: coupled %s with step_scale 0.5" % m, cand, coupled(m, **{"estimator.coupled.step_scale": 0.5}), duration=60)
    add("smoke_calib800_hold10_60s", S, "SMK", "60 s smoke: calibration num_samples 800 with hold 10", cand, calib(800), hold(10), duration=60)
    add("smoke_boot_seq_8x100_60s", S, "SMK", "60 s smoke: sequential bootstrap with inflation 8 x 100", cand, bmode("sequential"), boot(8, 100.0), duration=60)
    add("smoke_debiased_none_60s", S, "SMK", "60 s smoke: debiased none", cand, REASONS, DEBIASED, {P + "debias_mode": "none"}, duration=60)
    add("smoke_surface_dense_60s", S, "SMK", "60 s smoke: surface backend, cell 0.25, dense, sequential", cand, PFN1_DSOFF,
        dict(sk["backend_overrides"], **{sk["cell_size_key"]: 0.25, sk["bootstrap_key"]: "sequential"}), duration=60, surface=True)
    add("smoke_s2_scale_60s", S, "SMK", "60 s smoke: s2 gravity with accelerometer scale", cand, {"state.gravity_model": "s2", "calib.stationary.accel_excess_model": "scale"}, duration=60)
    add("smoke_openloop_motionq_60s", S, "SMK", "60 s smoke: open loop 5 s windows with motion-Q", cand, BEST, ol(5.0), duration=60)
    add("smoke_fixedq_grid_corner_60s", S, "SMK", "60 s smoke: fixed Q at the grid corner acc 2, gyro 1e-4", cand, fixedq(2.0, 1e-4), duration=60)

    # ================= stage 20: block A, start-up and onset
    A = "20"
    add("base_fixedq_full_A", A, "A", "baseline fixed Q (reference for block A); identical to id_baseline_full", cand, keep=True)
    cells = []
    for n in (100, 200, 400, 800, 1600, 3850):
        for h in (0, 10, 30, 50, 90):
            if n / 385.0 + 0.1 * h <= 10.0:
                cells.append((n, h))
    assert len(cells) == 23, len(cells)
    for qn, qf in (("fixedq", {}), ("motionq", BEST)):
        for n, h in cells:
            fr = [calib(n)] if n != 100 else []
            if h:
                fr.append(hold(h))
            add("A1_%s_calib%d_hold%d_full" % (qn, n, h), A, "A1",
                "calibration num_samples %d (%.2f s at 385 Hz) x startup hold %d scans (%.1f s), %s; start-up %.2f s (cap 10 s)" % (n, n / 385.0, h, 0.1 * h, qn, n / 385.0 + 0.1 * h),
                cand, qf, *fr, keep=(n, h) in ((800, 0), (3850, 0), (100, 50)) and qn == "fixedq")
    for mode in ("aggregate", "sequential"):
        for nn in (2, 4, 8, 16, 32):
            for f in (3, 10, 30, 100, 1000):
                add("A2_fixedq_%s_boot%d_x%d_full" % (mode, nn, f), A, "A2", "bootstrap mode %s, sigma^2 x%g for LIO scans 1..%d, fixed Q" % (mode, f, nn), cand, bmode(mode), boot(nn, f))
    for mode in ("aggregate", "sequential"):
        for nn in (2, 4, 8, 16, 32):
            add("A2_motionq_%s_boot%d_x100_full" % (mode, nn), A, "A2", "bootstrap mode %s, x100 for scans 1..%d, motion-Q (N ladder)" % (mode, nn), cand, BEST, bmode(mode), boot(nn, 100.0))
        for f in (3, 10, 30, 100, 1000):
            add("A2_motionq_%s_boot8_x%d_full" % (mode, f), A, "A2", "bootstrap mode %s, x%g for scans 1..8, motion-Q (factor ladder)" % (mode, f), cand, BEST, bmode(mode), boot(8, float(f)))
    for n in (100, 400, 1600):
        for mode in ("aggregate", "sequential"):
            for infl in (False, True):
                add("A3_fixedq_calib%d_%s_%s_full" % (n, mode, "boot8x100" if infl else "noinfl"), A, "A3",
                    "calibration num_samples %d x bootstrap mode %s x inflation %s, fixed Q" % (n, mode, "8 x 100" if infl else "none"),
                    cand, calib(n), bmode(mode), *([boot(8, 100.0)] if infl else []))
    for h in (10, 30):
        for infl in (False, True):
            for qn, qf in (("fixedq", {}), ("motionq", BEST)):
                add("A4_%s_hold%d_%s_full" % (qn, h, "boot8x100" if infl else "noinfl"), A, "A4", "hold %d x inflation %s, %s" % (h, "8 x 100" if infl else "none", qn), cand, qf, hold(h), *([boot(8, 100.0)] if infl else []))
    for h in (3, 5, 7, 10, 15, 20, 30):
        add("A5_motionq_hold%d_full" % h, A, "A5", "hold %d scans with motion-Q" % h, cand, BEST, hold(h), keep=(h == 10))

    # ================= stage 30: block B, Q models
    B = "30"
    for model in ("isotropic", "axis_aware"):
        for acc in (0, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100):
            add("B1_%s_beta09_acc%s_gyro10_capsoff_full" % (model, tag(acc)), B, "B1", "acc-only ladder: %s, beta 0.9, acc scale %g, gyro scale 10, caps off" % (model, acc), cand, q(model, 0.9, acc, 10), keep=(model == "isotropic" and acc in (0, 100)))
        for gyr in (2, 3, 5, 7, 10, 14, 20, 30, 45, 60, 100):
            add("B2_%s_beta09_acc0p3_gyro%d_capsoff_full" % (model, gyr), B, "B2", "gyro-only ladder: %s, beta 0.9, acc scale 0.3, gyro scale %d, caps off" % (model, gyr), cand, q(model, 0.9, 0.3, gyr))
    for model in ("isotropic", "axis_aware"):
        for acc in (0, 0.3):
            for gyr in (7, 10, 14, 20):
                for beta in (0.0, 0.5, 0.9, 0.99):
                    add("B3_%s_beta%s_acc%s_gyro%d_capsoff_full" % (model, tag(beta), tag(acc), gyr), B, "B3", "joint grid: %s beta %g acc scale %g gyro scale %d caps off" % (model, beta, acc, gyr), cand, q(model, beta, acc, gyr))
    for fa in (0.1, 1, 10, 100):
        for fg in (1, 10, 100, 1000):
            add("B4_motionq_accfloor%s_gyrfloor%s_full" % (tag(fa), tag(fg)), B, "B4", "floor-scale ladder on motion-Q: acc floor x%g, gyro floor x%g" % (fa, fg), cand, BEST,
                {"imu.process_noise.motion.acc_floor_scale": float(fa), "imu.process_noise.motion.gyro_floor_scale": float(fg)})
    rates = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
    G, AC = "imu.process_noise.bias_random_walk.gyro_variance_rate", "imu.process_noise.bias_random_walk.accel_variance_rate"
    for qn, qf in (("fixedq", {}), ("motionq", BEST)):
        for r in rates:
            add("B5_%s_gyrbias_rw_%s_full" % (qn, tag(r)), B, "B5", "gyro bias walk rate %g (rad/s)^2/s, accel rate default 1e-5, %s" % (r, qn), cand, qf, {G: r})
            add("B5_%s_accbias_rw_%s_full" % (qn, tag(r)), B, "B5", "accel bias walk rate %g (m/s^2)^2/s, gyro rate default 1e-5, %s" % (r, qn), cand, qf, {AC: r})
        for rg in (1e-7, 1e-5, 1e-3):
            for ra in (1e-7, 1e-5, 1e-3):
                add("B5_%s_bias_rw_gyr%s_acc%s_full" % (qn, tag(rg), tag(ra)), B, "B5", "bias walk cross: gyro %g, accel %g, %s" % (rg, ra, qn), cand, qf, {G: rg, AC: ra})
        for nm, fr in (("gyro_bias_off", {"calib.stationary.apply_to_state.gyro_bias": False}),
                       ("accel_bias_off", {"calib.stationary.apply_to_state.accel_bias": False}),
                       ("both_bias_off", {"calib.stationary.apply_to_state.gyro_bias": False, "calib.stationary.apply_to_state.accel_bias": False})):
            add("B5_%s_%s_full" % (qn, nm), B, "B5", "calibrated bias not applied to the state: %s, %s" % (nm, qn), cand, qf, fr)
    for av in (0.001, 0.01, 0.1, 0.5, 2.0):
        for gv in (1e-4, 1e-3, 1e-2, 0.3, 1.0):
            add("B6_fixed_acc%s_gyro%s_full" % (tag(av), tag(gv)), B, "B6", "fixed Q grid: acc variance %g, gyro variance %g" % (av, gv), cand, fixedq(av, gv))
    for k in (1, 3, 10, 30, 100, 300, 1000, 1e4, 1e5):
        add("B7_datasheet_x%s_full" % tag(k), B, "B7", "datasheet-derived fixed Q x%g (acc %.3g, gyro %.3g per-sample variances)" % (k, DS_ACC * k, DS_GYR * k), cand, fixedq(DS_ACC * k, DS_GYR * k))
    for f in (0.1, 10, 100):
        add("B8_p0_tilt_ambiguity_var_x%s_full" % tag(f), B, "B8", "P0 tilt/bias ambiguity accel std scaled so the tilt variance is x%g of the calibration value (std 0.03 x sqrt(%g))" % (f, f), cand,
            {"calib.p0.tilt_ba_ambiguity_accel_std": 0.03 * (f ** 0.5)})
    for vv in (1e-6, 1e-2):
        add("B8_p0_known_vel_var_%s_full" % tag(vv), B, "B8", "P0 velocity variance %g (calibration_derived P0; R65 value 1e-4)" % vv, cand, {"calib.p0.known_vel_variance": vv})
    for gm in ("vector3", "s2"):
        for ex in ("bias", "scale"):
            for qn, qf in (("fixedq", {}), ("motionq", BEST)):
                add("B9_%s_%s_%s_full" % (gm, ex, qn), B, "B9", "gravity model %s x accel excess model %s, %s" % (gm, ex, qn), cand, qf,
                    {"state.gravity_model": gm, "calib.stationary.accel_excess_model": ex})
    for mult in (1, 10, 100):
        for tau in (10.0, 100.0, 1000.0):
            for qn, qf in (("fixedq", {}), ("motionq", BEST)):
                add("B10_gm_sigma%dx_tau%d_%s_full" % (mult, int(tau), qn), B, "B10", "Gauss-Markov bias: sigma = %dx (0.664 x datasheet bias instability: gyro %.3g rad/s, accel %.3g m/s^2), tau %g s, %s" % (mult, mult * GM_GYR_SIGMA, mult * GM_ACC_SIGMA, tau, qn), cand, qf,
                    {"imu.process_noise.bias_model": "gauss_markov", "imu.process_noise.gauss_markov.gyro.correlation_time_s": tau,
                     "imu.process_noise.gauss_markov.gyro.sigma": mult * GM_GYR_SIGMA, "imu.process_noise.gauss_markov.accel.correlation_time_s": tau,
                     "imu.process_noise.gauss_markov.accel.sigma": mult * GM_ACC_SIGMA})
    for fo in (0.9, 0.99, 0.999):
        for sc in (0.01, 1.0, 100.0):
            add("B10_akf_forget%s_q0scale%s_full" % (tag(fo), tag(sc)), B, "B10", "AKF-style Q adaptation, forgetting %g, initial (nominal) fixed Q x%g (acc %.3g, gyro %.3g), fixed-Q base only (refused under motion-Q)" % (fo, sc, 0.5 * sc, 0.3 * sc), cand,
                fixedq(0.5 * sc, 0.3 * sc), {"akf_q.enable": True, "akf_q.forgetting": fo}, keep=(fo == 0.99 and sc == 1.0))
    add("B10_akf_forget0p99_maxratio10_full", B, "B10", "AKF Q, forgetting 0.99, applied value bounded to [0.01, 10] x nominal", cand, {"akf_q.enable": True, "akf_q.forgetting": 0.99, "akf_q.bounds.max_ratio": 10.0})
    add("B10_akf_forget0p99_maxratio1000_full", B, "B10", "AKF Q, forgetting 0.99, bound [0.01, 1000] x nominal", cand, {"akf_q.enable": True, "akf_q.forgetting": 0.99, "akf_q.bounds.max_ratio": 1000.0})
    add("B10_akf_forget0p99_warmup0_full", B, "B10", "AKF Q, forgetting 0.99, no warmup", cand, {"akf_q.enable": True, "akf_q.forgetting": 0.99, "akf_q.warmup_frames": 0})
    add("B10_akf_forget0p99_warmup100_full", B, "B10", "AKF Q, forgetting 0.99, warmup 100 frames", cand, {"akf_q.enable": True, "akf_q.forgetting": 0.99, "akf_q.warmup_frames": 100})
    # B11 open-loop Q calibration, 60 s, 5 s stationary windows
    olq = [("baseline", {}), ("motionq", BEST)]
    olq += [("datasheet_x%s" % tag(k), fixedq(DS_ACC * k, DS_GYR * k)) for k in (1, 100, 1e4)]
    olq += [("biaswalk_%s" % tag(r), {G: r, AC: r}) for r in (1e-9, 1e-7, 1e-3)]
    for mult, tau in ((10, 100.0), (100, 10.0)):
        olq.append(("gm_sigma%dx_tau%d" % (mult, int(tau)), {"imu.process_noise.bias_model": "gauss_markov", "imu.process_noise.gauss_markov.gyro.correlation_time_s": tau,
                    "imu.process_noise.gauss_markov.gyro.sigma": mult * GM_GYR_SIGMA, "imu.process_noise.gauss_markov.accel.correlation_time_s": tau,
                    "imu.process_noise.gauss_markov.accel.sigma": mult * GM_ACC_SIGMA}))
    for nm, fr in olq:
        add("B11_openloop5s_%s_60s" % nm, B, "B11", "open loop, 5 s stationary windows, 60 s: does predicted P match realised drift (%s)" % nm, cand, ol(5.0), fr, duration=60)

    # ================= stage 40: block C (voxel), map / plane / covariance
    C = "40"
    modes = (("pca", {}), ("debiased_full", DEBIASED), ("debiased_lambda0_only", dict(DEBIASED, **{P + "debias_mode": "lambda0_only"})),
             ("debiased_none", dict(DEBIASED, **{P + "debias_mode": "none"})))
    for qn, qf in (("fixedq", {}), ("motionq", BEST)):
        for mn, mf in modes:
            add("C1_%s_eigengap_%s_full" % (mn, qn), C, "C1", "plane fit %s, eigengap variance, %s" % (mn, qf and "motion-Q" or "fixed Q"), cand, REASONS, qf, mf, keep=(qn == "fixedq"))
            add("C1_%s_information_%s_full" % (mn, qn), C, "C1", "plane fit %s, information variance, %s" % (mn, "motion-Q" if qf else "fixed Q"), cand, REASONS, qf, mf, INFO)
            add("C1_%s_eigengap_posecombined_%s_full" % (mn, qn), C, "C1", "plane fit %s, eigengap, fit pose covariance combined, %s" % (mn, "motion-Q" if qf else "fixed Q"), cand, REASONS, qf, mf, COMB)
            if mn != "pca":
                add("C1_%s_eigengap_guardoff_%s_full" % (mn, qn), C, "C1", "plane fit %s, eigengap, debias_denominator_guard off, %s" % (mn, "motion-Q" if qf else "fixed Q"), cand, REASONS, qf, mf, {P + "debias_denominator_guard": False})
                add("C1_%s_eigengap_hyst2_%s_full" % (mn, qn), C, "C1", "plane fit %s, eigengap, revoke only above 2x threshold, %s" % (mn, "motion-Q" if qf else "fixed Q"), cand, REASONS, qf, mf, {P + "plane_threshold_hysteresis": 2.0})
    for qn, qf in (("fixedq", {}), ("motionq", BEST)):
        for m in ("end_pose", "propagated"):
            add("C2_sensoronly_%s_%s_full" % (m, qn), C, "C2", "point covariance mode %s, sensor-only plane fit, %s" % (m, "motion-Q" if qf else "fixed Q"), cand, qf, pcm(m), {"voxel_map.point_cov_stats_en": True}, keep=(qn == "fixedq" and m == "propagated"))
            for ins in ("posterior", "prior"):
                add("C2_fitcombined_%s_%s_%s_full" % (m, ins, qn), C, "C2", "point covariance mode %s, insertion source %s, combined plane-fit covariance, %s" % (m, ins, "motion-Q" if qf else "fixed Q"), cand, qf, COMB, pcm(m, ins), {"voxel_map.point_cov_stats_en": True})
    for qn, qf in (("fixedq", {}), ("motionq", BEST)):
        for cap in (2, 3, 5, 10):
            for nm, fr in (("exitoff", {}), ("norm1em3", {"lio.ekf.min_norm_dtheta": 1e-3, "lio.ekf.min_norm_dt": 1e-3}),
                           ("norm1em4", {"lio.ekf.min_norm_dtheta": 1e-4, "lio.ekf.min_norm_dt": 1e-4}),
                           ("norm1em5", {"lio.ekf.min_norm_dtheta": 1e-5, "lio.ekf.min_norm_dt": 1e-5}),
                           ("rel1em3", {"lio.ekf.min_diff_error": 1e-3})):
                add("C3_%s_iters%d_%s_full" % (qn, cap, nm), C, "C3", "LIO iteration cap %d with early exit %s, %s" % (cap, nm, "motion-Q" if qf else "fixed Q"), cand, qf, {"lio.ekf.max_iterations": cap}, fr)
    RHOS = (None, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)

    def rtag(r):
        return "independent" if r is None else "rho%s" % tag(r)
    for qn, qf in (("fixedq", {}), ("motionq", BEST)):
        for r in RHOS:
            add("C4_voxel_woodbury_%s_%s_full" % (rtag(r), qn), C, "C4", "voxel path, default density, Woodbury %s, %s" % (rtag(r), "motion-Q" if qf else "fixed Q"), cand, qf, woodbury(r), keep=(qn == "fixedq" and r in (None, 0.25)))
    for qn, qf in (("fixedq", {}), ("motionq", BEST)):
        for nm, fr in (("pfn1_dsfirst", PFN1_DSFIRST), ("pfn3_dsoff", PFN3_DSOFF), ("pfn1_dsoff", PFN1_DSOFF)):
            add("C6_%s_%s_full" % (nm, qn), C, "C6", "map density %s, pca voxel, %s" % (nm, "motion-Q" if qf else "fixed Q"), cand, qf, fr)

    # ================= stage 55: block F frame-1 and stationary-initialisation fragments (configured P0, motion-Q)
    F = "55"
    add("F_configuredP0_motionq_control_full", F, "F", "control for block F: configured P0, best motion-Q, no fragment", conf, BEST, keep=True)
    frs = sorted(os.listdir(os.path.join(a.repo, "config/experiments/frame1_covariance")))
    frs = [("frame1_covariance/" + x) for x in frs if x.endswith(".yaml")]
    frs += [("stationary_initialization_combinations/" + x) for x in sorted(os.listdir(os.path.join(a.repo, "config/experiments/stationary_initialization_combinations"))) if x.endswith(".yaml")]
    for rel in frs:
        nm = os.path.splitext(os.path.basename(rel))[0]
        add("F_%s_full" % nm, F, "F", "configured P0 + best motion-Q + fragment %s (R62 fragments were 30-60 s stationary only)" % rel, conf, BEST, flat(_load(os.path.join(a.repo, "config/experiments", rel))))

    # ================= stage 60: block D, Phase 4 spline / coupled sweep
    D = "60"
    for qn, qf in (("fixedq", {}), ("motionq", BEST)):
        add("D0_splineless_tbpn_off_%s_full" % qn, D, "D0", "splineless control with imu/undistort/time_based_process_noise none, %s (S-D7 confound control)" % qn, cand, qf, {"imu.undistort.time_based_process_noise": "none"})
        for mode in ("spline", "spline+refine"):
            mt = mode.replace("+", "_")
            for hz in (25, 50, 100, 200):
                add("D1_%s_hz%d_%s_full" % (mt, hz, qn), D, "D1", "decoupled %s, control-point rate %d Hz, %s" % (mode, hz, qn), cand, qf, spline(mode, hz), keep=(hz == 100 and qn == "fixedq"))
            add("D1_%s_hz100_endvel_off_%s_full" % (mt, qn), D, "D1", "decoupled %s, end_constraint/velocity false, %s" % (mode, qn), cand, qf, spline(mode, 100, **{"spline.end_constraint.velocity": False}))
            add("D1_%s_hz100_tbpn_off_%s_full" % (mt, qn), D, "D1", "decoupled %s with imu/undistort/time_based_process_noise none, %s" % (mode, qn), cand, qf, spline(mode, 100), {"imu.undistort.time_based_process_noise": "none"})
        add("D1_spline_refine_hz100_finalpass_%s_full" % qn, D, "D1", "spline+refine, final_pass true, %s" % qn, cand, qf, spline("spline+refine", 100, **{"spline.refine.final_pass": True}))
        for it in (2, 3):
            add("D1_spline_refine_hz100_iters%d_%s_full" % (it, qn), D, "D1", "spline+refine, refine iters %d (a step multiplier, S-D6), %s" % (it, qn), cand, qf, spline("spline+refine", 100, **{"spline.refine.iters": it}))
        for cw in (1e-3, 1e-2, 1e-1, 1.0):
            add("D1_spline_refine_hz100_curv%s_%s_full" % (tag(cw), qn), D, "D1", "spline+refine, curvature_weight %g (log ladder; the compiled default is 0, scale not derived), %s" % (cw, qn), cand, qf, spline("spline+refine", 100, **{"spline.refine.curvature_weight": cw}))
        for iw in (0.1, 1.0, 10.0, 100.0):
            add("D1_spline_refine_hz100_imuacc%s_%s_full" % (tag(iw), qn), D, "D1", "spline+refine, imu_acc_weight %g (log ladder; compiled default 0, scale not derived), %s" % (iw, qn), cand, qf, spline("spline+refine", 100, **{"spline.refine.imu_acc_weight": iw}))
    for acf in (0.35, 1.0):
        for beta in (0.3, 0.6, 1.0):
            for fl in ("allan", "off"):
                add("D2_adaptiveq_acf%s_beta%s_floor%s_full" % (tag(acf), tag(beta), fl), D, "D2", "decoupled spline with AdaptiveQ (fixed model), acf1_max %g, beta %g (beta_acc = beta_gyr), noise floor %s" % (acf, beta, fl), cand, spline("spline", 100),
                    {"adaptive_q.enable": True, "adaptive_q.acf1_max": acf, "adaptive_q.beta_acc": beta, "adaptive_q.beta_gyr": beta, "adaptive_q.noise_floor.mode": fl}, keep=(acf == 1.0 and beta == 0.3 and fl == "allan"))
    for qn, qf in (("fixedq", {}), ("motionq", BEST)):
        for m in ("measurement", "tail"):
            for gating in (False, True):
                for knots in (3, 4, 6, 8):
                    for it in (3, 5, 10):
                        add("D3_coupled_%s_gate%s_knots%d_iters%d_%s_full" % (m, "on" if gating else "off", knots, it, qn), D, "D3",
                            "coupled %s, gating_state_uncertainty %s, knot_count %d, max_iterations %d, %s (minimal diagnostics)" % (m, gating, knots, it, qn), cand, qf, coupled(m, gating, knots, it),
                            keep=(knots == 4 and it == 5 and not gating and qn == "fixedq"))
        for m in ("measurement", "tail"):
            for r in (0.25, 0.5):
                add("D4_coupled_%s_woodbury%s_%s_full" % (m, tag(r), qn), D, "D4", "coupled %s with Woodbury rho %g, %s" % (m, r, qn), cand, qf, coupled(m, rho=r))
    for m in ("measurement", "tail"):
        add("D6_coupled_%s_skipscan_fixedq_full" % m, D, "D6", "coupled %s, on_solver_error skip_scan (the counter shows whether it ever fires)" % m, cand, coupled(m, **{"estimator.coupled.on_solver_error": "skip_scan"}))
        for s in (0.5, 0.25):
            add("D6_coupled_%s_step%s_fixedq_full" % (m, tag(s)), D, "D6", "coupled %s, step_scale %g" % (m, s), cand, coupled(m, **{"estimator.coupled.step_scale": s}))
    for qn, qf in (("fixedq", {}), ("motionq", BEST)):
        for mode in ("spline", "spline+refine"):
            add("D6_%s_hz100_headpin_always_%s_full" % (mode.replace("+", "_"), qn), D, "D6", "decoupled %s, head_pin_update always, %s" % (mode, qn), cand, qf, spline(mode, 100, **{"spline.head_pin_update": "always"}))

    # ================= stage 70: block C5 surface backend (dense arms last)
    SU = "70"

    def surf(cell, rho, mode):
        fr = dict(sk["backend_overrides"])
        fr[sk["cell_size_key"]] = cell
        fr[sk["bootstrap_key"]] = mode
        fr.update(woodbury(rho))
        return fr
    for dense in (False, True):
        for cell in (0.25, 0.10):
            for mode in ("aggregate", "sequential"):
                for r in RHOS:
                    add("C5_surface_cell%s_%s_%s_%s_full" % (tag(cell), "dense" if dense else "default", mode, rtag(r)), SU, "C5",
                        "surface backend cell %g, %s density, bootstrap %s, Woodbury %s, fixed Q" % (cell, "dense (PFN 1, ds off)" if dense else "default (PFN 3, ds first)", mode, rtag(r)),
                        cand, (PFN1_DSOFF if dense else {}), surf(cell, r, mode), surface=True,
                        keep=(dense and cell == 0.25 and r in (None, 0.25)))

    # ================= stage 50: block E, nine other real sequences (pre-declared list, same on every bag)
    if bags:
        E = "50"
        elist = [("baseline", {}), ("motionq", BEST), ("axisaware_gyro30_capson", AXB30), ("hold10", hold(10)), ("boot8x100", boot(8, 100.0)),
                 ("motionq_boot8x100", dict(BEST, **boot(8, 100.0))), ("motionq_hold10", dict(BEST, **hold(10))),
                 ("iters10_norm1em4", {"lio.ekf.max_iterations": 10, "lio.ekf.min_norm_dtheta": 1e-4, "lio.ekf.min_norm_dt": 1e-4}),
                 ("debiased_none", dict(DEBIASED, **{P + "debias_mode": "none"})), ("spline_hz100", spline("spline", 100)),
                 ("coupled_tail", coupled("tail")), ("coupled_measurement", coupled("measurement"))]
        for b in bags:
            for nm, fr in elist:
                add("E_%s_%s_full" % (b, nm), E, "E", "other sequence %s: %s (fixed Q unless the name says motion-Q); same configuration list on every bag" % (b, nm), cand, fr, seq=b, keep=(nm in ("baseline", "motionq")))
            add("E_%s_baseline_rerun_full" % b, E, "E", "determinism control on %s: identical rerun of the baseline; digests must equal E_%s_baseline_full" % (b, b), cand, seq=b, keep=True)
    # ================= refusal checks (expected to fail at startup; the refusal text is the deliverable)
    R = "80"
    gmk = {"imu.process_noise.gauss_markov.gyro.correlation_time_s": 100.0, "imu.process_noise.gauss_markov.gyro.sigma": 1e-4,
           "imu.process_noise.gauss_markov.accel.correlation_time_s": 100.0, "imu.process_noise.gauss_markov.accel.sigma": 1e-3}
    for nm, fr, why in (
        ("RF_gm_keys_under_random_walk", gmk, "must be refused: gauss_markov keys set while bias_model=random_walk"),
        ("RF_gm_missing_keys", {"imu.process_noise.bias_model": "gauss_markov"}, "must be refused: gauss_markov requires the four numbers"),
        ("RF_gm_with_rw_rate", dict(gmk, **{"imu.process_noise.bias_model": "gauss_markov", G: 1e-6}), "must be refused: bias_random_walk rate set under gauss_markov"),
        ("RF_bias_model_unknown", {"imu.process_noise.bias_model": "bogus"}, "must be refused: not random_walk or gauss_markov"),
        ("RF_akf_under_motion_q", dict(BEST, **{"akf_q.enable": True}), "must be refused: akf_q needs imu/process_noise/model=fixed"),
        ("RF_akf_forgetting_out_of_scope", {"akf_q.forgetting": 0.9}, "must be refused: akf_q/forgetting set while akf_q/enable is false"),
        ("RF_adaptiveq_under_motion_q", dict(BEST, **spline("spline", 100), **{"adaptive_q.enable": True}), "must be refused: adaptive_q writes noise only the fixed model reads"),
        ("RF_head_pin_under_raw_imu", {"spline.head_pin_update": "always"}, "must be refused: spline/head_pin_update set while spline/mode=raw_imu"),
        ("RF_step_scale_zero", coupled("tail", **{"estimator.coupled.step_scale": 0.0}), "must be refused: step_scale must be in (0, 1]"),
        ("RF_step_scale_under_decoupled", {"estimator.coupled.step_scale": 0.5}, "must be refused: estimator/coupled/* set outside coupled mode")):
        add(nm, R, "RF", why, cand, fr, duration=10, refusal=True)

    names = [j["job_id"] for j in jobs]
    assert len(set(names)) == len(names), "duplicate job_id"
    json.dump(jobs, open(os.path.join(a.out, "jobs.json"), "w"), indent=1)
    with open(os.path.join(a.out, "arm_table.tsv"), "w") as f:
        f.write("arm\ttier\tarch\tkind\tduration_s\tseq\tnotes\tkeep_raw\tdup_of\n")
        for r in rows:
            nm, blk, kind, dur, seq, purpose, keep, dup = r
            f.write("\t".join(str(x) for x in [nm, blk, "dec_raw", kind, dur, seq, purpose.replace("\t", " "), keep, dup]) + "\n")
    c = Counter(r[1] for r in rows if r[2] != "dup")
    nd = sum(1 for r in rows if r[2] == "dup")
    json.dump({"n_jobs": len(jobs), "n_duplicates_not_written": nd, "per_block": dict(c), "duplicates": aliases}, open(os.path.join(a.out, "manifest_counts.json"), "w"), indent=1)
    print("wrote %d jobs to %s/jobs.json (%d duplicate configurations not written); per block: %s" % (len(jobs), a.out, nd, ", ".join("%s=%d" % kv for kv in sorted(c.items()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
