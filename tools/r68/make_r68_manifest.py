#!/usr/bin/env python3
"""R68: build jobs_smoke.json, jobs_refuse.json, jobs_main.json and arm_table.tsv from the R66 job list (r66_jobs.json, shipped in the package).

    python3 tools/r68/make_r68_manifest.py --jobs r68_inputs/r66_jobs.json --out r68_manifest --results-root /root/catkin_ws/livo_recon_results

Every R68 arm is an R66 job (same overrides) plus the delta written below. NO job sets calib.bootstrap.point_cov (the key is removed in R68 and would be refused).
Nothing edits the repo. No perturbation key anywhere (rule 3.11).

Blocks
  smoke    identity gates (R68 build vs the R67 sensor_model digests), default-change control, whole-run-inflation identity, GM engagement smoke
  refuse   two jobs that MUST fail at startup (removed key, removed spline mode)
  main
    DS   global LiDAR residual down-weighting (F-295): lio.bootstrap_inflate.scans = 100000 (whole run), factor F. eee_01 plus eee_03, nya_03; fixed-Q and motion-Q;
         pose-combined plane fit at F = 1 and 10; F = 300, 1000 on eee_01 to find where it stops helping
    GM   the 18 Gauss-Markov jobs of R66 block B10 and the 2 of B11, with the two bias_random_walk rate keys nulled (F-219)
    SF   surface cell 0.10 (F-294; F-223 wall cap) and dense PFN1 ds-off pca at rest
    ST   data_queues.log_sensor_stamps on one full baseline (stamp / short-frame data, F-173 F-179 F-190)
    TR   plane_fit_trace_frames = 20 on five arms (F-287)
"""
import argparse, copy, csv, json, os, sys

SEQS_OTHER = ["eee_03", "nya_03"]
FACTORS_E01 = [3, 10, 30, 100, 300, 1000]
FACTORS_BAG = [3, 10, 30, 100]
RW_NULL = {"imu.process_noise.bias_random_walk.gyro_variance_rate": None,
           "imu.process_noise.bias_random_walk.accel_variance_rate": None}
INFL = lambda f: {"lio.bootstrap_inflate.scans": 100000, "lio.bootstrap_inflate.factor": float(f)}


def arms(J):
    """yield dict(id, base, delta, seq(optional), dur(optional 'secs'), batch, group, tier, purpose, keep_raw)"""
    A = []
    def add(id_, base, delta=None, batch="main", group="", tier="", purpose="", keep=0, dur=None):
        A.append(dict(id=id_, base=base, delta=delta or {}, batch=batch, group=group, tier=tier, purpose=purpose, keep=keep, dur=dur))
    # ---- smoke
    add("00_id_baseline_48s", "00_id_baseline_48s", batch="smoke", group="ID", tier="ID", purpose="identity: digest must equal the R67 sensor_model digest", keep=1)
    add("00_id_baseline_60s", "00_id_baseline_60s", batch="smoke", group="ID", tier="ID", purpose="identity: digest must equal the R67 sensor_model digest", keep=1)
    add("00_id_baseline_full", "00_id_baseline_full", batch="smoke", group="ID", tier="ID", purpose="identity: digest must equal the R67 sensor_model digest", keep=1)
    add("00_id_motionq_full", "00_id_motionq_full", batch="smoke", group="ID", tier="ID", purpose="identity: digest must equal the R67 sensor_model digest", keep=1)
    add("00_id_default_posecov_60s", "00_id_baseline_60s", {"voxel_map.plane.plane_fit_pose_cov_mode": None}, batch="smoke", group="ID", tier="ID",
        purpose="plane_fit_pose_cov_mode key deleted: the new compiled default (sensor_only) must equal the pinned value; digest equals 00_id_baseline_60s")
    add("00_id_infl_whole_f1_60s", "00_id_baseline_60s", INFL(1.0), batch="smoke", group="ID", tier="ID",
        purpose="whole-run inflation at factor 1.0 is the identity: digest equals 00_id_baseline_60s")
    gm = dict(J["10_smoke_gm_gyro100_tau100_60s"]["overrides"])
    add("10_smoke_gm_gyro100_tau100_60s", "10_smoke_gm_gyro100_tau100_60s", RW_NULL, batch="smoke", group="GM", tier="SMK",
        purpose="Gauss-Markov engagement smoke with the two random-walk rate keys nulled (F-219): must run 60 s")
    # ---- refuse
    add("r68_refuse_point_cov_zero", "00_id_baseline_60s", {"calib.bootstrap.point_cov": "zero"}, batch="refuse", group="REF", tier="REF", dur=5,
        purpose="MUST FAIL at startup: removed key calib/bootstrap/point_cov")
    add("r68_refuse_spline_mode", "00_id_baseline_60s", {"spline.mode": "spline"}, batch="refuse", group="REF", tier="REF", dur=5,
        purpose="MUST FAIL at startup: spline/mode accepts only raw_imu")
    # ---- main: ID determinism
    add("00_id_baseline_full_rerun", "00_id_baseline_full", group="ID", tier="ID", purpose="determinism control (digest equals 00_id_baseline_full)", keep=1)
    # ---- DS
    for q, qb, suf in (("fq", "00_id_baseline_full", "fixedq"), ("mq", "00_id_motionq_full", "motionq")):
        for f in FACTORS_E01:
            add("r68_ds_eee_01_%s_f%d" % (q, f), qb, INFL(f), group="DS", tier="DS", keep=(1 if f in (10, 100) else 0),
                purpose="eee_01 %s, whole-run LiDAR residual variance x%d (information matrix / %d); reference = 00_id_%s_full" % (suf, f, f, "baseline" if q == "fq" else "motionq"))
    for sq in SEQS_OTHER:
        for q, suf in (("fq", "baseline"), ("mq", "motionq")):
            b = "50_E_%s_%s_full" % (sq, suf)
            add("r68_ds_%s_%s_f1" % (sq, q), b, group="DS", tier="DS", keep=1, purpose="%s %s control for the down-weighting arms (no inflation key)" % (sq, suf))
            for f in FACTORS_BAG:
                add("r68_ds_%s_%s_f%d" % (sq, q, f), b, INFL(f), group="DS", tier="DS", keep=(1 if f in (10, 100) else 0),
                    purpose="%s %s, whole-run LiDAR residual variance x%d" % (sq, suf, f))
    for q, b, suf in (("fq", "40_C1_pca_eigengap_posecombined_fixedq_full", "fixed-Q"), ("mq", "40_C1_pca_eigengap_posecombined_motionq_full", "motion-Q")):
        add("r68_ds_eee_01_%s_combined_f1" % q, b, group="DS", tier="DS", keep=1, purpose="plane_fit_pose_cov_mode combined, %s, no inflation (control)" % suf)
        add("r68_ds_eee_01_%s_combined_f10" % q, b, INFL(10), group="DS", tier="DS", keep=1, purpose="plane_fit_pose_cov_mode combined, %s, whole-run variance x10" % suf)
    # ---- GM
    for n in J:
        if n.startswith("30_B10_gm_") or n.startswith("30_B11_openloop5s_gm_"):
            add(n, n, RW_NULL, group="GM", tier="GM", purpose="R66 Gauss-Markov job re-issued with random-walk rate keys nulled (F-219)")
    # ---- SF
    for n in ("70_C5_surface_cell0p1_default_aggregate_independent_full", "70_C5_surface_cell0p1_default_sequential_independent_full",
              "70_C5_surface_cell0p1_dense_aggregate_independent_full", "70_C5_surface_cell0p1_dense_sequential_independent_full",
              "40_C6_pfn1_dsoff_fixedq_full", "40_C6_pfn1_dsoff_motionq_full"):
        add(n, n, group="SF", tier="SF", purpose="F-294 / F-223: re-run with the bootstrap fix; wall cap must not truncate it (see instructions section 7)")
    # ---- ST (F-173, F-179, F-190, F-236): per-message LiDAR/camera stamp log (data_queues/log_sensor_stamps, R65, write-only, never enabled before)
    add("r68_stamps_baseline_full", "00_id_baseline_full", {"data_queues.log_sensor_stamps": True}, group="ST", tier="ST", keep=1,
        purpose="00_id_baseline_full with data_queues/log_sensor_stamps on (write-only log): digest must equal 00_id_baseline_full (inert check) and sensor_stamps.csv is the data")
    # ---- TR
    tr = {"voxel_map.plane.plane_fit_trace_frames": 20, "voxel_map.plane.plane_fit_reasons_en": True}
    for id_, b, p in (("r68_tr_fixedq", "00_id_baseline_full", "baseline fixed-Q"), ("r68_tr_motionq", "00_id_motionq_full", "baseline motion-Q"),
                      ("r68_tr_posecombined_fixedq", "40_C1_pca_eigengap_posecombined_fixedq_full", "pose-combined fixed-Q"),
                      ("r68_tr_debiased_none_fixedq", "40_C1_debiased_none_eigengap_fixedq_full", "debiased none fixed-Q"),
                      ("r68_tr_information_fixedq", "40_C1_pca_information_fixedq_full", "pca information fixed-Q")):
        add(id_, b, tr, group="TR", tier="TR", keep=1, purpose="plane_fit_trace_frames=20 (frames 0..20, frame 0 = bootstrap window) on %s (F-287)" % p)
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", default="r68_manifest")
    ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    a = ap.parse_args()
    J = {j["job_id"]: j for j in json.load(open(a.jobs))}
    os.makedirs(a.out, exist_ok=True)
    out = {"smoke": [], "refuse": [], "main": []}; table = []; seen = set()
    for r in arms(J):
        if r["base"] not in J:
            sys.exit("base job %s not in r66_jobs.json" % r["base"])
        assert r["id"] not in seen, r["id"]; seen.add(r["id"])
        j = copy.deepcopy(J[r["base"]]); j["job_id"] = r["id"]
        o = j["overrides"]
        for k, v in r["delta"].items():
            # AGENT FIX (R68): a None delta value here was written assuming
            # gen_jobs.py's null-deletion mechanism can remove ANY key --
            # but apply_deletions() only strips a key from the BASE CONFIG
            # document (config/ntu_viral.yaml), for a key the base config
            # itself sets (e.g. imu/process_noise/bias_random_walk/*, used
            # correctly below by RW_NULL). voxel_map.plane.plane_fit_pose_cov_mode
            # is never in the base config -- it is only ever set via a job's
            # own override (inherited here from 00_id_baseline_60s) -- so
            # passing it through as a null override made gen_jobs.py's
            # apply_deletions() raise KeyError (confirmed live: "key
            # 'plane_fit_pose_cov_mode' does not exist in the base config at
            # that nesting"). For an override-only key, simply popping it
            # from this job's own override dict has the identical runtime
            # effect (the compiled default, sensor_only, applies) without
            # ever touching gen_jobs.py's base-config deletion path.
            if k == "voxel_map.plane.plane_fit_pose_cov_mode" and v is None:
                o.pop(k, None)
            else:
                o[k] = v
        if "calib.bootstrap.point_cov" in o and r["batch"] != "refuse":
            sys.exit("point_cov key forbidden outside the refuse batch")
        o["eval.state_trace_run_id"] = r["id"]
        path = "%s/r68_%s/%s" % (a.results_root, r["batch"], r["id"])
        o["outputs.path"] = path; o["outputs.debug_log_dir"] = path
        for k in o:
            if "perturb" in k.lower():
                sys.exit("perturbation key forbidden: " + k)
        if r["dur"]:
            j["duration_secs"] = r["dur"]
        out[r["batch"]].append(j)
        dur = str(j.get("duration_secs", "full"))
        table.append(dict(arm=r["id"], tier=r["tier"], arch="dec_raw", kind="closed", duration_s=dur, seq=j["seq"], notes=r["purpose"], keep_raw=str(r["keep"]),
                          dup_of="", r66_id=r["base"], group=r["group"], batch="r68_" + r["batch"]))
    for b, lst in out.items():
        json.dump(lst, open(os.path.join(a.out, "jobs_%s.json" % b), "w"), indent=1)
    cols = ["arm", "tier", "arch", "kind", "duration_s", "seq", "notes", "keep_raw", "dup_of", "r66_id", "group", "batch"]
    with open(os.path.join(a.out, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t"); w.writeheader(); w.writerows(table)
    print("smoke %d  refuse %d  main %d  total %d" % (len(out["smoke"]), len(out["refuse"]), len(out["main"]), len(table)))
    import collections
    print("main by group:", dict(collections.Counter(t["group"] for t in table if t["batch"] == "r68_main")))


main()
