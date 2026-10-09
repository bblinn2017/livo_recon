#!/usr/bin/env python3
"""make_r88_manifest.py -- R88: the "binned" map type (voxel_map/backend = binned; patches/r88_p1_binned.patch). Planner's script; never run against ROS by the planner (rule 1.1).
Base: the four R87 n55 jobs (min_init 5, min_update 5), r88_manifest/base_jobs_r87_n55_eee_01.json. 16 arms = 4 map / residual point-filter combinations x 4 bin settings:
  ac  cov_mode average, weight_mode count      the bin carries the average member covariance; a bin counts as its k returns
  au  cov_mode average, weight_mode uniform    the bin carries the average member covariance; a bin counts as one observation
  sc  cov_mode shrunk,  weight_mode count      the bin carries the covariance of a mean of k independent returns (C / k^2); a bin counts as its k returns
  su  cov_mode shrunk,  weight_mode uniform    the bin carries C / k^2; a bin counts as one observation
Every other key equals the R87 n55 job of the same tag, except voxel_map.backend = binned, voxel_map.binned.bin_size_fraction = 0.2, the two mode keys, the trace id and the output paths. Usage:
  python3 tools/r88/make_r88_manifest.py --base-jobs r88_manifest/base_jobs_r87_n55_eee_01.json --base-table r88_manifest/base_arm_table_r87.tsv --base-spec r88_manifest/base_listener_spec_r87.json --out r88_manifest_regen --results-root /root/catkin_ws/livo_recon_results
  python3 tools/r88/make_r88_manifest.py --selftest
"""
import argparse, copy, csv, json, os, sys

QUEUE = "r88_eee_01"
VARIANTS = [("ac", "average", "count"), ("au", "average", "uniform"), ("sc", "shrunk", "count"), ("su", "shrunk", "uniform")]
CORNERS = ("ac", "su")   # these two settings return the full frame-1 set (size budget); the other two return the compact files and the map
TAGS = ["m3r3", "m1r1", "m1r3", "m3r1"]
R87 = "r87_rt_sp_eee_01_v0p25_"
C = "r88_rt_sp_eee_01_v0p25_"
NEW_KEYS = ("voxel_map.backend", "voxel_map.binned.bin_size_fraction", "voxel_map.binned.cov_mode", "voxel_map.binned.weight_mode")
COMMON = ["exit_status.txt", "engagement_imu.txt", "post_calibration_voxelmap.csv"]      # all 16 arms
CORNER_EXTRA = ["state_trace.csv", "residual_dump.csv"]                                    # arms ac and su
ARCHIVE_FILES = ["residual_dump.csv", "residual_dump_state.csv", "post_calibration_voxelmap.csv", "pose_control_first_frame_solve_matrices.txt", "voxelmap_snapshot_f2.csv",
                 "pose_control_first_frame.csv", "pose_control_first_frame_state_chain.csv", "pose_control_first_frame_covariance_budget.csv", "pose_control_first_frame_covariance_budget.txt",
                 "calibration_p0.txt", "gravity_alignment.txt", "imu_first_scans.csv", "bootstrap_preparation.txt", "bootstrap_map_seed.txt",
                 "post_calibration_voxel_observation_support.csv", "state_trace.csv", "frame_stats.txt", "imu_cov_growth.csv", "lidar_msg_log.csv", "engagement_imu.txt", "exit_status.txt"]
ASKED = ("Bryce (2026-10-09): 'I would like to generate an addendum, not a new round, but in an addendum where we implement the new binned voxel map code and test it.'; 'I would like it to have the qualities that I described earlier.'; "
         "'I would like two options where we keep the average covariance and some sort of shrunk covariance. I also would like options where we do count weighted and uniform.'; 'Please draft all of these code changes and generate the addendum.'")


def aid(tag, v):
    return "%s%s_%s" % (C, tag, v)


def build(base_jobs, base_rows, results_root):
    by_tag = {j["job_id"][len(R87):-len("_n55")]: j for j in base_jobs}
    rb = {r["arm"][len(R87):-len("_n55")]: r for r in base_rows}
    assert sorted(by_tag) == sorted(TAGS) == sorted(rb)
    jobs, rows = [], []
    for v, cm, wm in VARIANTS:
        for tag in TAGS:
            j = copy.deepcopy(by_tag[tag]); o = j["overrides"]
            assert j["duration_secs"] == 60 and j["seq"] == "eee_01" and o["voxel_map.plane.plane_fit_mode"] == "debiased" and o["voxel_map.plane.plane_fit_pose_cov_mode"] == "sensor_only"
            assert o["voxel_map.points.min_init"] == 5 and o["voxel_map.points.min_update"] == 5
            for k in NEW_KEYS:
                assert k not in o, k
            a = aid(tag, v)
            j["job_id"] = a
            o["voxel_map.backend"] = "binned"; o["voxel_map.binned.bin_size_fraction"] = 0.2; o["voxel_map.binned.cov_mode"] = cm; o["voxel_map.binned.weight_mode"] = wm
            o["eval.state_trace_run_id"] = a
            path = "%s/%s/%s" % (results_root, QUEUE, a)
            o["outputs.path"] = path; o["outputs.debug_log_dir"] = path
            jobs.append(j)
            r = dict(rb[tag])
            ref = R87 + tag + "_n55"
            r.update(arm=a, batch=QUEUE, control="", same_batch_control="", ref=ref, asked=ASKED,
                     src_arm="%s (R87 job) with voxel_map.backend binned, cov_mode %s, weight_mode %s, bin_size_fraction 0.2" % (ref, cm, wm),
                     notes="R88 binned map: %s, cov_mode %s, weight_mode %s, min_init 5 / min_update 5 counted in bins, eee_01, edge 0.25, first 60 s, layer 0" % (tag, cm, wm))
            rows.append(r)
    return jobs, rows


def listener_spec(base_spec, results_root):
    s = json.loads(json.dumps(base_spec).replace("r87_manifest", "r88_manifest").replace("listener_spec_r87", "listener_spec_r88").replace("R87", "R88").replace("r87_eee_01", QUEUE))
    s["queues"] = [{"dir": "ablations/_job_queue/" + QUEUE, "results_root": "%s/%s" % (results_root, QUEUE), "expected_jobs": len(VARIANTS) * len(TAGS)}]
    d = s["raw"]["delete"]
    for f in ARCHIVE_FILES:
        if f != "exit_status.txt" and f not in d:
            d.append(f)
    return s


def required(arms_by_v):
    allarms = [a for v, _, _ in VARIANTS for a in arms_by_v[v]]
    corner = [a for v in CORNERS for a in arms_by_v[v]]
    R = []
    def add(title, file, where="return", source="arm", cols=None, min_rows=1, maxb=60000000, ar=None):
        R.append(dict(followup="binned map", title=title, arms=list(allarms if ar is None else ar), file=file, columns=cols or [], min_rows=min_rows, where=where, source=source,
                      contains={}, expected_bytes=0, max_bytes_per_file=maxb))
    add("per-scan position, yaw and quaternion", "pos_trace_all.csv.gz", cols=["scan_id", "t_abs", "t_rel_s", "px", "py", "pz", "yaw_rad", "qw", "qx", "qy", "qz"], min_rows=500, maxb=330000)
    add("per-scan residual and voxel-count table (r85_short_reduce.py)", "short_scan_all.csv.gz", cols=["scan", "t_s", "residual_count", "fs_n_residuals", "fs_n_voxels", "fs_n_voxels_is_plane", "fs_iters", "fs_dx_pos_mm", "fs_nis", "fs_reduced_chi2", "pos_err_mm"], min_rows=500, maxb=200000)
    add("state_reference.txt (first-frame reference)", "state_reference.txt", maxb=4000)
    add("engagement of all arms concatenated", "engagement_all.txt", source="generated", maxb=4000000, ar=[])
    add("per-arm completeness", "completeness.csv", source="generated", cols=["arm", "requires_missing"], min_rows=16, maxb=100000, ar=[])
    for f in COMMON:
        add("lossless gzip archive of %s (host archive, byte for byte)" % f, f + ".gz", where="archive", maxb=100000000, ar=allarms)
    for f in CORNER_EXTRA:
        add("lossless gzip archive of %s (host archive, byte for byte), settings ac and su only" % f, f + ".gz", where="archive", maxb=100000000, ar=corner)
    add("archive index (raw bytes, raw sha256, rows)", "archive_index.json", where="archive", maxb=100000)
    add("archive sha256 manifest", "archive.sha256", where="archive", maxb=100000)
    for f, t in (("r88_gates.txt", "gate results"), ("r88_binning.csv", "bin counters and map checks per arm"), ("r88_stream_engagement.csv", "stream engagement counters per arm"),
                 ("tree_check.txt", "tree used: R87_tree.tar.gz plus patches/r88_p1_binned.patch; repairs if any"), ("agent_edits.patch", "diff of the tested tree against R87_tree.tar.gz (the R88 patch and every repair)"),
                 ("R88_tree.tar.gz", "exact tested tree"), ("checks/build_tail.txt", "tail of the catkin build"), ("checks/ctest.txt", "ctest output incl. binned_bins_r88 and binned_node_r88"),
                 ("R88_REPORT.md", "report of the coding agent (facts only)"), ("code_answers.md", "answers to the open items of section 9"), ("harness_answers.md", "harness answers")):
        add(t, f, source="generated", ar=[], maxb=40000000 if f == "R88_tree.tar.gz" else 8000000)
    return R


def compliance():
    L = ["# R88 compliance table (what each choice is for, rule 4.19)", "", "| item | value | what it is for |", "|---|---|---|",
         "| bag, edge, window, arm type | eee_01, 0.25 m, first 60 s, retime_trim, debiased, lines on, motion-Q (acc_scale 0.3), layer 0, min_init 5 / min_update 5 | the cell of the R87 fix-on arms, so the only change is the binned map |",
         "| patches/r88_p1_binned.patch | new map type `binned` (`voxel_map/backend`), bin accumulator in its own header, `VoxelPlane::refitFromBins`, a binned branch in `VoxelNode`'s debiased path | Bryce 2026-10-09: bins as a different type of voxel map in separate files, with average or shrunk covariance and count or uniform weight |",
         "| voxel_map.backend = binned | selects `BinnedVoxelMap` | the default map stays untouched; a run selects the binned map explicitly |",
         "| voxel_map.binned.bin_size_fraction = 0.2 | bin edge 0.05 m at the 0.25 m root voxel | the value of the existing PCA binning default; five bins per voxel edge |",
         "| cov_mode average / shrunk | C / k or C / k^2 for a bin of k returns | average treats the members' errors as fully correlated (repeated beam), shrunk as independent (error of a mean falls as 1/k) |",
         "| weight_mode count / uniform | weight k or 1 for a bin in the fit sums | count: a bin stands for its returns; uniform: a bin is one observation (information grows with occupied bins) |",
         "| min_init 5 / min_update 5 counted in bins | the same keys as R87, counted in occupied bins and bins opened since the last fit | a voxel built from one repeated beam is one bin and gets no fit; the comparison arm is R87 n55 (raw points) |",
         "| four map / residual point-filter combinations | m3r3, m1r1, m1r3, m3r1 as R86B and R87 | the repeated-beam voxels differ by map point filter (F-531, F-535) |",
         "| dump keys | identical to R87 | write-only; the settings ac and su return the full frame-1 set (state trace, every-iteration residual dump); all 16 arms return the post-calibration map and the compact files (size budget) |",
         "| gates | BINNED_ENGAGED, MAP_BINS x4, MINBINS, PREIMU_SAME, SMALLSCANS, STREAM, DUMPS, EXIT, REQUIRED | the binned branch ran with the named modes; bins are fewer than raw points; no voxel with fewer than 5 bins fitted; the propagation is unchanged |", "", "Arms:", ""]
    for v, cm, wm in VARIANTS:
        for t in TAGS:
            L.append("- `%s`: cov_mode %s, weight_mode %s%s" % (aid(t, v), cm, wm, " (full frame-1 set returned)" if v in CORNERS else ""))
    return "\n".join(L) + "\n"


def write_all(out, base_jobs, base_rows, base_spec, results_root, cols):
    jobs, rows = build(base_jobs, base_rows, results_root)
    os.makedirs(out, exist_ok=True)
    json.dump(jobs, open(os.path.join(out, "jobs_%s.json" % QUEUE), "w"), indent=1)
    json.dump(listener_spec(base_spec, results_root), open(os.path.join(out, "listener_spec_r88.json"), "w"), indent=1)
    with open(os.path.join(out, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n"); w.writeheader()
        for r in rows: w.writerow(r)
    abl = {v: [aid(t, v) for t in TAGS] for v, _, _ in VARIANTS}
    json.dump(required(abl), open(os.path.join(out, "required_outputs.json"), "w"), indent=1)
    json.dump({"modes": {aid(t, v): {"cov_mode": cm, "weight_mode": wm} for v, cm, wm in VARIANTS for t in TAGS}}, open(os.path.join(out, "arm_modes.json"), "w"), indent=1)
    with open(os.path.join(out, "queue_order.tsv"), "w") as f:
        f.write("# listener spec\tqueue\tarms\nlistener_spec_r88.json\t%s\t%d arms\n" % (QUEUE, len(jobs)))
    open(os.path.join(out, "COMPLIANCE_TABLE.md"), "w").write(compliance())
    return jobs, rows


def selftest():
    import tempfile
    here = os.path.dirname(os.path.abspath(__file__)); man = os.path.join(here, "..", "..", "r88_manifest")
    bj = json.load(open(os.path.join(man, "base_jobs_r87_n55_eee_01.json")))
    rd = csv.DictReader(open(os.path.join(man, "base_arm_table_r87.tsv")), delimiter="\t"); cols = rd.fieldnames; br = list(rd)
    bs = json.load(open(os.path.join(man, "base_listener_spec_r87.json")))
    t = tempfile.mkdtemp(prefix="r88_man_")
    jobs, rows = write_all(t, bj, br, bs, "/root/catkin_ws/livo_recon_results", cols)
    fails = []; ck = lambda c, m: None if c else fails.append(m)
    ck(len(jobs) == 16 and len(rows) == 16, "16 arms")
    by = {j["job_id"]: j for j in jobs}; b87 = {j["job_id"][len(R87):-4]: j for j in bj}
    for v, cm, wm in VARIANTS:
        for tag in TAGS:
            o = by[aid(tag, v)]["overrides"]; ref = b87[tag]["overrides"]
            diff = {k for k in set(o) | set(ref) if o.get(k) != ref.get(k)}
            ck(diff == set(NEW_KEYS) | {"eval.state_trace_run_id", "outputs.path", "outputs.debug_log_dir"}, "%s differs from R87 n55 %s in %s" % (aid(tag, v), tag, sorted(diff)))
            ck(o["voxel_map.binned.cov_mode"] == cm and o["voxel_map.binned.weight_mode"] == wm and o["voxel_map.backend"] == "binned" and o["voxel_map.binned.bin_size_fraction"] == 0.2, "values " + aid(tag, v))
            ck(o["voxel_map.points.min_init"] == 5 and o["voxel_map.points.min_update"] == 5, "min_init / min_update " + aid(tag, v))
    ck(len({j["overrides"]["outputs.path"] for j in jobs}) == 16, "distinct paths")
    ck(len({(j["overrides"]["voxel_map.binned.cov_mode"], j["overrides"]["voxel_map.binned.weight_mode"]) for j in jobs}) == 4, "four mode pairs")
    req = json.load(open(os.path.join(t, "required_outputs.json")))
    arms_of = lambda f, w: {a for x in req if x["file"] == f and x["where"] == w for a in x["arms"]}
    corner = {aid(tg, v) for tg in TAGS for v in CORNERS}
    ck(arms_of("state_trace.csv.gz", "archive") == corner and arms_of("residual_dump.csv.gz", "archive") == corner, "state trace / residual dump for ac and su only")
    ck(arms_of("post_calibration_voxelmap.csv.gz", "archive") == set(by), "map for all 16")
    sp = json.load(open(os.path.join(t, "listener_spec_r88.json")))
    ck(sp["queues"][0]["expected_jobs"] == 16 and "r87" not in json.dumps(sp).lower().replace("r87_rt", ""), "listener spec")
    ck("exit_status.txt" not in sp["raw"]["delete"] and "voxelmap_snapshot_f2.csv" in sp["raw"]["delete"], "listener deletes")
    ck(len(json.load(open(os.path.join(t, "arm_modes.json")))["modes"]) == 16, "arm_modes.json")
    print("SELFTEST make_r88_manifest", ("FAIL " + "; ".join(fails)) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-jobs"); ap.add_argument("--base-table"); ap.add_argument("--base-spec"); ap.add_argument("--out"); ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    rd = csv.DictReader(open(a.base_table), delimiter="\t")
    jobs, rows = write_all(a.out, json.load(open(a.base_jobs)), list(rd), json.load(open(a.base_spec)), a.results_root, rd.fieldnames)
    print("wrote %d arms, 1 queue (%s)" % (len(jobs), QUEUE))
