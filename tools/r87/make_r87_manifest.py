#!/usr/bin/env python3
"""make_r87_manifest.py -- R87: the debiased fit path applies voxel_map/points/min_init and min_update (patches/r87_p1_admission.patch). Planner's script; never run against ROS by the planner (rule 1.1).
Base: the four R86B jobs (r87_manifest/base_jobs_r86b_eee_01.json, the jobs file of the R86B round). 8 arms = 4 map / residual point-filter combinations x 2 settings of the two admission keys:
  c31  voxel_map.points.min_init 3, min_update 1   control: with the patch these values reproduce R86B (the pre-patch debiased path fitted at 3 points and refitted on every batch)
  n55  voxel_map.points.min_init 5, min_update 5   the fix on (the values of config/ntu_viral.yaml, written out explicitly)
Every other key equals the R86B job of the same tag. Usage:
  python3 tools/r87/make_r87_manifest.py --base-jobs r87_manifest/base_jobs_r86b_eee_01.json --base-table r87_manifest/base_arm_table_r86b.tsv --base-spec r87_manifest/base_listener_spec_r86b.json --out r87_manifest_regen --results-root /root/catkin_ws/livo_recon_results
  python3 tools/r87/make_r87_manifest.py --selftest
"""
import argparse, copy, csv, json, os, sys

QUEUE = "r87_eee_01"
SETTINGS = [("c31", 3, 1), ("n55", 5, 5)]
KEYS = ("voxel_map.points.min_init", "voxel_map.points.min_update")
TAGS = ["m3r3", "m1r1", "m1r3", "m3r1"]
R86B = "r86b_rt_sp_eee_01_v0p25_"
C = "r87_rt_sp_eee_01_v0p25_"
COMMON = ["imu_cov_growth.csv", "exit_status.txt", "engagement_imu.txt", "residual_dump.csv", "post_calibration_voxelmap.csv"]      # all 8 arms
N55_EXTRA = ["state_trace.csv", "residual_dump_state.csv", "pose_control_first_frame.csv", "pose_control_first_frame_state_chain.csv", "pose_control_first_frame_covariance_budget.csv",
             "pose_control_first_frame_covariance_budget.txt", "pose_control_first_frame_solve_matrices.txt", "voxelmap_snapshot_f2.csv"]   # fix-on arms only
ARCHIVE_FILES = ["residual_dump.csv", "residual_dump_state.csv", "post_calibration_voxelmap.csv", "pose_control_first_frame_solve_matrices.txt", "voxelmap_snapshot_f2.csv",
                 "pose_control_first_frame.csv", "pose_control_first_frame_state_chain.csv", "pose_control_first_frame_covariance_budget.csv", "pose_control_first_frame_covariance_budget.txt",
                 "calibration_p0.txt", "gravity_alignment.txt", "imu_first_scans.csv", "bootstrap_preparation.txt", "bootstrap_map_seed.txt",
                 "post_calibration_voxel_observation_support.csv", "state_trace.csv", "frame_stats.txt", "imu_cov_growth.csv", "lidar_msg_log.csv", "engagement_imu.txt", "exit_status.txt"]
ASKED = ("Bryce (2026-10-09): 'the next change should be first to fix the debiased code to follow the min init and min update parameters'; 'we should also be applying the minimum at bootstrap. That should be the case in the PCA branch as well. "
         "Bootstrap should only be exempt from the update based parameters, and the min frames to init parameter.'; 'In this case, I would like the next update to be an addendum, and I'd like the next update to be the debiased fix. Let's do this now and try it.'")


def aid(tag, s):
    return "%s%s_%s" % (C, tag, s)


def build(base_jobs, base_rows, results_root):
    by_tag = {j["job_id"][len(R86B):]: j for j in base_jobs}
    rb = {r["arm"][len(R86B):]: r for r in base_rows}
    assert sorted(by_tag) == sorted(TAGS) == sorted(rb)
    jobs, rows = [], []
    for s, mi, mu in SETTINGS:
        for tag in TAGS:
            j = copy.deepcopy(by_tag[tag]); o = j["overrides"]
            assert j["duration_secs"] == 60 and j["seq"] == "eee_01" and o["imu.process_noise.motion.acc_scale"] == 0.3 and o["voxel_map.plane.plane_fit_mode"] == "debiased"
            for k in KEYS:
                assert k not in o, k
            a = aid(tag, s)
            j["job_id"] = a
            o[KEYS[0]] = mi; o[KEYS[1]] = mu
            o["eval.state_trace_run_id"] = a
            path = "%s/%s/%s" % (results_root, QUEUE, a)
            o["outputs.path"] = path; o["outputs.debug_log_dir"] = path
            jobs.append(j)
            r = dict(rb[tag])
            ctl = (R86B + tag) if s == "c31" else aid(tag, "c31")
            r.update(arm=a, batch=QUEUE, control=ctl, same_batch_control="" if s == "c31" else ctl, ref=ctl, asked=ASKED,
                     src_arm=R86B + tag + " (R86B job) with voxel_map.points.min_init %d and min_update %d" % (mi, mu),
                     notes="R87 debiased admission rule: %s, min_init %d, min_update %d, eee_01, edge 0.25, first 60 s, layer 0%s" % (tag, mi, mu, "; control, must reproduce the R86B arm of the same tag" if s == "c31" else "; fix on"))
            rows.append(r)
    return jobs, rows


def listener_spec(base_spec, results_root):
    s = json.loads(json.dumps(base_spec).replace("r86b_manifest", "r87_manifest").replace("listener_spec_r86b", "listener_spec_r87").replace("R86B", "R87").replace("r86b_eee_01", QUEUE))
    s["queues"] = [{"dir": "ablations/_job_queue/" + QUEUE, "results_root": "%s/%s" % (results_root, QUEUE), "expected_jobs": len(SETTINGS) * len(TAGS)}]
    d = s["raw"]["delete"]
    for f in ARCHIVE_FILES:
        if f != "exit_status.txt" and f not in d:
            d.append(f)
    return s


def required(arms_by_set):
    allarms = [a for s, _, _ in SETTINGS for a in arms_by_set[s]]
    R = []
    def add(title, file, where="return", source="arm", cols=None, min_rows=1, maxb=60000000, ar=None):
        R.append(dict(followup="F-535/F-531", title=title, arms=list(allarms if ar is None else ar), file=file, columns=cols or [], min_rows=min_rows, where=where, source=source,
                      contains={}, expected_bytes=0, max_bytes_per_file=maxb))
    add("per-scan position, yaw and quaternion", "pos_trace_all.csv.gz", cols=["scan_id", "t_abs", "t_rel_s", "px", "py", "pz", "yaw_rad", "qw", "qx", "qy", "qz"], min_rows=500, maxb=330000)
    add("per-scan residual and voxel-count table (r85_short_reduce.py)", "short_scan_all.csv.gz", cols=["scan", "t_s", "residual_count", "fs_n_residuals", "fs_n_voxels", "fs_n_voxels_is_plane", "fs_iters", "fs_dx_pos_mm", "fs_nis", "fs_reduced_chi2", "pos_err_mm"], min_rows=500, maxb=200000)
    add("state_reference.txt (first-frame reference)", "state_reference.txt", maxb=4000)
    add("engagement of all arms concatenated", "engagement_all.txt", source="generated", maxb=4000000, ar=[])
    add("per-arm completeness", "completeness.csv", source="generated", cols=["arm", "requires_missing"], min_rows=8, maxb=100000, ar=[])
    for f in COMMON:
        add("lossless gzip archive of %s (host archive, byte for byte)" % f, f + ".gz", where="archive", maxb=100000000, ar=allarms)
    for f in N55_EXTRA:
        add("lossless gzip archive of %s (host archive, byte for byte), fix-on arms only" % f, f + ".gz", where="archive", maxb=100000000, ar=arms_by_set["n55"])
    add("archive index (raw bytes, raw sha256, rows)", "archive_index.json", where="archive", maxb=100000)
    add("archive sha256 manifest", "archive.sha256", where="archive", maxb=100000)
    for f, t in (("r87_gates.txt", "gate results"), ("r87_identity.csv", "the four c31 arms against the shipped R86B traces and digests"), ("r87_admission.csv", "plane voxels of the post-calibration map of the n55 arms against the R86B map"),
                 ("r87_stream_engagement.csv", "stream engagement counters per arm"), ("tree_check.txt", "tree used: R86B_tree.tar.gz plus patches/r87_p1_admission.patch; repairs if any"),
                 ("agent_edits.patch", "diff of the tested tree against R86B_tree.tar.gz (the R87 patch and every repair)"), ("R87_tree.tar.gz", "exact tested tree"),
                 ("checks/build_tail.txt", "tail of the catkin build"), ("checks/ctest.txt", "ctest output incl. admission_rule_r87"), ("R87_REPORT.md", "report of the coding agent (facts only)"),
                 ("code_answers.md", "answers to the open items of section 9"), ("harness_answers.md", "harness answers")):
        add(t, f, source="generated", ar=[], maxb=40000000 if f == "R87_tree.tar.gz" else 8000000)
    return R


def write_all(out, base_jobs, base_rows, base_spec, results_root, cols):
    jobs, rows = build(base_jobs, base_rows, results_root)
    os.makedirs(out, exist_ok=True)
    json.dump(jobs, open(os.path.join(out, "jobs_%s.json" % QUEUE), "w"), indent=1)
    json.dump(listener_spec(base_spec, results_root), open(os.path.join(out, "listener_spec_r87.json"), "w"), indent=1)
    with open(os.path.join(out, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n"); w.writeheader()
        for r in rows: w.writerow(r)
    abl = {s: [aid(t, s) for t in TAGS] for s, _, _ in SETTINGS}
    json.dump(required(abl), open(os.path.join(out, "required_outputs.json"), "w"), indent=1)
    json.dump({aid(t, "c31"): R86B + t for t in TAGS}, open(os.path.join(out, "controls.json"), "w"), indent=1)
    with open(os.path.join(out, "queue_order.tsv"), "w") as f:
        f.write("# listener spec\tqueue\tarms\nlistener_spec_r87.json\t%s\t%d arms\n" % (QUEUE, len(jobs)))
    L = ["# R87 compliance table (what each choice is for, rule 4.19)", "", "| item | value | what it is for |", "|---|---|---|",
         "| bag, edge, window, arm type | eee_01, 0.25 m, first 60 s, retime_trim, debiased, lines on, motion-Q (acc_scale 0.3), layer 0 | the cell of R86B, so the only change is the admission rule |",
         "| patches/r87_p1_admission.patch | the debiased path (and the pca path in the bootstrap frame) decide whether to fit through `admission::fitWanted` | the debiased path ignored voxel_map/points/min_init and min_update (fit at 3 points, refit on every batch); Bryce 2026-10-09: it must follow them, with min_init applied in the bootstrap frame, bootstrap exempt from min_update and min_frames_to_init only |",
         "| voxel_map.points.min_init / min_update = 3 / 1 (arms c31) | the values at which the patch reproduces the pre-patch debiased behaviour | control: must be bit-identical to the R86B arm of the same tag (gate IDENT_R86B); it shows the patch changes nothing unless the keys say so |",
         "| voxel_map.points.min_init / min_update = 5 / 5 (arms n55) | the values of config/ntu_viral.yaml, written out | the fix on: voxels with fewer than 5 raw points get no fit; a fitted voxel refits after 5 further raw points (bootstrap frame excepted) |",
         "| four map / residual point-filter combinations | m3r3, m1r1, m1r3, m3r1 as R86B | F-531 is the pf1 versus pf3 difference, and the 3-point voxels (F-535) are a bootstrap-map effect; the fix is crossed with the combinations so its effect on each is measured |",
         "| dump keys | identical to R86B | write-only; the fix-on arms return the full frame-1 set (state trace, every-iteration residual dump, solve matrices, map before and after frame 1) so the same analysis as R86B can be repeated; the controls return only what the identity gate needs |",
         "| gates | IDENT_R86B x4, MAP_ADMISSION x4, ENGAGED, MINPOINTS, PREIMU_SAME, SMALLSCANS, STREAM, DUMPS, EXIT, REQUIRED | controls bit-identical; the new map is exactly the R86B map restricted to voxels with at least 5 points; the rule reached the LIO residuals |", "", "Arms:", ""]
    for s, mi, mu in SETTINGS:
        for t in TAGS:
            L.append("- `%s`: min_init %d, min_update %d%s" % (aid(t, s), mi, mu, ", control of `%s`" % (R86B + t) if s == "c31" else ""))
    open(os.path.join(out, "COMPLIANCE_TABLE.md"), "w").write("\n".join(L) + "\n")
    return jobs, rows


def selftest():
    import tempfile
    here = os.path.dirname(os.path.abspath(__file__)); man = os.path.join(here, "..", "..", "r87_manifest")
    bj = json.load(open(os.path.join(man, "base_jobs_r86b_eee_01.json")))
    rd = csv.DictReader(open(os.path.join(man, "base_arm_table_r86b.tsv")), delimiter="\t"); cols = rd.fieldnames; br = list(rd)
    bs = json.load(open(os.path.join(man, "base_listener_spec_r86b.json")))
    t = tempfile.mkdtemp(prefix="r87_man_")
    jobs, rows = write_all(t, bj, br, bs, "/root/catkin_ws/livo_recon_results", cols)
    fails = []; ck = lambda c, m: None if c else fails.append(m)
    ck(len(jobs) == 8 and len(rows) == 8, "8 arms")
    by = {j["job_id"]: j for j in jobs}; b86 = {j["job_id"][len(R86B):]: j for j in bj}
    for s, mi, mu in SETTINGS:
        for tag in TAGS:
            o = by[aid(tag, s)]["overrides"]; ref = b86[tag]["overrides"]
            diff = {k for k in set(o) | set(ref) if o.get(k) != ref.get(k)}
            ck(diff == set(KEYS) | {"eval.state_trace_run_id", "outputs.path", "outputs.debug_log_dir"}, "%s differs from R86B %s in %s" % (aid(tag, s), tag, sorted(diff)))
            ck(o[KEYS[0]] == mi and o[KEYS[1]] == mu, "values " + aid(tag, s))
            ck(by[aid(tag, s)]["duration_secs"] == 60 and by[aid(tag, s)]["seq"] == "eee_01", "duration/seq")
    ck(len({j["overrides"]["outputs.path"] for j in jobs}) == 8, "distinct paths")
    req = json.load(open(os.path.join(t, "required_outputs.json")))
    arms_of = lambda f, w: {a for x in req if x["file"] == f and x["where"] == w for a in x["arms"]}
    n55 = {aid(tg, "n55") for tg in TAGS}
    ck(arms_of("pose_control_first_frame_solve_matrices.txt.gz", "archive") == n55 and arms_of("state_trace.csv.gz", "archive") == n55, "solve matrices / state trace for n55 only")
    ck(arms_of("residual_dump.csv.gz", "archive") == set(by) and arms_of("post_calibration_voxelmap.csv.gz", "archive") == set(by), "residual dump and map for all 8")
    sp = json.load(open(os.path.join(t, "listener_spec_r87.json")))
    ck(sp["queues"][0]["expected_jobs"] == 8 and "r86b" not in json.dumps(sp).lower().replace("r86b_rt", ""), "listener spec")
    ck("exit_status.txt" not in sp["raw"]["delete"] and "voxelmap_snapshot_f2.csv" in sp["raw"]["delete"], "listener deletes")
    ck(sorted(json.load(open(os.path.join(t, "controls.json")))) == sorted(aid(tg, "c31") for tg in TAGS), "controls.json")
    print("SELFTEST make_r87_manifest", ("FAIL " + "; ".join(fails)) if fails else "PASS")
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
