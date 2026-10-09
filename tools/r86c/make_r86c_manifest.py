#!/usr/bin/env python3
"""make_r86c_manifest.py -- R86C addendum: the four R86B arms repeated at four values of imu.process_noise.motion.acc_scale (planner's script; never run against ROS by the planner, rule 1.1).
Config-only round on the R86B returned tree (R86B_tree.tar.gz): no patch, no code change.
Arms: {m3r3, m1r1, m1r3, m3r1} x acc_scale {0.3 (control = the R86B value), 0.1, 0.03, 0.0}; 16 jobs, queue r86c_eee_01. Every other key of an arm equals the R86B job of the same tag.
Usage:
  python3 tools/r86c/make_r86c_manifest.py --base-jobs r86c_manifest/base_jobs_r86b_eee_01.json --base-table r86c_manifest/base_arm_table_r86b.tsv --base-spec r86c_manifest/base_listener_spec_r86b.json --out r86c_manifest_regen --results-root /root/catkin_ws/livo_recon_results
  python3 tools/r86c/make_r86c_manifest.py --selftest
"""
import argparse, copy, csv, json, os, sys

QUEUE = "r86c_eee_01"
KEY = "imu.process_noise.motion.acc_scale"
LEVELS = [("a300", 0.3), ("a100", 0.1), ("a030", 0.03), ("a000", 0.0)]    # a<thousandths>; a300 is the R86B value (control)
TAGS = ["m3r3", "m1r1", "m1r3", "m3r1"]
R86B = "r86b_rt_sp_eee_01_v0p25_"
C = "r86c_rt_sp_eee_01_v0p25_"
# Return list (what is archived-and-returned per arm). Everything the listener archives stays on the host; only these names are copied into the return.
CONTROL_ARCHIVE = ["imu_cov_growth.csv", "exit_status.txt", "engagement_imu.txt"]                                   # a300: identity against R86B only
LEVEL_ARCHIVE = CONTROL_ARCHIVE + ["state_trace.csv", "pose_control_first_frame.csv", "pose_control_first_frame_state_chain.csv",
                                   "pose_control_first_frame_covariance_budget.csv", "pose_control_first_frame_covariance_budget.txt", "residual_dump_state.csv"]
ZERO_EXTRA = ["residual_dump.csv"]                                                                                 # only a000 returns the per-residual dump
# everything the listener should archive (superset; the R86B list, so the unchanged scorer archives the same names)
ARCHIVE_FILES = ["residual_dump.csv", "residual_dump_state.csv", "post_calibration_voxelmap.csv", "pose_control_first_frame_solve_matrices.txt", "voxelmap_snapshot_f2.csv",
                 "pose_control_first_frame.csv", "pose_control_first_frame_state_chain.csv", "pose_control_first_frame_covariance_budget.csv", "pose_control_first_frame_covariance_budget.txt",
                 "calibration_p0.txt", "gravity_alignment.txt", "imu_first_scans.csv", "bootstrap_preparation.txt", "bootstrap_map_seed.txt",
                 "post_calibration_voxel_observation_support.csv", "state_trace.csv", "frame_stats.txt", "imu_cov_growth.csv", "lidar_msg_log.csv", "engagement_imu.txt", "exit_status.txt"]
ASKED = ("Bryce (2026-10-09): 'I say right now we first generate an addendum where we try lowering Q acceleration scale.'; "
         "earlier the same day: 'we could consider scaling the acceleration scale factor for Q down to trust the IMU more'")


def aid(tag, lvl):
    return "%s%s_%s" % (C, tag, lvl)


def build(base_jobs, base_rows, results_root):
    by_tag = {j["job_id"][len(R86B):]: j for j in base_jobs}
    rb = {r["arm"][len(R86B):]: r for r in base_rows}
    assert sorted(by_tag) == sorted(TAGS) == sorted(rb), (sorted(by_tag), sorted(rb))
    jobs, rows = [], []
    for lvl, val in LEVELS:
        for tag in TAGS:
            j = copy.deepcopy(by_tag[tag])
            o = j["overrides"]
            assert o[KEY] == 0.3 and j["duration_secs"] == 60 and j["seq"] == "eee_01", tag
            a = aid(tag, lvl)
            j["job_id"] = a
            o[KEY] = val
            o["eval.state_trace_run_id"] = a
            path = "%s/%s/%s" % (results_root, QUEUE, a)
            o["outputs.path"] = path
            o["outputs.debug_log_dir"] = path
            jobs.append(j)
            r = dict(rb[tag])
            ctl = (R86B + tag) if lvl == "a300" else aid(tag, "a300")
            r.update(arm=a, batch=QUEUE, control=ctl, same_batch_control="" if lvl == "a300" else ctl, ref=ctl, asked=ASKED,
                     src_arm=R86B + tag + " (R86B job) with imu.process_noise.motion.acc_scale %g" % val,
                     notes="R86C acc_scale ladder: %s (map/residual point filter as in R86B %s), imu.process_noise.motion.acc_scale %g, eee_01, edge 0.25, first 60 s, layer 0%s" % (tag, tag, val, "; control of the R86B arm of the same tag" if lvl == "a300" else ""))
            rows.append(r)
    return jobs, rows


def listener_spec(base_spec, results_root):
    s = json.loads(json.dumps(base_spec).replace("r86b_manifest", "r86c_manifest").replace("listener_spec_r86b", "listener_spec_r86c").replace("R86B", "R86C").replace("r86b_eee_01", QUEUE))
    s["queues"] = [{"dir": "ablations/_job_queue/" + QUEUE, "results_root": "%s/%s" % (results_root, QUEUE), "expected_jobs": len(LEVELS) * len(TAGS)}]
    d = s["raw"]["delete"]
    for f in ARCHIVE_FILES:
        if f != "exit_status.txt" and f not in d:
            d.append(f)
    return s


def required(arms_by_lvl):
    allarms = [a for lvl, _ in LEVELS for a in arms_by_lvl[lvl]]
    R = []
    def add(title, file, where="return", source="arm", cols=None, min_rows=1, maxb=60000000, ar=None):
        R.append(dict(followup="F-531/F-537", title=title, arms=list(allarms if ar is None else ar), file=file, columns=cols or [], min_rows=min_rows, where=where, source=source,
                      contains={}, expected_bytes=0, max_bytes_per_file=maxb))
    add("per-scan position, yaw and quaternion", "pos_trace_all.csv.gz", cols=["scan_id", "t_abs", "t_rel_s", "px", "py", "pz", "yaw_rad", "qw", "qx", "qy", "qz"], min_rows=500, maxb=330000)
    add("per-scan residual and voxel-count table (r85_short_reduce.py)", "short_scan_all.csv.gz", cols=["scan", "t_s", "residual_count", "fs_n_residuals", "fs_n_voxels", "fs_n_voxels_is_plane", "fs_iters", "fs_dx_pos_mm", "fs_nis", "fs_reduced_chi2", "pos_err_mm"], min_rows=500, maxb=200000)
    add("state_reference.txt (first-frame reference)", "state_reference.txt", maxb=4000)
    add("engagement of all arms concatenated", "engagement_all.txt", source="generated", maxb=4000000, ar=[])
    add("per-arm completeness", "completeness.csv", source="generated", cols=["arm", "requires_missing"], min_rows=16, maxb=100000, ar=[])
    ctl = [a for a in arms_by_lvl["a300"]]
    new = [a for lvl, _ in LEVELS if lvl != "a300" for a in arms_by_lvl[lvl]]
    zero = list(arms_by_lvl["a000"])
    for f in CONTROL_ARCHIVE:
        add("lossless gzip archive of %s (host archive, byte for byte)" % f, f + ".gz", where="archive", maxb=100000000, ar=allarms)
    for f in [x for x in LEVEL_ARCHIVE if x not in CONTROL_ARCHIVE]:
        add("lossless gzip archive of %s (host archive, byte for byte)" % f, f + ".gz", where="archive", maxb=100000000, ar=new)
    for f in ZERO_EXTRA:
        add("lossless gzip archive of %s (host archive, byte for byte), acc_scale 0.0 arms only" % f, f + ".gz", where="archive", maxb=100000000, ar=zero)
    add("archive index (raw bytes, raw sha256, rows)", "archive_index.json", where="archive", maxb=100000)
    add("archive sha256 manifest", "archive.sha256", where="archive", maxb=100000)
    for f, t in (("r86c_gates.txt", "gate results"), ("r86c_identity.csv", "the four a300 arms against the shipped R86B traces and digests"),
                 ("r86c_acc_engagement.csv", "Q_acc (velocity block) per level against the a300 arm of the same tag"), ("r86c_stream_engagement.csv", "stream engagement counters per arm"),
                 ("tree_check.txt", "tree used: R86B_tree.tar.gz sha256 and the keys of the 16 jobs"), ("agent_edits.patch", "diff of the tested tree against R86B_tree.tar.gz (expected empty)"),
                 ("checks/ctest.txt", "ctest output of the unchanged tree"), ("R86C_REPORT.md", "report of the coding agent (facts only)"),
                 ("code_answers.md", "answers to the open items of section 9"), ("harness_answers.md", "harness answers")):
        add(t, f, source="generated", ar=[], maxb=8000000)
    return R


def write_all(out, base_jobs, base_rows, base_spec, results_root, cols):
    jobs, rows = build(base_jobs, base_rows, results_root)
    os.makedirs(out, exist_ok=True)
    json.dump(jobs, open(os.path.join(out, "jobs_%s.json" % QUEUE), "w"), indent=1)
    json.dump(listener_spec(base_spec, results_root), open(os.path.join(out, "listener_spec_r86c.json"), "w"), indent=1)
    with open(os.path.join(out, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    abl = {lvl: [aid(t, lvl) for t in TAGS] for lvl, _ in LEVELS}
    json.dump(required(abl), open(os.path.join(out, "required_outputs.json"), "w"), indent=1)
    json.dump({aid(t, "a300"): R86B + t for t in TAGS}, open(os.path.join(out, "controls.json"), "w"), indent=1)
    with open(os.path.join(out, "queue_order.tsv"), "w") as f:
        f.write("# listener spec\tqueue\tarms\nlistener_spec_r86c.json\t%s\t%d arms\n" % (QUEUE, len(jobs)))
    L = ["# R86C compliance table (what each choice is for, rule 4.19)", "", "| item | value | what it is for |", "|---|---|---|",
         "| bag, edge, window, arm type | eee_01, 0.25 m, first 60 s, retime_trim, debiased, lines on, motion-Q, layer 0 | the same cell as R86B, so the only change is the Q setting |",
         "| imu.process_noise.motion.acc_scale | 0.3, 0.1, 0.03, 0.0 | the strength of the data-driven (excitation-energy) part of the accelerometer process noise: step variance = floor term + acc_scale^2 x filtered excitation energy above the floor. Bryce (2026-10-09): scale it down to trust the IMU more. 0.3 is the R86B value (control); 0.0 leaves only the floor term. Validation in imu_processing.cpp accepts 0.0 (rejects only negative or non-finite) |",
         "| four map / residual point-filter combinations (m3r3, m1r1, m1r3, m3r1) | as R86B | the question of F-531 is the pf1 versus pf3 difference; the Q setting is crossed with it so the effect on each combination is measured, not assumed |",
         "| dump keys | identical to R86B on every arm | write-only; keeping them makes the a300 arms comparable bit for bit with R86B, and gives the frame-1 chain at every level |",
         "| returned per arm | pos trace, short-scan table, state_reference (compact); imu_cov_growth, engagement_imu, exit_status (all arms); state_trace, pose_control chain and budget, residual_dump_state (levels other than 0.3); residual_dump (level 0.0 only) | the position error after every scan, the Q actually applied, the frame-1 state before IMU / after IMU / after LIO, and the residuals at level 0.0 where the effect is largest. State_trace is 2.07 MB per arm, so 12 arms is 24.8 MB: that is why the return cannot be one zip under 30 MB (section 8) |",
         "| not returned | solve matrices, voxel map dumps, support file, frame_stats, lidar_msg_log of any arm | unchanged inputs (maps) or not needed for this question; they stay in the host archive |",
         "| gates | IDENT_R86B x4, ACC_ENGAGED, PREIMU_SAME, SMALLSCANS, STREAM, EXIT, REQUIRED | the a300 arms must reproduce R86B; the key must change Q in the expected direction; pre-IMU state must be unchanged by a Q key |", "", "Arms:", ""]
    for lvl, val in LEVELS:
        for t in TAGS:
            L.append("- `%s`: acc_scale %g%s" % (aid(t, lvl), val, ", control of `%s`" % (R86B + t) if lvl == "a300" else ""))
    open(os.path.join(out, "COMPLIANCE_TABLE.md"), "w").write("\n".join(L) + "\n")
    return jobs, rows


def selftest():
    import tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    man = os.path.join(here, "..", "..", "r86c_manifest")
    bj = json.load(open(os.path.join(man, "base_jobs_r86b_eee_01.json")))
    rd = csv.DictReader(open(os.path.join(man, "base_arm_table_r86b.tsv")), delimiter="\t")
    cols = rd.fieldnames; br = list(rd)
    bs = json.load(open(os.path.join(man, "base_listener_spec_r86b.json")))
    t = tempfile.mkdtemp(prefix="r86c_man_")
    jobs, rows = write_all(t, bj, br, bs, "/root/catkin_ws/livo_recon_results", cols)
    fails = []
    ck = lambda c, m: None if c else fails.append(m)
    ck(len(jobs) == 16 and len(rows) == 16, "16 arms")
    by = {j["job_id"]: j for j in jobs}
    b86 = {j["job_id"][len(R86B):]: j for j in bj}
    for lvl, val in LEVELS:
        for tag in TAGS:
            o = by[aid(tag, lvl)]["overrides"]; ref = b86[tag]["overrides"]
            diff = {k for k in set(o) | set(ref) if o.get(k) != ref.get(k)}
            allowed = {KEY, "eval.state_trace_run_id", "outputs.path", "outputs.debug_log_dir"} if lvl != "a300" else {"eval.state_trace_run_id", "outputs.path", "outputs.debug_log_dir"}
            ck(diff == allowed, "%s differs from R86B %s in %s (expected %s)" % (aid(tag, lvl), tag, sorted(diff), sorted(allowed)))
            ck(o[KEY] == val, "acc_scale value " + aid(tag, lvl))
            ck(by[aid(tag, lvl)]["duration_secs"] == 60 and by[aid(tag, lvl)]["seq"] == "eee_01", "duration/seq")
            ck(o["outputs.path"].startswith("/root/catkin_ws/livo_recon_results/" + QUEUE + "/" + aid(tag, lvl)), "path")
            ck(not any("perturb" in k.lower() for k in o), "no perturb key")
    ck(len({j["overrides"]["outputs.path"] for j in jobs}) == 16, "16 distinct output paths")
    ck(len({j["job_id"] for j in jobs}) == 16, "16 distinct ids")
    rs = list(csv.DictReader(open(os.path.join(t, "arm_table.tsv")), delimiter="\t"))
    ck(len(rs) == 16 and all(r["batch"] == QUEUE and r["tier"] == "EF" for r in rs), "arm table")
    ck(all("residual_dump.csv" in r["requires"].split() for r in rs), "scorer demands the dump files on every arm (same as R86B)")
    req = json.load(open(os.path.join(t, "required_outputs.json")))
    def arms_of(f, where):
        return {a for x in req if x["file"] == f and x["where"] == where for a in x["arms"]}
    z = {aid(tg, "a000") for tg in TAGS}
    ck(arms_of("residual_dump.csv.gz", "archive") == z, "residual_dump returned for a000 only")
    ck(arms_of("state_trace.csv.gz", "archive") == {aid(tg, l) for l, _ in LEVELS if l != "a300" for tg in TAGS}, "state_trace for the 12 non-control arms")
    ck(arms_of("imu_cov_growth.csv.gz", "archive") == set(by), "imu_cov_growth for all 16")
    ck(arms_of("pos_trace_all.csv.gz", "return") == set(by), "pos trace for all 16")
    ck(not any(x["file"] in ("pose_control_first_frame_solve_matrices.txt.gz", "post_calibration_voxelmap.csv.gz") for x in req), "no solve matrices or map dumps returned")
    sp = json.load(open(os.path.join(t, "listener_spec_r86c.json")))
    ck(sp["queues"][0]["expected_jobs"] == 16 and sp["queues"][0]["dir"].endswith(QUEUE), "listener queue")
    ck("r86b" not in json.dumps(sp).lower().replace("r86b_rt", ""), "listener spec has no r86b path left")
    ck("exit_status.txt" not in sp["raw"]["delete"] and all(f in sp["raw"]["delete"] for f in ("imu_cov_growth.csv", "state_trace.csv", "residual_dump.csv")), "listener deletes/archives the needed names")
    ctl = json.load(open(os.path.join(t, "controls.json")))
    ck(sorted(ctl) == sorted(aid(tg, "a300") for tg in TAGS), "controls.json")
    print("SELFTEST make_r86c_manifest", ("FAIL " + "; ".join(fails)) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-jobs"); ap.add_argument("--base-table"); ap.add_argument("--base-spec"); ap.add_argument("--out"); ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    rd = csv.DictReader(open(a.base_table), delimiter="\t")
    jobs, rows = write_all(a.out, json.load(open(a.base_jobs)), list(rd), json.load(open(a.base_spec)), a.results_root, rd.fieldnames)
    print("wrote %d arms, 1 queue (%s)" % (len(jobs), QUEUE))
