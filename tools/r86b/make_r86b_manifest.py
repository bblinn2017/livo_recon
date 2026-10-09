#!/usr/bin/env python3
"""make_r86b_manifest.py -- R86B addendum: the R86A runs repeated with the state, covariance, residual and map dumps of frame 1 (planner's script; never run against ROS by the planner, rule 1.1).
Builds the four R86B arms from the shipped R85 cell-A job (r86b_manifest/base_job_r85_A.json, copied verbatim from R85 jobs_r85_eee_01.json):
  eee_01, edge 0.25 m, first 60 s, retime_trim, debiased, lines on, motion-Q, layer 0, downsampling first (leaf 0.1), neighborhood_size 1.
The four arms differ ONLY in which decimation feeds the voxel map and which feeds the IEKF residuals:
  m3r3  map point filter 3, residual point filter 3   (stream primary)     control of r85_rt_sht_eee_01_v0p25_A
  m1r1  map point filter 1, residual point filter 1   (stream primary)     control of r85_rt_sht_eee_01_v0p25_B
  m1r3  map point filter 1, residual point filter 3   (stream second, dry_run_point_filter_num 3)
  m3r1  map point filter 3, residual point filter 1   (stream second, dry_run_point_filter_num 1)
Every arm adds the same write-only dump keys (nothing else changes; R86B adds to R86A: eval.state_trace_pre_imu_frames, lio.residual_dump_all_iters, voxel_map.map.snapshot_frames "2"; drops lio.residual_dump_last_iter). Usage:
  python3 tools/r86b/make_r86b_manifest.py --base r86b_manifest/base_job_r85_A.json --base-row r86b_manifest/base_arm_row_r85_A.json --out r86b_manifest --results-root /root/catkin_ws/livo_recon_results
  python3 tools/r86b/make_r86b_manifest.py --selftest
"""
import argparse, copy, csv, json, os, sys

BAG, EDGE, QUEUE = "eee_01", "0p25", "r86b_eee_01"
FRAMES = "1"
DUMP = {"lio.residual_dump_frames": FRAMES, "lio.residual_dump_voxel_info": True, "lio.residual_dump_all_iters": True,
        "lio.first_frame_dump_frames": 1, "voxel_map.map.snapshot_post_calibration": True, "voxel_map.map.snapshot_frames": "2",
        "eval.state_trace_pre_imu_frames": 1}
ARMS = [  # (tag, map pf, residual pf, control arm in R85)
    ("m3r3", 3, 3, "r85_rt_sht_eee_01_v0p25_A"),
    ("m1r1", 1, 1, "r85_rt_sht_eee_01_v0p25_B"),
    ("m1r3", 1, 3, ""),
    ("m3r1", 3, 1, ""),
]
# files the scorer demands of the arm (arm_table `requires`): only logs whose writers ran in R86 / R86A
DUMP_FILES = ["residual_dump.csv", "residual_dump_state.csv", "post_calibration_voxelmap.csv", "pose_control_first_frame_solve_matrices.txt", "voxelmap_snapshot_f2.csv"]
# existing node logs (written for frame 1 since R19, never archived before R86B) and small calibration files: archived losslessly, required by the gates, NOT demanded by the scorer
# (a wrong assumption about one of them must cost that file, not the arm)
NEW_LOGS = ["pose_control_first_frame.csv", "pose_control_first_frame_state_chain.csv", "pose_control_first_frame_covariance_budget.csv", "pose_control_first_frame_covariance_budget.txt"]
SMALL_FILES = ["calibration_p0.txt", "gravity_alignment.txt", "imu_first_scans.csv", "bootstrap_preparation.txt", "bootstrap_map_seed.txt"]
ARCHIVE_FILES = DUMP_FILES + NEW_LOGS + SMALL_FILES + ["post_calibration_voxel_observation_support.csv", "state_trace.csv", "frame_stats.txt", "imu_cov_growth.csv", "lidar_msg_log.csv", "engagement_imu.txt", "exit_status.txt"]
ASKED = ("Bryce (2026-10-09): 'Yes, I think we should redo this and we should make sure we have all the relevant information for before and after frame one correction. What I'm thinking is we do before IMU propagation, after IMU propagation, and after LIO correction for any of the state covariance, etc. information. And we would also like to do before frame one, i.e. after calibration, voxel map dump. As well, we would like to dump any of the residual information relevant to the LIO correction. Let's generate this as an addendum. Make sure that it is sound.'; "
         "earlier: 'let's only dump frame 1, not all 10 frames'; "
         "earlier: 'i think we can just test one of the bags and determine what the difference is between frame 1-10 residuals with pfn1 and pfn3. we can run 4 versions of the experiment. one where we do combos of res points and map points at pfn1/3 to get 4 runs with dumps.'")


def arm_id(tag):
    return "r86b_rt_sp_%s_v%s_%s" % (BAG, EDGE, tag)


def build(base, base_row, results_root):
    jobs, rows = [], []
    for tag, mpf, rpf, ctl in ARMS:
        j = copy.deepcopy(base)
        aid = arm_id(tag)
        j["job_id"] = aid
        o = j["overrides"]
        assert o["cbk.lidar.point_filter_num"] == 3 and o["voxel_map.map.voxel_size"] == 0.25 and o["voxel_map.map.max_layer"] == 0 and j["duration_secs"] == 60 and j["seq"] == BAG
        o["cbk.lidar.point_filter_num"] = mpf
        if mpf != rpf:
            o["lio.dry_run_point_filter_num"] = rpf
            o["lio.iekf_point_stream"] = "second"
        for k, v in DUMP.items():
            assert k not in o, k
            o[k] = v
        o["eval.state_trace_run_id"] = aid
        path = "%s/%s/%s" % (results_root, QUEUE, aid)
        o["outputs.path"] = path
        o["outputs.debug_log_dir"] = path
        for k in o:
            assert "perturb" not in k.lower(), k
        jobs.append(j)
        r = dict(base_row)
        req = [x for x in base_row["requires"].split()]
        for f in DUMP_FILES:
            if f not in req:
                req.append(f)
        r.update(arm=aid, batch=QUEUE, group="SHT", cell="S", nb="1", requires=" ".join(req), f417="0", plane_check="0", early="0", dump="0", msglog="1",
                 pfn=str(mpf), sampling="map_pf%d_res_pf%d" % (mpf, rpf), arm_kind="sht", control=ctl, same_batch_control=ctl, ref=ctl,
                 dump_frames=FRAMES, asked=ASKED, src_arm="r85_rt_sht_eee_01_v0p25_A (cell A job) with map/residual decimation and dump keys per the R86B design (arms repeat R86A)",
                 notes="R86B map-point / residual-point split: map point filter %d, residual point filter %d, eee_01, edge 0.25, first 60 s, layer 0, dumps for frame 1 only" % (mpf, rpf),
                 duration_s="60", seq=BAG, bag=BAG, edge="0.25", layer="0", l2_ref="", l2_source="")
        rows.append(r)
    return jobs, rows


def listener_spec(base_spec, results_root):
    s = copy.deepcopy(base_spec)
    s["round"] = "R86B-eee_01"
    s["queues"] = [{"dir": "ablations/_job_queue/" + QUEUE, "results_root": "%s/%s" % (results_root, QUEUE), "expected_jobs": len(ARMS)}]
    for st in s["stages"]:
        c = st["cmd"]
        for i, x in enumerate(c):
            if x == "src/livo_recon/r85_manifest/arm_table.tsv":
                c[i] = "src/livo_recon/r86b_manifest/arm_table.tsv"
            if x.endswith("listener_spec_eee_01.json"):
                c[i] = "{spec_dir}/listener_spec_r86b.json"
    for v in s.get("verify", []):
        v["cmd"] = ["{spec_dir}/listener_spec_r86b.json" if x.endswith("listener_spec_eee_01.json") else x for x in v["cmd"]]
    d = s["raw"]["delete"]
    for f in ARCHIVE_FILES:
        if f != "exit_status.txt" and f not in d:   # exit_status.txt is archived by the scorer (EXTRA_ARCHIVE) and never deleted
            d.append(f)
    s["raw"]["delete"] = d
    return s


def required(arms):
    R = []
    def add(title, file, where="return", source="arm", cols=None, min_rows=1, maxb=60000000, ar=None):
        R.append(dict(followup="F-531/F-532", title=title, arms=list(arms if ar is None else ar), file=file, columns=cols or [], min_rows=min_rows, where=where, source=source,
                      contains={}, expected_bytes=0, max_bytes_per_file=maxb))
    add("per-scan position, yaw and quaternion", "pos_trace_all.csv.gz", cols=["scan_id", "t_abs", "t_rel_s", "px", "py", "pz", "yaw_rad", "qw", "qx", "qy", "qz"], min_rows=500, maxb=330000)
    add("per-scan residual and voxel-count table (r85_short_reduce.py)", "short_scan_all.csv.gz", cols=["scan", "t_s", "residual_count", "fs_n_residuals", "fs_n_voxels", "fs_n_voxels_is_plane", "fs_iters", "fs_dx_pos_mm", "fs_nis", "fs_reduced_chi2", "pos_err_mm"], min_rows=500, maxb=200000)
    add("per-message LiDAR log (n_trimmed, verdict)", "lidar_msg_log_all.csv.gz", min_rows=500, maxb=900000)
    add("state_reference.txt (first-frame reference)", "state_reference.txt", maxb=4000)
    add("engagement of the four arms concatenated", "engagement_all.txt", source="generated", maxb=2000000, ar=[])
    add("per-arm completeness", "completeness.csv", source="generated", cols=["arm", "requires_missing"], min_rows=4, maxb=100000, ar=[])
    for f in [x for x in ARCHIVE_FILES if x not in ("engagement_imu.txt", "exit_status.txt")]:
        add("lossless gzip archive of %s (host archive, byte for byte)" % f, f + ".gz", where="archive", maxb=100000000)
    add("lossless gzip archive of engagement_imu.txt", "engagement_imu.txt.gz", where="archive", maxb=400000)
    add("lossless gzip archive of exit_status.txt (all four must be 0)", "exit_status.txt.gz", where="archive", maxb=40000)
    add("archive index (raw bytes, raw sha256, rows)", "archive_index.json", where="archive", maxb=100000)
    add("archive sha256 manifest", "archive.sha256", where="archive", maxb=100000)
    for f, t in (("r86b_gates.txt", "gate results"), ("r86b_identity.csv", "the four arms against the shipped R86A traces (write-only check)"), ("r86b_dump_crosschecks.csv", "cross-checks between independent dumps of the same quantity"), ("r86b_controls.csv", "patched-binary runs m3r3 and m1r1 against the shipped R85 cell A and B traces"),
                 ("r86b_stream_engagement.csv", "stream engagement counters per arm"), ("r86b_dry_run_stream.csv", "[dry_run_stream] counters of the last log line per arm"),
                 ("dry_run_stream_lines.txt", "every [dry_run_stream] line of each arm's run.log, grouped under ### <arm>"),
                 ("r86b_trim_first_frames.csv", "lidar_msg_log rows of the first 15 messages per arm"),
                 ("tree_check.txt", "patched tree: patch applied, diff against R85_tree.tar.gz"), ("agent_edits.patch", "diff of the tested tree against the R85 tree (the R86B patch and every repair)"),
                 ("R86B_tree.tar.gz", "exact tested tree"), ("checks/build_tail.txt", "tail of the catkin build"), ("checks/ctest.txt", "ctest output incl. dump_rules_r86b"),
                 ("R86B_REPORT.md", "report of the coding agent (facts only)"), ("code_answers.md", "answers to the open items of section 8"), ("harness_answers.md", "harness answers")):
        add(t, f, source="generated", ar=[], maxb=8000000)
    return R


def write_all(out, base, base_row, base_spec, results_root):
    jobs, rows = build(base, base_row, results_root)
    os.makedirs(out, exist_ok=True)
    json.dump(jobs, open(os.path.join(out, "jobs_%s.json" % QUEUE), "w"), indent=1)
    json.dump(listener_spec(base_spec, results_root), open(os.path.join(out, "listener_spec_r86b.json"), "w"), indent=1)
    cols = list(base_row.keys())
    with open(os.path.join(out, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump(required([arm_id(t) for t, *_ in ARMS]), open(os.path.join(out, "required_outputs.json"), "w"), indent=1)
    with open(os.path.join(out, "controls.json"), "w") as f:
        json.dump({arm_id(t): c for t, _, _, c in ARMS if c}, f, indent=1)
    with open(os.path.join(out, "queue_order.tsv"), "w") as f:
        f.write("# listener spec\tqueue\tarms\nlistener_spec_r86b.json\t%s\t%d arms\n" % (QUEUE, len(ARMS)))
    lines = ["# R86B compliance table (what each choice is for, rule 4.19)", "",
             "| item | value | what it is for |", "|---|---|---|",
             "| bag, edge, window | eee_01, 0.25 m, first 60 s | the same short cell as R85 A and B and R86A, so every arm can be checked against returned data |",
             "| arm type | retime_trim, debiased, lines on, motion-Q, layer 0, downsampling first, neighborhood 1 | unchanged from R86A; only the decimation of the two point sets varies |",
             "| eval/state_trace_pre_imu_frames 1 (new, patch) | one `pre_imu` row in state_trace.csv for frame 1: full state and the 18x18 covariance before the IMU propagation | with the existing post_imu and post_lio rows: state and covariance before IMU, after IMU, after LIO (Bryce 2026-10-09) |",
             "| existing logs now archived: pose_control_first_frame.csv, _state_chain.csv, _covariance_budget.csv/.txt | state before IMU, after IMU and after every LIO iteration; P before IMU, after IMU, after LIO and Q | the same quantities written by a second, independent writer, so the two can be compared (gates CHAIN and COVBUDGET) |",
             "| calibration_p0.txt, gravity_alignment.txt, imu_first_scans.csv, bootstrap_preparation.txt, bootstrap_map_seed.txt | small files written at calibration | the state, covariance, IMU samples and map seed that frame 1 starts from |",
             "| lio/residual_dump_frames 1, voxel_info, residual_dump_all_iters (new, patch) | per-residual dump of frame 1 at EVERY iteration (R86A: iteration 0 and the last) | residual set, Jacobians, variances and voxel support at each Gauss-Newton step of the frame-1 correction |",
             "| lio/first_frame_dump_frames 1 | solve matrices of frame 1, every iteration | information matrix, gradient, prior and solution per iteration (unchanged from R86A) |",
             "| voxel_map/map/snapshot_post_calibration true | voxel map before frame 1 (the map frame 1 queries) | unchanged from R86A |",
             "| voxel_map/map/snapshot_frames 2 | voxelmap_snapshot_f2.csv: the map after the insertion of frame 1 | map after the frame-1 correction (R86A did not write it) |",
             "| post_calibration_voxel_observation_support.csv | per-observation sums of the post-calibration map | support of the map frame 1 queries (written since R80, archived by the listener, not returned in R86A) |",
             "| data_queues.log_lidar_msgs, imu_cov_growth, frame_stats, state_trace | unchanged or already written | per-message trim log, per-scan covariance growth, per-frame statistics, per-scan state |",
             "| patches/r86b_p1_dumps.patch | write-only: pre_imu trace row, every-iteration residual dump | everything else in R86B is a key or an archive name; the four arms must reproduce R86A (gate IDENT_R86A) |",
             "| not set: lio.update_diag_en, lio.log_debug_en, lio.residual_dump_last_iter | | frame 1 only (Bryce 2026-10-08); the last-iteration dump is superseded by the every-iteration dump |", "",
             "Arms:", ""]
    for t, mpf, rpf, c in ARMS:
        lines.append("- `%s`: map point filter %d, residual point filter %d%s" % (arm_id(t), mpf, rpf, ", control of `%s`" % c if c else ""))
    open(os.path.join(out, "COMPLIANCE_TABLE.md"), "w").write("\n".join(lines) + "\n")
    return jobs, rows


def selftest():
    import tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    man = os.path.join(here, "..", "..", "r86b_manifest")
    base = json.load(open(os.path.join(man, "base_job_r85_A.json")))
    row = json.load(open(os.path.join(man, "base_arm_row_r85_A.json")))
    bs = json.load(open(os.path.join(man, "base_listener_spec_r85.json")))
    t = tempfile.mkdtemp(prefix="r86b_man_")
    jobs, rows = write_all(t, base, row, bs, "/root/catkin_ws/livo_recon_results")
    fails = []
    ck = lambda c, m: None if c else fails.append(m)
    ck(len(jobs) == 4 and len(rows) == 4, "4 arms")
    ref = jobs[0]["overrides"]
    for j in jobs:
        o = j["overrides"]
        diff = {k for k in set(o) | set(ref) if o.get(k) != ref.get(k)}
        allowed = {"cbk.lidar.point_filter_num", "lio.dry_run_point_filter_num", "lio.iekf_point_stream", "eval.state_trace_run_id", "outputs.path", "outputs.debug_log_dir"}
        ck(diff <= allowed, "%s differs outside the allowed keys: %s" % (j["job_id"], sorted(diff - allowed)))
        for k in DUMP:
            ck(o[k] == DUMP[k], "dump key " + k)
        ck(j["duration_secs"] == 60 and j["seq"] == BAG, "duration / seq")
    o = {j["job_id"]: j["overrides"] for j in jobs}
    a = lambda tag: o[arm_id(tag)]
    ck(a("m3r3")["cbk.lidar.point_filter_num"] == 3 and "lio.iekf_point_stream" not in a("m3r3"), "m3r3")
    ck(a("m1r1")["cbk.lidar.point_filter_num"] == 1 and "lio.iekf_point_stream" not in a("m1r1"), "m1r1")
    ck(a("m1r3")["cbk.lidar.point_filter_num"] == 1 and a("m1r3")["lio.dry_run_point_filter_num"] == 3 and a("m1r3")["lio.iekf_point_stream"] == "second", "m1r3")
    ck(a("m3r1")["cbk.lidar.point_filter_num"] == 3 and a("m3r1")["lio.dry_run_point_filter_num"] == 1 and a("m3r1")["lio.iekf_point_stream"] == "second", "m3r1")
    # the controls differ from the R85 cells only by the dump keys
    r85b = json.load(open(os.path.join(man, "ref_job_r85_B.json")))["overrides"]
    for tag, rb in (("m3r3", base["overrides"]), ("m1r1", r85b)):
        d = {k for k in set(a(tag)) | set(rb) if a(tag).get(k) != rb.get(k)}
        d -= {"eval.state_trace_run_id", "outputs.path", "outputs.debug_log_dir"}
        ck(d == set(DUMP), "%s control differs from its R85 cell in more than the dump keys: %s" % (tag, sorted(d ^ set(DUMP))))
    rs = list(csv.DictReader(open(os.path.join(t, "arm_table.tsv")), delimiter="\t"))
    ck(len(rs) == 4 and all(r["cell"] and r["tier"] == "EF" and r["group"] == "SHT" for r in rs), "arm table")
    ck(all("residual_dump.csv" in r["requires"].split() and "post_calibration_voxelmap.csv" in r["requires"].split() for r in rs), "requires")
    req = json.load(open(os.path.join(t, "required_outputs.json")))
    ck(any(x["file"] == "residual_dump.csv.gz" and x["where"] == "archive" for x in req), "required archive dump")
    sp = json.load(open(os.path.join(t, "listener_spec_r86b.json")))
    ck(sp["queues"][0]["expected_jobs"] == 4 and "post_calibration_voxelmap.csv" in sp["raw"]["delete"] and "residual_dump.csv" in sp["raw"]["delete"], "listener spec")
    ck(not any("eee_01.json" in str(x) for st in sp["stages"] for x in st["cmd"]), "listener spec points at the r86b spec")
    ck(all(f not in r["requires"].split() for r in rs for f in NEW_LOGS + SMALL_FILES), "new logs / small files are not demanded by the scorer")
    LIT = ["pose_control_first_frame.csv", "pose_control_first_frame_state_chain.csv", "pose_control_first_frame_covariance_budget.csv", "pose_control_first_frame_covariance_budget.txt",
           "calibration_p0.txt", "gravity_alignment.txt", "imu_first_scans.csv", "bootstrap_preparation.txt", "bootstrap_map_seed.txt",
           "voxelmap_snapshot_f2.csv", "post_calibration_voxel_observation_support.csv", "imu_cov_growth.csv"]   # literal on purpose: the check must not read the generator's own lists
    ck(all(f in sp["raw"]["delete"] for f in LIT), "listener deletes (and so archives) every new name")
    ck(all(any(x["file"] == f + ".gz" and x["where"] == "archive" for x in req) for f in LIT), "every new name is a required archive file")
    ck("exit_status.txt" not in sp["raw"]["delete"], "exit_status.txt stays out of raw.delete")
    ck(all(o[k] == v for o in [j["overrides"] for j in jobs] for k, v in (("lio.residual_dump_all_iters", True), ("eval.state_trace_pre_imu_frames", 1), ("voxel_map.map.snapshot_frames", "2"))), "new dump keys on every arm")
    ck(all("lio.residual_dump_last_iter" not in j["overrides"] for j in jobs), "last_iter is not set (superseded)")
    ck(any(x["file"] == "pose_control_first_frame_covariance_budget.txt.gz" and x["where"] == "archive" for x in req) and any(x["file"] == "voxelmap_snapshot_f2.csv.gz" for x in req), "required archive names")
    print("SELFTEST make_r86b_manifest", ("FAIL " + "; ".join(fails)) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base"); ap.add_argument("--base-row"); ap.add_argument("--base-spec"); ap.add_argument("--out"); ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    jobs, rows = write_all(a.out, json.load(open(a.base)), json.load(open(a.base_row)), json.load(open(a.base_spec)), a.results_root)
    print("wrote %d arms, 1 queue (%s)" % (len(jobs), QUEUE))
