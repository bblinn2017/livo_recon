#!/usr/bin/env python3
"""make_r86_manifest.py -- R86 (planner's script; never run against ROS by the planner, rule 1.1).
Builds the four R86 arms from the shipped R85 cell-A job (r86_manifest/base_job_r85_A.json, copied verbatim from R85 jobs_r85_eee_01.json):
  eee_01, edge 0.25 m, first 60 s, retime_trim, debiased, lines on, motion-Q, layer 0, downsampling first (leaf 0.1), neighborhood_size 1.
The four arms differ ONLY in which decimation feeds the voxel map and which feeds the IEKF residuals:
  m3r3  map point filter 3, residual point filter 3   (stream primary)     control of r85_rt_sht_eee_01_v0p25_A
  m1r1  map point filter 1, residual point filter 1   (stream primary)     control of r85_rt_sht_eee_01_v0p25_B
  m1r3  map point filter 1, residual point filter 3   (stream second, dry_run_point_filter_num 3)
  m3r1  map point filter 3, residual point filter 1   (stream second, dry_run_point_filter_num 1)
Every arm adds the same write-only dump keys (nothing else changes). Usage:
  python3 tools/r86/make_r86_manifest.py --base r86_manifest/base_job_r85_A.json --base-row r86_manifest/base_arm_row_r85_A.json --out r86_manifest --results-root /root/catkin_ws/livo_recon_results
  python3 tools/r86/make_r86_manifest.py --selftest
"""
import argparse, copy, csv, json, os, sys

BAG, EDGE, QUEUE = "eee_01", "0p25", "r86_eee_01"
FRAMES = ",".join(str(i) for i in range(1, 11))
SNAPS = ",".join(str(i) for i in range(2, 11))
DUMP = {"lio.residual_dump_frames": FRAMES, "lio.residual_dump_voxel_info": True, "lio.residual_dump_last_iter": True,
        "lio.first_frame_dump_frames": 10, "lio.update_diag_en": True,
        "voxel_map.map.snapshot_post_calibration": True, "voxel_map.map.snapshot_frames": SNAPS}
ARMS = [  # (tag, map pf, residual pf, control arm in R85)
    ("m3r3", 3, 3, "r85_rt_sht_eee_01_v0p25_A"),
    ("m1r1", 1, 1, "r85_rt_sht_eee_01_v0p25_B"),
    ("m1r3", 1, 3, ""),
    ("m3r1", 3, 1, ""),
]
DUMP_FILES = ["residual_dump.csv", "residual_dump_state.csv", "lio_update_diag.csv", "post_calibration_voxelmap.csv",
              "pose_control_first_frame_solve_matrices.txt"] + ["voxelmap_snapshot_f%d.csv" % i for i in range(2, 11)]
ARCHIVE_FILES = DUMP_FILES + ["state_trace.csv", "frame_stats.txt", "lidar_msg_log.csv", "line_rows_per_scan.csv", "engagement_imu.txt", "exit_status.txt", "plane_fit_reasons.csv"]
ASKED = ("Bryce (2026-10-08): 'i think we can just test one of the bags and determine what the difference is between frame 1-10 residuals with pfn1 and pfn3. we can run 4 versions of the experiment. "
         "one where we do combos of res points and map points at pfn1/3 to get 4 runs with dumps.' and 'let's build this and just this part as the next round. i approve.'")


def arm_id(tag):
    return "r86_rt_sp_%s_v%s_%s" % (BAG, EDGE, tag)


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
                 dump_frames=FRAMES, asked=ASKED, src_arm="r85_rt_sht_eee_01_v0p25_A (cell A job) with map/residual decimation and dump keys per the R86 design",
                 notes="R86 map-point / residual-point split: map point filter %d, residual point filter %d, eee_01, edge 0.25, first 60 s, layer 0, dumps for frames 1 to 10" % (mpf, rpf),
                 duration_s="60", seq=BAG, bag=BAG, edge="0.25", layer="0", l2_ref="", l2_source="")
        rows.append(r)
    return jobs, rows


def listener_spec(base_spec, results_root):
    s = copy.deepcopy(base_spec)
    s["round"] = "R86-eee_01"
    s["queues"] = [{"dir": "ablations/_job_queue/" + QUEUE, "results_root": "%s/%s" % (results_root, QUEUE), "expected_jobs": len(ARMS)}]
    for st in s["stages"]:
        c = st["cmd"]
        for i, x in enumerate(c):
            if x == "src/livo_recon/r85_manifest/arm_table.tsv":
                c[i] = "src/livo_recon/r86_manifest/arm_table.tsv"
            if x.endswith("listener_spec_eee_01.json"):
                c[i] = "{spec_dir}/listener_spec_r86.json"
    for v in s.get("verify", []):
        v["cmd"] = ["{spec_dir}/listener_spec_r86.json" if x.endswith("listener_spec_eee_01.json") else x for x in v["cmd"]]
    d = s["raw"]["delete"]
    for f in DUMP_FILES + ["lidar_msg_log.csv"]:
        if f not in d:
            d.append(f)
    s["raw"]["delete"] = d
    return s


def required(arms):
    R = []
    def add(title, file, where="return", source="arm", cols=None, min_rows=1, maxb=60000000, ar=None):
        R.append(dict(followup="F-531", title=title, arms=list(arms if ar is None else ar), file=file, columns=cols or [], min_rows=min_rows, where=where, source=source,
                      contains={}, expected_bytes=0, max_bytes_per_file=maxb))
    add("per-scan position, yaw and quaternion", "pos_trace_all.csv.gz", cols=["scan_id", "t_abs", "t_rel_s", "px", "py", "pz", "yaw_rad", "qw", "qx", "qy", "qz"], min_rows=500, maxb=330000)
    add("per-scan residual and voxel-count table (r85_short_reduce.py)", "short_scan_all.csv.gz", cols=["scan", "t_s", "residual_count", "fs_n_residuals", "fs_n_voxels", "fs_n_voxels_is_plane", "fs_iters", "fs_dx_pos_mm", "fs_nis", "fs_reduced_chi2", "pos_err_mm"], min_rows=500, maxb=200000)
    add("state_reference.txt (first-frame reference)", "state_reference.txt", maxb=4000)
    add("engagement of the four arms concatenated", "engagement_all.txt", source="generated", maxb=2000000, ar=[])
    add("per-arm completeness", "completeness.csv", source="generated", cols=["arm", "requires_missing"], min_rows=4, maxb=100000, ar=[])
    for f in DUMP_FILES + ["state_trace.csv", "frame_stats.txt", "lidar_msg_log.csv"]:
        add("lossless gzip archive of %s (host archive, byte-for-byte)" % f, f + ".gz", where="archive", maxb=400000000)
    add("lossless gzip archive of engagement_imu.txt", "engagement_imu.txt.gz", where="archive", maxb=400000)
    add("lossless gzip archive of exit_status.txt (all four must be 0)", "exit_status.txt.gz", where="archive", maxb=40000)
    add("archive index (raw bytes, raw sha256, rows)", "archive_index.json", where="archive", maxb=100000)
    add("archive sha256 manifest", "archive.sha256", where="archive", maxb=100000)
    for f, t in (("r86_gates.txt", "gate results"), ("r86_controls.csv", "runs m3r3 and m1r1 against the shipped R85 cell A and B traces"),
                 ("r86_stream_engagement.csv", "stream engagement counters per arm (frames, second, fallback, mean points for map and for the IEKF)"),
                 ("r86_trim_first_frames.csv", "lidar_msg_log rows of the first 15 LiDAR messages for each arm (n_trimmed, verdict)"),
                 ("tree_check.txt", "tested tree identity against R85_tree.tar.gz"), ("R86_REPORT.md", "report of the coding agent (facts only)"),
                 ("code_answers.md", "answers to the open items of section 9"), ("harness_answers.md", "harness answers")):
        add(t, f, source="generated", ar=[], maxb=2000000)
    return R


def write_all(out, base, base_row, base_spec, results_root):
    jobs, rows = build(base, base_row, results_root)
    os.makedirs(out, exist_ok=True)
    json.dump(jobs, open(os.path.join(out, "jobs_%s.json" % QUEUE), "w"), indent=1)
    json.dump(listener_spec(base_spec, results_root), open(os.path.join(out, "listener_spec_r86.json"), "w"), indent=1)
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
        f.write("# listener spec\tqueue\tarms\nlistener_spec_r86.json\t%s\t%d arms\n" % (QUEUE, len(ARMS)))
    lines = ["# R86 compliance table (what each choice is for, rule 4.19)", "",
             "| item | value | what it is for |", "|---|---|---|",
             "| bag, edge, window | eee_01, 0.25 m, first 60 s | the same short cell as R85 A and B, so runs 1 and 2 are controls against returned data |",
             "| arm type | retime_trim, debiased, lines on, motion-Q, layer 0, downsampling first, neighborhood 1 | unchanged from R85 cell A; only the decimation of the two point sets varies |",
             "| lio/residual_dump_frames 1..10, voxel_info, last_iter | per-residual dump at iteration 0 and the last iteration | residual sets, voxel point counts and distinct frames per residual (write-only) |",
             "| lio/first_frame_dump_frames 10, lio/update_diag_en | solve matrices of frames 1..10, one row per frame and iteration | information matrix and step per iteration (write-only) |",
             "| snapshot_post_calibration, snapshot_frames 2..10 | voxel map the frame queries, frames 1 to 10 | plane normal, offset and point count per voxel across the four maps (write-only) |",
             "| data_queues.log_lidar_msgs (already in the cell A job) | per-message log with n_trimmed | shows whether retime_trim trims messages inside frames 1 to 10 (second stream is not trimmed, SUSPECTED) |",
             "| lio/iekf_point_stream second + lio/dry_run_point_filter_num | runs m1r3 and m3r1 only | decouples residual points from map points |", "",
             "Arms:", ""]
    for t, mpf, rpf, c in ARMS:
        lines.append("- `%s`: map point filter %d, residual point filter %d%s" % (arm_id(t), mpf, rpf, ", control of `%s`" % c if c else ""))
    open(os.path.join(out, "COMPLIANCE_TABLE.md"), "w").write("\n".join(lines) + "\n")
    return jobs, rows


def selftest():
    import tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    man = os.path.join(here, "..", "..", "r86_manifest")
    base = json.load(open(os.path.join(man, "base_job_r85_A.json")))
    row = json.load(open(os.path.join(man, "base_arm_row_r85_A.json")))
    bs = json.load(open(os.path.join(man, "base_listener_spec_r85.json")))
    t = tempfile.mkdtemp(prefix="r86_man_")
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
    ck(all("residual_dump.csv" in r["requires"].split() and "voxelmap_snapshot_f10.csv" in r["requires"].split() for r in rs), "requires")
    req = json.load(open(os.path.join(t, "required_outputs.json")))
    ck(any(x["file"] == "residual_dump.csv.gz" and x["where"] == "archive" for x in req), "required archive dump")
    sp = json.load(open(os.path.join(t, "listener_spec_r86.json")))
    ck(sp["queues"][0]["expected_jobs"] == 4 and "voxelmap_snapshot_f10.csv" in sp["raw"]["delete"] and "residual_dump.csv" in sp["raw"]["delete"], "listener spec")
    ck(not any("eee_01.json" in str(x) for st in sp["stages"] for x in st["cmd"]), "listener spec points at the r86 spec")
    print("SELFTEST make_r86_manifest", ("FAIL " + "; ".join(fails)) if fails else "PASS")
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
