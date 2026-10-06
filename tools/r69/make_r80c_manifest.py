#!/usr/bin/env python3
"""R80C (F-484): job files, arm_table.tsv, listener specs, required_outputs.json, size ledger and compliance table for the retime addendum.

    python3 tools/r69/make_r80c_manifest.py --jobs r69_inputs/r66_jobs.json --grid r80c_grid.json --out r80c_manifest --results-root /root/catkin_ws/livo_recon_results

Planning agent's script; NOT run against the real host by me (a flow test against a stub job file only: 72 arms). Reuses the R80 generator's arm builders (make_r80_manifest as R80): the
36 BASE arms of R80 (BASE3 fixed-Q r80_mg_*, BASE3 motion-Q r80_b3_*, BASE1 fixed-Q and motion-Q r80_b1_*, nine bags) are cloned UNCHANGED (same base job, same overrides) and get only:
    cbk.lidar.overlap_policy = retime | retime_trim, cbk.lidar.retime_t0_s / retime_period_s / retime_seq0 (per bag, from r80c_grid.json), data_queues.log_lidar_msgs = true (write-only).
36 arms x 2 policies = 72 arms, nine per-bag queues of 8 arms (rule 1.16), batch name c_<bag> (results under <root>/r80_c_<bag>/<arm>).
Bryce, 2026-10-05: 'i say we generate another r80 addendum that implements the retime fix and re-records baselines over all 9 bags so we can compare';
  'i'd like to see what both retime and retime + trim do. let's try both and see.'; baselines: 'pfn 3 and pfn 1, both Q (36 arms)'.
"""
import argparse, copy, csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_r80_manifest as R80  # noqa: E402

R9, M = R80.R9, R80.M
BAGS = R80.BAGS
POLICIES = (("rt", "retime"), ("rtt", "retime_trim"))
SRC_PREFIX = ("r80_mg_", "r80_b3_", "r80_b1_")
MSG_COLS_C = R80.MSG_COLS + ["retime_shift_s", "n_trimmed"]
ASK_C = ("Bryce (2026-10-05): 'i say we generate another r80 addendum that implements the retime fix and re-records baselines over all 9 bags so we can compare'; "
         "'i'd like to see what both retime and retime + trim do. let's try both and see.'; baselines 'pfn 3 and pfn 1, both Q (36 arms)'")


def arms(grid):
    out = []
    src = [a for a in R80.arms() if a["id"].startswith(SRC_PREFIX)]
    assert len(src) == 36, len(src)
    for a in src:
        g = grid[a["bag"]]
        for tag, pol in POLICIES:
            b = copy.deepcopy(a)
            b["id"] = "r80c_%s_%s" % (tag, a["id"][len("r80_"):])
            b["batch"] = "c_" + a["bag"]
            d = dict(b["delta"])
            d.update({"cbk.lidar.overlap_policy": pol, "cbk.lidar.retime_t0_s": g["t0_s"], "cbk.lidar.retime_period_s": g["period_s"], "cbk.lidar.retime_seq0": g["seq0"],
                      "data_queues.log_lidar_msgs": True})
            b["delta"] = d
            b.update(kind="fb", ref=None, msglog=1, win=0, f417=0, dump=0, early=0)
            b["asked"] = ASK_C + "; same arm as %s (whole bag, 0.425 m, lines off, point filter %d, %s Q) with cbk/lidar/overlap_policy = %s" % (a["id"], a["pfn"], R80.Q_NAME[a["q"]], pol)
            b["purpose"] = "R80C F-484: %s, %s" % (pol, a["id"][len("r80_"):])
            b["src_arm"] = a["id"]; b["policy"] = pol
            out.append(b)
    return out


def required_outputs(table):
    n = {t["arm"]: R9.N_SCANS[t["bag"]] for t in table}
    allarm = [t["arm"] for t in table]
    R = []

    def add(fu, title, arms_, file, cols, min_rows, where="return", source="arm", eb=None, maxb=None):
        eb = eb if eb is not None else (sum(eb_f(a) for a in arms_) if (where == "return" and arms_) else 0)
        R.append(dict(followup=fu, title=title, arms=arms_, file=file, columns=cols, min_rows=min_rows, where=where, source=source, contains={}, expected_bytes=int(eb),
                      max_bytes_per_file=maxb or 400000))
    eb_f = lambda a: 100 * n[a]
    add("F-484", "per-message LiDAR log, every row, with the two R80C columns retime_shift_s (scan time used minus header + offset) and n_trimmed (points removed before the push)", allarm,
        "lidar_msg_log_all.csv.gz", MSG_COLS_C, 1500, maxb=900000)
    eb_f = lambda a: 56 * n[a]
    add("F-484", "per-scan position, yaw and quaternion, every arm (comparison with the R80 BASE arms of the same name)", allarm, "pos_trace_all.csv.gz", R80.POS_COLS, 1500, maxb=330000)
    add("F-484", "gates of R80C: grid identities, discard counts per policy, trim counts, row counts, exit status", [], "r80c_gates.txt", [], 1, source="generated")
    add("F-409", "engagement of every dispatched arm concatenated (includes the [lidar_overlap_policy] lines)", [], "engagement_all.txt", [], 10, source="generated")
    add("F-409", "per-arm completeness (arm_table 'requires' column)", [], "completeness.csv", ["arm", "requires_missing"], 10, source="generated")
    add("F-326", "exit_status.txt of every dispatched arm: the real node exit status, zero for all", allarm, "exit_status.txt", [], 1, where="archive")
    add("F-407", "engagement_imu.txt present for every dispatched arm", allarm, "engagement_imu.txt", [], 1, where="archive")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--grid", required=True); ap.add_argument("--out", default="r80c_manifest")
    ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--min-free-gb", type=float, default=150.0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    J = {j["job_id"]: j for j in json.load(open(a.jobs))}
    grid = json.load(open(a.grid))
    os.makedirs(a.out, exist_ok=True)
    out = {}; table = []; seen = set()
    for r in arms(grid):
        if r["base"] not in J:
            sys.exit("base job %s not in r66_jobs.json" % r["base"])
        assert r["id"] not in seen, r["id"]; seen.add(r["id"])
        j = copy.deepcopy(J[r["base"]]); j["job_id"] = r["id"]
        o = j["overrides"]
        for k, v in r["delta"].items():
            o[k] = v
        if o.get("estimator.mode") == "decoupled":
            o["estimator.mode"] = "iekf"
        o.pop("spline.mode", None)
        for k in o:
            if "perturb" in k.lower():
                sys.exit("perturbation key forbidden: " + k)
        o["eval.state_trace_run_id"] = r["id"]
        path = "%s/r80_%s/%s" % (a.results_root, r["batch"], r["id"])
        o["outputs.path"] = path; o["outputs.debug_log_dir"] = path
        out.setdefault(r["batch"], []).append(j)
        sampling = "pfn%s_ds%s" % (o.get("cbk.lidar.point_filter_num", "def"), o.get("imu.ds.mode", "def"))
        table.append(dict(arm=r["id"], tier=r["tier"], arch="iekf_raw", kind="closed", duration_s=str(j.get("duration_secs", "full")), seq=j["seq"], notes=r["purpose"], keep_raw="0", dup_of="",
                          r66_id=r["base"], group=r["group"], batch="r80_" + r["batch"], sampling=sampling, requires=" ".join(R80.req_of(o)), f417=str(r["f417"]), plane_check="0",
                          asked=r["asked"], early=str(r["early"]), pfn=str(r["pfn"]), pe="0", dump=str(r["dump"]), bag=r["bag"], win=str(r["win"]), msglog=str(r["msglog"])))
    for b, lst in out.items():
        json.dump(lst, open(os.path.join(a.out, "jobs_%s.json" % b), "w"), indent=1)
    cols = ["arm", "tier", "arch", "kind", "duration_s", "seq", "notes", "keep_raw", "dup_of", "r66_id", "group", "batch", "sampling", "requires", "f417", "plane_check", "asked", "early", "pfn", "pe", "dump", "bag", "win", "msglog"]
    with open(os.path.join(a.out, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t"); w.writeheader(); w.writerows(table)
    q = lambda b, n: {"dir": "ablations/_job_queue/r80_%s" % b, "results_root": "%s/r80_%s" % (a.results_root, b), "expected_jobs": n}
    order = []
    for b in BAGS:
        nm = "c_" + b
        sp = M.listener_spec("R80C-%s" % nm, [q(nm, len(out[nm]))], a.results_root, a.min_free_gb, a.workers)
        fn = "listener_spec_%s.json" % nm
        for blk in sp["stages"] + sp["verify"]:
            blk["cmd"] = [c.replace("%s", fn).replace("r74_manifest", "r80c_manifest") for c in blk["cmd"]]
        sp["raw"]["delete"] = sp["raw"]["delete"] + ["iekf_factor_iteration.csv", "lidar_msg_log.csv", "post_calibration_voxel_observation_support.csv", "line_rows_per_scan.csv"]
        json.dump(sp, open(os.path.join(a.out, fn), "w"), indent=1)
        order.append("%s\t%s\t%d arms" % (fn, nm, len(out[nm])))
    open(os.path.join(a.out, "queue_order.tsv"), "w").write("# listener spec\tqueue\tarms   (a preference, not a dependency: rule 1.16)\n" + "\n".join(order) + "\n")
    RO = required_outputs(table)
    json.dump(RO, open(os.path.join(a.out, "required_outputs.json"), "w"), indent=1)
    json.dump(R80.ledger(RO), open(os.path.join(a.out, "r80c_file_size_ledger.json"), "w"), indent=1)
    with open(os.path.join(a.out, "COMPLIANCE_TABLE.md"), "w") as f:
        f.write("# R80C compliance table (rule 1.10 item 2, 1.12)\n\n| arm | Bryce's words | policy | same as R80 arm | mark |\n|---|---|---|---|---|\n")
        for t in table:
            r = [x for x in arms(grid) if x["id"] == t["arm"]][0]
            f.write("| %s | %s | %s | %s | MATCHES |\n" % (t["arm"], "'implements the retime fix and re-records baselines over all 9 bags'; 'try both'; 'pfn 3 and pfn 1, both Q (36 arms)'", r["policy"], r["src_arm"]))
    cnt = {}
    for t in table:
        cnt[t["group"]] = cnt.get(t["group"], 0) + 1
    print("arms %d: %s; queues %d; listener specs %d" % (len(table), ", ".join("%s %d" % kv for kv in sorted(cnt.items())), len(out), len(BAGS)))


if __name__ == "__main__":
    main()
