#!/usr/bin/env python3
"""R80D (F-484, F-489, F-491): job files, arm_table.tsv, listener specs, required_outputs.json, size ledger and compliance table for the third-policy / offset-sweep / eee_03 addendum.

    python3 tools/r69/make_r80d_manifest.py --jobs r69_inputs/r66_jobs.json --grid-retime r80c_grid.json --grid-bag grid_check/r80d_grid_bag.json --out r80d_manifest \
        --results-root /root/catkin_ws/livo_recon_results

--grid-retime : the R80C grid file exactly as shipped (used for the R80C-grid retime arms of the eight bags that ran in R80C, so the centre arms of the offset sweep are bit-identical to R80C).
--grid-bag    : {bag: r80d_fit_grid output} for all nine bags, fitted from the bags by the coding agent (supplies the restamp grid for all bags and the retime grid for eee_03).
170 arms, nine per-bag queues d_<bag> (rule 1.16), results under <root>/r80_d_<bag>/<arm>:
  RS  36  r80d_rs_<src>                      overlap_policy restamp; <src> = the 36 R80 BASE arms (mg_<bag>_fq, b3_<bag>_mq, b1_<bag>_fq, b1_<bag>_mq; nine bags)
  OFF 126 r80d_off_<mgfq|b3mq>_<bag>_o<MMM>  overlap_policy retime_trim on the retime grid, cbk.lidar.time_offset = -MMM/1000 s, MMM in 115 110 105 100 095 090 085; src = mg_<bag>_fq (pfn 3 fixed-Q) and
                                             b3_<bag>_mq (pfn 3 motion-Q), nine bags
  E3  8   r80d_e3_<rt|rtt>_<src>             the eee_03 arms R80C did not run: overlap_policy retime | retime_trim, the four settings
Every arm is the R80 BASE arm of the same src unchanged (whole bag, 0.425 m, lines off) plus: cbk.lidar.overlap_policy, retime_t0_s / retime_period_s / retime_seq0 (grid), retime_ref_offset_s = -0.1
(the offset baked into t0_s), data_queues.log_lidar_msgs = true; OFF arms also cbk.lidar.time_offset.
Bryce, 2026-10-05 (see ASK_*): third policy restamp = 'restamp only' (36 arms); offset sweep 'Yes, all 9 bags' (126 arms); fixes 1, 5, 6: 'Yes, all three'.
"""
import argparse, copy, csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_r80_manifest as R80  # noqa: E402

R9, M = R80.R9, R80.M
BAGS = R80.BAGS
SRC_PREFIX = ("r80_mg_", "r80_b3_", "r80_b1_")
MSG_COLS_C = R80.MSG_COLS + ["retime_shift_s", "n_trimmed"]
REF_OFFSET = -0.1
OFFSETS_MMM = (115, 110, 105, 100, 95, 90, 85)          # time_offset = -MMM / 1000 s
SWEEP_SRC = (("mgfq", "mg_{bag}_fq"), ("b3mq", "b3_{bag}_mq"))
ASK_D1 = ("Bryce (2026-10-05): 'i would like it as a third policy for comparison'; answer to 'Which restamp arms do you want?' = 'restamp only' (36 arms: nine bags x four settings, discard rule unchanged)")
ASK_D2 = ("Bryce (2026-10-05): 'let's do another addendum before we settle this'; answer to the time-offset sweep question = 'Yes, all 9 bags' (retime_trim grid, pfn 3 fixed-Q and motion-Q, 7 offsets, 126 arms)")
ASK_D3 = ("Bryce (2026-10-05): answer to 'Fixes 1, 5 and 6' = 'Yes, all three'; fix 1 = the 8 missing eee_03 arms (F-489, four settings, retime and retime_trim, grid fitted from the bag)")


def src_arms():
    src = [a for a in R80.arms() if a["id"].startswith(SRC_PREFIX)]
    assert len(src) == 36, len(src)
    return {a["id"][len("r80_"):]: a for a in src}


def make(a, arm_id, pol, g, extra, asked, purpose, group):
    b = copy.deepcopy(a)
    b["id"] = arm_id; b["batch"] = "d_" + a["bag"]; b["group"] = group
    d = dict(b["delta"])
    d.update({"cbk.lidar.overlap_policy": pol, "cbk.lidar.retime_t0_s": g["t0_s"], "cbk.lidar.retime_period_s": g["period_s"], "cbk.lidar.retime_seq0": g["seq0"],
              "cbk.lidar.retime_ref_offset_s": REF_OFFSET, "data_queues.log_lidar_msgs": True})
    d.update(extra)
    b["delta"] = d
    b.update(kind="fb", ref=None, msglog=1, win=0, f417=0, dump=0, early=0)
    b["asked"] = asked + "; same arm as %s (whole bag, 0.425 m, lines off, point filter %d, %s Q) with cbk/lidar/overlap_policy = %s%s" % (a["id"], a["pfn"], R80.Q_NAME[a["q"]], pol, (", " + str(extra)) if extra else "")
    b["purpose"] = purpose
    b["src_arm"] = a["id"]; b["policy"] = pol; b["fam"] = group
    return b


def arms(grid_retime, grid_bag):
    S = src_arms(); out = []
    gr = lambda bag: grid_bag[bag]["retime"] | {"seq0": grid_bag[bag]["seq0"]} if bag == "eee_03" else grid_retime[bag]      # eee_03 from the bag fit, the eight R80C bags exactly as shipped
    gs = lambda bag: grid_bag[bag]["restamp"]
    for s, a in S.items():
        out.append(make(a, "r80d_rs_" + s, "restamp", gs(a["bag"]), {}, ASK_D1, "R80D F-484: restamp (third policy), %s" % s, "RS"))
    for bag in BAGS:
        for tag, tmpl in SWEEP_SRC:
            s = tmpl.format(bag=bag); a = S[s]
            for mmm in OFFSETS_MMM:
                out.append(make(a, "r80d_off_%s_%s_o%03d" % (tag, bag, mmm), "retime_trim", gr(bag), {"cbk.lidar.time_offset": round(-mmm / 1000.0, 3)}, ASK_D2,
                                "R80D F-491: time_offset -%.3f s, retime_trim grid, %s" % (mmm / 1000.0, s), "OFF"))
    for pol, tag in (("retime", "rt"), ("retime_trim", "rtt")):
        for s, a in S.items():
            if a["bag"] == "eee_03":
                out.append(make(a, "r80d_e3_%s_%s" % (tag, s), pol, gr("eee_03"), {}, ASK_D3, "R80D F-489: %s, %s" % (pol, s), "E3"))
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
    add("F-484", "per-message LiDAR log, every row, 17 columns incl. retime_shift_s and n_trimmed", allarm, "lidar_msg_log_all.csv.gz", MSG_COLS_C, 1500, maxb=900000)
    eb_f = lambda a: 56 * n[a]
    add("F-484", "per-scan position, yaw and quaternion, every arm (comparison with the R80 and R80C arms of the same src)", allarm, "pos_trace_all.csv.gz", R80.POS_COLS, 1500, maxb=330000)
    eb_f = lambda a: 16000
    add("F-484", "collector by-product per arm, now ledgered (R80C listed it UNLEDGERED)", allarm, "line_rows_per_scan_all.csv.gz", [], 1, maxb=40000)
    add("F-484", "gates of R80D: grid identities per family, discard counts, trim counts, offset term, restamp line, exit status", [], "r80d_gates.txt", [], 1, source="generated")
    add("F-484", "alignment-free stationary drift and departures per arm (r80d_alignfree.py)", [], "alignfree_summary.csv", [], 100, source="generated")
    add("F-409", "engagement of every dispatched arm concatenated", [], "engagement_all.txt", [], 10, source="generated")
    add("F-409", "per-arm completeness", [], "completeness.csv", ["arm", "requires_missing"], 10, source="generated")
    add("F-326", "exit_status.txt of every dispatched arm: zero for all", allarm, "exit_status.txt", [], 1, where="archive")
    add("F-407", "engagement_imu.txt present for every dispatched arm", allarm, "engagement_imu.txt", [], 1, where="archive")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--grid-retime", required=True); ap.add_argument("--grid-bag", required=True); ap.add_argument("--out", default="r80d_manifest")
    ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--min-free-gb", type=float, default=150.0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    J = {j["job_id"]: j for j in json.load(open(a.jobs))}
    grid_retime = json.load(open(a.grid_retime)); grid_bag = json.load(open(a.grid_bag))
    for bag in BAGS:
        if bag not in grid_bag:
            sys.exit("grid-bag has no entry for %s" % bag)
        if grid_bag[bag]["seq_gaps"] != 0:
            sys.exit("%s: seq_gaps = %d, restamp index (message count) would differ from seq - seq0; stop" % (bag, grid_bag[bag]["seq_gaps"]))
    os.makedirs(a.out, exist_ok=True)
    ALL = arms(grid_retime, grid_bag)
    out = {}; table = []; seen = set()
    for r in ALL:
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
        nm = "d_" + b
        sp = M.listener_spec("R80D-%s" % nm, [q(nm, len(out[nm]))], a.results_root, a.min_free_gb, a.workers)
        fn = "listener_spec_%s.json" % nm
        for blk in sp["stages"] + sp["verify"]:
            blk["cmd"] = [c.replace("%s", fn).replace("r74_manifest", "r80d_manifest") for c in blk["cmd"]]
        sp["raw"]["delete"] = sp["raw"]["delete"] + ["iekf_factor_iteration.csv", "lidar_msg_log.csv", "post_calibration_voxel_observation_support.csv", "line_rows_per_scan.csv"]
        json.dump(sp, open(os.path.join(a.out, fn), "w"), indent=1)
        order.append("%s\t%s\t%d arms" % (fn, nm, len(out[nm])))
    open(os.path.join(a.out, "queue_order.tsv"), "w").write("# listener spec\tqueue\tarms   (a preference, not a dependency: rule 1.16)\n" + "\n".join(order) + "\n")
    RO = required_outputs(table)
    json.dump(RO, open(os.path.join(a.out, "required_outputs.json"), "w"), indent=1)
    json.dump(R80.ledger(RO), open(os.path.join(a.out, "r80d_file_size_ledger.json"), "w"), indent=1)
    byid = {r["id"]: r for r in ALL}
    with open(os.path.join(a.out, "COMPLIANCE_TABLE.md"), "w") as f:
        f.write("# R80D compliance table (rule 1.10 item 2, 1.12)\n\n| arm | Bryce's words | policy | same as R80 arm | mark |\n|---|---|---|---|---|\n")
        for t in table:
            r = byid[t["arm"]]
            words = {"RS": "'i would like it as a third policy for comparison'; 'restamp only'", "OFF": "'let's do another addendum'; offset sweep 'Yes, all 9 bags'",
                     "E3": "fixes 1, 5, 6: 'Yes, all three' (fix 1: the 8 eee_03 arms)"}[r["group"]]
            f.write("| %s | %s | %s | %s | MATCHES |\n" % (t["arm"], words, r["policy"], r["src_arm"]))
    cnt = {}
    for t in table:
        cnt[t["group"]] = cnt.get(t["group"], 0) + 1
    print("arms %d: %s; queues %d; listener specs %d" % (len(table), ", ".join("%s %d" % kv for kv in sorted(cnt.items())), len(out), len(BAGS)))


if __name__ == "__main__":
    main()
