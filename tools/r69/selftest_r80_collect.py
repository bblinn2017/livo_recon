#!/usr/bin/env python3
"""selftest_r80_collect.py -- R80 return rules of r69_collect.py and the new reducers / archive line of r74_score_job.py. Synthetic data only; exits 1 on any failure.
    python3 tools/r69/selftest_r80_collect.py
Proves: SW arm gt_max_mm 100 -> no compact file; SW 900 -> pos_trace only; SW 500.0 (not above) -> nothing; SW gt unknown -> pos_trace only (undetermined);
  motion_q_scan_all.csv.gz returned for r80_b3_eee_02_mq only; ZD / NG / SC files returned (and only for their group); phase_summary.csv has a row for every arm (incl. SW);
  r74_analyses writes zupt_all.csv.gz / q_adapt_all.csv.gz (lossless, string-exact) only when the source exists; r74_score_job.main archives exit_status.txt (present: archived + listed;
  absent: stage still ok), and fails the arm when a required q_adapt.csv / zupt.csv is absent."""
import csv, gzip, json, os, subprocess, sys, tempfile
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r69_collect as C, r74_score_job as SJ   # noqa: E402
fails = []
def chk(c, m):
    if not c:
        fails.append(m)
def gz(p, text="a,b\n1,2\n"):
    with gzip.open(p, "wt") as f:
        f.write(text)

ALL = ["pos_trace_all.csv.gz", "residual_dump_all.csv.gz", "line_rows_per_scan_all.csv.gz", "window_state_all.csv.gz", "lidar_msg_log_all.csv.gz", "sensor_stamps_summary.json",
       "compress_manifest.json", "motion_q_scan_all.csv.gz", "zupt_all.csv.gz", "q_adapt_all.csv.gz", "step_cap.csv.gz", "foo_compact.csv.gz", "scan_log_all.csv.gz"]
ARMS = [  # arm, group, gt_max_mm (None = no value in the summary)
    ("r80_sw_eee_01_off_v0p5_pfn3_fq", "SW", 100.0), ("r80_sw_eee_01_off_v0p55_pfn3_fq", "SW", 900.0), ("r80_sw_eee_01_off_v0p6_pfn3_fq", "SW", 500.0),
    ("r80_sw_eee_01_off_v0p65_pfn3_fq", "SW", None), ("r80_sw_eee_01_hardabs_v0p7_pfn1_mq", "SW", 500.0001),
    ("r80_b3_eee_02_mq", "B3", 40.0), ("r80_b3_eee_01_mq", "B3", 40.0), ("r80_zd_eee_01_fq", "ZD", 40.0), ("r80_ng_eee_01_fq", "NG", 40.0), ("r80_sc_fq", "SC", 40.0),
    ("r80_mg_eee_01_fq", "MG", 40.0), ("r80_ec3_n250_pfn3", "EC3", 40.0), ("r80_sm_nis_60s", "SMK", 40.0)]
def got(root, arm):
    d = os.path.join(root, "compact", arm)
    return sorted(os.listdir(d)) if os.path.isdir(d) else []

with tempfile.TemporaryDirectory() as t:
    runs, out = os.path.join(t, "runs"), os.path.join(t, "out")
    rows = []
    for arm, grp, gt in ARMS:
        D = os.path.join(runs, arm); os.makedirs(D)
        for f in ALL:
            if f.endswith(".json"):
                open(os.path.join(D, f), "w").write("{}")
            else:
                gz(os.path.join(D, f))
        S = {"arm": arm, "kind": "fb", "n_scans": 10}
        if gt is not None:
            S["gt_max_mm"] = gt
        json.dump(S, open(os.path.join(D, "arm_summary.json"), "w"))
        pd.DataFrame({"scan": [1, 2, 3], "t": ["1.0000", "2.0000", "3.0000"], "ep_x_mm": [0.0, 1.0, 2.0], "ep_y_mm": [0.0] * 3, "ep_z_mm": [0.0] * 3}).to_csv(os.path.join(D, "arm_scan.csv"), index=False)
        rows.append(dict(arm=arm, tier="FB", kind="fb", duration_s="full", seq="eee_01", keep_raw=0, group=grp, dup_of="", notes=""))
    at = os.path.join(t, "arm_table.tsv")
    pd.DataFrame(rows).to_csv(at, sep="\t", index=False)
    subprocess.run([sys.executable, os.path.join(HERE, "r69_collect.py"), "--runs", runs, "--arms", at, "--out", out], check=True, capture_output=True)
    TR = ["pos_trace_all.csv.gz"]
    chk(got(out, "r80_sw_eee_01_off_v0p5_pfn3_fq") == [], "SW gt_max 100 must return nothing, got %s" % got(out, "r80_sw_eee_01_off_v0p5_pfn3_fq"))
    chk(got(out, "r80_sw_eee_01_off_v0p55_pfn3_fq") == TR, "SW gt_max 900 must return pos_trace only, got %s" % got(out, "r80_sw_eee_01_off_v0p55_pfn3_fq"))
    chk(got(out, "r80_sw_eee_01_off_v0p6_pfn3_fq") == [], "SW gt_max exactly 500 is not above the limit: nothing")
    chk(got(out, "r80_sw_eee_01_off_v0p65_pfn3_fq") == TR, "SW without gt_max_mm: pos_trace only (undetermined)")
    chk(got(out, "r80_sw_eee_01_hardabs_v0p7_pfn1_mq") == TR, "SW gt_max 500.0001 > 500: pos_trace only")
    mq2, mq1 = got(out, "r80_b3_eee_02_mq"), got(out, "r80_b3_eee_01_mq")
    chk("motion_q_scan_all.csv.gz" in mq2, "eee_02 mq arm must return motion_q_scan_all.csv.gz")
    chk("motion_q_scan_all.csv.gz" not in mq1, "another mq arm must not return motion_q_scan_all.csv.gz")
    chk("zupt_all.csv.gz" in got(out, "r80_zd_eee_01_fq"), "ZD must return zupt_all.csv.gz")
    chk("q_adapt_all.csv.gz" in got(out, "r80_ng_eee_01_fq"), "NG must return q_adapt_all.csv.gz")
    chk("step_cap.csv.gz" in got(out, "r80_sc_fq"), "SC must return step_cap.csv.gz")
    for arm, bad in (("r80_zd_eee_01_fq", ("q_adapt_all.csv.gz", "step_cap.csv.gz")), ("r80_ng_eee_01_fq", ("zupt_all.csv.gz", "step_cap.csv.gz")), ("r80_sc_fq", ("zupt_all.csv.gz", "q_adapt_all.csv.gz")),
                     ("r80_b3_eee_01_mq", ("zupt_all.csv.gz", "q_adapt_all.csv.gz", "step_cap.csv.gz"))):
        chk(not set(bad) & set(got(out, arm)), "%s returned a file of another group: %s" % (arm, got(out, arm)))
    for arm in ("r80_b3_eee_01_mq", "r80_mg_eee_01_fq", "r80_ec3_n250_pfn3", "r80_sm_nis_60s", "r80_zd_eee_01_fq"):
        extra = set(got(out, arm)) - C.F1_KEEP
        chk(not extra, "%s returned files outside F1_KEEP: %s" % (arm, sorted(extra)))
    chk("pos_trace_all.csv.gz" in got(out, "r80_ec3_n250_pfn3"), "EC3 returns pos_trace_all.csv.gz")
    ps = pd.read_csv(os.path.join(out, "phase_summary.csv"))
    chk(set(ps.arm) == {a[0] for a in ARMS}, "phase_summary.csv must have a row for every arm, has %d of %d" % (len(ps), len(ARMS)))
    chk(ps.set_index("arm").loc["r80_sw_eee_01_off_v0p5_pfn3_fq", "gt_max_mm"] == 100.0, "SW summary row must carry gt_max_mm")
    chk(len(json.load(open(os.path.join(out, "summaries_all.json")))) == len(ARMS), "summaries_all.json must hold every arm")
    asa = pd.read_csv(os.path.join(out, "arm_scan_all.csv.gz"))
    sw_arms = {a[0] for a in ARMS if a[1] == "SW"}
    chk(not (set(asa.arm) & sw_arms), "SW arms must leave no row in arm_scan_all.csv.gz: %s" % sorted(set(asa.arm) & sw_arms))
    chk(set(asa.arm) == {a[0] for a in ARMS} - sw_arms and (asa.groupby("arm").size() == 3).all(), "every non-SW arm keeps its rows in arm_scan_all.csv.gz")
    chk(sw_arms <= set(ps.arm) and sw_arms <= set(json.load(open(os.path.join(out, "summaries_all.json")))), "SW rows stay in phase_summary.csv and summaries_all.json")
    for g_ in ("B3", "B1", "ZD", "GP", "NG", "PC", "DB", "TL", "EC3", "SC", "SW", "SMK"):
        chk(g_ in C.F1_GROUPS, "group %s missing from F1_GROUPS" % g_)
    chk(C.sw_wants_trace(500.0) is False and C.sw_wants_trace(500.1) is True and C.sw_wants_trace(float("nan")) is True and C.sw_wants_trace("x") is True, "sw_wants_trace boundary")

    # ---- r74_score_job reducers
    d = os.path.join(t, "arm"); os.makedirs(d)
    open(os.path.join(d, "zupt.csv"), "w").write("frame,gyro_mean,acc_mean,run,stationary\n1,0.0100000000000000002,0.2,3,1\n2,0.5,0.1,0,0\n")
    open(os.path.join(d, "q_adapt.csv"), "w").write("frame,gain,nis,action\n1,1,0.9,hold\n")
    calls = []; orig = SJ.run; SJ.run = lambda cmd, log: (calls.append(cmd), 0)[1]
    try:
        info = SJ.r74_analyses(d, HERE, {"f417": "0", "plane_check": "0", "group": "ZD"}, 0)[1]
    finally:
        SJ.run = orig
    chk(info.get("zupt_all_rows") == 2 and info.get("q_adapt_all_rows") == 1, "gz copies row counts: %s" % info)
    chk(pd.read_csv(os.path.join(d, "zupt_all.csv.gz"), dtype=str).iloc[0].gyro_mean == "0.0100000000000000002", "zupt_all.csv.gz must be string-exact")
    d2 = os.path.join(t, "arm2"); os.makedirs(d2)
    SJ.run = lambda cmd, log: 0
    try:
        info2 = SJ.r74_analyses(d2, HERE, {"f417": "0", "plane_check": "0", "group": "SW"}, 0)[1]
    finally:
        SJ.run = orig
    chk(not os.path.exists(os.path.join(d2, "zupt_all.csv.gz")) and info2.get("zupt_all_rows") is None, "no source -> no zupt_all.csv.gz (no spurious file)")
    chk("zupt.csv" in SJ.R73_OPTIONAL and "q_adapt.csv" in SJ.R73_OPTIONAL and "zupt.csv" in SJ.BIG and "q_adapt.csv" in SJ.BIG, "zupt.csv / q_adapt.csv in BIG and R73_OPTIONAL")
    import make_r74_manifest as M74
    chk("zupt.csv" in M74.requires_of({"lio.zupt.mode": "detect_only"}, "x") and "q_adapt.csv" in M74.requires_of({"imu.process_noise.gain.mode": "nis_gate"}, "x")
        and "step_cap.csv" in M74.requires_of({"lio.ekf.step_cap.mode": "hard"}, "x"), "requires_of covers zupt / q_adapt / step_cap")

    # ---- r74_score_job.main end to end with the reducer subprocesses stubbed: exit_status archive + requires failure
    def run_main(name, es, requires, overrides_files):
        base = os.path.join(t, name); jd, od, so = (os.path.join(base, x) for x in ("job", "res", "stage"))
        for x in (jd, od, so):
            os.makedirs(x)
        open(os.path.join(jd, "DONE"), "w").write(""); open(os.path.join(jd, "job.env"), "w").write("")
        if es is not None:
            open(os.path.join(jd, "exit_status.txt"), "w").write(es)
        for f in overrides_files:
            open(os.path.join(jd, f), "w").write("frame,x\n1,2\n")
        spec = os.path.join(base, "spec.json"); json.dump({"raw": {"delete": ["zupt.csv", "q_adapt.csv"]}}, open(spec, "w"))
        tb = os.path.join(base, "arm_table.tsv")
        with open(tb, "w") as f:
            f.write("arm\ttier\tarch\tkind\tduration_s\tseq\tkeep_raw\trequires\tgroup\tf417\tearly\twin\tmsglog\tplane_check\nA\tEF\tx\tfb\t5\teee_01\t0\t%s\tZD\t0\t0\t0\t0\t0\n" % requires)
        pj = os.path.join(base, "p.json"); json.dump({"eee_01": {"onset_scan": 485, "stat_last": 1, "trans_last": 2}}, open(pj, "w"))
        argv = sys.argv; sys.argv = ["x", "--job-dir", jd, "--out-dir", od, "--stage-out", so, "--job-id", "A", "--spec", spec, "--arm-table", tb, "--params", pj, "--tools-dir", HERE]
        SJ.run = lambda cmd, log: 0
        try:
            try:
                SJ.main(); rc = 0
            except SystemExit as e:
                rc = e.code
        finally:
            SJ.run = orig; sys.argv = argv
        return rc, so
    rc, so = run_main("m1", "exit_code=0 signal=none\n", "zupt.csv", ["zupt.csv"])
    idx = [e["name"] for e in json.load(open(os.path.join(so, "archive", "archive_index.json")))]
    chk(rc == 0, "stage ok expected when the required zupt.csv exists, rc=%s" % rc)
    chk("exit_status.txt" in idx and "zupt.csv" in idx and os.path.exists(os.path.join(so, "archive", "exit_status.txt.gz")), "exit_status.txt archived: %s" % idx)
    chk("exit_status.txt" in open(os.path.join(so, "A", "raw_sha256.txt")).read(), "exit_status.txt listed in raw_sha256.txt")
    chk(gzip.open(os.path.join(so, "archive", "exit_status.txt.gz")).read() == b"exit_code=0 signal=none\n", "exit_status.txt archived losslessly")
    rc, so = run_main("m2", None, "zupt.csv", ["zupt.csv"])
    chk(rc == 0 and "exit_status.txt" not in [e["name"] for e in json.load(open(os.path.join(so, "archive", "archive_index.json")))], "absent exit_status.txt must not fail the stage (rc=%s)" % rc)
    rc, so = run_main("m3", "exit_code=0\n", "q_adapt.csv", ["zupt.csv"])
    chk(rc == 1 and "q_adapt.csv" in json.load(open(os.path.join(so, "summary_stage.json")))["requires_missing"], "required q_adapt.csv absent must fail the stage (rc=%s)" % rc)
    rc, so = run_main("m4", "exit_code=0\n", "zupt.csv", [])
    chk(rc == 1 and "zupt.csv" in json.load(open(os.path.join(so, "summary_stage.json")))["requires_missing"], "required zupt.csv absent must fail the stage (rc=%s)" % rc)

if fails:
    print("SELFTEST R80 COLLECT FAIL"); [print(" ", f) for f in fails]; sys.exit(1)
print("SELFTEST R80 COLLECT PASS")
