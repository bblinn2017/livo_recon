#!/usr/bin/env python3
"""R84: engagement tables for the sample-window key and the time-based process-noise key, extracted from each arm's output files and from the arm jobs. Planning agent's script;
selftest on synthetic text only (planner sandbox, not evidence). Nothing is computed from the values: they are copied as the node wrote them.

    python3 tools/r69/r84_engagement_tables.py --runs <dir, one folder per arm> --out <dir> [--arms r84_manifest/arm_table.tsv] [--jobs r84_manifest] [--jobs <jobs_*.json> ...] [--engagement-all <engagement_all.txt of the return>]
    python3 tools/r69/r84_engagement_tables.py --selftest

Writes
  r84_sample_window_engagement.csv  one row per arm whose job sets lio.point_cov.sample_window_s > 0 or whose engagement text carries the sample_window_s line. Columns:
      arm, sw_ms, tbpn, sample_window_s, nodes, point_evaluations, clamped_points, mean_trace_m2, max_trace_m2, inert_marker, not_engaged_marker
      The counters come from the engagement line the node writes ONLY when the key is > 0 (engagement_imu.txt first, then engagement.txt, then run.log; with --engagement-all the `### <arm>` blocks are a further source):
        `  lio/point_cov sample_window_s=<v> nodes=5 point_evaluations=<int> clamped_points=<int> mean_trace_m2=<float> max_trace_m2=<float>` optionally followed by `   <-- INERT: ...` or `   <-- NOT ENGAGED: ...`
      inert_marker / not_engaged_marker are 1 when that marker text follows the line, else 0. An arm whose job sets the key but whose files carry no such line gets blank counters and both markers 0.
      sample_window_s = the line's value, else the params echo `lio/point_cov/sample_window_s: <v>` of run.log, else the job's value. sw_ms = the arm table's sw_ms column, else sample_window_s x 1000 as a number text.
      tbpn = the arm table's tbpn column, else the job's imu.undistort.time_based_process_noise, else blank.
  r84_tbpn_engagement.csv           arm, job_tbpn (the job's value of imu.undistort.time_based_process_noise; blank when the job does not set it), echo_tbpn (the value in the params echo
      `imu/undistort/time_based_process_noise: <v>` of run.log / engagement_imu.txt / engagement.txt; blank when no echo was found). One row per arm that has a job value or an echo.
Jobs: --jobs takes a manifest directory (every jobs_*.json in it) or job files; each file is a list of {job_id, overrides}; the job id is the arm id. Dotted override keys.
Arms: every folder of --runs and every arm of the engagement-all blocks and the jobs, limited to the arm table's arms when --arms is given.
"""
import argparse, csv, glob, gzip, json, os, re, shutil, sys, tempfile

SW_KEY = "lio.point_cov.sample_window_s"
TB_KEY = "imu.undistort.time_based_process_noise"
SW_LINE = re.compile(r"lio/point_cov\s+sample_window_s=")
SW_ECHO = re.compile(r"lio/point_cov/sample_window_s\s*:\s*(\S+)")
TB_ECHO = re.compile(r"imu/undistort/time_based_process_noise\s*:\s*(\S+)")
KV = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)=([^\s,;]+)")
SRCS = ("engagement_imu.txt", "engagement.txt", "run.log")
SW_COLS = ["arm", "sw_ms", "tbpn", "sample_window_s", "nodes", "point_evaluations", "clamped_points", "mean_trace_m2", "max_trace_m2", "inert_marker", "not_engaged_marker"]
TB_COLS = ["arm", "job_tbpn", "echo_tbpn"]


def rd(p):
    for q in (p, p + ".gz"):
        if os.path.exists(q):
            return (gzip.open(q, "rt", errors="replace") if q.endswith(".gz") else open(q, errors="replace")).read()
    return None


def all_blocks(path):
    """engagement_all.txt -> {arm: {file name: text}} (blocks `### <arm>` and `--- <file>`)."""
    out, arm, fn = {}, None, None
    for ln in open(path, errors="replace").read().splitlines():
        if ln.startswith("### "):
            arm, fn = ln[4:].strip(), None; out.setdefault(arm, {})
        elif ln.startswith("--- ") and arm is not None:
            fn = ln[4:].strip(); out[arm].setdefault(fn, "")
        elif arm is not None and fn is not None:
            out[arm][fn] += ln + "\n"
    return out


def load_jobs(specs):
    """--jobs values (directories or files) -> {arm: overrides dict}."""
    files = []
    for s in specs or []:
        files += sorted(glob.glob(os.path.join(s, "jobs_*.json"))) if os.path.isdir(s) else [s]
    out = {}
    for f in files:
        d = json.load(open(f))
        for j in (d if isinstance(d, list) else d.get("jobs", [])):
            if isinstance(j, dict) and j.get("job_id"):
                out[j["job_id"]] = j.get("overrides") or {}
    return out


def texts(runs, arm, blocks):
    """-> [(file name, text)] in SRCS order: the arm's file when present, else the engagement_all block (run.log has no block)."""
    out = []
    for fn in SRCS:
        t = rd(os.path.join(runs, arm, fn)) if runs else None
        if t is None and fn != "run.log":
            t = (blocks.get(arm) or {}).get(fn)
        if t is not None:
            out.append((fn, t))
    return out


def num_text(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    return ("%.10g" % v)


def run(a):
    arms = sorted(d for d in os.listdir(a.runs) if os.path.isdir(os.path.join(a.runs, d))) if a.runs and os.path.isdir(a.runs) else []
    blocks = all_blocks(a.engagement_all) if a.engagement_all and os.path.exists(a.engagement_all) else {}
    jobs = load_jobs(a.jobs)
    table = {}
    if a.arms:
        table = {r["arm"]: r for r in csv.DictReader(open(a.arms), delimiter="\t")}
    arms = sorted(set(arms) | set(blocks) | set(jobs))
    if table:
        arms = [x for x in arms if x in table]
    os.makedirs(a.out, exist_ok=True)
    sw_rows, tb_rows = [], []
    for arm in arms:
        ov = jobs.get(arm, {})
        job_sw = ov.get(SW_KEY)
        job_tb = ov.get(TB_KEY)
        ts = texts(a.runs, arm, blocks)
        line = None
        for fn, t in ts:
            ls = [l for l in t.splitlines() if SW_LINE.search(l)]
            if ls:
                line = ls[0]; break
        echo_sw = echo_tb = None
        for fn, t in ts:
            m = SW_ECHO.search(t)
            if m and echo_sw is None:
                echo_sw = m.group(1)
            m = TB_ECHO.search(t)
            if m and echo_tb is None:
                echo_tb = m.group(1)
        on = isinstance(job_sw, (int, float)) and not isinstance(job_sw, bool) and job_sw > 0
        if on or line is not None:
            kv = dict(KV.findall(line)) if line else {}
            sw_s = kv.get("sample_window_s") or echo_sw or (num_text(job_sw) if job_sw is not None else "")
            tr = table.get(arm, {})
            sw_ms = (tr.get("sw_ms") or "").strip()
            if sw_ms.endswith(".0"):
                sw_ms = sw_ms[:-2]
            if not sw_ms and sw_s != "":
                sw_ms = num_text(float(sw_s) * 1000.0) if num_text(sw_s) != "" else ""
            tbpn = (tr.get("tbpn") or "").strip() or (job_tb if isinstance(job_tb, str) else "")
            sw_rows.append(dict(arm=arm, sw_ms=sw_ms, tbpn=tbpn, sample_window_s=sw_s, nodes=kv.get("nodes", ""), point_evaluations=kv.get("point_evaluations", ""),
                                clamped_points=kv.get("clamped_points", ""), mean_trace_m2=kv.get("mean_trace_m2", ""), max_trace_m2=kv.get("max_trace_m2", ""),
                                inert_marker=int(bool(line and "<-- INERT" in line)), not_engaged_marker=int(bool(line and "<-- NOT ENGAGED" in line))))
        if isinstance(job_tb, str) or echo_tb is not None:
            tb_rows.append(dict(arm=arm, job_tbpn=job_tb if isinstance(job_tb, str) else "", echo_tbpn=echo_tb or ""))
    for name, cols, rows in (("r84_sample_window_engagement.csv", SW_COLS, sw_rows), ("r84_tbpn_engagement.csv", TB_COLS, tb_rows)):
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
        print("%s: %d arms" % (name, len(rows)))
    return 0


def selftest():
    t = tempfile.mkdtemp(); fails = []; n = [0]

    def ck(c, m):
        n[0] += 1
        if not c:
            fails.append(m)
    try:
        R = os.path.join(t, "r"); J = os.path.join(t, "m")
        os.makedirs(J)

        def put(arm, fn, txt):
            os.makedirs(os.path.join(R, arm), exist_ok=True); open(os.path.join(R, arm, fn), "w").write(txt)
        LN = "  lio/point_cov sample_window_s=%s nodes=5 point_evaluations=%d clamped_points=%d mean_trace_m2=%s max_trace_m2=%s"
        put("w5", "engagement_imu.txt", "imu counters\n" + LN % ("0.005", 91234, 12, "1.5e-07", "4.2e-06") + "\n")
        put("w5", "run.log", "lio/point_cov/sample_window_s: 0.005\nimu/undistort/time_based_process_noise: var_acc\n")
        put("w1", "run.log", "imu/undistort/time_based_process_noise: none\n" + LN % ("0.001", 500, 0, "2e-08", "1e-07") + "\n")            # the line only in run.log: fallback
        put("inert", "engagement_imu.txt", LN % ("0.02", 0, 0, "0", "0") + "   <-- INERT: sample_window_s selected and no point was evaluated\n")
        put("ne", "engagement.txt", LN % ("0.01", 77, 3, "0", "0") + "   <-- NOT ENGAGED: every added covariance was zero (a valid null only if the platform never moved)\n")
        put("noline", "engagement_imu.txt", "imu counters only\n")                         # job sets the key, no line
        put("w0", "engagement_imu.txt", "imu counters only\n"); put("w0", "run.log", "imu/undistort/time_based_process_noise: var_acc\n")   # key 0: no line, no row in the first table
        put("plain", "engagement_imu.txt", "imu counters only\n")
        mk = lambda a, **ov: {"job_id": a, "overrides": ov}
        json.dump([mk("w5", **{SW_KEY: 0.005, TB_KEY: "var_acc"}), mk("w1", **{SW_KEY: 0.001, TB_KEY: "none"}), mk("inert", **{SW_KEY: 0.02}), mk("ne", **{SW_KEY: 0.01, TB_KEY: "var_acc"}),
                   mk("noline", **{SW_KEY: 0.002, TB_KEY: "none"}), mk("w0", **{TB_KEY: "var_acc"}), mk("plain", **{"voxel_map.map.voxel_size": 0.425})], open(os.path.join(J, "jobs_r84_x.json"), "w"))
        open(os.path.join(t, "at.tsv"), "w").write("arm\tsw_ms\ttbpn\nw5\t5.0\tvar_acc\nw1\t1.0\tnone\ninert\t\t\nne\t10.0\tvar_acc\nnoline\t2.0\tnone\nw0\t0.0\tvar_acc\nplain\t\t\n")
        run(argparse.Namespace(runs=R, out=os.path.join(t, "o"), arms=os.path.join(t, "at.tsv"), jobs=[J], engagement_all=None))
        sw = {r["arm"]: r for r in csv.DictReader(open(os.path.join(t, "o", "r84_sample_window_engagement.csv")))}
        tb = {r["arm"]: r for r in csv.DictReader(open(os.path.join(t, "o", "r84_tbpn_engagement.csv")))}
        ck(list(next(iter(sw.values()))) == SW_COLS, "sample-window columns")
        ck(set(sw) == {"w5", "w1", "inert", "ne", "noline"}, "rows: arms with the key > 0 only (w0 and plain have none): %r" % sorted(sw))
        r = sw["w5"]
        ck((r["sw_ms"], r["tbpn"], r["sample_window_s"], r["nodes"], r["point_evaluations"], r["clamped_points"], r["mean_trace_m2"], r["max_trace_m2"], r["inert_marker"], r["not_engaged_marker"]) ==
           ("5", "var_acc", "0.005", "5", "91234", "12", "1.5e-07", "4.2e-06", "0", "0"), "w5 row %r" % r)
        ck(sw["w1"]["point_evaluations"] == "500" and sw["w1"]["tbpn"] == "none", "line found in run.log when engagement_imu.txt has none: %r" % sw["w1"])
        ck(sw["inert"]["inert_marker"] == "1" and sw["inert"]["not_engaged_marker"] == "0" and sw["inert"]["point_evaluations"] == "0" and sw["inert"]["sw_ms"] == "20" and sw["inert"]["tbpn"] == "", "inert marker (sw_ms from the value, tbpn blank when neither table nor job names it): %r" % sw["inert"])
        ck(sw["ne"]["not_engaged_marker"] == "1" and sw["ne"]["inert_marker"] == "0" and sw["ne"]["clamped_points"] == "3", "not engaged marker, line in engagement.txt: %r" % sw["ne"])
        ck(sw["noline"]["point_evaluations"] == "" and sw["noline"]["inert_marker"] == "0" and sw["noline"]["not_engaged_marker"] == "0" and sw["noline"]["sample_window_s"] == "0.002", "job on, no line: blank counters, markers 0, window from the job: %r" % sw["noline"])
        ck(tb["w5"]["job_tbpn"] == "var_acc" and tb["w5"]["echo_tbpn"] == "var_acc" and tb["w1"]["job_tbpn"] == "none" and tb["w1"]["echo_tbpn"] == "none", "tbpn job and echo values: %r" % tb)
        ck(tb["w0"]["job_tbpn"] == "var_acc" and tb["noline"]["echo_tbpn"] == "" and tb["noline"]["job_tbpn"] == "none" and "plain" not in tb and "inert" not in tb, "tbpn rows: job value or echo only; blank echo when absent: %r" % sorted(tb))
        # no run dir: the engagement_all route (engagement_*.txt blocks; run.log has none) and jobs
        open(os.path.join(t, "eall.txt"), "w").write("### z\n--- engagement_imu.txt\n" + LN % ("0.05", 9, 1, "3e-05", "6e-05") + "\n\n### y\n--- engagement_imu.txt\nnothing\n\n")
        run(argparse.Namespace(runs=None, out=os.path.join(t, "o2"), arms=None, jobs=None, engagement_all=os.path.join(t, "eall.txt")))
        z = list(csv.DictReader(open(os.path.join(t, "o2", "r84_sample_window_engagement.csv"))))
        ck(len(z) == 1 and z[0]["arm"] == "z" and z[0]["sample_window_s"] == "0.05" and z[0]["sw_ms"] == "50" and z[0]["point_evaluations"] == "9", "engagement_all route %r" % z)
        # no arm has anything: header-only files
        run(argparse.Namespace(runs=os.path.join(t, "none"), out=os.path.join(t, "o3"), arms=None, jobs=None, engagement_all=None))
        ck(open(os.path.join(t, "o3", "r84_tbpn_engagement.csv")).read().strip() == ",".join(TB_COLS) and open(os.path.join(t, "o3", "r84_sample_window_engagement.csv")).read().strip() == ",".join(SW_COLS), "header-only files")
        ck(sorted(os.listdir(os.path.join(t, "o"))) == ["r84_sample_window_engagement.csv", "r84_tbpn_engagement.csv"], "only the two R84 files are written")
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("SELFTEST r84_engagement_tables", ("FAIL " + "; ".join(fails) + " | ") if fails else "PASS", "checks: %d, failures: %d" % (n[0], len(fails)),
          "" if fails else "(sample_window_s line from engagement_imu.txt, engagement.txt or run.log with INERT / NOT ENGAGED markers; job-on arm without a line gets blank counters; key 0 gives no row; tbpn job value and echo; engagement_all route; header-only)")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs"); ap.add_argument("--out", default="."); ap.add_argument("--arms"); ap.add_argument("--jobs", action="append", default=None)
    ap.add_argument("--engagement-all"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else run(a))
