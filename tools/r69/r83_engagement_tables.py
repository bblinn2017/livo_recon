#!/usr/bin/env python3
"""R83: engagement tables extracted from each arm's output files. Planning agent's script; selftest on synthetic text only (planner sandbox, not evidence).
    python3 tools/r69/r83_engagement_tables.py --runs <dir, one folder per arm> --out <dir> [--arms <arm_table.tsv>] [--engagement-all <engagement_all.txt of the return>]
    python3 tools/r69/r83_engagement_tables.py --selftest
Writes r83_d4_counters.csv (line 'line_residual/classify_at_fit = ... line_classified='; the D4P arms), r83_point_time_engagement.csv (line 'cbk/lidar/point_time: field=...'; the Hilti smoke arm),
r83_window_timer_all.txt (the profiler line that names 'imu/window_cov_profile', one tab-separated line `arm  src  line` per D1B arm; the name of required_outputs.json) and r83_peak_rss.csv (peak_rss.txt of the arms that have it: arm, text, aborted).
One row per arm that has the line: columns arm, src (the file the line came from), <every key=value token found in any row, union>, line (first 400 characters of the raw text). No arm has the line -> header only.
Nothing is computed from the values.

WHAT WAS WRONG IN R82 (planner's error): r82_engagement_tables.py read the three lines from `engagement_imu.txt` only. That file holds IMU-propagation and process-noise-switch counters. The D4 counters and the D2
block are written to `engagement.txt` and the point-time line is written to `run.log` at startup, so the shipped tool matched no arm for any of the three tables (the coding agent found this and patched its copy).
R83 does not rely on one file name: every table lists the files it may come from and the first file of an arm that has the line is used (column src says which). With --engagement-all the
`### <arm>` blocks of the return's engagement_all.txt are a further source for lines that live in engagement.txt / engagement_imu.txt (run.log is not part of that file).
"""
import argparse, csv, gzip, os, re, shutil, sys, tempfile

# (output file, line pattern, files to try in order)
TABLES = [("r83_d4_counters.csv", r"line_residual/classify_at_fit\s*=.*line_classified=", ("engagement.txt", "engagement_imu.txt", "run.log")),
          ("r83_point_time_engagement.csv", r"cbk/lidar/point_time:\s*field=", ("run.log", "engagement.txt", "engagement_imu.txt")),
          ]
TIMER_OUT = "r83_window_timer_all.txt"   # name of required_outputs.json (R83-D1B): one tab-separated line per arm that has the timer line: arm, src, the raw line
TIMER_PAT = r"imu/window_cov_profile"
TIMER_SRCS = ("profiler.txt", "run.log", "engagement.txt", "engagement_imu.txt")   # D1B arms set outputs.profiler.export_path = <arm folder>/profiler.txt (lines `<timer> <avg ms> ms (<n> frames)`); run.log has no [profiler] lines then
RSS_FILE = "peak_rss.txt"
RSS_OUT = "r83_peak_rss.csv"
KV = re.compile(r"([A-Za-z_][A-Za-z_0-9()/]*)\s*=\s*([^\s,;]+)")


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


def run(a):
    arms = sorted(d for d in os.listdir(a.runs) if os.path.isdir(os.path.join(a.runs, d))) if a.runs and os.path.isdir(a.runs) else []
    blocks = all_blocks(a.engagement_all) if a.engagement_all and os.path.exists(a.engagement_all) else {}
    arms = sorted(set(arms) | set(blocks))
    if a.arms:
        keep = {r["arm"] for r in csv.DictReader(open(a.arms), delimiter="\t")}
        arms = [x for x in arms if x in keep]
    os.makedirs(a.out, exist_ok=True)
    for name, pat, srcs in TABLES:
        rows, keys = [], []
        for arm in arms:
            hit = None
            for fn in srcs:
                t = rd(os.path.join(a.runs, arm, fn)) if a.runs else None
                if t is None and fn != "run.log":
                    t = (blocks.get(arm) or {}).get(fn)
                ls = [l for l in (t or "").splitlines() if re.search(pat, l)]
                if ls:
                    hit = (fn, ls[0]); break
            if not hit:
                continue
            kv = dict(KV.findall(hit[1]))
            for k in kv:
                if k not in keys:
                    keys.append(k)
            rows.append(dict(arm=arm, src=hit[0], line=hit[1].strip()[:400], **kv))
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, ["arm", "src"] + keys + ["line"], extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        print("%s: %d arms" % (name, len(rows)))
    lines = ["arm\tsrc\tline"]
    for arm in arms:
        for fn in TIMER_SRCS:
            t = rd(os.path.join(a.runs, arm, fn)) if a.runs else None
            if t is None and fn not in ("run.log", "profiler.txt"):
                t = (blocks.get(arm) or {}).get(fn)
            ls = [l for l in (t or "").splitlines() if re.search(TIMER_PAT, l)]
            if ls:
                lines.append("%s\t%s\t%s" % (arm, fn, ls[0].strip()[:400])); break
    open(os.path.join(a.out, TIMER_OUT), "w").write("\n".join(lines) + "\n")
    print("%s: %d arms" % (TIMER_OUT, len(lines) - 1))
    rows = []
    for arm in arms:
        t = rd(os.path.join(a.runs, arm, RSS_FILE)) if a.runs else None
        if t is None:
            continue
        d = os.path.join(a.runs, arm)
        ab = any(n.startswith("ABORTED_RSS") for n in os.listdir(d)) or "ABORTED_RSS" in t or "ABORTED_RSS" in (rd(os.path.join(d, "arm.json")) or "")
        rows.append(dict(arm=arm, text=" ".join(t.split())[:300], aborted=int(ab)))
    with open(os.path.join(a.out, RSS_OUT), "w", newline="") as f:
        w = csv.DictWriter(f, ["arm", "text", "aborted"]); w.writeheader(); w.writerows(rows)
    print("%s: %d arms" % (RSS_OUT, len(rows)))
    return 0


def selftest():
    t = tempfile.mkdtemp(); fails = []; n = [0]

    def ck(c, m):
        n[0] += 1
        if not c:
            fails.append(m)
    try:
        R = os.path.join(t, "r")

        def put(arm, fn, txt):
            os.makedirs(os.path.join(R, arm), exist_ok=True); open(os.path.join(R, arm, fn), "w").write(txt)
        # D4 line in engagement.txt (where the node writes it), point time in run.log, window timer in run.log
        put("a", "engagement.txt", "voxel_map/plane/line_residual/classify_at_fit = true line_classified=120 plane_kept=40 <-- x\n")
        put("a", "run.log", "cbk/lidar/point_time: field=time mode=absolute_s_f64 ref=min_point\n")
        put("a", "engagement_imu.txt", "imu counters only\n")
        put("b", "run.log", "[timers] imu/window_cov_profile  calls=3987 total_ms=812.5\n")
        put("f", "profiler.txt", "imu/other 2 ms (9 frames)\nimu/window_cov_profile 0.031 ms (3987 frames)\n"); put("f", "run.log", "[timers] imu/window_cov_profile calls=1 total_ms=1\n")
        put("b", "engagement_imu.txt", "line_residual/classify_at_fit = true line_classified=5 in the wrong file still found\n")
        put("c", "engagement_imu.txt", "nothing\n")
        put("d", "peak_rss.txt", "peak_rss_kb 41234567\n"); put("e", "peak_rss.txt", "peak_rss_kb 99\n"); open(os.path.join(R, "e", "ABORTED_RSS"), "w").write("x")
        run(argparse.Namespace(runs=R, out=os.path.join(t, "o"), arms=None, engagement_all=None))
        rd_ = lambda nm: list(csv.DictReader(open(os.path.join(t, "o", nm))))
        d4, pt, rss = rd_("r83_d4_counters.csv"), rd_("r83_point_time_engagement.csv"), rd_(RSS_OUT)
        wt = [x.split("\t") for x in open(os.path.join(t, "o", TIMER_OUT)).read().splitlines()[1:]]
        ck(len(d4) == 2 and {r["arm"] for r in d4} == {"a", "b"}, "d4 arms %r" % d4)
        da = [r for r in d4 if r["arm"] == "a"][0]
        ck(da["line_classified"] == "120" and da["plane_kept"] == "40" and da["src"] == "engagement.txt", "d4 row a %r" % da)
        ck([r for r in d4 if r["arm"] == "b"][0]["src"] == "engagement_imu.txt", "a line in another candidate file is still found")
        ck(len(pt) == 1 and pt[0]["mode"] == "absolute_s_f64" and pt[0]["ref"] == "min_point" and pt[0]["src"] == "run.log", "pt %r" % pt)
        wd = {x[0]: x for x in wt}
        ck(len(wt) == 2 and wd["b"][1] == "run.log" and "calls=3987" in wd["b"][2], "window timer, run.log fallback %r" % wt)
        ck(wd["f"][1] == "profiler.txt" and "3987 frames" in wd["f"][2], "window timer read from profiler.txt first %r" % wt)
        ck(len(rss) == 2 and {r["arm"]: r["aborted"] for r in rss} == {"d": "0", "e": "1"}, "rss rows %r" % rss)
        # the engagement_all.txt route (no run dir): engagement.txt blocks are read, run.log lines are not available there
        open(os.path.join(t, "eall.txt"), "w").write("### z\n--- engagement.txt\nvoxel_map/plane/line_residual/classify_at_fit = true line_classified=7\n\n### y\n--- engagement_imu.txt\nnothing\n\n")
        run(argparse.Namespace(runs=None, out=os.path.join(t, "o2"), arms=None, engagement_all=os.path.join(t, "eall.txt")))
        z = list(csv.DictReader(open(os.path.join(t, "o2", "r83_d4_counters.csv"))))
        ck(len(z) == 1 and z[0]["arm"] == "z" and z[0]["line_classified"] == "7", "engagement_all route %r" % z)
        # --arms filter
        open(os.path.join(t, "at.tsv"), "w").write("arm\nb\n")
        run(argparse.Namespace(runs=R, out=os.path.join(t, "o3"), arms=os.path.join(t, "at.tsv"), engagement_all=None))
        ck(len(list(csv.DictReader(open(os.path.join(t, "o3", "r83_d4_counters.csv"))))) == 1, "arms filter")
        ck(not any("zupt" in x[0] for x in TABLES), "no ZUPT table in R83")
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("SELFTEST r83_engagement_tables", ("FAIL " + "; ".join(fails) + " | ") if fails else "PASS", "checks: %d, failures: %d" % (n[0], len(fails)),
          "" if fails else "(D4 counters from engagement.txt, point-time and window-timer lines from run.log, a line in another candidate file found, peak RSS with the ABORTED_RSS marker, engagement_all route, arm filter; no ZUPT table)")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs"); ap.add_argument("--out", default="."); ap.add_argument("--arms"); ap.add_argument("--engagement-all"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else run(a))
