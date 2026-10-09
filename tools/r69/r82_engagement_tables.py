#!/usr/bin/env python3
"""R82: the three engagement tables the contract lists, extracted from each arm's engagement_imu.txt. Planning agent's script; selftest on synthetic text only (planner sandbox, not evidence).
    python3 tools/r69/r82_engagement_tables.py --runs <dir, one folder per arm> --out <dir> [--arms <arm_table.tsv>]
    python3 tools/r69/r82_engagement_tables.py --selftest
Writes r82_d4_counters.csv (line 'line_residual/classify_at_fit = ... line_classified=...'), r82_d2_diag.csv (block 'plane_var_diag_en rows=...') and r82_point_time_engagement.csv (line 'cbk/lidar/point_time: field=...').
One row per arm that has the line: columns arm, <every key=value token found in any row, union>, line (first 400 characters of the raw text). No arm has the line -> header only. Nothing is computed from the values."""
import argparse, csv, os, re, sys, gzip, tempfile, shutil

TABLES = [("r82_d4_counters.csv", r"line_residual/classify_at_fit\s*=.*line_classified=", "engagement.txt"), ("r82_d2_diag.csv", r"plane_var_diag_en rows=", "engagement.txt"), ("r82_point_time_engagement.csv", r"cbk/lidar/point_time:\s*field=", "run.log")]
# R82 AGENT FIX: the D4/D2 lines are written to engagement.txt (the general per-flag "did it do anything" summary) and the
# point_time line is written directly to run.log at startup -- none of the three ever appear in engagement_imu.txt (that file
# is IMU-propagation/process-noise-switch counters only). Confirmed by direct inspection of a dbcm (D4), pv (D2) and pton
# (point_time) arm's actual output files; the original single hardcoded "engagement_imu.txt" source matched zero arms for
# all three tables.


def rd(p):
    for q in (p, p + ".gz"):
        if os.path.exists(q):
            return (gzip.open(q, "rt", errors="replace") if q.endswith(".gz") else open(q, errors="replace")).read()
    return None


def run(a):
    arms = sorted(d for d in os.listdir(a.runs) if os.path.isdir(os.path.join(a.runs, d)))
    if a.arms:
        keep = {r["arm"] for r in csv.DictReader(open(a.arms), delimiter="\t")}
        arms = [x for x in arms if x in keep]
    os.makedirs(a.out, exist_ok=True)
    for name, pat, src in TABLES:
        rows, keys = [], []
        for arm in arms:
            t = rd(os.path.join(a.runs, arm, src))
            if not t:
                continue
            ls = [l for l in t.splitlines() if re.search(pat, l)]
            if not ls:
                continue
            kv = dict(re.findall(r"([A-Za-z_][A-Za-z_0-9()]*)\s*=\s*([^\s,;]+)", ls[0]))
            for k in kv:
                if k not in keys:
                    keys.append(k)
            rows.append(dict(arm=arm, line=ls[0].strip()[:400], **kv))
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, ["arm"] + keys + ["line"], extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        print("%s: %d arms" % (name, len(rows)))
    return 0


def selftest():
    t = tempfile.mkdtemp(); fails = []
    try:
        for arm, txt in (("a", "voxel_map/plane/line_residual/classify_at_fit = true line_classified=120 plane_kept=40 <-- x\ncbk/lidar/point_time: field=time mode=absolute_s_f64 ref=min_point\n"),
                         ("b", "voxel_map/plane/plane_var_diag_en rows=77 min=1e-06\n"), ("c", "nothing\n")):
            os.makedirs(os.path.join(t, "r", arm))
            for fn in ("engagement.txt", "run.log"):   # R82 AGENT FIX: D4/D2 come from engagement.txt, point_time from run.log -- write the same synthetic text to both so the selftest covers either source
                open(os.path.join(t, "r", arm, fn), "w").write(txt)
        run(argparse.Namespace(runs=os.path.join(t, "r"), out=os.path.join(t, "o"), arms=None))
        rd_ = lambda n: list(csv.DictReader(open(os.path.join(t, "o", n))))
        d4, d2, pt = rd_("r82_d4_counters.csv"), rd_("r82_d2_diag.csv"), rd_("r82_point_time_engagement.csv")
        if not (len(d4) == 1 and d4[0]["arm"] == "a" and d4[0]["line_classified"] == "120" and d4[0]["plane_kept"] == "40"): fails.append("d4 %r" % d4)
        if not (len(d2) == 1 and d2[0]["arm"] == "b" and d2[0]["rows"] == "77"): fails.append("d2 %r" % d2)
        if not (len(pt) == 1 and pt[0]["mode"] == "absolute_s_f64" and pt[0]["ref"] == "min_point"): fails.append("pt %r" % pt)
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("SELFTEST r82_engagement_tables", "FAIL " + "; ".join(fails) if fails else "PASS (D4 counters, D2 diagnostic rows and point-time lines extracted per arm; arms without the line are absent)")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs"); ap.add_argument("--out", default="."); ap.add_argument("--arms"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else run(a))
