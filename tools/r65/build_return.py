#!/usr/bin/env python3
"""R65: assemble the return zips and a checklist that says, item by item, what is present and what is missing. Nothing is interpreted.

python3 tools/r65/build_return.py --ret R65_RETURN --runs r65_runs --arms r65_manifest/arm_table.tsv --manifest r65_manifest \
        --queue-dir ablations/_job_queue/r65_eee01 --date 2026-09-30
Before running it, YOU write into --ret (agent-authored, see R65_INSTRUCTIONS.md section 9): report.md, STATE.md, agent_edits.patch,
code_answers.md, R65_tree.tar.gz, unit_tests/, checks/, bag_stamps/. This script adds: phase_summary.csv, arm_scan_all.csv.gz, arm_ol_tau_all.csv.gz,
plane_fit_reasons_all.csv.gz, plane_fit_trace_counts_all.csv.gz, completeness.csv, identity_controls.txt, compress_manifests.json,
sensor_stamps_summaries.json, summaries/, compact/, queue/, RETURN_CHECKLIST.md, then packs everything into zips of at most --limit-mb each:
  R65_RETURN_<date>_part1.zip   everything except the two big groups below (must fit; if not, the script says so and stops)
  R65_RETURN_<date>_part2.zip   arm_scan_all.csv.gz and the other *_all.csv.gz tables (split further if needed)
  R65_RETURN_<date>_part3[a,b,..].zip   compact/<arm>/ directories, greedily packed
Nothing agent-authored is ever dropped or thinned."""
import argparse, glob, os, shutil, subprocess, sys, zipfile

REQUIRED_AGENT = ["report.md", "STATE.md", "agent_edits.patch", "code_answers.md", "R65_tree.tar.gz",
                  "unit_tests/ctest_output.txt", "unit_tests/test_point_cov_output.txt", "unit_tests/test_voxel_history_output.txt", "checks/smoke_table.md", "bag_stamps"]
REQUIRED_DIRS_NONEMPTY = ["checks/param_dumps", "checks/refusals", "checks/smoke_arm_files"]
GENERATED = ["phase_summary.csv", "arm_scan_all.csv.gz", "arm_ol_tau_all.csv.gz", "plane_fit_reasons_all.csv.gz", "plane_fit_trace_counts_all.csv.gz",
             "completeness.csv", "identity_controls.txt", "compress_manifests.json", "sensor_stamps_summaries.json", "summaries", "compact", "queue"]
BIG_TABLES = ["arm_scan_all.csv.gz", "arm_ol_tau_all.csv.gz", "plane_fit_reasons_all.csv.gz", "plane_fit_trace_counts_all.csv.gz"]

def size(p):
    if os.path.isdir(p):
        return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)
    return os.path.getsize(p) if os.path.exists(p) else 0

def files_of(ret, rel):
    p = os.path.join(ret, rel)
    if os.path.isdir(p):
        return [os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs]
    return [p] if os.path.exists(p) else []

def write_zip(zp, ret, files):
    base = os.path.dirname(os.path.abspath(ret.rstrip("/")))
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, os.path.relpath(os.path.abspath(f), base))
    return os.path.getsize(zp) / 1e6

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ret", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--arms", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--queue-dir", required=True); ap.add_argument("--date", required=True)
    ap.add_argument("--limit-mb", type=float, default=28.0)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.check_call([sys.executable, os.path.join(here, "r65_collect.py"), "--runs", a.runs, "--arms", a.arms, "--out", a.ret])
    q = os.path.join(a.ret, "queue"); os.makedirs(q, exist_ok=True)
    for f in ("jobs.json", "arm_table.tsv"):
        if os.path.exists(os.path.join(a.manifest, f)):
            shutil.copy(os.path.join(a.manifest, f), q)
    with open(os.path.join(q, "queue_state.txt"), "w") as f:
        n = {"DONE": 0, "FAILED": 0, "neither": 0}
        for jd in sorted(glob.glob(os.path.join(a.queue_dir, "*/"))):
            st = "DONE" if os.path.exists(jd + "DONE") else ("FAILED" if os.path.exists(jd + "FAILED") else "neither")
            n[st] += 1
            f.write("%s\t%s\n" % (os.path.basename(jd.rstrip("/")), st))
        f.write("# totals: %s\n" % n)
    lines = ["# R65 return checklist (generated; presence and size only)", "", "| item | status | bytes |", "|---|---|---|"]
    miss = 0
    for it in REQUIRED_AGENT + REQUIRED_DIRS_NONEMPTY + GENERATED:
        p = os.path.join(a.ret, it)
        ok = os.path.exists(p) and (size(p) > 0 or it == "agent_edits.patch")
        if it in REQUIRED_DIRS_NONEMPTY or it in ("bag_stamps", "compact"):
            ok = os.path.isdir(p) and len(os.listdir(p)) > 0
        miss += 0 if ok else 1
        lines.append("| %s | %s | %d |" % (it, "PRESENT" if ok else "**MISSING**", size(p)))
    lines += ["", "MISSING items: %d. For each one write `NOT PRODUCED: <reason>` in a file of that name (or in report.md) and re-run this script." % miss]
    open(os.path.join(a.ret, "RETURN_CHECKLIST.md"), "w").write("\n".join(lines) + "\n")
    # ---- packing
    lim = a.limit_mb * 1e6
    allf = files_of(a.ret, ".")
    big = set()
    for t in BIG_TABLES:
        big.update(files_of(a.ret, t))
    compact = files_of(a.ret, "compact")
    part1 = [f for f in allf if f not in big and f not in set(compact)]
    made = []
    p1 = "R65_RETURN_%s_part1.zip" % a.date
    mb = write_zip(p1, a.ret, part1); made.append((p1, mb))
    if mb * 1e6 > lim:
        print("ERROR: part 1 is %.2f MB, over the %.1f MB limit; say so in report.md and name the largest items:" % (mb, a.limit_mb))
        for f in sorted(part1, key=os.path.getsize, reverse=True)[:8]:
            print("   %10d  %s" % (os.path.getsize(f), os.path.relpath(f, a.ret)))
        return 2
    def pack(prefix, files):
        out, cur, cur_sz, k = [], [], 0, 0
        for f in sorted(files):
            s = os.path.getsize(f)
            if cur and cur_sz + s > lim * 0.97:
                out.append(cur); cur, cur_sz = [], 0
            cur.append(f); cur_sz += s
        if cur:
            out.append(cur)
        names = []
        for i, fl in enumerate(out):
            nm = "R65_RETURN_%s_%s%s.zip" % (a.date, prefix, "" if len(out) == 1 else chr(ord("a") + i))
            mbi = write_zip(nm, a.ret, fl)   # inputs are already gzip/small json, so deflated size ~ raw size; 0.97 margin above
            names.append((nm, mbi))
        return names
    made += pack("part2", sorted(big))
    made += pack("part3", compact)
    print("wrote: " + ", ".join("%s (%.2f MB)" % m for m in made) + "; missing items %d" % miss)
    return 0

if __name__ == "__main__":
    sys.exit(main())
