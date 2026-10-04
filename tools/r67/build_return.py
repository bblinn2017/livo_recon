#!/usr/bin/env python3
"""R67: assemble the return as ONE zip and a checklist that says, item by item, what is present and what is missing. Nothing is interpreted.

python3 tools/r67/build_return.py --ret R67_RETURN --runs r67_runs --arms r67_manifest/arm_table.tsv --manifest r67_manifest \
        --queue-dir ablations/_job_queue/r67_all --date 2026-10-01 [--params bag_params.json]
Before running it, YOU write into --ret (agent-authored, see R67_INSTRUCTIONS.md section 10): report.md, STATE.md, agent_edits.patch, code_answers.md,
R67_tree.tar.gz, unit_tests/, checks/, bag_stats/, launch/ (launch_command.txt, containers_file.txt, launch_time.txt, checkup_cycles/ with every cycle's output).
This script adds: phase_summary.csv, arm_scan_all.csv.gz, arm_ol_tau_all.csv.gz, plane_fit_reasons_all.csv.gz, plane_fit_trace_counts_all.csv.gz,
completeness.csv, identity_controls.txt, duplicates.tsv, summaries_all.json, compress_manifests.json, sensor_stamps_summaries.json, compact/, queue/, RETURN_CHECKLIST.md,
then packs everything into ONE zip R67_RETURN_<date>.zip. It tries thinning levels 0..3 of arm_scan_all.csv.gz (r67_collect.py documents them; each keeps the
sufficiency statement) and stops at the first level whose single zip is under --limit-mb (default 29.5). Only if level 3 still does not fit it prints the largest
items and, as a LAST RESORT, splits (part files listed in SPLIT_LOG.txt; the reason and the items that forced it must be written into report.md section 8).
Nothing agent-authored is ever dropped or thinned."""
import argparse, glob, os, shutil, subprocess, sys, zipfile

REQUIRED_AGENT = ["report.md", "STATE.md", "agent_edits.patch", "code_answers.md", "R67_tree.tar.gz",
                  "unit_tests/ctest_output.txt", "checks/smoke_table.md", "checks/build_log_tail.txt",
                  "launch/launch_command.txt", "launch/containers_file.txt", "launch/launch_time.txt", "launch/checkup_cycles",
                  "P_RETURN/window_table.csv", "P_RETURN/manifest.json"]
REQUIRED_DIRS_NONEMPTY = ["checks/param_dumps", "checks/smoke_arm_files", "launch/checkup_cycles", "P_RETURN"]
GENERATED = ["phase_summary.csv", "arm_scan_all.csv.gz", "arm_ol_tau_all.csv.gz", "plane_fit_reasons_all.csv.gz", "plane_fit_trace_counts_all.csv.gz",
             "completeness.csv", "identity_controls.txt", "identity_gates.txt", "bootstrap_summary.csv", "duplicates.tsv", "summaries_all.json",
             "compress_manifests.json", "sensor_stamps_summaries.json", "compact", "queue"]
BIG_TABLES = ["arm_scan_all.csv.gz", "arm_ol_tau_all.csv.gz", "plane_fit_reasons_all.csv.gz", "plane_fit_trace_counts_all.csv.gz"]


def size(p):
    if os.path.isdir(p):
        return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)
    return os.path.getsize(p) if os.path.exists(p) else 0


def files_of(ret, rel):
    p = os.path.normpath(os.path.join(ret, rel))
    if os.path.isdir(p):
        return [os.path.normpath(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs]
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
    ap.add_argument("--params", default=None); ap.add_argument("--limit-mb", type=float, default=29.5)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    q = os.path.join(a.ret, "queue"); os.makedirs(q, exist_ok=True)
    for f in ("jobs_smoke.json", "jobs_main.json", "arm_table.tsv"):
        if os.path.exists(os.path.join(a.manifest, f)):
            shutil.copy(os.path.join(a.manifest, f), q)
    with open(os.path.join(q, "queue_state.txt"), "w") as f:
        n = {"DONE": 0, "FAILED": 0, "neither": 0}
        for jd in sorted(glob.glob(os.path.join(a.queue_dir, "*/"))):
            st = "DONE" if os.path.exists(jd + "DONE") else ("FAILED" if os.path.exists(jd + "FAILED") else "neither")
            n[st] += 1
            f.write("%s\t%s\n" % (os.path.basename(jd.rstrip("/")), st))
        f.write("# totals: %s\n" % n)
    lim = a.limit_mb * 1e6
    single = "R67_RETURN_%s.zip" % a.date
    fits, level_used, mb = False, None, 0.0
    for level in (0, 1, 2, 3):
        cmd = [sys.executable, os.path.join(here, "r67_collect.py"), "--runs", a.runs, "--arms", a.arms, "--out", a.ret, "--level", str(level)]
        if a.params:
            cmd += ["--params", a.params]
        subprocess.check_call(cmd)
        subprocess.call([sys.executable, os.path.join(here, "bootstrap_reduce.py"), "--collect", a.runs, "--out", a.ret])   # identity gates; exit 1 on a FAIL is reported in identity_gates.txt, not fatal here
        lines = ["# R67 return checklist (generated; presence and size only)", "", "thinning level used for arm_scan_all.csv.gz: %d" % level, "", "| item | status | bytes |", "|---|---|---|"]
        miss = 0
        for it in REQUIRED_AGENT + REQUIRED_DIRS_NONEMPTY + GENERATED:
            p = os.path.join(a.ret, it)
            ok = os.path.exists(p) and (size(p) > 0 or it == "agent_edits.patch")
            if it in REQUIRED_DIRS_NONEMPTY or it in ("compact",):
                ok = os.path.isdir(p) and len(os.listdir(p)) > 0
            miss += 0 if ok else 1
            lines.append("| %s | %s | %d |" % (it, "PRESENT" if ok else "**MISSING**", size(p)))
        lines += ["", "MISSING items: %d. For each one write `NOT PRODUCED: <reason>` in a file of that name (or in report.md) and re-run this script." % miss]
        open(os.path.join(a.ret, "RETURN_CHECKLIST.md"), "w").write("\n".join(lines) + "\n")
        allf = files_of(a.ret, ".")
        mb = write_zip(single, a.ret, allf)
        print("level %d: single zip %.2f MB (limit %.1f MB)" % (level, mb, a.limit_mb))
        if mb * 1e6 <= lim:
            fits, level_used = True, level
            break
        os.remove(single)
    if fits:
        print("wrote: %s (%.2f MB) at thinning level %d; missing items %d" % (single, mb, level_used, miss))
        return 0
    print("LAST RESORT SPLIT: even level 3 gives %.2f MB. Largest items (bytes):" % mb)
    for f in sorted(allf, key=os.path.getsize, reverse=True)[:15]:
        print("   %10d  %s" % (os.path.getsize(f), os.path.relpath(f, a.ret)))
    print("Before accepting the split: can the host-side compressors (tools/r67/r67_compress.py, r67_collect.py levels) thin the largest items further without losing "
          "what their sufficiency statement promises? If yes, tighten and re-run. If not, the split below is allowed; say in report.md section 8 which items forced it.")
    big = set()
    for t in BIG_TABLES:
        big.update(files_of(a.ret, t))
    compact = set(files_of(a.ret, "compact"))
    part1 = [f for f in allf if f not in big and f not in compact]
    made = []
    p1 = "R67_RETURN_%s_part1.zip" % a.date
    mb1 = write_zip(p1, a.ret, part1); made.append((p1, mb1))
    if mb1 * 1e6 > lim:
        print("ERROR: part 1 is %.2f MB, over the limit; nothing more can be split off. Name the largest items in report.md." % mb1)
        return 2

    def pack(prefix, files):
        out, cur, cur_sz = [], [], 0
        for f in sorted(files):
            sz = os.path.getsize(f)
            if cur and cur_sz + sz > lim * 0.97:
                out.append(cur); cur, cur_sz = [], 0
            cur.append(f); cur_sz += sz
        if cur:
            out.append(cur)
        names = []
        for i, fl in enumerate(out):
            nm = "R67_RETURN_%s_%s%s.zip" % (a.date, prefix, "" if len(out) == 1 else chr(ord("a") + i))
            names.append((nm, write_zip(nm, a.ret, fl)))
        return names
    made += pack("part2", sorted(big))
    made += pack("part3", sorted(compact))
    with open("SPLIT_LOG.txt", "w") as f:
        f.write("single zip at level 3 would have been %.2f MB (limit %.1f MB)\nparts: %s\n" % (mb, a.limit_mb, ", ".join("%s %.2f MB" % m for m in made)))
    print("wrote: " + ", ".join("%s (%.2f MB)" % m for m in made) + "; missing items %d" % miss)
    return 0


if __name__ == "__main__":
    sys.exit(main())
