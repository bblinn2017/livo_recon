#!/usr/bin/env python3
"""R64: assemble the return zip and a checklist that says, item by item, what is present and what is missing. Nothing is interpreted.

python3 tools/r64/build_return.py --ret R64_RETURN --runs r64_runs --arms r64_manifest/arm_table.tsv --manifest r64_manifest \
        --queue-dir ablations/_job_queue/r64_eee01 --date 2026-09-30
Before running it, YOU write into --ret (agent-authored, see R64_INSTRUCTIONS.md section 9): report.md, STATE.md, agent_edits.patch,
code_answers.md, unit_tests/, checks/, bag_probe/. This script then adds: phase_summary.csv, arm_scan_all.csv.gz, arm_ol_tau_all.csv.gz,
plane_fit_reasons_all.csv.gz, plane_fit_trace_counts_all.csv.gz, completeness.csv, identity_controls.txt, summaries/, raw_keep/, queue/,
writes RETURN_CHECKLIST.md, and zips to R64_RETURN_<date>.zip. If the zip exceeds 28 MB it applies the documented cuts in order and
writes CUT_LOG.txt saying exactly what was cut. It never deletes agent-authored files."""
import argparse, glob, gzip, os, shutil, subprocess, sys, zipfile

REQUIRED_AGENT = ["report.md", "STATE.md", "agent_edits.patch", "code_answers.md", "unit_tests/ctest_output.txt",
                  "unit_tests/test_debias_modes_output.txt", "checks/smoke_table.md", "bag_probe"]
REQUIRED_DIRS_NONEMPTY = ["checks/param_dumps", "checks/refusals", "checks/smoke_arm_files"]
GENERATED = ["phase_summary.csv", "arm_scan_all.csv.gz", "arm_ol_tau_all.csv.gz", "plane_fit_reasons_all.csv.gz",
             "plane_fit_trace_counts_all.csv.gz", "completeness.csv", "identity_controls.txt", "summaries", "raw_keep", "queue"]

def size(p):
    if os.path.isdir(p):
        return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)
    return os.path.getsize(p) if os.path.exists(p) else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ret", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--arms", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--queue-dir", required=True); ap.add_argument("--date", required=True)
    ap.add_argument("--limit-mb", type=float, default=28.0)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.check_call([sys.executable, os.path.join(here, "r64_collect.py"), "--runs", a.runs, "--arms", a.arms, "--out", a.ret])
    # raw_keep: every arm's kept raw files (only arms with keep_raw=1 have state_trace/frame_stats/gt; every arm with a trace has it)
    rk = os.path.join(a.ret, "raw_keep"); os.makedirs(rk, exist_ok=True)
    for d in sorted(glob.glob(os.path.join(a.runs, "*/"))):
        arm = os.path.basename(d.rstrip("/"))
        keep = [f for f in ("state_trace.csv.gz", "frame_stats.txt.gz", "lio_gt_matched.csv.gz", "plane_fit_trace.csv.gz", "plane_fit_reasons.csv",
                            "state_reference.txt", "calibration_p0.txt", "raw_sha256.txt", "run_log_tail.txt", "reduce.log", "arm.json") if os.path.exists(os.path.join(d, f))]
        if not any(f in keep for f in ("state_trace.csv.gz", "plane_fit_trace.csv.gz", "run_log_tail.txt")):
            keep = [f for f in keep if f in ("raw_sha256.txt", "reduce.log", "arm.json")]
        if keep:
            os.makedirs(os.path.join(rk, arm), exist_ok=True)
            for f in keep:
                shutil.copy(os.path.join(d, f), os.path.join(rk, arm, f))
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
    # checklist
    lines = ["# R64 return checklist (generated; presence and size only)", ""]
    miss = 0
    lines.append("| item | status | bytes |"); lines.append("|---|---|---|")
    for it in REQUIRED_AGENT + REQUIRED_DIRS_NONEMPTY + GENERATED:
        p = os.path.join(a.ret, it)
        ok = os.path.exists(p) and (size(p) > 0 or it == "agent_edits.patch")
        if it in REQUIRED_DIRS_NONEMPTY:
            ok = os.path.isdir(p) and len(os.listdir(p)) > 0
        miss += 0 if ok else 1
        lines.append("| %s | %s | %d |" % (it, "PRESENT" if ok else "**MISSING**", size(p)))
    lines += ["", "MISSING items: %d. For each one write `NOT PRODUCED: <reason>` in a file of that name (or in report.md) and re-run this script." % miss]
    open(os.path.join(a.ret, "RETURN_CHECKLIST.md"), "w").write("\n".join(lines) + "\n")
    # zip with cuts
    def make_zip():
        zp = "R64_RETURN_%s.zip" % a.date
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for d, _, fs in os.walk(a.ret):
                for f in fs:
                    z.write(os.path.join(d, f), os.path.relpath(os.path.join(d, f), os.path.dirname(a.ret.rstrip("/"))))
        return zp, os.path.getsize(zp) / 1e6
    zp, mb = make_zip()
    cuts = []
    if mb > a.limit_mb:   # cut 1: raw state_trace of all but the first three arms that have one
        traces = sorted(glob.glob(os.path.join(rk, "*", "state_trace.csv.gz")))
        for t in traces[3:]:
            cuts.append("removed " + os.path.relpath(t, a.ret)); os.remove(t)
        zp, mb = make_zip()
    if mb > a.limit_mb:   # cut 2: frame_stats and gt of kept arms
        for t in sorted(glob.glob(os.path.join(rk, "*", "frame_stats.txt.gz")) + glob.glob(os.path.join(rk, "*", "lio_gt_matched.csv.gz"))):
            cuts.append("removed " + os.path.relpath(t, a.ret)); os.remove(t)
        zp, mb = make_zip()
    if mb > a.limit_mb:   # cut 3: plane_fit_trace of arms other than P_pca / P_debiased_full
        for t in sorted(glob.glob(os.path.join(rk, "*", "plane_fit_trace.csv.gz"))):
            arm = os.path.basename(os.path.dirname(t))
            if arm not in ("P_pca_eigengap_60s", "P_debiased_full_eigengap_60s"):
                cuts.append("removed " + os.path.relpath(t, a.ret)); os.remove(t)
        zp, mb = make_zip()
    with open(os.path.join(a.ret, "CUT_LOG.txt"), "w") as f:
        f.write("zip size before cuts and after: see below\n" + "\n".join(cuts) + "\nfinal zip MB: %.2f (limit %.1f)\n" % (mb, a.limit_mb))
    zp, mb = make_zip()
    print("wrote %s (%.2f MB), missing items %d, cuts %d" % (zp, mb, miss, len(cuts)))
    if mb > 30:
        print("ERROR: zip over 30 MB even after cuts; say so in report.md and split the return in two zips (part 1: everything except raw_keep)")
        return 2

if __name__ == "__main__":
    sys.exit(main())
