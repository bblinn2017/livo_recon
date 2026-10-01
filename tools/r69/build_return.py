#!/usr/bin/env python3
"""R73: assemble the return, check it against required_outputs.json item by item, byte-split it when it is over the size limit. Nothing is interpreted.

python3 tools/r69/build_return.py --ret R73_RETURN --runs r73_runs --arms r73_manifest/arm_table.tsv --manifest r73_manifest \\
        --queue-dir ablations/_job_queue/r73_smoke,ablations/_job_queue/r73_refuse,ablations/_job_queue/r73_sh,... --date 2026-10-02 [--params r69_inputs/bag_params.json] [--allow-missing]
Before running it, YOU write into --ret (agent-authored, R73_INSTRUCTIONS.md section 10): report.md, STATE.md, agent_edits.patch, code_answers.md, R73_tree.tar.gz, unit_tests/, checks/
(incl. r73_symmetrize_gate.txt, the selftest outputs), launch/, listener/, stop_reasons.csv and STEP0/ (python tools/r69/r73_step0.py --out R73_RETURN/STEP0).
This script adds: phase_summary.csv, arm_scan_all.csv.gz, completeness.csv, identity_controls.txt, r73_gates.txt, r73_shadow_pairs.*, r73_freeze_check.csv, r73_p_table/, engagement_all.txt,
duplicates.tsv, summaries_all.json, compress_manifests.json, sensor_stamps_summaries.json, compact/, queue/, RETURN_CHECKLIST.md.
RETURN_CHECKLIST.md is derived from r73_manifest/required_outputs.json: one row per (follow-up, arm or table, file): follow-up id, file, columns, minimum rows, where it is, status.
If ANY required item is absent the script prints the follow-up ids and files and EXITS 3 WITHOUT packing (an item you cannot produce is written as `NOT PRODUCED: <reason>` into a file of that
name and the script re-run with --allow-missing; the reason also goes in report.md).
NOTHING is thinned or dropped. The zip is built once; if it is over --limit-mb (default 28) it is BYTE-SPLIT into R73_RETURN_<date>.zip.part00, .part01, ... (28 MB each, fixed size) with
R73_return.sha256 (sha256 and size of the whole zip and of every part), REASSEMBLE.txt and reassembly_check.txt (the parts are re-joined and the sha256 compared; a copy with one part dropped is
re-joined and MUST fail the comparison; both results are printed)."""
import argparse, csv, glob, gzip, hashlib, io, json, os, shutil, subprocess, sys, zipfile
import pandas as pd

REQUIRED_AGENT = ["report.md", "STATE.md", "agent_edits.patch", "code_answers.md", "R73_tree.tar.gz",
                  "unit_tests/ctest_output.txt", "checks/smoke_table.md", "checks/build_log_tail.txt", "checks/r73_symmetrize_gate.txt",
                  "launch/launch_command.txt", "launch/containers_file.txt", "launch/launch_time.txt", "launch/checkup_cycles",
                  "listener/selftest_output.txt", "listener/smoke_dry_run/listener_ledger.jsonl", "listener/smoke_delete/listener_ledger.jsonl",
                  "listener/smoke/listener_ledger.jsonl", "listener/smoke/listener_final.json", "stop_reasons.csv", "STEP0/step0_missing.txt"]
REQUIRED_DIRS_NONEMPTY = ["listener", "checks/param_dumps", "checks/smoke_arm_files", "launch/checkup_cycles", "STEP0"]
GENERATED = ["phase_summary.csv", "arm_scan_all.csv.gz", "completeness.csv", "identity_controls.txt", "r73_gates.txt", "duplicates.tsv", "summaries_all.json",
             "compress_manifests.json", "sensor_stamps_summaries.json", "engagement_all.txt", "compact", "queue"]
PART = 28 * 1000 * 1000


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
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for f in files:
            z.write(f, os.path.relpath(os.path.abspath(f), base))
    return os.path.getsize(zp)


def sha(path, start=0, length=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        left = length
        while True:
            n = 1 << 22 if left is None else min(1 << 22, left)
            if n == 0:
                break
            b = f.read(n)
            if not b:
                break
            h.update(b)
            if left is not None:
                left -= len(b)
    return h.hexdigest()


def byte_split(zp, part=PART):
    """Fixed-size byte split: zp.part00, zp.part01, ... (equivalent to `split -b 28M -d -a 2`). Returns [(name, bytes, sha256)]."""
    out = []
    with open(zp, "rb") as f:
        i = 0
        while True:
            b = f.read(part)
            if not b:
                break
            nm = "%s.part%02d" % (zp, i)
            open(nm, "wb").write(b)
            out.append((nm, len(b), hashlib.sha256(b).hexdigest()))
            i += 1
    return out


def join_sha(parts):
    h = hashlib.sha256()
    for p in parts:
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
    return h.hexdigest()


def reassembly_proof(zp, parts, out_txt):
    whole = sha(zp)
    full_ok = join_sha([p[0] for p in parts]) == whole
    drop_ok = None
    if len(parts) >= 2:
        drop_ok = join_sha([p[0] for p in parts[:1] + parts[2:]]) != whole      # one part dropped: the sha MUST differ
    lines = ["whole zip sha256 %s (%d bytes)" % (whole, os.path.getsize(zp)), "parts: %d" % len(parts),
             "reassembly of all parts equals the whole: %s" % ("PASS" if full_ok else "FAIL"),
             "reassembly with part01 dropped is rejected (sha differs): %s" % ("PASS" if drop_ok else ("FAIL" if drop_ok is not None else "n/a (single part)"))]
    open(out_txt, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return full_ok and drop_ok in (True, None)


def count_rows(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as f:
        return max(sum(1 for _ in f) - 1, 0)


def header(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as f:
        return f.readline().strip().split(",")


def check_table(path, cols, min_rows):
    if not os.path.exists(path):
        return "MISSING", "file absent"
    if os.path.getsize(path) == 0:
        return "MISSING", "empty file"
    if cols or min_rows > 1:
        if path.endswith((".csv", ".csv.gz")):
            h = header(path)
            miss = [c for c in cols if c not in h]
            if miss:
                return "FAIL", "columns absent: %s" % miss
            n = count_rows(path)
            if n < min_rows:
                return "FAIL", "rows %d < %d" % (n, min_rows)
            return "PRESENT", "rows %d" % n
    return "PRESENT", "%d bytes" % os.path.getsize(path)


def check_required(ret, runs, spec):
    """-> list of rows (followup, item, file, where, status, detail)."""
    rows = []
    compact = os.path.join(ret, "compact")
    for e in spec:
        fu, f, src, where = e["followup"], e["file"], e["source"], e["where"]
        arms = e["arms"]
        if src == "arm" and where == "return":
            for a in arms:
                st, dt = check_table(os.path.join(compact, a, f), e["columns"], e["min_rows"])
                rows.append((fu, a, f, where, st, dt))
        elif src == "arm" and where == "archive":
            for a in arms:
                raw = os.path.join(runs, a, "raw_sha256.txt")
                txt = open(raw).read() if os.path.exists(raw) else ""
                rows.append((fu, a, f, where, "PRESENT" if f in txt else "MISSING", "listed in raw_sha256.txt" if f in txt else "not in the lossless archive listing of the arm"))
        elif src == "generated":
            p = os.path.join(ret, f)
            st, dt = check_table(p, e["columns"], e["min_rows"])
            rows.append((fu, "-", f, where, st, dt))
            if st == "PRESENT" and e.get("contains"):
                txt = open(p, errors="replace").read()
                for a, keys in e["contains"].items():
                    blk = txt.split("### " + a)[1].split("### ")[0] if ("### " + a) in txt else ""
                    for k in keys:
                        rows.append((fu, a, f + " contains " + k, where, "PRESENT" if k in blk else "MISSING", ""))
        elif src == "agent":
            st, dt = check_table(os.path.join(ret, "checks", f), e["columns"], e["min_rows"])
            rows.append((fu, "-", "checks/" + f, where, st, dt))
        elif src in ("step0", "step0_raw"):
            for a in arms:
                st, dt = check_table(os.path.join(ret, "STEP0", a, f), e["columns"], e["min_rows"])
                rows.append((fu, a, "STEP0/%s/%s" % (a, f), where, st, dt))
        elif src == "step0_table":
            p = os.path.join(ret, "STEP0", f)
            st, dt = check_table(p, e["columns"], e["min_rows"])
            rows.append((fu, "-", "STEP0/" + f, where, st, dt))
            if st == "PRESENT":
                have = set(pd.read_csv(p, usecols=["arm"])["arm"].unique())
                for a in arms:
                    rows.append((fu, a, "STEP0/%s has rows for the arm" % f, where, "PRESENT" if a in have else "MISSING", ""))
    return rows


def engagement_all(runs, arms, out):
    with open(out, "w") as o:
        for a in arms:
            d = os.path.join(runs, a)
            o.write("### %s\n" % a)
            for n in ("engagement.txt", "engagement_imu.txt", "engagement_imu_shadow.txt"):
                p = os.path.join(d, n)
                if os.path.exists(p):
                    o.write("--- %s\n%s\n" % (n, open(p, errors="replace").read()))
            o.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ret", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--arms", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--queue-dir", required=True); ap.add_argument("--date", required=True)
    ap.add_argument("--params", default=None); ap.add_argument("--limit-mb", type=float, default=28.0); ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    q = os.path.join(a.ret, "queue"); os.makedirs(q, exist_ok=True)
    for f in glob.glob(os.path.join(a.manifest, "*")):
        if os.path.isfile(f):
            shutil.copy(f, q)
    with open(os.path.join(q, "queue_state.txt"), "w") as f:
        n = {"DONE": 0, "FAILED": 0, "neither": 0}
        for jd in sorted(j for qd in a.queue_dir.split(",") for j in glob.glob(os.path.join(qd, "*/"))):
            st = "DONE" if os.path.exists(jd + "DONE") else ("FAILED" if os.path.exists(jd + "FAILED") else "neither")
            n[st] += 1
            f.write("%s\t%s\n" % (os.path.basename(jd.rstrip("/")), st))
        f.write("# totals: %s\n" % n)
    cmd = [sys.executable, os.path.join(here, "r69_collect.py"), "--runs", a.runs, "--arms", a.arms, "--out", a.ret, "--level", "0"]
    if a.params:
        cmd += ["--params", a.params]
    subprocess.check_call(cmd)
    pairs_f = os.path.join(a.manifest, "coupled_pairs.txt")
    if os.path.exists(pairs_f):
        subprocess.call([sys.executable, os.path.join(here, "r71_spline_tools.py"), "pairs", "--scan-table", os.path.join(a.ret, "arm_scan_all.csv.gz"), "--pairs", open(pairs_f).read().strip(), "--out", a.ret])
    sp = os.path.join(a.manifest, "shadow_pairs.json")
    if os.path.exists(sp):
        subprocess.call([sys.executable, os.path.join(here, "r73_shadow_reduce.py"), "pairs", "--runs", a.runs, "--pairs", sp, "--out", a.ret])
    subprocess.call([sys.executable, os.path.join(here, "r73_gates.py"), "--runs", a.runs, "--out", os.path.join(a.ret, "r73_gates.txt")])   # FAIL is reported in the file, not fatal here
    if os.path.exists(os.path.join(a.ret, "phase_summary.csv")) and a.params:
        subprocess.call([sys.executable, os.path.join(here, "r73_p_table.py"), "--summary", os.path.join(a.ret, "phase_summary.csv"), "--compact", os.path.join(a.ret, "compact"), "--params", a.params, "--out", os.path.join(a.ret, "r73_p_table")])
        for f in ("r73_p2_cells.csv",):
            if os.path.exists(os.path.join(a.ret, "r73_p_table", f)):
                shutil.copy(os.path.join(a.ret, "r73_p_table", f), os.path.join(a.ret, f))
    arm_ids = [r["arm"] for r in csv.DictReader(open(a.arms), delimiter="\t")]
    engagement_all(a.runs, arm_ids, os.path.join(a.ret, "engagement_all.txt"))
    spec = json.load(open(os.path.join(a.manifest, "required_outputs.json")))
    rows = check_required(a.ret, a.runs, spec)
    miss = [r for r in rows if r[4] != "PRESENT"]
    lines = ["# R73 return checklist (generated from required_outputs.json; presence, columns and row counts only)", "", "| follow-up | arm | file | where | status | detail |", "|---|---|---|---|---|---|"]
    lines += ["| %s | %s | %s | %s | %s | %s |" % (r[0], r[1], r[2], r[3], r[4] if r[4] == "PRESENT" else "**%s**" % r[4], r[5]) for r in rows]
    lines += ["", "## agent-authored and generated items", "", "| item | status | bytes |", "|---|---|---|"]
    gmiss = 0
    for it in REQUIRED_AGENT + REQUIRED_DIRS_NONEMPTY + GENERATED:
        p = os.path.join(a.ret, it)
        ok = os.path.exists(p) and (size(p) > 0 or it == "agent_edits.patch")
        if it in REQUIRED_DIRS_NONEMPTY or it == "compact":
            ok = os.path.isdir(p) and len(os.listdir(p)) > 0
        gmiss += 0 if ok else 1
        lines.append("| %s | %s | %d |" % (it, "PRESENT" if ok else "**MISSING**", size(p)))
    fus = sorted({r[0] for r in miss})
    lines += ["", "MISSING or FAILED required-output rows: %d (follow-ups: %s). MISSING agent/generated items: %d." % (len(miss), ", ".join(fus) or "none", gmiss)]
    open(os.path.join(a.ret, "RETURN_CHECKLIST.md"), "w").write("\n".join(lines) + "\n")
    if (miss or gmiss) and not a.allow_missing:
        print("NOT PACKED: required items absent. Follow-ups: %s" % ", ".join(fus))
        for r in miss[:60]:
            print("   %s  %s  %s  [%s] %s" % (r[0], r[1], r[2], r[4], r[5]))
        print("   (%d more)" % max(0, len(miss) - 60) if len(miss) > 60 else "", "agent/generated missing: %d (see RETURN_CHECKLIST.md)" % gmiss)
        return 3
    zp = "R73_RETURN_%s.zip" % a.date
    allf = files_of(a.ret, ".")
    nbytes = write_zip(zp, a.ret, allf)
    print("zip %s %.2f MB (%d files)" % (zp, nbytes / 1e6, len(allf)))
    if nbytes <= a.limit_mb * 1e6:
        open("R73_return.sha256", "w").write("%s  %s  %d\n" % (sha(zp), zp, nbytes))
        print("single file; sha256 in R73_return.sha256")
        return 0
    parts = byte_split(zp)
    with open("R73_return.sha256", "w") as f:
        f.write("%s  %s  %d  (whole zip)\n" % (sha(zp), zp, nbytes))
        for nm, n, h in parts:
            f.write("%s  %s  %d\n" % (h, os.path.basename(nm), n))
    open("REASSEMBLE.txt", "w").write("cat %s.part* > %s && sha256sum %s   # the hash must equal the first line of R73_return.sha256; unzip only then\n" % (zp, zp, zp))
    ok = reassembly_proof(zp, parts, "reassembly_check.txt")
    os.remove(zp)       # only the parts and the checksum files are delivered
    print("byte-split into %d parts of %d bytes: %s" % (len(parts), PART, ", ".join(os.path.basename(p[0]) for p in parts[:3]) + (" ..." if len(parts) > 3 else "")))
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
