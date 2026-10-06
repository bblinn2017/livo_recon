#!/usr/bin/env python3
"""R74 (= R73 build_return.py ported: no STEP0, switch reducer in place of the shadow-pair reducer, no p_table): assemble the return, check it against required_outputs.json item by item, byte-split it when it is over the size limit. Nothing is interpreted.

python3 tools/r69/build_return.py --ret R74_RETURN --runs r74_runs --arms r74_manifest/arm_table.tsv --manifest r74_manifest --r73-runs <R73 results dir> \\
        --queue-dir ablations/_job_queue/r74_smoke,ablations/_job_queue/r74_ef,... --date 2026-10-02 [--params r69_inputs/bag_params.json] [--allow-missing]
Before running it, YOU write into --ret (agent-authored, R74_INSTRUCTIONS.md section 13): report.md, STATE.md, agent_edits.patch, code_answers.md, R74_tree.tar.gz, unit_tests/, checks/
(incl. r74_symmetrize_gate.txt, the selftest outputs), launch/, listener/ and stop_reasons.csv.
This script adds: phase_summary.csv, arm_scan_all.csv.gz, completeness.csv, identity_controls.txt, r74_gates.txt, r74_f426_part1_*.csv, r74_f426_part2_*.csv, engagement_all.txt,
duplicates.tsv, summaries_all.json, compress_manifests.json, sensor_stamps_summaries.json, compact/, queue/, RETURN_CHECKLIST.md.
RETURN_CHECKLIST.md is derived from r74_manifest/required_outputs.json: one row per (follow-up, arm or table, file): follow-up id, file, columns, minimum rows, where it is, status.
If ANY required item is absent the script prints the follow-up ids and files and EXITS 3 WITHOUT packing (an item you cannot produce is written as `NOT PRODUCED: <reason>` into a file of that
name and the script re-run with --allow-missing; the reason also goes in report.md).
NOTHING is thinned or dropped. The zip is built once; if it is over --limit-mb (default 28) it is BYTE-SPLIT into R74_RETURN_<date>.zip.part00, .part01, ... (28 MB each, fixed size) with
R74_return.sha256 (sha256 and size of the whole zip and of every part), REASSEMBLE.txt and reassembly_check.txt (the parts are re-joined and the sha256 compared; a copy with one part dropped is
re-joined and MUST fail the comparison; both results are printed)."""
import argparse, csv, glob, gzip, hashlib, io, json, os, shutil, subprocess, sys, zipfile
import pandas as pd

REQUIRED_AGENT = ["report.md", "STATE.md", "agent_edits.patch", "code_answers.md", "R74_tree.tar.gz",
                  "unit_tests/ctest_output.txt", "checks/smoke_table.md", "checks/build_log_tail.txt", "checks/r74_symmetrize_gate.txt",
                  "launch/launch_command.txt", "launch/containers_file.txt", "launch/launch_time.txt", "launch/checkup_cycles",
                  "listener/selftest_output.txt", "listener/smoke_dry_run/listener_ledger.jsonl", "listener/smoke_delete/listener_ledger.jsonl",
                  "listener/smoke/listener_ledger.jsonl", "listener/smoke/listener_final.json", "stop_reasons.csv", "checks/r74_selftest_output.txt"]
REQUIRED_DIRS_NONEMPTY = ["listener", "checks/param_dumps", "checks/smoke_arm_files", "launch/checkup_cycles"]
GENERATED = ["phase_summary.csv", "arm_scan_all.csv.gz", "completeness.csv", "identity_controls.txt", "r74_gates.txt", "duplicates.tsv", "summaries_all.json",
             "compress_manifests.json", "sensor_stamps_summaries.json", "engagement_all.txt", "compact", "queue"]
# R75: lists for --round R75 (the R74 listener proof and smoke stage are not repeated; one queue, r75_ln)
REQUIRED_AGENT_R75 = ["report.md", "STATE.md", "agent_edits.patch", "code_answers.md", "R75_tree.tar.gz", "unit_tests/ctest_output.txt", "checks/build_log_tail.txt",
                      "checks/r75_selftest_output.txt", "launch/launch_command.txt", "launch/containers_file.txt", "launch/launch_time.txt", "launch/checkup_cycles",
                      "listener/ln/listener_ledger.jsonl", "listener/ln/listener_final.json", "stop_reasons.csv", "harness_answers.md"]
REQUIRED_DIRS_NONEMPTY_R75 = ["listener", "launch/checkup_cycles", "checks"]
GENERATED_R75 = ["phase_summary.csv", "arm_scan_all.csv.gz", "completeness.csv", "identity_controls.txt", "r75_gates.txt", "duplicates.tsv", "summaries_all.json",
                 "compress_manifests.json", "sensor_stamps_summaries.json", "engagement_all.txt", "compact", "queue"]
# R76: two listener runs (id + refuse, then the full-bag grid), the voxel-size statistics, the F-426 step-0 outputs written by r76_step0_f426.py (not by this script)
REQUIRED_AGENT_R76 = ["report.md", "STATE.md", "agent_edits.patch", "code_answers.md", "R76_tree.tar.gz", "unit_tests/ctest_output.txt", "checks/build_log_tail.txt",
                      "checks/r76_selftest_output.txt", "launch/launch_command.txt", "launch/containers_file.txt", "launch/launch_time.txt", "launch/checkup_cycles",
                      "listener/id/listener_ledger.jsonl", "listener/id/listener_final.json", "listener/fb/listener_ledger.jsonl", "listener/fb/listener_final.json",
                      "stop_reasons.csv", "harness_answers.md", "step0_f426_log.txt", "voxel_stats/voxel_stats_run_log.txt"]
REQUIRED_DIRS_NONEMPTY_R76 = ["listener", "launch/checkup_cycles", "checks", "voxel_stats"]
GENERATED_R76 = ["phase_summary.csv", "arm_scan_all.csv.gz", "completeness.csv", "identity_controls.txt", "r76_gates.txt", "r76_identity_vs_old.csv", "duplicates.tsv", "summaries_all.json",
                 "compress_manifests.json", "sensor_stamps_summaries.json", "engagement_all.txt", "compact", "queue"]
# R77: id + refuse listener run, then the full-bag run (no F-426 step 0, no voxel statistics in this round)
REQUIRED_AGENT_R77 = ["report.md", "STATE.md", "agent_edits.patch", "code_answers.md", "R77_tree.tar.gz", "unit_tests/ctest_output.txt", "checks/build_log_tail.txt",
                      "checks/r77_selftest_output.txt", "launch/launch_command.txt", "launch/containers_file.txt", "launch/launch_time.txt", "launch/checkup_cycles",
                      "listener/id/listener_ledger.jsonl", "listener/id/listener_final.json", "listener/fb/listener_ledger.jsonl", "listener/fb/listener_final.json",
                      "stop_reasons.csv", "harness_answers.md"]
REQUIRED_DIRS_NONEMPTY_R77 = ["listener", "launch/checkup_cycles", "checks"]
GENERATED_R77 = ["phase_summary.csv", "arm_scan_all.csv.gz", "completeness.csv", "identity_controls.txt", "r77_gates.txt", "r77_identity_vs_old.csv", "duplicates.tsv", "summaries_all.json",
                 "compress_manifests.json", "sensor_stamps_summaries.json", "engagement_all.txt", "compact", "queue"]
# R78: id + refuse + smoke listener run (queues r78_id, r78_ref), then the full-bag run (r78_fb)
REQUIRED_AGENT_R78 = ["report.md", "STATE.md", "agent_edits.patch", "code_answers.md", "R78_tree.tar.gz", "unit_tests/ctest_output.txt", "checks/build_log_tail.txt",
                      "checks/r78_selftest_output.txt", "launch/launch_command.txt", "launch/containers_file.txt", "launch/launch_time.txt", "launch/checkup_cycles",
                      "listener/id/listener_ledger.jsonl", "listener/id/listener_final.json", "listener/fb/listener_ledger.jsonl", "listener/fb/listener_final.json",
                      "stop_reasons.csv", "harness_answers.md"]
REQUIRED_DIRS_NONEMPTY_R78 = ["listener", "launch/checkup_cycles", "checks"]
GENERATED_R78 = ["phase_summary.csv", "arm_scan_all.csv.gz", "completeness.csv", "identity_controls.txt", "r78_gates.txt", "r78_identity_vs_old.csv", "duplicates.tsv", "summaries_all.json",
                 "compress_manifests.json", "sensor_stamps_summaries.json", "engagement_all.txt", "compact", "queue"]
# R79: id + refuse + smoke listener run (queues r79_id, r79_ref), then the early + full-bag run (r79_ef, r79_fb)
REQUIRED_AGENT_R79 = ["report.md", "STATE.md", "agent_edits.patch", "code_answers.md", "R79_tree.tar.gz", "unit_tests/ctest_output.txt", "checks/build_log_tail.txt",
                      "checks/r79_selftest_output.txt", "launch/launch_command.txt", "launch/containers_file.txt", "launch/launch_time.txt", "launch/checkup_cycles",
                      "listener/id/listener_ledger.jsonl", "listener/id/listener_final.json", "listener/fb/listener_ledger.jsonl", "listener/fb/listener_final.json",
                      "stop_reasons.csv", "harness_answers.md"]
REQUIRED_DIRS_NONEMPTY_R79 = ["listener", "launch/checkup_cycles", "checks"]
GENERATED_R79 = ["phase_summary.csv", "arm_scan_all.csv.gz", "completeness.csv", "identity_controls.txt", "r79_gates.txt", "r79_identity_vs_old.csv", "duplicates.tsv", "summaries_all.json",
                 "compress_manifests.json", "sensor_stamps_summaries.json", "engagement_all.txt", "compact", "queue"]
# R80: id (the pfn 3 window arm: smoke + identity) then the run (window arms + whole-bag message logs)
REQUIRED_AGENT_R80 = ["report.md", "STATE.md", "agent_edits.patch", "code_answers.md", "R80_tree.tar.gz", "unit_tests/ctest_output.txt", "checks/build_log_tail.txt",
                      "checks/r80_selftest_output.txt", "launch/launch_command.txt", "launch/containers_file.txt", "launch/launch_time.txt", "launch/checkup_cycles",
                      "listener/id/listener_ledger.jsonl", "listener/id/listener_final.json",   # R81 fix (item 9): the R80 listener layout has only the id stream and the run stream is not a listener output
                      "stop_reasons.csv", "harness_answers.md"]
REQUIRED_DIRS_NONEMPTY_R80 = ["listener", "launch/checkup_cycles", "checks"]
GENERATED_R80 = ["phase_summary.csv", "arm_scan_all.csv.gz", "completeness.csv", "identity_controls.txt", "r80_gates.txt", "r80_identity_vs_old.csv", "duplicates.tsv", "summaries_all.json",
                 "compress_manifests.json", "sensor_stamps_summaries.json", "engagement_all.txt", "compact", "queue"]
# R81: id (identity + smoke + refuse arms, runs FIRST and alone) then nine bag listeners (one spec per bag covers its two queues r81_<bag> and r81_<bag>_s, rule 1.16).
# listener/<name>/ = the listener ledger of that spec. The identity-hold result (r81_identity_hold.py output) is agent-authored.
R81_BAGS = ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]
REQUIRED_AGENT_R81 = ["report.md", "STATE.md", "agent_edits.patch", "code_answers.md", "R81_tree.tar.gz", "unit_tests/ctest_output.txt", "checks/build_log_tail.txt",
                      "checks/r81_selftest_output.txt", "checks/r81_identity_hold.txt", "launch/launch_command.txt", "launch/containers_file.txt", "launch/launch_time.txt", "launch/checkup_cycles",
                      "listener/id/listener_ledger.jsonl", "listener/id/listener_final.json"] + \
                     ["listener/%s/%s" % (b, f) for b in R81_BAGS for f in ("listener_ledger.jsonl", "listener_final.json")] + ["stop_reasons.csv", "harness_answers.md"]
REQUIRED_DIRS_NONEMPTY_R81 = ["listener", "launch/checkup_cycles", "checks"]
GENERATED_R81 = ["phase_summary.csv", "arm_scan_all.csv.gz", "completeness.csv", "identity_controls.txt", "r81_gates.txt", "r81_identity_vs_old.csv", "duplicates.tsv", "summaries_all.json",
                 "compress_manifests.json", "sensor_stamps_summaries.json", "engagement_all.txt", "plane_fit_trace_counts_all.csv", "compact", "queue"]   # R81 (F-469): the collector writes plane_fit_trace_counts_all.csv itself
PART = 28 * 1000 * 1000
STORED_EXTS = (".gz", ".zip")   # already-compressed; ZIP_STORED saves time, not space (a .tar.gz is caught by .gz)


def excluded_from_zip(relpath):
    """Return-trim (Bryce, R74): not named in required_outputs.json, already returned elsewhere, or a local-only copy.
    relpath is the path as it will appear inside the zip (R74_RETURN/... or R75_RETURN/... form)."""
    parts = relpath.split(os.sep)
    if len(parts) >= 3 and parts[-3] == "smoke_arms" and parts[-1] == "arm_bundle.zip":
        return True   # already returned in stage 1b; keep only each arm's summary.json (R74 only -- R75 has no smoke_arms)
    if "queue" in parts[:-1] and os.path.basename(relpath) != "queue_state.txt":
        return True   # copies of the planner's own manifest/; only queue_state.txt (derived, not a copy) belongs in the return
    return False


def choose_part_size(nbytes, target=PART):
    """N-5/N-10: fixed-size parts, equivalent to `split -b 28M`. (Earlier versions spread the bytes evenly, which produced equal thirds against the documented 28 MB parts.)"""
    return target


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
    with zipfile.ZipFile(zp, "w", allowZip64=True) as z:
        for f in files:
            ct = zipfile.ZIP_STORED if f.lower().endswith(STORED_EXTS) else zipfile.ZIP_DEFLATED
            z.write(f, os.path.relpath(os.path.abspath(f), base), compress_type=ct)
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


STEP0_KEEP = ("STEP0/step0_missing.txt", "STEP0/STEP0_SIZES.txt", "STEP0/STEP0_SHA256.txt")
DELIVERED = "DELIVERED_IN_STAGE1B"


def step0_index(ret):
    """Paths listed in STEP0/STEP0_SHA256.txt (the index of what the stage-1b return already delivered); None if absent."""
    p = os.path.join(ret, "STEP0", "STEP0_SHA256.txt")
    if not os.path.exists(p):
        return None
    out = set()
    for ln in open(p):
        parts = ln.split(None, 1)
        if len(parts) == 2:
            out.add(parts[1].strip().replace("\\", "/"))
    return out


def check_required(ret, runs, spec, idx=None):
    """-> list of rows (followup, item, file, where, status, detail). idx: set of STEP0 paths already delivered in the stage-1b return (only with --step0-delivered):
    a step-0 item whose file is not here but IS in that index is DELIVERED_IN_STAGE1B (satisfied); a file in neither is MISSING."""
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
                rel = "STEP0/%s/%s" % (a, f)
                st, dt = check_table(os.path.join(ret, "STEP0", a, f), e["columns"], e["min_rows"])
                if st != "PRESENT" and idx is not None and rel in idx:
                    st, dt = DELIVERED, "listed in STEP0_SHA256.txt (delivered in the stage-1b return)"
                rows.append((fu, a, rel, where, st, dt))
        elif src == "step0_table":
            p = os.path.join(ret, "STEP0", f)
            st, dt = check_table(p, e["columns"], e["min_rows"])
            if st != "PRESENT" and idx is not None and "STEP0/" + f in idx:
                rows.append((fu, "-", "STEP0/" + f, where, DELIVERED, "listed in STEP0_SHA256.txt (delivered in the stage-1b return)"))
                continue
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
    ap.add_argument("--r73-runs", default=None, help="directory holding the R73 arm dirs (r73_sh_* shadow arms and plain controls) for the F-426 comparison")
    ap.add_argument("--round", default="R74", choices=["R74", "R75", "R76", "R77", "R78", "R79", "R80", "R81"], help="R75 / R76 / R77: that round's names, required-agent list and gates script (R7x_INSTRUCTIONS.md section 13)")
    ap.add_argument("--gate-args", default=None, help="R81: extra arguments for r81_gates.py as one quoted string (e.g. \"--grid-retime r80c_grid.json --grid-bag grid_check/r80d_grid_bag.json --early-grid <file>\")")
    ap.add_argument("--step0-delivered", action="store_true", help="STEP0 data was delivered in the stage-1b return: check it against STEP0/STEP0_SHA256.txt and do NOT pack it again (only the three index files are packed)")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    RND = a.round
    gates_py, gates_txt = {"R81": ("r81_gates.py", "r81_gates.txt"), "R80": ("r80_gates.py", "r80_gates.txt"), "R79": ("r79_gates.py", "r79_gates.txt"), "R78": ("r78_gates.py", "r78_gates.txt"), "R77": ("r77_gates.py", "r77_gates.txt"), "R76": ("r76_gates.py", "r76_gates.txt"), "R75": ("r75_gates.py", "r75_gates.txt")}.get(RND, ("r74_gates.py", "r74_gates.txt"))
    req_agent, req_dirs, gen_items = {"R81": (REQUIRED_AGENT_R81, REQUIRED_DIRS_NONEMPTY_R81, GENERATED_R81), "R80": (REQUIRED_AGENT_R80, REQUIRED_DIRS_NONEMPTY_R80, GENERATED_R80), "R79": (REQUIRED_AGENT_R79, REQUIRED_DIRS_NONEMPTY_R79, GENERATED_R79), "R78": (REQUIRED_AGENT_R78, REQUIRED_DIRS_NONEMPTY_R78, GENERATED_R78), "R77": (REQUIRED_AGENT_R77, REQUIRED_DIRS_NONEMPTY_R77, GENERATED_R77), "R76": (REQUIRED_AGENT_R76, REQUIRED_DIRS_NONEMPTY_R76, GENERATED_R76), "R75": (REQUIRED_AGENT_R75, REQUIRED_DIRS_NONEMPTY_R75, GENERATED_R75)}.get(RND, (REQUIRED_AGENT, REQUIRED_DIRS_NONEMPTY, GENERATED))
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
    cmd = [sys.executable, os.path.join(here, "r69_collect.py"), "--runs", a.runs, "--arms", a.arms, "--out", a.ret, "--level", ("-1" if a.round in ("R80", "R81") else "0")]   # R80 (F-381): no thinning; the SW sweep arms are kept out of arm_scan_all by the collector
    if a.params:
        cmd += ["--params", a.params]
    subprocess.check_call(cmd)
    pairs_f = os.path.join(a.manifest, "coupled_pairs.txt")
    if os.path.exists(pairs_f):
        subprocess.call([sys.executable, os.path.join(here, "r71_spline_tools.py"), "pairs", "--scan-table", os.path.join(a.ret, "arm_scan_all.csv.gz"), "--pairs", open(pairs_f).read().strip(), "--out", a.ret])
    sp = os.path.join(a.manifest, "switch_pairs.json")
    if os.path.exists(sp) and a.r73_runs and RND not in ("R76", "R77", "R78", "R79", "R80"):   # R76: the F-426 tables come from r76_step0_f426.py (the archive layout and the missing t_rel_s are handled there)
        subprocess.call([sys.executable, os.path.join(here, "r74_switch_reduce.py"), "part1", "--shadow-dirs", a.r73_runs, "--pairs", sp, "--out", a.ret])
        subprocess.call([sys.executable, os.path.join(here, "r74_switch_reduce.py"), "part2", "--runs", a.runs, "--controls", a.r73_runs, "--pairs", sp, "--out", a.ret])
    gcmd = [sys.executable, os.path.join(here, gates_py), "--runs", a.runs, "--out", os.path.join(a.ret, gates_txt)]
    if RND in ("R76", "R77", "R78", "R79"):
        gcmd += ["--refs", os.path.join(a.manifest, "identity_refs.json"), "--csv", os.path.join(a.ret, "%s_identity_vs_old.csv" % RND.lower())]
        if RND in ("R78", "R79"):
            gcmd += ["--refs-fb", os.path.join(a.manifest, "identity_refs_fb.csv.gz")]
    if RND == "R80":   # R80 gates: identity against R79 positions only (no identity_refs.json)
        gcmd += ["--refs-fb", os.path.join(a.manifest, "identity_refs_fb.csv.gz"), "--csv", os.path.join(a.ret, "r80_identity_vs_old.csv")]
        disp = os.path.join(q, "dispatched_arms.txt")   # R80 (rule 1.16): an undispatched queue is NOT RUN in the gates, not MISSING
        with open(disp, "w") as df:
            for jd in sorted(j for qd in a.queue_dir.split(",") for j in glob.glob(os.path.join(qd, "*/"))):
                df.write(os.path.basename(jd.rstrip("/")) + "\n")
        gcmd += ["--dispatched", disp]
    if RND == "R81":   # R81: the gate reads the manifest directory itself (arm_table.tsv, jobs_*.json, required_outputs.json); grids and reference returns come in through --gate-args
        gcmd = [sys.executable, os.path.join(here, gates_py), "--manifest", a.manifest, "--runs", a.runs, "--out", os.path.join(a.ret, gates_txt), "--csv", os.path.join(a.ret, "r81_identity_vs_old.csv"),
                "--dispatched", os.path.join(q, "dispatched_arms.txt")]
        with open(os.path.join(q, "dispatched_arms.txt"), "w") as df:
            for jd in sorted(j for qd in a.queue_dir.split(",") for j in glob.glob(os.path.join(qd, "*/"))):
                df.write(os.path.basename(jd.rstrip("/")) + "\n")
        if a.gate_args:
            import shlex
            gcmd += shlex.split(a.gate_args)
        gcmd += ["--hold-file", os.path.join(a.ret, "HOLD_FROM_GATES.txt")]   # a FAIL of a DET or IDENT row writes this file (nothing is deleted); the agent places it as the dispatcher's HOLD.txt (r81_hold_check.sh)
    subprocess.call(gcmd)   # FAIL is reported in the file, not fatal here
    arm_ids = [r["arm"] for r in csv.DictReader(open(a.arms), delimiter="\t")]
    engagement_all(a.runs, arm_ids, os.path.join(a.ret, "engagement_all.txt"))
    spec = json.load(open(os.path.join(a.manifest, "required_outputs.json")))
    idx = None
    if a.step0_delivered:
        idx = step0_index(a.ret)
        if idx is None:
            print("--step0-delivered needs STEP0/STEP0_SHA256.txt in --ret (the index sent with the stage-1b reply)")
            return 3
        for k in STEP0_KEEP:
            if not os.path.exists(os.path.join(a.ret, k)):
                print("--step0-delivered: %s is absent from --ret" % k)
                return 3
    rows = check_required(a.ret, a.runs, spec, idx)
    miss = [r for r in rows if r[4] not in ("PRESENT", DELIVERED)]
    lines = ["# %s return checklist (generated from required_outputs.json; presence, columns and row counts only)" % RND, "", "| follow-up | arm | file | where | status | detail |", "|---|---|---|---|---|---|"]
    lines += ["| %s | %s | %s | %s | %s | %s |" % (r[0], r[1], r[2], r[3], r[4] if r[4] == "PRESENT" else "**%s**" % r[4], r[5]) for r in rows]
    lines += ["", "## agent-authored and generated items", "", "| item | status | bytes |", "|---|---|---|"]
    gmiss = 0
    for it in req_agent + req_dirs + gen_items:
        p = os.path.join(a.ret, it)
        ok = os.path.exists(p) and (size(p) > 0 or it == "agent_edits.patch")
        if it in req_dirs or it == "compact":
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
    if RND == "R81":   # R81 (F-496): packing is done by r81_pack_return.py (split -b 28M -d -a 2, a .sha256 per part, reassembly proof, duplicates by pos_trace hash); this script stops here
        print("R81: checklist passed. Now run: python3 tools/r69/r81_pack_return.py --dir %s --required %s --name R81_RETURN_%s" % (a.ret, os.path.join(a.manifest, "required_outputs.json"), a.date))
        return 0
    zp = "%s_RETURN_%s.zip" % (RND, a.date)
    allf = files_of(a.ret, ".")
    base = os.path.dirname(os.path.abspath(a.ret.rstrip("/")))
    before_n = len(allf)
    allf = [f for f in allf if not excluded_from_zip(os.path.relpath(os.path.abspath(f), base))]
    print("return-trim: excluded %d files from the zip (kept on disk under --ret)" % (before_n - len(allf)))
    if a.step0_delivered:
        keep = {os.path.normpath(os.path.join(a.ret, k)) for k in STEP0_KEEP}
        stp = os.path.normpath(os.path.join(a.ret, "STEP0")) + os.sep
        allf = [f for f in allf if not f.startswith(stp) or f in keep]
        print("STEP0 data not packed (delivered in stage 1b); %d files kept" % len(allf))
    nbytes = write_zip(zp, a.ret, allf)
    print("zip %s %.2f MB (%d files)" % (zp, nbytes / 1e6, len(allf)))
    if nbytes <= a.limit_mb * 1e6:
        open("%s_return.sha256" % RND, "w").write("%s  %s  %d\n" % (sha(zp), zp, nbytes))
        print("single file; sha256 in %s_return.sha256" % RND)
        return 0
    parts = byte_split(zp, part=choose_part_size(nbytes))
    with open("%s_return.sha256" % RND, "w") as f:
        f.write("%s  %s  %d  (whole zip)\n" % (sha(zp), zp, nbytes))
        for nm, n, h in parts:
            f.write("%s  %s  %d\n" % (h, os.path.basename(nm), n))
    open("REASSEMBLE.txt", "w").write("cat %s.part* > %s && sha256sum %s   # the hash must equal the first line of %s_return.sha256; unzip only then\n" % (zp, zp, zp, RND))
    ok = reassembly_proof(zp, parts, "reassembly_check.txt")
    os.remove(zp)       # only the parts and the checksum files are delivered
    print("byte-split into %d parts of %d bytes: %s" % (len(parts), PART, ", ".join(os.path.basename(p[0]) for p in parts[:3]) + (" ..." if len(parts) > 3 else "")))
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
