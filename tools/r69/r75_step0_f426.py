#!/usr/bin/env python3
"""R75 Step 0 (F-435): produce the F-426 part 1 / part 2 tables from existing data. Python only; no new run.

    python3 tools/r69/r75_step0_f426.py --r73-archive <dir holding the R73 arm archive dirs> --r74-runs <R74 unbundled runs dir> --pairs r75_manifest/switch_pairs.json --out R75_RETURN --scratch <scratch dir>

Why: R74's part 1 reads shadow_scan_all.csv.gz or shadow_scan.csv, but the R73 archive holds shadow_scan.csv.gz (F-435); part 2 reads pos_trace_all.csv.gz of the two plain controls, but the archive holds
state_trace.csv.gz. For each R73 shadow arm and control named in the pairs file this script (1) copies shadow_scan.csv.gz to <scratch>/<arm>/shadow_scan_all.csv.gz, (2) runs
r74_frame_errors.py --no-errors on the archive dir with --out <scratch>/<arm> to write pos_trace_all.csv.gz (skipped when the scratch dir already has it), then runs r74_switch_reduce.py part1 and part2.
Outputs in --out: r74_f426_part1_frames.csv, r74_f426_part1_summary.csv, r74_f426_part2_frames.csv, r74_f426_part2_summary.csv, and step0_f426_log.txt (what was found, copied, run; a missing input is written, never invented).
Nothing in the archive is modified (scratch only).
"""
import argparse, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def find(root, arm, filename):
    """Return the first candidate dir that actually HOLDS filename (or filename+.gz) -- not just the first
    that exists as a directory: the top-level arm dir always exists (it holds config.yaml etc.), so a plain
    os.path.isdir() check on it short-circuits before ever reaching the archive subdirs (R75 Addendum 1, A1)."""
    for c in (os.path.join(root, arm), os.path.join(root, arm, "archive"), os.path.join(root, arm, "listener_out", "reduce", "archive")):
        if os.path.exists(os.path.join(c, filename)) or os.path.exists(os.path.join(c, filename + ".gz")):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r73-archive", required=True); ap.add_argument("--r74-runs", required=True); ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--scratch", required=True)
    a = ap.parse_args()
    pairs = [p for p in json.load(open(a.pairs)) if p.get("kind") == "switch"]
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.scratch, exist_ok=True)
    log = []
    shadows = sorted({p["shadow_arm_r73"] for p in pairs})
    ctls = sorted({p["control_before"] for p in pairs} | {p["control_after"] for p in pairs})
    for arm in shadows:
        src = find(a.r73_archive, arm, "shadow_scan.csv.gz"); dst = os.path.join(a.scratch, arm); os.makedirs(dst, exist_ok=True)
        f = os.path.join(src, "shadow_scan.csv.gz") if src else None
        if f and os.path.exists(f):
            shutil.copy(f, os.path.join(dst, "shadow_scan_all.csv.gz")); log.append("shadow %s: copied shadow_scan.csv.gz -> scratch shadow_scan_all.csv.gz" % arm)
        else:
            log.append("shadow %s: NO shadow_scan.csv.gz in %s" % (arm, src))
    for arm in ctls:
        src = find(a.r73_archive, arm, "state_trace.csv"); dst = os.path.join(a.scratch, arm); os.makedirs(dst, exist_ok=True)
        if os.path.exists(os.path.join(dst, "pos_trace_all.csv.gz")):
            log.append("control %s: pos_trace_all.csv.gz already in scratch" % arm); continue
        if not src:
            log.append("control %s: archive dir not found under %s" % (arm, a.r73_archive)); continue
        # state_reference.txt lives at the arm's top level, not alongside state_trace.csv.gz in the archive subdir
        # (R75 Addendum 1, A1) -- r74_frame_errors.py reads both from the same --dir, so stage a merge dir with both.
        merge = os.path.join(a.scratch, arm + "_merge"); os.makedirs(merge, exist_ok=True)
        for fn in ("state_trace.csv", "state_trace.csv.gz"):
            if os.path.exists(os.path.join(src, fn)):
                shutil.copy(os.path.join(src, fn), os.path.join(merge, fn))
        ref = os.path.join(a.r73_archive, arm, "state_reference.txt")
        if os.path.exists(ref):
            shutil.copy(ref, os.path.join(merge, "state_reference.txt"))
        rc = subprocess.call([sys.executable, os.path.join(HERE, "r74_frame_errors.py"), "--dir", merge, "--out", dst, "--arm", arm, "--no-errors"])
        log.append("control %s: r74_frame_errors.py --no-errors exit %d, pos_trace_all.csv.gz %s" % (arm, rc, "written" if os.path.exists(os.path.join(dst, "pos_trace_all.csv.gz")) else "ABSENT"))
    r1 = subprocess.call([sys.executable, os.path.join(HERE, "r74_switch_reduce.py"), "part1", "--shadow-dirs", a.scratch, "--pairs", a.pairs, "--out", a.out])
    r2 = subprocess.call([sys.executable, os.path.join(HERE, "r74_switch_reduce.py"), "part2", "--runs", a.r74_runs, "--controls", a.scratch, "--pairs", a.pairs, "--out", a.out])
    log.append("part1 exit %d, part2 exit %d" % (r1, r2))
    open(os.path.join(a.out, "step0_f426_log.txt"), "w").write("\n".join(log) + "\n"); print("\n".join(log))
    return 0 if r1 == 0 and r2 == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
