#!/usr/bin/env python3
"""R76 Step 0 (F-435 / F-446): produce the F-426 part 1 / part 2 tables from existing data. Python only; no new run.

    python3 tools/r69/r76_step0_f426.py --r73-archive <dir holding the R73 arm archive dirs> --r74-runs <R74 unbundled runs dir> --pairs r76_manifest/switch_pairs.json --out R76_RETURN --scratch <scratch dir>

Why: R74's part 1 reads shadow_scan_all.csv.gz or shadow_scan.csv, but the R73 archive holds shadow_scan.csv.gz (F-435); part 2 reads pos_trace_all.csv.gz of the two plain controls, but the archive holds
state_trace.csv.gz. For each R73 shadow arm and control named in the pairs file this script (1) copies shadow_scan.csv.gz to <scratch>/<arm>/shadow_scan_all.csv.gz, (2) runs
r74_frame_errors.py --no-errors on the archive dir with --out <scratch>/<arm> to write pos_trace_all.csv.gz (skipped when the scratch dir already has it), then runs r74_switch_reduce.py part1 and part2.
R76 (F-446): the R73 archive records neither t_rel_s nor bag_start_abs, so pos_trace_all.csv.gz carries t_rel_s = NaN there; part 2 keys on scan_id only.
Outputs in --out: r74_f426_part1_frames.csv, r74_f426_part1_summary.csv, r74_f426_part2_frames.csv, r74_f426_part2_summary.csv, and step0_f426_log.txt (what was found, copied, run; a missing input is written, never invented).
Nothing in the archive is modified (scratch only).
"""
import argparse, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


LAYOUTS = ("", "archive", os.path.join("listener_out", "reduce", "archive"))   # R76 (addendum A1): the R73 archive is <root>/<arm>/listener_out/reduce/archive


def find(root, arm):
    """The directory that holds the arm's archive files. R75 looked only at <root>/<arm> and <root>/<arm>/archive, found the four small top-level files and no data (A1)."""
    for lay in LAYOUTS[::-1]:
        c = os.path.join(root, arm, lay) if lay else os.path.join(root, arm)
        if os.path.isdir(c) and any(f.endswith((".csv.gz", ".csv")) for f in os.listdir(c)):
            return c
    return None


def selftest():
    import tempfile
    bad = []
    with tempfile.TemporaryDirectory() as t:
        for arm, lay in (("nested", LAYOUTS[2]), ("flat", ""), ("arch", "archive")):
            d = os.path.join(t, arm, lay) if lay else os.path.join(t, arm); os.makedirs(d)
            open(os.path.join(d, "shadow_scan.csv.gz"), "w").write("x")
            if lay:
                open(os.path.join(t, arm, "config.yaml"), "w").write("x")   # the small top-level files of the real layout (no data)
        for arm, lay in (("nested", LAYOUTS[2]), ("flat", ""), ("arch", "archive")):
            want = os.path.join(t, arm, lay) if lay else os.path.join(t, arm)
            if find(t, arm) != want:
                bad.append("find(%s) = %s, want %s" % (arm, find(t, arm), want))
        if find(t, "absent") is not None:
            bad.append("find(absent) is not None")
        os.makedirs(os.path.join(t, "onlytop")); open(os.path.join(t, "onlytop", "config.yaml"), "w").write("x")
        if find(t, "onlytop") is not None:
            bad.append("find() accepted a top-level dir with no data files")
    print("SELFTEST r76_step0_f426 " + ("FAIL: " + "; ".join(bad) if bad else "PASS (nested R73 layout, flat, archive/, absent and data-less top level)"))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r73-archive"); ap.add_argument("--r74-runs"); ap.add_argument("--pairs")
    ap.add_argument("--out"); ap.add_argument("--scratch")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not all((a.r73_archive, a.r74_runs, a.pairs, a.out, a.scratch)):
        ap.error("--r73-archive, --r74-runs, --pairs, --out and --scratch are required (or --selftest)")
    pairs = [p for p in json.load(open(a.pairs)) if p.get("kind") == "switch"]
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.scratch, exist_ok=True)
    log = []
    shadows = sorted({p["shadow_arm_r73"] for p in pairs})
    ctls = sorted({p["control_before"] for p in pairs} | {p["control_after"] for p in pairs})
    for arm in shadows:
        src = find(a.r73_archive, arm); dst = os.path.join(a.scratch, arm); os.makedirs(dst, exist_ok=True)
        f = os.path.join(src, "shadow_scan.csv.gz") if src else None
        if f and os.path.exists(f):
            shutil.copy(f, os.path.join(dst, "shadow_scan_all.csv.gz")); log.append("shadow %s: copied shadow_scan.csv.gz -> scratch shadow_scan_all.csv.gz" % arm)
        else:
            log.append("shadow %s: NO shadow_scan.csv.gz in %s" % (arm, src))
    for arm in ctls:
        src = find(a.r73_archive, arm); dst = os.path.join(a.scratch, arm); os.makedirs(dst, exist_ok=True)
        if os.path.exists(os.path.join(dst, "pos_trace_all.csv.gz")):
            log.append("control %s: pos_trace_all.csv.gz already in scratch" % arm); continue
        if not src:
            log.append("control %s: archive dir not found under %s" % (arm, a.r73_archive)); continue
        rc = subprocess.call([sys.executable, os.path.join(HERE, "r74_frame_errors.py"), "--dir", src, "--out", dst, "--arm", arm, "--no-errors"])
        log.append("control %s: r74_frame_errors.py --no-errors exit %d, pos_trace_all.csv.gz %s" % (arm, rc, "written" if os.path.exists(os.path.join(dst, "pos_trace_all.csv.gz")) else "ABSENT"))
    r1 = subprocess.call([sys.executable, os.path.join(HERE, "r74_switch_reduce.py"), "part1", "--shadow-dirs", a.scratch, "--pairs", a.pairs, "--out", a.out])
    r2 = subprocess.call([sys.executable, os.path.join(HERE, "r74_switch_reduce.py"), "part2", "--runs", a.r74_runs, "--controls", a.scratch, "--pairs", a.pairs, "--out", a.out])
    log.append("part1 exit %d, part2 exit %d" % (r1, r2))
    open(os.path.join(a.out, "step0_f426_log.txt"), "w").write("\n".join(log) + "\n"); print("\n".join(log))
    return 0 if r1 == 0 and r2 == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
