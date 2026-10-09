#!/usr/bin/env python3
"""R85: NOT NEEDED. Nothing in R85 calls this tool. The R84 version wrote the sample-window and time-based process-noise engagement tables; the sample-window key was removed from the build in R85
(patch 1, F-520; a job that sets it is a refuse arm) and the time-term comparison group is gone, so neither table exists in R85 and neither appears in r85_manifest/required_outputs.json.
build_return.py does not call it. It is kept only as a small helper that lists what an engagement_all.txt (build_return.py output, blocks `### <arm>` and `--- <file>`) contains.
Planning agent's script; selftest on synthetic text only (planner sandbox, not evidence). Nothing is interpreted.

    python3 tools/r69/r85_engagement_tables.py --engagement-all <engagement_all.txt> [--out <csv>]
    python3 tools/r69/r85_engagement_tables.py --selftest

Writes (to --out, or prints when --out is absent) one row per arm block: arm, files (the `--- <file>` names in the block, joined by space), lines (number of text lines of the block), no counter is parsed.
"""
import argparse, csv, os, shutil, sys, tempfile

COLS = ["arm", "files", "lines"]


def all_blocks(path):
    """engagement_all.txt -> {arm: {file name: text}} (blocks `### <arm>` and `--- <file>`)."""
    out, arm, fn = {}, None, None
    for ln in open(path, errors="replace").read().splitlines():
        if ln.startswith("### "):
            arm, fn = ln[4:].strip(), None
            out.setdefault(arm, {})
        elif ln.startswith("--- ") and arm is not None:
            fn = ln[4:].strip()
            out[arm].setdefault(fn, "")
        elif arm is not None and fn is not None:
            out[arm][fn] += ln + "\n"
    return out


def rows_of(blocks):
    return [[a, " ".join(fs), sum(t.count("\n") for t in fs.values())] for a, fs in blocks.items()]


def run(a):
    rows = rows_of(all_blocks(a.engagement_all))
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLS)
            w.writerows(rows)
    else:
        for r in rows:
            print(*r)
    print("%d arm blocks" % len(rows))
    return 0


def selftest():
    t = tempfile.mkdtemp()
    fails, n = [], [0]

    def ck(c, m):
        n[0] += 1
        if not c:
            fails.append(m)
    try:
        p = os.path.join(t, "e.txt")
        open(p, "w").write("### a\n--- engagement_imu.txt\nx\ny\n\n--- engagement.txt\nz\n\n### b\n\n### c\n--- engagement_imu.txt\n\n")
        b = all_blocks(p)
        ck(list(b) == ["a", "b", "c"], "arms in file order: %r" % list(b))
        ck(list(b["a"]) == ["engagement_imu.txt", "engagement.txt"] and b["a"]["engagement_imu.txt"] == "x\ny\n\n" and b["b"] == {}, "files and text of the blocks: %r" % b)
        r = rows_of(b)
        ck(r[0] == ["a", "engagement_imu.txt engagement.txt", 5] and r[1] == ["b", "", 0] and r[2][1] == "engagement_imu.txt", "rows %r" % r)
        o = os.path.join(t, "o.csv")
        ck(run(argparse.Namespace(engagement_all=p, out=o)) == 0 and open(o).read().splitlines()[0] == ",".join(COLS) and len(open(o).read().splitlines()) == 4, "csv written with header")
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("SELFTEST r85_engagement_tables", ("FAIL " + "; ".join(fails) + " | ") if fails else "PASS", "checks: %d, failures: %d" % (n[0], len(fails)),
          "" if fails else "(minimal helper: engagement_all.txt blocks listed; no R85 table is produced by this tool)")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engagement-all")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.engagement_all:
        ap.error("--engagement-all is required (or --selftest)")
    sys.exit(run(a))
