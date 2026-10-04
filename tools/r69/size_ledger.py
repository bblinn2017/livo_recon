#!/usr/bin/env python3
"""TEMPLATE (planning agent). File-size ledger: measure what a round wrote and compare with the ledger of expected sizes.

    python3 size_ledger.py measure <dir> --out sizes.tsv                 # path, bytes, is_gz   (agent runs this on STEP0/, a return dir, an arm's out dir)
    python3 size_ledger.py check   <dir> --ledger file_size_ledger.json  # every file must match a ledger pattern and be within its per-file and per-pattern budget
Ledger JSON: [{"pattern": "r73_f417_residuals.csv.gz", "max_bytes_per_file": 2e8, "max_total_bytes": 2e8, "basis": "measured R73 STEP0 145 MB" | "estimate", "note": ""}, ...]
`pattern` is an fnmatch pattern on the path relative to <dir> (basename also tried). A file matching NO pattern is UNLEDGERED: a new file or a changed log whose size nobody has
budgeted (rule 5.15: ask the agent to report its size before the round is relied on). Exit 1 on any UNLEDGERED or OVER row."""
import fnmatch, json, os, sys


def walk(d):
    for r, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(r, f)
            yield os.path.relpath(p, d), os.path.getsize(p)


def measure(d, out):
    tot = 0
    with open(out, "w") as o:
        o.write("path\tbytes\tgz\n")
        rows = sorted(walk(d), key=lambda x: -x[1])
        for p, b in rows:
            o.write("%s\t%d\t%s\n" % (p, b, "yes" if p.endswith(".gz") else "no")); tot += b
    print("%d files, %d bytes (%.1f MB) -> %s" % (len(rows), tot, tot / 1e6, out))
    return 0


def check(d, ledger):
    led = json.load(open(ledger))
    per = {i: [] for i in range(len(led))}
    bad = 0
    for p, b in sorted(walk(d)):
        hit = None
        for i, e in enumerate(led):
            if fnmatch.fnmatch(p, e["pattern"]) or fnmatch.fnmatch(os.path.basename(p), e["pattern"]):
                hit = i; break
        if hit is None:
            print("UNLEDGERED  %s  %d bytes" % (p, b)); bad += 1; continue
        per[hit].append(b)
        mx = led[hit].get("max_bytes_per_file")
        if mx is not None and b > mx:
            print("OVER  %s  %d > %d (pattern %s)" % (p, b, mx, led[hit]["pattern"])); bad += 1
    for i, e in enumerate(led):
        t = sum(per[i]); mt = e.get("max_total_bytes")
        if mt is not None and t > mt:
            print("OVER-TOTAL  pattern %s  %d > %d" % (e["pattern"], t, mt)); bad += 1
    print("check: %s (%d problems)" % ("PASS" if not bad else "FAIL", bad))
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "measure" and sys.argv[3] == "--out":
        sys.exit(measure(sys.argv[2], sys.argv[4]))
    if len(sys.argv) >= 5 and sys.argv[1] == "check" and sys.argv[3] == "--ledger":
        sys.exit(check(sys.argv[2], sys.argv[4]))
    print(__doc__); sys.exit(2)
