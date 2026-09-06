#!/usr/bin/env python3
"""CQ-18 item (1): the response-schema preflight.

TQ-4 declared sum_S/floor_share/sdiag_share/pvar_share/prior_pose_share/nis/
nis_est as responses; the writer emitted none of them; 1,988 cells ran
anyway and the gap was found at analysis time, after score-then-delete. The
card and the writer were never compared before dispatch.

This script IS that comparison. It takes a sweep's declared response list
and a freshly-generated data file from a single smoke cell, and asserts
every named response is a PRESENT and NON-NULL column -- printing a
per-column pass/fail table (rule 41b: a check must report numbers, not
"ok"), and exiting nonzero if anything is missing. A sweep generator calls
this and refuses to emit jobs if it fails (see tq12_skeleton.py's own
preflight call, or wire it into whatever generator dispatches next).

Usage:
    assert_responses.py --file frame_stats.txt --responses sum_S,floor_share,...
    assert_responses.py --file frame_stats.txt --responses-file responses.txt

The data file is assumed CSV-with-header (frame_stats.txt's own format:
comma-separated, header on line 1). "Non-null" means: the column exists,
and at least one row has a value that isn't empty, "nan", or "-1" (frame_stats.txt's
own convention for "not applicable this frame" on several diagnostic
fields -- see LioFrameDiag's doc comments -- so a column that is
UNIFORMLY -1/nan across every row is flagged the same as a genuinely
missing column: the response was declared but never actually populated,
which is the exact TQ-4 failure mode this script exists to catch).
"""
import argparse
import csv
import sys


def load_columns(path):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = list(reader)
    return header, rows


def is_populated(rows, col):
    """True iff at least one row has a real (non-empty, non-nan, non -1) value."""
    for row in rows:
        v = row.get(col)
        if v is None:
            continue
        v = v.strip()
        if v == "" or v.lower() == "nan":
            continue
        try:
            if float(v) == -1.0:
                continue
        except ValueError:
            pass
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="freshly-generated data file (CSV with header)")
    ap.add_argument("--responses", default="", help="comma-separated list of declared response column names")
    ap.add_argument("--responses-file", default="", help="path to a file with one response name per line")
    args = ap.parse_args()

    declared = []
    if args.responses:
        declared.extend(x.strip() for x in args.responses.split(",") if x.strip())
    if args.responses_file:
        with open(args.responses_file) as fh:
            declared.extend(line.strip() for line in fh if line.strip() and not line.startswith("#"))
    if not declared:
        print("ERROR: no declared responses given (--responses / --responses-file)", file=sys.stderr)
        sys.exit(2)

    header, rows = load_columns(args.file)
    header_set = set(header)

    print(f"=== assert_responses.py: {args.file} ({len(rows)} rows, {len(header)} columns) ===")
    n_fail = 0
    results = []
    for resp in declared:
        if resp not in header_set:
            results.append((resp, "MISSING", "column not present in header at all"))
            n_fail += 1
            continue
        if not is_populated(rows, resp):
            results.append((resp, "NULL", "column present but every row is empty/nan/-1"))
            n_fail += 1
            continue
        results.append((resp, "PASS", ""))

    width = max(len(r[0]) for r in results)
    for name, status, note in results:
        line = f"  {name:<{width}}  {status}"
        if note:
            line += f"  -- {note}"
        print(line)

    print(f"\n{len(declared) - n_fail}/{len(declared)} declared responses present and non-null; "
          f"{n_fail} missing/null")
    if n_fail:
        missing_names = [name for name, status, _ in results if status != "PASS"]
        print(f"REFUSED: {n_fail} response(s) not satisfied by {args.file}: {', '.join(missing_names)}",
              file=sys.stderr)
        sys.exit(1)
    print("OK: every declared response is a present, non-null column.")
    sys.exit(0)


if __name__ == "__main__":
    main()
