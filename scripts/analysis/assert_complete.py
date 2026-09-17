#!/usr/bin/env python3
"""CQ-38: the per-row completeness gate, distinct from assert_responses.py.

TQ-30's own DELIVERS-check reported `grep -c ',X,'` returns 0 on both
block-V CSVs -- that claim was about `ate_mm`; the command it names is
about the whole file, where 160 of 160 rows actually contained `,X,`
somewhere (all five update-level columns -- DIV_pos, rho_ref, ask_p50,
got_p50, kappa_eff -- were X on all 96 source=NEW rows). assert_responses.py
could not have caught this: its own is_populated() (see that module)
returns true the moment ANY ONE row has a real value, and the 64 REUSED
rows populated all five columns fine -- so that preflight would have
PASSED on the exact file this bug shipped in. It answers "does this
response exist at all" (a smoke-cell, pre-dispatch question); this module
answers "is every declared column populated on EVERY row of a named
population" (a filled-CSV, pre-filing question) -- a different check, not
a stricter call to the same one.

USAGE
-----
    assert_complete.py --file tq30_V_eee_01_FILLED.csv \
        --require-on source=NEW \
        --allow-x-on source=REUSED:h_rr_min_eig,htth_pos_trace,collapse_engaged,per_residual_engaged,sigma_scale_applied,DIV_pos_htth \
        --columns cell_id,seq,block,...,ate_mm,DIV_pos,rho_ref,ask_p50,got_p50,kappa_eff,...

--columns is the full manifest to check (every column the row schema
declares). --require-on POP=VAL selects the population every one of
--columns must be populated on (no exemptions) -- multiple --require-on
flags OR together if a row could match more than one population; typically
there is exactly one "the new work" population per file. --allow-x-on
POP=VAL:col1,col2,... names a population where the LISTED columns (and
only those) are permitted to read the literal X placeholder -- every
column NOT in that list is still required, even on that population. A row
matching neither --require-on nor any --allow-x-on population is treated
as --require-on (the conservative default: an unrecognized population must
be fully populated, not silently skipped).

A column reads "populated" the same way TQ-4/TQ-12's own convention does:
present, non-empty, not exactly the literal string "X" -- this is stricter
than assert_responses.py's is_populated() (which also excludes -1/nan as
"not applicable this frame" sentinels) because a pre-populated results CSV
uses X as its ONE placeholder-for-not-done marker (rule 45), not -1/nan,
and a real -1 or nan VALUE in a filled row is data, not a hole, and must
not be excused.

Exit 0 if every declared column is populated on every row of every
population it is required on (respecting named X exemptions). Exit 1
otherwise, after printing the full per-column, per-population failure
table (rule 41b: numbers, not "ok").
"""
import argparse
import csv
import sys


def load_rows(path):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = list(reader)
    return header, rows


def parse_population(spec):
    """'source=NEW' -> ('source', 'NEW')."""
    if "=" not in spec:
        raise ValueError(f"population spec {spec!r} must be COLUMN=VALUE")
    col, val = spec.split("=", 1)
    return col.strip(), val.strip()


def parse_allow_x(spec):
    """'source=REUSED:colA,colB' -> (('source','REUSED'), {'colA','colB'})."""
    if ":" not in spec:
        raise ValueError(f"--allow-x-on spec {spec!r} must be POP=VAL:col1,col2,...")
    pop_part, cols_part = spec.split(":", 1)
    pop = parse_population(pop_part)
    cols = {c.strip() for c in cols_part.split(",") if c.strip()}
    return pop, cols


def row_matches(row, population):
    col, val = population
    return row.get(col, "") == val


def is_x(value):
    return value is not None and value.strip() == "X"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="a filled results CSV (header on line 1)")
    ap.add_argument("--columns", required=True,
                     help="comma-separated full column manifest to check (every column the schema declares)")
    ap.add_argument("--require-on", action="append", default=[],
                     help="COLUMN=VALUE selecting a population that must be fully populated, no exemptions. Repeatable.")
    ap.add_argument("--allow-x-on", action="append", default=[],
                     help="COLUMN=VALUE:col1,col2,... naming a population and the specific columns allowed to read X on it. Repeatable.")
    args = ap.parse_args()

    manifest = [c.strip() for c in args.columns.split(",") if c.strip()]
    require_pops = [parse_population(s) for s in args.require_on]
    allow_x_pops = [parse_allow_x(s) for s in args.allow_x_on]

    header, rows = load_rows(args.file)
    missing_cols = [c for c in manifest if c not in header]

    print(f"[assert_complete] {args.file}: {len(rows)} rows, {len(manifest)} manifest columns, "
          f"{len(require_pops)} require-on population(s), {len(allow_x_pops)} allow-x-on population(s)")
    if missing_cols:
        print(f"  MISSING FROM HEADER (not even a column): {missing_cols}")

    ok = not missing_cols
    # failures[population_label][column] = count of rows in that population with X (and not exempted there)
    failures = {}

    for i, row in enumerate(rows):
        # Determine this row's exemption set: union of every allow-x-on
        # population it matches; if it matches none and no require-on
        # population either, it defaults to the conservative "fully
        # required" population labeled "(unclassified)".
        exempt_cols = set()
        matched_allow = False
        for pop, cols in allow_x_pops:
            if row_matches(row, pop):
                exempt_cols |= cols
                matched_allow = True

        matched_require = any(row_matches(row, pop) for pop in require_pops)

        if not matched_allow and not matched_require and (require_pops or allow_x_pops):
            label = "(unclassified -- treated as fully required)"
        elif matched_require:
            label = f"require-on:{'|'.join(f'{c}={v}' for c, v in require_pops if row_matches(row, (c, v)))}"
        else:
            label = f"allow-x-on:{'|'.join(f'{c}={v}' for c, v in [p for p, _ in allow_x_pops] if row_matches(row, (c, v)))}"

        for col in manifest:
            if col not in header:
                continue
            v = row.get(col)
            if is_x(v) and col not in exempt_cols:
                failures.setdefault(label, {}).setdefault(col, 0)
                failures[label][col] += 1

    if failures:
        ok = False
        print("  FAILURES (population -> column -> row count with unexempted X):")
        for label in sorted(failures):
            print(f"    {label}:")
            for col in sorted(failures[label]):
                print(f"      {col}: {failures[label][col]} rows")
    else:
        print("  every declared column is populated on every row of every named population.")

    print(f"[assert_complete] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
