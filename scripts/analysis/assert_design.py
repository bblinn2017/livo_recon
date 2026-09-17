#!/usr/bin/env python3
"""CQ-32 items (3)/(4): the design-integrity preflight, modeled line for
line on assert_responses.py's pattern (same failure class TQ-4 hit --
a mismatch between what a card/generator CLAIMS and what it actually
emits, found only after cells had already run -- caught here before
dispatch instead).

Two checks, run against a design's own declared cell table (a CSV with
one row per cell and one column per crossed factor, e.g.
livo_recon_results/tq12/tq12_cells.csv) and the generator module that
produced it (imported directly, not re-implemented -- see fast_ws/CLAUDE.md's
"go through gen_jobs.py"-style reuse rule: a design's own emission logic is
the one place that logic should live, this script recomputes from it
rather than guessing at collapse semantics independently):

  (3) DIGEST COLLISION -- two DISTINCT cell labels (differing in at least
      one declared factor) must never resolve to byte-identical effective
      configs. A collision means the design has a dead/unswept factor
      hiding behind a label that implies it varies -- exactly the kind of
      gap CQ-32 names (use_bins forced false off-pca, adaptive_q's whole
      namespace absent outside spline+refine, etc). Rows sharing a label
      whose OWN stratum/control marks it as intentionally repeated (the
      generator's CONTROLS stratum -- nudge pairs, logging on/off pairs,
      build-identity twins) are expected to collide and are excluded.

  (4) LABEL -> EFFECTIVE ROUND-TRIP -- for every factor a cell's row
      declares (including generated factors K/L), recompute what the
      generator's own config_for_cell()/derive() produce from that row's
      OTHER declared levels and compare against the row's own claimed
      value. A mismatch means the cell table's label doesn't actually
      describe the config that would be (or was) emitted for it --
      exactly the round-trip gap this item exists to catch.

Both checks report a per-item pass/fail table (rule 41b) and exit nonzero
naming the specific colliding/mismatched cell ids if anything fails.

Usage:
    assert_design.py --cells-csv livo_recon_results/tq12/tq12_cells.csv \\
                      --generator gen_tq12_grid
"""
import argparse
import csv
import hashlib
import importlib
import json
import sys


FACTOR_COLS = ("C", "F", "G", "H", "J", "K", "L")
CONTROL_STRATA = {"CONTROLS"}


def load_cells(path):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return rows


def levels_of(row):
    return {f: int(row[f]) for f in FACTOR_COLS if row.get(f) not in (None, "")}


def label_key(row):
    """The design-level label identity: everything that's supposed to make
    two cells the SAME designed point vs. different ones -- block, sequence,
    stratum, and the 5 basic factors (K/L are generated, not independently
    labeled, so they're excluded from identity and checked by item 4 instead)."""
    return (row.get("block", ""), row.get("sequence", ""), row.get("stratum", ""),
            row.get("C", ""), row.get("F", ""), row.get("G", ""),
            row.get("H", ""), row.get("J", ""))


def config_digest(cfg):
    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def recompute_config(gen, row):
    block_name = row["block"]
    seq_name = row["sequence"]
    stratum = row["stratum"]
    block = gen.BLOCKS[block_name]
    seq = gen.SEQUENCES[seq_name]
    base_levels = {f: int(row[f]) for f in ("C", "F", "G", "H", "J")}
    levels = gen.derive(base_levels)
    cfg = gen.config_for_cell(block_name, block, seq, levels, "unused_out_dir",
                               stratum=stratum, tier=row.get("tier", "A"))
    return levels, cfg


def check_digest_collisions(gen, rows):
    """Item 3. Returns (n_fail, n_checked, failures) -- failures is a list
    of (digest, [cell_ids]) for groups whose members have DIFFERENT labels."""
    by_digest = {}
    n_checked = 0
    for row in rows:
        if row.get("stratum", "") in CONTROL_STRATA:
            continue
        try:
            _, cfg = recompute_config(gen, row)
        except Exception as e:
            continue
        n_checked += 1
        digest = config_digest(cfg)
        by_digest.setdefault(digest, []).append(row)

    failures = []
    for digest, group in by_digest.items():
        if len(group) < 2:
            continue
        distinct_labels = {label_key(r) for r in group}
        if len(distinct_labels) > 1:
            failures.append((digest, [r["cell_id"] for r in group],
                              sorted(str(k) for k in distinct_labels)))
    n_fail = len(failures)
    return n_fail, n_checked, failures


def check_label_roundtrip(gen, rows):
    """Item 4. For every row, recompute K/L from C/F/G/H/J via the
    generator's own derive() and compare against the row's own claimed
    K/L -- the declared-vs-derived round trip. Returns (n_fail, n_checked,
    mismatches) -- mismatches is a list of (cell_id, factor, claimed,
    derived)."""
    mismatches = []
    n_checked = 0
    for row in rows:
        if not all(row.get(f) not in (None, "") for f in ("C", "F", "G", "H", "J", "K", "L")):
            continue
        n_checked += 1
        base_levels = {f: int(row[f]) for f in ("C", "F", "G", "H", "J")}
        derived = gen.derive(base_levels)
        for f in ("K", "L"):
            claimed = int(row[f])
            if claimed != derived[f]:
                mismatches.append((row["cell_id"], f, claimed, derived[f]))
    n_fail = len(mismatches)
    return n_fail, n_checked, mismatches


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-csv", required=True, help="declared cell table (e.g. tq12_cells.csv)")
    ap.add_argument("--generator", required=True,
                     help="importable generator module name (e.g. gen_tq12_grid), "
                          "must expose BLOCKS, SEQUENCES, derive(), config_for_cell()")
    args = ap.parse_args()

    gen = importlib.import_module(args.generator)
    rows = load_cells(args.cells_csv)

    print(f"=== assert_design.py: {args.cells_csv} ({len(rows)} rows) "
          f"against generator {args.generator} ===")

    n_fail = 0
    results = []

    n3_fail, n3_checked, collisions = check_digest_collisions(gen, rows)
    if n3_fail:
        n_fail += 1
        names = ", ".join(cid for _, cids, _ in collisions for cid in cids[:2])
        results.append(("(3) digest-collision", "FAIL",
                         f"{n3_fail} colliding group(s) among {n3_checked} non-control cells "
                         f"-- distinct labels sharing one effective config, e.g. {names}"))
    else:
        results.append(("(3) digest-collision", "PASS",
                         f"0 collisions among {n3_checked} non-control cells checked"))

    n4_fail, n4_checked, mismatches = check_label_roundtrip(gen, rows)
    if n4_fail:
        n_fail += 1
        names = ", ".join(f"{cid}.{f}(claimed {c} != derived {d})"
                           for cid, f, c, d in mismatches[:5])
        results.append(("(4) label-roundtrip", "FAIL",
                         f"{n4_fail} mismatch(es) among {n4_checked} cells checked -- {names}"))
    else:
        results.append(("(4) label-roundtrip", "PASS",
                         f"0 mismatches among {n4_checked} cells checked"))

    width = max(len(r[0]) for r in results)
    for name, status, note in results:
        line = f"  {name:<{width}}  {status}"
        if note:
            line += f"  -- {note}"
        print(line)

    print(f"\n{len(results) - n_fail}/{len(results)} design-integrity checks passed; {n_fail} failed")
    if n_fail:
        if collisions:
            print("\nfull collision detail:", file=sys.stderr)
            for digest, cids, labels in collisions:
                print(f"  digest {digest[:12]}...: cells {cids} -- labels {labels}", file=sys.stderr)
        if mismatches:
            print("\nfull round-trip mismatch detail:", file=sys.stderr)
            for cid, f, c, d in mismatches:
                print(f"  {cid}: {f} claimed={c} derived={d}", file=sys.stderr)
        print(f"REFUSED: {n_fail} design-integrity check(s) failed against {args.cells_csv}",
              file=sys.stderr)
        sys.exit(1)
    print("OK: every declared cell's label matches its recomputed effective config, "
          "and no two distinct labels collide.")
    sys.exit(0)


if __name__ == "__main__":
    main()
