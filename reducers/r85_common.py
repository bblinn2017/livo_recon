#!/usr/bin/env python3
"""r85_common: helpers shared by the R85 host-side reducers (r85_l0_reduce, r85_sht_reduce). Python 3 standard library only; reads returned CSV/TSV files (also .csv.gz) and
writes CSVs; nothing here feeds a run. Copied from the R84 helpers (r84_common), not imported from them. `python3 -I r85_common.py --selftest` runs its own checks.

Conventions kept by every R85 reducer
  * native units: every ATE and error is a length in mm written with three decimals; a difference is arm minus baseline in mm; no ratios, no noise bars.
  * collapse = gt_max_mm > 500 (COLLAPSE_GT_MM); an arm whose gt_max_mm is empty has no collapse value (blank), never a guess.
  * the baseline sits in the same row as the arm it is compared with.
  * missing arms/rows are named in a `*_missing.txt` next to the output and the process exits 1; the tables are still written; nothing is invented.
  * phase_summary.csv: if an arm appears more than once the first row wins.
"""
import csv
import gzip
import math
import os
import sys

COLLAPSE_GT_MM = 500.0
QS = ["fq", "mq"]


def num(x):
    """float or None (empty, 'nan', non-numeric -> None)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def f3(x):
    """three-decimal string, '' for None."""
    return "" if x is None else "%.3f" % x


def collapsed_of(gt_max):
    return None if gt_max is None else gt_max > COLLAPSE_GT_MM


def bflag(b):
    return "" if b is None else ("True" if b else "False")


def median(v):
    v = sorted(v)
    n = len(v)
    if n == 0:
        return None
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def q_of(arm):
    """q from the arm-name suffix: fq or mq."""
    s = arm.rsplit("_", 1)[-1]
    return s if s in QS else ""


def read_rows(path, delim=","):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="") as f:
        return list(csv.DictReader(f, delimiter=delim))


def edge_s(e):
    """voxel edge as the shortest text: 0.2, 0.25, 0.3."""
    return "%g" % float(e)


def load_phase(path):
    """{arm: {'gt_ate_mm': float|None, 'gt_max_mm': float|None}} from phase_summary.csv (first row of an arm wins)."""
    d = {}
    for r in read_rows(path):
        a = r.get("arm", "")
        if a and a not in d:
            d[a] = {"gt_ate_mm": num(r.get("gt_ate_mm")), "gt_max_mm": num(r.get("gt_max_mm"))}
    return d


def load_arm_table(manifest, group):
    """rows of arm_table.tsv whose `group` is `group` (a string) or in `group` (a collection)."""
    gs = {group} if isinstance(group, str) else set(group)
    return [r for r in read_rows(os.path.join(manifest, "arm_table.tsv"), "\t") if r.get("group") in gs]


def load_refs(manifest):
    """{arm: {'gt_ate_mm','gt_max_mm'}} from l2_reference_values.csv (the existing reference arms)."""
    d = {}
    for r in read_rows(os.path.join(manifest, "l2_reference_values.csv")):
        if r["arm"] not in d:
            d[r["arm"]] = {"gt_ate_mm": num(r.get("gt_ate_mm")), "gt_max_mm": num(r.get("gt_max_mm"))}
    return d


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_missing(path, lines):
    """write the missing list (or remove a stale one); return exit code 1 when anything is missing."""
    if lines:
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("MISSING (%d), listed in %s" % (len(lines), path), file=sys.stderr)
        return 1
    if os.path.exists(path):
        os.remove(path)
    return 0


class Checker:
    def __init__(self):
        self.fails = 0
        self.n = 0

    def __call__(self, name, ok, msg=""):
        self.n += 1
        print("%-60s %s  %s" % (name, "pass" if ok else "FAIL", msg))
        self.fails += (not ok)

    def done(self):
        print("%d checks, failures: %d" % (self.n, self.fails))
        return 1 if self.fails else 0


def selftest():
    c = Checker()
    c("num empty/nan/text", num("") is None and num("nan") is None and num("x") is None and num("1.5") == 1.5)
    c("f3", f3(1.23456) == "1.235" and f3(None) == "")
    c("median odd/even/empty", median([3, 1, 2]) == 2 and median([1, 2, 3, 4]) == 2.5 and median([]) is None)
    c("collapse threshold is strict >500", collapsed_of(500.0) is False and collapsed_of(500.1) is True and collapsed_of(None) is None)
    c("q_of", q_of("r85_rt_l0_eee_01_pca_off_v0p2_fq") == "fq" and q_of("x_mq") == "mq")
    c("edge_s", edge_s("0.200") == "0.2" and edge_s(0.25) == "0.25")
    return c.done()


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else 0)
