#!/usr/bin/env python3
"""r83_common: helpers shared by the R83 host-side reducers (r83_mxl_reduce, r83_vsw_reduce, r83_mqs_reduce, r83_pcl_reduce, r83_pcw_collect).
Python only, reads archived run folders and CSVs; nothing here feeds a run. Not a CLI (python3 r83_common.py --selftest runs its own checks).

Conventions kept by every reducer that uses it
  * native units: every ATE and error is a length in mm, written with three decimals; a difference is arm minus baseline, in mm; no ratios.
  * the baseline sits in the same row as the arm it is compared with, together with the number of scans (n) of each.
  * no pooling over bags (no mean over bags, no pooled median); no run is filtered out for diverging: gt_max_mm > 500 is shown in its own column
    `diverged_gt_max_gt_500` and the run stays in every table beside its control.
  * COLLAPSE FLAG (project rule 4.15 c, as r80_sweep_zones.py / r79_load.departure): collapse = 'yes' if gt_max_mm > 500 AND the estimated position departs
    by >= 500 mm from the healthy reference arm over scans onset .. onset+215 (start offset removed, alignment free); 'no' if gt_max_mm <= 500 (the rule is an
    AND, the departure is not needed); 'undetermined' if gt_max_mm > 500 and the trace of the arm or of its reference is not available (never read as healthy).
    The reference arm of each family is printed in the row (column collapse_ref) and is the R82 arm of the same bag and Q at 0.425 m, lines off.
  * missing arms/files are named in a `*_missing.txt` next to the output and the process exits 1; nothing is skipped silently.

Where a run's files are looked for: `--runs DIR [DIR ...]` (every dir is tried in order): DIR/<arm>/<file>, DIR/compact/<arm>/<file>, each also with `.gz` added or
removed. phase_summary.csv columns used: arm, gt_ate_mm, gt_max_mm, n_scans, gt_join_scans, wall_s, onset_scan_used (the R82A phase_summary.csv has NO bag, q or
overlap column: bag, Q and overlap mode are parsed from the arm name).
"""
import gzip
import json
import os
import re
import sys

import numpy as np
import pandas as pd

BAGS = ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]
QS = ["fq", "mq"]
ONSETS = {"eee_01": 485, "eee_02": 481, "eee_03": 310, "nya_01": 583, "nya_02": 539, "nya_03": 343, "sbs_01": 415, "sbs_02": 290, "sbs_03": 340}   # R82 onset_scans.json
COLLAPSE_DEP_MM = 500.0
COLLAPSE_GT_MM = 500.0
DEP_WINDOW = 215


def f3(x):
    """three decimals, NaN stays NaN (written as empty by to_csv)."""
    try:
        return np.nan if x is None or not np.isfinite(float(x)) else round(float(x), 3)
    except (TypeError, ValueError):
        return np.nan


def load_onsets(path=None):
    if path:
        d = json.load(open(path))
        return {k: int(v) for k, v in d.get("onset_scan_used", d).items()}
    return dict(ONSETS)


def read_csv_any(path, **kw):
    return pd.read_csv(path, **kw)   # pandas reads .gz by extension


def find_file(roots, arm, name):
    """first existing path of <root>/<arm>/<name>, <root>/compact/<arm>/<name>, with '.gz' toggled. None if absent."""
    alts = [name, name[:-3] if name.endswith(".gz") else name + ".gz"]
    for r in roots:
        for sub in (os.path.join(r, arm), os.path.join(r, "compact", arm)):
            for n in alts:
                p = os.path.join(sub, n)
                if os.path.isfile(p):
                    return p
    return None


class Phase:
    """phase_summary.csv rows by arm; several files may be given, the FIRST file that holds an arm wins (R83 return first, then the R82 return for controls)."""

    def __init__(self, paths):
        self.rows = {}
        self.source = {}
        self.dup = []
        for p in paths:
            d = pd.read_csv(p, low_memory=False)
            if "arm" not in d.columns:
                raise SystemExit("%s has no arm column" % p)
            for _, r in d.iterrows():
                a = r["arm"]
                if a in self.rows:
                    self.dup.append((a, p))
                    continue
                self.rows[a] = r
                self.source[a] = p

    def has(self, arm):
        return arm in self.rows

    def get(self, arm, col, default=np.nan):
        r = self.rows.get(arm)
        if r is None or col not in r.index:
            return default
        v = r[col]
        return default if pd.isna(v) else v


def departure_mm(arm_trace, ref_trace, onset, window=DEP_WINDOW):
    """r79_load.departure: over the common scans in [onset, onset+window], max norm of (arm pos - arm pos at the first common scan) - (ref pos - ref pos at the
    first common scan), in mm. Traces are pos_trace_all.csv(.gz) frames with scan_id, px, py, pz in metres. NaN when fewer than 2 common scans."""
    a = arm_trace.drop_duplicates("scan_id").set_index("scan_id")[["px", "py", "pz"]]
    b = ref_trace.drop_duplicates("scan_id").set_index("scan_id")[["px", "py", "pz"]]
    common = a.index.intersection(b.index)
    common = common[(common >= onset) & (common <= onset + window)].sort_values()
    if len(common) < 2:
        return np.nan
    pa = a.loc[common].to_numpy(float)
    pb = b.loc[common].to_numpy(float)
    d = (pa - pa[0]) - (pb - pb[0])
    return 1000.0 * float(np.max(np.linalg.norm(d, axis=1)))


def collapse_flag(gt_max_mm, dep_mm):
    if not np.isfinite(gt_max_mm):
        return "undetermined"
    if gt_max_mm <= COLLAPSE_GT_MM:
        return "no"
    if not np.isfinite(dep_mm):
        return "undetermined"
    return "yes" if dep_mm >= COLLAPSE_DEP_MM else "no"


class Ctx:
    """Everything a reducer needs: phase table, trace roots, onsets, a missing-input log, a trace cache."""

    def __init__(self, phase_paths, run_roots, onsets_path=None):
        self.phase = Phase(phase_paths)
        self.roots = list(run_roots or [])
        self.onsets = load_onsets(onsets_path)
        self.missing = []
        self._tr = {}

    def miss(self, what):
        if what not in self.missing:
            self.missing.append(what)

    def trace(self, arm):
        if arm not in self._tr:
            p = find_file(self.roots, arm, "pos_trace_all.csv.gz")
            self._tr[arm] = read_csv_any(p, usecols=["scan_id", "px", "py", "pz"]) if p else None
        return self._tr[arm]

    def collapse(self, arm, ref_arm, bag):
        """(flag, departure_mm) of arm against ref_arm for the bag's onset."""
        gm = self.phase.get(arm, "gt_max_mm")
        if not np.isfinite(gm) or gm <= COLLAPSE_GT_MM:
            return collapse_flag(gm, np.nan), np.nan
        ta, tr = self.trace(arm), self.trace(ref_arm)
        dep = departure_mm(ta, tr, self.onsets[bag]) if (ta is not None and tr is not None) else np.nan
        return collapse_flag(gm, dep), dep

    def arm_fields(self, arm, ref_arm, bag, tag=""):
        """standard per-arm columns with a suffix tag: ate, gt_max, diverged flag, collapse flag, departure, n_scans, gt_join_scans."""
        if not self.phase.has(arm):
            self.miss("phase_summary row: " + arm)
            return {}
        gm = self.phase.get(arm, "gt_max_mm")
        flag, dep = self.collapse(arm, ref_arm, bag)
        return {"ate_mm" + tag: f3(self.phase.get(arm, "gt_ate_mm")), "gt_max_mm" + tag: f3(gm),
                "diverged" + tag: bool(np.isfinite(gm) and gm > COLLAPSE_GT_MM), "collapse" + tag: flag, "departure_mm" + tag: f3(dep),
                "n_scans" + tag: self.phase.get(arm, "n_scans"), "n_gt_joined" + tag: self.phase.get(arm, "gt_join_scans")}

    def write_missing(self, out_path):
        mp = os.path.splitext(out_path)[0] + "_missing.txt"
        if self.missing:
            with open(mp, "w") as f:
                f.write("\n".join(self.missing) + "\n")
            print("MISSING INPUTS (%d), listed in %s:" % (len(self.missing), mp), *self.missing[:40], sep="\n  ", file=sys.stderr)
            if len(self.missing) > 40:
                print("  ... %d more in the file" % (len(self.missing) - 40), file=sys.stderr)
        elif os.path.exists(mp):
            os.remove(mp)
        return 1 if self.missing else 0


def add_common_args(ap):
    ap.add_argument("--phase", nargs="+", help="phase_summary.csv of the R83 return, then the R82 return's phase_summary.csv for the controls (first file that holds an arm wins)")
    ap.add_argument("--runs", nargs="*", default=[], help="dirs holding <arm>/ or compact/<arm>/ folders (pos_trace_all.csv.gz, motion_q_scan.csv, ...); R83 runs first, R82 compact after")
    ap.add_argument("--onsets", default=None, help="onset_scans.json (r82_manifest); default the nine R82 onsets")
    ap.add_argument("--selftest", action="store_true")


class Checker:
    def __init__(self):
        self.fails = 0
        self.n = 0

    def __call__(self, name, ok, msg=""):
        self.n += 1
        print("%-52s %s  %s" % (name, "pass" if ok else "FAIL", msg))
        self.fails += (not ok)

    def done(self):
        print("%d checks, failures: %d" % (self.n, self.fails))
        return 1 if self.fails else 0


def synth_trace(n=800, seed=0, jump_at=None, jump_mm=0.0):
    rng = np.random.default_rng(seed)
    p = np.cumsum(rng.normal(size=(n, 3)) * 0.002, axis=0)
    if jump_at:
        p[jump_at:, 0] += jump_mm / 1000.0
    return pd.DataFrame(dict(scan_id=np.arange(1, n + 1), t_abs=1.6e9 + 0.1 * np.arange(n), px=p[:, 0], py=p[:, 1], pz=p[:, 2]))


def write_arm_trace(root, arm, tr, compact=True):
    d = os.path.join(root, "compact", arm) if compact else os.path.join(root, arm)
    os.makedirs(d, exist_ok=True)
    tr.to_csv(os.path.join(d, "pos_trace_all.csv.gz"), index=False)
    return d


def selftest():
    import tempfile
    c = Checker()
    ta = synth_trace(seed=1)
    tb = ta.copy()
    tb["px"] = tb["px"] + np.where(tb.scan_id > 500, 0.8, 0.0)   # 800 mm jump at scan 501, inside 485..700
    c("departure of an identical trace is 0", departure_mm(ta, ta, 485) == 0.0)
    c("departure sees a 800 mm jump", abs(departure_mm(tb, ta, 485) - 800.0) < 1.0, "%.1f" % departure_mm(tb, ta, 485))
    c("collapse flag: gt_max below 500 is no", collapse_flag(300.0, np.nan) == "no")
    c("collapse flag: above 500 and no trace is undetermined", collapse_flag(900.0, np.nan) == "undetermined")
    c("collapse flag: above 500 and departure 800 is yes", collapse_flag(900.0, 800.0) == "yes")
    c("collapse flag: above 500, departure 100 is no", collapse_flag(900.0, 100.0) == "no")
    d = tempfile.mkdtemp()
    write_arm_trace(d, "ref", ta)
    write_arm_trace(d, "bad", tb)
    write_arm_trace(d, "plain", ta, compact=False)
    pc = os.path.join(d, "ph.csv")
    pd.DataFrame(dict(arm=["ref", "bad", "plain", "nr"], gt_ate_mm=[10.0, 100.0, 11.0, 5.0], gt_max_mm=[50.0, 900.0, 600.0, 700.0], n_scans=[800] * 4)).to_csv(pc, index=False)
    x = Ctx([pc], [d])
    f = x.arm_fields("bad", "ref", "eee_01")
    c("arm_fields collapse yes, departure 800", f["collapse"] == "yes" and abs(f["departure_mm"] - 800.0) < 1.0, str(f["collapse"]))
    f = x.arm_fields("plain", "ref", "eee_01")
    c("arm_fields diverged but not collapsed", f["diverged"] and f["collapse"] == "no")
    f = x.arm_fields("nr", "ref", "eee_01")
    c("arm without its own trace is undetermined", f["collapse"] == "undetermined")
    c("missing arm is logged and gives empty fields", x.arm_fields("nope", "ref", "eee_01") == {} and "phase_summary row: nope" in x.missing)
    c("find_file toggles .gz", find_file([d], "plain", "pos_trace_all.csv") is not None and find_file([d], "ref", "pos_trace_all.csv.gz") is not None)
    return c.done()


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else 0)
