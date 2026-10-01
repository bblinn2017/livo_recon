#!/usr/bin/env python3
"""stop_reasons.py -- one row per arm: how far it ran and how it ended (planning agent's, R69; fixes F-313). Facts only.

    python3 tools/r69/stop_reasons.py --runs r72_runs --arms r72_manifest/arm_table.tsv --out R72_RETURN/stop_reasons.csv [--extra-runs r68_runs:r68_manifest/arm_table.tsv ...]

Columns: arm, batch, seq, n_scans, first_scan, last_scan, seq_max_last_scan, last_scan_gap, wall_s, exit_code, sentinel, stop_reason, evidence_file.
`first_scan`/`last_scan` are ABSOLUTE scan indices from arm_summary.json, so an arm with a longer calibration (fewer scans after it) still ends at the same last scan as a
complete arm. `seq_max_last_scan` = the largest last_scan among the arms of the same sequence and duration class across every batch passed in (so a whole group cut by a cap is still seen against the complete arms). `last_scan_gap` = seq_max_last_scan - last_scan.
`stop_reason` is one of: `failed_run` (FAILED sentinel / exit code != 0), `ended_before_group_max` (gap > 2 scans), `reached_group_max`, `no_summary` (nothing reduced).
The old R68 rule compared n_scans with a constant (3,980) and so mislabelled calibration-length arms as truncated (F-313).
"""
import argparse, csv, json, os, sys


def rows_for(runs, table, batch_prefix=""):
    out = []
    tab = list(csv.DictReader(open(table), delimiter="\t"))
    for r in tab:
        D = os.path.join(runs, r["arm"])
        meta, S = {}, {}
        for fn, tgt in (("arm.json", "meta"), ("arm_summary.json", "S")):
            p = os.path.join(D, fn)
            if os.path.exists(p):
                try:
                    (meta if tgt == "meta" else S).update(json.load(open(p)))
                except Exception:
                    pass
        out.append(dict(arm=r["arm"], batch=r.get("batch", ""), seq=r["seq"], dur=r["duration_s"], n_scans=S.get("n_scans"), first_scan=S.get("first_scan"),
                        last_scan=S.get("last_scan"), wall_s=meta.get("wall_s"), exit_code=meta.get("exit_code"), sentinel=meta.get("sentinel"),
                        evidence_file=os.path.join(D, "arm_summary.json") if S else (os.path.join(D, "run_log_tail.txt") if os.path.exists(os.path.join(D, "run_log_tail.txt")) else "")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--arms", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--extra-runs", nargs="*", default=[], help="runs_dir:arm_table pairs (earlier rounds)")
    a = ap.parse_args()
    rows = rows_for(a.runs, a.arms)
    for pair in a.extra_runs:
        rd, tb = pair.split(":", 1)
        if os.path.isdir(rd) and os.path.exists(tb):
            rows += rows_for(rd, tb)
    mx = {}
    for r in rows:
        if r["last_scan"] is not None and r["exit_code"] in (0, None):
            k = (r["seq"], r["dur"])
            mx[k] = max(mx.get(k, 0), int(r["last_scan"]))
    cols = ["arm", "batch", "seq", "n_scans", "first_scan", "last_scan", "seq_max_last_scan", "last_scan_gap", "wall_s", "exit_code", "sentinel", "stop_reason", "evidence_file"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader()
        for r in rows:
            m = mx.get((r["seq"], r["dur"]))
            if r["last_scan"] is None:
                reason, gap = ("failed_run" if r["sentinel"] == "FAILED" or r["exit_code"] not in (0, None) else "no_summary"), None
            else:
                gap = (m - int(r["last_scan"])) if m is not None else None
                reason = "failed_run" if r["exit_code"] not in (0, None) else ("ended_before_group_max" if gap is not None and gap > 2 else "reached_group_max")
            w.writerow({**{k: r.get(k) for k in cols if k in r}, "seq_max_last_scan": m, "last_scan_gap": gap, "stop_reason": reason})
    print("wrote", a.out, "rows", len(rows))


main()
