#!/usr/bin/env python3
"""p_collect.py -- run p_reduce over a results root and pack a small, transferable return (planning agent's, R69 design).

    python3 p_collect.py <results_root> --out P_RETURN [--arms arms.txt] [--scan-arms scan_arms.txt] [--params bag_params.json]
                          [--gt] [--budget-mb 20] [--part-mb 18]

<results_root>/<arm>/ must hold state_trace.csv + state_reference.txt (and optionally imu_cov_growth.csv, arm.json, lio_gt_matched.csv).
Output P_RETURN/:
    window_table.csv     ONE ROW PER ARM, wide: <window>__<quantity>__<stat>  (every arm that has a state_trace)
    scan/<arm>.csv.gz    thinned per-scan series, only for arms in --scan-arms (default: none) -- added in the order given until --budget-mb
    windows/<arm>.json   full per-window JSON for --scan-arms arms
    manifest.json        schema, script digests, per-arm status, sizes, warnings, dropped files
    P_RETURN.zip (or .part00, .part01 ... when larger than --part-mb; reassemble with cat)
Nothing is deleted; nothing outside --out is written.
"""
import argparse, hashlib, json, os, sys, zipfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p_reduce  # noqa: E402


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def flat(S):
    row = {"arm": S["arm"], "n_scans": S.get("n_scans", 0), "warnings": "; ".join(S.get("warnings", []))}
    for wn, W in S.get("windows", {}).items():
        row["%s__n" % wn] = W.get("n")
        for q, d in W.items():
            if isinstance(d, dict):
                for st, v in d.items():
                    row["%s__%s__%s" % (wn, q, st)] = v
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="P_RETURN")
    ap.add_argument("--arms", default=None, help="file: one arm directory name per line (default: every subdirectory with a state_trace.csv)")
    ap.add_argument("--scan-arms", default=None, help="file: arms that also get the thinned per-scan series, in priority order")
    ap.add_argument("--params", default=None)
    ap.add_argument("--gt", action="store_true")
    ap.add_argument("--stat-last", type=int, default=480)
    ap.add_argument("--trans-last", type=int, default=540)
    ap.add_argument("--full-until", type=int, default=60)
    ap.add_argument("--until2", type=int, default=700)
    ap.add_argument("--step1", type=int, default=5)
    ap.add_argument("--step2", type=int, default=20)
    ap.add_argument("--budget-mb", type=float, default=20.0)
    ap.add_argument("--part-mb", type=float, default=18.0)
    a = ap.parse_args()
    arms = [l.strip() for l in open(a.arms)] if a.arms else sorted(d for d in os.listdir(a.root) if os.path.exists(os.path.join(a.root, d, "state_trace.csv")) or os.path.exists(os.path.join(a.root, d, "P", "p_window.json")))
    scan_arms = [l.strip() for l in open(a.scan_arms) if l.strip()] if a.scan_arms else []
    os.makedirs(os.path.join(a.out, "scan"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "windows"), exist_ok=True)
    rows, status, scan_bytes, dropped = [], {}, 0, []
    for arm in arms:
        d = os.path.join(a.root, arm)
        try:
            pw, ps = os.path.join(d, "P", "p_window.json"), os.path.join(d, "P", "p_scan.csv.gz")
            if os.path.exists(pw):   # R69: the listener's scorer already ran p_reduce on the raw before it was deleted
                S = json.load(open(pw))
                T = pd.read_csv(ps) if os.path.exists(ps) else None
            else:
                S, T = p_reduce.reduce_arm(d, a)
        except Exception as e:  # keep going; report
            status[arm] = "error: %s" % e
            print("ERROR in %s: %r" % (arm, e), file=sys.stderr)
            continue
        status[arm] = "ok" if T is not None else "skipped: " + "; ".join(S.get("warnings", []))
        if T is None:
            continue
        rows.append(flat(S))
        if arm in scan_arms:
            tmp = os.path.join(a.out, "scan", arm + ".csv.gz")
            p_reduce.write_outputs(S, T, os.path.join(a.out, "_tmp"))
            os.replace(os.path.join(a.out, "_tmp", "p_scan.csv.gz"), tmp)
            os.replace(os.path.join(a.out, "_tmp", "p_window.json"), os.path.join(a.out, "windows", arm + ".json"))
            sz = os.path.getsize(tmp)
            if (scan_bytes + sz) / 1e6 > a.budget_mb * 0.8:
                os.remove(tmp); os.remove(os.path.join(a.out, "windows", arm + ".json")); dropped.append(arm)
            else:
                scan_bytes += sz
    tmpdir = os.path.join(a.out, "_tmp")
    if os.path.isdir(tmpdir):
        for f in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, f))
        os.rmdir(tmpdir)
    wt = pd.DataFrame(rows)
    wt.to_csv(os.path.join(a.out, "window_table.csv"), index=False, float_format="%.5g")
    man = {"schema": p_reduce.SCHEMA, "n_arms_requested": len(arms), "n_arms_reduced": len(rows), "status": status,
           "scan_files": sorted(os.listdir(os.path.join(a.out, "scan"))), "scan_files_dropped_for_budget": dropped,
           "window_table_bytes": os.path.getsize(os.path.join(a.out, "window_table.csv")), "scan_bytes": scan_bytes,
           "script_sha256_16": {"p_reduce.py": sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p_reduce.py")),
                                "p_collect.py": sha(os.path.abspath(__file__)),
                                "common.py": sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common.py"))},
           "args": vars(a)}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1, default=str)
    zpath = a.out.rstrip("/") + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _, files in os.walk(a.out):
            for f in files:
                p = os.path.join(base, f)
                z.write(p, os.path.relpath(p, os.path.dirname(os.path.abspath(a.out))))
    size = os.path.getsize(zpath)
    if size > a.part_mb * 1e6:
        with open(zpath, "rb") as f:
            k = 0
            while True:
                chunk = f.read(int(a.part_mb * 1e6))
                if not chunk:
                    break
                open("%s.part%02d" % (zpath, k), "wb").write(chunk); k += 1
        print("zip %.2f MB split into %d parts; reassemble with: cat %s.part* > %s" % (size / 1e6, k, zpath, zpath))
    else:
        print("zip %.2f MB" % (size / 1e6))
    print("arms reduced %d / %d; scan files %d; dropped for budget %d" % (len(rows), len(arms), len(man["scan_files"]), len(dropped)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
