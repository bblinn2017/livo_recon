#!/usr/bin/env python3
"""H3 (F-223): stop_reasons.csv for every R67 and R68 arm.
Columns: arm, batch, n_scans, last_scan, wall_s, exit_code, stop_reason, evidence_file
stop_reason is derived, not guessed: 'full' if n_scans/last_scan is within 2 scans of the
batch's reference full length for that seq (from bag_params.json's n_scans, or the batch's
own observed max over arms of the same seq+duration=full if the seq isn't in bag_params.json),
'truncated' otherwise (with the scan deficit noted), 'error' if exit_code != 0, 'n/a' if the
arm's duration_s isn't 'full' (a 48s/60s arm ending early by design isn't a truncation) or if
arm_summary.json/arm.json is missing.
"""
import csv, json, os, sys

BAG_PARAMS = json.load(open("r68_inputs/bag_params.json"))

def ref_full_scans(seq):
    e = BAG_PARAMS.get(seq)
    return e["n_scans"] if e else None

def process(runs_dir, batch, rows, full_len_by_seq):
    for arm in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, arm)
        if not os.path.isdir(d):
            continue
        aj = os.path.join(d, "arm.json")
        sj = os.path.join(d, "arm_summary.json")
        if not os.path.exists(aj):
            rows.append({"arm": arm, "batch": batch, "n_scans": "", "last_scan": "", "wall_s": "",
                         "exit_code": "", "stop_reason": "n/a (no arm.json)", "evidence_file": aj})
            continue
        a = json.load(open(aj))
        exit_code = a.get("exit_code")
        wall_s = a.get("wall_s")
        seq = a.get("seq")
        duration = a.get("duration_s")
        n_scans = last_scan = None
        if os.path.exists(sj):
            s = json.load(open(sj))
            n_scans = s.get("n_scans")
            last_scan = s.get("last_scan")
        evidence = sj if os.path.exists(sj) else aj
        if exit_code not in (0, None) and exit_code != 0:
            reason = "error (exit_code=%s)" % exit_code
        elif str(duration) != "full":
            reason = "n/a (duration=%s, not a full-length arm)" % duration
        elif n_scans is None:
            reason = "n/a (no arm_summary.json / n_scans)"
        else:
            ref = full_len_by_seq.get(seq) or ref_full_scans(seq)
            if ref is None:
                reason = "n/a (no reference full-scan count for seq=%s)" % seq
            elif ref - n_scans <= 2:
                reason = "full"
            else:
                reason = "truncated (%d scans short of reference %d, %.1f%%)" % (ref - n_scans, ref, 100.0 * (ref - n_scans) / ref)
        rows.append({"arm": arm, "batch": batch, "n_scans": n_scans, "last_scan": last_scan,
                     "wall_s": wall_s, "exit_code": exit_code, "stop_reason": reason, "evidence_file": evidence})

def observed_full_by_seq(runs_dir):
    """Fallback reference: for seqs bag_params.json doesn't cover (e.g. sbs_*, non-eee/nya),
    use the max n_scans observed among this run's own duration=full arms of that seq."""
    best = {}
    for arm in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, arm)
        aj, sj = os.path.join(d, "arm.json"), os.path.join(d, "arm_summary.json")
        if not (os.path.isdir(d) and os.path.exists(aj) and os.path.exists(sj)):
            continue
        a = json.load(open(aj))
        if str(a.get("duration_s")) != "full":
            continue
        s = json.load(open(sj))
        n = s.get("n_scans")
        seq = a.get("seq")
        if n and (seq not in best or n > best[seq]):
            best[seq] = n
    return best

if __name__ == "__main__":
    rows = []
    full_by_seq = {}
    full_by_seq.update(observed_full_by_seq("r67_runs"))
    for seq, n in observed_full_by_seq("r68_runs").items():
        full_by_seq[seq] = max(full_by_seq.get(seq, 0), n)
    process("r67_runs", "R67", rows, full_by_seq)
    process("r68_runs", "R68", rows, full_by_seq)
    out = sys.argv[1] if len(sys.argv) > 1 else "R68_RETURN/stop_reasons.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, ["arm", "batch", "n_scans", "last_scan", "wall_s", "exit_code", "stop_reason", "evidence_file"])
        w.writeheader(); w.writerows(rows)
    n_trunc = sum(1 for r in rows if str(r["stop_reason"]).startswith("truncated"))
    print("wrote %d rows to %s; %d truncated" % (len(rows), out, n_trunc))
    for r in rows:
        if str(r["stop_reason"]).startswith("truncated"):
            print(" ", r["batch"], r["arm"], r["stop_reason"])
