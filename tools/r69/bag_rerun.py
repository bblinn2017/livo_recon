#!/usr/bin/env python3
"""R69 (F-286): host-only re-reduction of the R67 BAG arms with the per-bag windows from bag_params.json. Planning agent's script; not run by me.

  python3 tools/r69/bag_rerun.py --runs r67_runs --arms r67_manifest/arm_table.tsv --params r69_inputs/bag_params.json --out R69_RETURN/bag_rerun

For every R67 arm whose seq is not eee_01 (arm table column `seq`): copies arm.json, state_trace.csv (or .gz), state_reference.txt and lio_gt_matched.csv (or .gz)
where they exist into <out>/work/<arm>/ (the R67 arm directory is NOT modified), runs r69_reduce.py on the copy with --params, and writes
<out>/bag_rerun_summary.csv: arm, seq, onset_scan, stat_last, trans_last, excursion_mm, settled_offset_mm, wander_mm, gt_ate_mm, evo_ate_m, n_scans, status.
An arm whose raw inputs are missing gets status `no_raw_input:<file>` and empty numbers (nothing is invented). Also writes <out>/README.txt with the command.
Sufficiency: settled_offset_mm and wander_mm are the only R67 numbers that used the wrong windows; excursion and ATE do not depend on the window.
Control: for seq eee_01 nothing is re-reduced; the script re-reduces 00_id_baseline_full once with the eee_01 entry of bag_params.json (485/480/540) into
<out>/control_eee_01/ and writes its settled_offset_mm beside the value in r67_runs/00_id_baseline_full/arm_summary.json (they must be equal)."""
import argparse, csv, gzip, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NEED = ["state_trace.csv", "state_reference.txt", "lio_gt_matched.csv"]


def stage(src_dir, dst):
    os.makedirs(dst, exist_ok=True)
    miss = []
    for fn in NEED + ["arm.json"]:
        s = os.path.join(src_dir, fn)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(dst, fn))
        elif os.path.exists(s + ".gz"):
            with gzip.open(s + ".gz", "rb") as fi, open(os.path.join(dst, fn), "wb") as fo:
                shutil.copyfileobj(fi, fo)
        elif fn in ("state_trace.csv", "state_reference.txt"):
            miss.append(fn)
    return miss


def reduce(dst, params):
    subprocess.run([sys.executable, os.path.join(HERE, "r69_reduce.py"), dst, "--params", params], stdout=open(os.path.join(dst, "reduce.log"), "w"), stderr=subprocess.STDOUT)
    p = os.path.join(dst, "arm_summary.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--arms", required=True); ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = json.load(open(a.params)); os.makedirs(a.out, exist_ok=True)
    rows = []
    for r in csv.DictReader(open(a.arms), delimiter="\t"):
        if r["seq"] == "eee_01":
            continue
        d = os.path.join(a.runs, r["arm"]); dst = os.path.join(a.out, "work", r["arm"])
        row = {"arm": r["arm"], "seq": r["seq"], **{k: P[r["seq"]][k] for k in ("onset_scan", "stat_last", "trans_last")}}
        if not os.path.isdir(d):
            row["status"] = "no_arm_dir"; rows.append(row); continue
        miss = stage(d, dst)
        if miss:
            row["status"] = "no_raw_input:" + ",".join(miss); rows.append(row); continue
        if not os.path.exists(os.path.join(dst, "arm.json")):
            json.dump({"arm": r["arm"], "seq": r["seq"]}, open(os.path.join(dst, "arm.json"), "w"))
        S = reduce(dst, a.params)
        row.update(excursion_mm=S.get("excursion_mm"), settled_offset_mm=S.get("settled_offset_mm"), wander_mm=S.get("wander_mm"), gt_ate_mm=S.get("gt_ate_mm"),
                   evo_ate_m=S.get("evo_ate_m"), n_scans=S.get("n_scans"), status="ok" if "settled_offset_mm" in S else "reduce_failed")
        rows.append(row)
    cols = ["arm", "seq", "onset_scan", "stat_last", "trans_last", "excursion_mm", "settled_offset_mm", "wander_mm", "gt_ate_mm", "evo_ate_m", "n_scans", "status"]
    with open(os.path.join(a.out, "bag_rerun_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
    # eee_01 control
    c = os.path.join(a.out, "control_eee_01"); base = os.path.join(a.runs, "00_id_baseline_full")
    line = "control_eee_01: 00_id_baseline_full not available"
    if os.path.isdir(base) and not stage(base, c):
        json.dump({"arm": "00_id_baseline_full", "seq": "eee_01"}, open(os.path.join(c, "arm.json"), "w")) if not os.path.exists(os.path.join(c, "arm.json")) else None
        S = reduce(c, a.params); old = json.load(open(os.path.join(base, "arm_summary.json"))) if os.path.exists(os.path.join(base, "arm_summary.json")) else {}
        line = "control_eee_01: settled_offset_mm rerun %s  R67 %s  wander_mm rerun %s  R67 %s" % (S.get("settled_offset_mm"), old.get("settled_offset_mm"), S.get("wander_mm"), old.get("wander_mm"))
    open(os.path.join(a.out, "README.txt"), "w").write("python3 tools/r69/bag_rerun.py --runs %s --arms %s --params %s --out %s\n%s\n" % (a.runs, a.arms, a.params, a.out, line))
    print(line); print("arms: %d, ok: %d" % (len(rows), sum(1 for r in rows if r["status"] == "ok")))


main()
