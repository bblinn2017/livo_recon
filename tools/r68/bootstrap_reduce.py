#!/usr/bin/env python3
"""R68 bootstrap reducer + identity gates (planning agent's; not built or run by me).

  python3 bootstrap_reduce.py <arm_dir>                   -> <arm_dir>/bootstrap_summary.json
  python3 bootstrap_reduce.py --collect <runs_dir> --out <ret_dir>
        -> <ret_dir>/bootstrap_summary.csv   one row per arm: schema, hash, sensor_cov_trace_over_3_{mean,min,max}, zero count, deskew fallback flag, digest
        -> <ret_dir>/identity_gates.txt       PASS/FAIL per gate (see R68_INSTRUCTIONS.md section 6)
Gates (R68 build vs the R67 sensor_model runs; the R68 patch removes code and must not change arithmetic):
  G1 00_id_baseline_{48s,60s,full}, 00_id_motionq_full: blank-run-id state_trace digest == the R67 sensor_model digest
  G2 every non-refusal arm: bootstrap_preparation schema v2, flattened_prepared_population_hash == R67 sensor_model hash, zero count 0
  G3 00_id_default_posecov_60s digest == 00_id_baseline_60s digest (new compiled default sensor_only == the pinned value)
  G4 00_id_infl_whole_f1_60s digest == 00_id_baseline_60s digest (whole-run inflation at factor 1 is the identity)
  G5 00_id_baseline_full_rerun digest == 00_id_baseline_full digest (determinism)
  G7 r68_stamps_baseline_full digest == 00_id_baseline_full digest (the stamp log is write-only)
  G6 r68_refuse jobs: each refused at startup and run.log contains the expected text
"""
import argparse, csv, json, os, re, sys

R67_DIGEST = {"00_id_baseline_48s": "694fadd31260f9fb262e7e32690e9cc7aad80079b08c22ed1d2f3f9ec321d1be",
              "00_id_baseline_60s": "8cb066436baaedb45f2af20921ff56337a1b553e1aa2d8fafe9d4170a9043f19",
              "00_id_baseline_full": "8bc89b6bb69aaf32d2283a969ad6cffad798836cf40ccf0def2fe9712d14e5c2",
              "00_id_motionq_full": "98f3855cb2146bb06e9dc15b101a8699804a395b4807144323f99e2e509dbad1"}
R67_HASH = "1957524258770630969"
REFUSE = {"r68_refuse_point_cov_zero": "removed key: calib/bootstrap/point_cov",
          "r68_refuse_spline_mode": "spline/mode"}


def parse_prep(path):
    d = {}
    if not os.path.exists(path):
        return d
    for ln in open(path):
        p = ln.split()
        if len(p) == 2:
            d[p[0]] = p[1]
        elif len(p) == 1:
            d.setdefault("_flags", []).append(p[0])
        if p and p[0] == "observation":
            d["n_obs_lines"] = d.get("n_obs_lines", 0) + 1
    return d


def summarize(arm_dir):
    d = parse_prep(os.path.join(arm_dir, "bootstrap_preparation.txt"))
    fb = False
    rl = os.path.join(arm_dir, "run.log")
    if os.path.exists(rl):
        fb = "poses-empty fallback used" in open(rl, errors="replace").read()
    dg = os.path.join(arm_dir, "state_trace_blank_runid.sha256")
    return {"arm": os.path.basename(os.path.abspath(arm_dir)), "schema": d.get("schema"),
            "hash": d.get("flattened_prepared_population_hash"), "total_prepared": d.get("total_prepared_point_count"),
            "tr_mean": d.get("sensor_cov_trace_over_3_mean"), "tr_min": d.get("sensor_cov_trace_over_3_min"),
            "tr_max": d.get("sensor_cov_trace_over_3_max"), "n_zero": d.get("sensor_cov_zero_point_count"),
            "deskew_fallback": fb, "digest": open(dg).read().strip() if os.path.exists(dg) else None}


def collect(runs, out):
    os.makedirs(out, exist_ok=True)
    rows = [summarize(os.path.join(runs, a)) for a in sorted(os.listdir(runs)) if os.path.isdir(os.path.join(runs, a)) and not a.startswith("r68_refuse")]
    with open(os.path.join(out, "bootstrap_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    by = {r["arm"]: r for r in rows}
    L = []
    def gate(name, ok, why=""):
        L.append("%s  %s  %s" % ("PASS" if ok else "FAIL", name, why))
    for base, dig in R67_DIGEST.items():
        r = by.get(base)
        if r is None:
            gate("G1 %s present" % base, False, "arm missing"); continue
        gate("G1 %s digest == R67 sensor_model digest" % base, r["digest"] == dig, "got %s" % r["digest"])
    bad = [r["arm"] for r in rows if not r["arm"].startswith("r68_refuse") and not (r["schema"] == "bootstrap_preparation_v2" and r["hash"] == R67_HASH and r["n_zero"] == "0")]
    gate("G2 every arm: schema v2, hash == R67, zero count 0", not bad, "failing arms: %s" % bad[:20])
    def same(g, a_, b_):
        x, y = by.get(a_), by.get(b_)
        gate("%s %s digest == %s digest" % (g, a_, b_), x is not None and y is not None and x["digest"] is not None and x["digest"] == y["digest"],
             "got %s / %s" % (x and x["digest"], y and y["digest"]))
    same("G3", "00_id_default_posecov_60s", "00_id_baseline_60s")
    same("G4", "00_id_infl_whole_f1_60s", "00_id_baseline_60s")
    same("G5", "00_id_baseline_full_rerun", "00_id_baseline_full")
    same("G7", "r68_stamps_baseline_full", "00_id_baseline_full")
    for arm, txt in REFUSE.items():
        d = os.path.join(runs, arm)
        have = any(os.path.exists(os.path.join(d, f)) and txt in open(os.path.join(d, f), errors="replace").read() for f in ("run.log", "run_log_tail.txt"))
        gate("G6 %s refused at startup (run.log contains %r)" % (arm, txt), have, "run dir %s" % ("present" if os.path.isdir(d) else "missing"))
    fb = [r["arm"] for r in rows if r["deskew_fallback"]]
    L.append("INFO deskew poses-empty fallback used in %d arms: %s" % (len(fb), fb[:20]))
    open(os.path.join(out, "identity_gates.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if not any(l.startswith("FAIL") for l in L) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("arm_dir", nargs="?"); ap.add_argument("--collect"); ap.add_argument("--out", default="R68_RETURN")
    a = ap.parse_args()
    if a.collect:
        sys.exit(collect(a.collect, a.out))
    s = summarize(a.arm_dir); json.dump(s, open(os.path.join(a.arm_dir, "bootstrap_summary.json"), "w"), indent=1)
