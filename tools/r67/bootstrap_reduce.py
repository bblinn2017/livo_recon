#!/usr/bin/env python3
"""R67 bootstrap-covariance reducer + identity check (planning agent's; not built or run by me).

  python3 bootstrap_reduce.py <arm_dir>                   -> <arm_dir>/bootstrap_summary.json  (parsed bootstrap_preparation.txt, bootstrap_map_seed.txt head, deskew fallback line from run.log)
  python3 bootstrap_reduce.py --collect <runs_dir> --out <ret_dir>
        -> <ret_dir>/bootstrap_summary.csv   one row per arm: point_cov mode (from config.yaml), schema, hash, sensor_cov_trace_over_3_{mean,min,max}, zero count, deskew fallback flag
        -> <ret_dir>/identity_gates.txt       PASS/FAIL per gate (see R67_INSTRUCTIONS.md section 6)
Gates: G1 each '__zero' arm: blank-run-id state_trace digest == the R66 digest of its twin; G2 zero arms: bootstrap hash 8313440171814820807 (baseline twins) and
sensor_cov_zero_point_count == total_prepared_point_count; G3 every non-zero arm: schema v2, trace_mean > 0, zero count < total; G4 sensor_model baseline digest != zero digest.
"""
import argparse, csv, json, os, re, sys

R66_DIGEST = {"00_id_baseline_48s": "197d7f00e23f45160a1a7efbbc9e86dc85bfae069140d1eb3790f3dc75500341",
              "00_id_baseline_60s": "a096c8b9e581f3c47bfa184c2909aea450ff74700a77a766ab93fded3391c8ad",
              "00_id_baseline_full": "2d3eebfb2a84ae0fe8e03222232ea814bc323e02027de395681b54f32b4f6857",
              "00_id_motionq_full": "9463c218c1047a6a7f08b363b4687e5f5d3e8c595f50e64c8bb3c414d5073373"}
R66_HASH = "8313440171814820807"


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


def mode_of(arm_dir):
    p = os.path.join(arm_dir, "config.yaml")
    if os.path.exists(p):
        txt = open(p).read()
        m = re.search(r"point_cov:\s*(\w+)", txt)
        if m:
            return m.group(1)
    return "unknown"


def summarize(arm_dir):
    d = parse_prep(os.path.join(arm_dir, "bootstrap_preparation.txt"))
    fb = False
    rl = os.path.join(arm_dir, "run.log")
    if os.path.exists(rl):
        fb = "poses-empty fallback used" in open(rl, errors="replace").read()
    dg = os.path.join(arm_dir, "state_trace_blank_runid.sha256")
    return {"arm": os.path.basename(os.path.abspath(arm_dir)), "point_cov": mode_of(arm_dir), "schema": d.get("schema"),
            "hash": d.get("flattened_prepared_population_hash"), "total_prepared": d.get("total_prepared_point_count"),
            "tr_mean": d.get("sensor_cov_trace_over_3_mean"), "tr_min": d.get("sensor_cov_trace_over_3_min"),
            "tr_max": d.get("sensor_cov_trace_over_3_max"), "n_zero": d.get("sensor_cov_zero_point_count"),
            "deskew_fallback": fb, "digest": open(dg).read().strip() if os.path.exists(dg) else None}


def collect(runs, out):
    os.makedirs(out, exist_ok=True)
    rows = [summarize(os.path.join(runs, a)) for a in sorted(os.listdir(runs)) if os.path.isdir(os.path.join(runs, a))]
    with open(os.path.join(out, "bootstrap_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    by = {r["arm"]: r for r in rows}
    L = []
    def gate(name, ok, why=""):
        L.append("%s  %s  %s" % ("PASS" if ok else "FAIL", name, why))
    for base, dig in R66_DIGEST.items():
        z = by.get(base + "__zero")
        if z is None:
            gate("G1 %s__zero present" % base, False, "arm missing"); continue
        gate("G1 %s__zero digest == R66" % base, z["digest"] == dig, "got %s" % z["digest"])
        s = by.get(base)
        if s is not None:
            gate("G4 %s sensor_model digest != zero digest" % base, s["digest"] is not None and s["digest"] != z["digest"], "got %s" % s["digest"])
        if base != "00_id_motionq_full":
            gate("G2 %s__zero bootstrap hash == R66" % base, z["hash"] == R66_HASH, "got %s" % z["hash"])
        gate("G2 %s__zero every prepared point has zero sensor_cov" % base, z["n_zero"] is not None and z["n_zero"] == z["total_prepared"],
             "n_zero %s total %s" % (z["n_zero"], z["total_prepared"]))
    bad = [r["arm"] for r in rows if not r["arm"].endswith("__zero") and not (r["schema"] == "bootstrap_preparation_v2" and r["tr_mean"] not in (None, "0") and float(r["tr_mean"]) > 0 and r["n_zero"] != r["total_prepared"])]
    gate("G3 every non-zero arm carries nonzero bootstrap sensor_cov", not bad, "failing arms: %s" % bad[:20])
    fb = [r["arm"] for r in rows if r["deskew_fallback"]]
    L.append("INFO deskew poses-empty fallback used in %d arms: %s" % (len(fb), fb[:20]))
    open(os.path.join(out, "identity_gates.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if not any(l.startswith("FAIL") for l in L) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("arm_dir", nargs="?"); ap.add_argument("--collect"); ap.add_argument("--out", default="R67_RETURN")
    a = ap.parse_args()
    if a.collect:
        sys.exit(collect(a.collect, a.out))
    s = summarize(a.arm_dir); json.dump(s, open(os.path.join(a.arm_dir, "bootstrap_summary.json"), "w"), indent=1)
