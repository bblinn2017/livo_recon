#!/usr/bin/env python3
"""R67: build jobs_smoke.json, jobs_main.json and arm_table.tsv from the R66 job list (r66_jobs.json, shipped in the package).

    python3 tools/r67/make_r67_manifest.py --jobs r67_inputs/r66_jobs.json --r66-table r67_inputs/r66_arm_table.tsv --arms r67_inputs/arms_r67.tsv \
        --out r67_manifest --results-root /root/catkin_ws/livo_recon_results

Every R67 arm is an R66 job with the SAME overrides (so a difference is the bootstrap-covariance fix and nothing else), except:
  * output paths / run id are rewritten to the R67 batch;
  * every arm sets calib.bootstrap.point_cov explicitly (sensor_model, or zero for the four '__zero' identity controls).
Smoke batch = the four '__zero' controls + the two short sensor_model baselines (48 s, 60 s). Main batch = everything else.
No perturbation key is written anywhere (rule 3.11). Nothing edits the repo.
"""
import argparse, copy, csv, json, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--r66-table", required=True); ap.add_argument("--arms", required=True)
    ap.add_argument("--out", default="r67_manifest"); ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    a = ap.parse_args()
    jobs = {j["job_id"]: j for j in json.load(open(a.jobs))}
    arms = list(csv.DictReader(open(a.arms), delimiter="\t"))
    t66 = {r["arm"]: r for r in csv.DictReader(open(a.r66_table), delimiter="\t")}
    os.makedirs(a.out, exist_ok=True)
    smoke, main_, table, seen = [], [], [], set()
    for r in arms:
        aid = r["r66_id"]; zero = aid.endswith("__zero"); base = aid[:-6] if zero else aid
        if base not in jobs:
            sys.exit("arm %s not in R66 jobs" % base)
        assert aid not in seen; seen.add(aid)
        j = copy.deepcopy(jobs[base]); j["job_id"] = aid
        is_smoke = zero or base in ("00_id_baseline_48s", "00_id_baseline_60s")
        batch = "r67_smoke" if is_smoke else "r67_main"
        o = j["overrides"]
        o["calib.bootstrap.point_cov"] = "zero" if zero else "sensor_model"
        o["eval.state_trace_run_id"] = aid
        path = "%s/%s/%s" % (a.results_root, batch, aid)
        o["outputs.path"] = path; o["outputs.debug_log_dir"] = path
        for k in list(o):
            if "perturb" in k.lower(): sys.exit("perturbation key forbidden: " + k)
        (smoke if is_smoke else main_).append(j)
        t = dict(t66[base]); t.update(arm=aid, notes=r["purpose"], keep_raw=("1" if r["snapshot"] == "True" else "0"), dup_of="", r66_id=base, group=r["group"], batch=batch)
        t["kind"] = "closed" if t.get("kind") in ("", None) else t["kind"]
        table.append(t)
    json.dump(smoke, open(os.path.join(a.out, "jobs_smoke.json"), "w"), indent=1)
    json.dump(main_, open(os.path.join(a.out, "jobs_main.json"), "w"), indent=1)
    with open(os.path.join(a.out, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(table[0]), delimiter="\t"); w.writeheader(); w.writerows(table)
    print("smoke %d  main %d  total %d" % (len(smoke), len(main_), len(table)))
    full = sum(1 for t in table if t["duration_s"] == "full")
    print("full-length arms: %d" % full)
main()
