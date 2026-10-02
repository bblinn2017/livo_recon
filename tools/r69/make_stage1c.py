#!/usr/bin/env python3
"""R73b stage 1c: a manifest for re-running a few smoke arms after the R73b patch. Ids and all overrides except outputs.path/outputs.debug_log_dir are unchanged (those move to .../<batch>/<id> so
that gen_jobs' OUT_DIR, which comes from --batch, and outputs.path are the same directory); the job list, expected_jobs, queue dir and results_root of the spec are cut/renamed.

python3 tools/r69/make_stage1c.py --manifest r73_manifest --out r73_manifest_1c [--arms a,b,c]
default arms: r73_00_id_baseline_60s, r73_id_shadow_same_60s, r73_smoke_shadow_iso_60s, r73_smoke_shadow_fixed_mq_60s
Writes <out>/jobs_smoke.json (unchanged job ids and overrides) and <out>/listener_spec_smoke.json. Prints the arms written. Exit 2 if an arm is not in jobs_smoke.json."""
import argparse, json, os, sys

DEFAULT = ["r73_00_id_baseline_60s", "r73_id_shadow_same_60s", "r73_smoke_shadow_iso_60s", "r73_smoke_shadow_fixed_mq_60s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True); ap.add_argument("--arms", default=",".join(DEFAULT))
    ap.add_argument("--queue-dir", default="ablations/_job_queue/r73_smoke1c", help="queue directory of the 1c batch (gen_jobs --batch r73_smoke1c); a new directory so the 1b queue is never re-dispatched")
    a = ap.parse_args()
    want = [x for x in a.arms.split(",") if x]
    jobs = json.load(open(os.path.join(a.manifest, "jobs_smoke.json")))
    have = {j["job_id"] for j in jobs}
    bad = [x for x in want if x not in have]
    if bad:
        print("NOT IN jobs_smoke.json: %s" % bad)
        return 2
    sel = [j for j in jobs if j["job_id"] in set(want)]
    batch = os.path.basename(a.queue_dir.rstrip("/"))            # gen_jobs derives job.env OUT_DIR from --batch, so outputs.path must live under the same name
    for j in sel:
        for k in ("outputs.path", "outputs.debug_log_dir"):
            root, jid = os.path.split(j["overrides"][k].rstrip("/"))
            j["overrides"][k] = "%s/%s" % (os.path.dirname(root) + "/" + batch, jid)
    spec = json.load(open(os.path.join(a.manifest, "listener_spec_smoke.json")))
    spec["queues"] = [q for q in spec["queues"] if q["dir"].rstrip("/").endswith("_smoke")]   # the refuse queue is not re-run
    for q in spec["queues"]:
        q["expected_jobs"] = len(sel)
        q["dir"] = a.queue_dir
        q["results_root"] = os.path.dirname(q["results_root"].rstrip("/")) + "/" + batch   # = the directory the rewritten outputs.path points into = gen_jobs OUT_DIR
    os.makedirs(a.out, exist_ok=True)
    json.dump(sel, open(os.path.join(a.out, "jobs_smoke.json"), "w"), indent=1)
    json.dump(spec, open(os.path.join(a.out, "listener_spec_smoke.json"), "w"), indent=1)
    print("stage 1c: %d arms: %s" % (len(sel), ", ".join(j["job_id"] for j in sel)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
