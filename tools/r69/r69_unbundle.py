#!/usr/bin/env python3
"""r69_unbundle.py -- after the listener has handled the queues: gather each job's committed, reduced outputs into <runs>/<arm>/ (planning agent's, R69).

    python3 tools/r69/r69_unbundle.py --specs r71_manifest/listener_spec_smoke.json r71_manifest/listener_spec_main.json --root . --runs r71_runs
            [--matrices-arms tools/r69/matrices_arms.txt]

For every job in every queue of the specs:
  * committed (listener_out/summary.json exists): unzip listener_out/reduce/arm_bundle.zip into <runs>/<arm>/, copy archive_index.json
    (raw bytes, sha256, gz bytes, ratio per archived file) and summary.json to <runs>/<arm>/listener_summary.json; for arms listed in --matrices-arms also copy
    archive/pose_control_first_frame_solve_matrices.txt.gz to <runs>/_matrices/<arm>.txt.gz (the full-matrix dump of frames 1..5, lossless);
  * FAILED jobs (skipped by the listener, raw kept): <runs>/<arm>/arm.json (exit_code 1) and run_log_tail.txt (last 40 lines of run.log);
  * anything else: listed in <runs>/_unbundle_status.json as `not_committed:<listener state>`.
Reads only; the raw files stay where they are (or are gone, if the listener consumed them: the lossless archives are under listener_out/).
"""
import argparse, json, os, shutil, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "parallel"))
import sweep_listener as sl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True); ap.add_argument("--root", default="."); ap.add_argument("--runs", required=True)
    ap.add_argument("--matrices-arms", default=None)
    a = ap.parse_args()
    mats = set(l.strip() for l in open(a.matrices_arms) if l.strip()) if a.matrices_arms else set()
    os.makedirs(a.runs, exist_ok=True)
    status = {}
    for sp in a.specs:
        spec = sl.load_spec(sp, a.root)
        for q in spec["queues"]:
            qd = sl.absp(spec, q["dir"])
            if not os.path.isdir(qd):
                continue
            for n in sorted(os.listdir(qd)):
                if not os.path.isdir(os.path.join(qd, n)):
                    continue
                job = sl.Job(spec, q, n)
                D = os.path.join(a.runs, n)
                summ = os.path.join(job.lo, "summary.json")
                bundle = os.path.join(job.lo, "reduce", "arm_bundle.zip")
                if os.path.exists(summ) and os.path.exists(bundle):
                    shutil.rmtree(D, ignore_errors=True)
                    with zipfile.ZipFile(bundle) as z:
                        z.extractall(a.runs)
                    shutil.copy(summ, os.path.join(D, "listener_summary.json"))
                    ai = os.path.join(job.lo, "reduce", "archive", "archive_index.json")
                    if os.path.exists(ai):
                        shutil.copy(ai, os.path.join(D, "archive_index.json"))
                    mg = os.path.join(job.lo, "reduce", "archive", "pose_control_first_frame_solve_matrices.txt.gz")
                    if n in mats and os.path.exists(mg):
                        os.makedirs(os.path.join(a.runs, "_matrices"), exist_ok=True)
                        shutil.copy(mg, os.path.join(a.runs, "_matrices", n + ".txt.gz"))
                    es = os.path.join(job.out_dir, "exit_status.txt")   # R69B: real node exit status written by run_job.sh (F-326), if the agent added it
                    if os.path.exists(es):
                        shutil.copy(es, os.path.join(D, "exit_status.txt"))
                    status[n] = "ok"
                elif job.sentinel() == "FAILED":
                    os.makedirs(D, exist_ok=True)
                    json.dump({"arm": n, "exit_code": 1, "sentinel": "FAILED"}, open(os.path.join(D, "arm.json"), "w"))
                    es = os.path.join(job.out_dir, "exit_status.txt")
                    if os.path.exists(es):
                        shutil.copy(es, os.path.join(D, "exit_status.txt"))
                    rl = sl.os.path.join(job.out_dir, "run.log")
                    if not os.path.exists(rl):
                        rl = os.path.join(job.job_dir, "run.log")
                    if os.path.exists(rl):
                        # AGENT FIX (R69): a last-40-lines tail loses the actual refusal/
                        # exception line for a refuse-style job (the real "removed key: ..."/
                        # "[config] refused: ..." line fires early, then roslaunch's own
                        # trailing [params] dump + shutdown banner pushes it out of the last
                        # 40 lines) -- confirmed live: bootstrap_reduce.py's G6 gate failed for
                        # all 3 r71_refuse_* arms even though the full run.log (in the queue
                        # dir) does contain the expected text, just not within its own last 40
                        # lines. Same class of bug hit and fixed in R68 (run_log_tail.txt there
                        # had the identical problem). Copy the full run.log instead of a tail;
                        # these files are small (refuse jobs die at startup).
                        shutil.copy(rl, os.path.join(D, "run_log_tail.txt"))
                    status[n] = "failed_run_log_tail"
                else:
                    status[n] = "not_committed:" + (job.status() if os.path.isdir(job.state_dir) else "new")
    json.dump(status, open(os.path.join(a.runs, "_unbundle_status.json"), "w"), indent=1)
    bad = {k: v for k, v in status.items() if v not in ("ok", "failed_run_log_tail")}
    print("unbundled %d arms; not committed: %s" % (sum(1 for v in status.values() if v == "ok"), bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
