#!/usr/bin/env python3
"""make_broken_clone.py -- stage-1b failure-path proof helper (planning agent's, R70). Touches only the LISTENER TEST, never an estimator run.

    python3 tools/r69/make_broken_clone.py --spec r73_manifest/listener_spec_smoke.json --root . --src 00_id_baseline_60s --dst 00_id_broken_clone

Copies the FINISHED smoke job <src> (queue dir entry and out dir) to the queue `r73_smoke_broken` as <dst>, rewrites OUT_DIR in the copy's job.env, removes the
copy's `.listener/` and `listener_out/`, and deletes ONLY `state_reference.txt` in the copy's out dir (the reducer needs it, so the stage must fail and the raw
must survive). The arm-table row for <dst> exists (make_r73_manifest.py writes it), so the failure is caused by the missing file and nothing else.
Run it AFTER the dry-run pass of the smoke jobs finished and BEFORE the deleting pass (the copy needs the raw files).
Prints every path it created and the out dir the broken spec resolves to. Exit 0 only if that out dir exists and state_reference.txt is gone.
"""
import argparse, json, os, shutil, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True); ap.add_argument("--root", default=".")
    ap.add_argument("--broken-spec", default=None, help="default: listener_spec_broken.json next to --spec")
    ap.add_argument("--src", default="00_id_baseline_60s"); ap.add_argument("--dst", default="00_id_broken_clone")
    ap.add_argument("--listener-dir", default="scripts/parallel")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.root, a.listener_dir))
    import sweep_listener as sl
    spec = sl.load_spec(a.spec, a.root)
    bspec = sl.load_spec(a.broken_spec or os.path.join(os.path.dirname(os.path.abspath(a.spec)), "listener_spec_broken.json"), a.root)
    src_q = next(q for q in spec["queues"] if os.path.isdir(os.path.join(sl.absp(spec, q["dir"]), a.src)))
    sj = sl.Job(spec, src_q, a.src)
    bq = bspec["queues"][0]
    if not os.path.exists(os.path.join(sj.job_dir, "DONE")):
        sys.exit("source job has no DONE sentinel: %s" % sj.job_dir)
    dj_dir = os.path.join(sl.absp(bspec, bq["dir"]), a.dst)
    d_out = os.path.join(sl.absp(bspec, bq["results_root"]), a.dst)
    # R70: no path_maps translation of d_out (agent fix, R70): run this script via `docker exec` so /root/catkin_ws/... is the real bind-mounted path.
    for d in (dj_dir, d_out):
        if os.path.exists(d):
            sys.exit("destination exists, remove it first: " + d)
    os.makedirs(os.path.dirname(dj_dir), exist_ok=True); os.makedirs(os.path.dirname(d_out), exist_ok=True)
    shutil.copytree(sj.job_dir, dj_dir)
    shutil.copytree(sj.out_dir, d_out)
    for sub in (".listener", "listener_out"):
        shutil.rmtree(os.path.join(d_out, sub), ignore_errors=True)
    # job.env: OUT_DIR (container path) -> the copy
    env = os.path.join(dj_dir, "job.env")
    changed = 0
    if os.path.isfile(env):
        lines = []
        for line in open(env, errors="replace"):
            if line.startswith("OUT_DIR="):
                new = line.replace("/%s/" % os.path.basename(os.path.dirname(sj.out_dir)), "/r73_smoke_broken/").replace(a.src, a.dst)
                changed += int(new != line); line = new
            lines.append(line)
        open(env, "w").write("".join(lines))
    sr = os.path.join(d_out, "state_reference.txt")
    if os.path.exists(sr):
        os.remove(sr)
    bj = sl.Job(bspec, bq, a.dst)
    print("source job dir     :", sj.job_dir); print("source out dir     :", sj.out_dir)
    print("clone job dir      :", dj_dir); print("clone out dir      :", d_out)
    print("job.env OUT_DIR lines changed:", changed)
    print("broken spec resolves out dir to:", bj.out_dir)
    ok = os.path.isdir(bj.out_dir) and os.path.realpath(bj.out_dir) == os.path.realpath(d_out) and not os.path.exists(os.path.join(bj.out_dir, "state_reference.txt")) and changed == 1
    print("CLONE OK" if ok else "CLONE NOT OK (OUT_DIR rewrite or resolution failed; fix job.env by hand and say so)")
    sys.exit(0 if ok else 1)


main()
