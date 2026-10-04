#!/usr/bin/env python3
"""make_broken_clone.py -- stage-1b failure-path proof helper (planning agent's, R70; R74 addendum 3: no hardcoded batch name, config.yaml rewritten, self-check compares the literal paths). Touches only the LISTENER TEST, never an estimator run.

    python3 tools/r69/make_broken_clone.py --spec r73_manifest/listener_spec_smoke.json --root . --src 00_id_baseline_60s --dst 00_id_broken_clone

Copies the FINISHED smoke job <src> (queue dir entry and out dir) to the broken spec's queue as <dst>, rewrites the source out dir to the clone's in the copy's job.env and config.yaml, removes the
copy's `.listener/` and `listener_out/`, and deletes ONLY `state_reference.txt` in the copy's out dir (the reducer needs it, so the stage must fail and the raw
must survive). The arm-table row for <dst> exists (make_r73_manifest.py writes it), so the failure is caused by the missing file and nothing else.
Run it AFTER the dry-run pass of the smoke jobs finished and BEFORE the deleting pass (the copy needs the raw files).
Prints every path it created and the out dir the broken spec resolves to. Exit 0 only if that out dir exists and state_reference.txt is gone.
"""
import argparse, json, os, re, shutil, sys


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
    # R74 addendum 3: the clone's OUT_DIR (job.env) and outputs.path / outputs.debug_log_dir (config.yaml) are all rewritten from the SOURCE job's
    # own OUT_DIR string to the clone's out dir. (Before: job.env got a hardcoded R73 batch name and config.yaml was never rewritten, so
    # check_job_paths.py reported a mismatch that the listener's own fallback resolution hid.)
    env = os.path.join(dj_dir, "job.env")
    cfg = os.path.join(dj_dir, "config.yaml")

    def env_out(path):
        v = None
        for ln in open(path, errors="replace"):
            m = re.match(r'\s*(?:export\s+)?OUT_DIR\s*=\s*(.*?)\s*$', ln)
            if m:
                v = m.group(1).strip().strip('"').strip("'")
        return v

    def cfg_outputs(path):
        # R74 addendum 4: config.yaml is written by a YAML dumper that wraps a long value onto a second line, so a one-line regex reads it as empty
        # (addendum 3's first version did). Parse the YAML when a YAML module is present; otherwise match across the line break.
        try:
            import yaml
            o = (yaml.safe_load(open(path)) or {}).get("outputs", {}) or {}
            return o.get("path"), o.get("debug_log_dir")
        except Exception:
            pass
        txt = open(path, errors="replace").read()
        m = re.search(r'^outputs\s*:[ \t]*\n((?:[ \t]+.*\n?|\n)*)', txt, re.M)
        blk = m.group(1) if m else ""
        mp = re.search(r'(?:^|\s)path\s*:\s*(\S+)', blk)
        md = re.search(r'(?:^|\s)debug_log_dir\s*:\s*(\S+)', blk)
        return (mp.group(1).strip('"').strip("'") if mp else None), (md.group(1).strip('"').strip("'") if md else None)

    src_out = env_out(os.path.join(sj.job_dir, "job.env")) if os.path.isfile(os.path.join(sj.job_dir, "job.env")) else None
    changed = cfg_changed = 0
    if src_out and os.path.isfile(env):
        txt = open(env, errors="replace").read()
        new_txt = txt.replace(src_out, d_out)
        changed = txt.count(src_out)
        open(env, "w").write(new_txt)
    if src_out and os.path.isfile(cfg):
        txt = open(cfg, errors="replace").read()
        cfg_changed = txt.count(src_out)
        open(cfg, "w").write(txt.replace(src_out, d_out))
    sr = os.path.join(d_out, "state_reference.txt")
    if os.path.exists(sr):
        os.remove(sr)
    bj = sl.Job(bspec, bq, a.dst)
    print("source job dir     :", sj.job_dir); print("source out dir     :", sj.out_dir)
    print("clone job dir      :", dj_dir); print("clone out dir      :", d_out)
    e_out = env_out(env) if os.path.isfile(env) else None
    c_path, c_dbg = cfg_outputs(cfg) if os.path.isfile(cfg) else (None, None)
    print("job.env OUT_DIR occurrences rewritten:", changed, "; config.yaml occurrences rewritten:", cfg_changed)
    print("clone job.env OUT_DIR       :", e_out); print("clone config outputs.path   :", c_path); print("clone config debug_log_dir  :", c_dbg)
    print("broken spec resolves out dir to:", bj.out_dir)
    same = lambda x, y: x is not None and y is not None and x.rstrip("/") == y.rstrip("/")
    ok = (os.path.isdir(bj.out_dir) and os.path.realpath(bj.out_dir) == os.path.realpath(d_out) and not os.path.exists(os.path.join(bj.out_dir, "state_reference.txt"))
          and changed >= 1 and cfg_changed >= 1 and same(e_out, d_out) and same(c_path, d_out) and (c_dbg is None or same(c_dbg, d_out)))
    print("CLONE OK" if ok else "CLONE NOT OK (a path in the clone's job.env or config.yaml does not equal its out dir; stop, do not fix by hand)")
    sys.exit(0 if ok else 1)


main()
