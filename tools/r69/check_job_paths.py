#!/usr/bin/env python3
"""TEMPLATE (planning agent). Pre-dispatch check: for every generated job directory, job.env OUT_DIR must equal config.yaml outputs.path (and outputs.debug_log_dir).

    python3 check_job_paths.py ablations/_job_queue/<batch> [--root-map /root/catkin_ws=/home/bblinn/fast_ws]
Prints one row per job; exit 1 if any pair differs or is unreadable. Run it before every dispatch of a new batch (rule 1.11).
job.env: lines like `OUT_DIR=/x/y`, `export OUT_DIR="/x/y"`. config.yaml: top-level `outputs:` mapping with `path` (yaml module if present, else an indentation-based fallback)."""
import os, re, sys


def read_env(p):
    v = None
    for ln in open(p):
        m = re.match(r'\s*(?:export\s+)?OUT_DIR\s*=\s*(.*?)\s*$', ln)
        if m:
            v = m.group(1).strip().strip('"').strip("'")
    return v


def read_outputs(p):
    try:
        import yaml
        d = yaml.safe_load(open(p)) or {}
        o = d.get("outputs", {}) or {}
        return o.get("path"), o.get("debug_log_dir")
    except Exception:
        pass
    path = dbg = None
    inside = False
    for ln in open(p):
        if re.match(r'^outputs\s*:', ln):
            inside = True; continue
        if inside and re.match(r'^\S', ln):
            inside = False
        if inside:
            m = re.match(r'^\s+path\s*:\s*(.*?)\s*$', ln)
            if m: path = m.group(1).strip('"').strip("'")
            m = re.match(r'^\s+debug_log_dir\s*:\s*(.*?)\s*$', ln)
            if m: dbg = m.group(1).strip('"').strip("'")
    return path, dbg


def main(argv):
    if len(argv) < 2:
        print(__doc__); return 2
    q = argv[1]
    bad = 0
    n = 0
    for jd in sorted(os.listdir(q)):
        d = os.path.join(q, jd)
        env, cfg = os.path.join(d, "job.env"), os.path.join(d, "config.yaml")
        if not (os.path.isfile(env) and os.path.isfile(cfg)):
            continue
        n += 1
        out = read_env(env)
        path, dbg = read_outputs(cfg)
        ok = out is not None and out.rstrip("/") == (path or "").rstrip("/") and (dbg is None or dbg.rstrip("/") == out.rstrip("/"))
        bad += 0 if ok else 1
        print("%s  %s  OUT_DIR=%s  outputs.path=%s  debug_log_dir=%s" % ("OK  " if ok else "DIFF", jd, out, path, dbg))
    print("jobs checked: %d, mismatches: %d" % (n, bad))
    return 1 if (bad or n == 0) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
