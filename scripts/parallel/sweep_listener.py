#!/usr/bin/env python3
"""sweep_listener.py -- general-purpose job listener (planning agent, R69; design: claude/round-listener-design.md).

Waits for jobs in dispatch queues to finish, runs the processing stages a round's spec declares, validates every declared
output, runs independent verify stages against the still-present raw data, commits the outputs and summary.json, re-hashes
the raw files and ONLY THEN deletes the spec's raw-delete list. It contains no round-specific logic.

    sweep_listener.py --spec listener_spec.json [--follow] [--dry-run] [--status] [--selftest] [--root DIR]

  --follow    keep polling until every expected job is terminal (default: process what is ready now, then exit)
  --dry-run   do everything except deleting raw (step 7)
  --status    print handled / failed / pending counts, free disk, alarm; exit
  --selftest  scratch queue with one good job and deliberately broken ones; exit 0 only if the good job was consumed and every
              broken job kept its raw (and a crash after commit resumes)

Job state markers (one file each) live in <out_dir>/.listener/: seen.json staged validated verified committed consumed
failed.json attempts. Committed outputs live in <out_dir>/listener_out/<stage>/<declared path>; summary.json is
<out_dir>/listener_out/summary.json, written last. Ledger: <ledger_dir>/listener_ledger.jsonl; final: listener_final.json.
Nothing here launches or kills jobs. Python 3.6+ standard library only.
"""
import argparse, concurrent.futures as cf, csv, fnmatch, glob, gzip, hashlib, io, json, os, shutil, signal, subprocess
import sys, tempfile, threading, time, zipfile

SCHEMA = 1
TERMINAL = ("consumed", "failed_final", "skipped_failed_run", "dry_committed")


class StageError(Exception):
    pass


class Crash(Exception):
    """test hook: simulate the process dying right after commit"""


# ------------------------------------------------------------------ small helpers
def now():
    return time.time()


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def count_rows(path):
    if path.endswith((".gz", ".npz", ".zip", ".bin")):
        return None
    n = 0
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            n += b.count(b"\n")
    return n


def jwrite(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def jread(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def inside(path, roots):
    rp = os.path.realpath(path)
    for r in roots:
        rr = os.path.realpath(r)
        if rp == rr or rp.startswith(rr + os.sep):
            return True
    return False


def spec_hash(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ spec
def load_spec(path, root):
    spec = jread(path)
    if not isinstance(spec, dict):
        sys.exit("spec unreadable or not an object: %s" % path)
    for k in ("round", "queues", "stages", "raw", "summary", "limits"):
        if k not in spec:
            sys.exit("spec missing key: " + k)
    spec.setdefault("trigger", {})
    spec["trigger"].setdefault("sentinels", ["DONE", "FAILED"])
    spec["trigger"].setdefault("settle_seconds", 10)
    spec["trigger"].setdefault("settle_files", [])
    spec["trigger"].setdefault("quiet_files", ["run.log"])
    spec.setdefault("on_failed_job", "skip")
    spec.setdefault("verify", [])
    spec.setdefault("path_maps", [])
    spec["limits"].setdefault("workers", 4)
    spec["limits"].setdefault("max_failures", 20)
    spec["limits"].setdefault("max_attempts", 2)
    spec["limits"].setdefault("min_free_gb", 0)
    spec["limits"].setdefault("poll_interval_s", 5)
    spec["raw"].setdefault("sha256", "twice")
    spec["_dir"] = os.path.dirname(os.path.abspath(path))
    spec["_root"] = os.path.abspath(root or ".")
    spec["_hash"] = spec_hash({k: v for k, v in spec.items() if not k.startswith("_")})
    if spec["on_failed_job"] not in ("skip", "process", "process_keep_raw"):
        sys.exit("on_failed_job must be skip|process|process_keep_raw")
    return spec


def absp(spec, p):
    return p if os.path.isabs(p) else os.path.join(spec["_root"], p)


# ------------------------------------------------------------------ jobs
class Job:
    def __init__(self, spec, queue, name):
        self.spec, self.queue, self.name = spec, queue, name
        self.job_dir = os.path.join(absp(spec, queue["dir"]), name)
        self.out_dir = self._find_out_dir()
        self.state_dir = os.path.join(self.out_dir, ".listener")
        self.lo = os.path.join(self.out_dir, "listener_out")
        self.fp = None
        self.fp_time = 0.0

    def _find_out_dir(self):
        cands = []
        env = os.path.join(self.job_dir, "job.env")
        if os.path.isfile(env):
            for line in open(env, errors="replace"):
                if line.startswith("OUT_DIR="):
                    p = line.split("=", 1)[1].strip().strip("'\"")
                    for a, b in self.spec["path_maps"]:
                        if p.startswith(a):
                            p = b + p[len(a):]
                    cands.append(p)
        if self.queue.get("results_root"):
            cands.append(os.path.join(absp(self.spec, self.queue["results_root"]), self.name))
        for c in cands:
            if os.path.isdir(c):
                return c
        return cands[-1] if cands else self.job_dir

    def sentinel(self):
        for s in self.spec["trigger"]["sentinels"]:
            if os.path.exists(os.path.join(self.job_dir, s)):
                return s
        return None

    def marker(self, m):
        return os.path.join(self.state_dir, m)

    def has(self, m):
        return os.path.exists(self.marker(m))

    def status(self):
        for m, st in (("consumed", "consumed"), ("failed_final", "failed_final"), ("skipped", "skipped_failed_run"),
                      ("dry_committed", "dry_committed"), ("committed", "committed"), ("verified", "verified"),
                      ("validated", "validated"), ("staged", "staged"), ("seen.json", "seen")):
            if self.has(m):
                return st
        return "new"

    def fingerprint(self):
        t = self.spec["trigger"]
        items = []
        for base in (self.job_dir, self.out_dir):
            for n in list(t["settle_files"]) + list(t["quiet_files"]):
                p = os.path.join(base, n)
                if os.path.exists(p):
                    st = os.stat(p)
                    items.append((p, st.st_size, st.st_mtime))
        try:
            items.append(("out_dir", os.stat(self.out_dir).st_mtime))
        except OSError:
            pass
        return tuple(items)

    def ready(self):
        # R69 stage-1 fix: a job that already has committed outputs only needs its consume step; nothing is being written, so no settle wait
        if self.sentinel() is not None and self.status() in ("committed", "dry_committed"):
            return True
        if self.sentinel() is None:
            self.fp = None
            return False
        fp = self.fingerprint()
        t = now()
        if fp != self.fp:
            self.fp, self.fp_time = fp, t
        return (t - self.fp_time) >= float(self.spec["trigger"]["settle_seconds"])


# ------------------------------------------------------------------ raw list
def expand_raw(job):
    """Return [(abs_path, is_dir)] for the spec's raw.delete patterns, inside the job's own directories."""
    roots = [job.out_dir, job.job_dir]
    out, seen = [], set()
    for pat in job.spec["raw"]["delete"]:
        isdir = pat.startswith("dir:")
        if isdir:
            pat = pat[4:]
        for base in roots:
            for p in sorted(glob.glob(os.path.join(base, pat))):
                if not inside(p, roots) or os.path.islink(p) and not inside(os.path.realpath(p), roots):
                    continue
                if os.path.isdir(p) != isdir:
                    continue
                rp = os.path.realpath(p)
                if rp in seen:
                    continue
                seen.add(rp)
                out.append((p, isdir))
    return out


def hash_entry(path, isdir, do_hash=True):
    if isdir:
        files = []
        for dp, _, fn in os.walk(path):
            for n in sorted(fn):
                files.append(os.path.join(dp, n))
        h = hashlib.sha256()
        tot = 0
        for p in sorted(files):
            h.update(os.path.relpath(p, path).encode())
            h.update(sha256_file(p).encode() if do_hash else b"-")
            tot += os.path.getsize(p)
        return {"path": path, "dir": True, "bytes": tot, "sha256": h.hexdigest() if do_hash else None, "files": len(files)}
    return {"path": path, "dir": False, "bytes": os.path.getsize(path), "rows": count_rows(path),
            "sha256": sha256_file(path) if do_hash else None}


# ------------------------------------------------------------------ validators
def validate_output(path, d):
    """return None if fine, else an error string"""
    if not os.path.exists(path):
        return "missing"
    if os.path.getsize(path) < int(d.get("min_bytes", 1)):
        return "smaller than min_bytes"
    kind = d.get("kind", "exists")
    try:
        if kind == "json":
            obj = json.load(open(path))
            for k in d.get("required_fields", []):
                if not isinstance(obj, dict) or k not in obj or obj[k] is None:
                    return "json field missing or null: " + k
        elif kind == "csv":
            op = gzip.open if path.endswith(".gz") else open
            with op(path, "rt", newline="") as f:
                rd = csv.reader(f)
                hdr = next(rd, None)
                if hdr is None:
                    return "empty csv"
                for c in d.get("required_columns", []):
                    if c not in hdr:
                        return "csv column missing: " + c
                n = 0
                for row in rd:
                    n += 1
                    if len(row) != len(hdr):
                        return "csv row %d has %d fields, header has %d" % (n, len(row), len(hdr))
                if n < int(d.get("min_rows", 0)):
                    return "csv has %d rows, need %d" % (n, int(d.get("min_rows", 0)))
        elif kind == "gz":
            with gzip.open(path, "rb") as f:
                while f.read(1 << 22):
                    pass
        elif kind in ("zip", "npz"):
            with zipfile.ZipFile(path) as z:
                bad = z.testzip()
                if bad:
                    return "zip member corrupt: " + bad
        elif kind == "sha_manifest":
            base = os.path.dirname(path)
            for line in open(path):
                line = line.strip()
                if not line:
                    continue
                hx, rel = line.split(None, 1)
                rel = rel.strip().lstrip("*")
                p = os.path.join(base, rel)
                if not os.path.exists(p):
                    return "manifest path missing: " + rel
                if sha256_file(p) != hx:
                    return "manifest hash mismatch: " + rel
        elif kind in ("exists", "binary"):
            pass
        else:
            return "unknown validator kind: " + kind
    except Exception as e:  # any parse failure is a validation failure
        return "%s check failed: %s" % (kind, e)
    return None


# ------------------------------------------------------------------ running commands
def run_cmd(cmd, timeout, nice, cwd, logpath):
    def pre():
        os.setsid()
        if nice:
            try:
                os.nice(int(nice))
            except OSError:
                pass
    with open(logpath, "wb") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=cwd, preexec_fn=pre)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                p.kill()
            p.wait()
            raise StageError("timeout after %ss" % timeout)
    if rc != 0:
        raise StageError("exit code %d (see %s)" % (rc, logpath))


def fmt(cmd, ctx):
    out = []
    for c in cmd:
        for k, v in ctx.items():
            c = c.replace("{%s}" % k, v)
        out.append(c)
    return out


# ------------------------------------------------------------------ listener
class Listener:
    def __init__(self, spec, dry_run=False, follow=False, ledger_dir=None, crash_after_commit=()):
        self.spec, self.dry, self.follow = spec, dry_run, follow
        self.ledger_dir = ledger_dir or os.path.join(spec["_dir"], "listener_state")
        os.makedirs(self.ledger_dir, exist_ok=True)
        self.ledger_path = os.path.join(self.ledger_dir, "listener_ledger.jsonl")
        self.lock = threading.Lock()
        self.jobs = {}
        self.busy = set()
        self.failures = 0
        self.bytes_freed = 0
        self.alarm = None
        self.crash = set(crash_after_commit)
        self.min_free_seen = None
        self.crashed = set()

    # ---- ledger
    def ev(self, kind, job, **kw):
        rec = dict(t=round(now(), 3), event=kind, job=job.name if job else None, **kw)
        with self.lock:
            with open(self.ledger_path, "a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
                f.flush()
        return rec

    def log(self, msg):
        sys.stderr.write("[sweep_listener %s] %s\n" % (time.strftime("%H:%M:%S"), msg))
        sys.stderr.flush()

    # ---- discovery
    def discover(self):
        for q in self.spec["queues"]:
            qd = absp(self.spec, q["dir"])
            if not os.path.isdir(qd):
                continue
            for n in sorted(os.listdir(qd)):
                if os.path.isdir(os.path.join(qd, n)) and (q["dir"], n) not in self.jobs:
                    self.jobs[(q["dir"], n)] = Job(self.spec, q, n)

    def expected(self):
        return sum(int(q["expected_jobs"]) for q in self.spec["queues"])

    # ---- disk
    def free_gb(self):
        q = self.spec["queues"][0]
        p = absp(self.spec, q.get("results_root") or q["dir"])
        while not os.path.exists(p) and p != os.path.dirname(p):
            p = os.path.dirname(p)
        return shutil.disk_usage(p).free / 1e9

    def check_disk(self):
        f = self.free_gb()
        self.min_free_seen = f if self.min_free_seen is None else min(self.min_free_seen, f)
        if f < float(self.spec["limits"]["min_free_gb"]):
            self.alarm = "free disk %.1f GB below min_free_gb %s" % (f, self.spec["limits"]["min_free_gb"])
            open(os.path.join(self.ledger_dir, ".listener_alarm"), "w").write(self.alarm + "\n")
            self.log("FATAL " + self.alarm)
            return False
        return True

    # ---- one job
    def process(self, job):
        spec = self.spec
        os.makedirs(job.state_dir, exist_ok=True)
        st = job.status()
        failed_run = job.sentinel() == "FAILED"
        if st in TERMINAL and not (st == "dry_committed" and not self.dry):
            return
        if failed_run and spec["on_failed_job"] == "skip":
            open(job.marker("skipped"), "w").write("failed run, on_failed_job=skip\n")
            self.ev("skipped_failed_run", job)
            return
        keep_raw = failed_run and spec["on_failed_job"] != "process"
        # --- resume at consume when committed
        if st in ("committed", "dry_committed"):
            if not self.recheck_outputs(job):
                raise StageError("committed outputs no longer match summary.json; not deleting raw")
            return self.consume(job, keep_raw)
        # --- seen (restart from here for anything earlier)
        for m in ("staged", "validated", "verified"):
            if os.path.exists(job.marker(m)):
                os.remove(job.marker(m))
        shutil.rmtree(os.path.join(job.state_dir, "tmp"), ignore_errors=True)
        raw = expand_raw(job)
        do_hash = spec["raw"]["sha256"] != "none"
        seen = [hash_entry(p, d, do_hash) for p, d in raw]
        jwrite(job.marker("seen.json"), {"spec_hash": spec["_hash"], "raw": seen})
        self.ev("raw_seen", job, n=len(seen), bytes=sum(s["bytes"] for s in seen))
        ctx_base = dict(job_dir=job.job_dir, out_dir=job.out_dir, job_id=job.name, round=str(spec["round"]),
                        spec_dir=spec["_dir"])
        stage_dirs, stage_summaries = {}, {}
        # --- stages
        for sg in spec["stages"]:
            sdir = os.path.join(job.state_dir, "tmp", sg["name"])
            os.makedirs(sdir, exist_ok=True)
            stage_dirs[sg["name"]] = sdir
            ctx = dict(ctx_base, stage_out=sdir)
            for k, v in stage_dirs.items():
                ctx["stage_out_of:" + k] = v
            t0 = now()
            run_cmd(fmt(sg["cmd"], ctx), sg.get("timeout_s", 600), sg.get("nice", 0), spec["_root"],
                    os.path.join(job.state_dir, "tmp", sg["name"] + ".log"))
            self.ev("staged", job, stage=sg["name"], seconds=round(now() - t0, 2))
        open(job.marker("staged"), "w").write("ok\n")
        # --- validate
        outs = []
        for sg in spec["stages"]:
            for d in sg.get("outputs", []):
                p = os.path.join(stage_dirs[sg["name"]], d["path"])
                err = validate_output(p, d)
                if err:
                    raise StageError("validate %s/%s: %s" % (sg["name"], d["path"], err))
                outs.append((sg["name"], d, p))
                if d.get("kind") == "json" and d.get("merge_into_summary"):
                    stage_summaries[sg["name"]] = jread(p, {})
        open(job.marker("validated"), "w").write("ok\n")
        self.ev("validated", job, outputs=len(outs))
        # --- verify (raw still present)
        for vf in spec["verify"]:
            ctx = dict(ctx_base, stage_out=os.path.join(job.state_dir, "tmp", "verify_" + vf["name"]))
            os.makedirs(ctx["stage_out"], exist_ok=True)
            for k, v in stage_dirs.items():
                ctx["stage_out_of:" + k] = v
            run_cmd(fmt(vf["cmd"], ctx), vf.get("timeout_s", 600), vf.get("nice", 0), spec["_root"],
                    os.path.join(job.state_dir, "tmp", "verify_" + vf["name"] + ".log"))
            self.ev("verified", job, verify=vf["name"])
        open(job.marker("verified"), "w").write("ok\n")
        # --- commit
        os.makedirs(job.lo, exist_ok=True)
        osums = []
        for sname, d, p in outs:
            dst = os.path.join(job.lo, sname, d["path"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(p, dst)
            osums.append({"stage": sname, "path": os.path.join(sname, d["path"]), "bytes": os.path.getsize(dst),
                          "sha256": sha256_file(dst)})
        for vf in spec["verify"]:   # keep the verify logs: they are the control numbers
            lp = os.path.join(job.state_dir, "tmp", "verify_" + vf["name"] + ".log")
            if os.path.exists(lp):
                shutil.copy(lp, os.path.join(job.lo, "verify_" + vf["name"] + ".log"))
        summary = {}
        for sname in sorted(stage_summaries):
            for k, v in stage_summaries[sname].items():
                summary[k] = v
        summary["schema_version"] = spec["summary"]["schema_version"]
        summary["listener"] = {"spec_hash": spec["_hash"], "round": spec["round"], "job": job.name,
                               "raw": seen, "outputs": osums, "dry_run": bool(self.dry or keep_raw)}
        for k in spec["summary"].get("required_fields", []):
            if k not in summary or summary[k] is None:
                raise StageError("summary.json required field missing: " + k)
        jwrite(os.path.join(job.lo, "summary.json"), summary)   # LAST
        open(job.marker("committed"), "w").write("ok\n")
        self.ev("committed", job, outputs=len(osums))
        if job.name in self.crash:
            self.crash.discard(job.name)
            raise Crash(job.name)
        return self.consume(job, keep_raw)

    def recheck_outputs(self, job):
        s = jread(os.path.join(job.lo, "summary.json"))
        if not s or "listener" not in s or s["listener"].get("spec_hash") != self.spec["_hash"]:
            return False
        for o in s["listener"]["outputs"]:
            p = os.path.join(job.lo, o["path"])
            if not os.path.exists(p) or sha256_file(p) != o["sha256"]:
                return False
        return True

    def consume(self, job, keep_raw):
        if self.dry or keep_raw:
            open(job.marker("dry_committed"), "w").write("raw kept (dry-run or failed run)\n")
            self.ev("dry_committed", job)
            return
        seen = jread(job.marker("seen.json"), {}).get("raw", [])
        seen_paths = {os.path.realpath(s["path"]): s for s in seen}
        current = expand_raw(job)
        # --- re-hash every raw file against its seen hash
        for s in seen:
            if not os.path.exists(s["path"]):
                continue   # already gone (resume after partial delete)
            h = hash_entry(s["path"], s["dir"], s["sha256"] is not None)
            if s["sha256"] is not None and h["sha256"] != s["sha256"]:
                raise StageError("raw changed between seen and consume: " + s["path"])
        todo = [(p, d) for p, d in current if os.path.realpath(p) in seen_paths]
        unseen = [p for p, d in current if os.path.realpath(p) not in seen_paths]
        if unseen:
            self.ev("unseen_raw_kept", job, paths=unseen)
        plan = [{"path": p, "dir": d, "bytes": hash_entry(p, d, False)["bytes"]} for p, d in todo]
        self.ev("delete_plan", job, plan=plan)
        freed = 0
        for p, d in todo:
            freed += hash_entry(p, d, False)["bytes"]
            if d:
                shutil.rmtree(p)
            else:
                os.remove(p)
        with self.lock:
            self.bytes_freed += freed
        open(job.marker("consumed"), "w").write("freed %d\n" % freed)
        for m in ("dry_committed",):
            if os.path.exists(job.marker(m)):
                os.remove(job.marker(m))
        shutil.rmtree(os.path.join(job.state_dir, "tmp"), ignore_errors=True)
        self.ev("consumed", job, bytes_freed=freed)

    def run_one(self, job):
        key = job.name
        try:
            self.process(job)
        except Crash:
            self.crashed.add(key)
            self.log("simulated crash after commit: " + key)
        except Exception as e:
            n = int((jread(job.marker("attempts"), {}) or {}).get("n", 0)) + 1
            jwrite(job.marker("attempts"), {"n": n, "last_error": str(e)})
            self.ev("failed", job, error=str(e), attempt=n)
            self.log("FAILED %s attempt %d: %s" % (key, n, e))
            if n >= int(self.spec["limits"]["max_attempts"]):
                open(job.marker("failed_final"), "w").write(str(e) + "\n")
                with self.lock:
                    self.failures += 1
        finally:
            with self.lock:
                self.busy.discard((job.queue["dir"], job.name))

    # ---- main loop
    def counts(self):
        c = {}
        for j in self.jobs.values():
            s = j.status() if os.path.isdir(j.state_dir) else "new"
            c[s] = c.get(s, 0) + 1
        return c

    def is_term(self, st):
        # a dry-run result is terminal only in a dry-run pass; a deleting pass must still consume it
        return st in TERMINAL and not (st == "dry_committed" and not self.dry)

    def terminal_count(self):
        return sum(1 for j in self.jobs.values() if os.path.isdir(j.state_dir) and self.is_term(j.status()))

    def loop(self):
        lim = self.spec["limits"]
        ex = cf.ThreadPoolExecutor(max_workers=int(lim["workers"]))
        futs = []
        try:
            while True:
                if not self.check_disk():
                    break
                self.discover()
                if self.failures > int(lim["max_failures"]):
                    self.alarm = "more than max_failures jobs failed"
                    self.log("FATAL " + self.alarm)
                    break
                launched = 0
                pending_settle = False
                for k, j in self.jobs.items():
                    if k in self.busy:
                        continue
                    st = j.status() if os.path.isdir(j.state_dir) else "new"
                    if st in TERMINAL and not (st == "dry_committed" and not self.dry):
                        continue
                    if st == "failed_final":
                        continue
                    if k in self.crashed:
                        continue
                    if not j.ready():
                        if j.sentinel() is not None:
                            pending_settle = True   # finished but still inside its settle window: a non-follow pass must wait, not exit
                        continue
                    self.busy.add(k)
                    futs.append(ex.submit(self.run_one, j))
                    launched += 1
                futs = [f for f in futs if not f.done()]
                done = self.terminal_count()
                if done >= self.expected() and not futs:
                    break
                if not self.follow and not futs and launched == 0 and not pending_settle:
                    break
                time.sleep(float(lim["poll_interval_s"]) if (self.follow or futs or pending_settle) else 0)
        finally:
            ex.shutdown(wait=True)
        return self.finish()

    def finish(self):
        self.discover()
        nonterm, left = [], []
        for j in self.jobs.values():
            s = j.status() if os.path.isdir(j.state_dir) else "new"
            if not self.is_term(s):
                nonterm.append({"job": j.name, "state": s})
            if s in ("failed_final", "skipped_failed_run"):
                left.append(j.name)
        unscored = [j.name for j in self.jobs.values()
                    if (not os.path.isdir(j.state_dir) or j.status() not in (("consumed", "dry_committed", "skipped_failed_run") if self.dry else ("consumed", "skipped_failed_run")))
                    and expand_raw(j)]
        fin = {"round": self.spec["round"], "spec_hash": self.spec["_hash"], "counts": self.counts(),
               "expected": self.expected(), "bytes_freed": self.bytes_freed, "min_free_gb_seen": self.min_free_seen,
               "alarm": self.alarm, "failures": self.failures, "nonterminal": nonterm,
               "unscored_with_raw": unscored, "dry_run": self.dry}
        jwrite(os.path.join(self.ledger_dir, "listener_final.json"), fin)
        bad = bool(self.alarm) or self.failures > 0 or bool(unscored) or len(nonterm) > 0
        self.log("final: %s" % json.dumps(fin["counts"], sort_keys=True))
        return 1 if bad else 0

    def status_text(self):
        self.discover()
        free = self.free_gb()
        alarm = os.path.exists(os.path.join(self.ledger_dir, ".listener_alarm"))
        return "round %s  jobs seen %d / expected %d  states %s  free_disk_gb %.1f  alarm %s" % (
            self.spec["round"], len(self.jobs), self.expected(), json.dumps(self.counts(), sort_keys=True), free, alarm)


# ------------------------------------------------------------------ selftest
SELFTEST_STAGE = r'''
import csv, gzip, json, os, sys, time
job_dir, out_dir, stage_out = sys.argv[1:4]
mode = open(os.path.join(job_dir, "mode.txt")).read().strip()
raw = os.path.join(out_dir, "raw.csv")
if mode == "stage_exit1":
    sys.exit(1)
if mode == "timeout":
    time.sleep(60)
if mode == "raw_changed":
    open(raw, "a").write("9,9\n")        # the raw file changes after listener hashed it
rows = list(csv.reader(open(raw)))
with open(os.path.join(stage_out, "table.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["a"] if mode == "csv_missing_column" else ["a", "b"])
    for r in rows[1:]:
        w.writerow(r[:1] if mode == "csv_missing_column" else r)
json.dump({"n": len(rows) - 1, "ok": True}, open(os.path.join(stage_out, "s.json"), "w"))
if mode != "missing_output":
    with gzip.open(os.path.join(stage_out, "archive.csv.gz"), "wt") as g:
        g.write(open(raw).read())
'''

SELFTEST_VERIFY = r'''
import gzip, os, sys
raw, arch = sys.argv[1:3]
a = gzip.open(arch, "rt").read()
b = open(raw).read()
if a != b:
    sys.exit("archive does not reproduce raw")
print("archive roundtrip identical, %d bytes" % len(b))
'''


def selftest():
    base = tempfile.mkdtemp(prefix="listener_selftest_")
    sys.stderr.write("selftest scratch dir: %s\n" % base)
    open(os.path.join(base, "stage.py"), "w").write(SELFTEST_STAGE)
    open(os.path.join(base, "verify.py"), "w").write(SELFTEST_VERIFY)
    qdir, rdir = os.path.join(base, "queue"), os.path.join(base, "results")
    modes = ["good", "good_resume", "stage_exit1", "missing_output", "csv_missing_column", "raw_changed", "timeout",
             "failed_run"]
    for m in modes:
        jd, od = os.path.join(qdir, m), os.path.join(rdir, m)
        os.makedirs(jd), os.makedirs(od)
        open(os.path.join(jd, "mode.txt"), "w").write(m if m not in ("good_resume",) else "good")
        open(os.path.join(jd, "job.env"), "w").write("OUT_DIR=%s\n" % od)
        with open(os.path.join(od, "raw.csv"), "w") as f:
            f.write("a,b\n" + "".join("%d,%d\n" % (i, i * i) for i in range(200)))
        open(os.path.join(od, "other.txt"), "w").write("not on the delete list\n")
        open(os.path.join(jd, "run.log"), "w").write("log\n")
        open(os.path.join(jd, "FAILED" if m == "failed_run" else "DONE"), "w").write("")
    spec = {
        "spec_version": 1, "round": "selftest",
        "queues": [{"dir": qdir, "results_root": rdir, "expected_jobs": len(modes)}],
        "trigger": {"sentinels": ["DONE", "FAILED"], "settle_seconds": 1, "settle_files": ["raw.csv"], "quiet_files": ["run.log"]},
        "on_failed_job": "skip",
        "stages": [{"name": "reduce", "cmd": [sys.executable, os.path.join(base, "stage.py"), "{job_dir}", "{out_dir}", "{stage_out}"],
                    "timeout_s": 3,
                    "outputs": [{"path": "s.json", "kind": "json", "required_fields": ["n", "ok"], "merge_into_summary": True},
                                {"path": "table.csv", "kind": "csv", "required_columns": ["a", "b"], "min_rows": 10},
                                {"path": "archive.csv.gz", "kind": "gz", "min_bytes": 10}]}],
        "verify": [{"name": "roundtrip", "cmd": [sys.executable, os.path.join(base, "verify.py"), "{out_dir}/raw.csv",
                                                   "{stage_out_of:reduce}/archive.csv.gz"], "timeout_s": 20}],
        "summary": {"schema_version": 1, "required_fields": ["schema_version", "n", "listener"]},
        "raw": {"delete": ["raw.csv"], "sha256": "twice"},
        "limits": {"workers": 3, "max_failures": 50, "max_attempts": 1, "min_free_gb": 0, "poll_interval_s": 0.5},
    }
    sp = os.path.join(base, "listener_spec.json")
    json.dump(spec, open(sp, "w"))
    sp_loaded = load_spec(sp, base)
    L = Listener(sp_loaded, dry_run=False, follow=False, ledger_dir=os.path.join(base, "state"), crash_after_commit=["good_resume"])
    rc1 = L.loop()
    # second pass: the crashed job must resume from `committed` and be consumed
    L2 = Listener(load_spec(sp, base), dry_run=False, follow=False, ledger_dir=os.path.join(base, "state"))
    rc2 = L2.loop()
    res, ok = [], True
    for m in modes:
        raw_there = os.path.exists(os.path.join(rdir, m, "raw.csv"))
        other_there = os.path.exists(os.path.join(rdir, m, "other.txt"))
        committed = os.path.exists(os.path.join(rdir, m, "listener_out", "summary.json"))
        want_deleted = m in ("good", "good_resume")
        good = (raw_there != want_deleted) and other_there and (committed == want_deleted or m in ("raw_changed",))
        if m == "raw_changed":
            good = raw_there and other_there   # raw must survive; its summary may exist but must not be consumed
            good = good and not os.path.exists(os.path.join(rdir, m, ".listener", "consumed"))
        ok = ok and good
        res.append((m, "raw kept" if raw_there else "raw deleted", "summary" if committed else "no summary", "PASS" if good else "FAIL"))
    sys.stdout.write("%-20s %-10s %-11s %s\n" % ("job", "raw", "summary", "result"))
    for r in res:
        sys.stdout.write("%-20s %-10s %-11s %s\n" % r)
    sys.stdout.write("first pass rc %d (nonzero expected: broken jobs), second pass rc %d\n" % (rc1, rc2))
    sys.stdout.write("SELFTEST %s\n" % ("PASS" if ok and rc1 != 0 else "FAIL"))
    return 0 if ok and rc1 != 0 else 1


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec")
    ap.add_argument("--root", default=".", help="directory relative paths in the spec resolve against")
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ledger-dir", default=None)
    ap.add_argument("--min-free-gb", type=float, default=None, help="override spec limits.min_free_gb")
    ap.add_argument("--workers", type=int, default=None, help="override spec limits.workers")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.spec:
        ap.error("--spec required")
    spec = load_spec(a.spec, a.root)
    if a.min_free_gb is not None:
        spec["limits"]["min_free_gb"] = a.min_free_gb
    if a.workers is not None:
        spec["limits"]["workers"] = a.workers
    L = Listener(spec, dry_run=a.dry_run, follow=a.follow, ledger_dir=a.ledger_dir)
    if a.status:
        print(L.status_text())
        sys.exit(0)
    L.log("start: round %s spec_hash %s dry_run %s follow %s" % (spec["round"], spec["_hash"], a.dry_run, a.follow))
    sys.exit(L.loop())


if __name__ == "__main__":
    main()
