#!/usr/bin/env python3
"""verify_archive.py -- listener verify stage (planning agent's, R69). Run BEFORE the raw files are deleted.

    python3 tools/r69/verify_archive.py --job-dir J --out-dir O --archive-dir <stage_out_of:reduce>/archive --spec listener_spec.json

For every file the spec will delete (raw.delete) that exists: the archive must hold <name>.gz, and gunzip(<name>.gz) must reproduce the
raw file bit for bit (sha256 and byte count), and archive_index.json must agree. A raw.delete entry that is a glob or a dir: with
matching files is refused (the scorer only archives flat names). Prints one line per file (the control numbers returned to the planner).
Exit 0 only if every file passes.
"""
import argparse, glob, gzip, hashlib, json, os, sys

ap = argparse.ArgumentParser()
for k in ("--job-dir", "--out-dir", "--archive-dir", "--spec"):
    ap.add_argument(k, required=True)
a = ap.parse_args()
spec = json.load(open(a.spec))
roots = [a.out_dir, a.job_dir]
idx = {e["name"]: e for e in json.load(open(os.path.join(a.archive_dir, "archive_index.json")))}
bad, n = [], 0
for pat in spec["raw"]["delete"]:
    flat = not pat.startswith("dir:") and not any(c in pat for c in "*?[")
    hits = []
    for r in roots:
        hits += glob.glob(os.path.join(r, pat[4:] if pat.startswith("dir:") else pat))
    if not flat:
        if hits:
            bad.append("pattern %s matches files but the scorer archives flat names only" % pat)
        continue
    raw = next((h for h in hits if os.path.isfile(h)), None)
    if raw is None:
        continue
    gz = os.path.join(a.archive_dir, pat + ".gz")
    if not os.path.exists(gz):
        bad.append("%s: no archive" % pat); continue
    h, tot = hashlib.sha256(), 0
    with gzip.open(gz, "rb") as g:
        for b in iter(lambda: g.read(1 << 22), b""):
            h.update(b); tot += len(b)
    hr = hashlib.sha256()
    with open(raw, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            hr.update(b)
    e = idx.get(pat)
    ok = h.hexdigest() == hr.hexdigest() and tot == os.path.getsize(raw) and e and e["raw_sha256"] == hr.hexdigest()
    print("%-45s raw %12d B  gz %11d B  ratio %6.2f  sha256 %s  %s" % (pat, os.path.getsize(raw), os.path.getsize(gz),
          os.path.getsize(raw) / max(1, os.path.getsize(gz)), hr.hexdigest()[:16], "IDENTICAL" if ok else "MISMATCH"))
    n += 1
    if not ok:
        bad.append("%s: archive does not reproduce raw" % pat)
print("files verified: %d" % n)
if bad:
    sys.exit("VERIFY FAILED: " + "; ".join(bad))
print("VERIFY OK")
