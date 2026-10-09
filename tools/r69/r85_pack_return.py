#!/usr/bin/env python3
"""R85: pack the return (from r84_pack_return.py, itself from r83_pack_return.py; rule 5.12b of the planner, Bryce 2026-10-07). Planning agent's script; selftest on a synthetic folder only (planner sandbox, not evidence).

    python3 tools/r69/r85_pack_return.py --dir R85_RETURN --required r85_manifest/required_outputs.json --name R85_RETURN_<date> [--arm-table r85_manifest/arm_table.tsv] [--part-mb 28] [--standalone <file> ...] [--workdir <dir>]
    python3 tools/r69/r85_pack_return.py --verify <folder holding <name>.zip.NN and <name>.sha256, or a reassembled <name>.zip> --name R85_RETURN_<date> [--keep-zip]      # what the planner / a receiver runs
    python3 tools/r69/r85_pack_return.py --selftest

What it does, in this order (it stops with a non-zero exit at the first failed step and says which):
 1. REQUIRED: every file of required_outputs.json is resolved (where == "return", source == "arm" -> <dir>/compact/<arm>/<file>; "return" + "generated" -> <dir>/<file>; where == "archive" -> <dir>/archive/<arm>/<file>)
    and must exist and be non-empty. Also required: r85_gates.txt. Any miss -> REFUSED, every missing path named (exit 2), nothing is packed.
 2. DUPLICATES: sha256 of the DECOMPRESSED content of compact/<arm>/pos_trace_all.csv.gz for every arm that has one. Arms with the same hash are written to <dir>/duplicates.tsv (columns arm, dup_of, notes;
    rows already there are kept). A group is `expected` when --arm-table says one member is a determinism repeat (kind starts with det) of another member; every other group is `UNEXPECTED` and is printed.
 3. LEDGER: <dir>/return_ledger.tsv = relative path, bytes, sha256 of EVERY other file in the folder (the ledger and the duplicates file are written before it is computed; the ledger lists itself as `(this file)`).
    The ledger is INSIDE the zip. It is the only integrity record of the zip's content.
 4. ZIP the folder (stored for .gz/.zip members, deflated otherwise). Rule 5.12b: the zip is split ONLY when it is larger than one part (--part-mb, default 28 = 28 MiB, the unit of `split -b 28M`): then with the exact rule of
    `split -b 28M -d -a 2 <name>.zip <name>.zip.` (suffixes .00 .01 ...) and the whole zip is removed. A zip that fits in one part is handed over as the single <name>.zip.
 5. HASH FILE: exactly ONE text file <name>.sha256 is written next to the parts (format of `sha256sum -c`): one line per part (or the one line of the single zip) and one line per --standalone file. NO whole-zip hash file
    is written anywhere (no <name>.zip.sha256, no per-part .sha256, nothing inside the zip). The .sha256 file is never inside the zip.
 6. PROOF (printed, nothing written): the parts are joined in suffix order into a scratch zip (gaps in the suffix sequence fail), the scratch zip is checked against the ledger inside it, and a join with one part dropped MUST fail
    that check. Scratch files are deleted.
 7. OUTPUT LISTING: every file handed over, with name and size in bytes (parts, <name>.sha256, standalone files). The agent pastes this list into its final message.
The tool never deletes anything except the single whole zip it replaced by its parts and its own scratch files.

--verify (receiver side): given a folder with the parts (and <name>.sha256), the parts are checked against the lines of <name>.sha256 (when it is present), joined in suffix order, and the reassembled zip is checked against
return_ledger.tsv inside it (every listed file present with the listed bytes and sha256, no file in the zip that the ledger does not list). Given a .zip file, only the ledger check runs. There is no whole-zip hash to compare.
Exit 0 only when every step passes.
"""
import argparse, csv, glob, gzip, hashlib, io, json, os, re, shutil, subprocess, sys, tempfile, zipfile

MIB = 1024 * 1024
GATES = "r85_gates.txt"
LEDGER = "return_ledger.tsv"


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_parts(paths):
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
    return h.hexdigest()


def required_paths(root, req):
    out = []
    for r in req:
        if r["where"] == "return" and r["source"] == "arm":
            base = [os.path.join(root, "compact", a, r["file"]) for a in r["arms"]]
        elif r["where"] == "return":
            base = [os.path.join(root, r["file"])]
        else:
            base = [os.path.join(root, "archive", a, r["file"]) for a in r["arms"]]
        out += [(p, r["followup"], r["title"]) for p in base]
    return out


def csv_rows(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="", errors="replace") as f:
        return max(sum(1 for _ in f) - 1, 0)


def find_missing(root, req):
    miss = [(p, fu, t) for p, fu, t in required_paths(root, req) if not os.path.isfile(p) or os.path.getsize(p) == 0]
    g = os.path.join(root, GATES)
    if not os.path.isfile(g) or os.path.getsize(g) == 0:
        if not any(p == g for p, _, _ in miss):
            miss.append((g, "R85", "gates file"))
    return miss


# ---------------------------------------------------------------- duplicates
def pos_hash(path):
    h = hashlib.sha256(); n = 0
    with gzip.open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b); n += b.count(b"\n")
    return h.hexdigest(), max(n - 1, 0)


def det_pairs(arm_table):
    """set of frozenset({arm, ref}) for rows whose kind starts with det and that name a ref."""
    out = set()
    if arm_table and os.path.exists(arm_table):
        with open(arm_table, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if str(r.get("arm_kind") or r.get("kind", "")).startswith("det") and r.get("ref"):
                    out.add(frozenset((r["arm"], r["ref"])))
    return out


def find_duplicates(root, arm_table=None):
    """-> (groups: list of lists of arms with identical decompressed pos_trace, not_comparable arms, n compared)."""
    byh = {}; nocmp = []
    cdir = os.path.join(root, "compact")
    arms = sorted(d for d in os.listdir(cdir) if os.path.isdir(os.path.join(cdir, d))) if os.path.isdir(cdir) else []
    for a in arms:
        p = os.path.join(cdir, a, "pos_trace_all.csv.gz")
        if not os.path.isfile(p):
            nocmp.append(a); continue
        h, n = pos_hash(p)
        byh.setdefault((h, n), []).append(a)
    groups = [sorted(v) for v in byh.values() if len(v) > 1]
    return sorted(groups), nocmp, sum(len(v) for v in byh.values())


def write_duplicates(root, arm_table=None):
    groups, nocmp, ncmp = find_duplicates(root, arm_table)
    pairs = det_pairs(arm_table)
    p = os.path.join(root, "duplicates.tsv")
    old = []
    if os.path.exists(p):
        with open(p, newline="") as f:
            rd = csv.reader(f, delimiter="\t")
            hdr = next(rd, None)
            old = [r for r in rd if r]
    have = {r[0] for r in old}
    rows = list(old); unexpected = []
    for g in groups:
        canon = g[0]
        exp = any(frozenset((x, y)) in pairs for i, x in enumerate(g) for y in g[i + 1:]) and len(g) == 2
        for a in g[1:]:
            if a in have:
                continue
            rows.append([a, canon, "%s: identical pos_trace_all.csv.gz content (sha256 of the decompressed file) to %s" % ("expected (determinism repeat)" if exp else "UNEXPECTED", canon)])
        if not exp:
            unexpected.append(g)
    with open(p, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["arm", "dup_of", "notes"]); w.writerows(rows)
    return groups, unexpected, nocmp, ncmp


# ---------------------------------------------------------------- ledger, zip, split
def write_ledger(root):
    lp = os.path.join(root, LEDGER)
    open(lp, "w").write("")
    rows = []
    for d, _, fs in os.walk(root):
        for f in sorted(fs):
            fp = os.path.join(d, f)
            rel = os.path.relpath(fp, root).replace(os.sep, "/")
            if fp == lp:
                continue
            rows.append((rel, os.path.getsize(fp), sha_file(fp)))
    rows.sort()
    with open(lp, "w") as f:
        f.write("path\tbytes\tsha256\n")
        for r in rows:
            f.write("%s\t%d\t%s\n" % r)
        f.write("%s\t-\t(this file)\n" % LEDGER)
    return rows


def make_zip(root, zpath):
    base = os.path.dirname(os.path.abspath(root.rstrip("/")))
    with zipfile.ZipFile(zpath, "w", allowZip64=True) as z:
        for d, _, fs in os.walk(root):
            for f in sorted(fs):
                fp = os.path.join(d, f)
                ct = zipfile.ZIP_STORED if f.lower().endswith((".gz", ".zip")) else zipfile.ZIP_DEFLATED
                z.write(fp, os.path.relpath(os.path.abspath(fp), base), compress_type=ct)
    return os.path.getsize(zpath)


def split_like_coreutils(zpath, part_bytes):
    """`split -b <part_bytes> -d -a 2 zpath zpath.` -> list of part paths (suffix 00, 01, ...). More than 100 parts is refused (-a 2)."""
    parts = []
    with open(zpath, "rb") as f:
        i = 0
        while True:
            b = f.read(part_bytes)
            if not b:
                break
            if i > 99:
                raise SystemExit("more than 100 parts with -a 2: raise --part-mb")
            p = "%s.%02d" % (zpath, i)
            with open(p, "wb") as o:
                o.write(b)
            parts.append(p); i += 1
    return parts


def write_sha_file(directory, name, files):
    """THE one hash file <directory>/<name>.sha256: `<sha256>  <basename>` per file of `files` (the parts, or the single zip, and the standalone files). -> path."""
    p = os.path.join(directory, name + ".sha256")
    with open(p, "w") as f:
        for fp in files:
            f.write("%s  %s\n" % (sha_file(fp), os.path.basename(fp)))
    return p


def sha_check(shafile, directory):
    """`sha256sum -c` of the .sha256 file inside `directory`: True when every listed file exists and hashes to its line. Returns (ok, message)."""
    msgs = []; ok = True
    for ln in open(shafile).read().splitlines():
        m = re.match(r"^([0-9a-f]{64})  (.+)$", ln)
        if not m:
            ok = False; msgs.append("bad line %r" % ln[:60]); continue
        fp = os.path.join(directory, m.group(2))
        if not os.path.isfile(fp):
            ok = False; msgs.append("%s: file missing" % m.group(2))
        elif sha_file(fp) != m.group(1):
            ok = False; msgs.append("%s: FAILED" % m.group(2))
    return ok, "; ".join(msgs) or "OK"


def part_paths(directory, name):
    """the parts of <name>.zip in suffix order; raises ValueError when the suffix sequence has a gap or does not start at 00."""
    ps = sorted(glob.glob(os.path.join(directory, name + ".zip.[0-9][0-9]")))
    idx = [int(p[-2:]) for p in ps]
    if idx != list(range(len(idx))):
        raise ValueError("part suffixes %s are not 00..%02d without a gap" % (idx, len(idx) - 1))
    return ps


def join_to(parts, out):
    with open(out, "wb") as o:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, o, 1 << 20)
    return out


def ledger_check(zpath):
    """the receiver's check of a (reassembled) zip against return_ledger.tsv INSIDE it. -> (ok, lines). Every ledger row: file present in the zip, bytes equal, sha256 equal (the read also checks the member CRC);
    every zip member other than the ledger must be listed. Any exception while reading (a flipped byte can break the zip structure itself) is a FAIL."""
    L = []
    try:
        with zipfile.ZipFile(zpath) as z:
            names = [i.filename for i in z.infolist() if not i.filename.endswith("/")]
            led = [n for n in names if n == LEDGER or n.endswith("/" + LEDGER)]
            if len(led) != 1:
                return False, ["FAIL %s: %d ledger files in the zip (need exactly one)" % (os.path.basename(zpath), len(led))]
            prefix = led[0][:-len(LEDGER)]
            rows = list(csv.reader(io.StringIO(z.read(led[0]).decode("utf-8", "replace")), delimiter="\t"))
            hdr, rows = rows[0], [r for r in rows[1:] if r]
            if hdr[:3] != ["path", "bytes", "sha256"]:
                return False, ["FAIL ledger header %r" % hdr]
            listed = set(); bad = []; n_ok = 0
            for path, nbytes, h in rows:
                if h == "(this file)":
                    continue
                listed.add(prefix + path)
                if prefix + path not in names:
                    bad.append("%s: not in the zip" % path); continue
                try:
                    hh = hashlib.sha256(); n = 0
                    with z.open(prefix + path) as f:
                        for b in iter(lambda: f.read(1 << 20), b""):
                            hh.update(b); n += len(b)
                except Exception as e:                       # BadZipFile (CRC), zlib.error, ...
                    bad.append("%s: unreadable (%s)" % (path, str(e)[:60])); continue
                if str(n) != nbytes:
                    bad.append("%s: %d bytes, ledger %s" % (path, n, nbytes))
                elif hh.hexdigest() != h:
                    bad.append("%s: sha256 differs from the ledger" % path)
                else:
                    n_ok += 1
            extra = [n[len(prefix):] for n in names if n != led[0] and n not in listed]
            for x in extra:
                bad.append("%s: in the zip, not in the ledger" % x)
    except Exception as e:
        return False, ["FAIL %s: zip unreadable (%s)" % (os.path.basename(zpath), str(e)[:80])]
    ok = not bad
    L.append("%s ledger check of %s: %d files match, %d problem(s)" % ("PASS" if ok else "FAIL", os.path.basename(zpath), n_ok, len(bad)))
    L += ["  " + b for b in bad[:20]] + (["  ... %d more" % (len(bad) - 20)] if len(bad) > 20 else [])
    return ok, L


def verify(path, name, keep_zip=False):
    """receiver side. path = a folder (parts + <name>.sha256) or a .zip file. -> (ok, lines)."""
    L = []; ok = True
    if os.path.isfile(path):
        o, l = ledger_check(path)
        return o, l
    try:
        ps = part_paths(path, name)
    except ValueError as e:
        return False, ["FAIL parts: %s" % e]
    single = os.path.join(path, name + ".zip")
    files = ps or ([single] if os.path.isfile(single) else [])
    if not files:
        return False, ["FAIL neither parts nor a single zip found for %s in %s" % (name, path)]
    sf = os.path.join(path, name + ".sha256")
    if os.path.isfile(sf):
        g, m = sha_check(sf, path)
        L.append("%s %s (per-part hashes): %s" % ("PASS" if g else "FAIL", os.path.basename(sf), m)); ok &= g
        listed = {ln.split(None, 1)[1].strip() for ln in open(sf).read().splitlines() if len(ln.split(None, 1)) == 2}
        unl = [os.path.basename(p) for p in files if os.path.basename(p) not in listed]
        if unl:
            L.append("FAIL part(s) not listed in %s: %s" % (os.path.basename(sf), ", ".join(unl))); ok = False
    else:
        L.append("INFO %s not found: the part hashes were not checked (the ledger check below still decides)" % (name + ".sha256"))
    tmp = tempfile.mkdtemp(prefix="r85_verify_")
    try:
        z = join_to(files, os.path.join(tmp, name + ".zip"))
        o, l = ledger_check(z)
        L += l; ok &= o
        if keep_zip and o:
            shutil.copy(z, os.path.join(path, name + ".zip")); L.append("INFO reassembled zip kept: %s" % os.path.join(path, name + ".zip"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok, L


def pack(root, req_path, name, part_mb=28.0, limit_mb=None, workdir=None, arm_table=None, standalone=(), quiet=False):
    say = (lambda *a: None) if quiet else print
    req = json.load(open(req_path))
    miss = find_missing(root, req)
    if miss:
        say("REFUSED: %d required file(s) missing, empty or short:" % len(miss))
        for p, fu, t in miss[:80]:
            say("  %s  [%s %s]" % (p, fu, t))
        return 2
    groups, unexpected, nocmp, ncmp = write_duplicates(root, arm_table)
    say("duplicates.tsv: compared %d pos traces; %d identical group(s), %d UNEXPECTED; %d arm(s) with no pos trace (not comparable)" % (ncmp, len(groups), len(unexpected), len(nocmp)))
    for g in unexpected:
        say("  UNEXPECTED duplicate group: %s" % ", ".join(g))
    write_ledger(root)
    wd = workdir or os.path.dirname(os.path.abspath(root.rstrip("/"))) or "."
    os.makedirs(wd, exist_ok=True)
    zpath = os.path.join(wd, name + ".zip")
    size = make_zip(root, zpath)
    part_bytes = int(part_mb * MIB)
    limit_bytes = int((limit_mb if limit_mb is not None else part_mb) * MIB)
    parts = []
    if size > limit_bytes:
        parts = split_like_coreutils(zpath, part_bytes)
    handed = list(parts or [zpath])
    extra = []
    for sp in standalone:
        dst = os.path.join(wd, os.path.basename(sp))
        if os.path.abspath(sp) != os.path.abspath(dst):
            shutil.copy(sp, dst)
        extra.append(dst)
    shafile = write_sha_file(wd, name, handed + extra)
    # proof (nothing is written): join, check against the ledger inside, and a join with a part dropped must fail
    tmp = tempfile.mkdtemp(prefix="r85_pack_proof_"); lines = []; ok = True
    try:
        z = join_to(handed, os.path.join(tmp, "joined.zip"))
        o, l = ledger_check(z); lines += l; ok &= o
        if len(handed) >= 2:
            z2 = join_to(handed[:1] + handed[2:], os.path.join(tmp, "dropped.zip"))
            o2, _ = ledger_check(z2)
            lines.append("%s a join with part 01 dropped fails the ledger check" % ("PASS" if not o2 else "FAIL")); ok &= (not o2)
        else:
            lines.append("INFO single zip: dropped-part test not applicable")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    g, m = sha_check(shafile, wd)
    lines.append("%s %s lists %d file(s) and every hash verifies: %s" % ("PASS" if g else "FAIL", os.path.basename(shafile), len(handed + extra), m)); ok &= g
    if parts:
        os.remove(zpath)
    outputs = handed + [shafile] + extra
    say("FILES HANDED OVER (paste this list into the final message; nothing else is handed over):")
    for p in outputs:
        say("  %-60s %12d bytes" % (os.path.basename(p), os.path.getsize(p)))
    say("  (the zip is %d bytes: %s)" % (size, ("split into %d part(s) of at most %d bytes with split -b %gM -d -a 2" % (len(parts), part_bytes, part_mb)) if parts else "one file, not larger than one part, so not split"))
    for l in lines:
        say("  " + l)
    if not ok:
        say("FAILED: proof"); return 5
    say("packed %s: %d file(s) handed over, hashes only in %s.sha256, no whole-zip hash file" % (name, len(outputs), name))
    return 0


# ---------------------------------------------------------------- selftest
def selftest():
    fails = []; nck = [0]
    t = tempfile.mkdtemp(); root = os.path.join(t, "RET")
    for a in ("a1", "a2", "a3", "a4", "a5"):
        os.makedirs(os.path.join(root, "compact", a)); os.makedirs(os.path.join(root, "archive", a))
        open(os.path.join(root, "archive", a, "exit_status.txt"), "w").write("exit_code=0 signal=none\n")

    def trace(arm, val, mtime):
        p = os.path.join(root, "compact", arm, "pos_trace_all.csv.gz")
        with open(p, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=mtime) as g:
                g.write(("scan_id,px,py,pz,yaw_rad\n" + "".join("%d,%s,0,0,0\n" % (i, val + i * 1e-3) for i in range(1, 50))).encode())
    trace("a1", 0.0, 1); trace("a2", 0.0, 999)      # identical content, different gzip header time: still a duplicate
    trace("a3", 1.0, 5); trace("a4", 2.0, 5)
    trace("a5", 1.0, 7)                              # identical to a3: an UNEXPECTED duplicate unless the arm table says otherwise
    open(os.path.join(root, "r85_gates.txt"), "w").write("PASS 1  FAIL 0\n")
    req = [dict(followup="F-1", title="t", arms=["a1", "a3"], file="pos_trace_all.csv.gz", where="return", source="arm"),
           dict(followup="F-3", title="e", arms=["a1"], file="exit_status.txt", where="archive", source="arm")]
    rp = os.path.join(t, "req.json"); json.dump(req, open(rp, "w"))
    at = os.path.join(t, "arm_table.tsv")
    open(at, "w").write("arm\tkind\tref\na1\tclosed\t\na2\tdet\ta1\na3\tclosed\t\na4\tclosed\t\na5\tclosed\t\n")

    def ck(cond, msg):
        nck[0] += 1
        if not cond:
            fails.append(msg)

    def capture(fn, *a, **k):
        buf = io.StringIO(); so = sys.stdout; sys.stdout = buf
        try:
            rc = fn(*a, **k)
        finally:
            sys.stdout = so
        return rc, buf.getvalue()
    # 1. refuse paths
    os.remove(os.path.join(root, "r85_gates.txt"))
    ck(pack(root, rp, "T0", 28, None, t, at, quiet=True) == 2, "missing gates file must refuse")
    open(os.path.join(root, "r85_gates.txt"), "w").write("PASS 1  FAIL 0\n")
    gen = os.path.join(root, "r85_l0_rows.csv")                      # a generated required file (where == return, source == generated)
    req.append(dict(followup="R85-L0", title="g", arms=[], file="r85_l0_rows.csv", where="return", source="generated")); json.dump(req, open(rp, "w"))
    ck(pack(root, rp, "T0", 28, None, os.path.join(t, "w0"), at, quiet=True) == 2, "a missing generated required file must refuse")
    open(gen, "w").write("arm\na1\n")
    os.remove(os.path.join(root, "archive", "a1", "exit_status.txt"))
    ck(pack(root, rp, "T0", 28, None, t, at, quiet=True) == 2, "missing archive file must refuse")
    open(os.path.join(root, "archive", "a1", "exit_status.txt"), "w").write("exit_code=0 signal=none\n")
    # 2. no split: the zip is under one part -> the single zip and the one .sha256, nothing else
    w1 = os.path.join(t, "w1")
    rc, out = capture(pack, root, rp, "S1", 28, None, w1, at, ())
    ck(rc == 0 and sorted(os.listdir(w1)) == ["S1.sha256", "S1.zip"], "single zip: rc %s files %s" % (rc, sorted(os.listdir(w1))))
    ck("S1.zip" in out and "S1.sha256" in out and "FILES HANDED OVER" in out and "bytes" in out, "listing must name every handed-over file with its size: %s" % out[-400:])
    ck(sha_check(os.path.join(w1, "S1.sha256"), w1)[0] and len(open(os.path.join(w1, "S1.sha256")).read().splitlines()) == 1, "the .sha256 lists the single zip")
    dup = open(os.path.join(root, "duplicates.tsv")).read()
    ck("a2\ta1\texpected (determinism repeat)" in dup, "a2 = a1 determinism repeat must be listed as expected: %s" % dup)
    ck("a5\ta3\tUNEXPECTED" in dup and "a4" not in dup, "a5 = a3 must be listed UNEXPECTED, a4 not at all: %s" % dup)
    led = open(os.path.join(root, LEDGER)).read()
    ck("compact/a1/pos_trace_all.csv.gz" in led and "duplicates.tsv" in led and "r85_gates.txt" in led and "(this file)" in led, "ledger must list every other file and itself")
    with zipfile.ZipFile(os.path.join(w1, "S1.zip")) as z:
        ck(any(n.endswith(LEDGER) for n in z.namelist()) and not any(n.endswith(".sha256") for n in z.namelist()), "return_ledger.tsv is inside the zip, no .sha256 inside")
    ok, lines = verify(os.path.join(w1, "S1.zip"), "S1")
    ck(ok, "verify of the single zip: %s" % lines)
    ok, lines = verify(w1, "S1")
    ck(ok, "verify of the single-zip folder: %s" % lines)
    # 3. split: random data, parts of 2 MiB
    big = os.path.join(root, "compact", "a1", "big.bin.gz")          # .gz: stored, so a byte flip lands in the data of the member as it is
    open(big, "wb").write(os.urandom(5 * MIB))
    w2 = os.path.join(t, "w2")
    stand = os.path.join(t, "R85_notes.txt"); open(stand, "w").write("a standalone file\n")
    rc, out = capture(pack, root, rp, "S2", 2.0, None, w2, at, (stand,))
    parts = sorted(glob.glob(os.path.join(w2, "S2.zip.[0-9][0-9]")))
    ck(rc == 0 and len(parts) >= 3 and not os.path.exists(os.path.join(w2, "S2.zip")), "split: rc %s parts %d" % (rc, len(parts)))
    ck(sorted(os.listdir(w2)) == sorted([os.path.basename(p) for p in parts] + ["S2.sha256", "R85_notes.txt"]), "handed-over folder holds the parts, the one .sha256 and the standalone file only: %s" % sorted(os.listdir(w2)))
    ck(not glob.glob(os.path.join(w2, "*.zip.sha256")) and not glob.glob(os.path.join(w2, "*.zip.*.sha256")) and not os.path.exists(os.path.join(w2, "reassembly_check.txt")) and not os.path.exists(os.path.join(w2, "R85_return.sha256")), "no whole-zip hash file, no per-part .sha256, no reassembly file")
    sl = open(os.path.join(w2, "S2.sha256")).read().splitlines()
    ck(len(sl) == len(parts) + 1 and all(re.match(r"^[0-9a-f]{64}  \S+$", x) for x in sl) and not any(x.endswith("  S2.zip") for x in sl) and any(x.endswith("  R85_notes.txt") for x in sl), "S2.sha256: one line per part plus the standalone file, no whole-zip line: %s" % sl)
    ck(all(os.path.basename(p) in out for p in parts) and "S2.sha256" in out and "R85_notes.txt" in out, "the output listing must name every handed-over file")
    ck(all(os.path.getsize(p) == 2 * MIB for p in parts[:-1]) and 0 < os.path.getsize(parts[-1]) <= 2 * MIB, "parts are full-size except the last")
    # the parts are what coreutils split -b 2M -d -a 2 produces from the same zip
    cat = os.path.join(t, "cat.zip"); join_to(parts, cat)
    if shutil.which("split"):
        sp = os.path.join(t, "cu"); os.makedirs(sp)
        subprocess.check_call(["split", "-b", "2M", "-d", "-a", "2", cat, os.path.join(sp, "x.zip.")])
        cu = sorted(glob.glob(os.path.join(sp, "x.zip.*")))
        ck(len(cu) == len(parts) and all(open(a, "rb").read() == open(b, "rb").read() for a, b in zip(cu, parts)) and [os.path.basename(c)[-2:] for c in cu] == [os.path.basename(p)[-2:] for p in parts], "parts differ from coreutils split -b 2M -d -a 2")
    with zipfile.ZipFile(cat) as z:
        ck(z.testzip() is None, "joined zip invalid")
    # 4. receiver: pass, then failures
    ok, lines = verify(w2, "S2", keep_zip=True)
    ck(ok and os.path.exists(os.path.join(w2, "S2.zip")), "receiver verify on the good set must pass: %s" % lines)
    ck(verify(os.path.join(w2, "S2.zip"), "S2")[0], "verify of the kept reassembled zip against its ledger")
    os.remove(os.path.join(w2, "S2.zip"))
    # a byte flipped in the reassembled zip, inside the stored member big.bin.gz: ledger check fails (no hash file is involved)
    with zipfile.ZipFile(cat) as z:
        zi = [i for i in z.infolist() if i.filename.endswith("big.bin.gz")][0]
        off = zi.header_offset + 30 + len(zi.filename.encode()) + len(zi.extra) + 12345
    flipped = os.path.join(t, "flipped.zip"); shutil.copy(cat, flipped)
    with open(flipped, "r+b") as f:
        f.seek(off); b = f.read(1); f.seek(off); f.write(bytes([b[0] ^ 1]))
    ok, lines = verify(flipped, "S2")
    ck(not ok and any("big.bin.gz" in l for l in lines), "one flipped byte in the reassembled zip must fail the ledger check: %s" % lines)
    # the same flip in a part: the per-part hash fails, and with the hash file removed the ledger check still fails
    pi = off // (2 * MIB); po = off % (2 * MIB)
    with open(parts[pi], "r+b") as f:
        f.seek(po); b = f.read(1); f.seek(po); f.write(bytes([b[0] ^ 1]))
    ok, lines = verify(w2, "S2")
    ck(not ok and any(l.startswith("FAIL S2.sha256") for l in lines) and any(l.startswith("FAIL ledger check") for l in lines), "a flipped byte in a part must fail both the part hash and the ledger: %s" % lines)
    shutil.move(os.path.join(w2, "S2.sha256"), os.path.join(t, "S2.sha256.away"))
    ok, lines = verify(w2, "S2")
    ck(not ok and any(l.startswith("INFO S2.sha256 not found") for l in lines) and any(l.startswith("FAIL ledger check") for l in lines), "without the hash file the ledger check still fails on the flipped part: %s" % lines)
    shutil.move(os.path.join(t, "S2.sha256.away"), os.path.join(w2, "S2.sha256"))
    with open(parts[pi], "r+b") as f:                      # undo the flip
        f.seek(po); b = f.read(1); f.seek(po); f.write(bytes([b[0] ^ 1]))
    ck(verify(w2, "S2")[0], "after undoing the flip the set verifies again")
    # a flip inside a deflated member (its stream or CRC breaks) and a flip in the zip structure also fail
    with zipfile.ZipFile(cat) as z:
        zd = [i for i in z.infolist() if i.filename.endswith("return_ledger.tsv") is False and i.compress_type == zipfile.ZIP_DEFLATED][0]
        off2 = zd.header_offset + 30 + len(zd.filename.encode()) + len(zd.extra) + 3
    f2 = os.path.join(t, "flipped2.zip"); shutil.copy(cat, f2)
    with open(f2, "r+b") as f:
        f.seek(off2); b = f.read(1); f.seek(off2); f.write(bytes([b[0] ^ 0x55]))
    ck(not verify(f2, "S2")[0], "a flipped byte in a deflated member must fail")
    # dropped part, dropped last part
    os.rename(parts[1], parts[1] + ".gone")
    ck(not verify(w2, "S2")[0], "a dropped middle part (suffix gap) must fail")
    os.rename(parts[1] + ".gone", parts[1])
    os.rename(parts[-1], parts[-1] + ".gone")
    ck(not verify(w2, "S2")[0], "a dropped LAST part must fail (part hash list and ledger)")
    os.rename(parts[-1] + ".gone", parts[-1])
    # a file added to the zip but not in the ledger, and a file in the ledger that is not in the zip
    f3 = os.path.join(t, "extra.zip"); shutil.copy(cat, f3)
    with zipfile.ZipFile(f3, "a") as z:
        z.writestr("RET/surprise.txt", "x")
    ok, lines = verify(f3, "S2")
    ck(not ok and any("surprise.txt: in the zip, not in the ledger" in l for l in lines), "a file the ledger does not list must fail: %s" % lines)
    # 5. the threshold: a zip exactly at one part is not split, one byte over is
    zsize = os.path.getsize(cat)
    w3 = os.path.join(t, "w3")
    rc, _ = capture(pack, root, rp, "S3", zsize / MIB, None, w3, at, ())
    ck(rc == 0 and os.path.exists(os.path.join(w3, "S3.zip")) and not glob.glob(os.path.join(w3, "S3.zip.[0-9]*")), "a zip not larger than one part is handed over whole")
    w4 = os.path.join(t, "w4")
    rc, _ = capture(pack, root, rp, "S4", (zsize - 1) / MIB, None, w4, at, ())
    ck(rc == 0 and len(glob.glob(os.path.join(w4, "S4.zip.[0-9][0-9]"))) == 2 and not os.path.exists(os.path.join(w4, "S4.zip")), "a zip one byte over one part is split into two parts")
    ck(verify(w4, "S4")[0], "the two-part set verifies")
    shutil.rmtree(t, ignore_errors=True)
    print("SELFTEST r85_pack_return", ("FAIL " + "; ".join(fails) + " | ") if fails else "PASS", "checks: %d, failures: %d" % (nck[0], len(fails)),
          "" if fails else "(refuses on missing gates / archive file / missing generated file; duplicates by decompressed pos_trace hash; ledger inside the zip lists every other file; no split at or under one part, split over it; "
          "parts equal coreutils split -b 2M -d -a 2; ONE .sha256 with the part lines and the standalone file, no whole-zip hash file, no per-part hash files; listing names every file with its size; "
          "receiver verify passes; one flipped byte fails (stored member, deflated member, part with and without the hash file); dropped part, extra file, missing ledger row fail)")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir"); ap.add_argument("--required"); ap.add_argument("--name", default="R85_RETURN"); ap.add_argument("--part-mb", type=float, default=28.0)
    ap.add_argument("--limit-mb", type=float, default=None, help="split only when the zip is larger than this; default = --part-mb (rule 5.12b: only when a part would exceed 28 MB)")
    ap.add_argument("--arm-table", default=None); ap.add_argument("--workdir", default=None)
    ap.add_argument("--standalone", nargs="*", default=[], help="other files handed over next to the parts; their hashes go into the one .sha256")
    ap.add_argument("--verify", default=None, help="folder holding <name>.zip.NN (and <name>.sha256), or a reassembled .zip: the receiver's check against return_ledger.tsv inside")
    ap.add_argument("--keep-zip", action="store_true", help="with --verify on a folder: keep the reassembled <name>.zip in the folder when it verifies"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.verify:
        ok, lines = verify(a.verify, a.name, a.keep_zip)
        print("\n".join(lines)); print("VERIFY", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    if not (a.dir and a.required):
        sys.exit("--dir and --required are required")
    return pack(a.dir, a.required, a.name, a.part_mb, a.limit_mb, a.workdir, a.arm_table, a.standalone)


if __name__ == "__main__":
    sys.exit(main())
