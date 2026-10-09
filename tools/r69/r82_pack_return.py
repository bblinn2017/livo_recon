#!/usr/bin/env python3
"""R82: pack the return (from r80d_pack_return.py). Planning agent's script; selftest on a synthetic folder only (planner sandbox, not evidence).

    python3 tools/r69/r82_pack_return.py --dir R82_RETURN --required r82_manifest/required_outputs.json --name R82_RETURN_<date> [--arm-table r82_manifest/arm_table.tsv] [--limit-mb 30] [--part-mb 28]
    python3 tools/r69/r82_pack_return.py --verify <folder holding <name>.zip.NN, <name>.zip.NN.sha256, <name>.sha256> --name R82_RETURN_<date>      # what the planner / a receiver runs
    python3 tools/r69/r82_pack_return.py --selftest

What it does, in this order (it stops with a non-zero exit at the first failed step and says which):
 1. REQUIRED: every file of required_outputs.json is resolved (where == "return", source == "arm" -> <dir>/compact/<arm>/<file>; "return" + "generated" -> <dir>/<file>; where == "archive" -> <dir>/archive/<arm>/<file>)
    and must exist and be non-empty. Also required: r82_gates.txt, and (unless --no-imu-timing) imu_timing/imu_timing_summary.csv with at least 7 data rows (the F-492 rerun, r80d_imu_timing_v2.py). Any miss -> REFUSED, every
    missing path named (exit 2), nothing is packed.
 2. DUPLICATES: sha256 of the DECOMPRESSED content of compact/<arm>/pos_trace_all.csv.gz for every arm that has one (a gzip header carries a time stamp, so file bytes would miss identical content). Arms with the same
    hash are written to <dir>/duplicates.tsv (columns arm, dup_of, notes; rows already there, e.g. from the collector, are kept). R80D's list was built only from the arm table's kind == dup rows and was blind to two identical
    arms. A group is annotated `expected` when --arm-table says one member is a determinism repeat (kind starts with det) of another member; every other group is `UNEXPECTED` and is printed. SWT arms return no trace
    (summaries only) and cannot be compared this way: they are listed as `not comparable`.
 3. LEDGER: <dir>/return_ledger.tsv = relative path, bytes, sha256 of EVERY file in the folder (imu_timing/ included; the ledger and the duplicates file are written before it is computed, the ledger lists itself as
    `(this file)`).
 4. ZIP the folder (stored for .gz/.zip members, deflated otherwise). If the zip is over --limit-mb it is split with the exact rule of `split -b <part>M -d -a 2 <name>.zip <name>.zip.` (part = --part-mb MiB, default 28,
    suffixes .00 .01 ...) and the whole zip is removed; a <part>.sha256 file ("<sha256>  <part name>", the format of `sha256sum -c`) is written for EVERY part and <name>.sha256 for the whole zip (a single zip gets
    <name>.zip.sha256 too). The final output listing prints every output file INCLUDING the .sha256 files and the tool exits non-zero if any expected .sha256 is missing or does not verify (F-496).
 5. REASSEMBLY PROOF (reassembly_check.txt): the parts are joined in suffix order (cat <name>.zip.* equivalent; gaps in the suffix sequence fail), the result must equal the whole-zip hash, `sha256sum -c` equivalent
    on every part passes, and a copy with one part dropped (and one with the parts in the wrong order) MUST fail the comparison; both results are printed.
The tool never deletes anything except the single whole zip it replaced by its parts.
"""
import argparse, csv, glob, gzip, hashlib, io, json, os, re, shutil, subprocess, sys, tempfile, zipfile

MIB = 1024 * 1024
IMU_SUMMARY = "imu_timing/imu_timing_summary.csv"
IMU_MIN_ROWS = 7
GATES = "r82_gates.txt"


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


def find_missing(root, req, imu=True):
    miss = [(p, fu, t) for p, fu, t in required_paths(root, req) if not os.path.isfile(p) or os.path.getsize(p) == 0]
    g = os.path.join(root, GATES)
    if not os.path.isfile(g) or os.path.getsize(g) == 0:
        if not any(p == g for p, _, _ in miss):
            miss.append((g, "R82", "gates file"))
    if imu:
        p = os.path.join(root, IMU_SUMMARY)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            miss.append((p, "F-492", "IMU timing summary (r80d_imu_timing_v2.py)"))
        elif csv_rows(p) < IMU_MIN_ROWS:
            miss.append((p, "F-492", "IMU timing summary has %d data rows, at least %d (the seven remaining bags)" % (csv_rows(p), IMU_MIN_ROWS)))
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
    lp = os.path.join(root, "return_ledger.tsv")
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
        f.write("return_ledger.tsv\t-\t(this file)\n")
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


def write_sha_files(zpath, parts):
    """<part>.sha256 for every part, <name>.sha256 for the whole zip (single zip: <zip>.sha256). Format of `sha256sum -c`. -> list of sha files."""
    out = []
    whole = sha_parts(parts) if parts else sha_file(zpath)
    for p in parts:
        sp = p + ".sha256"
        open(sp, "w").write("%s  %s\n" % (sha_file(p), os.path.basename(p))); out.append(sp)
    wp = (zpath[:-len(".zip")] + ".sha256") if parts else zpath + ".sha256"
    open(wp, "w").write("%s  %s\n" % (whole, os.path.basename(zpath))); out.append(wp)
    return out, whole


def sha_check(shafile, directory):
    """`sha256sum -c` of one .sha256 file inside `directory`: True when every listed file exists and hashes to its line. Returns (ok, message)."""
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


def reassembly_proof(directory, name, out_txt=None):
    """the receiver's check. -> (ok, lines). Whole hash from <name>.sha256; every part against its own .sha256; join in suffix order; a copy with one part dropped and one with two parts swapped must FAIL."""
    L = []; ok = True
    try:
        ps = part_paths(directory, name)
    except ValueError as e:
        L.append("FAIL parts: %s" % e); ok = False; ps = []
    single = os.path.join(directory, name + ".zip")
    wf = os.path.join(directory, name + ".sha256") if ps else single + ".sha256"
    if not os.path.exists(wf):
        L.append("FAIL whole-zip sha256 file missing: %s" % os.path.basename(wf)); ok = False; whole = None
    else:
        whole = open(wf).read().split()[0]
    if not ps and not os.path.exists(single):
        L.append("FAIL neither parts nor a single zip found for %s" % name); ok = False
    files = ps or ([single] if os.path.exists(single) else [])
    for p in files:
        sp = p + ".sha256"
        if not os.path.exists(sp):
            L.append("FAIL %s missing" % os.path.basename(sp)); ok = False
        else:
            g, m = sha_check(sp, directory)
            L.append("%s %s: %s" % ("PASS" if g else "FAIL", os.path.basename(sp), m)); ok &= g
    if files and whole:
        joined = sha_parts(files)
        g = joined == whole
        L.append("%s reassembly (parts joined in suffix order) sha256 %s the recorded whole-zip hash" % ("PASS" if g else "FAIL", "equals" if g else "DIFFERS from")); ok &= g
        if len(files) >= 2:
            d = sha_parts(files[:1] + files[2:]) != whole
            L.append("%s reassembly with part 01 dropped is rejected (hash differs)" % ("PASS" if d else "FAIL")); ok &= d
            sw = sha_parts([files[1], files[0]] + files[2:]) != whole
            L.append("%s reassembly with parts 00 and 01 swapped is rejected (hash differs)" % ("PASS" if sw else "FAIL")); ok &= sw
        else:
            L.append("INFO single zip: dropped-part test not applicable")
    if out_txt:
        open(out_txt, "w").write("\n".join(L) + "\n")
    return ok, L


def pack(root, req_path, name, limit_mb=30.0, part_mb=28.0, workdir=None, arm_table=None, imu=True, quiet=False):
    say = (lambda *a: None) if quiet else print
    req = json.load(open(req_path))
    miss = find_missing(root, req, imu)
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
    zpath = os.path.join(wd, name + ".zip")
    size = make_zip(root, zpath)
    parts = []
    if size > limit_mb * MIB:
        parts = split_like_coreutils(zpath, int(part_mb * MIB))
        shas, whole = write_sha_files(zpath, parts)
        os.remove(zpath)
        if sha_parts(parts) != whole:
            say("REFUSED: joined parts differ from the whole zip"); return 3
    else:
        shas, whole = write_sha_files(zpath, [])
    outputs = (parts or [zpath]) + shas
    gone = [p for p in shas if not os.path.isfile(p)]
    ok, lines = reassembly_proof(wd, name, os.path.join(wd, "reassembly_check.txt"))
    outputs.append(os.path.join(wd, "reassembly_check.txt"))
    say("OUTPUT LISTING (every file the receiver gets; the .sha256 files are part of it):")
    for p in outputs:
        say("  %-60s %s" % (os.path.basename(p), ("%12d bytes" % os.path.getsize(p)) if os.path.isfile(p) else "MISSING"))
    for l in lines:
        say("  " + l)
    if gone:
        say("FAILED: .sha256 file(s) missing: %s" % ", ".join(os.path.basename(p) for p in gone)); return 4
    if not ok:
        say("FAILED: reassembly proof"); return 5
    n_sha = len([p for p in outputs if p.endswith(".sha256")])
    exp_sha = (len(parts) + 1) if parts else 1
    if n_sha != exp_sha:
        say("FAILED: %d .sha256 files listed, expected %d" % (n_sha, exp_sha)); return 4
    say("packed %s: %d bytes, sha256 %s, %d part(s), %d .sha256 file(s)" % (name, size, whole[:16], len(parts) or 1, n_sha))
    return 0


# ---------------------------------------------------------------- selftest
def selftest():
    import random
    fails = []
    t = tempfile.mkdtemp(); root = os.path.join(t, "RET")
    for a in ("a1", "a2", "a3", "a4", "a5"):
        os.makedirs(os.path.join(root, "compact", a)); os.makedirs(os.path.join(root, "archive", a))
        open(os.path.join(root, "archive", a, "exit_status.txt"), "w").write("exit_code=0 signal=none\n")
    os.makedirs(os.path.join(root, "imu_timing"))

    def trace(arm, val, mtime):
        p = os.path.join(root, "compact", arm, "pos_trace_all.csv.gz")
        with open(p, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=mtime) as g:
                g.write(("scan_id,px,py,pz,yaw_rad\n" + "".join("%d,%s,0,0,0\n" % (i, val + i * 1e-3) for i in range(1, 50))).encode())
    trace("a1", 0.0, 1); trace("a2", 0.0, 999)      # identical content, different gzip header time: still a duplicate
    trace("a3", 1.0, 5); trace("a4", 2.0, 5)         # different content
    trace("a5", 1.0, 7)                              # identical to a3: an UNEXPECTED duplicate unless the arm table says otherwise
    open(os.path.join(root, "r82_gates.txt"), "w").write("PASS 1  FAIL 0\n")
    open(os.path.join(root, "imu_timing", "imu_timing_summary.csv"), "w").write("bag,n\n" + "".join("b%d,100\n" % i for i in range(9)))
    req = [dict(followup="F-1", title="t", arms=["a1", "a3"], file="pos_trace_all.csv.gz", where="return", source="arm"),
           dict(followup="F-3", title="e", arms=["a1"], file="exit_status.txt", where="archive", source="arm")]
    rp = os.path.join(t, "req.json"); json.dump(req, open(rp, "w"))
    at = os.path.join(t, "arm_table.tsv")
    open(at, "w").write("arm\tkind\tref\na1\tclosed\t\na2\tdet\ta1\na3\tclosed\t\na4\tclosed\t\na5\tclosed\t\n")
    ck = lambda cond, msg: fails.append(msg) if not cond else None
    # 1. refuse paths (each shown to fire)
    os.remove(os.path.join(root, "r82_gates.txt"))
    ck(pack(root, rp, "T0", 30, 28, t, at, quiet=True) == 2, "missing gates file must refuse")
    open(os.path.join(root, "r82_gates.txt"), "w").write("PASS 1  FAIL 0\n")
    os.rename(os.path.join(root, "imu_timing", "imu_timing_summary.csv"), os.path.join(t, "imu.bak"))
    ck(pack(root, rp, "T0", 30, 28, t, at, quiet=True) == 2, "missing imu summary must refuse")
    open(os.path.join(root, "imu_timing", "imu_timing_summary.csv"), "w").write("bag,n\nb0,1\n")
    ck(pack(root, rp, "T0", 30, 28, t, at, quiet=True) == 2, "imu summary with 1 row must refuse")
    ck(pack(root, rp, "T0", 30, 28, t, at, imu=False, quiet=True) == 0, "--no-imu-timing waives the imu requirement")
    shutil.copy(os.path.join(t, "imu.bak"), os.path.join(root, "imu_timing", "imu_timing_summary.csv"))
    os.remove(os.path.join(root, "archive", "a1", "exit_status.txt"))
    ck(pack(root, rp, "T0", 30, 28, t, at, quiet=True) == 2, "missing archive file must refuse")
    open(os.path.join(root, "archive", "a1", "exit_status.txt"), "w").write("exit_code=0 signal=none\n")
    # 2. single zip: .sha256 listed
    buf = io.StringIO(); so = sys.stdout; sys.stdout = buf
    try:
        rc = pack(root, rp, "S1", 30, 28, t, at)
    finally:
        sys.stdout = so
    out = buf.getvalue()
    ck(rc == 0 and "S1.zip.sha256" in out and os.path.exists(os.path.join(t, "S1.zip")), "single zip: rc %s, listing %s" % (rc, out[-300:]))
    dup = open(os.path.join(root, "duplicates.tsv")).read()
    ck("a2\ta1\texpected (determinism repeat)" in dup, "a2 = a1 determinism repeat must be listed as expected: %s" % dup)
    ck("a5\ta3\tUNEXPECTED" in dup and "a4" not in dup, "a5 = a3 must be listed UNEXPECTED, a4 not at all: %s" % dup)
    led = open(os.path.join(root, "return_ledger.tsv")).read()
    ck("imu_timing/imu_timing_summary.csv" in led and "duplicates.tsv" in led and "compact/a1/pos_trace_all.csv.gz" in led, "ledger must list the imu_timing and duplicates files")
    # 3. split: big, random data
    open(os.path.join(root, "compact", "a1", "big.bin"), "wb").write(os.urandom(5 * MIB))
    buf = io.StringIO(); sys.stdout = buf
    try:
        rc = pack(root, rp, "S2", 1.0, 2.0, t, at)
    finally:
        sys.stdout = so
    out = buf.getvalue()
    parts = sorted(glob.glob(os.path.join(t, "S2.zip.[0-9][0-9]")))
    ck(rc == 0 and len(parts) >= 3 and not os.path.exists(os.path.join(t, "S2.zip")), "split: rc %s parts %d" % (rc, len(parts)))
    ck(all(os.path.exists(p + ".sha256") for p in parts) and os.path.exists(os.path.join(t, "S2.sha256")), "every part has a .sha256 and the whole zip has S2.sha256")
    ck(all(os.path.basename(p) + ".sha256" in out for p in parts) and "S2.sha256" in out, "the output listing must name every .sha256 file")
    # the parts are what coreutils split -b 2M -d -a 2 produces from the same zip
    if shutil.which("split"):
        cat = os.path.join(t, "cat.zip"); open(cat, "wb").write(b"".join(open(p, "rb").read() for p in parts))
        sp = os.path.join(t, "cu"); os.makedirs(sp)
        subprocess.check_call(["split", "-b", "2M", "-d", "-a", "2", cat, os.path.join(sp, "x.zip.")])
        cu = sorted(glob.glob(os.path.join(sp, "x.zip.*")))
        ck(len(cu) == len(parts) and all(open(a, "rb").read() == open(b, "rb").read() for a, b in zip(cu, parts)) and [os.path.basename(c)[-2:] for c in cu] == [os.path.basename(p)[-2:] for p in parts], "parts differ from coreutils split -b 2M -d -a 2")
    # the zip itself is valid and carries every file
    cat = os.path.join(t, "S2_joined.zip"); open(cat, "wb").write(b"".join(open(p, "rb").read() for p in parts))
    with zipfile.ZipFile(cat) as z:
        ck(z.testzip() is None and any(n.endswith("return_ledger.tsv") for n in z.namelist()), "joined zip invalid")
    # 4. receiver checks: dropped part, missing .sha256, swapped parts, corrupted part
    ok, lines = reassembly_proof(t, "S2")
    ck(ok and any("dropped is rejected" in l and l.startswith("PASS") for l in lines), "receiver proof on the good set: %s" % lines)
    os.rename(parts[1], parts[1] + ".gone")
    ok, lines = reassembly_proof(t, "S2")
    ck(not ok, "a dropped middle part (suffix gap or missing) must fail: %s" % lines)
    os.rename(parts[1] + ".gone", parts[1])
    os.rename(parts[-1], parts[-1] + ".gone")
    ok, lines = reassembly_proof(t, "S2")
    ck(not ok, "a dropped LAST part must fail (hash differs from the whole): %s" % lines)
    os.rename(parts[-1] + ".gone", parts[-1])
    os.rename(parts[0] + ".sha256", parts[0] + ".sha256.gone")
    ok, lines = reassembly_proof(t, "S2")
    ck(not ok and any("sha256 missing" in l for l in lines), "a missing .sha256 must fail: %s" % lines)
    os.rename(parts[0] + ".sha256.gone", parts[0] + ".sha256")
    with open(parts[0], "r+b") as f:
        f.seek(100); b = f.read(1); f.seek(100); f.write(bytes([b[0] ^ 1]))
    ok, lines = reassembly_proof(t, "S2")
    ck(not ok, "a corrupted part must fail: %s" % lines)
    # 4b. the pack tool fails when a .sha256 file it lists is missing (F-496): drop one right after it is written
    global write_sha_files
    orig = write_sha_files
    def broken(zp, ps):
        out_, wh = orig(zp, ps); os.remove(out_[0]); return out_, wh
    write_sha_files = broken
    try:
        rc = pack(root, rp, "S3", 1.0, 2.0, t, at, quiet=True)
    finally:
        write_sha_files = orig
    ck(rc != 0, "pack must fail (non-zero) when a .sha256 file is missing, rc %s" % rc)
    # 5. the pack tool itself fails when a .sha256 is missing after writing (checked through the final verification)
    ck(sha_check(os.path.join(t, "S1.zip.sha256"), t)[0], "sha_check of a good file")
    open(os.path.join(t, "bad.sha256"), "w").write("0" * 64 + "  S1.zip\n")
    ck(not sha_check(os.path.join(t, "bad.sha256"), t)[0], "sha_check must fail on a wrong hash")
    print("SELFTEST r82_pack_return", "FAIL " + "; ".join(fails) if fails else
          "PASS (refuses on missing gates / imu summary / archive file; duplicates by decompressed pos_trace hash: expected pair and UNEXPECTED pair found, different arm not; ledger lists imu_timing; single zip and split zip "
          "with a .sha256 per part listed in the output; parts equal coreutils split -b 2M -d -a 2; reassembly proven; dropped middle part, dropped last part, missing .sha256, corrupted part all rejected)")
    shutil.rmtree(t, ignore_errors=True)
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir"); ap.add_argument("--required"); ap.add_argument("--name", default="R82_RETURN"); ap.add_argument("--limit-mb", type=float, default=30.0); ap.add_argument("--part-mb", type=float, default=28.0)
    ap.add_argument("--arm-table", default=None); ap.add_argument("--no-imu-timing", action="store_true"); ap.add_argument("--workdir", default=None)
    ap.add_argument("--verify", default=None, help="directory holding <name>.zip.NN, <name>.zip.NN.sha256, <name>.sha256: run the receiver's reassembly proof"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.verify:
        ok, lines = reassembly_proof(a.verify, a.name)
        print("\n".join(lines)); print("VERIFY", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    if not (a.dir and a.required):
        sys.exit("--dir and --required are required")
    return pack(a.dir, a.required, a.name, a.limit_mb, a.part_mb, a.workdir, a.arm_table, not a.no_imu_timing)


if __name__ == "__main__":
    sys.exit(main())
