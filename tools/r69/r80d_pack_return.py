#!/usr/bin/env python3
"""R80D: pack the return. Reads required_outputs.json (generated), REFUSES and names every required file that is missing, zips the folder, splits byte-wise when over the limit, writes a .sha256.

    python3 tools/r69/r80d_pack_return.py --dir R80D_RETURN --required r80d_manifest/required_outputs.json --name R80D_RETURN [--limit-mb 30]
    python3 tools/r69/r80d_pack_return.py --selftest

Required files are resolved per the manifest: where == "return" and source == "arm" -> <dir>/compact/<arm>/<file> for every listed arm; where == "return" and source == "generated" -> <dir>/<file>; where == "archive"
is checked in <dir>/archive/<arm>/<file>. The gates file r80d_gates.txt and the .sha256 are ALWAYS produced or the pack refuses (R80C's zip had neither listed). Split: split -b 28M -d -a 2 equivalent, parts
<name>.zip.00, .01, ...; the reassembly (cat parts > x.zip) is proven here by hash before the tool says done. Rule 5.12c. Planning agent's script; selftest on a synthetic folder only.
"""
import argparse, hashlib, json, os, shutil, sys, tempfile, zipfile


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


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def pack(root, req_path, name, limit_mb, workdir=None, part_mb=28):
    req = json.load(open(req_path))
    miss = [(p, fu, t) for p, fu, t in required_paths(root, req) if not os.path.isfile(p) or os.path.getsize(p) == 0]
    gates = os.path.join(root, "r80d_gates.txt")
    if not any(p == gates for p, _, _ in miss) and not os.path.isfile(gates):
        miss.append((gates, "F-484", "gates file"))
    if miss:
        print("REFUSED: %d required file(s) missing or empty:" % len(miss))
        for p, fu, t in miss[:50]: print("  %s  [%s %s]" % (p, fu, t))
        return 2
    wd = workdir or os.path.dirname(os.path.abspath(root)) or "."
    z = os.path.join(wd, name + ".zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for d, _, fs in os.walk(root):
            for f in sorted(fs): zf.write(os.path.join(d, f), os.path.relpath(os.path.join(d, f), os.path.dirname(os.path.abspath(root))))
    size = os.path.getsize(z); digest = sha(z)
    parts = [z]
    if size > limit_mb * 1024 * 1024:
        step = int(part_mb * 1024 * 1024); parts = []
        with open(z, "rb") as f:
            i = 0
            while True:
                b = f.read(step)
                if not b: break
                p = "%s.%02d" % (z, i); open(p, "wb").write(b); parts.append(p); i += 1
        rt = os.path.join(wd, ".reassembled.tmp")
        with open(rt, "wb") as o:
            for p in parts: o.write(open(p, "rb").read())
        ok = sha(rt) == digest; os.remove(rt)
        if not ok: print("REFUSED: reassembly hash differs"); return 3
        os.remove(z)
    open(os.path.join(wd, name + ".sha256"), "w").write("%s  %s.zip\n" % (digest, name))
    print("packed %s.zip (%d bytes, sha256 %s) in %d part(s): %s; reassembly %s" % (name, size, digest[:16], len(parts), [os.path.basename(p) for p in parts], "proven" if len(parts) > 1 else "n/a (single file)"))
    return 0


def selftest():
    t = tempfile.mkdtemp(); root = os.path.join(t, "RET"); os.makedirs(os.path.join(root, "compact", "a1")); os.makedirs(os.path.join(root, "archive", "a1"))
    req = [dict(followup="F-1", title="t", arms=["a1"], file="x.csv", where="return", source="arm"), dict(followup="F-2", title="g", arms=[], file="r80d_gates.txt", where="return", source="generated"),
           dict(followup="F-3", title="e", arms=["a1"], file="exit_status.txt", where="archive", source="arm")]
    rp = os.path.join(t, "req.json"); json.dump(req, open(rp, "w"))
    assert pack(root, rp, "T", 30, t) == 2                      # everything missing -> refuse
    open(os.path.join(root, "compact", "a1", "x.csv"), "w").write("a\n1\n"); open(os.path.join(root, "archive", "a1", "exit_status.txt"), "w").write("exit_code=0\n")
    assert pack(root, rp, "T", 30, t) == 2                      # gates still missing -> refuse
    open(os.path.join(root, "r80d_gates.txt"), "w").write("SUMMARY\n")
    assert pack(root, rp, "T", 30, t) == 0 and os.path.exists(os.path.join(t, "T.zip")) and os.path.exists(os.path.join(t, "T.sha256"))
    open(os.path.join(root, "compact", "a1", "big.bin"), "wb").write(os.urandom(5 * 1024 * 1024))
    assert pack(root, rp, "S", 1, t, 2) == 0 and os.path.exists(os.path.join(t, "S.zip.00")) and os.path.exists(os.path.join(t, "S.zip.01")) and not os.path.exists(os.path.join(t, "S.zip"))
    h = open(os.path.join(t, "S.sha256")).read().split()[0]
    cat = b"".join(open(os.path.join(t, "S.zip.%02d" % i), "rb").read() for i in range(10) if os.path.exists(os.path.join(t, "S.zip.%02d" % i)))
    assert hashlib.sha256(cat).hexdigest() == h
    shutil.rmtree(t); print("selftest: refuses when a required file or the gates file is missing; packs with sha256; split parts reassemble to the recorded hash")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir"); ap.add_argument("--required"); ap.add_argument("--name", default="R80D_RETURN"); ap.add_argument("--limit-mb", type=float, default=30.0); ap.add_argument("--part-mb", type=float, default=28.0); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: selftest(); return
    if not (a.dir and a.required): sys.exit("--dir and --required are required")
    sys.exit(pack(a.dir, a.required, a.name, a.limit_mb, None, a.part_mb))


if __name__ == "__main__":
    main()
