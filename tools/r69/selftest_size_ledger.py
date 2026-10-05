#!/usr/bin/env python3
"""N-2 (planning agent): shows size_ledger.py check FIRES. Cases: complete dir -> PASS; unledgered file -> FAIL; over-size file -> FAIL; over-total -> FAIL.
NOT covered by design: a MISSING file passes size_ledger.py (it only measures what exists); missing items are build_return.py's job (exits 3 against required_outputs.json).
Run: python3 selftest_size_ledger.py   (exit 0 only if every case behaves as stated)"""
import json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); SL = os.path.join(HERE, "size_ledger.py")
led = [{"pattern": "a.csv", "max_bytes_per_file": 100, "max_total_bytes": 100}, {"pattern": "b/*.txt", "max_bytes_per_file": 50, "max_total_bytes": 80}]
def run(files):
    with tempfile.TemporaryDirectory() as d:
        for p, n in files.items():
            os.makedirs(os.path.dirname(os.path.join(d, p)) or d, exist_ok=True); open(os.path.join(d, p), "wb").write(b"x" * n)
        lp = os.path.join(d, "..", "led_%d.json" % os.getpid()); json.dump(led, open(lp, "w"))
        r = subprocess.run([sys.executable, SL, "check", d, "--ledger", lp], capture_output=True, text=True); os.remove(lp)
        return r.returncode, r.stdout
cases = [("complete", {"a.csv": 10, "b/x.txt": 10}, 0), ("unledgered", {"a.csv": 10, "z.bin": 5}, 1),
         ("over_file", {"a.csv": 101}, 1), ("over_total", {"b/x.txt": 45, "b/y.txt": 45}, 1)]
bad = 0
for name, files, want in cases:
    rc, out = run(files); ok = rc == want; bad += not ok
    print("%-11s expected exit %d got %d  %s" % (name, want, rc, "OK" if ok else "WRONG")); print("   " + out.strip().replace("\n", "\n   "))
print("selftest_size_ledger: %s" % ("PASS" if not bad else "FAIL")); sys.exit(1 if bad else 0)
