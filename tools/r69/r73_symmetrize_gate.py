#!/usr/bin/env python3
"""R73 (F-419) grep gate: every covariance assignment is symmetrised through include/livo_recon/utils/algo/symmetrize.h.

    python3 tools/r69/r73_symmetrize_gate.py --src <livo_recon tree> [--out r73_symmetrize_gate.txt]
    python3 tools/r69/r73_symmetrize_gate.py --selftest

FAIL when (1) a `covMut() =` / `covMut() +=` assignment's statement does not contain `symmetrized(` or `symmetrizeInPlace(`, (2) a hand-written `0.5 * (X + X.transpose())` or
`0.5 * (X + X.transpose().eval())` pattern occurs outside symmetrize.h, (3) symmetrize.h is absent. Two known non-symmetrising uses of 0.5*( are whitelisted by file+needle.
"""
import argparse, os, re, sys, tempfile

WHITELIST = [("lio_iekf.cpp", "dx.transpose() *"), ("init_consistency.cpp", ".trace()")]
HAND = re.compile(r"0\.5\s*\*\s*\(\s*[\w\.\->\[\]\(\)]+\s*\+\s*[\w\.\->\[\]\(\)]*\.transpose\(\)")
ASSIGN = re.compile(r"covMut\(\)\s*(\+|-)?=")


def statement(lines, i):
    s = lines[i]
    j = i
    while ";" not in lines[j] and j + 1 < len(lines):
        j += 1
        s += lines[j]
    return s


def scan(src):
    probs, n_assign = [], 0
    if not os.path.exists(os.path.join(src, "include", "livo_recon", "utils", "algo", "symmetrize.h")):
        probs.append("symmetrize.h absent")
    roots = [os.path.join(src, d) for d in ("src", "include")]
    if not any(os.path.isdir(r) for r in roots):
        roots = [src]      # selftest layout
    for dp, _, fn in (x for r in roots for x in os.walk(r)):
        for f in fn:
            if not f.endswith((".cpp", ".h", ".hpp", ".cu")) or f == "symmetrize.h":
                continue
            p = os.path.join(dp, f)
            L = open(p, errors="replace").read().split("\n")
            for i, l in enumerate(L):
                code = l.split("//")[0]
                if ASSIGN.search(code):
                    n_assign += 1
                    st = statement(L, i)
                    if "symmetrized(" not in st and "symmetrizeInPlace(" not in st:
                        probs.append("%s:%d covMut() assignment without symmetrize: %s" % (os.path.relpath(p, src), i + 1, st.strip()[:100]))
                if HAND.search(code) and not any(f == w[0] and w[1] in code for w in WHITELIST):
                    probs.append("%s:%d hand-written symmetrisation: %s" % (os.path.relpath(p, src), i + 1, code.strip()[:100]))
    return probs, n_assign


def selftest():
    fails = []
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(os.path.join(t, "include/livo_recon/utils/algo")); os.makedirs(os.path.join(t, "src")); open(os.path.join(t, "include/livo_recon/utils/algo/symmetrize.h"), "w").write("// x\n")
        open(os.path.join(t, "src", "a.cpp"), "w").write("void f(){ state_->covMut() = symmetrized(P);\n state_->covMut() = symmetrized(\n A * B);\n}\n")
        p, n = scan(t)
        if p or n != 2:
            fails.append("good tree: %s n=%d" % (p, n))
        open(os.path.join(t, "src", "b.cpp"), "w").write("void g(){ state_->covMut() = P;\n}\n")
        p, _ = scan(t)
        if not any("without symmetrize" in x for x in p):
            fails.append("bare covMut assignment not caught")
        os.remove(os.path.join(t, "src", "b.cpp"))
        open(os.path.join(t, "src", "c.cpp"), "w").write("auto Q = 0.5 * (Q0 + Q0.transpose());\n")
        p, _ = scan(t)
        if not any("hand-written" in x for x in p):
            fails.append("hand-written symmetrisation not caught")
        os.remove(os.path.join(t, "src", "c.cpp")); os.remove(os.path.join(t, "include/livo_recon/utils/algo/symmetrize.h"))
        p, _ = scan(t)
        if "symmetrize.h absent" not in p:
            fails.append("absent header not caught")
    if fails:
        print("SELFTEST r73_symmetrize_gate FAIL"); [print(" ", f) for f in fails]; return 1
    print("SELFTEST r73_symmetrize_gate PASS"); return 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src"); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    probs, n = scan(a.src)
    txt = "symmetrize gate: %d covMut() assignments checked, %d problems\n%s\n%s\n" % (n, len(probs), "\n".join(probs), "FAIL" if probs else "PASS")
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
