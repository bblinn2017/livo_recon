#!/usr/bin/env python3
"""R74 (F-419 extension) source gate for covariance symmetrisation.

    python3 tools/r69/r74_symmetrize_gate.py --src <livo_recon tree> [--exemptions tools/r69/r74_symmetrize_exemptions.json] [--out r74_symmetrize_gate.txt]
    python3 tools/r69/r74_symmetrize_gate.py --selftest

Two checks (the R73 gate, kept, plus a wider one):
 A. (R73) every `covMut() =` / `covMut() +=` statement contains symmetrized( or symmetrizeInPlace(; no hand-written `0.5 * (X + X.transpose())` outside symmetrize.h;
    symmetrize.h exists.
 B. (R74) every other assignment whose left-hand side NAMES a covariance or information matrix (cov, covariance, plane_var_, P_inverse, Gamma, posterior, K1, P0, P, P_new,
    P_acc, .P, .A, P[...]) and whose right-hand side forms a matrix product, a transpose, an inverse or a solve (or accumulates with noalias) must either contain symmetrized( /
    symmetrizeInPlace( or be listed in the EXEMPTION TABLE (file, a needle contained in the statement, and a reason). The table is listed in the output, and a table entry
    that matches no statement FAILS (a stale exemption hides nothing).
Statements whose right-hand side only reads a scalar (.trace(), .norm(), .value(), ...) or streams (<<) are ignored. This is a text heuristic, not a parser: it finds the
statements a reviewer should look at; it does not prove symmetry.
"""
import argparse, json, os, re, sys, tempfile

WHITELIST = [("lio_iekf.cpp", "dx.transpose() *"), ("init_consistency.cpp", ".trace()")]
HAND = re.compile(r"0\.5\s*\*\s*\(\s*[\w\.\->\[\]\(\)]+\s*\+\s*[\w\.\->\[\]\(\)]*\.transpose\(\)")
ASSIGN_COV = re.compile(r"covMut\(\)\s*(\+|-)?=")
ASSIGN = re.compile(r"(?<![=!<>+\-*/])(\+=|-=|=)(?!=)")
NAME = re.compile(r"(cov|covariance|plane_var_?|P_inverse|Gamma|posterior|\bK1\b|\bP0\b|\bP\b|P_new|P_acc|\.P\b|\.A\b|\bP\[)", re.I)
SKIP_RHS = re.compile(r"\.trace\(\)|\.norm\(\)|\.value\(\)|\.determinant\(\)|<<|\.coeff|\.maxCoeff|\.minCoeff")
NEED = re.compile(r"transpose\(\)|inverse\(\)|\.solve\(|noalias")
SYM = ("symmetrized(", "symmetrizeInPlace(")


def statement(lines, i, limit=8):
    s = lines[i].split("//")[0]
    j = i
    while ";" not in lines[j].split("//")[0] and j + 1 < len(lines) and j - i < limit:
        j += 1
        s += " " + lines[j].split("//")[0]
    return s, j


LEAD = re.compile(r"^\s*(else\s+)?(if\s*\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)\s*)?(else\s+)?")


def clean(st):
    """The statement proper: drop text before the last { or } and a leading if (...) / else."""
    k = max(st.rfind("{"), st.rfind("}"))
    if k >= 0 and ";" in st[k:]:
        st = st[k + 1:]
    for _ in range(3):
        st2 = LEAD.sub("", st, count=1)
        if st2 == st:
            break
        st = st2
    return st


def files(src):
    roots = [os.path.join(src, d) for d in ("src", "include")]
    if not any(os.path.isdir(r) for r in roots):
        roots = [src]
    for dp, _, fn in (x for r in roots for x in os.walk(r)):
        for f in sorted(fn):
            if f.endswith((".cpp", ".h", ".hpp", ".cu")) and f != "symmetrize.h":
                yield os.path.join(dp, f)


def scan(src, exemptions):
    probs, n_assign, n_b, used = [], 0, 0, set()
    if not os.path.exists(os.path.join(src, "include", "livo_recon", "utils", "algo", "symmetrize.h")) and not any(f.endswith("symmetrize.h") for f in [os.path.join(dp, x) for dp, _, fn in os.walk(src) for x in fn]):
        probs.append("symmetrize.h absent")
    for p in files(src):
        L = open(p, errors="replace").read().split("\n")
        base = os.path.basename(p)
        seen_end = -1
        for i, l in enumerate(L):
            code = l.split("//")[0]
            if ASSIGN_COV.search(code):
                n_assign += 1
                st, _ = statement(L, i)
                if not any(k in st for k in SYM):
                    probs.append("%s:%d covMut() assignment without symmetrize: %s" % (os.path.relpath(p, src), i + 1, st.strip()[:100]))
            if HAND.search(code) and not any(base == w[0] and w[1] in code for w in WHITELIST):
                probs.append("%s:%d hand-written symmetrisation: %s" % (os.path.relpath(p, src), i + 1, code.strip()[:100]))
            if i <= seen_end:
                continue
            st, j = statement(L, i)
            if ";" not in st:
                continue
            seen_end = j
            st = clean(st)
            m = ASSIGN.search(st)
            if not m or ASSIGN_COV.search(st):
                continue
            lhs, rhs = st[:m.start()].strip(), st[m.end():]
            if len(lhs) >= 80 or lhs.startswith(("for", "if", "while", "return", "else")):
                continue
            if not NAME.search(lhs) or not NEED.search(rhs + lhs) or SKIP_RHS.search(rhs) or any(k in st for k in SYM):
                continue
            n_b += 1
            hit = None
            for k, e in enumerate(exemptions):
                if e["file"] == base and e["needle"] in st:
                    hit = k
                    break
            if hit is None:
                probs.append("%s:%d covariance-like assignment neither symmetrized nor exempt: %s" % (os.path.relpath(p, src), i + 1, " ".join(st.split())[:110]))
            else:
                used.add(hit)
    for k, e in enumerate(exemptions):
        if k not in used:
            probs.append("stale exemption (matches no statement): %s | %s" % (e["file"], e["needle"]))
    return probs, n_assign, n_b, used


def selftest():
    fails = []
    ex = [{"file": "d.cpp", "needle": "P_acc = J * C", "reason": "test"}]
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(os.path.join(t, "include/livo_recon/utils/algo")); os.makedirs(os.path.join(t, "src"))
        open(os.path.join(t, "include/livo_recon/utils/algo/symmetrize.h"), "w").write("// x\n")
        w = lambda n, s: open(os.path.join(t, "src", n), "w").write(s)
        w("a.cpp", "void f(){ state_->covMut() = symmetrized(P);\n state_->covMut() = symmetrized(\n A * B);\n}\n")
        p, n, nb, _ = scan(t, [])
        if p or n != 2:
            fails.append("good tree: %s n=%d" % (p, n))
        w("b.cpp", "void g(){ state_->covMut() = P;\n}\n"); p, *_ = scan(t, [])
        if not any("without symmetrize" in x for x in p): fails.append("bare covMut assignment not caught")
        os.remove(os.path.join(t, "src", "b.cpp"))
        w("c.cpp", "auto Q = 0.5 * (Q0 + Q0.transpose());\n"); p, *_ = scan(t, [])
        if not any("hand-written" in x for x in p): fails.append("hand-written symmetrisation not caught")
        os.remove(os.path.join(t, "src", "c.cpp"))
        # B: covariance-like assignments
        w("d.cpp", "void h(){\n  P_acc = J * C * J.transpose();\n}\n"); p, _, nb, _ = scan(t, [])
        if not (nb == 1 and any("neither symmetrized nor exempt" in x for x in p)): fails.append("B: unsymmetrised product not caught: %s" % p)
        p, _, nb, used = scan(t, ex)
        if p or used != {0}: fails.append("B: exemption not honoured: %s" % p)
        w("d.cpp", "void h(){\n  P_acc = symmetrized(J * C * J.transpose());\n}\n"); p, _, nb, _ = scan(t, [])
        if p or nb != 0: fails.append("B: symmetrised assignment flagged: %s" % p)
        p, *_ = scan(t, ex)
        if not any("stale exemption" in x for x in p): fails.append("B: stale exemption not caught")
        w("d.cpp", "void h(){\n  double s = (J * P * J.transpose()).trace();\n  Gamma_trace = Gamma.trace();\n  os << cov.transpose();\n}\n"); p, *_ = scan(t, [])
        if p: fails.append("B: scalar and stream statements flagged: %s" % p)
        w("d.cpp", "void h(){\n  out.Gamma.noalias() += w * H.transpose() * H;\n}\n"); p, *_ = scan(t, [])
        if not any("neither" in x for x in p): fails.append("B: noalias accumulation not caught")
        w("d.cpp", "void h(){\n  if (a == b) x = y;\n  int n = 3;\n  M3D R = Q.transpose();\n}\n"); p, *_ = scan(t, [])
        if p: fails.append("B: unrelated statements flagged: %s" % p)
        w("d.cpp", "void h(){\n  if (mode == \"full\") {\n    covariance_ = A / N - m * m.transpose();\n  } else {\n    covariance_ = raw;\n  }\n  if (x)\n    cov_end.noalias() += s * V.transpose();\n}\n"); p, _, nb, _ = scan(t, [])
        if nb != 2: fails.append("B: statements after if/else not seen (nb=%d): %s" % (nb, p))
        os.remove(os.path.join(t, "src", "d.cpp")); os.remove(os.path.join(t, "include/livo_recon/utils/algo/symmetrize.h"))
        p, *_ = scan(t, [])
        if "symmetrize.h absent" not in p: fails.append("absent header not caught")
    if fails:
        print("SELFTEST r74_symmetrize_gate FAIL"); [print(" ", f) for f in fails]; return 1
    print("SELFTEST r74_symmetrize_gate PASS"); return 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src"); ap.add_argument("--out"); ap.add_argument("--exemptions", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "r74_symmetrize_exemptions.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    ex = json.load(open(a.exemptions))
    probs, n, nb, used = scan(a.src, ex)
    lines = ["symmetrize gate (R74): %d covMut() assignments checked, %d other covariance-like statements checked, %d exemptions listed (%d used), %d problems" % (n, nb, len(ex), len(used), len(probs))]
    lines += probs
    lines.append("EXEMPTION TABLE (file | needle | reason):")
    lines += ["  %s | %s | %s" % (e["file"], e["needle"], e["reason"]) for e in ex]
    lines.append("FAIL" if probs else "PASS")
    txt = "\n".join(lines) + "\n"
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
