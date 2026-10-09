#!/usr/bin/env python3
"""R82 (F-506): r82_hilti_metrics.csv from the [evo] line of each Hilti arm. Planning agent's script; selftest on synthetic text only (planner sandbox, not evidence).
    python3 tools/r69/r82_hilti_reduce.py --runs <dir, one folder per arm> --manifest r82_manifest --out r82_hilti_metrics.csv
    python3 tools/r69/r82_hilti_reduce.py --selftest
Arms: rows of <manifest>/arm_table.tsv with group HIL. Per arm the LAST line of engagement_imu.txt / run_log_tail.txt / run.log (first file that has one) matching '[evo...]' and 'ATE=' is parsed
for ATE, RTE, ROE, ARE (the shapes of the existing [evo] line), ATE_first, ARE_first (F-506, appended at the end). Columns: arm, exp, ds, pt, q, ate_m, rte_m, roe_deg, are_deg, ate_first_m, are_first_deg, src, line.
A metric that is not printed is left EMPTY (the gate HILCOMP then FAILs it); nothing is guessed. exp/ds/pt/q come from the arm id r82_dc_hl_<exp>_ds<first|off>_pt<off|on>_<fq|mq>."""
import argparse, csv, os, re, sys, gzip, tempfile, shutil

NUM = r"(-?[0-9]+(?:\.[0-9]*)?(?:[eE][-+]?[0-9]+)?)"
ID_RE = re.compile(r"^r82_dc_hl_(?P<exp>exp\d+)_ds(?P<ds>first|off)_pt(?P<pt>off|on)_(?P<q>fq|mq)$")
COLS = ["arm", "exp", "ds", "pt", "q", "ate_m", "rte_m", "roe_deg", "are_deg", "ate_first_m", "are_first_deg", "src", "line"]
FILES = ("engagement_imu.txt", "run_log_tail.txt", "run.log")


def rd(p):
    for q in (p, p + ".gz"):
        if os.path.exists(q):
            return (gzip.open(q, "rt", errors="replace") if q.endswith(".gz") else open(q, errors="replace")).read()
    return None


def parse(line):
    g = lambda k: (re.search(r"(?<![A-Za-z_])%s=%s" % (k, NUM), line) or [None, ""])[1]
    return dict(ate_m=g("ATE"), rte_m=g("RTE"), roe_deg=g("ROE"), are_deg=g("ARE"), ate_first_m=g("ATE_first"), are_first_deg=g("ARE_first"))


def find_line(d):
    for fn in FILES:
        t = rd(os.path.join(d, fn))
        if not t:
            continue
        ls = [l for l in t.splitlines() if re.search(r"\[evo[^\]]*\]", l) and re.search(r"(?<![A-Za-z_])ATE=", l)]
        if ls:
            return fn, ls[-1].strip()
    return None, ""


def run(a):
    arms = [r["arm"] for r in csv.DictReader(open(os.path.join(a.manifest, "arm_table.tsv")), delimiter="\t") if r.get("group") == "HIL"]
    rows = []
    for arm in arms:
        d = os.path.join(a.runs, arm)
        m = ID_RE.match(arm); m = m.groupdict() if m else {}
        src, line = find_line(d) if os.path.isdir(d) else (None, "")
        row = dict(arm=arm, exp=m.get("exp", ""), ds=m.get("ds", ""), pt=m.get("pt", ""), q=m.get("q", ""), src=src or "", line=line[:400])
        row.update(parse(line) if line else {k: "" for k in COLS[5:11]})
        rows.append(row)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(rows)
    miss = [r["arm"] for r in rows if not r["ate_m"]]
    print("hilti arms %d, without an [evo] ATE line %d %s" % (len(rows), len(miss), miss[:5]))
    return 0 if rows and not miss else 1


def selftest():
    t = tempfile.mkdtemp(); fails = []
    try:
        mf = os.path.join(t, "m"); os.makedirs(mf)
        with open(os.path.join(mf, "arm_table.tsv"), "w") as f:
            f.write("arm\tgroup\nr82_dc_hl_exp04_dsfirst_ptoff_fq\tHIL\nr82_dc_hl_exp04_dsoff_pton_mq\tHIL\nr82_rt_ds_eee_01_off_fq\tF503\n")
        for arm, extra in (("r82_dc_hl_exp04_dsfirst_ptoff_fq", "  ATE_first=0.0412m ARE_first=1.25deg (first-pose alignment, secondary)"), ("r82_dc_hl_exp04_dsoff_pton_mq", "")):
            os.makedirs(os.path.join(t, "r", arm))
            open(os.path.join(t, "r", arm, "run.log"), "w").write("[evo/gt] old ATE=9.9m\n[evo/file] ATE=0.0301m  RTE=0.012m  ROE=0.5deg  ARE=0.61deg (ROE ...)" + extra + "\n")
        out = os.path.join(t, "o.csv")
        rc = run(argparse.Namespace(runs=os.path.join(t, "r"), manifest=mf, out=out))
        R = {r["arm"]: r for r in csv.DictReader(open(out))}
        a, b = R["r82_dc_hl_exp04_dsfirst_ptoff_fq"], R["r82_dc_hl_exp04_dsoff_pton_mq"]
        if not (a["ate_m"] == "0.0301" and a["rte_m"] == "0.012" and a["roe_deg"] == "0.5" and a["are_deg"] == "0.61" and a["ate_first_m"] == "0.0412" and a["are_first_deg"] == "1.25"):
            fails.append("parse %r" % a)
        if b["ate_first_m"] != "" or b["ate_m"] != "0.0301":
            fails.append("absent ATE_first must be empty: %r" % b)
        if (a["exp"], a["ds"], a["pt"], a["q"]) != ("exp04", "first", "off", "fq") or "F503" in str(R):
            fails.append("id parse / group filter")
        os.remove(os.path.join(t, "r", "r82_dc_hl_exp04_dsoff_pton_mq", "run.log"))
        if run(argparse.Namespace(runs=os.path.join(t, "r"), manifest=mf, out=out)) != 1:
            fails.append("a missing line must exit 1")
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("SELFTEST r82_hilti_reduce", "FAIL " + "; ".join(fails) if fails else "PASS (ATE RTE ROE ARE ATE_first ARE_first parsed from the last [evo] ATE line; an absent metric is an empty cell; non-HIL rows ignored; a missing line exits 1)")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs"); ap.add_argument("--manifest", default="r82_manifest"); ap.add_argument("--out", default="r82_hilti_metrics.csv"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else run(a))
