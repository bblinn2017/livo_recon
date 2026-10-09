#!/usr/bin/env python3
"""R83 (F-506): r83_hilti_metrics.csv from the [evo] line of each Hilti arm. Self-contained copy of the parsing of r82_hilti_reduce.py (no import, so the R82 file need not be present).
    python3 tools/r69/r83_hilti_reduce.py --runs <dir, one folder per arm> --manifest r83_manifest --out r83_hilti_metrics.csv
    python3 tools/r69/r83_hilti_reduce.py --selftest
Differences from R82: the arms are the rows of <manifest>/arm_table.tsv whose id holds `_hl_` (the R83 Hilti smoke arm r83_dc_sm_hl_exp04_60s is group OWN, not HIL); exp comes from the id (exp<N>);
ds, pt, q are read from the id when it has the R82 shape and are empty otherwise. Columns are those of  A metric that is not printed is an EMPTY cell; the exit code is 1 when no arm or an arm has no ATE line.
Planning agent's script; selftest on synthetic text only (planner sandbox, not evidence)."""
import argparse, csv, gzip, os, re, sys, tempfile, shutil

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
    arms = [r["arm"] for r in csv.DictReader(open(os.path.join(a.manifest, "arm_table.tsv")), delimiter="\t") if "_hl_" in r["arm"]]
    rows = []
    for arm in arms:
        d = os.path.join(a.runs, arm)
        m = ID_RE.match(arm.replace("r83_", "r82_", 1)); m = m.groupdict() if m else {}
        e = re.search(r"_hl_(exp\d+)", arm)
        src, line = find_line(d) if os.path.isdir(d) else (None, "")
        row = dict(arm=arm, exp=e.group(1) if e else "", ds=m.get("ds", ""), pt=m.get("pt", ""), q=m.get("q", ""), src=src or "", line=line[:400])
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
        open(os.path.join(mf, "arm_table.tsv"), "w").write("arm\tgroup\nr83_dc_sm_hl_exp04_60s\tOWN\nr83_rt_ds_eee_01_first_fq\tFB\n")
        arm = "r83_dc_sm_hl_exp04_60s"
        os.makedirs(os.path.join(t, "r", arm))
        open(os.path.join(t, "r", arm, "run.log"), "w").write("[evo/gt] old ATE=9.9m\n[evo/file] ATE=0.0301m  RTE=0.012m  ROE=0.5deg  ARE=0.61deg (ROE ...)  ATE_first=0.0412m ARE_first=1.25deg\n")
        out = os.path.join(t, "o.csv")
        rc = run(argparse.Namespace(runs=os.path.join(t, "r"), manifest=mf, out=out))
        R = list(csv.DictReader(open(out)))
        if rc != 0 or len(R) != 1 or R[0]["arm"] != arm or R[0]["exp"] != "exp04":
            fails.append("rows %r rc %s" % (R, rc))
        elif not (R[0]["ate_m"] == "0.0301" and R[0]["rte_m"] == "0.012" and R[0]["roe_deg"] == "0.5" and R[0]["are_deg"] == "0.61" and R[0]["ate_first_m"] == "0.0412" and R[0]["are_first_deg"] == "1.25"):
            fails.append("parse %r" % R[0])
        os.remove(os.path.join(t, "r", arm, "run.log"))
        if run(argparse.Namespace(runs=os.path.join(t, "r"), manifest=mf, out=out)) != 1:
            fails.append("a missing line must exit 1")
        if list(csv.DictReader(open(out)))[0]["ate_m"] != "":
            fails.append("an absent metric must be an empty cell")
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("SELFTEST r83_hilti_reduce", "FAIL " + "; ".join(fails) if fails else "PASS (r83_ smoke arm found by its _hl_ id although its group is OWN; ATE RTE ROE ARE ATE_first ARE_first from the last [evo] ATE line; other arms ignored; a missing line exits 1 with empty cells)")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs"); ap.add_argument("--manifest", default="r83_manifest"); ap.add_argument("--out", default="r83_hilti_metrics.csv"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else run(a))
