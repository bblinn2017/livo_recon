#!/usr/bin/env python3
"""R80D: alignment-free per-arm summary (no global alignment anywhere; ATE is not used because it moves with the transition, rule 4.12 items 6 and 11).

    python3 tools/r69/r80d_alignfree.py --root <dir, one folder per r80d arm with pos_trace_all.csv.gz> --ref-root <R80 compact dir, folders r80_<src>> \
        --phase-summary <R80 phase_summary.csv> --out alignfree_summary.csv [--extra-ref-root <R80C compact dir>]
    python3 tools/r69/r80d_alignfree.py --selftest

Per arm (src = the R80 arm it re-runs: rs -> r80d_rs_<src>; off -> mgfq = mg_<bag>_fq, b3mq = b3_<bag>_mq; e3 -> r80d_e3_<rt|rtt>_<src>):
  onset_scan       : onset_scan_used of the R80 phase_summary row of r80_<src> (first scan after the stationary start)
  stat_drift_mm    : max over scans 1..onset-1 of |p(scan) - p(scan 1)|, millimetres, positions as written by the run (no alignment, no rotation, no scaling)
  stat_settle_mm   : |mean p over scans 1..onset-1 - p(scan 1)| (a drift of the average, same frame)
  dep_ctl_mm       : max over joined scans of | (p - p(first joined)) - (q - q(first joined)) | where q = the R80 control arm r80_<src> (same data, discard policy); joined on scan_id, never on t_abs
  dep_ctl_scan50   : first scan_id whose departure from the control is >= 50 mm (blank if none)
  dep_ref_mm       : the same departure from the healthy reference r80_b3_<bag>_mq (R80 pfn 3 motion-Q baseline; for a b3 src, r80_mg_<bag>_fq)
  n_scans, n_joined: scans in the arm, scans joined to the control
  dep_flag         : 1 when dep_ref_mm >= 500 (rule 4.15c needs a second condition, largest error > 500 mm against ground truth, which this tool does not have: read it from phase_summary gt_max_mm)
The numbers are positions only. Planning agent's script; run by me only on synthetic traces.
"""
import argparse, csv, gzip, math, os, re, sys
import numpy as np

BAG = r"[a-z]+_\d\d"


def src_of(arm):
    m = re.match(r"r80d_rs_(.+)$", arm)
    if m: return "rs", m.group(1)
    m = re.match(r"r80d_off_(mgfq|b3mq)_(%s)_o(\d{3})$" % BAG, arm)
    if m: return "off", ("mg_%s_fq" if m.group(1) == "mgfq" else "b3_%s_mq") % m.group(2)
    m = re.match(r"r80d_e3_(?:rtt|rt)_(.+)$", arm)
    if m: return "e3", m.group(1)
    return None, None


def read_trace(path):
    op = gzip.open if path.endswith(".gz") else open
    ids, xyz = [], []
    with op(path, "rt", newline="") as f:
        for r in csv.DictReader(f):
            ids.append(int(r["scan_id"])); xyz.append((float(r["px"]), float(r["py"]), float(r["pz"])))
    return np.array(ids, dtype=np.int64), np.array(xyz, dtype=np.float64)


def departure(ia, pa, ib, pb):
    """-> (max departure mm, first scan >= 50 mm or None, n joined). Start offset removed at the first joined scan."""
    common = np.intersect1d(ia, ib)
    if len(common) < 2:
        return float("nan"), None, int(len(common))
    A = pa[np.searchsorted(ia, common)]; B = pb[np.searchsorted(ib, common)]
    d = 1000.0 * np.linalg.norm((A - A[0]) - (B - B[0]), axis=1)
    hit = np.nonzero(d >= 50.0)[0]
    return float(d.max()), (int(common[hit[0]]) if len(hit) else None), int(len(common))


def summarize(arm, ia, pa, onset, ctl, ref):
    fam, src = src_of(arm)
    bag = re.search(BAG, src).group(0)
    ons = int(onset)
    k = int(np.searchsorted(ia, ons))                  # scans with scan_id < onset
    st = pa[:k] if k >= 2 else pa[:2]
    drift = 1000.0 * float(np.linalg.norm(st - st[0], axis=1).max())
    settle = 1000.0 * float(np.linalg.norm(st.mean(axis=0) - st[0]))
    dc, sc, nj = departure(ia, pa, *ctl)
    dr, _, _ = departure(ia, pa, *ref)
    return dict(arm=arm, fam=fam, src=src, bag=bag, onset_scan=ons, n_scans=int(len(ia)), n_joined=nj, stat_drift_mm=round(drift, 3), stat_settle_mm=round(settle, 3),
                dep_ctl_mm=round(dc, 3), dep_ctl_scan50="" if sc is None else sc, dep_ref_mm=round(dr, 3), dep_flag=int(dr >= 500.0))


COLS = ["arm", "fam", "src", "bag", "onset_scan", "n_scans", "n_joined", "stat_drift_mm", "stat_settle_mm", "dep_ctl_mm", "dep_ctl_scan50", "dep_ref_mm", "dep_flag"]


def ref_src(src):
    bag = re.search(BAG, src).group(0)
    return "mg_%s_fq" % bag if src.startswith("b3_") else "b3_%s_mq" % bag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--ref-root"); ap.add_argument("--phase-summary"); ap.add_argument("--out", default="alignfree_summary.csv"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not (a.root and a.ref_root and a.phase_summary):
        sys.exit("--root, --ref-root and --phase-summary are required")
    ons = {r["arm"]: r["onset_scan_used"] for r in csv.DictReader(open(a.phase_summary))}
    arms = sorted(d for d in os.listdir(a.root) if d.startswith("r80d_") and os.path.isdir(os.path.join(a.root, d)))
    rows, bad = [], []
    cache = {}

    def ref_trace(src):
        if src not in cache:
            p = os.path.join(a.ref_root, "r80_" + src, "pos_trace_all.csv.gz")
            if not os.path.exists(p):
                sys.exit("missing reference trace %s" % p)
            cache[src] = read_trace(p)
        return cache[src]
    for arm in arms:
        fam, src = src_of(arm)
        p = os.path.join(a.root, arm, "pos_trace_all.csv.gz")
        if fam is None or not os.path.exists(p):
            bad.append(arm); continue
        key = "r80_" + src
        if key not in ons or ons[key] in ("", None):
            sys.exit("no onset_scan_used for %s in %s" % (key, a.phase_summary))
        ia, pa = read_trace(p)
        rows.append(summarize(arm, ia, pa, float(ons[key]), ref_trace(src), ref_trace(ref_src(src))))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(rows)
    print("alignfree rows %d, arms skipped (bad name or no pos_trace) %d %s" % (len(rows), len(bad), bad[:5]))
    sys.exit(1 if bad else 0)


def selftest():
    n = 600; ids = np.arange(1, n + 1)
    base = np.zeros((n, 3)); base[300:, 0] = 0.01 * np.arange(n - 300)           # still, then moves in x
    arm_p = base.copy(); arm_p[:, 1] += np.where(ids <= 100, 0.002 * np.sin(ids / 7.0), 0.0)   # 2 mm wobble while stationary
    arm_p += np.array([5.0, 5.0, 0.0])                                      # a different start offset: must not matter
    d, s, nj = departure(ids, arm_p, ids, base)
    assert nj == n and d < 3.0 and s is None, (d, s, nj)
    arm_q = arm_p.copy(); arm_q[400:, 0] += 0.2                             # 200 mm step at scan 401
    d, s, nj = departure(ids, arm_q, ids, base)
    assert abs(d - 200.0) < 5.0 and s == 401, (d, s)
    # joined on scan_id, not on position in the file: drop scans 10..19 from the arm
    keep = (ids < 10) | (ids >= 20)
    d2, _, nj2 = departure(ids[keep], arm_q[keep], ids, base)
    assert nj2 == n - 10 and abs(d2 - 200.0) < 5.0
    r = summarize("r80d_rs_mg_eee_01_fq", ids, arm_q, 301, (ids, base), (ids, base))
    assert r["fam"] == "rs" and r["src"] == "mg_eee_01_fq" and r["bag"] == "eee_01" and 1.0 < r["stat_drift_mm"] < 3.0 and r["dep_flag"] == 0 and r["dep_ctl_scan50"] == 401, r
    big = arm_p.copy(); big[450:, 0] += 1.0
    assert summarize("r80d_off_b3mq_nya_02_o085", ids, big, 301, (ids, base), (ids, base))["dep_flag"] == 1
    assert src_of("r80d_off_b3mq_nya_02_o085") == ("off", "b3_nya_02_mq") and src_of("r80d_e3_rtt_mg_eee_03_fq") == ("e3", "mg_eee_03_fq") and src_of("junk") == (None, None)
    assert ref_src("b3_nya_02_mq") == "mg_nya_02_fq" and ref_src("b1_eee_01_fq") == "b3_eee_01_mq"
    print("selftest: start offset ignored; 200 mm step reported as 200 mm at scan 401; join on scan_id with missing scans; collapse flag; name mapping")


if __name__ == "__main__":
    main()
