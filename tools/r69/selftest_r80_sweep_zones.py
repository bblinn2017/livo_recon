#!/usr/bin/env python3
"""Selftest of r80_sweep_zones.py on synthetic summaries (no real data). python3 selftest_r80_sweep_zones.py"""
import os, sys, tempfile, subprocess
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r80_sweep_zones as Z

fails = []
def chk(c, m):
    if not c:
        fails.append(m)

# tags and grid
chk([Z.edge_tag(c) for c in (20, 35, 100, 50)] == ["0p2", "0p35", "1p0", "0p5"], "edge tags")
ex = Z.expected_arms()
chk(len(ex) == 9 * 2 * 2 * 2 * 17 and len({e[0] for e in ex}) == len(ex), "grid size/uniqueness")
chk(all(Z.ARM_RE.match(e[0]) for e in ex), "arm ids match the naming contract")

def write_pos(root, arm, drift_mm_per_scan, n=800, yaw=0.0):
    d = os.path.join(root, "compact", arm); os.makedirs(d, exist_ok=True)
    s = np.arange(1, n + 1)
    pd.DataFrame({"scan_id": s, "px": drift_mm_per_scan * 1e-3 * np.maximum(s - 485, 0), "py": 0.0, "pz": 0.0, "yaw_rad": yaw * (s <= 10) * (s / 10.0)}).to_csv(
        os.path.join(d, "pos_trace_all.csv.gz"), index=False)

with tempfile.TemporaryDirectory() as T:
    ret = os.path.join(T, "ret"); out = os.path.join(T, "out"); os.makedirs(ret)
    # cell A: eee_01 fq off pfn3, ATE by edge (cm): best 0.40 = 30.0; 0.35 = 34.0 (in zone); 0.45 = 35.0 (in zone, exactly +5); 0.50 = 35.001 (out); 0.30 = 20.0 but COLLAPSED (not ranked)
    ate = {c: 60.0 + c * 0.1 for c in Z.EDGES_CM}
    ate.update({40: 30.0, 35: 34.0, 45: 35.0, 50: 35.001, 30: 20.0})
    rows = []
    for c in Z.EDGES_CM:
        a = Z.arm_id("eee_01", "off", c, 3, "fq")
        if c == 70:
            continue                                   # missing arm
        gm = 600.0 if c == 30 else 100.0
        rows.append(dict(arm=a, gt_ate_mm=ate[c], gt_max_mm=gm, excursion_mm=1.5, st_lio_nees9_mean=200.0, settled_offset_mm=3.0, onset_scan_used=485))
        write_pos(ret, a, 10.0 if c == 30 else 0.0, yaw=0.01)    # 10 mm/scan from 485: 215 scans -> 2150 mm departure
    # a large-gt_max arm with small departure is NOT collapsed (AND rule), still ranked; and one with no trace is undetermined
    a = Z.arm_id("eee_01", "off", 25, 3, "fq"); rows = [r for r in rows if r["arm"] != a]
    rows.append(dict(arm=a, gt_ate_mm=10.0, gt_max_mm=900.0, onset_scan_used=485)); write_pos(ret, a, 0.0)       # departure 0 -> not collapsed, best ATE 10
    a = Z.arm_id("eee_01", "off", 95, 3, "fq"); rows = [r for r in rows if r["arm"] != a]
    os.remove(os.path.join(ret, "compact", a, "pos_trace_all.csv.gz"))
    rows.append(dict(arm=a, gt_ate_mm=5.0, gt_max_mm=900.0, onset_scan_used=485))                                  # no trace -> undetermined, not ranked even though ATE is lowest
    # cell B: ties. eee_01 mq hardabs pfn1: edges 0.50 and 0.60 both 40.0 -> best_edge 0.5 ; 0.55 = 41 in zone
    for c, v in ((50, 40.0), (55, 41.0), (60, 40.0)):
        rows.append(dict(arm=Z.arm_id("eee_01", "hardabs", c, 1, "mq"), gt_ate_mm=v, gt_max_mm=80.0, onset_scan_used=485))
    # reference (R79 name) is a flat trajectory
    write_pos(ret, "r79_nb_eee_01_off_v0p425_pfn3_fq", 0.0)
    # an unexpected r80_sw_ row, and a row with no ATE
    rows.append(dict(arm="r80_sw_eee_01_off_v0p425_pfn3_fq", gt_ate_mm=1.0, gt_max_mm=1.0))
    rows.append(dict(arm=Z.arm_id("eee_02", "off", 20, 3, "fq"), gt_ate_mm=np.nan, gt_max_mm=np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(ret, "phase_summary.csv"), index=False)

    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "r80_sweep_zones.py"), "--return-dir", ret, "--out", out], capture_output=True, text=True)
    chk(r.returncode == 0, "CLI failed: " + r.stderr[-400:])
    Zc = pd.read_csv(os.path.join(out, "r80_sweep_zones.csv"), dtype=str, keep_default_na=False)
    All = pd.read_csv(os.path.join(out, "r80_sweep_all.csv"), keep_default_na=False)
    chk(len(Zc) == 72 and len(All) == 1224, "cell/arm counts %d %d" % (len(Zc), len(All)))
    cell = lambda b, q, l, p: Zc[(Zc.bag == b) & (Zc.q == q) & (Zc.line == l) & (Zc.pfn == str(p))].iloc[0]
    A_ = cell("eee_01", "fq", "off", 3)
    # ranked: 0.25 (10.0, gt_max 900 but departure 0 -> not collapsed) is best. zone = within 5 mm of 10 -> only 0.25
    chk(A_.best_edge == "0.25" and A_.best_ate_mm == "10.000" and A_.zone_edges == "0.25" and A_.zone_width_edges == "1", "cell A best/zone: %s" % A_.to_dict())
    chk(A_.n_collapsed == "1" and A_.n_undetermined == "1", "cell A collapsed/undetermined counts %s %s" % (A_.n_collapsed, A_.n_undetermined))
    chk(A_.incomplete == "1" and A_.missing_edges == "0.7" and A_.n_present == "16", "cell A missing 0.7 reported")
    # second evaluation with the 0.25 override removed isolates the known best 0.40 / zone {0.35,0.40,0.45}
    P = pd.read_csv(os.path.join(ret, "phase_summary.csv")).set_index("arm")
    P = P.drop(Z.arm_id("eee_01", "off", 25, 3, "fq"))
    A2, Z2 = Z.score(P, ret, ret, Z.DEFAULT_REF, 5.0)
    z = Z2[(Z2.bag == "eee_01") & (Z2.q == "fq") & (Z2.line == "off") & (Z2.pfn == 3)].iloc[0]
    chk(z.best_edge == "0.4" and z.zone_edges == "0.35;0.4;0.45" and z.zone_width_edges == 3, "known zone: %s" % z.to_dict())
    chk(z.n_collapsed == 1 and z.n_missing == 2, "known zone counts (0.25 now missing, 0.7 missing): %s" % z.to_dict())
    c30 = A2[A2.arm == Z.arm_id("eee_01", "off", 30, 3, "fq")].iloc[0]
    chk(c30.status == "COLLAPSED" and c30.in_zone == "n/a" and abs(c30.departure_mm - 2150.0) < 1e-6, "collapsed arm excluded (ATE 20.0 would have won): %s" % c30.to_dict())
    chk(A2[A2.arm == Z.arm_id("eee_01", "off", 50, 3, "fq")].iloc[0].in_zone == "no", "0.50 at +5.001 is out of the zone")
    chk(A2[A2.arm == Z.arm_id("eee_01", "off", 45, 3, "fq")].iloc[0].in_zone == "yes", "0.45 at exactly +5 is in the zone (inclusive)")
    y = A2[A2.arm == Z.arm_id("eee_01", "off", 40, 3, "fq")].iloc[0].yaw_move_mrad
    chk(abs(y - 9.0) < 1e-6, "yaw movement (0.01 rad at scan 10 minus 0.001 rad at scan 1) -> 9 mrad, got %s" % y)
    # tie
    B_ = cell("eee_01", "mq", "hardabs", 1)
    chk(B_.best_edge == "0.5" and B_.zone_edges == "0.5;0.55;0.6" and B_.zone_contiguous == "1" and B_.incomplete == "1" and B_.n_present == "3", "tie handling: %s" % B_.to_dict())
    # cell with nothing
    N_ = cell("sbs_03", "mq", "off", 1)
    chk(N_.best_edge == "" and N_.zone_width_edges == "0" and N_.n_missing == "17", "empty cell reported, not dropped")
    # missing statuses in the all table
    chk((All.status == "MISSING_NO_ATE").sum() == 1 and (All.status == "MISSING_NO_ROW").sum() > 1000, "missing statuses")
    # txt outputs
    ctxt = open(os.path.join(out, "r80_sweep_collapsed.txt")).read()
    chk(Z.arm_id("eee_01", "off", 30, 3, "fq") in ctxt and Z.arm_id("eee_01", "off", 95, 3, "fq") in ctxt.split("UNDETERMINED")[1], "collapsed.txt lists collapsed and undetermined")
    mtxt = open(os.path.join(out, "r80_sweep_missing.txt")).read()
    chk("MISSING eee_01 fq off pfn3" in mtxt and "UNEXPECTED" in mtxt and "r80_sw_eee_01_off_v0p425_pfn3_fq" in mtxt, "missing.txt")
    W = pd.read_csv(os.path.join(out, "r80_sweep_ate_by_edge.csv"), dtype=str, keep_default_na=False)
    chk(W.shape == (72, 4 + 17), "wide shape %s" % (W.shape,))
    w = W[(W.bag == "eee_01") & (W.q == "fq") & (W.line == "off") & (W.pfn == "3")].iloc[0]
    chk(w["0.30"] == "20.000*" and w["0.70"] == "MISSING" and w["0.95"] == "5.000?" and w["0.40"] == "30.000", "wide marks: %s" % w.to_dict())
    chk("5 mm" in open(os.path.join(out, "r80_sweep_provenance.txt")).read() or "tol_mm = 5 mm" in open(os.path.join(out, "r80_sweep_provenance.txt")).read(), "provenance states tol")
    # determinism: run again, identical bytes
    out2 = os.path.join(T, "out2"); Z.run(ret, out2)
    chk(open(os.path.join(out, "r80_sweep_zones.csv")).read() == open(os.path.join(out2, "r80_sweep_zones.csv")).read(), "deterministic zones")
    # collapse_status unit
    chk(Z.collapse_status(499.9, 9999) == "no" and Z.collapse_status(500.1, 499.9) == "no" and Z.collapse_status(500.1, 500.0) == "yes"
        and Z.collapse_status(900, np.nan) == "undetermined" and Z.collapse_status(np.nan, 1) == "undetermined", "collapse_status AND rule")

if fails:
    print("FAIL"); [print(" -", f) for f in fails]; sys.exit(1)
print("selftest_r80_sweep_zones: PASS")
