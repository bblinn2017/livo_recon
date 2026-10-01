#!/usr/bin/env python3
"""R73 (F-400 Stage 2): tables for the motion-Q scale grid (r73_p2_<bag>_mq_a<acc>_g<gyro>, centre control r73_p2_<bag>_mq_a0p3_g10).

    python3 r73_p_table.py --summary phase_summary.csv --compact compact --params r69_inputs/bag_params.json --out r73_p_table

Per bag the reference is the SAME bag's centre control of THIS round (acc 0.3, gyro 10). Rule 4.12/4.15: no noise bar; every difference is a value; counts are counts.
Collapse timing is ALIGNMENT-FREE: displacement(scan) = || pos_arm(scan) - pos_control(scan) || from pos_trace_all.csv.gz (posterior position, same scan ids, same start state, no alignment).
  transition_disp_max_mm = max displacement over the bag's transition window (stat_last+1 .. trans_last from bag_params.json); disp_max_mm over the whole run;
  collapse_first_scan = first scan with displacement >= 500 mm (None if none); collapsed = transition_disp_max_mm >= 500 mm (the F-400 rule fixed before the data).
A collapsed arm is REPORTED (own row, own numbers) and is NOT RANKED: it is excluded from the cell medians and from the winners; n_collapsed is a column of the cell table.
Writes r73_p2_long.csv, r73_p2_cells.csv (per cell, over non-collapsed completed bags: n_bags, n_collapsed, median/mean/worst ratio of gt_ate_mm, excursion_mm, settled ratio, the transition
displacement median/max), r73_p2_winners.json (per metric the cell with the smallest median ratio over >= 5 non-collapsed bags, and that cell's worst-bag ratio; NO composite score).
"""
import argparse, json, os, re, sys
import numpy as np
import pandas as pd

METRICS = ["gt_ate_mm", "excursion_mm", "settled_offset_mm", "wander_mm"]
COLLAPSE_MM = 500.0
MIN_BAGS = 5


def parse_arm(a):
    m = re.match(r"r73_p2_(eee_\d\d|nya_\d\d|sbs_\d\d)_mq_a(\w+?)_g(\d+)$", a)
    return (m.group(1), float(m.group(2).replace("p", ".")), float(m.group(3))) if m else (None, None, None)


def disp(arm_pos, ctl_pos):
    a = arm_pos.set_index("scan_id")[["px", "py", "pz"]]
    c = ctl_pos.set_index("scan_id")[["px", "py", "pz"]]
    j = a.join(c, lsuffix="_a", rsuffix="_c", how="inner")
    d = np.linalg.norm(j[["px_a", "py_a", "pz_a"]].to_numpy(float) - j[["px_c", "py_c", "pz_c"]].to_numpy(float), axis=1) * 1e3
    return pd.Series(d, index=j.index)


def load_pos(compact, arm):
    for n in ("pos_trace_all.csv.gz", "pos_trace_all.csv"):
        p = os.path.join(compact, arm, n)
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


def build(S, compact, params):
    P = S[S["arm"].astype(str).str.startswith("r73_p2_")].copy()
    if P.empty:
        return P, pd.DataFrame(), {}
    P[["bag", "acc", "gyro"]] = P["arm"].apply(lambda a: pd.Series(parse_arm(a)))
    P = P[P["bag"].notna()].copy()
    P["cell"] = ["a%g_g%g" % (r.acc, r.gyro) for r in P.itertuples()]
    ctl = {r.bag: r for r in P.itertuples() if (r.acc, r.gyro) == (0.3, 10.0)}
    for m in METRICS:
        if m not in P:
            P[m] = np.nan
        P[m + "_ratio"] = [(getattr(r, m) / getattr(ctl[r.bag], m)) if r.bag in ctl and pd.notna(getattr(ctl[r.bag], m)) and getattr(ctl[r.bag], m) != 0 else np.nan for r in P.itertuples()]
    P["completed"] = P["n_scans"].notna() & (P["exit_code"].fillna(0) == 0) if "exit_code" in P else P["n_scans"].notna()
    tdm, dm, cfs = [], [], []
    for r in P.itertuples():
        pa, pc = load_pos(compact, r.arm), load_pos(compact, ctl[r.bag].arm) if r.bag in ctl else None
        w = (params or {}).get(r.bag, {})
        if pa is None or pc is None or not w:
            tdm.append(np.nan); dm.append(np.nan); cfs.append(None); continue
        d = disp(pa, pc)
        tr = d[(d.index > int(w["stat_last"])) & (d.index <= int(w["trans_last"]))]
        tdm.append(float(tr.max()) if len(tr) else np.nan); dm.append(float(d.max()) if len(d) else np.nan)
        hit = d[d >= COLLAPSE_MM]
        cfs.append(int(hit.index[0]) if len(hit) else None)
    P["transition_disp_max_mm"], P["disp_max_mm"], P["collapse_first_scan"] = tdm, dm, cfs
    P["collapsed"] = P["transition_disp_max_mm"] >= COLLAPSE_MM
    P["ranked"] = P["completed"] & ~P["collapsed"].fillna(False) & P["transition_disp_max_mm"].notna()
    rows = []
    for (acc, gyro), G in P.groupby(["acc", "gyro"]):
        R = G[G["ranked"]]
        r = {"cell": "a%g_g%g" % (acc, gyro), "acc": acc, "gyro": gyro, "n_bags": int(len(G)), "n_completed": int(G["completed"].sum()), "n_collapsed": int(G["collapsed"].fillna(False).sum()), "n_ranked": int(len(R))}
        for m in ("gt_ate_mm", "excursion_mm", "settled_offset_mm"):
            x = R[m + "_ratio"].dropna()
            r[m + "_ratio_median"] = float(x.median()) if len(x) else np.nan
            r[m + "_ratio_mean"] = float(x.mean()) if len(x) else np.nan
            r[m + "_ratio_worst"] = float(x.max()) if len(x) else np.nan
            r[m + "_n_lower"] = int((x < 1).sum()); r[m + "_n_higher"] = int((x > 1).sum())
        r["transition_disp_median_mm"] = float(R["transition_disp_max_mm"].median()) if len(R) else np.nan
        r["transition_disp_max_mm"] = float(R["transition_disp_max_mm"].max()) if len(R) else np.nan
        rows.append(r)
    C = pd.DataFrame(rows).sort_values(["acc", "gyro"]).reset_index(drop=True)
    W = {"rule": "per metric: the cell with the smallest median ratio over >= %d non-collapsed bags; its worst-bag ratio and n_collapsed are reported; no composite score" % MIN_BAGS}
    ok = C[(C["n_ranked"] >= MIN_BAGS) & ~((C["acc"] == 0.3) & (C["gyro"] == 10.0))]
    for m in ("gt_ate_mm", "excursion_mm", "settled_offset_mm"):
        if ok[m + "_ratio_median"].notna().any():
            b = ok.loc[ok[m + "_ratio_median"].idxmin()]
            W[m] = {"cell": b["cell"], "median_ratio": float(b[m + "_ratio_median"]), "worst_bag_ratio": float(b[m + "_ratio_worst"]), "n_ranked": int(b["n_ranked"]), "n_collapsed": int(b["n_collapsed"])}
    return P, C, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True); ap.add_argument("--compact", required=True); ap.add_argument("--params", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default=None)
    a = ap.parse_args()
    P, C, W = build(pd.read_csv(a.summary), a.compact, json.load(open(a.params)))
    os.makedirs(a.out, exist_ok=True)
    P.to_csv(os.path.join(a.out, "r73_p2_long.csv"), index=False)
    C.to_csv(os.path.join(a.out, "r73_p2_cells.csv"), index=False)
    json.dump(W, open(os.path.join(a.out, "r73_p2_winners.json"), "w"), indent=1)
    print("p2 arms %d  cells %d" % (len(P), len(C)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
