#!/usr/bin/env python3
"""R72 (F-400): Stage-1 table for the P0 / Q sweep.

    python3 r72_p_table.py --summary phase_summary.csv --arms r72_manifest/arm_table.tsv --out <dir>

Inputs: phase_summary.csv from r69_collect.py (columns arm, seq?, evo_ate_m, gt_ate_mm, excursion_mm, settled_offset_mm, wander_mm, n_scans, exit_code), arm_table.tsv (arm -> seq, r66_id).
Arm naming: r72_p_<bag>_<cell>, cell = <q>_<name>, q in fq|mq. Reference for a cell = the SAME bag's centre cell of the SAME Q family in THIS round: fq_p0d, mq_p0d
(in-round reference, so no cross-round drift). No noise bar (rule 3.10): every difference is reported as a value; "n_better/n_worse" are counts, not significance.
Writes: r72_p_long.csv (arm x metrics + ratios to the in-round reference), r72_p_cells.csv (per cell: across-bag medians/means/worst of the ATE and excursion ratios, counts,
settled and wander reported), r72_p_winners.json (the ATE winner and the excursion winner per Q family and overall, with the bag-level numbers, for choosing the stage-2 neighbourhood).
Metric definitions: excursion = max_{scans 1..10} ||e_p||; settled = ||mean e_p over scans 50..stat_last||; wander = ||mean(400..480) - mean(50..100)|| (r69_reduce.py).
"""
import argparse, json, os, re, sys
import numpy as np
import pandas as pd

METRICS = ["gt_ate_mm", "evo_ate_m", "excursion_mm", "settled_offset_mm", "wander_mm"]


def parse_arm(a):
    m = re.match(r"r72_p_(eee_\d\d|nya_\d\d|sbs_\d\d)_(fq|mq)_(.+)$", a)
    return (m.group(1), m.group(2), m.group(2) + "_" + m.group(3)) if m else (None, None, None)


def build(S, arms=None):
    S = S.copy()
    P = S[S["arm"].astype(str).str.startswith("r72_p_")].copy()
    if P.empty:
        return P, pd.DataFrame(), {}
    P[["bag", "q", "cell"]] = P["arm"].apply(lambda a: pd.Series(parse_arm(a)))
    P = P[P["bag"].notna()]
    ref = {(r.bag, r.q): r for r in P.itertuples() if r.cell in ("fq_p0d", "mq_p0d")}
    for m in METRICS:
        if m not in P:
            P[m] = np.nan
        P[m + "_ratio"] = [(getattr(r, m) / getattr(ref[(r.bag, r.q)], m)) if (r.bag, r.q) in ref and getattr(ref[(r.bag, r.q)], m) not in (0, None)
                           and pd.notna(getattr(ref[(r.bag, r.q)], m)) else np.nan for r in P.itertuples()]
    P["completed"] = P["n_scans"].notna() & (P["exit_code"].fillna(0) == 0) if "exit_code" in P else P["n_scans"].notna()
    rows = []
    for (q, cell), G in P.groupby(["q", "cell"]):
        r = {"q": q, "cell": cell, "n_bags": int(len(G)), "n_completed": int(G["completed"].sum())}
        for m in ("gt_ate_mm", "excursion_mm"):
            x = G[m + "_ratio"].dropna()
            r[m + "_ratio_median"] = float(x.median()) if len(x) else np.nan
            r[m + "_ratio_mean"] = float(x.mean()) if len(x) else np.nan
            r[m + "_ratio_worst"] = float(x.max()) if len(x) else np.nan
            r[m + "_n_lower"] = int((x < 1).sum()); r[m + "_n_higher"] = int((x > 1).sum())
        for m in ("gt_ate_mm", "excursion_mm", "settled_offset_mm", "wander_mm"):
            r[m + "_median"] = float(G[m].median()) if G[m].notna().any() else np.nan
            r[m + "_max"] = float(G[m].max()) if G[m].notna().any() else np.nan
        rows.append(r)
    C = pd.DataFrame(rows).sort_values(["q", "gt_ate_mm_ratio_median"])
    W = {}
    for key, sub in (("fq", C[C.q == "fq"]), ("mq", C[C.q == "mq"]), ("all", C)):
        sub = sub[sub.n_completed > 0]
        if sub.empty:
            continue
        a = sub.loc[sub["gt_ate_mm_ratio_median"].idxmin()]
        e = sub.loc[sub["excursion_mm_ratio_median"].idxmin()]
        W[key] = {"ate_winner": a["cell"], "ate_ratio_median": float(a["gt_ate_mm_ratio_median"]), "ate_ratio_worst_bag": float(a["gt_ate_mm_ratio_worst"]),
                  "excursion_winner": e["cell"], "excursion_ratio_median": float(e["excursion_mm_ratio_median"]),
                  "excursion_winner_settled_median_mm": float(e["settled_offset_mm_median"]), "ate_winner_settled_median_mm": float(a["settled_offset_mm_median"])}
    return P, C, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True); ap.add_argument("--arms"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P, C, W = build(pd.read_csv(a.summary))
    os.makedirs(a.out, exist_ok=True)
    P.to_csv(os.path.join(a.out, "r72_p_long.csv"), index=False)
    C.to_csv(os.path.join(a.out, "r72_p_cells.csv"), index=False)
    json.dump(W, open(os.path.join(a.out, "r72_p_winners.json"), "w"), indent=1)
    print("p arms %d  cells %d" % (len(P), len(C)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
