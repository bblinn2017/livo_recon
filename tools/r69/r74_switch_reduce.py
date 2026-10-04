#!/usr/bin/env python3
"""R74 (F-426): switch-arm comparison and the per-frame table of the existing shadow files.

    python3 r74_switch_reduce.py part1  --shadow-dirs <dir of R73 shadow arms> --pairs r74_manifest/switch_pairs.json --out <dir>
    python3 r74_switch_reduce.py part2  --runs <R74 runs dir> --controls <dir with the R73 plain-control arms> --pairs r74_manifest/switch_pairs.json --out <dir>

part1 (no new run): for pairs a, b, d, e, the step difference |step_shadow - step_driver| (mm) of every frame of the window, with the onset scan marked (column `at_or_after_onset`),
      and the max / median over the frames before and from the onset. Input: shadow_scan_all.csv.gz (or shadow_scan.csv) of the R73 shadow arm, columns frame, role, start_pos_*, post_pos_*.
part2: the switch arm ran config A up to onset-1 and config B from the onset on. Per frame (scan_id) of the window, alignment-free:
      step_X(f) = pos_X(f) - pos_X(f-1) (mm, from pos_trace_all.csv.gz, matched by scan_id);
      dA(f) = |step_switch - step_ctlA| and dB(f) = |step_switch - step_ctlB|;  before the onset the switch arm must follow A (dA ~ 0), from the onset it either follows B
      (update fails on the healthy state: dB small) or A (needs accumulated history: dB large).  Facts only: max and median of dA and dB over frames before and from the onset, per arm; no verdict threshold.
Time columns are not used; the key is scan_id.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def rd(d, *names):
    for n in names:
        p = os.path.join(d, n)
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


def summ(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (float(v.max()), float(np.median(v)), int(len(v))) if len(v) else (np.nan, np.nan, 0)


def part1(a):
    pairs = json.load(open(a.pairs))
    out, tabs = [], []
    for pr in pairs:
        if pr.get("kind") != "switch":
            continue
        S = rd(os.path.join(a.shadow_dirs, pr["shadow_arm_r73"]), "shadow_scan_all.csv.gz", "shadow_scan.csv")
        if S is None:
            out.append(dict(pair=pr["pair"], note="MISSING shadow_scan")); continue
        lo, hi = pr["window"]; on = pr["onset"]
        d, s = S[S["role"] == "driver"].set_index("frame"), S[S["role"] == "shadow"].set_index("frame")
        fr = sorted(f for f in set(d.index) & set(s.index) if lo <= f <= hi)
        cols = lambda D, p: D.loc[fr, [p + "_x", p + "_y", p + "_z"]].to_numpy(float)
        sd, ss = (cols(d, "post_pos") - cols(d, "start_pos")) * 1e3, (cols(s, "post_pos") - cols(s, "start_pos")) * 1e3
        diff = np.linalg.norm(ss - sd, axis=1)
        T = pd.DataFrame({"pair": pr["pair"], "frame": fr, "step_diff_mm": diff, "at_or_after_onset": [int(f >= on) for f in fr]})
        tabs.append(T)
        b, aft = diff[np.array(fr) < on], diff[np.array(fr) >= on]
        out.append(dict(pair=pr["pair"], window_lo=lo, window_hi=hi, onset=on, n_before=len(b), n_from_onset=len(aft),
                        max_before_mm=summ(b)[0], median_before_mm=summ(b)[1], max_from_onset_mm=summ(aft)[0], median_from_onset_mm=summ(aft)[1],
                        max_frame=int(fr[int(np.argmax(diff))]) if len(fr) else None))
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(out).to_csv(os.path.join(a.out, "r74_f426_part1_summary.csv"), index=False, float_format="%.9g")
    if tabs:
        pd.concat(tabs).to_csv(os.path.join(a.out, "r74_f426_part1_frames.csv"), index=False, float_format="%.9g")
    return 0


def steps_by_scan(P, frames):
    P = P.drop_duplicates("scan_id").set_index("scan_id")
    f = np.asarray(frames, int); ok = np.isin(f, P.index.to_numpy()); pos = np.full((len(f), 3), np.nan)
    pos[ok] = P.loc[f[ok], ["px", "py", "pz"]].to_numpy(float)
    st = np.full((len(f), 3), np.nan); st[1:] = (pos[1:] - pos[:-1]) * 1e3
    return st


def part2(a):
    pairs = json.load(open(a.pairs))
    out, tabs = [], []
    for pr in pairs:
        if pr.get("kind") != "switch":
            continue
        P = rd(os.path.join(a.runs, pr["switch_arm"]), "pos_trace_all.csv.gz")
        A = rd(os.path.join(a.controls, pr["control_before"]), "pos_trace_all.csv.gz")
        B = rd(os.path.join(a.controls, pr["control_after"]), "pos_trace_all.csv.gz")
        if P is None or A is None or B is None:
            out.append(dict(pair=pr["pair"], switch_arm=pr["switch_arm"], note="MISSING pos_trace_all.csv.gz of the switch arm or a control")); continue
        lo, hi = pr["window"]; on = pr["onset"]
        fr = np.arange(lo, hi + 1)
        s0, sa, sb = steps_by_scan(P, fr), steps_by_scan(A, fr), steps_by_scan(B, fr)
        dA, dB = np.linalg.norm(s0 - sa, axis=1), np.linalg.norm(s0 - sb, axis=1)
        dAB = np.linalg.norm(sa - sb, axis=1)
        T = pd.DataFrame({"pair": pr["pair"], "switch_arm": pr["switch_arm"], "frame": fr, "dA_mm": dA, "dB_mm": dB, "dAB_mm": dAB, "at_or_after_onset": (fr >= on).astype(int)})
        tabs.append(T)
        bef, aft = fr < on, fr >= on
        r = dict(pair=pr["pair"], switch_arm=pr["switch_arm"], control_before=pr["control_before"], control_after=pr["control_after"], onset=on, window_lo=lo, window_hi=hi)
        for nm, v in (("dA", dA), ("dB", dB), ("dAB", dAB)):
            mx, md, n = summ(v[bef]); r.update({nm + "_before_max_mm": mx, nm + "_before_median_mm": md})
            mx, md, n2 = summ(v[aft]); r.update({nm + "_from_onset_max_mm": mx, nm + "_from_onset_median_mm": md})
        r["n_before"], r["n_from_onset"] = int(np.isfinite(dA[bef]).sum()), int(np.isfinite(dA[aft]).sum())
        out.append(r)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(out).to_csv(os.path.join(a.out, "r74_f426_part2_summary.csv"), index=False, float_format="%.9g")
    if tabs:
        pd.concat(tabs).to_csv(os.path.join(a.out, "r74_f426_part2_frames.csv"), index=False, float_format="%.9g")
    return 0


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p1 = sp.add_parser("part1"); p1.add_argument("--shadow-dirs", required=True); p1.add_argument("--pairs", required=True); p1.add_argument("--out", required=True)
    p2 = sp.add_parser("part2"); p2.add_argument("--runs", required=True); p2.add_argument("--controls", required=True); p2.add_argument("--pairs", required=True); p2.add_argument("--out", required=True)
    a = ap.parse_args()
    return part1(a) if a.cmd == "part1" else part2(a)


if __name__ == "__main__":
    sys.exit(main())
