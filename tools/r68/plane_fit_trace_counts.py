#!/usr/bin/env python3
"""plane_fit_trace_counts.py -- build plane_fit_trace_counts_all.csv.gz (planning agent; written 2026-09-30, compile- and synthetic-tested only).

    python3 plane_fit_trace_counts.py --runs r67_runs --out R67_RETURN [--arms arms.txt]

Output  <out>/plane_fit_trace_counts_all.csv.gz   columns: arm, path, frame, reason, reason_name, rows
        <out>/plane_fit_trace_counts_status.csv   one row per arm: source used, rows, note
One row per (arm, path, frame, reason) = how many plane-fit ATTEMPT ROWS of that outcome were logged in that frame (path = pca | debiased).

Per arm, the first source that exists is used:
  1. <arm>/plane_fit_trace_counts.csv         (r6x_reduce.py writes it from the raw trace)
  2. <arm>/plane_fit_trace.csv or .csv.gz     (raw trace written by the C++: voxelplane.cpp:609, header frame,path,reason,N,F,...) -- counted in chunks
  3. <arm>/plane_trace_compact.csv.gz         (r6x_compress.py output; has frame,path,reason,rows)
No source = status row "no_source"; nothing is invented.

WHY THE FILE WAS EMPTY IN R66: the raw trace plane_fit_trace.csv is only written when voxel_map/plane/plane_fit_trace_frames > 0
(voxelmap.cpp:218, default 0; fitDiagWanted() in voxelplane.cpp) and NO R66 job set it (only voxel_map.plane.plane_fit_reasons_en, 37 jobs, which writes
plane_fit_reasons.csv -- a different file: per-frame counters, already returned as plane_fit_reasons_all.csv.gz). The reducer code that turns the trace into counts
exists (r66_reduce.py ~line 307); its input was never produced. To get data, add to a job's overrides, e.g.
    "voxel_map.plane.plane_fit_trace_frames": 700, "voxel_map.plane.plane_fit_reasons_en": true
(trace rows are capped at plane_fit_trace_max_rows = 200000 shared over frames 0..N, i.e. about 285 rows per frame for N=700; counts are therefore of LOGGED
rows, not of all fits -- plane_fit_reasons.csv has the complete per-frame counters).
"""
import argparse, gzip, os, sys
import pandas as pd

REASONS = {0: "accept", 1: "n_lt_3", 2: "solver_fail", 3: "eig12_small", 4: "floor_reject", 5: "non_finite",
           6: "threshold", 7: "denom", 8: "ceiling", 9: "info_reject"}   # FitReason enum, voxelplane.cpp


def from_raw(path):
    acc = None
    for ch in pd.read_csv(path, usecols=["frame", "path", "reason"], chunksize=500000):
        g = ch.groupby(["path", "frame", "reason"]).size()
        acc = g if acc is None else acc.add(g, fill_value=0)
    if acc is None:
        return pd.DataFrame(columns=["path", "frame", "reason", "rows"])
    return acc.astype(int).reset_index(name="rows")


def arm_counts(d):
    p = os.path.join(d, "plane_fit_trace_counts.csv")
    if os.path.exists(p):
        return pd.read_csv(p)[["path", "frame", "reason", "rows"]], "plane_fit_trace_counts.csv"
    for fn in ("plane_fit_trace.csv", "plane_fit_trace.csv.gz"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return from_raw(p), fn
    p = os.path.join(d, "plane_trace_compact.csv.gz")
    if os.path.exists(p):
        c = pd.read_csv(p, usecols=["frame", "path", "reason", "rows"])
        return c[["path", "frame", "reason", "rows"]], "plane_trace_compact.csv.gz"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--out", default=".")
    ap.add_argument("--arms", default=None, help="file with one arm directory name per line (default: every subdirectory of --runs)")
    a = ap.parse_args()
    arms = [l.strip() for l in open(a.arms) if l.strip()] if a.arms else sorted(x for x in os.listdir(a.runs) if os.path.isdir(os.path.join(a.runs, x)))
    os.makedirs(a.out, exist_ok=True)
    frames, status = [], []
    for arm in arms:
        d = os.path.join(a.runs, arm)
        try:
            df, src = arm_counts(d)
        except Exception as e:   # report, never hide
            status.append(dict(arm=arm, source="error", rows=0, note=repr(e))); print("ERROR", arm, repr(e), file=sys.stderr); continue
        if df is None:
            status.append(dict(arm=arm, source="no_source", rows=0, note="no plane_fit_trace*.csv in the arm directory (plane_fit_trace_frames unset?)")); continue
        df = df.copy(); df.insert(0, "arm", arm)
        df["reason_name"] = df["reason"].map(REASONS).fillna("unknown")
        frames.append(df[["arm", "path", "frame", "reason", "reason_name", "rows"]])
        status.append(dict(arm=arm, source=src, rows=int(df["rows"].sum()), note=""))
    st = pd.DataFrame(status); st.to_csv(os.path.join(a.out, "plane_fit_trace_counts_status.csv"), index=False)
    n_src = int((st.source.isin(["no_source", "error"]) == False).sum())
    if frames:
        pd.concat(frames).to_csv(os.path.join(a.out, "plane_fit_trace_counts_all.csv.gz"), index=False, compression="gzip")
        print("wrote plane_fit_trace_counts_all.csv.gz: %d of %d arms had a source" % (n_src, len(arms)))
    else:
        print("NOT PRODUCED: 0 of %d arms have a plane_fit_trace source. The raw trace is only written when voxel_map.plane.plane_fit_trace_frames > 0 "
              "(default 0). See the docstring; status file written." % len(arms))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
