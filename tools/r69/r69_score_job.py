#!/usr/bin/env python3
"""r69_score_job.py -- the listener's `reduce` stage for ONE finished job (planning agent's, R69; not built or run on real data by me).

    python3 tools/r69/r69_score_job.py --job-dir <queue/job> --out-dir <results/job> --stage-out <tmp> --job-id <id>
            --spec <listener_spec.json> --arm-table <arm_table.tsv> --params <bag_params.json> [--tools-dir tools/r69]

Does what post_queue.sh did to ONE arm, but writes only into --stage-out, and archives every raw file the spec will delete:
  1. builds <stage-out>/<job-id>/ (small files copied, big raw files symlinked read-only), writes arm.json;
  2. runs r69_reduce.py (with --params for the arm's sequence), r69_compress.py, bootstrap_reduce.py, p_reduce.py (--gt);
  3. asserts the windows used equal bag_params.json for the sequence (F-305) and records them;
  4. archives every raw.delete file LOSSLESSLY (gzip) to <stage-out>/archive/<name>.gz with sha256 of the raw stream, and writes
     archive/archive.sha256 (sha_manifest of the .gz files) and archive/archive_index.json (raw bytes, raw sha256, gz bytes, ratio,
     rows, lzma ratio of a 4 MB head sample = measured compressibility, F-301);
  5. removes every symlink, zips the arm directory to <stage-out>/arm_bundle.zip (reduced outputs only, no raw), and writes
     <stage-out>/summary_stage.json (schema_version, arm, seq, n_scans, windows, window_check_ok, p_reduce_ok, ate_m, ...).
Exit code 0 only if the reducers ran, the window check passed and every archived file's sha256 was written. Facts only; no interpretation.
"""
import argparse, csv, fnmatch, glob, gzip, hashlib, json, lzma, os, shutil, subprocess, sys, zipfile

SCHEMA = 3
SMALL = ["bootstrap_preparation.txt", "bootstrap_map_seed.txt", "state_reference.txt", "results_lio.txt", "gravity_alignment.txt",
         "imu_first_scans.csv", "calibration_p0.txt", "config.yaml", "engagement.txt", "run.log", "pose_control_first_frame_solve.csv",
         "plane_fit_reasons.csv"]
BIG = ["state_trace.csv", "lio_gt_matched.csv", "imu_cov_growth.csv", "frame_stats.txt", "plane_fit_trace.csv",
       "sensor_stamps.csv", "point_cov_stats.csv", "surface_map_frames.csv", "frame_window.csv", "akf_q.csv", "coupled_scan_status.csv",
       "motion_q_scan.csv", "post_calibration_voxelmap.csv", "lio_update_diag.csv", "live_point_cov.csv",
       "pose_control_first_frame_solve_matrices.txt",
       "residual_geometry.csv", "info_scale.csv", "redundancy_renorm.csv",   # R70 write-only logs
       "residual_dump.csv", "residual_dump_state.csv",                        # R71 per-residual dump (selected frames)
       "joint_knot_all_scans.csv", "joint_knot_scan_summary.csv"]            # coupled knot diagnostics (full diag only)


def find(name, roots):
    for r in roots:
        p = os.path.join(r, name)
        if os.path.isfile(p):
            return p
    for r in roots:
        for dp, _, fn in os.walk(r):
            if "/.listener" in dp or "listener_out" in dp:
                continue
            if name in fn:
                return os.path.join(dp, name)
    return None


def run(cmd, log):
    with open(log, "ab") as lf:
        lf.write(("$ " + " ".join(cmd) + "\n").encode())
        return subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)


def archive(src, dst_dir, name):
    h, n_raw, rows = hashlib.sha256(), 0, 0
    gz = os.path.join(dst_dir, name + ".gz")
    with open(src, "rb") as f, gzip.open(gz, "wb", compresslevel=6) as g:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b); n_raw += len(b); rows += b.count(b"\n"); g.write(b)
    with open(src, "rb") as f:
        head = f.read(4 << 20)
    xz = len(lzma.compress(head, preset=6)) if head else 0
    return {"name": name, "raw_bytes": n_raw, "raw_sha256": h.hexdigest(), "gz_bytes": os.path.getsize(gz), "rows": rows,
            "gz_ratio": round(n_raw / max(1, os.path.getsize(gz)), 3), "head_sample_bytes": len(head),
            "head_lzma_ratio": round(len(head) / max(1, xz), 3) if xz else None}


def sel_frames(df):
    """Frames of the per-residual dump that are RETURNED whole (return budget, rule 5.12a): 1, 2, 3, 5, 100, 480 for sparse maps, 2 and 100 when a frame has
    more than 8,000 residuals (dense maps). Every frame is analysed in every arm and every frame is in the lossless archive."""
    per = df.groupby("frame").size()
    want = [2, 100] if per.max() > 8000 else [1, 2, 3, 5, 100, 480]
    return [f for f in want if f in per.index]


def compact_diag(D, keep_arm=False):
    """Return-sized, numeric-only reductions of the two new per-frame dumps (rule 5.12b; the full files are archived losslessly).
    lio_update_diag_compact.csv.gz: frames 1..60 every iteration, then every 10th frame, last iteration only.
    live_point_cov_compact.csv.gz : frames 1..60 every frame, then every 10th frame."""
    import pandas as pd
    out = {}
    p = os.path.join(D, "lio_update_diag.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        last = df.groupby("frame", as_index=False).tail(1)
        first_it = df.groupby("frame", as_index=False).head(1)
        keep = df[df.frame <= 60]
        # R71: the whole stationary window (frames <= 500) keeps iteration 0 and the last iteration of every frame; the R70 thinning
        # (every 10th frame above 60) left 11 stationary frames and made per-frame LiDAR-only increments untestable.
        mid = pd.concat([first_it, last]).drop_duplicates(["frame", "iter"])
        mid = mid[(mid.frame > 60) & (mid.frame <= 500)]
        late = last[(last.frame > 500) & (last.frame % 10 == 0)]
        c = pd.concat([keep, mid, late]).sort_values(["frame", "iter"])
        c.to_csv(os.path.join(D, "lio_update_diag_compact.csv.gz"), index=False, float_format="%.7g", compression="gzip")
        out["update_diag_rows"] = int(len(df)); out["update_diag_frames"] = int(df.frame.nunique()); out["update_diag_compact_rows"] = int(len(c))
    p = os.path.join(D, "live_point_cov.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        c = df[(df.frame <= 60) | (df.frame % 10 == 0)]
        c.to_csv(os.path.join(D, "live_point_cov_compact.csv.gz"), index=False, float_format="%.7g", compression="gzip")
        out["live_point_cov_rows"] = int(len(df)); out["live_point_cov_compact_rows"] = int(len(c))
    # R70 (F-340, F-320, F-317): small per-frame logs, thinned the same way (all of frames 1..60, then every 10th frame).
    for nm in ("residual_geometry", "info_scale", "redundancy_renorm"):
        p = os.path.join(D, nm + ".csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            fcol = "frame" if "frame" in df.columns else df.columns[0]
            c = df[(df[fcol] <= 500) | (df[fcol] % 10 == 0)]   # R71: all frames of the stationary window
            c.to_csv(os.path.join(D, nm + "_compact.csv.gz"), index=False, float_format="%.7g", compression="gzip")
            out[nm + "_rows"] = int(len(df)); out[nm + "_compact_rows"] = int(len(c))
    # R71: per-residual dump (selected frames, small) is kept whole; coupled knot diagnostics keep the last iteration of every
    # scan up to 500 (the stationary window), then every 10th scan.
    # R71 (F-377): surface census, small and always returned: every 10th frame to 480, every frame 485..620 (the onset window), every 50th after; plus window means.
    p = os.path.join(D, "surface_map_frames.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        if "frame" in df.columns:
            c = df[((df.frame <= 480) & (df.frame % 10 == 0)) | ((df.frame >= 485) & (df.frame <= 620)) | ((df.frame > 620) & (df.frame % 50 == 0))]
            c.to_csv(os.path.join(D, "surface_census_compact.csv.gz"), index=False, float_format="%.7g", compression="gzip")
            num = df.select_dtypes("number")
            rows = []
            for nm, lo, hi in (("stationary_1_480", 1, 480), ("onset_485_620", 485, 620), ("motion_621_end", 621, 10 ** 9)):
                w = num[(df.frame >= lo) & (df.frame <= hi)]
                if len(w):
                    r = w.mean().to_dict(); r.update(window=nm, n_frames=int(len(w))); rows.append(r)
            if rows:
                pd.DataFrame(rows).to_csv(os.path.join(D, "surface_census_windows.csv.gz"), index=False, float_format="%.7g", compression="gzip")
            out["surface_census_rows"] = int(len(df))
    # The per-residual dump is analysed in every arm (r71_analyses) and returned whole only for keep_raw arms (return budget, rule 5.12a);
    # the lossless gzip archive keeps every dump either way.
    p = os.path.join(D, "residual_dump.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        out["residual_dump_rows"] = int(len(df))
        if keep_arm:
            df = df[df.frame.isin(sel_frames(df))]
            df.to_csv(os.path.join(D, "residual_dump_compact.csv.gz"), index=False, float_format="%.9g", compression="gzip")
    p = os.path.join(D, "residual_dump_state.csv")
    if os.path.exists(p) and keep_arm:
        pd.read_csv(p).to_csv(os.path.join(D, "residual_dump_state_compact.csv.gz"), index=False, float_format="%.9g", compression="gzip")
    for nm in ("joint_knot_all_scans", "joint_knot_scan_summary"):
        p = os.path.join(D, nm + ".csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "iteration" in df.columns and "scan_id" in df.columns:
                df = df.sort_values(["scan_id", "iteration"]).groupby(["scan_id"] + (["knot_index"] if "knot_index" in df.columns else []), as_index=False).tail(1)
            sc = "scan_id" if "scan_id" in df.columns else df.columns[0]
            c = df[(df[sc] <= 500) | (df[sc] % 10 == 0)]
            c.to_csv(os.path.join(D, nm + "_compact.csv.gz"), index=False, float_format="%.7g", compression="gzip")
            out[nm + "_rows"] = int(len(df)); out[nm + "_compact_rows"] = int(len(c))
    return out


def r71_analyses(D, T):
    """R71: run the offline analysis tools on the raw logs of this arm (python only; failures are returned, not swallowed)."""
    res = {}
    for nm, cmd, trig in (
        ("residual_info", [sys.executable, os.path.join(T, "r71_residual_info.py"), "dump", "--dump", os.path.join(D, "residual_dump.csv"), "--out", D], "residual_dump.csv"),
        ("crossframe", [sys.executable, os.path.join(T, "r71_residual_info.py"), "cross", "--diag", os.path.join(D, "lio_update_diag.csv"), "--out", D], "lio_update_diag.csv"),
        ("knots", [sys.executable, os.path.join(T, "r71_spline_tools.py"), "knots", "--knots", os.path.join(D, "joint_knot_all_scans.csv"), "--out", D], "joint_knot_all_scans.csv")):
        if os.path.exists(os.path.join(D, trig)):
            res[nm] = run(cmd, os.path.join(D, "score.log"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True); ap.add_argument("--out-dir", required=True); ap.add_argument("--stage-out", required=True)
    ap.add_argument("--job-id", required=True); ap.add_argument("--spec", required=True); ap.add_argument("--arm-table", required=True)
    ap.add_argument("--params", required=True); ap.add_argument("--tools-dir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    roots = [a.out_dir, a.job_dir]
    spec = json.load(open(a.spec))
    raw_names = [p for p in spec["raw"]["delete"] if not p.startswith("dir:") and not any(c in p for c in "*?[")]
    D = os.path.join(a.stage_out, a.job_id); os.makedirs(D, exist_ok=True)   # directory name = arm name (bootstrap_reduce uses it)
    log = os.path.join(D, "score.log")
    ok = True
    problems = []
    # ---- 1. arm dir
    missing = []
    for n in SMALL:
        s = find(n, roots)
        if s:
            shutil.copy(s, os.path.join(D, n))
        elif n not in ("engagement.txt", "pose_control_first_frame_solve.csv", "plane_fit_reasons.csv"):
            missing.append(n)
    for n in BIG:
        s = find(n, roots)
        if s:
            os.symlink(os.path.abspath(s), os.path.join(D, n))
        else:
            missing.append(n)
    sent = "FAILED" if os.path.exists(os.path.join(a.job_dir, "FAILED")) else "DONE"
    rc_run = 1 if sent == "FAILED" else 0
    fin = os.path.join(a.job_dir, sent)
    envp = os.path.join(a.job_dir, "job.env")
    wall = int(os.stat(fin).st_mtime - os.stat(envp).st_mtime) if os.path.exists(envp) and os.path.exists(fin) else None
    ate = None
    try:
        import re
        m = re.search(r"ATE=([0-9.]+)", open(os.path.join(D, "results_lio.txt")).read())
        ate = float(m.group(1)) if m else None
    except Exception:
        pass
    meta = {"arm": a.job_id, "job_dir": a.job_id, "exit_code": rc_run, "missing": missing, "sentinel": sent}
    if wall is not None:
        meta["wall_s"] = wall
    row = None
    with open(a.arm_table) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["arm"] == a.job_id:
                row = r
                meta.update({"tier": r["tier"], "arch": r["arch"], "kind": r["kind"], "duration_s": r["duration_s"], "seq": r["seq"]})
    if row is None:
        problems.append("arm not in arm table"); ok = False
    if ate is not None:
        meta["ate_m"] = ate
    json.dump(meta, open(os.path.join(D, "arm.json"), "w"))
    seq = meta.get("seq", "eee_01")
    bp = json.load(open(a.params))
    want = bp.get(seq)
    if not want:
        problems.append("no bag_params entry for seq %s" % seq); ok = False
    keep = row is not None and row.get("keep_raw") == "1"
    onset = int(want["onset_scan"]) if want else 485
    T = a.tools_dir
    rcs = {}
    # ---- 2. reducers (successful runs only, as post_queue did)
    if rc_run == 0:
        rcs["reduce"] = run([sys.executable, os.path.join(T, "r69_reduce.py"), D, "--params", a.params], log)
        kinds = "stamps,plane_trace,numeric,gt,small_logs" if keep else "stamps,plane_trace,small_logs_min"
        rcs["compress"] = run([sys.executable, os.path.join(T, "r69_compress.py"), D, "--kinds", kinds, "--onset", str(onset)], log)
        if os.path.exists(os.path.join(D, "state_trace.csv")):
            # AGENT FIX (R69): state_trace_blank_runid.sha256 must be written BEFORE the
            # bootstrap_reduce.py subprocess call, not after -- bootstrap_reduce.py <arm_dir>
            # (single-arm mode)'s summarize() reads this exact file to populate the "digest"
            # field of bootstrap_summary.json; with the digest file not yet on disk, every
            # arm's bootstrap_summary.json baked in "digest": null permanently (confirmed
            # live: all G1/G3/G4/G5/G8 gates read "got None" at --collect time, even though
            # state_trace_blank_runid.sha256 was present and correct on disk by then -- the
            # digest field just preserves the stale None from an earlier write, since
            # bootstrap_reduce.py --collect's summarize() short-circuits to an existing
            # bootstrap_summary.json instead of recomputing). Reordered so the digest exists
            # before bootstrap_reduce.py runs.
            h = hashlib.sha256()
            with open(os.path.join(D, "state_trace.csv"), newline="") as f:
                for i, rw in enumerate(csv.reader(f)):
                    if i > 0:
                        rw[0] = ""
                    h.update((",".join(rw) + "\n").encode())
            open(os.path.join(D, "state_trace_blank_runid.sha256"), "w").write(h.hexdigest() + "\n")
            rcs["bootstrap_reduce"] = run([sys.executable, os.path.join(T, "bootstrap_reduce.py"), D], log)
        pdir = os.path.join(D, "P"); os.makedirs(pdir, exist_ok=True)
        rcs["p_reduce"] = run([sys.executable, os.path.join(T, "p_reduce.py"), D, "--out", pdir, "--params", a.params, "--gt"], log)
    else:
        try:
            lines = open(os.path.join(D, "run.log"), errors="replace").read().splitlines()[-40:]
            open(os.path.join(D, "run_log_tail.txt"), "w").write("\n".join(lines) + "\n")
        except Exception:
            pass
    try:
        dg = compact_diag(D, keep)
        for k_, rc_ in r71_analyses(D, T).items():
            dg["analysis_" + k_ + "_rc"] = rc_
            if rc_ != 0:
                problems.append("R71 analysis %s failed rc=%s (see score.log)" % (k_, rc_)); ok = False
    except Exception as e:
        dg = {}
        problems.append("compact_diag failed: %s" % e); ok = False
    # ---- 3. window assertion (F-305)
    S = {}
    try:
        S = json.load(open(os.path.join(D, "arm_summary.json")))
    except Exception:
        pass
    defs = S.get("definitions", {})
    wc = {"expected": {k: want[k] for k in ("onset_scan", "stat_last", "trans_last")} if want else None, "used": defs}
    wok = bool(want) and all(int(defs.get(k, -1)) == int(want[k]) for k in ("onset_scan", "stat_last", "trans_last"))
    if rc_run == 0 and not wok:
        problems.append("window mismatch: used %s expected %s" % (defs, wc["expected"])); ok = False
    print("windows used %s expected %s -> %s" % (defs, wc["expected"], "OK" if wok else "MISMATCH"))
    for k, v in rcs.items():
        if v != 0:
            problems.append("%s exit %d" % (k, v)); ok = False
    # ---- 4. lossless archive of everything the spec will delete
    adir = os.path.join(a.stage_out, "archive"); os.makedirs(adir, exist_ok=True)
    index = []
    for n in raw_names:
        s = find(n, roots)
        if s:
            index.append(archive(s, adir, n))
    with open(os.path.join(adir, "archive.sha256"), "w") as f:
        for e in index:
            hx = hashlib.sha256(open(os.path.join(adir, e["name"] + ".gz"), "rb").read()).hexdigest() if e["gz_bytes"] < (1 << 28) else None
            if hx is None:
                h = hashlib.sha256()
                with open(os.path.join(adir, e["name"] + ".gz"), "rb") as g:
                    for b in iter(lambda: g.read(1 << 22), b""):
                        h.update(b)
                hx = h.hexdigest()
            f.write("%s  %s.gz\n" % (hx, e["name"]))
    json.dump(index, open(os.path.join(adir, "archive_index.json"), "w"), indent=1)
    # ---- 5. bundle (no raw): remove symlinks first
    for n in os.listdir(D):
        p = os.path.join(D, n)
        if os.path.islink(p):
            os.remove(p)
    if os.path.exists(os.path.join(D, "point_cov_stats.csv")):
        pass
    with open(os.path.join(D, "raw_sha256.txt"), "w") as f:
        for e in index:
            f.write("%s  %s\n" % (e["raw_sha256"], e["name"]))
    bundle = os.path.join(a.stage_out, "arm_bundle.zip")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fn in os.walk(D):
            for n in sorted(fn):
                p = os.path.join(dp, n)
                z.write(p, os.path.join(a.job_id, os.path.relpath(p, D)))
    summ = {"schema_version": SCHEMA, "arm": a.job_id, "seq": seq, "n_scans": S.get("n_scans", 0), "first_scan": S.get("first_scan"),
            "last_scan": S.get("last_scan"), "windows_used": defs, "windows_expected": wc["expected"], "window_check_ok": wok,
            "p_reduce_ok": rcs.get("p_reduce", 1) == 0 and os.path.exists(os.path.join(D, "P", "p_window.json")),
            "reducer_exit_codes": rcs, "ate_m": ate, "sentinel": sent, "exit_code": rc_run, "wall_s": wall, "missing_files": missing,
            "diag": dg, "archive_files": len(index), "archive_raw_bytes": sum(e["raw_bytes"] for e in index),
            "archive_gz_bytes": sum(e["gz_bytes"] for e in index), "problems": problems,
            "stage_ok": ok}
    json.dump(summ, open(os.path.join(a.stage_out, "summary_stage.json"), "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
