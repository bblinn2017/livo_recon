#!/usr/bin/env python3
"""stage1b_checks.py -- facts about the new files, read from the RAW files of the finished smoke jobs (planning agent's, R70). No interpretation.

    python3 tools/r69/stage1b_checks.py --spec r71_manifest/listener_spec_smoke.json --root . --out R71_STAGE1B/checks

Run it AFTER the dry-run pass of the smoke jobs and BEFORE the deleting pass (it needs the raw files; the archives are checked by the listener's verify stage).
Writes <out>/stage1b_checks.md and stage1b_checks.json. Each check is independent; a failing check writes `NOT PRODUCED: <reason>` and the others continue.

C1 frame_stats: first two lines (truncated), field counts of the header and of the first data row by whitespace and by comma, total rows.
C2 new files: first 3 lines and row counts of lio_update_diag.csv and live_point_cov.csv (diag-on job), set of frames, iterations per frame (min/max/histogram), last frame.
C3 K=1 vs K=5: the frame-1 blocks of pose_control_first_frame_solve_matrices.txt of the diag-on job (K=5) against the whole K=1 file of the baseline 60 s job. Reported
   byte-for-byte AND with the `plane_id=` field masked (it is a heap address, expected to differ between processes). Number of differing lines and the first 3 differing lines.
C4 HtH/Htz cross-check: block n of frame 1 in the matrices dump against row n of frame 1 in lio_update_diag.csv (HtH upper triangle, Htz): max abs and max relative difference.
C5 refuse jobs: first 3 lines of run.log matching refus|renamed|removed key|error|exception, and the last line.
C6 timing (S7): mtime of DONE and of every file in each smoke job's out dir, in seconds relative to DONE.
C7 first-block counts: `=== iekf_solve_snapshot ===` blocks per frame_idx in the K=5 dump, size of the dump.
"""
import argparse, collections, json, os, re, sys, time


def first_lines(p, n):
    out = []
    with open(p, errors="replace") as f:
        for _ in range(n):
            l = f.readline()
            if not l:
                break
            out.append(l.rstrip("\n"))
    return out


def blocks(p):
    """list of (frame_idx, iteration, text lines) of every iekf_solve_snapshot block"""
    res, cur = [], None
    with open(p, errors="replace") as f:
        for line in f:
            if line.startswith("=== iekf_solve_snapshot ==="):
                cur = [line]
            elif cur is not None:
                cur.append(line)
                if line.startswith("=== end_iekf_solve_snapshot ==="):
                    fr = it = None
                    for l in cur[:12]:
                        if l.startswith("frame_idx="): fr = int(l.split("=")[1])
                        if l.startswith("iteration="): it = int(l.split("=")[1])
                    res.append((fr, it, cur)); cur = None
    return res


def mat(lines, name):
    for i, l in enumerate(lines):
        if l.startswith("matrix %s " % name):
            m = re.search(r"rows=(\d+) cols=(\d+)", l)
            r, c = int(m.group(1)), int(m.group(2))
            return [[float(x) for x in lines[i + 1 + k].split()] for k in range(r)]
    return None


def vec(lines, name):
    for i, l in enumerate(lines):
        if l.startswith("vector %s " % name):
            return [float(x) for x in lines[i + 1].split()]
    return None


def mask(lines):
    return [re.sub(r"plane_id=\d+", "plane_id=MASKED", l) for l in lines]


def csv_rows(p):
    with open(p, errors="replace") as f:
        hdr = f.readline().rstrip("\n").split(",")
        rows = [l.rstrip("\n").split(",") for l in f if l.strip()]
    return hdr, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True); ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True); ap.add_argument("--listener-dir", default="scripts/parallel")
    ap.add_argument("--diag-job", default="r71_diag_on_baseline_60s"); ap.add_argument("--k1-job", default="00_id_baseline_60s")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.root, a.listener_dir))
    import sweep_listener as sl
    spec = sl.load_spec(a.spec, a.root)
    jobs = {}
    for q in spec["queues"]:
        qd = sl.absp(spec, q["dir"])
        if os.path.isdir(qd):
            for n in sorted(os.listdir(qd)):
                if os.path.isdir(os.path.join(qd, n)):
                    jobs[n] = sl.Job(spec, q, n)
    os.makedirs(a.out, exist_ok=True)
    R, md = {}, ["# stage-1b checks (facts only)\n"]

    def run(tag, title, fn):
        md.append("\n## %s %s\n" % (tag, title))
        try:
            r = fn()
            R[tag] = r
            md.extend(r.get("md", []))
        except Exception as e:
            R[tag] = {"error": repr(e)}
            md.append("NOT PRODUCED: %r\n" % (e,))

    def f_in(job, name):
        for base in (job.out_dir, job.job_dir):
            p = os.path.join(base, name)
            if os.path.exists(p):
                return p
        raise FileNotFoundError("%s not in %s or %s" % (name, job.out_dir, job.job_dir))

    def c1():
        p = f_in(jobs[a.k1_job], "frame_stats.txt")
        L = first_lines(p, 2)
        hdr, row = L[0], L[1] if len(L) > 1 else ""
        n = sum(1 for _ in open(p, errors="replace"))
        cnt = lambda s: dict(whitespace=len(s.split()), comma=len(s.split(",")))
        return {"file": p, "header": cnt(hdr), "row1": cnt(row), "lines": n,
                "md": ["file: `%s`; lines %d" % (p, n), "header fields %s; first data row fields %s" % (cnt(hdr), cnt(row)), "header (first 300 chars): `%s`" % hdr[:300], "row1 (first 300 chars): `%s`" % row[:300]]}

    def c2():
        D = jobs[a.diag_job]; out = {"md": []}
        for nm, key in (("lio_update_diag.csv", "frame,iter"), ("live_point_cov.csv", "frame,n_pts")):
            p = f_in(D, nm); hdr, rows = csv_rows(p)
            fr = [int(float(r[0])) for r in rows]
            per = collections.Counter(fr)
            hist = collections.Counter(per.values())
            out[nm] = {"rows": len(rows), "header_fields": len(hdr), "first_frame": min(fr), "last_frame": max(fr), "distinct_frames": len(per), "rows_per_frame_hist": dict(hist)}
            out["md"] += ["**%s**: %d rows, %d header fields, frames %d..%d (%d distinct), rows-per-frame histogram %s" % (nm, len(rows), len(hdr), min(fr), max(fr), len(per), dict(hist))]
            out["md"] += ["```"] + first_lines(p, 3) + ["```"]
        return out

    def c3():
        K1 = f_in(jobs[a.k1_job], "pose_control_first_frame_solve_matrices.txt")
        K5 = f_in(jobs[a.diag_job], "pose_control_first_frame_solve_matrices.txt")
        b1 = blocks(K1); b5 = [b for b in blocks(K5) if b[0] == 1]
        l1 = [l for b in b1 for l in b[2]]; l5 = [l for b in b5 for l in b[2]]
        raw_eq = (l1 == l5); m1, m5 = mask(l1), mask(l5)
        diff = [(i, x, y) for i, (x, y) in enumerate(zip(m1, m5)) if x != y]
        raw_diff = sum(1 for x, y in zip(l1, l5) if x != y)
        r = {"k1_file": K1, "k5_file": K5, "k1_blocks": len(b1), "k5_frame1_blocks": len(b5), "k1_lines": len(l1), "k5_frame1_lines": len(l5),
             "bytes_equal_raw": raw_eq, "differing_lines_raw": raw_diff + abs(len(l1) - len(l5)), "differing_lines_plane_id_masked": len(diff) + abs(len(l1) - len(l5))}
        r["md"] = ["K=1 file `%s`: %d blocks, %d lines. K=5 file `%s`: %d frame-1 blocks, %d lines." % (K1, len(b1), len(l1), K5, len(b5), len(l5)),
                   "byte-for-byte equal: **%s**; differing lines raw %d; differing lines with `plane_id=` masked %d" % (raw_eq, r["differing_lines_raw"], r["differing_lines_plane_id_masked"])]
        for i, x, y in diff[:3]:
            r["md"] += ["first differences (masked): line %d  K1: `%s`  K5: `%s`" % (i, x[:200].rstrip(), y[:200].rstrip())]
        return r

    def c4():
        K5 = f_in(jobs[a.diag_job], "pose_control_first_frame_solve_matrices.txt")
        hdr, rows = csv_rows(f_in(jobs[a.diag_job], "lio_update_diag.csv"))
        ix = {h: i for i, h in enumerate(hdr)}
        d1 = [r for r in rows if int(float(r[ix["frame"]])) == 1]
        b1 = [b for b in blocks(K5) if b[0] == 1]
        out = {"matrices_blocks_frame1": len(b1), "diag_rows_frame1": len(d1), "per_iteration": [], "md": []}
        out["md"].append("frame 1: %d matrix blocks, %d lio_update_diag rows; matrix block `iteration=` values %s; diag `iter` values %s" %
                         (len(b1), len(d1), [b[1] for b in b1], [int(float(r[ix["iter"]])) for r in d1]))
        for n, (b, r) in enumerate(zip(b1, d1)):
            H = mat(b[2], "HtH"); z = vec(b[2], "Htz")
            mx_a = mx_r = 0.0
            for i in range(6):
                for j in range(i, 6):
                    dv = float(r[ix["HtH_%d%d" % (i, j)]]); mv = H[i][j]
                    mx_a = max(mx_a, abs(dv - mv)); mx_r = max(mx_r, abs(dv - mv) / max(abs(mv), 1e-300))
            for i in range(6):
                dv = float(r[ix["Htz_%d" % i]]); mv = z[i]
                mx_a = max(mx_a, abs(dv - mv)); mx_r = max(mx_r, abs(dv - mv) / max(abs(mv), 1e-300))
            sym = max(abs(H[i][j] - H[j][i]) for i in range(6) for j in range(6))
            out["per_iteration"].append({"n": n, "max_abs": mx_a, "max_rel": mx_r, "HtH_asym": sym})
            out["md"].append("block %d: max |diag - matrices| abs %.3e, relative %.3e (HtH + Htz, 27 numbers); matrices HtH max asymmetry %.3e" % (n, mx_a, mx_r, sym))
        return out

    def c5():
        out = {"md": []}
        for n, j in sorted(jobs.items()):
            if not n.startswith("r71_refuse"):
                continue
            p = None
            for base in (j.out_dir, j.job_dir):
                if os.path.exists(os.path.join(base, "run.log")):
                    p = os.path.join(base, "run.log"); break
            if not p:
                out["md"].append("%s: NOT PRODUCED: no run.log" % n); continue
            lines = open(p, errors="replace").read().splitlines()
            hits = [l for l in lines if re.search(r"(?i)refus|renamed|removed key|error|exception", l)][:3]
            out[n] = {"first_matches": hits, "last_line": lines[-1] if lines else ""}
            out["md"] += ["**%s** (`%s`, %d lines)" % (n, p, len(lines))] + ["- `%s`" % h[:300] for h in hits] + ["- last line: `%s`" % (lines[-1][:300] if lines else "")]
        return out

    def c6():
        out = {"md": []}
        for n, j in sorted(jobs.items()):
            d = os.path.join(j.job_dir, "DONE")
            if not os.path.exists(d):
                d = os.path.join(j.job_dir, "FAILED")
            if not os.path.exists(d):
                out["md"].append("%s: no sentinel" % n); continue
            t0 = os.stat(d).st_mtime; rel = {}
            for base in (j.out_dir, j.job_dir):
                if os.path.isdir(base):
                    for fn in os.listdir(base):
                        p = os.path.join(base, fn)
                        if os.path.isfile(p):
                            rel[fn] = round(os.stat(p).st_mtime - t0, 3)
            late = {k: v for k, v in rel.items() if v > 0}
            out[n] = {"sentinel": os.path.basename(d), "files_after_sentinel_s": late, "max_after_s": max(late.values()) if late else 0.0}
            out["md"].append("%s: sentinel `%s`; files modified after it: %s (max %.3f s); settle window in the spec %s s" % (n, os.path.basename(d), late or "none", out[n]["max_after_s"], spec["trigger"]["settle_seconds"]))
        return out

    def c7():
        K5 = f_in(jobs[a.diag_job], "pose_control_first_frame_solve_matrices.txt")
        b = blocks(K5); per = collections.Counter(x[0] for x in b)
        return {"blocks_per_frame": dict(per), "bytes": os.path.getsize(K5),
                "md": ["blocks per frame_idx: %s; file size %d bytes" % (dict(sorted(per.items())), os.path.getsize(K5))]}

    def c8():
        out = {"md": []}
        for jn, files in ((a.diag_job, ("residual_geometry.csv",)),
                          ("r71_smoke_block_f12_2_60s", ("info_scale.csv", "engagement.txt")),
                          ("r71_smoke_block_f10_10_60s", ("info_scale.csv",)),
                          ("r71_smoke_renorm_60s", ("redundancy_renorm.csv", "engagement.txt")),
                          ("r71_smoke_surface0p425_60s", ("surface_map_frames.csv",))):
            if jn not in jobs:
                out["md"].append("%s: job not in the queues" % jn); continue
            for fnm in files:
                try:
                    p = f_in(jobs[jn], fnm)
                except Exception as e:
                    out["md"].append("%s / %s: NOT PRODUCED %r" % (jn, fnm, e)); continue
                if fnm.endswith(".csv"):
                    hdr, rows = csv_rows(p)
                    out[jn + "/" + fnm] = {"rows": len(rows), "header_fields": len(hdr), "bytes": os.path.getsize(p)}
                    out["md"] += ["**%s / %s**: %d rows, %d header fields, %d bytes" % (jn, fnm, len(rows), len(hdr), os.path.getsize(p)), "```"] + first_lines(p, 3) + ["```"]
                else:
                    out["md"] += ["**%s / %s** (lines mentioning info_scale / renorm / INERT):" % (jn, fnm), "```"] + [l.rstrip() for l in open(p, errors="replace") if any(k in l for k in ("info_scale", "renorm", "INERT"))][:12] + ["```"]
        return out

    DUMP_FRAMES = [1, 2, 3, 5, 10, 50, 100, 200, 400, 480]    # must equal the frames set on r71_smoke_dump_on_baseline_60s in make_r71_manifest.py

    def c9():
        """R71 residual dump: requested frames present, per-frame row count, Lambda rebuilt from the dump equals HtH of lio_update_diag at iteration 0
        (relative < 1e-5), CONTROLS: a frame list with a frame beyond the run reports it absent, the same comparison against another frame's HtH FAILS, and a
        job with the key unset has no dump file (so the checks can fail)."""
        import numpy as np
        out = {"md": []}
        jn = "r71_smoke_dump_on_baseline_60s"
        if jn not in jobs:
            return {"md": ["%s: job not in the queues" % jn]}
        dump = f_in(jobs[jn], "residual_dump.csv"); st = f_in(jobs[jn], "residual_dump_state.csv"); ud = f_in(jobs[jn], "lio_update_diag.csv")
        hdr, rows = csv_rows(dump); ci = {h: i for i, h in enumerate(hdr)}
        frames = sorted(set(int(r[ci["frame"]]) for r in rows))
        out["md"].append("requested %s; present %s; missing %s; rows %d" % (DUMP_FRAMES, frames, sorted(set(DUMP_FRAMES) - set(frames)), len(rows)))
        uh, ur = csv_rows(ud); ui = {h: i for i, h in enumerate(uh)}
        hk = ["HtH_%d%d" % (i, j) for i in range(6) for j in range(i, 6)]
        def lam_dump(f):
            R = [r for r in rows if int(r[ci["frame"]]) == f]
            H = np.array([[float(r[ci[c]]) for c in ("hrot_x", "hrot_y", "hrot_z", "hpos_x", "hpos_y", "hpos_z")] for r in R]); w = 1.0 / np.array([float(r[ci["sigma_squared"]]) for r in R])
            return (H * w[:, None]).T @ H, len(R)
        def lam_diag(f):
            r = next(x for x in ur if int(x[ui["frame"]]) == f and int(x[ui["iter"]]) == 0)
            L = np.zeros((6, 6)); k = 0
            for i in range(6):
                for j in range(i, 6):
                    L[i, j] = L[j, i] = float(r[ui[hk[k]]]); k += 1
            return L, int(float(r[ui["n_resid"]]))
        worst = 0.0; ctl = []
        for f in frames:
            A, n1 = lam_dump(f); B, n2 = lam_diag(f)
            rel = float(np.abs(A - B).max() / np.abs(B).max()); worst = max(worst, rel)
            out["md"].append("frame %d: rows in dump %d, n_resid in update diag %d, max relative |Lambda_dump - HtH| %.3e" % (f, n1, n2, rel))
        if len(frames) >= 2:
            A, _ = lam_dump(frames[0]); B, _ = lam_diag(frames[-1])
            ctl = float(np.abs(A - B).max() / np.abs(B).max())
            out["md"].append("CONTROL (must be large): Lambda of frame %d against HtH of frame %d: relative %.3e" % (frames[0], frames[-1], ctl))
        out["worst_rel"] = worst
        jb = "r71_smoke_dump_frame_beyond_60s"
        if jb in jobs:
            h2, r2 = csv_rows(f_in(jobs[jb], "residual_dump.csv")); c2 = h2.index("frame")
            fr2 = sorted(set(int(r[c2]) for r in r2))
            out["md"].append("CONTROL frame list [1, 99999]: frames present %s (99999 must be absent, 1 present)" % fr2)
        for jz in ("00_id_baseline_60s",):
            if jz in jobs:
                present = os.path.exists(os.path.join(jobs[jz].out_dir, "residual_dump.csv")) or os.path.exists(os.path.join(jobs[jz].job_dir, "residual_dump.csv"))
                out["md"].append("CONTROL key unset (%s): residual_dump.csv present = %s (must be False)" % (jz, present))
        return out

    def c10():
        out = {"md": []}
        jn = "r71_smoke_surface0p425_60s"
        want = ["n_cells_pts_1_2", "n_cells_pts_3_to_min", "n_cells_inv_secondary_eig", "n_cells_inv_plane_threshold", "n_cells_inv_denom", "n_cells_inv_var_ceiling"]
        if jn in jobs:
            hdr, rows = csv_rows(f_in(jobs[jn], "surface_map_frames.csv"))
            out["md"].append("%s surface_map_frames.csv: new census columns present %s; header ends with %s; rows %d" % (jn, [w in hdr for w in want], hdr[-6:], len(rows)))
            ci = {h: i for i, h in enumerate(hdr)}
            if all(w in ci for w in want) and "n_cells_lt_min_points" in ci:
                bad1 = sum(1 for r in rows if int(r[ci["n_cells_lt_min_points"]]) != int(r[ci["n_cells_pts_1_2"]]) + int(r[ci["n_cells_pts_3_to_min"]]))
                bad2 = sum(1 for r in rows if int(r[ci["n_cells_fit_invalid"]]) != sum(int(r[ci[w]]) for w in want[2:]))
                out["md"].append("census identities over %d rows: lt_min_points == pts_1_2 + pts_3_to_min fails in %d rows; fit_invalid == sum of the four inv columns fails in %d rows (both must be 0)" % (len(rows), bad1, bad2))
        jc = "r71_co_tail_fulldiag_60s"
        if jc in jobs:
            hdr, rows = csv_rows(f_in(jobs[jc], "joint_knot_all_scans.csv"))
            sc = sorted(set(int(r[hdr.index("scan_id")]) for r in rows))
            out["md"].append("%s joint_knot_all_scans.csv: absolute-state columns present %s; rows %d; distinct scans %d (first %s, last %s; must exceed 1)" % (jc, [c in hdr for c in ("after_v_x", "after_bg_x", "after_ba_x", "after_theta_x")], len(rows), len(sc), sc[:1], sc[-1:]))
        return out

    run("C1", "frame_stats field counts (K=1 baseline 60 s job)", c1)
    run("C2", "first rows and counts of the new files (diagnostics-on job)", c2)
    run("C3", "frame-1 blocks of the K=5 dump against the K=1 dump", c3)
    run("C4", "HtH / Htz in lio_update_diag against the matrices dump", c4)
    run("C5", "refuse jobs: first error lines", c5)
    run("C6", "timing of files after the sentinel (S7)", c6)
    run("C7", "blocks per frame in the K=5 dump", c7)
    run("C8", "new R71 files: rows, headers, engagement lines", c8)
    run("C9", "residual dump: frames, Lambda against update diag, controls", c9)
    run("C10", "surface census columns and coupled knot absolute columns", c10)
    open(os.path.join(a.out, "stage1b_checks.md"), "w").write("\n".join(md) + "\n")
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "md"} for k, v in R.items()}, open(os.path.join(a.out, "stage1b_checks.json"), "w"), indent=1, default=str)
    print("\n".join(md))


main()
