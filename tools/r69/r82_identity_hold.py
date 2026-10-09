#!/usr/bin/env python3
"""R82 identity hold: run by the coding agent AFTER queue r82_id has drained and BEFORE any other queue is dispatched. Planning agent's script; selftest on synthetic data only (planner sandbox, not evidence).

    python3 tools/r69/r82_identity_hold.py --runs <dir with one folder per r82_id arm> --manifest r82_manifest --ref-return R80_RETURN --ref-return R80C_RETURN \
        [--ref-runs <dir with <ref arm>/run.log and state_trace.csv, if the old host archive is still on the node>] [--ref-manifest r80_manifest --ref-manifest r80c_manifest] \
        [--hold-file ablations/_job_queue/HOLD.txt] [--tol-mm 0.005] [--out checks/r82_identity_hold.txt] [--arms <arm>=<ref>,<arm>=<ref>]
    python3 tools/r69/r82_identity_hold.py --selftest

Identity arms: the rows of <manifest>/arm_table.tsv with kind == id (columns arm, ref) unless --arms is given. SPEC: (1) r82_dc_id_mg_eee_01_fq against R80 r80_mg_eee_01_fq, (2) r82_rt_id_mg_eee_01_fq against the R80C
retime_trim arm of the same src (the ref column of the table names it). Reference data comes from the old returns (--ref-return, repeatable; the first one that has the ref arm is used):
  <ret>/phase_summary.csv (row of the ref arm: excursion_mm, settled_offset_mm), <ret>/compact/<ref>/pos_trace_all.csv.gz (or <ret>/<ref>/pos_trace_all.csv.gz), <ret>/completeness.csv
  (column state_trace_sha256_run_id_blanked = the digest of state_trace.csv with the run_id column blanked).
Arm data comes from <runs>/<arm>/: pos_trace_all.csv.gz (else state_trace.csv, post_lio rows), arm_summary.json (excursion_mm, settled_offset_mm), state_trace_blank_runid.sha256, run.log, state_trace.csv.
Checks per arm (tolerance --tol-mm, default 0.005 mm, the project tolerance of F-419):
  POS  every scan of the reference exists in the arm and the largest |position difference| is <= tol (yaw difference is printed as INFO: no yaw tolerance has been decided by Bryce)
  EXC  |excursion_mm - reference| <= tol        SET  |settled_offset_mm - reference| <= tol
  DIG  state_trace digest equals the reference digest WHEN BOTH exist (otherwise INFO). Note: the decision of F-419 (2026-10-01) made digest equality a fact, not a verdict; this tool follows the R82 spec
       text ("plus digest equality where a digest exists") and says in its FAIL line when every metric is within tolerance and only the digest differs.
A check that cannot be made (a file is absent) is MISSING and counts as FAIL: an identity that cannot be shown is not a pass. The tool prints PASS or FAIL per arm and an overall line, exits 0 only when every
identity arm is PASS, and exits 1 otherwise. On FAIL it writes HOLD.txt (--hold-file; default ablations/_job_queue/HOLD.txt when that directory exists, else ./HOLD.txt) with: the arm, its reference, every check,
the FIRST DIVERGING SCAN (first scan where the position differs by more than tol in the pos trace; and, when both raw state_trace.csv files are available, the first scan and column where any state value differs at all),
and the CONFIG DIFFERENCES it can detect: (a) every new R82 key in the arm's run.log that is not at its default, (b) the effective-config lines (key = value and /ns/key: value) that differ from the reference arm's run.log
(--ref-runs), (c) the override keys that differ between the arm's job and the reference arm's job (job files of this manifest and of --ref-manifest dirs). It never deletes, moves or cancels anything and never removes an
existing HOLD.txt (only a person, on the planner's word, releases a hold; r82_hold_check.sh ignores a renamed HOLD.released_* file).
"""
import argparse, csv, glob, gzip, io, json, os, re, shutil, sys, tempfile, time
import numpy as np
import pandas as pd

TOL_MM = 0.005
# every new R82 key and its default (inert at the default, rule 1.7): key (slash form) -> list of values that mean "at default / absent"
NEW_KEY_DEFAULTS = {
    "voxel_map/plane/plane_var_min": ["0", "0.0", "0.0e+00", "0.000000", None],
    "voxel_map/plane/plane_var_scan_s2": ["0", "0.0", "0.0e+00", "0.000000", None],
    "lio/residual_dump_voxel_info": ["false", "0", None],
    "imu/process_noise/gain/mode": ["off", None],       # nis_smooth is the new value; the identity arms use gain off (any other mode is reported)
    "imu/process_noise/gain/nis_slope": [None],
    "imu/process_noise/gain/nis_clip": [None],
    "lio/time_cov/mode": ["off", None],
    "lio/time_cov/sigma_s": [None],
    "voxel_map/plane/plane_var_diag_en": ["false", "0", None],
    "voxel_map/plane/line_residual/classify_at_fit": ["false", "0", None],
    "voxel_map/plane/debiased_line_equiv": ["off", None],
    "lio/zupt/threshold_mode": ["fixed", None],
    "evo/dense_gt_metrics": ["false", "0", None],
    "cbk/lidar/point_time_mode": ["offset_ns_u32", None],
}
IGNORE_KEYS = re.compile(r"(^|[./])(outputs[./](path|debug_log_dir)|eval[./]state_trace_run_id|results_root|job_id|log_dir|debug_log_dir|run_id)$|timestamp|launch_time|wall")


def rd_text(path):
    if os.path.exists(path):
        return (gzip.open(path, "rt", errors="replace") if path.endswith(".gz") else open(path, errors="replace")).read()
    if os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt", errors="replace").read()
    return None


def rd_csv(path):
    t = rd_text(path)
    return pd.read_csv(io.StringIO(t)) if t else None


def first_file(dirs, rels):
    for d in dirs:
        for r in rels:
            p = os.path.join(d, r)
            if os.path.exists(p) or os.path.exists(p + ".gz"):
                return p
    return None


# ------------------------------------------------------------------ data access
def pos_of(arm_dir):
    P = rd_csv(os.path.join(arm_dir, "pos_trace_all.csv.gz"))
    if P is not None and {"scan_id", "px", "py", "pz"} <= set(P.columns):
        return P, "pos_trace_all.csv.gz"
    S = rd_csv(os.path.join(arm_dir, "state_trace.csv"))
    if S is not None and {"scan_id", "px", "py", "pz"} <= set(S.columns):
        if "phase" in S.columns:
            S = S[S["phase"] == "post_lio"]
        return S.drop_duplicates("scan_id"), "state_trace.csv (post_lio)"
    return None, None


def ref_pos(ref_dirs, ref):
    p = first_file(ref_dirs, [os.path.join("compact", ref, "pos_trace_all.csv.gz"), os.path.join(ref, "pos_trace_all.csv.gz")])
    return (rd_csv(p), p) if p else (None, None)


def ref_row(ref_dirs, ref):
    for d in ref_dirs:
        p = os.path.join(d, "phase_summary.csv")
        if os.path.exists(p):
            T = pd.read_csv(p)
            if "arm" in T.columns and (T["arm"] == ref).any():
                return T[T["arm"] == ref].iloc[0].to_dict()
    return None


def ref_digest(ref_dirs, ref):
    for d in ref_dirs:
        p = os.path.join(d, "completeness.csv")
        if os.path.exists(p):
            T = pd.read_csv(p)
            c = "state_trace_sha256_run_id_blanked"
            if "arm" in T.columns and c in T.columns and (T["arm"] == ref).any():
                v = str(T[T["arm"] == ref].iloc[0][c])
                if v and v != "nan":
                    return v
    return None


def arm_digest(arm_dir):
    t = rd_text(os.path.join(arm_dir, "state_trace_blank_runid.sha256"))
    return t.split()[0] if t and t.split() else None


def arm_metrics(arm_dir):
    p = os.path.join(arm_dir, "arm_summary.json")
    S = json.load(open(p)) if os.path.exists(p) else {}
    return S.get("excursion_mm"), S.get("settled_offset_mm")


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ comparisons
def pos_compare(P, R, tol_mm):
    """-> dict(n_ref, n_arm, n_common, max_mm, first_scan (first scan with a difference > tol or None), missing (reference scans absent in the arm), yaw_mrad)."""
    a = P.drop_duplicates("scan_id").set_index("scan_id"); b = R.drop_duplicates("scan_id").set_index("scan_id")
    com = b.index.intersection(a.index).sort_values()
    out = dict(n_ref=len(b), n_arm=len(a), n_common=len(com), max_mm=float("nan"), first_scan=None, missing=len(b.index.difference(a.index)), yaw_mrad=None)
    if not len(com):
        return out
    d = np.linalg.norm(a.loc[com, ["px", "py", "pz"]].to_numpy() - b.loc[com, ["px", "py", "pz"]].to_numpy(), axis=1) * 1e3
    cd = np.max(np.abs(a.loc[com, ["px", "py", "pz"]].to_numpy() - b.loc[com, ["px", "py", "pz"]].to_numpy()), axis=1) * 1e3
    out["max_mm"] = float(np.max(np.maximum(d, cd)))
    bad = np.where(np.maximum(d, cd) > tol_mm)[0]
    out["first_scan"] = int(com[bad[0]]) if len(bad) else None
    if "yaw_rad" in a.columns and "yaw_rad" in b.columns:
        dy = a.loc[com, "yaw_rad"].to_numpy() - b.loc[com, "yaw_rad"].to_numpy()
        out["yaw_mrad"] = float(np.max(np.abs((dy + np.pi) % (2 * np.pi) - np.pi)) * 1e3)
    return out


def state_first_divergence(arm_csv, ref_csv):
    """first scan and column where any numeric state value of the two raw state_trace.csv files differs AT ALL (the digest test is bit equality). None if identical on the common scans / not comparable."""
    A, B = rd_csv(arm_csv), rd_csv(ref_csv)
    if A is None or B is None or "scan_id" not in A.columns or "scan_id" not in B.columns:
        return None
    key = ["scan_id"] + (["phase"] if "phase" in A.columns and "phase" in B.columns else [])
    A = A.drop_duplicates(key).set_index(key); B = B.drop_duplicates(key).set_index(key)
    com = A.index.intersection(B.index)
    cols = [c for c in A.columns if c in B.columns and c != "run_id" and pd.api.types.is_numeric_dtype(A[c]) and pd.api.types.is_numeric_dtype(B[c])]
    best = None
    for c in cols:
        x, y = A.loc[com, c].to_numpy(dtype=float), B.loc[com, c].to_numpy(dtype=float)
        ne = np.where(~((x == y) | (np.isnan(x) & np.isnan(y))))[0]
        if len(ne):
            i = ne[0]; k = com[i]
            sc = int(k[0] if isinstance(k, tuple) else k)
            if best is None or sc < best[0]:
                best = (sc, c, float(x[i]), float(y[i]), str(k[1]) if isinstance(k, tuple) and len(k) > 1 else "")
    if best is None and len(A) != len(B):
        return (None, "row count", float(len(A)), float(len(B)), "")
    return best


def cfg_dict(text):
    """effective-config lines of a run.log: `key = value [note]` and `/ns/key: value` -> {slash key: value} (the last occurrence wins)."""
    d = {}
    for ln in (text or "").splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_./]+)\s*=\s*(\S+)(?:\s+\[.*\])?\s*$", ln)
        if m and "/" in m.group(1):
            d[m.group(1)] = m.group(2).strip("'\""); continue
        m = re.match(r"^\s*/[A-Za-z0-9_]+/([A-Za-z0-9_./]+):\s*(\S+)\s*$", ln)
        if m:
            d["" + m.group(1)] = m.group(2).strip("'\"")
    return d


def new_key_findings(log):
    """new R82 keys that the arm's run.log shows at a non-default value."""
    d = cfg_dict(log); out = []
    for k, ok_vals in NEW_KEY_DEFAULTS.items():
        v = d.get(k)
        if v is None:
            continue
        if v not in ok_vals and not (num(v) == 0.0 and k.endswith(("plane_var_min", "plane_var_scan_s2"))) and not (k == "lio/residual_dump_voxel_info" and v.lower() in ("false", "0")):
            out.append("%s = %s (default %s)" % (k, v, ok_vals[0]))
    for k in ("imu/process_noise/gain/nis_slope", "imu/process_noise/gain/nis_clip"):
        if k in d and d.get("imu/process_noise/gain/mode") != "nis_smooth":
            out.append("%s printed although gain mode is not nis_smooth" % k)
    return out


def log_diff(a_log, b_log, limit=40):
    A, B = cfg_dict(a_log), cfg_dict(b_log)
    out = []
    for k in sorted(set(A) | set(B)):
        if IGNORE_KEYS.search(k):
            continue
        if k not in A:
            out.append("%s: only in the reference (%s)" % (k, B[k]))
        elif k not in B:
            out.append("%s: only in the arm (%s)" % (k, A[k]))
        elif A[k] != B[k] and not (num(A[k]) is not None and num(B[k]) is not None and abs(num(A[k]) - num(B[k])) <= 1e-12 * max(1.0, abs(num(B[k])))):
            out.append("%s: arm %s, reference %s" % (k, A[k], B[k]))
    return out[:limit] + (["... %d more" % (len(out) - limit)] if len(out) > limit else [])


def find_job(manifest_dirs, arm):
    for d in manifest_dirs:
        for fp in sorted(glob.glob(os.path.join(d, "jobs_*.json"))):
            try:
                for j in json.load(open(fp)):
                    if j.get("job_id") == arm:
                        return j.get("overrides") or {}
            except Exception:
                continue
    return None


def overrides_diff(a, b):
    out = []
    for k in sorted(set(a) | set(b)):
        if IGNORE_KEYS.search(k):
            continue
        if k not in a:
            out.append("%s: only in the reference job (%r)" % (k, b[k]))
        elif k not in b:
            out.append("%s: only in the arm's job (%r)" % (k, a[k]))
        elif a[k] != b[k]:
            out.append("%s: arm %r, reference %r" % (k, a[k], b[k]))
    return out


# ------------------------------------------------------------------ main logic
def check_arm(arm, ref, runs, ref_dirs, ref_runs, manifests, ref_manifests, tol):
    """-> dict(arm, ref, status, rows [(name, status, detail)], first_scan, state_div, cfg_new, cfg_log, cfg_jobs)"""
    d = os.path.join(runs, arm)
    rows = []
    add = lambda n, s, det="": rows.append((n, s, det))
    P, psrc = pos_of(d) if os.path.isdir(d) else (None, None)
    R, rsrc = ref_pos(ref_dirs, ref)
    first_scan = None
    if P is None or R is None:
        add("POS per-scan position equals %s (tol %.4g mm)" % (ref, tol), "MISSING", "arm trace %s, reference trace %s" % (psrc or "absent", rsrc or "absent (no --ref-return holds %s)" % ref))
    else:
        c = pos_compare(P, R, tol)
        first_scan = c["first_scan"]
        ok = c["missing"] == 0 and c["n_common"] > 0 and c["max_mm"] <= tol
        det = "arm %d scans (%s), reference %d scans, common %d, reference scans absent in the arm %d, max |difference| %.6g mm" % (c["n_arm"], psrc, c["n_ref"], c["n_common"], c["missing"], c["max_mm"])
        if c["missing"] and first_scan is None:
            first_scan = int(R.drop_duplicates("scan_id").set_index("scan_id").index.difference(P.drop_duplicates("scan_id").set_index("scan_id").index).min())
            det += "; first reference scan absent in the arm %d" % first_scan
        add("POS per-scan position equals %s (tol %.4g mm)" % (ref, tol), "PASS" if ok else "FAIL", det)
        if c["yaw_mrad"] is not None:
            add("POS yaw difference (INFO: no tolerance decided)", "INFO", "max %.6g mrad" % c["yaw_mrad"])
    ex, st = arm_metrics(d)
    rr = ref_row(ref_dirs, ref)
    for nm, a_v, key in (("EXC excursion_mm", ex, "excursion_mm"), ("SET settled_offset_mm", st, "settled_offset_mm")):
        r_v = num(rr.get(key)) if rr else None
        a_v = num(a_v)
        if a_v is None or r_v is None:
            add("%s equals %s (tol %.4g mm)" % (nm, ref, tol), "MISSING", "arm %s, reference %s" % (a_v, "no phase_summary row" if rr is None else r_v))
        else:
            add("%s equals %s (tol %.4g mm)" % (nm, ref, tol), "PASS" if abs(a_v - r_v) <= tol else "FAIL", "arm %.6f, reference %.6f, difference %.3g" % (a_v, r_v, abs(a_v - r_v)))
    da, db = arm_digest(d), ref_digest(ref_dirs, ref)
    metrics_ok = all(s == "PASS" for n, s, _ in rows if s != "INFO")
    if da and db:
        add("DIG state_trace digest (run_id blanked) equals %s" % ref, "PASS" if da == db else "FAIL",
            ("%s vs %s" % (da[:16], db[:16])) + ("" if da == db or not metrics_ok else "; every metric above is within tolerance and ONLY the digest differs (F-419 treated digest equality as a fact, the R82 spec as a verdict: planner decides)"))
    else:
        add("DIG state_trace digest equals %s" % ref, "INFO", "digest not available on both sides (arm %s, reference %s)" % ("yes" if da else "no", "yes" if db else "no"))
    status = "PASS" if all(s in ("PASS", "INFO") for _, s, _ in rows) else "FAIL"
    out = dict(arm=arm, ref=ref, status=status, rows=rows, first_scan=first_scan, state_div=None, cfg_new=[], cfg_log=[], cfg_jobs=[])
    if status == "FAIL":
        sa = os.path.join(d, "state_trace.csv"); sb = first_file(([os.path.join(x, ref) for x in ref_runs] + [os.path.join(x, "compact", ref) for x in ref_dirs] + [os.path.join(x, ref) for x in ref_dirs]), ["state_trace.csv"])
        if os.path.exists(sa) and sb:
            out["state_div"] = state_first_divergence(sa, sb)
        log = rd_text(os.path.join(d, "run.log")) or ""
        out["cfg_new"] = new_key_findings(log)
        rl = first_file([os.path.join(x, ref) for x in ref_runs], ["run.log"])
        out["cfg_log"] = log_diff(log, rd_text(rl) or "") if rl and log else ([] if rl else ["(reference run.log not available: pass --ref-runs to compare the effective configs)"])
        ja, jb = find_job(manifests, arm), find_job(manifests + ref_manifests, ref)
        out["cfg_jobs"] = overrides_diff(ja, jb) if ja is not None and jb is not None else ["(job overrides not comparable: arm job %s, reference job %s)" % ("found" if ja is not None else "not found", "found" if jb is not None else "not found")]
    return out


def hold_text(failed, tol):
    L = ["HOLD: R82 identity check FAILED. Written %s by r82_identity_hold.py." % time.strftime("%Y-%m-%d %H:%M:%S %z"), "",
         "Do NOT dispatch any other queue (r82_hold_check.sh refuses while this file is named HOLD.txt). Let running jobs finish. Cancel nothing, delete nothing, change nothing in the tree.",
         "Investigate the cause fully (diff of config dump, run.log, state trace first-divergence scan, tree diff) and report to the planner. Only the planner's written release lifts this hold:",
         "then rename this file to HOLD.released_<date>.txt (never delete it).", ""]
    for r in failed:
        L += ["ARM %s   reference %s   tolerance %.4g mm" % (r["arm"], r["ref"], tol)]
        L += ["  %-7s %s   [%s]" % (s, n, dt) for n, s, dt in r["rows"]]
        L.append("  first diverging scan (position, pos trace, > tol): %s" % (r["first_scan"] if r["first_scan"] is not None else "none found / not determinable"))
        sd = r["state_div"]
        L.append("  first scan where any raw state value differs at all: %s" % ("scan %s, column %s, arm %.17g, reference %.17g %s" % (sd[0], sd[1], sd[2], sd[3], sd[4]) if sd else "not determinable (state_trace.csv missing on one side, or equal)"))
        L.append("  config differences detected:")
        L += ["    new key not at default: " + x for x in r["cfg_new"]] or ["    new keys: all at default in the arm's run.log (or not printed)"]
        L += ["    run.log vs reference run.log: " + x for x in r["cfg_log"]] or ["    run.log vs reference run.log: no difference in the effective-config lines"]
        L += ["    job overrides: " + x for x in r["cfg_jobs"]] or ["    job overrides: identical (after ignoring paths and run ids)"]
        L.append("")
    return "\n".join(L) + "\n"


def run(runs, manifest, ref_returns, ref_runs, ref_manifests, hold_file, tol, out, arms_arg=None):
    pairs = []
    if arms_arg:
        pairs = [tuple(x.split("=", 1)) for x in arms_arg.split(",") if "=" in x]
    else:
        tp = os.path.join(manifest, "arm_table.tsv")
        if not os.path.exists(tp):
            print("no arm_table.tsv in %s and no --arms" % manifest); return 2
        with open(tp, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if (r.get("arm_kind") or r.get("kind")) == "id" and r.get("ref"):
                    pairs.append((r["arm"], r["ref"]))
    if not pairs:
        print("no identity arm (kind == id with a ref) found: nothing was checked; this is a FAIL, the id queue must contain them"); return 1
    res = [check_arm(a, r, runs, ref_returns, ref_runs, [manifest] if manifest else [], ref_manifests, tol) for a, r in pairs]
    lines = []
    for r in res:
        lines.append("%s %s (reference %s)" % (r["status"], r["arm"], r["ref"]))
        lines += ["    %-7s %s   [%s]" % (s, n, dt) for n, s, dt in r["rows"]]
    failed = [r for r in res if r["status"] != "PASS"]
    lines.append("")
    lines.append("IDENTITY %s: %d of %d identity arms PASS" % ("PASS" if not failed else "FAIL", len(res) - len(failed), len(res)))
    txt = "\n".join(lines)
    print(txt)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        open(out, "w").write(txt + "\n")
    if failed:
        hf = hold_file or (os.path.join("ablations", "_job_queue", "HOLD.txt") if os.path.isdir(os.path.join("ablations", "_job_queue")) else "HOLD.txt")
        body = hold_text(failed, tol)
        if os.path.exists(hf):
            body = "\n" + "=" * 70 + "\nAPPENDED %s\n" % time.strftime("%Y-%m-%d %H:%M:%S") + body     # an existing hold is never replaced
            open(hf, "a").write(body)
        else:
            open(hf, "w").write(body)
        print("HOLD written: %s  (no other queue may be dispatched; nothing was cancelled or deleted)" % os.path.abspath(hf))
        return 1
    return 0


# ------------------------------------------------------------------ selftest
def selftest():
    fails = []
    ck = lambda c, m: fails.append(m) if not c else None
    t = tempfile.mkdtemp()
    N = 60
    sid = np.arange(1, N + 1)
    base = pd.DataFrame({"scan_id": sid, "px": np.cumsum(np.full(N, 1e-3)), "py": 0.0, "pz": 0.0, "yaw_rad": 0.01})
    state = pd.DataFrame({"run_id": "x", "scan_id": sid, "phase": "post_lio", "px": base.px, "py": 0.0, "pz": 0.0, "vx": 0.1})

    def mk_ret(d, arms):
        os.makedirs(d, exist_ok=True)
        pd.DataFrame([dict(arm=a, excursion_mm=82.003503, settled_offset_mm=65.332038) for a in arms]).to_csv(os.path.join(d, "phase_summary.csv"), index=False)
        pd.DataFrame([dict(arm=a, state_trace_sha256_run_id_blanked="ab" * 32) for a in arms]).to_csv(os.path.join(d, "completeness.csv"), index=False)
        for a in arms:
            os.makedirs(os.path.join(d, "compact", a), exist_ok=True)
            base.to_csv(os.path.join(d, "compact", a, "pos_trace_all.csv.gz"), index=False, compression="gzip")

    def mk_arm(arm, shift_scan=None, shift_mm=0.0, exc=82.003503, dig="ab" * 32, log="", state_shift=None):
        d = os.path.join(runs, arm); os.makedirs(d, exist_ok=True)
        P = base.copy(); S = state.copy()
        if shift_scan:
            P.loc[P.scan_id >= shift_scan, "px"] += shift_mm * 1e-3
        if state_shift:
            S.loc[S.scan_id >= state_shift, "vx"] += 1e-12
        P.to_csv(os.path.join(d, "pos_trace_all.csv.gz"), index=False, compression="gzip"); S.to_csv(os.path.join(d, "state_trace.csv"), index=False)
        json.dump({"excursion_mm": exc, "settled_offset_mm": 65.332038}, open(os.path.join(d, "arm_summary.json"), "w"))
        if dig:
            open(os.path.join(d, "state_trace_blank_runid.sha256"), "w").write(dig + "\n")
        open(os.path.join(d, "run.log"), "w").write(log)
    runs = os.path.join(t, "runs"); os.makedirs(runs)
    mf = os.path.join(t, "man"); os.makedirs(mf); rm = os.path.join(t, "rman"); os.makedirs(rm)
    open(os.path.join(mf, "arm_table.tsv"), "w").write("arm\tkind\tref\nr82_dc_id_mg_eee_01_fq\tid\tr80_mg_eee_01_fq\nr82_rt_id_mg_eee_01_fq\tid\tr80c_rtt_mg_eee_01_fq\nr81_rt_mg_eee_01_fq\tclosed\t\n")
    json.dump([dict(job_id="r82_dc_id_mg_eee_01_fq", overrides={"cbk.lidar.point_filter_num": 3, "outputs.path": "/x/a", "voxel_map.plane.plane_var_min": 4e-4}),
               dict(job_id="r82_rt_id_mg_eee_01_fq", overrides={"cbk.lidar.point_filter_num": 3, "cbk.lidar.overlap_policy": "retime_trim"})], open(os.path.join(mf, "jobs_r82_id.json"), "w"))
    json.dump([dict(job_id="r80_mg_eee_01_fq", overrides={"cbk.lidar.point_filter_num": 3, "outputs.path": "/y/b"}), dict(job_id="r80c_rtt_mg_eee_01_fq", overrides={"cbk.lidar.point_filter_num": 3, "cbk.lidar.overlap_policy": "retime_trim"})],
              open(os.path.join(rm, "jobs_x.json"), "w"))
    r80, r80c = os.path.join(t, "R80"), os.path.join(t, "R80C")
    mk_ret(r80, ["r80_mg_eee_01_fq"]); mk_ret(r80c, ["r80c_rtt_mg_eee_01_fq"])
    hold = os.path.join(t, "HOLD.txt")
    call = lambda: run(runs, mf, [r80, r80c], [], [rm], hold, TOL_MM, os.path.join(t, "out.txt"))
    quiet = io.StringIO(); so = sys.stdout

    def runq():
        sys.stdout = quiet
        try:
            return call()
        finally:
            sys.stdout = so
    # 1. both identical -> PASS, no HOLD.txt
    mk_arm("r82_dc_id_mg_eee_01_fq"); mk_arm("r82_rt_id_mg_eee_01_fq")
    ck(runq() == 0 and not os.path.exists(hold), "identical arms must PASS and write no HOLD.txt")
    # 2. 0.004 mm shift (inside tolerance) -> PASS ; 0.006 mm -> FAIL with the first scan
    mk_arm("r82_dc_id_mg_eee_01_fq", 20, 0.004)
    ck(runq() == 0, "a 0.004 mm shift is inside the 0.005 mm tolerance")
    mk_arm("r82_dc_id_mg_eee_01_fq", 33, 0.006, log="voxel_map/plane/plane_var_min = 0.0004\n  imu/process_noise/gain/mode = off\n")
    rc = runq()
    ht = open(hold).read() if os.path.exists(hold) else ""
    ck(rc == 1 and "ARM r82_dc_id_mg_eee_01_fq" in ht, "a 0.006 mm shift must FAIL and write HOLD.txt (rc %s)" % rc)
    ck("first diverging scan (position, pos trace, > tol): 33" in ht, "HOLD.txt must name first diverging scan 33: %s" % ht[:1500])
    ck("voxel_map/plane/plane_var_min = 0.0004" in ht and "voxel_map.plane.plane_var_min: only in the arm's job" in ht, "HOLD.txt must list the new key not at default and the override difference")
    ck("ARM r82_rt_id_mg_eee_01_fq" not in ht, "the identical rt arm must not be in the HOLD")
    # 3. never deletes: the arm files are all still there; an existing HOLD is appended not replaced
    ck(os.path.exists(os.path.join(runs, "r82_dc_id_mg_eee_01_fq", "pos_trace_all.csv.gz")) and os.path.exists(os.path.join(r80, "phase_summary.csv")), "nothing may be deleted")
    first_len = len(ht)
    runq()
    ck(len(open(hold).read()) > first_len and open(hold).read().startswith(ht[:200]), "an existing HOLD.txt is appended, never replaced")
    os.rename(hold, hold + ".released")
    # 4. excursion off by 0.01 mm -> FAIL EXC
    mk_arm("r82_dc_id_mg_eee_01_fq", exc=82.013503)
    ck(runq() == 1, "excursion difference must fail")
    o = open(os.path.join(t, "out.txt")).read()
    ck(any(l.strip().startswith("FAIL") and "EXC excursion_mm" in l for l in o.splitlines()), "EXC row must be FAIL: " + o)
    os.remove(hold) if os.path.exists(hold) else None
    # 5. digest differs with equal metrics -> FAIL with the explanatory text; state first divergence is named
    mk_arm("r82_dc_id_mg_eee_01_fq", dig="cd" * 32, state_shift=44)
    os.makedirs(os.path.join(t, "refruns", "r80_mg_eee_01_fq"), exist_ok=True); state.to_csv(os.path.join(t, "refruns", "r80_mg_eee_01_fq", "state_trace.csv"), index=False)
    open(os.path.join(t, "refruns", "r80_mg_eee_01_fq", "run.log"), "w").write("cbk/lidar/point_filter_num = 3\nvoxel_map/plane/plane_var_min = 0\n")
    open(os.path.join(runs, "r82_dc_id_mg_eee_01_fq", "run.log"), "w").write("cbk/lidar/point_filter_num = 3\nvoxel_map/plane/plane_var_min = 0.0004\n")
    sys.stdout = quiet
    try:
        rc = run(runs, mf, [r80, r80c], [os.path.join(t, "refruns")], [rm], hold, TOL_MM, os.path.join(t, "out.txt"))
    finally:
        sys.stdout = so
    ht = open(hold).read()
    o = open(os.path.join(t, "out.txt")).read()
    ck(rc == 1 and "ONLY the digest differs" in o, "digest-only difference must FAIL with the explanatory text: " + o)
    ck("scan 44, column vx" in ht, "first raw state divergence must be scan 44 column vx: " + ht)
    ck("voxel_map/plane/plane_var_min: arm 0.0004, reference 0" in ht, "run.log difference must be listed")
    os.remove(hold)
    # 6. no reference return -> MISSING counts as FAIL; no identity arm -> FAIL
    mk_arm("r82_dc_id_mg_eee_01_fq")
    sys.stdout = quiet
    try:
        rc = run(runs, mf, [r80], [], [rm], hold, TOL_MM, None)         # R80C reference absent: the rt identity arm has no reference
        rc2 = run(runs, os.path.join(t, "empty"), [r80, r80c], [], [], hold, TOL_MM, None)
    finally:
        sys.stdout = so
    ck(rc == 1 and "ARM r82_rt_id_mg_eee_01_fq" in open(hold).read(), "a missing reference must be a FAIL with HOLD")
    ck(rc2 == 2, "missing arm table must be rc 2")
    os.remove(hold)
    # 7. an arm directory missing altogether
    shutil.rmtree(os.path.join(runs, "r82_rt_id_mg_eee_01_fq"))
    sys.stdout = quiet
    try:
        rc = run(runs, mf, [r80, r80c], [], [rm], hold, TOL_MM, None)
    finally:
        sys.stdout = so
    ck(rc == 1, "a missing identity arm directory must FAIL")
    print("SELFTEST r82_identity_hold", "FAIL " + "; ".join(fails) if fails else
          "PASS (identical arms PASS and write nothing; 0.004 mm inside tol; 0.006 mm FAIL with first diverging scan 33, new key not at default and override difference named; HOLD appended not replaced, nothing deleted; "
          "excursion FAIL; digest-only FAIL with the explanation, first raw state divergence scan 44 column vx, run.log difference listed; missing reference / arm / table FAIL)")
    shutil.rmtree(t, ignore_errors=True)
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs"); ap.add_argument("--manifest", default="r82_manifest"); ap.add_argument("--ref-return", action="append", default=[]); ap.add_argument("--ref-runs", action="append", default=[])
    ap.add_argument("--ref-manifest", action="append", default=[]); ap.add_argument("--hold-file", default=None); ap.add_argument("--tol-mm", type=float, default=TOL_MM)
    ap.add_argument("--out", default=None); ap.add_argument("--arms", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.runs:
        sys.exit("--runs is required")
    return run(a.runs, a.manifest, a.ref_return, a.ref_runs, a.ref_manifest, a.hold_file, a.tol_mm, a.out, a.arms)


if __name__ == "__main__":
    sys.exit(main())
