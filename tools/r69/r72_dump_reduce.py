#!/usr/bin/env python3
"""R72 (F-398): reduce coupled_record_f<N>.txt and coupled_failure_dump.txt (lossless matrix dumps, see matrix_dump.h) to compact JSON + a text table.

    python3 r72_dump_reduce.py --dir <arm dir> --out <arm dir>      (reads coupled_record_f*.txt and coupled_failure_dump.txt if present)

Block format:  "## <name> <rows> <cols>" then <rows> lines of <cols> numbers;  "# key: text" lines are metadata.
Pure numpy; planning-agent tool, checked only on synthetic round trips (selftest_r72.py) before the real files exist.
"""
import argparse, glob, json, os, re, sys
import numpy as np


def parse(path):
    """-> (blocks {name: ndarray (rows, cols)}, meta {key: text}). Raises ValueError on a malformed block (never silently truncates)."""
    blocks, meta = {}, {}
    with open(path) as f:
        lines = f.read().split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("## "):
            parts = ln[3:].split()
            if len(parts) != 3:
                raise ValueError("bad block header %r" % ln)
            name, r, c = parts[0], int(parts[1]), int(parts[2])
            if i + r >= len(lines):
                raise ValueError("block %s truncated" % name)
            M = np.zeros((r, c))
            for k in range(r):
                toks = lines[i + 1 + k].split()
                if len(toks) != c:
                    raise ValueError("block %s row %d has %d values, expected %d" % (name, k, len(toks), c))
                M[k] = [float(t) for t in toks]
            blocks[name] = M          # a repeated name (appended iterations) keeps the LAST; iteration blocks are prefixed i<k>_ so they do not collide
            i += 1 + r
        else:
            m = re.match(r"# ([^:]+): (.*)$", ln)
            if m:
                meta.setdefault(m.group(1), m.group(2))
            i += 1
    return blocks, meta


def kv(text):
    return dict(re.findall(r"(\w+)=(\S+)", text or ""))


def eig_stats(A):
    A = 0.5 * (A + A.T)
    w = np.linalg.eigvalsh(A)
    pos = w[w > 0]
    return {"min": float(w[0]), "max": float(w[-1]), "n_neg": int((w <= 0).sum()),
            "cond": float(w[-1] / w[0]) if w[0] > 0 else None, "min_pos": float(pos.min()) if len(pos) else None}


def reduce_record(path):
    B, M = parse(path)
    meta = kv(M.get("meta", ""))
    its = sorted({int(m.group(1)) for k in B for m in [re.match(r"i(\d+)_", k)] if m})
    out = {"file": os.path.basename(path), "meta": meta, "iterations": []}
    if "P_joint_prior" in B:
        out["prior_eig"] = eig_stats(B["P_joint_prior"])
    if "P18_scan_head_controlled" in B:
        out["head_P18_diag_sqrt"] = np.sqrt(np.maximum(np.diag(B["P18_scan_head_controlled"]), 0)).tolist()
    if "P18_after_imu_propagation" in B:
        out["propagated_P18_diag_sqrt"] = np.sqrt(np.maximum(np.diag(B["P18_after_imu_propagation"]), 0)).tolist()
    for k in its:
        p = "i%d_" % k
        g = lambda n: B.get(p + n)
        row = {"iter": k}
        if g("n_residuals") is not None:
            row["n_residuals"] = float(g("n_residuals")[0, 0])
        if g("knots_before") is not None and g("knots_after") is not None:
            d = g("knots_after") - g("knots_before")
            row["knot_change_max_abs"] = float(np.abs(d).max())
            row["knot_change_per_knot_norm"] = np.linalg.norm(d, axis=1).tolist()
        for nm in ("delta", "lidar_delta", "prior_delta"):
            if g(nm) is not None:
                row[nm + "_norm"] = float(np.linalg.norm(g(nm)))
        if g("A_system") is not None:
            row["A_eig"] = eig_stats(g("A_system"))
        if g("HtH_lidar") is not None:
            row["HtH_eig"] = eig_stats(g("HtH_lidar"))
        if g("P_joint_posterior") is not None:
            row["posterior_eig"] = eig_stats(g("P_joint_posterior"))
            if "P_joint_prior" in B:
                # directions in which the posterior is LARGER than the prior would be an information loss; report trace and the max ratio of diagonals
                dp, dq = np.diag(B["P_joint_prior"]), np.diag(g("P_joint_posterior"))
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(dp > 0, dq / dp, np.nan)
                row["posterior_over_prior_diag_max"] = float(np.nanmax(ratio))
                row["posterior_over_prior_diag_min"] = float(np.nanmin(ratio))
                row["posterior_trace_over_prior_trace"] = float(dq.sum() / dp.sum()) if dp.sum() > 0 else None
        if g("A_positive_definite_pass") is not None:
            row["A_pd_pass"] = bool(g("A_positive_definite_pass")[0, 0] == 1.0)
        if g("step_scale_applied") is not None:
            row["step_scale"] = float(g("step_scale_applied")[0, 0])
        out["iterations"].append(row)
    return out


def reduce_failure(path):
    B, M = parse(path)
    out = {"file": os.path.basename(path), "meta": kv(M.get("meta", "")), "what": M.get("what"), "capture_valid": M.get("capture_valid"),
           "capture_where": M.get("capture_where")}
    F = B.get("failed_matrix_symmetrised")
    if F is not None:
        w, V = np.linalg.eigh(0.5 * (F + F.T))
        out["failed_eig"] = eig_stats(F)
        out["failed_n"] = int(F.shape[0])
        bad = np.where(w <= 0)[0]
        out["negative_eigenvalues"] = w[bad].tolist()
        k = bad[0] if len(bad) else 0
        v = V[:, k]
        order = np.argsort(-np.abs(v))[:12]
        out["min_eigvec_top_indices"] = [[int(i), float(v[i])] for i in order]      # which state coordinates carry the offending direction
        out["min_eigvec_mass_top12"] = float((v[order] ** 2).sum())
        d = np.diag(F)
        out["diag_min"] = float(d.min()); out["diag_argmin"] = int(d.argmin())
        out["diag_nonpositive_indices"] = [int(i) for i in np.where(d <= 0)[0]]
        out["asymmetry_max_abs"] = float(np.abs(F - F.T).max())
    for nm in ("failed_matrix_ldlt_info", "failed_matrix_ldlt_positive"):
        if nm in B:
            out[nm] = float(B[nm][0, 0])
    if "failed_matrix_ldlt_diag" in B:
        d = B["failed_matrix_ldlt_diag"].ravel()
        out["ldlt_diag_min"] = float(d.min()); out["ldlt_diag_argmin"] = int(d.argmin()); out["ldlt_n_nonpositive"] = int((d <= 0).sum())
    for nm in ("P18_prior_cov_before_correction", "P18_after_imu_propagation", "P18_scan_head_controlled"):
        if nm in B and B[nm].shape[0] == B[nm].shape[1]:
            out[nm + "_eig"] = eig_stats(B[nm])
    for nm in ("P_joint_prior", "A_system"):
        if nm in B:
            out[nm + "_eig"] = eig_stats(B[nm])
    for nm in ("delta_previous_iteration", "lidar_delta_previous_iteration", "prior_delta_previous_iteration"):
        if nm in B:
            out[nm + "_norm"] = float(np.linalg.norm(B[nm]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"records": [], "failure": None, "errors": []}
    for p in sorted(glob.glob(os.path.join(a.dir, "coupled_record_f*.txt"))):
        try:
            res["records"].append(reduce_record(p))
        except Exception as e:
            res["errors"].append("%s: %s" % (os.path.basename(p), e))
    fp = os.path.join(a.dir, "coupled_failure_dump.txt")
    if os.path.exists(fp):
        try:
            res["failure"] = reduce_failure(fp)
        except Exception as e:
            res["errors"].append("failure dump: %s" % e)
    if not res["records"] and res["failure"] is None and not res["errors"]:
        print("no dump files in", a.dir); return 0
    json.dump(res, open(os.path.join(a.out, "r72_dump_summary.json"), "w"), indent=1)
    with open(os.path.join(a.out, "r72_dump_summary.txt"), "w") as f:
        f.write("frame iter n_resid  A_min_eig  A_cond  |delta|  |lidar|  |prior|  knot_max_change  post/prior_trace\n")
        for r in res["records"]:
            fr = r["meta"].get("frame", "?")
            for it in r["iterations"]:
                A = it.get("A_eig", {})
                f.write("%s %d %s %.3e %s %.3e %.3e %.3e %.3e %s\n" % (fr, it["iter"], it.get("n_residuals"), A.get("min", float("nan")), "%.2e" % A["cond"] if A.get("cond") else "NA",
                        it.get("delta_norm", float("nan")), it.get("lidar_delta_norm", float("nan")), it.get("prior_delta_norm", float("nan")),
                        it.get("knot_change_max_abs", float("nan")), "%.3f" % it["posterior_trace_over_prior_trace"] if it.get("posterior_trace_over_prior_trace") else "NA"))
        if res["failure"]:
            f.write("\nFAILURE %s\n" % json.dumps({k: v for k, v in res["failure"].items() if k not in ("min_eigvec_top_indices",)}, indent=1))
        for e in res["errors"]:
            f.write("ERROR " + e + "\n")
    return 1 if res["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
