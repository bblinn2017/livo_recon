#!/usr/bin/env python3
"""D-1 -- "Is directional coverage real?"

Rebuilds the per-plane directional Fisher information
    H = sum_i a_i a_i^T / sigma_i^2,   a_i = [a0_i, a1_i, 1]
from corr.csv's retained a0/a1/plane_id columns (see voxelplane.cpp's
CorrInfoCols / debugLogConsistencyCorr -- the comment there names this
exact reconstruction as D-1's reason for existing), then regresses
log(NIS) on the resulting extrapolation leverage a(q)^T H^-1 a(q) against
the incumbent covariates log(aniso), log(occ_aniso), log(|d_center|),
cluster-robust by scan_id.

sigma_i^2 = S_sensor_i + roughness_i, matching the card's
sigma_i^2 = n^T Sigma_i n + lambda0^deb (S_sensor is exactly n^T Sigma_i n;
roughness is lambda0 already noise-corrected, i.e. lambda0^deb -- NOT the
raw per-plane "lambda0" column, which still includes noise).

d_center = sqrt(a0^2 + a1^2) -- the query's own tangent-plane offset from
plane center, which a0/a1 already are (a(q) = [d.v, d.u, 1]).

D-1's SECOND falsifier (--gravity-check, meaningful on eee_01 specifically):
"a forward-looking scanner on a vertical wall must show H with a small
eigenvalue in the vertical-tilt direction." H's tangent 2x2 sub-block
(the [a0,a1] x [a0,a1] corner of the full 3x3 H -- the "1" row/column is
the plane's own d-offset sensitivity, not a spatial tilt direction) has
eigenvectors expressed in the plane's own (y_normal_, x_normal_) chart
(a0 <-> y_normal_, a1 <-> x_normal_, per voxelplane.cpp's J_nq(0)/J_nq(1)
convention). Mapping the eigenvector with the SMALLEST eigenvalue back to
world frame (e0*y_normal_ + e1*x_normal_) and comparing it against gravity
(this codebase's world convention: gravity = [0,0,-9.81], i.e. vertical is
the world Z axis) tests the claim directly: on a wall plane (normal
roughly horizontal, |normal.z| small; normal recovered as
x_normal_ x y_normal_) scanned by horizontal rings, the least-constrained
tilt direction should be near-vertical.

Needs x_normal_/y_normal_ per row, added to corr.csv alongside a0/a1 --
NOT present in any corr.csv generated before this addition (D-1's own
56-cell dispatch predates it and had to be deleted/rerun for this check).

Usage:
  d1_leverage_report.py --corr-csv <path/to/corr.csv> --job-id <id> [--gravity-check]
Prints one line of JSON per job to stdout.
"""
import argparse, json, sys
import numpy as np
import pandas as pd

USECOLS = ["scan_id", "nu", "S", "gated", "dropped_by_ablation",
           "S_sensor", "aniso", "occ_aniso", "a0", "a1", "plane_id", "roughness",
           "S_prior_pose"]
GRAVITY_COLS = ["x_normal_x", "x_normal_y", "x_normal_z",
                "y_normal_x", "y_normal_y", "y_normal_z"]
WORLD_UP = np.array([0.0, 0.0, 1.0])   # vertical axis; gravity = [0,0,-9.81] in this codebase

MIN_PLANE_ROWS = 5          # need >=5 correspondences to trust a 3x3 H
MAX_COND = 1e10              # refuse an ill-conditioned H rather than report a bogus leverage
WALL_NORMAL_Z_MAX = 0.3      # |normal.z| below this => "roughly vertical wall"


def build_leverage(df):
    """Per plane_id: H = sum a a^T / sigma^2 (using ALL of that plane's
    retained rows), then leverage_i = a_i^T H^-1 a_i for every row in that
    plane. Returns a leverage Series aligned to df.index, NaN where the
    plane has too few rows or H is ill-conditioned."""
    a0 = df["a0"].to_numpy()
    a1 = df["a1"].to_numpy()
    sigma2 = df["S_sensor"].to_numpy() + df["roughness"].to_numpy()
    valid_sigma = sigma2 > 0

    leverage = np.full(len(df), np.nan)
    for pid, idx in df.groupby("plane_id").indices.items():
        idx = idx[valid_sigma[idx]]
        if len(idx) < MIN_PLANE_ROWS:
            continue
        A = np.stack([a0[idx], a1[idx], np.ones(len(idx))], axis=1)  # (n,3)
        w = 1.0 / sigma2[idx]                                        # (n,)
        H = (A * w[:, None]).T @ A                                    # (3,3)
        try:
            evals = np.linalg.eigvalsh(H)
        except np.linalg.LinAlgError:
            continue
        if evals[0] <= 0 or evals[-1] / max(evals[0], 1e-300) > MAX_COND:
            continue
        Hinv = np.linalg.inv(H)
        lev = np.einsum("ij,jk,ik->i", A, Hinv, A)
        leverage[idx] = lev
    return leverage


def run_gravity_check(df):
    """D-1's second falsifier. Per plane: build the tangent 2x2 sub-block
    of H, take the eigenvector with the SMALLEST eigenvalue (the least-
    constrained tilt direction), map it to world frame via that plane's
    own (y_normal_, x_normal_) chart, and score |v_world . up|. Restricted
    to planes classified as roughly-vertical walls (|normal.z| small,
    normal recovered as x_normal_ x y_normal_). One row per usable plane;
    the caller summarizes (median alignment etc.) -- kept per-plane here
    so a skewed/bimodal distribution isn't hidden behind one number."""
    a0 = df["a0"].to_numpy()
    a1 = df["a1"].to_numpy()
    sigma2 = df["S_sensor"].to_numpy() + df["roughness"].to_numpy()
    valid_sigma = sigma2 > 0
    xn_cols = [df[c].to_numpy() for c in GRAVITY_COLS[:3]]
    yn_cols = [df[c].to_numpy() for c in GRAVITY_COLS[3:]]

    rows = []
    for pid, idx in df.groupby("plane_id").indices.items():
        idx = idx[valid_sigma[idx]]
        if len(idx) < MIN_PLANE_ROWS:
            continue
        A = np.stack([a0[idx], a1[idx], np.ones(len(idx))], axis=1)
        w = 1.0 / sigma2[idx]
        H = (A * w[:, None]).T @ A
        Ht = H[:2, :2]
        try:
            evals, evecs = np.linalg.eigh(Ht)  # ascending
        except np.linalg.LinAlgError:
            continue
        if evals[0] <= 0:
            continue

        x_normal = np.array([c[idx].mean() for c in xn_cols])
        y_normal = np.array([c[idx].mean() for c in yn_cols])
        xn_norm, yn_norm = np.linalg.norm(x_normal), np.linalg.norm(y_normal)
        if xn_norm < 1e-6 or yn_norm < 1e-6:
            continue
        xn, yn = x_normal / xn_norm, y_normal / yn_norm

        normal = np.cross(xn, yn)
        normal_z = float(normal[2] / max(np.linalg.norm(normal), 1e-12))
        if abs(normal_z) > WALL_NORMAL_Z_MAX:
            continue  # not a roughly-vertical wall -- the falsifier doesn't apply here

        e0, e1 = evecs[:, 0]  # smallest-eigenvalue eigenvector, in (a0=y_normal_, a1=x_normal_) coords
        v_world = e0 * yn + e1 * xn
        vnorm = np.linalg.norm(v_world)
        if vnorm < 1e-9:
            continue
        v_world /= vnorm
        alignment = float(abs(np.dot(v_world, WORLD_UP)))  # 1.0 = smallest-eig direction is exactly vertical

        rows.append({"plane_id": int(pid), "n_rows": int(len(idx)), "normal_z": normal_z,
                     "min_eig": float(evals[0]), "max_eig": float(evals[1]), "alignment": alignment})
    return rows


def summarize_gravity_check(rows):
    if not rows:
        return {"n_wall_planes": 0, "note": "no planes passed the |normal.z| < %.2f vertical-wall filter" % WALL_NORMAL_Z_MAX}
    align = np.array([r["alignment"] for r in rows])
    return {
        "n_wall_planes": len(rows),
        "alignment_median": float(np.median(align)),
        "alignment_p10": float(np.percentile(align, 10)),
        "alignment_p90": float(np.percentile(align, 90)),
        "frac_aligned_gt_0.7": float(np.mean(align > 0.7)),
        "frac_aligned_lt_0.3": float(np.mean(align < 0.3)),
    }


def _cluster_ols(y, X_cols, X_names, clusters):
    """Cluster-robust OLS. X_cols is a list of 1-D numpy arrays (already
    log-transformed etc by the caller); returns coef/se/pvalue/r2 by name."""
    import statsmodels.api as sm
    X = sm.add_constant(np.column_stack(X_cols))
    fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    names = ["const"] + X_names
    return {
        "coef": dict(zip(names, fit.params.tolist())),
        "se_cluster": dict(zip(names, fit.bse.tolist())),
        "pvalue": dict(zip(names, fit.pvalues.tolist())),
        "r_squared": float(fit.rsquared),
    }


def run_one(corr_csv, job_id, gravity_check=False, d1r=False):
    usecols = USECOLS + GRAVITY_COLS if gravity_check else USECOLS
    df = pd.read_csv(corr_csv, usecols=usecols)
    n_rows_total = len(df)
    df = df[(df["dropped_by_ablation"] == 0) & (df["gated"] == 0) & (df["S"] > 0)].copy()
    n_rows_used_pre_leverage = len(df)
    if n_rows_used_pre_leverage < 50:
        return {"job_id": job_id, "error": f"too few usable rows ({n_rows_used_pre_leverage})"}

    df["leverage"] = build_leverage(df)
    df["nis"] = df["nu"] ** 2 / df["S"]
    df["d_center"] = np.sqrt(df["a0"] ** 2 + df["a1"] ** 2)
    # D-1R.  S = floor_term + sigma_diag_squared + plane_var_term +
    # s_prior_pose (voxelplane.cpp's own S formula) -- S_prior_pose is
    # already its own logged column (P2), so S_est = S - S_prior_pose
    # exactly, no need to reconstruct floor_term/sigma_diag_squared/
    # plane_var_term individually.
    df["S_est"] = df["S"] - df["S_prior_pose"]
    df["nis_est"] = np.where(df["S_est"] > 0, df["nu"] ** 2 / df["S_est"], np.nan)

    mask = (
        (df["leverage"] > 0) & np.isfinite(df["leverage"])
        & (df["nis"] > 0) & np.isfinite(df["nis"])
        & (df["aniso"] > 0) & np.isfinite(df["aniso"])
        & (df["occ_aniso"] > 0) & np.isfinite(df["occ_aniso"])
        & (df["d_center"] > 0) & np.isfinite(df["d_center"])
    )
    d = df[mask]
    n_rows_regressed = len(d)
    n_planes_used = d["plane_id"].nunique()
    n_clusters = d["scan_id"].nunique()
    if n_rows_regressed < 50 or n_clusters < 5:
        return {"job_id": job_id, "error": f"too few regressable rows ({n_rows_regressed}) or clusters ({n_clusters})",
                "n_rows_total": n_rows_total, "n_rows_used_pre_leverage": n_rows_used_pre_leverage}

    base_cols = [np.log(d["leverage"].to_numpy()), np.log(d["aniso"].to_numpy()),
                 np.log(d["occ_aniso"].to_numpy()), np.log(d["d_center"].to_numpy())]
    base_names = ["log_leverage", "log_aniso", "log_occ_aniso", "log_d_center"]
    clusters = d["scan_id"].to_numpy()

    result = {
        "job_id": job_id,
        "n_rows_total": n_rows_total,
        "n_rows_used_pre_leverage": n_rows_used_pre_leverage,
        "n_rows_regressed": n_rows_regressed,
        "n_planes_used": int(n_planes_used),
        "n_clusters_scan_id": int(n_clusters),
    }
    result.update(_cluster_ols(np.log(d["nis"].to_numpy()), base_cols, base_names, clusters))

    if d1r:
        # D-1R.  D-1's own S (spec 1, above -- this IS the original
        # regression) is confounded: S_prior_pose is 81-98% of S and grows
        # with the SAME lever arm a(q) the leverage regressor is built
        # from, so a positive log_leverage coefficient on spec 1 could be
        # the confound, not real directional coverage. Specs 2/3 test that.
        mask2 = mask & np.isfinite(df["nis_est"]) & (df["nis_est"] > 0) & (df["S_prior_pose"] > 0)
        d2 = df[mask2]
        result["d1r"] = {"n_rows_spec23": int(len(d2))}
        if len(d2) >= 50 and d2["scan_id"].nunique() >= 5:
            cols2 = [np.log(d2["leverage"].to_numpy()), np.log(d2["aniso"].to_numpy()),
                     np.log(d2["occ_aniso"].to_numpy()), np.log(d2["d_center"].to_numpy())]
            clusters2 = d2["scan_id"].to_numpy()
            # spec 2: same covariates, S_est (excludes s_prior_pose) instead of S.
            result["d1r"]["spec2_S_est"] = _cluster_ols(
                np.log(d2["nis_est"].to_numpy()), cols2, base_names, clusters2)
            # spec 3: spec 2 + log(S_prior_pose) as an explicit covariate,
            # controlling for the shared lever-arm factor directly instead
            # of just excluding its effect from the denominator.
            cols3 = cols2 + [np.log(d2["S_prior_pose"].to_numpy())]
            result["d1r"]["spec3_S_est_plus_prior_pose_covariate"] = _cluster_ols(
                np.log(d2["nis_est"].to_numpy()), cols3, base_names + ["log_S_prior_pose"], clusters2)
        else:
            result["d1r"]["error"] = f"too few rows/clusters for specs 2/3 ({len(d2)} rows)"
        result["d1r"]["note"] = ("age/fill covariates from P10 NOT yet included -- corr.csv only "
                                  "has them aggregated per-scan (corr_scan.csv), not per-row; adding "
                                  "per-row age/fill to corr.csv needs a rebuild, deferred while DX-2 "
                                  "is actively dispatching (never rebuild mid-sweep).")

    if gravity_check:
        if not all(c in df.columns for c in GRAVITY_COLS):
            result["gravity_check"] = {"error": "corr.csv is missing x_normal_/y_normal_ columns -- "
                                                  "generated before that logging was added, cannot compute"}
        else:
            rows = run_gravity_check(df)
            result["gravity_check"] = summarize_gravity_check(rows)
            result["gravity_check_per_plane"] = rows
    else:
        result["gravity_check"] = "not requested (pass --gravity-check)"
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr-csv", required=True)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--gravity-check", action="store_true",
                     help="D-1's second falsifier -- meaningful on eee_01, needs corr.csv's "
                          "x_normal_/y_normal_ columns (added alongside a0/a1 for this).")
    ap.add_argument("--d1r", action="store_true",
                     help="D-1R: run the regression 3 ways (spec1=D-1's original S, "
                          "spec2=S_est excluding s_prior_pose, spec3=spec2+log(s_prior_pose) "
                          "as a covariate) to test whether D-1's positive log_leverage "
                          "coefficient survives removing the S_prior_pose confound.")
    a = ap.parse_args()
    result = run_one(a.corr_csv, a.job_id, gravity_check=a.gravity_check, d1r=a.d1r)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
