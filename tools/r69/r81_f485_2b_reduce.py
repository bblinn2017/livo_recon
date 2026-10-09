#!/usr/bin/env python3
"""R81 F-485 Part 2B reducer. Host analysis of the ground-row dump of frames 400..410 (eee_01, 60 s, fq/mq x pfn 1/3) with the two added columns voxel_point_count and voxel_distinct_frames
(written when lio/residual_dump_voxel_info is true). Reads returned files only; the output is a table of derived numbers. NOTHING here feeds a run.

    python3 r81_f485_2b_reduce.py --dir <arm dir> [--dir <arm dir> ...] --out <dir> [--file NAME] [--frames 400 410] [--iter first|all|N] [--min-rows 1] [--ground-height H] [--sensor-col sigma_diag_squared]
    python3 r81_f485_2b_reduce.py --root <dir holding the arm directories> --out <dir> ...
    python3 r81_f485_2b_reduce.py --selftest

Definitions (claude/f485-plane-variance-part1-2a.md, Part 2; the design text in r81-designs-pending.md)
  plane      one (arm, frame, iteration, node_id): the dump rows of one voxel in one frame at one iteration. The default iteration is the FIRST dumped one of each frame (--iter first): the iterations of a
             frame are the same points seen again, pooling them would count each row twice.
  n          number of rows of the plane (the points of this frame that matched the plane).
  r-bar      mean of the residual column r over the n rows.
  V          model variance of r-bar for an average of n independent points:  V = plane_var_term + (floor_term + sensor_part) / n,
             plane_var_term = mean of the column over the rows (it is a per-voxel value), floor_term and sensor_part = mean over the rows. sensor_part = --sensor-col (default sigma_diag_squared).
  P          pose-prior term: mean over the rows of s_prior_pose. The weight leaves it out (R80: sigma_squared has no prior term; sigma_gate_squared has it) and it is a COMMON error of every row of the plane,
             so it does not shrink with n:  V_with_prior = V + P.  Every statistic is reported WITHOUT it (the model of the weight, 'wo') and WITH it ('wp').
  R          ratio = mean over planes of r-bar^2 / mean over planes of V (1 if the model is right and the points are independent).
  excess     e = r-bar^2 - V (or - V_with_prior). Regressions (ordinary least squares, one row per plane, no weights; slope, intercept, their classical standard errors, R^2, n planes):
               e on 1/n                    (a slope that is the same sign and size at every pfn = the n-dependent part is mis-stated; an intercept = a part that does not shrink with n)
               e on 1/distinct_frames      (distinct_frames = voxel_distinct_frames, the scans that built the voxel)
               e on both
  s2 (DERIVED, NOT USED for the run): the R81 key plane_var_scan_s2 adds s2 / n_scans to the plane variance with n_scans = distinct frames. Two derived numbers: s2_slope = the slope of e on 1/distinct_frames (with
             an intercept) and s2_origin = sum(e / df) / sum(1 / df^2) (the same regression through the origin = the whole excess attributed to the scan term). A negative value is reported as is (no clipping).
             Bryce decides s2; the arms of N485C use his values 2.0e-4 and 4.0e-4. The dump of these arms has plane_var_term without any scan term (the default), so s2 here is the excess ON TOP OF the dump's plane_var_term.
  scatter_over_noise  (rows >= 2) mean over planes of the within-plane variance of r (ddof 1) / mean over planes of (floor_term + sensor_part). The 2A result was 2.2 to 6.5 in frames 1 to 5.
Outputs (--out)
  f485_2b_summary.csv   one row per (arm, scope) with scope = 'all' (frames pooled) and every frame; columns below. Regressions only when the scope has at least 4 planes with at least 2 distinct values of the regressor.
  f485_2b_planes.csv    one row per plane (arm, frame, iter, node_id): n, rbar, V, P, voxel_point_count, distinct_frames, excess_wo, excess_wp  (so every summary number can be recomputed).
A missing file or column, an empty frame range or a plane whose voxel_point_count / voxel_distinct_frames differ between its rows is a nonzero exit (the latter is counted and reported in n_inconsistent_planes when
--allow-inconsistent is given, then the first row's value is used)."""
import argparse, os, sys
import numpy as np
import pandas as pd

NEED = ["frame", "iter", "node_id", "r", "plane_var_term", "floor_term", "s_prior_pose", "voxel_point_count", "voxel_distinct_frames"]
FILES = ["residual_dump_all.csv.gz", "residual_dump_all.csv", "residual_dump.csv", "residual_dump.csv.gz", "window_ground_rows_all.csv.gz", "f485_2b_ground_rows.csv.gz"]
PLANE_COLS = ["arm", "frame", "iter", "node_id", "n", "rbar", "V", "P", "voxel_point_count", "distinct_frames", "excess_wo", "excess_wp"]
REG = ["inv_n", "inv_df", "both"]
SUMMARY_COLS = (["arm", "scope", "n_planes", "n_rows", "mean_n", "mean_distinct_frames", "mean_rbar2", "mean_V", "mean_P", "R_wo", "R_wp", "scatter_over_noise", "n_inconsistent_planes"]
                + ["%s_%s_%s" % (k, rg, st) for k in ("wo", "wp") for rg in REG for st in (("slope", "slope_se", "intercept", "intercept_se", "r2", "n") if rg != "both" else ("b_inv_n", "b_inv_df", "intercept", "r2", "n"))]
                + ["s2_slope_wo", "s2_origin_wo", "s2_slope_wp", "s2_origin_wp"])


def _first(d, names):
    for n in names:
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p
    return None


def load(path, frames, it, ground_height, sensor_col):
    D = pd.read_csv(path)
    miss = [c for c in NEED + [sensor_col] if c not in D.columns]
    if miss:
        raise SystemExit("%s lacks columns %s (2B needs the R81 build with lio/residual_dump_voxel_info true, and the R80 dump columns)" % (os.path.basename(path), miss))
    D = D[(D.frame >= frames[0]) & (D.frame <= frames[1])]
    if len(D) == 0:
        raise SystemExit("%s has no rows with frame in [%d, %d]" % (os.path.basename(path), frames[0], frames[1]))
    if it == "first":
        D = D[D["iter"] == D.groupby("frame")["iter"].transform("min")]
    elif it != "all":
        D = D[D["iter"] == int(it)]
    if ground_height is not None:
        if not np.isfinite(ground_height) or "wz" not in D.columns:
            raise SystemExit("--ground-height needs a finite H and the dump column wz")
        D = D[D.wz <= ground_height]
    if len(D) == 0:
        raise SystemExit("%s: no rows left after the iteration / height selection" % os.path.basename(path))
    return D


def planes(D, arm, sensor_col, min_rows, allow_inc):
    g = D.groupby(["frame", "iter", "node_id"], sort=True)
    agg = g.agg(n=("r", "size"), rbar=("r", "mean"), pv=("plane_var_term", "mean"), floor=("floor_term", "mean"), sens=(sensor_col, "mean"), P=("s_prior_pose", "mean"),
                pc_min=("voxel_point_count", "min"), pc_max=("voxel_point_count", "max"), df_min=("voxel_distinct_frames", "min"), df_max=("voxel_distinct_frames", "max"),
                pc=("voxel_point_count", "first"), dfr=("voxel_distinct_frames", "first"), var_r=("r", lambda x: float(np.var(x.to_numpy(float), ddof=1)) if len(x) > 1 else np.nan)).reset_index()
    inc = (agg.pc_min != agg.pc_max) | (agg.df_min != agg.df_max)
    if inc.any() and not allow_inc:
        raise SystemExit("arm %s: %d planes whose voxel_point_count / voxel_distinct_frames differ between rows (join or writer problem; --allow-inconsistent to count and continue)" % (arm, int(inc.sum())))
    agg["inconsistent"] = inc
    agg = agg[agg.n >= min_rows].copy()
    if len(agg) == 0:
        raise SystemExit("arm %s: no plane with at least %d rows" % (arm, min_rows))
    agg["V"] = agg.pv + (agg.floor + agg.sens) / agg.n
    agg["noise"] = agg.floor + agg.sens
    agg["excess_wo"] = agg.rbar ** 2 - agg.V
    agg["excess_wp"] = agg.rbar ** 2 - agg.V - agg.P
    agg.insert(0, "arm", arm)
    return agg


def ols(y, X):
    """y (m,), X (m,p) with the intercept column included -> (coef, se, r2). se is NaN when m <= p."""
    m, p = X.shape
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    ss, tss = float(res @ res), float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss / tss if tss > 0 else float("nan")
    if m > p:
        try:
            se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * ss / (m - p))
        except np.linalg.LinAlgError:
            se = np.full(p, np.nan)
    else:
        se = np.full(p, np.nan)
    return coef, se, r2


def regress(ex, invn, invdf):
    out = {}
    for rg, cols in (("inv_n", [invn]), ("inv_df", [invdf]), ("both", [invn, invdf])):
        ok = len(ex) >= 4 and all(np.ptp(c) > 0 for c in cols)
        if rg == "both" and ok:
            ok = abs(np.corrcoef(invn, invdf)[0, 1]) < 0.9999
        if not ok:
            continue
        X = np.column_stack([np.ones(len(ex))] + cols)
        coef, se, r2 = ols(ex, X)
        if rg == "both":
            out.update({"both_b_inv_n": coef[1], "both_b_inv_df": coef[2], "both_intercept": coef[0], "both_r2": r2, "both_n": len(ex)})
        else:
            out.update({rg + "_slope": coef[1], rg + "_slope_se": se[1], rg + "_intercept": coef[0], rg + "_intercept_se": se[0], rg + "_r2": r2, rg + "_n": len(ex)})
    return out


def summarize(A, arm):
    rows = []
    for scope, sub in [("all", A)] + [(str(int(f)), A[A.frame == f]) for f in sorted(A.frame.unique())]:
        r = dict(arm=arm, scope=scope, n_planes=int(len(sub)), n_rows=int(sub.n.sum()), mean_n=float(sub.n.mean()), mean_distinct_frames=float(sub.dfr.mean()), mean_rbar2=float((sub.rbar ** 2).mean()),
                 mean_V=float(sub.V.mean()), mean_P=float(sub.P.mean()), n_inconsistent_planes=int(sub.inconsistent.sum()))
        r["R_wo"] = r["mean_rbar2"] / r["mean_V"] if r["mean_V"] > 0 else float("nan")
        r["R_wp"] = r["mean_rbar2"] / (r["mean_V"] + r["mean_P"]) if r["mean_V"] + r["mean_P"] > 0 else float("nan")
        m2 = sub[sub.n >= 2]
        r["scatter_over_noise"] = float(m2.var_r.mean() / m2.noise.mean()) if len(m2) and m2.noise.mean() > 0 else float("nan")
        invn, invdf = 1.0 / sub.n.to_numpy(float), 1.0 / sub.dfr.to_numpy(float)
        for k, col in (("wo", "excess_wo"), ("wp", "excess_wp")):
            ex = sub[col].to_numpy(float)
            r.update({"%s_%s" % (k, kk): v for kk, v in regress(ex, invn, invdf).items()})
            r["s2_slope_" + k] = r.get("%s_inv_df_slope" % k, float("nan"))
            den = float((invdf ** 2).sum())
            r["s2_origin_" + k] = float((ex * invdf).sum() / den) if den > 0 else float("nan")
        rows.append(r)
    return pd.DataFrame(rows).reindex(columns=SUMMARY_COLS)


def run(arms, out, frames=(400, 410), it="first", min_rows=1, ground_height=None, sensor_col="sigma_diag_squared", fname=None, allow_inc=False):
    S, P = [], []
    for name, d in arms:
        p = os.path.join(d, fname) if fname else _first(d, FILES)
        if not p or not os.path.exists(p):
            raise SystemExit("arm %s: no dump file (%s) in %s" % (name, fname or " / ".join(FILES), d))
        A = planes(load(p, frames, it, ground_height, sensor_col), name, sensor_col, min_rows, allow_inc)
        P.append(A); S.append(summarize(A, name))
    os.makedirs(out, exist_ok=True)
    S = pd.concat(S, ignore_index=True)[SUMMARY_COLS]
    S.to_csv(os.path.join(out, "f485_2b_summary.csv"), index=False, float_format="%.8g")
    pd.concat(P, ignore_index=True).rename(columns={"pc": "voxel_point_count", "dfr": "distinct_frames"})[PLANE_COLS].to_csv(os.path.join(out, "f485_2b_planes.csv"), index=False, float_format="%.9g")
    return S


# ---------------------------------------------------------------------------------------------- selftest
def _synth(path, rng, s2=3e-4, a_pt=2e-3, pv0=1e-6, prior=2e-3, frames=range(400, 411), nplanes=60, ground_split=True):
    """Truth: r-bar of a plane = common error ~ N(0, pv0 + s2/df + prior) + mean of n point errors N(0, a_pt). Model (the dump) plane_var_term = pv0, floor+sensor = a_pt."""
    rows = []
    for f in frames:
        for k in range(nplanes):
            node = 1000 * f + k
            n = int(rng.integers(1, 40)); dfr = int(rng.integers(1, 12)); pc = int(n * rng.integers(3, 30))
            common = rng.normal(0, np.sqrt(pv0 + s2 / dfr + prior))
            for it in (0, 2):
                for j in range(n):
                    rows.append(dict(frame=f, iter=it, node_id=node, r=common + rng.normal(0, np.sqrt(a_pt)) if it == 0 else 9.0, plane_var_term=pv0, floor_term=a_pt / 2, sigma_diag_squared=a_pt / 2,
                                     s_prior_pose=prior, voxel_point_count=pc, voxel_distinct_frames=dfr, hpos_z=0.99, wz=-0.2 if (k % 5) else 1.5))
    pd.DataFrame(rows).to_csv(path, index=False)


def selftest():
    import tempfile, subprocess
    fails = []
    rng = np.random.default_rng(5)
    root = tempfile.mkdtemp()
    d = os.path.join(root, "armA"); os.makedirs(d)
    _synth(os.path.join(d, "residual_dump_all.csv"), rng, nplanes=120)
    S = run([("armA", d)], os.path.join(root, "o"))
    a = S[S.scope == "all"].iloc[0]
    # iteration 2 holds r = 9: pooling both iterations would blow R up; first iteration only -> R_wp ~ 1 (the planted truth contains the prior and the scan term)
    if not (0.6 < a.R_wp < 1.6):
        fails.append("R_wp %r (planted ~1 incl. prior, s2 3e-4)" % a.R_wp)
    if not a.R_wo > a.R_wp:
        fails.append("R_wo %r should exceed R_wp %r" % (a.R_wo, a.R_wp))
    # s2 recovery: excess WITH prior = s2/df + n-dependent noise (zero mean) -> slope on 1/df ~ 3e-4 (the noise of a single plane is large, the mean over 1320 planes is not)
    if not (1.5e-4 < a.s2_slope_wp < 4.5e-4) or not (1.5e-4 < a.s2_origin_wp < 4.5e-4):
        fails.append("s2 wp slope %r origin %r (planted 3e-4)" % (a.s2_slope_wp, a.s2_origin_wp))
    # WITHOUT the prior the excess carries the prior as a constant (intercept ~ 2e-3), the 1/df slope is still s2
    if not (1.2e-3 < a.wo_inv_df_intercept < 2.8e-3) or not (1.5e-4 < a.wo_inv_df_slope < 4.5e-4):
        fails.append("wo intercept %r slope %r" % (a.wo_inv_df_intercept, a.wo_inv_df_slope))
    # 1/n carries no excess slope in the planted truth (the n-dependence of the model is right): slope of e_wp on 1/n ~ 0 within a few SE
    if abs(a.wp_inv_n_slope) > 4 * a.wp_inv_n_slope_se + 1e-9:
        fails.append("e_wp on 1/n slope %r se %r (planted 0)" % (a.wp_inv_n_slope, a.wp_inv_n_slope_se))
    # F2: plant a 1/n excess (correlated points: an offset that does not average) and show the 1/n slope fires
    d2 = os.path.join(root, "armB"); os.makedirs(d2)
    X = pd.read_csv(os.path.join(d, "residual_dump_all.csv"))
    g = X.groupby(["frame", "iter", "node_id"])
    X.loc[X.iter == 0, "r"] = X.loc[X.iter == 0, "r"] * 1.0   # keep
    X["floor_term"] = X["floor_term"] * 0.25; X["sigma_diag_squared"] = X["sigma_diag_squared"] * 0.25   # model under-states the per-point variance by 4: excess = 1.5e-3 / n
    X.to_csv(os.path.join(d2, "residual_dump_all.csv"), index=False)
    Sb = run([("armB", d2)], os.path.join(root, "o2")); b = Sb[Sb.scope == "all"].iloc[0]
    if not (b.wp_inv_n_slope > 2 * b.wp_inv_n_slope_se and 0.7e-3 < b.wp_inv_n_slope < 2.4e-3):   # planted 1.5e-3 = 2e-3 - 5e-4
        fails.append("planted 1/n excess not detected: slope %r se %r" % (b.wp_inv_n_slope, b.wp_inv_n_slope_se))
    # per-plane arithmetic on a hand-made plane
    hm = pd.DataFrame([dict(frame=400, iter=0, node_id=7, r=x, plane_var_term=1e-6, floor_term=4e-4, sigma_diag_squared=2e-4, s_prior_pose=1e-3, voxel_point_count=80, voxel_distinct_frames=4) for x in (0.01, 0.03, 0.02, 0.04)])
    d3 = os.path.join(root, "armC"); os.makedirs(d3); hm.to_csv(os.path.join(d3, "residual_dump.csv"), index=False)
    run([("armC", d3)], os.path.join(root, "o3")); PL = pd.read_csv(os.path.join(root, "o3", "f485_2b_planes.csv")).iloc[0]
    wantV = 1e-6 + (4e-4 + 2e-4) / 4
    if PL.n != 4 or abs(PL.rbar - 0.025) > 1e-12 or abs(PL.V - wantV) > 1e-12 or abs(PL.excess_wo - (0.025 ** 2 - wantV)) > 1e-12 or abs(PL.excess_wp - (0.025 ** 2 - wantV - 1e-3)) > 1e-12 or PL.distinct_frames != 4:
        fails.append("plane arithmetic %s" % PL.to_dict())
    # ground-height option keeps wz <= H (k % 5 != 0 rows at -0.2), the frame range selects, iter all doubles the planes
    Sg = run([("armA", d)], os.path.join(root, "o4"), ground_height=0.0)
    if Sg[Sg.scope == "all"].n_planes.iloc[0] >= a.n_planes:
        fails.append("ground-height did not reduce the planes")
    Sf = run([("armA", d)], os.path.join(root, "o5"), frames=(402, 404))
    if sorted(Sf.scope) != ["402", "403", "404", "all"]:
        fails.append("frame range scopes %s" % list(Sf.scope))
    Sa = run([("armA", d)], os.path.join(root, "o6"), it="all")
    if Sa[Sa.scope == "all"].n_planes.iloc[0] != 2 * a.n_planes:
        fails.append("--iter all planes")
    # failures: missing columns (old build without the voxel info), no file, empty frame range, inconsistent voxel values
    old = pd.read_csv(os.path.join(d, "residual_dump_all.csv")).drop(columns=["voxel_distinct_frames"]); d4 = os.path.join(root, "armD"); os.makedirs(d4); old.to_csv(os.path.join(d4, "residual_dump_all.csv"), index=False)
    inc = pd.read_csv(os.path.join(d, "residual_dump_all.csv")); k = inc.index[(inc.node_id == inc.node_id.iloc[0]) & (inc["iter"] == 0)][0]; inc.loc[k, "voxel_distinct_frames"] += 5
    d5 = os.path.join(root, "armE"); os.makedirs(d5); inc.to_csv(os.path.join(d5, "residual_dump_all.csv"), index=False)
    for what, fn in (("old build (no voxel columns)", lambda: run([("armD", d4)], os.path.join(root, "o"))), ("no file", lambda: run([("x", tempfile.mkdtemp())], os.path.join(root, "o"))),
                     ("empty frame range", lambda: run([("armA", d)], os.path.join(root, "o"), frames=(1, 2))), ("inconsistent voxel info", lambda: run([("armE", d5)], os.path.join(root, "o")))):
        try:
            fn(); fails.append("%s did not fail" % what)
        except SystemExit:
            pass
    Si = run([("armE", d5)], os.path.join(root, "o7"), allow_inc=True)
    if Si[Si.scope == "all"].n_inconsistent_planes.iloc[0] != 1:
        fails.append("--allow-inconsistent count")
    # CLI
    r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--dir", d, "--out", os.path.join(root, "cli")], capture_output=True, text=True)
    if r.returncode != 0 or list(pd.read_csv(os.path.join(root, "cli", "f485_2b_summary.csv")).columns) != SUMMARY_COLS:
        fails.append("cli rc %s %s" % (r.returncode, r.stderr[-200:]))
    print("SELFTEST r81_f485_2b_reduce", "FAIL " + "; ".join(fails) if fails else
          "PASS (synthetic planted truth: R with and without the pose-prior term, s2 = 3e-4 recovered as the 1/distinct_frames slope and through the origin, the prior appearing as the intercept without it, "
          "a planted 1/n excess is detected and a model-consistent one is not, first iteration only, hand-checked V / r-bar / excess, height and frame selections; old build, no file, empty range and inconsistent voxel info fail)")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", default=[]); ap.add_argument("--root"); ap.add_argument("--out"); ap.add_argument("--file")
    ap.add_argument("--frames", type=int, nargs=2, default=[400, 410]); ap.add_argument("--iter", default="first"); ap.add_argument("--min-rows", type=int, default=1)
    ap.add_argument("--ground-height", type=float, default=None, help="keep rows with wz <= H (F-480); default: all rows of the file (it is already a ground-row dump)")
    ap.add_argument("--sensor-col", default="sigma_diag_squared", help="the dump column used as the sensor part of the per-point variance (ASSUMPTION: sigma_diag_squared)")
    ap.add_argument("--allow-inconsistent", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    arms = [(os.path.basename(d.rstrip("/")), d) for d in a.dir]
    if a.root:
        arms += [(n, os.path.join(a.root, n)) for n in sorted(os.listdir(a.root)) if os.path.isdir(os.path.join(a.root, n))]
    if not arms or not a.out:
        sys.exit("--out and --dir (or --root) are required")
    S = run(arms, a.out, tuple(a.frames), a.iter, a.min_rows, a.ground_height, a.sensor_col, a.file, a.allow_inconsistent)
    print(S[S.scope == "all"][["arm", "n_planes", "n_rows", "R_wo", "R_wp", "scatter_over_noise", "s2_slope_wo", "s2_origin_wo", "s2_slope_wp", "s2_origin_wp"]].to_string(index=False, float_format=lambda x: "%.4g" % x))
