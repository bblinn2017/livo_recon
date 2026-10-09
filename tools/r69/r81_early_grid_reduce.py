#!/usr/bin/env python3
"""R81 (F-474): early-bag grid reducer. One row per ARM per FRAME 1..20 (early_grid_summary.csv), one row per arm (early_grid_arms.csv) and the axis tables as markdown
(early_grid_axis_tables.md). Python only, reads the arms' raw early files (the archive keeps them), no rerun. Definitions follow r72_early_frames.py (F-396 / F-416) and the
metric rule (Bryce 2026-10-05): position and yaw errors are against the FIRST FRAME BEFORE IMU PROPAGATION (the calibration state), NOT against ground truth, in the arm's
own first-frame coordinates (heading removed, see r81_alignfree.py).

    python3 tools/r69/r81_early_grid_reduce.py --root <dir of arm folders> [--glob 'r81_*'] [--arm-table arm_table.tsv] [--ground-height 0.0] [--out-dir <dir>]
    python3 tools/r69/r81_early_grid_reduce.py --selftest

INPUTS per arm folder (the first name found; a missing REQUIRED file is a nonzero exit, never silently absent)
  state_trace.csv[.gz] | state_trace_early_all.csv.gz     phase post_lio rows (the posterior) of scans 1..20: qw..qz, px..pz, P_2_2 (rotation-z variance)         REQUIRED
  state_reference.txt                                      q_wxyz, p of the state BEFORE IMU propagation (written by LioProcBase::ensureStationaryReference)    REQUIRED, see below
  lio_update_diag.csv[.gz] | lio_update_diag_early_all.csv.gz   per frame and iteration: dtheta_x/y/z (rad)                                                       REQUIRED
  residual_dump.csv[.gz]                                   iteration 0 rows of the dumped frames: hpos_x/y/z (world plane normal), hrot_*, sigma_squared, wz       REQUIRED for the wall and ground counts
                                                           (`--allow-missing-residuals` writes blanks and says so on stdout)
  CAUTION: r79_early_reduce.py --early keeps scan_id <= EARLY_LAST = 10 (state_trace_early_all, lio_update_diag_early_all), the grid needs frames to 20: run this tool on the raw arm
  folder, or set EARLY_LAST = 20 in r79_early_reduce.py (listed as a question in C2_REPORT.md).
  state_reference.txt absent -> FALLBACK: the post_imu row of scan 1 (state after the first IMU propagation, the closest row that state_trace.csv holds); flagged in `first_frame_src`
  (state_reference | post_imu_scan1). The calibration state itself is state_reference.txt only.

COLUMNS of early_grid_summary.csv
  arm, bag, samples (calibration samples), ds_mode (imu.ds.mode first|off), pfn (cbk.lidar.point_filter_num), Q (fq|mq), mode (dc|rt): from the arm id, an --arm-table row
      (columns bag, samples, ds, pfn, q, mode override the id) -- expected id: r81_<mode>_<family>_<bag>_n<samples>_ds<first|off>_pfn<1|3>_<fq|mq>
  frame            1..20 (state_trace scan_id = lio_update_diag frame = residual_dump frame); have_state 0 when the arm has no posterior row for it (blank values)
  pos_err_mm       |p - p_first_frame| of the posterior (mm); ex_mm, ey_mm, ez_mm its components in first-frame coordinates (heading removed)
  yaw_err_mrad     wrap(yaw - yaw_first_frame) in mrad (yaw = atan2(R10, R00)); rot_err_mrad = |Log(R_ff^T R)| in mrad (full rotation error)
  post_sd_yaw_mrad sqrt(P_2_2) of the posterior (mrad)
  n_iter, yaw_step_first_mrad, yaw_step_last_mrad   iterations of the frame, dtheta_z of iteration 0 and of the last iteration (mrad); rot_step_last_mrad = |dtheta| of the last iteration
  converged        1 when |yaw_step_last| < 1 mrad (Bryce / F-382 wording: last step < 1 mrad; yaw step). converged_rotnorm: the same with the rotation step norm (stricter), recorded beside it
  n_resid, n_ground, n_wall, n_slanted   iteration-0 residual counts of the frame by plane normal n = (hpos_x, hpos_y, hpos_z): ground |nz| >= 0.8, wall |nz| < 0.35 (F-417; r72 used <=, equal only at
                   exactly 0.35), slanted otherwise
  with --ground-height H (world z of the floor; F-480 provisional 0.0): n_above_h (world z wz > H), n_at_or_below_h, n_ground_label_above_h (|nz| >= 0.8 AND wz > H: the facade/tree points the
                   normal calls ground), n_wall_above_h
  believed_sd_yaw_mrad   r72 definition: LiDAR-only pinv(Lambda), Lambda = sum h h^T / sigma^2 over the iteration-0 residuals of the frame, yaw = rotation axis 2, mrad
  first_wall_ge1 / ge10 / ge100   first frame of 1..20 with >= 1 / 10 / 100 wall residuals (blank when never reached; the same value on every row of the arm)
  excursion_10_mm  max over scans 1..10 of pos_err_mm (the same on every row of the arm)
  first_frame_src

AXIS TABLES (markdown): per mode (dc, rt: modes are never pooled), per metric (excursion over scans 1..10; mean |yaw error| over frames 1..20; share of frames converged; first frame with >= 10
wall residuals): rows = axis level (Q, pfn, ds mode, samples), columns = one per bag and the pooled column (pooled over the bags is a convenience: the per-bag columns are the data, rule 4.1);
cell = median over the arms of the group (n arms). Paired tables for the two-level axes (Q, pfn, ds mode, and mode itself): the ratio of the second level to the first of the excursion on
cells matched on every other axis (the F-416 definition): median ratio, and k of N pairs where the second level is lower.
Planning agent's script. Run by me only on synthetic arm folders.
"""
import argparse, csv, fnmatch, math, os, re, sys
import numpy as np
import pandas as pd

BAG = r"[a-z]+_\d\d"
FIRST, LAST = 1, 20
ARM_COLS = ["arm", "bag", "samples", "ds_mode", "pfn", "Q", "mode"]
FRAME_COLS = ARM_COLS + ["frame", "have_state", "pos_err_mm", "ex_mm", "ey_mm", "ez_mm", "yaw_err_mrad", "rot_err_mrad", "post_sd_yaw_mrad", "n_iter", "yaw_step_first_mrad", "yaw_step_last_mrad",
                         "rot_step_last_mrad", "converged", "converged_rotnorm", "n_resid", "n_ground", "n_wall", "n_slanted"]
HEIGHT_COLS = ["n_above_h", "n_at_or_below_h", "n_ground_label_above_h", "n_wall_above_h"]
TAIL_COLS = ["believed_sd_yaw_mrad", "first_wall_ge1", "first_wall_ge10", "first_wall_ge100", "excursion_10_mm", "first_frame_src"]


# ----------------------------------------------------------------------------------------------------------------- small maths
def wrap(a):
    return (np.asarray(a, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def quat_R(q):
    """(N,4) w,x,y,z -> (N,3,3)."""
    q = np.atleast_2d(np.asarray(q, dtype=np.float64)); q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def log_so3(R):
    c = min(1.0, max(-1.0, (np.trace(R) - 1.0) / 2.0)); th = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-9:
        return 0.5 * v
    if abs(math.pi - th) < 1e-6:
        A = (R + np.eye(3)) / 2.0; ax = np.sqrt(np.maximum(np.diag(A), 0)); k = int(np.argmax(ax)); ax = A[:, k] / max(ax[k], 1e-12)
        return th * ax / np.linalg.norm(ax)
    return th / (2.0 * math.sin(th)) * v


def pinv_psd(L, rel=1e-10):
    w, V = np.linalg.eigh(0.5 * (L + L.T)); keep = w > rel * max(w.max(), 1e-300)
    return (V[:, keep] / w[keep]) @ V[:, keep].T


def first_file(d, names):
    for n in names:
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p
    return None


def read_reference(path):
    dd = {}
    for line in open(path):
        t = line.split()
        if t and t[0] in ("q_wxyz", "p"):
            dd[t[0]] = [float(x) for x in t[1:]]
    if len(dd.get("q_wxyz", [])) != 4 or len(dd.get("p", [])) != 3:
        raise SystemExit("%s lacks q_wxyz / p" % path)
    return quat_R(np.array([dd["q_wxyz"]]))[0], np.array(dd["p"])


def parse_arm(arm, row=None):
    """-> dict(bag, samples, ds_mode, pfn, Q, mode); an --arm-table row overrides. Unparsed fields are None."""
    row = row or {}
    g = lambda pat: (re.search(pat, arm).group(1) if re.search(pat, arm) else None)
    m = re.match(r"r81_(dc|rt)_", arm)
    q = re.findall(r"(?:^|_)(fq|mq)(?=_|$)", arm)
    d = dict(bag=g("(%s)" % BAG), samples=g(r"_n(\d+)(?:_|$)"), ds_mode=g(r"_ds(first|off)(?:_|$)"), pfn=g(r"_pfn(\d+)(?:_|$)"), Q=(q[-1] if q else None), mode=(m.group(1) if m else None))
    for k, c in (("bag", "bag"), ("samples", "samples"), ("ds_mode", "ds"), ("pfn", "pfn"), ("Q", "q"), ("mode", "mode")):
        if row.get(c):
            d[k] = row[c]
    if d["samples"] is not None: d["samples"] = int(d["samples"])
    if d["pfn"] is not None: d["pfn"] = int(d["pfn"])
    return d


# ----------------------------------------------------------------------------------------------------------------- one arm
def arm_frames(arm_dir, arm, meta, ground_height=None, allow_missing_residuals=False):
    """-> (DataFrame rows frame 1..20, notes list)."""
    notes = []
    tp = first_file(arm_dir, ["state_trace.csv", "state_trace.csv.gz", "state_trace_early_all.csv.gz"])
    dp = first_file(arm_dir, ["lio_update_diag.csv", "lio_update_diag.csv.gz", "lio_update_diag_early_all.csv.gz"])
    rp = first_file(arm_dir, ["residual_dump.csv", "residual_dump.csv.gz"])
    sp = first_file(arm_dir, ["state_reference.txt"])
    if tp is None: raise SystemExit("%s: no state_trace file" % arm_dir)
    if dp is None: raise SystemExit("%s: no lio_update_diag file" % arm_dir)
    if rp is None and not allow_missing_residuals: raise SystemExit("%s: no residual_dump.csv (--allow-missing-residuals to write blank wall/ground counts)" % arm_dir)
    ST = pd.read_csv(tp)
    for c in ("scan_id", "phase", "qw", "qx", "qy", "qz", "px", "py", "pz"):
        if c not in ST.columns: raise SystemExit("%s lacks column %s" % (tp, c))
    post = ST[ST.phase == "post_lio"].drop_duplicates("scan_id", keep="last").set_index("scan_id").sort_index()
    if len(post) == 0: raise SystemExit("%s has no post_lio rows" % tp)
    if sp is not None:
        Rff, pff = read_reference(sp); src = "state_reference"
    else:
        imu = ST[ST.phase == "post_imu"].drop_duplicates("scan_id", keep="last").set_index("scan_id").sort_index()
        if len(imu) == 0: raise SystemExit("%s: no state_reference.txt and no post_imu row" % arm_dir)
        r0 = imu.iloc[0]; Rff = quat_R(np.array([[r0.qw, r0.qx, r0.qy, r0.qz]]))[0]; pff = np.array([r0.px, r0.py, r0.pz]); src = "post_imu_scan1"
        notes.append("first frame = post_imu row of scan %d (no state_reference.txt)" % imu.index[0])
    yaw_ff = math.atan2(Rff[1, 0], Rff[0, 0]); c, s = math.cos(yaw_ff), math.sin(yaw_ff)
    UD = pd.read_csv(dp)
    for col in ("frame", "iter", "dtheta_x", "dtheta_y", "dtheta_z"):
        if col not in UD.columns: raise SystemExit("%s lacks column %s" % (dp, col))
    RD = None
    if rp is not None:
        RD = pd.read_csv(rp)
        for col in ("frame", "hpos_x", "hpos_y", "hpos_z", "hrot_x", "hrot_y", "hrot_z", "sigma_squared") + (("wz",) if ground_height is not None else ()):
            if col not in RD.columns: raise SystemExit("%s lacks column %s" % (rp, col))
        if "iter" in RD.columns: RD = RD[RD["iter"] == 0]
    rows = []
    for fr in range(FIRST, LAST + 1):
        r = dict(arm=arm, **meta, frame=fr)
        have = fr in post.index
        r["have_state"] = int(have)
        if have:
            q = post.loc[fr]
            R = quat_R(np.array([[q.qw, q.qx, q.qy, q.qz]]))[0]; d = np.array([q.px, q.py, q.pz]) - pff
            e = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]]) * 1000.0
            r.update(pos_err_mm=float(np.linalg.norm(e)), ex_mm=e[0], ey_mm=e[1], ez_mm=e[2],
                     yaw_err_mrad=float(wrap(math.atan2(R[1, 0], R[0, 0]) - yaw_ff)) * 1000.0, rot_err_mrad=float(np.linalg.norm(log_so3(Rff.T @ R))) * 1000.0)
            if "P_2_2" in post.columns and pd.notna(q.P_2_2):
                r["post_sd_yaw_mrad"] = math.sqrt(max(float(q.P_2_2), 0.0)) * 1000.0
        U = UD[UD.frame == fr].sort_values("iter")
        r["n_iter"] = int(len(U))
        if len(U):
            dz = U["dtheta_z"].to_numpy(float); th = U[["dtheta_x", "dtheta_y", "dtheta_z"]].to_numpy(float)
            r.update(yaw_step_first_mrad=dz[0] * 1e3, yaw_step_last_mrad=dz[-1] * 1e3, rot_step_last_mrad=float(np.linalg.norm(th[-1])) * 1e3,
                     converged=int(abs(dz[-1]) * 1e3 < 1.0), converged_rotnorm=int(np.linalg.norm(th[-1]) * 1e3 < 1.0))
        if RD is not None:
            g = RD[RD.frame == fr]
            n = len(g); r["n_resid"] = n
            if n:
                nz = np.abs(g["hpos_z"].to_numpy(float))
                gm, wm = nz >= 0.8, nz < 0.35
                r.update(n_ground=int(gm.sum()), n_wall=int(wm.sum()), n_slanted=int(n - gm.sum() - wm.sum()))
                H = g[["hrot_x", "hrot_y", "hrot_z", "hpos_x", "hpos_y", "hpos_z"]].to_numpy(float); w = 1.0 / g["sigma_squared"].to_numpy(float)
                inv = pinv_psd((H * w[:, None]).T @ H)
                r["believed_sd_yaw_mrad"] = math.sqrt(max(inv[2, 2], 0.0)) * 1e3
                if ground_height is not None:
                    up = g["wz"].to_numpy(float) > ground_height
                    r.update(n_above_h=int(up.sum()), n_at_or_below_h=int(n - up.sum()), n_ground_label_above_h=int((gm & up).sum()), n_wall_above_h=int((wm & up).sum()))
        rows.append(r)
    T = pd.DataFrame(rows)
    T["first_frame_src"] = src
    for k in (1, 10, 100):
        hit = T[(T.get("n_wall", pd.Series(dtype=float)) >= k)] if "n_wall" in T else T.iloc[0:0]
        T["first_wall_ge%d" % k] = int(hit.frame.iloc[0]) if len(hit) else np.nan
    m10 = T[(T.frame <= 10) & (T.have_state == 1)]
    T["excursion_10_mm"] = float(m10.pos_err_mm.max()) if len(m10) else np.nan
    if int(T.have_state.sum()) < LAST: notes.append("posterior rows for %d of %d frames" % (T.have_state.sum(), LAST))
    return T, notes


# ----------------------------------------------------------------------------------------------------------------- tables
def arm_table(F):
    rows = []
    for arm, g in F.groupby("arm", sort=True):
        h = g.iloc[0]
        have = g[g.have_state == 1]
        rows.append(dict(arm=arm, bag=h.bag, samples=h.samples, ds_mode=h.ds_mode, pfn=h.pfn, Q=h.Q, mode=h["mode"], excursion_10_mm=h.excursion_10_mm,
                         yaw_abs_mean_mrad=float(have.yaw_err_mrad.abs().mean()) if len(have) else np.nan,
                         pos_err_frame10_mm=float(g[g.frame == 10].pos_err_mm.iloc[0]) if (g.frame == 10).any() else np.nan,
                         pos_err_frame20_mm=float(g[g.frame == 20].pos_err_mm.iloc[0]) if (g.frame == 20).any() else np.nan,
                         frac_converged=float(g.converged.dropna().mean()) if g.converged.notna().any() else np.nan,
                         first_wall_ge1=h.first_wall_ge1, first_wall_ge10=h.first_wall_ge10, first_wall_ge100=h.first_wall_ge100,
                         n_wall_frame1=(g[g.frame == 1].n_wall.iloc[0] if "n_wall" in g and (g.frame == 1).any() else np.nan),
                         believed_sd_yaw_frame1_mrad=(g[g.frame == 1].believed_sd_yaw_mrad.iloc[0] if "believed_sd_yaw_mrad" in g else np.nan)))
    return pd.DataFrame(rows)


METRICS = [("excursion_10_mm", "Excursion over scans 1..10 (mm; max position error against the first frame), median over arms"),
           ("yaw_abs_mean_mrad", "Mean |yaw error| over frames 1..20 (mrad; against the first frame), median over arms"),
           ("frac_converged", "Share of frames 1..20 that converged (last yaw step < 1 mrad), median over arms"),
           ("first_wall_ge10", "First frame with >= 10 wall residuals (median over the arms that reach it; k/N reached)")]
AXES = [("Q", "Q"), ("pfn", "pfn"), ("ds_mode", "ds mode"), ("samples", "samples")]
BINARY = [("Q", ["fq", "mq"], "mq / fq"), ("pfn", [1, 3], "pfn 3 / pfn 1"), ("ds_mode", ["first", "off"], "ds off / ds first")]


def cell(v, metric):
    v = pd.Series(v)
    if metric == "first_wall_ge10":
        r = v.dropna()
        return "- (0/%d)" % len(v) if not len(r) else "%g (%d/%d)" % (r.median(), len(r), len(v))
    v = v.dropna()
    return "-" if not len(v) else ("%.3g (%d)" % (v.median(), len(v)))


def metric_table(A, metric, title):
    bags = sorted(A.bag.dropna().unique())
    L = ["### " + title, "", "| axis = level | " + " | ".join(bags) + " | pooled |", "|---|" + "---|" * (len(bags) + 1)]
    for col, nm in AXES:
        for lv in sorted(A[col].dropna().unique(), key=lambda x: (isinstance(x, str), x)):
            sel = A[A[col] == lv]
            L.append("| %s = %s | " % (nm, lv) + " | ".join(cell(sel[sel.bag == b][metric], metric) for b in bags) + " | " + cell(sel[metric], metric) + " |")
    return "\n".join(L) + "\n"


def pairs(A, col, lv, metric="excursion_10_mm", ignore=None):
    """matched pairs (first level, second level) on every other axis; -> DataFrame bag, a, b."""
    keys = [c for c in ("bag", "samples", "ds_mode", "pfn", "Q", "mode") if c != col]
    a, b = A[A[col] == lv[0]], A[A[col] == lv[1]]
    M = a.merge(b, on=keys, suffixes=("_a", "_b"))
    return M[["bag", metric + "_a", metric + "_b"]].rename(columns={metric + "_a": "a", metric + "_b": "b"}) if len(M) else pd.DataFrame(columns=["bag", "a", "b"])


def pair_cell(M):
    M = M.dropna()
    M = M[M.a > 0]
    if not len(M): return "-"
    r = (M.b / M.a)
    return "%.3g (%d/%d lower)" % (r.median(), int((M.b < M.a).sum()), len(M))


def paired_table(A, title, specs, metric="excursion_10_mm"):
    bags = sorted(A.bag.dropna().unique())
    L = ["### " + title, "", "| ratio | " + " | ".join(bags) + " | pooled |", "|---|" + "---|" * (len(bags) + 1)]
    for col, lv, nm in specs:
        M = pairs(A, col, lv, metric)
        L.append("| %s | " % nm + " | ".join(pair_cell(M[M.bag == b]) for b in bags) + " | " + pair_cell(M) + " |")
    return "\n".join(L) + "\n"


def axis_markdown(A):
    out = ["# F-474 early-bag grid: axis tables", "",
           "Position and yaw are against the first frame before IMU propagation (the calibration state), in each arm's own first-frame coordinates; ground truth is not used. "
           "Cells: median over the arms of the group (n arms in brackets). Per-bag columns are the data; the pooled column pools the bags and is a convenience (rule 4.1). "
           "dc and rt are never pooled.", ""]
    for mode in sorted(A["mode"].dropna().unique()):
        S = A[A["mode"] == mode]
        out += ["## mode %s (%d arms)" % (mode, len(S)), ""]
        for m, t in METRICS:
            out.append(metric_table(S, m, t))
        out.append(paired_table(S, "Paired excursion ratio on cells matched on every other axis (second / first level; k of N pairs lower)", BINARY))
        out.append(paired_table(S, "Paired mean |yaw error| ratio", BINARY, "yaw_abs_mean_mrad"))
    if set(A["mode"].dropna().unique()) >= {"dc", "rt"}:
        out += ["## mode axis: rt / dc on matched cells", "", paired_table(A, "Excursion ratio rt / dc", [("mode", ["dc", "rt"], "rt / dc")]),
                paired_table(A, "Mean |yaw error| ratio rt / dc", [("mode", ["dc", "rt"], "rt / dc")], "yaw_abs_mean_mrad")]
    return "\n".join(out) + "\n"


def run(a):
    table = {}
    if a.arm_table:
        table = {r["arm"]: r for r in csv.DictReader(open(a.arm_table), delimiter="\t")}
    arms = sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d)) and fnmatch.fnmatch(d, a.glob))
    frames, problems = [], []
    for arm in arms:
        meta = parse_arm(arm, table.get(arm))
        miss = [k for k, v in meta.items() if v is None]
        if miss:
            problems.append("%s: not parsed %s" % (arm, miss)); continue
        T, notes = arm_frames(os.path.join(a.root, arm), arm, meta, a.ground_height, a.allow_missing_residuals)
        frames.append(T)
        for n in notes: print("[%s] %s" % (arm, n))
    if not frames:
        raise SystemExit("no arm folder matched %s in %s (%s)" % (a.glob, a.root, problems[:3]))
    F = pd.concat(frames, ignore_index=True)
    cols = FRAME_COLS + (HEIGHT_COLS if a.ground_height is not None else []) + TAIL_COLS
    for c in cols:
        if c not in F.columns: F[c] = np.nan
    os.makedirs(a.out_dir, exist_ok=True)
    F[cols].to_csv(os.path.join(a.out_dir, "early_grid_summary.csv"), index=False, float_format="%.9g")
    A = arm_table(F)
    A.to_csv(os.path.join(a.out_dir, "early_grid_arms.csv"), index=False, float_format="%.9g")
    open(os.path.join(a.out_dir, "early_grid_axis_tables.md"), "w").write(axis_markdown(A))
    print("early grid: %d arms, %d rows (frames %d..%d), problems %d %s" % (len(A), len(F), FIRST, LAST, len(problems), problems[:4]))
    return 1 if problems else 0


# ----------------------------------------------------------------------------------------------------------------- selftest
def _rz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def _make_arm(d, yaw_ff, p_ff, e_mm, dyaw_mrad, steps, walls, ground_up, with_ref=True, sd_mrad=2.0):
    """synthetic arm folder. Frame k (1..20): posterior = first frame + Rz(yaw_ff) [e_mm[k-1] mm, 0, 0], yaw = yaw_ff + dyaw_mrad[k-1]/1000.
    steps[k] = list of yaw steps (mrad) per iteration; walls[k] wall residuals; ground_up[k] ground-label residuals above the floor."""
    os.makedirs(d, exist_ok=True)
    rows = []
    for k in range(0, 21):
        e = (e_mm[k - 1] if k >= 1 else 0.0) / 1000.0; dy = (dyaw_mrad[k - 1] if k >= 1 else 0.0) / 1000.0
        p = np.array(p_ff) + _rz(yaw_ff) @ np.array([e, 0, 0]); y = yaw_ff + dy
        for ph in ("post_imu", "post_lio"):
            rows.append(dict(run_id="x", scan_id=k, t_abs=k * 0.1, phase=ph, qw=math.cos(y / 2), qx=0.0, qy=0.0, qz=math.sin(y / 2), px=p[0], py=p[1], pz=p[2], P_2_2=(sd_mrad / 1000.0) ** 2))
    pd.DataFrame(rows).to_csv(os.path.join(d, "state_trace.csv"), index=False)
    if with_ref:
        with open(os.path.join(d, "state_reference.txt"), "w") as f:
            f.write("schema state_reference_v1\nq_wxyz %.15g 0 0 %.15g\np %.15g %.15g %.15g\nv 0 0 0\nbag_start_abs 0\nP0_dim 1\nP0_row_major 1\n" % (math.cos(yaw_ff / 2), math.sin(yaw_ff / 2), *p_ff))
    U = []
    for k in range(1, 21):
        for i, st in enumerate(steps.get(k, [0.5])):
            U.append(dict(frame=k, iter=i, dtheta_x=0.0, dtheta_y=0.0, dtheta_z=st / 1000.0, dt_x=0.0, dt_y=0.0, dt_z=0.0))
    pd.DataFrame(U).to_csv(os.path.join(d, "lio_update_diag.csv"), index=False)
    rng = np.random.default_rng(3); R = []
    for k in range(1, 21):
        for i in range(60):
            nz = 0.99 if i < 40 else 0.6
            wz = 0.5 if i < ground_up.get(k, 0) else -0.4
            R.append((k, 0, 1e-3, 0.0, 0.0, nz, wz))
        for i in range(walls.get(k, 0)):
            R.append((k, 0, 1e-3, 1.0, 0.0, 0.1, 5.0))
        R.append((k, 1, 1e-3, 0.0, 0.0, 0.99, -0.4))      # an iteration-1 row: must be ignored
    RD = pd.DataFrame(R, columns=["frame", "iter", "sigma_squared", "hpos_x", "hpos_y", "hpos_z", "wz"])
    for c in ("hrot_x", "hrot_y", "hrot_z"):
        RD[c] = rng.normal(size=len(RD))
    RD.to_csv(os.path.join(d, "residual_dump.csv"), index=False)


def selftest():
    import tempfile, shutil
    fails = []

    def chk(c, m):
        if not c: fails.append(m)
    t = tempfile.mkdtemp()
    try:
        e = [1.0 * k for k in range(1, 21)]                            # 1, 2, ..., 20 mm
        dy = [3.0] * 20; dy[4] = -12.0                                  # yaw error 3 mrad, -12 mrad at frame 5
        steps = {1: [30.0, -20.0, 5.0], 2: [4.0, 0.4], 3: [-53.6, 48.1, -24.9, 24.9, -25.0]}
        walls = {k: min(k * 4, 150) for k in range(1, 21)}              # 4, 8, ..., 80, 84.. ; >= 100 first at frame 25 -> never inside 1..20? 4*20=80 < 100
        walls[20] = 120                                                 # reaches 100 at frame 20
        up = {k: 7 for k in range(1, 21)}
        base = os.path.join(t, "root")
        arms = {"r81_dc_ef_eee_01_n200_dsfirst_pfn1_fq": (0.7, (2.0, 3.0, 0.1), 1.0), "r81_dc_ef_eee_01_n200_dsfirst_pfn1_mq": (-2.0, (0.0, 0.0, 0.0), 0.1),
                "r81_dc_ef_eee_01_n200_dsoff_pfn1_fq": (2.5, (1.0, 1.0, 0.0), 0.7), "r81_rt_ef_eee_01_n200_dsfirst_pfn1_fq": (1.1, (2.0, 3.0, 0.1), 0.5)}
        for nm, (yff, pff, sc) in arms.items():
            _make_arm(os.path.join(base, nm), yff, pff, [x * sc for x in e], dy, steps, walls, up, with_ref=True)
        # an arm without state_reference.txt: falls back to the post_imu row of scan 0/1 and is flagged
        _make_arm(os.path.join(base, "r81_dc_ef_nya_01_n400_dsoff_pfn3_mq"), 0.2, (0, 0, 0), e, dy, steps, walls, up, with_ref=False)
        _make_arm(os.path.join(base, "r81_dc_ef_nya_01_n400_dsoff_pfn3_fq"), 0.2, (0, 0, 0), [x * 3 for x in e], dy, steps, walls, up, with_ref=True)
        T, notes = arm_frames(os.path.join(base, "r81_dc_ef_eee_01_n200_dsfirst_pfn1_fq"), "a", parse_arm("r81_dc_ef_eee_01_n200_dsfirst_pfn1_fq"), 0.0)
        chk(len(T) == 20 and list(T.frame) == list(range(1, 21)), "20 rows frame 1..20")
        chk(abs(T.pos_err_mm[2] - 3.0) < 1e-6 and abs(T.ex_mm[2] - 3.0) < 1e-6 and abs(T.ey_mm[2]) < 1e-6, "position error in first-frame coordinates (heading removed): %r %r %r" % (T.pos_err_mm[2], T.ex_mm[2], T.ey_mm[2]))
        chk(abs(T.yaw_err_mrad[4] + 12.0) < 1e-6 and abs(T.yaw_err_mrad[0] - 3.0) < 1e-6, "yaw error vs the first frame: %r %r" % (T.yaw_err_mrad[4], T.yaw_err_mrad[0]))
        chk(abs(T.rot_err_mrad[4] - 12.0) < 1e-6, "rot err norm")
        chk(abs(T.post_sd_yaw_mrad[0] - 2.0) < 1e-9, "post sd yaw from P_2_2")
        chk(abs(T.excursion_10_mm.iloc[0] - 10.0) < 1e-9, "excursion over scans 1..10: %r" % T.excursion_10_mm.iloc[0])
        chk(T.n_iter[2] == 5 and abs(T.yaw_step_first_mrad[2] + 53.6) < 1e-9 and abs(T.yaw_step_last_mrad[2] + 25.0) < 1e-9 and T.converged[2] == 0, "frame 3 steps: first -53.6 last -25.0, not converged")
        chk(T.converged[0] == 0 and T.converged[1] == 1 and T.yaw_step_last_mrad[0] == 5.0, "converged flag: |last| < 1 mrad (frame 1 last 5 -> 0; frame 2 last 0.4 -> 1)")
        chk(list(T.n_wall[:3]) == [4, 8, 12] and T.n_ground[0] == 40 and T.n_slanted[0] == 20 and T.n_resid[0] == 64, "class counts frame 1: %r" % T.iloc[0][["n_wall", "n_ground", "n_slanted", "n_resid"]].to_dict())
        chk(T.n_ground_label_above_h[0] == 7 and T.n_wall_above_h[0] == 4 and T.n_above_h[0] == 11 and T.n_at_or_below_h[0] == 53, "height classes: %r" % T.iloc[0][HEIGHT_COLS].to_dict())
        chk(T.first_wall_ge1.iloc[0] == 1 and T.first_wall_ge10.iloc[0] == 3 and T.first_wall_ge100.iloc[0] == 20, "first frame with 1/10/100 walls: %r %r %r" % (T.first_wall_ge1.iloc[0], T.first_wall_ge10.iloc[0], T.first_wall_ge100.iloc[0]))
        # believed sd against an independent pinv of the same Lambda (frame 1)
        RD = pd.read_csv(os.path.join(base, "r81_dc_ef_eee_01_n200_dsfirst_pfn1_fq", "residual_dump.csv")); g = RD[(RD.frame == 1) & (RD.iter == 0)]
        H = g[["hrot_x", "hrot_y", "hrot_z", "hpos_x", "hpos_y", "hpos_z"]].to_numpy(); L = (H / g.sigma_squared.to_numpy()[:, None]).T @ H
        chk(abs(T.believed_sd_yaw_mrad[0] - 1e3 * math.sqrt(np.linalg.pinv(L, hermitian=True)[2, 2])) < 1e-6 * T.believed_sd_yaw_mrad[0], "believed sd yaw")
        chk(T.first_frame_src.iloc[0] == "state_reference" and not notes, "source state_reference, no notes")
        T2, notes2 = arm_frames(os.path.join(base, "r81_dc_ef_nya_01_n400_dsoff_pfn3_mq"), "b", parse_arm("r81_dc_ef_nya_01_n400_dsoff_pfn3_mq"))
        chk(T2.first_frame_src.iloc[0] == "post_imu_scan1" and notes2 and "post_imu" in notes2[0], "fallback flagged: %r" % notes2)
        chk("n_above_h" not in T2.columns or T2.n_above_h.isna().all(), "no height columns without --ground-height")
        # CLI end to end
        out = os.path.join(t, "out")
        ns = argparse.Namespace(root=base, glob="r81_*", arm_table=None, ground_height=0.0, out_dir=out, allow_missing_residuals=False)
        rc = run(ns)
        S = pd.read_csv(os.path.join(out, "early_grid_summary.csv")); A = pd.read_csv(os.path.join(out, "early_grid_arms.csv")); md = open(os.path.join(out, "early_grid_axis_tables.md")).read()
        chk(rc == 0 and len(S) == 6 * 20 and len(A) == 6, "6 arms x 20 rows: rc %r rows %d" % (rc, len(S)))
        chk(list(S.columns[:7]) == ARM_COLS and "n_ground_label_above_h" in S.columns, "columns")
        a1 = A[A.arm == "r81_dc_ef_eee_01_n200_dsfirst_pfn1_fq"].iloc[0]
        chk(a1.samples == 200 and a1.ds_mode == "first" and a1.pfn == 1 and a1.Q == "fq" and a1["mode"] == "dc" and a1.bag == "eee_01", "arm id parsed: %r" % a1.to_dict())
        # Axis tables: eee_01 dc, Q pair (mq scale 0.1 of fq) -> ratio 0.1 (1/1 lower); ds pair fq: first 10 mm vs off 7 mm -> 0.7 (1/1 lower); rt/dc: 5/10 = 0.5
        chk("## mode dc" in md and "## mode rt" in md and "## mode axis: rt / dc" in md, "markdown sections")
        chk("| mq / fq | 0.1 (1/1 lower)" in md.replace("0.100", "0.1"), "paired Q ratio mq/fq on eee_01 = 0.1: %s" % [l for l in md.splitlines() if "mq / fq" in l][:1])
        chk("| ds off / ds first | 0.7 (1/1 lower)" in md, "paired ds ratio 0.7: %s" % [l for l in md.splitlines() if "ds off" in l][:1])
        chk("| rt / dc | 0.5 (1/1 lower)" in md, "rt/dc ratio 0.5: %s" % [l for l in md.splitlines() if "rt / dc" in l][:2])
        # a missing required file is an error, never silent
        os.remove(os.path.join(base, "r81_dc_ef_eee_01_n200_dsoff_pfn1_fq", "lio_update_diag.csv"))
        try:
            run(ns); fails.append("missing lio_update_diag did not fail")
        except SystemExit:
            pass
        _make_arm(os.path.join(base, "r81_dc_ef_eee_01_n200_dsoff_pfn1_fq"), 2.5, (1, 1, 0), e, dy, steps, walls, up)
        os.remove(os.path.join(base, "r81_dc_ef_eee_01_n200_dsoff_pfn1_fq", "residual_dump.csv"))
        try:
            run(ns); fails.append("missing residual_dump did not fail")
        except SystemExit:
            pass
        ns.allow_missing_residuals = True
        ns.glob = "r81_dc_ef_eee_01_n200_dsoff_pfn1_fq"
        chk(run(ns) == 0 and pd.read_csv(os.path.join(out, "early_grid_summary.csv")).n_wall.isna().all(), "allow-missing-residuals writes blank counts")
        ns.glob = "no_such_*"
        try:
            run(ns); fails.append("no matching arm did not fail")
        except SystemExit:
            pass
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("SELFTEST r81_early_grid_reduce", ("FAIL " + "; ".join(fails)) if fails else
          "PASS (20 frames per arm; position and yaw error against the first frame in first-frame coordinates; iteration steps and converged flag; wall/ground/slanted and height counts at iteration 0; "
          "first frame with 1/10/100 walls; believed sd equals an independent pinv; excursion over scans 1..10; post_imu fallback flagged; arm id parsing; paired ratio tables for Q, ds, rt/dc; "
          "missing files fail)")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--glob", default="r81_*"); ap.add_argument("--arm-table"); ap.add_argument("--ground-height", type=float, default=None)
    ap.add_argument("--out-dir", default="."); ap.add_argument("--allow-missing-residuals", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.root:
        sys.exit("--root is required")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
