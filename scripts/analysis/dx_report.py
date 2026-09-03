#!/usr/bin/env python3
"""P9 -- the DX-1R analysis, and it emits blocks to paste into the register.

DX-1 logged everything and reported one number per cell because the join
across frame_stats.txt, scan.csv, corr_scan.csv, spline_q.csv, pose_pair.csv
and GT was left as an exercise. This is that join, run once per cell inside
the scoring pass (before corr.csv is deleted), emitting three machine-parsed
blocks: a per-cell time series (Block 1, ladder cells only), a per-cell
summary (Block 2, all cells), and a pairwise attribution across adjacent
rungs (Block 3, the "why").

Reuses traj_divergence.py's analyse()/prefix_error/window_error and
bench_eval/evaluate.py's umeyama_no_scale/resample directly, so the
alignment convention is identical to the scorer's (rule: verify #1 below).

Usage (single cell):
  dx_report.py --run-dir livo_recon_results/dx1r/dx1r_<cell>_eee_01 \
               --gt bench_eval/gt/eee_01.csv --gt-format ntu_csv \
               --frame-correction ntu_prism --cell <cell> --out <dir>

Usage (the full DX-1R batch, all three blocks in one shot):
  dx_report.py --batch-root livo_recon_results/dx1r --gt ... --out <dir>
"""
import argparse, csv, json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _c in [os.environ.get("BENCH_EVAL_DIR"),
           os.path.join(_HERE, "..", "..", "bench_eval"),
           os.path.join(_HERE, "..", "..", "..", "bench_eval"),
           os.path.join(_HERE, "..", "..", "..", "..", "bench_eval")]:
    if _c and os.path.exists(os.path.join(_c, "evaluate.py")):
        sys.path.insert(0, _c)
        break
try:
    from evaluate import load_tum, LOADERS, apply_frame_correction, resample, umeyama_no_scale
except ImportError:
    sys.exit("dx_report.py: cannot find bench_eval/evaluate.py. Set BENCH_EVAL_DIR.")

sys.path.insert(0, _HERE)
from traj_divergence import prefix_error, window_error, detect_onset  # noqa: E402

LADDER_CELLS = ["l0", "l1", "l2", "l3", "l4", "l5", "l6",
                "l0d", "l1d", "l2d", "l3d", "l4d"]
# l5d dropped: P7 refuses debiased+information_directional as unreachable
# at startup (see voxelplane.cpp/voxelmap.cpp), so no DX-1R dispatch ever
# produces this cell -- it isn't just missing data, it structurally can't
# exist. Was previously left in this list as a no-data placeholder
# ("comes back None/skipped gracefully"), which silently inflated every
# "want N" footer below (13 ladder cells * 40 = 520 vs the true 12*40=480,
# 21 total cells vs the true 20, 13 pairs vs the true 11) -- fixed by
# removing it here instead of padding the footer to match.
ALL_CELLS = LADDER_CELLS + ["s1", "s2", "n_l0", "n_l2", "n_l5", "r_l0", "r_l2", "r_l5"]
# S3 dropped per rule 28c -- it resolved to exactly L3, so it contributes
# nothing Block 2/3 wouldn't already have from L3 itself.

# Built programmatically, not as a literal list, to avoid a copy/paste typo
# in 13 hand-written tuples. 6 pca-arm adjacent (L0->L1..L5->L6) + 5
# debiased-arm adjacent (L0d->L1d..L4d->L5d) + 2 cross-arm (L4->L4d,
# L5->L5d) = 13. The card's own prose says "the same six on the debiased
# arm", but the debiased ladder (L0d..L5d, 6 rungs) only has 5 adjacent
# gaps -- there is no L6d. Implemented as 13 pairs, not 14; flagged here and
# in the report rather than silently padding to match the card's count.
PAIRS = (list(zip(["l0", "l1", "l2", "l3", "l4"], ["l1", "l2", "l3", "l4", "l5"]))
         + [("l5", "l6")]
         + list(zip(["l0d", "l1d", "l2d", "l3d"], ["l1d", "l2d", "l3d", "l4d"]))
         + [("l4", "l4d")])
# l4d->l5d and l5->l5d dropped along with l5d itself (see LADDER_CELLS).
assert len(PAIRS) == 11, len(PAIRS)

# trajectory_mode: discrete cells (spline.enable=false in gen_dx1_manifest.py's
# DISCRETE override) -- used by DX1R-VERIFY check 3 (refit_dtraj_rms must be 0
# on these, since there is no spline to refit at all).
DISCRETE_CELLS = {"l0", "l0d", "r_l0", "n_l0"}

DIAGNOSTICS = ["refusal", "div_pos", "div_rot", "nis", "sdiag_share", "pvar_share",
               "floor_share", "prior_pose_share", "n_res", "w_per_res", "cov_acc",
               "cov_gyr", "refit_dtraj_rms", "boundary_dpos", "trP_drop", "gain", "v_err"]
# DX-2 preflight fix: this was "refit_dtraj" (no "_rms" suffix), a name that
# never existed as a diag dict key (only "refit_dtraj_rms"/"refit_dtraj_max"
# do -- see load_cell()'s own assignment) -- d_refit_dtraj came back NaN on
# every pair for exactly this reason, not the corr_scan join issue P10 fixed.
assert len(DIAGNOSTICS) == 17, len(DIAGNOSTICS)


def _read_csv_dict(path):
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    cols = {}
    for k in rows[0]:
        try:
            cols[k] = np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
        except ValueError:
            pass
    return cols


def _resample_at(src_t, src_v, dst_t, max_gap=1.0):
    """Nearest-earlier-sample resample of a per-scan series onto dst_t
    (GT timestamps). NaN where the nearest sample is further than max_gap."""
    if src_t is None or len(src_t) == 0:
        return np.full(len(dst_t), np.nan)
    idx = np.clip(np.searchsorted(src_t, dst_t) - 1, 0, len(src_t) - 1)
    out = np.array(src_v)[idx].astype(float)
    gap = np.abs(src_t[idx] - dst_t)
    out[gap > max_gap] = np.nan
    return out


def load_cell(run_dir, gt_path, gt_format, frame_correction, window=10.0, max_gap=0.1):
    """One cell's full join: alignment, gain, and the per-scan diagnostics,
    all resampled onto the GT time grid analyse() itself uses.

    V-3 (2026-09-03): max_gap here feeds resample()'s bracket-gap tolerance
    (a direct `gap <= max_gap` check) for the SAME estimate-onto-GT
    interpolation the live C++ scorer does (EvoProc's opts_.max_time_diff,
    config key evo/max_time_diff, compiled default 0.05s) -- but the C++
    path (and evo/fastlivo_evo.py's own lookup_est(), which mirrors it)
    both gate on bracket_span <= 2.0*max_time_diff, i.e. the DOUBLED value,
    not the raw one. max_gap here must therefore be 2*0.05 = 0.1, not
    0.05 -- an off-by-factor-of-2 in an earlier pass at this same fix
    caused every cell to come back "too few matched samples" (odometry.txt's
    own ~83ms inter-frame spacing already exceeds a raw 0.05s tolerance).

    This was 0.5 before -- 5x looser than the correct 0.1 -- which silently
    matched ~3.5x more GT samples than the live scorer (n=6616 vs n=1868 on
    dx1r_l5_eee_01) and was the ENTIRE explanation for every disagreement
    V-3 found between results_lio.txt's ATE and this script's own `ate`
    column (verified via an independent third re-score,
    evo/fastlivo_evo.py's --incremental Kabsch path with --max-time-diff
    0.05: n and ATE landed bit-identical to results_lio.txt at every point
    along the progressive curve, not just the endpoint, on both a spline
    cell (L5, was 5.5% off) and a discrete no-spline cell (L0, was 11.5%
    off) once the tolerance matched). Fixed to 0.1 so this script's `ate`
    column is the same statistic as the live scorer reports -- the R1
    "determinism finding" (L1/L2/L3 missing their dx1 values by 16-21mm)
    was very likely this same convention mismatch, not a real
    non-determinism bug.
    """
    est_path = os.path.join(run_dir, "odometry.txt")
    if not os.path.exists(est_path):
        return None

    t_e, p_e, q_e = load_tum(est_path)
    t_g, p_g, q_g = LOADERS[gt_format](gt_path)
    p_e, q_e = apply_frame_correction(p_e, q_e, frame_correction)
    # V-3: linear (lerp), not the default pchip -- matches the live C++
    # scorer's own interp mode exactly (EvoProc: lerp(position)/slerp
    # (rotation), see evo_processing.cpp's "interp" comment). Quaternion
    # interpolation is always Slerp regardless of this argument, already
    # matching the live scorer's slerpRot -- only position differs.
    p_i, q_i, valid = resample(t_e, p_e, q_e, t_g, max_gap, method="linear")
    t = t_g[valid]
    src, dst = p_i[valid], p_g[valid]
    if len(t) < 20:
        return None

    Rm, tv = umeyama_no_scale(src, dst)
    aligned = (Rm @ src.T).T + tv
    e_glob = np.linalg.norm(aligned - dst, axis=1)
    e_pre = prefix_error(t, src, dst)
    e_win = window_error(t, src, dst, window)

    ate_rmse = float(np.sqrt(np.mean(e_glob ** 2)))

    # gain(t): needs P8's pose_pair.csv, resampled onto the SAME grid, then
    # put through the SAME (Rm, tv) the posterior trajectory was aligned
    # with -- gain is only meaningful inside one common frame.
    pp = _read_csv_dict(os.path.join(run_dir, "pose_pair.csv"))
    gain = np.full(len(t), np.nan)
    v_err = np.full(len(t), np.nan)
    if pp is not None and "t" in pp:
        prior_xyz = np.stack([_resample_at(pp["t"], pp["ppx"], t),
                              _resample_at(pp["t"], pp["ppy"], t),
                              _resample_at(pp["t"], pp["ppz"], t)], axis=1)
        prior_aligned = (Rm @ prior_xyz.T).T + tv
        e_prior = np.linalg.norm(prior_aligned - dst, axis=1)
        gain = e_prior - e_glob  # NOTE: uses the GLOBAL alignment's dst/e_glob
        # (posterior at the matched GT sample), matching e_glob's own frame --
        # this is intentionally the global-frame gain, not a prefix/window
        # variant, since gain answers "closer to truth", a single-frame
        # question, not a departure-timing one.
        vx = _resample_at(pp["t"], pp["vx"], t)
        vy = _resample_at(pp["t"], pp["vy"], t)
        vz = _resample_at(pp["t"], pp["vz"], t)
        v_post_aligned = (Rm[:, :3] @ np.stack([vx, vy, vz], axis=1).T).T  # rotation only, no translation for a velocity
        dgt = np.gradient(dst, t, axis=0)
        v_err = np.linalg.norm(v_post_aligned - dgt, axis=1)

    fs = _read_csv_dict(os.path.join(run_dir, "frame_stats.txt"))
    cs = _read_csv_dict(os.path.join(run_dir, "corr_scan.csv"))
    sc = _read_csv_dict(os.path.join(run_dir, "scan.csv"))
    sq = _read_csv_dict(os.path.join(run_dir, "spline_q.csv"))
    # _read_csv_dict silently drops any column that doesn't parse as float
    # (its own ValueError guard) -- spline_q.csv's "status" column is a
    # string enum (ok/below_floor/warmup/no_window/not_white), so it needs
    # its own pass to survive into sq for the active/clamped derivation
    # below (C-7).
    sq_status_path = os.path.join(run_dir, "spline_q.csv")
    if sq is not None and os.path.exists(sq_status_path):
        with open(sq_status_path) as _f:
            _rows = list(csv.DictReader(_f))
        if _rows and "status" in _rows[0]:
            sq["status"] = np.array([r["status"] for r in _rows], dtype=object)

    diag = {}
    if fs is not None and "t" in fs:
        diag["refusal"] = _resample_at(fs["t"], fs["refusal"], t)
        diag["ask"] = _resample_at(fs["t"], fs["ask"], t)
        diag["got"] = _resample_at(fs["t"], fs["got"], t)
        diag["dx_pos_mm"] = _resample_at(fs["t"], fs["dx_pos_mm"], t)
        diag["htz_pos_norm"] = _resample_at(fs["t"], fs["htz_pos_norm"], t)
        diag["iters"] = _resample_at(fs["t"], fs["iters"], t)
        trP_pre = _resample_at(fs["t"], fs["trP_pos_pre"], t)
        # DX-2 preflight fix: trP_pos_drop_p50 was landing at exactly 0 on
        # every DX-1R cell -- not an EKF/estimator finding, a join bug.
        # GT is denser than frame_stats.txt's own scan rate (~20Hz vs
        # ~10-12Hz), so multiple adjacent GT-grid samples resample to the
        # SAME underlying scan; computing pre-post on the RESAMPLED
        # (duplicated) series then diffs a value against itself over half
        # the time, producing bit-identical zeros. Fixed by computing the
        # pre->post delta on frame_stats.txt's own native per-scan grid
        # FIRST, then resampling the already-computed delta series (safe
        # to duplicate across nearby GT samples -- it's now a well-defined
        # per-scan quantity, not a raw level that can self-cancel).
        fs_trP_drop_native = fs["trP_pos_pre"][:-1] - fs["trP_pos_pre"][1:]
        fs_trP_drop_native = np.append(fs_trP_drop_native, np.nan)  # last scan has no "next"
        diag["trP_drop"] = _resample_at(fs["t"], fs_trP_drop_native, t)
        # DX-2 preflight fix: boundary_dpos is now a real per-scan column
        # (frame_stats.txt, -1 sentinel = unavailable -- first spline scan
        # or spline disabled) instead of a hardcoded NaN placeholder.
        if "boundary_dpos" in fs:
            bd = np.where(fs["boundary_dpos"] >= 0, fs["boundary_dpos"], np.nan)
            diag["boundary_dpos"] = _resample_at(fs["t"], bd, t)
        n_res = _resample_at(fs["t"], fs["n_residuals"], t)
        n_planes = _resample_at(fs["t"], fs["n_planes"], t)
        h_pp = _resample_at(fs["t"], fs["h_pp_min_eig"], t)
        h_rr = _resample_at(fs["t"], fs["h_rr_min_eig"], t)
        h_rr_trace = _resample_at(fs["t"], fs["h_rr_trace"], t)
        sum_weight = _resample_at(fs["t"], fs["sum_weight"], t)
        diag["n_res"] = n_res
        diag["n_planes"] = n_planes
        diag["div_pos"] = np.where(sum_weight > 0, 3.0 * h_pp / sum_weight, np.nan)
        diag["div_rot"] = np.where(h_rr_trace > 0, 3.0 * h_rr / h_rr_trace, np.nan)
        diag["w_per_res"] = np.where(n_res > 0, sum_weight / n_res, np.nan)

    if cs is not None and "scan_id" in cs:
        # corr_scan.csv has no t column of its own (only scan_id) -- join it
        # onto frame_stats.txt's t via an actual frame_idx/scan_id VALUE
        # lookup, not a row-count coincidence. The two files are NOT
        # guaranteed 1:1 in length (confirmed on the DX-1R rerun: some
        # scans get logged to frame_stats.txt but never reach
        # flushCorrScan(), e.g. a scan with no accepted correspondences) --
        # the old length-match fallback silently produced NaN for every
        # corr_scan-derived diagnostic whenever that happened, which is
        # exactly what the t-column precision-loss bug was masking (every
        # length compared equal-but-wrong before that fix, this compares
        # equal-but-still-wrong after it). Missing scan_ids resolve to NaN
        # here, not the whole diagnostic.
        cs_t = None
        if fs is not None and "t" in fs and "frame_idx" in fs:
            idx_to_t = dict(zip(fs["frame_idx"], fs["t"]))
            cs_t_raw = np.array([idx_to_t.get(sid, np.nan) for sid in cs["scan_id"]])
            # _resample_at's searchsorted needs a strictly-sorted src_t --
            # drop any scan_id that didn't resolve (NaN) rather than let it
            # break the ordering assumption.
            keep = ~np.isnan(cs_t_raw)
            cs_t = cs_t_raw[keep]
            cs = {k: v[keep] for k, v in cs.items()}
        if cs_t is not None and len(cs_t) > 0:
            sum_S = cs["sum_S"]
            diag["sdiag_share"] = _resample_at(cs_t, np.where(sum_S > 0, cs["sum_sdiag"] / sum_S, np.nan), t)
            diag["pvar_share"] = _resample_at(cs_t, np.where(sum_S > 0, cs["sum_pvar"] / sum_S, np.nan), t)
            diag["floor_share"] = _resample_at(cs_t, np.where(sum_S > 0, cs["sum_floor"] / sum_S, np.nan), t)
            diag["prior_pose_share"] = _resample_at(cs_t, np.where(sum_S > 0, cs["sum_prior_pose"] / sum_S, np.nan), t)
            diag["nis"] = _resample_at(cs_t, np.where(cs["n_nis_finite"] > 0, cs["sum_nis"] / cs["n_nis_finite"], np.nan), t)
            if "n_vis_hit" in cs:
                vtot = cs["n_vis_hit"] + cs["n_vis_free"] + cs["n_vis_unobs"]
                diag["vis_free"] = _resample_at(cs_t, np.where(vtot > 0, cs["n_vis_free"] / vtot, np.nan), t)
                diag["vis_unobs"] = _resample_at(cs_t, np.where(vtot > 0, cs["n_vis_unobs"] / vtot, np.nan), t)
            # P10 (block v2).  n_vis_hit/free/unobs above count every
            # candidate whose ray was classified, accepted or not -- can't
            # separate "the arm's known-free difference is a fact about the
            # scene" from "...a fact about how long a chart lives" without
            # the accepted-only counts, chart age, and chart fill P10 adds.
            if "n_vis_hit_thru" in cs:
                diag["vis_hit_thru"] = _resample_at(
                    cs_t, np.where(cs["n_vis_hit"] > 0, cs["n_vis_hit_thru"] / cs["n_vis_hit"], np.nan), t)
            else:
                diag["vis_hit_thru"] = np.full(len(t), np.nan)  # pre-P10 corr_scan.csv -- column doesn't exist
            if "n_vis_free_acc" in cs:
                vtot = cs["n_vis_hit"] + cs["n_vis_free"] + cs["n_vis_unobs"]
                vtot_acc = cs["n_vis_hit_acc"] + cs["n_vis_free_acc"] + cs["n_vis_unobs_acc"]
                diag["vis_free_acc_frac"] = _resample_at(
                    cs_t, np.where(vtot > 0, cs["n_vis_free_acc"] / vtot, np.nan), t)
                diag["vis_free_of_acc"] = _resample_at(
                    cs_t, np.where(vtot_acc > 0, cs["n_vis_free_acc"] / vtot_acc, np.nan), t)
                diag["age_free"] = _resample_at(
                    cs_t, np.where(cs["n_age_free"] > 0, cs["sum_age_free"] / cs["n_age_free"], np.nan), t)
                diag["age_hit"] = _resample_at(
                    cs_t, np.where(cs["n_age_hit"] > 0, cs["sum_age_hit"] / cs["n_age_hit"], np.nan), t)
                diag["fill_free"] = _resample_at(
                    cs_t, np.where(cs["n_vis_free"] > 0, cs["sum_fill_free"] / cs["n_vis_free"], np.nan), t)
                diag["fill_hit"] = _resample_at(
                    cs_t, np.where(cs["n_vis_hit"] > 0, cs["sum_fill_hit"] / cs["n_vis_hit"], np.nan), t)

    if sc is not None and "t" in sc:
        diag["omega_norm"] = _resample_at(sc["t"], sc["omega_norm"], t)
        diag["acc_norm"] = _resample_at(sc["t"], sc["acc_norm"], t)

    if sq is not None and "t_abs" in sq:
        diag["cov_acc_pre"] = _resample_at(sq["t_abs"], sq["cov_acc_pre"], t)
        diag["cov_acc_post"] = _resample_at(sq["t_abs"], sq["cov_acc_meas"], t)
        diag["cov_gyr_pre"] = _resample_at(sq["t_abs"], sq["cov_gyr_pre"], t)
        diag["cov_gyr_post"] = _resample_at(sq["t_abs"], sq["cov_gyr_meas"], t)
        diag["refit_dtraj_rms"] = _resample_at(sq["t_abs"], sq["refit_dtraj_rms"], t)
        diag["refit_dtraj_max"] = _resample_at(sq["t_abs"], sq["refit_dtraj_max"], t)
        # DX-5 (commit 43b598a): the fitted bias/gravity delta magnitude,
        # never logged before this round -- 0 on every pre-43b598a spline_q.csv
        # (column absent) and on any cell with imu_fit_mode=off (biasAccDelta()
        # etc. return Zero() when the joint solve never ran).
        if "d_bias_acc_norm" in sq:
            diag["d_bias_acc_norm"] = _resample_at(sq["t_abs"], sq["d_bias_acc_norm"], t)
            diag["d_bias_gyr_norm"] = _resample_at(sq["t_abs"], sq["d_bias_gyr_norm"], t)
            diag["d_gravity_norm"] = _resample_at(sq["t_abs"], sq["d_gravity_norm"], t)
        # C-7 (second half): the AdaptiveQ GATE columns -- distinct from
        # cov_*_pre/post above, which csvRow() writes from the raw residual
        # whether or not update() ran (see Q-1b) and therefore cannot tell
        # "flat because clamped" from "flat because nothing drives it".
        # z_acc/z_gyr are the module's own whitened-innovation gate signal;
        # acf1_acc/acf1_gyr its residual-autocorrelation whiteness check;
        # status ("ok"/"below_floor"/"warmup"/"no_window"/"not_white")
        # is the module's own per-frame verdict -- active means the EMA
        # update actually ran and moved the applied value ("ok" only);
        # clamped means it ran but the floor pinned the output
        # ("below_floor") rather than nothing driving it at all (the other
        # three statuses -- warmup/no_window/not_white -- are neither).
        if "z_acc" in sq:
            diag["z_acc"] = _resample_at(sq["t_abs"], sq["z_acc"], t)
            diag["z_gyr"] = _resample_at(sq["t_abs"], sq["z_gyr"], t)
            diag["acf1_acc"] = _resample_at(sq["t_abs"], sq["acf1_acc"], t)
            diag["acf1_gyr"] = _resample_at(sq["t_abs"], sq["acf1_gyr"], t)
        if "status" in sq:
            status = np.asarray(sq["status"], dtype=object)
            diag["active"] = _resample_at(sq["t_abs"], (status == "ok").astype(float), t)
            diag["clamped"] = _resample_at(sq["t_abs"], (status == "below_floor").astype(float), t)

    diag["cov_acc"] = diag.get("cov_acc_post", np.full(len(t), np.nan))
    diag["cov_gyr"] = diag.get("cov_gyr_post", np.full(len(t), np.nan))
    diag.setdefault("boundary_dpos", np.full(len(t), np.nan))  # pre-DX-2 frame_stats.txt (no column)
    diag["gain"] = gain
    diag["v_err"] = v_err

    return dict(t=t, e_glob=e_glob, e_pre=e_pre, e_win=e_win, gain=gain, v_err=v_err,
                ate_rmse=ate_rmse, diag=diag)


def series_block(cell_data):
    """Block 1: 40 equal-span medians per cell, ladder cells only."""
    rows = []
    for cell, d in cell_data.items():
        if cell not in LADDER_CELLS or d is None:
            continue
        t = d["t"]
        t0, t1 = t[0], t[-1]
        edges = np.linspace(t0, t1, 41)
        gain_cum = np.nancumsum(np.nan_to_num(d["gain"], nan=0.0))
        for k in range(40):
            lo, hi = edges[k], edges[k + 1]
            m = (t >= lo) & (t < hi if k < 39 else t <= hi)
            if not m.any():
                continue

            def med(arr):
                v = arr[m]
                v = v[np.isfinite(v)]
                return float(np.median(v)) if len(v) else float("nan")

            def rms(arr):
                # P-1 item 0 / P-2 item 1: a bucket median of e_glob discards
                # the within-bucket spikes ATE is actually built from -- an
                # RMS over these bucket RMS values reproduces harness ATE to
                # 0.4-4.1% on dx1r's twelve cells (verified P-2, nineteenth
                # round), where the median route ran 0.65-0.90x low on the
                # well-behaved cells. Emitted alongside the median, not in
                # place of it, per rule 5b.
                v = arr[m]
                v = v[np.isfinite(v)]
                return float(np.sqrt(np.mean(v ** 2))) if len(v) else float("nan")

            def frac(arr):
                # C-7: active/clamped are 0/1 per-scan indicators -- a
                # bucket's MEDIAN of a binary series just picks the
                # majority state (or ties), losing exactly the "how much
                # of this bucket" question the fraction is for. Mean, not
                # med(), is the right reduction here.
                v = arr[m]
                v = v[np.isfinite(v)]
                return float(np.mean(v)) if len(v) else float("nan")

            rows.append(dict(
                cell=cell.upper(), t_s=float(t[m][len(t[m]) // 2] - t0),
                e_glob=med(d["e_glob"]), e_glob_rms=rms(d["e_glob"]),
                e_pre=med(d["e_pre"]), e_win=med(d["e_win"]),
                gain_cum=float(gain_cum[m][-1]) if m.any() else float("nan"),
                v_err=med(d["v_err"]), refusal=med(d["diag"].get("refusal", np.full(len(t), np.nan))),
                div_pos=med(d["diag"].get("div_pos", np.full(len(t), np.nan))),
                nis=med(d["diag"].get("nis", np.full(len(t), np.nan))),
                sdiag_share=med(d["diag"].get("sdiag_share", np.full(len(t), np.nan))),
                pvar_share=med(d["diag"].get("pvar_share", np.full(len(t), np.nan))),
                floor_share=med(d["diag"].get("floor_share", np.full(len(t), np.nan))),
                n_res=med(d["diag"].get("n_res", np.full(len(t), np.nan))),
                # P10 (block v2)
                vis_free=med(d["diag"].get("vis_free", np.full(len(t), np.nan))),
                vis_free_acc_frac=med(d["diag"].get("vis_free_acc_frac", np.full(len(t), np.nan))),
                age_free=med(d["diag"].get("age_free", np.full(len(t), np.nan))),
                age_hit=med(d["diag"].get("age_hit", np.full(len(t), np.nan))),
                # C-7: AdaptiveQ gate columns -- see load_cell()'s own
                # comment for why these, not cov_*_pre/post, are what
                # separates "flat because clamped" from "flat because
                # nothing drives it".
                z_acc=med(d["diag"].get("z_acc", np.full(len(t), np.nan))),
                z_gyr=med(d["diag"].get("z_gyr", np.full(len(t), np.nan))),
                acf1_acc=med(d["diag"].get("acf1_acc", np.full(len(t), np.nan))),
                acf1_gyr=med(d["diag"].get("acf1_gyr", np.full(len(t), np.nan))),
                active_frac=frac(d["diag"].get("active", np.full(len(t), np.nan))),
                clamped_frac=frac(d["diag"].get("clamped", np.full(len(t), np.nan))),
            ))
    return rows


def summary_block(cell_data):
    """Block 2: one row per cell (S3 excluded -- caller's cell_data shouldn't
    include it, or it's silently identical to L3 and skipped here too)."""
    rows = []
    for cell, d in cell_data.items():
        if cell == "s3" or d is None:
            continue
        diag = d["diag"]

        def p50(k):
            v = diag.get(k)
            if v is None:
                return float("nan")
            v = v[np.isfinite(v)]
            return float(np.median(v)) if len(v) else float("nan")

        def p90(k):
            v = diag.get(k)
            if v is None:
                return float("nan")
            v = v[np.isfinite(v)]
            return float(np.percentile(v, 90)) if len(v) else float("nan")

        def frac_nan(k):
            v = diag.get(k)
            if v is None or len(v) == 0:
                return float("nan")
            return float(np.mean(~np.isfinite(v)))

        gain = d["gain"]
        gain_fin = gain[np.isfinite(gain)]
        rows.append(dict(
            cell=cell.upper(), ate=d["ate_rmse"],
            e_pre_p50=p50_arr(d["e_pre"]), e_pre_p90=p90_arr(d["e_pre"]),
            e_win_p50=p50_arr(d["e_win"]),
            gain_sum=float(np.sum(gain_fin)) if len(gain_fin) else float("nan"),
            gain_neg_frac=float(np.mean(gain_fin < 0)) if len(gain_fin) else float("nan"),
            v_err_p50=p50_arr(d["v_err"]),
            refusal_p50=p50("refusal"), refusal_nan_frac=frac_nan("refusal"),
            div_pos_p50=p50("div_pos"), div_rot_p50=p50("div_rot"), nis_p50=p50("nis"),
            sdiag_share_p50=p50("sdiag_share"), pvar_share_p50=p50("pvar_share"),
            floor_share_p50=p50("floor_share"), prior_pose_share_p50=p50("prior_pose_share"),
            n_res_p50=p50("n_res"), n_planes_p50=p50("n_planes"), w_per_res_p50=p50("w_per_res"),
            cov_acc_pre_p50=p50("cov_acc_pre"), cov_acc_post_p50=p50("cov_acc_post"),
            cov_gyr_pre_p50=p50("cov_gyr_pre"), cov_gyr_post_p50=p50("cov_gyr_post"),
            refit_dtraj_rms_p50=p50("refit_dtraj_rms"), refit_dtraj_max=(
                float(np.nanmax(diag["refit_dtraj_max"])) if "refit_dtraj_max" in diag and np.any(np.isfinite(diag["refit_dtraj_max"])) else float("nan")),
            boundary_dpos_p50=p50("boundary_dpos"), boundary_dpos_p90=p90("boundary_dpos"),
            vis_free_frac=p50("vis_free"), vis_unobs_frac=p50("vis_unobs"),
            vis_hit_thru_frac=p50("vis_hit_thru"),
            iters_p50=p50("iters"), trP_pos_drop_p50=p50("trP_drop"),
            # P10 (block v2): accepted-only known-free rate ("the number
            # that actually decides P6b"), plus chart age/fill so a known-
            # free difference between arms can be told apart from a chart-
            # persistence difference.
            vis_free_acc_frac=p50("vis_free_acc_frac"), vis_free_of_acc=p50("vis_free_of_acc"),
            age_free_p50=p50("age_free"), age_hit_p50=p50("age_hit"),
            fill_free_p50=p50("fill_free"), fill_hit_p50=p50("fill_hit"),
            # DX-5: p50/p90 of the fitted bias/gravity delta magnitude
            # (commit 43b598a) -- nan on any cell whose spline_q.csv predates
            # that commit (column absent, diag key never set).
            d_bias_acc_norm_p50=p50("d_bias_acc_norm"), d_bias_acc_norm_p90=p90("d_bias_acc_norm"),
            d_bias_gyr_norm_p50=p50("d_bias_gyr_norm"), d_bias_gyr_norm_p90=p90("d_bias_gyr_norm"),
            d_gravity_norm_p50=p50("d_gravity_norm"), d_gravity_norm_p90=p90("d_gravity_norm"),
        ))
    return rows


def p50_arr(v):
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else float("nan")


def p90_arr(v):
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 90)) if len(v) else float("nan")


def _cluster_se(x, y, t, block_s=10.0):
    """Pearson r with cluster-robust SE by block_s-second blocks."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return float("nan"), float("nan"), int(m.sum())
    x, y, tt = x[m], y[m], t[m]
    r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    blocks = np.floor((tt - tt[0]) / block_s).astype(int)
    ubl = np.unique(blocks)
    if len(ubl) < 2:
        return r, float("nan"), int(m.sum())
    xs, ys = (x - x.mean()) / (x.std() + 1e-300), (y - y.mean()) / (y.std() + 1e-300)
    prod = xs * ys
    block_means = np.array([prod[blocks == b].mean() for b in ubl])
    se = float(np.std(block_means, ddof=1) / np.sqrt(len(ubl)))
    return r, se, int(m.sum())


def _best_lag(x, y, max_lag=20):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        # Register rule: never use 0 as a stand-in for "could not compute" --
        # 0 is also a legitimate answer (no lag), so the two must not be
        # conflated. float("nan") for both signals "not computable" instead.
        return float("nan"), float("nan")
    x, y = x[m], y[m]
    best_lag, best_r = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            xs, ys = x[-lag:], y[:lag]
        elif lag > 0:
            xs, ys = x[:-lag], y[lag:]
        else:
            xs, ys = x, y
        if len(xs) < 10 or np.std(xs) == 0 or np.std(ys) == 0:
            continue
        r = float(np.corrcoef(xs, ys)[0, 1])
        if np.isfinite(r) and r > best_r:
            best_r, best_lag = r, lag
    return best_lag, (best_r if best_r > -2.0 else float("nan"))


def pairs_block(cell_data):
    """Block 3: pairwise attribution. d(t) = the per-scan INCREMENT of
    delta_e(t) = e_pre^B(t) - e_pre^A(t), correlated against each
    diagnostic's own per-scan delta (B-A)."""
    rows = []
    for a_id, b_id in PAIRS:
        A, B = cell_data.get(a_id), cell_data.get(b_id)
        if A is None or B is None:
            continue
        tA, tB = A["t"], B["t"]
        t, ia, ib = np.intersect1d(tA, tB, return_indices=True)
        if len(t) < 10:
            continue
        delta_e = B["e_pre"][ib] - A["e_pre"][ia]
        d_t = np.diff(delta_e, prepend=delta_e[0])

        def dget(cell, key, idx):
            v = cell["diag"].get(key)
            if v is None:
                return np.full(len(idx), np.nan)
            return v[idx]

        for diag in DIAGNOSTICS:
            if diag == "gain":
                xA, xB = dget(A, "gain", ia) if False else A["gain"][ia], B["gain"][ib]
            elif diag == "v_err":
                xA, xB = A["v_err"][ia], B["v_err"][ib]
            else:
                xA, xB = dget(A, diag, ia), dget(B, diag, ib)
            dx = xB - xA
            r, se, n = _cluster_se(d_t, dx, t)
            lag, r_lag = _best_lag(d_t, dx)
            rows.append(dict(pair=f"{a_id.upper()}->{b_id.upper()}", diagnostic=f"d_{diag}",
                             r=r, se_cluster=se, n_scans=n, best_lag_scans=lag, r_at_best_lag=r_lag))
    return rows


def _fnum(v, dp):
    """Register rule: a value that could not be computed is the literal
    token 'nan' -- never blank, never 0, never -1, never 'N/A'."""
    return "nan" if not np.isfinite(v) else f"{v:.{dp}f}"


def _fint(v):
    return "nan" if not np.isfinite(v) else str(int(round(v)))


def emit_series(rows):
    # P10: bumped to v2 -- vis_free/vis_free_acc_frac/age_free/age_hit
    # appended, existing columns unchanged/unreordered.
    # P-2 (nineteenth round): bumped to v3 -- e_glob_rms appended.  A bucket
    # median discards the within-bucket spikes ATE is built from; e_glob_rms
    # is the statistic that reconciles against harness ATE (P-1 item 0).
    # C-7 (nineteenth round): bumped to v4 -- z_acc/z_gyr/acf1_acc/acf1_gyr/
    # active_frac/clamped_frac appended -- the AdaptiveQ GATE columns, not
    # yet present anywhere (cov_acc/cov_gyr landed already but cannot
    # distinguish a live module from a disabled one -- see Q-1b).
    out = ["[DX1R-SERIES v4]",
           "cell,t_s,e_glob,e_glob_rms,e_pre,e_win,gain_cum,v_err,refusal,div_pos,nis,sdiag_share,pvar_share,floor_share,n_res,"
           "vis_free,vis_free_acc_frac,age_free,age_hit,z_acc,z_gyr,acf1_acc,acf1_gyr,active_frac,clamped_frac"]
    for r in rows:
        out.append(f"{r['cell']},{r['t_s']:.3f},{_fnum(r['e_glob'],5)},{_fnum(r['e_glob_rms'],5)},{_fnum(r['e_pre'],5)},{_fnum(r['e_win'],5)},"
                   f"{_fnum(r['gain_cum'],5)},{_fnum(r['v_err'],4)},{_fnum(r['refusal'],3)},{_fnum(r['div_pos'],3)},"
                   f"{_fnum(r['nis'],3)},{_fnum(r['sdiag_share'],3)},{_fnum(r['pvar_share'],3)},{_fnum(r['floor_share'],3)},"
                   f"{_fint(r['n_res'])},"
                   f"{_fnum(r['vis_free'],4)},{_fnum(r['vis_free_acc_frac'],4)},{_fnum(r['age_free'],2)},{_fnum(r['age_hit'],2)},"
                   f"{_fnum(r['z_acc'],3)},{_fnum(r['z_gyr'],3)},{_fnum(r['acf1_acc'],3)},{_fnum(r['acf1_gyr'],3)},"
                   f"{_fnum(r['active_frac'],3)},{_fnum(r['clamped_frac'],3)}")
    out.append("[/DX1R-SERIES]")
    return "\n".join(out)


def emit_summary(rows):
    # P10: bumped to v2 -- vis_free_acc_frac/vis_free_of_acc/age_{free,hit}_p50/
    # fill_{free,hit}_p50 appended, existing columns unchanged/unreordered.
    # DX-5: bumped to v3 -- d_bias_acc_norm/d_bias_gyr_norm/d_gravity_norm
    # p50/p90 appended (commit 43b598a), same "append at the end" convention.
    cols = ["cell", "ate", "e_pre_p50", "e_pre_p90", "e_win_p50", "gain_sum", "gain_neg_frac",
            "v_err_p50", "refusal_p50", "refusal_nan_frac", "div_pos_p50", "div_rot_p50", "nis_p50",
            "sdiag_share_p50", "pvar_share_p50", "floor_share_p50", "prior_pose_share_p50",
            "n_res_p50", "n_planes_p50", "w_per_res_p50", "cov_acc_pre_p50", "cov_acc_post_p50",
            "cov_gyr_pre_p50", "cov_gyr_post_p50", "refit_dtraj_rms_p50", "refit_dtraj_max",
            "boundary_dpos_p50", "boundary_dpos_p90", "vis_free_frac", "vis_unobs_frac",
            "vis_hit_thru_frac", "iters_p50", "trP_pos_drop_p50",
            "vis_free_acc_frac", "vis_free_of_acc", "age_free_p50", "age_hit_p50",
            "fill_free_p50", "fill_hit_p50",
            "d_bias_acc_norm_p50", "d_bias_acc_norm_p90", "d_bias_gyr_norm_p50",
            "d_bias_gyr_norm_p90", "d_gravity_norm_p50", "d_gravity_norm_p90"]
    out = ["[DX1R-SUMMARY v3]", ",".join(cols)]
    for r in rows:
        def fmt(c):
            v = r[c]
            if isinstance(v, float):
                return "nan" if not np.isfinite(v) else f"{v:.5g}"
            return str(v)
        vals = [r["cell"]] + [fmt(c) for c in cols[1:]]
        out.append(",".join(vals))
    out.append("[/DX1R-SUMMARY]")
    return "\n".join(out)


def emit_pairs(rows):
    out = ["[DX1R-PAIRS v1]", "pair,diagnostic,r,se_cluster,n_scans,best_lag_scans,r_at_best_lag"]
    for r in rows:
        out.append(f"{r['pair']},{r['diagnostic']},{_fnum(r['r'],2)},{_fnum(r['se_cluster'],2)},"
                   f"{r['n_scans']},{_fint(r['best_lag_scans'])},{_fnum(r['r_at_best_lag'],2)}")
    out.append("[/DX1R-PAIRS]")
    return "\n".join(out)


def _results_lio_ate(run_dir):
    """Parse ATE=<x>m out of results_lio.txt's evo console line (ANSI codes
    and all -- it's a raw redirected console log, not a clean data file)."""
    import re
    path = os.path.join(run_dir, "results_lio.txt")
    if not os.path.exists(path):
        return float("nan")
    with open(path) as f:
        text = f.read()
    m = re.findall(r"ATE=([0-9.]+)m", text)
    return float(m[-1]) if m else float("nan")


def _script_commit():
    import subprocess
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_HERE,
                                       stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _adaptive_q_echo(run_dir):
    """C-10: no artifact in this project recorded which cells had AdaptiveQ
    on -- its shipped default is false and eighteen rounds argued about
    cov_acc/cov_gyr without it. Reads the retained config.yaml directly (the
    only place the resolved value lives) rather than inferring it from
    engagement.txt (which is absent whenever spline is off, per its own
    nested-under-spline design -- see Q-1's card)."""
    path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(path):
        return "enable=nan"
    try:
        import yaml
        cfg = yaml.safe_load(open(path)) or {}
    except Exception:
        return "enable=nan(unparsable)"
    aq = cfg.get("adaptive_q", {}) or {}
    enable = aq.get("enable", False)  # shipped default
    nf = aq.get("noise_floor", {}) or {}
    return (f"enable={str(enable).lower()}"
            f",noise_floor_mode={nf.get('mode', 'allan')}"
            f",beta_acc={aq.get('beta_acc', 'default')}"
            f",beta_gyr={aq.get('beta_gyr', 'default')}"
            f",warmup_frames={aq.get('warmup_frames', 'default')}")


def meta_block(cell_data, batch, seq, build_commit, batch_root, dispatched_cells, run_dirs=None):
    import datetime
    completed = [c for c in dispatched_cells if cell_data.get(c) is not None]
    failed = [c for c in dispatched_cells if cell_data.get(c) is None]
    series = series_block(cell_data)
    short = []
    for cell in LADDER_CELLS:
        n = sum(1 for r in series if r["cell"] == cell.upper())
        if cell_data.get(cell) is not None and n < 40:
            short.append(f"{cell}(series={n}/40)")
    # C-10: per-cell AdaptiveQ scope echo. run_dirs lets a caller (e.g.
    # emit_batch_blocks.py) pass explicit NAME->run_dir pairs; falls back to
    # dx_report.py's own batch-root/batch_cell_seq convention otherwise.
    aq_lines = []
    for cell in dispatched_cells:
        if cell_data.get(cell) is None:
            continue
        if run_dirs and cell in run_dirs:
            rd = run_dirs[cell]
        elif batch_root:
            rd = os.path.join(batch_root, f"{batch}_{cell}_{seq}")
        else:
            continue
        aq_lines.append(f"adaptive_q[{cell}]: {_adaptive_q_echo(rd)}")
    out = ["[DX1R-META v2]",
           f"batch={batch}", f"seq={seq}",
           f"commit={build_commit}", f"script_commit={_script_commit()}",
           f"date_utc={datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
           f"cells_dispatched={len(dispatched_cells)}  cells_completed={len(completed)}  cells_failed={len(failed)}",
           f"failed={','.join(failed) if failed else 'NONE'}",
           f"short_blocks={','.join(short) if short else 'NONE'}",
           *aq_lines,
           "[/DX1R-META]"]
    return "\n".join(out)


def verify_block(cell_data, series, dispatched_cells):
    """The four checks, PER CELL, AS DATA -- see the register's DX-1R-D card,
    step 1's FILE LAYOUT for [DX1R-VERIFY v1]'s exact semantics:
      (1) e_glob's RMS over all scans must equal results_lio.txt's ATE to 4dp.
      (2) gain_sum_m -- NOT a verified identity (C-11 fix, nineteenth round).
          The original spec here claimed a telescoping identity
          gain_cum[-1] + e_glob[0] =~= e_prior[-1] with e_prior := gain +
          e_glob -- but gain[k] is a PER-SCAN counterfactual (this scan's
          own prior error minus its own posterior error, both against GT at
          the SAME instant), not a running trajectory quantity, so it does
          not telescope: gain_cum[-1] sums ~4,000 per-scan terms (order
          10^2-10^3) while e_prior[-1] is a single scan's value (order
          10^-2), so the "error" was ALWAYS approximately gain_cum[-1]
          itself and could not fail regardless of whether the underlying
          data was right. Reported honestly now: the raw sum, not an "error"
          that implies a check passed.
      (3) on a trajectory_mode: discrete cell, refit_dtraj_rms is 0 (there is
          no spline to refit at all -- see DISCRETE_CELLS).
      (4) the row count this cell contributed to DX1R-SERIES (0 for any cell
          not in LADDER_CELLS -- SERIES is ladder-only by construction).
    """
    rows = []
    for cell in dispatched_cells:
        d = cell_data.get(cell)
        if d is None:
            rows.append(dict(cell=cell.upper(), e_glob_rms=float("nan"), ate_results_lio=float("nan"),
                              gain_sum_m=float("nan"), refit_dtraj_zero_ok="nan", n_series_rows=0))
            continue
        e_glob = d["e_glob"]
        e_glob_rms = float(np.sqrt(np.mean(e_glob[np.isfinite(e_glob)] ** 2))) if np.any(np.isfinite(e_glob)) else float("nan")

        gain = d["gain"]
        gain_cum = np.nancumsum(np.nan_to_num(gain, nan=0.0))
        gain_sum_m = float(gain_cum[-1]) if len(gain_cum) else float("nan")

        if cell in DISCRETE_CELLS:
            rd = d["diag"].get("refit_dtraj_rms")
            if rd is None or not np.any(np.isfinite(rd)):
                zero_ok = "1"  # no spline data at all on a discrete cell -- trivially satisfied
            else:
                zero_ok = "1" if np.nanmax(np.abs(rd)) < 1e-9 else "0"
        else:
            zero_ok = "nan"  # check does not apply outside DISCRETE_CELLS

        n_series_rows = sum(1 for r in series if r["cell"] == cell.upper())

        rows.append(dict(
            cell=cell.upper(),
            e_glob_rms=e_glob_rms,
            ate_results_lio=float("nan"),  # filled by caller (needs run_dir, not in cell_data)
            gain_sum_m=gain_sum_m,
            refit_dtraj_zero_ok=zero_ok,
            n_series_rows=n_series_rows,
        ))
    return rows


def emit_verify(rows):
    # C-11: bumped to v2 -- gain_telescope_err_m renamed gain_sum_m (it was
    # never a verified identity, see verify_block()'s doc comment).
    out = ["[DX1R-VERIFY v2]",
           "cell,e_glob_rms,ate_results_lio,gain_sum_m,refit_dtraj_zero_ok,n_series_rows"]
    for r in rows:
        out.append(f"{r['cell']},{_fnum(r['e_glob_rms'],4)},{_fnum(r['ate_results_lio'],4)},"
                   f"{_fnum(r['gain_sum_m'],4)},{r['refit_dtraj_zero_ok']},{r['n_series_rows']}")
    out.append("[/DX1R-VERIFY]")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-root", required=True, help="dir containing dx1r_<cell>_eee_01/ subdirs")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--gt-format", default="ntu_csv", choices=list(LOADERS))
    ap.add_argument("--frame-correction", default="ntu_prism",
                    choices=["none", "ntu_prism", "hilti_tip"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="eee_01")
    ap.add_argument("--batch-id", default="dx1r",
                    help="prefix for run dirs (dx1r_<cell>_<seq>) and DX1R-META's batch= field")
    ap.add_argument("--emit-blocks", default=None,
                    help="write all five blocks (META, VERIFY, SERIES, SUMMARY, PAIRS), "
                         "in that order, nothing else, to this single path -- the register's "
                         "docs/results/<BATCH-ID>.txt convention")
    ap.add_argument("--build-commit", default=None,
                    help="git rev-parse HEAD of the livo_recon build the cells actually RAN "
                         "on (not detectable from data -- must be supplied). Required with "
                         "--emit-blocks.")
    a = ap.parse_args()

    if a.emit_blocks and not a.build_commit:
        sys.exit("--emit-blocks requires --build-commit (the commit the cells ran on)")

    os.makedirs(a.out, exist_ok=True)
    cell_data = {}
    for cell in ALL_CELLS:
        run_dir = os.path.join(a.batch_root, f"{a.batch_id}_{cell}_{a.seq}")
        d = load_cell(run_dir, a.gt, a.gt_format, a.frame_correction)
        cell_data[cell] = d
        if d is None:
            print(f"WARNING: {cell} not loaded (missing odometry.txt or too few matched samples)",
                  file=sys.stderr)

    series = series_block(cell_data)
    summary = summary_block(cell_data)
    pairs = pairs_block(cell_data)

    with open(os.path.join(a.out, "dx1r_series.txt"), "w") as f:
        f.write(emit_series(series) + "\n")
    with open(os.path.join(a.out, "dx1r_summary.txt"), "w") as f:
        f.write(emit_summary(summary) + "\n")
    with open(os.path.join(a.out, "dx1r_pairs.txt"), "w") as f:
        f.write(emit_pairs(pairs) + "\n")

    print(emit_series(series))
    print()
    print(emit_summary(summary))
    print()
    print(emit_pairs(pairs))

    print(f"\n# row counts: series={len(series)} (want {40*len(LADDER_CELLS)}) "
          f"summary={len(summary)} (want {len(ALL_CELLS)}) "
          f"pairs={len(pairs)} (want 17*{len(PAIRS)}={17*len(PAIRS)})",
          file=sys.stderr)

    if a.emit_blocks:
        verify_rows = verify_block(cell_data, series, ALL_CELLS)
        for r in verify_rows:
            cell = r["cell"].lower()
            run_dir = os.path.join(a.batch_root, f"{a.batch_id}_{cell}_{a.seq}")
            r["ate_results_lio"] = _results_lio_ate(run_dir)
        meta = meta_block(cell_data, a.batch_id, a.seq, a.build_commit, a.batch_root, ALL_CELLS)
        blocks = [meta, emit_verify(verify_rows), emit_series(series),
                  emit_summary(summary), emit_pairs(pairs)]
        with open(a.emit_blocks, "w", newline="\n") as f:
            f.write("\n".join(blocks) + "\n")
        print(f"\n# wrote {a.emit_blocks}", file=sys.stderr)


if __name__ == "__main__":
    main()
