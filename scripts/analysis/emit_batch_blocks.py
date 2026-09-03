#!/usr/bin/env python3
"""Generic [BATCH-META]/[BATCH-VERIFY]/[*-SERIES]/[*-SUMMARY]/[*-PAIRS] block
emitter for a batch whose cells do NOT follow DX-1R's fixed 20-cell ladder
naming (dx_report.py's ALL_CELLS/LADDER_CELLS/PAIRS are hardcoded to that one
batch). Reuses dx_report.py's load_cell()/summary_block()/meta_block()/
verify_block() directly (already generic over an arbitrary cell_data dict +
dispatched_cells list) and supplies its own SERIES (no ladder filter) and
PAIRS (arbitrary pair list, register rule 5/5a: a round's numbers need their
per-scan/per-bucket series in the register, not just a hand-typed summary
table).

Register rule 5a: "A SUMMARY TABLE IS NOT A DATA BLOCK." This exists because
that rule was violated for DX-5/DV-1(b)/T3-0e/Q-1 in the seventeenth round --
those sections shipped hand-computed p50/p90 tables with no block behind
them, exactly the DX-3 defect the bug ledger already calls out.

Usage:
  emit_batch_blocks.py --batch-id dx5 --seq eee_01 \
      --cell off_off=livo_recon_results/dx5/dx5_l5_ex_off_off_eee_01 \
      --cell b01_off=livo_recon_results/dx5/dx5_l5_ex_b01_off_eee_01 \
      ... \
      --pair off_off:b01_off --pair off_off:b10_off --pair off_off:off_on \
      --pair b01_off:b01_on --pair b10_off:b10_on \
      --gt bench_eval/gt/eee_01.csv --gt-format ntu_csv \
      --frame-correction ntu_prism --build-commit <sha> \
      --out livo_recon_results/dx5/dx5_blocks.txt
"""
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import dx_report as dxr  # noqa: E402

# The diagnostics load_cell() actually populates today (register rule: name
# what is real, do not invent columns that are not logged). d_bias_*/d_gravity
# are the new DX-5 columns (43b598a); nan on any cell whose spline_q.csv
# predates that commit or whose imu_fit_mode is off.
GENERIC_DIAGNOSTICS = [
    "refusal", "div_pos", "div_rot", "nis", "sdiag_share", "pvar_share",
    "floor_share", "prior_pose_share", "n_res", "w_per_res", "cov_acc",
    "cov_gyr", "refit_dtraj_rms", "boundary_dpos", "trP_drop",
    "d_bias_acc_norm", "d_bias_gyr_norm", "d_gravity_norm",
]


def series_block_generic(cell_data):
    """Same 40-bucket median scheme as dx_report.series_block(), but over
    EVERY cell in cell_data -- no LADDER_CELLS membership filter, since a
    non-DX-1R batch's cell names were never in that list to begin with."""
    rows = []
    for cell, d in cell_data.items():
        if d is None:
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
                v = arr[m]
                v = v[np.isfinite(v)]
                return float(np.sqrt(np.mean(v ** 2))) if len(v) else float("nan")

            def frac(arr):
                v = arr[m]
                v = v[np.isfinite(v)]
                return float(np.mean(v)) if len(v) else float("nan")

            rows.append(dict(
                cell=cell.upper(), t_s=float(t[m][len(t[m]) // 2] - t0),
                # P-1 item 0: RMS shipped alongside the median -- a median
                # over a bucket discards exactly the within-bucket spikes ATE
                # is built from (bug-ledger row, this round).
                e_glob=med(d["e_glob"]), e_glob_rms=rms(d["e_glob"]),
                e_pre=med(d["e_pre"]), e_pre_rms=rms(d["e_pre"]),
                e_win=med(d["e_win"]), e_win_rms=rms(d["e_win"]),
                gain_cum=float(gain_cum[m][-1]) if m.any() else float("nan"),
                v_err=med(d["v_err"]),
                refusal=med(d["diag"].get("refusal", np.full(len(t), np.nan))),
                div_pos=med(d["diag"].get("div_pos", np.full(len(t), np.nan))),
                nis=med(d["diag"].get("nis", np.full(len(t), np.nan))),
                n_res=med(d["diag"].get("n_res", np.full(len(t), np.nan))),
                cov_acc=med(d["diag"].get("cov_acc", np.full(len(t), np.nan))),
                cov_gyr=med(d["diag"].get("cov_gyr", np.full(len(t), np.nan))),
                d_bias_acc_norm=med(d["diag"].get("d_bias_acc_norm", np.full(len(t), np.nan))),
                d_bias_gyr_norm=med(d["diag"].get("d_bias_gyr_norm", np.full(len(t), np.nan))),
                d_gravity_norm=med(d["diag"].get("d_gravity_norm", np.full(len(t), np.nan))),
                # C-7 (nineteenth round): AdaptiveQ gate columns -- see
                # dx_report.py's load_cell()/series_block() for why these
                # (not cov_acc/cov_gyr) tell a live module from a disabled
                # one.
                z_acc=med(d["diag"].get("z_acc", np.full(len(t), np.nan))),
                z_gyr=med(d["diag"].get("z_gyr", np.full(len(t), np.nan))),
                acf1_acc=med(d["diag"].get("acf1_acc", np.full(len(t), np.nan))),
                acf1_gyr=med(d["diag"].get("acf1_gyr", np.full(len(t), np.nan))),
                active_frac=frac(d["diag"].get("active", np.full(len(t), np.nan))),
                clamped_frac=frac(d["diag"].get("clamped", np.full(len(t), np.nan))),
            ))
    return rows


def emit_series_generic(name, rows):
    # C-7 (nineteenth round): bumped to v2 -- z_acc/z_gyr/acf1_acc/acf1_gyr/
    # active_frac/clamped_frac appended.
    out = [f"[{name}-SERIES v2]",
           "cell,t_s,e_glob,e_glob_rms,e_pre,e_pre_rms,e_win,e_win_rms,gain_cum,v_err,"
           "refusal,div_pos,nis,n_res,cov_acc,cov_gyr,"
           "d_bias_acc_norm,d_bias_gyr_norm,d_gravity_norm,"
           "z_acc,z_gyr,acf1_acc,acf1_gyr,active_frac,clamped_frac"]
    for r in rows:
        out.append(
            f"{r['cell']},{r['t_s']:.3f},{dxr._fnum(r['e_glob'],5)},{dxr._fnum(r['e_glob_rms'],5)},"
            f"{dxr._fnum(r['e_pre'],5)},{dxr._fnum(r['e_pre_rms'],5)},"
            f"{dxr._fnum(r['e_win'],5)},{dxr._fnum(r['e_win_rms'],5)},"
            f"{dxr._fnum(r['gain_cum'],5)},{dxr._fnum(r['v_err'],4)},"
            f"{dxr._fnum(r['refusal'],3)},{dxr._fnum(r['div_pos'],3)},{dxr._fnum(r['nis'],3)},"
            f"{dxr._fint(r['n_res'])},{dxr._fnum(r['cov_acc'],4)},{dxr._fnum(r['cov_gyr'],6)},"
            f"{dxr._fnum(r['d_bias_acc_norm'],6)},{dxr._fnum(r['d_bias_gyr_norm'],6)},"
            f"{dxr._fnum(r['d_gravity_norm'],6)},"
            f"{dxr._fnum(r['z_acc'],3)},{dxr._fnum(r['z_gyr'],3)},{dxr._fnum(r['acf1_acc'],3)},{dxr._fnum(r['acf1_gyr'],3)},"
            f"{dxr._fnum(r['active_frac'],3)},{dxr._fnum(r['clamped_frac'],3)}")
    out.append(f"[/{name}-SERIES]")
    return "\n".join(out)


def emit_summary_generic(name, rows):
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
    out = [f"[{name}-SUMMARY v1]", ",".join(cols)]
    for r in rows:
        def fmt(c):
            v = r[c]
            if isinstance(v, float):
                return "nan" if not np.isfinite(v) else f"{v:.5g}"
            return str(v)
        out.append(",".join([r["cell"]] + [fmt(c) for c in cols[1:]]))
    out.append(f"[/{name}-SUMMARY]")
    return "\n".join(out)


def pairs_block_generic(cell_data, pairs, diagnostics):
    rows = []
    for a_id, b_id in pairs:
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

        for diag in diagnostics:
            if diag == "gain":
                xA, xB = A["gain"][ia], B["gain"][ib]
            elif diag == "v_err":
                xA, xB = A["v_err"][ia], B["v_err"][ib]
            else:
                xA, xB = dget(A, diag, ia), dget(B, diag, ib)
            dx = xB - xA
            r, se, n = dxr._cluster_se(d_t, dx, t)
            lag, r_lag = dxr._best_lag(d_t, dx)
            rows.append(dict(pair=f"{a_id.upper()}->{b_id.upper()}", diagnostic=f"d_{diag}",
                             r=r, se_cluster=se, n_scans=n, best_lag_scans=lag, r_at_best_lag=r_lag))
    return rows


def emit_pairs_generic(name, rows):
    out = [f"[{name}-PAIRS v1]", "pair,diagnostic,r,se_cluster,n_scans,best_lag_scans,r_at_best_lag"]
    for r in rows:
        out.append(f"{r['pair']},{r['diagnostic']},{dxr._fnum(r['r'],2)},{dxr._fnum(r['se_cluster'],2)},"
                   f"{r['n_scans']},{dxr._fint(r['best_lag_scans'])},{dxr._fnum(r['r_at_best_lag'],2)}")
    out.append(f"[/{name}-PAIRS]")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", action="append", required=True, metavar="NAME=RUN_DIR",
                    help="repeatable: one cell's name and run_dir")
    ap.add_argument("--pair", action="append", default=[], metavar="A:B",
                    help="repeatable: a pair of --cell NAMEs for the PAIRS block")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--gt-format", default="ntu_csv", choices=list(dxr.LOADERS))
    ap.add_argument("--frame-correction", default="ntu_prism",
                    choices=["none", "ntu_prism", "hilti_tip"])
    ap.add_argument("--batch-id", required=True, help="also the block-name PREFIX, upper-cased")
    ap.add_argument("--seq", default="eee_01")
    ap.add_argument("--build-commit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    name = a.batch_id.upper()
    cell_data, dispatched, run_dirs = {}, [], {}
    for spec in a.cell:
        cname, run_dir = spec.split("=", 1)
        dispatched.append(cname)
        run_dirs[cname] = run_dir
        d = dxr.load_cell(run_dir, a.gt, a.gt_format, a.frame_correction)
        cell_data[cname] = d
        if d is None:
            print(f"WARNING: {cname} not loaded ({run_dir})", file=sys.stderr)

    series = series_block_generic(cell_data)
    summary = dxr.summary_block(cell_data)
    pairs_list = [tuple(p.split(":", 1)) for p in a.pair]
    pairs = pairs_block_generic(cell_data, pairs_list, GENERIC_DIAGNOSTICS) if pairs_list else []

    verify_rows = dxr.verify_block(cell_data, series, dispatched)
    for r in verify_rows:
        cname = None
        for spec in a.cell:
            if spec.split("=", 1)[0].upper() == r["cell"]:
                cname = spec.split("=", 1)[0]
                run_dir = spec.split("=", 1)[1]
                break
        r["ate_results_lio"] = dxr._results_lio_ate(run_dir) if cname else float("nan")
        # DX-1R-VERIFY's check (3) is DISCRETE_CELLS-specific and does not
        # apply outside that one batch's naming -- leave as the "nan" (not
        # applicable) the generic verify_block already emits for anything
        # not in dxr.DISCRETE_CELLS.

    meta = dxr.meta_block(cell_data, a.batch_id, a.seq, a.build_commit, None, dispatched, run_dirs=run_dirs)
    meta = meta.replace("[DX1R-META v2]", f"[{name}-META v2]").replace("[/DX1R-META]", f"[/{name}-META]")
    verify = dxr.emit_verify(verify_rows).replace("[DX1R-VERIFY v2]", f"[{name}-VERIFY v2]").replace(
        "[/DX1R-VERIFY]", f"[/{name}-VERIFY]")

    blocks = [meta, verify, emit_series_generic(name, series), emit_summary_generic(name, summary)]
    if pairs:
        blocks.append(emit_pairs_generic(name, pairs))

    text = "\n".join(blocks) + "\n"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="\n") as f:
        f.write(text)
    print(text)
    print(f"# wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
