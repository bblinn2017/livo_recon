#!/usr/bin/env python3
"""P-1: ATE over time -- the deliverable this project had never produced.

Four PNGs per batch, built from data dx_report.py's load_cell() already
computes (register rule: "no extra file, no GT export, nothing that can
be blocked by retention"):

  1  ate_k.png            item 2, THE FIGURE.  Rolling-prefix ATE_k
                           (traj_divergence.rolling_prefix_ate, C-6) per
                           cell, one panel per arm.  "What ATE would have
                           read if the run had stopped at k."
  2  instantaneous.png     item 1 + 3.  Per-scan e_glob, log y, cells
                           overlaid per arm, first bucket shaded.  Per-
                           scan (not bucket-median) per the register's
                           own item-0 finding: a median-then-RMS double
                           aggregation runs 0.65-1.18x off the real
                           per-scan scalar.
  3  diagnostics.png       item 4.  n_res/refusal/nis/div_pos sharing the
                           x-axis with instantaneous error, one named
                           cell (default L1, the largest transient).
  4  q_channels.png        item 5.  cov_acc/cov_gyr over time with the
                           Allan floor as a band, for whichever cell
                           pairs are passed via --q-pair (adaptive_q on
                           vs off, e.g. L5 vs Q-L5).

Usage:
  p1_plots.py --batch-root livo_recon_results/dx1r --batch-id dx1r \
      --cells l0 l1 l2 l3 l4 l5 l6 l0d l1d l2d l3d l4d \
      --gt bench_eval/gt/eee_01.csv --gt-format ntu_csv \
      --frame-correction ntu_prism --out livo_recon_results/dx1r/p1
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import dx_report as dxr                                   # noqa: E402
from traj_divergence import rolling_prefix_ate             # noqa: E402

# Allan-deviation IMU noise floor, from the bags' own static segments
# (register constraint row "T7 step 3") -- variance, not std, since
# cov_acc/cov_gyr are variances.
ALLAN_GYR = (1.65e-4, 7.39e-4)
ALLAN_ACC = (7.04e-4, 7.31e-3)

def _split_arms(cells):
    pca = [c for c in cells if not c.endswith("d")]
    deb = [c for c in cells if c.endswith("d")]
    return pca, deb


def fig_ate_k(cell_data, aligned_src_dst, out_path, first_bucket_s):
    pca, deb = _split_arms(list(cell_data))
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=False)
    for ax, cells, title in [(axes[0], pca, "pca arm"), (axes[1], deb, "debiased arm")]:
        for cell in cells:
            if cell not in aligned_src_dst:
                continue
            t = cell_data[cell]["t"]
            ate_k = aligned_src_dst[cell]
            ax.plot(t - t[0], ate_k, label=cell.upper(), linewidth=1.1)
        ax.axvspan(0, first_bucket_s, color="0.85", zorder=0, label="first bucket" if title == "pca arm" else None)
        ax.set_title(f"ATE_k -- {title}")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("ATE_k (m)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    fig.suptitle("P-1 item 2 -- rolling-prefix ATE_k: \"what ATE would have read if the run had stopped at k\"")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_instantaneous(cell_data, out_path, first_bucket_s):
    pca, deb = _split_arms(list(cell_data))
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=False)
    for ax, cells, title in [(axes[0], pca, "pca arm"), (axes[1], deb, "debiased arm")]:
        for cell in cells:
            d = cell_data[cell]
            t = d["t"] - d["t"][0]
            e = np.clip(d["e_glob"], 1e-4, None)
            ax.plot(t, e, label=cell.upper(), linewidth=0.8, alpha=0.85)
        ax.axvspan(0, first_bucket_s, color="0.85", zorder=0)
        ax.set_yscale("log")
        ax.set_title(f"instantaneous e_glob (log y) -- {title}")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("e_glob (m)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("P-1 item 1 -- per-scan e_glob, not bucket medians (item 0's own finding)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_diagnostics(cell_data, cell, out_path, first_bucket_s):
    if cell not in cell_data:
        print(f"diagnostics: {cell} not loaded, skipping", file=sys.stderr)
        return
    d = cell_data[cell]
    t = d["t"] - d["t"][0]
    diag = d["diag"]
    fig, axes = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    axes[0].plot(t, np.clip(d["e_glob"], 1e-4, None), color="k", linewidth=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("e_glob (m)")
    panels = [("n_res", "n_res"), ("refusal", "refusal"), ("nis", "nis"), ("div_pos", "DIV_pos")]
    for ax, (key, label) in zip(axes[1:], panels):
        v = diag.get(key, np.full(len(t), np.nan))
        ax.plot(t, v, linewidth=0.8, color="C1")
        ax.set_ylabel(label)
    for ax in axes:
        ax.axvspan(0, first_bucket_s, color="0.9", zorder=0)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle(f"P-1 item 4 -- diagnostic strip, {cell.upper()}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_q_channels(pairs, out_path):
    """pairs: list of (label, run_dir_on, run_dir_off) -- spline_q.csv read
    directly (not via load_cell(), which only carries the GT-resampled
    subset) so the FULL per-scan cadence is visible, not just what
    survived matching to GT."""
    import csv
    n = len(pairs)
    if n == 0:
        print("q_channels: no pairs given, skipping", file=sys.stderr)
        return
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.2 * n), squeeze=False)
    for i, (label, on_dir, off_dir) in enumerate(pairs):
        for j, (chan, floor) in enumerate([("cov_acc", ALLAN_ACC), ("cov_gyr", ALLAN_GYR)]):
            ax = axes[i][j]
            for name, run_dir, style in [(f"{label} (Q on)", on_dir, "-"), (f"{label} (Q off)", off_dir, "--")]:
                path = os.path.join(run_dir, "spline_q.csv")
                if not os.path.exists(path):
                    continue
                rows = list(csv.DictReader(open(path)))
                t = np.array([float(r["t_abs"]) for r in rows])
                v = np.array([float(r[f"{chan}_pre"]) for r in rows])
                ax.plot(t - t[0], np.clip(v, 1e-6, None), style, linewidth=0.7, label=name)
            ax.axhspan(floor[0], floor[1], color="0.85", zorder=0, label="Allan floor" if i == 0 and j == 0 else None)
            ax.set_yscale("log")
            ax.set_title(f"{chan}_pre -- {label}")
            ax.set_xlabel("t (s)")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, which="both")
    fig.suptitle("P-1 item 5 -- Q channels over time, Allan floor band")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-root", required=True)
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--seq", default="eee_01")
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--gt-format", default="ntu_csv", choices=list(dxr.LOADERS))
    ap.add_argument("--frame-correction", default="ntu_prism", choices=["none", "ntu_prism", "hilti_tip"])
    ap.add_argument("--out", required=True, help="output dir for the 3 batch-level PNGs")
    ap.add_argument("--diag-cell", default=None, help="cell for diagnostics.png (default: first cell)")
    ap.add_argument("--first-bucket-s", type=float, default=8.3, help="R-0's first-bucket width (332s/40)")
    ap.add_argument("--q-pair", nargs=3, action="append", default=[], metavar=("LABEL", "ON_DIR", "OFF_DIR"),
                    help="repeatable: label, adaptive_q-ON run dir, adaptive_q-OFF run dir")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)

    cell_data = {}
    ate_k_by_cell = {}
    for cell in a.cells:
        run_dir = os.path.join(a.batch_root, f"{a.batch_id}_{cell}_{a.seq}")
        d = dxr.load_cell(run_dir, a.gt, a.gt_format, a.frame_correction)
        if d is None:
            print(f"WARNING: {cell} not loaded ({run_dir})", file=sys.stderr)
            continue
        cell_data[cell] = d
        # ATE_k needs the SAME aligned src/dst pair load_cell() used --
        # reconstruct via the resample it already did, by re-deriving
        # from e_glob's own inputs is wasteful; instead recompute directly
        # from odometry.txt/GT here since load_cell() doesn't return src/dst.
        est_path = os.path.join(run_dir, "odometry.txt")
        t_e, p_e, q_e = dxr.load_tum(est_path)
        t_g, p_g, q_g = dxr.LOADERS[a.gt_format](a.gt)
        p_e, q_e = dxr.apply_frame_correction(p_e, q_e, a.frame_correction)
        p_i, q_i, valid = dxr.resample(t_e, p_e, q_e, t_g, 0.1, method="linear")
        src, dst = p_i[valid], p_g[valid]
        ate_k, meta = rolling_prefix_ate(src, dst)
        ate_k_by_cell[cell] = ate_k
        if not np.isfinite(meta["max_drift"]) or meta["max_drift"] > 1e-6:
            print(f"  {cell}: rolling_prefix_ate max_drift={meta['max_drift']:.3e} "
                  f"over {meta['n_checkpoints']} checkpoints", file=sys.stderr)

    fig_ate_k(cell_data, ate_k_by_cell, os.path.join(a.out, "ate_k.png"), a.first_bucket_s)
    fig_instantaneous(cell_data, os.path.join(a.out, "instantaneous.png"), a.first_bucket_s)
    diag_cell = a.diag_cell or a.cells[0]
    fig_diagnostics(cell_data, diag_cell, os.path.join(a.out, "diagnostics.png"), a.first_bucket_s)

    pairs = [(label, on_dir, off_dir) for label, on_dir, off_dir in a.q_pair]
    fig_q_channels(pairs, os.path.join(a.out, "q_channels.png"))

    print(f"wrote 4 PNGs to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
