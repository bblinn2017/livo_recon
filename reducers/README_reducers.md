# R85 host-side reducers: order and exact commands

Python 3 standard library only. They read the returned `phase_summary.csv`, the per-arm `compact/<arm>/short_scan_all.csv.gz` files and the manifest, and write small CSVs into the return dir. None feeds a run. Keep the four files `r85_common.py`, `r85_l0_reduce.py`, `r85_sht_reduce.py`, `r85_run_reducers.sh` in one folder (`r85_common.py` is a copy of the helpers; the scripts add their own folder to the path so `python3 -I` works).
Each script has `--selftest` (SYNTHETIC data built by the test; a sandbox check of the code, not data from any run).

## Variables
```bash
RED=<this folder>
RET=<R85 return dir>            # holds phase_summary.csv and compact/<arm>/short_scan_all.csv.gz; the reducers write here
MAN=<path>/r85_manifest         # arm_table.tsv, l2_reference_values.csv, onset_scans.json
R84=<path>/r84_layer0_vs_layer2_all_arms.csv     # optional, held by Bryce
```

## Order
1. Selftests (once): `for t in r85_common r85_l0_reduce r85_sht_reduce; do python3 -I $RED/$t.py --selftest | tail -1; done`
2. Layer 0 against layer 2 (288 L0 arms, edges 0.20 and 0.25 m):
   `python3 -I $RED/r85_l0_reduce.py --phase-summary $RET/phase_summary.csv --manifest $MAN --out $RET [--r84-all-arms $R84]`
   Writes `r85_l0_rows.csv`, `r85_l0_collapse_counts.csv`, `r85_l0_paired.csv`, `r85_l0_unified.csv`, `r85_l0_notes.txt`, and `r85_l0_missing.txt` only when something is missing (exit 1).
3. Short tier (144 SHT arms):
   `python3 -I $RED/r85_sht_reduce.py --ret $RET --manifest $MAN --out $RET`
   Writes `r85_sht_checkpoints.csv`, `r85_sht_paired.csv`, and `r85_sht_missing.txt` only when something is missing (exit 1).
Both steps in one go: `bash $RED/r85_run_reducers.sh $RET $MAN [$R84]`. Both steps always run; the exit code is 1 when either reported anything missing. `tools/r69/build_return.py --round R85` calls this script itself (option `--r84-all-arms`).
Return all files written, including `r85_l0_notes.txt` and any `*_missing.txt`.

## Common definitions
Native units: ATE and differences in mm (three decimals), counts as counts. A difference is the first-named arm minus the second-named arm. Collapsed = gt_max_mm > 500. A blank cell means the value is not available (missing arm, row or scan), never zero. phase_summary.csv: first row of an arm wins. No ratios, no percentages, no noise bars or confidence intervals; sign counts and counts of cells are given instead.

## r85_l0_rows.csv (one row per L0 arm, 288)
- arm, bag: arm name, bag. overlap: arm_table `mode`, dc = discard, rt = retime_trim. fit: pca | deb. lines: off | on. q: fq | mq from the arm-name suffix. edge: voxel edge (m), 0.20 or 0.25.
- gt_ate_mm, gt_max_mm: whole-run Umeyama ATE and maximum error of the layer-0 (max_layer 0) arm, from phase_summary.csv. collapsed: gt_max_mm > 500.
- l2_ref: layer-2 reference arm of the same cell. l2_source: R81 | R82A | R83 (existing arm; numbers from `l2_reference_values.csv`) or R85 (the twin L2R arm of this round; numbers from phase_summary.csv).
- l2_gt_ate_mm, l2_gt_max_mm, l2_collapsed: the same three quantities for the layer-2 reference.
- d_ate_mm: layer-0 gt_ate_mm minus layer-2 gt_ate_mm (mm); blank when either is missing.

## r85_l0_collapse_counts.csv (one row per overlap, fit, lines, q, edge; counts over the bags)
n_arms; n_collapsed_layer0 / n_collapsed_layer2: bags where that layer's arm has gt_max_mm > 500; n_missing: bags where the layer-0 or the layer-2 value is missing (n_missing_layer0, n_missing_layer2 separately); collapsed_by_bag: tokens `<bag>:<a><b>`, a = layer 0, b = layer 2, 1 collapsed, 0 not collapsed, ? missing.

## r85_l0_paired.csv (one row per overlap, fit, lines, q, edge)
Over the bags where both arms have a value and neither collapsed: n_pairs; n_layer0_lower (d_ate_mm < 0); n_layer0_higher (d_ate_mm > 0); n_equal (d_ate_mm = 0.000); median_d_ate_mm (mm). The columns ending `_all` are the same over every bag with both values, collapsed or not.

## r85_l0_unified.csv (one row per arm type, edge and layer)
Arm type = overlap (rt | dc) x fit (pca | deb) x lines (off | on) x Q (fq | mq), written as `arm_type` (for example `rt_deb_lines-on_mq`) and in its four parts. layer 0 = the L0 arms; layer 2 = their layer-2 references. Columns: n_cells (bags in the group), n_with_value, n_missing, n_collapsed (gt_max_mm > 500), n_noncollapsed, median_gt_ate_noncollapsed_mm (median over the non-collapsed cells that have a value), median_gt_ate_all_mm (median over every cell that has a value), worst_gt_ate_mm and worst_bag (the largest gt_ate_mm of any cell with a value, and its bag), source (`R85` for this round and the manifest reference values, `R84_all_arms` for rows taken from the R84 file).
With `--r84-all-arms` the rows for the edges 0.30 to 0.70 m are added from `r84_layer0_vs_layer2_all_arms.csv` (R84 values; the reducer identifies the bag, overlap, fit, lines, Q, edge and the gt_ate_mm / gt_max_mm columns by header name or by the R84 arm name; layer-0 and layer-2 column pairs, or a `layer` column, are accepted). If the header cannot be identified the line `NOT JOINED: <reason>` is printed and written to `r85_l0_notes.txt`, and the table holds the edges 0.20 and 0.25 only. Rows of that file at 0.20 and 0.25 m are ignored (those come from this round).

## r85_sht_checkpoints.csv (one row per SHT arm and checkpoint scan)
Checkpoint scans 1, 2, 3, 5, 10, 20, 50, 100, 300, 480, 600. Columns: arm, bag, edge, cell (A to H), pfn (point_filter_num), ds (imu.ds.mode: first | off), nb (neighborhood_size), scan, residual_count (state_trace), fs_n_residuals, fs_n_voxels, fs_n_voxels_is_plane (frame_stats), pos_err_mm (distance of the arm's position from its own first-frame reference, mm). A value cell is blank when the arm's table has no such scan or the value is empty. Cells: A pfn 3 / first / nb 1 (counterpart), B pfn 1, C ds off, D pfn 1 + ds off, E nb 2, F pfn 1 + nb 2, G ds off + nb 2, H all three.

## r85_sht_paired.csv (one row per edge, cell other than A, metric and scan)
Metrics fs_n_residuals, fs_n_voxels, fs_n_voxels_is_plane, pos_err_mm. scan = each checkpoint scan, and `stationary` = the arm's median of the metric over its scans 1 to onset-10 (onset = `onset_scan_used` of the bag, `r85_manifest/onset_scans.json`). The difference is cell minus cell A of the same bag and edge, in native units. Columns: factors_changed (which of pfn, ds, nb differ from A) and factor_count (1 single-factor: B, C, E; 2 two-factor: D, F, G; 3 three-factor: H), pfn, ds, nb, n_pairs, median_diff, n_lower (diff < 0), n_higher (diff > 0), n_equal (diff = 0.000), n_arms_ended_before (arms of that edge, cell arm or A arm, with a file, whose table ends before the scan; for `stationary` before onset-10). The 60 s arms have no ATE; nothing in this file is an ATE.

## Missing / notes files
`r85_l0_missing.txt`, `r85_sht_missing.txt`: one line per arm, reference or file without a value (exit 1); an SHT arm without a readable short_scan_all.csv.gz is listed there and left out of the pairs. `r85_l0_notes.txt`: whether the R84 file was joined (no effect on the exit code).
