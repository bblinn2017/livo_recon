# R37 follow-up ledger

## R37-F01 — Build and permanent tests

The coding agent must compile the repaired tree and run all four registered
CTest targets. In particular, confirm the retained `test_spline` and
`test_ds_csr` targets compile under the production Release configuration and
that the new joint-knot marginal-prior/Jacobian checks pass.

## R37-F02 — Repeat the matched four-cell campaign

Repeat measurement/tail × sensor-only/gating-state on the identical `eee_01`
window. The purpose is to determine whether preserving scan-head uncertainty
in the mutable-knot marginal corrects the previously tiny update and the
357.8 m measurement-time divergence. Return complete, not excerpted,
diagnostic CSVs and odometry.

Required comparisons include `joint_knot_corrections.csv` and
`joint_knot_scan_summary.csv`: signed LiDAR/prior/realized corrections,
residual timestamp counts by interval, zero-Jacobian counts, covariance and
information eigenvalues, convergence reason, and distance toward the pre-IMU
stationary state.

## R37-F03 — Do not accept tail-time as a fallback without control parity

The prior 3.202 m tail result remains unacceptable relative to the established
centimeter-scale controls. Both residual-time policies must be compared to a
same-build splineless/decoupled control before either is called correct.

## R37-F04 — Knot-count sweep

Keep the 3/5/6-knot sweep blocked until the default four-knot campaign is
stable and understood.

## R37-F05 — Live configuration rejection and diagnostic collection

Verify removed coupled keys fail at startup. Update the job copy-out list for
`joint_knot_corrections.csv` and `joint_knot_scan_summary.csv` if
`outputs/debug_log_dir` is not set directly to the per-job output directory.
