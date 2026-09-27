# R34 follow-up ledger

## R34-F01 — Complete coupled-mode source separation

**Status: In progress in the working candidate; validation and final extraction remain open.**

Keep one implementation file for each coupled family:

- `lio_coupled_local_spline.cpp`
- `lio_coupled_single_tail.cpp`
- `lio_coupled_direct_lidar_imu.cpp`
- `lio_coupled_covariance_all_knots.cpp`

Each file must ultimately own both its legacy control and matched physical-RPV numerical logic, family-specific state/covariance mutation, and family-specific diagnostics. `lio_coupled.cpp` should retain only scan lifecycle, shared factor construction, dispatch, common trust-region/application ordering, and common logging. Continue extracting the still-inline legacy branches only after deterministic equivalence checkpoints. Do not move mode behavior into `LioBase`, and do not introduce lifecycle inheritance merely for dispatch.

Acceptance requires clean/incremental compile time and peak-memory measurements proving that changing one mode does not rebuild the former monolith, plus deterministic pre/post-refactor equivalence for every legacy mode.

## R34-F02 — Four genuine matched physical-RPV counterparts

**Status: Implemented in the working candidate; build/runtime validation open.**

The four counterparts must remain architecturally distinct:

- local spline: locally supported minimum-norm realization through the ordinary endpoint spline Jacobian;
- single tail: final-control-point-only realization with explicit nine-dimensional residual;
- direct LiDAR-to-IMU: complete EKF-state conditional mean/covariance update, including bias/gravity cross-covariances, followed by covariance-consistent spline reconciliation;
- covariance all knots: dense covariance-mediated all-free-knot realization.

Permanent tests must prove coefficient support, full-state cross-covariance behavior, identical physical measurement-stage equations, head constraints, and requested/scaled/applied/nonlinear realization consistency. A same-run campaign must confirm that no counterpart is a pass-through alias.

## R34-F03 — Production scan-entry physical reference

**Status: Corrected in the working candidate; validation open and blocking.**

Use the immutable `MeasureGroup::{rot,pos,vel}_after_imu` snapshot, never `coupled_prop_`, as the physical restoring reference. At GN iteration zero require the complete relative `[theta,p,v]` vector to be near zero. Log the exact snapshot source and frames. Add an integration test that exercises the real production caller with a large nonidentity gravity-aligned attitude; a helper-level test alone is insufficient.

## R34-F04 — Correctly propagated IMU process covariance

**Status: Open.**

Derive and validate `Lambda_Q`, uncertain-head injection, between-window propagation, endpoint covariance, and all R/P/V cross blocks against the production residual-derived information and scan-entry EKF covariance. Until closed, retain the label `IMU-residual-derived head-conditioned spline covariance`.

## R34-F05 — Post-fix matched campaign

**Status: Open; blocked on build/tests for F02 and F03.**

Rerun all controls and all four genuine counterparts from one resolved configuration. The prior R33 physical-RPV performance numbers are invalid for estimator assessment because the production reference was identity/zero or stale. Preserve first-frame per-iteration state chains and compare whether position/velocity return toward zero/x0 without treating the nonzero gravity-aligned absolute attitude as error.

## R34-F06 — Complete diagnostic extraction

**Status: Shared writer implemented; migration of historical diagnostics remains open.**

`PoseControlDiagnosticWriter` now owns named matrix/vector serialization and supports extensible records containing labels, scalars, vectors, and matrices under `src/diagnostics/pose_control`. Existing coupled matrix/vector dumps route through the shared serializer, and the existing unified CSV row emitter remains the common scalar diagnostic path.

Continue moving the older first-frame solve, covariance-budget, knot-map, PSD, and scan-specific formatting functions out of `lio_coupled.cpp` in equivalence-preserving checkpoints. Maintain one persistent writer per basename so multiple modes cannot independently truncate or interleave the same file. Add golden-schema tests before intentionally changing field names or file formats.
