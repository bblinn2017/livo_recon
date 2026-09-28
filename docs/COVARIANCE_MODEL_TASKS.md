# Covariance-model implementation checklist

The work is deliberately staged so that changes to the prior, propagation,
and LiDAR information are identifiable in real-data experiments.

## Stage 1 — calibration-derived P0

- [x] Require stationary initialization and remove the ambiguous
  `use_calib`, `use_calib_var`, and `use_calib_bias` switches.
- [x] Split per-bias state application under `calib/stationary`.
- [x] Move runtime acc/gyro and bias-random-walk noise out of `state/cov`
  into `imu/process_noise`; reject removed keys instead of silently falling
  back to defaults.
- [x] Audit calibration, state initialization, and propagation ownership.
- [x] Preserve `calib/p0/mode: configured` as the exact baseline path.
- [x] Add `calibration_derived` P0 mode after the required stationary window.
- [x] Estimate accelerometer/gyroscope covariance of the mean with a
  correlation-aware Bartlett/HAC estimator.
- [x] Propagate accelerometer-mean covariance through the production gravity
  alignment and accelerometer-bias calculation.
- [x] Preserve the induced attitude–accelerometer-bias cross covariance.
- [x] Keep yaw and gravity uncertainty configured: stationary specific force
  cannot identify yaw or separate gravity from accelerometer bias.
- [x] Encode the required initialization semantics: map-origin position and
  stationary velocity are known, with small positive numerical variances.
- [x] Log the actual pre-propagation P0, labeled blocks, full matrix,
  eigenvalues, state mean, and calibration mean covariances for both modes.
- [x] Represent the unobservable transverse tilt–accelerometer-bias direction
  with one balanced acceleration-equivalent uncertainty and its negative
  cross-covariance; retain separate gyro-bias and radial-bias floors.
- [x] Real-data A/B: configured baseline versus calibration-derived P0 on
  eee_01, first post-calibration frame and 30 s stationary window.
- [x] Validate per-block NEES before IMU propagation, post-IMU, and post-LIO.
- [x] Preserve the actual post-calibration P0 and its mode in StateGroup so
  later diagnostics cannot mistake configured inputs for effective P0.
- [x] Make `initialization_consistency_all_scans.csv` report both configured
  inputs and actual effective P0 scalar projections.
- [x] Run the full-factorial R43 P0 sweep over known p/v variance, balanced
  ambiguity scale, yaw variance, bias floors, and gating on/off. 288/384
  cells completed; the 96 `tilt_ba_ambiguity_accel_std=0` cells crash
  (`scan-head covariance is not positive definite` in
  `buildJointKnotImuPrior`'s LDLT check) — see R43's REPORT.md for the
  root-cause finding.
- [x] Stabilize every configured or calibration-derived P0 by symmetric
  eigendecomposition and a separately reported numerical eigenvalue floor.
- [x] Preserve the R43 candidate and configured-P0 control as named overlays.
- [x] Treat the zero-ambiguity P0 cells as an invalid overconfident boundary,
  not as a missing candidate: they were stabilized numerically and were worse
  than the selected nonzero-ambiguity region on real data.
- [x] Retain the best observed R43 region as the named Phase-1 candidate while
  keeping configured P0 as the baseline control.

## Stage 2 — motion-dependent Q_k

- [x] Add one mode flag: `fixed`, `isotropic`, or `axis_aware`.
- [x] Retain calibrated per-axis stationary variance as the noise floor.
- [x] Define gravity-removed specific-force excitation using the production
  frame/sign convention and verify it is near zero while stationary.
- [x] Add acceleration and angular-rate scale parameters `s_a`, `s_omega`.
- [ ] Determine which scales are available from the VN-100/NTU VIRAL sensor
  specification; do not infer dynamic scale error from stationary data.
- [ ] Add programmatic real-data calibration/ablation for unidentified scales.
- [x] Add optional low-pass excitation filtering; `beta=0` disables it.
- [x] Bound each dynamic variance contribution independently.
- [x] Subtract the calibrated stationary measurement-noise energy before
  applying motion scale/cap so the stationary floor is not counted twice.
- [x] Implement isotropic dynamic contribution (norm-squared/3 times identity).
- [x] Implement axis-aware dynamic contribution while retaining full frame
  transformations into the propagation covariance.
- [x] Keep bias random-walk noise distinct from acc/gyr measurement noise.
- [ ] Verify fixed mode reproduces the existing propagation exactly.
- [ ] Compare beta=0 and filtered variants on the same real-data windows.

## Stage 3 — correlation-aware Gamma_L

- [x] Audit the existing residual-redundancy/Woodbury implementations and
  document which production paths currently use them.
- [x] Port the existing matched-plane shared-uncertainty model to the full
  joint-knot Jacobian as an experimental first correlation model.
- [x] Accumulate each correlated group as `H_g^T C_g^-1 H_g` and
  `-H_g^T C_g^-1 r_g` in the one canonical joint solve.
- [x] Preserve an independent-residual baseline mode.
- [x] Add a dense-`C^-1` algebra oracle and compact all-scan engagement and
  information diagnostics.
- [x] Replace R45's scalar `sqrt(var_i var_j)` approximation with the signed,
  rank-at-most-three `J_i P_plane J_j^T` covariance in both estimator paths.
- [x] Remove the trace-discount clamp and the misleading effective-residual
  count; report exact information ratios and information-increase groups.
- [x] Preserve knot IDs/timestamps, immutable-head labels, knot-specific
  biases, and gravity in the first-frame trajectory export.
- [ ] Compile and run the dense covariance oracle through the coding agent.
- [ ] Re-run the Phase-3 real-data comparison; R45's scalar results are not
  evidence for or against the corrected model.
- [ ] Validate information growth versus residual density/redundancy.
- [ ] Identify and validate correlations beyond shared map-plane uncertainty
  before enabling a broader live covariance model; do not fold prior-state or
  deskew uncertainty into measurement covariance and count it twice.
- [ ] Re-evaluate P0/Q_k consistency only after Gamma_L is corrected.

The model groups by matched map-plane identity and represents the shared
plane-fit error with its actual three-parameter covariance. For residuals
`i,j` on one `VoxelPlane`, the correlated term is
`rho * J_nq_i * P_plane * J_nq_j^T`. Factoring `P_plane=L L^T` gives a
rank-at-most-three Woodbury factor `U_i=sqrt(rho) J_nq_i L`. A scalar
`sqrt(plane_var_i*plane_var_j)` construction is incorrect because it discards
the signed/directional cross-covariance. R45 used that scalar approximation;
R46 replaces it in both the joint-knot and legacy decoupled accumulators.

The exact covariance update is not trace-clamped: blending it back toward the
independent result would no longer correspond to the stated measurement
model. A trace ratio is retained as a diagnostic, together with a count of
groups whose signed correlations add information in residual-difference
directions. It is not reported as an "effective residual count", since that
interpretation is invalid when the exact trace can increase.

### Incremental Stage-3 sequence

Do not introduce the broad surface/RBF model in the same behavioral experiment
as the first per-plane correction. At every step retain the preceding mode as
a control and measure whether covariance consistency improves without making
the stationary trajectory worse.

1. **Independent residual control.** Retain the current diagonal `Gamma_L`.
2. **Per-`VoxelPlane` information aggregation.** Marginalize the shared
   plane-fit uncertainty within each exact `plane_id` and add the resulting
   group `(Gamma_g,b_g)` once to the canonical joint solve. This is implemented
   by the joint-knot Woodbury path. It is information aggregation, not a
   literal averaged scalar residual, so point-specific timestamps and knot
   Jacobians are preserved.
3. **Validate overconfidence reduction.** Compare information ratio,
   posterior covariance, normalized errors, first-frame correction, and 30 s
   stationary drift against the independent control. Do not proceed merely
   because trace information decreased.
4. **Optional exact group compression.** If storage or assembly cost warrants
   it, add covariance-whitened QR/SVD pseudo-residual compression and verify it
   reproduces the uncompressed per-plane `(Gamma_g,b_g)` to numerical
   tolerance. This is not required for estimator correctness.
5. **Cross-voxel surface association.** Determine when adjacent `VoxelPlane`
   objects represent one physical surface using normal, offset, adjacency, and
   observation-history evidence. Keep this diagnostic-only until validated.
6. **Broader surface covariance.** Only after step 5, add a surface-patch or
   low-rank RBF latent term between plane-level groups. Use a dense small-case
   oracle and a deterministic scalable approximation for live runs.

The legacy `plane_averaged` residual-collapse mode is not step 2. It replaces
multiple measurements with one scalar and is intentionally rejected by the
coupled estimator because residuals at different point times have different
joint-knot Jacobians.

The residual-mode audit removed `plane_averaged`, both count-weighted modes,
per-residual `info_gain`, and the legacy Woodbury rescale/directional
workarounds. It also removed the density and adaptive-chi2 global residual
scalers while retaining reduced chi-square as a read-only diagnostic. See
`docs/RESIDUAL_MODE_AUDIT.md`. The only remaining
`residual_redundancy` choices are the independent control and direct
per-`VoxelPlane` covariance marginalization.

## Experiment and reporting constraints

- Use real data for behavioral validation; synthetic tests are only for
  matrix/Jacobian identities.
- Coding-agent reports contain implementation/build/run status and raw data;
  interpretation is performed here.
- Keep first-post-calibration-frame and 30 s stationary statistics compact.
- Note explicitly that covariance/trajectory performance depends on process
  noise until Stage 2 is tuned, and on residual correlation until Stage 3.
