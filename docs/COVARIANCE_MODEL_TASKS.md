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
- [x] Implement isotropic dynamic contribution (norm-squared/3 times identity).
- [x] Implement axis-aware dynamic contribution while retaining full frame
  transformations into the propagation covariance.
- [x] Keep bias random-walk noise distinct from acc/gyr measurement noise.
- [ ] Verify fixed mode reproduces the existing propagation exactly.
- [ ] Compare beta=0 and filtered variants on the same real-data windows.

## Stage 3 — correlation-aware Gamma_L

- [ ] Audit the existing residual-redundancy/Woodbury implementations and
  document which production paths currently use them.
- [ ] Port the chosen correlation model to the full joint-knot Jacobian.
- [ ] Accumulate each correlated group as `H_g^T C_g^-1 H_g` and
  `-H_g^T C_g^-1 r_g` in the one canonical joint solve.
- [ ] Preserve an independent-residual baseline mode.
- [ ] Validate information growth versus residual density/redundancy.
- [ ] Re-evaluate P0/Q_k consistency only after Gamma_L is corrected.

## Experiment and reporting constraints

- Use real data for behavioral validation; synthetic tests are only for
  matrix/Jacobian identities.
- Coding-agent reports contain implementation/build/run status and raw data;
  interpretation is performed here.
- Keep first-post-calibration-frame and 30 s stationary statistics compact.
- Note explicitly that covariance/trajectory performance depends on process
  noise until Stage 2 is tuned, and on residual correlation until Stage 3.
