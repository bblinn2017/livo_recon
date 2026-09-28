# Canonical coupled joint-knot estimator

The coupled pipeline has one estimator. Its mutable state is

`[theta_1,p_1,v_1,...,theta_(N-1),p_(N-1),v_(N-1),`
`bg_0,ba_0,...,bg_(N-1),ba_(N-1),g]`.

Knot 0 is the scan head and is present only as an immutable interpolation
boundary. It has no optimization columns. Knot `N-1` is the scan tail. Actual
knot timestamps are stored, and the default `N` is four.

Translation uses cubic Hermite interpolation between adjacent physical knot
states. Rotation uses SO(3) geodesic interpolation with right-multiplicative
perturbations. Gravity remains an additive, unconstrained three-vector.

Every iteration performs the single fixed-prior information solve

`delta = (P^-1 + Gamma_L)^-1 (b_L + P^-1 vec)`.

There is no state clipping or component-wise damping. The knot posterior is
retained for diagnostics; only its tail-state/tail-bias/shared-gravity marginal is committed to
the production `StateGroup` covariance.

## Experimental axes

- `residual_evaluation_time: measurement`: each point's Jacobian is evaluated
  through the trajectory at its capture timestamp.
- `residual_evaluation_time: tail`: the nominal point is deskewed correctly,
  then its tail-frame coordinate is frozen while differentiating through the
  tail pose.
- `gating_state_uncertainty: false`: association uses sensor and plane
  uncertainty only.
- `gating_state_uncertainty: true`: propagated trajectory uncertainty is added
  to association/gating only. It is never added to the solve weight.

`all_time` is intentionally absent until its measurement model is defined.

## Implementation checklist

- [x] Remove the coupled estimator selector and make joint knots unconditional.
- [x] Remove raw-IMU, coefficient-pose, four legacy, and four physical-RPV
  coupled implementations from the production source and build graph.
- [x] Represent four physical knots by default and store their timestamps.
- [x] Eliminate the head perturbation structurally.
- [x] Use Hermite position/velocity and geodesic attitude interpolation.
- [x] Represent `bg` and `ba` at every knot as production Markov states and
  keep one unconstrained three-vector gravity shared over the scan.
- [x] Record and compose the production IMU propagation's exact per-interval
  `F` and complete `Q`, marginalize the uncertain but mean-immutable physical
  head into the retained future knots, and retain every cross-time covariance.
  Bias random-walk blocks are not projected out; they connect successive
  knot-specific bias states.
- [x] Use one fixed-prior information solve for every residual-time policy.
- [x] Keep state-derived covariance out of solve weights.
- [x] Provide sensor-only and gating-only state-uncertainty variants.
- [x] Apply the complete joint increment without clipping.
- [x] Commit only the posterior tail physical state, tail biases, and shared
  gravity marginal to `StateGroup`.
- [x] Log the signed requested and realized correction for every physical and
  nuisance block, split into LiDAR and prior-restoring contributions.
- [x] Log residual counts by knot interval, zero-Jacobian residuals,
  prior/LiDAR/posterior eigenvalue summaries, scan stop reason, and distance
  toward the pre-IMU stationary reference.
- [x] Add independent analytic Hermite and scalar point-to-plane Jacobian
  checks to the permanent joint-knot test.
- [x] Register every retained active test in CTest and remove fixtures for
  deleted spline APIs/modes.

## Diagnostic outputs

- `joint_knot_iterations.csv`: one solve row, including the LiDAR/prior split,
  interval support, and covariance/information eigenvalue summaries.
- `joint_knot_corrections.csv`: signed three-vector rows for every knot
  `theta/p/v` block, every knot's `bg/ba`, and shared `g`, with requested, LiDAR, prior, and
  realized values.
- `joint_knot_scan_summary.csv`: final correction from the post-IMU state,
  convergence reason, and before/after distance to the pre-IMU state.
- `joint_knot_states.csv`: absolute before/after state of every knot, including
  knot ID/time, immutable-head marker, knot-specific `bg/ba`, and gravity.
- `joint_knot_first_frame_state_chain.csv`: gravity-aligned pre-IMU,
  post-IMU, and current tail state without assuming attitude starts at zero.
- `joint_knot_first_frame_matrices.txt`: complete first-frame matrices and
  correction decomposition.
