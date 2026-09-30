# Tests

`test_spline` (scan spline and AdaptiveQ) was removed together with that code on 2026-09-30. `test_ds_csr` and
`test_bootstrap_observation_moments` are built through `tests/active/CMakeLists.txt`.

## Joint-knot coupled estimator

The canonical coupled estimator is covered by `test_joint_knot_estimator.cpp`
and `tests/joint_knots/CMakeLists.txt`. It checks immutable-head structure,
Hermite/geodesic interpolation, analytic Hermite Jacobian blocks, an
independent scalar point-to-plane directional derivative, knot-specific bias
Markov blocks, full-production-Q prior assembly, tail-marginal equivalence,
measurement-time versus tail-time Jacobians, and the
LiDAR/prior correction decomposition. Build and run it through the normal
project configuration so it uses the same Release flags as production.
