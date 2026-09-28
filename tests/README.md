# Standalone validation for the scan spline and adaptive Q

This test deliberately does NOT need ROS, PCL, OpenCV libraries or a catkin
workspace. They compile the new numerical code against Eigen alone, so the
maths can be checked on any machine in a few seconds, independently of
whether the full node builds.

```sh
# from the repo root
g++ -std=c++17 -O2 -I include -I /usr/include/eigen3 \
    tests/test_spline.cpp src/lio/spline.cpp src/lio/adaptive_q.cpp \
    -o /tmp/test_spline && /tmp/test_spline

```

`spline.cpp` includes `utils/data/data_wrappers.h`, which pulls in OpenCV for
`ImageData`. If OpenCV headers are not on the include path, put a minimal
stand-in earlier on it — only `ImuSample`, `PointXYZT`, `PointXYZCov` and
`Pose6D` are actually used.

## test_spline — correctness and safety

Ground truth is an analytic trajectory (Lissajous position, fixed-axis
rotation at a time-varying rate) whose acceleration and body-frame angular
velocity are known in closed form, so every check compares against a number
that is right by construction rather than against another run of the same
code. It asserts:

- the uniform cubic basis is a partition of unity and both derivative sets
  sum to zero, with support inside the control-point range;
- the fit recovers position to ~1.7e-9 m, rotation to ~4.0e-10 rad,
  acceleration to ~2.8e-4 m/s² and body angular velocity to ~7.8e-8 rad/s;
- the IMU residual recovers injected white noise σ to within 20% across
  three decades of σ, on both channels;
- `AdaptiveQ` holds every safety property: an extreme measurement stays
  inside the bounded excursion, a correlated residual is refused
  (`not_white`), a sub-floor residual is refused rather than clamped
  (`below_floor`), warm-up applies nothing, the rate limit holds, and
  `enable: false` is a hard no-op.

## Joint-knot coupled estimator

The canonical coupled estimator is covered by `test_joint_knot_estimator.cpp`
and `tests/joint_knots/CMakeLists.txt`. It checks immutable-head structure,
Hermite/geodesic interpolation, analytic Hermite Jacobian blocks, an
independent scalar point-to-plane directional derivative, knot-specific bias
Markov blocks, full-production-Q prior assembly, tail-marginal equivalence,
measurement-time versus tail-time Jacobians, and the
LiDAR/prior correction decomposition. Build and run it through the normal
project configuration so it uses the same Release flags as production.
