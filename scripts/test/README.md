# Standalone validation for the scan spline and adaptive Q

These two tests deliberately do NOT need ROS, PCL, OpenCV or a catkin
workspace. They compile the new numerical code against Eigen alone, so the
maths can be checked on any machine in a few seconds, independently of
whether the full node builds.

```sh
# from the repo root
g++ -std=c++17 -O2 -I include -I /usr/include/eigen3 \
    scripts/test/test_spline.cpp src/lio/spline.cpp src/lio/adaptive_q.cpp \
    -o /tmp/test_spline && /tmp/test_spline

g++ -std=c++17 -O2 -I include -I /usr/include/eigen3 \
    scripts/test/test_indirect.cpp src/lio/spline.cpp \
    -o /tmp/test_indirect && /tmp/test_indirect
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
- `anchorTo()` lands exactly on the target pose and preserves the fitted
  shape (|acc| drift ~1e-12);
- `AdaptiveQ` holds every safety property: an extreme measurement stays
  inside the bounded excursion, a correlated residual is refused
  (`not_white`), a sub-floor residual is refused rather than clamped
  (`below_floor`), warm-up applies nothing, the rate limit holds, and
  `enable: false` is a hard no-op.

## test_indirect — the production coupling, and what it corrected

`test_spline`'s control-point sweep fits the spline to *noiseless* poses,
which exercises only the direct path. In the real system `mg.poses` are
IMU-integrated, so IMU noise reaches the fit indirectly. This test
dead-reckons the poses from the same noisy IMU the residual is measured
against, exactly as `ImuProc::propagate()` does.

It is the test that corrected a wrong expectation. The naive argument says
that as the knot rate approaches the IMU rate the spline interpolates the IMU
and the residual collapses to zero. That is wrong here, because the quantity
compared against the accelerometer is a **second derivative**: a noise-chasing
interpolant is amplified by 1/Δ², so the estimate **explodes** instead. At
true σ_a = 0.0200:

| n_cp | σ̂_a | ratio | acf1_a | fit residual |
|---|---|---|---|---|
| 4 | 0.017484 | 0.87 | +0.010 | 1.09e-06 |
| 6 | 0.017174 | 0.86 | −0.064 | 3.90e-07 |
| 8 | 0.016115 | 0.81 | −0.158 | 1.60e-07 |
| 10 | 0.017572 | 0.88 | −0.073 | 1.05e-07 |
| 14 | 0.095285 | 4.76 | +0.307 | 3.99e-08 |
| 18 | 0.268005 | 13.40 | +0.227 | 8.22e-09 |

The fit getting monotonically better while the estimate gets catastrophically
worse is the signature. Two consequences, both now encoded in the defaults:

1. There is a flat, usable plateau at n_cp 4–10 — the estimate sits at
   0.81–0.96 of truth with |acf1| ≤ 0.23. `n_control_points: 8` is its
   middle.
2. **Whiteness, not the floor, is the anchor that binds.** The floor guards
   the under-resolved and genuinely-quiet-sensor cases; it does nothing on
   the over-resolved side, where the estimate is too *large*. `acf1` moving
   from ~0 to +0.31 / −0.51 across the cliff is what catches it, which is why
   `AdaptiveQ::update()` checks whiteness first and refuses on it outright.

Re-run this on real bags before trusting any `n_control_points` choice: the
plateau's width is a property of the motion and the IMU rate, not a constant.
