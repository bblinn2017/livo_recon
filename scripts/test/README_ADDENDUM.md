# Addendum — rotation, channels, and the refinement

Three fixtures added alongside `test_spline.cpp` / `test_indirect.cpp`. No ROS,
PCL or catkin workspace; build them the same way as the other two. OpenCV
*headers* are needed, not the libraries -- `spline.h` pulls in
`utils/data/data_wrappers.h` for `Pose6D` and `ImuSample`, and that header
includes `<opencv2/opencv.hpp>` for `ImageData`. See the note in `README.md`
for the stub route if OpenCV is not on the include path.

```sh
OPENCV=$(pkg-config --cflags opencv4 2>/dev/null || echo -I/usr/include/opencv4)
for t in test_spline_rotation test_spline_refine test_spline_channels; do
  g++ -std=c++17 -O2 -Wall -Wextra -I include -I /usr/include/eigen3 $OPENCV \
      scripts/test/$t.cpp src/lio/spline.cpp -o /tmp/$t && /tmp/$t
done
```

All three build clean and pass against this tree (gcc 12, Eigen 3.4, OpenCV 4);
the only warnings are Eigen `-Wmaybe-uninitialized` false positives raised
inside the fixtures' own `Pose6D p{}` construction, none from `spline.cpp`.

## `test_spline_rotation.cpp` — the fixture the first suite was missing

`test_spline.cpp`'s rotation truth was `Exp(axis * theta(t))` with a **fixed
axis**. That is precisely the case where a cubic spline in one tangent chart is
an *exact* representation, so the original suite could not distinguish the
tangent parameterisation from the cumulative SO(3) one, and its 7.8e-8 rad/s
angular-velocity result was measuring nothing about the choice.

This fixture uses `R(t) = Exp(e_z α(t)) · Exp(e_x β(t))`, whose axis genuinely
rotates and whose body rate has an exact closed form, and sweeps the total
rotation swept per scan. Result at n_cp = 8, max error over the scan:

| chord | tangent rot | cumulative rot | tangent ω | cumulative ω |
|---|---|---|---|---|
| 8.5° | 1.24e-05 | 1.24e-05 | 6.05e-03 | 6.27e-03 |
| 33.8° | 7.17e-05 | 9.80e-05 | 2.94e-02 | 3.28e-02 |
| 128.3° | 1.09e-03 | 3.89e-03 | 4.54e-01 | 8.48e-01 |
| 165.7° | 7.58e-03 | 8.57e-02 | 5.50e+00 | 1.49e+01 |

**Tangent wins everywhere**, and cumulative is fully converged there —
`rot_fit_iters` 2, 4, 8, 16, 32 agree to five digits. Over one scan the chart
never approaches its |φ| → π failure, so the exactness of the linear least
squares beats the manifold-correctness of the parameterisation. The
literature's preference for cumulative is about *global* splines over long
windows; that argument does not transfer to a per-scan fit.

## `test_spline_channels.cpp` — the two channels want opposite n_cp

Same moving-axis truth, white noise injected at σ_a = 0.0200 and σ_g = 0.00200,
poses dead-reckoned from that same noisy IMU (the production coupling).

At 8.5° chord — which brackets eee_01's measured `rot_chord_deg` max of ~6°:

| n_cp | σ̂_a ratio | acf1_a | σ̂_g ratio | acf1_g |
|---|---|---|---|---|
| 4 | 1.27 | +0.268 | **39.07** | +0.565 |
| 6 | 1.12 | +0.112 | 5.92 | +0.293 |
| 8 | 1.17 | +0.049 | **1.08** | +0.086 |
| 10 | 1.33 | +0.114 | 1.32 | +0.201 |
| 12 | 0.93 | −0.256 | 2.29 | +0.208 |
| 16 | **11.56** | +0.305 | 17.91 | +0.090 |

At 33.8° chord the gyro column never reaches 1 at any n_cp: 181, 32.2, 4.67,
3.90, 6.65, 67.1.

The mechanism is that the two channels fail for opposite reasons.

- **Accelerometer** reads the *second* derivative of the position spline, so
  over-resolution amplifies noise by 1/Δ². It wants **low** n_cp, and is
  insensitive to rotation magnitude.
- **Gyro** reads the *first* derivative of the rotation spline, so
  under-resolution leaves representation error. It wants **high** n_cp, and its
  error scales with rotation magnitude.

So there is a *window*, not a plateau, it is narrow, and it **closes as rotation
increases**. `n_cp = 8` is the gyro optimum at low rotation by luck: it was
chosen from an accelerometer-only, fixed-axis bench where the gyro channel's
representation error was identically zero by construction.

For scale, the Allan gyro floor read off the bags is σ = 1.65e-4 to 7.39e-4
rad/s. At 8.5° chord the spline's own ω representation error is 6.3e-3 rad/s at
n_cp = 8 — **8–38× the noise it is trying to measure** — falling to 2.4e-4 at
n_cp = 16, i.e. only at the top of the range does the instrument become quieter
than its subject.

`AdaptiveQ`'s whiteness gate behaves correctly throughout: acf1_g is +0.565 at
n_cp = 4 and +0.296 to +0.546 at 33.8°, at or above the 0.35 `acf1_max`, so the
gyro estimate is *refused* rather than believed in exactly the cases where it
would be wrong. The gate makes the estimate unavailable, not correct.

**A ceiling nobody had noticed:** the fit shrinks n_cp to `n_samples − 1`, so at
a 10 Hz scan and a 200 Hz IMU it silently caps at ~19. A sweep asking for 20,
24 and 32 gets three identical cells. `spline_q.csv` now logs `n_cp_req`
alongside `n_cp` so this is visible.

## `test_spline_refine.cpp` — the refinement, and its guards

Asserts that `refineWithLidar()` recovers a known 2.7 cm intra-scan shape error
to 9.4e-14 m in three steps; that an oversized step is rejected *whole* with the
control points bit-unchanged; that the single-normal (corridor) case stays
bounded in the unconstrained directions because of the prior; and that
`lidar_refine_cp: false` is a hard no-op.
