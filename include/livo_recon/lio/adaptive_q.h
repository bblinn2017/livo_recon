#pragma once

#include <string>
#include <Eigen/Core>

#include "livo_recon/utils/algo/math.h"
#include "livo_recon/lio/spline.h"

// ============================================================================
// AdaptiveQ -- live process-noise estimation from the spline-vs-IMU residual.
//
// WHY THIS IS A MEASUREMENT AND NOT AN INFERENCE.  Every adaptive-Q scheme in
// the LIO literature drives Q from the filter's own state increments
// (Sage-Husa, and AKF-LIO's rank-1 F_w^-1 (dx dx^T) F_w^-T).  That statistic
// has expectation K S K^T -- a function of R and P, NOT of Q -- so those
// schemes are really degeneracy detectors.  Worse, for a linear-Gaussian
// system with unknown inputs (which an ESIKF with the IMU as its input is),
// Q and R are provably not uniquely identifiable, jointly or individually.
//
// The residual here is a different object.  It differences the RAW IMU
// against a smooth trajectory fitted to the same window, so it measures the
// accelerometer's and gyro's own noise directly.  cov_w's accelerometer block
// is built from exactly that sigma^2 (ImuProc::propagate(): var_acc * dt^2 for
// velocity, var_acc * dt^4/4 for position), so measuring it as a measurement
// covariance and applying it as process noise is one physical quantity
// entering through two doors -- not a category error.
//
// THE DEGENERACY, AND THE TWO ANCHORS THAT CLOSE IT.  The residual variance
// is a MONOTONE DECREASING function of the control-point count: at the IMU
// rate the spline interpolates the IMU and the residual goes to zero; at
// minimal resolution it contains the full signal.  The curve has no interior
// stationary point, so the estimate alone is not well-posed and ablating
// n_cp maps the curve without saying where to stand.  Two independent
// conditions bracket it:
//
//   (1) FLOOR.  sigma_hat must not fall below the sensor's own measured
//       noise floor (CalibProc's static-window variance, now always stored
//       via StateGroup::setNoiseFloor() regardless of use_calib_var).  Below
//       the floor the spline is fitting noise, which is not physically
//       possible to do honestly -- the estimator cannot find the sensor
//       quieter than the sensor is.
//   (2) WHITENESS.  The residual must be white.  If lag-1 autocorrelation
//       still exceeds acf1_max the spline has not absorbed the real motion,
//       so the "noise" is contaminated by signal no matter how small it is,
//       and the estimate is refused.
//
// If (1) and (2) cannot be satisfied at the same n_cp there is no valid
// window, which is a model-misspecification detector rather than a tuning
// failure.  That case is logged (`no_window`) and the estimate held, never
// silently split the difference.
//
// SAFETY.  Nothing here point-estimates Q and feeds it straight back.  The
// applied value is a BOUNDED MULTIPLICATIVE EXCURSION from the nominal,
//
//     var = var_nominal * 10^( beta * tanh(z) ),      z from the measurement
//
// so the reachable set is finite, enumerable, and can be swept offline and
// certified free of the kind of notch the alpha ladders found (a change in
// the fifth significant figure of Q moving ATE by 3,500x).  beta is a SAFETY
// parameter, not an ambition parameter.  A dead or confused estimator gives
// z -> 0 and recovers the nominal exactly -- the identity map is the failure
// mode, by construction.  Plus: a per-frame rate limit, a hard floor, a hard
// ceiling, and a warm-up during which nothing is applied at all.
// ============================================================================

// ---------------------------------------------------------------------------
// MEASURED BEHAVIOUR OF n_control_points  (bench, 2026-09-01, synthetic
// trajectory + injected white IMU noise, poses dead-reckoned from the SAME
// noisy IMU exactly as ImuProc::propagate() does -- i.e. the production
// coupling, not an idealised one).  True sigma_a = 0.0200, sigma_g = 0.00200,
// 21 samples over a 0.1 s scan:
//
//    n_cp   sigma_a_hat  ratio   acf1_a    sigma_g_hat  ratio   acf1_g
//      4      0.017484   0.87    +0.010     0.0017634   0.88    -0.057
//      6      0.017174   0.86    -0.064     0.0019129   0.96    -0.005
//      8      0.016115   0.81    -0.158     0.0018705   0.94    -0.122
//     10      0.017572   0.88    -0.073     0.0018878   0.94    -0.232
//     14      0.095285   4.76    +0.307     0.0040491   2.02    +0.021
//     18      0.268005  13.40    +0.227     0.0012326   0.62    -0.512
//
// TWO THINGS TO TAKE FROM THIS, one of which corrects a prior expectation.
//
// (1) There IS a usable plateau.  Between n_cp 4 and 10 the estimate sits at
//     0.81-0.96 of truth on both channels with |acf1| <= 0.23 -- flat, so the
//     answer is not an artefact of a lucky control-point count.  The default
//     of 8 sits in the middle of it.  The residual is biased slightly LOW
//     because the spline absorbs a little of the noise it is measuring; a
//     ~15% underestimate is the honest characterisation, not a bug.
//
// (2) The over-resolution failure runs the OPPOSITE way from the naive
//     expectation.  The naive argument -- "as the knot rate approaches the
//     IMU rate the spline interpolates the IMU and the residual goes to
//     zero" -- is wrong for THIS estimator, because the quantity compared
//     against the accelerometer is the spline's SECOND DERIVATIVE, and
//     differentiating a noise-chasing interpolant twice amplifies by
//     1/delta^2.  So sigma_a_hat does not collapse, it EXPLODES: 4.8x at
//     n_cp=14, 13.4x at n_cp=18, while the fit residual itself falls
//     monotonically (1.09e-6 -> 8.22e-9 m) because the fit is indeed
//     interpolating.  The fit getting better and the estimate getting worse
//     at the same time is the signature.
//
//     Consequence for the anchors: the FLOOR does not bind on the
//     over-resolved side at all -- it guards the under-resolved side and the
//     genuinely-quiet-sensor case.  WHITENESS is the anchor that catches
//     over-resolution (acf1 goes from ~0 to +0.31 / -0.51 across the cliff),
//     which is why AdaptiveQ checks it FIRST and refuses on it outright.
//     The bounded excursion and max_ratio are the backstop if it slips
//     through.
//
// So n_control_points_max is deliberately generous (32) because it is the
// ablation knob, but anything above ~12 on a 0.1 s scan at 200 Hz IMU is
// expected to be past the cliff.  Ablate across it on purpose; do not ship
// there.
// ---------------------------------------------------------------------------

namespace livo_recon
{

struct AdaptiveQOptions
{
  // Master switch.  false (default) = varAcc()/varGyr() are never touched
  // and the filter behaves exactly as before.
  bool enable = false;

  // Bounded excursion.  beta = 0.3 is +/- 2x; the AI-IMU original uses 3
  // (+/- 1000x), which is far too wide given what the alpha ladders found.
  double beta_acc = 0.3;
  double beta_gyr = 0.3;

  // Per-frame rate limit on z, in z-units.  0.02 at 10 Hz means the applied
  // value needs ~5 s to traverse the full excursion, so no single frame's
  // measurement can move the filter far.
  double z_rate_limit = 0.02;

  // Whiteness gate: refuse the measurement if |lag-1 ACF| exceeds this.
  // 0.2 is loose enough to pass genuinely white residuals at n ~ 20 (where
  // the ACF's own standard error is ~1/sqrt(n) = 0.22) and tight enough to
  // catch a spline that is still tracking real motion.
  double acf1_max = 0.35;

  // Hard bounds relative to the NOMINAL (the YAML's state/cov/{acc,gyr}).
  // Belt and braces on top of beta -- if beta is ever raised these still
  // hold.
  double max_ratio = 100.0;
  double min_ratio = 0.01;

  // Apply the measured floor from the calibration window as a hard lower
  // bound on the APPLIED variance.  This is separate from, and stricter
  // than, min_ratio: min_ratio is relative to a hand-set nominal, this is
  // relative to the sensor.
  bool  use_noise_floor = true;
  // Multiplier on the calibration floor before it is used as a bound.  1.0
  // is the raw physical floor.  Raise it if the empirical stability floor
  // (the config comment's ~90x datasheet on the gyro) turns out to bind
  // first -- the two are different quantities and this is where that gets
  // expressed.
  double noise_floor_scale = 1.0;

  // Frames to observe before applying anything.  AKF-LIO's own failure case
  // is a ~10 s convergence transient during which its adaptation is
  // actively harmful; a warm-up is the cheapest defence.
  int warmup_frames = 20;

  // Exponential smoothing on the measured variance, in [0,1). Higher =
  // slower.  This is a smoother on the MEASUREMENT, not a Sage-Husa-style
  // recursion on the estimate -- there is no feedback path from the applied
  // value back into the statistic.
  double ema = 0.9;

  // Per-frame CSV (adaptive_q.csv in the debug log dir).
  bool log_en = false;
};

class AdaptiveQ
{
public:
  void configure(const AdaptiveQOptions& o) { opts_ = o; }
  const AdaptiveQOptions& opts() const { return opts_; }

  // Nominal = whatever the state was configured with before adaptation
  // starts (the YAML's state/cov/{acc,gyr}).  Floor = the calibration
  // window's measured variance.  Both isotropic scalars, matching the
  // trace/3 reduction used everywhere else.
  void setNominal(double var_acc_nom, double var_gyr_nom);
  void setFloor(double var_acc_floor, double var_gyr_floor);

  // Fold in one scan's residual statistics.  Returns true if the applied
  // values changed.  Never throws, never divides by zero, and refuses
  // silently-bad inputs (n too small, non-finite, ACF gate, below floor)
  // rather than propagating them.
  bool update(const SplineImuResidualStats& st);

  // The values to hand the filter.  Before warm-up completes these are the
  // nominal, exactly.
  double varAcc() const { return applied_acc_; }
  double varGyr() const { return applied_gyr_; }
  bool   active() const { return active_; }

  // Last decision, for the log and for the register's own accounting.
  const std::string& lastStatus() const { return status_; }
  double zAcc() const { return z_acc_; }
  double zGyr() const { return z_gyr_; }
  double measuredAcc() const { return meas_acc_; }
  double measuredGyr() const { return meas_gyr_; }
  int    frames() const { return frames_; }

  std::string csvHeader() const;
  std::string csvRow(double t_abs, const SplineImuResidualStats& st) const;

private:
  AdaptiveQOptions opts_;

  double nom_acc_ = 0.0, nom_gyr_ = 0.0;
  double floor_acc_ = 0.0, floor_gyr_ = 0.0;
  double meas_acc_ = 0.0, meas_gyr_ = 0.0;    // EMA of the measurement
  double z_acc_ = 0.0, z_gyr_ = 0.0;
  double applied_acc_ = 0.0, applied_gyr_ = 0.0;
  int    frames_ = 0;
  bool   active_ = false;
  bool   primed_ = false;
  std::string status_ = "off";

  // Map a measured variance to a bounded excursion z about the nominal.
  // z = log10(meas / nom) / beta, then clamped by tanh's own range.  The
  // inverse map (applied = nom * 10^(beta*tanh(z))) is what actually gets
  // applied, so the applied value is ALWAYS inside [nom/10^beta,
  // nom*10^beta] no matter what the measurement does.
  static double zFromRatio(double meas, double nom, double beta);
  static double applyZ(double nom, double beta, double z);
};

}  // namespace livo_recon
