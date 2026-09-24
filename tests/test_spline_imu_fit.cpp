// Two new mechanisms, each checked against the property that justifies it.
//
//   reintegratePoses()      -- replaying the stored pose recurrence under a
//                              corrected bias must equal having integrated
//                              with that bias in the first place.  If it does
//                              not, the spline's shape is still wrong, just
//                              differently.
//   SplineOptions::imu_fit_w_{acc,gyr}
//                           -- zero weight must be a hard no-op, and non-zero
//                              weight must actually pull the spline toward the
//                              raw IMU (and, as the header warns, pull the
//                              AdaptiveQ estimate DOWN while doing it).
//
// Eigen + OpenCV headers only.
#include "livo_recon/lio/spline.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

using namespace livo_recon;

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
  if (!ok) ++failures;
}

// Analytic truth: Lissajous position, moving-axis rotation.
struct Truth {
  M3D rot(double t) const {
    return Exp(V3D(0,0,1) * (0.35*std::sin(24.0*t) + 0.8*t))
         * Exp(V3D(1,0,0) * (0.28*std::cos(31.0*t) + 0.6*t));
  }
  V3D accW(double t) const {
    return V3D(-0.9*2.1*2.1*std::sin(2.1*t),
               -0.6*1.7*1.7*std::cos(1.7*t),
               -0.25*3.3*3.3*std::sin(3.3*t));
  }
  V3D posW(double t) const {
    return V3D(0.9*std::sin(2.1*t)+0.3*t, 0.6*std::cos(1.7*t), 0.25*std::sin(3.3*t));
  }
  V3D velW(double t) const {
    return V3D(0.9*2.1*std::cos(2.1*t)+0.3, -0.6*1.7*std::sin(1.7*t), 0.25*3.3*std::cos(3.3*t));
  }
  V3D omega(double t) const {
    const double ad = 0.35*24.0*std::cos(24.0*t) + 0.8;
    const double bd = -0.28*31.0*std::sin(31.0*t) + 0.6;
    return Exp(V3D(1,0,0)*(0.28*std::cos(31.0*t)+0.6*t)).transpose()*V3D(0,0,1)*ad
         + V3D(1,0,0)*bd;
  }
};

// Mirror of ImuProc::propagate()'s dead-reckoning, bias-corrected, so the
// fixture can integrate the SAME stream under two different biases.
static std::vector<Pose6D> propagate(const std::vector<ImuSample>& imu,
                                     const M3D& R0, const V3D& p0, const V3D& v0,
                                     const V3D& ba, const V3D& bg, const V3D& g)
{
  std::vector<Pose6D> out;
  out.reserve(imu.size());
  M3D R = R0; V3D pos = p0, vel = v0;
  for (size_t k = 0; k + 1 < imu.size(); ++k)
  {
    const ImuSample& head = imu[k];
    const ImuSample& tail = imu[k+1];
    const double dt = tail.t - head.t, dt2 = dt*dt;
    const V3D angvel = 0.5*(head.gyro + tail.gyro) - bg;
    const V3D acc_wh = R * (head.acc - ba) + g;
    const M3D R_next = R * Exp(angvel, dt);
    const V3D acc_wt = R_next * (tail.acc - ba) + g;
    const V3D acc_avr = 0.5*(acc_wh + acc_wt);
    out.push_back(Pose6D{ head.t, acc_wh, acc_wt, angvel, vel, pos, R, dt });
    pos = pos + vel*dt + 0.5*acc_avr*dt2;
    vel = vel + acc_avr*dt;
    R = R_next;
  }
  return out;
}

static double maxPoseDiff(const std::vector<Pose6D>& a, const std::vector<Pose6D>& b,
                          double& rot_err)
{
  double dp = 0.0; rot_err = 0.0;
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) {
    dp = std::max(dp, (a[i].pos - b[i].pos).norm());
    rot_err = std::max(rot_err, Log(a[i].rot.transpose()*b[i].rot).norm());
  }
  return dp;
}

int main()
{
  const Truth T;
  const double t0 = 0.0, t1 = 0.1;
  const V3D g(0, 0, -9.81);
  const int N = 21;
  const double dt = (t1 - t0) / (N - 1);

  // Clean IMU stream carrying a TRUE bias, plus a little white noise so the
  // fit has something to smooth.
  const V3D ba_true(0.031, -0.017, 0.024);
  const V3D bg_true(0.0021, 0.0013, -0.0035);
  std::mt19937 rng(90210);
  std::normal_distribution<double> na(0.0, 0.02), ng(0.0, 0.002);

  std::vector<ImuSample> imu(N);
  for (int i = 0; i < N; ++i) {
    const double t = t0 + i*dt;
    imu[i].t = t;
    imu[i].acc  = T.rot(t).transpose()*(T.accW(t) - g) + ba_true
                + V3D(na(rng), na(rng), na(rng));
    imu[i].gyro = T.omega(t) + bg_true + V3D(ng(rng), ng(rng), ng(rng));
  }

  const M3D R0 = T.rot(t0); const V3D p0 = T.posW(t0), v0 = T.velW(t0);

  // ── reintegratePoses ────────────────────────────────────────────────────
  std::printf("=== reintegratePoses ===\n");

  // What propagate() actually produced this frame: the WRONG (pre-update)
  // bias.  What the ESIKF has converged to: the true one.
  const V3D ba_old = ba_true + V3D(0.004, -0.003, 0.005);
  const V3D bg_old = bg_true + V3D(-0.0006, 0.0009, 0.0004);

  const std::vector<Pose6D> poses_old  = propagate(imu, R0, p0, v0, ba_old,  bg_old,  g);
  const std::vector<Pose6D> poses_true = propagate(imu, R0, p0, v0, ba_true, bg_true, g);

  double rot_err = 0.0;
  const double stale_dp = maxPoseDiff(poses_old, poses_true, rot_err);
  std::printf("  stale-bias shape error over %.0f ms: %.3e m, %.3e rad\n",
              (t1-t0)*1e3, stale_dp, rot_err);

  std::vector<Pose6D> replayed;
  const bool ran = reintegratePoses(poses_old,
                                    ba_true - ba_old, bg_true - bg_old, V3D::Zero(),
                                    g, 1e-9, 1e-9, replayed);
  check(ran, "a non-zero correction actually replays");

  double rot_after = 0.0;
  const double dp_after = maxPoseDiff(replayed, poses_true, rot_after);
  std::printf("  after replay: %.3e m, %.3e rad\n", dp_after, rot_after);
  check(dp_after < 1e-12 && rot_after < 1e-12,
        "replay reproduces integrating with the corrected bias from the start");
  check(replayed.size() == poses_old.size(), "replay preserves the sample count");

  std::vector<Pose6D> untouched;
  const bool ran0 = reintegratePoses(poses_old, V3D::Zero(), V3D::Zero(), V3D::Zero(),
                                     g, 1e-9, 1e-9, untouched);
  check(!ran0 && untouched.empty(), "a zero correction is a hard no-op, not a round-trip");

  // Gravity moves too, and it moves the shape the same way an accel bias does.
  std::vector<Pose6D> gshift;
  const V3D dg(0.0, 0.0, 0.01);
  check(reintegratePoses(poses_old, V3D::Zero(), V3D::Zero(), dg, g, 1e-9, 1e-9, gshift),
        "a gravity correction replays");
  double rg = 0.0;
  const double dpg = maxPoseDiff(gshift, poses_old, rg);
  const double predicted = 0.5 * dg.norm() * (t1-t0) * (t1-t0);
  std::printf("  gravity shift %.3f m/s^2 -> %.3e m (predicted ~%.3e m)\n",
              dg.norm(), dpg, predicted);
  check(dpg > 0.2*predicted && dpg < 5.0*predicted,
        "position shift from a gravity delta is the predicted 0.5*dg*T^2 order");

  // ── imu_fit_w_{acc,gyr} ─────────────────────────────────────────────────
  std::printf("\n=== raw-IMU term in the fit ===\n");

  SplineImuFitData ifd;
  ifd.samples  = &imu;
  ifd.bias_acc = ba_true;
  ifd.bias_gyr = bg_true;
  ifd.gravity  = g;

  SplineOptions o;
  o.n_control_points = 8;
  o.enable = true;

  ScanSpline s_off, s_null, s_acc, s_gyr;
  check(s_off.fit(poses_true, t0, t1, o), "baseline fit succeeds");
  check(s_null.fit(poses_true, t0, t1, o, &ifd), "fit with imu data but zero weight succeeds");

  double worst = 0.0;
  for (int i = 0; i <= 40; ++i) {
    const double t = t0 + (t1-t0)*i/40.0;
    worst = std::max(worst, (s_off.posAt(t) - s_null.posAt(t)).norm());
    worst = std::max(worst, (s_off.accAt(t) - s_null.accAt(t)).norm());
    worst = std::max(worst, (s_off.omegaBodyAt(t) - s_null.omegaBodyAt(t)).norm());
  }
  check(worst == 0.0, "zero weight is BIT-identical to not passing IMU data at all");

  // Residual against the raw stream, the thing the weight is supposed to shrink.
  auto imuRes = [&](const ScanSpline& sp, double& ra, double& rw) {
    ra = 0.0; rw = 0.0; int n = 0;
    for (const auto& sm : imu) {
      if (sm.t < sp.t0() || sm.t > sp.t1()) continue;
      ra += (sp.rotAt(sm.t).transpose()*(sp.accAt(sm.t) - g) + ba_true - sm.acc).squaredNorm();
      rw += (sp.omegaBodyAt(sm.t) + bg_true - sm.gyro).squaredNorm();
      ++n;
    }
    if (n) { ra = std::sqrt(ra/n); rw = std::sqrt(rw/n); }
  };

  double ra0, rw0; imuRes(s_off, ra0, rw0);
  std::printf("  w=0            acc %.4e  gyr %.4e\n", ra0, rw0);

  bool acc_monotone = true, gyr_monotone = true;
  double prev_a = ra0, prev_w = rw0;
  for (double w : {0.1, 1.0, 10.0}) {
    SplineOptions oa = o; oa.imu_fit_w_acc = w;
    SplineOptions og = o; og.imu_fit_w_gyr = w;
    ScanSpline sa, sg;
    sa.fit(poses_true, t0, t1, oa, &ifd);
    sg.fit(poses_true, t0, t1, og, &ifd);
    double a1, w1, a2, w2;
    imuRes(sa, a1, w1); imuRes(sg, a2, w2);
    std::printf("  w_acc=%-5.1f    acc %.4e            w_gyr=%-5.1f  gyr %.4e\n",
                w, a1, w, w2);
    if (!(a1 < prev_a + 1e-15)) acc_monotone = false;
    if (!(w2 < prev_w + 1e-15)) gyr_monotone = false;
    prev_a = a1; prev_w = w2;
  }
  check(acc_monotone, "raising w_acc monotonically shrinks the accel residual");
  check(gyr_monotone, "raising w_gyr monotonically shrinks the gyro residual");

  // The header's warning, made falsifiable: pulling the spline onto the raw
  // IMU means the spline absorbs some of the noise AdaptiveQ is about to be
  // asked to report, so sigma_a_hat must move DOWN.  Whether down is an
  // improvement depends on which side the baseline sits, and that differs by
  // fixture -- test_indirect dead-reckons its poses from the same noisy IMU
  // and lands ~15% LOW, this one integrates from a clean initial condition on
  // a moving-axis rotation and lands HIGH because representation error
  // inflates the accel residual.  So the assertion here is on the DIRECTION
  // only.  Do not read "closer to truth" off this fixture as a general result.
  SplineOptions ohi = o; ohi.imu_fit_w_acc = 10.0; ohi.imu_fit_w_gyr = 10.0;
  ScanSpline shi; shi.fit(poses_true, t0, t1, ohi, &ifd);
  const auto st0 = computeSplineImuResidual(s_off, imu, ba_true, bg_true, g);
  const auto sthi = computeSplineImuResidual(shi,  imu, ba_true, bg_true, g);
  std::printf("  sigma_a_hat  w=0 %.4e   w=10 %.4e   (true 2.000e-02)\n",
              std::sqrt(st0.cov_acc), std::sqrt(sthi.cov_acc));
  check(sthi.cov_acc < st0.cov_acc,
        "high IMU weight moves sigma_a_hat DOWN (direction only -- see comment)");

  std::printf("\n%s (%d failures)\n", failures ? "FAILURES" : "ALL CHECKS PASSED", failures);
  return failures ? 1 : 0;
}
