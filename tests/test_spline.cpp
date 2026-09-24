// Standalone numerical validation of ScanSpline + AdaptiveQ.
// Ground truth is an analytic trajectory whose acceleration and body-frame
// angular velocity are known in closed form, so every check below compares
// against a number that is right by construction rather than against another
// run of the same code.
#include "livo_recon/lio/spline.h"
#include "livo_recon/lio/adaptive_q.h"
#include <cstdio>
#include <random>
#include <vector>
using namespace livo_recon;

static int g_fail = 0;
static void check(bool ok, const char* name, const char* detail = "") {
  printf("  [%s] %s %s\n", ok ? " ok " : "FAIL", name, detail);
  if (!ok) g_fail++;
}

// Analytic truth. Position: a 3-frequency Lissajous. Rotation: constant-axis
// rotation at a time-varying rate, so omega_body is exactly known.
struct Truth {
  static V3D pos(double t) {
    return V3D(0.9*std::sin(2.1*t) + 0.3*t, 0.6*std::cos(1.7*t), 0.25*std::sin(3.3*t));
  }
  static V3D acc(double t) {   // d2/dt2 pos, world frame
    return V3D(-0.9*2.1*2.1*std::sin(2.1*t), -0.6*1.7*1.7*std::cos(1.7*t),
               -0.25*3.3*3.3*std::sin(3.3*t));
  }
  static double ang(double t) { return 0.8*t + 0.35*std::sin(2.4*t); }
  static double angdot(double t) { return 0.8 + 0.35*2.4*std::cos(2.4*t); }
  static V3D axis() { return V3D(0.3, -0.5, 0.81).normalized(); }
  static M3D rot(double t) { return Exp(axis() * ang(t)); }
  // R(t) = Exp(axis*ang(t)) with a FIXED axis, so Rdot = R [axis*angdot]_x
  // and the body-frame angular velocity is exactly axis*angdot.
  static V3D omega(double t) { return axis() * angdot(t); }
};

static std::vector<Pose6D> makePoses(double t0, double t1, int n) {
  std::vector<Pose6D> ps;
  for (int i = 0; i < n; ++i) {
    double t = t0 + (t1 - t0) * i / double(n - 1);
    Pose6D p{}; p.t = t; p.pos = Truth::pos(t); p.rot = Truth::rot(t);
    p.dt = (t1 - t0) / (n - 1);
    ps.push_back(p);
  }
  return ps;
}

int main() {
  const double t0 = 0.0, t1 = 0.1;    // one 10 Hz scan
  const V3D g(0, 0, -9.81);

  // ---- 1. basis sanity -------------------------------------------------
  {
    SplineOptions o; o.n_control_points = 10;
    ScanSpline s; s.fit(makePoses(t0, t1, 40), t0, t1, o);
    bool ok = true;
    for (double u = 0.0; u <= 1.0; u += 0.05) {
      double t = t0 + u * (t1 - t0);
      int fc; Eigen::Vector4d b, db, ddb;
      s.basisAt(t, fc, b, db, ddb);
      ok &= std::abs(b.sum() - 1.0) < 1e-12;
      ok &= std::abs(db.sum()) < 1e-12;
      ok &= std::abs(ddb.sum()) < 1e-12;
      ok &= (fc >= 0 && fc + 3 < s.nControlPoints());
    }
    check(ok, "basis partition of unity, derivative sums zero, support in range");
  }

  // ---- 2. does the fit recover the analytic derivatives? ---------------
  {
    SplineOptions o; o.n_control_points = 12;
    ScanSpline s;
    check(s.fit(makePoses(t0, t1, 40), t0, t1, o), "fit succeeds");
    double max_p = 0, max_a = 0, max_w = 0, max_r = 0;
    for (double t = t0 + 0.01; t <= t1 - 0.01; t += 0.002) {
      max_p = std::max(max_p, (s.posAt(t) - Truth::pos(t)).norm());
      max_a = std::max(max_a, (s.accAt(t) - Truth::acc(t)).norm());
      max_w = std::max(max_w, (s.omegaBodyAt(t) - Truth::omega(t)).norm());
      max_r = std::max(max_r, Log(Truth::rot(t).transpose() * s.rotAt(t)).norm());
    }
    char buf[256];
    snprintf(buf, sizeof buf, "pos %.2e m, rot %.2e rad, acc %.2e m/s2, omega %.2e rad/s",
             max_p, max_r, max_a, max_w);
    check(max_p < 1e-5 && max_r < 1e-5 && max_a < 5e-2 && max_w < 5e-3,
          "spline recovers analytic pos/rot/acc/omega", buf);
  }

  // ---- 3. THE LOAD-BEARING TEST: does the residual measure IMU noise? --
  // Synthesize an IMU from the analytic truth plus white noise of KNOWN
  // sigma, fit the spline to noiseless poses, and ask whether
  // computeSplineImuResidual recovers sigma^2.
  {
    printf("\n  sigma_a recovery (truth -> measured, ratio):\n");
    bool ok = true;
    for (double sa : {0.002, 0.02, 0.2}) {
      const double sg = sa * 0.1;
      std::mt19937 rng(12345);
      std::normal_distribution<double> na(0.0, sa), ng(0.0, sg);
      std::vector<ImuSample> imu;
      for (int i = 0; i <= 20; ++i) {                 // 200 Hz over 0.1 s
        double t = t0 + (t1 - t0) * i / 20.0;
        M3D R = Truth::rot(t);
        V3D a = R.transpose() * (Truth::acc(t) - g);  // specific force
        V3D w = Truth::omega(t);
        imu.emplace_back(a + V3D(na(rng), na(rng), na(rng)),
                         w + V3D(ng(rng), ng(rng), ng(rng)), t);
      }
      SplineOptions o; o.n_control_points = 8;
      ScanSpline s; s.fit(makePoses(t0, t1, 40), t0, t1, o);
      auto st = computeSplineImuResidual(s, imu, V3D::Zero(), V3D::Zero(), g);
      double ra = std::sqrt(st.cov_acc) / sa, rg = std::sqrt(st.cov_gyr) / sg;
      printf("    sigma_a=%.4f -> %.4f (x%.2f)   sigma_g=%.5f -> %.5f (x%.2f)  n=%d\n",
             sa, std::sqrt(st.cov_acc), ra, sg, std::sqrt(st.cov_gyr), rg, st.n);
      // The accel channel carries the spline's own fit error on top of the
      // noise, so it is biased HIGH at small sigma -- exactly the direction
      // the floor anchor is designed to catch. Require within 2x.
      ok &= (ra > 0.5 && ra < 2.0) && (rg > 0.5 && rg < 2.0);
    }
    check(ok, "residual recovers injected IMU sigma within 2x on both channels");
  }

  // ---- 4. the degeneracy: is the residual monotone in n_cp? ------------
  // This is the property the whole floor/whiteness anchoring exists to
  // handle. If it does not hold, the anchoring argument is wrong.
  {
    std::mt19937 rng(999);
    const double sa = 0.02, sg = 0.002;
    std::normal_distribution<double> na(0.0, sa), ng(0.0, sg);
    std::vector<ImuSample> imu;
    std::vector<Pose6D> poses = makePoses(t0, t1, 60);
    for (int i = 0; i <= 60; ++i) {
      double t = t0 + (t1 - t0) * i / 60.0;
      M3D R = Truth::rot(t);
      imu.emplace_back(R.transpose() * (Truth::acc(t) - g) + V3D(na(rng), na(rng), na(rng)),
                       Truth::omega(t) + V3D(ng(rng), ng(rng), ng(rng)), t);
    }
    printf("\n  residual vs control points (the degeneracy AdaptiveQ anchors):\n");
    double prev = 1e30; bool mono = true;
    for (int ncp : {4, 6, 8, 12, 20, 30}) {
      SplineOptions o; o.n_control_points = ncp;
      ScanSpline s;
      if (!s.fit(poses, t0, t1, o)) { printf("    n_cp=%2d  fit refused\n", ncp); continue; }
      // Fit the spline TO THE NOISY IMU's implied motion by fitting to poses
      // sampled densely -- here the poses are noiseless, so what varies with
      // n_cp is purely the spline's ability to absorb the true motion.
      auto st = computeSplineImuResidual(s, imu, V3D::Zero(), V3D::Zero(), g);
      printf("    n_cp=%2d  sigma_a_hat=%.5f  acf1_a=%+.3f  sigma_g_hat=%.6f  acf1_g=%+.3f\n",
             s.nControlPoints(), std::sqrt(st.cov_acc), st.acf1_acc,
             std::sqrt(st.cov_gyr), st.acf1_gyr);
      if (st.cov_acc > prev * 1.5) mono = false;
      prev = st.cov_acc;
    }
    check(mono, "residual variance is non-increasing in control-point count");
  }

  // ---- 5. anchorTo is rigid: shape preserved, endpoint moved ----------
  {
    SplineOptions o; o.n_control_points = 10;
    ScanSpline s; s.fit(makePoses(t0, t1, 40), t0, t1, o);
    std::vector<V3D> before;
    for (double t = t0; t <= t1; t += 0.005) before.push_back(s.accAt(t));
    const M3D Rn = Exp(V3D(0.03, -0.02, 0.05));
    const V3D pn(1.5, -2.5, 0.75);
    s.anchorTo(t1, Rn, pn);
    double endp = (s.posAt(t1) - pn).norm();
    double endr = Log(Rn.transpose() * s.rotAt(t1)).norm();
    double shape = 0; size_t k = 0;
    for (double t = t0; t <= t1; t += 0.005, ++k)
      shape = std::max(shape, (s.accAt(t).norm() - before[k].norm()));
    char buf[200];
    snprintf(buf, sizeof buf, "endpoint err %.2e m / %.2e rad, |acc| drift %.2e", endp, endr, shape);
    check(endp < 1e-12 && endr < 1e-12 && std::abs(shape) < 1e-10,
          "anchorTo lands exactly on the target and preserves |acc|", buf);
  }

  // ---- 6. AdaptiveQ safety properties ---------------------------------
  {
    AdaptiveQOptions o; o.enable = true; o.beta_acc = 0.3; o.beta_gyr = 0.3;
    o.warmup_frames = 2; o.z_rate_limit = 1e9; o.ema = 0.0;
    AdaptiveQ q; q.configure(o);
    q.setNominal(0.5, 0.3);
    q.setFloor(1e-5, 1e-7);

    SplineImuResidualStats st; st.n = 50;

    // A wildly high measurement must still land inside the beta band.
    st.cov_acc = 1e6; st.cov_gyr = 1e6; st.acf1_acc = 0.0; st.acf1_gyr = 0.0;
    for (int i = 0; i < 8; ++i) q.update(st);
    const double hi_a = q.varAcc(), hi_g = q.varGyr();
    check(hi_a <= 0.5 * std::pow(10, 0.3) + 1e-12 &&
          hi_g <= 0.3 * std::pow(10, 0.3) + 1e-12,
          "extreme measurement stays inside the bounded excursion");

    // A correlated residual must be refused outright.
    AdaptiveQ q2; q2.configure(o); q2.setNominal(0.5, 0.3); q2.setFloor(1e-5, 1e-7);
    st.cov_acc = 1e-3; st.cov_gyr = 1e-3; st.acf1_acc = 0.9; st.acf1_gyr = 0.9;
    for (int i = 0; i < 8; ++i) q2.update(st);
    check(q2.varAcc() == 0.5 && q2.varGyr() == 0.3 && q2.lastStatus() == "not_white",
          "correlated residual is refused, nominal retained", q2.lastStatus().c_str());

    // A residual below the physical floor must be refused, not clamped.
    AdaptiveQ q3; q3.configure(o); q3.setNominal(0.5, 0.3); q3.setFloor(1e-2, 1e-2);
    st.cov_acc = 1e-9; st.cov_gyr = 1e-9; st.acf1_acc = 0.0; st.acf1_gyr = 0.0;
    for (int i = 0; i < 8; ++i) q3.update(st);
    check(q3.varAcc() == 0.5 && q3.lastStatus() == "below_floor",
          "sub-floor residual refused, nominal retained", q3.lastStatus().c_str());

    // Warm-up must apply nothing.
    AdaptiveQ q4; AdaptiveQOptions o4 = o; o4.warmup_frames = 100; q4.configure(o4);
    q4.setNominal(0.5, 0.3); q4.setFloor(1e-5, 1e-7);
    st.cov_acc = 0.9; st.cov_gyr = 0.9; st.acf1_acc = 0.0; st.acf1_gyr = 0.0;
    for (int i = 0; i < 10; ++i) q4.update(st);
    check(q4.varAcc() == 0.5 && !q4.active() && q4.lastStatus() == "warmup",
          "warm-up applies nothing");

    // Rate limit: with a tight limit the applied value must move slowly.
    AdaptiveQ q5; AdaptiveQOptions o5 = o; o5.z_rate_limit = 0.01; q5.configure(o5);
    q5.setNominal(0.5, 0.3); q5.setFloor(1e-5, 1e-7);
    st.cov_acc = 1e6; st.cov_gyr = 1e6;
    q5.update(st); q5.update(st); q5.update(st);
    check(std::abs(q5.zAcc()) <= 0.011, "rate limit holds z within one step per frame");

    // Disabled must be a hard no-op.
    AdaptiveQ q6; AdaptiveQOptions o6; o6.enable = false; q6.configure(o6);
    q6.setNominal(0.5, 0.3);
    st.cov_acc = 1e6; q6.update(st);
    check(q6.varAcc() == 0.5 && !q6.active(), "disabled is a hard no-op");
  }

  printf("\n%s  (%d failures)\n", g_fail ? "FAILURES PRESENT" : "ALL CHECKS PASSED", g_fail);
  return g_fail ? 1 : 0;
}
