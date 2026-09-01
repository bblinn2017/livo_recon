// Follow-up to test 4. In the real system mg.poses are IMU-INTEGRATED, so
// IMU noise reaches the spline fit indirectly through the poses. The main
// test fed analytic (noiseless) poses, which exercises only the direct path
// -- and the direct path is absent by construction here, because the IMU is
// not in the fit cost. This test drives the poses by integrating the SAME
// noisy IMU the residual is measured against, which is the production
// coupling, and asks whether sigma_hat still collapses as n_cp rises.
#include "livo_recon/lio/spline.h"
#include <cstdio>
#include <random>
using namespace livo_recon;

int main() {
  const double t0 = 0.0, t1 = 0.1;
  const V3D g(0, 0, -9.81);
  const V3D axis = V3D(0.3, -0.5, 0.81).normalized();
  auto ang    = [](double t){ return 0.8*t + 0.35*std::sin(2.4*t); };
  auto angdot = [](double t){ return 0.8 + 0.35*2.4*std::cos(2.4*t); };
  auto accW   = [](double t){ return V3D(-0.9*2.1*2.1*std::sin(2.1*t),
                                         -0.6*1.7*1.7*std::cos(1.7*t),
                                         -0.25*3.3*3.3*std::sin(3.3*t)); };
  const double sa = 0.02, sg = 0.002;
  const int N = 20;                       // 200 Hz over 0.1 s
  const double dt = (t1 - t0) / N;

  std::mt19937 rng(2024);
  std::normal_distribution<double> na(0.0, sa), ng(0.0, sg);

  // Build the noisy IMU stream, then dead-reckon the poses from it exactly
  // as ImuProc::propagate() does (right-multiplied Exp, trapezoid accel).
  std::vector<ImuSample> imu;
  for (int i = 0; i <= N; ++i) {
    double t = t0 + i * dt;
    M3D R = Exp(axis * ang(t));
    imu.emplace_back(R.transpose() * (accW(t) - g) + V3D(na(rng), na(rng), na(rng)),
                     axis * angdot(t) + V3D(ng(rng), ng(rng), ng(rng)), t);
  }

  std::vector<Pose6D> poses;
  M3D R = Exp(axis * ang(t0));
  V3D p(0.9*std::sin(2.1*t0) + 0.3*t0, 0.6*std::cos(1.7*t0), 0.25*std::sin(3.3*t0));
  V3D v(0.9*2.1*std::cos(2.1*t0) + 0.3, -0.6*1.7*std::sin(1.7*t0), 0.25*3.3*std::cos(3.3*t0));
  for (int i = 0; i < N; ++i) {
    Pose6D ps{}; ps.t = imu[i].t; ps.rot = R; ps.pos = p; ps.vel = v; ps.dt = dt;
    poses.push_back(ps);
    const V3D aw0 = R * imu[i].acc + g;
    R = R * Exp(0.5 * (imu[i].gyro + imu[i+1].gyro), dt);
    const V3D aw1 = R * imu[i+1].acc + g;
    const V3D aw = 0.5 * (aw0 + aw1);
    p += v * dt + 0.5 * aw * dt * dt;
    v += aw * dt;
  }

  printf("IMU-integrated poses (the production coupling), true sigma_a=%.4f sigma_g=%.5f\n", sa, sg);
  printf("  n_cp  sigma_a_hat   ratio   acf1_a   sigma_g_hat   ratio   acf1_g   fit_res_pos\n");
  for (int ncp : {4, 6, 8, 10, 14, 18}) {
    SplineOptions o; o.n_control_points = ncp;
    ScanSpline s;
    if (!s.fit(poses, poses.front().t, t1, o)) { printf("  %4d  fit refused\n", ncp); continue; }
    auto st = computeSplineImuResidual(s, imu, V3D::Zero(), V3D::Zero(), g);
    if (!st.valid()) { printf("  %4d  residual n too small (%d)\n", ncp, st.n); continue; }
    printf("  %4d  %10.6f  %6.2f  %+7.3f  %11.7f  %6.2f  %+7.3f  %.2e\n",
           s.nControlPoints(), std::sqrt(st.cov_acc), std::sqrt(st.cov_acc)/sa, st.acf1_acc,
           std::sqrt(st.cov_gyr), std::sqrt(st.cov_gyr)/sg, st.acf1_gyr, s.fitResidualPos());
  }
  return 0;
}
