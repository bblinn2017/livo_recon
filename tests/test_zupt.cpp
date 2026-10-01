// R72 unit test (F-401): zero-velocity pseudo-update (include/livo_recon/lio/zupt.h). Isolated algebraic identities only.
//   ZU1 information identity: P'^-1 = P^-1 + H^T (sigma^2 I)^-1 H (1e-8 relative), over random SPD P.
//   ZU2 sigma_v -> 0 drives the post-update velocity to zero: v + dx_v == 0 (1e-6 relative to |v|), and the velocity block of P'
//       vanishes.
//   ZU3 P' is symmetric and positive definite; the velocity block of P' is not larger than that of P (trace).
//   ZU4 the correction is the weighted pull: dx_v = -P_vv S^-1 v exactly.
//   ZU5 sigma_v -> infinity leaves state and covariance unchanged (dx ~ 0, P' ~ P).
//   ZU6 the check fires: the plain (I - K H) P form with a deliberately wrong gain (K scaled by 0.5) fails the information identity
//       by a wide margin, so ZU1 would catch a mis-built gain.
//   ZD1 detector: needs enter_scans consecutive candidates; a single non-candidate scan exits immediately and resets the run.
//   ZD2 mode helpers: off -> on() false; detect_only -> on() true, apply() false; apply -> both true.
#include "livo_recon/lio/zupt.h"

#include <cstdio>
#include <cstdlib>
#include <random>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

static Eigen::MatrixXd randomSpd(std::mt19937& rng, int n)
{
  std::normal_distribution<double> N(0, 1);
  Eigen::MatrixXd B(n, n);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) B(i, j) = N(rng);
  return 1e-2 * (B * B.transpose() + 0.3 * Eigen::MatrixXd::Identity(n, n));
}

int main()
{
  std::mt19937 rng(5);
  const int dim = 18, iv = 6;
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, dim);
  H.block<3, 3>(0, iv).setIdentity();
  for (int trial = 0; trial < 20; ++trial) {
    const Eigen::MatrixXd P = randomSpd(rng, dim);
    const Eigen::Vector3d v(0.12, -0.05, 0.03);
    const double sigma = 0.05;
    Eigen::VectorXd dx; Eigen::MatrixXd Pn;
    zuptVelocityUpdate(P, v, iv, sigma, dx, Pn);
    const Eigen::MatrixXd info_expect = P.inverse() + H.transpose() * H / (sigma * sigma);
    const double rel = (Pn.inverse() - info_expect).norm() / info_expect.norm();
    if (rel > 1e-8) { CHECK(false, "ZU1 information identity"); }
    Eigen::LLT<Eigen::MatrixXd> llt(Pn);
    if (llt.info() != Eigen::Success || (Pn - Pn.transpose()).cwiseAbs().maxCoeff() > 1e-15) { CHECK(false, "ZU3 symmetric PD"); }
    if (Pn.block<3, 3>(iv, iv).trace() > P.block<3, 3>(iv, iv).trace()) { CHECK(false, "ZU3 velocity trace does not grow"); }
    const Eigen::Matrix3d S = P.block<3, 3>(iv, iv) + sigma * sigma * Eigen::Matrix3d::Identity();
    const Eigen::Vector3d dv_expect = -P.block<3, 3>(iv, iv) * S.inverse() * v;
    if ((dx.segment<3>(iv) - dv_expect).norm() > 1e-12 * v.norm()) { CHECK(false, "ZU4 weighted pull"); }
  }
  CHECK(true, "ZU1 information identity over 20 random SPD covariances");
  CHECK(true, "ZU3 symmetric positive definite, velocity trace does not grow (20 draws)");
  CHECK(true, "ZU4 dx_v = -P_vv S^-1 v (20 draws)");
  {
    const Eigen::MatrixXd P = randomSpd(rng, dim);
    const Eigen::Vector3d v(0.3, 0.2, -0.1);
    Eigen::VectorXd dx; Eigen::MatrixXd Pn;
    zuptVelocityUpdate(P, v, iv, 1e-7, dx, Pn);
    CHECK((v + dx.segment<3>(iv)).norm() < 1e-6 * v.norm(), "ZU2 sigma -> 0: velocity driven to zero");
    CHECK((Pn.block<3, 3>(iv, iv).cwiseAbs().maxCoeff() < 1e-9), "ZU2 sigma -> 0: velocity block of P' vanishes");
    zuptVelocityUpdate(P, v, iv, 1e6, dx, Pn);
    CHECK(dx.cwiseAbs().maxCoeff() < 1e-9 && (Pn - P).norm() < 1e-9 * P.norm(), "ZU5 sigma -> infinity: no change");
    // ZU6 control
    const double sigma = 0.05;
    const Eigen::Matrix3d S = P.block<3, 3>(iv, iv) + sigma * sigma * Eigen::Matrix3d::Identity();
    const Eigen::MatrixXd K = 0.5 * P.block(0, iv, dim, 3) * S.inverse();   // wrong gain
    Eigen::MatrixXd IKH = Eigen::MatrixXd::Identity(dim, dim);
    IKH.block(0, iv, dim, 3) -= K;
    const Eigen::MatrixXd Pwrong = IKH * P * IKH.transpose() + sigma * sigma * K * K.transpose();
    const Eigen::MatrixXd info_expect = P.inverse() + H.transpose() * H / (sigma * sigma);
    const double rel = (Pwrong.inverse() - info_expect).norm() / info_expect.norm();
    CHECK(rel > 1e-3, "ZU6 control: a wrong gain breaks the information identity (the check fires)");
  }
  {
    ZuptOptions o; o.gyro_thresh = 0.02; o.acc_thresh = 0.15; o.enter_scans = 3;
    ZuptDetector d;
    CHECK(!d.update(0.01, 0.1, o) && !d.update(0.01, 0.1, o), "ZD1 not stationary before enter_scans");
    CHECK(d.update(0.01, 0.1, o), "ZD1 stationary on the third candidate");
    CHECK(d.update(0.01, 0.1, o), "ZD1 stays stationary");
    CHECK(!d.update(0.5, 0.1, o) && d.run() == 0, "ZD1 one non-candidate exits at once and resets the run");
    CHECK(!d.update(0.01, 0.1, o), "ZD1 re-entry needs enter_scans again");
    CHECK(!d.update(0.01, 0.5, o) && d.run() == 0, "ZD1 the acceleration criterion also exits");
  }
  {
    ZuptOptions o;
    CHECK(!o.on() && !o.apply(), "ZD2 off");
    o.mode = "detect_only";
    CHECK(o.on() && !o.apply(), "ZD2 detect_only");
    o.mode = "apply";
    CHECK(o.on() && o.apply(), "ZD2 apply");
  }
  std::printf("ALL PASS\n");
  return 0;
}
