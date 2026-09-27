#include "livo_recon/lio/pose_control_covariance.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>

using namespace livo_recon;

namespace
{
int failures = 0;

void check(bool ok, const char* name, double value = 0.0, double tol = 0.0)
{
  std::printf("  [%s] %-70s %.6e tol %.6e\n", ok ? "PASS" : "FAIL", name, value, tol);
  if (!ok) ++failures;
}

Eigen::Matrix<double,9,9> makeJointCovariance()
{
  Eigen::Matrix<double,9,9> L = Eigen::Matrix<double,9,9>::Identity();
  L.block<3,3>(6,0) = 0.18 * Eigen::Matrix3d::Identity();
  L.block<3,3>(6,3) = -0.27 * Eigen::Matrix3d::Identity();
  L.block<3,3>(3,0) = 0.09 * Eigen::Matrix3d::Identity();
  return L * L.transpose() + 0.2 * Eigen::Matrix<double,9,9>::Identity();
}

void testInformationFormDecomposition()
{
  const Eigen::Matrix<double,9,9> P = makeJointCovariance();
  const Eigen::Matrix<double,9,9> Pinv = P.inverse();
  Eigen::Matrix<double,6,6> Lambda = Eigen::Matrix<double,6,6>::Identity();
  Lambda.diagonal() << 5.0, 7.0, 9.0, 13.0, 17.0, 19.0;
  Eigen::Matrix<double,9,9> Hfull = Eigen::Matrix<double,9,9>::Zero();
  Hfull.topLeftCorner<6,6>() = Lambda;
  const Eigen::Matrix<double,9,9> K1 = (Pinv + Hfull).inverse();

  Eigen::Matrix<double,6,1> b;
  b << 0.2, -0.1, 0.05, -0.3, 0.12, 0.08;
  Eigen::Matrix<double,9,1> vec;
  vec << 0.01, -0.02, 0.03, 0.2, -0.1, 0.04, -0.05, 0.02, 0.08;
  Eigen::Matrix<double,9,1> bfull = Eigen::Matrix<double,9,1>::Zero();
  bfull.head<6>() = b;

  const Eigen::Matrix<double,9,1> decomposed =
      K1.leftCols<6>() * b + vec - K1.leftCols<6>() * Lambda * vec.head<6>();
  const Eigen::Matrix<double,9,1> direct = K1 * (bfull + Pinv * vec);
  const double err = (decomposed - direct).norm();
  check(err < 1e-12, "9-D decomposition equals direct information solve", err, 1e-12);
  check(decomposed.tail<3>().norm() > 1e-6,
        "pose-only LiDAR induces velocity through joint covariance",
        decomposed.tail<3>().norm(), 1e-6);
}

void testAllKnotsCovarianceRealization()
{
  constexpr int n = 15;
  Eigen::Matrix<double,9,n> J = Eigen::Matrix<double,9,n>::Zero();
  J.leftCols<9>().setIdentity();
  J.rightCols<6>().setRandom();
  Eigen::Matrix<double,n,n> B = Eigen::Matrix<double,n,n>::Random();
  const Eigen::Matrix<double,n,n> Pz = B * B.transpose() + 0.5 * Eigen::Matrix<double,n,n>::Identity();
  Eigen::Matrix<double,9,1> target;
  target << 0.02, -0.01, 0.03, 0.2, -0.1, 0.05, -0.06, 0.04, 0.01;
  const Eigen::Matrix<double,9,9> Py = J * Pz * J.transpose();
  const Eigen::Matrix<double,n,1> dz = Pz * J.transpose() * generalPseudoInverse(Py, 1e-12) * target;
  const double err = (J * dz - target).norm();
  check(err < 1e-10, "all-knots covariance realization reproduces full-rank target", err, 1e-10);
}

void testTailResidualAndAppliedScaling()
{
  Eigen::Matrix<double,9,6> Jtail = Eigen::Matrix<double,9,6>::Zero();
  Jtail.topRows<6>().setIdentity();
  Jtail.bottomRows<3>() = 0.4 * Eigen::Matrix<double,3,6>::Random();
  const Eigen::Matrix<double,6,6> Pt = Eigen::Matrix<double,6,6>::Identity();
  Eigen::Matrix<double,9,1> target;
  target << 0.1, -0.2, 0.05, 0.4, -0.1, 0.2, 0.7, -0.5, 0.3;
  const Eigen::Matrix<double,6,1> dt =
      Pt * Jtail.transpose() * generalPseudoInverse(Jtail * Pt * Jtail.transpose(), 1e-12) * target;
  const Eigen::Matrix<double,9,1> requested = Jtail * dt;
  check((requested - target).norm() > 1e-6,
        "six-DOF tail realization exposes nonzero nine-DOF residual",
        (requested - target).norm(), 1e-6);

  const double scale = 0.25;
  const Eigen::Matrix<double,9,1> applied = Jtail * (scale * dt);
  const double scale_err = (applied - scale * requested).norm();
  check(scale_err < 1e-12, "applied linear realization follows trust-region scale", scale_err, 1e-12);
  check((applied - target).norm() >= (requested - target).norm(),
        "applied error is evaluated against the unscaled physical target");
}
}  // namespace

int main()
{
  std::puts("pose-control physical-RPV update tests");
  testInformationFormDecomposition();
  testAllKnotsCovarianceRealization();
  testTailResidualAndAppliedScaling();
  std::printf("failures: %d\n", failures);
  return failures == 0 ? 0 : 1;
}
