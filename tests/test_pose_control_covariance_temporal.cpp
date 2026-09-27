#include "livo_recon/diagnostics/pose_control/pose_control_physical_diagnostics.h"
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>

using namespace livo_recon;

namespace {
int failures = 0;
void check(bool ok, const char* name, double value, double tol)
{
  if (!ok) {
    ++failures;
    std::printf("  [FAIL] %s %.6e tol %.6e\n", name, value, tol);
  } else {
    std::printf("  [PASS] %s %.6e tol %.6e\n", name, value, tol);
  }
}
}

int main()
{
  std::printf("Pose-control covariance and temporal-coupling diagnostics tests\n");

  Eigen::Matrix3d Pk;
  Pk << 2.0, 0.2, 0.1,
        0.2, 1.5, 0.0,
        0.1, 0.0, 1.0;
  Eigen::Matrix3d A;
  A << 0.8, 0.1, 0.0,
       0.0, 0.7, 0.2,
       0.0, 0.0, 0.9;
  Eigen::Matrix3d Q;
  Q << 0.3, 0.02, 0.0,
       0.02, 0.2, 0.01,
       0.0, 0.01, 0.1;
  Eigen::Matrix3d Pl = A * Pk * A.transpose() + Q;
  Eigen::Matrix3d Plk = A * Pk;

  const Eigen::MatrixXd cond = poseControlConditionalCovariance(Pl, Pk, Plk);
  check((cond-Q).norm() < 1e-12, "conditional covariance matches independently propagated Q", (cond-Q).norm(), 1e-12);

  Eigen::Matrix<double,2,3> H;
  H << 1.0, 0.0, 0.0,
       0.0, 0.0, 1.0;
  Eigen::Matrix2d R = 0.25 * Eigen::Matrix2d::Identity();
  const Eigen::MatrixXd Reff = poseControlEffectiveMeasurementCovariance(R, H, cond);
  const Eigen::MatrixXd Reff_ref = R + H * Q * H.transpose();
  check((Reff-Reff_ref).norm() < 1e-12, "effective covariance matches direct reference", (Reff-Reff_ref).norm(), 1e-12);

  Eigen::Matrix3d P = Eigen::Matrix3d::Identity();
  const Eigen::MatrixXd reduction = poseControlCovarianceReduction(P, H, R);
  Eigen::MatrixXd S = H*P*H.transpose()+R;
  const Eigen::MatrixXd reduction_ref = P*H.transpose()*S.inverse()*H*P;
  check((reduction-reduction_ref).norm() < 1e-12, "covariance reduction matches Joseph-equivalent reference", (reduction-reduction_ref).norm(), 1e-12);
  check((reduction-reduction.transpose()).norm() < 1e-14, "covariance reduction is symmetric", (reduction-reduction.transpose()).norm(), 1e-14);

  const Eigen::MatrixXd P_post = P - reduction;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5*(P_post+P_post.transpose()));
  check(es.eigenvalues().minCoeff() > 0.0, "posterior covariance remains positive definite", es.eigenvalues().minCoeff(), 0.0);

  return failures == 0 ? 0 : 1;
}
