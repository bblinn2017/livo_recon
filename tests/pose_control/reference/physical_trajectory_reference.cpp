#include "physical_trajectory_reference.h"

namespace livo_recon_test
{

Eigen::Matrix<double, 6, 6> whiteAccelerationProcessCovariance(double dt, double q)
{
  using M6 = Eigen::Matrix<double, 6, 6>;
  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;

  M6 Q = M6::Zero();
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  Q.block<3, 3>(0, 0) = (dt3 / 3.0) * q * I;
  Q.block<3, 3>(0, 3) = (dt2 / 2.0) * q * I;
  Q.block<3, 3>(3, 0) = Q.block<3, 3>(0, 3);
  Q.block<3, 3>(3, 3) = dt * q * I;
  return Q;
}

Eigen::Matrix<double, 6, 6> propagatePVWhiteAcceleration(
    const Eigen::Matrix<double, 6, 6>& P0, double dt, double q)
{
  using M6 = Eigen::Matrix<double, 6, 6>;
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  M6 F = M6::Identity();
  F.block<3, 3>(0, 3) = dt * I;
  return F * P0 * F.transpose() + whiteAccelerationProcessCovariance(dt, q);
}

}  // namespace livo_recon_test
