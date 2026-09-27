#include "livo_recon/processing/pose_control_coupled_modes.h"
#include "livo_recon/utils/algo/math.h"
#include "livo_recon/utils/state/state.h"

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

Eigen::MatrixXd makeStateCovariance()
{
  Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(9, 9) * 0.01;
  // Real p-v and R-p cross terms so the R/P-to-V coupling check below is
  // exercising a genuinely coupled covariance, not a block-diagonal one.
  cov.block<3, 3>(3, 6) = 0.004 * Eigen::Matrix3d::Identity();
  cov.block<3, 3>(6, 3) = cov.block<3, 3>(3, 6).transpose();
  cov.block<3, 3>(0, 3) = 0.001 * Eigen::Matrix3d::Identity();
  cov.block<3, 3>(3, 0) = cov.block<3, 3>(0, 3).transpose();
  return cov;
}

Eigen::MatrixXd makeFullStateCovariance()
{
  Eigen::MatrixXd cov=Eigen::MatrixXd::Identity(18,18)*0.01;
  cov.topLeftCorner(9,9)=makeStateCovariance();
  // Correlate R/P/V with two nuisance-state blocks.  A direct-IMU update
  // must not silently discard the resulting mean/covariance changes.
  cov.block<9,3>(0,9).setConstant(2e-4);
  cov.block<3,9>(9,0)=cov.block<9,3>(0,9).transpose();
  cov.block<9,3>(0,12).setConstant(-1e-4);
  cov.block<3,9>(12,0)=cov.block<9,3>(0,12).transpose();
  return cov;
}

PoseControlPhysicalLidarInformation makeLidarInfo()
{
  PoseControlPhysicalLidarInformation lidar;
  lidar.Lambda = Eigen::Matrix<double, 6, 6>::Identity();
  lidar.Lambda.diagonal() << 4.0, 6.0, 8.0, 10.0, 12.0, 14.0;
  lidar.b << 0.3, -0.15, 0.05, -0.2, 0.1, -0.05;
  return lidar;
}

// Item: identical current/propagated rotation must produce exactly zero
// relative restoring rotation (StateGroup::boxminusFromPropagat()'s own
// convention, reused inline by solveDirectLidarImuPhysicalRpv).
void testIdenticalRotationGivesZeroRelativeRotation()
{
  const M3D R = Exp(V3D(0.9, -1.4, 2.1));  // large, nonidentity absolute attitude
  StateGroup current, propagated;
  current.setPropagatedState(R, V3D(1.0, 2.0, 3.0), V3D(0.1, 0.2, 0.3));
  propagated.setPropagatedState(R, V3D(1.0, 2.0, 3.0), V3D(0.1, 0.2, 0.3));

  PhysicalRpvUpdate rpv;
  const bool ok = solveDirectLidarImuPhysicalRpv(current, propagated, makeStateCovariance(), makeLidarInfo(), rpv);
  check(ok, "solve succeeds for identical current/propagated state");
  check(rpv.vec.head<3>().norm() < 1e-12,
        "identical large nonidentity rotations give zero relative restoring rotation",
        rpv.vec.head<3>().norm(), 1e-12);
}

// Item: a known small relative rotation on top of a large nonidentity
// absolute attitude must produce the correct sign/magnitude (not merely
// something nonzero) -- checked against Log(R_current^T R_propagated)
// computed independently here, not reusing the production call.
void testSmallRelativeRotationOnLargeAbsoluteAttitude()
{
  const M3D R0 = Exp(V3D(0.8, 1.6, -2.3));       // large absolute attitude
  const V3D small_delta(0.004, -0.007, 0.002);   // small RIGHT perturbation
  const M3D R1 = R0 * Exp(small_delta);
  StateGroup current, propagated;
  current.setPropagatedState(R0, V3D::Zero(), V3D::Zero());
  propagated.setPropagatedState(R1, V3D::Zero(), V3D::Zero());

  PhysicalRpvUpdate rpv;
  const bool ok = solveDirectLidarImuPhysicalRpv(current, propagated, makeStateCovariance(), makeLidarInfo(), rpv);
  check(ok, "solve succeeds for small-delta-on-large-attitude case");
  const double err = (rpv.vec.head<3>() - small_delta).norm();
  check(err < 1e-9, "relative rotation recovers the exact small delta, sign and magnitude", err, 1e-9);
}

// Item: physical solve sign convention. lidar.b already carries the
// production -H^T*W*r convention (see pose_control_lidar_factor.cpp's own
// accumulation); the measurement term must use it with NO additional
// negation, i.e. measurement_term == K1_pose * b exactly, cross-checked
// against an independently-assembled direct information solve.
void testSignConventionMeasurementTermNoExtraNegation()
{
  StateGroup current, propagated;
  const M3D R = Exp(V3D(0.3, -0.2, 0.5));
  current.setPropagatedState(R, V3D(0.5, -0.3, 0.2), V3D(0.05, -0.02, 0.03));
  propagated.setPropagatedState(R, V3D(0.5, -0.3, 0.2), V3D(0.05, -0.02, 0.03));
  const Eigen::MatrixXd P = makeStateCovariance();
  const PoseControlPhysicalLidarInformation lidar = makeLidarInfo();

  PhysicalRpvUpdate rpv;
  const bool ok = solveDirectLidarImuPhysicalRpv(current, propagated, P, lidar, rpv);
  check(ok, "solve succeeds for sign-convention check");

  // Independent reconstruction: A = H_full + P^-1, K1 = A^-1,
  // measurement_term = K1_pose * b (no sign flip).
  Eigen::Matrix<double, 9, 9> Pfull = Eigen::Matrix<double, 9, 9>::Zero();
  Pfull = P;
  const Eigen::Matrix<double, 9, 9> Pinv = Pfull.inverse();
  Eigen::Matrix<double, 9, 9> Hfull = Eigen::Matrix<double, 9, 9>::Zero();
  Hfull.topLeftCorner<6, 6>() = lidar.Lambda;
  const Eigen::Matrix<double, 9, 9> K1 = (Pinv + Hfull).inverse();
  const Eigen::Matrix<double, 9, 1> expected_measurement_term = K1.leftCols<6>() * lidar.b;

  const double err = (rpv.measurement_term - expected_measurement_term).norm();
  check(err < 1e-9, "measurement_term == +K1_pose*b, matching production's b=-H^TWr convention", err, 1e-9);
}

// Item: R/P-to-V covariance coupling produces the expected velocity
// correction even though LiDAR never observes velocity directly
// (H_full's bottom-right 3x3 block is exactly zero).
void testVelocityCorrectionFromCovarianceCoupling()
{
  StateGroup current, propagated;
  const M3D R = Exp(V3D(0.1, 0.2, -0.1));
  current.setPropagatedState(R, V3D::Zero(), V3D::Zero());
  propagated.setPropagatedState(R, V3D::Zero(), V3D::Zero());

  PhysicalRpvUpdate rpv;
  const bool ok = solveDirectLidarImuPhysicalRpv(current, propagated, makeStateCovariance(), makeLidarInfo(), rpv);
  check(ok, "solve succeeds for velocity-coupling check");
  check(rpv.H_full.bottomRightCorner<3, 3>().norm() < 1e-15,
        "LiDAR information has exactly zero direct velocity block");
  check(rpv.solution.tail<3>().norm() > 1e-6,
        "nonzero velocity correction is induced purely through R/P-to-V covariance coupling",
        rpv.solution.tail<3>().norm(), 1e-6);
}

// Item: covariance symmetry/PSD/dimensions of the returned posterior.
void testPosteriorCovarianceSymmetricPsdCorrectDimension()
{
  StateGroup current, propagated;
  const M3D R = Exp(V3D(0.4, -0.3, 0.2));
  current.setPropagatedState(R, V3D(1, 1, 1), V3D(0, 0, 0));
  propagated.setPropagatedState(R, V3D(1, 1, 1), V3D(0.02, -0.01, 0.03));

  PhysicalRpvUpdate rpv;
  const bool ok = solveDirectLidarImuPhysicalRpv(current, propagated, makeStateCovariance(), makeLidarInfo(), rpv);
  check(ok, "solve succeeds for posterior covariance check");
  check(rpv.P_post.rows() == 9 && rpv.P_post.cols() == 9, "posterior covariance is 9x9");
  const double asym = (rpv.P_post - rpv.P_post.transpose()).norm();
  check(asym < 1e-10, "posterior covariance is symmetric", asym, 1e-10);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(rpv.P_post);
  check(es.eigenvalues().minCoeff() > -1e-9, "posterior covariance is PSD", es.eigenvalues().minCoeff(), -1e-9);
}

void testFullStateConditionalLiftPreservesCrossCovariance()
{
  StateGroup current, propagated;
  const M3D R=Exp(V3D(0.2,-0.4,1.1));
  current.setPropagatedState(R,V3D::Zero(),V3D::Zero());
  propagated.setPropagatedState(R,V3D::Zero(),V3D::Zero());
  PhysicalRpvUpdate rpv;
  const bool ok=solveDirectLidarImuPhysicalRpv(current,propagated,makeFullStateCovariance(),makeLidarInfo(),rpv);
  check(ok,"full-state conditional physical solve succeeds");
  check(rpv.state_delta.size()==18,"full-state correction retains StateGroup dimension");
  check(rpv.state_delta.segment(9,6).norm()>1e-9,
        "R/P/V cross covariance induces nuisance-state correction",
        rpv.state_delta.segment(9,6).norm(),1e-9);
  check(rpv.P_full_post.rows()==18 && rpv.P_full_post.cols()==18,
        "full posterior covariance retains StateGroup dimension");
  check((rpv.P_full_post-rpv.P_full_post.transpose()).norm()<1e-10,
        "full posterior covariance is symmetric",
        (rpv.P_full_post-rpv.P_full_post.transpose()).norm(),1e-10);
}

// Item: parseCoupledMode's mapping for all ten accepted names (mirrors
// test_pose_control_coupled_mode_dispatch.cpp's own check; kept here too
// since this file is the one that exercises the family enum against the
// actual solve/realization entry points, not just name parsing).
void testAllTenModeNamesResolveToKnownFamilies()
{
  const char* names[] = {
      "local_spline", "single_tail", "direct_lidar_imu", "covariance_all_knots",
      "physical_rpv_local_spline", "physical_rpv_single_tail",
      "physical_rpv_direct_lidar_imu", "physical_rpv_covariance_all_knots",
      "physical_rpv_tail", "physical_rpv_all_knots"};
  bool all_ok = true;
  for (const char* name : names) {
    CoupledModeSelection sel;
    if (!parseCoupledMode(name, sel)) all_ok = false;
  }
  check(all_ok, "all ten accepted configuration names parse successfully");
}

}  // namespace

int main()
{
  std::puts("pose-control coupled-family physical-solve tests");
  testIdenticalRotationGivesZeroRelativeRotation();
  testSmallRelativeRotationOnLargeAbsoluteAttitude();
  testSignConventionMeasurementTermNoExtraNegation();
  testVelocityCorrectionFromCovarianceCoupling();
  testPosteriorCovarianceSymmetricPsdCorrectDimension();
  testFullStateConditionalLiftPreservesCrossCovariance();
  testAllTenModeNamesResolveToKnownFamilies();
  std::printf("failures: %d\n", failures);
  return failures == 0 ? 0 : 1;
}
