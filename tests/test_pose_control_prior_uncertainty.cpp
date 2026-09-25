#include "livo_recon/lio/pose_control_spline.h"
#include "livo_recon/lio/pose_control_layout.h"
#include "livo_recon/lio/pose_control_imu_prior_builder.h"
#include "livo_recon/lio/pose_control_covariance.h"
#include "physical_trajectory_reference.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace livo_recon;

namespace
{
int failures = 0;

void check(bool ok, const char* name, double value = 0.0, double tol = 0.0)
{
  std::printf("  [%s] %-72s %.6e tol %.6e\n", ok ? "PASS" : "FAIL", name, value, tol);
  if (!ok) ++failures;
}

PoseControlSpline makeSpline()
{
  PoseControlSpline s;
  s.init(7, 0.0, 0.1);
  s.R_anchor = M3D::Identity();
  for (int k = 0; k < s.N(); ++k) {
    s.cp_p.col(k) = V3D(0.01 * k, -0.002 * k, 0.001 * k);
    s.cp_phi.col(k) = V3D(0.001 * k, -0.0005 * k, 0.0002 * k);
  }
  return s;
}

std::vector<ImuSample> perfectImu(const PoseControlSpline& s, const V3D& gravity)
{
  std::vector<ImuSample> out;
  for (double t = s.t0(); t <= s.t1() + 1e-12; t += 0.001) {
    ImuSample m;
    m.t = std::min(t, s.t1());
    m.acc = s.rotAt(m.t).transpose() * (s.accAt(m.t) - gravity);
    m.gyro = s.omegaBodyAt(m.t);
    out.push_back(m);
    if (m.t >= s.t1()) break;
  }
  return out;
}

// Independent closed-form first-segment head mapping for the cubic B-spline.
// This test reference intentionally retypes the three boundary basis rows so
// that the test does not call poseControlHeadPosJacobians() on both sides.
void independentHeadPVJacobian(double delta, double t,
                               Eigen::Matrix<double, 6, 6>& J)
{
  const double u = t / delta;
  const double inv_delta = 1.0 / delta;
  const Eigen::Matrix3d M = (Eigen::Matrix3d() <<
      1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0,
      -0.5 * inv_delta, 0.0, 0.5 * inv_delta,
      inv_delta * inv_delta, -2.0 * inv_delta * inv_delta,
      inv_delta * inv_delta).finished();

  const Eigen::Matrix3d Minv = M.inverse();
  Eigen::RowVector3d b, db;
  const double u2 = u * u;
  const double om = 1.0 - u;
  b << om * om * om / 6.0,
       (3.0 * u * u2 - 6.0 * u2 + 4.0) / 6.0,
       (-3.0 * u * u2 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0;
  db << -0.5 * om * om,
        0.5 * (3.0 * u2 - 4.0 * u),
        0.5 * (-3.0 * u2 + 2.0 * u + 1.0);

  // The three-by-two inverse columns give the control-point response to p0/v0.
  J.setZero();
  for (int k = 0; k < 3; ++k) {
    const double m_p = Minv(k, 0);
    const double m_v = Minv(k, 1);
    J.block<3, 3>(0, 0) += b(k) * m_p * Eigen::Matrix3d::Identity();
    J.block<3, 3>(0, 3) += b(k) * m_v * Eigen::Matrix3d::Identity();
    J.block<3, 3>(3, 0) += db(k) * inv_delta * m_p * Eigen::Matrix3d::Identity();
    J.block<3, 3>(3, 3) += db(k) * inv_delta * m_v * Eigen::Matrix3d::Identity();
  }
}

void testHeadSensitivityIndependentReference()
{
  const PoseControlSpline s = makeSpline();
  const auto hs = poseControlHeadPosSensitivity(s);
  const double t = 0.037;

  M3D dp_dp0, dp_dv0, dv_dp0, dv_dv0;
  poseControlHeadPosJacobians(s, hs, t, dp_dp0, dp_dv0, dv_dp0, dv_dv0);

  Eigen::Matrix<double, 6, 6> J_ref;
  independentHeadPVJacobian(s.delta(), t - s.t0(), J_ref);

  Eigen::Matrix<double, 6, 6> J_prod = Eigen::Matrix<double, 6, 6>::Zero();
  J_prod.block<3, 3>(0, 0) = dp_dp0;
  J_prod.block<3, 3>(0, 3) = dp_dv0;
  J_prod.block<3, 3>(3, 0) = dv_dp0;
  J_prod.block<3, 3>(3, 3) = dv_dv0;

  check((J_prod - J_ref).norm() < 1e-12,
        "head p/v sensitivity matches independent cubic-boundary reference",
        (J_prod - J_ref).norm(), 1e-12);
}

struct PriorAssembly
{
  Eigen::MatrixXd P_full;
  Eigen::VectorXd delta_z_mean;
  Eigen::MatrixXd P_z;
  PoseControlHeadNullspace hns;
};

PriorAssembly assemblePrior(const PoseControlSpline& spline,
                            const PoseControlFreeLayout& layout,
                            const Eigen::MatrixXd& P0)
{
  const V3D gravity(0, 0, -9.81);
  const V3D var_a = V3D::Constant(0.02 * 0.02);
  const V3D var_w = V3D::Constant(0.005 * 0.005);
  const auto imu = perfectImu(spline, gravity);
  const PoseControlHeadNullspace hns =
      buildPoseControlHeadNullspace(spline, spline.posAt(spline.t0()), spline.velAt(spline.t0()));

  PoseControlFreeLayout lay = layout;
  const int raw_dim = lay.dim();
  const int d_eta = hns.freeDim();
  const int d_st = lay.dimST();
  const int d_z = d_eta + d_st;

  Eigen::MatrixXd A_raw = Eigen::MatrixXd::Zero(raw_dim, raw_dim);
  Eigen::VectorXd b_raw = Eigen::VectorXd::Zero(raw_dim);
  PoseControlPriorHeadBlock hb;
  buildPoseControlContinuousImuPrior(
      spline, lay, imu, V3D::Zero(), V3D::Zero(), gravity,
      var_a, var_w, A_raw, b_raw, &hb, nullptr);

  const Eigen::MatrixXd Omega0 = generalPseudoInverse(P0, 1e-9);
  Eigen::MatrixXd A_hh = hb.A_hh + Omega0;
  Eigen::MatrixXd A_hf = hb.A_hf;
  if (A_hf.size() == 0) A_hf = Eigen::MatrixXd::Zero(9, raw_dim);

  if (d_st > 0) {
    A_raw.block(lay.dimCFree(), lay.dimCFree(), d_st, d_st) +=
        Omega0.block(9, 9, d_st, d_st);
    A_hf.block(0, lay.dimCFree(), 9, d_st) += Omega0.block(0, 9, 9, d_st);
  }

  Eigen::MatrixXd Ps = Eigen::MatrixXd::Zero(raw_dim, d_z);
  Ps.block(0, 0, hns.rawDim(), d_eta) = hns.Z;
  if (d_st > 0)
    Ps.block(hns.rawDim(), d_eta, d_st, d_st) = Eigen::MatrixXd::Identity(d_st, d_st);

  const Eigen::MatrixXd A_ff = Ps.transpose() * A_raw * Ps;
  const Eigen::MatrixXd A_hf_z = A_hf * Ps;
  Eigen::MatrixXd Lambda_full = Eigen::MatrixXd::Zero(9 + d_z, 9 + d_z);
  Lambda_full.block(0, 0, 9, 9) = A_hh;
  Lambda_full.block(0, 9, 9, d_z) = A_hf_z;
  Lambda_full.block(9, 0, d_z, 9) = A_hf_z.transpose();
  Lambda_full.block(9, 9, d_z, d_z) = A_ff;

  PriorAssembly out;
  out.P_full = generalPseudoInverse(Lambda_full, 1e-9);
  out.P_z = out.P_full.bottomRightCorner(d_z, d_z);
  out.delta_z_mean = generalPseudoInverse(A_ff, 1e-9) * (Ps.transpose() * b_raw);
  out.hns = hns;
  return out;
}

void testFixedHeadMeanUncertainCovariance()
{
  const PoseControlSpline s = makeSpline();
  PoseControlFreeLayout layout;
  layout.N = s.N();
  layout.has_bg = layout.has_ba = layout.has_g = false;

  Eigen::MatrixXd P0 = Eigen::MatrixXd::Zero(9, 9);
  P0.block<3, 3>(0, 0) = 1e-5 * Eigen::Matrix3d::Identity();
  P0.block<3, 3>(3, 3) = 1e-5 * Eigen::Matrix3d::Identity();
  P0.block<3, 3>(6, 6) = 1e-4 * Eigen::Matrix3d::Identity();

  const PriorAssembly a = assemblePrior(s, layout, P0);
  const PriorAssembly b = assemblePrior(s, layout, 100.0 * P0);

  check((a.delta_z_mean - b.delta_z_mean).norm() < 1e-10,
        "changing fixed-head covariance leaves fixed-head conditional mean unchanged",
        (a.delta_z_mean - b.delta_z_mean).norm(), 1e-10);

  check((a.P_z - b.P_z).norm() > 1e-10,
        "changing fixed-head covariance changes spline uncertainty",
        (a.P_z - b.P_z).norm(), 1e-10);

  const double t = 0.037;
  Eigen::Matrix<double, 6, 6> J_h;
  independentHeadPVJacobian(s.delta(), t - s.t0(), J_h);
  const auto sample = evaluatePoseControlPhysicalSample(
      s, layout, a.hns, t, V3D(0, 0, -9.81));

  Eigen::Matrix<double, 6, 6> P_head_effect = J_h * P0 * J_h.transpose();
  const Eigen::MatrixXd J_eta_pv = [&]() {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, sample.dp_deta.cols());
    J.topRows(3) = sample.dp_deta;
    J.bottomRows(3) = sample.dv_deta;
    return J;
  }();

  const Eigen::MatrixXd P_eta_a = a.P_z.topLeftCorner(sample.dp_deta.cols(), sample.dp_deta.cols());
  const Eigen::MatrixXd P_eta_b = b.P_z.topLeftCorner(sample.dp_deta.cols(), sample.dp_deta.cols());
  const Eigen::Matrix<double, 6, 6> P_pv_a = J_eta_pv * P_eta_a * J_eta_pv.transpose() + P_head_effect;
  const Eigen::Matrix<double, 6, 6> P_pv_b = J_eta_pv * P_eta_b * J_eta_pv.transpose() + P_head_effect;

  check((P_pv_a - P_pv_b).norm() > 1e-12,
        "physical p/v covariance changes when only head uncertainty changes",
        (P_pv_a - P_pv_b).norm(), 1e-12);
}

void testWhiteAccelerationReference()
{
  Eigen::Matrix<double, 6, 6> P0 = Eigen::Matrix<double, 6, 6>::Zero();
  P0.block<3, 3>(0, 0) = 0.01 * Eigen::Matrix3d::Identity();
  P0.block<3, 3>(3, 3) = 0.02 * Eigen::Matrix3d::Identity();
  const double dt = 0.01;
  const double q = 0.5;

  const auto Q = livo_recon_test::whiteAccelerationProcessCovariance(dt, q);
  const auto P = livo_recon_test::propagatePVWhiteAcceleration(P0, dt, q);

  check((Q - Q.transpose()).norm() < 1e-14,
        "white-acceleration reference Q is symmetric", (Q - Q.transpose()).norm(), 1e-14);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(0.5 * (Q + Q.transpose()));
  check(es.eigenvalues().minCoeff() >= -1e-14,
        "white-acceleration reference Q is PSD", -es.eigenvalues().minCoeff(), 1e-14);

  const auto P_twice = livo_recon_test::propagatePVWhiteAcceleration(P, dt, q);
  const auto P_once_2dt = livo_recon_test::propagatePVWhiteAcceleration(P0, 2.0 * dt, q);
  const double semigroup_error = (P_twice - P_once_2dt).norm();
  check(semigroup_error < 1e-12,
        "physical covariance propagation satisfies the white-acceleration semigroup identity",
        semigroup_error, 1e-12);
}

void testHighFrequencyMahalanobis()
{
  const PoseControlSpline s = makeSpline();
  PoseControlFreeLayout layout;
  layout.N = s.N();
  layout.has_bg = layout.has_ba = layout.has_g = false;

  const V3D gravity(0, 0, -9.81);
  const V3D var_a = V3D::Constant(0.02 * 0.02);
  const V3D var_w = V3D::Constant(0.005 * 0.005);
  const auto imu = perfectImu(s, gravity);
  const int d = layout.dim();

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(d, d);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(d);
  buildPoseControlContinuousImuPrior(s, layout, imu, V3D::Zero(), V3D::Zero(), gravity,
                                     var_a, var_w, A, b, nullptr, nullptr);

  Eigen::VectorXd high = Eigen::VectorXd::Zero(d);
  Eigen::VectorXd low = Eigen::VectorXd::Zero(d);
  const double amp = 1e-3;
  for (int k = 3; k < s.N(); ++k) {
    high(3 * k) = (k & 1) ? -amp : amp;
    low(3 * k) = amp;
  }

  const double cost_high = (high.transpose() * A * high)(0);
  const double cost_low = (low.transpose() * A * low)(0);
  check(cost_high > 10.0 * std::max(cost_low, 1e-12),
        "high-frequency knot mode has substantially larger IMU Mahalanobis cost",
        cost_high / std::max(cost_low, 1e-12), 10.0);
}

}  // namespace

int main()
{
  std::printf("Pose-control physical prior uncertainty validation\n");
  testHeadSensitivityIndependentReference();
  testFixedHeadMeanUncertainCovariance();
  testWhiteAccelerationReference();
  testHighFrequencyMahalanobis();
  return failures ? 1 : 0;
}
