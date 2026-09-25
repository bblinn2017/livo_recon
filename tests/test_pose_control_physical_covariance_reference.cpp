#include "livo_recon/lio/pose_control_spline.h"
#include "livo_recon/lio/pose_control_layout.h"
#include "livo_recon/lio/pose_control_imu_prior_builder.h"
#include "livo_recon/lio/pose_control_covariance.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
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
  std::printf("  [%s] %-72s %.6e tol %.6e\n",
              ok ? "PASS" : "FAIL", name, value, tol);
  if (!ok) ++failures;
}

PoseControlSpline makeStationarySpline(int N)
{
  PoseControlSpline s;
  s.init(N, 0.0, 0.1);
  s.R_anchor = M3D::Identity();
  const V3D p0(0.3, -0.2, 0.1);
  for (int k = 0; k < N; ++k) {
    s.cp_p.col(k) = p0;
    s.cp_phi.col(k).setZero();
  }
  return s;
}

std::vector<ImuSample> perfectImu(const PoseControlSpline& s, double dt)
{
  std::vector<ImuSample> out;
  for (double t = s.t0(); t < s.t1() - 0.25 * dt; t += dt) {
    ImuSample m;
    m.t = t;
    m.acc = V3D::Zero();
    m.gyro = V3D::Zero();
    out.push_back(m);
  }
  return out;
}

// Independent re-typing of the cubic B-spline second derivative basis. This
// reference intentionally does not call PoseControlSpline::dAccDcp().
void independentSecondDerivativeBasis(double u, Eigen::Vector4d& ddb)
{
  const double om = 1.0 - u;
  ddb << om, 3.0 * u - 2.0, -3.0 * u + 1.0, u;
}

Eigen::MatrixXd independentAccelerationDesign(
    const PoseControlSpline& spline,
    const std::vector<ImuSample>& imu)
{
  const int N = spline.N();
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3 * static_cast<int>(imu.size()), 6 * N);
  const double inv_delta2 = 1.0 / (spline.delta() * spline.delta());

  for (int i = 0; i < static_cast<int>(imu.size()); ++i) {
    int seg = 0;
    double u = 0.0;
    spline.locate(imu[i].t, seg, u);

    Eigen::Vector4d ddb;
    independentSecondDerivativeBasis(u, ddb);
    for (int k = 0; k < 4; ++k) {
      const int abs_k = seg + k;
      if (abs_k < 0 || abs_k >= N) continue;
      H.block<3, 3>(3 * i, 3 * abs_k) =
          ddb[k] * inv_delta2 * M3D::Identity();
    }
  }
  return H;
}

Eigen::MatrixXd inversePositiveDefinite(const Eigen::MatrixXd& A)
{
  const Eigen::MatrixXd As = 0.5 * (A + A.transpose());
  Eigen::LDLT<Eigen::MatrixXd> ldlt(As);
  if (ldlt.info() != Eigen::Success) return Eigen::MatrixXd();
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(A.rows(), A.cols());
  return ldlt.solve(I);
}

// Independent physical p/v/a mapping using the explicitly typed scalar basis
// and the already-computed head-nullspace Z. No production physical-Jacobian
// accessor is used here.
Eigen::MatrixXd independentPhysicalJacobian(
    const PoseControlSpline& spline,
    const PoseControlHeadNullspace& hns,
    double t)
{
  const int N = spline.N();
  const int p_eta = 3 * N - 6;
  const Eigen::MatrixXd Zp = hns.Z.topRows(3 * N).leftCols(p_eta);

  int seg = 0;
  double u = 0.0;
  spline.locate(t, seg, u);

  const double inv_delta = 1.0 / spline.delta();
  const double inv_delta2 = inv_delta * inv_delta;
  const double u2 = u * u;
  const double om = 1.0 - u;
  Eigen::Vector4d b, db, ddb;
  b << om * om * om / 6.0,
       (3.0 * u * u * u - 6.0 * u2 + 4.0) / 6.0,
       (-3.0 * u * u * u + 3.0 * u2 + 3.0 * u + 1.0) / 6.0,
       u * u * u / 6.0;
  db << -0.5 * om * om,
        0.5 * (3.0 * u2 - 4.0 * u),
        0.5 * (-3.0 * u2 + 2.0 * u + 1.0),
        0.5 * u2;
  independentSecondDerivativeBasis(u, ddb);

  Eigen::MatrixXd Jc = Eigen::MatrixXd::Zero(9, 3 * N);
  for (int k = 0; k < 4; ++k) {
    const int abs_k = seg + k;
    if (abs_k < 0 || abs_k >= N) continue;
    Jc.block<3, 3>(0, 3 * abs_k) = b[k] * M3D::Identity();
    Jc.block<3, 3>(3, 3 * abs_k) = (db[k] * inv_delta) * M3D::Identity();
    Jc.block<3, 3>(6, 3 * abs_k) = (ddb[k] * inv_delta2) * M3D::Identity();
  }
  return Jc * Zp;
}

void buildProductionAndReference(const PoseControlSpline& spline,
                                 const std::vector<ImuSample>& imu,
                                 double q_acc,
                                 Eigen::MatrixXd& Aprod_eta,
                                 Eigen::MatrixXd& Aref_eta,
                                 PoseControlHeadNullspace& hns)
{
  PoseControlFreeLayout layout;
  layout.N = spline.N();
  layout.has_bg = layout.has_ba = layout.has_g = false;

  hns = buildPoseControlHeadNullspace(
      spline, spline.posAt(spline.t0()), spline.velAt(spline.t0()));

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(layout.dim(), layout.dim());
  Eigen::VectorXd b = Eigen::VectorXd::Zero(layout.dim());
  const V3D gravity = V3D::Zero();
  const V3D var_acc = V3D::Constant(q_acc / 0.001);
  const V3D var_gyr = V3D::Constant((0.005 * 0.005) / 0.001);

  buildPoseControlContinuousImuPrior(
      spline, layout, imu, V3D::Zero(), V3D::Zero(), gravity,
      var_acc, var_gyr, A, b, nullptr, nullptr);
  Aprod_eta = hns.Z.transpose() * A * hns.Z;

  const Eigen::MatrixXd H = independentAccelerationDesign(spline, imu);
  const double W = 1.0 / var_acc.x();
  const Eigen::MatrixXd Aref_raw = W * H.transpose() * H;
  Aref_eta = hns.Z.transpose() * Aref_raw * hns.Z;
}

void testIndependentPositionInformationReference()
{
  const PoseControlSpline spline = makeStationarySpline(7);
  const auto imu = perfectImu(spline, 0.001);

  Eigen::MatrixXd Aprod, Aref;
  PoseControlHeadNullspace hns;
  buildProductionAndReference(spline, imu, 4e-4, Aprod, Aref, hns);

  const int p_eta = 3 * spline.N() - 6;
  const Eigen::MatrixXd Aprod_p = Aprod.topLeftCorner(p_eta, p_eta);
  const Eigen::MatrixXd Aref_p = Aref.topLeftCorner(p_eta, p_eta);
  const double rel = (Aprod_p - Aref_p).norm() /
                     std::max(1.0, std::max(Aprod_p.norm(), Aref_p.norm()));

  check(rel < 1e-10,
        "production position-information Hessian matches independent analytic reference",
        rel, 1e-10);
}

void testIndependentPhysicalCovarianceReference()
{
  const PoseControlSpline spline = makeStationarySpline(7);
  const auto imu = perfectImu(spline, 0.001);

  Eigen::MatrixXd Aprod, Aref;
  PoseControlHeadNullspace hns;
  buildProductionAndReference(spline, imu, 4e-4, Aprod, Aref, hns);

  const int p_eta = 3 * spline.N() - 6;
  const Eigen::MatrixXd Pprod = inversePositiveDefinite(Aprod.topLeftCorner(p_eta, p_eta));
  const Eigen::MatrixXd Pref = inversePositiveDefinite(Aref.topLeftCorner(p_eta, p_eta));
  check(Pprod.size() > 0 && Pref.size() > 0,
        "independent position covariance references are invertible");
  if (Pprod.size() == 0 || Pref.size() == 0) return;

  double worst_cov = 0.0;
  for (double u : {0.0, 0.25, 0.5, 0.75, 1.0}) {
    const double t = spline.t0() + u * (spline.t1() - spline.t0());
    const Eigen::MatrixXd J = independentPhysicalJacobian(spline, hns, t);
    const Eigen::MatrixXd Cp = J * Pprod * J.transpose();
    const Eigen::MatrixXd Cr = J * Pref * J.transpose();
    const double rel = (Cp - Cr).norm() /
                       std::max(1.0, std::max(Cp.norm(), Cr.norm()));
    worst_cov = std::max(worst_cov, rel);
  }

  check(worst_cov < 1e-8,
        "mapped physical p/v/a covariance matches independent reference",
        worst_cov, 1e-8);
}

void testCovarianceNoiseScaling()
{
  const PoseControlSpline spline = makeStationarySpline(7);
  const auto imu = perfectImu(spline, 0.001);

  Eigen::MatrixXd A1, R1, A4, R4;
  PoseControlHeadNullspace h1, h4;
  buildProductionAndReference(spline, imu, 4e-4, A1, R1, h1);
  buildProductionAndReference(spline, imu, 16e-4, A4, R4, h4);

  const int p_eta = 3 * spline.N() - 6;
  const Eigen::MatrixXd P1 = inversePositiveDefinite(A1.topLeftCorner(p_eta, p_eta));
  const Eigen::MatrixXd P4 = inversePositiveDefinite(A4.topLeftCorner(p_eta, p_eta));
  if (P1.size() == 0 || P4.size() == 0) {
    check(false, "noise-scaling covariance matrices are invertible");
    return;
  }

  const double ratio = P4.trace() / std::max(P1.trace(), 1e-18);
  check(std::isfinite(ratio),
        "physical covariance noise-density scaling ratio is finite", ratio, 0.0);
  check(std::abs(ratio - 4.0) < 1e-6,
        "physical covariance scales linearly with noise density",
        ratio, 1e-6);
}

}  // namespace

int main()
{
  std::printf("Pose-control independent physical covariance reference tests\n");
  testIndependentPositionInformationReference();
  testIndependentPhysicalCovarianceReference();
  testCovarianceNoiseScaling();
  std::printf("%d failure(s)\n", failures);
  return failures ? 1 : 0;
}
