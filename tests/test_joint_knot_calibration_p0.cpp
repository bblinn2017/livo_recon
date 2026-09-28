// Isolated mathematical identities for the calibration-derived P0 model
// (src/processing/calib_processing.cpp's applyCalibrationDerivedP0()/
// covarianceOfMean()). CalibProc's own methods are private and tied to a
// live ROS node handle / sample buffer, so these tests independently
// reconstruct the same formulas (documented at each site below) rather than
// calling into production code -- the same "isolated identity" convention
// already used by test_joint_knot_estimator.cpp/test_joint_knot_init_consistency.cpp.
#include "livo_recon/utils/algo/math.h"

#include <Eigen/Eigenvalues>
#include <cstdio>
#include <random>

using namespace livo_recon;

namespace
{
int fail(const char* name, double value = 0.0, double tol = 0.0)
{
  std::printf("[FAIL] %s %.6e tol %.6e\n", name, value, tol);
  return 1;
}

double minEig(const Eigen::MatrixXd& M)
{
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (M + M.transpose()));
  return es.eigenvalues().minCoeff();
}
}  // namespace

int main()
{
  // Same construction as applyCalibrationDerivedP0(): gravity axis in the
  // aligned body frame, its skew map A = [g*axis]_x, and the transverse
  // tangent projector.
  const double g = 9.81;
  const V3D axis = V3D(0.2, -0.6, 0.77).normalized();
  M3D A;
  A << SKEW_SYM_MATRX(g * axis);
  const M3D A_pinv = -A / (g * g);
  const M3D tangent = M3D::Identity() - axis * axis.transpose();
  const double ambiguity_std = 0.0856;
  const M3D ambiguity_accel_cov = ambiguity_std * ambiguity_std * tangent;

  const M3D P_theta_amb = A_pinv * ambiguity_accel_cov * A_pinv.transpose();
  const M3D P_ba_amb = ambiguity_accel_cov;
  const M3D P_theta_ba_amb = -A_pinv * ambiguity_accel_cov;

  // ------------------------------------------------------------------
  // 1. The balanced ambiguity covariance (the full 6x6 [theta;ba] block) is
  //    symmetric PSD.
  // ------------------------------------------------------------------
  {
    Eigen::MatrixXd P6(6, 6);
    P6.block<3, 3>(0, 0) = P_theta_amb;
    P6.block<3, 3>(3, 3) = P_ba_amb;
    P6.block<3, 3>(0, 3) = P_theta_ba_amb;
    P6.block<3, 3>(3, 0) = P_theta_ba_amb.transpose();
    if ((P6 - P6.transpose()).norm() >= 1e-12)
      return fail("ambiguity covariance symmetric", (P6 - P6.transpose()).norm(), 1e-12);
    const double me = minEig(P6);
    if (me < -1e-12)
      return fail("ambiguity covariance PSD (min eigenvalue)", me, -1e-12);
  }

  // ------------------------------------------------------------------
  // 6. A zero-ambiguity PSD prior is made strictly PD by the same symmetric
  //    eigenvalue floor used by production before information-form solves.
  // ------------------------------------------------------------------
  {
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(18, 18);
    P.diagonal().setConstant(1e-4);
    P(0,0) = 0.0;  // exact null direction exposed by R43's ambiguity=0 cells
    P(1,1) = -1e-20;  // representative floating-point roundoff
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (P + P.transpose()));
    if (es.info() != Eigen::Success) return fail("P0 floor eigensolve succeeded");
    const double floor = 1e-12;
    P = es.eigenvectors() * es.eigenvalues().cwiseMax(floor).asDiagonal() *
        es.eigenvectors().transpose();
    const double me = minEig(P);
    if (me < floor * (1.0 - 1e-10))
      return fail("P0 numerical floor makes zero-ambiguity prior PD", me, floor);
  }

  // ------------------------------------------------------------------
  // 2. Tilt and b_a marginal variances are equal in acceleration-equivalent
  //    units: g^2 * P_theta_amb == P_ba_amb exactly (derivation: A_pinv =
  //    -skew(axis)/g, so A_pinv*tangent*A_pinv^T = tangent/g^2 using
  //    skew(axis)*tangent=skew(axis) and skew(axis)*skew(axis)^T=tangent
  //    for a unit axis).
  // ------------------------------------------------------------------
  {
    const M3D lhs = g * g * P_theta_amb;
    if ((lhs - P_ba_amb).norm() >= 1e-9 * P_ba_amb.norm())
      return fail("tilt/b_a marginal std equal in accel-equivalent units",
                  (lhs - P_ba_amb).norm(), 1e-9 * P_ba_amb.norm());
  }

  // ------------------------------------------------------------------
  // 3. A*dtheta + dba cancels along the modeled ambiguity direction: the
  //    model is A*dtheta=-e, dba=+e for e in the tangent plane, i.e.
  //    dtheta=-A_pinv*e (since A*A_pinv is the tangent-plane projector).
  // ------------------------------------------------------------------
  {
    const V3D e = tangent * V3D(0.03, -0.02, 0.05);  // an arbitrary tangent-plane vector
    const V3D dtheta = -A_pinv * e;
    const V3D dba = e;
    const V3D residual = A * dtheta + dba;
    if (residual.norm() >= 1e-10)
      return fail("A*dtheta + dba cancels along ambiguity direction",
                  residual.norm(), 1e-10);
  }

  // ------------------------------------------------------------------
  // 4. P_theta_ba has the compensating sign: since A*dtheta+dba==0 always
  //    (test 3), Cov(A*dtheta+dba, dba) must be exactly zero, i.e.
  //    A*P_theta_ba_amb + P_ba_amb == 0. A same-sign (non-compensating)
  //    cross-covariance would fail this identity.
  // ------------------------------------------------------------------
  {
    const M3D residual = A * P_theta_ba_amb + P_ba_amb;
    if (residual.norm() >= 1e-9 * P_ba_amb.norm())
      return fail("P_theta_ba has the compensating sign (A*P_theta_ba + P_ba == 0)",
                  residual.norm(), 1e-9 * P_ba_amb.norm());
  }

  // ------------------------------------------------------------------
  // 5. Bartlett covariance-of-the-mean estimator is symmetric PSD.
  //    Reimplements covarianceOfMean()'s Bartlett-window HAC construction
  //    directly against a synthetic AR(1)-correlated 3-vector sequence
  //    (temporally correlated, unlike white noise, to actually exercise the
  //    lag terms).
  // ------------------------------------------------------------------
  {
    std::mt19937 rng(777);
    std::normal_distribution<double> noise(0.0, 1.0);
    const int n = 500;
    std::vector<V3D> x(n);
    V3D state = V3D::Zero();
    const double phi = 0.7;  // AR(1) coefficient -- induces real autocorrelation
    for (int i = 0; i < n; ++i) {
      const V3D eps(noise(rng), noise(rng), noise(rng));
      state = phi * state + eps;
      x[i] = state;
    }
    V3D mean = V3D::Zero();
    for (const auto& v : x) mean += v;
    mean /= static_cast<double>(n);

    const int lags = 20;
    const int lmax = std::min(lags, n - 1);
    M3D spectral = M3D::Zero();
    for (int lag = 0; lag <= lmax; ++lag) {
      M3D gamma = M3D::Zero();
      for (int i = lag; i < n; ++i) {
        const V3D xi = x[i] - mean;
        const V3D xj = x[i - lag] - mean;
        gamma += xi * xj.transpose();
      }
      gamma /= static_cast<double>(n);
      if (lag == 0) {
        spectral += gamma;
      } else {
        const double w = 1.0 - static_cast<double>(lag) / static_cast<double>(lmax + 1);
        spectral += w * (gamma + gamma.transpose());
      }
    }
    M3D cov_mean = 0.5 * (spectral + spectral.transpose()) / static_cast<double>(n);
    if ((cov_mean - cov_mean.transpose()).norm() >= 1e-12)
      return fail("Bartlett covariance-of-the-mean symmetric",
                  (cov_mean - cov_mean.transpose()).norm(), 1e-12);
    // Project onto the PSD cone exactly as production code does, then verify
    // the projection actually is PSD (production discards negative
    // round-off eigenvalues rather than trusting the raw Bartlett estimate).
    Eigen::SelfAdjointEigenSolver<M3D> es(cov_mean);
    if (es.info() != Eigen::Success) return fail("Bartlett eigensolve succeeded");
    const M3D projected = es.eigenvectors() * es.eigenvalues().cwiseMax(0.0).asDiagonal() *
                           es.eigenvectors().transpose();
    if (minEig(projected) < -1e-12)
      return fail("Bartlett covariance-of-the-mean PSD after projection", minEig(projected), -1e-12);
  }

  std::printf("[PASS] test_joint_knot_calibration_p0\n");
  return 0;
}
