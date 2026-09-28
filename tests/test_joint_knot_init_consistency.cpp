#include "livo_recon/diagnostics/init_consistency.h"
#include "livo_recon/utils/algo/chi_square.h"

#include <cstdio>
#include <random>

using namespace livo_recon;

namespace
{
// Same NDEBUG-immune fail()-return convention as test_joint_knot_estimator.cpp
// (this repo builds permanent tests under Release/-DNDEBUG, which compiles
// out assert()'s argument entirely).
int fail(const char* name, double value = 0.0, double tol = 0.0)
{
  std::printf("[FAIL] %s %.6e tol %.6e\n", name, value, tol);
  return 1;
}
}  // namespace

int main()
{
  // ------------------------------------------------------------------
  // 1. 3-DOF block NEES against a known SPD matrix.
  // ------------------------------------------------------------------
  {
    // P_pp = diag(1,4,9), e_p = (1,2,3) -> eps_p = 1/1 + 4/4 + 9/9 = 3
    // exactly. Route through computeInitConsistency's own p-block path by
    // zeroing the R/V errors (R_ref==R, v_ref==v) and using a well-
    // conditioned P_RR/P_VV so only eps_p is under test.
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(9, 9);
    P.block<3, 3>(3, 3) = V3D(1.0, 4.0, 9.0).asDiagonal();
    const M3D R_ref = M3D::Identity(), R = M3D::Identity();
    const V3D v_ref = V3D::Zero(), v = V3D::Zero();
    const V3D p_ref = V3D::Zero(), p(1.0, 2.0, 3.0);
    const auto res = computeInitConsistency(R_ref, p_ref, v_ref, R, p, v, P);
    if (!res.pp_solve_valid) return fail("3-DOF NEES: pp_solve_valid");
    if (std::abs(res.eps_p - 3.0) >= 1e-10)
      return fail("3-DOF NEES against known SPD matrix", res.eps_p - 3.0, 1e-10);
  }

  // ------------------------------------------------------------------
  // 2. 9-DOF joint NEES including nonzero cross-covariances, checked
  //    against an independently-computed explicit inverse (not the
  //    production LDLT path) on the same SPD matrix.
  // ------------------------------------------------------------------
  {
    Eigen::MatrixXd A(9, 9);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < 9; ++i)
      for (int j = 0; j < 9; ++j) A(i, j) = dist(rng);
    Eigen::MatrixXd P = A * A.transpose() + 1e-2 * Eigen::MatrixXd::Identity(9, 9);

    const M3D R_ref = Exp(V3D(0.3, -0.2, 0.5));
    const M3D R = R_ref * Exp(V3D(0.01, 0.02, -0.015));
    const V3D p_ref(0.1, -0.2, 0.3), p(0.12, -0.18, 0.31);
    const V3D v_ref(0.0, 0.0, 0.0), v(0.01, -0.005, 0.002);
    const auto res = computeInitConsistency(R_ref, p_ref, v_ref, R, p, v, P);
    if (!res.rpv_solve_valid) return fail("9-DOF NEES: rpv_solve_valid");

    Eigen::VectorXd e(9);
    e << Log(M3D(R_ref.transpose() * R)), (p - p_ref), (v - v_ref);
    const double expected = e.dot(P.inverse() * e);
    if (std::abs(res.eps_RPV - expected) >= 1e-6 * std::max(1.0, std::abs(expected)))
      return fail("9-DOF joint NEES vs explicit inverse", res.eps_RPV - expected, 1e-6);
  }

  // ------------------------------------------------------------------
  // 3. Gravity-axis yaw/tilt projection at a nonidentity attitude, using an
  //    EXACT (not small-angle) decomposition: Log(Exp(x)) == x for any x
  //    within Exp's domain, so picking dtheta = yaw*a + tilt (tilt perp a)
  //    gives an exactly-known eR.
  // ------------------------------------------------------------------
  {
    const M3D R_ref = Exp(V3D(0.7, -1.1, 0.4));
    const V3D a = (R_ref.transpose() * V3D(0.0, 0.0, -1.0)).normalized();
    const V3D tilt_true = a.unitOrthogonal() * 0.05 + a.cross(a.unitOrthogonal()) * (-0.03);
    const double yaw_true = 0.08;
    const V3D dtheta = yaw_true * a + tilt_true;  // tilt_true already perp a
    const M3D R = R_ref * Exp(dtheta);

    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(9, 9);
    const auto res = computeInitConsistency(
        R_ref, V3D::Zero(), V3D::Zero(), R, V3D::Zero(), V3D::Zero(), P);
    if ((res.eR - dtheta).norm() >= 1e-10)
      return fail("gravity-axis projection: eR recovers exact dtheta",
                  (res.eR - dtheta).norm(), 1e-10);
    if (std::abs(res.yaw_error - yaw_true) >= 1e-10)
      return fail("gravity-axis projection: yaw_error", res.yaw_error - yaw_true, 1e-10);
    if ((res.tilt - tilt_true).norm() >= 1e-10)
      return fail("gravity-axis projection: tilt vector",
                  (res.tilt - tilt_true).norm(), 1e-10);
  }

  // ------------------------------------------------------------------
  // 4. Invariance of the 2D tilt-plane NEES/eigenvalues under a change of
  //    orthonormal basis for the plane perpendicular to the gravity axis --
  //    a similarity transform (P' = Q^T P Q, e' = Q^T e for orthogonal Q)
  //    leaves both the quadratic form and the eigenvalue SET unchanged.
  // ------------------------------------------------------------------
  {
    const V3D a = V3D(0.2, -0.5, 0.8).normalized();
    const V3D u1 = a.unitOrthogonal();
    const V3D u2 = a.cross(u1);
    Eigen::Matrix<double, 3, 2> U1;
    U1.col(0) = u1; U1.col(1) = u2;
    // A second, rotated orthonormal basis for the same plane.
    const double theta = 0.9;
    Eigen::Matrix2d Q;
    Q << std::cos(theta), -std::sin(theta), std::sin(theta), std::cos(theta);
    const Eigen::Matrix<double, 3, 2> U2 = U1 * Q;

    const M3D P_RR = (V3D(0.5, 2.0, 1.3).asDiagonal().toDenseMatrix() +
                      0.3 * V3D(1, 1, 1) * V3D(1, 1, 1).transpose());
    const V3D e3(0.02, -0.015, 0.03);
    const V3D e_tilt3 = e3 - a.dot(e3) * a;

    const Eigen::Matrix2d P1 = U1.transpose() * P_RR * U1;
    const Eigen::Matrix2d P2 = U2.transpose() * P_RR * U2;
    const Eigen::Vector2d e1 = U1.transpose() * e_tilt3;
    const Eigen::Vector2d e2 = U2.transpose() * e_tilt3;

    Eigen::LDLT<Eigen::Matrix2d> ldlt1(P1), ldlt2(P2);
    if (!(ldlt1.info() == Eigen::Success && ldlt1.isPositive()) ||
        !(ldlt2.info() == Eigen::Success && ldlt2.isPositive()))
      return fail("tilt basis invariance: both bases must be PD");
    const double eps1 = e1.dot(ldlt1.solve(e1));
    const double eps2 = e2.dot(ldlt2.solve(e2));
    if (std::abs(eps1 - eps2) >= 1e-10)
      return fail("tilt-plane NEES invariant under orthonormal basis change",
                  eps1 - eps2, 1e-10);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es1(P1), es2(P2);
    const Eigen::Vector2d ev1 = es1.eigenvalues(), ev2 = es2.eigenvalues();
    if ((ev1 - ev2).norm() >= 1e-10)
      return fail("tilt-plane eigenvalues invariant under basis change",
                  (ev1 - ev2).norm(), 1e-10);
  }

  // ------------------------------------------------------------------
  // 5. Invalid/non-positive-definite covariance reporting: preserve the
  //    row (eigenvalues), mark the solve invalid, never regularize.
  // ------------------------------------------------------------------
  {
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(9, 9);
    P.block<3, 3>(3, 3) = V3D(1.0, -2.0, 3.0).asDiagonal();  // non-PD P_pp
    const auto res = computeInitConsistency(
        M3D::Identity(), V3D::Zero(), V3D::Zero(),
        M3D::Identity(), V3D(1.0, 2.0, 3.0), V3D::Zero(), P);
    if (res.pp_solve_valid) return fail("non-PD P_pp must be reported invalid");
    if (res.eps_p != 0.0) return fail("non-PD P_pp: eps_p must not be silently computed", res.eps_p);
    // Eigenvalues must be preserved verbatim (including the negative one),
    // not clamped/regularized to something positive.
    bool saw_negative = false;
    for (double e : res.Ppp_eig) if (e < 0.0) saw_negative = true;
    if (!saw_negative) return fail("non-PD P_pp: negative eigenvalue must be preserved, not regularized");
    for (bool f : res.chi2_p) if (f) return fail("non-PD P_pp: chi2 flags must all be false when invalid");
  }

  // ------------------------------------------------------------------
  // 6. Correct separation of post_imu and final post_lio: two distinct
  //    (state, covariance) inputs must not alias into the same result, and
  //    the error/covariance contraction comparison must use the actual
  //    final state, not an intermediate one.
  // ------------------------------------------------------------------
  {
    const M3D R_ref = M3D::Identity();
    const V3D p_ref = V3D::Zero(), v_ref = V3D::Zero();
    // post_imu: larger error, larger covariance.
    const V3D p_imu(0.10, 0.0, 0.0);
    Eigen::MatrixXd P_imu = 1e-2 * Eigen::MatrixXd::Identity(9, 9);
    // post_lio: smaller error, smaller covariance (both contracted).
    const V3D p_lio(0.01, 0.0, 0.0);
    Eigen::MatrixXd P_lio = 1e-4 * Eigen::MatrixXd::Identity(9, 9);

    const auto imu = computeInitConsistency(R_ref, p_ref, v_ref, R_ref, p_imu, v_ref, P_imu);
    const auto lio = computeInitConsistency(R_ref, p_ref, v_ref, R_ref, p_lio, v_ref, P_lio);
    if (std::abs(imu.ep_norm - lio.ep_norm) < 1e-9)
      return fail("post_imu/post_lio separation: results must differ for differing inputs");

    const double error_before = std::sqrt(imu.eR_norm * imu.eR_norm +
        imu.ep_norm * imu.ep_norm + imu.ev_norm * imu.ev_norm);
    const double error_after = std::sqrt(lio.eR_norm * lio.eR_norm +
        lio.ep_norm * lio.ep_norm + lio.ev_norm * lio.ev_norm);
    const double cov_before = imu.Prr_trace + imu.Ppp_trace + imu.Pvv_trace;
    const double cov_after = lio.Prr_trace + lio.Ppp_trace + lio.Pvv_trace;
    if (!(error_after < error_before && cov_after < cov_before))
      return fail("post_imu->post_lio: constructed scenario must show both contracted");
  }

  // ------------------------------------------------------------------
  // Chi-square inverse CDF sanity: the exact DOF=1 identity
  // P(|z|<=1)=0.6827 <=> chi2_inv(0.6827,1)==1.
  // ------------------------------------------------------------------
  {
    const double q = chiSquareInverseCdf(0.6827, 1);
    if (std::abs(q - 1.0) >= 1e-3)
      return fail("chi-square DOF=1 68% quantile is exactly 1.0", q - 1.0, 1e-3);
    // Monotonic pair check.
    if (!(chiSquareInverseCdf(0.95, 3) > chiSquareInverseCdf(0.68 /*approx*/, 3)))
      return fail("chi-square quantile must increase with confidence level");
  }

  std::printf("[PASS] test_joint_knot_init_consistency\n");
  return 0;
}
