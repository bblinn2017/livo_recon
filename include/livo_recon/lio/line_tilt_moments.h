#pragma once
// ============================================================================
// R82 (D4, F-500): the plane-equivalent line / smooth row variance (VoxelPlane::computeLineTilt, R77/R80) on the DEBIASED fit, where the points are gone and only
// moment accumulators remain. ROS-free, Eigen only, so tests/test_line_tilt_r82.cpp can check it against the per-point reference.
//
// The reference (pca path, computeLineTilt): for row i (u_i the row direction, tilt axis b, and for smooth rows also axis a), with z_k = p_k - mean, equal weights w = 1/N,
//     g_b,k = (w / gap_b) (u_i u_b^T + u_b u_i^T) z_k,   g_d,k = -w u_i,   gap_x = floored(e_i - e_x) (|gap| >= fl, sign of the eigenvalue order),
//     L_i(1,1) = sum_k g_b^T C_k g_b,   L_i(1,2) = sum_k g_b^T C_k g_d,   L_i(2,2) = sum_k g_d^T C_k g_d,   (smooth: L_i(0,.) with g_a likewise),
//     X (hard cross matrix of rows 0 and 1) = the same sums with g of row 0 on the left and g of row 1 on the right.
// Everything is a sum over points of (linear in z_k)^T C_k (linear in z_k) or of (linear in z_k)^T C_k u, so it needs only
//     M0 = sum_k C_k,   M1_a = sum_k p_k(a) C_k,   M2_ab = sum_k p_k(a) p_k(b) C_k      (the debiased fit's Scov_, W_, V_: already accumulated)
// shifted to the mean:  U_ab = sum_k z_a z_b C_k = M2_ab - mu_b M1_a - mu_a M1_b + mu_a mu_b M0,   T_a = sum_k z_a C_k = M1_a - mu_a M0.
//   form "moment"   (exact): U and T as above. Equal to the per-point reference to rounding (the test checks 1e-9 relative).
//   form "mean_cov" (approximation): C_k replaced by the mean C = M0 / N for every point, so U_ab = N raw(a,b) C with raw = the centred scatter / N, and T_a = 0 (the points are centred).
//                    The cross terms L(.,2) vanish; the offset term is u^T C u / N. A HYPOTHESIS-grade shortcut, tested against "moment" on the same fit.
// The eigenvalue gaps are floored exactly as computeLineTilt floors them, so a line voxel's near-zero gap does not blow up.
// ============================================================================
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace livo_recon
{

struct LineTiltResult
{
  Eigen::Matrix3d L[3] = {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d X = Eigen::Matrix3d::Zero();   // hard rows only
  long floored = 0;                               // gaps lifted to the floor (same counting rule as computeLineTilt)
};

namespace line_tilt_detail
{
inline double flooredGap(const double e[3], int i, int a, double fl, bool count, long& floored)
{
  const double g = e[i] - e[a];
  const double m = std::max(std::fabs(g), fl);
  if (count && std::fabs(g) < fl) ++floored;
  return (i < a) ? -m : m;   // ascending eigenvalues: e_i <= e_a when i < a
}
inline Eigen::Matrix3d symM(const Eigen::Vector3d& ui, const Eigen::Vector3d& ub) { return ui * ub.transpose() + ub * ui.transpose(); }
}  // namespace line_tilt_detail

// Shared core. U[a][b] (3x3 each) = sum_k z_a z_b C_k, T[a] (3x3) = sum_k z_a C_k, M0 = sum_k C_k, all RAW sums over N points (not divided). `moment` selects nothing here: the caller
// passes the exact or the approximate U / T / M0.
inline LineTiltResult lineTiltFromSums(const Eigen::Vector3d u[3], const double e[3], bool smooth, double fl, double N, const Eigen::Matrix3d U[3][3],
                                       const Eigen::Matrix3d T[3], const Eigen::Matrix3d& M0)
{
  using namespace line_tilt_detail;
  LineTiltResult out;
  const int rows = smooth ? 3 : 2;
  const double invN2 = 1.0 / (N * N);
  int ia[3], ib[3];
  double gap_a[3] = {0, 0, 0}, gap_b[3] = {0, 0, 0};
  for (int i = 0; i < rows; ++i)
  {
    ia[i] = (i == 0) ? 1 : 0;
    ib[i] = (i == 2) ? 1 : 2;
    gap_a[i] = flooredGap(e, i, ia[i], fl, smooth, out.floored);   // hard rows do not use axis a: not counted
    gap_b[i] = flooredGap(e, i, ib[i], fl, true, out.floored);
  }
  // Q(Ma, Mb) = invN2 sum_ab Ma.col(a)^T U_ab Mb.col(b);  D(M, v) = invN2 sum_a M.col(a)^T T_a v;  S(v, w) = invN2 v^T M0 w.
  auto Q = [&](const Eigen::Matrix3d& Ma, const Eigen::Matrix3d& Mb) {
    double s = 0.0;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) s += Ma.col(a).dot(U[a][b] * Mb.col(b));
    return s * invN2;
  };
  auto D = [&](const Eigen::Matrix3d& M, const Eigen::Vector3d& v) {
    double s = 0.0;
    for (int a = 0; a < 3; ++a) s += M.col(a).dot(T[a] * v);
    return s * invN2;
  };
  auto S = [&](const Eigen::Vector3d& v, const Eigen::Vector3d& w) { return invN2 * v.dot(M0 * w); };
  Eigen::Matrix3d Mb[3], Ma[3];
  for (int i = 0; i < rows; ++i)
  {
    Mb[i] = symM(u[i], u[ib[i]]);
    Ma[i] = symM(u[i], u[ia[i]]);
    Eigen::Matrix3d& L = out.L[i];
    L(1, 1) = Q(Mb[i], Mb[i]) / (gap_b[i] * gap_b[i]);
    L(1, 2) = -D(Mb[i], u[i]) / gap_b[i];
    L(2, 2) = S(u[i], u[i]);
    if (smooth)
    {
      L(0, 0) = Q(Ma[i], Ma[i]) / (gap_a[i] * gap_a[i]);
      L(0, 1) = Q(Ma[i], Mb[i]) / (gap_a[i] * gap_b[i]);
      L(0, 2) = -D(Ma[i], u[i]) / gap_a[i];
    }
    L(1, 0) = L(0, 1);
    L(2, 0) = L(0, 2);
    L(2, 1) = L(1, 2);
  }
  if (!smooth)
  {
    // X: row 0 on the left, row 1 on the right (computeLineTilt's gbv[0], gdv[0] against gbv[1], gdv[1]); both rows tilt toward u2.
    out.X(1, 1) = Q(Mb[0], Mb[1]) / (gap_b[0] * gap_b[1]);
    out.X(1, 2) = -D(Mb[0], u[1]) / gap_b[0];
    out.X(2, 1) = -D(Mb[1], u[0]) / gap_b[1];
    out.X(2, 2) = S(u[0], u[1]);
  }
  return out;
}

// Exact-moment form. M1[a] = sum_k p_k(a) C_k, M2[a][b] = sum_k p_k(a) p_k(b) C_k, M0 = sum_k C_k (world coordinates, RAW sums), mean = true world mean.
inline LineTiltResult lineTiltFromMoments(const Eigen::Vector3d u[3], const double e[3], bool smooth, double fl, double N, const Eigen::Vector3d& mean,
                                          const Eigen::Matrix3d M2[3][3], const Eigen::Matrix3d M1[3], const Eigen::Matrix3d& M0)
{
  Eigen::Matrix3d U[3][3], T[3];
  for (int a = 0; a < 3; ++a)
  {
    T[a] = M1[a] - mean(a) * M0;
    for (int b = 0; b < 3; ++b) U[a][b] = M2[a][b] - mean(b) * M1[a] - mean(a) * M1[b] + mean(a) * mean(b) * M0;
  }
  return lineTiltFromSums(u, e, smooth, fl, N, U, T, M0);
}

// Mean-covariance approximation. raw_cov = centred scatter / N in world coordinates (sum_k z z^T / N); mean_cov = M0 / N.
inline LineTiltResult lineTiltFromMeanCov(const Eigen::Vector3d u[3], const double e[3], bool smooth, double fl, double N, const Eigen::Matrix3d& raw_cov,
                                          const Eigen::Matrix3d& mean_cov)
{
  Eigen::Matrix3d U[3][3], T[3];
  for (int a = 0; a < 3; ++a)
  {
    T[a].setZero();
    for (int b = 0; b < 3; ++b) U[a][b] = (N * raw_cov(a, b)) * mean_cov;
  }
  return lineTiltFromSums(u, e, smooth, fl, N, U, T, N * mean_cov);
}

}  // namespace livo_recon
