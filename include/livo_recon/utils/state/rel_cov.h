#pragma once
// R65 (F-169). Noise accrued from each IMU step's head to the scan end, restricted to the 6x6 (R,P) block.
//
// For steps k = 0..n-1 with per-step transition F_k and process covariance Q_k (P_{k+1} = F_k P_k F_k^T + Q_k), the noise
// covariance accrued from the head of step k to the end of step n-1 is
//     N_k = sum_{j>=k} Phi_{j+1} Q_j Phi_{j+1}^T,   Phi_{j+1} = F_{n-1} ... F_{j+1}  (Phi_n = I),
// computed by one backward sweep:  N_k = N_{k+1} + M_{k+1} Q_k M_{k+1}^T,  M_k = M_{k+1} F_k,  M_n = I.
// Header-only and ROS-free so it can be unit tested against an independent forward simulation.
#include <Eigen/Dense>
#include <vector>

namespace livo_recon
{

inline std::vector<Eigen::Matrix<double, 6, 6>> relCovBlocksBackward(
    const std::vector<Eigen::MatrixXd>& F, const std::vector<Eigen::MatrixXd>& Q, int idx_r)
{
  const std::size_t n = F.size();
  std::vector<Eigen::Matrix<double, 6, 6>> out(n, Eigen::Matrix<double, 6, 6>::Zero());
  if (n == 0 || Q.size() != n) return out;
  const int dim = static_cast<int>(F[0].rows());
  Eigen::MatrixXd M = Eigen::MatrixXd::Identity(dim, dim);
  Eigen::MatrixXd N = Eigen::MatrixXd::Zero(dim, dim);
  for (int k = static_cast<int>(n) - 1; k >= 0; --k)
  {
    N += M * Q[k] * M.transpose();
    out[k] = N.block<6, 6>(idx_r, idx_r);
    M = M * F[k];
  }
  return out;
}

}  // namespace livo_recon
