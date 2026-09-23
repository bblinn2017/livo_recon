#include "livo_recon/lio/pose_control_covariance.h"

namespace livo_recon
{

Eigen::MatrixXd generalPseudoInverse(const Eigen::MatrixXd& M, double rel_thresh)
{
  const int n = static_cast<int>(M.rows());
  const Eigen::MatrixXd Ms = 0.5 * (M + M.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Ms);
  const auto& evals = es.eigenvalues();
  const auto& evecs = es.eigenvectors();
  const double thresh = rel_thresh * std::max(evals.maxCoeff(), 0.0);
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n, n);
  for (int k = 0; k < n; ++k)
    if (evals(k) > thresh) out += (1.0 / evals(k)) * (evecs.col(k) * evecs.col(k).transpose());
  return out;
}

bool schurComplementFreeCovariance(
    const Eigen::MatrixXd& A_hh, const Eigen::MatrixXd& A_hf,
    const Eigen::MatrixXd& A_ff, Eigen::MatrixXd& P_free,
    double rel_thresh)
{
  const Eigen::MatrixXd A_hh_pinv = generalPseudoInverse(A_hh, rel_thresh);
  const Eigen::MatrixXd A_hh_inv_A_hf = A_hh_pinv * A_hf;
  if (!A_hh_inv_A_hf.allFinite()) return false;

  const Eigen::MatrixXd A_ff_schur = A_ff - A_hf.transpose() * A_hh_inv_A_hf;
  P_free = generalPseudoInverse(A_ff_schur, rel_thresh);
  return P_free.allFinite();
}

}  // namespace livo_recon
