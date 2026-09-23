#include "livo_recon/lio/pose_control_covariance.h"

namespace livo_recon
{

bool schurComplementFreeCovariance(
    const Eigen::MatrixXd& A_hh, const Eigen::MatrixXd& A_hf,
    const Eigen::MatrixXd& A_ff, Eigen::MatrixXd& P_free)
{
  Eigen::LDLT<Eigen::MatrixXd> ldlt_hh(A_hh);
  if (ldlt_hh.info() != Eigen::Success) return false;
  const Eigen::MatrixXd A_hh_inv_A_hf = ldlt_hh.solve(A_hf);
  if (!A_hh_inv_A_hf.allFinite()) return false;

  const Eigen::MatrixXd A_ff_schur = A_ff - A_hf.transpose() * A_hh_inv_A_hf;

  Eigen::LDLT<Eigen::MatrixXd> ldlt_ff(A_ff_schur);
  if (ldlt_ff.info() != Eigen::Success) return false;
  P_free = ldlt_ff.solve(Eigen::MatrixXd::Identity(A_ff.rows(), A_ff.cols()));
  return P_free.allFinite();
}

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

}  // namespace livo_recon
