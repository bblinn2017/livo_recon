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

}  // namespace livo_recon
