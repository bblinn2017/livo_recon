#include "livo_recon/utils/algo/covariance_math.h"

namespace livo_recon
{
namespace
{
void eigCheck(const Eigen::MatrixXd& M, double tol, bool& sym, bool& psd,
              double& min_eig, double& max_eig, double& asym)
{
  asym = (M - M.transpose()).norm();
  sym = asym < tol * std::max(1.0, M.norm());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (M + M.transpose()));
  min_eig = es.eigenvalues().minCoeff();
  max_eig = es.eigenvalues().maxCoeff();
  psd = min_eig > -tol * std::max(1.0, max_eig);
}
}

bool covarianceInformationUpdate(
    const Eigen::MatrixXd& P_prior, const Eigen::MatrixXd& Lambda_meas,
    Eigen::MatrixXd& P_post, CovarianceUpdateDiagnostics& diag, double tol)
{
  const int n = static_cast<int>(P_prior.rows());
  eigCheck(P_prior, tol, diag.prior_symmetric, diag.prior_psd,
           diag.min_eig_prior, diag.max_eig_prior, diag.asymmetry_prior);
  eigCheck(Lambda_meas, tol, diag.meas_symmetric, diag.meas_psd,
           diag.min_eig_meas, diag.max_eig_meas, diag.asymmetry_meas);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
      0.5 * (Lambda_meas + Lambda_meas.transpose()));
  const Eigen::VectorXd d = es.eigenvalues().cwiseMax(0.0);
  const Eigen::MatrixXd S = d.cwiseSqrt().asDiagonal() * es.eigenvectors().transpose();
  const Eigen::MatrixXd SP = S * P_prior;
  const Eigen::MatrixXd M = Eigen::MatrixXd::Identity(n,n) + SP * S.transpose();
  Eigen::LDLT<Eigen::MatrixXd> ldlt(M);
  if (ldlt.info() != Eigen::Success) return false;
  const Eigen::MatrixXd Y = ldlt.solve(SP);
  if (!Y.allFinite()) return false;
  P_post = P_prior - SP.transpose() * Y;
  P_post = 0.5 * (P_post + P_post.transpose());
  diag.post_finite = P_post.allFinite();
  if (!diag.post_finite) return false;
  bool ignored = false;
  eigCheck(P_post, tol, ignored, diag.post_psd, diag.min_eig_post,
           diag.max_eig_post, diag.asymmetry_post);
  diag.post_symmetric = true;
  return true;
}

Eigen::MatrixXd generalPseudoInverse(const Eigen::MatrixXd& M,
                                     double relative_threshold)
{
  const int n = static_cast<int>(M.rows());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (M + M.transpose()));
  const double threshold = relative_threshold *
      std::max(es.eigenvalues().maxCoeff(), 0.0);
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n,n);
  for (int k = 0; k < n; ++k)
    if (es.eigenvalues()(k) > threshold)
      out.noalias() += (1.0 / es.eigenvalues()(k)) *
          es.eigenvectors().col(k) * es.eigenvectors().col(k).transpose();
  return out;
}

}  // namespace livo_recon
