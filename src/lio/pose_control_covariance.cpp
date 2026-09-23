#include "livo_recon/lio/pose_control_covariance.h"

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
}  // namespace

bool covarianceInformationUpdate(
    const Eigen::MatrixXd& P_prior, const Eigen::MatrixXd& Lambda_meas,
    Eigen::MatrixXd& P_post, CovarianceUpdateDiagnostics& diag, double tol)
{
  const int n = static_cast<int>(P_prior.rows());
  eigCheck(P_prior, tol, diag.prior_symmetric, diag.prior_psd, diag.min_eig_prior, diag.max_eig_prior, diag.asymmetry_prior);
  eigCheck(Lambda_meas, tol, diag.meas_symmetric, diag.meas_psd, diag.min_eig_meas, diag.max_eig_meas, diag.asymmetry_meas);

  // S from a symmetric eigendecomposition of Lambda_meas -- negative
  // eigenvalues (floating-point noise only; a genuine negative eigenvalue
  // here would mean Lambda_meas itself is not a valid information matrix,
  // already flagged via diag.meas_psd) are clipped to exactly 0, not
  // thresholded away as "unobserved". This is NOT the discard-weak-
  // information mechanism this update replaces -- a clipped-to-0 eigenvalue
  // contributes literally zero rows to S, which correctly contributes zero
  // measurement information (not zero POSTERIOR covariance) in that
  // direction via the Woodbury form below.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_meas(0.5 * (Lambda_meas + Lambda_meas.transpose()));
  Eigen::VectorXd d = es_meas.eigenvalues().cwiseMax(0.0);
  const Eigen::MatrixXd S = d.cwiseSqrt().asDiagonal() * es_meas.eigenvectors().transpose();

  const Eigen::MatrixXd SP = S * P_prior;                          // n x n
  const Eigen::MatrixXd M = Eigen::MatrixXd::Identity(n, n) + SP * S.transpose();  // I + S P S^T, always SPD
  Eigen::LDLT<Eigen::MatrixXd> ldlt_M(M);
  if (ldlt_M.info() != Eigen::Success) return false;
  const Eigen::MatrixXd Y = ldlt_M.solve(SP);                      // M^-1 S P_prior
  if (!Y.allFinite()) return false;

  P_post = P_prior - SP.transpose() * Y;
  P_post = 0.5 * (P_post + P_post.transpose());
  diag.post_finite = P_post.allFinite();
  if (!diag.post_finite) return false;

  bool sym_unused;
  eigCheck(P_post, tol, sym_unused, diag.post_psd, diag.min_eig_post, diag.max_eig_post, diag.asymmetry_post);
  diag.post_symmetric = true;   // exactly symmetric by construction (explicit symmetrization above)
  return true;
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
