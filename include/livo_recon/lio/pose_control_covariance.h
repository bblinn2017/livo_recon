#pragma once

#include <Eigen/Dense>

// ============================================================================
// Information-form covariance update for the pose-control estimator.
//
// 2026-09-22 correction (supersedes the earlier schurComplementFreeCovariance
// + generalPseudoInverse(1e-2) approach, which used an ad hoc pseudo-inverse
// threshold to avoid a numerical blow-up -- that threshold silently
// converted weak-but-real information into ZERO covariance in some
// directions, which is not a principled fix, just a smaller symptom of the
// same underlying issue).
//
// The correct, threshold-free update is the stable Woodbury/Kalman-style
// covariance form. Given a prior covariance P_prior (SPD, possibly with a
// huge but FINITE condition number) and measurement information
//     Lambda_meas = S^T S     (S: any square-root factorization, e.g.
//                              eigendecomposition Lambda_meas=V D V^T,
//                              S = sqrt(D) V^T)
// the posterior covariance is
//     M = I + S P_prior S^T                      (always SPD: I + PSD)
//     Y = M^-1 S P_prior
//     P_post = P_prior - (S P_prior)^T Y
// This is the Woodbury identity for (P_prior^-1 + Lambda_meas)^-1 -- it
// NEVER inverts P_prior^-1+Lambda_meas directly (which can be
// ill-conditioned when Lambda_meas has near-zero eigenvalues in some
// directions and huge ones in others), only M (size = dim(z), but always
// well-conditioned since its eigenvalues are all >= 1). A weak measurement
// direction (S small there) leaves M's corresponding contribution small,
// so P_post naturally stays close to P_prior there -- no threshold needed,
// no risk of converting weak-but-nonzero information into exactly-zero
// covariance.
// ============================================================================

namespace livo_recon
{

// Diagnostics from covarianceInformationUpdate() -- reported, never
// silently discarded (item 5: "do not silently clamp negative covariance
// eigenvalues to zero... if a material negative eigenvalue occurs, fail
// the diagnostic and report").
struct CovarianceUpdateDiagnostics
{
  bool prior_symmetric = false, prior_psd = false;
  bool meas_symmetric = false, meas_psd = false;
  bool post_finite = false, post_symmetric = false, post_psd = false;
  double min_eig_prior = 0.0, min_eig_meas = 0.0, min_eig_post = 0.0;
  double max_eig_prior = 0.0, max_eig_meas = 0.0, max_eig_post = 0.0;
  double asymmetry_prior = 0.0, asymmetry_meas = 0.0, asymmetry_post = 0.0;
};

// Computes P_post = P_prior - (S P_prior)^T (I + S P_prior S^T)^-1 (S P_prior),
// symmetrized. S is built internally from Lambda_meas via a symmetric
// eigendecomposition (Lambda_meas = V*max(D,0)*V^T, S = sqrt(max(D,0))*V^T)
// -- negative eigenvalues from floating-point noise are clipped to 0 (NOT
// thresholded away as "no information", genuinely zero is zero), never
// silently forced positive. tol is the numerical tolerance for the
// symmetry/PSD diagnostic checks (NOT a pinv threshold -- nothing here
// discards weak information). Returns false only if P_post comes out
// non-finite; a PSD violation is reported in diag, not treated as failure
// on its own (the caller decides what to do with a diagnostic finding).
bool covarianceInformationUpdate(
    const Eigen::MatrixXd& P_prior, const Eigen::MatrixXd& Lambda_meas,
    Eigen::MatrixXd& P_post, CovarianceUpdateDiagnostics& diag, double tol = 1e-6);

// Dynamic-size SPD pseudo-inverse via eigendecomposition of the symmetrized
// matrix, relative eigenvalue floor: for eigenvalue lambda_k > rel_thresh *
// lambda_max, contributes (1/lambda_k) * v_k v_k^T; for lambda_k at or
// below that threshold, contributes EXACTLY ZERO (standard Moore-Penrose
// behavior) -- NOT a large value. Still used for computing P_z_prior FROM
// a prior/process-only information matrix.
//
// 2026-09-24 CLARIFICATION (found via test_pose_control_prior_math.cpp's
// testStructuralVsNumericalNullspace(), which numerically checks this):
// an EARLIER version of this comment said a discarded (near-zero-
// eigenvalue) direction represents "infinite prior variance". That is true
// only in the RESTRICTED-PSEUDO-INVERSE sense used when this matrix is
// later COMBINED with other information (e.g. covarianceInformationUpdate's
// Woodbury update reading a genuinely-zero row/column here as "this factor
// contributes nothing here, defer entirely to whatever else covers this
// direction") -- it is NOT true as a standalone claim about a query vector
// along that exact eigendirection: v_k^T * generalPseudoInverse(M) * v_k is
// EXACTLY ZERO for a discarded eigenvalue, not large. A weak-but-RETAINED
// eigenvalue (above rel_thresh) DOES get a large-but-finite 1/lambda_k
// contribution, which is the case the "infinite variance" language actually
// describes accurately. Read call sites accordingly: a hard structural
// zero and a numerically-weak-but-real mode are OPPOSITE outcomes here
// (zero vs. large), which is a different situation from thresholding away
// MEASUREMENT information in the posterior (what the old, superseded
// mechanism did and this file's covarianceInformationUpdate() above never
// does).
Eigen::MatrixXd generalPseudoInverse(const Eigen::MatrixXd& M, double rel_thresh);

}  // namespace livo_recon
