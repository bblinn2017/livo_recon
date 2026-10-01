#pragma once

#include <Eigen/Dense>
#include <string>

// ============================================================================
// R72 (F-398): what the joint-knot solver knew at the moment a matrix failed its positive-definiteness check.
//
// WHY. R71's coupled arms aborted with "joint-knot information matrix is not positive definite" and left nothing behind: the matrix
// that failed, its spectrum and the terms that built it were gone with the stack. symmetricInverse() (joint_knot_estimator.cpp)
// now calls captureNonPd() just before it throws, so the caller that catches the exception can dump it.
//
// WHAT. The symmetrised matrix that was factorised, its eigenvalues (ascending), the LDLT pivots and the LDLT status. Thread-local,
// overwritten by the next failure; valid == false until a failure has happened. Nothing is written or copied on the success path.
// ============================================================================

namespace livo_recon
{

struct NonPdCapture
{
  bool valid = false;
  std::string where;             // "scan_head" | "prior_P" | "posterior_A"
  Eigen::MatrixXd matrix;        // the symmetrised matrix that was factorised
  Eigen::VectorXd eigenvalues;   // ascending
  Eigen::VectorXd ldlt_diag;     // LDLT pivots (D)
  int ldlt_info = 0;             // Eigen::ComputationInfo of the LDLT
  bool ldlt_positive = false;
};

inline NonPdCapture& lastNonPdCapture()
{
  static thread_local NonPdCapture capture;
  return capture;
}

template <typename LdltT>
inline void captureNonPd(const Eigen::MatrixXd& symmetrised, const LdltT& ldlt, const char* where)
{
  NonPdCapture& c = lastNonPdCapture();
  c.valid = true;
  c.where = where;
  c.matrix = symmetrised;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(symmetrised, Eigen::EigenvaluesOnly);
  c.eigenvalues = es.eigenvalues();
  c.ldlt_diag = ldlt.vectorD();
  c.ldlt_info = static_cast<int>(ldlt.info());
  c.ldlt_positive = ldlt.isPositive();
}

}  // namespace livo_recon
