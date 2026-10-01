#pragma once
// R73 (F-419): the ONE way a covariance (or information) matrix is symmetrised in this code base.
//   symmetrized(X)  returns 0.5 (X + X^T) as an evaluated temporary (safe to assign back to X: no aliasing).
//   symmetrizeInPlace(X) does the same on X itself.
// Standing rule 3.12: every assignment to a covariance, in any file, goes through symmetrized(); the plain product F P F^T + Q is
// never stored. tools/r69/r73_symmetrize_gate.py greps the tree and fails when a covariance assignment or a hand-written
// 0.5*(X+X^T) appears outside this header.
#include <Eigen/Dense>

namespace livo_recon
{
template <typename Derived>
inline typename Derived::PlainObject symmetrized(const Eigen::MatrixBase<Derived>& X)
{
  typename Derived::PlainObject S = X;                  // evaluate once (X may be an expression that references the destination)
  S = 0.5 * (S + S.transpose().eval());
  return S;
}

template <typename Derived>
inline void symmetrizeInPlace(Eigen::MatrixBase<Derived>& X)
{
  X = symmetrized(X);
}

// Largest absolute asymmetry |X_ij - X_ji|, for the diagnostics that report it.
template <typename Derived>
inline double asymmetryMax(const Eigen::MatrixBase<Derived>& X)
{
  return (X - X.transpose()).cwiseAbs().maxCoeff();
}
}  // namespace livo_recon
