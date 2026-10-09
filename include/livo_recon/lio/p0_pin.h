#pragma once
// R89 (Bryce 2026-10-09): known blocks of the post-calibration covariance P0 may be EXACTLY zero.
//
// Position, velocity and attitude are known exactly in the calibration and stationary windows, so their P0 blocks (and every cross term
// with them) can be zero. Nothing factorises or inverts P0: the first matrix that is inverted is the covariance after the scan's IMU
// propagation, which process noise has made positive. P0 only has to be positive SEMI-definite, and a matrix with rows and columns
// removed (set to zero) from a positive semi-definite matrix stays positive semi-definite.
//
// The numerical eigenvalue floor (calib/p0/numerical_eigenvalue_floor) therefore acts on the FREE states only (the biases, gravity, ...);
// a pinned block is written back as exact zero after the floor. With no pinned block stabilize() is the earlier CalibProc::stabilizeP0()
// operation for operation (same calls, same order), so a run without pinned blocks is bit-identical.
//
// ROS-free, Eigen only: tests/test_p0_pin_r89.cpp checks it.
#include "livo_recon/utils/algo/symmetrize.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace livo_recon
{
namespace p0
{

// A 3-state block of P (start index) whose variance is a known constant.
struct KnownBlock
{
  int start = 0;
  double variance = -1.0;   // < 0: not a known block (left to the construction); 0: pinned exactly at zero; > 0: constant diagonal
};

// Set every listed block (variance >= 0) to variance * I and clear every cross term of its rows and columns.
inline void applyKnownBlocks(Eigen::MatrixXd& P, const std::vector<KnownBlock>& blocks)
{
  for (const KnownBlock& b : blocks)
  {
    if (!(b.variance >= 0.0)) continue;
    P.middleRows(b.start, 3).setZero();
    P.middleCols(b.start, 3).setZero();
    P.block<3, 3>(b.start, b.start) = b.variance * Eigen::Matrix3d::Identity();
  }
}

// Indices of the states whose block is pinned at exactly zero.
inline std::vector<int> pinnedIndices(const std::vector<KnownBlock>& blocks)
{
  std::vector<int> idx;
  for (const KnownBlock& b : blocks)
    if (b.variance == 0.0)
      for (int k = 0; k < 3; ++k) idx.push_back(b.start + k);
  std::sort(idx.begin(), idx.end());
  return idx;
}

struct StabilizeResult
{
  double min_before = 0.0;
  double min_after = 0.0;
};

// Symmetrise P, raise every eigenvalue of the FREE part to at least `floor`, verify, and write the pinned rows and columns as exact zero.
inline StabilizeResult stabilize(Eigen::MatrixXd& Pfull, const std::vector<int>& pinned, double floor)
{
  std::vector<char> is_pinned(static_cast<std::size_t>(Pfull.rows()), 0);
  for (int i : pinned) is_pinned[static_cast<std::size_t>(i)] = 1;
  std::vector<int> free_idx;
  for (int i = 0; i < Pfull.rows(); ++i)
    if (!is_pinned[static_cast<std::size_t>(i)]) free_idx.push_back(i);
  const bool all_free = pinned.empty();

  Eigen::MatrixXd sub;
  if (!all_free)
  {
    const int n = static_cast<int>(free_idx.size());
    sub.resize(n, n);
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < n; ++c) sub(r, c) = Pfull(free_idx[static_cast<std::size_t>(r)], free_idx[static_cast<std::size_t>(c)]);
  }
  Eigen::MatrixXd& P = all_free ? Pfull : sub;

  P = symmetrized(P);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(P);
  if (es.info() != Eigen::Success) throw std::runtime_error("P0 stabilization eigensolve failed");
  StabilizeResult out;
  out.min_before = es.eigenvalues().minCoeff();
  if (out.min_before < floor)
  {
    const Eigen::VectorXd values = es.eigenvalues().cwiseMax(floor);
    P = es.eigenvectors() * values.asDiagonal() * es.eigenvectors().transpose();
    P = symmetrized(P);
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> check(P);
  if (check.info() != Eigen::Success) throw std::runtime_error("stabilized P0 verification eigensolve failed");
  out.min_after = check.eigenvalues().minCoeff();
  if (out.min_after < floor)
  {
    const double lift = floor - out.min_after + std::numeric_limits<double>::epsilon() * std::max(1.0, floor);
    P.diagonal().array() += lift;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> lifted(P);
    if (lifted.info() != Eigen::Success) throw std::runtime_error("lifted P0 verification eigensolve failed");
    out.min_after = lifted.eigenvalues().minCoeff();
  }

  if (!all_free)
  {
    Pfull.setZero();
    const int n = static_cast<int>(free_idx.size());
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < n; ++c) Pfull(free_idx[static_cast<std::size_t>(r)], free_idx[static_cast<std::size_t>(c)]) = P(r, c);
  }
  return out;
}

}  // namespace p0
}  // namespace livo_recon
