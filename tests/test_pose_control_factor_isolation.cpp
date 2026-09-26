#include "livo_recon/diagnostics/pose_control/pose_control_physical_diagnostics.h"

#include <Eigen/Core>
#include <cmath>
#include <cstdio>

using namespace livo_recon;

namespace {
int failures = 0;
void check(bool ok, const char* name, double value = 0.0, double tol = 0.0)
{
  if (!ok) {
    ++failures;
    std::printf("  [FAIL] %s %.6e tol %.6e\n", name, value, tol);
  } else {
    std::printf("  [PASS] %s %.6e tol %.6e\n", name, value, tol);
  }
}
}

int main()
{
  std::printf("Pose-control factor-isolation and GT-direction diagnostics tests\n");

  {
    const V3D before(1.0, 2.0, 3.0);
    const V3D after(2.0, 2.0, 3.0);
    const V3D gt(4.0, 2.0, 3.0);
    const auto m = computePoseControlCorrectionMetrics(before, after, gt);
    check(m.gt_direction_valid, "GT direction is valid");
    check(std::abs(m.gt_cosine - 1.0) < 1e-12, "aligned correction cosine", m.gt_cosine, 1e-12);
    check(std::abs(m.gt_parallel - 1.0) < 1e-12, "aligned parallel component", m.gt_parallel, 1e-12);
    check(std::abs(m.gt_perpendicular) < 1e-12, "aligned perpendicular component", m.gt_perpendicular, 1e-12);
    check(std::abs(m.gt_error_reduction - 1.0) < 1e-12, "aligned GT error reduction", m.gt_error_reduction, 1e-12);
  }

  {
    const V3D before = V3D::Zero();
    const V3D after(1.0, 0.0, 0.0);
    const V3D gt(0.0, 1.0, 0.0);
    const auto m = computePoseControlCorrectionMetrics(before, after, gt);
    check(std::abs(m.gt_cosine) < 1e-12, "orthogonal correction cosine", m.gt_cosine, 1e-12);
    check(std::abs(m.gt_parallel) < 1e-12, "orthogonal parallel component", m.gt_parallel, 1e-12);
    check(std::abs(m.gt_perpendicular - 1.0) < 1e-12, "orthogonal perpendicular component", m.gt_perpendicular, 1e-12);
    // FIX (test-only, no production change): a purely orthogonal correction
    // does NOT leave GT distance unchanged -- by Pythagoras, new_distance =
    // sqrt(gt_distance_before^2 + correction_norm^2) always exceeds
    // gt_distance_before for any nonzero orthogonal step (here
    // sqrt(1^2+1^2)=sqrt(2), so reduction = 1-sqrt(2) = -0.41421356...).
    // The original assertion here (expecting ~0) was mathematically
    // incorrect; computePoseControlCorrectionMetrics()'s own formula is
    // unchanged and independently verified to satisfy this identity, matching
    // the same orthogonal-case check already established in
    // test_pose_control_correction_diagnostics.cpp ("orthogonal correction
    // increases GT distance").
    check(m.gt_error_reduction < 0.0 && std::abs(m.gt_error_reduction - (1.0 - std::sqrt(2.0))) < 1e-12,
          "orthogonal correction increases GT distance by the Pythagorean amount",
          m.gt_error_reduction, 1e-12);
  }

  {
    const V3D before = V3D::Zero();
    const V3D after(-1.0, 0.0, 0.0);
    const V3D gt(1.0, 0.0, 0.0);
    const auto m = computePoseControlCorrectionMetrics(before, after, gt);
    check(std::abs(m.gt_cosine + 1.0) < 1e-12, "away correction cosine", m.gt_cosine, 1e-12);
    check(std::abs(m.gt_error_reduction + 1.0) < 1e-12, "away GT error reduction", m.gt_error_reduction, 1e-12);
  }

  {
    Eigen::Matrix2d A;
    A << 4.0, 1.0, 1.0, 3.0;
    Eigen::Vector2d b(5.0, 4.0);
    const auto s = solvePoseControlFactorStep(A, b, 1e-12);
    const Eigen::Vector2d ref = A.inverse() * b;
    check(s.effective_rank == 2, "factor solve retains full-rank system", s.effective_rank, 0.0);
    check((s.delta - ref).norm() < 1e-12, "factor solve matches independent inverse", (s.delta - ref).norm(), 1e-12);
  }

  {
    Eigen::Matrix2d A;
    A << 4.0, 0.0, 0.0, 1e-15;
    Eigen::Vector2d b(4.0, 1.0);
    const auto s = solvePoseControlFactorStep(A, b, 1e-12);
    check(s.effective_rank == 1, "weak factor mode is explicitly thresholded", s.effective_rank, 0.0);
    check(std::abs(s.delta(0) - 1.0) < 1e-12, "strong factor component retained", s.delta(0), 1e-12);
    check(std::abs(s.delta(1)) < 1e-12, "thresholded weak factor component suppressed", s.delta(1), 1e-12);
  }

  // Verify that a joint quadratic solution is generally NOT the sum of
  // independent factor-only solutions, which is exactly why factor isolation
  // must be logged rather than inferred from the final step.
  {
    Eigen::Matrix2d Al, Ai;
    Al << 3.0, 1.0, 1.0, 2.0;
    Ai << 2.0, -0.5, -0.5, 1.0;
    Eigen::Vector2d bl(1.0, 2.0), bi(2.0, -1.0);
    const auto sl = solvePoseControlFactorStep(Al, bl);
    const auto si = solvePoseControlFactorStep(Ai, bi);
    const auto sj = solvePoseControlFactorStep(Al + Ai, bl + bi);
    check((sj.delta - sl.delta - si.delta).norm() > 1e-3,
          "joint step is not generally the sum of factor-only steps",
          (sj.delta - sl.delta - si.delta).norm(), 1e-3);
  }

  return failures == 0 ? 0 : 1;
}
