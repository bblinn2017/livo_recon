// Phase 10 (pose_control_mechanism_validation_campaign): synthetic ablation
// of the mean-solver pseudo-inverse threshold, mirroring exactly the
// A_ff_priorS_pinv * xi_z_priorS computation at lio_coupled.cpp's
// delta_z_prior site (the ONE call site pose_control_mean_pinv_rel_thresh
// governs), with KNOWN weak-mode eigenvalues and gradient components so the
// "correct" answer in each scenario is derivable by construction, not by
// which threshold happens to give a lower real-data ATE.
//
// generalPseudoInverse(M, rel_thresh) discards any eigendirection whose
// eigenvalue is <= rel_thresh * lambda_max(M) (treated as structurally
// singular / zero information) and otherwise inverts it exactly. The OLD
// mean threshold (1e-6) is coarser -- it discards more directions as
// "structural" -- while the CORRECTED covariance threshold (1e-12) retains
// far more, including directions with real but small information.
#include "livo_recon/lio/pose_control_covariance.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <cstdio>
#include <string>

using namespace livo_recon;

namespace
{
int failures = 0;
void check(bool ok, const char* name, double value = 0.0, double tol = 0.0)
{
  std::printf("  [%s] %-95s %.6e tol %.6e\n", ok ? "PASS" : "FAIL", name, value, tol);
  if (!ok) ++failures;
}

constexpr double kOldMeanThresh = 1e-6;
constexpr double kCorrectedThresh = 1e-12;

// Build a diagonal information matrix (in its own eigenbasis, so the
// eigenvectors are just the standard basis and every quantity below is
// trivially exact) with one "dominant" well-conditioned direction and one
// deliberately weak direction whose relative eigenvalue is controlled by
// the caller.
Eigen::MatrixXd diagonalInfo(double lambda_dominant, double lambda_weak, int n = 4)
{
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
  A(0, 0) = lambda_dominant;
  A(1, 1) = lambda_weak;
  for (int i = 2; i < n; ++i) A(i, i) = lambda_dominant;   // keep the rest well-conditioned/irrelevant
  return A;
}

// Scenario A: the weak mode (index 1) is REAL and INFORMATIVE -- its
// relative eigenvalue (1e-9) sits strictly between the two thresholds, and
// it carries a genuine, bounded gradient component. The corrected
// (1e-12) threshold retains it and produces a finite, bounded step; the
// old (1e-6) threshold discards it entirely (zero step along that mode),
// which is the WRONG answer here because the direction is not actually
// singular -- it is just weakly observed.
void testWeakButRealInformativeMode()
{
  const double lambda_dominant = 1.0;
  const double lambda_weak = 1e-9;   // relative eigenvalue 1e-9: between 1e-12 and 1e-6
  const Eigen::MatrixXd A = diagonalInfo(lambda_dominant, lambda_weak);
  Eigen::VectorXd xi = Eigen::VectorXd::Zero(4);
  xi(1) = 1e-9;   // a genuine, small-but-nonzero gradient along the weak mode itself

  const Eigen::MatrixXd Ainv_old = generalPseudoInverse(A, kOldMeanThresh);
  const Eigen::MatrixXd Ainv_new = generalPseudoInverse(A, kCorrectedThresh);
  const Eigen::VectorXd step_old = Ainv_old * xi;
  const Eigen::VectorXd step_new = Ainv_new * xi;

  check(std::abs(step_old(1)) < 1e-15, "OLD (1e-6) threshold discards the weak-but-real mode entirely (step=0)", step_old(1), 1e-15);
  const double expected_new = xi(1) / lambda_weak;   // = 1.0, exact since eigenbasis is the standard basis
  check(std::abs(step_new(1) - expected_new) < 1e-6 * std::abs(expected_new),
        "CORRECTED (1e-12) threshold retains it and produces the exact, bounded, informative step", step_new(1), expected_new);
  check(std::isfinite(step_new(1)) && std::abs(step_new(1)) < 1e6,
        "the corrected-threshold step is finite and NOT a numerical blow-up -- this is genuine information, not noise",
        step_new(1));
}

// Scenario B: the weak mode is NUMERICALLY NEGLIGIBLE -- its relative
// eigenvalue (1e-14) sits below BOTH thresholds' reasonable floor for
// "real information" and its associated gradient component is pure
// floating-point-scale noise, not a real signal (in a real problem this is
// exactly a structural nullspace direction with residual roundoff, not an
// observed quantity). Retaining it and inverting would AMPLIFY that noise
// by 1/lambda into an enormous, meaningless step -- the defensible policy
// discards it, and BOTH thresholds here in fact agree it should be
// discarded, which is itself the point: 1e-12 is not "retain everything",
// it is "retain what current floating-point precision can still resolve",
// consistent with pose_control_q_pinv_rel_thresh's own documented margin
// in lio_coupled.h.
void testNumericallyNegligibleMode()
{
  const double lambda_dominant = 1.0;
  const double lambda_weak = 1e-14;   // relative eigenvalue below the 1e-12 floor
  const Eigen::MatrixXd A = diagonalInfo(lambda_dominant, lambda_weak);
  Eigen::VectorXd xi = Eigen::VectorXd::Zero(4);
  xi(1) = 1e-14;   // roundoff-scale, not a real observed gradient

  const Eigen::MatrixXd Ainv_old = generalPseudoInverse(A, kOldMeanThresh);
  const Eigen::MatrixXd Ainv_new = generalPseudoInverse(A, kCorrectedThresh);
  const Eigen::VectorXd step_old = Ainv_old * xi;
  const Eigen::VectorXd step_new = Ainv_new * xi;

  check(std::abs(step_old(1)) < 1e-15, "OLD threshold discards the negligible mode (step=0)", step_old(1), 1e-15);
  check(std::abs(step_new(1)) < 1e-15, "CORRECTED threshold ALSO discards it (1e-14 relative eigenvalue is below its own 1e-12 floor)", step_new(1), 1e-15);
  std::printf("  -> both thresholds agree here: 1e-12 is a precision floor, not \"retain everything.\"\n");
}

// Scenario C: LiDAR and IMU/prior factors actively disagree along a weak
// mode (mirrors the mode_gradient diagnostic's disagreement_strength
// concept, but applied to the mean solve). Construct the weak mode with a
// real (1e-9 relative) eigenvalue and a LARGER gradient than scenario A, so
// under the corrected threshold the resulting step is meaningfully larger
// -- i.e. genuine factor disagreement along a retained weak mode can
// produce a materially different mean trajectory than the old threshold's
// "discard and ignore" policy. This does not by itself say which policy is
// "right" for real data; it demonstrates the mechanism precisely.
void testStronglyDisagreeingFactorsAlongWeakMode()
{
  const double lambda_dominant = 1.0;
  const double lambda_weak = 1e-9;
  const Eigen::MatrixXd A = diagonalInfo(lambda_dominant, lambda_weak);
  // xi here represents the COMBINED (lidar+imu) gradient at this weak
  // mode; construct it as the sum of two individually large, oppositely-
  // signed contributions that happen to leave a modest net residual --
  // i.e. both factors want to move this mode substantially, and what's
  // left over (xi) is still non-trivial relative to lambda_weak.
  const double g_lidar = 5e-6, g_imu = -4.999e-6;   // large individually, small net
  Eigen::VectorXd xi = Eigen::VectorXd::Zero(4);
  xi(1) = g_lidar + g_imu;   // net residual gradient along the weak mode

  const Eigen::MatrixXd Ainv_old = generalPseudoInverse(A, kOldMeanThresh);
  const Eigen::MatrixXd Ainv_new = generalPseudoInverse(A, kCorrectedThresh);
  const Eigen::VectorXd step_old = Ainv_old * xi;
  const Eigen::VectorXd step_new = Ainv_new * xi;

  check(std::abs(step_old(1)) < 1e-15, "OLD threshold: net factor disagreement along this weak mode is silently thrown away", step_old(1), 1e-15);
  const double expected_new = xi(1) / lambda_weak;
  check(std::abs(step_new(1) - expected_new) < 1e-6 * std::abs(expected_new),
        "CORRECTED threshold: the net disagreement is retained and correctly amplified by 1/lambda_weak (exact, by construction)",
        step_new(1), expected_new);
  // Individual factor "wants" (g/lambda) are both far larger in magnitude
  // than the realized step -- this is exactly the tug-of-war geometry
  // mode_gradient's disagreement_strength is built to flag; the mean
  // solve's realized step is the NET of that tug-of-war, not either side's
  // individual demand.
  const double step_lidar_only = g_lidar / lambda_weak;
  const double step_imu_only = g_imu / lambda_weak;
  check(std::abs(step_lidar_only) > 100.0 * std::abs(step_new(1)) && std::abs(step_imu_only) > 100.0 * std::abs(step_new(1)),
        "each factor's individual pull is >>100x the realized net step -- genuine tug-of-war, small net residual",
        std::abs(step_lidar_only), 100.0 * std::abs(step_new(1)));
}

// Cross-cutting check: for a fully well-conditioned problem (no weak modes
// at all), the two thresholds must agree exactly -- the split must be a
// genuine no-op away from the weak-mode regime, matching lio_coupled.h's
// requirement that this new option default to identical behavior.
void testThresholdsAgreeAwayFromWeakModes()
{
  const Eigen::MatrixXd A = diagonalInfo(1.0, 0.5);   // no weak mode at all
  Eigen::VectorXd xi = Eigen::VectorXd::Zero(4);
  xi(0) = 0.3; xi(1) = -0.2; xi(2) = 0.1; xi(3) = 0.05;
  const Eigen::VectorXd step_old = generalPseudoInverse(A, kOldMeanThresh) * xi;
  const Eigen::VectorXd step_new = generalPseudoInverse(A, kCorrectedThresh) * xi;
  check((step_old - step_new).norm() < 1e-12, "away from any weak mode, OLD and CORRECTED thresholds produce IDENTICAL steps", (step_old - step_new).norm(), 1e-12);
}

}  // namespace

int main()
{
  std::printf("Pose-control mean-solver pseudo-inverse threshold synthetic ablation suite\n");
  testWeakButRealInformativeMode();
  testNumericallyNegligibleMode();
  testStronglyDisagreeingFactorsAlongWeakMode();
  testThresholdsAgreeAwayFromWeakModes();
  std::printf("%d failure(s)\n", failures);
  return failures ? 1 : 0;
}
