#pragma once

#include <Eigen/Dense>
#include <string>

// ============================================================================
// R70 (F-320, F-340): separate rotation and position scales on the LiDAR information.
//
// WHY. The global information scale F (lio/bootstrap_inflate with scans=100000) divides every admitted residual's weight by F.
// R69B (F-334, F-336): with combined pose covariance the F at which stationary rotation NEES reaches its expected value (3)
// is about 4 to 12, the F for position NEES about 1.5 to 2.3; with the sensor-only plane fit rotation needs F > 1000 and
// position about 21. One scalar cannot fit both blocks, and a large F costs offset and ATE (F-307).
//
// WHAT. With Lambda = HtH (6x6, rows 0-2 rotation, 3-5 position) and g = Htz as built from the admitted residuals, and
// S = diag(1/sqrt(F_rot) I3, 1/sqrt(F_pos) I3):
//   Lambda' = S Lambda S          g' = Lambda' pinv(Lambda) g
// g' keeps the measurement-only least-squares solution pinv(Lambda) g unchanged and changes only how much information backs it
// (block by block) against the prior. For F_rot = F_pos = F this equals Lambda/F and g/F, i.e. the global F exactly (up to
// rounding). Cross rotation-position information is scaled by 1/sqrt(F_rot F_pos), which keeps Lambda' positive semidefinite.
// pinv uses the numerical-rank cut 6 eps lambda_max (not a tuning constant); discarded modes are counted and logged.
//
// WHERE. LioProcBase::solveSystem / solveSystem_cuda after accumulation and the redundancy module, before applyMeanUpdate, so
// the mean update, the covariance update and every HtH-based diagnostic see the scaled quantities. iekf estimator only.
// mode=off (default) never touches HtH/Htz (bit-identical).
// ============================================================================

namespace livo_recon
{

struct InfoScaleOptions
{
  std::string mode = "off";   // "off" | "block"
  double rotation = 1.0;      // F_rot >= 1
  double position = 1.0;      // F_pos >= 1
  bool on() const { return mode != "off"; }
};

struct InfoScaleStats
{
  double trace_before = 0.0, trace_after = 0.0;
  double gnorm_before = 0.0, gnorm_after = 0.0;
  int discarded_modes = 0;
};

// In place on (HtH, Htz). Returns magnitudes for the engagement log.
InfoScaleStats applyBlockInfoScale(const InfoScaleOptions& opts,
                                   Eigen::Matrix<double, 6, 6>& HtH,
                                   Eigen::Matrix<double, 6, 1>& Htz);

}  // namespace livo_recon
