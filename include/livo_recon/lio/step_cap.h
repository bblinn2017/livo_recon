#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <string>

// ============================================================================
// R72 (F-403, F-391): opt-in hard cap on the size of one iterated-EKF update step.
//
// WHY. R71 (F-403) found single iterations of the iekf update that move the state by far more than the scan-to-scan motion.
// A cap is an experiment, never a default (Bryce, F-391): it tests whether the large steps are what produces the excursion.
//
// WHAT. `solution` is the per-iteration correction EkfUpdate::applyMeanUpdate is about to apply (rotation rows idx_r..+2 in
// rad, position rows idx_p..+2 in m). With
//   s = min(1, cap_rot / ||solution_rot||, cap_pos / ||solution_pos||)
// the applied step is s * solution. One scalar for the whole vector keeps the DIRECTION of the step (a per-component clamp
// would rotate it). s == 1 exactly when both norms are within their caps, so a step inside the caps is untouched.
//
// WHERE. EkfUpdate::applyMeanUpdate, only when EkfUpdate::step_cap.on(). mode=off (default) never evaluates this
// (bit-identical). iekf estimator only; the coupled estimator does not use EkfUpdate.
// ============================================================================

namespace livo_recon
{

struct StepCapOptions
{
  std::string mode = "off";     // "off" | "hard"
  double rotation_rad = 1e-2;   // cap on ||rotation step||
  double position_m = 1e-2;     // cap on ||position step||
  bool on() const { return mode != "off"; }
};

struct StepCapResult
{
  double rot_norm = 0.0;   // raw step norms (before the cap)
  double pos_norm = 0.0;
  double scale = 1.0;      // s in (0, 1]
  bool bound = false;      // true when s < 1
};

inline StepCapResult stepCapScale(const StepCapOptions& o, const Eigen::VectorXd& solution, int idx_r, int idx_p)
{
  StepCapResult r;
  r.rot_norm = solution.segment<3>(idx_r).norm();
  r.pos_norm = solution.segment<3>(idx_p).norm();
  double s = 1.0;
  if (r.rot_norm > o.rotation_rad) s = std::min(s, o.rotation_rad / r.rot_norm);
  if (r.pos_norm > o.position_m) s = std::min(s, o.position_m / r.pos_norm);
  r.scale = s;
  r.bound = s < 1.0;
  return r;
}

}  // namespace livo_recon
