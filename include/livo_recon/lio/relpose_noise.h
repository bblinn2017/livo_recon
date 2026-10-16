#pragma once

#include <vector>

#include "livo_recon/utils/algo/math.h"

// R97A R2: relative-pose process-noise term for the deskew's point covariance
// (imu/undistort/time_based_process_noise = relpose_q | relpose_full). See
// R97A_HANDOFF/code_specs/R2_relpose.md for the derivation. Pure math, no ROS/StateGroup
// dependency, so it is unit-testable against a synthetic set of steps.
namespace livo_recon
{

// One IMU propagation step's process-noise blocks, ALREADY expressed in the frame the caller
// wants (deskewPoints() rotates the raw world-frame cov_w blocks into the scan-end body frame
// before building this record -- see R2's own "rotate them into the body axes of the scan end").
struct RelposeStepNoise
{
  double t = 0.0;    // step start time t_k
  double dt = 0.0;   // step duration
  M3D Q_theta = M3D::Zero();
  M3D Q_v     = M3D::Zero();
  M3D Q_p     = M3D::Zero();  // zero when imu/second_order is off
  M3D Q_pv    = M3D::Zero();  // zero when imu/second_order is off

  // (t_end - t_k)^2 * Q_v + Q_p + (t_end - t_k) * (Q_pv + Q_pv^T), the per-step contribution to
  // Sigma_p -- depends on t_end, computed once per scan by buildRelposeSuffixTables().
};

// Direct O(K) sum, per R2's own definition (including the straddling-step fraction f) -- the
// brute-force reference the suffix-sum form is tested against.
void relposeProcessNoiseBruteForce(
    const std::vector<RelposeStepNoise>& steps, double t_point, double t_end,
    M3D& sigma_theta, M3D& sigma_p);

// Suffix-sum tables, built once per scan (O(K)); relposeProcessNoiseFromSuffix() then answers
// each point in O(log K) via binary search on step start times.
struct RelposeSuffixTables
{
  std::vector<double> step_start;     // steps[k].t, ascending
  std::vector<double> step_end;       // steps[k].t + steps[k].dt
  std::vector<M3D> term_theta;        // term_theta[k] = Q_theta_k
  std::vector<M3D> term_p;            // (t_end-t_k)^2 Q_v_k + Q_p_k + (t_end-t_k)(Q_pv_k+Q_pv_k^T)
  std::vector<M3D> suffix_theta;      // suffix_theta[k] = sum_{j=k}^{K-1} term_theta[j]; size K+1
  std::vector<M3D> suffix_p;          // suffix_p[k] = sum_{j=k}^{K-1} term_p[j]; size K+1
};

RelposeSuffixTables buildRelposeSuffixTables(
    const std::vector<RelposeStepNoise>& steps, double t_end);

// O(log K): binary search for the straddling step, fraction f, suffix lookup.
void relposeProcessNoiseFromSuffix(
    const RelposeSuffixTables& tabs, double t_point, M3D& sigma_theta, M3D& sigma_p);

}  // namespace livo_recon
