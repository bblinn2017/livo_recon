#pragma once

#include "livo_recon/node_context.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/log/profiler.h"

#include <Eigen/Core>

namespace livo_recon
{

struct ImuProcOptions
{
  // History (12-15): see docs/livo_recon_changelog.md#include-livo_recon-processing-imu_processing.h-12
  bool   log_debug_en = false;
  // History (17-20): see docs/livo_recon_changelog.md#include-livo_recon-processing-imu_processing.h-17
  bool   log_qhat_en   = false;
  bool   second_order = true;

  // Runtime accelerometer/gyroscope process-noise model. "fixed" is the
  // production control. The dynamic modes use the stationary-calibration
  // variance as their floor and add a bounded motion-dependent variance.
  std::string process_noise_model = "fixed";  // fixed|isotropic|axis_aware
  // R97B: motion/beta (EMA smoothing of the per-step dynamic excitation
  // energy) and the acc/gyro_max_dynamic_variance clamp were both removed
  // -- the mid360 beta=0 vs beta=0.9 acc/gyro_scale sweep comparison found
  // no meaningful difference in best-achievable ATE or in acc/gyro_scale's
  // effect size either way, so the per-step instantaneous energy is used
  // directly and unclamped.
  // 2026-10-28: isotropic (scalar, broadcast to all 3 axes -- Constant(k) reproduces the old
  // scalar behavior exactly) or per-axis (anisotropic). config.yaml's imu/process_noise/motion/
  // {acc,gyro}_scale accepts EITHER a single number or a 3-element [x,y,z] array for this reason --
  // see imu_processing.cpp's loadScaleParam().
  Eigen::Vector3d motion_acc_scale = Eigen::Vector3d::Constant(0.3);   // R80 F-477: was 1.0
  Eigen::Vector3d motion_gyr_scale = Eigen::Vector3d::Constant(10.0);  // R80 F-477: was 1.0
  // 2026-10-29: how motion_{acc,gyr}_scale turns the debiased dynamic signal into a variance.
  // "quadratic" (the original/only form before this date): variance = scale^2 * debiased_energy,
  // i.e. variance grows with the SQUARE of the dynamic signal's magnitude (matches qscale_fit's
  // "s"/acc_scale/gyro_scale, the excess-ENERGY term). "linear" (new default): variance =
  // scale * debiased_magnitude, i.e. variance grows LINEARLY with magnitude (matches qscale_fit's
  // "w", the excess-MAGNITUDE term). qscale_fit's own joint fits this session found the two terms
  // highly collinear (X vs X2, r~0.9), and the real acc/gyro error-vs-motion-magnitude relationship
  // tracks the linear term, not the quadratic one -- see scratchpad/TODO_investigation.md. "linear"
  // is the new default; "quadratic" reproduces this file's exact pre-2026-10-29 behavior unchanged.
  std::string motion_scale_form = "linear";  // "quadratic" | "linear"

  // Per-IMU-step trace of the first three propagations (imu_first_scans.csv).
  bool first_scans_log_en = false;
  // R83B (A6/B5): per-IMU-step motion-Q + per-axis clamp flags, every scan, not just the first three
  // (motion_q_step.csv). Independent of first_scans_log_en. Default off.
  bool motion_q_step_log_en = false;
};

// T7-a: the two quantities the Myers-Tapley process-noise estimator needs,
// accumulated across one frame's IMU propagation steps.
//
//   Q-hat = (1/N) SUM dx_k dx_k^T  -  (1/N) SUM [ Phi P+_{k-1} Phi^T - P+_k ]
//
// dx_k is read on the LIO side (it is the posterior boxminus the propagated
// state, available only after the IEKF loop converges). The bracketed term is
// read here. Note that propagate() runs once per IMU SAMPLE, not once per
// frame, so the frame-level Phi is a product and the frame-level process
// noise is the recursion  A <- F A F^T + cov_w  -- not a sum of cov_w. Adding
// the cov_w's directly would understate the term by the amount the earlier
// steps' noise is amplified by the later steps' Jacobians, which is exactly
// the regime (high angular rate) the estimator is meant to be informative in.
//
// Reading CONSUMES the accumulator: it is zeroed on read, so one read per
// frame makes the accumulation span exactly one frame with no separate
// begin-frame call to keep in sync. Returns false when disarmed or when no
// propagation has happened since the last read.
//
// CQ-71 item 0: p_before is P at the START of this frame's propagation
// (before the first IMU sample is consumed) -- the fourth quantity the
// card's own information-budget ask needs, alongside phi_p_phit (F P F^T,
// the propagated prior) and accum_cov_w (Q_eff, the noise actually
// injected): phi_p_phit + accum_cov_w == P_after_IMU by construction.
bool imuProcQhatRead(Eigen::MatrixXd& phi_p_phit, Eigen::MatrixXd& accum_cov_w,
                      Eigen::MatrixXd& p_before);

// CQ-76 T2.1: a NON-DESTRUCTIVE read of the same p_before snapshot
// imuProcQhatRead() above consumes -- does not touch g_qhat_primed or
// g_qhat_accum_cov_w, so it can be called EARLIER in a scan (at Pi_ss's
// own construction site, before the solve) without disturbing the later,
// destructive imuProcQhatRead() call CQ-76 T1.0/T1.1 already added at the
// posterior-write site. Same g_qhat_enabled/imu_samples-empty gating as
// the read above (returns false, p_before left untouched, if not primed).
bool imuProcQhatPeekPBefore(Eigen::MatrixXd& p_before);

bool imuProcQhatPeekAll(Eigen::MatrixXd& p_before, Eigen::MatrixXd& phi_p_phit,
                        Eigen::MatrixXd& q_eff, Eigen::MatrixXd& p_after);

class ImuProc
{
public:
  explicit ImuProc(NodeContext& ctx);
  ~ImuProc();   // R72 (F-401): writes the engagement report

  std::string loadParameters(ros::NodeHandle& pnh);

  void propagate(MeasureGroup& mg);
  // History (78-82): see docs/livo_recon_changelog.md#include-livo_recon-processing-imu_processing.h-78
  void processIMU(MeasureGroup& mg);

private:
  StateGroupPtr state_;
  ProfilerPtr profiler_;
  DataQueuesPtr data_queues_;  // for start_time -- see debugLogImu()'s absolute timestamps

  ImuProcOptions opts_;
  ImuSample last_imu_sample_;
  // R63 (F-118): first-interval head is always seeded from the first IMU sample.
  bool head_seeded_ = false;
  long first_head_seed_count_ = 0;
  long acc_scale_applied_samples_ = 0;   // R63 engagement counter (accel scale)
  bool acc_scale_logged_ = false;
  size_t propagation_index_ = 0;
public:
  std::string engagementReport() const;
};

}  // namespace livo_recon
