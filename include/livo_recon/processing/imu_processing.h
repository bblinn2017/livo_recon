#pragma once

#include "livo_recon/node_context.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/log/profiler.h"

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
  double motion_beta = 0.9;  // 0 = instantaneous; larger = more smoothing (R80 F-477: default 0.9, was 0.0)
  double motion_acc_scale = 0.3;   // R80 F-477: was 1.0
  double motion_gyr_scale = 10.0;  // R80 F-477: was 1.0
  // R97A Q3-fix: restored after bisection found R97A Q3's removal of this cap caused a real
  // divergence (not a no-op) on r97a_c_nya_02_id1 -- the motion-dependent dynamic variance
  // legitimately exceeds the configured cap at some point in that trajectory, and leaving it
  // uncapped lets a transient huge process noise destabilize the filter. See REPORT.md.
  double motion_acc_max_dynamic_variance = 0.5;
  double motion_gyr_max_dynamic_variance = 0.3;

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
  bool s2_logged_ = false;   // engagement counter: 1 iff the seed was applied
  // Low-pass state is excitation ENERGY before the stationary noise floor is
  // subtracted. Filtering the already-rectified dynamic variance would retain
  // the positive bias of squared stationary sensor noise.
  V3D filtered_acc_excitation_energy_ = V3D::Zero();
  V3D filtered_gyr_excitation_energy_ = V3D::Zero();
  bool motion_noise_primed_ = false;
  size_t propagation_index_ = 0;
public:
  std::string engagementReport() const;
};

}  // namespace livo_recon
