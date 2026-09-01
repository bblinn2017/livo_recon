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

  // History (24-44): see docs/livo_recon_changelog.md#include-livo_recon-processing-imu_processing.h-24
  double q_alpha_acc = 1.0;
  double q_alpha_gyr = 1.0;
  double q_alpha_bias = 1.0;
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
bool imuProcQhatRead(Eigen::MatrixXd& phi_p_phit, Eigen::MatrixXd& accum_cov_w);

class ImuProc
{
public:
  explicit ImuProc(NodeContext& ctx);

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
};

}  // namespace livo_recon
