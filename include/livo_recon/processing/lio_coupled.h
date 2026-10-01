#pragma once

#include "livo_recon/lio/joint_knot_estimator.h"
#include "livo_recon/processing/lio_base.h"

namespace livo_recon
{

struct LioProcCoupledOptions
{
  int knot_count = 4;
  std::string residual_evaluation_time = "measurement";
  bool gating_state_uncertainty = false;
  std::string test_id = "unlabeled";
  // When true, skip every large per-iteration/per-scan diagnostic writer
  // (state-chain, all-scan gating/knot dumps, first-frame matrix/covariance
  // text dumps, stationary_iteration.csv, the scan summary) and write only
  // writeInitializationConsistencyDiagnostics()'s compact post_imu/post_lio
  // pair per scan. Trajectory/odometry export, resolved config, and
  // engagement reporting are untouched -- those aren't gated by this flag.
  // Default false preserves every existing round's full-diagnostic behavior.
  bool minimal_diagnostics = false;

  // R66 (C-D4): "throw" (default, existing behaviour) lets a solver exception abort the node; "skip_scan" catches it,
  // restores the state to its value at the start of the scan, logs it in coupled_scan_status.csv and continues.
  std::string on_solver_error = "throw";
  // R66 (C-D3): multiplier on every Gauss-Newton delta (1.0 = the existing full step, bit-identical). Damps the
  // state step only; the posterior covariance is still the full-step one (a stated inconsistency of this test).
  double step_scale = 1.0;

  // R72 (F-398). failure_dump: on a solver exception write coupled_failure_dump.txt (every matrix and term of the failing update,
  // see writeFailureDump) and engagement.txt BEFORE the exception continues, so a node that then aborts still leaves both behind.
  // Behaviour when nothing fails is unchanged. record_frames N > 0: for map-backed frames 1..N write coupled_record_f<k>.txt with the
  // per-iteration matrices and vectors of the joint solve (writeIterationRecord); write-only, 0 (default) = off.
  bool failure_dump = false;
  int record_frames = 0;

  JointKnotResidualTime residualTime() const {
    return residual_evaluation_time == "tail"
        ? JointKnotResidualTime::Tail
        : JointKnotResidualTime::Measurement;
  }
};

class LioProcCoupled : public LioProcBase
{
public:
  explicit LioProcCoupled(NodeContext& ctx);
  ~LioProcCoupled() override;

  std::string loadParameters(ros::NodeHandle& pnh) override;
  std::string engagementReport() const override;
  void deskewAndDownsample(MeasureGroup& mg) override;
  std::string processLIO(MeasureGroup& mg) override;

private:
  std::string processLIOBody(MeasureGroup& mg);
  // R72 (F-398): write-only diagnostics, see LioProcCoupledOptions::failure_dump / record_frames.
  void writeFailureDump(const MeasureGroup& mg, const std::string& what, int completed_iterations);
  void writeIterationRecord(const MeasureGroup& mg, int iteration, const JointKnotTrajectory& before);
  int iterations_this_scan_ = 0;   // completed GN iterations of the scan being processed (for the failure dump)
  std::string failure_note_;       // appended to engagementReport() after a failure
  long solver_errors_skipped_ = 0;
  const char* firstFrameArchitectureName() const override { return "coupled_joint_knots"; }
  void makeEvaluationPoints(const MeasureGroup& mg,
                            std::vector<PointXYZCov>& points,
                            std::vector<M3D>& gating_covariances) const;
  void commitTrajectoryToState();
  void writeFinalDeskew(MeasureGroup& mg) const;

  LioProcCoupledOptions copts_;
  JointKnotTrajectory trajectory_;
  JointKnotTrajectory imu_prior_mean_;
  JointKnotPrior joint_prior_;
  JointKnotSolve last_solve_;
  bool trajectory_valid_ = false;
  int iterations_ = 0;
};

}  // namespace livo_recon
