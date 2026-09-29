#include "livo_recon/processing/lio_coupled.h"

#include "livo_recon/diagnostics/init_consistency.h"
#include "livo_recon/diagnostics/joint_knots/joint_knot_diagnostic_writer.h"
#include "livo_recon/diagnostics/stationary_diagnostic_writer.h"
#include "livo_recon/lio/residual_redundancy.h"
#include "livo_recon/map/voxelmap.h"

#include <cmath>
#include <limits>
#include <sstream>

namespace livo_recon
{

std::string LioProcCoupled::processLIO(MeasureGroup& mg)
{
  mg.prior_pos = state_->pos();
  mg.prior_rot = state_->rot();
  mg.prior_vel = state_->vel();
  if (voxel_map_->isEmpty()) return {};
  ensureStationaryReference(mg);
  logStateTrace("post_imu", mg, 0, 0);
  if (openLoopActive() || startupHoldActive(static_cast<long>(voxel_map_->frame_idx_))) return openLoopFinishScan(mg);

  TimedScope scope(profiler_, "lio/joint_knots");
  prior_cov_ = state_->cov();
  applyPriorScalarControls(prior_cov_, opts_.prior_scalar);
  state_propagat_ = *state_;
  trP_pos_pre_ = prior_cov_.block<3,3>(StateGroup::idxP(), StateGroup::idxP()).trace();
  if (!copts_.minimal_diagnostics)
    writeJointKnotStateChainDiagnostics(
        copts_.test_id, voxel_map_->frame_idx_, -1, "pre_update",
        mg.image.t + data_queues_->start_time, mg, *state_);

  trajectory_valid_ = trajectory_.initialize(
      mg.poses, mg.image.t, copts_.knot_count, state_propagat_);
  if (!trajectory_valid_) return "joint_knots:no_trajectory";
  if (mg.pose_covariances.empty() ||
      mg.imu_state_transitions.size() != mg.poses.size() ||
      mg.imu_process_covariances.size() != mg.poses.size())
    return "joint_knots:missing_imu_covariance_records";
  imu_prior_mean_ = trajectory_;
  // Apply the same configured covariance controls to the covariance that
  // actually seeds the joint cross-time prior.  Previously prior_cov_ was
  // adjusted for diagnostics while the solve silently used the unadjusted
  // scan-head snapshot, creating two different meanings of "fixed prior".
  Eigen::MatrixXd controlled_head_covariance = mg.pose_covariances.front();
  applyPriorScalarControls(controlled_head_covariance, opts_.prior_scalar);
  joint_prior_ = buildJointKnotImuPrior(
      imu_prior_mean_, state_propagat_, mg.poses, mg.imu_state_transitions,
      mg.imu_process_covariances, controlled_head_covariance);

  bool any_solved = false;
  std::string stop = "max_iter";
  int completed = 0;
  for (int iteration = 0; iteration < opts_.max_iterations; ++iteration) {
    const StateGroup iteration_before = *state_;
    setDiagnosticGnIteration(iteration);
    std::vector<PointXYZCov> evaluated_points;
    std::vector<M3D> gating_covariances;
    makeEvaluationPoints(mg, evaluated_points, gating_covariances);
    double mean_gating_cov_trace = 0.0, max_gating_cov_trace = 0.0;
    if (!gating_covariances.empty()) {
      for (const M3D& covariance : gating_covariances) {
        mean_gating_cov_trace += covariance.trace();
        max_gating_cov_trace = std::max(max_gating_cov_trace, covariance.trace());
      }
      mean_gating_cov_trace /= gating_covariances.size();
    }
    buildResiduals(evaluated_points, residuals_, iteration == 0,
                   copts_.gating_state_uncertainty,
                   copts_.gating_state_uncertainty ? &gating_covariances : nullptr);
    const std::string architecture = "coupled_" + copts_.residual_evaluation_time;
    if (residuals_.empty()) {
      if (!copts_.minimal_diagnostics)
        writeGatingIterationDiagnostics(
            copts_.test_id, architecture, voxel_map_->frame_idx_, iteration,
            mg.image.t + data_queues_->start_time,
            copts_.gating_state_uncertainty,
            static_cast<int>(evaluated_points.size()),
            n_statistical_gate_candidates_, n_statistical_gate_rejections_,
            n_miss_coverage_, n_miss_mismatch_, mean_gating_cov_trace,
            max_gating_cov_trace, residuals_);
      stop = "no_residuals";
      break;
    }

    // The accepted residual covariance is consumed directly. Deprecated
    // global density/chi2 scalers and per-residual reweighting are absent.
    if (!copts_.minimal_diagnostics)
      writeGatingIterationDiagnostics(
          copts_.test_id, architecture, voxel_map_->frame_idx_, iteration,
          mg.image.t + data_queues_->start_time,
          copts_.gating_state_uncertainty, static_cast<int>(evaluated_points.size()),
          n_statistical_gate_candidates_, n_statistical_gate_rejections_,
          n_miss_coverage_, n_miss_mismatch_, mean_gating_cov_trace,
          max_gating_cov_trace, residuals_);

    const JointKnotTrajectory before = trajectory_;
    last_solve_ = solveJointKnotInformation(
        trajectory_, imu_prior_mean_, joint_prior_, residuals_,
        evaluated_points, copts_.residualTime(), opts_.residual_redundancy);
    // Intentionally no clipping, blockwise limiting or damping: this is the
    // same full-step fixed-prior MAP convention used by the ordinary IEKF.
    trajectory_.applyDelta(last_solve_.delta);
    commitTrajectoryToState();
    any_solved = true;
    ++completed;
    ++iterations_;

    if (!copts_.minimal_diagnostics) {
      writeJointKnotLidarInformationDiagnostics(
          copts_.test_id, voxel_map_->frame_idx_, iteration,
          mg.image.t + data_queues_->start_time, last_solve_,
          static_cast<int>(residuals_.size()));
      writeJointKnotIterationDiagnostics(
          copts_.test_id, voxel_map_->frame_idx_, iteration,
          mg.image.t + data_queues_->start_time,
          copts_.residual_evaluation_time,
          copts_.gating_state_uncertainty, before, trajectory_,
          joint_prior_, last_solve_, static_cast<int>(residuals_.size()),
          mean_gating_cov_trace, max_gating_cov_trace);
      writeJointKnotAllScanDiagnostics(
          copts_.test_id, voxel_map_->frame_idx_, iteration,
          mg.image.t + data_queues_->start_time, stationary_reference_R_,
          stationary_reference_p_, stationary_reference_v_, before, trajectory_);
      writeStationaryIterationDiagnostics(
          copts_.test_id, architecture, voxel_map_->frame_idx_, iteration,
          mg.image.t + data_queues_->start_time, stationary_reference_R_,
          stationary_reference_p_, stationary_reference_v_, iteration_before,
          *state_, residuals_);
      writeJointKnotStateChainDiagnostics(
          copts_.test_id, voxel_map_->frame_idx_, iteration, "post_iteration",
          mg.image.t + data_queues_->start_time, mg, *state_);
    }

    // Experiment policy: run exactly max_iterations whenever residuals remain
    // available. Step/error values are diagnostics only, not stop conditions.
  }

  if (any_solved) {
    state_->covMut() = extractTailStateCovariance(
        trajectory_, last_solve_.posterior, *state_);
    writeFinalDeskew(mg);
    if (!copts_.minimal_diagnostics)
      writeJointKnotCovarianceDiagnostics(
          copts_.test_id, voxel_map_->frame_idx_,
          mg.image.t + data_queues_->start_time, trajectory_, joint_prior_,
          last_solve_.posterior, state_propagat_.cov(), state_->cov(),
          controlled_head_covariance, mg.poses, mg.imu_state_transitions,
          mg.imu_process_covariances);
  }
  mg.pos_after_lio = state_->pos();
  mg.rot_after_lio = state_->rot();
  mg.vel_after_lio = state_->vel();
  mg.cov_after_lio = state_->cov();
  if (!copts_.minimal_diagnostics)
    writeJointKnotScanSummaryDiagnostics(
        copts_.test_id, voxel_map_->frame_idx_,
        mg.image.t + data_queues_->start_time, completed, stop,
        state_propagat_, *state_, stationary_reference_R_,
        stationary_reference_p_, stationary_reference_v_);
  logStateTrace("post_lio", mg, static_cast<int>(residuals_.size()), completed);
  writeInitializationConsistencyDiagnostics(
      copts_.test_id, voxel_map_->frame_idx_,
      mg.image.t + data_queues_->start_time,
      state_->initialCovarianceMode(), state_->initialCovarianceActual(),
      state_->initCovPos(), state_->initCovVel(), state_->initCovRotTilt(),
      state_->initCovRotYaw(), state_->initCovGravity(), state_->initCovBg(),
      state_->initCovBa(), static_cast<int>(residuals_.size()), completed,
      stationary_reference_R_, stationary_reference_p_, stationary_reference_v_,
      state_propagat_.rot(), state_propagat_.pos(), state_propagat_.vel(),
      state_propagat_.cov().block<9, 9>(StateGroup::idxR(), StateGroup::idxR()),
      state_->rot(), state_->pos(), state_->vel(),
      state_->cov().block<9, 9>(StateGroup::idxR(), StateGroup::idxR()));

  std::ostringstream report;
  report << "joint_knots: iterations=" << completed
         << " residuals=" << residuals_.size()
         << " stop=" << stop;
  return report.str();
}

}  // namespace livo_recon
