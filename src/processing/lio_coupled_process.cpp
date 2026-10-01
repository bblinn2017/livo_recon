#include "livo_recon/processing/lio_coupled.h"
#include "livo_recon/utils/algo/symmetrize.h"

#include "livo_recon/diagnostics/init_consistency.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"
#include "livo_recon/diagnostics/matrix_dump.h"
#include "livo_recon/lio/failure_capture.h"
#include "livo_recon/diagnostics/joint_knots/joint_knot_diagnostic_writer.h"
#include "livo_recon/diagnostics/r66_writers.h"
#include "livo_recon/diagnostics/stationary_diagnostic_writer.h"
#include "livo_recon/lio/residual_redundancy.h"
#include "livo_recon/map/voxelmap.h"

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

namespace livo_recon
{

std::string LioProcCoupled::processLIO(MeasureGroup& mg)
{
  const int iterations_before = iterations_;
  const double t_abs = mg.image.t + data_queues_->start_time;
  std::string report;
  iterations_this_scan_ = 0;
  if (copts_.failure_dump) lastNonPdCapture().valid = false;   // R72 (F-398): a stale capture from an earlier scan must not be mislabelled
  if (copts_.on_solver_error == "throw")
  {
    try
    {
      report = processLIOBody(mg);
    }
    catch (const std::exception& e)
    {
      // R72 (F-398): write and flush the dump and the engagement report BEFORE the exception continues (the node then aborts and no
      // destructor runs, which is why R71's aborted arms left no engagement.txt).
      if (copts_.failure_dump) writeFailureDump(mg, e.what(), iterations_this_scan_);
      throw;
    }
    writeCoupledScanStatus(static_cast<long>(voxel_map_->frame_idx_), t_abs, report,
                           iterations_ - iterations_before, 0, std::string());
    return report;
  }
  const StateGroup snapshot = *state_;
  try
  {
    report = processLIOBody(mg);
    writeCoupledScanStatus(static_cast<long>(voxel_map_->frame_idx_), t_abs, report,
                           iterations_ - iterations_before, 0, std::string());
  }
  catch (const std::exception& e)
  {
    if (copts_.failure_dump) writeFailureDump(mg, e.what(), iterations_this_scan_);   // R72 (F-398): before the state is restored
    *state_ = snapshot;
    mg.pos_after_lio = state_->pos();
    mg.rot_after_lio = state_->rot();
    mg.vel_after_lio = state_->vel();
    mg.cov_after_lio = state_->cov();
    ++solver_errors_skipped_;
    report = "joint_knots:solver_error_skipped";
    ROS_ERROR_STREAM("[coupled] solver error on scan " << voxel_map_->frame_idx_ << ", state restored, scan skipped: "
                     << e.what());
    writeCoupledScanStatus(static_cast<long>(voxel_map_->frame_idx_), t_abs, report,
                           iterations_ - iterations_before, 1, e.what());
  }
  return report;
}

std::string LioProcCoupled::processLIOBody(MeasureGroup& mg)
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
  // R73 (F-410): the joint IMU prior (a 114-dim covariance, built and inverted) is needed only to SOLVE. A scan with no LiDAR residual has
  // no H^T H, no H^T z and nothing to solve, so the prior is not built: the propagated covariance stays as the scan's covariance and the
  // prior inversion cannot throw. It is built after the first iteration's residual count is known to be positive. The one exception is
  // estimator/coupled/gating_state_uncertainty=true, whose residual admission reads joint_prior_.P: there it is built up front (no R73
  // arm sets that key).
  joint_prior_built_ = false;
  const auto buildPrior = [&]() {
    joint_prior_ = buildJointKnotImuPrior(
        imu_prior_mean_, state_propagat_, mg.poses, mg.imu_state_transitions,
        mg.imu_process_covariances, controlled_head_covariance);
    if (copts_.imu_prior_info_scale != 1.0)   // R73 (F-415): experimental scale of the IMU prior's information (see the option's comment)
      joint_prior_.P_inverse *= copts_.imu_prior_info_scale;
    joint_prior_built_ = true;
    ++prior_built_scans_;
  };
  if (copts_.gating_state_uncertainty) buildPrior();

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
      if (iteration == 0) ++prior_skipped_scans_;   // R73 (F-410): nothing solved, nothing built, covariance left as propagated
      break;
    }
    if (!joint_prior_built_) buildPrior();

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
    if (copts_.step_scale == 1.0)
      trajectory_.applyDelta(last_solve_.delta);
    else
    {
      const Eigen::VectorXd scaled_delta = last_solve_.delta * copts_.step_scale;
      trajectory_.applyDelta(scaled_delta);
    }
    commitTrajectoryToState();
    any_solved = true;
    ++completed;
    ++iterations_;
    iterations_this_scan_ = completed;
    if (copts_.record_frames > 0 && static_cast<long>(voxel_map_->frame_idx_) >= 1 &&
        static_cast<long>(voxel_map_->frame_idx_) <= copts_.record_frames)
      writeIterationRecord(mg, iteration, before);   // R72 (F-398): write-only
    if (copts_.update_log) writeUpdateLogIteration(mg, iteration, static_cast<int>(residuals_.size()));   // R73 (F-415): write-only

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
    state_->covMut() = symmetrized(extractTailStateCovariance(
        trajectory_, last_solve_.posterior, *state_));   // R73 F-419
    writeFinalDeskew(mg);
    if (copts_.update_log) writeUpdateLogScan(mg, stop, completed);   // R73 (F-415): write-only
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

namespace
{

// One row per knot: [t, R (row-major 9), p (3), v (3)].
Eigen::MatrixXd knotsMatrix(const JointKnotTrajectory& tr)
{
  Eigen::MatrixXd m(tr.knotCount(), 16);
  for (int k = 0; k < tr.knotCount(); ++k)
  {
    const JointKnot& kn = tr.knots()[k];
    m(k, 0) = kn.t;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) m(k, 1 + 3 * i + j) = kn.R(i, j);
    for (int i = 0; i < 3; ++i) { m(k, 10 + i) = kn.p(i); m(k, 13 + i) = kn.v(i); }
  }
  return m;
}

// One row per knot: [bg (3), ba (3)]; gravity is separate.
Eigen::MatrixXd biasMatrix(const JointKnotTrajectory& tr)
{
  Eigen::MatrixXd m(tr.knotCount(), 6);
  for (int k = 0; k < tr.knotCount(); ++k)
  {
    m.block<1, 3>(k, 0) = tr.biasGyr(k).transpose();
    m.block<1, 3>(k, 3) = tr.biasAcc(k).transpose();
  }
  return m;
}

Eigen::VectorXd ascendingEigenvalues(const Eigen::MatrixXd& M)
{
  const Eigen::MatrixXd S = symmetrized(M);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S, Eigen::EigenvaluesOnly);
  return es.eigenvalues();
}

}  // namespace

// R72 (F-398): per-iteration record of the joint solve for map-backed frames 1..record_frames. Everything the question "what exactly
// changes in the spline" needs: the knots before and after, the LiDAR information (Htz = b_L, HtH = Gamma_L) and the prior
// information (P^-1, written once per frame) separately, the prior mean pull (vec, prior_rhs), the system matrix A and its spectrum
// (the positive-definiteness check), the posterior P after this iteration, the step split into LiDAR and prior parts and per-knot
// step norms, and the tail 18x18 posterior. Write-only; appends to one file per frame; flushed every call.
void LioProcCoupled::writeIterationRecord(const MeasureGroup& mg, int iteration, const JointKnotTrajectory& before)
{
  const long frame = static_cast<long>(voxel_map_->frame_idx_);
  std::ofstream o(debugLogPath("coupled_record_f" + std::to_string(frame) + ".txt"), std::ios::app);
  if (!o) return;
  const JointKnotSolve& sv = last_solve_;
  const int D = trajectory_.dim();
  if (iteration == 0)
  {
    std::ostringstream meta;
    meta << "frame=" << frame << " t_abs=" << std::setprecision(17) << (mg.image.t + data_queues_->start_time)
         << " knot_count=" << copts_.knot_count << " residual_evaluation_time=" << copts_.residual_evaluation_time
         << " D=" << D << " test_id=" << copts_.test_id << " imu_steps=" << mg.poses.size();
    dumpText(o, "meta", meta.str());
    Eigen::MatrixXd head = mg.pose_covariances.empty() ? Eigen::MatrixXd::Zero(18, 18) : mg.pose_covariances.front();
    applyPriorScalarControls(head, opts_.prior_scalar);
    dumpMatrix(o, "P18_scan_head_controlled", head);
    dumpMatrix(o, "P18_after_imu_propagation", state_propagat_.cov());
    dumpMatrix(o, "P_joint_prior", joint_prior_.P);
    dumpMatrix(o, "P_joint_prior_inverse", joint_prior_.P_inverse);
    dumpVector(o, "P_joint_prior_eigenvalues", ascendingEigenvalues(joint_prior_.P));
    dumpMatrix(o, "knots_prior_mean", knotsMatrix(imu_prior_mean_));
    dumpMatrix(o, "bias_prior_mean", biasMatrix(imu_prior_mean_));
    dumpVector(o, "gravity_prior_mean", imu_prior_mean_.gravity());
    Eigen::MatrixXd steps(mg.poses.size(), 2);
    for (size_t m = 0; m < mg.poses.size(); ++m) { steps(m, 0) = mg.poses[m].t; steps(m, 1) = mg.poses[m].dt; }
    dumpMatrix(o, "imu_step_t_dt", steps);
    for (size_t m = 0; m < mg.imu_state_transitions.size() && m < mg.imu_process_covariances.size(); ++m)
    {
      dumpMatrix(o, "imu_F_" + std::to_string(m), mg.imu_state_transitions[m]);
      dumpMatrix(o, "imu_Q_" + std::to_string(m), mg.imu_process_covariances[m]);
    }
  }
  const std::string pre = "i" + std::to_string(iteration) + "_";
  dumpMatrix(o, pre + "knots_before", knotsMatrix(before));
  dumpMatrix(o, pre + "knots_after", knotsMatrix(trajectory_));
  dumpMatrix(o, pre + "bias_after", biasMatrix(trajectory_));
  dumpVector(o, pre + "gravity_after", trajectory_.gravity());
  dumpScalar(o, pre + "n_residuals", static_cast<double>(residuals_.size()));
  {
    Eigen::VectorXd by_interval(sv.residual_count_by_interval.size());
    for (size_t i = 0; i < sv.residual_count_by_interval.size(); ++i) by_interval(i) = sv.residual_count_by_interval[i];
    dumpVector(o, pre + "residual_count_by_interval", by_interval);
  }
  dumpVector(o, pre + "Htz_lidar", sv.b_L);
  dumpMatrix(o, pre + "HtH_lidar", sv.Gamma_L);
  dumpVector(o, pre + "vec_mean_minus_prior_mean", sv.vec);
  dumpVector(o, pre + "prior_rhs", sv.prior_rhs);
  dumpMatrix(o, pre + "A_system", sv.A);
  dumpVector(o, pre + "A_eigenvalues", ascendingEigenvalues(sv.A));
  dumpScalar(o, pre + "A_positive_definite_pass", 1.0);   // reaching this line means symmetricInverse(A) did not throw
  dumpMatrix(o, pre + "P_joint_posterior", sv.posterior);
  dumpVector(o, pre + "P_joint_posterior_eigenvalues", ascendingEigenvalues(sv.posterior));
  dumpVector(o, pre + "delta", sv.delta);
  dumpVector(o, pre + "lidar_delta", sv.lidar_delta);
  dumpVector(o, pre + "prior_delta", sv.prior_delta);
  dumpScalar(o, pre + "delta_norm", sv.delta.norm());
  dumpScalar(o, pre + "step_scale_applied", copts_.step_scale);
  {
    Eigen::MatrixXd per_knot(std::max(0, trajectory_.knotCount() - 1), 3);
    for (int k = 1; k < trajectory_.knotCount(); ++k)
    {
      const int off = trajectory_.knotOffset(k);
      per_knot(k - 1, 0) = sv.delta.segment<3>(off).norm();       // rotation
      per_knot(k - 1, 1) = sv.delta.segment<3>(off + 3).norm();   // position
      per_knot(k - 1, 2) = sv.delta.segment<3>(off + 6).norm();   // velocity
    }
    dumpMatrix(o, pre + "delta_norm_per_knot_rot_pos_vel", per_knot);
  }
  dumpMatrix(o, pre + "tail_P18_posterior", extractTailStateCovariance(trajectory_, sv.posterior, *state_));
  o.flush();
}

// R72 (F-398): everything about a failing scan, written and flushed before the exception continues. The matrix that failed (and its
// spectrum and LDLT pivots), P before the correction (the 18x18 fixed prior and the joint prior with its inverse), every term of the
// update that was formed (Htz, HtH, vec, the right-hand-side parts, A), the IMU F/Q records the prior was built from, the knots,
// and, when an earlier iteration of the same scan had succeeded, that iteration's posterior and step. No size cap: the code throws
// at the first non-PD matrix, so this is written once per run.
void LioProcCoupled::writeFailureDump(const MeasureGroup& mg, const std::string& what, int completed_iterations)
{
  const long frame = static_cast<long>(voxel_map_->frame_idx_);
  const NonPdCapture& cap = lastNonPdCapture();
  {
    std::ofstream o(debugLogPath("coupled_failure_dump.txt"), std::ios::trunc);
    if (o)
    {
      std::ostringstream meta;
      meta << "frame=" << frame << " t_abs=" << std::setprecision(17) << (mg.image.t + data_queues_->start_time)
           << " completed_iterations_before_failure=" << completed_iterations << " knot_count=" << copts_.knot_count
           << " residual_evaluation_time=" << copts_.residual_evaluation_time << " test_id=" << copts_.test_id
           << " on_solver_error=" << copts_.on_solver_error;
      dumpText(o, "meta", meta.str());
      dumpText(o, "what", what);
      dumpText(o, "capture_valid", cap.valid ? "1" : "0");
      dumpText(o, "capture_where", cap.valid ? cap.where : std::string("none (the exception did not come from a positive-definiteness check)"));
      if (cap.valid)
      {
        dumpMatrix(o, "failed_matrix_symmetrised", cap.matrix);
        dumpVector(o, "failed_matrix_eigenvalues_ascending", cap.eigenvalues);
        dumpVector(o, "failed_matrix_ldlt_diag", cap.ldlt_diag);
        dumpScalar(o, "failed_matrix_ldlt_info", cap.ldlt_info);
        dumpScalar(o, "failed_matrix_ldlt_positive", cap.ldlt_positive ? 1.0 : 0.0);
      }
      dumpMatrix(o, "P18_prior_cov_before_correction", prior_cov_);
      dumpMatrix(o, "P18_after_imu_propagation", state_propagat_.cov());
      Eigen::MatrixXd head = mg.pose_covariances.empty() ? Eigen::MatrixXd::Zero(18, 18) : mg.pose_covariances.front();
      applyPriorScalarControls(head, opts_.prior_scalar);
      dumpMatrix(o, "P18_scan_head_controlled", head);
      Eigen::MatrixXd steps(mg.poses.size(), 2);
      for (size_t m = 0; m < mg.poses.size(); ++m) { steps(m, 0) = mg.poses[m].t; steps(m, 1) = mg.poses[m].dt; }
      dumpMatrix(o, "imu_step_t_dt", steps);
      for (size_t m = 0; m < mg.imu_state_transitions.size() && m < mg.imu_process_covariances.size(); ++m)
      {
        dumpMatrix(o, "imu_F_" + std::to_string(m), mg.imu_state_transitions[m]);
        dumpMatrix(o, "imu_Q_" + std::to_string(m), mg.imu_process_covariances[m]);
      }
      for (size_t m = 0; m < mg.pose_covariances.size(); ++m) dumpMatrix(o, "pose_cov_" + std::to_string(m), mg.pose_covariances[m]);
      if (trajectory_valid_)
      {
        dumpMatrix(o, "knots_current", knotsMatrix(trajectory_));
        dumpMatrix(o, "bias_current", biasMatrix(trajectory_));
        dumpVector(o, "gravity_current", trajectory_.gravity());
        dumpMatrix(o, "knots_imu_prior_mean", knotsMatrix(imu_prior_mean_));
        dumpMatrix(o, "bias_imu_prior_mean", biasMatrix(imu_prior_mean_));
      }
      if (cap.valid && cap.where == "posterior_A")
      {
        const JointKnotSolve& p = jointKnotLastFailedSolve();
        dumpMatrix(o, "P_joint_prior", joint_prior_.P);
        dumpMatrix(o, "P_joint_prior_inverse", joint_prior_.P_inverse);
        dumpMatrix(o, "HtH_lidar_Gamma_L", p.Gamma_L);
        dumpVector(o, "Htz_lidar_b_L", p.b_L);
        dumpVector(o, "vec_mean_minus_prior_mean", p.vec);
        dumpVector(o, "prior_rhs", p.prior_rhs);
        dumpVector(o, "lidar_rhs", p.lidar_rhs);
        dumpVector(o, "rhs", p.rhs);
        dumpMatrix(o, "A_system", p.A);
        dumpScalar(o, "mean_abs_residual", p.mean_abs_residual);
        Eigen::VectorXd by_interval(p.residual_count_by_interval.size());
        for (size_t i = 0; i < p.residual_count_by_interval.size(); ++i) by_interval(i) = p.residual_count_by_interval[i];
        dumpVector(o, "residual_count_by_interval", by_interval);
        dumpScalar(o, "zero_jacobian_residual_count", p.zero_jacobian_residual_count);
      }
      if (completed_iterations > 0)
      {
        dumpMatrix(o, "P_joint_posterior_previous_iteration", last_solve_.posterior);
        dumpVector(o, "delta_previous_iteration", last_solve_.delta);
        dumpVector(o, "lidar_delta_previous_iteration", last_solve_.lidar_delta);
        dumpVector(o, "prior_delta_previous_iteration", last_solve_.prior_delta);
      }
      o.flush();
    }
  }   // closed: the file is complete on disk before the exception continues
  {
    std::ostringstream note;
    note << what << " (frame " << frame << ", after " << completed_iterations << " completed iterations; coupled_failure_dump.txt written)";
    failure_note_ = note.str();
    std::ofstream eng(debugLogPath("engagement.txt"), std::ios::trunc);
    if (eng) { eng << engagementReport() << '\n'; eng.flush(); }
  }
}

}  // namespace livo_recon
