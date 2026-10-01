#include "livo_recon/processing/lio_coupled.h"

#include <stdexcept>

#include "livo_recon/diagnostics/log/debug_log_dir.h"
#include "livo_recon/utils/log/config_resolve.h"
#include "livo_recon/utils/log/param_warn.h"

#include <fstream>
#include <sstream>

namespace livo_recon
{

LioProcCoupled::LioProcCoupled(NodeContext& ctx) : LioProcBase(ctx) {}

LioProcCoupled::~LioProcCoupled()
{
  const std::string report = engagementReport();
  ROS_WARN_STREAM("\n" << report);
  std::ofstream out(debugLogPath("engagement.txt"), std::ios::trunc);
  if (out) out << report << '\n';
}

std::string LioProcCoupled::loadParameters(ros::NodeHandle& pnh)
{
  ConfigResolver cfg(pnh);
  loadSharedParameters(cfg, pnh);
  if (opts_.shadow.on())
    throw std::invalid_argument("lio/shadow/mode=on is implemented for estimator/mode=iekf only (R73 F-414); refused under coupled");
  if (opts_.scan_log_en)
    throw std::invalid_argument("lio/scan_log_en is implemented for estimator/mode=iekf only (R73); refused under coupled");

  cfg.nested<int>(true, "estimator/mode=coupled",
                  "estimator/coupled/knot_count", copts_.knot_count, 4);
  cfg.nestedMode(true, "estimator/mode=coupled",
                 "estimator/coupled/residual_evaluation_time",
                 copts_.residual_evaluation_time, "measurement",
                 {"measurement", "tail"});
  cfg.nested<bool>(true, "estimator/mode=coupled",
                   "estimator/coupled/gating_state_uncertainty",
                   copts_.gating_state_uncertainty, false);
  cfg.nested<std::string>(true, "estimator/mode=coupled",
                          "estimator/coupled/test_id", copts_.test_id,
                          std::string("unlabeled"));
  cfg.nested<bool>(true, "estimator/mode=coupled",
                   "estimator/coupled/minimal_diagnostics",
                   copts_.minimal_diagnostics, false);
  cfg.nestedMode(true, "estimator/mode=coupled", "estimator/coupled/on_solver_error",
                 copts_.on_solver_error, "throw", {"throw", "skip_scan"});
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/step_scale", copts_.step_scale, 1.0);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/failure_dump", copts_.failure_dump, false);
  cfg.nested<int>(true, "estimator/mode=coupled", "estimator/coupled/record_frames", copts_.record_frames, 0);
  cfg.nested<bool>(true, "estimator/mode=coupled", "estimator/coupled/update_log_en", copts_.update_log, false);                 // R73
  cfg.nested<double>(true, "estimator/mode=coupled", "estimator/coupled/imu_prior_info_scale", copts_.imu_prior_info_scale, 1.0);   // R73
  if (!(copts_.imu_prior_info_scale > 0.0) || !std::isfinite(copts_.imu_prior_info_scale))
    cfg.requireCombination("estimator/coupled/imu_prior_info_scale must be finite and > 0");
  if (copts_.record_frames < 0)
    cfg.requireCombination("estimator/coupled/record_frames must be >= 0");
  if (!(copts_.step_scale > 0.0 && copts_.step_scale <= 1.0))
    cfg.requireCombination("estimator/coupled/step_scale must be in (0, 1]");
  if (copts_.knot_count < 2)
    cfg.requireCombination("estimator/coupled/knot_count must be at least 2");
  if (opts_.residual_redundancy.renorm != "off")
    cfg.requireCombination(
        "lio/residual_redundancy/renorm is implemented for estimator/mode=iekf only (the coupled estimator builds joint-knot "
        "information inside solveJointKnotInformation)");
  if (opts_.info_scale.on())
    cfg.requireCombination(
        "lio/info_scale/mode=block is implemented for estimator/mode=iekf only (the coupled estimator accumulates "
        "joint-knot information, not the 6x6 HtH the block scale acts on)");
  if (opts_.step_cap.on())
    cfg.requireCombination(
        "lio/ekf/step_cap/mode=hard is implemented for estimator/mode=iekf only (the coupled estimator does not use "
        "EkfUpdate::applyMeanUpdate; its own step knob is estimator/coupled/step_scale)");
  if (state_->gravityS2())
    cfg.requireCombination(
        "state/gravity_model=s2 is not implemented for the coupled estimator "
        "(joint knots carry a 3-vector shared gravity)");
  if (!state_->estBA() || !state_->estBG() || !state_->estGravity())
    cfg.requireCombination(
        "coupled_joint_knots requires state/est/ba=true, state/est/bg=true, "
        "and state/est/gravity=true because knot-specific [bg,ba] and shared g "
        "are part of the joint state");
  if (opts_.residual_redundancy.mode != "off" &&
      opts_.residual_redundancy.mode != "woodbury")
    cfg.requireCombination(
        "coupled_joint_knots supports lio/residual_redundancy/mode=off or "
        "woodbury");
  if (cuda_enable_)
    cfg.requireCombination(
        "coupled_joint_knots currently requires cuda/enable=false");

  // No legacy coupled selector or legacy coupled option namespace is
  // claimed. Removed configurations therefore fail ConfigResolver's
  // unclaimed-key check instead of silently selecting an obsolete estimator.
  return finalizeConfig(cfg, {"lio/ekf", "voxel_map"});
}

std::string LioProcCoupled::engagementReport() const
{
  std::ostringstream out;
  out << "[engagement] estimator=coupled_joint_knots"
      << " knot_count=" << copts_.knot_count
      << " residual_evaluation_time=" << copts_.residual_evaluation_time
      << " gating_state_uncertainty="
      << (copts_.gating_state_uncertainty ? "true" : "false")
      << " gating_state_uncertainty_scope=admission_only_direct_override"
      << " solve_uses_gating_state_covariance=false"
      << " iterations=" << iterations_
      << " head_state=immutable"
      << " biases=knot_specific_markov"
      << " process_covariance=full_production_Q"
      << " gravity=unconstrained_additive_3vector"
      << " lidar_information=" << opts_.residual_redundancy.mode
      << " lidar_correlation_group=matched_plane_id"
      << " lidar_correlation_rho=" << opts_.residual_redundancy.rho
      << " clipping=none"
      << " on_solver_error=" << copts_.on_solver_error
      << " solver_errors_skipped=" << solver_errors_skipped_
      << " prior_built_scans=" << prior_built_scans_ << " prior_skipped_no_residual_scans=" << prior_skipped_scans_
      << " step_scale=" << copts_.step_scale
      << " update_log_en=" << (copts_.update_log ? "true" : "false") << " imu_prior_info_scale=" << copts_.imu_prior_info_scale
      << " failure_dump=" << (copts_.failure_dump ? "true" : "false")
      << " record_frames=" << copts_.record_frames;
  if (!failure_note_.empty()) out << "\n  FAILURE: " << failure_note_;
  return out.str();
}

void LioProcCoupled::deskewAndDownsample(MeasureGroup& mg)
{
  TimedScope scope(profiler_, "lio/deskew");
  std::vector<PointXYZCov> deskewed;
  deskewPoints(state_, mg.poses, mg.image.t, mg.lidar_points,
               opts_.deskew, deskewed);
  if (opts_.dsOn()) {
    const DsMode mode = opts_.ds_mode == "average" ? DsMode::AVERAGE : DsMode::FIRST;
    voxelDownsample(deskewed, mg.points,
                    PointXYZCovKeyFn{opts_.ds_leaf_size}, mode);
  } else {
    mg.points = std::move(deskewed);
  }
}

void LioProcCoupled::makeEvaluationPoints(
    const MeasureGroup& mg, std::vector<PointXYZCov>& points,
    std::vector<M3D>& gating_covariances) const
{
  points = mg.points;
  gating_covariances.assign(points.size(), M3D::Zero());
  const JointKnotEvaluation tail = trajectory_.evaluate(
      trajectory_.knots().back().t);
  for (size_t i = 0; i < points.size(); ++i) {
    const V3D q = points[i].raw_body_point;
    const V3D world = trajectory_.worldPoint(points[i].t, q,
                                             copts_.residualTime());
    points[i].point = tail.R.transpose() * (world - tail.p);
    if (copts_.gating_state_uncertainty) {
      const Eigen::MatrixXd J = trajectory_.worldPointJacobian(
          points[i].t, q, JointKnotResidualTime::Measurement);
      gating_covariances[i] = J * joint_prior_.P * J.transpose();
    }
  }
}

void LioProcCoupled::commitTrajectoryToState()
{
  const JointKnot& tail = trajectory_.knots().back();
  Eigen::VectorXd dx = Eigen::VectorXd::Zero(state_->dimState());
  dx.segment<3>(StateGroup::idxR()) = Log(M3D(state_->rot().transpose() * tail.R));
  dx.segment<3>(StateGroup::idxP()) = tail.p - state_->pos();
  dx.segment<3>(StateGroup::idxV()) = tail.v - state_->vel();
  if (state_->idxBG() >= 0)
    dx.segment<3>(state_->idxBG()) = trajectory_.tailBiasGyr() - state_->biasGyr();
  if (state_->idxBA() >= 0)
    dx.segment<3>(state_->idxBA()) = trajectory_.tailBiasAcc() - state_->biasAcc();
  if (state_->idxG() >= 0)
    dx.segment<3>(state_->idxG()) = trajectory_.gravity() - state_->gravity();
  state_->applyDelta(dx);
}

void LioProcCoupled::writeFinalDeskew(MeasureGroup& mg) const
{
  if (!trajectory_valid_) return;
  const JointKnot& tail = trajectory_.knots().back();
  for (PointXYZCov& point : mg.points) {
    const V3D world = trajectory_.worldPoint(
        point.t, point.raw_body_point, JointKnotResidualTime::Measurement);
    point.point = tail.R.transpose() * (world - tail.p);
  }
}

}  // namespace livo_recon
