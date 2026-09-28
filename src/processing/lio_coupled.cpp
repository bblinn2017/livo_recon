#include "livo_recon/processing/lio_coupled.h"

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
  if (copts_.knot_count < 2)
    cfg.requireCombination("estimator/coupled/knot_count must be at least 2");
  if (!state_->estBA() || !state_->estBG() || !state_->estGravity())
    cfg.requireCombination(
        "coupled_joint_knots requires state/est/ba=true, state/est/bg=true, "
        "and state/est/gravity=true because knot-specific [bg,ba] and shared g "
        "are part of the joint state");
  if (opts_.residual_weighting.collapseOn())
    cfg.requireCombination(
        "coupled_joint_knots requires lio/residual_weighting/collapse=off: "
        "collapsing measurements with different timestamps would destroy "
        "the unique knot Jacobian for each residual");
  if (opts_.residual_weighting.perResidualOn())
    cfg.requireCombination(
        "coupled_joint_knots currently requires lio/residual_weighting/per_residual=off: "
        "the legacy reweighter does not evaluate measurement-time knot Jacobians");
  if (opts_.residual_redundancy.on())
    cfg.requireCombination(
        "coupled_joint_knots currently requires lio/residual_redundancy/mode=off");
  if (opts_.cov_redundancy_discount.on())
    cfg.requireCombination(
        "coupled_joint_knots currently requires lio/ekf/cov_redundancy_discount=off");
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
      << " clipping=none";
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
