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
