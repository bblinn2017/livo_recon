#pragma once

#include "livo_recon/utils/algo/math.h"
#include "livo_recon/utils/data/data_wrappers.h"
#include "livo_recon/utils/map/voxelmap_utils.h"
#include "livo_recon/utils/state/state.h"
#include "livo_recon/lio/residual_redundancy.h"

#include <Eigen/Dense>
#include <algorithm>
#include <string>
#include <vector>

namespace livo_recon
{

enum class JointKnotResidualTime { Measurement, Tail };

struct JointKnot
{
  double t = 0.0;
  M3D R = M3D::Identity();
  V3D p = V3D::Zero();
  V3D v = V3D::Zero();
};

struct JointKnotEvaluation
{
  M3D R = M3D::Identity();
  V3D p = V3D::Zero();
  V3D v = V3D::Zero();
};

// Physical scan trajectory.  Knot zero exists in the trajectory but has no
// columns in the optimization vector: dtheta_0, dp_0 and dv_0 are therefore
// structurally zero. Biases are knot-specific Markov states so production
// gyro/accelerometer bias random-walk process noise is represented without
// collapsing time-local increments into one scan-wide variable. Gravity is
// one shared three-vector over the complete scan.
class JointKnotTrajectory
{
public:
  bool initialize(const std::vector<Pose6D>& poses, double scan_tail,
                  int knot_count, const StateGroup& propagated_state);

  int knotCount() const { return static_cast<int>(knots_.size()); }
  int dim() const
  {
    return 9 * std::max(0, knotCount() - 1) + 6 * knotCount() + 3;
  }
  int knotOffset(int k) const { return k <= 0 ? -1 : 9 * (k - 1); }
  int biasOffset(int k) const
  {
    return 9 * std::max(0, knotCount() - 1) + 6 * k;
  }
  int gravityOffset() const
  {
    return 9 * std::max(0, knotCount() - 1) + 6 * knotCount();
  }
  const std::vector<JointKnot>& knots() const { return knots_; }
  const V3D& biasGyr(int k) const { return bg_.at(k); }
  const V3D& biasAcc(int k) const { return ba_.at(k); }
  const V3D& tailBiasGyr() const { return bg_.back(); }
  const V3D& tailBiasAcc() const { return ba_.back(); }
  const V3D& gravity() const { return g_; }

  JointKnotEvaluation evaluate(double t) const;
  V3D worldPoint(double point_time, const V3D& q_body,
                 JointKnotResidualTime residual_time) const;
  Eigen::MatrixXd worldPointJacobian(double point_time, const V3D& q_body,
                                     JointKnotResidualTime residual_time) const;
  Eigen::VectorXd boxminus(const JointKnotTrajectory& prior) const;
  void applyDelta(const Eigen::VectorXd& dx);

private:
  int interval(double t, double& u) const;
  std::vector<JointKnot> knots_;
  std::vector<V3D> bg_, ba_;
  V3D g_ = V3D::Zero();
};

struct JointKnotPrior
{
  Eigen::MatrixXd P;
  Eigen::MatrixXd P_inverse;
};

// Linearized IMU propagation prior.  Cross-time covariance is created by
// propagating the production 18-state chain through every knot interval.
// Full production process noise is injected once per interval. Biases remain
// distinct at every knot while gravity is retained once because it is shared
// and has no process noise. Knot zero has no physical mean-update columns,
// but its bias columns remain mutable and its physical uncertainty is
// marginalized into every retained future knot rather than conditioned away.
JointKnotPrior buildJointKnotImuPrior(const JointKnotTrajectory& trajectory,
                                     const StateGroup& propagated_state,
                                     const std::vector<Pose6D>& poses,
                                     const std::vector<Eigen::MatrixXd>& transitions,
                                     const std::vector<Eigen::MatrixXd>& process_covariances,
                                     const Eigen::MatrixXd& scan_head_covariance);

struct JointKnotSolve
{
  Eigen::MatrixXd Gamma_L;
  Eigen::VectorXd b_L;
  Eigen::MatrixXd A;
  Eigen::MatrixXd K1;
  Eigen::VectorXd vec;
  Eigen::VectorXd rhs;
  Eigen::VectorXd delta;
  // Split of the canonical correction.  These add exactly to delta and are
  // retained explicitly so a small update can be attributed to weak LiDAR
  // information, a strong prior, or cancellation between the two.
  Eigen::VectorXd lidar_rhs;
  Eigen::VectorXd prior_rhs;
  Eigen::VectorXd lidar_delta;
  Eigen::VectorXd prior_delta;
  Eigen::MatrixXd posterior;
  double mean_abs_residual = 0.0;
  std::vector<int> residual_count_by_interval;
  int zero_jacobian_residual_count = 0;
  std::string lidar_information_mode = "independent";
  ResidualRedundancyStats redundancy_stats;
};

// Pre-linearized scalar LiDAR row used by the shared joint information
// accumulator. Keeping this independent of trajectory interpolation makes the
// correlated-group algebra directly testable against dense C^-1.
struct JointLidarRow
{
  Eigen::RowVectorXd H;
  double residual = 0.0;
  double sigma_squared = 0.0;
  double plane_var_term = 0.0;
  const void* plane_id = nullptr;
  V3D plane_jacobian = V3D::Zero();
  M3D plane_covariance = M3D::Zero();
};

struct JointLidarInformation
{
  Eigen::MatrixXd Gamma;
  Eigen::VectorXd b;
  ResidualRedundancyStats redundancy_stats;
  std::string mode = "independent";
};

JointLidarInformation accumulateJointLidarInformation(
    const std::vector<JointLidarRow>& rows, int state_dimension,
    const ResidualRedundancyOptions& options);

JointKnotSolve solveJointKnotInformation(
    const JointKnotTrajectory& current,
    const JointKnotTrajectory& imu_prior_mean,
    const JointKnotPrior& prior,
    const std::vector<Residual>& residuals,
    const std::vector<PointXYZCov>& evaluated_points,
    JointKnotResidualTime residual_time,
    const ResidualRedundancyOptions& redundancy_options = {});

Eigen::MatrixXd extractTailStateCovariance(const JointKnotTrajectory& trajectory,
                                           const Eigen::MatrixXd& joint_cov,
                                           const StateGroup& state);

}  // namespace livo_recon
