#pragma once

#include "livo_recon/lio/pose_control_lidar_factor.h"
#include "livo_recon/lio/pose_control_spline.h"
#include "livo_recon/utils/state/state.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <string>
#include <vector>

namespace livo_recon
{

enum class CoupledModeFamily { LocalSpline, SingleTail, DirectLidarImu, CovarianceAllKnots };

struct CoupledModeSelection
{
  CoupledModeFamily family = CoupledModeFamily::LocalSpline;
  bool physical_rpv = false;
};

bool parseCoupledMode(const std::string& name, CoupledModeSelection& selection);
const char* coupledModeFamilyName(CoupledModeFamily family);

void assembleLocalSplineControl(const PoseControlSpline&, const PoseControlFreeLayout&,
                                const std::vector<PoseControlLidarObs>&,
                                Eigen::MatrixXd&, Eigen::VectorXd&, double&,
                                std::vector<PoseControlLidarRecord>* records=nullptr);

struct CoupledModeStepResult
{
  Eigen::VectorXd delta_z;
  Eigen::MatrixXd sigma_post;
  Eigen::MatrixXd physical_gain;
  Eigen::MatrixXd physical_covariance;
  Eigen::MatrixXd physical_jacobian;
  Eigen::Matrix<double,6,6> innovation_transform=Eigen::Matrix<double,6,6>::Zero();
  Eigen::Matrix<double,6,1> innovation_dual=Eigen::Matrix<double,6,1>::Zero();
  Eigen::Matrix<double,6,1> physical_increment=Eigen::Matrix<double,6,1>::Zero();
  Eigen::MatrixXd lidar_information_z;
  Eigen::VectorXd lidar_rhs_z;
  Eigen::VectorXd full_state_delta;
  Eigen::MatrixXd full_state_covariance;
};

bool solveSingleTailControl(const PoseControlSpline&, const PoseControlFreeLayout&,
                            const PoseControlHeadNullspace&, const Eigen::MatrixXd& sigma_prior,
                            const Eigen::MatrixXd& J_tail_full,
                            const Eigen::Matrix<double,6,6>& P_tail,
                            const Eigen::Matrix<double,6,1>& current_minus_prior,
                            const PoseControlPhysicalLidarInformation&,
                            CoupledModeStepResult&);

bool solveDirectLidarImuControl(const PoseControlSpline&, const PoseControlFreeLayout&,
                                const PoseControlHeadNullspace&, const StateGroup&,
                                const Eigen::MatrixXd& J_tail_full,
                                const PoseControlPhysicalLidarInformation&,
                                CoupledModeStepResult&);

struct CovarianceAllKnotsResidualTrace
{
  std::size_t residual_index=0;
  Eigen::MatrixXd J_phys;
  Eigen::RowVectorXd H;
  Eigen::Matrix<double,6,1> h6=Eigen::Matrix<double,6,1>::Zero();
  Eigen::VectorXd gain;
  Eigen::VectorXd accumulated_before;
  double measurement_variance=0.0, innovation=0.0, innovation_variance=0.0, nis=0.0;
};

struct CovarianceAllKnotsStatistics
{
  std::size_t count=0, above_3p84=0, above_6p63=0;
  double sum=0.0, square_sum=0.0, maximum=0.0;
};

bool solveCovarianceAllKnotsControl(const PoseControlSpline&, const PoseControlFreeLayout&,
                                    const PoseControlHeadNullspace&, const V3D& gravity,
                                    const std::vector<Residual>&, const Eigen::MatrixXd& sigma_prior,
                                    double q_pos, double q_rot, CoupledModeStepResult&,
                                    CovarianceAllKnotsStatistics&,
                                    std::vector<CovarianceAllKnotsResidualTrace>* traces=nullptr);

struct PhysicalRpvUpdate
{
  Eigen::Matrix<double,9,9> P_prior = Eigen::Matrix<double,9,9>::Zero();
  Eigen::Matrix<double,9,9> P_post = Eigen::Matrix<double,9,9>::Zero();
  Eigen::Matrix<double,9,9> H_full = Eigen::Matrix<double,9,9>::Zero();
  Eigen::Matrix<double,9,9> P_prior_inverse = Eigen::Matrix<double,9,9>::Zero();
  Eigen::Matrix<double,9,9> A = Eigen::Matrix<double,9,9>::Zero();
  Eigen::Matrix<double,9,9> K1 = Eigen::Matrix<double,9,9>::Zero();
  Eigen::Matrix<double,9,6> K1_pose = Eigen::Matrix<double,9,6>::Zero();
  Eigen::Matrix<double,9,6> G = Eigen::Matrix<double,9,6>::Zero();
  Eigen::Matrix<double,9,1> vec = Eigen::Matrix<double,9,1>::Zero();
  Eigen::Matrix<double,9,1> measurement_term = Eigen::Matrix<double,9,1>::Zero();
  Eigen::Matrix<double,9,1> prior_term = Eigen::Matrix<double,9,1>::Zero();
  Eigen::Matrix<double,9,1> solution = Eigen::Matrix<double,9,1>::Zero();
  Eigen::Matrix<double,6,1> desired_pose = Eigen::Matrix<double,6,1>::Zero();
  // Full-state counterparts preserve R/P/V-to-bias/gravity cross covariance.
  // The R/P/V entries agree with solution/P_post; the remaining entries are
  // the conditional full-state correction/posterior implied by the same solve.
  Eigen::VectorXd state_delta;
  Eigen::MatrixXd P_full_post;
};

bool solveDirectLidarImuPhysicalRpv(const StateGroup& current,
                                    const StateGroup& propagated,
                                    const Eigen::MatrixXd& propagated_covariance,
                                    const PoseControlPhysicalLidarInformation& lidar,
                                    PhysicalRpvUpdate& out);

bool realizeLocalSplinePhysicalRpv(const PoseControlSpline&, const PoseControlFreeLayout&,
                                   const PoseControlHeadNullspace&, const Eigen::MatrixXd&,
                                   const Eigen::Matrix<double,9,1>&, Eigen::VectorXd&,
                                   Eigen::Matrix<double,9,Eigen::Dynamic>&,
                                   Eigen::Matrix<double,9,1>&, Eigen::Matrix<double,9,1>&);
bool realizeSingleTailPhysicalRpv(const PoseControlSpline&, const PoseControlFreeLayout&,
                                 const PoseControlHeadNullspace&, const Eigen::MatrixXd&,
                                 const Eigen::Matrix<double,9,1>&, Eigen::VectorXd&,
                                 Eigen::Matrix<double,9,Eigen::Dynamic>&,
                                 Eigen::Matrix<double,9,1>&, Eigen::Matrix<double,9,1>&);
bool realizeDirectLidarImuPhysicalRpv(const PoseControlSpline&, const PoseControlFreeLayout&,
                                     const PoseControlHeadNullspace&, const Eigen::MatrixXd&,
                                     const Eigen::Matrix<double,9,1>&, Eigen::VectorXd&,
                                     Eigen::Matrix<double,9,Eigen::Dynamic>&,
                                     Eigen::Matrix<double,9,1>&, Eigen::Matrix<double,9,1>&);
bool realizeCovarianceAllKnotsPhysicalRpv(const PoseControlSpline&, const PoseControlFreeLayout&,
                                         const PoseControlHeadNullspace&, const Eigen::MatrixXd&,
                                         const Eigen::Matrix<double,9,1>&, Eigen::VectorXd&,
                                         Eigen::Matrix<double,9,Eigen::Dynamic>&,
                                         Eigen::Matrix<double,9,1>&, Eigen::Matrix<double,9,1>&);

} // namespace livo_recon
