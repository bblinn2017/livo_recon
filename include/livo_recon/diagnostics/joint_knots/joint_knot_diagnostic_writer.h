#pragma once

#include "livo_recon/lio/joint_knot_estimator.h"
#include "livo_recon/utils/data/measures.h"
#include <string>

namespace livo_recon
{

void writeJointKnotIterationDiagnostics(
    const std::string& test_id, int scan_id, int iteration, double t_abs,
    const std::string& residual_time, bool gating_state_uncertainty,
    const JointKnotTrajectory& before, const JointKnotTrajectory& after,
    const JointKnotPrior& prior, const JointKnotSolve& solve,
    int residual_count, double mean_gating_cov_trace,
    double max_gating_cov_trace);

// Compact all-scan record of the exact LiDAR information admitted to the
// canonical solve.  Unlike the large first-frame matrix dump this is intended
// for 30 s real-data comparisons of independent and correlated Gamma_L.
void writeJointKnotLidarInformationDiagnostics(
    const std::string& test_id, int scan_id, int iteration, double t_abs,
    const JointKnotSolve& solve, int residual_count);

void writeJointKnotAllScanDiagnostics(
    const std::string& test_id, int scan_id, int iteration, double t_abs,
    const M3D& reference_R, const V3D& reference_p, const V3D& reference_v,
    const JointKnotTrajectory& before, const JointKnotTrajectory& after);

void writeJointKnotStateChainDiagnostics(
    const std::string& test_id, int scan_id, int iteration,
    const std::string& phase, double t_abs,
    const MeasureGroup& measures, const StateGroup& current);

void writeJointKnotScanSummaryDiagnostics(
    const std::string& test_id, int scan_id, double t_abs,
    int completed_iterations, const std::string& stop_reason,
    const StateGroup& post_imu, const StateGroup& final_state,
    const M3D& reference_R, const V3D& reference_p,
    const V3D& reference_v);

void writeJointKnotCovarianceDiagnostics(
    const std::string& test_id, int scan_id, double t_abs,
    const JointKnotTrajectory& trajectory, const JointKnotPrior& prior,
    const Eigen::MatrixXd& posterior_extended,
    const Eigen::MatrixXd& state_post_imu,
    const Eigen::MatrixXd& state_post_lio,
    const Eigen::MatrixXd& scan_head_covariance,
    const std::vector<Pose6D>& poses,
    const std::vector<Eigen::MatrixXd>& transitions,
    const std::vector<Eigen::MatrixXd>& process_covariances);

}  // namespace livo_recon
