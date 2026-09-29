#pragma once
// R61: per-scan raw state and FULL covariance trace, plus the per-scan IMU covariance growth split.
//
// state_trace.csv: two rows per scan (phase = post_imu | post_lio) with the raw state (rotation as a quaternion
// w,x,y,z of R body->world, position, velocity, gyro bias, accel bias, gravity) and the upper triangle of the state
// covariance for the FIRST 18 state dimensions (layout R(0-2) P(3-5) V(6-8) BG(9-11) BA(12-14) G(15-17), valid when
// bg, ba and gravity are all estimated; `dim` is written so a different layout is detectable and the missing columns
// are left empty). Unlike initialization_consistency_all_scans.csv this file has NO reference in it: errors are formed
// offline against the stationary reference (constant) or against GT (position only, see the analysis scripts).
//
// imu_cov_growth.csv: one row per scan: diag(P) at the start of the scan's IMU propagation, diag(P) at its end, and
// diag of the scan-accumulated process covariance Qacc (Qacc <- F Qacc F^T + Q per IMU step), so that
// P_end = Phi P_start Phi^T + Qacc holds exactly and the growth caused by Q can be read directly.
//
// Both are numerics-inert (write-only).
#include <Eigen/Dense>
#include <string>

#include "livo_recon/utils/state/state.h"

namespace livo_recon
{

void writeStateTraceRow(const std::string& run_id, int scan_id, double t_abs, const char* phase,
                        const StateGroup& s, int residual_count, int completed_iterations,
                        int open_loop_window, int open_loop_reset);

void writeImuCovGrowthRow(size_t scan_index, double t_abs, const Eigen::MatrixXd& P_start,
                          const Eigen::MatrixXd& P_end, const Eigen::MatrixXd& Q_acc);

// One-shot: the stationary reference (R,p,v) and the first-frame head covariance P0 the analysis needs to form errors
// from state_trace.csv. Written to state_reference.txt by LioProcBase::ensureStationaryReference().
void writeStateReference(const Eigen::Matrix3d& R_ref, const Eigen::Vector3d& p_ref, const Eigen::Vector3d& v_ref,
                         const Eigen::MatrixXd& P0);

}  // namespace livo_recon
