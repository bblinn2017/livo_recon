#pragma once

#include "livo_recon/utils/algo/math.h"
#include "livo_recon/utils/data/data_wrappers.h"

// ============================================================================
// Shared 9-dim ([theta,p,v]) single-IMU-step process model, extracted
// 2026-09-23 out of the now-removed obsolete coupled B-spline pose-knot
// estimator (pose_knot_spline.h/.cpp, retired -- see
// pose_control_implementation_report_v3.md's estimator-retirement section).
//
// These two free functions are estimator-independent: a marginal
// [theta,p,v] 9x9 block of the EXACT SAME F_x/cov_w construction
// ImuProc::propagate() (imu_processing.cpp) uses for the full state. They
// are the ONLY part of the old pose-knot file that current production code
// depends on -- pose_control_imu_prior_builder.cpp uses them to build the
// scan-start joint IMU prior (buildPoseControlImuPriorContribution(), see
// pose_control_imu_prior_builder.h). Bias/gravity rows/columns are absent
// (held fixed at their scan-start ESIKF values by both callers).
// ============================================================================

namespace livo_recon
{

// P_{j+1}^- = F_j P_j^+ F_j^T + Q_j, for ONE raw IMU sample interval.
void buildImuStep9x9(const M3D& rot_imu, const V3D& acc_avr, const V3D& angvel_avr,
                     double dt, const V3D& var_acc, const V3D& var_gyr,
                     double q_alpha_acc, double q_alpha_gyr, bool second_order,
                     Eigen::Matrix<double, 9, 9>& F9, Eigen::Matrix<double, 9, 9>& Q9);

// Advances (rot,pos,vel) by ONE raw-IMU-sample-pair micro-step [head,tail]
// via the same trapezoidal integration ImuProc::propagate() uses for the
// state itself, and simultaneously accumulates F9/Q9 into the caller's
// running segment accumulators via buildImuStep9x9() above.
void integrateAndAccumulateStep(
    const ImuSample& head, const ImuSample& tail,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    const V3D& var_acc, const V3D& var_gyr,
    double q_alpha_acc, double q_alpha_gyr, bool second_order,
    M3D& rot_imu, V3D& pos_imu, V3D& vel_imu,
    Eigen::Matrix<double, 9, 9>& F_seg, Eigen::Matrix<double, 9, 9>& Q_seg);

}  // namespace livo_recon
