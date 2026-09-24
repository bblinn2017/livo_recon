#pragma once

#include "livo_recon/lio/pose_control_spline.h"
#include "livo_recon/lio/pose_control_layout.h"
#include "livo_recon/diagnostics/pose_control/pose_control_physical_diagnostics.h"

namespace livo_recon
{

struct PoseControlPriorHeadBlock
{
  Eigen::Matrix<double, 9, 9> A_hh = Eigen::Matrix<double, 9, 9>::Zero();
  Eigen::MatrixXd A_hf;
};

struct PoseControlContinuousImuPriorStats
{
  int n_samples = 0;
  double E_imu = 0.0;
};

/// Accumulates the continuous-time accelerometer/gyro prior in reduced spline coordinates.
PoseControlContinuousImuPriorStats buildPoseControlContinuousImuPrior(
    const PoseControlSpline& spline, const PoseControlFreeLayout& layout,
    const std::vector<ImuSample>& imu_samples,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    const V3D& var_acc, const V3D& var_gyr,
    Eigen::MatrixXd& A, Eigen::VectorXd& b,
    PoseControlPriorHeadBlock* head_block,
    std::vector<ImuSplineResidualSample>* out_samples = nullptr);

}  // namespace livo_recon
