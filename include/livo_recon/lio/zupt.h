#pragma once

#include <Eigen/Dense>
#include <string>

// ============================================================================
// R72 (F-401): zero-velocity pseudo-update, as ROS-free pure functions (tests/test_zupt.cpp).
//
// WHAT. When a scan is judged stationary, the world-frame velocity of the tail state is observed to be zero with variance
// sigma_v^2 I3:  z = 0 = v + n, H = [0 .. I3 (at the velocity rows) .. 0].  Joseph-form update on the full error-state covariance:
//   S = P_vv + sigma_v^2 I,  K = P[:, v] S^-1,  dx = K (0 - v),  P' = (I - K H) P (I - K H)^T + sigma_v^2 K K^T.
// The information form is exact: P'^-1 = P^-1 + H^T (sigma_v^2 I)^-1 H.
//
// DETECTOR. A scan is a candidate when its mean angular-rate norm is below gyro_thresh and its mean world-frame specific-force
// residual norm below acc_thresh; it is stationary after enter_scans consecutive candidates and leaves stationary immediately on a
// non-candidate scan (so a late exit never constrains motion).
//
// MODES. off (default; never evaluated, bit-identical); detect_only (detector and log run, state untouched: this is the control
// arm and also the data that sets the thresholds); apply.
// Threshold and sigma_v values here are PROVISIONAL (rule 1.7); the manifest sets them explicitly.
// ============================================================================

namespace livo_recon
{

struct ZuptOptions
{
  std::string mode = "off";    // "off" | "detect_only" | "apply"
  double gyro_thresh = 0.02;   // rad/s, mean |gyro| over the scan
  double acc_thresh = 0.15;    // m/s^2, mean |world specific force + gravity| over the scan
  int enter_scans = 3;         // consecutive candidate scans needed
  double sigma_v = 0.02;       // m/s, pseudo-measurement standard deviation
  bool on() const { return mode != "off"; }
  bool apply() const { return mode == "apply"; }
};

class ZuptDetector
{
public:
  // Returns the stationary flag for this scan.
  bool update(double gyro_mean, double acc_mean, const ZuptOptions& o)
  {
    const bool candidate = gyro_mean < o.gyro_thresh && acc_mean < o.acc_thresh;
    run_ = candidate ? run_ + 1 : 0;
    stationary_ = run_ >= o.enter_scans;
    return stationary_;
  }
  bool stationary() const { return stationary_; }
  int run() const { return run_; }

private:
  int run_ = 0;
  bool stationary_ = false;
};

// P is dim x dim (error-state covariance), v the current world-frame velocity, idx_v the first velocity row. Outputs the error-state
// correction dx (dim) and the updated covariance.
inline void zuptVelocityUpdate(const Eigen::MatrixXd& P, const Eigen::Vector3d& v, int idx_v, double sigma_v,
                               Eigen::VectorXd& dx, Eigen::MatrixXd& P_new)
{
  const int dim = static_cast<int>(P.rows());
  const double s2 = sigma_v * sigma_v;
  const Eigen::Matrix3d S = P.block<3, 3>(idx_v, idx_v) + s2 * Eigen::Matrix3d::Identity();
  const Eigen::MatrixXd PHt = P.block(0, idx_v, dim, 3);                       // P H^T
  const Eigen::MatrixXd K = PHt * S.inverse();                                  // dim x 3
  dx = K * (-v);
  Eigen::MatrixXd IKH = Eigen::MatrixXd::Identity(dim, dim);
  IKH.block(0, idx_v, dim, 3) -= K;                                             // K H has K in the velocity columns
  P_new = IKH * P * IKH.transpose() + s2 * K * K.transpose();
  P_new = 0.5 * (P_new + P_new.transpose());
}

}  // namespace livo_recon
