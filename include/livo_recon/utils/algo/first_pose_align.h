#pragma once
// R82 (F-506): the SECONDARY ground-truth alignment of evo/dense_gt_metrics, ROS-free and templated on any pose record with est_pos, est_rot, gt_pos, gt_rot
// (EvoProc's MatchedPose) so tests/test_first_pose_align_r82.cpp can check it without ROS. The est pose of the FIRST element is mapped onto its gt pose exactly:
// R0 = gt_rot0 * est_rot0^T, t0 = gt_pos0 - R0 * est_pos0. Errors are then RMS position error (m) and RMS rotation angle of (R0 * est_rot)^T * gt_rot (deg).
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

namespace livo_recon
{
template <class M>
inline bool firstPoseAlignedErrors(const std::vector<M>& matched, double& ate_first_m, double& are_first_deg, Eigen::Matrix3d& R0, Eigen::Vector3d& t0)
{
  const int n = static_cast<int>(matched.size());
  if (n < 3) return false;
  R0 = matched.front().gt_rot * matched.front().est_rot.transpose();
  t0 = matched.front().gt_pos - R0 * matched.front().est_pos;
  double sp = 0.0, sr = 0.0;
  for (const auto& m : matched)
  {
    sp += ((R0 * m.est_pos + t0) - m.gt_pos).squaredNorm();
    const Eigen::Matrix3d err_rot = (R0 * m.est_rot).transpose() * m.gt_rot;
    const double a = std::acos(std::clamp((err_rot.trace() - 1.0) / 2.0, -1.0, 1.0));
    sr += a * a;
  }
  ate_first_m = std::sqrt(sp / n);
  are_first_deg = std::sqrt(sr / n) * (180.0 / M_PI);
  return true;
}
}  // namespace livo_recon
