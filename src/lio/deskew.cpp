#include "livo_recon/lio/deskew.h"

#include "livo_recon/utils/algo/math.h"
#include "livo_recon/utils/algo/hashing.h"
#include "livo_recon/utils/log/debug_log_dir.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <fstream>

namespace livo_recon
{

M3D getBodyCov(const V3D& p_lidar, double sigma_r2, double sigma_a2)
{
  double range = p_lidar.norm();
  if (range < 1e-6)
    return M3D::Identity() * 1e-6;

  V3D dir = p_lidar / range;

  V3D base1;
  if (std::abs(dir.z()) < 0.9)
    base1 = V3D(-dir.y(), dir.x(), 0.0);
  else
    base1 = V3D(0.0, -dir.z(), dir.y());
  base1.normalize();

  V3D base2 = dir.cross(base1);
  base2.normalize();

  M3D dir_hat;
  dir_hat <<     0.0,     -dir.z(),  dir.y(),
              dir.z(),      0.0,    -dir.x(),
             -dir.y(),   dir.x(),     0.0;

  Eigen::Matrix<double, 3, 2> N;
  N.col(0) = base1;
  N.col(1) = base2;

  Eigen::Matrix<double, 3, 2> A = range * dir_hat * N;

  return dir * sigma_r2 * dir.transpose() +
        A * (sigma_a2 * Eigen::Matrix2d::Identity()) * A.transpose();
}

void deskewPoints(
    const StateGroupPtr& state,
    const std::vector<Pose6D>& poses,
    double scan_end_time,
    const std::vector<PointXYZT>& points,
    const DeskewOptions& opts,
    std::vector<PointXYZCov>& points_out)
{
  if (points.empty()) { points_out.clear(); return; }
  points_out.resize(points.size());

  if (poses.empty()) {
    for (size_t i = 0; i < points.size(); ++i)
      points_out[i] = PointXYZCov{ state->lidarToImu(points[i].p), M3D::Zero() };
    return;
  }

  int pose_idx = static_cast<int>(poses.size()) - 1;
  int pt_idx   = static_cast<int>(points.size()) - 1;

  const M3D R_end_T = state->rot().transpose();

  // Computed once, reused per-point below -- see DeskewOptions::
  // time_based_process_noise's docs for the "state" vs "var_acc" modes.
  M3D vel_noise_end = M3D::Zero();
  if (opts.time_based_process_noise == "state") {
    const M3D P_VV = state->cov().block<3, 3>(StateGroup::idxV(), StateGroup::idxV());
    vel_noise_end = R_end_T * P_VV * R_end_T.transpose();
  } else if (opts.time_based_process_noise == "var_acc") {
    vel_noise_end = state->varAcc().asDiagonal();
  }

  // Per-point transform against a given (head, R_base, t_base, vel_term)
  // bracket.
  auto deskewOnePoint = [&](const PointXYZT& pt, const Pose6D& head,
                             const M3D& R_base, const V3D& t_base, const V3D& vel_term) -> PointXYZCov
  {
    const double dt = pt.t - head.t;

    const double alpha    = (head.dt > 1e-9) ? dt / head.dt : 0.0;
    const V3D acc_world   = (1.0 - alpha) * head.acc_head + alpha * head.acc_tail;
    const M3D R_rel       = R_base * Exp(head.gyr, dt);
    const V3D t_rel       = t_base + vel_term * dt + 0.5 * R_end_T * acc_world * dt * dt;
    const V3D p_imu_i   = state->lidarToImu(pt.p);
    const V3D p_imu_end = R_rel * p_imu_i + t_rel;

    const M3D cov_lidar_i   = getBodyCov(pt.p, opts.sigma_r2, opts.sigma_a2);
    const M3D cov_lidar_end = R_rel * cov_lidar_i * R_rel.transpose();
    M3D cov_imu_end   = state->lidarToImu(cov_lidar_end);

    if (opts.time_based_process_noise != "none") {
      const double total_dt = scan_end_time - pt.t;
      if (opts.time_based_process_noise == "state")
        cov_imu_end.noalias() += (total_dt * total_dt) * vel_noise_end;
      else  // "var_acc"
        cov_imu_end.noalias() += (total_dt * total_dt * total_dt) * vel_noise_end;
    }

    return PointXYZCov{ p_imu_end, cov_imu_end };
  };

  while (pose_idx >= 0 && pt_idx >= 0)
  {
    const Pose6D& head = poses[pose_idx--];

    const M3D R_base   = R_end_T * head.rot;
    const V3D t_base   = R_end_T * (head.pos - state->pos());
    const V3D vel_term = R_end_T * head.vel;

    while (pt_idx >= 0 && points[pt_idx].t > head.t)
    {
      points_out[pt_idx] = deskewOnePoint(points[pt_idx], head, R_base, t_base, vel_term);
      pt_idx--;
    }
  }

  // Every point must be matched by construction (poses spans the whole
  // scan window) -- a real, always-active check instead of assert()
  // (compiled out in this Release build, CMAKE_BUILD_TYPE Release, so it
  // would silently leave points_out[0..pt_idx] as Eigen's UNINITIALIZED
  // default-constructed V3D/M3D if this were ever violated). Loud abort
  // with diagnostics, not a silent fallback -- if this fires,
  // propagate()'s pose coverage is genuinely broken and needs fixing,
  // not masking.
  if (pt_idx != -1)
  {
    fprintf(stderr,
        "[deskewPoints] FATAL: %d point(s) unmatched by poses coverage -- "
        "poses span [%.6f, %.6f] (%zu poses), first unmatched point t=%.6f "
        "(points span [%.6f, %.6f], %zu points)\n",
        pt_idx + 1, poses.front().t, poses.back().t, poses.size(),
        points[pt_idx].t, points.back().t, points.front().t, points.size());
    std::abort();
  }
}

}  // namespace livo_recon
