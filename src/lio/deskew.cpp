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
    for (size_t i = 0; i < points.size(); ++i) {
      points_out[i] = PointXYZCov{ state->lidarToImu(points[i].p), M3D::Zero() };
      points_out[i].t = points[i].t;
    }
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

    PointXYZCov out{ p_imu_end, cov_imu_end };
    out.t = pt.t;
    return out;
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

namespace
{

// One point against the spline.  Shared by the full and subset variants so
// there is exactly one definition of what "spline deskew" means.
inline PointXYZCov deskewOnePointSpline(
    const StateGroupPtr& state, const ScanSpline& spline,
    const M3D& R_end_T, const V3D& p_end,
    double scan_end_time, const PointXYZT& pt,
    const DeskewOptions& opts, bool keep_time_noise,
    const M3D& vel_noise_end)
{
  // Pose at the point's own capture time, relative to the scan-end frame.
  const M3D R_i   = spline.rotAt(pt.t);
  const V3D p_i   = spline.posAt(pt.t);
  const M3D R_rel = R_end_T * R_i;
  const V3D t_rel = R_end_T * (p_i - p_end);

  const V3D p_imu_i   = state->lidarToImu(pt.p);
  const V3D p_imu_end = R_rel * p_imu_i + t_rel;

  const M3D cov_lidar_i   = getBodyCov(pt.p, opts.sigma_r2, opts.sigma_a2);
  const M3D cov_lidar_end = R_rel * cov_lidar_i * R_rel.transpose();
  M3D cov_imu_end = state->lidarToImu(cov_lidar_end);

  if (keep_time_noise && opts.time_based_process_noise != "none")
  {
    const double total_dt = scan_end_time - pt.t;
    if (opts.time_based_process_noise == "state")
      cov_imu_end.noalias() += (total_dt * total_dt) * vel_noise_end;
    else
      cov_imu_end.noalias() += (total_dt * total_dt * total_dt) * vel_noise_end;
  }

  PointXYZCov out{ p_imu_end, cov_imu_end };
  out.t = pt.t;
  return out;
}

// The legacy inflation matrix, built once per call.  Zero (and never used)
// unless keep_time_noise is on.
inline M3D splineVelNoise(const StateGroupPtr& state, const DeskewOptions& opts,
                          bool keep_time_noise)
{
  if (!keep_time_noise) return M3D::Zero();
  if (opts.time_based_process_noise == "state") {
    const M3D R_end_T = state->rot().transpose();
    const M3D P_VV = state->cov().block<3, 3>(StateGroup::idxV(), StateGroup::idxV());
    return R_end_T * P_VV * R_end_T.transpose();
  }
  if (opts.time_based_process_noise == "var_acc")
    return state->varAcc().asDiagonal();
  return M3D::Zero();
}

}  // namespace

void deskewPointsSpline(
    const StateGroupPtr& state,
    const ScanSpline& spline,
    double scan_end_time,
    const std::vector<PointXYZT>& points,
    const DeskewOptions& opts,
    bool keep_time_noise,
    std::vector<PointXYZCov>& points_out)
{
  points_out.resize(points.size());
  if (points.empty()) return;

  const M3D R_end_T = spline.rotAt(scan_end_time).transpose();
  const V3D p_end   = spline.posAt(scan_end_time);
  const M3D vel_noise_end = splineVelNoise(state, opts, keep_time_noise);

  for (size_t i = 0; i < points.size(); ++i)
    points_out[i] = deskewOnePointSpline(state, spline, R_end_T, p_end,
                                          scan_end_time, points[i], opts,
                                          keep_time_noise, vel_noise_end);
}

void deskewPointsSplineSubset(
    const StateGroupPtr& state,
    const ScanSpline& spline,
    double scan_end_time,
    const std::vector<PointXYZT>& points,
    const std::vector<int>& indices,
    const DeskewOptions& opts,
    bool keep_time_noise,
    std::vector<PointXYZCov>& points_out)
{
  points_out.resize(indices.size());
  if (indices.empty()) return;

  const M3D R_end_T = spline.rotAt(scan_end_time).transpose();
  const V3D p_end   = spline.posAt(scan_end_time);
  const M3D vel_noise_end = splineVelNoise(state, opts, keep_time_noise);

  for (size_t i = 0; i < indices.size(); ++i)
  {
    const int idx = indices[i];
    if (idx < 0 || idx >= static_cast<int>(points.size())) continue;
    points_out[i] = deskewOnePointSpline(state, spline, R_end_T, p_end,
                                          scan_end_time, points[idx], opts,
                                          keep_time_noise, vel_noise_end);
  }
}

void deskewPointsSplineCsr(
    const StateGroupPtr& state,
    const ScanSpline& spline,
    double scan_end_time,
    const std::vector<PointXYZT>& points,
    const std::vector<int>& offsets,
    const std::vector<int>& members,
    const DeskewOptions& opts,
    bool keep_time_noise,
    std::vector<PointXYZCov>& points_out)
{
  if (offsets.size() < 2) { points_out.clear(); return; }
  const size_t n_out = offsets.size() - 1;
  points_out.resize(n_out);

  const M3D R_end_T = spline.rotAt(scan_end_time).transpose();
  const V3D p_end   = spline.posAt(scan_end_time);
  const M3D vel_noise_end = splineVelNoise(state, opts, keep_time_noise);

  for (size_t i = 0; i < n_out; ++i)
  {
    const int b = offsets[i], e = offsets[i + 1];
    if (e <= b) continue;

    // Single-member cell: identical to deskewPointsSplineSubset().  This is
    // every cell in FIRST mode, so that mode stays bit-for-bit what it was.
    if (e - b == 1)
    {
      const int idx = members[b];
      if (idx < 0 || idx >= static_cast<int>(points.size())) continue;
      points_out[i] = deskewOnePointSpline(state, spline, R_end_T, p_end,
                                           scan_end_time, points[idx], opts,
                                           keep_time_noise, vel_noise_end);
      continue;
    }

    // Multi-member cell: re-place each member from its RAW coordinates, then
    // average exactly as voxelDownsample(AVERAGE) does -- literal mean
    // position, literal mean of each covariance component kept separate, and
    // mean capture time to match the mean position.
    V3D sum_point = V3D::Zero();
    M3D sum_sensor_cov = M3D::Zero();
    M3D sum_pos_cov    = M3D::Zero();
    double sum_t = 0.0;
    int n = 0;
    for (int k = b; k < e; ++k)
    {
      const int idx = members[k];
      if (idx < 0 || idx >= static_cast<int>(points.size())) continue;
      const PointXYZCov d = deskewOnePointSpline(state, spline, R_end_T, p_end,
                                                 scan_end_time, points[idx], opts,
                                                 keep_time_noise, vel_noise_end);
      sum_point      += d.point;
      sum_sensor_cov += d.sensor_cov;
      sum_pos_cov    += d.pos_cov;
      sum_t          += d.t;
      ++n;
    }
    if (n == 0) continue;

    PointXYZCov out{};
    const double dn = static_cast<double>(n);
    out.point      = sum_point / dn;
    out.sensor_cov = sum_sensor_cov / dn;
    out.pos_cov    = sum_pos_cov / dn;
    out.t          = sum_t / dn;
    points_out[i]  = out;
  }
}

}  // namespace livo_recon
