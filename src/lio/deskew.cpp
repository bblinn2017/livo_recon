#include "livo_recon/lio/deskew.h"

#include "livo_recon/utils/algo/math.h"
#include "livo_recon/utils/algo/hashing.h"
#include "livo_recon/utils/algo/symmetrize.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"
#include "livo_recon/lio/relpose_noise.h"

#include <Eigen/Geometry>
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
    std::vector<PointXYZCov>& points_out,
    const std::vector<Eigen::MatrixXd>* imu_process_covariances,
    const Eigen::MatrixXd* p0_at_scan_start,
    RelposeStats* relpose_stats)
{
  const bool relpose_on = opts.time_based_process_noise == "relpose_q" ||
                          opts.time_based_process_noise == "relpose_full";

  if (points.empty()) { points_out.clear(); return; }
  points_out.resize(points.size());

  if (poses.empty()) {
    // No IMU poses to deskew against: points keep their sensor model covariance (never zero).
    for (size_t i = 0; i < points.size(); ++i) {
      points_out[i] = PointXYZCov{ state->lidarToImu(points[i].p), state->lidarToImu(getBodyCov(points[i].p, opts.sigma_r2, opts.sigma_a2)) };
      points_out[i].t = points[i].t;
      points_out[i].raw_body_point = points_out[i].point;  // no motion to compensate
    }
    if (relpose_on && relpose_stats) { ++relpose_stats->scans_total; ++relpose_stats->scans_skipped_no_imu; }
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

  // R97A R2: relpose_q / relpose_full build a suffix-sum table of the per-step process-noise
  // blocks (rotated into the scan-end body frame), used in place of the var_acc/state term above.
  RelposeSuffixTables relpose_tabs;
  bool relpose_have_steps = false;
  M3D relpose_P_vv_end = M3D::Zero(), relpose_P_baba_end = M3D::Zero(),
      relpose_P_vba_end = M3D::Zero(), relpose_P_bgbg_end = M3D::Zero();
  bool relpose_full_have_p0 = false;
  if (relpose_on) {
    if (relpose_stats) ++relpose_stats->scans_total;
    if (imu_process_covariances && imu_process_covariances->size() == poses.size() && !poses.empty()) {
      std::vector<RelposeStepNoise> steps(poses.size());
      for (size_t k = 0; k < poses.size(); ++k) {
        const Eigen::MatrixXd& Qw = (*imu_process_covariances)[k];
        RelposeStepNoise& s = steps[k];
        s.t = poses[k].t;
        s.dt = poses[k].dt;
        s.Q_theta = R_end_T * Qw.block<3, 3>(StateGroup::idxR(), StateGroup::idxR()) * R_end_T.transpose();
        s.Q_v     = R_end_T * Qw.block<3, 3>(StateGroup::idxV(), StateGroup::idxV()) * R_end_T.transpose();
        s.Q_p     = R_end_T * Qw.block<3, 3>(StateGroup::idxP(), StateGroup::idxP()) * R_end_T.transpose();
        s.Q_pv    = R_end_T * Qw.block<3, 3>(StateGroup::idxP(), StateGroup::idxV()) * R_end_T.transpose();
      }
      relpose_tabs = buildRelposeSuffixTables(steps, scan_end_time);
      relpose_have_steps = true;
    }
    if (opts.time_based_process_noise == "relpose_full" && p0_at_scan_start && p0_at_scan_start->rows() > 0) {
      const Eigen::MatrixXd& P0 = *p0_at_scan_start;
      const M3D P_vv   = P0.block<3, 3>(StateGroup::idxV(), StateGroup::idxV());
      const M3D P_baba = P0.block<3, 3>(StateGroup::idxBA(), StateGroup::idxBA());
      const M3D P_vba  = P0.block<3, 3>(StateGroup::idxV(), StateGroup::idxBA());
      const M3D P_bgbg = P0.block<3, 3>(StateGroup::idxBG(), StateGroup::idxBG());
      relpose_P_vv_end   = R_end_T * P_vv * R_end_T.transpose();
      relpose_P_baba_end = R_end_T * P_baba * R_end_T.transpose();
      relpose_P_vba_end  = R_end_T * P_vba * R_end_T.transpose();
      relpose_P_bgbg_end = R_end_T * P_bgbg * R_end_T.transpose();
      relpose_full_have_p0 = true;
    }
    if (!relpose_have_steps && relpose_stats) ++relpose_stats->scans_skipped_no_imu;
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
    const M3D cov_lidar_end = symmetrized(M3D(R_rel * cov_lidar_i * R_rel.transpose()));   // R80 F-419 (Bryce): per-point covariance, no exemption
    M3D cov_imu_end   = state->lidarToImu(cov_lidar_end);

    if (relpose_on) {
      // R97A R2: process-noise-only relative-pose covariance, in place of the var_acc/state term.
      M3D sigma_theta = M3D::Zero(), sigma_p = M3D::Zero();
      if (relpose_have_steps)
        relposeProcessNoiseFromSuffix(relpose_tabs, pt.t, sigma_theta, sigma_p);
      if (relpose_full_have_p0) {
        const double Delta = scan_end_time - pt.t;
        const double a = Delta, b = 0.5 * Delta * Delta;
        sigma_p.noalias() += a * a * relpose_P_vv_end + b * b * relpose_P_baba_end +
            a * b * (relpose_P_vba_end + relpose_P_vba_end.transpose());
        sigma_theta.noalias() += a * a * relpose_P_bgbg_end;
      }
      M3D q_skew;
      q_skew <<            0.0, -p_imu_end.z(),  p_imu_end.y(),
                  p_imu_end.z(),            0.0, -p_imu_end.x(),
                 -p_imu_end.y(),  p_imu_end.x(),            0.0;
      const M3D sigma_add = sigma_p + q_skew * sigma_theta * q_skew.transpose();
      cov_imu_end.noalias() += sigma_add;
    } else if (opts.time_based_process_noise != "none") {
      const double total_dt = scan_end_time - pt.t;
      if (opts.time_based_process_noise == "state")
        cov_imu_end.noalias() += (total_dt * total_dt) * vel_noise_end;
      else  // "var_acc"
        cov_imu_end.noalias() += (total_dt * total_dt * total_dt) * vel_noise_end;
    }
    cov_imu_end = symmetrized(cov_imu_end);   // R80 F-419 (Bryce): covers lidarToImu() and the process-noise sums above

    PointXYZCov out{ p_imu_end, cov_imu_end };
    out.t = pt.t;
    out.raw_body_point = p_imu_i;  // CQ-50: pre-warp, own-capture-time body point
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

}  // namespace livo_recon
