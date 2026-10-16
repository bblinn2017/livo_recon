#pragma once

#include <string>
#include <vector>

#include "livo_recon/utils/data/data_wrappers.h"
#include "livo_recon/utils/state/state.h"
#include <Eigen/Dense>

// History (9-22): see docs/livo_recon_changelog.md#include-livo_recon-lio-deskew.h-9
namespace livo_recon
{

struct DeskewOptions
{
  // "none" | "state" | "var_acc" | "relpose_q" | "relpose_full" -- see deskewPoints()'s own doc comment.
  std::string time_based_process_noise = "var_acc";

  // Per-point sensor noise model params (range-noise variance, transverse
  // angular-noise variance) -- mirrors ImuProcOptions::sigma_r2/sigma_a2
  // exactly, just passed in directly instead of read off an ImuProc
  // instance, since this file has no ImuProc dependency.
  double sigma_r2 = 0.0025;   // (0.05m)^2, ImuProcOptions' own default
  double sigma_a2 = 1.218e-5; // sin(0.2deg)^2, ImuProcOptions' own default
};

// R97A R2: engagement counter for the relpose_* modes -- scans where no relative-pose term could
// be added at all (no IMU steps this scan, e.g. the first scan after calibration).
struct RelposeStats
{
  long scans_total = 0;
  long scans_skipped_no_imu = 0;
};

// Isotropic-in-range, anisotropic-in-angle per-point sensor noise model
// (range along the ray, angle transverse to it) in the LiDAR's own body
// frame -- moved verbatim from ImuProc::getBodyCov(), only decoupled from
// ImuProcOptions (sigma_r2/sigma_a2 passed directly instead).
M3D getBodyCov(const V3D& p_lidar, double sigma_r2, double sigma_a2);

// Core deskew: `points` (ASCENDING-time-sorted -- true of raw sensor
// output like mg.lidar_points) into `state`'s CURRENT end-of-scan
// reference frame, using `poses` (the fixed, once-per-frame IMU-
// integrated intra-scan pose sequence -- ImuProc::propagate()'s output)
// as the piecewise trajectory this reference frame is expressed relative
// to. `scan_end_time` is the frame's own reference timestamp
// (mg.image.t), used only for the time_based_process_noise covariance-
// inflation term below.
//
// Called ONCE per frame, with `state` exactly as ImuProc::propagate()
// left it.
//
// `time_based_process_noise`: inflates each point's returned covariance
// by a term growing with (scan_end_time - point's own capture time) --
// "state" rotates the state's own P_VV (velocity-uncertainty block) into
// the scan-end body frame; "var_acc" uses a fixed diag(varAcc) proxy
// instead (see ImuProcOptions::time_based_process_noise's original doc
// comment for the full derivation of both). "none" disables it.
// R97A R2: `imu_process_covariances`/`p0_at_scan_start` are only used (and may be null/empty)
// under time_based_process_noise = relpose_q | relpose_full -- see the spec. `imu_process_covariances`
// must align with `poses` (same length, same order) when non-null. `p0_at_scan_start` is the
// error-state covariance at the START of this scan's IMU propagation (before any step of it),
// needed by relpose_full only. `relpose_stats`, when non-null, is updated with engagement counts
// (scans where no relative-pose term could be added -- no IMU steps this scan).
void deskewPoints(
    const StateGroupPtr& state,
    const std::vector<Pose6D>& poses,
    double scan_end_time,
    const std::vector<PointXYZT>& points,
    const DeskewOptions& opts,
    std::vector<PointXYZCov>& points_out,
    const std::vector<Eigen::MatrixXd>* imu_process_covariances = nullptr,
    const Eigen::MatrixXd* p0_at_scan_start = nullptr,
    RelposeStats* relpose_stats = nullptr);

}  // namespace livo_recon
