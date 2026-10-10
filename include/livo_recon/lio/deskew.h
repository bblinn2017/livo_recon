#pragma once

#include <string>
#include <vector>

#include "livo_recon/utils/data/data_wrappers.h"
#include "livo_recon/utils/state/state.h"

// History (9-22): see docs/livo_recon_changelog.md#include-livo_recon-lio-deskew.h-9
namespace livo_recon
{

struct DeskewOptions
{
  // "none" | "state" | "var_acc" -- see deskewPoints()'s own doc comment.
  std::string time_based_process_noise = "var_acc";

  // Per-point sensor noise model params (range-noise variance, transverse
  // angular-noise variance) -- mirrors ImuProcOptions::sigma_r2/sigma_a2
  // exactly, just passed in directly instead of read off an ImuProc
  // instance, since this file has no ImuProc dependency.
  double sigma_r2 = 0.0025;   // (0.05m)^2, ImuProcOptions' own default
  double sigma_a2 = 1.218e-5; // sin(0.2deg)^2, ImuProcOptions' own default

  // R91H (diagnostic, experimental): scale of the ROTATION part of the within-scan warp. Each point is carried to the scan end by
  // R_rel = R_end^T * R_head * Exp(gyr, dt); with a scale s != 1 the rotation actually applied is the same axis with the angle times s (Eigen::AngleAxis).
  // s = 1 is the baseline and takes the unchanged code path (R_rel is not touched). s = 0 applies no rotation. The translation part of the warp
  // (t_rel) is NOT changed. Valid as a test only while the sensor is known to be stationary (there the within-scan IMU rotation is IMU error).
  double rot_warp_scale = 1.0;
};

// R91H: counters of the rotation part of the warp for one call (filled only when a pointer is passed). theta = the rotation angle of R_rel in radians.
struct DeskewRotStats
{
  long   points = 0;            // points warped (poses non-empty)
  long   points_rescaled = 0;   // points whose applied rotation differs from R_rel (scale != 1 and theta > 0)
  double sum_theta_raw = 0.0;   // sum over points of theta
  double sum_theta_applied = 0.0;
  double max_theta_raw = 0.0;
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
void deskewPoints(
    const StateGroupPtr& state,
    const std::vector<Pose6D>& poses,
    double scan_end_time,
    const std::vector<PointXYZT>& points,
    const DeskewOptions& opts,
    std::vector<PointXYZCov>& points_out,
    DeskewRotStats* rot_stats = nullptr);   // R91H: optional counters; nullptr (the default) changes nothing

}  // namespace livo_recon
