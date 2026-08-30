#pragma once

#include <string>
#include <vector>

#include "livo_recon/utils/data/data_wrappers.h"
#include "livo_recon/utils/state/state.h"

// Per-point LiDAR deskewing. Only ONE strategy lives here now:
// deskewPoints(), a one-shot deskew against the piecewise-IMU-integrated
// trajectory `poses` (ImuProc::propagate()'s output), using the IMU-only
// propagated state. Never re-evaluated once the IEKF loop starts
// correcting state_ -- only the single scan-end pose ever gets corrected.
//
// 2026-08-24: REMOVED an experimental "iterative deskew" alternative
// (deskewAndSelect()/evalSpline(), a re-evaluated-every-IEKF-iteration
// cubic Hermite spline over the whole scan) -- never enabled in any
// production config, and its one validated result came with a known,
// unresolved regression on 3 HILTI sequences. Full design, math, and
// removal rationale preserved at
// docs/removed_livo_recon_spline_deskew_2026aug24.md; literal pre-removal
// code at refactor_snapshots/remove_livo_recon_spline_deskew_2026aug24/.
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
    std::vector<PointXYZCov>& points_out);

}  // namespace livo_recon
