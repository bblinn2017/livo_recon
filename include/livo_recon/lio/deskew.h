#pragma once

#include <string>
#include <vector>

#include "livo_recon/utils/data/data_wrappers.h"
#include "livo_recon/utils/state/state.h"
#include "livo_recon/lio/spline.h"

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

// Spline deskew.  Places every point at the pose the spline gives for the
// point's OWN timestamp, expressed in the scan-end IMU frame, and rotates
// its sensor covariance by the same relative rotation.  Unlike
// deskewPoints() this is safe -- and intended -- to call repeatedly within
// one frame: re-anchor the spline to the corrected state (ScanSpline::
// anchorTo()) and call this again, and every point moves with the
// correction instead of only the scan-end pose moving.
//
// The time_based_process_noise inflation is deliberately NOT applied here.
// That term exists to stand in for trajectory uncertainty that the one-shot
// deskew cannot express, and it does so by putting a PROCESS-noise quantity
// (state->varAcc(), the same variable cov_w's accelerometer block is built
// from) inside a MEASUREMENT covariance -- the same number on both sides of
// S = H P H^T + R.  With the spline carrying intra-scan trajectory shape
// explicitly, the term has nothing left to stand in for, and dropping it
// restores a clean separation between Q and R.  That separation is a
// precondition for AdaptiveQ meaning anything: with var_acc in R as well as
// in Q, "measure the noise, apply it as process noise" would feed back into
// its own measurement.  Set spline/keep_time_noise to restore the legacy
// term for an A/B against this reasoning.
void deskewPointsSpline(
    const StateGroupPtr& state,
    const ScanSpline& spline,
    double scan_end_time,
    const std::vector<PointXYZT>& points,
    const DeskewOptions& opts,
    bool keep_time_noise,
    std::vector<PointXYZCov>& points_out);

// As above, but only for the points at `indices` into `points`, writing into
// points_out[i] for i in [0, indices.size()).  Used for the per-IEKF-
// iteration re-deskew: the voxel downsample runs ONCE (on iteration 0) and
// its surviving raw indices are reused every iteration afterwards, so the
// re-deskew costs one spline evaluation per kept point and no re-hashing.
// Re-running the downsample itself each iteration would also let the KEPT
// SET change between iterations, which would make the residual count --
// and therefore the update -- move for reasons unrelated to the state.
// As deskewPointsSplineSubset(), but over the CSR membership set that
// voxelDownsampleIndexedCsr() produces, so it covers ds_mode:=average as
// well as first.  points_out[i] is rebuilt from the members of output point
// i: each member is re-placed from its RAW coordinates against the current
// spline, and the results are then averaged exactly as
// voxelDownsample(AVERAGE) averages them.  A single-member cell -- every
// cell in FIRST mode -- takes the same code path as the old subset call and
// is bit-identical to it.
void deskewPointsSplineCsr(
    const StateGroupPtr& state,
    const ScanSpline& spline,
    double scan_end_time,
    const std::vector<PointXYZT>& points,
    const std::vector<int>& offsets,
    const std::vector<int>& members,
    const DeskewOptions& opts,
    bool keep_time_noise,
    std::vector<PointXYZCov>& points_out);

void deskewPointsSplineSubset(
    const StateGroupPtr& state,
    const ScanSpline& spline,
    double scan_end_time,
    const std::vector<PointXYZT>& points,
    const std::vector<int>& indices,
    const DeskewOptions& opts,
    bool keep_time_noise,
    std::vector<PointXYZCov>& points_out);

}  // namespace livo_recon
