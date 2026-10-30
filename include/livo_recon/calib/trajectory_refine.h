// Batch continuous-time refinement of a LiDAR-only trajectory, for IMU
// Q-scale calibration. See docs/QSCALE_REFINE_RESTRUCTURE_PLAN.md for the
// full rationale: this replaces LiDAR_IMU_Init's ct_refine.cpp (same
// uniform cubic B-spline approach -- cumulative on SO(3), plain on R^3,
// fitted to raw per-point timestamps against point-to-plane residuals)
// but associates against a livo_recon MapBackend (VoxelMap::
// findPlaneResidual(), already built by a prior LIO pass) instead of
// ct_refine's own from-scratch kd-tree + PCA plane fit, and weights each
// residual by that call's real per-point sigma_squared (sensor noise +
// plane-fit uncertainty) instead of ct_refine's single flat 1/sigma.
//
// Residual = n . (R(t_j) x_j + p(t_j)) - d, weight = 1/sqrt(sigma_squared),
// against the FROZEN map built by the prior pass (never updated during
// the solve -- association and optimization are deliberately decoupled,
// same pattern ct_refine already used: re-associate against the current
// trajectory estimate, then solve with those associations' {normal, d,
// weight} held fixed, repeat for opt.outer iterations).
#pragma once

#include "livo_recon/common_lib.h"
#include "livo_recon/map/map_backend.h"

#include <array>
#include <vector>

namespace livo_recon
{
namespace calib
{

// One LiDAR return, body frame, at ITS OWN capture time (not deskewed to
// scan-end) -- matches PointXYZCov::raw_body_point/t/sensor_cov exactly,
// so a caller can build this straight from MeasureGroup::points after
// LioProcBase::deskewAndDownsample() with no extra computation.
struct TrajPoint
{
  V3D raw_body_point = V3D::Zero();
  M3D sensor_cov = M3D::Identity();
  double t = 0.0;   // absolute capture time [s], same base as TrajPose::t
};

struct TrajPose
{
  double t = 0.0;
  M3D rot = M3D::Identity();
  V3D pos = V3D::Zero();
  double var_rot = -1.0;   // -1 = unknown (no covariance pass implemented yet)
  double var_pos = -1.0;
};

struct TrajectoryRefineOptions
{
  double knot = 0.05;             // knot spacing [s]
  double min_dt = 0.5;            // (unused directly here -- the map is already
                                   // frozen from a prior pass, so "no self-match"
                                   // is implicit; kept for parity/future use)
  double prior_rot = 0.05, prior_pos = 0.10;   // weak control-point prior [rad], [m]
  int outer = 6;
  double window_s = 25;           // solve in windows of this length [s] (0: all at once)
  bool verbose = true;

  // Known-still window: every control point whose time falls in here is
  // held FIXED to one shared pose (the Lie-algebra mean of its own
  // initial values) for every outer iteration, instead of being free --
  // ported from LiDAR_IMU_Init's CtOptions::rigid_still, confirmed in
  // that investigation to matter materially (3-11x residual reduction in
  // a genuinely stationary window; ordinary point-to-plane refinement
  // alone does not converge flat there). ABSOLUTE epoch seconds, matching
  // TrajPose::t. Both < 0 (default): disabled.
  double t0_still = -1, t1_still = -1;

  // Diagnostic only: if non-empty, dumps one CSV row per matched point
  // from the LAST outer iteration's association pass (before the final
  // solve updates the spline further) -- t, body-frame range/azimuth,
  // unweighted residual, weight. Empty (default): no dump, no behavior
  // change. See docs/QSCALE_REFINE_RESTRUCTURE_PLAN.md's envelope-
  // artifact investigation for why this was added.
  std::string debug_dump_path;

  // Visual (epipolar) correction, ported from LiDAR_IMU_Init's ct_refine.cpp
  // CameraEpipolarCost -- which was itself derived from livo_recon's own
  // VioAccumulator::computeEpipolarResidualAndJacobian geometry (see that
  // file's header comment), just rewritten as a Ceres-AutoDiff residual over
  // two FREE spline poses instead of one free pose against a fixed EKF
  // anchor. Default disabled (enable=false), so existing callers/behavior
  // are unchanged.
  struct Camera
  {
    bool enable = false;
    std::string track_cache_path;   // livo_recon .rlivtrackcache (cache_tracker_output.cpp)
    double fx = 0, fy = 0, cx = 0, cy = 0;   // camera intrinsics (StateGroup::cameraFx() etc.)
    M3D R_l2c = M3D::Identity();            // StateGroup::R_l2c()/t_l2c() -- LiDAR-to-camera extrinsic
    V3D t_l2c = V3D::Zero();
    double weight_px = 1.0;   // residual weight, in 1/pixel (matches VioProc's own
                               // epipolar_residual_weight convention -- see vio_accumulator.h)
    int max_matches_per_pair = 64;   // cap per (t_a, t_b) frame pair, like ct_refine's own

    // Outlier rejection -- livo_recon's own track-cache match source has
    // NEITHER of these by default (unlike LiDAR_IMU_Init's COLMAP match
    // path, which gets RANSAC-verified two-view geometry for free from
    // COLMAP's own sequential_matcher) -- confirmed this session that the
    // resulting unfiltered outlier tail (residuals stuck at ~300-440px
    // even post-convergence) measurably WORSENS trajectory quality (8-12Hz
    // ripple envelope max +158%) despite the Cauchy robust loss. Both
    // default enabled, so a caller that just sets enable=true gets the
    // same protection LI_Init's COLMAP path gets for free.

    // 1) Per-(t_a,t_b)-pair essential-matrix RANSAC (cv::findEssentialMat),
    // run ONCE when matches are loaded, before any residual ever reaches
    // the solve -- mirrors COLMAP's own two_view_geometries verification,
    // just applied to livo_recon's own tracker output instead of SIFT
    // matches. Threshold is in PIXELS (converted internally to the
    // normalized-bearing scale findEssentialMat needs via pixel_scale).
    // <= 0 disables.
    double ransac_threshold_px = 2.0;
    int ransac_min_matches = 8;   // pairs with fewer matches skip RANSAC (not enough for a
                                   // reliable essential-matrix fit) and are kept as-is

    // 2) Per-outer-iteration residual-based rejection: re-evaluates every
    // match's epipolar residual against the CURRENT trajectory estimate
    // before each outer iteration's solve (same philosophy as the LiDAR
    // plane association's own re-association + outlier rejection, which
    // the camera residual previously had no equivalent of at all) and
    // drops any match whose residual exceeds this threshold FOR THAT
    // iteration only (a match outside the threshold now may re-enter once
    // the trajectory improves). <= 0 disables.
    double outlier_px_threshold = 10.0;
  } camera;
};

struct TrajectoryRefineReport
{
  int outer_done = 0;
  std::vector<int> n_res;
  std::vector<double> rms_before, rms_after;   // weighted point-to-plane rms [m]
  double seconds = 0;
};

class SplineTrajectory
{
public:
  // control points on t_start + (k - 1) * dt, initialised from init_poses
  // (must be time-sorted, t covering [t_start, t_end]).
  void init(const std::vector<TrajPose>& init_poses, double t_start, double t_end, double dt);
  bool valid(double t) const;
  void eval(double t, M3D& R, V3D& p) const;
  double tStart() const { return t0_; }
  double tEnd() const { return t1_; }
  double dt() const { return dt_; }
  size_t size() const { return q_.size(); }
  void locate(double t, int& c, double& u) const;

  std::vector<std::array<double, 4>> q_;   // w, x, y, z
  std::vector<std::array<double, 3>> p_;

  double still_t0_ = -1, still_t1_ = -1;   // rigid-still window actually used, if any

private:
  double t0_ = 0, t1_ = 0, dt_ = 0.05;
};

TrajectoryRefineReport refineTrajectory(const std::vector<TrajPoint>& points,
                                         const std::vector<TrajPose>& init_poses,
                                         MapBackend& map,
                                         SplineTrajectory& traj,
                                         const TrajectoryRefineOptions& opt);

// Samples the trajectory at `rate` Hz (or at the given times if times != nullptr).
std::vector<TrajPose> sampleTrajectory(const SplineTrajectory& traj, double rate,
                                        const std::vector<double>* times = nullptr);

}  // namespace calib
}  // namespace livo_recon
