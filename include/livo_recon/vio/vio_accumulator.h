#pragma once

#include "livo_recon/vio/tracked_frame.h"
#include "livo_recon/utils/state/state.h"
#include "livo_recon/utils/algo/ekf.h"

// VIO's epipolar-line residual -> HtH/Htz accumulation, lifted out of
// VioProc (2026-08-14) so both VioProc's own sequential solve AND
// CombinedProc's joint LIO+VIO solve can call the exact same math without
// CombinedProc depending on VioProc's private internals -- mirrors
// lio/lio_accumulator.h's split for the LIO side. Needs more per-frame
// context than the LIO accumulator (camera intrinsics, tracked anchors),
// so it's a small stateful class (holding state_, like VioProc itself
// did) rather than a single free function. Deliberately does NOT hold a
// TrackerPtr (2026-08-14) -- frame.rot_anchor/pos_anchor are already
// resolved by Tracker::resolveAnchorPose() before accumulate() is ever
// called (see Tracker's own doc comment for that mechanism), so this
// class has no need to reach back into Tracker itself.
namespace livo_recon
{

// Quantities used by computeEpipolarResidualAndJacobian that depend only
// on the current pose estimate (not on any individual point), computed
// once per accumulate() call and reused across every point.
struct ReprojectionPrecomputed
{
  M3D R_common;  // rot_curr^T * frame.rot_anchor -- maps an anchor-body-frame vector into the current body frame
  V3D t_common;  // rot_curr^T * (frame.pos_anchor - pos_curr)
  M3D Ric;
  V3D tic;
  M3D Rcw;       // == Rcw_curr; d(p_cam)/d(dt) = -Rcw
};

// Two-ray closest-approach solve output (VioAccumulator::solveClosestRayPoints) --
// s is the depth along the anchor ray, u along the compared (current or,
// under bidirectional, reverse-anchor) ray, both floored at min_depth.
struct RayClosestApproach
{
  double s = 0.0;
  double u = 0.0;
  bool s_clamped = false;
  bool u_clamped = false;
  bool parallel = false;
};

// A lean subset of VioProcOptions that the accumulator actually needs --
// declared here (not in vio_processing.h) so this file doesn't depend on
// VioProc's full option set (logging toggles, etc. that only the
// sequential-mode caller cares about). VioProc constructs one of these
// from its own VioProcOptions each call.
struct VioAccumulateOptions
{
  double pos_epipolar_residual_var = 0.0556;
  double neg_epipolar_residual_var = 0.03125;
  double anchor_split_depth        = 0.0;
  bool   bidirectional              = true;
  bool   distortion_weight_on       = false;
  double weight_pixel_noise_px      = 0.5;

  // DOF-split weighting (2026-08-15): a single epipolar residual's weight
  // used to land identically on BOTH the rotation and position columns of
  // J when forming HtH/Htz (thread_HtH += J^T * weight * J), even though
  // epipolar geometry is fundamentally scale-ambiguous -- it constrains
  // translation DIRECTION and rotation well, but has no way to observe
  // absolute translation MAGNITUDE the way LIO's metric point-to-plane
  // residuals can. Letting VIO's position-column contribution carry the
  // same confidence as its rotation-column contribution lets a single
  // marginal-quality frame's position "vote" get baked permanently into
  // the shared voxel map (see vio_processing.cpp/livo_recon_node.cpp's
  // updateMaps()-after-estimateState() ordering), corrupting LIO's own
  // future residuals against a map LIO never agreed to.
  //
  // pos_weight_scale multiplies ONLY the position columns of J before
  // forming HtH/Htz -- rotation columns always keep the full computed
  // weight. 1.0 (default) reproduces the old behavior exactly (no DOF
  // split). Set < 1.0 to let VIO keep correcting rotation at full
  // confidence while trusting it less for position, deferring position
  // consistency to LIO's global map.
  double pos_weight_scale          = 1.0;

  // residual_mode (2026-08-16, ported from FAST-LIVO2's livo_vio ResidualMode
  // ::kMixed -- see vio_proc.cpp's own accumulation loop): "epipolar"
  // (default) is the original single epipolar-line residual above, whose
  // pos_weight_scale knob is a blunt way to distrust its position-column
  // contribution. "mixed" instead runs TWO independently-weighted residuals
  // per point -- the same epipolar-line residual with its POSITION columns
  // zeroed (rotation-only), plus a new ray-gap residual (perpendicular
  // distance between the anchor and current rays' closest-approach points,
  // computed via the SAME solveClosestRayPoints() already used for the
  // epipolar depth-clamp decision) with its ROTATION columns zeroed
  // (position-only) -- a principled decoupling rather than a scale-down.
  // pos_weight_scale is ignored in "mixed" mode (the split is already
  // structural, not a weight knob). A point is accepted only if BOTH
  // residuals compute successfully (matches FAST-LIVO2's own behavior).
  std::string residual_mode         = "epipolar";
  double pos_ray_distance_residual_var = 1000.0;  // free-solve (unclamped) ray-gap weight -- FAST-LIVO2's own tuned default
  double neg_ray_distance_residual_var = 10.0;    // clamped (either ray) ray-gap weight -- FAST-LIVO2's own tuned default

  // Diagnostic logging -- see VioProcOptions's matching fields for the
  // full doc comments. Kept here (not stripped) so accumulate() can
  // reproduce IDENTICAL log output to before this extraction.
  bool        log_point_residuals  = false;
  bool        log_residual_counts  = false;
  double      log_norm_l_low_frac  = 0.3;
  std::string log_path             = "/tmp/livo_recon_myvio_log.txt";
};

class VioAccumulator
{
public:
  explicit VioAccumulator(StateGroupPtr state);

  // Builds this frame's residuals from `frame`/`anchors` against the
  // CURRENT state_ estimate, accumulates them into `out` (does NOT call
  // out.applyMeanUpdate()/applyCovarianceUpdate() -- caller owns the EKF
  // solve step). avg_error/n_valid are the same diagnostics VioProc's
  // sequential solveSystem() has always computed. Returns false if there
  // were no valid residuals this call (out/avg_error untouched).
  bool accumulate(const TrackedFrame& frame, const std::vector<AnchorPoint>& anchors,
                   const VioAccumulateOptions& opts, EkfUpdate& out,
                   float& avg_error, int& n_valid, double t_abs, int iter) const;

private:
  bool computeEpipolarResidualAndJacobian(const ReprojectionPrecomputed& pre,
                                          double fx, double fy, double cx, double cy,
                                          const cv::Point2f& uv_seed,
                                          const cv::Point2f& uv_tracked,
                                          const Eigen::Matrix2d& anchor_uv_jacobian,
                                          const Eigen::Matrix2d& curr_uv_jacobian,
                                          const VioAccumulateOptions& opts,
                                          double& residual_out,
                                          Eigen::Matrix<double, 1, 6>& jacobian_out,
                                          double& norm_l_out,
                                          bool& depth_clamped_out,
                                          double* weight_var_out) const;

  bool computeDirectionalEpipolar(const V3D& t_cc, const V3D& y,
                                  const M3D& dtcc_dtheta, const M3D& dtcc_dt, const M3D& dy_dtheta,
                                  const V3D& x_compare, const VioAccumulateOptions& opts,
                                  double& residual_out, Eigen::Matrix<double, 1, 6>& jacobian_out,
                                  double& norm_l_out, bool* depth_clamped_out) const;

  double epipolarLineWeightVar(const V3D& t_cc, const V3D& y, const V3D& x1, const M3D& R_cc,
                               const Eigen::Matrix2d& anchor_uv_jacobian,
                               const Eigen::Matrix2d& curr_uv_jacobian,
                               double fx, double fy, const VioAccumulateOptions& opts) const;

  // Ray-gap residual for "mixed" residual_mode -- see VioAccumulateOptions::
  // residual_mode's doc comment. Returns the FULL 6-column Jacobian (both
  // rotation and position blocks populated, like computeEpipolarResidual
  // AndJacobian's own jacobian_out) -- accumulate() is responsible for
  // zeroing whichever columns this residual shouldn't influence (rotation,
  // for the position-only role this residual plays in "mixed" mode).
  bool computeRayDistanceResidualAndJacobian(const ReprojectionPrecomputed& pre,
                                             double fx, double fy, double cx, double cy,
                                             const cv::Point2f& uv_seed,
                                             const cv::Point2f& uv_tracked,
                                             const Eigen::Matrix2d& anchor_uv_jacobian,
                                             const Eigen::Matrix2d& curr_uv_jacobian,
                                             const VioAccumulateOptions& opts,
                                             double& residual_out,
                                             Eigen::Matrix<double, 1, 6>& jacobian_out,
                                             bool& depth_clamped_out,
                                             double* weight_var_out) const;

  RayClosestApproach solveClosestRayPoints(const V3D& y, const V3D& x_compare, const V3D& t_cc,
                                           double min_depth) const;

  ReprojectionPrecomputed computeReprojectionPrecomputed(const M3D& rot_curr, const V3D& pos_curr,
                                                         const M3D& Rcw_curr,
                                                         const M3D& rot_anchor, const V3D& pos_anchor) const;

  StateGroupPtr state_;
};

}  // namespace livo_recon
