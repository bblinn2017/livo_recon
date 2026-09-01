#pragma once

#include "livo_recon/utils/map/voxelmap_utils.h"
#include <cstdint>

namespace livo_recon
{

class VoxelPlane
{
public:
  VoxelPlane(VoxelOptsPtr opts);

  bool isInit() const { return points_size_ > opts_->min_init_points; }
  bool isPlane() const { return is_plane_; }
  bool isFull() const { return points_size_ >= opts_->max_points; }

  // scan_id (default -1 -> no logging regardless of opts_ flags): the
  // calling frame's VoxelMap::frame_idx_, threaded down only for T0-D's
  // corr.csv (see VoxelOpts::log_consistency_corr_en) -- not used for
  // anything else. Callers outside the VoxelMap frame-processing path
  // (none today) can safely omit it.
  bool computeResidual(const WorldPointCov& pt, Residual& res, int scan_id = -1) const;
  bool getVizInfo(PlaneVizInfo& info) const;

  int pointsSize() const { return points_size_; }
  int lastFitJ() const { return last_fit_j_; }

  // Diagnostic-only accessors (see the point_filter_num=1 single-frame-
  // init failure-mode investigation): expose the raw PCA outputs so a
  // caller can log them without duplicating the eigen-decomposition.
  const V3D& eigenValues() const { return eigen_values_; }
  const M3D& planeVar() const { return plane_var_; }
  float radius() const { return radius_; }

  // total_count (default -1 -> use points.size()): the TRUE number of raw
  // points `points` summarizes, for isInit()/isFull() gating -- lets a
  // caller pass a much smaller set of pre-binned representative points
  // (see VoxelNode::always_update's bin accumulator) while still gating
  // convergence against the real accumulated point count, not the
  // (much smaller) representative-point count.
  //
  // weights (default nullptr): per-entry weight, parallel to `points`.
  // When null, falls back to today's behavior exactly (unweighted) --
  // unchanged for callers that still pass raw points (normal/frame_gated
  // modes, where points_ stays bounded by max_points anyway). When
  // provided, points/weights are used directly with NO internal binning:
  // mean = sum(w_i*p_i)/sum(w_i), covariance = sum(w_i*(p_i-mean)(p_i-mean)^T)/sum(w_i),
  // and the plane_var_ Jacobian uses w_i/sum(w_i) per entry. Passing
  // weight_i = each bin's own raw point count gives an EXACT reconstruction
  // of the unweighted PCA over every raw point the bins summarize (see
  // VoxelNode::buildBinReps()).
  //
  // running (default nullptr): when non-null (VoxelNode's default
  // equal-weight points_ path only -- see RunningMoments' docs), mean/
  // covariance are derived from it in O(1) instead of the loops above --
  // exact, not an approximation. `points` is still required alongside it
  // for the per-point plane_var_ Jacobian loop, which this does not
  // affect. Must not be passed together with `weights`.
  // `var_weights`: separate weight vector for the plane_var_ Jacobian
  // loop only (see T3-0c) -- lets a caller change the fitted plane
  // (center/covariance/normal, via `weights`) independently of the
  // uncertainty scaling attached to it. Defaults to `weights` itself
  // when null, matching pre-2026-08-30 behavior (one weight vector for
  // both fit and uncertainty).
  void update(const std::vector<PointXYZCov>& points, int total_count = -1,
              const std::vector<double>* weights = nullptr,
              const RunningMoments* running = nullptr,
              const std::vector<double>* var_weights = nullptr);

  // opts_->plane_fit_mode == "debiased" path: folds `points` into this
  // VoxelPlane's own PERSISTENT O(1) accumulators (never cleared/replayed)
  // and refits normal_/plane_/plane_var_ from them directly -- see
  // docs/debiased_voxel_plane_fit_2026aug24.md for the full derivation.
  // Unlike update(), safe (and intended) to call on every incoming batch
  // forever; never needs the caller to retain raw points.
  //
  // distinct_frames (default -1 -> treated as 1, i.e. maximally
  // conservative/no between-frame correction): VoxelNode's count of
  // distinct frames that have contributed to this voxel so far. Needed for
  // the shared-pose-noise shrinkage correction in refitDebiased() -- see
  // docs/debiased_voxel_plane_fit_2026aug24.md's "single-frame
  // over-subtraction" section. Cheap to pass every call; only the latest
  // value is kept.
  //
  // trust_sensor_noise (default true): false for points from the
  // calibration/bootstrap window (VoxelNode's g_current_frame_idx == 0) --
  // that window aggregates many scans from a STATIONARY sensor looking at
  // the same geometry from the same vantage point repeatedly, so per-point
  // sensor noise is correlated across those points (shared range/
  // incidence-angle-driven error), not independent the way it is across
  // genuinely different frames/viewpoints once the sensor is moving.
  // Subtracting the full per-point sensor_cov sum as if independent
  // over-corrects there the same way naively subtracting shared pose_cov
  // per point did (see refitDebiased()'s pose_shrink) -- confirmed
  // empirically 2026-08-24 (docs/debiased_voxel_plane_fit_2026aug24.md).
  // Points with trust_sensor_noise=false still contribute to Sp_/Spp_ (the
  // fit itself) but not to Scov_sensor_ (what gets subtracted) -- a voxel
  // fed only by bootstrap points gets plain, uncorrected-for-sensor-noise
  // scatter (safe/conservative, matches pca-mode's own untouched
  // behavior), and the correction phases in normally once real,
  // independent-viewpoint frames start contributing.
  void addPoints(const std::vector<PointXYZCov>& points, int total_count = -1,
                 int distinct_frames = -1, bool trust_sensor_noise = true);

private:
  void refitDebiased();

  // Persistent accumulators for addPoints()/refitDebiased() -- RAW
  // (unweighted, w_i=1) sums, never reset. Normalization (by N or N^2, see
  // refitDebiased()) is applied only at query time.
  double N_acc_ = 0.0;
  V3D    Sp_    = V3D::Zero();  // Sigma p_i
  M3D    Spp_   = M3D::Zero();  // Sigma p_i p_i^T
  M3D    Scov_  = M3D::Zero();  // Sigma (pt.sensor_cov+pt.pos_cov), combined -- plane_var_'s accumulators only
  M3D    Scov_sensor_ = M3D::Zero(); // Sigma pt.sensor_cov, independent -- M_debiased's fit correction
  M3D    V_[3][3];              // V_[a][b] = Sigma p_i(a) p_i(b) Cov_i  (combined cov, plane_var_ only)
  M3D    W_[3];                 // W_[a]    = Sigma p_i(a) Cov_i        (combined cov, plane_var_ only)
  double sum_sensor_var_ = 0.0; // Sigma sensor_var_i (see sensor_noise_floor_eig0)
  int    distinct_frames_ = 1;  // latest VoxelNode::distinct_frames_ seen by addPoints()

  // sigma_diag_squared: independent (sensor+pose) noise VARIANCE only.
  // plane_var_term: this plane's own fit-uncertainty contribution, returned
  // separately so the caller (VoxelPlane::computeResidual()) can add it
  // into Residual::sigma_squared. The accept/reject gate still uses the
  // full (sigma_diag_squared + plane_var_term) combined variance.
  // is_candidate (default nullptr): set true as soon as this point clears
  // the purely-geometric checks (finite r, within max_radius*radius_ of
  // this plane's center) -- i.e. "this is a real correspondence, whatever
  // the chi2 test below decides" -- regardless of gate()'s own return
  // value (which still means the full accept/reject decision, unchanged).
  // T0-D's corr.csv wants exactly this population (pre-outlier-gate), see
  // computeResidual().
  // dropped_by_ablation (default nullptr): set true when T3-0e's
  // occ_aniso_drop_mode rejected this correspondence (distinct from the
  // ordinary sigma-gate rejection) -- set AFTER *is_candidate, so a drop
  // still reaches corr.csv instead of silently vanishing (code-review
  // fix, 2026-08-31 -- T0-D's own "log before the gate" guardrail).
  bool gate(const V3D& p, const M3D& sensor_cov, const M3D& pose_cov,
            double& r, double& sigma_diag_squared, double& plane_var_term,
            Eigen::Matrix<double, 1, 3>& J_nq, bool* is_candidate = nullptr,
            bool* dropped_by_ablation = nullptr) const;

  VoxelOptsPtr opts_;

  // Minimal, lossless plane-uncertainty state over [theta1, theta2, d] --
  // theta1/theta2 are the only 2 dimensions today's ambient 6x6 J_n ever
  // actually carried information along (its output is always confined to
  // span{y_normal_, x_normal_}, never along the normal's own radial
  // direction), plus d, the plane's scalar offset. The other 4 dimensions
  // of the old 6x6 (normal's radial direction + center's 2 in-plane
  // directions) never carried real information: normal's radial direction
  // can't move without breaking unit norm, and center's in-plane
  // components are exactly orthogonal to the residual r=n.(p-center) (see
  // gate()'s dr/dcenter=-n). A lossless 3x6 projection T (rows
  // [y_normal_^T,0], [x_normal_^T,0], [0,-normal_^T]) compresses the old
  // state down to this one with no loss -- see update()'s per-point
  // Jmin construction (the direct, already-projected equivalent of
  // T * J_old).
  M3D plane_var_;
  M3D covariance_;
  PlaneInfo plane_;

  V3D y_normal_;
  V3D x_normal_;
  float radius_ = 0;
  V3D eigen_values_ = V3D::Ones();

  int points_size_ = 0;
  bool is_plane_ = false;

  // Occupied-bin count (J) from this plane's last update() call that used
  // weights (0 for an unweighted/unbinned fit, or for a refitDebiased()
  // fit -- debiased mode has no binning concept) -- stored so
  // computeResidual() can log it per-correspondence without recomputing
  // (see VoxelOpts::log_consistency_covariates_en). Mirrors
  // debugLogPlaneFitStats()'s own `j` local exactly, just persisted.
  int last_fit_j_ = 0;

  // T3-0e (2026-08-31): 8x8 tangent-frame occupancy bitmask, cell size
  // opts_->voxel_size/8 -- see updateOccupancy()'s doc comment (in the
  // .cpp) for the full design, anchoring, and reset rule. Same structure
  // T8-b's card independently proposes reusing later (J_eff = popcount,
  // the gap detector, the m2 leverage term); built here first because
  // T3-0e needs it now and T8-b is not yet scheduled.
  uint64_t occupancy_bitmask_ = 0;
  V3D occ_anchor_normal_   = V3D::Zero();
  V3D occ_anchor_x_normal_ = V3D::Zero();
  V3D occ_anchor_y_normal_ = V3D::Zero();
  V3D occ_anchor_center_   = V3D::Zero();
  bool occ_anchored_ = false;

  // Code-review fix, 2026-08-31: occupancyAnisotropy() used to heap-
  // allocate a std::vector and scan all 64 cells on EVERY call, including
  // from computeResidual() inside LioProc::buildResiduals()'s OMP
  // parallel-for -- once per correspondence, twice when covariates
  // logging is on. Cached here instead (recomputed once per fit, in
  // recomputeOccupancyCache(), called at the end of update()/addPoints()
  // after the occupancy-update loop) -- same pattern last_fit_j_ already
  // uses. -2.0 is "not yet computed this fit" (distinct from -1.0,
  // occupancyAnisotropy()'s own "undefined, <3 cells" sentinel).
  double cached_occ_aniso_ = -2.0;
  int cached_occ_cells_ = 0;
  void recomputeOccupancyCache();

  void updateOccupancy(const V3D& world_point);

public:
  // a = lambda1(M_cov)/lambda2(M_cov) of the OCCUPIED cell centers (one
  // sample per occupied cell, unweighted by point count) in the anchored
  // tangent frame -- T3-0e's in-plane COVERAGE anisotropy, distinct from
  // eigenValues()'s point-scatter (density-weighted) anisotropy. Returns
  // -1.0 if fewer than 3 cells are occupied (covariance undefined/
  // degenerate below that). O(1) -- returns a value cached at the last
  // fit, see recomputeOccupancyCache().
  double occupancyAnisotropy() const { return cached_occ_aniso_ < -1.5 ? -1.0 : cached_occ_aniso_; }
  int occupiedCellCount() const { return cached_occ_cells_; }
};

// T0-F-2b (2026-08-31): per-frame aggregates across every VoxelPlane fit
// this frame, for LioProc's per-frame diagnostic log -- file-scope atomics
// in voxelplane.cpp rather than threading a stats object through every
// VoxelPlane constructor call site (this diagnostic is opt-in and rare;
// changing the constructor signature would touch every caller for a
// feature most runs never use). Call voxelPlaneFrameStatsReset() once at
// the start of each frame (before any VoxelPlane::update()/refitDebiased()
// calls), then voxelPlaneFrameStatsRead() at the end of that same frame.
// Thread-safe (the accumulation calls run inside VoxelMap's OMP
// parallel-for over voxels); NOT re-entrant across overlapping frames --
// this codebase processes one frame at a time, so that never happens.
void voxelPlaneFrameStatsReset();
void voxelPlaneFrameStatsRead(int& denom_rejected_count, double& max_plane_var_trace);

}  // namespace livo_recon
