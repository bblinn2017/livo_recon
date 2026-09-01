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

  // History (36-65): see docs/livo_recon_changelog.md#include-livo_recon-lio-voxelplane.h-36
  void update(const std::vector<PointXYZCov>& points, int total_count = -1,
              const std::vector<double>* weights = nullptr,
              const RunningMoments* running = nullptr,
              const std::vector<double>* var_weights = nullptr);

  // History (71-102): see docs/livo_recon_changelog.md#include-livo_recon-lio-voxelplane.h-71
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

  // History (122-138): see docs/livo_recon_changelog.md#include-livo_recon-lio-voxelplane.h-122
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

  // History (180-185): see docs/livo_recon_changelog.md#include-livo_recon-lio-voxelplane.h-180
  uint64_t occupancy_bitmask_ = 0;
  V3D occ_anchor_normal_   = V3D::Zero();
  V3D occ_anchor_x_normal_ = V3D::Zero();
  V3D occ_anchor_y_normal_ = V3D::Zero();
  V3D occ_anchor_center_   = V3D::Zero();
  bool occ_anchored_ = false;

  // History (193-201): see docs/livo_recon_changelog.md#include-livo_recon-lio-voxelplane.h-193
  double cached_occ_aniso_ = -2.0;
  int cached_occ_cells_ = 0;
  // T8-b: per-axis occupancy second moments, in the (x_normal_, y_normal_)
  // frame the bitmask is anchored in. recomputeOccupancyCache() already
  // computes these on its way to the anisotropy ratio; caching them
  // separately is what lets the coverage term act per tangent axis
  // instead of as one scalar. Units are m^2.
  double cached_occ_var_u_ = 0.0;   // along occ_anchor_x_normal_ (lambda2 dir)
  double cached_occ_var_v_ = 0.0;   // along occ_anchor_y_normal_ (lambda1 dir)
  // Diagnostic only: the scalar trace ratio applied by applyPlaneConfidence()
  // on the most recent fit. 1.0 means the terms were off or inert.
  double last_plane_conf_factor_ = 1.0;
  void recomputeOccupancyCache();

  // T8-b: applies the redundancy and coverage terms to plane_var_ in place.
  // MUST be called after recomputeOccupancyCache(), never before -- it reads
  // the occupancy cache this fit just rebuilt.
  void applyPlaneConfidence();

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
  double occupancyVarU() const { return cached_occ_var_u_; }
  double occupancyVarV() const { return cached_occ_var_v_; }
  double planeConfFactor() const { return last_plane_conf_factor_; }
};

// History (238-248): see docs/livo_recon_changelog.md#include-livo_recon-lio-voxelplane.h-238
void voxelPlaneFrameStatsReset();
void voxelPlaneFrameStatsRead(int& denom_rejected_count, double& max_plane_var_trace);

// History (252-256): see docs/livo_recon_changelog.md#include-livo_recon-lio-voxelplane.h-252
void debugFlushConsistencyCorr();

}  // namespace livo_recon
