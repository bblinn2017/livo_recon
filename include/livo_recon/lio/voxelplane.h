#pragma once

#include "livo_recon/utils/map/voxelmap_utils.h"
#include <atomic>
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
  // corr.csv (see VoxelOpts::log_consistency_mode) -- not used for
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
  // P5.  1 = exact per-point directional weighting used this fit, 0 =
  // equal-weight fallback (only meaningful under plane_var_mode =
  // "information_directional"; always 0 otherwise).
  int infoPath() const { return info_path_; }

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
            const V3D& body_dir, const V3D& body_normal,
            double& r, double& sigma_diag_squared, double& plane_var_term,
            Eigen::Matrix<double, 1, 3>& J_nq, bool* is_candidate = nullptr,
            bool* dropped_by_ablation = nullptr) const;

  // Additive along-normal variance floor, m^2 -- see
  // VoxelOpts::weight_floor_mode.  The SAME value feeds gate()'s admission
  // threshold and computeResidual()'s res.sigma_squared; they used to differ.
  // `in_gate` distinguishes the admission threshold from the weight the
  // admitted correspondence is then given.  Every mode returns the same value
  // for both EXCEPT "legacy", which reproduces the historical asymmetry (gate
  // none, weight 1e-3) so that a byte-identity control against pre-5c93cc6
  // code exists at all.
  double weightFloor(const V3D& body_dir, const V3D& body_normal,
                     bool in_gate) const;

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
  // (see VoxelOpts::log_consistency_mode == "corr+covariates"). Mirrors
  // debugLogPlaneFitStats()'s own `j` local exactly, just persisted.
  int last_fit_j_ = 0;

  // ── the unified (information) plane model ────────────────────────────
  // See VoxelOpts::plane_var_mode.  All three are written by
  // buildInformationCovariance() and are otherwise zero, so they double as
  // the "was this mode live" engagement counters and as D-1's covariates.
  //
  // roughness_ is lambda0 AFTER the measurement noise is subtracted -- the
  // plane's own surface roughness, as a variance.  weightFloor()'s
  // "roughness" mode returns it, which is what puts it in the residual for
  // the first time.
  double roughness_ = 0.0;
  // sigma_bar^2 = mean projected measurement noise + roughness_.
  double sigma_bar2_ = 0.0;
  // The DESIGN-EFFECT corrected sample size actually used in I.  Raw N
  // overstates the information whenever many returns in this voxel came from
  // one sweep: they share that instant's pose error, so they are not N
  // independent observations.  This is the scalar (equicorrelated) special
  // case of the low-rank correction I = A^T(D + H P_xi H^T)^-1 A -- the
  // directional generalisation is deferred until D-1 says the effect is
  // worth its complexity, and info_n_raw_/distinct_frames_ are logged so
  // D-1 can size what is being left on the table.
  double info_n_eff_ = 0.0;
  double info_n_raw_ = 0.0;
  double info_rho_   = 0.0;   // intra-scan correlation used for the design effect

  // Mean per-point covariance over the points this plane was fitted from,
  // split so the shared (pose) part can be told from the independent
  // (sensor) part -- the debiased path already keeps these as Scov_/
  // Scov_sensor_; these are update()'s equivalents for the pca path, which
  // has no persistent accumulator.  Write-only unless plane_var_mode is
  // "information".
  M3D mean_cov_all_    = M3D::Zero();
  M3D mean_cov_sensor_ = M3D::Zero();

  // P5.  Per-point directional weighted moments (plane_var_mode =
  // "information_directional" only) -- the exact chart-space Fisher
  // information from update()'s own point vector, in place of applying one
  // isotropic sigma_bar^2 to every point regardless of the angle it was
  // observed from. Sw is the weighted analogue of n_raw (a design-effect
  // scale still applies on top, same as the equal-weight path); Swd/Swdd
  // are the weighted first/second moments about plane_.center, in WORLD
  // coordinates -- the chart (y_normal_/x_normal_) is applied only when
  // buildInformationCovariance() projects these into H.
  //
  // update() has the point vector in hand and can form these exactly;
  // refitDebiased() runs from unweighted persistent sums after the points
  // are gone and cannot, so info_path_ says which path actually ran:
  // 1 = exact (weighted moments used), 0 = equal-weight fallback (the
  // existing diagonal closed form, unchanged). Logged so the two paths'
  // disagreement -- what the equal-weight approximation costs -- is
  // measurable rather than silently averaged away.
  double Sw_    = 0.0;
  V3D    Swd_   = V3D::Zero();
  M3D    Swdd_  = M3D::Zero();
  int    info_path_ = 0;

  // I = (N_eff/sigma_bar^2)*diag(lambda1, lambda2, 1); plane_var_ = I^-1.
  // Sets is_plane_ = false if the model cannot be formed.
  void buildInformationCovariance(double n_raw, const M3D& mean_cov_all,
                                  const M3D& mean_cov_sensor,
                                  bool eig0_already_debiased);

public:
  // VoxelNode's distinct-frame counter, needed by the design-effect
  // correction on the pca path (the debiased path receives it through
  // addPoints()).  No-op unless plane_var_mode is "information".
  void noteFrames(int distinct_frames)
  { if (distinct_frames > 0) distinct_frames_ = distinct_frames; }

  double roughness()   const { return roughness_; }
  double sigmaBar2()   const { return sigma_bar2_; }
  double infoNEff()    const { return info_n_eff_; }
  double infoNRaw()    const { return info_n_raw_; }
  double infoRho()     const { return info_rho_; }
  int    distinctFrames() const { return distinct_frames_; }

private:


  // History (180-185): see docs/livo_recon_changelog.md#include-livo_recon-lio-voxelplane.h-180
  uint64_t occupancy_bitmask_ = 0;      // a point RETURNED from this cell
  // P6a.  A ray CROSSED this cell and kept going (positive evidence the
  // surface is absent there), reset alongside occupancy_bitmask_ whenever
  // the occupancy chart re-anchors -- see updateOccupancy(). Together the
  // two bitmasks give three states per cell instead of two:
  //   hit & !thru   surface observed here
  //  !hit &  thru   KNOWN FREE
  //  !hit & !thru   UNOBSERVED (honest ignorance -- "no bit" used to mean
  //                 this AND known-free both, which is the category error
  //                 this patch exists to fix)
  //   hit &  thru   edge / thin surface / mixed pixel
  // atomic: classifyVisibility() is called from computeResidual(), which
  // T3-0d's comment documents runs inside LioProc::buildResiduals()'s OMP
  // parallel loop -- unlike occupancy_bitmask_ (only ever mutated from the
  // separate, sequential updateOccupancy() pass), concurrent candidates on
  // this same plane can classify at the same time, so this needs fetch_or,
  // not |=.
  // mutable: written from classifyVisibility(), which computeResidual()
  // (const) calls -- same reasoning as any other const-context accumulator
  // in this codebase (e.g. the OMP-safe corr_scan accumulators elsewhere).
  mutable std::atomic<uint64_t> occ_thru_bitmask_{0};
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

  // P6a.  MEASUREMENT ONLY -- classifies one candidate's ray against this
  // plane's own occupancy chart and sets occ_thru_bitmask_ if it crosses a
  // KNOWN-FREE cell; never touches is_plane_/plane_var_/anything the state
  // estimate reads. o_world/p_world are the sensor origin and the
  // candidate point, both in world frame; sigma_r is the range noise at
  // this point's range (getBodyCov()'s own sigma_r), used as the crossing
  // margin so a return exactly at the surface doesn't misclassify from
  // sensor noise alone. Returns 0=hit (this cell has a return), 1=known
  // free (ray crossed and kept going), 2=unobserved (neither).
  int classifyVisibility(const V3D& o_world, const V3D& p_world, double sigma_r) const;

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

// Total planes fitted under plane_var_mode = "information" since process
// start.  Zero on a run configured for it means the mode never ran: an INERT
// cell, which the scorer must treat as a validity failure rather than a null.
long voxelPlaneInformationFitCount();

// History (252-256): see docs/livo_recon_changelog.md#include-livo_recon-lio-voxelplane.h-252
void debugFlushConsistencyCorr();

}  // namespace livo_recon
