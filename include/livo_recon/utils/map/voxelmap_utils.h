#pragma once

#include "livo_recon/utils/algo/math.h"
#include "livo_recon/utils/data/data_wrappers.h"
#include "livo_recon/utils/algo/hashing.h"

#include <atomic>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Eigenvalues>

namespace livo_recon
{

struct Residual
{
  // Index of the input point that produced this residual.  The joint-knot
  // estimator uses it to evaluate the measurement Jacobian at the same
  // timestamp and raw body point that was used for association.
  int source_index = -1;
  double r;
  V3D normal;

  // CQ-46: this residual's own world-frame point position, for the
  // pair-correlation diagnostic's Euclidean-separation binning. Not used by
  // any live estimator path -- purely a read of pt_world.point, copied in by
  // LioProc::buildResiduals() right after the residual is accepted.
  V3D world_point = V3D::Zero();

  // Capture time of the point this residual came from, copied straight from
  // PointXYZCov::t in LioProc::buildResiduals().  Diagnostic only
  // now that the scan-spline refinement that consumed it is gone.  0.0 on any
  // path that does not set it.
  double t = 0.0;
  // Rotation Jacobian: point.cross(R1^T*normal), the R1 (rotation)
  // Jacobian column (lio_accumulator.cpp's accumulateLioResiduals()).
  V3D point_cross_normal;
  // CQ-50: this point's body-frame position at ITS OWN capture time t
  // (above), before deskew's per-point warp to the scan-end (t1) reference
  // -- copied straight from PointXYZCov::raw_body_point in buildResiduals().
  // point_cross_normal above is built at t1 (the deskewed point + the
  // scan-end state_->rot()); this field lets a consumer build the ANALOGOUS
  // cross product at t_k instead (raw_body_point.cross(R(t_k)^T*normal)),
  // for chaining honestly with a per-point-time Jacobian rather than
  // mismatching a t1 derivative against a t_k sensitivity. Zero on any path
  // that predates this field. Not read by buildResiduals()'s own H or by
  // the iekf estimator -- see PointXYZCov::raw_body_point's doc.
  V3D raw_body_point = V3D::Zero();
  // VARIANCE (not standard deviation), despite the name -- independent
  // (sensor+pose) noise, plus this plane's own fit uncertainty
  // (plane_var_term below) folded in directly. Used as 1/sigma_squared,
  // the inverse-variance weight in the weighted normal equations
  // (solveSystem()'s HtH/Htz accumulation) and as the variance in gate()'s
  // N-sigma accept/reject test.
  double sigma_squared;

  // Identity of the VoxelPlane this residual matched against (the VoxelPlane
  // instance's own address, stable for the lifetime of that voxel). Null
  // only for a residual that hasn't been matched to a plane yet.
  const void* plane_id = nullptr;

  // Stable diagnostic identity of the owning octree node. Unlike plane_id,
  // this can be joined across compact CSV files without serializing an
  // allocator address. It never participates in gating or the solve.
  int32_t plane_node_id = -1;

  // P6a.  This candidate's ray classified against its own plane's occupancy
  // chart: 0=hit, 1=known free, 2=unobserved, -1=not classified (not a
  // gate() candidate). Measurement only -- never read by anything that
  // feeds the state estimate.
  int vis_state = -1;

  // R81 (F-485 2B): the owning voxel's point count (VoxelPlane::pointsSize()) and distinct-frame count (VoxelPlane::distinctFrames()) at the time the row was built,
  // copied in VoxelPlane::computeResidual(). Write-only: read only by the residual dump when log/lio/residual_dump/voxel_info is true. -1 on a path that does not set them.
  int voxel_point_count = -1;
  int voxel_distinct_frames = -1;
  // R83 (F-428 follow-up, Bryce 2026-10-06 on whether subdividing voxels is useful): octree layer of the owning node (0 = root), copied in VoxelNode::findPlaneResidual().
  // Write-only, read only by the residual dump under log/lio/residual_dump/voxel_info. -1 on a path that does not set it.
  int voxel_layer = -1;
  // R91C (Bryce 2026-10-09, binning the residuals by direction): the unit vector u2 = VoxelPlane::x_normal_ of the owning voxel (third eigenvector of the point scatter: the line axis of a
  // line voxel) and the axial coordinate u2 . (p - plane centre) of the point. Write-only: read only by the residual dump under log/lio/residual_dump/line_geometry. Zero on a path that does not set them.
  Eigen::Vector3d line_axis = Eigen::Vector3d::Zero();
  double line_axial = 0.0;

  // This plane's own fit-uncertainty contribution (n * plane_var * n),
  // added into sigma_squared by LioProc::buildResiduals().
  double plane_var_term = 0.0;

  // Exact ingredients behind plane_var_term. For two residuals i,j matched
  // to the same VoxelPlane, their shared plane-fit covariance is
  //   J_i * plane_covariance * J_j^T,
  // not sqrt(plane_var_term_i*plane_var_term_j). Keeping the signed
  // three-parameter Jacobian and covariance permits the correlation-aware
  // information solve to reproduce the actual VoxelPlane model.
  V3D plane_jacobian = V3D::Zero();
  M3D plane_covariance = M3D::Zero();

  // CQ-18 item (2): sigma_squared above is floor_term+sigma_diag_squared
  // already summed -- these two components are split out separately (and
  // s_prior_pose, the H P- H^T term never folded into sigma_squared/the
  // real accept-reject weighting, see VoxelPlane::computeResidual()'s own
  // comment on this) so frame_stats.txt can report the S = floor_term +
  // sigma_diag_squared + plane_var_term + s_prior_pose decomposition's four
  // shares per frame. -1.0 (never negative for a real residual) marks "not
  // set" for a residual predating this field, e.g. constructed elsewhere.
  double floor_term         = -1.0;
  double sigma_diag_squared = -1.0;
  double s_prior_pose       = -1.0;
  // R80 (F-449a / gate puzzle): the variance the admission test used, floor_term + sigma_gate_diag_squared + plane_var_term (the gate's diag part includes the pose covariance when
  // the estimator passes lio/gate_state_uncertainty, which is true by default); sigma_squared above is what the WEIGHT uses (never the pose covariance). -1 = not set. Write-only.
  double sigma_gate_squared = -1.0;

  // Which tier of VoxelMap::findPlaneResidual()'s fallback matched this
  // point: 0 = primary voxel, 1 = single directional neighbor (the only
  // fallback FAST-LIVO2 also has), 2 = the full neighborhood_size box
  // search (livo_recon-only, picks the best-scoring candidate among up to
  // ~(2*neighborhood_size+1)^3 voxels rather than giving up). Debug-only,
  // for attributing residual-quality issues to a specific tier.
  int match_tier = 0;

  // R74 (F-430, F-431): which kind of row this is (see lio/line_residual.h LineRowKind): 0 = the ordinary single point-to-plane row (the
  // default; everything above describes it), 1/2 = the u0/u1 rows of a line voxel, 3/4/5 = the u0/u1/u2 rows of the smooth point-to-
  // distribution form. For a point that gets more than one row, VoxelPlane::computeResidual() fills this residual (the u0 row) and puts
  // the other rows in extra_* (n_extra of them); LioProcBase::buildResiduals() turns each into its own Residual right after this one.
  // Default 0 / n_extra 0: nothing changes for a run that does not select a line mode.
  int row_kind = 0;
  int n_extra = 0;
  V3D extra_normal[2] = {V3D::Zero(), V3D::Zero()};
  double extra_r[2] = {0.0, 0.0};
  double extra_sigma[2] = {0.0, 0.0};      // floor + point noise projected on the row direction (the fit part is extra_fit_var)
  double extra_fit_var[2] = {0.0, 0.0};
  int extra_kind[2] = {0, 0};              // LineRowKind of each extra row
  // R77 (F-453): the tilt part of each row's fit variance (already included in plane_var_term / extra_fit_var); 0 unless line_variance = plane_equivalent.
  double tilt_var_term = 0.0;
  double extra_tilt_var[2] = {0.0, 0.0};
};

// A world-frame point with its covariance kept as two separate
// contributions rather than pre-summed:
//   - sensor_cov: the point's own independent per-point sensor noise,
//     rotated to world frame (StateGroup::toWorld()).
//   - pose_cov: the current state's pose (rotation+position) uncertainty's
//     contribution to this point's world-frame position (StateGroup::
//     poseCovAt()) -- NOT independent per point (every point in the frame
//     shares the same pose uncertainty), and already accounted for once,
//     correctly, via the EKF's own prior-covariance term in the update --
//     so it belongs in plane-match *gating* (a wider gate is reasonable
//     when the prior pose is less certain) but must NOT also inflate the
//     residual *weight*, which would double-count it.
struct GateAudit
{
  bool reached_statistical_gate = false;
  bool rejected_by_statistical_gate = false;
};

struct WorldPointCov
{
  V3D point;
  M3D sensor_cov;
  M3D pose_cov;

  // Estimator-local, gate-only switch (lio/gate_state_uncertainty).  When true, pose_cov contributes
  // to correspondence admission.  It must never be
  // copied into Residual::sigma_squared: the state prior already appears in
  // the information solve and adding it to the solve weight would count it
  // twice.
  bool include_pose_cov_in_gate = false;

  // Optional per-input-point audit sink. VoxelPlane may be queried against
  // several candidate planes; flags are therefore OR-accumulated. The
  // caller decides whether the point was ultimately admitted by any plane.
  GateAudit* gate_audit = nullptr;

  // History (66-87): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-66
  V3D body_point = V3D::Zero();
  M3D rot_transpose = M3D::Identity();
  Eigen::Matrix<double, 6, 6> prior_cov_rp = Eigen::Matrix<double, 6, 6>::Zero();

  M3D total() const { return sensor_cov + pose_cov; }
};

inline double pointPlaneGateMeasurementVariance(
    const V3D& normal, const M3D& sensor_cov, const M3D& pose_cov,
    bool include_pose_cov_in_gate)
{
  double variance = normal.dot(sensor_cov * normal);
  if (include_pose_cov_in_gate)
    variance += normal.dot(pose_cov * normal);
  return variance;
}

struct PlaneInfo
{
  V3D normal = V3D::Zero();
  V3D center = V3D::Zero();
  double d   = 0.0;  // -normal.dot(center)
};

struct PlaneVizInfo
{
  V3D   center;
  V3D   normal;
  float radius;
  bool  is_converged;
};

struct PlaneUpdate
{
  int32_t      node_id;
  bool         deleted;  // true = remove from cache; false = add/update
  PlaneVizInfo info;     // valid when !deleted
};

// One read-only row of the live octree/plane state. Used only by the
// post-calibration snapshot; keeping it separate from PlaneVizInfo avoids
// turning the visualization cache into estimator state.
struct VoxelPlaneSnapshot
{
  int32_t node_id = -1;
  VoxelKey root_key{};
  int layer = 0;
  V3D cell_center = V3D::Zero();
  V3D cell_min = V3D::Zero();
  V3D cell_max = V3D::Zero();
  int status = 0;
  bool retired = false;
  bool initialized = false;
  bool is_plane = false;
  bool full = false;
  V3D center = V3D::Zero();
  V3D normal = V3D::Zero();
  V3D tangent_x = V3D::Zero();
  V3D tangent_y = V3D::Zero();
  double d = 0.0;
  double radius = 0.0;
  V3D eigenvalues = V3D::Zero();
  M3D scatter = M3D::Zero();
  M3D plane_covariance = M3D::Zero();
  int point_count = 0;
  int distinct_frames = 0;
  int last_fit_j = 0;
  double roughness = 0.0;
  double sigma_bar2 = 0.0;
  double info_n_raw = 0.0;
  double info_n_eff = 0.0;
  double info_rho = 0.0;
  int info_path = 0;
  int occupied_cells = 0;
  double occupancy_anisotropy = -1.0;
  double occupancy_var_u = 0.0;
  double occupancy_var_v = 0.0;
  double plane_conf_factor = 1.0;
};

// R73 (F-414): whole-map statistics and order-independent digests, computed on demand (never inside the residual loop).
//   digest_exact:   sum over every octree node of a 64-bit mix of (root key, layer, status, retired, is_plane, point_count,
//                   distinct_frames, plane center, normal, d) with the doubles hashed by bit pattern. Order independent (sum), so the
//                   hash map's iteration order does not matter. Equal digests mean bit-identical plane parameters.
//   digest_rounded: the same with center, normal, d rounded to 1e-3 and node_id left out: equal digests mean agreement to 1 mm.
//   planes_changed: nodes whose rounded hash differs from the previous call (or are new); voxels_created: change in n_root_voxels.
struct VoxelMapScanStats
{
  long n_root_voxels = 0, n_nodes = 0, n_plane_nodes = 0, n_retired = 0;
  long voxels_created = 0, planes_changed = 0;
  unsigned long long digest_exact = 0, digest_rounded = 0;
};

// Running sufficient statistics for VoxelNode's DEFAULT (unweighted,
// equal-weight-per-point) points_ accumulation path, maintained
// incrementally in O(1) per inserted point (see VoxelNode::insertPoints())
// so VoxelPlane::update() can derive mean/covariance in O(1) at fit time
// instead of re-scanning every accumulated point on every refit -- exact,
// not an approximation. Only valid when every point contributing to a fit
// carries equal weight, which is true for VoxelNode's raw points_ path but
// NOT the binned map's weighted bin path (voxel_map/backend = binned) --
// VoxelPlane::update() only consults this when explicitly passed a
// non-null pointer, and callers must not pass one alongside `weights`.
struct RunningMoments
{
  double sum_w  = 0.0;          // == point count so far (equal-weight path)
  V3D    sum_p  = V3D::Zero();  // Sigma (p_i - ref)
  M3D    sum_pp = M3D::Zero();  // Sigma (p_i - ref)(p_i - ref)^T
  V3D    ref    = V3D::Zero();  // reference point subtracted before accumulating (conditioning only, exact either way)

  void reset(const V3D& new_ref)
  {
    sum_w = 0.0;
    sum_p.setZero();
    sum_pp.setZero();
    ref = new_ref;
  }

  void add(const V3D& p)
  {
    const V3D d = p - ref;
    sum_w += 1.0;
    sum_p += d;
    sum_pp.noalias() += d * d.transpose();
  }
};

// Resolved-once-at-load enum mirror of the hottest per-point mode string
// (VoxelOpts::weight_floor_mode below) -- see VoxelMap::loadParameters()'s
// doc comment where this is resolved, and voxelplane.cpp's weightFloor(),
// which switches on this instead of re-comparing the std::string form on
// every candidate point, every IEKF iteration. The string field itself is
// unchanged and remains the source of truth (config loading, logging,
// validation) -- this is a derived cache, not a replacement.
// R97A: PlaneGateMode is removed; "disc" is the only plane_gate_mode now.
enum class WeightFloorMode { SensorRange, Incidence, Constant, None, Legacy, Roughness };

struct VoxelOpts
{
  int    max_layer                 = 2;
  double voxel_size                = 0.5;
  double plane_threshold           = 0.01;
  int    min_init_points           = 5;
  int    min_update_points         = 5;
  int    max_points                = 50;
  double sigma_num_squared         = 9.0;
  double max_radius                = 3.0;

  // R93A (Bryce 2026-10-10): voxel_map/residual/pose_cov_in_sigma is removed (it was false in every job). The residual weight sigma_squared never contains the pose covariance; the
  // admission gate may (lio/gate_state_uncertainty, an estimator option, per-point WorldPointCov::include_pose_cov_in_gate).

  // R65 (F-169). Write-only: one row per map insertion in point_cov_stats.csv (frame, n_points, mean/min/max of
  // trace(pos_cov) over the inserted points). Off by default; no effect on the estimator.
  bool point_cov_stats_en          = false;

  // History (177-181): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-177
  bool log_debug_en                = false;

  // History (184-195): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-184
  bool log_variance_shares_en      = false;

  // History (198-213): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-198

  // History (215-229): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-215

  // Tiered fallback when the primary voxel's own findPlaneResidual() misses
  // (see VoxelMap::findPlaneResidual()): (1) primary voxel -- (2) the single
  // FAST-LIVO2-style directional neighbor the point geometrically leans
  // toward (one extra hash lookup, catches the common case cheaply) --
  // (3) only if that also misses, an exhaustive box search over
  // neighborhood_size voxels in each direction, scored and best-of-N. Tier 3
  // alone (skipping tier 2) was the original livo_recon behavior; tier 2
  // alone regressed ATE badly on eee_01 under dynamic motion (residual
  // count collapse -- a single wrong directional guess had no second
  // chance). Both tiers together give tier 2's low steady-state cost with
  // tier 3's robustness as a real safety net.
  int    neighborhood_size         = 1;

  // "normal" (default): once a node's accumulated points_size_ reaches
  // max_points, lock permanently into CONVERGED (if now_plane) or DISABLED
  // (if not) -- matches FAST-LIVO2's own max_points_num_/update_enable_=
  // false mechanism exactly (voxel_map.cpp there uses the same default,
  // 50).
  // "always_update": a node that currently has a valid plane (now_plane)
  // never locks into CONVERGED -- points_ keeps growing and
  // VoxelPlane::update() keeps refitting from scratch every
  // min_update_points new points, indefinitely. The non-planar/DISABLED
  // dead-end path (gives up at max_layer once truly full) is unaffected --
  // this only removes the "freeze a good plane forever" behavior, not the
  // "give up on a hopeless voxel" one. No upper bound on points_'s growth
  // in this mode -- fine for the short investigation runs this was added
  // for, not intended for long/production runs without a rethink of the
  // O(N) full-refit-from-scratch cost in VoxelPlane::update().
  // R97A: "frame_gated" and min_frames_to_converge/min_frames_to_init are removed (the latter was a no-op at its default, 1); an
  // unrecognized convergence_mode value is now REFUSED at start (previously fell through to "normal" silently).
  std::string convergence_mode      = "normal";

  // false (default) -- VoxelPlane::update()'s acceptance test trusts the
  // raw empirical eigen_values_(0) (perpendicular/out-of-plane variance)
  // unconditionally, however small a sample produced it. true -- floor it
  // at this voxel's own points' average per-point measurement noise
  // (PointXYZCov::cov, already carrying sensor+deskew-integration
  // uncertainty from ImuProc::deskewPoints()) projected onto the candidate
  // normal, before the plane_threshold comparison. Targets the failure
  // mode found investigating point_filter_num=1's ATE regression: a small,
  // single-viewpoint sample (all points from one lidar sweep/frame) can
  // look implausibly flat -- empirical eig0 orders of magnitude below what
  // that sensor's own noise model says is achievable -- not because the
  // surface is that flat, but because points from one instant/vantage
  // point share correlated noise sources undecorrelated by genuine
  // multi-frame view diversity (confirmed empirically: median eig0 for
  // single/few-frame inits was ~200x smaller than for 10+-frame inits on
  // NTU_VIRAL eee_01, despite similar in-plane point spread). If the
  // floored eig0 would reach or exceed eigen_values_(1), the fit is
  // rejected outright instead (the noise floor alone already explains the
  // observed "flatness" -- this candidate can't be distinguished from
  // non-planar). Complements, not replaces, min_frames_to_init: doesn't
  // fix the underlying single-viewpoint estimation problem, just stops the
  // acceptance test and plane_var_ from trusting an eig0 the sensor's own
  // model says is implausible.
  bool   sensor_noise_floor_eig0    = false;

  // R93A (Bryce 2026-10-10): plane_fit_pose_cov_mode is removed. Every fit uses the sensor covariance only (the former sensor_only); pos_cov stays on a point for plane_var_mode = information and diagnostics.

  // R64. Plane-fit diagnostics (write-only, default off). plane_fit_reasons_en: per-frame counts of fit outcomes by
  // reason -> plane_fit_reasons.csv. plane_fit_trace_frames > 0: one row per fit attempt for
  // frames 0..plane_fit_trace_frames (frame 0 = bootstrap window) -> plane_fit_trace.csv, at most
  // plane_fit_trace_max_rows rows.
  bool plane_fit_reasons_en = false;
  int  plane_fit_trace_frames = 0;
  int  plane_fit_trace_max_rows = 200000;
  // R83B (A1/B4): same frame window as plane_fit_trace_frames, same row cap, but one row per fit attempt with the
  // voxel's layer and the sorted eigenvalues of the raw (uncorrected) scatter -> plane_fit_eigs.csv. Default off.
  bool plane_fit_eigs_en = false;

  // ── the plane covariance: which model produces plane_var_ ────────────
  //
  // "eigengap" is the incumbent: differentiate the fitted eigenVECTOR with
  // respect to each point, which introduces 1/(lambda0 - lambda_m) and is
  // why the gap floor and the
  // kPlaneVarCeiling reject exist.  It blows up when the plane is
  // nearly isotropic -- i.e. when it is barely a plane -- rather than when
  // the plane is badly SAMPLED, which is the quantity a residual weight
  // actually wants.
  //
  // "information" is the unified model.  Work directly in the local chart
  // (theta1, theta2, d) that the lever arm J_nq already uses and accumulate
  // the directional Fisher information
  //
  //     I = sum_i a_i a_i^T / sigma_i^2 ,   a_i = [d_i.y_normal, d_i.x_normal, 1]
  //     sigma_i^2 = n^T Sigma_i n  +  lambda0_roughness
  //
  // then plane_var_ = I^-1.  Because the points are centred on plane_.center
  // (so sum d_i = 0) and (x_normal_, y_normal_) are eigenvectors of the
  // second moment, every off-diagonal term vanishes EXACTLY and this
  // reduces to a closed form needing no new accumulator at all:
  //
  //     I = (N_eff / sigma_bar^2) * diag(lambda1, lambda2, 1)
  //
  // Three things the register has been treating as separate corrections are
  // three projections of that one matrix: surface ROUGHNESS is the scalar in
  // the denominator, sampling ANISOTROPY is I's conditioning, and
  // directional COVERAGE is I^-1's eigendirections read against the query's
  // own lever arm -- a query off the sampled band lands in the weak
  // eigendirection and its variance grows in proportion to how far outside
  // the support it sits, with no tuning constant anywhere.
  //
  // TRANSITIONAL.  Both modes exist only so the incumbent is a
  // representable control (standing rule 10b, the lesson of weight_floor's
  // missing "legacy").  When INT-1/D-1 decide, the loser is deleted along
  // with sensor_noise_floor_eig0.
  std::string plane_var_mode = "eigengap";

  // R90: the pca-only bin path (use_bins, bin_size_fraction, bin_weight_mode_fit/_var) is removed; binning is the binned map type (voxel_map/backend = binned) for both plane fits. The old keys are refused at start-up.

  // History (412-425): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-412
  // History (428-445): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-428
  //
  // The consistency log is a TIER LADDER, not two independent switches, and
  // writing it as two booleans was a live defect.  corr_scan.csv (per-scan
  // exact aggregates: mean NIS, accept fraction, dropped-by-ablation count)
  // is cheap -- one accumulator update per candidate, one line per scan --
  // and corr.csv (a full per-correspondence row) is not: it is ~1 GB/job and
  // halves sweep throughput.  Both sat inside a single `if
  // (log_consistency_corr_en)`, so the cheap per-scan aggregate was
  // UNREACHABLE without paying for the expensive per-correspondence rows,
  // and every sweep that turned Tier B off to make its throughput budget
  // silently lost the level statistics too.  A boolean pair can also express
  // nonsense (covariates on, corr off = inert), which the enum cannot.
  //   off             -- nothing
  //   scan            -- corr_scan.csv only (Tier A: cheap, safe in a sweep)
  //   corr            -- Tier A + per-correspondence rows, no covariates
  //   corr+covariates -- Tier A + rows carrying the plane/geometry covariates
  std::string log_consistency_mode = "off";
  static constexpr const char* LOG_CONSISTENCY_MODES[] = {
      "off", "scan", "corr", "corr+covariates" };
  // Tier A: the per-scan accumulator. Live in every mode but "off".
  bool logCorrScan() const { return log_consistency_mode != "off"; }
  // Tier B: individual corr.csv rows, subject to the stride below.
  bool logCorrRows() const
  { return log_consistency_mode == "corr" || log_consistency_mode == "corr+covariates"; }
  // The covariate columns, which cost the plane-level geometry queries.
  bool logCorrCovariates() const
  { return log_consistency_mode == "corr+covariates"; }

  // History (448-456): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-448
  // Tier B only: striding a log that is not being written is meaningless, and
  // the per-scan aggregates above are deliberately NOT strided.
  int log_consistency_corr_stride = 1;

  // R97A: voxel_map/plane/plane_gate_mode is removed; "disc" (VoxelPlane::gate()'s isotropic radius gate) is the only mode now.

  // ── The residual-weight floor ───────────────────────────────────────────
  // MODE: "sensor_range" (default) | "incidence" | "constant" | "none".
  //
  // WHAT WAS THERE BEFORE, AND WHY IT WAS WRONG.  The along-normal variance
  // is computed from a real range/bearing model (getBodyCov(): sigma_r on the
  // ray, R*sigma_a across it), and then a flat literal 1e-3 was added on top
  // of it in TWO places in this file.  1e-3 is (3.2 cm)^2, it is in no
  // datasheet, and whether it dominates is pure geometry:
  //
  //   near-normal incidence   n^T S n ~ sigma_r^2            = 2.5e-3
  //   grazing at range R      n^T S n ~ R^2 sigma_a^2 sin^2  = 3e-4 at 5 m,
  //                                                            1.1e-2 at 30 m
  //
  // so the constant dominates exactly the SHORT-RANGE, OBLIQUE returns.  That
  // predicts the measured split with nothing else assumed -- 2-3% of
  // correspondences floored on eee_01 (NTU-VIRAL, outdoor, long range) against
  // 55-70% on exp05 (HILTI, indoor construction) -- which means that on HILTI
  // the majority of the update was weighted by a constant, i.e. very nearly
  // uniformly, and every plane-confidence mechanism that acts through the
  // weight had almost nothing to act on.
  //
  // THE PHYSICS THE MODEL WAS MISSING.  The projection sigma_r^2 cos^2(theta)
  // sends the range error to ZERO at grazing incidence, which is backwards: a
  // tilted surface stretches the return pulse in time (dt = 2 rho tan(theta)/c
  // over a footprint of radius rho), so range noise DEGRADES with incidence
  // rather than projecting away.  Model that as sigma_r_eff^2(theta) =
  // sigma_r^2 (cos^2 + k sin^2) along the normal.  At k = 1 it collapses to
  // sigma_r^2 exactly -- a floor, but the SENSOR's own range variance rather
  // than a literal, and per-dataset because imu/sensor/range_err already is.
  //
  //   "sensor_range"  floor at sigma_r^2 (= weight_sigma_r2).  The default:
  //                   the constant, tuned by the physics instead of guessed.
  //   "incidence"     no floor; add sigma_r^2 (cos^2 + k sin^2) - the k = 1
  //                   case is identical to sensor_range, so k IS the ablation
  //   "constant"      the historical flat weight_floor_constant.  Kept as the
  //                   control arm for W-1, not because it is defensible
  //   "none"          no floor at all -- the other control, and the only way
  //                   to see what the floor was doing
  //
  // APPLIED IN BOTH THE GATE AND THE WEIGHT, which closes the bug-ledger row
  // "gate and weight use different floors": gate() tested
  // sigma_diag_squared + plane_var_term with no floor while the weight was
  // 1e-3 + the same, so the admission threshold and the variance the admitted
  // correspondence was then given disagreed.
  //   "legacy"        THE REPRODUCIBILITY CONTROL, and its absence was a
  //                   defect.  The historical code did not use one floor: the
  //                   gate had NONE and the weight had 1e-3.  The first four
  //                   modes above all tie the two together, so none of them
  //                   reproduces the pre-restructure behaviour and V-2's
  //                   byte-identity check had nothing to run against --
  //                   "constant" isolates the gate change and "none" isolates
  //                   the weight change, and neither is the old pair.  This
  //                   mode is that pair, provided ONLY so the control exists.
  //                   It is not defensible as a setting; do not ship it.
  //   "roughness"   The unified model's answer, and the reason this whole
  //                   axis exists.  The floor IS the plane's own measured
  //                   surface roughness lambda0_deb -- the out-of-plane
  //                   second moment after the measurement noise has been
  //                   subtracted off.  The ledger's standing complaint is
  //                   that "lambda0 is a binary admission test only: surface
  //                   roughness never enters the residual variance"; here it
  //                   is the variance component it always was, measured per
  //                   plane instead of chosen once as a literal.  The noise
  //                   subtraction is done at query time.
  //                   Requires plane_var_mode = "information", which is what
  //                   computes it; refuses otherwise rather than silently
  //                   floor-ing at zero.
  //                   FALSIFIABLE, and cheaply: if this is right, the
  //                   per-plane optimal constant floor should CORRELATE with
  //                   that plane's lambda0_deb -- W-1's retained data can be
  //                   re-read to check it before any of this is trusted.
  std::string weight_floor_mode = "sensor_range";
  static constexpr const char* WEIGHT_FLOOR_MODES[] = {
      "sensor_range", "incidence", "constant", "none", "legacy", "roughness" };
  // Resolved once at load, in VoxelMap::loadParameters() -- see this
  // header's own enum doc comment above VoxelOpts.
  WeightFloorMode weight_floor_mode_enum = WeightFloorMode::SensorRange;
  // (m^2) -- the historical literal, live only under "constant".
  double weight_floor_constant = 1e-3;
  // (m^2) -- sigma_r^2, mirrored from imu/sensor/range_err by LioProc so the
  // two never drift apart.  0.05 m -> 2.5e-3.
  double weight_sigma_r2 = 2.5e-3;
  // Incidence exponent; live only under "incidence".  1.0 reproduces
  // "sensor_range" exactly, > 1 makes grazing returns progressively less
  // trusted, which is the physical claim under test.
  double weight_incidence_k = 1.0;

  // History (479-507, removal): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-487-removed

  // R93A: plane_var_denom_floor_en is removed (it was false in every job).

  // R97A: voxel_map/map/shuffle_insertion_seed is removed (it was 0 in every job).

  // History (570-575): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-570
  bool log_frame_stats_en = false;

  // One-shot diagnostic dump after frame 0 has seeded the map. This is the
  // exact map queried by the first post-calibration LIO frame.
  bool snapshot_post_calibration = false;

  // R79 (F-464, F-441, F-440): frames N >= 2 whose queried map is written as voxelmap_snapshot_f<N>.csv (see map/snapshot_frames.h). Empty = off (default,
  // bit-identical). Needs snapshot_post_calibration; parsed and validated in VoxelMap::loadParameters.
  std::vector<int> snapshot_frames;

  // R97A: voxel_map/map/freeze_after_calibration is removed (it was false in every job).

  // R76 (F-445): residual form for voxels that may be lines. R97A: always on (voxel_map/plane/line_residual/enabled is removed; the
  // no-line path is gone -- every voxel is a candidate line or a plane, decided at fit time, see lio/voxel_class.h).
  // line_form: hard (default; a line voxel gets two rows) | smooth (every matched point gets three rows).
  // line_rule (read only when form is hard): absolute (default; second eigenvalue < plane_threshold, the same constant that tests the first) |
  // relative (e1 not distinguishable from e0, rel_sigmas) | relative_noise (relative with e0 floored at the sensor noise variance) |
  // shape (e1 <= line_shape_ratio * e2). See lio/line_residual.h. line_residual_enum is resolved once at load.
  std::string line_form = "hard";
  std::string line_rule = "absolute";
  double line_rel_sigmas = 3.0;
  double line_shape_ratio = 0.1;   // used only with line_rule = shape
  int line_residual_enum = 0;   // LineMode as an int (0 = Off), set by VoxelMap::loadParameters()

  // R77 (F-453): how the map-side variance of a line / smooth row is formed (read only when line_enabled).
  //   fit_only (default, R76 behaviour, bit-identical): hard rows e_i/N; smooth rows max(e_i, sensor noise variance)(1+1/N).
  //   plane_equivalent (R80, replaces the R77 tilt-only form): the same fit_only variance PLUS what the point-to-plane row's J plane_var_ J^T carries: the variance of the row's own direction
  //   u_i, propagated from the points' sensor / pose covariance as a quadratic form in l = (u_a . d, u_b . d, 1) -- the tilt of u_i toward the tilt axes (first-order eigenvector perturbation,
  //   gap floored at the sensor noise variance), the offset variance of the centroid along u_i (n^T (sum w^2 C) n for a plane), and their cross term. Hard rows i = 0, 1 tilt toward u2 only
  //   (the line axis); smooth rows tilt toward the other two axes. R77 plane_equivalent results (tilt only, no offset or cross term) are not reproducible with this key.
  // Read only with plane_var_mode = eigengap (the information model has no per-point fit to propagate): refused otherwise.
  std::string line_variance = "fit_only";
  bool line_var_plane_equivalent = false;   // resolved once at load from line_variance
  // R80 (F-453, Bryce 2026-10-05): rows = per_normal (default; a hard line voxel gives the two rows u0 and u1, bit for bit what R74 to R79 built) | distance (a hard line voxel gives ONE row:
  // the shortest distance from the point to the line (the voxel's axis u2), along the direction q from the line to the point; q = (d - (u2.d) u2) / |.|). Read only for form hard.
  // Fit variance of the distance row: fit_only = sum_i (q.u_i)^2 e_i / N; plane_equivalent adds Var(q-weighted row perturbation) = the two plane-normal point covariances projected on q.
  std::string line_rows = "per_normal";
  bool line_rows_distance = false;   // resolved once at load from line_rows

  // R82 (D4, F-500, Bryce 2026-10-06), unconditional since R93A (the key voxel_map/plane/line_residual/classify_at_fit is removed). Line model at FIT time.
  //   The fit (VoxelPlane::update()) decides reject / plane / line ONCE, right after the eigenvalues are final, with the one function classifyVoxel (include/livo_recon/lio/voxel_class.h), and store the class (VoxelPlane::isLine()):
  //     (1) with the hard line form a voxel with e1 < 1e-8 <= e2 is a perfect line and is kept; e2 < 1e-8 rejects (with any other line setting e1 < 1e-8 also rejects, as before).
  //     (2) after the sensor floor, e0 >= e1 is not a reject: e0 is set equal to e1 (the order the pca fit always had).
  //     (3) the gaps are floored by the one gap floor (below).

  // R77 (F-433), unconditional since R93A (the key voxel_map/plane/plane_gap_floor_en and the eigengap-denominator guard are removed): the eigenvalue gaps |e0-e1| and |e0-e2| of the eigengap plane variance are floored at the sensor noise
  // variance (weight_sigma_r2, = imu/sensor/range_err^2), keeping the plane and the sign of the eigenvalue order. Needs range_err > 0 with plane_var_mode = eigengap (refused otherwise). Not applied under the information models.

  // R97A: voxel_map/plane/plane_var_min and plane_var_scan_s2 are removed (both were 0.0 in every job).
  // R82 (D2, F-502): write-only diagnostic of the plane variance term; see lio/plane_var_diag.h. Default false.
  bool plane_var_diag_en = false;

  // R88: the binned map accumulator (src/map/binned_voxelmap.cpp; map/binned/binned_bins.h). R97A: it is the only map (voxel_map/backend
  // is removed); binned_enabled is gone with it -- VoxelNode::insertPoints() always keeps returns in bins and fits from the occupied bins.
  //   binned_bin_fraction   : bin edge = this x the node's own edge (0 < f <= 1).
  //   binned_cov_mode       : 0 average (C / k), 1 shrunk (C / k^2), 2 fused (R97A R3, precision-
  //                           weighted), 3 fused_pose (R97A R3, fused + a fixed scene term) --
  //                           matches binned::CovMode's own enum values exactly.
  //   (R95: binned_weight_mode was removed; the weight of a bin is always its return count k.)
  double binned_bin_fraction = 0.2;
  int    binned_cov_mode = 0;
  // R97A R3: voxel_map/binned/fused_pose_sigma_m -- only meaningful (and validated) under binned_cov_mode == 3 (fused_pose).
  double binned_fused_pose_sigma_m = 0.0;
  // R92: min_init / min_update in a binned map always count the UNIQUE bins that received a return since the last fit (a bin hit in several scans counts once; before the first fit = occupied bins). The R91 member binned_update_count_mode was removed.
};
using VoxelOptsPtr = std::shared_ptr<VoxelOpts>;

enum class VoxelStatus { OPEN, PARENT, CONVERGED, DISABLED };

// Per-frame LIO-side diagnostics carried into frame_stats.txt.  Kept here
// rather than in lio_processing.h because VoxelMap owns the writer and must
// not depend on LioProc.  All four are cheap scalars computed from state the
// update already forms.
struct LioFrameDiag
{
  int    n_residuals  = 0;    // point-to-plane residuals the ESIKF accumulated
  double h_pp_min_eig = -1.0; // min eigenvalue of HtH's position block
  double h_rr_min_eig = -1.0; // min eigenvalue of HtH's rotation block
  // CQ-81 item B: max eigenvalue of HtH's position block, alongside the
  // pre-existing min above -- H_pp = sum_i w_i * n_i * n_i^T over this
  // scan's accepted residuals (dr/dpos == n_i^T for point-to-plane, so
  // H_pp already IS the weighted plane-normal-diversity matrix CQ-81 asks
  // for), so lambda_min/lambda_max together are the diversity spread
  // without any new accumulation pass -- both read off the same
  // SelfAdjointEigenSolver the min already used.
  double h_pp_max_eig = -1.0;
  double sum_weight   = 0.0;  // total residual weight, i.e. how much information
  // CQ-36 item 1: trace(HtH's position block), read directly off ekf_.HtH --
  // unlike sum_weight above (summed from residuals_, structurally blind to
  // any per-plane covariance correction applies, see residual_redundancy.h),
  // this DOES move under that correction. Equal to sum_weight at
  // lio/residual_redundancy/mode=="off"; divergent otherwise. h_rr_trace below is
  // the rotation-block counterpart (already existed, kept unrenamed).
  double htth_pos_trace = -1.0;
  // P1.  All from objects the update already holds; one 6x6 solve per frame.
  double h_rr_trace   = 0.0;  // denominator for DIV_rot; trace(H_pp)==sum_weight already
  double htz_rot_norm = 0.0;  // |Htz(0..2)| -- rotation drive
  double htz_pos_norm = 0.0;  // |Htz(3..5)| -- position drive
  double ask          = -1.0; // Htz^T HtH^-1 Htz : what the measurements alone would move
  double got          = -1.0; // dx^T HtH dx     : what the filter actually applied
  double refusal       = std::numeric_limits<double>::quiet_NaN();  // 1 - got/ask
  int    iters        = 0;
  double dx_rot_deg   = 0.0;
  double dx_pos_mm    = 0.0;
  double trP_pos_pre  = -1.0; // BEFORE the update -- turns a level into a delta
  // P-F (BASE, 2026-09-04): L1 (DECIMATION) had no column at all -- n_residuals
  // above is L2 (ASSOCIATION). n_points_after_pfn = mg.lidar_points.size() (post
  // cbk/lidar/point_filter_num, pre imu/ds); n_points_after_ds = mg.points.size()
  // (post imu/ds, the set buildResiduals() actually consumes). Both already
  // computed by the time LioProc builds this struct -- no new counting logic,
  // just capturing sizes that existed. n_points_raw (pre-point_filter_num) is
  // NOT captured here -- would need a queue-level accumulator in DataQueues
  // since mg.lidar_points can bundle multiple raw scans per frame; left for a
  // later patch.
  int n_points_after_pfn = -1;
  int n_points_after_ds  = -1;
  // CQ-85 item 2: mg.n_imu_samples, read straight through -- settles the
  // standing 39-vs-20 samples-per-scan discrepancy directly rather than
  // leaving it uncounted.
  int n_imu_samples = -1;
  // Phase-6A diagnostic campaign (2026-09-22, user request): straight
  // through from LioProcBase::n_miss_coverage_/n_miss_mismatch_/
  // n_tier0_miss_coverage_/n_tier0_miss_mismatch_ (already computed by
  // buildResiduals(), never logged before) -- lets a correspondence-
  // starvation failure (few/degenerate residuals) be distinguished from a
  // conditioning failure on an otherwise-healthy correspondence set,
  // shared across the IEKF and coupled estimators since buildResiduals() is
  // common machinery. -1 = not populated (caller didn't set it).
  int n_miss_coverage = -1;
  int n_miss_mismatch = -1;
  int n_tier0_miss_coverage = -1;
  int n_tier0_miss_mismatch = -1;

  // CQ-18 item (2).  S = floor_term + sigma_diag_squared + plane_var_term +
  // s_prior_pose, summed across this frame's residuals_; the four *_share
  // fields are each component's fraction of sum_S (sum to 1.0 to printing
  // precision by construction). nis/nis_est are this frame's MEAN per-
  // residual nu^2/S and nu^2/(S-s_prior_pose) (see voxelplane.cpp's
  // per-candidate nis/nis_est, which this mirrors at frame granularity
  // instead of corr_scan.csv's per-scan aggregate). -1.0 = no residuals /
  // sum_S <= 0 this frame.
  double sum_S           = -1.0;
  double floor_share     = -1.0;
  double sdiag_share     = -1.0;
  double pvar_share      = -1.0;
  double prior_pose_share = -1.0;
  double nis             = -1.0;
  double nis_est         = -1.0;

  // CQ-19(a).  P (prior_cov_, pre-update) and state_->cov() (post-update)
  // decomposed beyond the position block's bare trace (trP_pos_pre above):
  // the position block's own eigenvalues (an anisotropic prior is a
  // different object from an isotropic one of the same trace), the
  // rotation block's trace/min-eigenvalue, and the Frobenius norm of the
  // pos-vel and pos-bias cross-blocks (where IMU bias error reaches
  // position). -1.0 = unavailable (e.g. bias not estimated, so no pos-bias
  // block exists).
  double p_pos_eig_min_pre  = -1.0, p_pos_eig_mid_pre  = -1.0, p_pos_eig_max_pre  = -1.0;
  double p_pos_eig_min_post = -1.0, p_pos_eig_mid_post = -1.0, p_pos_eig_max_post = -1.0;
  double p_rot_trace_pre    = -1.0, p_rot_eig_min_pre  = -1.0;
  double p_pos_vel_fro_pre  = -1.0, p_pos_bias_fro_pre = -1.0;
  // CQ-74 item 1: the rotation block's POST-update counterpart to
  // p_rot_trace_pre/p_rot_eig_min_pre above (the rotation block previously
  // had NO post-update columns at all, despite the weakest eigenvector
  // being a rotation axis on ~100% of scans measured this session --
  // standing/weak-direction-is-always-rotation). p_rot_eig_mid_pre/
  // p_rot_eig_max_pre fill out the PRE side's own already-computed
  // eigenvalue spectrum (only the min was previously kept). trP_pos_post
  // makes the position block's own post-update trace explicit rather than
  // requiring a reader to sum p_pos_eig_*_post themselves.
  double p_rot_eig_mid_pre  = -1.0, p_rot_eig_max_pre  = -1.0;
  double trP_pos_post       = -1.0;
  double p_rot_trace_post   = -1.0;
  double p_rot_eig_min_post = -1.0, p_rot_eig_mid_post = -1.0, p_rot_eig_max_post = -1.0;

  // CQ-19(b).  |v_min(P_pp) . v_min(HtH_pp)| -- the one number that signs
  // rho_ref: 1 if the prior's tightest axis coincides with HtH's weakest
  // (benign -- the prior fills exactly what the scan can't see), 0 if they
  // are orthogonal (a miscalibration, tight where the data is weak). -1.0
  // = unavailable (no residuals this frame, or P's position block missing).
  double cos_pmin_hmin = -1.0;

  // CQ-28: residual-redundancy-correction engagement/magnitude, from
  // ResidualRedundancyStats (see residual_redundancy.h) -- 0/0/0/1.0 both
  // when the mode is "off" and on a frame where no plane group had >= 2
  // matched residuals (nothing to correct is not an error condition).
  int    redund_groups     = 0;
  int    redund_n_raw      = 0;
  double redund_info_ratio = 1.0;
  int    information_increase_groups = 0;

  // CQ-34 item 4: a zero redund_groups conflates three distinct facts --
  // "no plane got >= 2 matched residuals", "some did but rho*plane_var_term
  // was non-positive (degenerate_pv)", and "some did but a residual's
  // independent variance went non-positive after subtracting the shared
  // term (degenerate_var)". redund_groups_seen counts group.size()>=2
  // BEFORE either degeneracy test, so seen - groups - degenerate_pv -
  // degenerate_var == 0 always, and the three failure modes are visible
  // separately instead of collapsing into one ambiguous zero.
  int redund_groups_seen           = 0;
  int redund_groups_degenerate_pv  = 0;
  int redund_groups_degenerate_var = 0;

  // CQ-31 item 7: naive vs Woodbury-corrected information-trace, always
  // populated (including mode=="off") -- see ResidualRedundancyStats.
  double naive_info_gain    = 0.0;
  double woodbury_info_gain = 0.0;
  double gamma_correction_trace = 0.0;
  double gamma_correction_frobenius = 0.0;
  double gamma_correction_min_eigenvalue = 0.0;
  double gamma_correction_max_eigenvalue = 0.0;
  double b_correction_norm = 0.0;

  // CQ-31 item 8: reduced_chi2 = sum(r^2/sigma_squared)/n_residuals, ~1 for
  // a correctly-calibrated residual model. Computed twice already elsewhere
  // in this codebase for a console-only debug line; this is the first time
  // it is retained per-frame (TQ-23 flagged its absence explicitly).
  double reduced_chi2 = 0.0;

  // TQ-20 item 1: kappa = P^-1/HtH, the ratio of prior information to
  // measurement information -- exactly 1/sqrt(1-rho_ref)-1 (rho_ref ==
  // refusal above, same formula), so no new matrix work is needed beyond
  // what ask/got/refusal already compute; kept as its own named column
  // rather than making a caller re-derive it from refusal. -1.0 when
  // refusal/got are unavailable (matches this struct's -1.0 sentinel
  // convention, not refusal's NaN one, since kappa_eff is a ratio a
  // consumer is more likely to filter with a numeric comparison).
  double kappa_eff = -1.0;

  // TQ-20 item 1/6: the 6 generalized eigenvalues of the pencil
  // (HtH, P) -- i.e. of P^-1*HtH, without ever forming P^-1 explicitly
  // (GeneralizedSelfAdjointEigenSolver(HtH, P) solves HtH*v=lambda*P*v
  // directly) -- ascending order, matching every other eigenvalue column
  // in this struct. A single joint 6-DOF measure of how much each
  // direction's measurement information dominates its prior, as opposed
  // to h_pp_min_eig/p_pos_eig_min_pre's separate 3x3 position-only views
  // above. -1.0 (all six) when P is unavailable or not positive-definite
  // (GeneralizedSelfAdjointEigenSolver requires the second matrix PD; a
  // failure here is itself diagnostic -- see kappa_gev_ok below).
  double kappa_gev0 = -1.0, kappa_gev1 = -1.0, kappa_gev2 = -1.0;
  double kappa_gev3 = -1.0, kappa_gev4 = -1.0, kappa_gev5 = -1.0;
  bool   kappa_gev_ok = false;
};

struct VoxelStats
{
  std::atomic<int>     open{0};
  std::atomic<int>     parent{0};
  std::atomic<int>     converged{0};
  std::atomic<int>     disabled{0};
  std::atomic<int>     planes{0};  // non-disabled, non-parent nodes currently fitting a plane
  std::atomic<int32_t> next_node_id{0};

  // init_frames_sum/init_count: VoxelNode::distinct_frames_'s value (total
  // distinct frames seen since node creation) at the moment of each node's
  // FIRST VoxelPlane::update() call -- answers "how many distinct frames
  // of view diversity had accumulated by the time this voxel was first
  // judged", the same quantity min_init_points is meant to proxy for via
  // raw point count alone.
  //
  // update_frames_sum/update_count: for every SUBSEQUENT refit, the number
  // of NEW distinct frames since that node's previous update() call (see
  // VoxelNode::last_update_distinct_frames_) -- i.e. the actual refit
  // INTERVAL in frames, not distinct_frames_'s own cumulative since-
  // creation total (which would conflate "how long has this voxel
  // existed" with "how often does it get refit").
  std::atomic<long long> init_frames_sum{0};
  std::atomic<long long> init_count{0};
  std::atomic<long long> update_frames_sum{0};
  std::atomic<long long> update_count{0};

  // R88 (binned map only): returns put into bins, bins opened, fits made from bins. Written by VoxelNode, read by the engagement report (an enabled map with binned_points == 0 is INERT).
  std::atomic<long long> binned_points{0};
  std::atomic<long long> binned_new_bins{0};
  std::atomic<long long> binned_fits{0};
  std::atomic<long long> binned_touched_at_fit{0};   // R91: sum over binned fits of the unique bins touched since the previous fit (R92: touched is the only count)
  // R97A R3: cumulative count of returns, under cov_mode fused/fused_pose, whose covariance needed the numerical floor before inversion. Always 0 under average/shrunk.
  std::atomic<long long> binned_fused_clamped{0};

  int total() const { return open + converged + disabled; }

  void increment(VoxelStatus s)
  {
    switch (s) {
      case VoxelStatus::OPEN:      open.fetch_add(1,      std::memory_order_relaxed); break;
      case VoxelStatus::PARENT:    parent.fetch_add(1,    std::memory_order_relaxed); break;
      case VoxelStatus::CONVERGED: converged.fetch_add(1, std::memory_order_relaxed); break;
      case VoxelStatus::DISABLED:  disabled.fetch_add(1,  std::memory_order_relaxed); break;
    }
  }

  void decrement(VoxelStatus s)
  {
    switch (s) {
      case VoxelStatus::OPEN:      open.fetch_sub(1,      std::memory_order_relaxed); break;
      case VoxelStatus::PARENT:    parent.fetch_sub(1,    std::memory_order_relaxed); break;
      case VoxelStatus::CONVERGED: converged.fetch_sub(1, std::memory_order_relaxed); break;
      case VoxelStatus::DISABLED:  disabled.fetch_sub(1,  std::memory_order_relaxed); break;
    }
  }

  void transition(VoxelStatus from, VoxelStatus to)
  {
    decrement(from);
    increment(to);
  }
};
using VoxelStatsPtr = std::shared_ptr<VoxelStats>;

struct BucketEntry
{
  std::vector<PointXYZCov> world;
  std::vector<V3D>         body;
};

class VoxelMap;
using VoxelMapPtr = std::shared_ptr<VoxelMap>;

}
