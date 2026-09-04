#pragma once

#include "livo_recon/utils/algo/math.h"
#include "livo_recon/utils/data/data_wrappers.h"
#include "livo_recon/utils/algo/hashing.h"

#include <atomic>
#include <limits>
#include <memory>
#include <vector>
#include <Eigen/Eigenvalues>

namespace livo_recon
{

struct Residual
{
  double r;
  V3D normal;

  // Capture time of the point this residual came from, copied straight from
  // PointXYZCov::t in LioProc::buildResiduals().  Only the scan-spline
  // control-point refinement reads it (lio/spline.h, refineWithLidar): a
  // residual's time is what selects the four control points it acts on, and
  // without it the refinement cannot be built from the residual set the IEKF
  // already accumulated.  0.0 on any path that does not set it, which
  // ScanSpline treats as "outside the window" and skips rather than
  // mis-attributing to the scan start.
  double t = 0.0;
  // Rotation Jacobian: point.cross(R1^T*normal), the R1 (rotation)
  // Jacobian column (lio_accumulator.cpp's accumulateLioResiduals()).
  V3D point_cross_normal;
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

  // P6a.  This candidate's ray classified against its own plane's occupancy
  // chart: 0=hit, 1=known free, 2=unobserved, -1=not classified (not a
  // gate() candidate). Measurement only -- never read by anything that
  // feeds the state estimate.
  int vis_state = -1;

  // This plane's own fit-uncertainty contribution (n * plane_var * n),
  // added into sigma_squared by LioProc::buildResiduals().
  double plane_var_term = 0.0;

  // Which tier of VoxelMap::findPlaneResidual()'s fallback matched this
  // point: 0 = primary voxel, 1 = single directional neighbor (the only
  // fallback FAST-LIVO2 also has), 2 = the full neighborhood_size box
  // search (livo_recon-only, picks the best-scoring candidate among up to
  // ~(2*neighborhood_size+1)^3 voxels rather than giving up). Debug-only,
  // for attributing residual-quality issues to a specific tier.
  int match_tier = 0;
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
struct WorldPointCov
{
  V3D point;
  M3D sensor_cov;
  M3D pose_cov;

  // History (66-87): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-66
  V3D body_point = V3D::Zero();
  M3D rot_transpose = M3D::Identity();
  Eigen::Matrix<double, 6, 6> prior_cov_rp = Eigen::Matrix<double, 6, 6>::Zero();

  M3D total() const { return sensor_cov + pose_cov; }
};

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

// Running sufficient statistics for VoxelNode's DEFAULT (unweighted,
// equal-weight-per-point) points_ accumulation path, maintained
// incrementally in O(1) per inserted point (see VoxelNode::insertPoints())
// so VoxelPlane::update() can derive mean/covariance in O(1) at fit time
// instead of re-scanning every accumulated point on every refit -- exact,
// not an approximation. Only valid when every point contributing to a fit
// carries equal weight, which is true for VoxelNode's raw points_ path but
// NOT the separate weighted bin path (see PointBin/useBins()) --
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

  // true -- VoxelPlane::gate()'s sigma_diag_squared includes pose_cov (the
  // point's own deskew-transform uncertainty, itself derived from the
  // state's prior covariance) alongside sensor_cov, despite that being a
  // double-count against solveSystem's own P_before^-1 prior term.
  // Originally kept true because excluding it regressed badly in isolation
  // (pre-info_gain/pre-var_acc-based-time_based_process_noise), acting as
  // necessary damping. false (now the default) -- exclude pose_cov from
  // sigma_diag_squared entirely (gate() still uses sensor_cov only); confirmed on
  // eee_01 (90s, clean/fair harness, imu/undistort/time_based_process_noise
  // ="var_acc") to further improve ATE 0.090m -> 0.078m on top of the
  // var_acc-based noise-floor fix, which now supplies the damping this
  // double-count was standing in for, without inheriting the collapse
  // risk of using pose_cov to do it.
  bool pose_cov_in_sigma           = false;

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
  // "frame_gated": same lock-in as "normal", but ALSO requires
  // distinct_frames_ >= min_frames_to_converge (VoxelNode's count of
  // distinct frames that have contributed points since node creation) --
  // targets the specific failure found when undecimated map insertion lets
  // a single frame's dense point cluster fill a voxel to max_points before
  // it's seen enough independent viewpoints/scan passes.
  std::string convergence_mode      = "normal";
  int    min_frames_to_converge     = 5;

  // 1 (default, no-op) -- a voxel's FIRST plane fit (isInit() threshold,
  // min_init_points above) may happen off points from a single frame.
  // >1 -- ALSO require VoxelNode::distinct_frames_ >= min_frames_to_init
  // before that first fit, same spirit as min_frames_to_converge but
  // applied at init instead of lock-in. Targets a failure min_frames_to_
  // converge/max_points can't: with point_filter_num relaxed toward
  // "keep everything", a single dense frame near the sensor can hand a
  // voxel min_init_points-worth of points in one shot -- a PCA fit off
  // points from one instant/viewing angle, immediately live for residuals
  // (VoxelNode::hasConvergedPlane()'s docs: usable the moment isPlane()
  // is true, not gated on CONVERGED) -- confirmed empirically: raising
  // min_init_points alone (a point-count proxy for view diversity) fixed
  // an NTU_VIRAL point_filter_num=1 ATE regression (eee_01: 1.48m ->
  // 0.12m) where min_frames_to_converge/max_points changes had zero
  // effect, but a point-count threshold doesn't generalize across lidars
  // with different angular resolution/point density the way a frame-count
  // requirement does.
  //
  // Skipped entirely for the calibration/bootstrap frame (global frame
  // index 0 -- see VoxelNode::insertPoints()'s g_current_frame_idx check,
  // and CalibProc::estimateFromBuffer()/skipCalibration(), which both push
  // their very first MeasureGroup as frame 0): that frame already
  // legitimately aggregates many real distinct lidar frames captured
  // while the sensor was stationary (CalibProc::collectSamples()'
  // num_samples), and LIO needs SOME usable planes the instant calibration
  // ends, not after min_frames_to_init more real frames separately elapse.
  int    min_frames_to_init         = 1;

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

  // "pca" (default): today's unchanged behavior -- unweighted PCA on raw
  // point positions, plane_var_ (fit uncertainty) a separate O(N)
  // per-point Jacobian sandwich propagating each point's PointXYZCov::cov.
  // "debiased": VoxelPlane fits the normal DIRECTLY from a debiased
  // (noise-corrected) structure tensor, M_debiased = Spp/N - mean*mean^T -
  // Scov/N (Scov = sum of point covariances) -- a closed-form correction
  // for the fact that E[(p-mean)(p-mean)^T] = TrueScatter + Cov under
  // additive per-point measurement noise, so subtracting the accumulated
  // noise term gives a consistent (unbiased) estimate of the noise-free
  // plane structure without needing IRLS/iteration. plane_var_ is
  // reconstructed from a second set of O(1) fixed-size accumulators
  // (cross-moments of point position against point covariance) instead of
  // replaying raw points -- see docs/debiased_voxel_plane_fit_2026aug24.md
  // for the full derivation. VoxelNode never locks a "debiased" voxel into
  // CONVERGED (no O(N) cost to keep refitting, so no reason to stop and
  // discard further points) -- min_update_points/max_points/
  // convergence_mode/min_frames_to_converge are PCA-mode-only under this
  // setting. KNOWN CAVEAT: the debiasing correction assumes independent
  // per-point noise; points from a single lidar frame/viewpoint have
  // documented correlated noise (see sensor_noise_floor_eig0's docs above)
  // that can make Scov over-subtract and drive the debiased eigenvalue
  // negative on early/single-frame fits -- min_frames_to_init and the
  // sensor_noise_floor_eig0 floor both still apply to guard this, but
  // debiased mode may reject/collapse more often on early voxels than pca
  // mode until this is characterized by a real comparative sweep.
  std::string plane_fit_mode        = "pca";

  // History (348-361): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-348
  std::string plane_fit_pose_cov_mode = "combined";

  // History (364-384): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-364
  std::string bin_weight_mode_fit = "count";
  std::string bin_weight_mode_var = "count";

  // History (388-396): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-388
  double bin_size_fraction = 0.2;

  // ── the plane covariance: which model produces plane_var_ ────────────
  //
  // "eigengap" is the incumbent: differentiate the fitted eigenVECTOR with
  // respect to each point, which introduces 1/(lambda0 - lambda_m) and is
  // why plane_var_denom_floor_en, the debiased denom guard and the
  // kPlaneVarCeiling reject all exist.  It blows up when the plane is
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
  // with plane_var_denom_floor_en and sensor_noise_floor_eig0.
  std::string plane_var_mode = "eigengap";

  // History (399-409): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-399
  //
  // pca ONLY.  VoxelNode::insertPoints()'s debiased branch hands the raw
  // points_world straight to VoxelPlane::addPoints() and never reaches
  // bins_/buildBinReps(), so this flag has no effect there.  It is nested
  // under plane_fit_mode=pca in VoxelMap::loadParameters() and setting it on
  // debiased REFUSES at startup rather than being silently ignored -- see
  // the comment there for why that confound matters and what replaces
  // binning.  bin_size_fraction and bin_weight_mode_fit/_var are in turn
  // nested under this flag.
  bool use_bins = false;

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

  // History (459-476): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-459
  std::string plane_gate_mode = "disc";
  static constexpr const char* PLANE_GATE_MODES[] = {
      "disc", "ellipse", "ellipse_area_matched" };

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
  //                   plane instead of chosen once as a literal.  Available
  //                   on BOTH fit modes: on the debiased path lambda0 is
  //                   already noise-corrected, on pca the same subtraction
  //                   is done at query time (which is one reason the unified
  //                   model does not need plane_fit_mode as an arm).
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

  // History (550-554): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-550
  bool plane_var_denom_floor_en = false;

  // History (557-567): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-557
  int shuffle_insertion_seed = 0;

  // History (570-575): see docs/livo_recon_changelog.md#include-livo_recon-utils-map-voxelmap_utils.h-570
  bool log_frame_stats_en = false;
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
  double sum_weight   = 0.0;  // total residual weight, i.e. how much information
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
  // DX-2 preflight fix.  THIS scan's spline-window start (t0) vs the
  // PREVIOUS scan's finalized end (t1) -- the inter-scan discontinuity
  // nothing had measured before. -1 if unavailable (first spline scan,
  // spline disabled, or previous scan's spline failed).
  double boundary_dpos     = -1.0;  // metres
  double boundary_drot_deg = -1.0;  // degrees
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
