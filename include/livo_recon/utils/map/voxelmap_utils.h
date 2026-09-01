#pragma once

#include "livo_recon/utils/algo/math.h"
#include "livo_recon/utils/data/data_wrappers.h"
#include "livo_recon/utils/algo/hashing.h"

#include <atomic>
#include <memory>
#include <vector>
#include <Eigen/Eigenvalues>

namespace livo_recon
{

struct Residual
{
  double r;
  V3D normal;
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

  // T0-D (2026-08-31): the frame-level context needed to compute this
  // correspondence's full innovation covariance S = H P- H^T + R for
  // corr.csv (see VoxelOpts::log_consistency_corr_en) -- NOT used by any
  // real accept/reject or EKF-weighting logic, which never needed the
  // H P- H^T term (this codebase's IEKF is a batch-WLS normal-equation
  // solve, not a per-correspondence sequential Kalman update -- see
  // ekf.h's applyMeanUpdate()). Populated unconditionally (cheap: a V3D
  // copy plus two small matrices set ONCE per frame and copied per point)
  // by LioProc::buildResiduals() so VoxelPlane::computeResidual() can
  // reconstruct this point's Jacobian and reach into the frame's prior
  // covariance -- without touching VoxelPlane's own constructor/call
  // chain, since WorldPointCov already flows unmodified through the
  // whole findPlaneResidual() chain. body_point: this point in
  // SENSOR/body frame (pre-toWorld) -- combined with plane_.normal (only
  // known inside VoxelPlane) via rot_transpose to reconstruct
  // point_cross_normal exactly as LioProc::buildResiduals() does for its
  // own HtH accumulation. rot_transpose: state_->rot().transpose() at
  // this frame's prior. prior_cov_rp: the frame's FIXED prior_cov_'s
  // [R,P]x[R,P] 6x6 block (R,P are contiguous starting at
  // StateGroup::idxR(), see ekf.h) -- the only sub-block a point-to-plane
  // residual's Jacobian (nonzero only in R,P columns) can have any
  // quadratic-form interaction with.
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

  // Gates VoxelNode::debugLogPlaneInit()'s per-voxel-first-plane-fit dump
  // (/tmp/plane_init.txt) and VoxelPlane's noise-floor dump
  // (/tmp/noise_floor.txt) -- both were previously unconditional (2026-08-09
  // cleanup, task #149: unlike most of this codebase's debug logs, these
  // had no gate at all). Off by default.
  bool log_debug_en                = false;

  // T3-0d (2026-08-30): does plane_var_ have enough authority in the
  // residual variance budget to move ATE at all? Gates two debug dumps,
  // off by default. /tmp/variance_shares.txt (one line per accepted
  // correspondence, from VoxelPlane::computeResidual()): sigma_diag_squared,
  // plane_var_term, and plane_var_term's share of their sum -- the direct
  // measurement T3-0/T3-0b/T0-B-2/T0-B-3 never took, all four having
  // turned knobs that act on plane_var_ and reported only ATE.
  // /tmp/plane_fit_stats.txt (one line per VoxelPlane::update() call that
  // used weights): N (points/bins seen), effective sample size N_eff =
  // (sum w)^2 / sum(w^2), and trace(plane_var_) -- lets a bin_size_fraction
  // or bin_weight_mode sweep be read against how much N_eff actually moved,
  // instead of inferring it from ATE alone.
  bool log_variance_shares_en      = false;

  // use_disc_distance (2026-08-04, tried and REMOVED): treated each
  // VoxelPlane as a finite disc (radius_ around center) rather than an
  // infinite plane, computing r as the true 3D point-to-disc distance
  // (Pythagorean combination of perpendicular + excess in-plane distance
  // beyond radius_, with the residual's gradient direction rotating
  // toward the in-plane radial direction for points past the edge) instead
  // of today's plane-only r gated by a hard max_radius*radius_ cutoff.
  // Motivated by nya_02's cov_acc collapse investigation (a genuine
  // physical argument -- the flat-plane assumption is shakiest right at a
  // patch's edge) but regressed badly on nya_02 across every cov_acc
  // tested, including ones that were otherwise clean without it: 0.01
  // (0.81m -> 8498.7m), 0.1 (0.033m -> 0.200m), gradually-compounding not
  // an instant blowup. Consistent with this session's broader pattern --
  // every attempt to make residual weighting more "statistically honest"
  // (info_gain, Woodbury, cross_frame_reuse, this) has hurt real-world
  // tracking robustness rather than helped it.

  // cross_frame_reuse_ref (2026-08-03, tried and REMOVED): a per-plane
  // discount based on VoxelPlane::match_streak_ (how many consecutive
  // frames had matched at least one residual against a plane), motivated
  // by trace(P_PP)'s anti-correlation with error persisting over a ~1-2s
  // (many-frame) window rather than a single-frame spike -- unlike every
  // other mechanism tried this investigation (density_sigma_ref, the
  // abandoned plane_averaged/count_weighted/info_gain/
  // woodbury_plane_correction), this was the only one that looked at
  // cross-frame (not within-frame) correlation. It DID change the
  // qualitative oscillation character (at cross_frame_reuse_ref=2, eee_01:
  // mean-crossings of trace(P_PP) dropped ~60%, 113->45, and coefficient
  // of variation dropped ~23%, 6.09->4.67) but produced no meaningful ATE
  // improvement across ref values 2/5/10/20 (0.0256/0.0254/0.0254/0.0252m
  // vs 0.0254m baseline, all within noise) -- removed as not worth the
  // added complexity given the effect size.

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

  // Controls plane_var_'s own residual-weighting propagation -- a
  // DIFFERENT quantity from pose_cov_in_sigma above (which controls only
  // gate()'s sigma_diag_squared). Applies to BOTH plane_fit_mode paths:
  // the PCA Jmin loop (VoxelPlane::update()) and, since 2026-08-24 pass4
  // (see docs/debiased_voxel_plane_fit_2026aug24.md), the debiased path's
  // own Scov_/V_/W_ accumulators too -- initially scoped PCA-only, but
  // live diagnostics showed plane_var_ blowups (traces >900,000) driven
  // by pos_cov's own magnitude for fresh/small-N voxels, independent of
  // the denom1/denom2 eigengap guard, so debiased mode needs this
  // exclusion option too. "combined" (default, unchanged) -- sensor_cov+
  // pos_cov, the full noise budget. "sensor_only" -- excludes pos_cov
  // from this propagation entirely (pos_cov, poseCovAt()'s range^2
  // lever-arm term, is documented to "dwarf true sensor noise for far
  // points" -- the same reason pose_cov_in_sigma defaults false).
  std::string plane_fit_pose_cov_mode = "combined";

  // Controls VoxelNode::buildBinReps()'s two per-bin weight vectors,
  // consumed by VoxelPlane::update() as w_i/weight_sum (always self-
  // normalizing, so only the RELATIVE weighting across bins matters, not
  // the absolute scale). "count" -- weight = bin.count, i.e. the sum is
  // the inverse participation ratio over bins, an effective sample size
  // bounded above by the bin count J but shaped by point density within
  // bins (the Kish effective-N). "uniform" -- weight = 1 for every
  // occupied bin regardless of how many points landed in it, i.e. true
  // occupancy counting (effective sample size is exactly J). Only takes
  // effect when use_bins is true (see VoxelNode::useBins()).
  //
  // Split into _fit and _var 2026-08-30 (T3-0c): before this, one
  // "bin_weight_mode" fed BOTH the plane fit (center/covariance/normal,
  // in VoxelPlane::update()'s weighted mean/covariance) and the
  // plane_var_ Jacobian in the same call -- so T3-0's "uniform" arm moved
  // the ESTIMATE and its UNCERTAINTY together, the identical confound T3-0
  // was built to strip out of T0-B-2. T3's actual proposal only reweights
  // the uncertainty term (its directional M_cov substitutes for this
  // scalar case) -- (fit=count, var=count) reproduces pre-split behavior;
  // (fit=count, var=uniform) is the arm that actually tests T3's premise
  // in isolation.
  std::string bin_weight_mode_fit = "count";
  std::string bin_weight_mode_var = "count";

  // VoxelNode's PointBin accumulator bin size, as a fraction of that
  // node's own voxel extent (opts->voxel_size / 2^layer) -- see
  // VoxelNode::density_weight_leaf_. Was a hard-coded
  // kDensityWeightLeafFraction constant chosen only to bound refit cost;
  // exposed as a config option 2026-08-30 (T3-0b) since it also sets the
  // effective sample size (occupied-bin count J) that bin_weight_mode
  // above operates over, and had never been swept. Smaller -> finer bins,
  // more of them (J closer to N, less aggregation, higher refit cost).
  // Larger -> coarser bins, fewer of them (J smaller, more aggregation).
  double bin_size_fraction = 0.2;

  // Whether VoxelNode uses the PointBin accumulator (bins_) instead of
  // raw points_ storage -- see VoxelNode::useBins(). Split out from
  // convergence_mode 2026-08-30 (T0-B-4): before this, "always_update"
  // implied binning as a side effect (binning exists to bound refit cost
  // for a voxel that never locks), which meant every prior experiment
  // that varied convergence_mode to turn binning "on/off" also changed
  // whether the voxel could converge and freeze -- two effects in one
  // knob. Defaults false (raw points_, matching every non-always_update
  // config's actual historical behavior); a config that wants the OLD
  // "always_update implies binning" behavior must set this explicitly
  // alongside convergence_mode: "always_update".
  bool use_bins = false;

  // T0-D (2026-08-31): corr.csv -- one row per point-to-plane
  // correspondence, BEFORE the outlier (sigma_num_squared) gate, for
  // offline filter-consistency diagnosis (scripts/analysis/consistency.py
  // -- NIS/whiteness/Q-vs-R triage, no ground truth needed). Core columns
  // only: scan_id (=this frame's VoxelMap::frame_idx_), nu (=r), S (=full
  // residual variance, sensor[+pose]+plane_var_term), gated (0/1 -- did
  // this correspondence pass the chi2 test, logged either way, unlike
  // variance_shares.txt which only ever sees accepted ones -- the
  // register's own note is that gating biases NIS DOWN by truncating the
  // upper tail, so scoring must see the pre-gate population). Deliberately
  // split from log_variance_shares_en (a different, narrower log already
  // in production use) and from log_consistency_covariates_en below, so a
  // run can log the cheap core NIS columns without paying for the extra
  // per-point breakdown. Off by default.
  bool log_consistency_corr_en = false;

  // Extra covariate columns appended to log_consistency_corr_en's core
  // row: S_sensor, S_pose (gate()'s two independent variance sources,
  // logged separately regardless of pose_cov_in_sigma), S_plane_tilt,
  // S_plane_d (plane_var_term's quadratic form split by J_nq's tilt vs. d
  // components, off-diagonal tilt/d cross-terms divided evenly between the
  // two -- S_plane_tilt+S_plane_d reconstructs plane_var_term exactly), N
  // (this plane's total accumulated point count), J (occupied bin count if
  // this plane is bin-fit, else 0), aniso (eigen_values_(2)/
  // eigen_values_(1), in-plane coverage anisotropy -- T3's premise
  // directly: consistency.py's panel 5 bins NIS by this), lambda0
  // (eigen_values_(0), T0-G 2026-08-31 -- the plane fit's own is_plane_
  // threshold quantity; T8-0b's/T8-d's outcome variable). range/incidence
  // are NOT logged -- computing them needs the sensor's world position,
  // which isn't available at VoxelPlane::computeResidual()'s call depth
  // without threading it through the whole MapBackend virtual interface;
  // consistency.py degrades gracefully (skips that one covariate) when the
  // column is absent, so this was scoped out rather than done partially.
  // No-op unless log_consistency_corr_en is also true. Off by default.
  bool log_consistency_covariates_en = false;

  // T8-a (2026-08-31): "disc" (default) -- today's isotropic admission
  // test, range_dis > max_radius*radius_ with radius_ = sqrt(lambda2) (the
  // LARGEST in-plane eigenvalue) -- a sliver sampled 1m along one axis and
  // 3cm along the other admits correspondences 3m out along the axis it
  // barely observed, since the disc radius is set by the axis with the
  // MOST support. "ellipse" -- Mahalanobis ellipse of the fit's own
  // sampling, using the SAME x_normal_/y_normal_ basis J_nq already uses
  // (no new geometry): m2 = u1^2/lambda2 + u2^2/lambda1, gated against
  // max_radius^2. Bit-identical to "disc" only in the degenerate case
  // lambda1==lambda2; otherwise a real behavior change, so it must be
  // opted into explicitly.
  // "ellipse_area_matched" (2026-09-01): "ellipse" above is strictly
  // CONTAINED in the disc (admits only sqrt(lambda1/lambda2) of its area),
  // so an ellipse-vs-disc comparison confounds shape with admitted-area --
  // this mode rescales the ellipse's threshold by sqrt(lambda2/lambda1)
  // (capped at 4x, see gate()'s AUDIT comment) so the admitted AREA matches
  // the disc's, isolating the shape change the coverage hypothesis is
  // actually about.
  std::string plane_gate_mode = "disc";

  // T3-0e (2026-08-31): test T3's DIRECTIONAL premise directly -- drop
  // correspondences from planes whose occupancy anisotropy (see
  // VoxelPlane::occupancyAnisotropy()) is judged "bad" by one of two
  // rules, applied at gate() time (a genuine drop from the residual, not
  // just a logged diagnostic):
  //   "none" (default): no dropping, bit-identical to today.
  //   "top": drop if occupancyAnisotropy() > occ_aniso_drop_threshold.
  //     The threshold must be supplied externally (occ_aniso_drop_threshold
  //     has no sensible default) -- computed from a baseline pass's own
  //     occ_aniso distribution (e.g. its 90th percentile, for a "top
  //     decile" cut). Leaving it at -1 (a value no real anisotropy can be
  //     below) makes "top" mode a no-op, matching "none" -- a safe default
  //     rather than an easy-to-miss misconfiguration.
  //   "random": drop with probability occ_aniso_drop_fraction, decided
  //     once per PLANE (not per correspondence) via a fixed-seed hash of
  //     plane_.center -- deterministic and reproducible, but only
  //     approximately size-matched to "top" mode's actual drop count (in
  //     EXPECTATION, not exact count) since this codebase has no
  //     first-class per-plane identity to do an exact two-pass match
  //     against. Document this approximation plainly if this arm's own
  //     result is ever quoted.
  std::string occ_aniso_drop_mode = "none";
  double occ_aniso_drop_threshold = -1.0;
  double occ_aniso_drop_fraction = 0.1;
  int occ_aniso_drop_seed = 0;

  // Code-review fix, 2026-08-31: occupancyAnisotropy() returns -1.0
  // (undefined) when fewer than 3 cells are occupied -- typically a plane
  // whose normal/anchor is still settling. Under "top" mode, -1.0 used to
  // silently never exceed occ_aniso_drop_threshold, so undefined planes
  // were ALWAYS kept -- systematically protecting the least-converged
  // planes, plausibly the ones the ablation most wants to be able to
  // remove. false (default): undefined counts as "keep" (conservative,
  // matches the old accidental behavior, now a deliberate choice). true:
  // undefined counts as "top decile" (drop). Either way, the count of
  // undefined-and-therefore-policy-decided planes is now logged (corr.csv
  // gains a dropped_by_ablation column) so this bucket's size is visible,
  // not just its existence.
  bool occ_aniso_undefined_as_top = false;

  // ------------------------------------------------------------------
  // T8-b (2026-09-01): the three plane-confidence terms, each on its own
  // switch so the grid can turn them on and off independently. A
  // point-to-plane update can be wrong for three separable reasons --
  // the points were noisy, the points were redundant, the points did not
  // cover the plane -- and livo_recon has had a partial answer to each
  // that could not compose. These two flags supply the missing two.
  //
  // Both are EXACTLY behaviour-preserving when off, and the coverage term
  // is additionally an identity when coverage is isotropic, so a null
  // result is distinguishable from an inert switch: see the occ_cells
  // and plane_conf_factor columns in corr.csv.
  //
  // redundancy: plane_var_ scales as 1/N. On the pca path the bin
  // weights already discount repeated returns; on the debiased path
  // N_acc_ counts every redundant return, so debiased is over-confident
  // by the redundancy factor precisely where binning exists to prevent
  // it. Replace the raw count with a coverage-derived effective sample
  // size (occupied 8x8 cells) and inflate by the ratio.
  bool   plane_conf_redundancy_en  = false;
  double plane_conf_redundancy_cap = 16.0;  // hard ceiling on the inflation ratio

  // coverage: inflate each TANGENT axis of plane_var_ by how thinly the
  // plane was sampled along that axis's lever arm. Uses the occupancy
  // bitmask's per-axis second moments, which are density-independent by
  // construction -- that is the whole reason to read coverage off the
  // bitmask rather than off the scatter eigenvalues.
  bool   plane_conf_coverage_en   = false;
  double plane_conf_coverage_beta = 1.0;    // 0 = identity, 1 = full ratio
  double plane_conf_coverage_cap  = 100.0;  // ceiling on either axis factor

  // T1 (2026-09-01): refitDebiased() rejects a fit whose eigengap
  // denominators are within eps of zero; update() -- the pca path -- has
  // NO such guard and will happily divide plane_var_ by an arbitrarily
  // small denom1. That asymmetry sits underneath every pca-vs-debiased
  // comparison in the register. On: apply the identical guard to both.
  bool plane_var_denom_floor_en = false;

  // T0-G (2026-08-31): diagnostic-only. 0 (default) -- no-op, today's
  // behavior unchanged. Nonzero -- deterministically shuffle each frame's
  // point order (std::mt19937 seeded from this value XOR'd with the frame
  // index, so every frame gets a distinct but reproducible permutation)
  // immediately before VoxelMap::updateMap()'s bucket-build step. Point
  // order affects nothing SEMANTICALLY (a voxel's accumulated sums are
  // order-independent in exact arithmetic), but floating-point summation
  // is order-dependent -- this exists purely to measure how much that
  // last-bit-level nondeterminism can move ATE, i.e. this register's first
  // error bar. See VoxelMap::updateMap()'s call site for the actual
  // shuffle.
  int shuffle_insertion_seed = 0;

  // T0-F-2b (2026-08-31): logs frame_stats.txt (t, frame_idx,
  // denom_rejected_count, max_plane_var_trace) bracketing each
  // VoxelMap::updateMap() call -- see voxelPlaneFrameStats{Reset,Read}()
  // in voxelplane.h for what's aggregated and why here specifically
  // (ordering-independent w.r.t. common/insert_map_after_lio). Off by
  // default.
  bool log_frame_stats_en = false;
};
using VoxelOptsPtr = std::shared_ptr<VoxelOpts>;

enum class VoxelStatus { OPEN, PARENT, CONVERGED, DISABLED };

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
