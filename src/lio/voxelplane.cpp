#include "livo_recon/lio/voxelplane.h"
#include "livo_recon/utils/log/debug_log_dir.h"
#include <fstream>
#include <sstream>
#include <mutex>
#include <atomic>
#include <limits>
#include <vector>
#include <cmath>
#include <functional>

namespace livo_recon
{

namespace
{
// T0-F-2b: see voxelplane.h's voxelPlaneFrameStats{Reset,Read}() doc
// comment. atomic<double> has no fetch_max pre-C++20 -- CAS loop instead.
std::atomic<int> g_denom_rejected_count{0};
std::atomic<double> g_max_plane_var_trace{-1.0};

void updateMaxPlaneVarTrace(double trace)
{
  double cur = g_max_plane_var_trace.load(std::memory_order_relaxed);
  while (trace > cur &&
         !g_max_plane_var_trace.compare_exchange_weak(cur, trace, std::memory_order_relaxed)) {}
}
void debugLogNoiseFloor(const std::string& msg)
{
  static bool first_call = true;
  std::ofstream ofs(debugLogPath("noise_floor.txt"), first_call ? std::ios::trunc : std::ios::app);
  first_call = false;
  ofs << msg << "\n";
}

// T3-0d: computeResidual() runs inside LioProc::buildResiduals()'s OMP
// parallel-for loop, so this needs its own lock -- debugLogNoiseFloor()
// above is only ever called from single-threaded contexts and has no such
// guard. Off by default (log_variance_shares_en), so the lock is never
// taken in a normal run.
void debugLogVarianceShare(double sigma_diag_squared, double plane_var_term)
{
  static bool first_call = true;
  static std::mutex mtx;
  const double total = sigma_diag_squared + plane_var_term;
  const double share = total > 0.0 ? plane_var_term / total : 0.0;
  std::lock_guard<std::mutex> lock(mtx);  // guards first_call too -- see comment above
  std::ofstream ofs(debugLogPath("variance_shares.txt"), first_call ? std::ios::trunc : std::ios::app);
  first_call = false;
  ofs << sigma_diag_squared << " " << plane_var_term << " " << share << "\n";
}

// T3-0d: N/J/N_eff/trace(plane_var_) per VoxelPlane::update() call that
// commits a plane -- lets a bin_size_fraction or bin_weight_mode sweep be
// read against how much the effective sample size actually moved, rather
// than inferred from ATE alone. May be called concurrently across
// different voxels during map insertion, hence the same mutex pattern as
// debugLogVarianceShare() above.
void debugLogPlaneFitStats(int n, int j, double n_eff, double trace_plane_var)
{
  static bool first_call = true;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  std::ofstream ofs(debugLogPath("plane_fit_stats.txt"), first_call ? std::ios::trunc : std::ios::app);
  first_call = false;
  ofs << n << " " << j << " " << n_eff << " " << trace_plane_var << "\n";
}

// T0-D (2026-08-31): corr.csv for scripts/analysis/consistency.py --
// header written once on first call (trunc), core-vs-full row shape
// decided by whether the covariate fields are finite/non-empty (see
// computeResidual()'s call site: covariates are all left at their default
// -1.0/-1 sentinel when log_consistency_covariates_en is off, and this
// function omits those columns from the header AND every row in that
// case, since consistency.py keys off column presence, not sentinel
// values). Same per-call mutex pattern as the two functions above --
// computeResidual() runs inside LioProc::buildResiduals()'s OMP
// parallel-for.
void debugLogConsistencyCorr(bool with_covariates, int scan_id, double nu, double S,
                              double s_sensor, double s_plane_tilt, double s_plane_d,
                              double s_pose, int n, int j, double aniso, int gated,
                              double s_prior_pose, double lambda0, double occ_aniso, int occ_cells,
                              int dropped_by_ablation, double occ_var_u, double occ_var_v,
                              double plane_conf_factor)
{
  static bool first_call = true;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  std::ofstream ofs(debugLogPath("corr.csv"), first_call ? std::ios::trunc : std::ios::app);
  if (first_call) {
    // dropped_by_ablation (2026-08-31, code-review fix): unconditional,
    // not gated behind with_covariates -- T3-0e's ablation arms need this
    // even in a plain (non-covariates) NIS run, to know which rows were
    // excluded from the residual by the ablation itself vs by the
    // ordinary sigma-gate (gated=1 without this =0 means the LATTER).
    // When this is 1, S/s_sensor/.../s_prior_pose are all the -1.0
    // sentinel (gate() does not finish computing them for a dropped
    // correspondence) -- only nu, N/J/aniso/lambda0/occ_aniso/occ_cells
    // (plane-level, valid regardless) are meaningful on that row.
    ofs << "scan_id,nu,S,gated,dropped_by_ablation";
    // T0-G (2026-08-31): lambda0 added -- eigen_values_(0), the plane fit's
    // smallest eigenvalue (its own is_plane_ = eigen_values_(0) <
    // plane_threshold test uses this directly). T8-0b/T8-d's outcome
    // variable ("median lambda0"); previously computed but never logged
    // anywhere. T3-0e (2026-08-31): occ_aniso/occ_cells added --
    // VoxelPlane::occupancyAnisotropy()/occupiedCellCount(), the OCCUPANCY-
    // based (density-independent) in-plane coverage anisotropy, distinct
    // from `aniso` (point-scatter/density-weighted).
    if (with_covariates) ofs << ",S_sensor,S_plane_tilt,S_plane_d,S_pose,S_prior_pose,N,J,aniso,lambda0,occ_aniso,occ_cells,occ_var_u,occ_var_v,plane_conf_factor";
    ofs << "\n";
  }
  first_call = false;
  ofs << scan_id << "," << nu << "," << S << "," << gated << "," << dropped_by_ablation;
  if (with_covariates)
    ofs << "," << s_sensor << "," << s_plane_tilt << "," << s_plane_d << "," << s_pose
        << "," << s_prior_pose << "," << n << "," << j << "," << aniso << "," << lambda0
        << "," << occ_aniso << "," << occ_cells
        << "," << occ_var_u << "," << occ_var_v << "," << plane_conf_factor;
  ofs << "\n";
}
}  // namespace

void voxelPlaneFrameStatsReset()
{
  g_denom_rejected_count.store(0, std::memory_order_relaxed);
  g_max_plane_var_trace.store(-1.0, std::memory_order_relaxed);
}

void voxelPlaneFrameStatsRead(int& denom_rejected_count, double& max_plane_var_trace)
{
  denom_rejected_count = g_denom_rejected_count.load(std::memory_order_relaxed);
  max_plane_var_trace = g_max_plane_var_trace.load(std::memory_order_relaxed);
}

VoxelPlane::VoxelPlane(VoxelOptsPtr opts)
  : opts_(opts)
{
  plane_var_ = M3D::Zero();
  covariance_ = M3D::Zero();
  for (int a = 0; a < 3; ++a) {
    W_[a] = M3D::Zero();
    for (int b = 0; b < 3; ++b) V_[a][b] = M3D::Zero();
  }
}

bool VoxelPlane::gate(const V3D& p, const M3D& sensor_cov, const M3D& pose_cov,
                      double& r, double& sigma_diag_squared, double& plane_var_term,
                      Eigen::Matrix<double, 1, 3>& J_nq, bool* is_candidate,
                      bool* dropped_by_ablation) const
{
  const V3D& n = plane_.normal;
  r = n.dot(p) + plane_.d;
  if (!std::isfinite(r)) return false;

  const V3D d_center = p - plane_.center;

  if (opts_->plane_gate_mode == "ellipse" ||
      opts_->plane_gate_mode == "ellipse_area_matched") {
    // T8-a: Mahalanobis ellipse of the fit's own sampling, replacing the
    // isotropic disc -- same x_normal_/y_normal_ basis J_nq uses below, no
    // new geometry. eigen_values_(2)/(1) are the largest/second-largest
    // in-plane eigenvalues (x_normal_/y_normal_'s own axes respectively).
    constexpr double eps = 1e-12;
    const double l2 = std::max(eigen_values_(2), eps);   // major in-plane
    const double l1 = std::max(eigen_values_(1), eps);   // minor in-plane
    const double u1 = d_center.dot(x_normal_);           // along the l2 axis
    const double u2 = d_center.dot(y_normal_);           // along the l1 axis
    const double m2 = u1 * u1 / l2 + u2 * u2 / l1;
    double thr2 = opts_->max_radius * opts_->max_radius;
    // AUDIT, 2026-09-01. "ellipse" is not a shape-only change. Its region
    // has semi-axes max_radius*sqrt(l2) and max_radius*sqrt(l1) against the
    // disc's radius max_radius*sqrt(l2), so it is strictly CONTAINED in the
    // disc and admits sqrt(l1/l2) of its area. On a sliver at aniso 1000
    // that is 1/32 of the correspondences. An ellipse-vs-disc comparison
    // therefore confounds "the gate is the wrong shape" with "the gate
    // admitted far fewer points", and the size of the confound scales with
    // the very anisotropy under test -- exactly the population-matching
    // defect the T3-0e review found in the drop ablation.
    //
    // ellipse area = pi * T * sqrt(l1*l2);  disc area = pi * R^2 * l2
    // equal  =>  T = R^2 * sqrt(l2/l1)
    // "ellipse_area_matched" is the shape change at matched admitted area,
    // and is the arm the coverage hypothesis is actually about.
    if (opts_->plane_gate_mode == "ellipse_area_matched") {
      // Capped at 4x. Beyond aniso ~16 the areas are only PARTIALLY matched
      // and the arm drifts back toward the plain ellipse -- which is the
      // safe direction to fail, since the alternative is admitting points
      // arbitrarily far from a barely-sampled axis just to hit exact area
      // parity (audit finding A2, 2026-09-01).
      thr2 *= std::min(4.0, std::sqrt(l2 / l1));
    }
    if (m2 > thr2) return false;
  } else {
    const double dis_to_center = d_center.squaredNorm();
    const double range_dis = std::sqrt(std::max(0.0, dis_to_center - r * r));
    if (range_dis > opts_->max_radius * radius_) return false;
  }

  // T0-D's "log before the gate" guardrail: mark candidacy (this point
  // cleared the purely-geometric test) BEFORE any T3-0e ablation drop
  // below, so a dropped correspondence still reaches corr.csv (as
  // dropped_by_ablation=1) instead of vanishing from the population
  // entirely -- code-review fix, 2026-08-31 (the original version
  // returned false above *is_candidate=true, so drops were unlogged and
  // any NIS/count comparison against baseline was over a silently
  // truncated population).
  if (is_candidate) *is_candidate = true;

  // T3-0e: drop this whole PLANE from the residual (not just this one
  // correspondence -- the decision is per-plane, evaluated identically
  // for every correspondence that hits it) if the coverage-anisotropy
  // ablation says so. See VoxelOpts::occ_aniso_drop_mode's doc comment.
  if (opts_->occ_aniso_drop_mode == "top") {
    const double a = occupancyAnisotropy();
    // Code-review fix: a<3 occupied cells) -> -1.0 sentinel, which used to
    // silently survive "> threshold" (never true), systematically
    // PROTECTING planes whose normals are still rotating/unstable -- the
    // planes T3-0e most wants to be able to drop. Explicit policy instead:
    // occ_aniso_undefined_as_top controls whether an undefined value
    // counts as top-decile (drop) or not (keep); default false (keep,
    // conservative) but now a deliberate, documented, logged choice, not
    // an accident of the sentinel's sign.
    const bool undefined = (a < 0.0);
    if (undefined ? opts_->occ_aniso_undefined_as_top : (a > opts_->occ_aniso_drop_threshold)) {
      if (dropped_by_ablation) *dropped_by_ablation = true;
      return false;
    }
  } else if (opts_->occ_aniso_drop_mode == "random") {
    // Code-review fix: hash on occ_anchor_center_ (only reassigned on a
    // >10-degree occupancy reset, see updateOccupancy()) instead of
    // plane_.center (reassigned on EVERY refitDebiased()/update() call --
    // the original version redrew this "random" decision at scan rate,
    // ~10Hz, instead of once per plane; see the bug ledger entry this
    // fixes). Still only APPROXIMATELY stable (a rare reset changes the
    // draw), not a first-class per-plane identity, but no longer redrawn
    // every single fit.
    // POPULATION MATCH (T3-0e blocking fix, 2026-09-01). The "top" arm can
    // only ever select planes whose occ_aniso is DEFINED -- >= 3 occupied
    // cells -- unless occ_aniso_undefined_as_top says otherwise. Drawing
    // "random" from every plane makes arm 3 a draw from a strictly larger
    // population than arm 2, so the two arms remove different KINDS of
    // plane and the comparison has no control. Gate eligibility on the same
    // predicate "top" uses, so the two arms stay matched under either
    // setting of occ_aniso_undefined_as_top rather than only under one.
    const bool eligible = opts_->occ_aniso_undefined_as_top
                        || (occupancyAnisotropy() >= 0.0);
    if (eligible) {
      std::size_t h = std::hash<double>{}(occ_anchor_center_.x() * 1e6 + opts_->occ_aniso_drop_seed);
      h ^= std::hash<double>{}(occ_anchor_center_.y() * 1e6 + opts_->occ_aniso_drop_seed * 2.0) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= std::hash<double>{}(occ_anchor_center_.z() * 1e6 + opts_->occ_aniso_drop_seed * 3.0) + 0x9e3779b9 + (h << 6) + (h >> 2);
      const double u01 = static_cast<double>(h % 1000000ULL) / 1000000.0;
      if (u01 < opts_->occ_aniso_drop_fraction) {
        if (dropped_by_ablation) *dropped_by_ablation = true;
        return false;
      }
    }
  }

  J_nq(0, 0) = d_center.dot(y_normal_);
  J_nq(0, 1) = d_center.dot(x_normal_);
  J_nq(0, 2) = 1.0;

  // sigma_diag_squared: sensor + (optionally) pose VARIANCE. See
  // VoxelOpts::pose_cov_in_sigma's docs for the double-counting tradeoff
  // this toggles.
  sigma_diag_squared = n.dot(sensor_cov * n);
  if (opts_->pose_cov_in_sigma)
    sigma_diag_squared += n.dot(pose_cov * n);
  if (!std::isfinite(sigma_diag_squared) || sigma_diag_squared <= 0.0) return false;
  if (sigma_diag_squared < 1e-6) sigma_diag_squared = 1e-6;

  plane_var_term = (J_nq * plane_var_ * J_nq.transpose()).value();
  const double sigma_gate_squared = sigma_diag_squared + plane_var_term;
  if (!std::isfinite(sigma_gate_squared) || sigma_gate_squared <= 0.0) return false;

  return r * r <= opts_->sigma_num_squared * sigma_gate_squared;
}

bool VoxelPlane::computeResidual(const WorldPointCov& pt, Residual& res, int scan_id) const
{
  if (!is_plane_) return false;

  double r = 0.0, sigma_diag_squared = 0.0, plane_var_term = 0.0;
  Eigen::Matrix<double, 1, 3> J_nq = Eigen::Matrix<double, 1, 3>::Zero();
  bool is_candidate = false;
  bool dropped_by_ablation = false;
  const bool accepted = gate(pt.point, pt.sensor_cov, pt.pose_cov, r, sigma_diag_squared,
                              plane_var_term, J_nq, &is_candidate, &dropped_by_ablation);

  if (opts_->log_consistency_corr_en && is_candidate && scan_id >= 0) {
    const bool cov = opts_->log_consistency_covariates_en;
    double s_sensor = -1.0, s_tilt = -1.0, s_d = -1.0, s_pose = -1.0, aniso = -1.0, lambda0 = -1.0;
    double occ_aniso = -1.0;
    int n = -1, j = -1, occ_cells = -1;
    // T8-b: logged unconditionally on the covariate path so that an inert
    // switch and a null effect are distinguishable after the fact. A run
    // where plane_conf_factor is 1.0 everywhere did not test the hypothesis.
    double occ_var_u = -1.0, occ_var_v = -1.0, plane_conf_factor = -1.0;
    if (cov) {
      // Plane-level quantities (independent of THIS correspondence's own
      // gate() outcome) are always valid, even for a dropped-by-ablation
      // row -- log them regardless, so the drop's own population (which
      // planes, how anisotropic) is auditable.
      lambda0 = eigen_values_(0);
      occ_aniso = occupancyAnisotropy();
      occ_cells = occupiedCellCount();
      occ_var_u = occupancyVarU();
      occ_var_v = occupancyVarV();
      plane_conf_factor = planeConfFactor();
      n = points_size_;
      j = last_fit_j_;
      aniso = eigen_values_(2) / std::max(eigen_values_(1), 1e-12);
    }
    // s_sensor/s_tilt/s_d/s_pose/S/s_prior_pose all depend on J_nq/
    // sigma_diag_squared/plane_var_term, which gate() does NOT finish
    // computing when it returns early on a T3-0e drop (they're zero-
    // initialized above, not meaningful) -- skip computing/logging these
    // for a dropped row rather than logging garbage-looking zeros dressed
    // up as real numbers. S itself logs as the -1.0 sentinel for a
    // dropped row (code-review fix, 2026-08-31).
    double s_prior_pose = -1.0;
    double S = -1.0;
    if (!dropped_by_ablation) {
      if (cov) {
        const V3D& normal = plane_.normal;
        s_sensor = normal.dot(pt.sensor_cov * normal);
        s_pose   = normal.dot(pt.pose_cov * normal);
        // Exact split of plane_var_term = J_nq*plane_var_*J_nq^T: tilt
        // covers the [theta1,theta2] 2x2 sub-block, d covers plane_var_
        // (2,2) alone, and the tilt/d cross terms are divided evenly
        // between the two so s_tilt+s_d == plane_var_term exactly.
        const double tilt_only = J_nq(0) * J_nq(0) * plane_var_(0, 0)
                                + J_nq(1) * J_nq(1) * plane_var_(1, 1)
                                + 2.0 * J_nq(0) * J_nq(1) * plane_var_(0, 1);
        const double d_only = J_nq(2) * J_nq(2) * plane_var_(2, 2);
        const double cross = 2.0 * J_nq(2) * (J_nq(0) * plane_var_(0, 2) + J_nq(1) * plane_var_(1, 2));
        s_tilt = tilt_only + 0.5 * cross;
        s_d    = d_only + 0.5 * cross;
      }
      // H P- H^T: the frame's PRIOR pose uncertainty projected through this
      // correspondence's own Jacobian -- the textbook S = H P- H^T + R term
      // this codebase's batch-WLS IEKF has no other use for (see
      // WorldPointCov::prior_cov_rp's doc comment), computed here purely so
      // corr.csv's S column matches the register's definition, not folded
      // into res.sigma_squared/the real gate() accept-reject decision
      // (unchanged, still measurement-noise-only, matching this codebase's
      // actual production weighting). point_cross_normal mirrors LioProc::
      // buildResiduals()'s own res.point_cross_normal formula exactly.
      const V3D point_cross_normal = pt.body_point.cross(pt.rot_transpose * plane_.normal);
      Eigen::Matrix<double, 1, 6> H_i;
      H_i << point_cross_normal.transpose(), plane_.normal.transpose();
      s_prior_pose = (H_i * pt.prior_cov_rp * H_i.transpose()).value();
      S = 1e-3 + sigma_diag_squared + plane_var_term + s_prior_pose;
    }
    debugLogConsistencyCorr(cov, scan_id, r, S, s_sensor, s_tilt, s_d, s_pose, n, j, aniso,
                             accepted ? 0 : 1, s_prior_pose, lambda0, occ_aniso, occ_cells,
                             dropped_by_ablation ? 1 : 0, occ_var_u, occ_var_v, plane_conf_factor);
  }

  if (!accepted) return false;

  res.r              = r;
  res.normal         = plane_.normal;
  res.sigma_squared  = 1e-3 + sigma_diag_squared;
  res.plane_id       = this;
  res.plane_var_term = plane_var_term;
  if (opts_->log_variance_shares_en) debugLogVarianceShare(sigma_diag_squared, plane_var_term);
  return true;
}

bool VoxelPlane::getVizInfo(PlaneVizInfo& info) const
{
  if (!is_plane_) return false;
  info.center       = plane_.center;
  info.normal       = plane_.normal;
  info.radius       = radius_;
  info.is_converged = isFull();
  return true;
}

void VoxelPlane::update(const std::vector<PointXYZCov>& points, int total_count,
                        const std::vector<double>* weights_in,
                        const RunningMoments* running,
                        const std::vector<double>* var_weights_in)
{
  plane_var_.setZero();
  covariance_.setZero();
  plane_ = {};
  is_plane_ = false;

  const int N = (int)points.size();
  points_size_ = (total_count >= 0) ? total_count : N;
  if (N < 3) return;

  // `weights`: supplied by the caller (VoxelNode always pre-bins whenever
  // convergence_mode=="always_update" -- see useBins()) or left null for
  // the plain unweighted path below.
  const std::vector<double>* weights = weights_in;
  double weight_sum = N;
  const bool use_weights = (weights_in != nullptr);

  // Independent weighting for the plane_var_ Jacobian loop below (see
  // T3-0c) -- falls back to `weights`/`weight_sum` when not supplied, so
  // a caller that only passes `weights` gets the pre-2026-08-30 behavior
  // (fit and uncertainty share one weighting) unchanged. weight_sum is
  // finalized by the fit block below, so var_weight_sum's fallback is
  // resolved after it, not here.
  const std::vector<double>* var_weights = var_weights_in ? var_weights_in : weights;
  const bool use_var_weights = (var_weights != nullptr);
  double var_weight_sum = 0.0;
  if (var_weights_in) {
    for (double w : *var_weights) var_weight_sum += w;
  }

  if (running) {
    // O(1) mean/covariance from pre-accumulated running sums (see
    // RunningMoments' docs) instead of the two O(N) loops below -- exact,
    // not an approximation (only ever passed for VoxelNode's equal-weight
    // points_ path, so this and use_weights are mutually exclusive).
    const V3D center_rel = running->sum_p / running->sum_w;
    covariance_   = running->sum_pp / running->sum_w - center_rel * center_rel.transpose();
    plane_.center = running->ref + center_rel;
  } else if (use_weights) {
    weight_sum = 0.0;
    for (double w : *weights) weight_sum += w;

    for (int i = 0; i < N; ++i)
      plane_.center += (*weights)[i] * points[i].point;
    plane_.center /= weight_sum;

    for (int i = 0; i < N; ++i) {
      V3D d = points[i].point - plane_.center;
      covariance_ += (*weights)[i] * (d * d.transpose());
    }
    covariance_ /= weight_sum;
    if (!var_weights_in) var_weight_sum = weight_sum;  // fallback resolved now that weight_sum is final
  } else {
    for (const auto& pt : points)
      plane_.center += pt.point;
    plane_.center /= N;

    for (const auto& pt : points) {
      V3D d = pt.point - plane_.center;
      covariance_ += d * d.transpose();
    }
    covariance_ /= N;
  }

  Eigen::SelfAdjointEigenSolver<M3D> solver(covariance_);
  if (solver.info() != Eigen::Success) return;

  eigen_values_ = solver.eigenvalues();
  const M3D evecs = solver.eigenvectors();

  if (eigen_values_(1) < 1e-8 || eigen_values_(2) < 1e-8)
    return;

  if (opts_->sensor_noise_floor_eig0) {
    // Uses points[i].sensor_cov's isotropic (trace/3) proxy rather than
    // points[i].sensor_cov+points[i].pos_cov (combined): pose
    // uncertainty is common to every point in a frame
    // and reflects "how sure am I of my own pose", not per-point
    // measurement precision, and its range^2 lever-arm scaling can dwarf
    // true sensor noise for far points, which made an earlier cov-based
    // version of this floor reject almost everything (confirmed
    // empirically: median floor ~119 vs plane_threshold=0.0025, a ~5
    // order-of-magnitude mismatch).
    double sensor_floor = 0.0;
    for (int i = 0; i < N; ++i) {
      const double w = use_weights ? ((*weights)[i] / weight_sum) : (1.0 / N);
      sensor_floor += w * (points[i].sensor_cov.trace() / 3.0);
    }
    if (opts_->log_debug_en)
    {
      std::ostringstream dbg;
      dbg << "N=" << N << " raw_eig0=" << eigen_values_(0) << " sensor_floor=" << sensor_floor
          << " plane_threshold=" << opts_->plane_threshold;
      debugLogNoiseFloor(dbg.str());
    }
    eigen_values_(0) = std::max(eigen_values_(0), sensor_floor);
    // The floor alone already explains the observed "flatness" -- this
    // candidate can't be distinguished from non-planar, so reject rather
    // than risk denom1=eigen_values_(0)-eigen_values_(1) collapsing to
    // ~zero in the plane_var_ Jacobian below.
    if (eigen_values_(0) >= eigen_values_(1)) return;
  }

  // Acceptance is min-eigenvalue-only, matching FAST-LIVO2's init_plane()
  // exactly (see voxel_map.cpp there) -- an extra eigen0/eigen1 ratio check
  // this used to also require was confirmed (via a 4-run A/B on NTU VIRAL
  // eee_01) to reject ~90% of livo_recon's plane candidates that this
  // eigen0-only test accepts (44% vs 94% acceptance rate), starving the
  // map of usable planes and driving a 2.16m full-run ATE divergence
  // FAST-LIVO2 doesn't share; removing it (alone) cut that to 0.093m.
  is_plane_ = eigen_values_(0) < opts_->plane_threshold;
  if (!is_plane_) return;

  // NOTE: no denom1/denom2 near-degenerate-eigengap guard here, unlike
  // refitDebiased() below -- tried applying the same guard to this PCA
  // path too (this failure mode isn't theoretically unique to debiasing),
  // but it measurably changed PCA-mode's own output (eee_01 ATE 0.0260m
  // -> 0.0249m), i.e. it DOES trigger occasionally even in plain PCA.
  // PCA has long-validated production behavior with no evidence this
  // mechanism is a real problem for it (the debiased-mode blowup is
  // specifically because debiasing shrinks eig0 toward eig1/eig2
  // together, making a small gap far more common there) -- reverted here
  // to preserve PCA's exact original byte-identical behavior. Revisit
  // only if PCA mode is later found to have its own denom-blowup problem.
  const double denom1 = eigen_values_(0) - eigen_values_(1);
  const double denom2 = eigen_values_(0) - eigen_values_(2);

  plane_.normal = evecs.col(0).normalized();
  y_normal_     = evecs.col(1);
  x_normal_     = evecs.col(2);
  plane_.d      = -plane_.normal.dot(plane_.center);
  radius_       = (float)std::sqrt(eigen_values_(2));

  const double inv_N  = 1.0 / N;

  for (int i = 0; i < N; ++i) {
    const auto& pt = points[i];
    const V3D z  = pt.point - plane_.center;
    const double a0 = plane_.normal.dot(z);
    const double a1 = y_normal_.dot(z);
    const double a2 = x_normal_.dot(z);

    // Each point's true contribution to d(mean)/d(point_i) and
    // d(covariance)/d(point_i) is w_i/weight_sum under the weighted
    // mean/covariance computed above -- inv_N (equal 1/N weighting) is
    // only correct in the unweighted path. Using inv_N unconditionally
    // here when weighting is on would over-count dense/high-weight
    // entries and under-count sparse/low-weight ones in this
    // fit-uncertainty propagation. `var_weights` (T3-0c) deliberately
    // decouples this from the fit's own weights when the caller supplies
    // it separately -- that makes this no longer the exact derivative of
    // THIS fit, but an intentional independent uncertainty-scaling
    // experiment (T3's actual proposal: reweight the uncertainty without
    // touching the fitted normal/center/covariance at all).
    const double w = use_var_weights ? ((*var_weights)[i] / var_weight_sum) : inv_N;

    // Minimal 3x3 Jmin, the direct equivalent of T * J_old (see
    // plane_var_'s docs in voxelplane.h) -- row0/row1 are
    // y_normal_^T*J_n / x_normal_^T*J_n (J_n's output is confined to
    // span{y_normal_,x_normal_}, so these pick out its two components
    // exactly), row2 is -normal_^T*(w*I) (the center-perturbation block's
    // projection onto d).
    Eigen::Matrix<double, 1, 3> row_theta1 =
        (w / denom1) * (a1 * plane_.normal.transpose() + a0 * y_normal_.transpose());
    Eigen::Matrix<double, 1, 3> row_theta2 =
        (w / denom2) * (a2 * plane_.normal.transpose() + a0 * x_normal_.transpose());
    Eigen::Matrix<double, 1, 3> row_d = -w * plane_.normal.transpose();

    M3D Jmin;
    Jmin.row(0) = row_theta1;
    Jmin.row(1) = row_theta2;
    Jmin.row(2) = row_d;

    // "combined" (default, unchanged behavior): full sensor+pose noise
    // budget, same spirit as always. "sensor_only" (new ablation, see
    // VoxelOpts::plane_fit_pose_cov_mode's docs): excludes pos_cov from
    // this fit-uncertainty propagation entirely, testing the same
    // exclusion idea already validated for gate()'s sigma_diag_squared
    // (pose_cov_in_sigma=false) but applied to plane_var_ instead -- a
    // DIFFERENT quantity, unconditionally combined here regardless of
    // pose_cov_in_sigma. Unlike refitDebiased()'s fit-BIAS correction
    // (which must not naively combine sensor+pose, see that function),
    // this term is a residual-weighting VARIANCE, where combining is
    // always at least conservative -- "sensor_only" is a genuine
    // ablation, not a correctness fix.
    const M3D residual_cov = (opts_->plane_fit_pose_cov_mode == "sensor_only")
        ? pt.sensor_cov : M3D(pt.sensor_cov + pt.pos_cov);
    plane_var_.noalias() += Jmin * residual_cov * Jmin.transpose();
  }

  last_fit_j_ = use_weights ? N : 0;

  // T1: refitDebiased() rejects a fit whose eigengap denominators are within
  // eps of zero; this path had no such guard and would divide plane_var_ by
  // an arbitrarily small denom1. Same eps, same counter, same rejection --
  // so that the two arms are finally doing the same thing here. Off by
  // default (see the NOTE above -- this measurably changes PCA's own output
  // when on, which is the whole point of it being an ablatable switch now
  // instead of a permanent behavior change).
  if (opts_->plane_var_denom_floor_en) {
    const double eps_denom = std::max(1e-8, 0.1 * opts_->plane_threshold);
    if (std::fabs(denom1) < eps_denom || std::fabs(denom2) < eps_denom) {
      is_plane_ = false;
      plane_var_.setZero();
      g_denom_rejected_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  }

  if (opts_->log_variance_shares_en) {
    double n_eff = N;
    if (use_var_weights) {
      double sum_sq = 0.0;
      for (double w : *var_weights) sum_sq += w * w;
      n_eff = sum_sq > 0.0 ? (var_weight_sum * var_weight_sum) / sum_sq : 0.0;
    }
    debugLogPlaneFitStats(points_size_, last_fit_j_, n_eff, plane_var_.trace());
  }

  // T3-0e: same occupancy update as addPoints()'s debiased path, using the
  // tangent frame just fitted above. `points` may be bin representatives
  // rather than raw points (see `running`/use_weights above) -- fine for
  // occupancy purposes, a bin rep already represents one occupied region.
  for (int i = 0; i < N; ++i) updateOccupancy(points[i].point);
  recomputeOccupancyCache();
  applyPlaneConfidence();
  // Moved BELOW applyPlaneConfidence so frame_stats' max_plane_var_trace
  // reports the covariance the residual actually uses, not the pre-inflation
  // one. With both terms off this is the same number as before, in the same
  // place in the sequence of writes.
  updateMaxPlaneVarTrace(plane_var_.trace());
}

void VoxelPlane::addPoints(const std::vector<PointXYZCov>& points, int total_count,
                           int distinct_frames, bool trust_sensor_noise)
{
  for (const auto& pt : points) {
    const V3D& p = pt.point;
    // Combined sensor+pose covariance -- for plane_var_'s accumulators
    // only (Scov_/V_/W_), which legitimately want the full residual-noise
    // budget, same as the PCA path's Jmin loop above. sensor_cov alone
    // (independent per point) is what M_debiased's own fit-bias
    // correction uses, via Scov_sensor_ below -- see refitDebiased() for
    // why the shared pose component can't be folded in there the same
    // naive way.
    // 2026-08-24 pass4 diagnostic finding: plane_var_ blowups (traces up
    // to ~965,882 seen live) persist even when the denom1/denom2 guard
    // below passes comfortably (e.g. denom1=-0.003, eps=0.00025) -- the
    // real driver is pos_cov's own magnitude, not the eigengap. pos_cov
    // (poseCovAt()'s range^2 lever-arm term) is documented elsewhere in
    // this codebase to "dwarf true sensor noise for far points"
    // (VoxelOpts::pose_cov_in_sigma's docs, which is why THAT flag
    // defaults false) -- every blown-up sample here was F=1, N=4-6: a
    // fresh, barely-initialized voxel whose Scov_/V_/W_ accumulators are
    // raw sums of only a handful of points, not yet averaged down, so one
    // large pos_cov contribution dominates. Respect the same
    // plane_fit_pose_cov_mode ablation used for the PCA path's Jmin loop
    // here too (deviating from the original spec, which scoped it
    // PCA-only, based on this evidence -- see
    // docs/debiased_voxel_plane_fit_2026aug24.md).
    const M3D combined_cov = (opts_->plane_fit_pose_cov_mode == "sensor_only")
        ? pt.sensor_cov : M3D(pt.sensor_cov + pt.pos_cov);
    N_acc_ += 1.0;
    Sp_    += p;
    Spp_.noalias() += p * p.transpose();
    Scov_  += combined_cov;
    // Only fold this point's sensor_cov into the SUBTRACTED correction
    // (Scov_sensor_/sum_sensor_var_) when it comes from a trusted,
    // independent-viewpoint source -- see addPoints()'s trust_sensor_noise
    // docs. Bootstrap points still contribute to Sp_/Spp_/Scov_ above (the
    // fit itself, and plane_var_'s combined-cov accumulators, which stay
    // conservative either way since they only ever ADD uncertainty) --
    // just not to the bias-correction term that gets subtracted.
    if (trust_sensor_noise) {
      Scov_sensor_ += pt.sensor_cov;
      sum_sensor_var_ += pt.sensor_cov.trace() / 3.0;
    }
    for (int a = 0; a < 3; ++a) {
      W_[a] += p(a) * combined_cov;
      for (int b = a; b < 3; ++b) {
        const M3D term = (p(a) * p(b)) * combined_cov;
        V_[a][b] += term;
        if (b != a) V_[b][a] += term;
      }
    }
  }
  points_size_ = (total_count >= 0) ? total_count : (int)N_acc_;
  if (distinct_frames > 0) distinct_frames_ = distinct_frames;
  refitDebiased();

  // T3-0e: occupancy uses whatever tangent frame refitDebiased() just left
  // (is_plane_ may still be false on early/rejected fits -- updateOccupancy()
  // itself no-ops until an anchor exists, so this is safe to call
  // unconditionally).
  if (is_plane_) {
    for (const auto& pt : points) updateOccupancy(pt.point);
    recomputeOccupancyCache();
    applyPlaneConfidence();
    // refitDebiased() already recorded the pre-inflation trace; this second
    // call (no-op when both T8-b terms are off) updates it to the
    // post-inflation value actually used downstream, same reasoning as
    // update()'s call site above.
    updateMaxPlaneVarTrace(plane_var_.trace());
  }
}

void VoxelPlane::refitDebiased()
{
  plane_var_.setZero();
  covariance_.setZero();
  plane_ = {};
  is_plane_ = false;
  // Bug ledger 2026-08-31: this was never set in debiased mode, leaving
  // the J column of corr.csv stale/zero on every debiased run. Reset here
  // and set to N_acc_ (this fit's effective point count) only on a
  // committed success below, mirroring update()'s PCA-path convention
  // (last_fit_j_ = use_weights ? N : 0).
  last_fit_j_ = 0;

  const double N = N_acc_;
  if (N < 3) return;

  const V3D mean = Sp_ / N;
  // M_debiased: noise-corrected structure tensor -- see
  // docs/debiased_voxel_plane_fit_2026aug24.md. covariance_ is reused here
  // (same field PCA mode fills) purely for getVizInfo()/diagnostics parity.
  //
  // Sensor noise (independent per point) is subtracted in full, every
  // time -- Scov_sensor_/N. Pose noise (Scov_ - Scov_sensor_, SHARED
  // across every point within one frame, not independent) is subtracted
  // with a (F-1)/F shrinkage factor, F = distinct_frames_: a shared
  // per-frame shift cancels out of a SINGLE frame's own scatter entirely
  // (subtracting it in full there is a pure over-correction -- confirmed
  // numerically, drives eig0 negative), and only becomes a genuine
  // between-frame variance contribution as more independent frames
  // accumulate (F=1 -> 0 correction, F->inf -> full correction). See the
  // doc file's "single-frame over-subtraction" section for the derivation
  // and numeric verification.
  const M3D Scov_pose = Scov_ - Scov_sensor_;
  const double pose_shrink = (distinct_frames_ > 1)
      ? (distinct_frames_ - 1.0) / distinct_frames_ : 0.0;
  covariance_ = Spp_ / N - mean * mean.transpose()
              - Scov_sensor_ / N - pose_shrink * (Scov_pose / N);

  Eigen::SelfAdjointEigenSolver<M3D> solver(covariance_);
  if (solver.info() != Eigen::Success) return;

  eigen_values_ = solver.eigenvalues();
  const M3D evecs = solver.eigenvectors();

  if (eigen_values_(1) < 1e-8 || eigen_values_(2) < 1e-8) return;

  // A population covariance's eigenvalues can never be negative -- a
  // negative SAMPLE eigenvalue here is definitionally sampling noise (the
  // "smallest of three correlated noisy quantities" selection is a biased
  // estimator of the true smallest eigenvalue, especially at low N/F), not
  // evidence the true scatter is negative. Clamping (rather than the old
  // reject-on-any-negative behavior) lets a mildly-negative result --
  // exactly what a genuinely flat surface's corrected estimate looks like
  // some fraction of the time -- correctly read as "flat" instead of being
  // thrown out. Confirmed empirically 2026-08-24: rejected candidates at
  // N=8-24, F=2 had raw (uncorrected) eig0 of 1e-8 to 1e-4 -- genuinely
  // near-flat -- pushed to ~-0.0002 by a sensor-noise correction of
  // comparable magnitude to its own sampling noise, not by a real
  // over-subtraction. See docs/debiased_voxel_plane_fit_2026aug24.md.
  // Guarding against the SEPARATE, larger-magnitude over-subtraction
  // failure mode (correlated noise from non-independent samples) is now
  // handled at the source, via trust_sensor_noise excluding bootstrap
  // points from the correction entirely, rather than by this clamp.
  eigen_values_(0) = std::max(eigen_values_(0), 0.0);

  if (opts_->sensor_noise_floor_eig0) {
    const double sensor_floor = sum_sensor_var_ / N;
    if (opts_->log_debug_en) {
      std::ostringstream dbg;
      dbg << "[debiased] N=" << N << " raw_eig0=" << eigen_values_(0)
          << " sensor_floor=" << sensor_floor << " plane_threshold=" << opts_->plane_threshold;
      debugLogNoiseFloor(dbg.str());
    }
    eigen_values_(0) = std::max(eigen_values_(0), sensor_floor);
    if (eigen_values_(0) >= eigen_values_(1)) return;
  }

  // Non-finite is still a hard reject (NaN/Inf from a degenerate solve) --
  // the negative case is handled by the clamp above now.
  if (!std::isfinite(eigen_values_(0))) return;

  is_plane_ = eigen_values_(0) < opts_->plane_threshold;
  if (!is_plane_) return;

  const double denom1 = eigen_values_(0) - eigen_values_(1);
  const double denom2 = eigen_values_(0) - eigen_values_(2);
  // Near-degenerate-eigengap guard, debiased-mode-only (see update()'s
  // comment on why this was NOT kept for PCA mode -- it measurably changed
  // PCA's own output there, unlike here where it's the confirmed fix for
  // the plane_var_ blowup (traces of 13.998/18.541 seen for a 9-point
  // debiased fit at eig0~2.3e-7): debiasing shrinks eig0 (and often
  // eig1/eig2 together, since the same noise floor gets subtracted from
  // all three), making a small denom1/denom2 far more common than in
  // plain PCA.
  const double eps_denom = std::max(1e-8, 0.1 * opts_->plane_threshold);
  const bool denom_rejected = std::fabs(denom1) < eps_denom || std::fabs(denom2) < eps_denom;
  if (opts_->log_debug_en) {
    std::ostringstream dbg;
    dbg << "[debiased_denom] N=" << N << " F=" << distinct_frames_
        << " eig0=" << eigen_values_(0) << " eig1=" << eigen_values_(1)
        << " eig2=" << eigen_values_(2) << " denom1=" << denom1 << " denom2=" << denom2
        << " eps=" << eps_denom << " rejected=" << (denom_rejected ? 1 : 0);
    debugLogNoiseFloor(dbg.str());
  }
  if (denom_rejected) {
    is_plane_ = false;
    g_denom_rejected_count.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  plane_.normal = evecs.col(0).normalized();
  y_normal_     = evecs.col(1);
  x_normal_     = evecs.col(2);
  plane_.center = mean;
  plane_.d      = -plane_.normal.dot(plane_.center);
  radius_       = (float)std::sqrt(eigen_values_(2));

  const double inv_N2 = 1.0 / (N * N);
  M3D Vw[3][3], Ww[3];
  for (int a = 0; a < 3; ++a) {
    Ww[a] = W_[a] * inv_N2;
    for (int b = 0; b < 3; ++b) Vw[a][b] = V_[a][b] * inv_N2;
  }
  const M3D Scov_w = Scov_ * inv_N2;

  auto U = [&](int a, int b) -> M3D {
    return Vw[a][b] - mean(a) * Ww[b] - mean(b) * Ww[a] + mean(a) * mean(b) * Scov_w;
  };
  auto Wc = [&](int a) -> V3D {
    return (Ww[a] - mean(a) * Scov_w) * plane_.normal;
  };

  const M3D M1 = plane_.normal * y_normal_.transpose() + y_normal_ * plane_.normal.transpose();
  const M3D M2 = plane_.normal * x_normal_.transpose() + x_normal_ * plane_.normal.transpose();

  double s_t1t1 = 0.0, s_t2t2 = 0.0, s_t1t2 = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const M3D Uab = U(a, b);
      s_t1t1 += M1.col(a).dot(Uab * M1.col(b));
      s_t2t2 += M2.col(a).dot(Uab * M2.col(b));
      s_t1t2 += M1.col(a).dot(Uab * M2.col(b));
    }
  }

  double s_t1d = 0.0, s_t2d = 0.0;
  for (int a = 0; a < 3; ++a) {
    s_t1d += M1.col(a).dot(Wc(a));
    s_t2d += M2.col(a).dot(Wc(a));
  }

  plane_var_(0, 0) = s_t1t1 / (denom1 * denom1);
  plane_var_(1, 1) = s_t2t2 / (denom2 * denom2);
  plane_var_(0, 1) = plane_var_(1, 0) = s_t1t2 / (denom1 * denom2);
  plane_var_(2, 2) = plane_.normal.dot(Scov_w * plane_.normal);
  plane_var_(0, 2) = plane_var_(2, 0) = -s_t1d / denom1;
  plane_var_(1, 2) = plane_var_(2, 1) = -s_t2d / denom2;

  // Sanity ceiling on plane_var_ itself, not just the denom1/denom2 gap
  // that feeds it -- confirmed live (2026-08-24) that denom1/denom2 can sit
  // comfortably outside eps_denom (e.g. denom1=-0.003, eps=0.001) while
  // plane_var_'s trace still reaches ~965,882: a barely-initialized voxel
  // (F=1, N=4-6) has Scov_/V_/W_ built from only a handful of points, so
  // one large pos_cov contribution (documented elsewhere in this codebase
  // to "dwarf true sensor noise for far points", see pose_cov_in_sigma's
  // docs) dominates a still-small denominator's amplification. Rather than
  // trying to characterize every combination of small-N/large-pos_cov/
  // small-denom that can produce this, cap the actual output directly --
  // this is the quantity that widens gate()'s acceptance
  // (sigma_gate_squared = sigma_diag_squared + plane_var_term) and lets
  // garbage-magnitude residuals through, which is the direct divergence
  // mechanism. 1.0 is generous (orientation variance in [theta1,theta2] is
  // radians^2-scale and d's variance is meters^2-scale; both should be
  // small fractions for anything worth trusting as a residual source) --
  // reject (not clamp) so an under-converged voxel just waits for more
  // points/frames rather than silently supplying an overconfident-looking
  // but wrong uncertainty. addPoints() keeps accumulating regardless, so
  // this voxel can still become usable once F/N grow past this point.
  constexpr double kPlaneVarCeiling = 1.0;
  if (!plane_var_.allFinite() || plane_var_.trace() > kPlaneVarCeiling) {
    is_plane_ = false;
    if (opts_->log_debug_en) {
      std::ostringstream dbg;
      dbg << "[debiased_ceiling_reject] N=" << N << " F=" << distinct_frames_
          << " plane_var_trace=" << plane_var_.trace()
          << " denom1=" << denom1 << " denom2=" << denom2;
      debugLogNoiseFloor(dbg.str());
    }
    return;
  }

  last_fit_j_ = (int)N;
  updateMaxPlaneVarTrace(plane_var_.trace());

  if (opts_->log_debug_en) {
    std::ostringstream dbg;
    dbg << "[debiased_fit] N=" << N << " F=" << distinct_frames_
        << " eig0=" << eigen_values_(0) << " eig1=" << eigen_values_(1)
        << " eig2=" << eigen_values_(2) << " plane_var_trace=" << plane_var_.trace()
        << " normal=[" << plane_.normal.transpose() << "]";
    debugLogNoiseFloor(dbg.str());
  }
}

// T3-0e (2026-08-31): 8x8 tangent-frame occupancy bitmask. Cell size
// opts_->voxel_size/8, grid spans [-4,4) cells each axis (i.e. +/-
// voxel_size/2 centered on the anchor). The basis is ANCHORED (frozen at
// the first call after a successful fit) rather than re-derived from the
// current x_normal_/y_normal_ every call, for two reasons: (1) a tangent
// frame that shifts slightly every refit would make "which cell is this"
// answer differently for the same physical point across calls, corrupting
// the occupancy pattern instead of accumulating it; (2) it matches T8-b's
// own card, which anchors for the same reason and resets only when the
// normal has rotated >~10 degrees (a reset reads as low coverage, which
// inflates uncertainty -- the safe direction, not a correctness bug).
void VoxelPlane::updateOccupancy(const V3D& world_point)
{
  if (!is_plane_) return;

  constexpr double kResetCosThreshold = 0.9848;  // cos(10 deg)
  if (!occ_anchored_ || occ_anchor_normal_.dot(plane_.normal) < kResetCosThreshold) {
    occ_anchor_normal_   = plane_.normal;
    occ_anchor_x_normal_ = x_normal_;
    occ_anchor_y_normal_ = y_normal_;
    occ_anchor_center_   = plane_.center;
    occupancy_bitmask_   = 0;
    occ_anchored_ = true;
  }

  const double cell_size = opts_->voxel_size / 8.0;
  const V3D d = world_point - occ_anchor_center_;
  const double u = d.dot(occ_anchor_x_normal_);
  const double v = d.dot(occ_anchor_y_normal_);
  const int cu = static_cast<int>(std::floor(u / cell_size)) + 4;
  const int cv = static_cast<int>(std::floor(v / cell_size)) + 4;
  if (cu < 0 || cu >= 8 || cv < 0 || cv >= 8) return;  // outside the tracked window -- not an error, just untracked
  occupancy_bitmask_ |= (uint64_t(1) << (cu * 8 + cv));
}

// Code-review fix, 2026-08-31: recomputes both cached_occ_cells_ and
// cached_occ_aniso_ ONCE per fit (called from update()/addPoints() right
// after the occupancy-update loop), rather than on every
// occupancyAnisotropy()/occupiedCellCount() call -- see those two
// methods' header doc comment. Logic is otherwise unchanged from the
// original per-call implementation.
void VoxelPlane::recomputeOccupancyCache()
{
  cached_occ_cells_ = __builtin_popcountll(occupancy_bitmask_);

  const double cell_size = opts_->voxel_size / 8.0;
  std::vector<std::pair<double, double>> centers;
  for (int cu = 0; cu < 8; ++cu) {
    for (int cv = 0; cv < 8; ++cv) {
      if (occupancy_bitmask_ & (uint64_t(1) << (cu * 8 + cv))) {
        // cell CENTER in tangent (u,v) -- inverse of updateOccupancy()'s
        // floor((u/cell_size)) + 4 binning.
        centers.emplace_back((cu - 4 + 0.5) * cell_size, (cv - 4 + 0.5) * cell_size);
      }
    }
  }
  if (centers.size() < 3) {
    cached_occ_aniso_ = -1.0;
    cached_occ_var_u_ = 0.0;
    cached_occ_var_v_ = 0.0;
    return;
  }
  double mean_u = 0.0, mean_v = 0.0;
  for (const auto& c : centers) { mean_u += c.first; mean_v += c.second; }
  mean_u /= centers.size();
  mean_v /= centers.size();
  double var_u = 0.0, var_v = 0.0, cov_uv = 0.0;
  for (const auto& c : centers) {
    const double du = c.first - mean_u, dv = c.second - mean_v;
    var_u += du * du;
    var_v += dv * dv;
    cov_uv += du * dv;
  }
  const double n = static_cast<double>(centers.size());
  var_u /= n; var_v /= n; cov_uv /= n;
  cached_occ_var_u_ = var_u;
  cached_occ_var_v_ = var_v;
  // 2x2 symmetric eigenvalues, closed form.
  const double tr = var_u + var_v;
  const double det = var_u * var_v - cov_uv * cov_uv;
  const double disc = std::sqrt(std::max(0.0, tr * tr / 4.0 - det));
  const double lambda1 = tr / 2.0 + disc;
  const double lambda2 = tr / 2.0 - disc;
  constexpr double eps = 1e-12;
  cached_occ_aniso_ = lambda1 / std::max(lambda2, eps);
}

// T8-b (2026-09-01). The two plane-confidence terms livo_recon did not have.
//
// Both are multiplicative on plane_var_ and both are the IDENTITY when their
// switch is off. The coverage term is additionally the identity when coverage
// is isotropic (var_u == var_v), which is what makes a null result on this
// switch distinguishable from an inert switch: if occ_cells is healthy and
// plane_conf_factor is still 1.0 across a run, the geometry -- not the code --
// is what made it inert.
//
// Applied AFTER the fit and AFTER recomputeOccupancyCache(), never inside the
// fit itself, so the eigen-decomposition, the plane_threshold acceptance test
// and (on the debiased path) the ceiling rejection all still see the RAW fit.
// A switch that could change which planes exist would not be ablatable.
void VoxelPlane::applyPlaneConfidence()
{
  last_plane_conf_factor_ = 1.0;
  if (!is_plane_) return;
  if (!opts_->plane_conf_redundancy_en && !opts_->plane_conf_coverage_en) return;

  const double trace_before = plane_var_.trace();

  // ---- redundancy -------------------------------------------------------
  // plane_var_ ~ 1/N. The occupied-cell count is a density-INDEPENDENT
  // measure of how much distinct surface the fit actually saw, so N/cells is
  // the redundancy factor the debiased path never removes. Capped, because on
  // a dense return this ratio is unbounded and an unbounded inflation would
  // silently become a gate (every residual would pass sigma_num).
  if (opts_->plane_conf_redundancy_en) {
    const double n_raw = (last_fit_j_ > 0) ? static_cast<double>(last_fit_j_)
                                           : static_cast<double>(points_size_);
    if (cached_occ_cells_ >= 3 && n_raw > 0.0) {
      double f = n_raw / static_cast<double>(cached_occ_cells_);
      if (f < 1.0) f = 1.0;
      // AUDIT A3: NOT comparable across arms. On the pca path (without
      // use_bins, which defaults false) last_fit_j_ is 0 so n_raw falls
      // through to points_size_, capped at max_points (~50). On the debiased
      // path last_fit_j_ is N_acc_, uncapped and growing for the voxel's
      // whole life. The inflation ratio therefore has a hard ceiling of
      // ~max_points/64 on one arm and none on the other -- defensible as a
      // MODEL (debiased really does accumulate unboundedly) but it means
      // this factor is not comparable ACROSS arms. Compare within an arm
      // only; plane_conf_factor is logged so the number is visible rather
      // than hidden behind the cap.
      if (f > opts_->plane_conf_redundancy_cap) f = opts_->plane_conf_redundancy_cap;
      plane_var_ *= f;
    }
  }

  // ---- coverage ---------------------------------------------------------
  // plane_var_ is expressed over [theta1, theta2, d]. From gate():
  //   J_nq(0) = d_center . y_normal_   -> component 0's lever arm is along v
  //   J_nq(1) = d_center . x_normal_   -> component 1's lever arm is along u
  // A tilt is poorly determined when the plane was sampled thinly along the
  // direction that would have revealed it, so component 0 is inflated by how
  // thin the footprint is along v, and component 1by how thin it is along u.
  // Scaling as D * plane_var_ * D with D diagonal keeps the result symmetric
  // and positive semi-definite, which a naive per-entry scaling would not.
  //
  // AUDIT A1: the bitmask's (u, v) axes are a SNAPSHOT of (x_normal_,
  // y_normal_) taken at anchor time. Re-anchoring is triggered by normal
  // rotation only (see updateOccupancy()), so the in-plane axes -- the
  // eigenvectors of a 2x2 whose eigenvalues are nearly equal on a
  // well-covered plane -- can rotate or swap freely while the normal never
  // moves at all. Applying the term through a stale frame would attribute
  // cached_occ_var_u_/v_ to the wrong tangent component and inflate the
  // WELL-sampled axis instead of the thin one: not a weaker effect, a
  // sign-inverted one, on exactly the isotropic planes where it should have
  // been inert. Guard on the anchor frame still being aligned with the
  // CURRENT fit's own tangent basis (same 10-degree threshold
  // updateOccupancy() uses, for consistency) before trusting it.
  constexpr double kFrameAlignThreshold = 0.9848;  // cos(10 deg)
  const bool frame_fresh =
      occ_anchored_ &&
      std::fabs(occ_anchor_x_normal_.dot(x_normal_)) > kFrameAlignThreshold &&
      std::fabs(occ_anchor_y_normal_.dot(y_normal_)) > kFrameAlignThreshold;
  if (opts_->plane_conf_coverage_en && cached_occ_cells_ >= 3 && frame_fresh) {
    constexpr double eps = 1e-12;
    const double vu = std::max(cached_occ_var_u_, eps);
    const double vv = std::max(cached_occ_var_v_, eps);
    const double vref = std::max(vu, vv);
    const double b = opts_->plane_conf_coverage_beta;
    double f0 = std::pow(vref / vv, b);   // theta1: lever arm along v
    double f1 = std::pow(vref / vu, b);   // theta2: lever arm along u
    const double cap = opts_->plane_conf_coverage_cap;
    if (!std::isfinite(f0) || f0 > cap) f0 = cap;
    if (!std::isfinite(f1) || f1 > cap) f1 = cap;
    if (f0 < 1.0) f0 = 1.0;
    if (f1 < 1.0) f1 = 1.0;
    V3D d(std::sqrt(f0), std::sqrt(f1), 1.0);
    const M3D D = d.asDiagonal();
    plane_var_ = D * plane_var_ * D;
  }

  if (!plane_var_.allFinite()) {
    // An inflation that produced a non-finite covariance is a bug, not a
    // result. Fail loudly rather than letting it reach the residual.
    is_plane_ = false;
    last_plane_conf_factor_ = 0.0;
    return;
  }
  last_plane_conf_factor_ = (trace_before > 0.0) ? (plane_var_.trace() / trace_before) : 1.0;
}

}  // namespace livo_recon
