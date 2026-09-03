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
#include <cstdint>

namespace livo_recon
{

namespace
{
// T0-F-2b: see voxelplane.h's voxelPlaneFrameStats{Reset,Read}() doc
// comment. atomic<double> has no fetch_max pre-C++20 -- CAS loop instead.
std::atomic<int> g_denom_rejected_count{0};
// Engagement counter for the information plane model.  A run configured with
// plane_var_mode = "information" that never increments this built no plane
// under the model and is an INERT cell, not a null result -- the same
// discipline SM-1 established for the spline flags, applied at the point the
// mechanism actually runs.
std::atomic<long> g_info_fits{0};
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

// History (70-79): see docs/livo_recon_changelog.md#src-lio-voxelplane.cpp-70
// D-1's covariates.  corr.csv logged everything about a correspondence
// EXCEPT the two things the information model is a statement about: the
// query's own lever arm, and which plane it hit.  Without the lever arm the
// leverage a^T I^-1 a cannot be reconstructed after the fact; without a
// plane id the per-plane information cannot be grouped at all.  Bundled in a
// struct rather than threaded as eight more positional arguments through a
// twenty-argument call.
struct CorrInfoCols {
  double   a0 = 0.0, a1 = 0.0;   // J_nq(0), J_nq(1); J_nq(2) is 1 by construction
  uint64_t plane_id = 0;         // groups correspondences by plane WITHIN one run
  double   roughness = -1.0;     // lambda0 after noise subtraction, m^2
  double   sigma_bar2 = -1.0;    // roughness + mean projected measurement noise
  double   n_eff = -1.0;         // design-effect corrected sample size
  double   n_raw = -1.0;         // raw return count, for sizing the correction
  double   rho = -1.0;           // intra-scan correlation used for the design effect
  int      frames = -1;          // distinct frames this plane was built from
  double   floor_term = -1.0;    // weightFloor()'s own contribution to S, separable from sigma_diag_squared/plane_var_term
  double   lambda1 = -1.0;       // second eigenvalue (with lambda0 already logged, completes I per plane)
  double   lambda2 = -1.0;       // third eigenvalue
  int      info_path = -1;       // P5: 1=exact directional weighting, 0=equal-weight fallback, -1=n/a (not information_directional)
};

void debugLogConsistencyCorr(bool with_covariates, int scan_id, double nu, double S,
                              double s_sensor, double s_plane_tilt, double s_plane_d,
                              double s_pose, int n, int j, double aniso, int gated,
                              double s_prior_pose, double lambda0, double occ_aniso, int occ_cells,
                              int dropped_by_ablation, double occ_var_u, double occ_var_v,
                              double plane_conf_factor, const CorrInfoCols& info)
{
  // History (87-94): see docs/livo_recon_changelog.md#src-lio-voxelplane.cpp-87
  static std::mutex mtx;
  static std::vector<char> buf(1 << 20);
  static std::ofstream ofs;
  static bool first_call = true;
  std::lock_guard<std::mutex> lock(mtx);
  if (first_call) {
    ofs.rdbuf()->pubsetbuf(buf.data(), static_cast<std::streamsize>(buf.size()));
    ofs.open(debugLogPath("corr.csv"), std::ios::trunc);
    // History (103-111): see docs/livo_recon_changelog.md#src-lio-voxelplane.cpp-103
    ofs << "scan_id,nu,S,gated,dropped_by_ablation";
    // History (113-120): see docs/livo_recon_changelog.md#src-lio-voxelplane.cpp-113
    if (with_covariates) ofs << ",S_sensor,S_plane_tilt,S_plane_d,S_pose,S_prior_pose,N,J,aniso,lambda0,occ_aniso,occ_cells,occ_var_u,occ_var_v,plane_conf_factor"
                             << ",a0,a1,plane_id,roughness,sigma_bar2,n_eff,n_raw,rho,frames"
                             << ",floor_term,lambda1,lambda2,info_path";
    ofs << "\n";
  }
  first_call = false;
  ofs << scan_id << "," << nu << "," << S << "," << gated << "," << dropped_by_ablation;
  if (with_covariates)
    ofs << "," << s_sensor << "," << s_plane_tilt << "," << s_plane_d << "," << s_pose
        << "," << s_prior_pose << "," << n << "," << j << "," << aniso << "," << lambda0
        << "," << occ_aniso << "," << occ_cells
        << "," << occ_var_u << "," << occ_var_v << "," << plane_conf_factor
        << "," << info.a0 << "," << info.a1 << "," << info.plane_id
        << "," << info.roughness << "," << info.sigma_bar2
        << "," << info.n_eff << "," << info.n_raw << "," << info.rho
        << "," << info.frames
        << "," << info.floor_term << "," << info.lambda1 << "," << info.lambda2
        << "," << info.info_path;
  ofs << "\n";
}

// History (134-140): see docs/livo_recon_changelog.md#src-lio-voxelplane.cpp-134
struct CorrScanAccum {
  int scan_id = -1;
  long n_candidates = 0, n_accepted = 0, n_dropped = 0, n_nis_finite = 0;
  double sum_nis = 0.0, sum_nis2 = 0.0, sum_log_nis = 0.0, max_nis = 0.0;
  long n_share = 0; double sum_share = 0.0;
  // P2.  The variance decomposition, retained per scan instead of only
  // ever existing inside corr.csv's per-correspondence rows (0.7-1 GB/job,
  // deleted after scoring). floor_share/sensor_share/plane_share =
  // sum_{floor,sdiag,pvar}/sum_S answer "how much was each term ever
  // contributing", which a batch's ATE alone cannot.
  //
  // sum_prior_pose added after the smoke test found the original 3-term
  // decomposition summing to ~1.5% of sum_S, not 1: S actually has a FOURTH
  // term (s_prior_pose, "H P- H^T", see its own comment at the call site)
  // that the register's original P2 hunk didn't know about and doesn't sum
  // to 1 without. This is exactly the "fourth term entering S" case P2's
  // own verify text names as the most valuable possible outcome.
  double sum_S = 0.0, sum_floor = 0.0, sum_sdiag = 0.0, sum_pvar = 0.0;
  double sum_prior_pose = 0.0;
  int    n_S   = 0;
};
std::mutex g_corr_scan_mtx;
CorrScanAccum g_corr_scan;

void flushCorrScan(CorrScanAccum& a)
{
  if (a.scan_id < 0) return;
  static bool first = true;
  static std::ofstream ofs;
  if (first) {
    ofs.open(debugLogPath("corr_scan.csv"), std::ios::trunc);
    ofs << "scan_id,n_candidates,n_accepted,n_dropped,n_nis_finite,"
           "sum_nis,sum_nis2,sum_log_nis,max_nis,n_share,sum_share,"
           "n_S,sum_S,sum_floor,sum_sdiag,sum_pvar,sum_prior_pose\n";
    first = false;
  }
  ofs << a.scan_id << ',' << a.n_candidates << ',' << a.n_accepted << ','
      << a.n_dropped << ',' << a.n_nis_finite << ',' << a.sum_nis << ','
      << a.sum_nis2 << ',' << a.sum_log_nis << ',' << a.max_nis << ','
      << a.n_share << ',' << a.sum_share << ','
      << a.n_S << ',' << a.sum_S << ',' << a.sum_floor << ','
      << a.sum_sdiag << ',' << a.sum_pvar << ',' << a.sum_prior_pose << '\n';
  a = CorrScanAccum{};
}

// Called for EVERY candidate, always, regardless of the corr.csv stride.
// Cheap: a few adds under a lock, no I/O except once per scan.
void debugAccumConsistencyCorr(int scan_id, double nu, double S, int gated,
                               int dropped, double plane_share,
                               double floor_term, double sigma_diag_squared,
                               double plane_var_term, double s_prior_pose)
{
  std::lock_guard<std::mutex> lock(g_corr_scan_mtx);
  if (scan_id != g_corr_scan.scan_id) {
    flushCorrScan(g_corr_scan);
    g_corr_scan.scan_id = scan_id;
  }
  auto& a = g_corr_scan;
  a.n_candidates++;
  if (dropped) { a.n_dropped++; return; }
  if (gated == 0) {
    a.n_accepted++;
    if (S > 0.0 && std::isfinite(nu) && std::isfinite(S)) {
      const double nis = nu * nu / S;
      if (std::isfinite(nis) && nis > 0.0) {
        a.n_nis_finite++;
        a.sum_nis += nis;
        a.sum_nis2 += nis * nis;
        a.sum_log_nis += std::log(nis);
        if (nis > a.max_nis) a.max_nis = nis;
      }
    }
    if (plane_share >= 0.0 && std::isfinite(plane_share)) {
      a.n_share++; a.sum_share += plane_share;
    }
    if (S > 0.0 && std::isfinite(S)) {
      a.n_S++;  a.sum_S += S;  a.sum_floor += floor_term;
      a.sum_sdiag += sigma_diag_squared;  a.sum_pvar += plane_var_term;
      a.sum_prior_pose += s_prior_pose;
    }
  }
}

}  // namespace

// The final scan never sees a scan_id change, so it needs an explicit flush.
// Wired into VoxelMap's shutdown path -- see the call site added near this
// codebase's other end-of-run debug-log finalization. External linkage
// (unlike debugAccumConsistencyCorr()/flushCorrScan(), only ever called from
// this file) so a different translation unit can trigger the final flush.
void debugFlushConsistencyCorr()
{
  std::lock_guard<std::mutex> lock(g_corr_scan_mtx);
  flushCorrScan(g_corr_scan);
}

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

long voxelPlaneInformationFitCount()
{
  return g_info_fits.load(std::memory_order_relaxed);
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

// `body_dir` is the unit ray from the sensor to the point, in the SAME frame
// pt.body_point is expressed in.  Used only by weight_floor_mode "incidence",
// which needs the incidence angle; every other mode ignores it.  Pass a null
// vector to fall back to the isotropic form.
bool VoxelPlane::gate(const V3D& p, const M3D& sensor_cov, const M3D& pose_cov,
                      const V3D& body_dir, const V3D& body_normal,
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
    // History (259-272): see docs/livo_recon_changelog.md#src-lio-voxelplane.cpp-259
    if (opts_->plane_gate_mode == "ellipse_area_matched") {
      // History (274-278): see docs/livo_recon_changelog.md#src-lio-voxelplane.cpp-274
      thr2 *= std::min(4.0, std::sqrt(l2 / l1));
    }
    if (m2 > thr2) return false;
  } else {
    const double dis_to_center = d_center.squaredNorm();
    const double range_dis = std::sqrt(std::max(0.0, dis_to_center - r * r));
    if (range_dis > opts_->max_radius * radius_) return false;
  }

  // History (288-295): see docs/livo_recon_changelog.md#src-lio-voxelplane.cpp-288
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
    // History (318-333): see docs/livo_recon_changelog.md#src-lio-voxelplane.cpp-318
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

  // ONE floor, used by the gate below AND by res.sigma_squared in
  // computeResidual() -- see VoxelOpts::weight_floor_mode.  They used to
  // differ (the gate had none, the weight had a flat 1e-3), so the admission
  // threshold and the variance the admitted correspondence was then given
  // disagreed about S.
  const double floor_term = weightFloor(body_dir, body_normal, /*in_gate=*/true);
  const double sigma_gate_squared = floor_term + sigma_diag_squared + plane_var_term;
  if (!std::isfinite(sigma_gate_squared) || sigma_gate_squared <= 0.0) return false;

  return r * r <= opts_->sigma_num_squared * sigma_gate_squared;
}

// See VoxelOpts::weight_floor_mode for the derivation.  Returns the additive
// along-normal variance term, in m^2.
double VoxelPlane::weightFloor(const V3D& body_dir, const V3D& body_normal,
                               bool in_gate) const
{
  const std::string& m = opts_->weight_floor_mode;
  // The historical asymmetry, reproduced exactly and only on request -- see
  // VoxelOpts::weight_floor_mode's "legacy" note.
  if (m == "legacy")   return in_gate ? 0.0 : opts_->weight_floor_constant;
  if (m == "none")     return 0.0;
  if (m == "constant") return opts_->weight_floor_constant;
  // The unified model's floor: the plane's own measured surface roughness,
  // as a variance.  This is the term the ledger has been recording as
  // missing -- "lambda0 is a binary admission test only: surface roughness
  // never enters the residual variance".  It does now, per plane, measured,
  // instead of a literal chosen once for every plane in every scene.
  // roughness_ is only written under plane_var_mode = "information", which
  // VoxelMap::loadParameters() enforces, so a zero here means the plane has
  // not been fitted yet rather than that the mode is inert.
  if (m == "roughness") return roughness_;

  const double sr2 = opts_->weight_sigma_r2;
  if (m != "incidence") return sr2;                 // "sensor_range"

  // sigma_r^2 (cos^2 theta + k sin^2 theta).  cos theta is the angle between
  // the ray and the plane normal, both in the body frame.  If either vector
  // is degenerate the geometry is unavailable and we fall back to the k = 1
  // value, which is exactly "sensor_range" -- never to zero.
  const double dn = body_dir.norm(), nn = body_normal.norm();
  if (!(dn > 1e-9) || !(nn > 1e-9)) return sr2;
  double c = std::abs(body_dir.dot(body_normal) / (dn * nn));
  c = std::min(1.0, std::max(0.0, c));
  const double c2 = c * c;
  return sr2 * (c2 + opts_->weight_incidence_k * (1.0 - c2));
}

bool VoxelPlane::computeResidual(const WorldPointCov& pt, Residual& res, int scan_id) const
{
  if (!is_plane_) return false;

  double r = 0.0, sigma_diag_squared = 0.0, plane_var_term = 0.0;
  Eigen::Matrix<double, 1, 3> J_nq = Eigen::Matrix<double, 1, 3>::Zero();
  bool is_candidate = false;
  bool dropped_by_ablation = false;
  // Ray direction and plane normal in the body frame, for weight_floor_mode
  // "incidence".  pt.body_point is the return in the body frame and the
  // sensor origin is within centimetres of that frame's origin, so its
  // direction is the ray direction to well within the angular resolution
  // this is used at.  pt.rot_transpose is R_wb^T.
  const V3D body_normal = pt.rot_transpose * plane_.normal;
  const bool accepted = gate(pt.point, pt.sensor_cov, pt.pose_cov,
                              pt.body_point, body_normal,
                              r, sigma_diag_squared,
                              plane_var_term, J_nq, &is_candidate, &dropped_by_ablation);
  const double floor_term = weightFloor(pt.body_point, body_normal, /*in_gate=*/false);

  // Tier A (the per-scan accumulator below) is reachable on its own now.
  // It used to sit inside a guard that also gated Tier B's ~1 GB/job of
  // per-correspondence rows, so a sweep that could not afford Tier B lost
  // the cheap level statistics with it. See VoxelOpts::log_consistency_mode.
  if (opts_->logCorrScan() && is_candidate && scan_id >= 0) {
    const bool cov = opts_->logCorrCovariates();
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
    // History (403-409): see docs/livo_recon_changelog.md#src-lio-voxelplane.cpp-403
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
      // Deliberately NOT written as pt.body_point.cross(body_normal), even though
  // body_normal holds the same value.  Hoisting the product out of the
  // expression can change whether the compiler contracts it into an FMA, and
  // on this system a bit-level change is not harmless: T0-G measured up to
  // 18.88% ATE movement from a semantically inert 0.1% nudge.  Keeping the
  // original expression form leaves the "legacy" weight-floor mode a genuine
  // byte-identity control instead of one confounded by a refactor.
  const V3D point_cross_normal = pt.body_point.cross(pt.rot_transpose * plane_.normal);
      Eigen::Matrix<double, 1, 6> H_i;
      H_i << point_cross_normal.transpose(), plane_.normal.transpose();
      s_prior_pose = (H_i * pt.prior_cov_rp * H_i.transpose()).value();
      S = floor_term + sigma_diag_squared + plane_var_term + s_prior_pose;
    }

    // 14b: exact per-scan aggregates for EVERY candidate, regardless of the
    // corr.csv stride below -- level statistics (mean NIS, accept fraction,
    // dropped-by-ablation count) must never depend on which rows happened
    // to survive striding.
    const double plane_share = (cov && S > 0.0 && s_tilt >= 0.0 && s_d >= 0.0)
        ? (s_tilt + s_d) / S : -1.0;
    debugAccumConsistencyCorr(scan_id, r, S, accepted ? 0 : 1,
                              dropped_by_ablation ? 1 : 0, plane_share,
                              floor_term, sigma_diag_squared, plane_var_term,
                              s_prior_pose);

    // 14c: full per-correspondence rows only every Nth candidate. The
    // stride is PRIME on purpose -- LiDAR returns arrive in ring/azimuth
    // order, so a round stride (50, 64, 100) can alias with the beam count
    // (a 64-beam sensor sampled every 64th return is one ring, not a
    // sample of the scan). DISTRIBUTIONS (percentiles, decile cuts) are
    // the only thing that needs individual rows -- level statistics come
    // from corr_scan.csv above and are unaffected by the stride.
    if (opts_->logCorrRows()) {
      static std::atomic<uint64_t> corr_row_counter{0};
      const uint64_t k = corr_row_counter.fetch_add(1, std::memory_order_relaxed);
      const int stride = std::max(1, opts_->log_consistency_corr_stride);
      if ((k % static_cast<uint64_t>(stride)) == 0) {
        CorrInfoCols info;
      if (cov) {
        // The lever arm this correspondence actually presented to the plane.
        // gate() has already filled J_nq; J_nq(2) is 1 by construction and is
        // not logged.  These two columns are what let D-1 rebuild
        // I = sum a a^T / sigma^2 per plane and score the leverage
        // a^T I^-1 a against the incumbent covariates, from retained data,
        // with no estimator change.
        info.a0 = J_nq(0);
        info.a1 = J_nq(1);
        info.plane_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(this));
        info.roughness  = roughness_;
        info.sigma_bar2 = sigma_bar2_;
        info.n_eff      = info_n_eff_;
        info.n_raw      = info_n_raw_;
        info.rho        = info_rho_;
        info.frames     = distinct_frames_;
        info.info_path  = (opts_->plane_var_mode == "information_directional") ? info_path_ : -1;
        // Without this, S is logged but the floor's share of it is not
        // separable -- so "does the roughness floor dominate?" (W-1's own
        // question, asked again of a different floor) is unanswerable from
        // the log that exists to answer it.
        info.floor_term = floor_term;
        info.lambda1    = eigen_values_(1);
        info.lambda2    = eigen_values_(2);
      }
      debugLogConsistencyCorr(cov, scan_id, r, S, s_sensor, s_tilt, s_d, s_pose, n, j, aniso,
                                 accepted ? 0 : 1, s_prior_pose, lambda0, occ_aniso, occ_cells,
                                 dropped_by_ablation ? 1 : 0, occ_var_u, occ_var_v, plane_conf_factor,
                                 info);
      }
    }
  }

  if (!accepted) return false;

  res.r              = r;
  res.normal         = plane_.normal;
  res.sigma_squared  = floor_term + sigma_diag_squared;
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

  // History (513-518): see docs/livo_recon_changelog.md#src-lio-voxelplane.cpp-513
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

  // The information model needs the fit's mean point covariance, split so
  // the shared (pose) part is separable from the independent (sensor) part.
  // The debiased path keeps these persistently as Scov_/Scov_sensor_; the
  // pca path has no persistent accumulator, so form them here.  Cheap, and
  // write-only unless plane_var_mode is "information".
  // The two plane models are alternatives, so they are written as one
  // branch.  An earlier revision of this threaded `!info_mode` through the
  // Jacobian loop's own condition and through two later guards, which is
  // three places to get wrong and a loop that reads as if it might run.
  if (opts_->plane_var_mode == "information" ||
      opts_->plane_var_mode == "information_directional") {
    mean_cov_all_.setZero();
    mean_cov_sensor_.setZero();
    // P5.  Weighted moment accumulators for the directional path, in WORLD
    // coordinates so they are independent of the chart -- the chart
    // (y_normal_/x_normal_) is applied only when buildInformationCovariance()
    // projects these into H.
    const bool directional = (opts_->plane_var_mode == "information_directional");
    double Sw = 0.0; V3D Swd = V3D::Zero(); M3D Swdd = M3D::Zero();
    const V3D& n = plane_.normal;
    // Pass 1: mean covariances (as before).
    for (int i = 0; i < N; ++i) {
      const M3D Ci = points[i].sensor_cov + points[i].pos_cov;
      mean_cov_all_    += Ci;
      mean_cov_sensor_ += points[i].sensor_cov;
    }
    mean_cov_all_    /= N;
    mean_cov_sensor_ /= N;
    // Correction (2026-09-03, register-flagged defect in the first draft of
    // this hunk): lam0 here must be ROUGHNESS, not the raw smallest
    // eigenvalue. On the pca arm eigen_values_(0) still contains the mean
    // projected measurement noise (sigma_meas2), which n^T*Ci*n ALSO
    // contributes per point below -- using raw eigen_values_(0) as lam0
    // double-counts that noise and deflates every weight. roughness_ itself
    // is computed inside buildInformationCovariance(), which runs AFTER
    // this whole loop and has no points by then -- so it's recomputed here
    // identically (same formula, pre-debiased path) rather than restructured
    // into a third function, since N is small and this is O(1) extra work.
    const double sigma_meas2 = std::max(0.0, n.dot(mean_cov_all_ * n));
    const double lam0 = std::max(0.0, eigen_values_(0) - sigma_meas2);  // = roughness
    // Pass 2: per-point directional weights, now that lam0 is correct.
    for (int i = 0; i < N; ++i) {
      if (!directional) break;
      const M3D Ci = points[i].sensor_cov + points[i].pos_cov;
      const double s2 = n.dot(Ci * n) + lam0;
      if (!(s2 > 0.0) || !std::isfinite(s2)) continue;
      const double w = 1.0 / s2;
      const V3D d = points[i].point - plane_.center;
      Sw += w;  Swd += w * d;  Swdd += w * d * d.transpose();
    }
    if (directional && Sw > 0.0) {
      Sw_ = Sw; Swd_ = Swd; Swdd_ = Swdd; info_path_ = 1;  // 1 = exact
    } else {
      info_path_ = 0;  // 0 = equal-weight fallback
    }
    // eig0 here is the RAW out-of-plane second moment -- the noise has not
    // been subtracted on this path, so buildInformationCovariance() does it.
    buildInformationCovariance(static_cast<double>(points_size_), mean_cov_all_,
                               mean_cov_sensor_, /*eig0_already_debiased=*/false);
    last_fit_j_ = use_weights ? N : 0;
  } else {

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

  }  // end eigengap branch

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
    // History (734-755): see docs/livo_recon_changelog.md#src-lio-voxelplane.cpp-734
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
  // History (808-812): see docs/livo_recon_changelog.md#src-lio-voxelplane.cpp-808
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

  // History (848-864): see docs/livo_recon_changelog.md#src-lio-voxelplane.cpp-848
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
  const bool info_mode = (opts_->plane_var_mode == "information" ||
                          opts_->plane_var_mode == "information_directional");
  const double eps_denom = std::max(1e-8, 0.1 * opts_->plane_threshold);
  // Under the information model there is no eigengap denominator to guard:
  // plane_var_ is assembled directly in the (theta1, theta2, d) chart and
  // never divides by lambda0 - lambda_m.  This guard, plane_var_denom_floor_en
  // and kPlaneVarCeiling below are all artefacts of the incumbent form.
  const bool denom_rejected = !info_mode &&
      (std::fabs(denom1) < eps_denom || std::fabs(denom2) < eps_denom);
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

  if (info_mode) {
    // P5.  refitDebiased() runs from unweighted persistent sums after the
    // points themselves are gone -- it can never populate the exact
    // per-point weighted moments, so this is unconditionally the
    // equal-weight fallback. Reset explicitly rather than leaving whatever
    // update() set on an earlier fit of this same (persistent, reused)
    // VoxelPlane instance -- otherwise buildInformationCovariance() could
    // read a STALE info_path_==1 and use Sw_/Swd_/Swdd_ from a fit that is
    // no longer this one.
    info_path_ = 0;
    // Scov_ is the mean-scaled sum of combined per-point covariances and
    // Scov_sensor_ the independent (sensor-only) part -- exactly the split
    // buildInformationCovariance() needs, already accumulated by addPoints()
    // for the debiasing's own over-subtraction guard.  eig0 here IS the
    // debiased out-of-plane moment, so no further subtraction.
    buildInformationCovariance(N, M3D(Scov_ / N), M3D(Scov_sensor_ / N),
                               /*eig0_already_debiased=*/true);
    if (is_plane_) {
      last_fit_j_ = (int)N;
      updateMaxPlaneVarTrace(plane_var_.trace());
    }
    return;
  }

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

  // History (960-979): see docs/livo_recon_changelog.md#src-lio-voxelplane.cpp-960
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

// The unified plane model.  See VoxelOpts::plane_var_mode for the argument;
// this is the whole implementation, and its smallness is the point.
//
// I = sum_i a_i a_i^T / sigma_i^2 with a_i = [d_i.y_normal_, d_i.x_normal_, 1].
// Two exact simplifications collapse it:
//   * the points are centred on plane_.center, so sum_i d_i = 0 and both
//     cross terms between a tilt row and the offset row vanish;
//   * (y_normal_, x_normal_) are eigenvectors of the second moment, so
//     sum_i (d_i.y)(d_i.x) = 0 and the tilt block is diagonal too.
// What is left is  I = (N_eff/sigma_bar^2) * diag(lambda1, lambda2, 1),
// i.e. plane_var_ = (sigma_bar^2/N_eff) * diag(1/lambda1, 1/lambda2, 1).
//
// Read that against the incumbent: the eigengap form diverges as
// lambda0 -> lambda1, i.e. when the patch is barely a plane at all -- a case
// plane_threshold already rejects on its own terms.  This form instead
// diverges as lambda1 -> 0, i.e. when the plane was barely SAMPLED along the
// axis that would have pinned that tilt down.  A forward-looking scanner
// laying horizontal rings on a vertical wall has a large lambda2 along the
// rings and a tiny lambda1 across them, so the tilt about the horizontal
// axis is the uncertain one and a query above or below the sampled band
// picks that variance up through its own lever arm.  No occupancy grid, no
// exponent, no cap.
void VoxelPlane::buildInformationCovariance(double n_raw, const M3D& mean_cov_all,
                                            const M3D& mean_cov_sensor,
                                            bool eig0_already_debiased)
{
  const V3D& n = plane_.normal;
  // Only the along-normal component of a point's 3D covariance ever reaches
  // a point-to-plane residual.  This is the same projection gate() already
  // forms per correspondence as sigma_diag_squared, taken here as a mean
  // over the fit's own points.
  const double sigma_meas2  = std::max(0.0, n.dot(mean_cov_all * n));
  const double sigma_sens2  = std::max(0.0, n.dot(mean_cov_sensor * n));
  const double sigma_pose2  = std::max(0.0, sigma_meas2 - sigma_sens2);

  // Roughness as a variance rather than an admission test.  The debiased
  // path has already subtracted the noise out of lambda0; the pca path has
  // not, so the same subtraction happens here -- which is precisely why this
  // model does not need plane_fit_mode as an arm.
  roughness_ = eig0_already_debiased
      ? std::max(0.0, eigen_values_(0))
      : std::max(0.0, eigen_values_(0) - sigma_meas2);

  sigma_bar2_ = sigma_meas2 + roughness_;
  info_n_raw_ = n_raw;
  if (!(sigma_bar2_ > 0.0) || !std::isfinite(sigma_bar2_) || n_raw < 3.0) {
    is_plane_ = false;
    return;
  }

  // Design effect.  Returns from one sweep share that instant's pose error,
  // so a voxel holding n_raw returns from F frames does not hold n_raw
  // independent observations.  With intra-scan correlation rho and mean
  // group size m = n_raw/F, the standard equicorrelated effective size is
  // n_raw / (1 + (m-1)rho).  rho is not a tuning constant -- it is the share
  // of the residual variance that is common-mode, read off the same
  // accumulators the debiased fit already keeps split for its own
  // over-subtraction guard.
  const double F = std::max(1.0, static_cast<double>(distinct_frames_));
  const double m = n_raw / F;
  info_rho_ = (sigma_bar2_ > 0.0) ? std::min(1.0, sigma_pose2 / sigma_bar2_) : 0.0;
  info_n_eff_ = n_raw / std::max(1.0, 1.0 + (m - 1.0) * info_rho_);

  // P5.  The exact directional path, when update() populated it on this
  // fit (info_path_ == 1): the same design-effect discount the equal-weight
  // path applies (n_raw/n_eff), but on the full weighted 3x3 moment instead
  // of an isotropic sigma_bar2_ split diagonally by eigenvalue. Off-
  // diagonals are the whole point -- a plane sampled unevenly in direction
  // has real tilt/offset correlation the diagonal form cannot represent.
  const double design_effect = n_raw / info_n_eff_;
  if (info_path_ == 1) {
    Eigen::Matrix<double, 3, 3> H;
    const V3D& u = y_normal_;  const V3D& v = x_normal_;
    H(0, 0) = u.dot(Swdd_ * u);  H(0, 1) = u.dot(Swdd_ * v);  H(0, 2) = u.dot(Swd_);
    H(1, 0) = H(0, 1);           H(1, 1) = v.dot(Swdd_ * v);  H(1, 2) = v.dot(Swd_);
    H(2, 0) = H(0, 2);           H(2, 1) = H(1, 2);           H(2, 2) = Sw_;
    H /= design_effect;
    // Guard exactly as the diagonal path does: refuse the plane if H is
    // ill-conditioned (no support in one direction) or plane_var_ ends up
    // non-finite -- an ill-conditioned H must not be admitted with a huge
    // variance, it must be refused.
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es(H, Eigen::EigenvaluesOnly);
    const auto& hev = es.eigenvalues();
    const bool ill_conditioned = !(hev(0) > 0.0) || !std::isfinite(hev(0)) ||
        (hev(2) / std::max(hev(0), 1e-300)) > 1e12;
    if (ill_conditioned) {
      is_plane_ = false;
      return;
    }
    plane_var_ = H.inverse();
    if (!plane_var_.allFinite()) { is_plane_ = false; return; }
    g_info_fits.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  const double l1 = std::max(eigen_values_(1), 1e-12);   // along y_normal_ -> theta1
  const double l2 = std::max(eigen_values_(2), 1e-12);   // along x_normal_ -> theta2
  const double sc = sigma_bar2_ / info_n_eff_;

  plane_var_.setZero();
  plane_var_(0, 0) = sc / l1;
  plane_var_(1, 1) = sc / l2;
  plane_var_(2, 2) = sc;

  if (!plane_var_.allFinite()) { is_plane_ = false; return; }
  g_info_fits.fetch_add(1, std::memory_order_relaxed);
}

// History (1006-1016): see docs/livo_recon_changelog.md#src-lio-voxelplane.cpp-1006
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

// History (1041-1046): see docs/livo_recon_changelog.md#src-lio-voxelplane.cpp-1041
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

// History (1093-1105): see docs/livo_recon_changelog.md#src-lio-voxelplane.cpp-1093
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
