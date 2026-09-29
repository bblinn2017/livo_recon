#pragma once
#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
namespace livo_recon::stationary_map {
using V3 = Eigen::Vector3d;
using M3 = Eigen::Matrix3d;

// Set of distinct observation ids that contributed points (bitset; union is
// an OR). Needed by the debiased fit's per-frame pose-noise shrinkage
// (F-1)/F, where F = distinct frames, which is NOT additive across merged
// patches as a count but is as a set union.
struct ObsSet {
  std::vector<std::uint64_t> w;
  void add(std::uint64_t o) {
    const std::size_t i = static_cast<std::size_t>(o >> 6);
    if (w.size() <= i) w.resize(i + 1, 0ULL);
    w[i] |= (1ULL << (o & 63));
  }
  void merge(const ObsSet& b) {
    if (w.size() < b.w.size()) w.resize(b.w.size(), 0ULL);
    for (std::size_t i = 0; i < b.w.size(); ++i) w[i] |= b.w[i];
  }
  std::size_t count() const {
    std::size_t c = 0;
    for (auto x : w) c += static_cast<std::size_t>(__builtin_popcountll(x));
    return c;
  }
};

// Additive plane statistics. Sums are accumulated about `ref` (the first point
// ever added), exactly like the production VoxelMap running sums
// (`running->ref`), so sum_outer/n - mean*mean^T does not cancel
// catastrophically at map-frame coordinates. merge() re-expresses the other
// operand's sums about this operand's ref (exact algebra, no raw points).
//
// The covariance sums mirror the production VoxelPlane accumulators:
//   sum_cov          = sum_i (sensor_cov_i + pose_cov_i)   (Scov_)
//   sum_sensor_cov   = sum_i sensor_cov_i                  (Scov_sensor_)
//   sum_sensor_var   = sum_i trace(sensor_cov_i)/3         (sum_sensor_var_)
// These are translation-invariant, so they need no re-referencing on merge.
struct PlaneStats {
  std::uint64_t n = 0;
  V3 ref = V3::Zero();
  V3 sum = V3::Zero();        // sum of (p - ref)
  M3 sum_outer = M3::Zero();  // sum of (p - ref)(p - ref)^T
  M3 sum_cov = M3::Zero();
  M3 sum_sensor_cov = M3::Zero();
  double sum_sensor_var = 0.0;
  ObsSet obs;
  // R57 covariance tensors (eigengap plane covariance only; empty unless addCovTensors() is called).
  // Accumulated about the ORIGIN of the cache frame (not `ref`), so merge() is plain addition:
  //   covS = sum C_i, covW[a] = sum p_a C_i, covV[a][b] = sum p_a p_b C_i   (production V_/W_ analogue)
  // where C_i is the per-point covariance chosen by the caller (sensor, or sensor+pose).
  bool has_tensors = false;
  M3 covS = M3::Zero();
  std::array<M3, 3> covW = zeroArr3();
  std::array<std::array<M3, 3>, 3> covV = zeroArr33();
  static std::array<M3, 3> zeroArr3() { std::array<M3, 3> a; for (auto& m : a) m.setZero(); return a; }
  static std::array<std::array<M3, 3>, 3> zeroArr33() {
    std::array<std::array<M3, 3>, 3> a;
    for (auto& r : a) for (auto& m : r) m.setZero();
    return a;
  }
  void addCovTensors(const V3& p, const M3& C) {
    has_tensors = true;
    covS += C;
    for (int a = 0; a < 3; ++a) {
      covW[a] += p[a] * C;
      for (int b = a; b < 3; ++b) {
        const M3 t = (p[a] * p[b]) * C;
        covV[a][b] += t;
        if (b != a) covV[b][a] += t;
      }
    }
  }
  void add(const V3& p) { add(p, M3::Zero(), M3::Zero()); }
  void add(const V3& p, const M3& sensor_cov, const M3& pose_cov) {
    if (n == 0) ref = p;
    const V3 q = p - ref;
    ++n;
    sum += q;
    sum_outer += q * q.transpose();
    sum_cov += sensor_cov + pose_cov;
    sum_sensor_cov += sensor_cov;
    sum_sensor_var += sensor_cov.trace() / 3.0;
  }
  void noteObs(std::uint64_t o) { obs.add(o); }
  void merge(const PlaneStats& o) {
    if (o.n == 0) return;
    if (n == 0) { *this = o; return; }
    const V3 s = o.ref - ref;
    const double m = static_cast<double>(o.n);
    n += o.n;
    sum += o.sum + m * s;
    sum_outer += o.sum_outer + s * o.sum.transpose() + o.sum * s.transpose() + m * s * s.transpose();
    sum_cov += o.sum_cov;
    sum_sensor_cov += o.sum_sensor_cov;
    sum_sensor_var += o.sum_sensor_var;
    obs.merge(o.obs);
    if (o.has_tensors) {
      has_tensors = true;
      covS += o.covS;
      for (int a = 0; a < 3; ++a) {
        covW[a] += o.covW[a];
        for (int b = 0; b < 3; ++b) covV[a][b] += o.covV[a][b];
      }
    }
  }
  V3 mean() const { return n ? V3(ref + sum / double(n)) : V3::Zero(); }
  M3 covariance() const {
    if (n < 2) return M3::Zero();
    const V3 m = sum / double(n);
    return sum_outer / double(n) - m * m.transpose();
  }
};

// Pose-noise context for the debiased fit (stationary map: one fixed pose).
// Points are supplied in the IMU/body frame (the frame of the cache). Per point
// the backend computes what StateGroup::poseCovAt(p_body) computes in production
// (a WORLD-frame covariance):
//   pose_cov_w = -R K P_RR K R^T + P_PP - (R K P_RP) - (R K P_RP)^T,  K = [p_b]x
// and rotates it back into the body frame, pose_cov = R^T pose_cov_w R, so that
// all sums, fits and reported geometry stay in the cache frame (eigenvalues are
// frame-invariant; normals and centres stay comparable to the reference map).
// sensor_cov = sensor_var * I (isotropic; production's stationary calibration
// path builds sensor_cov = 0, calib_processing.cpp:508). R is the fixed world-
// from-body rotation of the stationary state; P_* are its calibrated covariance
// blocks (cov_ blocks idxR/idxP of the state).
struct PoseCovContext {
  M3 R = M3::Identity();
  M3 P_RR = M3::Zero(), P_PP = M3::Zero(), P_RP = M3::Zero();
};
M3 poseCovAtBody(const PoseCovContext& c, const V3& p_body);

// Plane-validity rule. Defaults reproduce the production VoxelMap acceptance
// test (src/lio/voxelplane.cpp): eig1 and eig2 must both be >= 1e-8 (rejects
// rank-1/collinear cells, which have an undefined normal) and eig0 < 0.01 m^2
// (absolute variance, NOT a scale-free ratio). The R52 harness used only a
// scale-free ratio lambda0/sum <= 0.10, which admits rank-1 cells; that rule
// is still reproducible with plane_eig_max=1e30, min_secondary_eig=0,
// max_planarity=0.10.
struct ValidityRule {
  std::size_t min_points = 3;
  double plane_eig_max = 0.01;
  double min_secondary_eig = 1e-8;
  double max_planarity = 1.0;  // lambda0 / sum(lambda); 1.0 = off
};

struct DebiasRule {
  bool sensor_noise_floor_eig0 = false;  // opts_->sensor_noise_floor_eig0 (default false)
  double denom_floor_scale = 0.1;        // eps_denom = max(1e-8, 0.1 * plane_eig_max)
};

struct PlaneFit {
  bool valid = false;
  bool rank2 = false;  // eig1, eig2 >= 1e-8 regardless of the rule in force (diagnostic)
  V3 center = V3::Zero(), normal = V3::UnitZ();
  double d = 0, planarity = 0;
  Eigen::Vector3d eigenvalues = Eigen::Vector3d::Zero();
  // R57: the other two eigenvectors (production y_normal_ = eigenvector 1, x_normal_ = eigenvector 2) and the
  // plane covariance over [theta1 (along y_axis), theta2 (along x_axis), h (offset along normal)].
  // cov_ok is false unless a covariance model was attached (merge_test=d2) and succeeded.
  V3 y_axis = V3::UnitY(), x_axis = V3::UnitX();
  bool cov_ok = false;
  M3 cov = M3::Zero();
};
// Normal is oriented to point toward the cache-frame origin (the sensor), so
// d = -normal.center >= 0. Comparisons never rely on this orientation: use
// planeAngleDeg / planeOffset below, which are sign-invariant.
PlaneFit fitPlane(const PlaneStats& s, const ValidityRule& rule);
// Debiased fit, mirroring VoxelPlane::refitDebiased():
//   C = S/N - mu mu^T - sum_sensor_cov/N - pose_shrink * (sum_cov - sum_sensor_cov)/N,
//   pose_shrink = (F-1)/F for F = distinct observations > 1 else 0.
// Rejects if eig1 or eig2 < 1e-8 (after debiasing), clamps eig0 to >= 0, optionally
// floors eig0 at sum_sensor_var/N (and rejects if the floor reaches eig1),
// requires eig0 < plane_eig_max and |eig0-eig1|, |eig0-eig2| >= eps_denom.
PlaneFit fitPlaneDebiased(const PlaneStats& s, const ValidityRule& rule, const DebiasRule& debias);
// R52 wrapper: ratio-only rule, admits rank-deficient cells. Kept only so the
// R52 numbers can be reproduced; not a credibility test.
PlaneFit fitIncrementalPca(const PlaneStats& s, double max_ratio = 0.08);

// R57 plane covariance models, both over [theta1, theta2, h] at the plane's own centre.
//  information (production plane_var_mode=information, diagonal path):
//      Sigma = (sigma_bar^2 / n_eff) * diag(1/lambda1, 1/lambda2, 1), sigma_bar^2 = n^T Cbar n + roughness,
//      roughness = max(0, lambda0 - n^T Cbar n) (fit not debiased) or max(0, lambda0) (debiased fit),
//      n_eff = N / max(1, 1 + (N/F - 1) rho), rho = min(1, sigma_pose^2 / sigma_bar^2), F = distinct observations.
//      Cbar = sum_cov/N when use_pose else sum_sensor_cov/N.
//  eigengap (production pca-path plane_var_, equal weights): sum_i J_i C_i J_i^T with
//      J_i rows (1/(N d1)) (n y^T + y n^T) z_i, (1/(N d2)) (n x^T + x n^T) z_i, -(1/N) n^T, d1 = l0 - l1, d2 = l0 - l2,
//      z_i = p_i - mean, computed EXACTLY from the additive tensors in PlaneStats (needs has_tensors).
//      Returns false (no covariance) when |d1| or |d2| < eps_denom.
bool planeCovInformation(const PlaneStats& s, const PlaneFit& f, bool debiased, bool use_pose, M3& out);
bool planeCovEigengap(const PlaneStats& s, const PlaneFit& f, double eps_denom, M3& out);
// Covariance-aware plane discrepancy: 2-dof tilt test (B's normal in A's tangent frame against
// Sigma_theta,A + J Sigma_theta,B J^T) plus 1-dof offset test at the midpoint of the two centres
// (difference of signed distances against the offset variance of each plane there). The two parts are
// treated as independent (approximation). Returns +inf if either fit lacks a covariance.
double planeD2(const PlaneFit& a, const PlaneFit& b);
double planeAngleDeg(const V3& na, const V3& nb);              // 0..90, sign-invariant
double planeOffset(const PlaneFit& a, const PlaneFit& b);      // max of the two centre-to-plane distances, sign-invariant

struct RansacResult { bool found = false; V3 normal = V3::UnitZ(); double d = 0; std::vector<int> inlier_idx; };
RansacResult ransacPlane(const std::vector<V3>& pts, double dist_thresh, int iters, std::uint64_t seed);

struct Patch {
  std::uint64_t id = 0;
  PlaneStats stats;  // in a Snapshot: the effective stats the fit was computed from
  PlaneFit fit;
  V3 bb_min = V3::Constant(1e30), bb_max = V3::Constant(-1e30);
  std::uint64_t surface_id = 0;
  std::vector<std::uint64_t> children;  // valid patch ids sharing this patch's surface
  std::uint64_t rejected_points = 0, reservoir_pending = 0;  // robust families only
  int level = 0;  // 0 = leaf cell; >0 = sub-cell created by robust recursion (cell size leaf / 2^level)
};
// members = the two surface ids merged (merge) or {split-out patch id} (split).
struct MergeEvent { std::string kind; std::uint64_t surface_id = 0; std::vector<std::uint64_t> members; std::string reason; };
struct Snapshot {
  std::string backend;
  std::size_t patches = 0, surfaces = 0;
  std::size_t cells_total = 0, cells_rank_deficient = 0;  // cells with n>=3 whose eig1 or eig2 < 1e-8
  std::uint64_t merges = 0, splits = 0, unmerges = 0;
  double insert_ms = 0;
  std::vector<Patch> data;
  std::vector<MergeEvent> events;
  // R57 (all zero for default options). Counters are cumulative over the run, except with
  // rebuild_mode=scratch where merges/splits/unmerges and these merge-test counters are per snapshot.
  double snapshot_ms = 0;               // wall time of this snapshot() call
  std::uint64_t state_bytes = 0;        // approximate retained state (cells, halves, reservoirs, surfaces)
  std::uint64_t d2_pairs = 0, d2_rejects = 0, d2_nocov_pairs = 0, support_rejects = 0;
  std::uint64_t obs_tested = 0, obs_skipped = 0, obs_rejects = 0, validity_rejects = 0, sub_cells = 0;
  std::uint64_t cells_valid_no_cov = 0; // valid cells (this snapshot) without a plane covariance (d2 mode)
  struct SurfaceStat {                  // one row per surface with >= 2 valid member patches (report_surface_stats)
    std::uint64_t surface_id = 0;
    std::uint64_t n_patches = 0, n_d2 = 0;
    double points = 0, max_angle_deg = 0, max_offset = 0, max_d2 = 0, mean_d2 = 0;  // member vs union plane
  };
  std::vector<SurfaceStat> surface_stats;
};
class Backend {
 public:
  virtual ~Backend() = default;
  virtual std::string name() const = 0;
  virtual void insert(std::uint64_t obs, const std::vector<V3>& pts) = 0;
  virtual Snapshot snapshot() const = 0;
};

struct Options {
  double leaf = 0.25;
  // Validity (see ValidityRule).
  std::size_t min_points = 3;
  double plane_eig_max = 0.01;
  double min_secondary_eig = 1e-8;
  double max_planarity = 1.0;
  // Debiased fit (production `plane_fit_mode: debiased`). Requires `pose` for a
  // non-trivial correction; sensor_var is the isotropic per-point sensor
  // variance (m^2), 0 to reproduce production's stationary calibration path.
  bool debiased = false;
  bool sensor_noise_floor_eig0 = false;
  double sensor_var = 0.0;
  PoseCovContext pose;
  // Merge/split.
  double merge_angle_deg = 5, merge_offset = 0.05, merge_gap = 0.75;
  // "combined_fit": a merge is accepted only if the additively combined
  // statistics of the two surfaces form a valid plane and EVERY member patch
  // stays within merge_angle_deg / merge_offset of that combined plane (no
  // transitive chaining). "pairwise": R52 behaviour, patch-to-patch single
  // linkage only.
  std::string merge_criterion = "combined_fit";
  double split_planarity_max = 0.20;
  // Robust families.
  std::size_t robust_reservoir = 96;
  double robust_ransac_dist = 0.02;
  int robust_ransac_iters = 200;

  // ---- R57 (every default reproduces R56 byte-for-byte) ----
  // merge_test: "threshold" (angle/offset gates, R55/R56) or "d2" (covariance-aware plane discrepancy).
  std::string merge_test = "threshold";
  // d2 only: "information" or "eigengap" (see planeCovInformation / planeCovEigengap).
  std::string plane_var_mode = "information";
  double d2_tau = 11.34;       // chi2(3) 99%: pair test, member-vs-union (combined_fit) test and split test
  double d2_tau_half = 16.81;  // chi2(6) 99%: split-half observation-consistency test (obs_test)
  bool d2_pose_cov = false;    // include the pose covariance (needs --debias-context) in C; else sensor_var*I only
  // Footprint support test (both merge tests): 3-D gap between the two patches' point bounding boxes <= support_gap.
  bool support_test = false;
  double support_gap = 0.25;
  // Observation consistency (d2 only): every patch also keeps stats for even/odd observation ids; a merge needs
  // sum over the two halves of D2(A_h, B_h) <= d2_tau_half. Skipped (counted) if any half fit is unusable.
  bool obs_test = false;
  // "incremental" (sticky, R55/R56) or "scratch" (singletons at every snapshot, then merge/split again).
  std::string rebuild_mode = "incremental";
  // Robust families. depth > 0: RANSAC-rejected points go to sub-cells of size leaf/2^level (level <= depth)
  // instead of the leftover tier. validity_test: the inlier set projected on its plane (grid = cell/grid_div,
  // 8-connectivity) must have its largest connected component >= conn_min of the occupied grid cells.
  int robust_recursion_depth = 0;
  bool robust_validity_test = false;
  double robust_grid_div = 5.0;
  double robust_conn_min = 0.5;
  bool report_surface_stats = false;
};
std::unique_ptr<Backend> makeBackend(const std::string& family, const Options& o);
}  // namespace livo_recon::stationary_map
