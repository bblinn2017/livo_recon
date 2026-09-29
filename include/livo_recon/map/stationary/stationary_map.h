#pragma once
#include <Eigen/Dense>
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
};
std::unique_ptr<Backend> makeBackend(const std::string& family, const Options& o);
}  // namespace livo_recon::stationary_map
