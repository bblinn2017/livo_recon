#include "livo_recon/map/stationary/stationary_map.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

// Stationary-map research library. Families (see makeBackend):
//   incremental_pca   per-cell additive stats, no merging
//   mergeable_voxel   + sticky surface aggregation
//   robust_voxel      per-cell block buffer + dominant-plane RANSAC, no merging
//   robust_mergeable  robust cells + sticky surface aggregation
//   gaussian_surface  mergeable_voxel + split re-evaluation (reversible merge)
//
// R53 changes relative to the R52 return (each traceable to the R52 audit):
//  - fitPlane(): production-parity validity (rank-2 + absolute eig0), rank2 flag.
//  - PlaneStats: reference-shifted sums (numerical cancellation).
//  - planeOffset()/planeAngleDeg(): sign-invariant; R52 compared |d_a - d_b|
//    with arbitrary eigenvector signs.
//  - SurfaceAgg.combined is recomputed from the current child stats at every
//    snapshot (R52 left it empty forever, so gaussian splits never fired).
//  - merge_criterion=combined_fit stops transitive chaining.
//  - robust families: snapshot() also counts the pending (not yet promoted)
//    block via a non-destructive RANSAC, so cells with < robust_reservoir
//    points are no longer silently absent.
namespace livo_recon::stationary_map {

namespace {
constexpr double kRad2Deg = 57.29577951308232;
constexpr double kRank2Eig = 1e-8;
}  // namespace

double planeAngleDeg(const V3& na, const V3& nb) {
  const double c = std::clamp(std::abs(na.dot(nb)), 0.0, 1.0);
  return std::acos(c) * kRad2Deg;
}

double planeOffset(const PlaneFit& a, const PlaneFit& b) {
  const V3 dc = b.center - a.center;
  return std::max(std::abs(a.normal.dot(dc)), std::abs(b.normal.dot(dc)));
}

PlaneFit fitPlane(const PlaneStats& s, const ValidityRule& rule) {
  PlaneFit f;
  if (s.n < 3) return f;
  const M3 cov = s.covariance();
  if (!cov.allFinite()) return f;
  Eigen::SelfAdjointEigenSolver<M3> es(cov);
  if (es.info() != Eigen::Success) return f;
  f.eigenvalues = es.eigenvalues();  // ascending
  f.center = s.mean();
  f.normal = es.eigenvectors().col(0).normalized();
  f.y_axis = es.eigenvectors().col(1).normalized();
  f.x_axis = es.eigenvectors().col(2).normalized();
  if (f.normal.dot(f.center) > 0.0) f.normal = -f.normal;  // point toward origin
  f.d = -f.normal.dot(f.center);
  const double den = std::max(f.eigenvalues.sum(), 1e-15);
  f.planarity = f.eigenvalues[0] / den;
  f.rank2 = f.eigenvalues[1] >= kRank2Eig && f.eigenvalues[2] >= kRank2Eig;
  f.valid = s.n >= rule.min_points &&
            f.eigenvalues[1] >= rule.min_secondary_eig && f.eigenvalues[2] >= rule.min_secondary_eig &&
            f.eigenvalues[0] < rule.plane_eig_max && f.planarity <= rule.max_planarity;
  return f;
}

M3 poseCovAtBody(const PoseCovContext& c, const V3& p) {
  M3 K;
  K << 0.0, -p.z(), p.y(),
       p.z(), 0.0, -p.x(),
       -p.y(), p.x(), 0.0;
  const M3 Rp = c.R * K;
  const M3 JRP = Rp * c.P_RP;
  return -c.R * K * c.P_RR * K * c.R.transpose() + c.P_PP - JRP - JRP.transpose();
}

PlaneFit fitPlaneDebiased(const PlaneStats& s, const ValidityRule& rule, const DebiasRule& debias) {
  PlaneFit f;
  if (s.n < 3) return f;
  const double N = static_cast<double>(s.n);
  const double F = static_cast<double>(s.obs.count());
  const double pose_shrink = F > 1.0 ? (F - 1.0) / F : 0.0;
  const V3 m = s.sum / N;
  const M3 pose_sum = s.sum_cov - s.sum_sensor_cov;
  const M3 cov = s.sum_outer / N - m * m.transpose() - s.sum_sensor_cov / N - pose_shrink * (pose_sum / N);
  if (!cov.allFinite()) return f;
  Eigen::SelfAdjointEigenSolver<M3> es(cov);
  if (es.info() != Eigen::Success) return f;
  Eigen::Vector3d ev = es.eigenvalues();
  f.center = s.mean();
  f.normal = es.eigenvectors().col(0).normalized();
  f.y_axis = es.eigenvectors().col(1).normalized();
  f.x_axis = es.eigenvectors().col(2).normalized();
  if (f.normal.dot(f.center) > 0.0) f.normal = -f.normal;
  f.d = -f.normal.dot(f.center);
  f.rank2 = ev[1] >= kRank2Eig && ev[2] >= kRank2Eig;
  f.eigenvalues = ev;
  f.planarity = std::max(ev[0], 0.0) / std::max(ev.sum(), 1e-15);
  if (ev[1] < rule.min_secondary_eig || ev[2] < rule.min_secondary_eig) return f;
  ev[0] = std::max(ev[0], 0.0);
  if (debias.sensor_noise_floor_eig0) {
    ev[0] = std::max(ev[0], s.sum_sensor_var / N);
    if (ev[0] >= ev[1]) { f.eigenvalues = ev; return f; }
  }
  f.eigenvalues = ev;
  if (!std::isfinite(ev[0])) return f;
  if (!(ev[0] < rule.plane_eig_max)) return f;
  const double eps_denom = std::max(1e-8, debias.denom_floor_scale * rule.plane_eig_max);
  if (std::fabs(ev[0] - ev[1]) < eps_denom || std::fabs(ev[0] - ev[2]) < eps_denom) return f;
  f.valid = s.n >= rule.min_points && f.planarity <= rule.max_planarity;
  return f;
}


bool planeCovInformation(const PlaneStats& s, const PlaneFit& f, bool debiased, bool use_pose, M3& out) {
  if (s.n < 3) return false;
  const double N = static_cast<double>(s.n);
  const M3 Cm = (use_pose ? s.sum_cov : s.sum_sensor_cov) / N;
  const M3 Cs = s.sum_sensor_cov / N;
  const V3& n = f.normal;
  const double sm2 = std::max(0.0, n.dot(Cm * n));
  const double ss2 = std::max(0.0, n.dot(Cs * n));
  const double sp2 = std::max(0.0, sm2 - ss2);
  const double rough = debiased ? std::max(0.0, f.eigenvalues[0]) : std::max(0.0, f.eigenvalues[0] - sm2);
  const double sb2 = sm2 + rough;
  if (!(sb2 > 0.0) || !std::isfinite(sb2)) return false;
  const double F = std::max(1.0, static_cast<double>(s.obs.count()));
  const double m = N / F;
  const double rho = std::min(1.0, sp2 / sb2);
  const double neff = N / std::max(1.0, 1.0 + (m - 1.0) * rho);
  const double sc = sb2 / neff;
  const double l1 = std::max(f.eigenvalues[1], 1e-12), l2 = std::max(f.eigenvalues[2], 1e-12);
  out = M3::Zero();
  out(0, 0) = sc / l1;
  out(1, 1) = sc / l2;
  out(2, 2) = sc;
  return out.allFinite();
}

bool planeCovEigengap(const PlaneStats& s, const PlaneFit& f, double eps_denom, M3& out) {
  if (s.n < 3 || !s.has_tensors) return false;
  const double N = static_cast<double>(s.n);
  const double d1 = f.eigenvalues[0] - f.eigenvalues[1];
  const double d2 = f.eigenvalues[0] - f.eigenvalues[2];
  if (std::fabs(d1) < eps_denom || std::fabs(d2) < eps_denom) return false;
  const V3 mu = s.mean();  // origin coordinates, same as the tensors
  const V3& n = f.normal;
  const V3& y = f.y_axis;
  const V3& x = f.x_axis;
  const M3& S1 = s.covS;
  // U_a = sum z_a C, T_ab = sum z_a z_b C with z = p - mu.
  std::array<M3, 3> U;
  std::array<std::array<M3, 3>, 3> T;
  for (int a = 0; a < 3; ++a) U[a] = s.covW[a] - mu[a] * S1;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      T[a][b] = s.covV[a][b] - mu[a] * s.covW[b] - mu[b] * s.covW[a] + mu[a] * mu[b] * S1;
  const M3 A1 = n * y.transpose() + y * n.transpose();
  const M3 A2 = n * x.transpose() + x * n.transpose();
  // Q(P, R) = sum_i z_i^T P C_i R z_i = sum_{a,b,c,d} P(a,c) R(d,b) T[a][b](c,d)
  auto Q = [&](const M3& P, const M3& R) {
    double q = 0.0;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        for (int c = 0; c < 3; ++c)
          for (int d = 0; d < 3; ++d) q += P(a, c) * R(d, b) * T[a][b](c, d);
    return q;
  };
  // Lc(A) = sum_i (A z_i)^T C_i n = sum_{a,c,d} A(c,a) n_d U[a](c,d)
  auto Lc = [&](const M3& A) {
    double q = 0.0;
    for (int a = 0; a < 3; ++a)
      for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d) q += A(c, a) * n[d] * U[a](c, d);
    return q;
  };
  out = M3::Zero();
  out(0, 0) = Q(A1, A1) / (N * N * d1 * d1);
  out(1, 1) = Q(A2, A2) / (N * N * d2 * d2);
  out(0, 1) = out(1, 0) = Q(A1, A2) / (N * N * d1 * d2);
  out(2, 2) = n.dot(S1 * n) / (N * N);
  out(0, 2) = out(2, 0) = -Lc(A1) / (N * N * d1);
  out(1, 2) = out(2, 1) = -Lc(A2) / (N * N * d2);
  return out.allFinite();
}

double planeD2(const PlaneFit& a, const PlaneFit& b) {
  const double inf = std::numeric_limits<double>::infinity();
  if (!a.cov_ok || !b.cov_ok) return inf;
  V3 nb = b.normal;
  if (nb.dot(a.normal) < 0.0) nb = -nb;
  Eigen::Vector2d dth(a.y_axis.dot(nb), a.x_axis.dot(nb));
  Eigen::Matrix2d J;
  J << a.y_axis.dot(b.y_axis), a.y_axis.dot(b.x_axis),
       a.x_axis.dot(b.y_axis), a.x_axis.dot(b.x_axis);
  const Eigen::Matrix2d S = a.cov.topLeftCorner<2, 2>() + J * b.cov.topLeftCorner<2, 2>() * J.transpose();
  const double det = S.determinant();
  if (!(det > 1e-300) || !std::isfinite(det)) return inf;
  const double t2 = dth.dot(S.inverse() * dth);
  const V3 mid = 0.5 * (a.center + b.center);
  const double delta = a.normal.dot(mid - a.center) - nb.dot(mid - b.center);
  auto var_at = [&](const PlaneFit& f) {
    const V3 dm = mid - f.center;
    const V3 l(f.y_axis.dot(dm), f.x_axis.dot(dm), 1.0);
    return l.dot(f.cov * l);
  };
  const double v = var_at(a) + var_at(b);
  if (!(v > 0.0) || !std::isfinite(v)) return inf;
  const double r = t2 + delta * delta / v;
  return std::isfinite(r) ? r : inf;
}

PlaneFit fitIncrementalPca(const PlaneStats& s, double max_ratio) {
  ValidityRule r;
  r.plane_eig_max = std::numeric_limits<double>::infinity();
  r.min_secondary_eig = -std::numeric_limits<double>::infinity();
  r.max_planarity = max_ratio;
  return fitPlane(s, r);
}

RansacResult ransacPlane(const std::vector<V3>& pts, double dist_thresh, int iters, std::uint64_t seed) {
  RansacResult best;
  const int n = static_cast<int>(pts.size());
  if (n < 3) return best;
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> pick(0, n - 1);
  int best_count = -1;
  for (int it = 0; it < iters; ++it) {
    int i0 = pick(rng), i1 = pick(rng), i2 = pick(rng);
    if (i0 == i1 || i1 == i2 || i0 == i2) continue;
    const V3& p0 = pts[i0]; const V3& p1 = pts[i1]; const V3& p2 = pts[i2];
    V3 normal = (p1 - p0).cross(p2 - p0);
    const double norm = normal.norm();
    if (norm < 1e-12) continue;
    normal /= norm;
    const double d = -normal.dot(p0);
    int count = 0;
    for (const auto& p : pts) if (std::abs(normal.dot(p) + d) <= dist_thresh) ++count;
    if (count > best_count) {
      best_count = count;
      best.normal = normal; best.d = d; best.found = true;
    }
  }
  if (!best.found) return best;
  best.inlier_idx.clear();
  for (int i = 0; i < n; ++i) if (std::abs(best.normal.dot(pts[i]) + best.d) <= dist_thresh) best.inlier_idx.push_back(i);
  return best;
}

namespace {

struct Key {
  int x, y, z;
  int l = 0;  // R57: refinement level of a robust sub-cell (0 for every leaf cell, so R56 ordering is unchanged)
  bool operator<(const Key& o) const { return std::tie(x, y, z, l) < std::tie(o.x, o.y, o.z, o.l); }
};
Key key(const V3& p, double size, int level = 0) {
  return {int(std::floor(p.x() / size)), int(std::floor(p.y() / size)), int(std::floor(p.z() / size)), level};
}
double boxGap(const V3& amin, const V3& amax, const V3& bmin, const V3& bmax) {
  const V3 g = (amin - bmax).cwiseMax(bmin - amax).cwiseMax(0.0);
  return g.norm();
}

// Per-cell block buffer for the robust families. NOTE: this is a fixed-size
// block (filled to `cap`, RANSAC'd, cleared), not a rolling reservoir sample;
// the R52 replacement branch was unreachable because the buffer is cleared at
// cap, and is removed.
struct Reservoir {
  std::vector<V3> primary;   // block awaiting a RANSAC decision
  std::vector<std::uint64_t> primary_obs;
  std::vector<V3> leftover;  // one extra tier for RANSAC-rejected points
  std::vector<std::uint64_t> leftover_obs;
  std::uint64_t seen = 0;
  std::uint64_t rejected_total = 0;
  void offer(const V3& p, std::uint64_t o) { ++seen; primary.push_back(p); primary_obs.push_back(o); }
};

struct SurfaceAgg {
  std::uint64_t id = 0;
  PlaneStats combined;  // sum of the CURRENT effective stats of VALID children
  std::vector<std::uint64_t> children;  // patch ids (may include currently-invalid cells)
  PlaneFit fit;
};

class MapImpl : public Backend {
 public:
  MapImpl(std::string n, Options o, bool merge, bool robust, bool gaussian)
      : n_(std::move(n)), o_(o), merge_(merge), robust_(robust), gaussian_(gaussian) {
    rule_.min_points = o_.min_points;
    rule_.plane_eig_max = o_.plane_eig_max;
    rule_.min_secondary_eig = o_.min_secondary_eig;
    rule_.max_planarity = o_.max_planarity;
    debias_.sensor_noise_floor_eig0 = o_.sensor_noise_floor_eig0;
    R_T_ = o_.pose.R.transpose();
    if (o_.merge_criterion != "combined_fit" && o_.merge_criterion != "pairwise")
      throw std::runtime_error("merge_criterion must be combined_fit or pairwise");
    if (o_.merge_test != "threshold" && o_.merge_test != "d2")
      throw std::runtime_error("merge_test must be threshold or d2");
    if (o_.plane_var_mode != "information" && o_.plane_var_mode != "eigengap")
      throw std::runtime_error("plane_var_mode must be information or eigengap");
    if (o_.rebuild_mode != "incremental" && o_.rebuild_mode != "scratch")
      throw std::runtime_error("rebuild_mode must be incremental or scratch");
    d2_ = (o_.merge_test == "d2");
    if (o_.obs_test && !d2_) throw std::runtime_error("obs_test requires merge_test=d2");
    tensors_ = d2_ && o_.plane_var_mode == "eigengap";
    if (tensors_ && o_.debiased) throw std::runtime_error("plane_var_mode=eigengap is defined for the non-debiased fit only");
    if (tensors_ && o_.sensor_var <= 0.0 && !o_.d2_pose_cov)
      throw std::runtime_error("plane_var_mode=eigengap needs a non-zero covariance: set sensor_var > 0 or d2_pose_cov");
    if (o_.robust_recursion_depth < 0 || o_.robust_recursion_depth > 4)
      throw std::runtime_error("robust_recursion_depth must be 0..4");
    eps_denom_ = std::max(1e-8, debias_.denom_floor_scale * o_.plane_eig_max);
  }
  std::string name() const override { return n_; }

  // Merge/split work is deferred to snapshot(): only checkpoint states are read.
  void insert(std::uint64_t obs, const std::vector<V3>& pts) override {
    auto t0 = std::chrono::steady_clock::now();
    for (const auto& p : pts) {
      auto k = key(p, o_.leaf);
      auto& pa = getCell(k);
      pa.bb_min = pa.bb_min.cwiseMin(p);
      pa.bb_max = pa.bb_max.cwiseMax(p);
      if (robust_) {
        auto& res = reservoirs_[k];
        res.offer(p, obs);
        maybeRunRansac(pa, res, k, 0);
      } else {
        addPoint(pa.stats, p, obs, o_.obs_test ? &half_[k] : nullptr);
      }
    }
    ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }

  Snapshot snapshot() const override {
    auto t0 = std::chrono::steady_clock::now();
    refreshFits();
    if (o_.rebuild_mode == "scratch") resetSurfaces();
    refreshSurfaceStats();
    if (merge_) rebuildSurfaces();
    if (gaussian_) reevaluateSplits();
    const double snap_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ms_ += snap_ms;

    Snapshot s;
    s.backend = n_; s.insert_ms = ms_; s.merges = merges_; s.splits = splits_; s.unmerges = unmerges_;
    s.events = events_;
    s.snapshot_ms = snap_ms;
    s.d2_pairs = d2_pairs_; s.d2_rejects = d2_rejects_; s.d2_nocov_pairs = d2_nocov_pairs_;
    s.support_rejects = support_rejects_; s.obs_tested = obs_tested_; s.obs_skipped = obs_skipped_;
    s.obs_rejects = obs_rejects_; s.validity_rejects = validity_rejects_; s.sub_cells = sub_cells_;
    s.cells_total = cells_.size();
    std::set<std::uint64_t> ids;
    for (const auto& kv : cells_) {
      const Patch& c = kv.second;
      const PlaneStats& eff = eff_.at(kv.first);
      if (eff.n >= 3 && !c.fit.rank2) ++s.cells_rank_deficient;
      if (!c.fit.valid) continue;
      if (d2_ && !c.fit.cov_ok) ++s.cells_valid_no_cov;
      Patch p = c;
      p.stats = eff;
      if (robust_) {
        auto it = reservoirs_.find(kv.first);
        if (it != reservoirs_.end()) {
          p.rejected_points = it->second.rejected_total;
          p.reservoir_pending = it->second.primary.size();
        }
      }
      p.children.clear();
      auto sit = surfaces_.find(p.surface_id);
      if (sit != surfaces_.end())
        for (auto cid : sit->second.children)
          if (cellOf(cid).fit.valid) p.children.push_back(cid);
      std::sort(p.children.begin(), p.children.end());
      s.data.push_back(p);
      ids.insert(p.surface_id);
    }
    s.patches = s.data.size();
    s.surfaces = ids.size();
    s.state_bytes = stateBytes();
    if (o_.report_surface_stats) fillSurfaceStats(s);
    return s;
  }

 private:
  // Per-point covariance terms (debiased mode only; zero otherwise).
  // `halves` (obs_test): the same point is also added to halves[obs & 1] (even/odd observation ids).
  void addPoint(PlaneStats& st, const V3& p, std::uint64_t obs, std::array<PlaneStats, 2>* halves = nullptr) const {
    M3 sensor = M3::Zero(), pose = M3::Zero();
    const bool need_cov = o_.debiased || d2_;
    if (need_cov) {
      sensor = o_.sensor_var * M3::Identity();
      if (o_.debiased || o_.d2_pose_cov) pose = R_T_ * poseCovAtBody(o_.pose, p) * o_.pose.R;
    }
    const M3 C = o_.d2_pose_cov ? M3(sensor + pose) : sensor;
    auto feed = [&](PlaneStats& t) {
      if (need_cov) t.add(p, sensor, pose); else t.add(p);
      t.noteObs(obs);
      if (tensors_) t.addCovTensors(p, C);
    };
    feed(st);
    if (halves) feed((*halves)[obs & 1ULL]);
  }
  PlaneFit fit(const PlaneStats& st) const {
    PlaneFit f = o_.debiased ? fitPlaneDebiased(st, rule_, debias_) : fitPlane(st, rule_);
    if (d2_ && f.valid) {
      M3 c;
      const bool ok = tensors_ ? planeCovEigengap(st, f, eps_denom_, c)
                               : planeCovInformation(st, f, o_.debiased, o_.d2_pose_cov, c);
      f.cov_ok = ok;
      if (ok) f.cov = c;
    }
    return f;
  }

  Patch& getCell(const Key& k) {
    auto& pa = cells_[k];
    if (!pa.id) {
      pa.id = ++next_;
      pa.level = k.l;
      pa.surface_id = pa.id;
      SurfaceAgg sa; sa.id = pa.id; sa.children = {pa.id};
      surfaces_[pa.id] = sa;
      id_to_key_[pa.id] = k;
      if (k.l > 0) ++sub_cells_;
    }
    return pa;
  }

  Patch& cellOf(std::uint64_t patch_id) const { return cells_.at(id_to_key_.at(patch_id)); }

  // Effective per-cell statistics = promoted stats (+ inliers of the pending
  // block for robust families, computed non-destructively), then the fit.
  void refreshFits() const {
    eff_.clear();
    for (auto& kv : cells_) {
      Patch& pa = kv.second;
      PlaneStats eff = pa.stats;
      std::array<PlaneStats, 2> eh;
      if (o_.obs_test) {
        auto hit = half_.find(kv.first);
        if (hit != half_.end()) eh = hit->second;
      }
      if (robust_) {
        auto it = reservoirs_.find(kv.first);
        if (it != reservoirs_.end() && it->second.primary.size() >= 3) {
          auto r = ransacPlane(it->second.primary, o_.robust_ransac_dist, o_.robust_ransac_iters,
                               pa.id * 7919ULL + it->second.seen);
          if (r.found)
            for (int idx : r.inlier_idx)
              addPoint(eff, it->second.primary[idx], it->second.primary_obs[idx], o_.obs_test ? &eh : nullptr);
        }
      }
      eff_[kv.first] = eff;
      pa.fit = fit(eff);
      if (o_.obs_test) {
        eff_half_[kv.first] = eh;
        half_fit_[kv.first] = {fit(eh[0]), fit(eh[1])};
      }
    }
  }

  void refreshSurfaceStats() const {
    for (auto& kv : surfaces_) {
      SurfaceAgg& sa = kv.second;
      sa.combined = PlaneStats();
      for (auto cid : sa.children) {
        const Patch& c = cellOf(cid);
        if (c.fit.valid) sa.combined.merge(eff_.at(id_to_key_.at(cid)));
      }
      sa.fit = fit(sa.combined);
    }
  }

  struct Consistency {
    bool ok = true;
    std::uint64_t worst = 0;
    double worst_score = -1.0;  // max over children of max(angle/merge_angle, offset/merge_offset); inf for invalid children
    double worst_ang = 0, worst_off = 0, planarity = 0;
  };
  // Is `children` (with combined stats `u`) still one plane? `invalid_violates`:
  // a currently-invalid child counts as a violation (split) instead of being ignored (merge).
  Consistency evaluate(const PlaneStats& u, const std::vector<std::uint64_t>& children, bool invalid_violates) const {
    Consistency r;
    const PlaneFit f = fit(u);
    r.planarity = f.planarity;
    const bool degraded = !f.valid || f.planarity > o_.split_planarity_max;
    for (auto cid : children) {
      const Patch& c = cellOf(cid);
      double score, ang = 0, off = 0;
      if (!c.fit.valid) {
        if (!invalid_violates) continue;
        score = std::numeric_limits<double>::infinity();
      } else if (!f.valid) {
        score = 0.0;  // no reference plane; the degraded flag decides
      } else {
        ang = planeAngleDeg(c.fit.normal, f.normal);
        off = std::abs(f.normal.dot(c.fit.center - f.center));
        if (d2_) {
          // member vs union: no covariance on either side -> no evidence, score 0 (never blocks/splits)
          score = (c.fit.cov_ok && f.cov_ok) ? planeD2(c.fit, f) / std::max(o_.d2_tau, 1e-12) : 0.0;
        } else {
          score = std::max(ang / std::max(o_.merge_angle_deg, 1e-12), off / std::max(o_.merge_offset, 1e-12));
        }
      }
      if (score > r.worst_score) { r.worst_score = score; r.worst = cid; r.worst_ang = ang; r.worst_off = off; }
    }
    r.ok = !degraded && r.worst_score <= 1.0;
    // d2 merge context: a union whose plane covariance cannot be computed gives no evidence, so refuse the merge
    // (splits, invalid_violates=true, are never forced by a missing covariance).
    if (d2_ && !invalid_violates && f.valid && !f.cov_ok) r.ok = false;
    return r;
  }

  // Robust extraction: promote a full block's RANSAC inliers into the cell's
  // permanent additive stats; rejected points go to a bounded leftover tier
  // that gets exactly one further independent RANSAC attempt.
  // With robust_recursion_depth > level, the rejected points of a full block are offered to the sub-cell of
  // size leaf / 2^(level+1) containing them (which becomes its own patch) instead of the leftover tier.
  void maybeRunRansac(Patch& pa, Reservoir& res, const Key& k, int level) {
    if (res.primary.size() < o_.robust_reservoir) return;
    std::array<PlaneStats, 2>* hp = o_.obs_test ? &half_[k] : nullptr;
    const double cell = o_.leaf / static_cast<double>(1 << level);
    auto run = [&](std::vector<V3>& pool, std::vector<std::uint64_t>& pool_obs,
                   std::vector<V3>& rej, std::vector<std::uint64_t>& rej_obs) {
      auto r = ransacPlane(pool, o_.robust_ransac_dist, o_.robust_ransac_iters, pa.id * 7919ULL + res.seen);
      if (r.found && o_.robust_validity_test && !distributionValid(pool, r, cell)) {
        r.found = false;
        ++validity_rejects_;
      }
      if (r.found) {
        std::vector<bool> is_inlier(pool.size(), false);
        for (int idx : r.inlier_idx) is_inlier[idx] = true;
        for (std::size_t i = 0; i < pool.size(); ++i) {
          if (is_inlier[i]) addPoint(pa.stats, pool[i], pool_obs[i], hp);
          else { rej.push_back(pool[i]); rej_obs.push_back(pool_obs[i]); }
        }
      } else {
        rej = pool; rej_obs = pool_obs;
      }
      pool.clear(); pool_obs.clear();
    };
    std::vector<V3> rej_p; std::vector<std::uint64_t> rej_p_obs;
    run(res.primary, res.primary_obs, rej_p, rej_p_obs);
    if (level < o_.robust_recursion_depth) {
      for (std::size_t i = 0; i < rej_p.size(); ++i) routeRejected(rej_p[i], rej_p_obs[i], level + 1);
      return;
    }
    for (std::size_t i = 0; i < rej_p.size(); ++i) {
      if (res.leftover.size() < o_.robust_reservoir) { res.leftover.push_back(rej_p[i]); res.leftover_obs.push_back(rej_p_obs[i]); }
      else ++res.rejected_total;
    }
    if (res.leftover.size() >= o_.robust_reservoir) {
      std::vector<V3> rej_l; std::vector<std::uint64_t> rej_l_obs;
      run(res.leftover, res.leftover_obs, rej_l, rej_l_obs);
      res.rejected_total += rej_l.size();
    }
  }

  void routeRejected(const V3& p, std::uint64_t obs, int level) {
    const Key k = key(p, o_.leaf / static_cast<double>(1 << level), level);
    Patch& pa = getCell(k);
    pa.bb_min = pa.bb_min.cwiseMin(p);
    pa.bb_max = pa.bb_max.cwiseMax(p);
    auto& res = reservoirs_[k];
    res.offer(p, obs);
    maybeRunRansac(pa, res, k, level);
  }

  // R-VoxelMap-style validity test: the RANSAC inliers projected on their plane, binned at cell/robust_grid_div,
  // must form one dominant 8-connected occupied region (largest component / occupied cells >= robust_conn_min).
  bool distributionValid(const std::vector<V3>& pool, const RansacResult& r, double cell) const {
    if (r.inlier_idx.size() < 3) return false;
    const V3 n = r.normal.normalized();
    const V3 t1 = n.cross(std::fabs(n.z()) < 0.9 ? V3::UnitZ() : V3::UnitX()).normalized();
    const V3 t2 = n.cross(t1);
    const double g = std::max(cell / std::max(o_.robust_grid_div, 1.0), 1e-6);
    std::set<std::pair<int, int>> occ;
    for (int idx : r.inlier_idx) {
      const V3& p = pool[idx];
      occ.insert({int(std::floor(p.dot(t1) / g)), int(std::floor(p.dot(t2) / g))});
    }
    std::set<std::pair<int, int>> seen;
    std::size_t best = 0;
    for (const auto& c0 : occ) {
      if (seen.count(c0)) continue;
      std::vector<std::pair<int, int>> stack{c0};
      seen.insert(c0);
      std::size_t cnt = 0;
      while (!stack.empty()) {
        const auto c = stack.back(); stack.pop_back(); ++cnt;
        for (int dx = -1; dx <= 1; ++dx)
          for (int dy = -1; dy <= 1; ++dy) {
            const std::pair<int, int> nb{c.first + dx, c.second + dy};
            if (occ.count(nb) && !seen.count(nb)) { seen.insert(nb); stack.push_back(nb); }
          }
      }
      best = std::max(best, cnt);
    }
    return static_cast<double>(best) / static_cast<double>(occ.size()) >= o_.robust_conn_min;
  }

  void rebuildSurfaces() const {
    std::vector<Patch*> v;
    for (auto& kv : cells_) if (kv.second.fit.valid) v.push_back(&kv.second);
    std::sort(v.begin(), v.end(), [](const Patch* a, const Patch* b) { return a->id < b->id; });
    std::map<Key, std::vector<Patch*>> buckets;
    for (auto* p : v) buckets[key(p->fit.center, o_.merge_gap)].push_back(p);
    for (Patch* a : v) {
      const Key ka = key(a->fit.center, o_.merge_gap);
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dz = -1; dz <= 1; ++dz) {
            auto bit = buckets.find(Key{ka.x + dx, ka.y + dy, ka.z + dz});
            if (bit == buckets.end()) continue;
            for (Patch* b : bit->second) {
              if (b->id <= a->id || a->surface_id == b->surface_id) continue;
              if (d2_) {
                if ((a->fit.center - b->fit.center).norm() > o_.merge_gap) continue;
                if (o_.support_test && boxGap(a->bb_min, a->bb_max, b->bb_min, b->bb_max) > o_.support_gap) {
                  ++support_rejects_; continue;
                }
                if (!a->fit.cov_ok || !b->fit.cov_ok) { ++d2_nocov_pairs_; continue; }
                ++d2_pairs_;
                if (planeD2(a->fit, b->fit) > o_.d2_tau) { ++d2_rejects_; continue; }
                if (o_.obs_test && !obsCompatible(*a, *b)) continue;
              } else {
                if (planeAngleDeg(a->fit.normal, b->fit.normal) > o_.merge_angle_deg) continue;
                if (planeOffset(a->fit, b->fit) > o_.merge_offset) continue;
                if ((a->fit.center - b->fit.center).norm() > o_.merge_gap) continue;
                if (o_.support_test && boxGap(a->bb_min, a->bb_max, b->bb_min, b->bb_max) > o_.support_gap) {
                  ++support_rejects_; continue;
                }
              }
              tryMerge(a->surface_id, b->surface_id);
            }
          }
    }
  }

  // Split-half observation consistency: both the even-observation and the odd-observation statistics must
  // separately agree that A and B are one plane (sum of the two D2 values against d2_tau_half, 6 dof).
  bool obsCompatible(const Patch& a, const Patch& b) const {
    const auto& ha = half_fit_.at(id_to_key_.at(a.id));
    const auto& hb = half_fit_.at(id_to_key_.at(b.id));
    double sum = 0.0;
    for (int h = 0; h < 2; ++h) {
      if (!ha[h].valid || !hb[h].valid || !ha[h].cov_ok || !hb[h].cov_ok) { ++obs_skipped_; return true; }
      sum += planeD2(ha[h], hb[h]);
    }
    ++obs_tested_;
    if (!(sum <= o_.d2_tau_half)) { ++obs_rejects_; return false; }
    return true;
  }

  // rebuild_mode=scratch: every cell back to a singleton surface; per-snapshot counters restart.
  void resetSurfaces() const {
    surfaces_.clear();
    for (auto& kv : cells_) {
      Patch& pa = kv.second;
      pa.surface_id = pa.id;
      SurfaceAgg sa; sa.id = pa.id; sa.children = {pa.id};
      surfaces_[pa.id] = sa;
    }
    events_.clear();
    merges_ = splits_ = unmerges_ = 0;
    d2_pairs_ = d2_rejects_ = d2_nocov_pairs_ = support_rejects_ = 0;
    obs_tested_ = obs_skipped_ = obs_rejects_ = 0;
  }

  std::uint64_t stateBytes() const {
    std::uint64_t b = cells_.size() * (sizeof(Patch) + sizeof(PlaneStats));
    if (o_.obs_test) b += cells_.size() * 2 * sizeof(PlaneStats);
    for (const auto& kv : reservoirs_)
      b += (kv.second.primary.size() + kv.second.leftover.size()) * (sizeof(V3) + sizeof(std::uint64_t));
    b += surfaces_.size() * sizeof(SurfaceAgg) + id_to_key_.size() * (sizeof(std::uint64_t) + sizeof(Key));
    return b;
  }

  void fillSurfaceStats(Snapshot& s) const {
    for (const auto& kv : surfaces_) {
      const SurfaceAgg& sa = kv.second;
      std::vector<const Patch*> ch;
      for (auto cid : sa.children) { const Patch& c = cellOf(cid); if (c.fit.valid) ch.push_back(&c); }
      if (ch.size() < 2 || !sa.fit.valid) continue;
      Snapshot::SurfaceStat st;
      st.surface_id = sa.id; st.n_patches = ch.size();
      double sum_d2 = 0.0;
      for (const Patch* c : ch) {
        st.points += static_cast<double>(eff_.at(id_to_key_.at(c->id)).n);
        st.max_angle_deg = std::max(st.max_angle_deg, planeAngleDeg(c->fit.normal, sa.fit.normal));
        st.max_offset = std::max(st.max_offset, std::abs(sa.fit.normal.dot(c->fit.center - sa.fit.center)));
        if (d2_ && c->fit.cov_ok && sa.fit.cov_ok) {
          const double d = planeD2(c->fit, sa.fit);
          st.max_d2 = std::max(st.max_d2, d); sum_d2 += d; ++st.n_d2;
        }
      }
      st.mean_d2 = st.n_d2 ? sum_d2 / static_cast<double>(st.n_d2) : 0.0;
      s.surface_stats.push_back(st);
    }
  }

  void tryMerge(std::uint64_t ida, std::uint64_t idb) const {
    if (ida == idb) return;
    SurfaceAgg& A = surfaces_.at(ida);
    SurfaceAgg& B = surfaces_.at(idb);
    PlaneStats u = A.combined;
    u.merge(B.combined);
    if (o_.merge_criterion == "combined_fit") {
      std::vector<std::uint64_t> ch = A.children;
      ch.insert(ch.end(), B.children.begin(), B.children.end());
      if (!evaluate(u, ch, /*invalid_violates=*/false).ok) return;
    }
    const std::uint64_t keep_id = std::min(ida, idb), drop_id = std::max(ida, idb);
    SurfaceAgg& K = surfaces_.at(keep_id);
    SurfaceAgg& D = surfaces_.at(drop_id);
    for (auto cid : D.children) {
      cellOf(cid).surface_id = keep_id;
      K.children.push_back(cid);
    }
    K.combined = u;
    K.fit = fit(K.combined);
    MergeEvent ev; ev.kind = "merge"; ev.surface_id = keep_id; ev.members = {keep_id, drop_id};
    ev.reason = o_.merge_criterion;
    events_.push_back(ev);
    surfaces_.erase(drop_id);
    ++merges_;
  }

  // gaussian_surface only: while a multi-child surface is inconsistent
  // (combined fit invalid/degraded, or some child deviates beyond the merge
  // thresholds from the combined plane, or a child cell has become invalid),
  // split the worst child back out as its own singleton, re-summing the
  // remaining children's retained stats (never raw points).
  void reevaluateSplits() const {
    std::vector<std::uint64_t> ids;
    for (auto& kv : surfaces_) ids.push_back(kv.first);
    for (auto sid : ids) {
      while (true) {
        auto it = surfaces_.find(sid);
        if (it == surfaces_.end() || it->second.children.size() < 2) break;
        SurfaceAgg& sa = it->second;
        Consistency c = evaluate(sa.combined, sa.children, /*invalid_violates=*/true);
        if (c.ok || c.worst == 0) break;
        const std::uint64_t w = c.worst;
        Patch& wc = cellOf(w);
        std::vector<std::uint64_t> remaining;
        for (auto cid : sa.children) if (cid != w) remaining.push_back(cid);
        PlaneStats rebuilt;
        for (auto cid : remaining) if (cellOf(cid).fit.valid) rebuilt.merge(eff_.at(id_to_key_.at(cid)));
        // If the split-out child carries the surface's own id, re-id the rest.
        std::uint64_t rest_id = sid;
        if (w == sid) rest_id = *std::min_element(remaining.begin(), remaining.end());
        SurfaceAgg rest; rest.id = rest_id; rest.children = remaining; rest.combined = rebuilt;
        rest.fit = fit(rebuilt);
        surfaces_.erase(sid);
        for (auto cid : remaining) cellOf(cid).surface_id = rest_id;
        surfaces_[rest_id] = rest;
        SurfaceAgg single; single.id = w; single.children = {w};
        if (wc.fit.valid) single.combined = eff_.at(id_to_key_.at(w));
        single.fit = fit(single.combined);
        wc.surface_id = w;
        surfaces_[w] = single;
        MergeEvent ev; ev.kind = "split"; ev.surface_id = rest_id; ev.members = {w};
        std::ostringstream oss;
        oss << "worst child angle " << c.worst_ang << "deg offset " << c.worst_off << "m; combined planarity "
            << c.planarity << " (split_planarity_max " << o_.split_planarity_max << ")";
        ev.reason = oss.str();
        events_.push_back(ev);
        ++splits_; ++unmerges_;
        sid = rest_id;
      }
    }
  }

  std::string n_;
  Options o_;
  ValidityRule rule_;
  DebiasRule debias_;
  M3 R_T_ = M3::Identity();
  bool merge_, robust_, gaussian_;
  mutable std::map<Key, Patch> cells_;
  mutable std::map<Key, PlaneStats> eff_;
  // R57 obs_test: promoted even/odd-observation stats, effective (with pending robust inliers) copies and their fits.
  std::map<Key, std::array<PlaneStats, 2>> half_;
  mutable std::map<Key, std::array<PlaneStats, 2>> eff_half_;
  mutable std::map<Key, std::array<PlaneFit, 2>> half_fit_;
  std::map<std::uint64_t, Key> id_to_key_;
  std::map<Key, Reservoir> reservoirs_;
  mutable std::map<std::uint64_t, SurfaceAgg> surfaces_;
  mutable std::vector<MergeEvent> events_;
  std::uint64_t next_ = 0;
  mutable std::uint64_t merges_ = 0, splits_ = 0, unmerges_ = 0;
  bool d2_ = false, tensors_ = false;
  double eps_denom_ = 1e-8;
  mutable std::uint64_t d2_pairs_ = 0, d2_rejects_ = 0, d2_nocov_pairs_ = 0, support_rejects_ = 0;
  mutable std::uint64_t obs_tested_ = 0, obs_skipped_ = 0, obs_rejects_ = 0;
  std::uint64_t validity_rejects_ = 0, sub_cells_ = 0;
  mutable double ms_ = 0;
};

}  // namespace

std::unique_ptr<Backend> makeBackend(const std::string& f, const Options& o) {
  if (f == "mergeable_voxel") return std::make_unique<MapImpl>(f, o, true, false, false);
  if (f == "robust_voxel") return std::make_unique<MapImpl>(f, o, false, true, false);
  if (f == "robust_mergeable") return std::make_unique<MapImpl>(f, o, true, true, false);
  if (f == "gaussian_surface") return std::make_unique<MapImpl>(f, o, true, false, true);
  if (f == "incremental_pca") return std::make_unique<MapImpl>(f, o, false, false, false);
  throw std::runtime_error("unknown stationary map backend: " + f);
}

}  // namespace livo_recon::stationary_map
