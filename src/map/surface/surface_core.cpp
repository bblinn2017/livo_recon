#include "livo_recon/map/surface/surface_core.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Eigenvalues>

namespace livo_recon
{
namespace surface
{

namespace
{
constexpr double kRad2Deg = 57.29577951308232;
}

// ---------------------------------------------------------------------------------------------------------------------
// SurfaceStats
// ---------------------------------------------------------------------------------------------------------------------
SurfaceStats::SurfaceStats()
{
  for (int a = 0; a < 3; ++a) {
    w[a].setZero();
    for (int b = 0; b < 3; ++b) v[a][b].setZero();
  }
}

void SurfaceStats::add(const V3& p, const M3& cov)
{
  const V3 d = p - ref;
  n += 1.0;
  sp += d;
  spp.noalias() += d * d.transpose();
  scov += cov;
  for (int a = 0; a < 3; ++a) {
    w[a] += p(a) * cov;
    for (int b = a; b < 3; ++b) {
      const M3 term = (p(a) * p(b)) * cov;
      v[a][b] += term;
      if (b != a) v[b][a] += term;
    }
  }
}

void SurfaceStats::merge(const SurfaceStats& o)
{
  if (o.n <= 0.0) return;
  if (n <= 0.0) {
    *this = o;
    return;
  }
  const V3 s = o.ref - ref;   // (p - ref) = (p - o.ref) + s
  n += o.n;
  sp += o.sp + o.n * s;
  spp += o.spp + o.sp * s.transpose() + s * o.sp.transpose() + o.n * (s * s.transpose());
  scov += o.scov;
  for (int a = 0; a < 3; ++a) {
    w[a] += o.w[a];
    for (int b = 0; b < 3; ++b) v[a][b] += o.v[a][b];
  }
}

M3 SurfaceStats::covariance() const
{
  const V3 m = sp / n;
  return spp / n - m * m.transpose() - scov / n;
}

// ---------------------------------------------------------------------------------------------------------------------
// Fit: VoxelPlane::refitDebiased() with the pose-covariance terms removed (sensor_only => Scov == Scov_sensor, and the
// pose shrink term is identically zero).
// ---------------------------------------------------------------------------------------------------------------------
Fit fitStats(const SurfaceStats& s, const Options& o)
{
  Fit f;
  f.n = s.n;
  if (s.n < 3.0 || s.n < static_cast<double>(o.min_points)) return f;

  const V3 mean = s.mean();
  const M3 cov = s.covariance();
  if (!cov.allFinite()) return f;
  Eigen::SelfAdjointEigenSolver<M3> es(cov);
  if (es.info() != Eigen::Success) return f;
  V3 ev = es.eigenvalues();          // ascending
  const M3 evec = es.eigenvectors();

  f.center = mean;
  f.normal = evec.col(0).normalized();
  f.y_axis = evec.col(1).normalized();
  f.x_axis = evec.col(2).normalized();
  f.d = -f.normal.dot(f.center);

  if (ev(1) < o.min_secondary_eig || ev(2) < o.min_secondary_eig) { f.eig = ev; return f; }
  ev(0) = std::max(ev(0), 0.0);
  f.eig = ev;
  const double den = std::max(ev.sum(), 1e-15);
  f.planarity = ev(0) / den;
  f.radius = static_cast<double>(static_cast<float>(std::sqrt(ev(2))));
  if (!std::isfinite(ev(0))) return f;
  if (!(ev(0) < o.plane_threshold)) return f;

  const double denom1 = ev(0) - ev(1);
  const double denom2 = ev(0) - ev(2);
  const double eps_denom = std::max(1e-8, o.denom_floor_scale * o.plane_threshold);
  if (std::fabs(denom1) < eps_denom || std::fabs(denom2) < eps_denom) {
    f.denom_rejected = true;
    return f;
  }

  const double N = s.n;
  const double inv_N2 = 1.0 / (N * N);
  M3 Vw[3][3], Ww[3];
  for (int a = 0; a < 3; ++a) {
    Ww[a] = s.w[a] * inv_N2;
    for (int b = 0; b < 3; ++b) Vw[a][b] = s.v[a][b] * inv_N2;
  }
  const M3 Scov_w = s.scov * inv_N2;
  auto U = [&](int a, int b) -> M3 {
    return Vw[a][b] - mean(a) * Ww[b] - mean(b) * Ww[a] + mean(a) * mean(b) * Scov_w;
  };
  auto Wc = [&](int a) -> V3 { return (Ww[a] - mean(a) * Scov_w) * f.normal; };

  const M3 M1 = f.normal * f.y_axis.transpose() + f.y_axis * f.normal.transpose();
  const M3 M2 = f.normal * f.x_axis.transpose() + f.x_axis * f.normal.transpose();

  double s_t1t1 = 0.0, s_t2t2 = 0.0, s_t1t2 = 0.0;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      const M3 Uab = U(a, b);
      s_t1t1 += M1.col(a).dot(Uab * M1.col(b));
      s_t2t2 += M2.col(a).dot(Uab * M2.col(b));
      s_t1t2 += M1.col(a).dot(Uab * M2.col(b));
    }
  double s_t1d = 0.0, s_t2d = 0.0;
  for (int a = 0; a < 3; ++a) {
    s_t1d += M1.col(a).dot(Wc(a));
    s_t2d += M2.col(a).dot(Wc(a));
  }

  M3 pv = M3::Zero();
  pv(0, 0) = s_t1t1 / (denom1 * denom1);
  pv(1, 1) = s_t2t2 / (denom2 * denom2);
  pv(0, 1) = pv(1, 0) = s_t1t2 / (denom1 * denom2);
  pv(2, 2) = f.normal.dot(Scov_w * f.normal);
  pv(0, 2) = pv(2, 0) = -s_t1d / denom1;
  pv(1, 2) = pv(2, 1) = -s_t2d / denom2;
  if (!pv.allFinite() || pv.trace() > o.plane_var_ceiling) {
    f.ceiling_rejected = true;
    return f;
  }
  f.plane_var = pv;
  f.valid = true;
  return f;
}

double planeAngleDeg(const V3& na, const V3& nb)
{
  const double c = std::min(1.0, std::max(0.0, std::abs(na.dot(nb))));
  return std::acos(c) * kRad2Deg;
}

double planeOffset(const Fit& a, const Fit& b)
{
  const V3 dc = b.center - a.center;
  return std::max(std::abs(a.normal.dot(dc)), std::abs(b.normal.dot(dc)));
}

double boxGap(const V3& amin, const V3& amax, const V3& bmin, const V3& bmax)
{
  const V3 g = (amin - bmax).cwiseMax(bmin - amax).cwiseMax(0.0);
  return g.norm();
}

// ---------------------------------------------------------------------------------------------------------------------
// SurfaceLayer
// ---------------------------------------------------------------------------------------------------------------------
SurfaceLayer::SurfaceLayer(const Options& o) : o_(o)
{
  const double cs = std::max(o_.cell_size, 1e-6);
  int R_gap = static_cast<int>(std::ceil(o_.merge_gap / cs - 1e-9)) + 1;
  int R = R_gap;
  if (o_.support_test) {
    const int R_sup = 1 + static_cast<int>(std::floor(o_.support_gap / cs + 1e-9));
    R = std::min(R, R_sup);
  }
  R = std::max(1, std::min(R, 8));
  for (int dx = -R; dx <= R; ++dx)
    for (int dy = -R; dy <= R; ++dy)
      for (int dz = -R; dz <= R; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        CellKey k;
        k.x = dx; k.y = dy; k.z = dz;
        offsets_.push_back(k);
      }
}

Summary SurfaceLayer::summary() const
{
  Summary m;
  m.cells = cells_.size();
  m.surfaces = surfaces_.size();
  for (const Cell& c : cells_) {
    m.points_total += c.st.n;
    if (c.fit.valid) {
      ++m.cells_valid;
      m.points_valid += c.st.n;
    }
  }
  for (const auto& kv : surfaces_) {
    std::size_t valid_children = 0;
    for (std::uint64_t cid : kv.second.children)
      if (cellById(cid).fit.valid) ++valid_children;
    m.largest = std::max(m.largest, valid_children);
    if (valid_children >= 2) {
      ++m.surfaces_multi;
      m.patches_in_multi += valid_children;
    }
  }
  return m;
}

CellKey SurfaceLayer::keyOf(const V3& p) const
{
  CellKey k;
  k.x = static_cast<int>(std::floor(p.x() / o_.cell_size));
  k.y = static_cast<int>(std::floor(p.y() / o_.cell_size));
  k.z = static_cast<int>(std::floor(p.z() / o_.cell_size));
  return k;
}

const Cell* SurfaceLayer::findCell(const CellKey& k) const
{
  auto it = index_.find(k);
  return it == index_.end() ? nullptr : &cells_[it->second];
}

const Surface* SurfaceLayer::findSurface(std::uint64_t id) const
{
  auto it = surfaces_.find(id);
  return it == surfaces_.end() ? nullptr : &it->second;
}

const Fit* SurfaceLayer::planeFor(const Cell& c, const void** owner) const
{
  if (!c.fit.valid) return nullptr;
  const Surface* s = findSurface(c.surface_id);
  if (s && s->children.size() > 1 && s->fit.valid) {
    if (owner) *owner = s;
    return &s->fit;
  }
  if (owner) *owner = &c;
  return &c.fit;
}

std::vector<std::uint64_t> SurfaceLayer::surfaceIds() const
{
  std::vector<std::uint64_t> ids;
  ids.reserve(surfaces_.size());
  for (const auto& kv : surfaces_) ids.push_back(kv.first);
  std::sort(ids.begin(), ids.end());
  return ids;
}

Cell& SurfaceLayer::getCell(const CellKey& k)
{
  auto it = index_.find(k);
  if (it != index_.end()) return cells_[it->second];
  cells_.emplace_back();
  Cell& c = cells_.back();
  c.id = static_cast<std::uint64_t>(cells_.size());
  c.key = k;
  c.st.setRef(V3((k.x + 0.5) * o_.cell_size, (k.y + 0.5) * o_.cell_size, (k.z + 0.5) * o_.cell_size));
  c.surface_id = c.id;
  Surface s;
  s.id = c.id;
  s.children = {c.id};
  surfaces_.emplace(c.id, std::move(s));
  index_.emplace(k, static_cast<std::uint32_t>(cells_.size() - 1));
  return c;
}

void SurfaceLayer::addPoint(const V3& p, const M3& sensor_cov)
{
  Cell& c = getCell(keyOf(p));
  c.st.add(p, sensor_cov);
  c.bb_min = c.bb_min.cwiseMin(p);
  c.bb_max = c.bb_max.cwiseMax(p);
  if (!c.dirty) {
    c.dirty = true;
    dirty_.push_back(c.id);
  }
}

void SurfaceLayer::recomputeSurface(Surface& s)
{
  s.combined = SurfaceStats();
  for (std::uint64_t cid : s.children) {
    const Cell& c = cellById(cid);
    if (c.fit.valid) s.combined.merge(c.st);
  }
  s.fit = fitStats(s.combined, o_);
}

SurfaceLayer::Consistency SurfaceLayer::evaluate(const SurfaceStats& u, const std::vector<std::uint64_t>& children,
                                                 bool invalid_violates) const
{
  Consistency r;
  const Fit f = fitStats(u, o_);
  r.planarity = f.planarity;
  const bool degraded = !f.valid || f.planarity > o_.split_planarity_max;
  for (std::uint64_t cid : children) {
    const Cell& c = cellById(cid);
    double score, ang = 0.0, off = 0.0;
    if (!c.fit.valid) {
      if (!invalid_violates) continue;
      score = std::numeric_limits<double>::infinity();
    } else if (!f.valid) {
      score = 0.0;   // no reference plane; the degraded flag decides
    } else {
      ang = planeAngleDeg(c.fit.normal, f.normal);
      off = std::abs(f.normal.dot(c.fit.center - f.center));
      score = std::max(ang / std::max(o_.merge_angle_deg, 1e-12), off / std::max(o_.merge_offset, 1e-12));
    }
    if (score > r.worst_score) {
      r.worst_score = score;
      r.worst = cid;
      r.worst_ang = ang;
      r.worst_off = off;
    }
  }
  r.ok = !degraded && r.worst_score <= 1.0;
  return r;
}

bool SurfaceLayer::tryMerge(std::uint64_t ida, std::uint64_t idb, UpdateStats& us)
{
  if (ida == idb) return false;
  const Surface& A = surfaces_.at(ida);
  const Surface& B = surfaces_.at(idb);
  SurfaceStats u = A.combined;
  u.merge(B.combined);
  std::vector<std::uint64_t> ch = A.children;
  ch.insert(ch.end(), B.children.begin(), B.children.end());
  std::sort(ch.begin(), ch.end());
  if (!evaluate(u, ch, /*invalid_violates=*/false).ok) {
    ++us.combined_rejects;
    return false;
  }
  const std::uint64_t keep = std::min(ida, idb), drop = std::max(ida, idb);
  for (std::uint64_t cid : ch) cells_[cid - 1].surface_id = keep;
  surfaces_.erase(drop);            // invalidates A or B: not used below
  Surface& K = surfaces_.at(keep);
  K.children = std::move(ch);
  K.combined = u;
  K.fit = fitStats(u, o_);
  ++us.merges;
  return true;
}

void SurfaceLayer::reevaluateSplits(std::uint64_t sid, UpdateStats& us)
{
  while (true) {
    auto it = surfaces_.find(sid);
    if (it == surfaces_.end() || it->second.children.size() < 2) break;
    Surface& sa = it->second;
    const Consistency c = evaluate(sa.combined, sa.children, /*invalid_violates=*/true);
    if (c.ok || c.worst == 0) break;
    const std::uint64_t w = c.worst;
    std::vector<std::uint64_t> remaining;
    for (std::uint64_t cid : sa.children)
      if (cid != w) remaining.push_back(cid);
    SurfaceStats rebuilt;
    for (std::uint64_t cid : remaining)
      if (cellById(cid).fit.valid) rebuilt.merge(cellById(cid).st);
    const std::uint64_t rest_id = (w == sid) ? remaining.front() : sid;
    Surface rest;
    rest.id = rest_id;
    rest.children = remaining;
    rest.combined = rebuilt;
    rest.fit = fitStats(rebuilt, o_);
    surfaces_.erase(it);            // sa is dangling from here on
    for (std::uint64_t cid : remaining) cells_[cid - 1].surface_id = rest_id;
    surfaces_[rest_id] = std::move(rest);
    Cell& wc = cells_[w - 1];
    Surface single;
    single.id = w;
    single.children = {w};
    if (wc.fit.valid) single.combined = wc.st;
    single.fit = fitStats(single.combined, o_);
    wc.surface_id = w;
    surfaces_[w] = std::move(single);
    ++us.splits;
    sid = rest_id;
  }
}

UpdateStats SurfaceLayer::update()
{
  UpdateStats us;
  std::sort(dirty_.begin(), dirty_.end());
  us.dirty_cells = dirty_.size();

  // 1. refit dirty cells
  for (std::uint64_t id : dirty_) {
    Cell& c = cells_[id - 1];
    c.fit = fitStats(c.st, o_);
    if (c.fit.valid) ++us.cells_valid_dirty;
    if (c.fit.denom_rejected) ++us.denom_rejected;
  }

  // 2. recompute the surfaces that contain a dirty cell
  {
    std::vector<std::uint64_t> touched;
    for (std::uint64_t id : dirty_) touched.push_back(cells_[id - 1].surface_id);
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    for (std::uint64_t sid : touched) recomputeSurface(surfaces_.at(sid));
  }

  // 3. merge candidates: pairs with at least one dirty cell
  for (std::uint64_t id : dirty_) {
    Cell& a = cells_[id - 1];
    if (!a.fit.valid) continue;
    for (const CellKey& off : offsets_) {
      CellKey kb;
      kb.x = a.key.x + off.x; kb.y = a.key.y + off.y; kb.z = a.key.z + off.z;
      auto it = index_.find(kb);
      if (it == index_.end()) continue;
      Cell& b = cells_[it->second];
      if (!b.fit.valid) continue;
      if (b.dirty && b.id < a.id) continue;          // that pair is handled when b is the current cell
      if (a.surface_id == b.surface_id) continue;
      ++us.pair_tests;
      if (planeAngleDeg(a.fit.normal, b.fit.normal) > o_.merge_angle_deg ||
          planeOffset(a.fit, b.fit) > o_.merge_offset ||
          (a.fit.center - b.fit.center).norm() > o_.merge_gap) {
        ++us.gate_rejects;
        continue;
      }
      if (o_.support_test && boxGap(a.bb_min, a.bb_max, b.bb_min, b.bb_max) > o_.support_gap) {
        ++us.support_rejects;
        continue;
      }
      tryMerge(a.surface_id, b.surface_id, us);
    }
  }

  // 4. split re-check for the surfaces that now contain a dirty cell
  {
    std::vector<std::uint64_t> touched;
    for (std::uint64_t id : dirty_) touched.push_back(cells_[id - 1].surface_id);
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    for (std::uint64_t sid : touched) reevaluateSplits(sid, us);
  }

  for (std::uint64_t id : dirty_) cells_[id - 1].dirty = false;
  dirty_.clear();
  merges_ += us.merges;
  splits_ += us.splits;
  return us;
}

}  // namespace surface
}  // namespace livo_recon
