#include "livo_recon/map/stationary/stationary_map.h"
#include <cassert>
#include <cmath>
#include <vector>

// R52: extends the supplied test to explicitly verify the additive
// PlaneStats accumulator reproduces a BATCH (direct, non-incremental)
// mean/covariance/PCA computation on the same deterministic synthetic
// point cloud, not just that merge(a,b) reproduces the union's own
// re-accumulated sum (which the original assertions already covered).
static livo_recon::stationary_map::PlaneFit batchFit(const std::vector<livo_recon::stationary_map::V3>& pts) {
  using namespace livo_recon::stationary_map;
  V3 mean = V3::Zero();
  for (const auto& p : pts) mean += p;
  mean /= double(pts.size());
  M3 cov = M3::Zero();
  for (const auto& p : pts) { V3 d = p - mean; cov += d * d.transpose(); }
  cov /= double(pts.size());
  Eigen::SelfAdjointEigenSolver<M3> es(cov);
  PlaneFit f;
  f.eigenvalues = es.eigenvalues();
  f.center = mean;
  f.normal = es.eigenvectors().col(0).normalized();
  f.d = -f.normal.dot(mean);
  f.planarity = f.eigenvalues[0] / std::max(f.eigenvalues.sum(), 1e-15);
  f.valid = true;
  return f;
}

namespace { void checkFamiliesBehaveDistinctly(); }

int main(){
  using namespace livo_recon::stationary_map;
  PlaneStats a,b,all;
  std::vector<V3> raw;
  for (int i = 0; i < 20; ++i) {
    V3 p(i*.01, 2*i*.01, 1.0);
    raw.push_back(p);
    all.add(p);
    (i < 10 ? a : b).add(p);
  }
  a.merge(b);
  assert(a.n == all.n);
  assert((a.sum - all.sum).norm() < 1e-12);
  assert((a.sum_outer - all.sum_outer).norm() < 1e-12);

  auto f = fitIncrementalPca(a, 1.0);
  assert(f.valid);

  // Cross-check against an independent batch computation on the raw points.
  auto batch = batchFit(raw);
  assert((f.center - batch.center).norm() < 1e-9);
  assert((f.eigenvalues - batch.eigenvalues).norm() < 1e-9);
  // Normal sign is only defined up to +/-1 by PCA -- compare via
  // orientation-invariant dot product magnitude.
  assert(std::abs(std::abs(f.normal.dot(batch.normal)) - 1.0) < 1e-9);
  assert(std::abs(f.planarity - batch.planarity) < 1e-9);

  checkFamiliesBehaveDistinctly();

  return 0;
}

// R52: behavioral smoke checks for the 5 completed backend families --
// not exhaustive validation, but enough to catch "the flag is unused"
// class of bug this round's own instructions warned about.
namespace {
void checkFamiliesBehaveDistinctly() {
  using namespace livo_recon::stationary_map;
  Options o; o.leaf = 1.0; o.robust_reservoir = 20; o.robust_ransac_dist = 0.03; o.robust_ransac_iters = 200;

  // A clean planar patch (z=1 plane) with a handful of gross outliers mixed
  // in -- robust_voxel must reject the outliers (fewer accepted points than
  // total inserted), while incremental_pca has no rejection mechanism at
  // all (accepts everything, degrading its own fit).
  std::vector<V3> pts;
  for (int i = 0; i < 30; ++i) pts.emplace_back(i * 0.01, 0.0, 1.0 + (i % 3 == 0 ? 0.001 : -0.001));
  for (int i = 0; i < 8; ++i) pts.emplace_back(0.1, 0.1, 5.0 + i);  // gross outliers, same cell

  auto robust = makeBackend("robust_voxel", o);
  robust->insert(0, pts);
  auto snap_r = robust->snapshot();
  assert(!snap_r.data.empty());
  std::uint64_t total_accepted = 0;
  for (const auto& p : snap_r.data) total_accepted += p.stats.n;
  assert(total_accepted < pts.size());  // outliers must have been rejected, not all 38 accepted

  auto plain = makeBackend("incremental_pca", o);
  plain->insert(0, pts);
  auto snap_p = plain->snapshot();
  std::uint64_t plain_accepted = 0;
  for (const auto& p : snap_p.data) plain_accepted += p.stats.n;
  assert(plain_accepted == pts.size());  // no rejection mechanism -- everything lands in one cell's stats

  // Merge: two adjacent, coplanar patches (same z=1 plane, different xy
  // cells) must end up under ONE surface_id.
  Options om = o; om.leaf = 0.5;
  auto merge_be = makeBackend("mergeable_voxel", om);
  std::vector<V3> a, b;
  for (int i = 0; i < 20; ++i) { a.emplace_back(i * 0.01, 0.1, 1.0); b.emplace_back(i * 0.01, 0.6, 1.0); }
  merge_be->insert(0, a); merge_be->insert(0, b);
  auto snap_m = merge_be->snapshot();
  assert(snap_m.patches >= 2);
  assert(snap_m.surfaces < snap_m.patches);  // the two coplanar patches must have merged into fewer surfaces
  assert(snap_m.merges >= 1);

  // gaussian_surface: force a merge, then feed enough new points into one
  // child to rotate its OWN fit away from the shared plane -- it must
  // eventually split back out (splits_/unmerges_ > 0), which mergeable_
  // voxel (no reversibility) never does regardless of how it degrades.
  Options og = om; og.split_planarity_max = 0.05;
  auto gauss = makeBackend("gaussian_surface", og);
  gauss->insert(0, a); gauss->insert(0, b);
  std::vector<V3> tilt;
  for (int i = 0; i < 40; ++i) tilt.emplace_back(i * 0.01, 0.1, 1.0 + i * 0.05);  // steadily tilts patch a's own cell
  gauss->insert(0, tilt);
  auto snap_g = gauss->snapshot();
  assert(snap_g.splits + snap_g.unmerges > 0);
}
}  // namespace
