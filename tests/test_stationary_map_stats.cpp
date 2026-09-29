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

  return 0;
}
