// R51: deterministic test for VoxelNode's per-(node, bootstrap observation)
// sufficient-statistics accumulator (BootstrapObservationStatsRow: N,
// sum_p, sum_pp) added by R51_phase3_bootstrap_observation_moments.patch.
// Constructs a bare VoxelNode directly (no ROS, no NodeContext -- its
// constructor takes only plain, default-constructible option/stats
// structs) and calls the real production insertPoints()/
// appendBootstrapObservationSupport() -- does not reimplement or duplicate
// any plane-fitting math.

#include "livo_recon/lio/voxelnode.h"

#include <cstdio>
#include <cmath>
#include <vector>

using namespace livo_recon;

namespace {

bool fail(const char* msg)
{
  std::fprintf(stderr, "[FAIL] %s\n", msg);
  return false;
}

bool nearlyEqual(double a, double b, double tol = 1e-9)
{
  return std::abs(a - b) <= tol;
}

}  // namespace

int main()
{
  auto opts = std::make_shared<VoxelOpts>();
  auto stats = std::make_shared<VoxelStats>();
  setCurrentFrame(0);
  setBootstrapInsertion(false);

  VoxelNode node(opts, stats, /*layer=*/0, /*center=*/V3D(0, 0, 0));

  // Two distinct bootstrap observations (ids 0 and 1) plus one ordinary
  // live point (id -1, must NOT be counted at all), all landing in the
  // same voxel cell so they all reach the SAME node's insertPoints() call.
  std::vector<PointXYZCov> points;
  auto addPoint = [&](double x, double y, double z, int obs_id) {
    PointXYZCov p;
    p.point = V3D(x, y, z);
    p.sensor_cov = M3D::Zero();
    p.bootstrap_observation_id = obs_id;
    points.push_back(p);
  };
  // All points are tight, near-coplanar clusters around the origin
  // (millimeter-scale offsets) so this single insertPoints() call cannot
  // trigger octree subdivision (passToChildren()) -- the test is about
  // the per-node accumulator, not the plane-fit/subdivision decision, so
  // it deliberately avoids exercising that path at all.
  // Observation 0: three known points.
  addPoint(0.0010, 0.0020, 0.0000, 0);
  addPoint(0.0020, 0.0000, 0.0000, 0);
  addPoint(0.0000, 0.0040, 0.0000, 0);
  // Observation 1: two known points.
  addPoint(0.0050, 0.0050, 0.0000, 1);
  addPoint(-0.0010, -0.0010, 0.0000, 1);
  // Ordinary live point (bootstrap_observation_id == -1 default) -- must
  // not appear in the support accumulator at all.
  addPoint(0.0030, 0.0030, 0.0000, -1);

  std::vector<PlaneUpdate> updates;
  node.insertPoints(points, updates);

  std::vector<BootstrapObservationStatsRow> rows;
  node.appendBootstrapObservationSupport(VoxelKey{0, 0, 0}, rows);

  if (rows.size() != 2) return fail("expected exactly 2 observation rows (ids 0 and 1)"), 1;

  const BootstrapObservationStatsRow* r0 = nullptr;
  const BootstrapObservationStatsRow* r1 = nullptr;
  for (const auto& r : rows) {
    if (r.observation_id == 0) r0 = &r;
    if (r.observation_id == 1) r1 = &r;
  }
  if (!r0 || !r1) return fail("missing expected observation_id in output rows"), 1;

  // Observation 0: hand-computed expected sufficient statistics.
  const V3D expected_sum_p0(0.0030, 0.0060, 0.0000);
  M3D expected_sum_pp0 = M3D::Zero();
  {
    const V3D pts[3] = {V3D(0.0010, 0.0020, 0.0000), V3D(0.0020, 0.0000, 0.0000), V3D(0.0000, 0.0040, 0.0000)};
    for (const auto& p : pts) expected_sum_pp0.noalias() += p * p.transpose();
  }

  if (r0->point_count != 3) return fail("observation 0: wrong N"), 1;
  if (!(r0->sum_p - expected_sum_p0).isZero(1e-9)) return fail("observation 0: wrong sum_p"), 1;
  if (!(r0->sum_pp - expected_sum_pp0).isZero(1e-9)) return fail("observation 0: wrong sum_pp"), 1;

  // Observation 1.
  const V3D expected_sum_p1(0.0040, 0.0040, 0.0000);
  M3D expected_sum_pp1 = M3D::Zero();
  {
    const V3D pts[2] = {V3D(0.0050, 0.0050, 0.0000), V3D(-0.0010, -0.0010, 0.0000)};
    for (const auto& p : pts) expected_sum_pp1.noalias() += p * p.transpose();
  }
  if (r1->point_count != 2) return fail("observation 1: wrong N"), 1;
  if (!(r1->sum_p - expected_sum_p1).isZero(1e-9)) return fail("observation 1: wrong sum_p"), 1;
  if (!(r1->sum_pp - expected_sum_pp1).isZero(1e-9)) return fail("observation 1: wrong sum_pp"), 1;

  // Downstream reconstruction the offline analysis script relies on: mean
  // and centered scatter from (N, sum_p, sum_pp) alone.
  const V3D mean0 = r0->sum_p / r0->point_count;
  const M3D scatter0 = r0->sum_pp / r0->point_count - mean0 * mean0.transpose();
  const V3D expected_mean0 = expected_sum_p0 / 3.0;
  const M3D expected_scatter0 = expected_sum_pp0 / 3.0 - expected_mean0 * expected_mean0.transpose();
  if (!(mean0 - expected_mean0).isZero(1e-9)) return fail("observation 0: wrong reconstructed mean"), 1;
  if (!(scatter0 - expected_scatter0).isZero(1e-9)) return fail("observation 0: wrong reconstructed centered scatter"), 1;

  // sum_pp must be exactly symmetric (raw second moment, p*p^T).
  if (!(r0->sum_pp - r0->sum_pp.transpose()).isZero(1e-12)) return fail("observation 0: sum_pp not symmetric"), 1;
  if (!(r1->sum_pp - r1->sum_pp.transpose()).isZero(1e-12)) return fail("observation 1: sum_pp not symmetric"), 1;

  std::printf("[PASS] test_bootstrap_observation_moments\n");
  return 0;
}
