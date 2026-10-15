// R88: the binned map type below the map level: VoxelPlane::refitFromBins() and VoxelNode's binned debiased branch, driven directly (links src/lio/voxelnode.cpp and voxelplane.cpp; no ROS).
// Planner sandbox on synthetic points: NOT evidence about any bag. Identities and rules only (rule 3.1).
//   BP1 average + count on bins of identical points gives the same fit as VoxelPlane::addPoints() on the raw points (eigenvalues, plane variance, centre, normal, d).
//   BP2 (R95: the uniform weight was removed) the shrunk covariance gives a finite plane variance on the same bins.
//   BP3 refit = false keeps the previous fit and moves only pointsSize() (occupied bins).
//   BN1 the admission rule counts BINS: 12 returns in 3 bins do not fit at min_init 5; the default (raw) node fits them. 6 bins fit.
//   BN2 (R92: touched count) returns into bins that are already open count once each: 4 open bins hit twice do not refit at min_update 5, 6 distinct bins touched since the fit do; 4 new bins do not, a 5th does.
//   BN3 the bootstrap frame (global frame 0) is exempt from min_update only: one new bin refits an admitted voxel there, and min_init still holds.
//   BN4 30 returns of one beam are one bin; the counters say so.
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "livo_recon/lio/voxelnode.h"
#include "livo_recon/lio/voxelplane.h"

using namespace livo_recon;
static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

static VoxelOptsPtr makeOpts(bool binned, int cov_mode, int min_init, int min_update)
{
  auto o = std::make_shared<VoxelOpts>();
  o->plane_threshold = 0.01;
  o->plane_fit_mode = "debiased";
  o->plane_var_mode = "eigengap";
  o->weight_sigma_r2 = 4e-4;
  o->min_init_points = min_init;
  o->min_update_points = min_update;
  o->voxel_size = 0.25;
  o->max_layer = 0;
  o->binned_enabled = binned;
  o->binned_bin_fraction = 0.2;
  o->binned_cov_mode = cov_mode;
  return o;
}

// A planar patch (z = 0.01 x - 0.02 y + 0.03) sampled one point per 0.05 m bin, `n` bins starting at index `first` (bin k of a 25-bin 5 x 5 grid, visited in the order k*7 mod 25), `reps` identical returns per bin.
static std::vector<PointXYZCov> patch(int first, int n, int reps, double cov_scale, bool vary_cov)
{
  std::vector<PointXYZCov> v;
  for (int k = first; k < first + n; ++k) {
    const int kk = (k * 7) % 25;   // spreads the first few bins over both directions of the 5 x 5 grid
    const double x = -0.1 + 0.05 * (kk % 5) + 0.0125, y = -0.1 + 0.05 * (kk / 5) + 0.0125;
    for (int r = 0; r < reps; ++r) {
      PointXYZCov p;
      p.point = V3D(x, y, 0.01 * x - 0.02 * y + 0.03);
      p.sensor_cov = M3D::Identity() * cov_scale * (vary_cov ? (1.0 + 0.3 * r + 0.1 * (k % 3)) : 1.0);
      p.pos_cov = M3D::Zero();
      v.push_back(p);
    }
  }
  return v;
}

int main()
{
  // ---------------------------------------------------------------- plane level
  {
    const auto pts = patch(0, 25, 3, 1e-6, true);
    auto o = makeOpts(false, 0, 3, 1);
    VoxelPlane raw(o);
    raw.addPoints(pts, -1, 1, true, 1, 0);
    CHECK(raw.isPlane());

    binned::BinSet bins(0.05);
    for (const auto& p : pts) bins.add(p.point, p.sensor_cov, true);
    CHECK(bins.numBins() == 25);
    binned::Options bo;   // average + count
    VoxelPlane vb(o);
    vb.refitFromBins(bins, bo, pts, 1, 1, 0, true);
    CHECK(vb.isPlane());
    CHECK(vb.pointsSize() == 25);
    CHECK((vb.eigenValues() - raw.eigenValues()).cwiseAbs().maxCoeff() < 1e-9);   // within-bin scatter is zero here (identical returns), so the sums are the raw sums
    CHECK((vb.planeVar() - raw.planeVar()).cwiseAbs().maxCoeff() < 1e-9 * (1.0 + raw.planeVar().cwiseAbs().maxCoeff()));
    VoxelPlaneSnapshot sr, sb;
    raw.fillSnapshot(sr); vb.fillSnapshot(sb);
    CHECK((sr.center - sb.center).norm() < 1e-9);
    CHECK((sr.normal - sb.normal).norm() < 1e-9);
    CHECK(std::fabs(sr.d - sb.d) < 1e-9);

    // BP2
    binned::Options bs; bs.cov_mode = binned::CovMode::Shrunk;
    VoxelPlane vs(o);
    vs.refitFromBins(bins, bs, pts, 1, 1, 0, true);
    CHECK(vs.isPlane());
    CHECK(std::isfinite(vs.planeVar().trace()));

    // BP3
    VoxelPlane k(o);
    binned::BinSet half(0.05);
    const auto first12 = patch(0, 12, 1, 1e-6, false);
    for (const auto& p : first12) half.add(p.point, p.sensor_cov, true);
    k.refitFromBins(half, bo, first12, 1, 1, 0, true);
    CHECK(k.isPlane());
    const V3D e_before = k.eigenValues();
    const M3D pv_before = k.planeVar();
    const auto more = patch(12, 6, 1, 1e-6, false);
    for (const auto& p : more) half.add(p.point, p.sensor_cov, true);
    k.refitFromBins(half, bo, more, 1, 2, 0, false);
    CHECK(k.isPlane());
    CHECK(k.pointsSize() == 18);
    CHECK((k.eigenValues() - e_before).cwiseAbs().maxCoeff() == 0.0);
    CHECK((k.planeVar() - pv_before).cwiseAbs().maxCoeff() == 0.0);
  }

  // ---------------------------------------------------------------- node level
  std::vector<PlaneUpdate> upd;
  {   // BN1: bins vs raw admission
    auto stats = std::make_shared<VoxelStats>();
    setBootstrapInsertion(false);
    setCurrentFrame(1);
    VoxelNode nb(makeOpts(true, 0, 5, 5), stats, 0, V3D(0, 0, 0));
    nb.insertPoints(patch(0, 4, 3, 1e-6, false), upd);          // 12 returns, 4 bins
    CHECK(!nb.hasConvergedPlane());
    CHECK(stats->binned_points.load() == 12);
    CHECK(stats->binned_new_bins.load() == 4);
    CHECK(stats->binned_fits.load() == 0);
    setCurrentFrame(2);
    nb.insertPoints(patch(4, 2, 1, 1e-6, false), upd);          // 2 more bins: 6 occupied > 5
    CHECK(stats->binned_new_bins.load() == 6);
    CHECK(stats->binned_fits.load() == 1);
    CHECK(nb.hasConvergedPlane());

    auto stats_raw = std::make_shared<VoxelStats>();
    setCurrentFrame(1);
    VoxelNode nr(makeOpts(false, 0, 5, 5), stats_raw, 0, V3D(0, 0, 0));
    nr.insertPoints(patch(0, 4, 3, 1e-6, false), upd);          // the same 12 returns, raw: 12 >= 5 raw points, so the default node fits at once
    CHECK(nr.hasConvergedPlane());
    CHECK(stats_raw->binned_points.load() == 0);
  }
  {   // BN2: updates count unique bins touched since the fit (R92; before R92 this block expected the opened count)
    auto stats = std::make_shared<VoxelStats>();
    setBootstrapInsertion(false);
    setCurrentFrame(1);
    VoxelNode n(makeOpts(true, 0, 5, 5), stats, 0, V3D(0, 0, 0));
    n.insertPoints(patch(0, 8, 1, 1e-6, false), upd);           // 8 bins: fitted
    CHECK(stats->binned_fits.load() == 1);
    setCurrentFrame(2);
    n.insertPoints(patch(0, 4, 2, 1e-6, false), upd);           // 8 returns into 4 open bins: 4 unique bins touched < min_update 5 (the returns count once per bin)
    CHECK(stats->binned_fits.load() == 1);
    CHECK(stats->binned_new_bins.load() == 8);
    setCurrentFrame(3);
    n.insertPoints(patch(2, 4, 1, 1e-6, false), upd);           // bins 2..5 (all open): unique since the fit {0,1,2,3,4,5} = 6 >= 5, no bin opened
    CHECK(stats->binned_fits.load() == 2);
    CHECK(stats->binned_new_bins.load() == 8);
    CHECK(stats->binned_touched_at_fit.load() == 8 + 6);
    setCurrentFrame(4);
    n.insertPoints(patch(8, 4, 1, 1e-6, false), upd);           // 4 new bins < min_update 5
    CHECK(stats->binned_fits.load() == 2);
    setCurrentFrame(5);
    n.insertPoints(patch(12, 1, 1, 1e-6, false), upd);          // the 5th unique bin since the fit
    CHECK(stats->binned_fits.load() == 3);
    CHECK(stats->binned_touched_at_fit.load() == 8 + 6 + 5);
    CHECK(stats->binned_new_bins.load() == 13);
  }
  {   // BN3: bootstrap frame
    auto stats = std::make_shared<VoxelStats>();
    setBootstrapInsertion(true);
    setCurrentFrame(0);
    VoxelNode n(makeOpts(true, 0, 5, 5), stats, 0, V3D(0, 0, 0));
    n.insertPoints(patch(0, 4, 1, 1e-6, false), upd);           // 4 bins at min_init 5: not even in the bootstrap frame
    CHECK(stats->binned_fits.load() == 0);
    n.insertPoints(patch(4, 2, 1, 1e-6, false), upd);           // 6 bins: first fit
    CHECK(stats->binned_fits.load() == 1);
    n.insertPoints(patch(6, 1, 1, 1e-6, false), upd);           // 1 new bin < min_update 5, but frame 0 is exempt
    CHECK(stats->binned_fits.load() == 2);
    setBootstrapInsertion(false);
  }
  {   // BN4
    auto stats = std::make_shared<VoxelStats>();
    setBootstrapInsertion(true);
    setCurrentFrame(0);
    VoxelNode n(makeOpts(true, 0, 5, 5), stats, 0, V3D(0, 0, 0));
    n.insertPoints(patch(0, 1, 30, 1e-6, false), upd);
    CHECK(stats->binned_points.load() == 30);
    CHECK(stats->binned_new_bins.load() == 1);
    CHECK(stats->binned_fits.load() == 0);
    setBootstrapInsertion(false);
  }
  std::printf("test_binned_node_r88 %s\n", g_fail ? "FAIL" : "PASS");
  return g_fail ? 1 : 0;
}
