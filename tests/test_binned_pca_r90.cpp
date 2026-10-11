// R90: the pca fit on the binned map type: BinSet::representatives() and VoxelNode's binned pca branch (links src/lio/voxelnode.cpp and voxelplane.cpp; no ROS).
// Planner sandbox on synthetic points: NOT evidence about any bag. Identities and rules only (rule 3.1).
//   PC1 representatives(): one point per occupied bin (the bin mean), weight k (count) or 1 (uniform), covariance C/k (average) or C/k^2 (shrunk).
//   PC2 VoxelPlane::update() on the representatives (average + count) equals update() on the raw returns when the returns of a bin are identical in eigenvalues, centre and normal. The plane variance is three times the raw one for bins of
//       three returns (average = the returns of a bin are fully correlated, so a bin counts as one noise draw with the weight of three); with shrunk + count it equals the raw plane variance (a mean of three independent returns).
//   PC3 the node counts unique BINS touched for min_init / min_update (R92): 12 returns in 4 bins do not fit at min_init 5; 2 more bins do; the raw pca node (binned off) fits the same 12 returns at once.
//   PC4 (R92: touched count) 4 open bins hit twice do not refit at min_update 5; 6 distinct bins touched since the fit do; 4 new bins do not, a 5th does.
//   PC5 points/max counts occupied bins: with max 6 a voxel of 8 bins locks after its first fit and takes no more fits; with max 50 it keeps fitting.
//   PC8 (step 2) the older pca-only bin path is gone: VoxelOpts has no use_bins, bin_size_fraction, bin_weight_mode_fit or bin_weight_mode_var member (compile-time check).
//   PC7 the node passes the configured modes to the fit: the plane covariance of the fitted node differs between count and uniform weight and between average and shrunk covariance.
//   PC6 on bins of unequal counts the pca plane variance is larger under count weight than under uniform weight (the weights concentrate on the few crowded bins: sum w^2 / (sum w)^2 is larger).
#include <cmath>
#include <cstdio>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "livo_recon/lio/voxelnode.h"
#include "livo_recon/lio/voxelplane.h"

using namespace livo_recon;
static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

// PC8: detectors for the removed members (a member that exists makes the static_asserts below fail to compile).
template <class T, class = void> struct HasUseBins : std::false_type {};
template <class T> struct HasUseBins<T, std::void_t<decltype(std::declval<T&>().use_bins)>> : std::true_type {};
template <class T, class = void> struct HasBinSizeFraction : std::false_type {};
template <class T> struct HasBinSizeFraction<T, std::void_t<decltype(std::declval<T&>().bin_size_fraction)>> : std::true_type {};
template <class T, class = void> struct HasBinWeightFit : std::false_type {};
template <class T> struct HasBinWeightFit<T, std::void_t<decltype(std::declval<T&>().bin_weight_mode_fit)>> : std::true_type {};
template <class T, class = void> struct HasBinWeightVar : std::false_type {};
template <class T> struct HasBinWeightVar<T, std::void_t<decltype(std::declval<T&>().bin_weight_mode_var)>> : std::true_type {};
static_assert(!HasUseBins<VoxelOpts>::value, "VoxelOpts::use_bins was removed in R90");
static_assert(!HasBinSizeFraction<VoxelOpts>::value, "VoxelOpts::bin_size_fraction was removed in R90 (the binned map uses binned_bin_fraction)");
static_assert(!HasBinWeightFit<VoxelOpts>::value, "VoxelOpts::bin_weight_mode_fit was removed in R90");
static_assert(!HasBinWeightVar<VoxelOpts>::value, "VoxelOpts::bin_weight_mode_var was removed in R90");

static VoxelOptsPtr makeOpts(bool binned, int cov_mode, int weight_mode, int min_init, int min_update, int max_points)
{
  auto o = std::make_shared<VoxelOpts>();
  o->plane_threshold = 0.01;
  o->plane_fit_mode = "pca";
  o->plane_var_mode = "eigengap";
  o->weight_sigma_r2 = 4e-4;
  o->min_init_points = min_init;
  o->min_update_points = min_update;
  o->max_points = max_points;
  o->voxel_size = 0.25;
  o->max_layer = 0;
  o->binned_enabled = binned;
  o->binned_bin_fraction = 0.2;
  o->binned_cov_mode = cov_mode;
  o->binned_weight_mode = weight_mode;
  return o;
}

// A planar patch (z = 0.01 x - 0.02 y + 0.03) sampled one point per 0.05 m bin, `n` bins starting at index `first` (bin k of a 25-bin 5 x 5 grid, visited in the order k*7 mod 25), `reps` identical returns per bin.
static std::vector<PointXYZCov> patch(int first, int n, int reps, double cov_scale)
{
  std::vector<PointXYZCov> v;
  for (int k = first; k < first + n; ++k) {
    const int kk = (k * 7) % 25;
    const double x = -0.1 + 0.05 * (kk % 5) + 0.0125, y = -0.1 + 0.05 * (kk / 5) + 0.0125;
    for (int r = 0; r < reps; ++r) {
      PointXYZCov p;
      p.point = V3D(x, y, 0.01 * x - 0.02 * y + 0.03);
      p.sensor_cov = M3D::Identity() * cov_scale;
      p.pos_cov = M3D::Zero();
      v.push_back(p);
    }
  }
  return v;
}

static std::vector<PointXYZCov> toPoints(const binned::Reps& r)
{
  std::vector<PointXYZCov> v(r.p.size());
  for (std::size_t i = 0; i < r.p.size(); ++i) { v[i].point = r.p[i]; v[i].sensor_cov = r.cov[i]; v[i].pos_cov = M3D::Zero(); }
  return v;
}

int main()
{
  {   // PC1
    binned::BinSet bins(0.05);
    const auto pts = patch(0, 4, 3, 2e-6);
    for (const auto& p : pts) bins.add(p.point, p.sensor_cov, true);
    binned::Options avg_count;                                       // average + count
    const binned::Reps a = bins.representatives(avg_count);
    CHECK(a.p.size() == 4 && a.cov.size() == 4 && a.w.size() == 4);
    for (std::size_t i = 0; i < 4; ++i) { CHECK(a.w[i] == 3.0); CHECK(std::fabs(a.cov[i](0, 0) - 2e-6) < 1e-18); }
    binned::Options sh_un; sh_un.cov_mode = binned::CovMode::Shrunk; sh_un.weight_mode = binned::WeightMode::Uniform;
    const binned::Reps b = bins.representatives(sh_un);
    for (std::size_t i = 0; i < 4; ++i) { CHECK(b.w[i] == 1.0); CHECK(std::fabs(b.cov[i](0, 0) - 2e-6 / 3.0) < 1e-18); CHECK((a.p[i] - b.p[i]).norm() == 0.0); }
    CHECK((a.p[0] - pts[0].point).norm() < 1e-15);                   // the mean of three identical returns is the return
  }
  {   // PC2
    const auto pts = patch(0, 25, 3, 1e-6);
    auto o = makeOpts(false, 0, 0, 3, 1, 50);
    VoxelPlane raw(o);
    raw.update(pts, -1, nullptr, nullptr, nullptr, 1, 0);
    CHECK(raw.isPlane());
    binned::BinSet bins(0.05);
    for (const auto& p : pts) bins.add(p.point, p.sensor_cov, true);
    CHECK(bins.numBins() == 25);
    const binned::Reps r = bins.representatives(binned::Options());
    const auto reps = toPoints(r);
    VoxelPlane vb(o);
    vb.update(reps, static_cast<int>(bins.numBins()), &r.w, nullptr, &r.w, 1, 0);
    CHECK(vb.isPlane());
    CHECK(vb.pointsSize() == 25);
    CHECK((vb.eigenValues() - raw.eigenValues()).cwiseAbs().maxCoeff() < 1e-12);
    CHECK((vb.planeVar() - 3.0 * raw.planeVar()).cwiseAbs().maxCoeff() < 1e-9 * (1.0 + raw.planeVar().cwiseAbs().maxCoeff()));
    binned::Options shr; shr.cov_mode = binned::CovMode::Shrunk;
    const binned::Reps rs = bins.representatives(shr);
    VoxelPlane vs(o);
    vs.update(toPoints(rs), static_cast<int>(bins.numBins()), &rs.w, nullptr, &rs.w, 1, 0);
    CHECK(vs.isPlane());
    CHECK((vs.planeVar() - raw.planeVar()).cwiseAbs().maxCoeff() < 1e-9 * (1.0 + raw.planeVar().cwiseAbs().maxCoeff()));
    VoxelPlaneSnapshot sr, sb;
    raw.fillSnapshot(sr); vb.fillSnapshot(sb);
    CHECK((sr.center - sb.center).norm() < 1e-12);
    CHECK((sr.normal - sb.normal).norm() < 1e-12);
    CHECK(std::fabs(sr.d - sb.d) < 1e-12);
  }
  std::vector<PlaneUpdate> upd;
  {   // PC3
    auto stats = std::make_shared<VoxelStats>();
    setBootstrapInsertion(false);
    setCurrentFrame(1);
    VoxelNode nb(makeOpts(true, 0, 0, 5, 5, 50), stats, 0, V3D(0, 0, 0));
    nb.insertPoints(patch(0, 4, 3, 1e-6), upd);              // 12 returns, 4 bins
    CHECK(!nb.hasConvergedPlane());
    CHECK(stats->binned_points.load() == 12);
    CHECK(stats->binned_new_bins.load() == 4);
    CHECK(stats->binned_fits.load() == 0);
    setCurrentFrame(2);
    nb.insertPoints(patch(4, 2, 1, 1e-6), upd);              // 2 more bins: 6 occupied
    CHECK(stats->binned_new_bins.load() == 6);
    CHECK(stats->binned_fits.load() == 1);
    CHECK(nb.hasConvergedPlane());

    auto stats_raw = std::make_shared<VoxelStats>();
    setCurrentFrame(1);
    VoxelNode nr(makeOpts(false, 0, 0, 5, 5, 50), stats_raw, 0, V3D(0, 0, 0));
    nr.insertPoints(patch(0, 4, 3, 1e-6), upd);              // the same 12 returns, raw: 12 >= 5, fits at once
    CHECK(nr.hasConvergedPlane());
    CHECK(stats_raw->binned_points.load() == 0);
  }
  {   // PC4 (R92: touched count; before R92 this block expected the opened count)
    auto stats = std::make_shared<VoxelStats>();
    setBootstrapInsertion(false);
    setCurrentFrame(1);
    VoxelNode n(makeOpts(true, 0, 0, 5, 5, 50), stats, 0, V3D(0, 0, 0));
    n.insertPoints(patch(0, 8, 1, 1e-6), upd);               // 8 bins: fitted
    CHECK(stats->binned_fits.load() == 1);
    setCurrentFrame(2);
    n.insertPoints(patch(0, 4, 2, 1e-6), upd);               // 8 returns into 4 open bins: 4 unique < 5
    CHECK(stats->binned_fits.load() == 1);
    CHECK(stats->binned_new_bins.load() == 8);
    setCurrentFrame(3);
    n.insertPoints(patch(2, 4, 1, 1e-6), upd);               // bins 2..5 (open): unique since the fit {0..5} = 6 >= 5
    CHECK(stats->binned_fits.load() == 2);
    CHECK(stats->binned_new_bins.load() == 8);
    setCurrentFrame(4);
    n.insertPoints(patch(8, 4, 1, 1e-6), upd);               // 4 new bins < 5
    CHECK(stats->binned_fits.load() == 2);
    setCurrentFrame(5);
    n.insertPoints(patch(12, 1, 1, 1e-6), upd);              // the 5th unique bin since the fit
    CHECK(stats->binned_fits.load() == 3);
  }
  {   // PC5
    auto s_lock = std::make_shared<VoxelStats>();
    auto s_open = std::make_shared<VoxelStats>();
    setBootstrapInsertion(false);
    setCurrentFrame(1);
    VoxelNode lock(makeOpts(true, 0, 0, 5, 5, 6), s_lock, 0, V3D(0, 0, 0));
    VoxelNode open(makeOpts(true, 0, 0, 5, 5, 50), s_open, 0, V3D(0, 0, 0));
    lock.insertPoints(patch(0, 8, 1, 1e-6), upd);
    open.insertPoints(patch(0, 8, 1, 1e-6), upd);
    CHECK(s_lock->binned_fits.load() == 1 && s_open->binned_fits.load() == 1);
    setCurrentFrame(2);
    lock.insertPoints(patch(8, 6, 1, 1e-6), upd);            // 6 new bins
    open.insertPoints(patch(8, 6, 1, 1e-6), upd);
    CHECK(s_lock->binned_fits.load() == 1);                  // locked: 8 occupied bins >= max 6
    CHECK(s_open->binned_fits.load() == 2);
  }
  {   // PC6
    auto pts = patch(0, 10, 1, 1e-6);
    for (int k = 0; k < 4; ++k) { auto more = patch(k, 1, 6, 1e-6); pts.insert(pts.end(), more.begin(), more.end()); }   // four bins hold 7 returns, six hold 1
    binned::BinSet bins(0.05);
    for (const auto& p : pts) bins.add(p.point, p.sensor_cov, true);
    auto o = makeOpts(true, 0, 0, 3, 1, 50);
    binned::Options bc, bu; bu.weight_mode = binned::WeightMode::Uniform;
    const binned::Reps rc = bins.representatives(bc), ru = bins.representatives(bu);
    VoxelPlane vc(o), vu(o);
    vc.update(toPoints(rc), static_cast<int>(bins.numBins()), &rc.w, nullptr, &rc.w, 1, 0);
    vu.update(toPoints(ru), static_cast<int>(bins.numBins()), &ru.w, nullptr, &ru.w, 1, 0);
    CHECK(vc.isPlane() && vu.isPlane());
    CHECK(vc.planeVar().trace() > vu.planeVar().trace());
  }
  {   // PC7
    auto pc = [&](int cov_mode, int weight_mode) {
      auto st = std::make_shared<VoxelStats>();
      setBootstrapInsertion(false);
      setCurrentFrame(1);
      VoxelNode n(makeOpts(true, cov_mode, weight_mode, 5, 5, 50), st, 0, V3D(0, 0, 0));
      auto pts = patch(0, 10, 1, 1e-6);
      for (int k = 0; k < 4; ++k) { auto more = patch(k, 1, 6, 1e-6); pts.insert(pts.end(), more.begin(), more.end()); }
      n.insertPoints(pts, upd);
      std::vector<VoxelPlaneSnapshot> out;
      n.appendPlaneSnapshots(VoxelKey(), out);
      CHECK(out.size() == 1);
      return out.empty() ? 0.0 : out[0].plane_covariance.trace();
    };
    const double ac = pc(0, 0), au = pc(0, 1), sc = pc(1, 0);
    CHECK(ac > 0.0 && au > 0.0 && sc > 0.0);
    CHECK(ac > au);          // count concentrates the weight on the crowded bins (PC6)
    CHECK(ac > sc);          // shrunk divides each bin covariance by k^2 instead of k
  }
  std::printf("test_binned_pca_r90 %s\n", g_fail ? "FAIL" : "PASS");
  return g_fail ? 1 : 0;
}
