// R91, rewritten in R92: min_init / min_update count UNIQUE BINS TOUCHED since the last fit. R92 removed the R88 to R90 count of bins newly OPENED and the key voxel_map/binned/update_count_mode, so this is the only count.
// Planner sandbox on synthetic points: NOT evidence about any bag. Identities and rules only (rule 3.1). Links src/lio/voxelnode.cpp and voxelplane.cpp; no ROS.
//   TB1 BinSet::dirtyCount(): returns into the same bin count once; markFitted() restarts the period; a bin opened in the period counts; numBins() is not touched by markFitted().
//   TN1 debiased fit, touched: returns into bins that are already open DO count; the same bin hit in two scans counts once (3 unique bins from 5 hits do not reach min_update 4; a 4th unique bin does).
//   TN2 (R92) VoxelOpts has no binned_update_count_mode member (compile-time check): the opened count is gone.
//   TN3 pca fit on the binned map: the same rules as TN1.
//   TN4 binned_touched_at_fit is the sum over fits of the unique bins touched in that period; binned_new_bins still counts bins opened.
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

// TN2: detector for the removed member (a member that exists makes the static_assert below fail to compile).
template <class T, class = void> struct HasCountMode : std::false_type {};
template <class T> struct HasCountMode<T, std::void_t<decltype(std::declval<T&>().binned_update_count_mode)>> : std::true_type {};
static_assert(!HasCountMode<VoxelOpts>::value, "VoxelOpts::binned_update_count_mode was removed in R92 (touched is the only count)");

static VoxelOptsPtr makeOpts(const char* fit, int min_init, int min_update)
{
  auto o = std::make_shared<VoxelOpts>();
  o->plane_threshold = 0.01;
  o->plane_fit_mode = fit;
  o->plane_var_mode = "eigengap";
  o->weight_sigma_r2 = 4e-4;
  o->min_init_points = min_init;
  o->min_update_points = min_update;
  o->voxel_size = 0.25;
  o->max_layer = 0;
  o->binned_enabled = true;
  o->binned_bin_fraction = 0.2;
  o->binned_cov_mode = 0;
  return o;
}

// one return per 0.05 m bin of a 5 x 5 grid on the plane z = 0.01 x - 0.02 y + 0.03, bins first .. first+n-1 (order k*7 mod 25), `reps` identical returns each
static std::vector<PointXYZCov> patch(int first, int n, int reps)
{
  std::vector<PointXYZCov> v;
  for (int k = first; k < first + n; ++k) {
    const int kk = (k * 7) % 25;
    const double x = -0.1 + 0.05 * (kk % 5) + 0.0125, y = -0.1 + 0.05 * (kk / 5) + 0.0125;
    for (int r = 0; r < reps; ++r) {
      PointXYZCov p;
      p.point = V3D(x, y, 0.01 * x - 0.02 * y + 0.03);
      p.sensor_cov = M3D::Identity() * 1e-6;
      p.pos_cov = M3D::Zero();
      v.push_back(p);
    }
  }
  return v;
}

// The sequence: frame 1: bins 0..3 (4 unique; first fit needs min_init 3).  frame 2: the same 4 bins again.  frame 3: bins 0..1.  frame 4: bins 0..2.  frame 5: bin 4 (new).
// Fits made after each frame are returned in `fits`, the stats are read at the end.
struct Run { long long fits[6]; long long touched_at_fit; long long new_bins; };
static Run run(const char* fit, int min_update)
{
  auto stats = std::make_shared<VoxelStats>();
  std::vector<PlaneUpdate> upd;
  setBootstrapInsertion(false);
  VoxelNode n(makeOpts(fit, 3, min_update), stats, 0, V3D(0, 0, 0));
  Run r{};
  setCurrentFrame(1); n.insertPoints(patch(0, 4, 1), upd); r.fits[1] = stats->binned_fits.load();
  setCurrentFrame(2); n.insertPoints(patch(0, 4, 2), upd); r.fits[2] = stats->binned_fits.load();
  setCurrentFrame(3); n.insertPoints(patch(0, 2, 1), upd); r.fits[3] = stats->binned_fits.load();
  setCurrentFrame(4); n.insertPoints(patch(0, 3, 1), upd); r.fits[4] = stats->binned_fits.load();
  setCurrentFrame(5); n.insertPoints(patch(4, 1, 1), upd); r.fits[5] = stats->binned_fits.load();
  r.touched_at_fit = stats->binned_touched_at_fit.load();
  r.new_bins = stats->binned_new_bins.load();
  return r;
}

int main()
{
  {   // TB1
    binned::BinSet b(0.05);
    CHECK(b.dirtyCount() == 0);
    CHECK(b.add(V3D(0.01, 0.01, 0.01), M3D::Identity() * 1e-6, true));       // opens bin A
    CHECK(!b.add(V3D(0.012, 0.012, 0.012), M3D::Identity() * 1e-6, true));   // same bin A
    CHECK(b.add(V3D(0.11, 0.01, 0.01), M3D::Identity() * 1e-6, true));       // opens bin B
    CHECK(b.dirtyCount() == 2 && b.numBins() == 2);
    b.markFitted();
    CHECK(b.dirtyCount() == 0 && b.numBins() == 2);
    CHECK(!b.add(V3D(0.011, 0.011, 0.011), M3D::Identity() * 1e-6, true));   // bin A again: touched, not opened
    CHECK(b.dirtyCount() == 1);
    CHECK(!b.add(V3D(0.013, 0.013, 0.013), M3D::Identity() * 1e-6, true));   // bin A a second time: still one
    CHECK(b.dirtyCount() == 1);
    CHECK(b.add(V3D(0.21, 0.01, 0.01), M3D::Identity() * 1e-6, true));       // opens bin C
    CHECK(b.dirtyCount() == 2);
    b.markFitted(); b.markFitted();
    CHECK(b.dirtyCount() == 0);
    CHECK(!b.add(V3D(0.11, 0.011, 0.011), M3D::Identity() * 1e-6, true));   // bin B after two empty periods
    CHECK(b.dirtyCount() == 1);
  }
  for (int pca = 0; pca < 2; ++pca) {
    const char* fit = pca ? "pca" : "debiased";
    // min_update 3. Frame 1 first fit (4 unique >= min_init 3); frame 2 re-hits 4 bins (4 >= 3) fit; frame 3 hits 2 bins (2 < 3) none; frame 4 hits bins 0..2 -> unique since the fit {0,1,2} = 3 >= 3 fit;
    // frame 5 opens bin 4 (1 < 3) none.
    const Run t = run(fit, 3);
    CHECK(t.fits[1] == 1);
    CHECK(t.fits[2] == 2);
    CHECK(t.fits[3] == 2);   // 2 unique bins only, however many returns
    CHECK(t.fits[4] == 3);   // bins 0,1 (frame 3) and 0,1,2 (frame 4): 3 unique, not 5 hits
    CHECK(t.fits[5] == 3);
    CHECK(t.touched_at_fit == 4 + 4 + 3);
    // frame 1..5 opened bins: 4 + 0 + 0 + 0 + 1
    CHECK(t.new_bins == 5);
    // touched with min_update 5 on the same sequence: frame 2 (4) no, frame 3 (still the same 4 unique since the first fit) no, frame 4 no; frame 5 adds bin 4: 5 unique >= 5 fit.
    const Run t5 = run(fit, 5);
    CHECK(t5.fits[1] == 1 && t5.fits[2] == 1 && t5.fits[3] == 1 && t5.fits[4] == 1);
    CHECK(t5.fits[5] == 2);
    CHECK(t5.touched_at_fit == 4 + 5);
  }
  std::printf("test_binned_touch_r91 %s\n", g_fail ? "FAIL" : "PASS");
  return g_fail ? 1 : 0;
}
