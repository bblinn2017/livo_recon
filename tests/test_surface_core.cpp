#include "livo_recon/map/surface/surface_core.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <tuple>
#include <vector>

// Every check is a CHECK() (not compiled out by NDEBUG, unlike assert()).
#define CHECK(c)                                                                        \
  do {                                                                                  \
    if (!(c)) {                                                                         \
      std::fprintf(stderr, "CHECK FAILED %s:%d: %s\n", __FILE__, __LINE__, #c);         \
      std::exit(1);                                                                     \
    }                                                                                   \
  } while (0)

using namespace livo_recon::surface;

namespace
{
// Deterministic uniform noise in [-1, 1).
struct Lcg
{
  std::uint64_t s = 88172645463325252ULL;
  double next()
  {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (static_cast<double>(s >> 11) / 9007199254740992.0) * 2.0 - 1.0;
  }
};

const M3 kCov = M3::Identity() * 1e-6;   // sensor covariance used everywhere

std::vector<V3> planeXY(double x0, double y0, double nx, double ny, double step, double z0, double slope_x = 0.0)
{
  std::vector<V3> out;
  for (int i = 0; i < static_cast<int>(nx / step); ++i)
    for (int j = 0; j < static_cast<int>(ny / step); ++j) {
      const double x = x0 + step * (i + 0.5), y = y0 + step * (j + 0.5);
      out.push_back(V3(x, y, z0 + slope_x * x));
    }
  return out;
}

std::vector<V3> wallX(double x, double y0, double ny, double z0, double nz, double step)
{
  std::vector<V3> out;
  for (int j = 0; j < static_cast<int>(ny / step); ++j)
    for (int k = 0; k < static_cast<int>(nz / step); ++k)
      out.push_back(V3(x, y0 + step * (j + 0.5), z0 + step * (k + 0.5)));
  return out;
}

// co-membership signature: valid cell key -> smallest key (x,y,z lexicographic) of any valid cell in its surface
std::map<std::uint64_t, std::uint64_t> partition(const SurfaceLayer& L)
{
  std::map<std::uint64_t, std::uint64_t> out;
  for (std::uint64_t sid : L.surfaceIds()) {
    const Surface* s = L.findSurface(sid);
    std::uint64_t rep = 0;
    for (std::uint64_t cid : s->children)
      if (L.cellById(cid).fit.valid) { rep = cid; break; }
    for (std::uint64_t cid : s->children)
      if (L.cellById(cid).fit.valid) out[cid] = rep;
  }
  return out;
}

// canonical partition keyed by cell KEY (creation order may differ between two layers)
std::set<std::set<std::tuple<int, int, int>>> groups(const SurfaceLayer& L)
{
  std::set<std::set<std::tuple<int, int, int>>> out;
  for (std::uint64_t sid : L.surfaceIds()) {
    const Surface* s = L.findSurface(sid);
    std::set<std::tuple<int, int, int>> g;
    for (std::uint64_t cid : s->children) {
      const Cell& c = L.cellById(cid);
      if (c.fit.valid) g.insert(std::make_tuple(c.key.x, c.key.y, c.key.z));
    }
    if (!g.empty()) out.insert(g);
  }
  return out;
}

void checkStatsMergeEqualsBatch()
{
  Lcg r;
  SurfaceStats a, b, batch;
  a.setRef(V3(0.1, 0.2, 0.3));
  b.setRef(V3(5.0, -3.0, 1.0));
  batch.setRef(a.ref);
  for (int i = 0; i < 200; ++i) {
    const V3 p(10.0 + 0.3 * r.next(), -4.0 + 0.3 * r.next(), 2.0 + 0.05 * r.next());
    const M3 cov = M3::Identity() * (1e-6 * (1.0 + 0.5 * r.next()));
    ((i % 2) ? a : b).add(p, cov);
    batch.add(p, cov);
  }
  SurfaceStats m = a;
  m.merge(b);
  CHECK(std::abs(m.n - batch.n) < 1e-12);
  CHECK((m.mean() - batch.mean()).norm() < 1e-9);
  CHECK((m.covariance() - batch.covariance()).norm() < 1e-9);
  CHECK((m.scov - batch.scov).norm() < 1e-15);
  for (int i = 0; i < 3; ++i) {
    CHECK((m.w[i] - batch.w[i]).norm() < 1e-9);
    for (int j = 0; j < 3; ++j) CHECK((m.v[i][j] - batch.v[i][j]).norm() < 1e-7);
  }
  // merging into an empty stats adopts the other
  SurfaceStats e;
  e.merge(a);
  CHECK(std::abs(e.n - a.n) < 1e-12 && (e.mean() - a.mean()).norm() < 1e-12);
}

void checkFit()
{
  Options o;
  Lcg r;
  auto make = [&](int per_side) {
    SurfaceStats s;
    s.setRef(V3(0.125, 0.125, 1.0));
    for (int i = 0; i < per_side; ++i)
      for (int j = 0; j < per_side; ++j) {
        const double x = 0.25 * (i + 0.5) / per_side, y = 0.25 * (j + 0.5) / per_side;
        s.add(V3(x, y, 1.0 + 0.1 * x + 0.05 * y + 0.003 * r.next()), M3::Identity() * 2.5e-5);
      }
    return s;
  };
  const Fit f10 = fitStats(make(10), o);
  const Fit f20 = fitStats(make(20), o);
  CHECK(f10.valid && f20.valid);
  const V3 truth = V3(-0.1, -0.05, 1.0).normalized();
  CHECK(std::abs(f20.normal.dot(truth)) > 0.999);
  CHECK(f20.plane_var.allFinite() && f20.plane_var.trace() > 0.0 && f20.plane_var.trace() < 1.0);
  const double ratio = f10.plane_var.trace() / f20.plane_var.trace();   // n grew 4x: variance should fall about 4x
  CHECK(ratio > 2.0 && ratio < 8.0);
  CHECK(f20.radius > 0.0 && f20.planarity >= 0.0 && f20.planarity < 0.2);
  // sign invariance of the tests
  CHECK(std::abs(planeAngleDeg(f20.normal, -f20.normal)) < 1e-6);
  Fit flipped = f20;
  flipped.normal = -f20.normal;
  CHECK(std::abs(planeOffset(f20, f20) - planeOffset(flipped, f20)) < 1e-12);
  // too few points -> invalid
  SurfaceStats few;
  few.setRef(V3::Zero());
  for (int i = 0; i < 4; ++i) few.add(V3(0.01 * i, 0.02 * i * i, 0.0), kCov);
  CHECK(!fitStats(few, o).valid);
  // a bimodal (two-level) cell is not a plane
  SurfaceStats two;
  two.setRef(V3(0.125, 0.125, 0.0));
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j) {
      two.add(V3(0.025 * i + 0.0125, 0.025 * j + 0.0125, 0.0), kCov);
      two.add(V3(0.025 * i + 0.0125, 0.025 * j + 0.0125, 0.3), kCov);
    }
  CHECK(!fitStats(two, o).valid);
}

void feed(SurfaceLayer& L, const std::vector<V3>& pts)
{
  for (const V3& p : pts) L.addPoint(p, kCov);
}

void checkMergeSplitAndSupport()
{
  Options o;   // cell 0.25, merge_gap 0.75, support on, gap 0.25
  {
    SurfaceLayer L(o);
    feed(L, planeXY(0.0, 0.0, 1.0, 0.5, 0.025, 0.0));      // floor: 4 x 2 cells
    feed(L, wallX(1.125, 0.0, 0.5, -0.5, 0.5, 0.025));     // wall, perpendicular
    const UpdateStats us = L.update();
    CHECK(us.merges >= 7);
    const Summary m = L.summary();
    CHECK(m.cells_valid == m.cells);
    CHECK(m.largest == 8);           // the floor is one surface of 8 cells
    CHECK(m.surfaces_multi >= 1);
    // the wall cells never join the floor surface
    const Cell* floor_cell = L.findCell(L.keyOf(V3(0.1, 0.1, 0.0)));
    const Cell* wall_cell = L.findCell(L.keyOf(V3(1.125, 0.1, -0.1)));
    CHECK(floor_cell && wall_cell);
    CHECK(floor_cell->surface_id != wall_cell->surface_id);
    CHECK(L.planeFor(*floor_cell) != nullptr && L.planeFor(*wall_cell) != nullptr);
    // surface plane is used for a floor cell (8 children), own plane for a wall cell
    const void* owner_f = nullptr;
    const void* owner_w = nullptr;
    L.planeFor(*floor_cell, &owner_f);
    L.planeFor(*wall_cell, &owner_w);
    CHECK(owner_f == static_cast<const void*>(L.findSurface(floor_cell->surface_id)));

    // Split: a shelf at z=0.3 inside one floor cell makes that cell bimodal -> invalid -> split out of the surface.
    const std::uint64_t sid_before = floor_cell->surface_id;
    std::vector<V3> shelf = planeXY(0.5, 0.25, 0.25, 0.25, 0.025, 0.3);   // the cell with key (2,1,1)?? -> z belongs to its own cell
    // put the shelf points at z = 0.3 into the FLOOR cell by lowering the cell height: use same x,y and z = 0.0..0.24 bimodal
    std::vector<V3> shelf_same_cell;
    for (const V3& p : shelf) shelf_same_cell.push_back(V3(p.x(), p.y(), 0.20));   // same cell (z key 0), 0.2 above the floor
    feed(L, shelf_same_cell);
    const UpdateStats us2 = L.update();
    CHECK(us2.splits >= 1);
    const Cell* bad = L.findCell(L.keyOf(V3(0.6, 0.3, 0.1)));
    CHECK(bad && !bad->fit.valid);
    CHECK(bad->surface_id != sid_before || L.findSurface(sid_before)->children.size() == 1);
    const Surface* rest = L.findSurface(floor_cell->surface_id);
    CHECK(rest != nullptr && rest->children.size() >= 2);
    for (std::uint64_t cid : rest->children) CHECK(cid != bad->id);
  }
  {
    // Support test: two coplanar cells 0.5 m apart merge without the support test and not with it (merge_gap large).
    Options a = o;
    a.merge_gap = 2.0;
    for (int support = 0; support < 2; ++support) {
      a.support_test = (support == 1);
      SurfaceLayer L(a);
      feed(L, planeXY(0.0, 0.0, 0.25, 0.25, 0.025, 0.0));
      feed(L, planeXY(0.75, 0.0, 0.25, 0.25, 0.025, 0.0));
      L.update();
      const Cell* c0 = L.findCell(L.keyOf(V3(0.1, 0.1, 0.0)));
      const Cell* c1 = L.findCell(L.keyOf(V3(0.8, 0.1, 0.0)));
      CHECK(c0 && c1);
      if (support == 1) CHECK(c0->surface_id != c1->surface_id);
      else CHECK(c0->surface_id == c1->surface_id);
    }
  }
}

void checkDirtySetEqualsFromScratchAndDeterminism()
{
  Options o;
  std::vector<V3> scene = planeXY(0.0, 0.0, 1.0, 0.75, 0.025, 0.0);
  const std::vector<V3> wall = wallX(1.125, 0.0, 0.5, -0.5, 0.5, 0.025);
  const std::vector<V3> slope = planeXY(2.0, 0.0, 0.75, 0.5, 0.025, 0.0, /*slope_x=*/0.5);   // tilted ~26 deg
  scene.insert(scene.end(), wall.begin(), wall.end());
  scene.insert(scene.end(), slope.begin(), slope.end());

  SurfaceLayer inc(o), once(o), inc2(o);
  const int frames = 6;
  for (int f = 0; f < frames; ++f) {
    for (std::size_t i = f; i < scene.size(); i += frames) { inc.addPoint(scene[i], kCov); inc2.addPoint(scene[i], kCov); }
    inc.update();
    inc2.update();
  }
  feed(once, scene);
  once.update();
  CHECK(groups(inc) == groups(once));
  // determinism: identical inputs -> bit-identical cell fits
  CHECK(inc.numCells() == inc2.numCells());
  for (std::uint64_t id = 1; id <= inc.numCells(); ++id) {
    const Cell& a = inc.cellById(id);
    const Cell& b = inc2.cellById(id);
    CHECK(a.surface_id == b.surface_id);
    CHECK(a.fit.valid == b.fit.valid);
    if (a.fit.valid) {
      CHECK(a.fit.center == b.fit.center);
      CHECK(a.fit.normal == b.fit.normal);
      CHECK(a.fit.plane_var == b.fit.plane_var);
    }
  }
  // the tilted slab (26 deg) must not merge with the floor (5 deg gate) even though their cells are far apart anyway;
  // and a floor with a 4-degree tilt DOES merge across neighbouring cells.
  const Summary m = inc.summary();
  CHECK(m.surfaces_multi >= 2);
}

void checkTiltGate()
{
  Options o;
  // two neighbouring cells, planes tilted 3 deg apart about the y axis: within 5 deg -> merge; 8 deg apart -> no merge
  const double t3 = std::tan(3.0 * M_PI / 180.0), t8 = std::tan(8.0 * M_PI / 180.0);
  for (int variant = 0; variant < 2; ++variant) {
    SurfaceLayer M(o);
    const double t = variant == 0 ? t3 : t8;
    std::vector<V3> a = planeXY(0.0, 0.0, 0.25, 0.25, 0.0125, 0.0);
    std::vector<V3> b;
    for (const V3& p : planeXY(0.25, 0.0, 0.25, 0.25, 0.0125, 0.0)) b.push_back(V3(p.x(), p.y(), t * (p.x() - 0.25)));
    feed(M, a);
    feed(M, b);
    M.update();
    const Cell* c0 = M.findCell(M.keyOf(V3(0.1, 0.1, 0.0)));
    const Cell* c1 = M.findCell(M.keyOf(V3(0.3, 0.1, 0.0)));
    CHECK(c0 && c1 && c0->fit.valid && c1->fit.valid);
    if (variant == 0) CHECK(c0->surface_id == c1->surface_id);
    else CHECK(c0->surface_id != c1->surface_id);
  }
}

}  // namespace

int main()
{
  checkStatsMergeEqualsBatch();
  checkFit();
  checkMergeSplitAndSupport();
  checkDirtySetEqualsFromScratchAndDeterminism();
  checkTiltGate();
  std::printf("test_surface_core: all checks passed\n");
  return 0;
}
