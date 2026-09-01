// Does the CSR-indexed downsample reproduce the plain one, exactly, in BOTH
// modes?  That is the only property the re-deskew rests on: it replaces
// voxelDownsample() on the path where the spline is active, so if the two
// disagree by so much as a bit, turning the spline on changes the point set
// for a reason that has nothing to do with the spline -- and every
// spline-on/off comparison in the register becomes uninterpretable.
//
// Eigen + OpenCV headers only (data_wrappers.h pulls opencv2 for ImageData).
#include "livo_recon/utils/algo/hashing.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace livo_recon;

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
  if (!ok) ++failures;
}

// Deterministic LCG -- no <random> implementation differences across libstdc++
// versions, so this fixture's verdict is reproducible on any machine.
static uint64_t rng_state = 0x2545F4914F6CDD1DULL;
static double urand() {
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return double((rng_state >> 11) & ((1ULL << 53) - 1)) / double(1ULL << 53);
}

static std::vector<PointXYZCov> makeCloud(int n, double extent) {
  std::vector<PointXYZCov> c(n);
  for (int i = 0; i < n; ++i) {
    c[i].point = V3D(urand() * extent, urand() * extent, urand() * extent);
    c[i].t = 0.05 * urand() - 0.02;              // straddles zero on purpose
    c[i].sensor_cov = M3D::Identity() * (1e-4 + 1e-5 * urand());
    c[i].pos_cov    = M3D::Identity() * (1e-3 * urand());
  }
  return c;
}

static bool same(const PointXYZCov& a, const PointXYZCov& b) {
  return a.point == b.point && a.t == b.t
      && a.sensor_cov == b.sensor_cov && a.pos_cov == b.pos_cov;
}

static void runMode(DsMode mode, const char* name, double leaf) {
  std::printf("\n-- %s (leaf %.2f) --\n", name, leaf);
  const std::vector<PointXYZCov> in = makeCloud(4000, 2.0);

  std::vector<PointXYZCov> ref;
  voxelDownsample(in, ref, PointXYZCovKeyFn{leaf}, mode);

  std::vector<PointXYZCov> got;
  std::vector<int> off, mem;
  voxelDownsampleIndexedCsr(in, got, off, mem, PointXYZCovKeyFn{leaf}, mode);

  check(got.size() == ref.size(), "same number of surviving points");
  bool identical = got.size() == ref.size();
  for (size_t i = 0; identical && i < got.size(); ++i) identical = same(got[i], ref[i]);
  check(identical, "every surviving point is BIT-identical, in the same order");

  check(off.size() == got.size() + 1, "offsets has size n_out + 1");
  check(!off.empty() && off.front() == 0, "offsets starts at 0");
  check(!off.empty() && off.back() == int(mem.size()), "offsets ends at n_members");

  bool mono = true;
  for (size_t i = 0; i + 1 < off.size(); ++i) mono = mono && (off[i + 1] > off[i]);
  check(mono, "every cell has at least one member and offsets are increasing");

  // The membership set must be a PARTITION of the input: every raw point
  // named once and only once. If a point were named twice the re-deskew
  // would weight it twice; if a point vanished the averaged position would
  // silently drift away from what the first downsample produced.
  std::vector<int> seen(in.size(), 0);
  bool in_range = true;
  for (int m : mem) {
    if (m < 0 || m >= int(in.size())) { in_range = false; break; }
    ++seen[m];
  }
  check(in_range, "every member index is in range");
  int once = 0, twice = 0, never = 0;
  for (int s : seen) { if (s == 1) ++once; else if (s > 1) ++twice; else ++never; }
  check(twice == 0, "no raw point is claimed by two cells");
  check(mode == DsMode::FIRST ? (never == int(in.size()) - once) : (never == 0),
        mode == DsMode::FIRST ? "FIRST keeps one member per cell, drops the rest"
                              : "AVERAGE claims every raw point exactly once");

  if (mode == DsMode::FIRST)
    check(mem.size() == got.size(), "FIRST: exactly one member per output point");
  else
    check(mem.size() == in.size(), "AVERAGE: members account for the whole cloud");

  // Re-averaging the members must reproduce the output point. This is what
  // deskewPointsSplineCsr() does after re-placing each member, so if it does
  // not hold at the identity re-placement it cannot hold at any other.
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const int b = off[i], e = off[i + 1];
    if (e - b == 1) continue;
    V3D sum = V3D::Zero();
    for (int k = b; k < e; ++k) sum += in[mem[k]].point;
    worst = std::max(worst, (sum / double(e - b) - got[i].point).norm());
  }
  std::printf("       worst re-average error %.3e m\n", worst);
  check(worst < 1e-12, "re-averaging the members reproduces the output point");
}

int main() {
  std::printf("=== voxelDownsampleIndexedCsr vs voxelDownsample ===\n");
  runMode(DsMode::FIRST,   "DsMode::FIRST",   0.25);
  runMode(DsMode::AVERAGE, "DsMode::AVERAGE", 0.25);
  runMode(DsMode::AVERAGE, "DsMode::AVERAGE", 1.00);   // fewer, fuller cells
  std::printf("\n%s (%d failures)\n", failures ? "FAILURES" : "ALL CHECKS PASSED", failures);
  return failures ? 1 : 0;
}
