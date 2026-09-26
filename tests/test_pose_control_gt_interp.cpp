// GT-to-query-timestamp interpolation validation (Phase 7): exact timestamp,
// midpoint, irregular spacing, start/end boundaries, out-of-range rejection,
// and duplicate/degenerate timestamps. NTU-VIRAL GT is position-only -- this
// utility never manufactures velocity/acceleration/orientation.
#include "livo_recon/diagnostics/gt_interp.h"

#include <cmath>
#include <cstdio>

using namespace livo_recon;

namespace
{
int failures = 0;
void check(bool ok, const char* name, double value = 0.0, double tol = 0.0)
{
  std::printf("  [%s] %-80s %.6e tol %.6e\n", ok ? "PASS" : "FAIL", name, value, tol);
  if (!ok) ++failures;
}

std::vector<GtSample3> regularSamples()
{
  std::vector<GtSample3> s;
  for (int i = 0; i <= 10; ++i) {
    GtSample3 g; g.t = i * 0.1; g.pos = V3D(i * 1.0, i * 2.0, i * 0.5);
    s.push_back(g);
  }
  return s;
}

void testExactGtTimestamp()
{
  const auto s = regularSamples();
  const auto r = interpolateGtToQueryTime(s, 0.3, 0.5);
  check(r.available, "exact-timestamp query is available");
  check((r.pos - V3D(3.0, 6.0, 1.5)).norm() < 1e-12, "exact-timestamp query returns the exact GT sample, no interpolation", (r.pos - V3D(3.0, 6.0, 1.5)).norm());
  check(std::abs(r.fraction) < 1e-12, "fraction is 0 at an exact bracket sample", r.fraction);
}

void testMidpointInterpolation()
{
  const auto s = regularSamples();
  const auto r = interpolateGtToQueryTime(s, 0.35, 0.5);   // exactly between t=0.3 and t=0.4
  check(r.available, "midpoint query is available");
  check(std::abs(r.fraction - 0.5) < 1e-12, "fraction is exactly 0.5 at the true midpoint", r.fraction, 0.5);
  const V3D expected(3.5, 7.0, 1.75);
  check((r.pos - expected).norm() < 1e-9, "midpoint position is the exact linear interpolation", (r.pos - expected).norm());
  check(std::abs(r.t_lo - 0.3) < 1e-9 && std::abs(r.t_hi - 0.4) < 1e-9, "bracketing timestamps are correctly identified");
  check(std::abs(r.gap - 0.1) < 1e-9, "interpolation gap is reported correctly", r.gap, 0.1);
}

void testIrregularSpacing()
{
  std::vector<GtSample3> s = {
    {0.0, V3D(0, 0, 0)}, {0.05, V3D(1, 0, 0)}, {0.5, V3D(2, 0, 0)}, {0.55, V3D(3, 0, 0)}, {1.0, V3D(4, 0, 0)},
  };
  // Query inside the LARGE gap [0.05, 0.5] -- gap=0.45.
  auto r = interpolateGtToQueryTime(s, 0.2, 0.5);   // max_gap=0.5 allows it
  check(r.available, "irregular spacing: query inside a wide-but-allowed gap is available");
  const double frac_expected = (0.2 - 0.05) / 0.45;
  check(std::abs(r.fraction - frac_expected) < 1e-9, "irregular spacing: fraction reflects the true (unequal) bracket width", r.fraction, frac_expected);

  // Same query, now with a TIGHTER max_gap that should reject it.
  auto r2 = interpolateGtToQueryTime(s, 0.2, 0.3);
  check(!r2.available, "irregular spacing: a gap exceeding max_gap is correctly rejected", r2.gap, 0.3);
  check(std::abs(r2.gap - 0.45) < 1e-9, "the rejected gap is still reported for diagnostic purposes", r2.gap, 0.45);
}

void testStartEndBoundaries()
{
  const auto s = regularSamples();
  auto r_start = interpolateGtToQueryTime(s, 0.0, 0.5);
  check(r_start.available && (r_start.pos - V3D(0, 0, 0)).norm() < 1e-12, "exact start boundary is available and exact");
  auto r_end = interpolateGtToQueryTime(s, 1.0, 0.5);
  check(r_end.available && (r_end.pos - V3D(10, 20, 5)).norm() < 1e-9, "exact end boundary is available and exact");
}

void testOutOfRangeRejected()
{
  const auto s = regularSamples();
  auto r_before = interpolateGtToQueryTime(s, -0.01, 0.5);
  check(!r_before.available, "a query strictly before the first GT sample is rejected (never extrapolated)");
  auto r_after = interpolateGtToQueryTime(s, 1.01, 0.5);
  check(!r_after.available, "a query strictly after the last GT sample is rejected (never extrapolated)");
  auto r_empty = interpolateGtToQueryTime({}, 0.5, 0.5);
  check(!r_empty.available, "an empty GT sample set is always unavailable, never a crash");
}

void testDuplicateDegenerateTimestamps()
{
  std::vector<GtSample3> s = {
    {0.0, V3D(0, 0, 0)}, {0.1, V3D(1, 0, 0)}, {0.1, V3D(1, 1, 0)}, {0.2, V3D(2, 0, 0)},
  };
  // Query exactly at the duplicated timestamp: bracket width 0, must not
  // divide by zero, and must report SOME defined position (the earlier of
  // the two duplicate samples), not NaN.
  auto r = interpolateGtToQueryTime(s, 0.1, 0.5);
  check(r.available, "a query landing exactly on a duplicate timestamp is available (no NaN/crash)");
  check(std::isfinite(r.pos.x()) && std::isfinite(r.pos.y()) && std::isfinite(r.pos.z()),
        "duplicate-timestamp interpolation result is finite", r.pos.norm());
}

}  // namespace

int main()
{
  std::printf("Pose-control GT-to-query-timestamp interpolation validation suite\n");
  testExactGtTimestamp();
  testMidpointInterpolation();
  testIrregularSpacing();
  testStartEndBoundaries();
  testOutOfRangeRejected();
  testDuplicateDegenerateTimestamps();
  std::printf("%d failure(s)\n", failures);
  return failures ? 1 : 0;
}
