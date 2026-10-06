// R80C (F-484): unit test of the retime grid time and of the retime_trim point count (header only, ROS-free). Algebraic identities only (rule 3.1).
#include <cmath>
#include <cstdio>
#include <vector>

#include "livo_recon/utils/data/lidar_msg_log.h"

using namespace livo_recon;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

struct P { double t; };

int main()
{
  // retimeScanTime (tolerance 1e-6 s: the clock is 1.6e9 s, one double step is 2.4e-7 s): seq0 maps to t0; consecutive seq differ by exactly the period; a gap in seq leaves a gap in time (no re-indexing).
  const double t0 = 1609059013.0668645, period = 0.099997231748;
  CHECK(retimeScanTime(119, 119, t0, period) == t0);
  CHECK(std::fabs(retimeScanTime(120, 119, t0, period) - (t0 + period)) < 1e-6);
  CHECK(std::fabs((retimeScanTime(1000, 119, t0, period) - retimeScanTime(999, 119, t0, period)) - period) < 1e-6);
  CHECK(std::fabs((retimeScanTime(205, 119, t0, period) - retimeScanTime(200, 119, t0, period)) - 5.0 * period) < 1e-6);   // seq 201..204 missing: five periods
  CHECK(retimeScanTime(100, 119, t0, period) < t0);   // a seq below seq0 maps before t0 (no unsigned wrap)

  // countPointsBefore: strictly earlier counts, equal stays, empty and all/none.
  std::vector<P> v = {{10.0}, {10.1}, {10.2}, {10.2}, {10.3}};
  CHECK(countPointsBefore(v, 9.0) == 0);
  CHECK(countPointsBefore(v, 10.0) == 0);       // equal to the first point: stays
  CHECK(countPointsBefore(v, 10.15) == 2);
  CHECK(countPointsBefore(v, 10.2) == 2);       // equal (twice) stays
  CHECK(countPointsBefore(v, 11.0) == 5);
  CHECK(countPointsBefore(std::vector<P>(), 1.0) == 0);

  // After trimming against the previous end, the pushLidar rule cannot fire (the point of equal time is kept and equal is not an overlap).
  for (double prev_end : {0.0, 9.0, 10.0, 10.15, 10.2, 11.0})
  {
    std::vector<P> w = v;
    const std::size_t n = countPointsBefore(w, prev_end);
    w.erase(w.begin(), w.begin() + static_cast<std::ptrdiff_t>(n));
    if (!w.empty()) CHECK(!lidarMessageOverlapsPrevious(w.front().t, prev_end));
  }
  // ... and without the trim the rule does fire on the same input (the check can fail).
  CHECK(lidarMessageOverlapsPrevious(v.front().t, 10.15));

  if (g_fail == 0) std::printf("ALL PASS\n");
  return g_fail == 0 ? 0 : 1;
}
