// R80D (F-484 / F-491): unit test of the offset term added to the retime grid time (header only, ROS-free). Algebraic identities only (rule 3.1).
#include <cmath>
#include <cstdio>

#include "livo_recon/utils/data/lidar_msg_log.h"

using namespace livo_recon;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

int main()
{
  const double t0 = 1609059013.0668645, period = 0.099997231748;
  // time_offset equal to the reference: the term is exactly 0.0, so the result is bit for bit the R80C retimeScanTime (the centre arm of the offset sweep must equal the R80C arm).
  for (unsigned seq : {119u, 120u, 1000u, 4105u})
    CHECK(retimeScanTimeWithOffset(seq, 119, t0, period, -0.1, -0.1) == retimeScanTime(seq, 119, t0, period));
  CHECK(retimeScanTimeWithOffset(119, 119, t0, period, 0.0, 0.0) == t0);
  // a swept offset moves the grid by exactly (offset - ref): -0.095 is 5 ms later, -0.115 is 15 ms earlier (tolerance 1e-6 s: one double step at 1.6e9 s is 2.4e-7 s).
  CHECK(std::fabs((retimeScanTimeWithOffset(500, 119, t0, period, -0.095, -0.1) - retimeScanTime(500, 119, t0, period)) - 0.005) < 1e-6);
  CHECK(std::fabs((retimeScanTimeWithOffset(500, 119, t0, period, -0.115, -0.1) - retimeScanTime(500, 119, t0, period)) + 0.015) < 1e-6);
  // the offset does not change the spacing between messages.
  CHECK(std::fabs((retimeScanTimeWithOffset(201, 119, t0, period, -0.09, -0.1) - retimeScanTimeWithOffset(200, 119, t0, period, -0.09, -0.1)) - period) < 1e-6);
  if (g_fail == 0) std::printf("ALL PASS\n");
  return g_fail == 0 ? 0 : 1;
}
