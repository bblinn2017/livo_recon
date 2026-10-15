// R95: the per-scan z squared summary (include/livo_recon/diagnostics/zsq_summary.h). Header-only; no ROS, no Eigen (mock type); own check macro (does not depend on NDEBUG).
// Planner sandbox on hand-made rows: NOT evidence about any bag. Identities only (rule 3.1).
//   ZS1 the header names the 20 columns in the documented order.
//   ZS2 a hand-computed example (8 rows: 6 valid, one zero variance, one NaN residual; plane rows 0, line rows 1, 2, 6; one row without a gate variance) gives the exact expected line.
//   ZS3 control (the check can fail): changing one residual changes the line, and the changed line equals its own hand-computed value.
//   ZS4 an empty vector writes a row of zeros with n_rows 0; a vector of only invalid rows writes n_invalid = n_rows and zero sums.
//   ZS5 the quantile index is floor(q (n - 1) + 0.5): for 6 valid rows p50 is the 4th and p90 the 6th smallest; for 1 valid row both are that row.
#include "livo_recon/diagnostics/zsq_summary.h"

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

struct MockRes { double r = 0, sigma_squared = 1, sigma_gate_squared = -1; int row_kind = 0; };

static std::string rowText(int scan, int iter, const std::vector<MockRes>& v)
{
  std::ostringstream o;
  o << std::setprecision(10);
  livo_recon::zsq_summary::writeRow(o, scan, iter, v);
  return o.str();
}

static std::vector<MockRes> example()
{
  std::vector<MockRes> v(8);
  v[0] = {1.0, 1.0, 2.0, 0};      // z2 1, zg 0.5
  v[1] = {2.0, 1.0, -1.0, 0};     // z2 4 (not > 4), no gate
  v[2] = {3.0, 1.0, 9.0, 0};      // z2 9 (> 4, not > 9), zg 1
  v[3] = {-4.0, 2.0, 4.0, 1};     // z2 8, zg 4 (not > 9)
  v[4] = {5.0, 1.0, 0.25, 2};     // z2 25 (> 9), zg 100 (> 9)
  v[5] = {1.0, 0.0, 1.0, 0};      // invalid: zero variance
  v[6] = {std::numeric_limits<double>::quiet_NaN(), 1.0, 1.0, 0};   // invalid: NaN residual
  v[7] = {0.5, 0.25, 1.0, 6};     // z2 1, zg 0.25
  return v;
}

int main()
{
  // ZS1
  CHECK(std::string(livo_recon::zsq_summary::header()) ==
        "scan_id,iter,n_rows,n_invalid,sum_r2,sum_z2,z2_p50,z2_p90,z2_max,n_z2_gt4,n_z2_gt9,n_plane,sum_z2_plane,n_plane_gt9,n_line,sum_z2_line,n_line_gt9,n_gate,sum_z2_gate,n_gate_gt9\n");
  // ZS2: valid z2 = {1, 4, 9, 8, 25, 1}; sum 48; sum r^2 = 1 + 4 + 9 + 16 + 25 + 0.25 = 55.25; sorted 1 1 4 8 9 25: p50 = index 3 = 8, p90 = index 5 = 25; > 4: {9, 8, 25} = 3; > 9: {25} = 1.
  // plane (kind 0, rows 0-2): n 3, sum 14, > 9: 0.  line (kinds 1, 2, 6): n 3, sum 8 + 25 + 1 = 34, > 9: 1.  gate: rows 0, 2, 3, 4, 7: 0.5 + 1 + 4 + 100 + 0.25 = 105.75, n 5, > 9: 1.
  const std::string expect = "7,0,8,2,55.25,48,8,25,25,3,1,3,14,0,3,34,1,5,105.75,1\n";
  CHECK(rowText(7, 0, example()) == expect);
  // ZS3: row 4 residual 5 -> 6: z2 36, zg 144; sum r^2 = 66.25; z2 sum 59; sorted 1 1 4 8 9 36; p50 8, p90 36, max 36; > 4: 3; > 9: 1; plane 3 / 14 / 0; line 3 / 8 + 36 + 1 = 45 / 1; gate sum 0.5 + 1 + 4 + 144 + 0.25 = 149.75.
  auto v2 = example();
  v2[4].r = 6.0;
  const std::string expect2 = "7,0,8,2,66.25,59,8,36,36,3,1,3,14,0,3,45,1,5,149.75,1\n";
  CHECK(rowText(7, 0, v2) != expect);
  CHECK(rowText(7, 0, v2) == expect2);
  // ZS4
  CHECK(rowText(3, 2, std::vector<MockRes>()) == "3,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n");
  std::vector<MockRes> bad(2);
  bad[0].sigma_squared = 0.0; bad[1].sigma_squared = std::numeric_limits<double>::infinity();
  CHECK(rowText(4, 1, bad) == "4,1,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n");
  // ZS5
  std::vector<MockRes> one(1);
  one[0].r = 2.0;
  CHECK(rowText(9, 0, one) == "9,0,1,0,4,4,4,4,4,0,0,1,4,0,0,0,0,0,0,0\n");
  std::printf("test_zsq_summary_r95 %s\n", g_fail ? "FAIL" : "PASS");
  return g_fail ? 1 : 0;
}
