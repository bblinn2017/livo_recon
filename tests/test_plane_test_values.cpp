// R74 unit test (F-428): the plane and eigengap test values written for every octree node (include/livo_recon/lio/plane_test_values.h).
//   PT1 edge_m = voxel_size / 2^layer for voxel 1.0 (layers 0..3: 1, 0.5, 0.25, 0.125) and voxel 0.425 (layer 2: 0.10625), exact.
//   PT2 planarity_value is e0; planarity_pass follows is_plane; second_eig_below_threshold is (e1 < plane_threshold) with equality NOT below.
//   PT3 eigengap_denom1 = e0 - e1 and eigengap_denom2 = e0 - e2 exactly (the code's own denominators).
//   PT4 ratios: e1/e0 and e2/e1 on a hand case (exact); a zero divisor gives nan, not inf and not 0.
//   PT5 rel_gap_threshold = 1 + 3 sqrt(2/(N-1)): N = 5 gives 1 + 3 sqrt(0.5); N = 201 gives 1.3 (1e-15); N < 2 gives nan; strictly decreasing in N.
//   R74 agent fix: the hardcoded check below originally read N = 101, but 1 + 3*sqrt(2/(101-1)) = 1.4242640687119286, not 1.3 --
//   solving 3*sqrt(2/(N-1)) = 0.3 gives N - 1 = 200, i.e. N = 201 is the value that actually equals 1.3. Verified: python3 -c
//   "import math; print(1+3*math.sqrt(2/200))" -> 1.3 exactly. The implementation (relativeGapThreshold(), matching the header's own
//   documented formula and design note D2) was not changed -- only this test's arithmetic.
//   PT6 the statistic behind it: the sample variance of N Gaussian draws has relative standard deviation sqrt(2/(N-1)) (N = 50, 40000 repetitions,
//       within 3 percent). Control: a wrong constant (sqrt(1/(N-1))) is 29 percent away from the measured value, so the check can fail.
//   PT7 the header and a written row have the same number of comma separated fields (10), and the row parses back to the values.
#include "livo_recon/lio/plane_test_values.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

static int fields(const std::string& s)
{
  int n = 1;
  for (char c : s) if (c == ',') ++n;
  return n;
}

int main()
{
  CHECK(planeTestValues(0, 1, 2, 0, 1.0, 2.5e-3, true, 10).edge_m == 1.0, "PT1 layer 0 voxel 1.0");
  CHECK(planeTestValues(0, 1, 2, 1, 1.0, 2.5e-3, true, 10).edge_m == 0.5, "PT1 layer 1");
  CHECK(planeTestValues(0, 1, 2, 2, 1.0, 2.5e-3, true, 10).edge_m == 0.25, "PT1 layer 2");
  CHECK(planeTestValues(0, 1, 2, 3, 1.0, 2.5e-3, true, 10).edge_m == 0.125, "PT1 layer 3");
  CHECK(planeTestValues(0, 1, 2, 2, 0.425, 2.5e-3, true, 10).edge_m == 0.425 / 4.0, "PT1 voxel 0.425 layer 2");

  const PlaneTestValues a = planeTestValues(1.0e-3, 2.0e-3, 4.0e-2, 1, 1.0, 2.5e-3, true, 30);
  CHECK(a.planarity_value == 1.0e-3 && a.planarity_pass == 1 && a.second_eig_below_threshold == 1, "PT2 plane with e1 below the threshold");
  const PlaneTestValues b = planeTestValues(1.0e-3, 2.5e-3, 4.0e-2, 1, 1.0, 2.5e-3, false, 30);
  CHECK(b.planarity_pass == 0 && b.second_eig_below_threshold == 0, "PT2 e1 equal to the threshold is not below it; is_plane false gives pass 0");

  CHECK(a.eigengap_denom1 == 1.0e-3 - 2.0e-3 && a.eigengap_denom2 == 1.0e-3 - 4.0e-2, "PT3 denominators e0 - e1 and e0 - e2");
  CHECK(a.ratio_e1_over_e0 == 2.0e-3 / 1.0e-3 && a.ratio_e2_over_e1 == 4.0e-2 / 2.0e-3, "PT4 ratios on the hand case");
  const PlaneTestValues z = planeTestValues(0.0, 0.0, 1.0, 0, 1.0, 2.5e-3, false, 30);
  CHECK(std::isnan(z.ratio_e1_over_e0) && std::isnan(z.ratio_e2_over_e1), "PT4 zero divisors give nan");

  CHECK(std::abs(relativeGapThreshold(5) - (1.0 + 3.0 * std::sqrt(0.5))) < 1e-15, "PT5 N = 5");
  CHECK(std::abs(relativeGapThreshold(201) - 1.3) < 1e-15, "PT5 N = 201 gives 1.3");
  CHECK(std::isnan(relativeGapThreshold(1)) && std::isnan(relativeGapThreshold(0)), "PT5 N < 2 gives nan");
  bool dec = true;
  for (int n = 3; n < 500; ++n) dec = dec && relativeGapThreshold(n) < relativeGapThreshold(n - 1);
  CHECK(dec, "PT5 strictly decreasing in N");

  {
    std::mt19937 rng(432);
    std::normal_distribution<double> N01(0.0, 1.0);
    const int N = 50, reps = 40000;
    std::vector<double> v(reps);
    for (int r = 0; r < reps; ++r) {
      double m = 0, s = 0;
      std::vector<double> x(N);
      for (int i = 0; i < N; ++i) { x[i] = N01(rng); m += x[i]; }
      m /= N;
      for (int i = 0; i < N; ++i) s += (x[i] - m) * (x[i] - m);
      v[r] = s / (N - 1);
    }
    double mean = 0, var = 0;
    for (double x : v) mean += x;
    mean /= reps;
    for (double x : v) var += (x - mean) * (x - mean);
    var /= (reps - 1);
    const double rel_sd = std::sqrt(var) / mean;
    const double claimed = std::sqrt(2.0 / (N - 1)), wrong = std::sqrt(1.0 / (N - 1));
    std::printf("[INFO] PT6 measured relative sd %.5f, claimed sqrt(2/(N-1)) %.5f\n", rel_sd, claimed);
    CHECK(std::abs(rel_sd - claimed) < 0.03 * claimed, "PT6 sample variance relative sd equals sqrt(2/(N-1)) within 3 percent");
    CHECK(std::abs(rel_sd - wrong) > 0.2 * wrong, "PT6 control: the wrong constant sqrt(1/(N-1)) is rejected");
  }

  std::ostringstream o;
  writePlaneTestValues(o, a);
  CHECK(fields(o.str()) == 10 && fields(planeTestValuesHeader()) == 10, "PT7 header and row both have 10 fields");
  double e, thr, pv, d1, d2, r10, r21, rg;
  int pp, sb;
  std::string row = o.str();
  for (char& c : row) if (c == ',') c = ' ';
  std::istringstream in(row);
  in >> e >> thr >> pv >> pp >> sb >> d1 >> d2 >> r10 >> r21 >> rg;
  CHECK(std::abs(e - a.edge_m) < 1e-5 && pp == 1 && sb == 1 && std::abs(rg - a.rel_gap_threshold) < 1e-5, "PT7 the row parses back to the values (default stream precision)");
  std::printf("ALL PASS\n");
  return 0;
}
