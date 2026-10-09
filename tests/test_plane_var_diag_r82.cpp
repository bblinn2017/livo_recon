// R82 (D2): unit test of the plane variance diagnostic (header only, ROS-free). Algebraic identities (rule 3.1). Planner sandbox, not evidence.
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

#include "livo_recon/lio/plane_var_diag.h"

using namespace livo_recon;
static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

int main()
{
  // Bins: monotone, ends, and a value is within 10^(1/40) of its bin centre.
  CHECK(pvdBin(0.0) == 0 && pvdBin(-1.0) == 0 && pvdBin(1e-12) == 0 && pvdBin(1.0) == kPvdBins - 1 && pvdBin(5.0) == kPvdBins - 1);
  int prev = 0; for (double v = 2e-9; v < 0.99; v *= 1.01) { const int b = pvdBin(v); CHECK(b >= prev); prev = b; }
  for (double v : {3e-9, 1e-6, 2.4e-5, 4e-4, 0.07}) { const double c = pvdBinCentre(pvdBin(v)); CHECK(c / v < 1.0600 * 1.0001 && v / c < 1.0600 * 1.0001); }
  // Quantiles of 100000 log-uniform values agree with the exact sample quantiles to the bin resolution.
  std::mt19937 rng(3); std::uniform_real_distribution<double> U(-7.0, -2.0);
  PlaneVarDiag d; std::vector<double> xs;
  for (int i = 0; i < 100000; ++i) { const double x = std::pow(10.0, U(rng)); xs.push_back(x); d.add(x, 2.0 * x, 1 + (i % 4)); }
  std::sort(xs.begin(), xs.end());
  const PlaneVarDiagSnapshot s = d.snapshot();
  CHECK(s.rows == 100000);
  for (double q : {0.1, 0.25, 0.5, 0.7, 0.9})
  {
    const double exact = xs[static_cast<size_t>(q * (xs.size() - 1))];
    const double got = pvdQuantile(s.fit, s.rows, q);
    CHECK(got / exact < 1.07 && exact / got < 1.07);
    CHECK(std::fabs(pvdQuantile(s.post, s.rows, q) / (2.0 * got) - 1.0) < 0.13);   // post = 2 fit: one bin step is 12 percent
  }
  // Exact floor counters.
  for (int k = 0; k < kPvdFloors; ++k) { const long exact = std::count_if(xs.begin(), xs.end(), [&](double x) { return x < kPvdFloorValues[k]; }); CHECK(s.fit_below[k] == exact); }
  // mean(1/n): n cycles 1,2,3,4 -> mean(1/n) = (1 + 1/2 + 1/3 + 1/4)/4.
  CHECK(std::fabs(s.sum_inv_n / s.rows - (1.0 + 0.5 + 1.0 / 3.0 + 0.25) / 4.0) < 1e-12);
  // distinct_frames < 1 is treated as 1; an empty diagnostic has zero quantiles.
  PlaneVarDiag e; CHECK(pvdQuantile(e.snapshot().fit, 0, 0.5) == 0.0); e.add(1e-5, 1e-5, 0); CHECK(std::fabs(e.snapshot().sum_inv_n - 1.0) < 1e-15);
  std::printf(g_fail ? "FAILED %d\n" : "ALL PASS\n", g_fail);
  return g_fail ? 1 : 0;
}
