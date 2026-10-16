#pragma once
// R82 (D2, F-502): write-only diagnostic of the plane variance term, as pure code (no Eigen, no ROS) so tests/test_plane_var_diag_r82.cpp can check the binning and the
// quantiles. VoxelPlane::gate() calls PlaneVarDiag::add() ONLY when log/voxel_map/plane_var_diag is true (default false: the call is skipped and nothing changes).
// What is recorded per gate() row that reaches the variance term: the plane variance term the fit gave (`fit`), the term after the scan term and before the clamp (`post`,
// equal to `fit` when plane_var_scan_s2 is 0), and 1 / max(1, distinct frames) (so mean s2/n = s2 * mean(1/n) for any s2). Histograms are log-spaced, 20 bins per decade over
// [1e-9, 1] (values outside go to the end bins), so a quantile read from them is exact to a factor of 10^(1/40) = 1.06 either way (geometric bin centre). The fractions below the
// candidate floors are EXACT counters (not read from the bins).
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>

namespace livo_recon
{

constexpr int kPvdBinsPerDecade = 20;
constexpr int kPvdDecades = 9;                                   // 1e-9 .. 1
constexpr int kPvdBins = kPvdBinsPerDecade * kPvdDecades + 2;    // + underflow, overflow
constexpr int kPvdFloors = 4;
constexpr double kPvdFloorValues[kPvdFloors] = {1e-5, 3e-5, 1e-4, 4e-4};   // the D2 candidates and the R81 clamp

inline int pvdBin(double v)
{
  if (!(v > 1e-9)) return 0;
  if (v >= 1.0) return kPvdBins - 1;
  const int b = 1 + static_cast<int>(std::floor((std::log10(v) + 9.0) * kPvdBinsPerDecade));
  return std::min(std::max(b, 1), kPvdBins - 2);
}

// Geometric centre of bin b (the underflow bin reports 1e-9, the overflow bin 1).
inline double pvdBinCentre(int b)
{
  if (b <= 0) return 1e-9;
  if (b >= kPvdBins - 1) return 1.0;
  return std::pow(10.0, -9.0 + (b - 0.5) / kPvdBinsPerDecade);
}

struct PlaneVarDiagSnapshot
{
  long rows = 0;
  std::array<long, kPvdBins> fit{}, post{};
  std::array<long, kPvdFloors> fit_below{};   // rows with fit < floor
  double sum_inv_n = 0.0;
};

// q in [0, 1]; the value of the bin that holds the q-th row (0 when there are no rows).
inline double pvdQuantile(const std::array<long, kPvdBins>& h, long total, double q)
{
  if (total <= 0) return 0.0;
  const long target = std::max<long>(1, static_cast<long>(std::ceil(q * static_cast<double>(total))));
  long cum = 0;
  for (int b = 0; b < kPvdBins; ++b)
  {
    cum += h[b];
    if (cum >= target) return pvdBinCentre(b);
  }
  return pvdBinCentre(kPvdBins - 1);
}

class PlaneVarDiag
{
public:
  void add(double fit, double post, int distinct_frames)
  {
    rows_.fetch_add(1, std::memory_order_relaxed);
    fit_[pvdBin(fit)].fetch_add(1, std::memory_order_relaxed);
    post_[pvdBin(post)].fetch_add(1, std::memory_order_relaxed);
    for (int k = 0; k < kPvdFloors; ++k)
      if (fit < kPvdFloorValues[k]) below_[k].fetch_add(1, std::memory_order_relaxed);
    const double inv = 1.0 / static_cast<double>(std::max(1, distinct_frames));
    double cur = sum_inv_n_.load(std::memory_order_relaxed);
    while (!sum_inv_n_.compare_exchange_weak(cur, cur + inv, std::memory_order_relaxed)) {}
  }
  PlaneVarDiagSnapshot snapshot() const
  {
    PlaneVarDiagSnapshot s;
    s.rows = rows_.load();
    for (int b = 0; b < kPvdBins; ++b) { s.fit[b] = fit_[b].load(); s.post[b] = post_[b].load(); }
    for (int k = 0; k < kPvdFloors; ++k) s.fit_below[k] = below_[k].load();
    s.sum_inv_n = sum_inv_n_.load();
    return s;
  }

private:
  std::atomic<long> rows_{0};
  std::array<std::atomic<long>, kPvdBins> fit_{}, post_{};
  std::array<std::atomic<long>, kPvdFloors> below_{};
  std::atomic<double> sum_inv_n_{0.0};
};

}  // namespace livo_recon
