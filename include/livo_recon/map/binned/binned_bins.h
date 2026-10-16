#pragma once
// R88: the bin accumulator of the "binned" map type. Header-only, Eigen + std only (no ROS), so tests/test_binned_bins_r88.cpp can check it alone.
//
// What it is. A voxel keeps its returns in small cubic bins (edge = bin_size_fraction x the voxel's own edge) instead of as raw points. Each bin holds the number of returns, the sum of their positions and the sums of their
// sensor covariances. When the voxel is fitted, every OCCUPIED bin is turned into one representative point (the mean of its returns) with a covariance and a weight, and the PCA fit (VoxelPlane::update())
// takes those representatives as weighted points.
//
// One choice (the config key voxel_map/binned/cov_mode). R95: the weight of a bin is always the number of returns it holds; the R88 `uniform` weight (w = 1) and its key voxel_map/binned/weight_mode were removed.
//
//   cov_mode (what covariance the representative carries; k = returns in the bin, C = the sum of their sensor covariances)
//     average : C / k.      The noise of the representative is the average member noise. This is the assumption that the members' errors are FULLY CORRELATED (a stationary sensor repeating the same beam),
//                           so averaging does not reduce the error of the mean.
//     shrunk  : C / k^2.    The noise of the representative is the noise of a mean of k INDEPENDENT returns, so it falls as 1/k.
//   weight (how much one bin contributes to the sums; not a choice since R95)
//     w = k.      A bin counts as the k returns it holds.
//
// Every statistic that a raw point would add with weight 1 is added with weight w = k by the representative; the covariance sums are w x (the representative's covariance). So the two combinations are:
//   average : sum of covariances = C (the raw sum), N = total returns.   (the raw statistics, with each bin's own scatter replaced by its mean)
//   shrunk  : sum of covariances = C / k, N = total returns.
//
// R90 (PCA on the binned map): representatives() hands the same occupied bins to the PCA fit (VoxelPlane::update()) as one point per bin (the mean of its returns), with the covariance of cov_mode and the weight
// w = k. The PCA fit then takes the weighted mean and covariance of those points and propagates the covariance of each into the plane variance.
//
// Not kept (and so not supported): the pose covariance. Every fit uses the sensor covariance only.
// Determinism: bins are visited in ascending key order (std::map), so the sums do not depend on hash order or thread count.
#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace livo_recon
{
namespace binned
{

using V3 = Eigen::Vector3d;
using M3 = Eigen::Matrix3d;

enum class CovMode { Average = 0, Shrunk = 1 };

inline const char* covModeName(CovMode m) { return m == CovMode::Shrunk ? "shrunk" : "average"; }
inline bool parseCovMode(const std::string& s, CovMode& out)
{
  if (s == "average") { out = CovMode::Average; return true; }
  if (s == "shrunk") { out = CovMode::Shrunk; return true; }
  return false;
}

struct Options
{
  CovMode cov_mode = CovMode::Average;
};

struct BinKey
{
  int x = 0, y = 0, z = 0;
  bool operator<(const BinKey& o) const
  {
    if (x != o.x) return x < o.x;
    if (y != o.y) return y < o.y;
    return z < o.z;
  }
  bool operator==(const BinKey& o) const { return x == o.x && y == o.y && z == o.z; }
};

inline BinKey binKeyOf(const V3& p, double bin_size)
{
  BinKey k;
  k.x = static_cast<int>(std::floor(p.x() / bin_size));
  k.y = static_cast<int>(std::floor(p.y() / bin_size));
  k.z = static_cast<int>(std::floor(p.z() / bin_size));
  return k;
}

struct Bin
{
  int count = 0;                 // returns in the bin
  V3 sum_p = V3::Zero();
  M3 sum_cov = M3::Zero();           // sensor covariances, all members
  long long touch_epoch = -1;        // R91: the fit epoch in which this bin last received a return (-1: never)
};

// One point per occupied bin, in ascending key order: the mean of the bin's returns, the covariance of cov_mode, and its weight.
struct Reps
{
  std::vector<V3> p;
  std::vector<M3> cov;
  std::vector<double> w;
};

class BinSet
{
public:
  explicit BinSet(double bin_size) : bin_size_(bin_size > 1e-9 ? bin_size : 1e-9) {}

  double binSize() const { return bin_size_; }
  std::size_t numBins() const { return bins_.size(); }
  long long numPoints() const { return n_points_; }
  const std::map<BinKey, Bin>& bins() const { return bins_; }

  // R91: UNIQUE bins that received at least one return since the last markFitted() (or since construction). A bin hit by several scans counts once; a bin opened in this period counts (it received a return).
  // Before the first fit this equals numBins(). R92: this is the count min_init / min_update read in a binned map (the only count).
  long long dirtyCount() const { return dirty_; }
  // Called by the owner when a fit has been made: the next period starts with no bin touched.
  void markFitted() { ++fit_epoch_; dirty_ = 0; }

  // Adds one return. Returns true when the return opened a new bin.
  bool add(const V3& p, const M3& sensor_cov)
  {
    const BinKey key = binKeyOf(p, bin_size_);
    auto it = bins_.find(key);
    const bool is_new = (it == bins_.end());
    if (is_new) it = bins_.emplace(key, Bin()).first;
    Bin& b = it->second;
    if (b.touch_epoch != fit_epoch_) { b.touch_epoch = fit_epoch_; ++dirty_; }
    ++b.count;
    b.sum_p += p;
    b.sum_cov += sensor_cov;
    ++n_points_;
    return is_new;
  }

  // The occupied bins as weighted points (see Reps). w = k; cov = C / k (average) or C / k^2 (shrunk).
  Reps representatives(const Options& o) const
  {
    Reps r;
    r.p.reserve(bins_.size());
    r.cov.reserve(bins_.size());
    r.w.reserve(bins_.size());
    for (const auto& kv : bins_) {
      const Bin& b = kv.second;
      const double k = static_cast<double>(b.count);
      r.p.push_back(b.sum_p / k);
      r.cov.push_back(b.sum_cov / ((o.cov_mode == CovMode::Shrunk) ? k * k : k));
      r.w.push_back(k);
    }
    return r;
  }

private:
  double bin_size_;
  std::map<BinKey, Bin> bins_;
  long long n_points_ = 0;
  long long fit_epoch_ = 0;   // R91: incremented by markFitted()
  long long dirty_ = 0;       // R91: bins whose touch_epoch == fit_epoch_
};

}  // namespace binned
}  // namespace livo_recon
