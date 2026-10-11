#pragma once
// R88: the bin accumulator of the "binned" map type. Header-only, Eigen + std only (no ROS), so tests/test_binned_bins_r88.cpp can check it alone.
//
// What it is. A voxel keeps its returns in small cubic bins (edge = bin_size_fraction x the voxel's own edge) instead of as raw points. Each bin holds the number of returns, the sum of their positions and the sums of their
// sensor covariances. When the voxel is fitted, every OCCUPIED bin is turned into one representative point (the mean of its returns) with a covariance and a weight, and those representatives are summed into the same additive
// statistics that the debiased plane fit keeps for raw points (VoxelPlane::N_acc_, Sp_, Spp_, Scov_, Scov_sensor_, V_, W_, sum_sensor_var_). The debiased fit itself is not changed.
//
// Two independent choices (both are config keys, voxel_map/binned/cov_mode and voxel_map/binned/weight_mode):
//
//   cov_mode (what covariance the representative carries; k = returns in the bin, C = the sum of their sensor covariances)
//     average : C / k.      The noise of the representative is the average member noise. This is the assumption that the members' errors are FULLY CORRELATED (a stationary sensor repeating the same beam),
//                           so averaging does not reduce the error of the mean.
//     shrunk  : C / k^2.    The noise of the representative is the noise of a mean of k INDEPENDENT returns, so it falls as 1/k.
//   weight_mode (how much one bin contributes to the sums)
//     count   : w = k.      A bin counts as the k returns it holds.
//     uniform : w = 1.      A bin counts as one observation however many returns fell in it.
//
// Every statistic that a raw point would add with weight 1 is added with weight w by the representative; the covariance sums are w x (the representative's covariance). So the four combinations are:
//   average + count   : sum of covariances = C (the raw sum), N = total returns.   (the raw statistics, with each bin's own scatter replaced by its mean)
//   average + uniform : sum of covariances = C / k, N = number of bins.
//   shrunk  + count   : sum of covariances = C / k, N = total returns.
//   shrunk  + uniform : sum of covariances = C / k^2, N = number of bins.
//
// Trusted returns. The debiased fit subtracts only the covariance of returns from a trusted (post-bootstrap) frame (VoxelPlane::Scov_sensor_). A bin therefore keeps a second covariance sum over its trusted members.
// The representative's trusted covariance is that sum divided by the same divisor as the all-member sum (k or k^2), so a bin that mixes bootstrap and live returns subtracts less, never more.
//
// R90 (PCA on the binned map): representatives() hands the same occupied bins to the PCA fit (VoxelPlane::update()) as one point per bin (the mean of its returns), with the same covariance (cov_mode) and the same weight
// (weight_mode) as moments() uses. The PCA fit then takes the weighted mean and covariance of those points and propagates the covariance of each into the plane variance.
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
enum class WeightMode { Count = 0, Uniform = 1 };

inline const char* covModeName(CovMode m) { return m == CovMode::Shrunk ? "shrunk" : "average"; }
inline const char* weightModeName(WeightMode m) { return m == WeightMode::Uniform ? "uniform" : "count"; }
inline bool parseCovMode(const std::string& s, CovMode& out)
{
  if (s == "average") { out = CovMode::Average; return true; }
  if (s == "shrunk") { out = CovMode::Shrunk; return true; }
  return false;
}
inline bool parseWeightMode(const std::string& s, WeightMode& out)
{
  if (s == "count") { out = WeightMode::Count; return true; }
  if (s == "uniform") { out = WeightMode::Uniform; return true; }
  return false;
}

struct Options
{
  CovMode cov_mode = CovMode::Average;
  WeightMode weight_mode = WeightMode::Count;
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
  int count_trusted = 0;         // of which from a trusted (post-bootstrap) frame
  V3 sum_p = V3::Zero();
  M3 sum_cov = M3::Zero();           // sensor covariances, all members
  M3 sum_cov_trusted = M3::Zero();   // sensor covariances, trusted members only
  long long touch_epoch = -1;        // R91: the fit epoch in which this bin last received a return (-1: never)
};

// The additive statistics the debiased fit keeps, built from the occupied bins. sp, spp are about `ref`; scov, scov_sensor, v, w are origin-referenced (as VoxelPlane's Scov_, Scov_sensor_, V_, W_).
struct Moments
{
  double n = 0.0;                    // sum of weights (VoxelPlane::N_acc_)
  std::size_t n_bins = 0;
  long long n_points = 0;
  V3 sp = V3::Zero();
  M3 spp = M3::Zero();
  M3 scov = M3::Zero();
  M3 scov_sensor = M3::Zero();
  double sum_sensor_var = 0.0;
  M3 v[3][3];
  M3 w[3];
  Moments()
  {
    for (int a = 0; a < 3; ++a) {
      w[a].setZero();
      for (int b = 0; b < 3; ++b) v[a][b].setZero();
    }
  }
};

// One point per occupied bin, in ascending key order: the mean of the bin's returns, the covariance that moments() would add for it (all members), and its weight.
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

  // Adds one return. `trusted` = the return comes from a post-bootstrap frame (its covariance may be subtracted by the debiased fit). Returns true when the return opened a new bin.
  bool add(const V3& p, const M3& sensor_cov, bool trusted)
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
    if (trusted) {
      ++b.count_trusted;
      b.sum_cov_trusted += sensor_cov;
    }
    ++n_points_;
    return is_new;
  }

  // The statistics of the occupied bins under `o`, with positions taken relative to `ref` in sp/spp (pass the plane's reference point, or zero).
  Moments moments(const Options& o, const V3& ref) const
  {
    Moments m;
    m.n_bins = bins_.size();
    m.n_points = n_points_;
    for (const auto& kv : bins_) {
      const Bin& b = kv.second;
      const double k = static_cast<double>(b.count);
      const V3 pbar = b.sum_p / k;
      const double w = (o.weight_mode == WeightMode::Count) ? k : 1.0;
      const double div = (o.cov_mode == CovMode::Shrunk) ? k * k : k;
      const M3 cov_all = b.sum_cov / div;
      const M3 cov_tr = b.sum_cov_trusted / div;
      const V3 d = pbar - ref;
      m.n += w;
      m.sp += w * d;
      m.spp.noalias() += w * (d * d.transpose());
      m.scov += w * cov_all;
      m.scov_sensor += w * cov_tr;
      m.sum_sensor_var += w * cov_tr.trace() / 3.0;
      for (int a = 0; a < 3; ++a) {
        m.w[a] += (w * pbar(a)) * cov_all;
        for (int c = a; c < 3; ++c) {
          const M3 term = (w * pbar(a) * pbar(c)) * cov_all;
          m.v[a][c] += term;
          if (c != a) m.v[c][a] += term;
        }
      }
    }
    return m;
  }

  // The occupied bins as weighted points (see Reps). Same weights and covariances as moments(): w = k (count) or 1 (uniform); cov = C / k (average) or C / k^2 (shrunk).
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
      r.w.push_back((o.weight_mode == WeightMode::Count) ? k : 1.0);
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
