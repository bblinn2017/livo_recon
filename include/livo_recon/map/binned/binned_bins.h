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
//     fused   : R97A R3. Precision-weighted fusion, treating each return as an INDEPENDENT measurement of one point: with
//               Lambda_i = Sigma_i^-1 (each return's own covariance, clamped before inverting if near-singular), L = sum
//               Lambda_i, Lp = sum Lambda_i p_i, the representative is p* = L^-1 Lp with covariance L^-1. Unlike average/
//               shrunk, the representative's POSITION also changes (it is no longer the plain arithmetic mean) -- it is the
//               precision-weighted mean, so an anisotropic member with low noise along one axis pulls p* towards it on that
//               axis specifically.
//     fused_pose : as fused, plus a scene term voxel_map/binned/fused_pose_sigma_m^2 * I that does not shrink with k (the
//               part of the error shared by every return of the bin, e.g. pose error) -- see R97A_HANDOFF/code_specs/
//               R3_fused_bins.md.
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
#include <Eigen/Eigenvalues>

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

// R97A R3: Fused/FusedPose add precision-weighted fusion of a bin's returns (see this file's
// own header comment update below and R97A_HANDOFF/code_specs/R3_fused_bins.md).
enum class CovMode { Average = 0, Shrunk = 1, Fused = 2, FusedPose = 3 };

inline const char* covModeName(CovMode m)
{
  switch (m) {
    case CovMode::Shrunk: return "shrunk";
    case CovMode::Fused: return "fused";
    case CovMode::FusedPose: return "fused_pose";
    default: return "average";
  }
}
inline bool parseCovMode(const std::string& s, CovMode& out)
{
  if (s == "average") { out = CovMode::Average; return true; }
  if (s == "shrunk") { out = CovMode::Shrunk; return true; }
  if (s == "fused") { out = CovMode::Fused; return true; }
  if (s == "fused_pose") { out = CovMode::FusedPose; return true; }
  return false;
}
inline bool covModeIsFused(CovMode m) { return m == CovMode::Fused || m == CovMode::FusedPose; }

struct Options
{
  CovMode cov_mode = CovMode::Average;
  // R97A R3: voxel_map/binned/fused_pose_sigma_m -- only read/validated under cov_mode == FusedPose.
  double fused_pose_sigma_m = 0.0;
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
  // R97A R3: precision-weighted fusion sums (Fused/FusedPose only; left at zero and unused by
  // average/shrunk). L = sum Lambda_i (Lambda_i = Sigma_i^-1, clamped per BinSet::add()'s own
  // doc), Lp = sum Lambda_i * p_i -- both additive, kept per bin as returns arrive.
  M3 sum_L = M3::Zero();
  V3 sum_Lp = V3::Zero();
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
  // R97A R3: `opts.cov_mode` decides, for this BinSet's whole lifetime, whether add() computes
  // and accumulates the per-return precision (Lambda_i) -- average/shrunk do no extra work, so
  // the mode is fixed at construction rather than re-decided on every representatives() call.
  explicit BinSet(double bin_size, const Options& opts = Options())
    : bin_size_(bin_size > 1e-9 ? bin_size : 1e-9), opts_(opts) {}

  double binSize() const { return bin_size_; }
  std::size_t numBins() const { return bins_.size(); }
  long long numPoints() const { return n_points_; }
  const std::map<BinKey, Bin>& bins() const { return bins_; }

  // R91: UNIQUE bins that received at least one return since the last markFitted() (or since construction). A bin hit by several scans counts once; a bin opened in this period counts (it received a return).
  // Before the first fit this equals numBins(). R92: this is the count min_init / min_update read in a binned map (the only count).
  long long dirtyCount() const { return dirty_; }
  // Called by the owner when a fit has been made: the next period starts with no bin touched.
  void markFitted() { ++fit_epoch_; dirty_ = 0; }

  // R97A R3: cumulative count (never reset by markFitted()) of returns whose covariance needed
  // the numerical floor (smallest eigenvalue <= 1e-12 * trace) before inversion. Always 0 under
  // average/shrunk (Lambda_i is never computed).
  long long fusedClampedCount() const { return fused_clamped_; }

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
    if (covModeIsFused(opts_.cov_mode)) {
      M3 sigma = sensor_cov;
      Eigen::SelfAdjointEigenSolver<M3> es(sigma);
      const double trace = sigma.trace();
      if (es.eigenvalues().minCoeff() <= 1e-12 * trace) {
        sigma += (1e-12 * trace) * M3::Identity();
        ++fused_clamped_;
      }
      const M3 lambda = sigma.inverse();
      b.sum_L += lambda;
      b.sum_Lp += lambda * p;
    }
    ++n_points_;
    return is_new;
  }

  // The occupied bins as weighted points (see Reps). w = k; cov = C / k (average), C / k^2
  // (shrunk), L^-1 (fused), or L^-1 + s^2*I (fused_pose, s = o.fused_pose_sigma_m) -- L/Lp are
  // the precision-weighted sums add() accumulated (requires o.cov_mode to match the mode this
  // BinSet was constructed with, or L/Lp are all zero and this falls back to a degenerate
  // L^-1 that is not meaningful).
  Reps representatives(const Options& o) const
  {
    Reps r;
    r.p.reserve(bins_.size());
    r.cov.reserve(bins_.size());
    r.w.reserve(bins_.size());
    for (const auto& kv : bins_) {
      const Bin& b = kv.second;
      const double k = static_cast<double>(b.count);
      if (covModeIsFused(o.cov_mode)) {
        const M3 l_inv = b.sum_L.inverse();
        r.p.push_back(l_inv * b.sum_Lp);
        M3 cov = l_inv;
        if (o.cov_mode == CovMode::FusedPose)
          cov += (o.fused_pose_sigma_m * o.fused_pose_sigma_m) * M3::Identity();
        r.cov.push_back(cov);
      } else {
        r.p.push_back(b.sum_p / k);
        r.cov.push_back(b.sum_cov / ((o.cov_mode == CovMode::Shrunk) ? k * k : k));
      }
      r.w.push_back(k);
    }
    return r;
  }

private:
  double bin_size_;
  Options opts_;
  std::map<BinKey, Bin> bins_;
  long long n_points_ = 0;
  long long fit_epoch_ = 0;   // R91: incremented by markFitted()
  long long dirty_ = 0;       // R91: bins whose touch_epoch == fit_epoch_
  long long fused_clamped_ = 0;   // R97A R3
};

}  // namespace binned
}  // namespace livo_recon
