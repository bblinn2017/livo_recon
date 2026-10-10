#pragma once

// ============================================================================
// R91E: residual binning by constraint direction ("a shared shift per direction bin").
//
// WHY (planner's analysis of the R91D dumps; MEASURED offline on dumped rows, not yet seen in the node). The solver adds up the LiDAR rows as if they were independent
// (accumulateLioResiduals: HtH = sum w h h^T, Htz = sum w r h, w = 1/sigma_squared). With the two-row line form (R91D) the data-only position offset of the stationary frames
// is below its claimed covariance for m3r3 (position statistic 1.49 and 1.31, expected 3) but only because the claimed row sd (22 to 27 mm, the sensor-range weight floor) is
// larger than the real post-fit row scatter (3.7, 4.9, 16.8 mm for plane, line u0, line u1 rows). With row variance set to the measured scatter the statistic would be 6.4, 7.4
// and 14.7 (m3r3 b0, m3r3 z0, m1r1 b0): rows are not independent. A shared shift of 0.34, 0.38, 0.62 mm per direction bin brings it to 3. This module adds that shift.
//
// MODEL. Every row has a constraint direction h (the position part of its Jacobian: the plane normal, or the u0 / u1 axis of a line voxel). Fold h by sign onto the nearest of
// 13 fixed axes a_k (the sign-folded directions of the 3x3x3 lattice, table below). All rows of bin k share one unknown shift c_k along a_k with variance tau^2, as if the
// part of the map seen in that direction were displaced by c_k a_k. Then the residual vector z of the bin has covariance W^-1 + tau^2 u u^T, with u_i = h_i . a_k (signed
// projection of row i on the bin axis). Woodbury, per bin, with
//     v = sum_i w_i u_i J_i          (J_i = [point_cross_normal; normal], 6 values: the Jacobian row, rotation first, as accumulateLioResiduals uses it)
//     s = sum_i w_i u_i^2
//     t = sum_i w_i u_i r_i
// gives the corrected normal equations
//     HtH' = HtH - sum_k v_k v_k^T / (1/tau^2 + s_k)          Htz' = Htz - sum_k v_k t_k / (1/tau^2 + s_k).
// Each bin's term is positive semidefinite and cannot exceed that bin's own information (Cauchy-Schwarz: v v^T <= s * sum w J J^T), so HtH' stays positive semidefinite.
// Information along a bin axis saturates near 1/tau^2 when s grows: more rows in one direction no longer add evidence without limit. Directions away from the bin axes are
// reduced only by the rows' own overlap with the axis. tau -> 0 is the identity (the key refuses 0; mode "off" is the identity path).
//
// WHAT IT IS FOR. The experiment (R91E): does bounding the evidence per direction make the claimed covariance honest where it fails (m1r1) without changing the estimator
// elsewhere, and what does it do to the motion onset. It does NOT touch dependence across scans (per-voxel persistence, slow wander): that needs another mechanism.
//
// DETERMINISM AND COST. One serial pass over the residuals in index order (the same order as accumulateLioResiduals), 13 fixed accumulators on the stack (no heap, no hash
// map, no pointer keys, no OMP, no atomics, no thread_local), bin chosen by the largest |h . a_k| with ties to the lowest k, then the 13 corrections applied in the fixed order
// k = 0..12. The result does not depend on thread count, allocator or pointer values. Cost: 13 dot products and one 6-vector update per row, about the cost of the accumulator
// itself. mode "off" never enters the function (the caller gates on on()) and never touches HtH or Htz.
//
// Header-only and templated on the residual type (livo_recon::Residual in the node, a mock with the same member names in tests/test_residual_binning_r91e.cpp).
// Members read: normal (V3D), point_cross_normal (V3D), sigma_squared, r, row_kind.
// ============================================================================

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "livo_recon/lio/line_residual.h"   // kRowLineDist

namespace livo_recon
{

struct ResidualBinningOptions
{
  std::string mode = "off";        // "off" | "direction"
  std::string axes = "lattice13";  // the only table in R91E
  double shift_sd_m = 0.0006;      // tau: sd of the shared shift of a bin, metres (> 0)
  int min_rows = 20;               // a bin with fewer rows than this is left alone
  bool on() const { return mode != "off"; }
};

namespace residual_binning
{
constexpr int kAxes = 13;

// The 13 sign-folded axes of the 3x3x3 lattice, in this fixed order (a sign flip of an axis changes nothing: v v^T and v t are invariant).
inline const double (&axisTable())[kAxes][3]
{
  static const double k[kAxes][3] = {
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {0.0, 0.0, 1.0},
      {0.70710678118654757, 0.70710678118654757, 0.0},
      {0.70710678118654757, -0.70710678118654757, 0.0},
      {0.70710678118654757, 0.0, 0.70710678118654757},
      {0.70710678118654757, 0.0, -0.70710678118654757},
      {0.0, 0.70710678118654757, 0.70710678118654757},
      {0.0, 0.70710678118654757, -0.70710678118654757},
      {0.57735026918962573, 0.57735026918962573, 0.57735026918962573},
      {0.57735026918962573, 0.57735026918962573, -0.57735026918962573},
      {0.57735026918962573, -0.57735026918962573, 0.57735026918962573},
      {-0.57735026918962573, 0.57735026918962573, 0.57735026918962573}};
  return k;
}

// Bin of a direction: the axis with the largest |h . a_k|, ties to the lowest k (strict > comparison). u receives the signed projection h . a_k.
inline int binOf(double hx, double hy, double hz, double& u)
{
  const double (&a)[kAxes][3] = axisTable();
  int best = 0;
  double best_abs = -1.0, best_u = 0.0;
  for (int k = 0; k < kAxes; ++k) {
    const double d = hx * a[k][0] + hy * a[k][1] + hz * a[k][2];
    const double ad = d < 0.0 ? -d : d;
    if (ad > best_abs) { best_abs = ad; best = k; best_u = d; }
  }
  u = best_u;
  return best;
}

struct BinOut
{
  int rows = 0;
  double s = 0.0, t = 0.0;
  double v[6] = {0, 0, 0, 0, 0, 0};
  bool applied = false;
};
}  // namespace residual_binning

struct ResidualBinningStats
{
  int n_rows = 0;            // residuals seen
  int n_corrected_rows = 0;  // rows that belong to a corrected bin
  int n_skipped_rows = 0;    // distance rows (direction set by the noise) and rows with a non-finite or non-positive variance or a non-finite direction
  int n_bins_active = 0;     // bins that reached min_rows
  double trace_before = 0.0, trace_after = 0.0;
  double gnorm_before = 0.0, gnorm_after = 0.0;
  std::array<residual_binning::BinOut, residual_binning::kAxes> bins;
};

// In place on (HtH, Htz) as built by accumulateLioResiduals (and nothing else yet). Returns the magnitudes for the engagement log and the per-bin sums for the detail log.
template <typename ResidualT>
ResidualBinningStats applyResidualBinning(const std::vector<ResidualT>& residuals, const ResidualBinningOptions& opts,
                                          Eigen::Matrix<double, 6, 6>& HtH, Eigen::Matrix<double, 6, 1>& Htz)
{
  using namespace residual_binning;
  ResidualBinningStats st;
  st.n_rows = static_cast<int>(residuals.size());
  st.trace_before = HtH.trace();
  st.gnorm_before = Htz.norm();
  st.trace_after = st.trace_before;
  st.gnorm_after = st.gnorm_before;
  if (!opts.on()) return st;

  std::array<int, kAxes> n{};
  std::array<double, kAxes> s{}, t{};
  std::array<Eigen::Matrix<double, 6, 1>, kAxes> v;
  for (int k = 0; k < kAxes; ++k) v[k].setZero();

  for (std::size_t i = 0; i < residuals.size(); ++i) {
    const ResidualT& res = residuals[i];
    if (res.row_kind == kRowLineDist) { ++st.n_skipped_rows; continue; }
    const double sig = res.sigma_squared;
    const double hx = res.normal.x(), hy = res.normal.y(), hz = res.normal.z();
    if (!(sig > 0.0) || !std::isfinite(sig) || !std::isfinite(hx + hy + hz + res.r)) { ++st.n_skipped_rows; continue; }
    double u = 0.0;
    const int k = binOf(hx, hy, hz, u);
    const double w = 1.0 / sig;   // the same expression as accumulateLioResiduals
    const double wu = w * u;
    v[k].template segment<3>(0).noalias() += wu * res.point_cross_normal;
    v[k].template segment<3>(3).noalias() += wu * res.normal;
    s[k] += wu * u;
    t[k] += wu * res.r;
    ++n[k];
  }

  const double inv_tau2 = 1.0 / (opts.shift_sd_m * opts.shift_sd_m);
  for (int k = 0; k < kAxes; ++k) {
    BinOut& b = st.bins[k];
    b.rows = n[k]; b.s = s[k]; b.t = t[k];
    for (int j = 0; j < 6; ++j) b.v[j] = v[k](j);
    if (n[k] < opts.min_rows || n[k] < 1) continue;
    const double denom = inv_tau2 + s[k];
    HtH.noalias() -= (v[k] * v[k].transpose()) / denom;
    Htz.noalias() -= v[k] * (t[k] / denom);
    b.applied = true;
    ++st.n_bins_active;
    st.n_corrected_rows += n[k];
  }
  st.trace_after = HtH.trace();
  st.gnorm_after = Htz.norm();
  return st;
}

}  // namespace livo_recon
