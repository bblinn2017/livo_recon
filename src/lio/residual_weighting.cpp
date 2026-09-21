#include "livo_recon/lio/residual_weighting.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace livo_recon
{

namespace
{

using M66 = Eigen::Matrix<double, 6, 6>;
using V6  = Eigen::Matrix<double, 6, 1>;

// Same layout as residual_redundancy.cpp's own jacobianOf() -- rotation
// columns 0-2, position columns 3-5 -- duplicated rather than shared across
// a translation-unit boundary for a 6-line function; if this drifts from
// residual_redundancy.cpp's copy the two modules' notion of "direction" for
// a residual would silently disagree, so keep them textually identical.
V6 jacobianOf(const Residual& r)
{
  V6 j;
  j.segment<3>(0) = r.point_cross_normal;
  j.segment<3>(3) = r.normal;
  return j;
}

// Uninformative seed covariance for info_gain's per-frame running Sigma --
// large relative to any realistic sigma_squared (typically 1e-4 to 1e-2 in
// this codebase's units). NOT negligible in practice: CQ-80 (sections/
// round-80) MEASURED frame 1's own leverage directly over its 94 residuals
// and found a mean discount factor of 0.783 -- a ~22% discount on the very
// first frame, not the "leverage ~= 0 / no discount" this comment used to
// claim. Not tuned against data; a from-scratch design choice (rule 51a),
// documented so a reader can judge it directly rather than reverse-
// engineer it from behavior. (This whole arm is dead regardless -- CQ-70
// found it drives eee_01 from 28.000mm to 153,812,000.000mm -- so this is
// documentation of a dead path, not a live-default concern.)
constexpr double kInfoGainSeedCov     = 1.0e6;
constexpr double kInfoGainMinDiscount = 1.0e-3;

}  // namespace

CollapseStats applyResidualCollapse(std::vector<Residual>& residuals)
{
  CollapseStats stats{};

  std::unordered_map<const void*, std::vector<Residual*>> by_plane;
  for (Residual& r : residuals) {
    if (r.plane_id != nullptr) by_plane[r.plane_id].push_back(&r);
  }

  std::vector<Residual> kept;
  kept.reserve(residuals.size());

  // Residuals with no plane_id (never grouped) or in a singleton group pass
  // through untouched; every >= 2 group is replaced by its collapsed form.
  std::unordered_map<const void*, bool> is_multi;
  for (const auto& kv : by_plane) is_multi[kv.first] = kv.second.size() >= 2;

  for (const Residual& r : residuals) {
    if (r.plane_id == nullptr || !is_multi[r.plane_id]) kept.push_back(r);
  }

  for (const auto& kv : by_plane) {
    const auto& group = kv.second;
    if (group.size() < 2) continue;

    double sum_w = 0.0, sum_wr = 0.0;
    V3D sum_w_pcn = V3D::Zero(), sum_w_n = V3D::Zero();
    for (const Residual* r : group) {
      const double w = (r->sigma_squared > 0.0) ? (1.0 / r->sigma_squared) : 0.0;
      sum_w   += w;
      sum_wr  += w * r->r;
      sum_w_pcn.noalias() += w * r->point_cross_normal;
      sum_w_n.noalias()   += w * r->normal;
    }
    if (!(sum_w > 0.0)) continue;  // degenerate group (every member's sigma_squared <= 0) -- leave uncollapsed

    Residual c{};
    c.r                  = sum_wr / sum_w;
    c.point_cross_normal = sum_w_pcn / sum_w;
    c.normal             = (sum_w_n / sum_w).normalized();
    c.plane_id           = kv.first;
    c.plane_var_term     = group.front()->plane_var_term;  // identical across the group by construction
    c.t                  = group.front()->t;
    c.vis_state          = group.front()->vis_state;
    c.match_tier         = group.front()->match_tier;
    // Joint variance of the weighted mean under "all k members share one
    // true value": 1/sigma_c^2 = sum_i(1/sigma_i^2) / k^2 -- see this file's
    // header comment.
    const double k = static_cast<double>(group.size());
    c.sigma_squared = (k * k) / sum_w;

    kept.push_back(c);
    ++stats.groups_collapsed;
    stats.residuals_removed += static_cast<int>(group.size()) - 1;
  }

  residuals.swap(kept);
  return stats;
}

PerResidualStats applyPerResidualReweight(std::vector<Residual>& residuals,
                                          const std::string& per_residual_mode)
{
  PerResidualStats stats{};
  if (per_residual_mode == "off" || residuals.empty()) return stats;

  if (per_residual_mode == "count_weighted" || per_residual_mode == "count_weighted_renorm") {
    std::unordered_map<const void*, int> group_size;
    for (const Residual& r : residuals) {
      if (r.plane_id != nullptr) ++group_size[r.plane_id];
    }

    double sum_w_before = 0.0, sum_w_after = 0.0;
    std::vector<double> k_of(residuals.size(), 1.0);
    for (std::size_t i = 0; i < residuals.size(); ++i) {
      const Residual& r = residuals[i];
      const double k = (r.plane_id != nullptr) ? static_cast<double>(group_size[r.plane_id]) : 1.0;
      k_of[i] = k;
      if (r.sigma_squared > 0.0) sum_w_before += 1.0 / r.sigma_squared;
    }
    for (std::size_t i = 0; i < residuals.size(); ++i) {
      const double k = k_of[i];
      if (k <= 1.0) continue;  // no-op on an ungrouped/singleton residual
      Residual& r = residuals[i];
      r.sigma_squared *= k;
      if (r.plane_var_term > 0.0) r.plane_var_term *= k;  // item 1c
      ++stats.residuals_touched;
    }

    if (per_residual_mode == "count_weighted_renorm") {
      for (const Residual& r : residuals)
        if (r.sigma_squared > 0.0) sum_w_after += 1.0 / r.sigma_squared;
      if (sum_w_after > 0.0 && sum_w_before > 0.0) {
        const double renorm = sum_w_before / sum_w_after;  // > 1 whenever count_weighted shrank the total weight
        stats.renorm_factor = renorm;
        // Apply as a sigma_squared scale (1/renorm) so it composes with the
        // per-residual k-scale above via the same multiplicative mechanism
        // item 1c requires -- plane_var_term moves with it identically.
        for (Residual& r : residuals) {
          r.sigma_squared /= renorm;
          if (r.plane_var_term > 0.0) r.plane_var_term /= renorm;
        }
      }
    }

    double sum_scale = 0.0;
    for (std::size_t i = 0; i < residuals.size(); ++i) {
      if (k_of[i] <= 1.0) continue;
      const double applied = k_of[i] / ((per_residual_mode == "count_weighted_renorm") ? stats.renorm_factor : 1.0);
      sum_scale += applied;
    }
    if (stats.residuals_touched > 0) stats.mean_applied_scale = sum_scale / stats.residuals_touched;
    return stats;
  }

  if (per_residual_mode == "info_gain") {
    M66 Sigma = M66::Identity() * kInfoGainSeedCov;  // uninformative, reseeded every frame -- see header comment
    double sum_discount = 0.0;
    for (Residual& r : residuals) {
      if (!(r.sigma_squared > 0.0)) continue;
      const V6 j = jacobianOf(r);
      const double w = 1.0 / r.sigma_squared;
      const V6 Sigma_j = Sigma * j;
      const double leverage = w * j.dot(Sigma_j);
      double discount = 1.0 / (1.0 + leverage);
      discount = std::max(discount, kInfoGainMinDiscount);

      r.sigma_squared /= discount;
      if (r.plane_var_term > 0.0) r.plane_var_term /= discount;  // item 1c
      ++stats.residuals_touched;
      sum_discount += discount;

      // Sherman-Morrison covariance downdate for admitting this
      // observation (weight w, direction j) into the running Sigma -- uses
      // the UNFLOORED leverage/denominator (1+leverage), not the floored
      // discount, so the floor is purely a division-safety clamp on the
      // WEIGHT change and never distorts the covariance recursion itself.
      const double denom = 1.0 + leverage;
      if (denom > 0.0) Sigma.noalias() -= (w / denom) * (Sigma_j * Sigma_j.transpose());
    }
    if (stats.residuals_touched > 0) stats.mean_applied_scale = sum_discount / stats.residuals_touched;
    return stats;
  }

  return stats;  // unrecognized mode -- caller's cfg.mode() already refused this at config time
}

}  // namespace livo_recon
