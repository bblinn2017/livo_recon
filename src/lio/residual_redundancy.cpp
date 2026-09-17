#include "livo_recon/lio/residual_redundancy.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace livo_recon
{

namespace
{

using M66 = Eigen::Matrix<double, 6, 6>;
using V6  = Eigen::Matrix<double, 6, 1>;

V6 jacobianOf(const Residual& r)
{
  V6 j;
  j.segment<3>(0) = r.point_cross_normal;  // rotation columns, matches accumulateLioResiduals()
  j.segment<3>(3) = r.normal;              // position columns
  return j;
}

// Minimum eigenvalue of HtH's position block (rows/cols 3-5), the same
// quantity frame_stats.txt's h_pp_min_eig column reports.
double positionMinEig(const M66& HtH)
{
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(HtH.block<3, 3>(3, 3));
  return es.eigenvalues().minCoeff();
}

enum class DegenerateReason { kNone, kPlaneVar, kResidualVar };

struct GroupCorrection
{
  M66 naive_HtH = M66::Zero();
  V6  naive_Htz = V6::Zero();
  M66 corrected_HtH = M66::Zero();
  V6  corrected_Htz = V6::Zero();
  int n_raw = 0;
  DegenerateReason degenerate = DegenerateReason::kNone;  // set iff n_raw==0
};

// Computes one plane group's naive (what accumulateLioResiduals() already
// added for these indices) and Woodbury-corrected contributions, with the
// per-group max_discount cap applied to the correction's information-trace.
// Returns std::nullopt (via n_raw==0) if the group is degenerate (rho*pv
// non-positive, or would make some residual's independent variance
// non-positive) -- such a group is left uncorrected, not force-fit.
GroupCorrection computeGroupCorrection(const std::vector<const Residual*>& group,
                                        double rho, double max_discount)
{
  GroupCorrection out;
  const double pv = group.front()->plane_var_term;
  const double shared = rho * pv;
  if (shared <= 0.0) {
    out.degenerate = DegenerateReason::kPlaneVar;  // n_raw stays 0 -- caller skips
    return out;
  }

  std::vector<double> w_indep;
  w_indep.reserve(group.size());
  for (const Residual* r : group) {
    const double sigma_indep2 = r->sigma_squared - shared;
    if (!(sigma_indep2 > 0.0)) {
      out.degenerate = DegenerateReason::kResidualVar;  // shared term consumes the whole variance
      return out;
    }
    w_indep.push_back(1.0 / sigma_indep2);
  }

  V6 sumJ = V6::Zero();
  double sumWr = 0.0, sumW = 0.0;
  for (size_t i = 0; i < group.size(); ++i) {
    const Residual& r = *group[i];
    const V6 j = jacobianOf(r);
    const double w = w_indep[i];
    out.naive_HtH.noalias() += (1.0 / r.sigma_squared) * (j * j.transpose());
    out.naive_Htz.noalias() += (1.0 / r.sigma_squared) * r.r * j;
    out.corrected_HtH.noalias() += w * (j * j.transpose());
    out.corrected_Htz.noalias() += w * r.r * j;
    sumJ.noalias() += w * j;
    sumWr += w * r.r;
    sumW += w;
  }

  const double c = shared / (1.0 + shared * sumW);
  out.corrected_HtH.noalias() -= c * (sumJ * sumJ.transpose());
  out.corrected_Htz.noalias() -= c * sumWr * sumJ;

  const double naive_trace = out.naive_HtH.trace();
  if (naive_trace > 0.0) {
    const double discount = 1.0 - out.corrected_HtH.trace() / naive_trace;
    if (discount > max_discount) {
      // Shrink the correction (not the naive baseline) so the realized
      // discount is exactly max_discount -- a convex blend between the
      // uncapped correction and the naive (zero-discount) baseline.
      const double scale = max_discount / discount;
      out.corrected_HtH = out.naive_HtH + scale * (out.corrected_HtH - out.naive_HtH);
      out.corrected_Htz = out.naive_Htz + scale * (out.corrected_Htz - out.naive_Htz);
    }
  }

  out.n_raw = static_cast<int>(group.size());
  return out;
}

}  // namespace

namespace
{

// Position-block (rows/cols 3-5) minimum eigenvalue AND its eigenvector,
// embedded back into 6D (rotation components zero) -- woodbury_directional
// needs the direction, not just the value positionMinEig() returns.
V6 positionMinEigVector(const M66& HtH)
{
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(HtH.block<3, 3>(3, 3));
  Eigen::Index idx;
  es.eigenvalues().minCoeff(&idx);
  V6 v = V6::Zero();
  v.segment<3>(3) = es.eigenvectors().col(idx).normalized();
  return v;
}

}  // namespace

ResidualRedundancyStats applyResidualRedundancyCorrection(
    const std::vector<Residual>& residuals,
    const ResidualRedundancyOptions& opts,
    EkfUpdate& ekf)
{
  if (opts.mode == "woodbury_divpos") {
    throw std::runtime_error(
        "lio/residual_redundancy/mode=woodbury_divpos is RETIRED (CQ-31): the "
        "Woodbury group correction is a rank-one downdate of HtH, so by Weyl's "
        "inequality h_pp_min_eig can only fall, never be preserved by any "
        "choice of GAMMA except on a measure-zero alignment -- confirmed "
        "empirically in CQ-28's own sweep (GAMMA search degenerated to 0 on "
        "every tested frame). Use woodbury_rescale or woodbury_directional.");
  }

  ResidualRedundancyStats stats{};

  std::unordered_map<const void*, std::vector<const Residual*>> by_plane;
  for (const Residual& r : residuals) {
    if (r.plane_id != nullptr) by_plane[r.plane_id].push_back(&r);
  }

  M66 delta_HtH_total = M66::Zero();
  V6  delta_Htz_total = V6::Zero();
  double naive_trace_total = 0.0, corrected_trace_total = 0.0;

  // CQ-31 item 7: this loop, and everything through redund_info_ratio below,
  // runs UNCONDITIONALLY -- including opts.mode == "off" -- so naive_
  // info_gain/woodbury_info_gain are always populated. Only the ekf.HtH/Htz
  // mutation further down is mode-gated.
  for (const auto& kv : by_plane) {
    const auto& group = kv.second;
    if (group.size() < 2) continue;
    ++stats.redund_groups_seen;

    const GroupCorrection gc = computeGroupCorrection(group, opts.rho, opts.max_discount);
    if (gc.n_raw == 0) {
      // degenerate group, left uncorrected -- CQ-34 item 4: tally WHICH
      // degeneracy fired, so a zero redund_groups can be told apart from
      // "plenty of groups, all declined" rather than reading identically
      // to "no group ever reached size >= 2".
      if (gc.degenerate == DegenerateReason::kPlaneVar) {
        ++stats.redund_groups_degenerate_pv;
      } else if (gc.degenerate == DegenerateReason::kResidualVar) {
        ++stats.redund_groups_degenerate_var;
      }
      continue;
    }

    ++stats.redund_groups;
    stats.redund_n_raw += gc.n_raw;

    const double naive_trace = gc.naive_HtH.trace();
    const double corrected_trace = gc.corrected_HtH.trace();
    naive_trace_total += naive_trace;
    corrected_trace_total += corrected_trace;
    if (naive_trace > 0.0)
      stats.redund_n_eff += static_cast<int>(std::ceil(gc.n_raw * (corrected_trace / naive_trace)));

    delta_HtH_total.noalias() += (gc.corrected_HtH - gc.naive_HtH);
    delta_Htz_total.noalias() += (gc.corrected_Htz - gc.naive_Htz);
  }

  // naive_info_gain/woodbury_info_gain are traces over the FULL 6x6, matching
  // h_pp_min_eig's own convention of reading off the position block from the
  // full matrix rather than a separately-tracked sub-accumulator.
  stats.naive_info_gain = naive_trace_total;
  stats.woodbury_info_gain = corrected_trace_total;

  if (stats.redund_groups == 0) return stats;  // nothing to apply; info_ratio/gains stay at defaults

  stats.redund_info_ratio = (naive_trace_total > 0.0) ? (corrected_trace_total / naive_trace_total) : 1.0;

  if (!opts.on()) return stats;  // mode == "off": stats computed above, ekf untouched

  if (opts.mode == "woodbury") {
    ekf.HtH += delta_HtH_total;
    ekf.Htz += delta_Htz_total;
    return stats;
  }

  if (opts.mode == "woodbury_rescale") {
    // Apply the downdate, then rescale the WHOLE post-correction system
    // (HtH and Htz together, so the solved mean x = HtH^-1 Htz is UNCHANGED
    // -- only the resulting covariance/information magnitude moves) by the
    // scalar that puts the position block's minimum eigenvalue back to
    // exactly its pre-correction value. One scalar, no search.
    const double naive_min_eig = positionMinEig(ekf.HtH);
    const M66 HtH_corrected = ekf.HtH + delta_HtH_total;
    const V6  Htz_corrected = ekf.Htz + delta_Htz_total;
    const double corrected_min_eig = positionMinEig(HtH_corrected);
    const double s = (corrected_min_eig > 0.0) ? (naive_min_eig / corrected_min_eig) : 1.0;
    ekf.HtH = s * HtH_corrected;
    ekf.Htz = s * Htz_corrected;
    stats.redund_info_ratio *= s;
    return stats;
  }

  if (opts.mode == "woodbury_directional") {
    // Project the downdate onto the complement of the NAIVE HtH's own
    // position-block minimum eigenvector v, so the Rayleigh quotient along v
    // -- v^T HtH v -- is unchanged EXACTLY (v^T delta_proj v == 0 by
    // construction): the starved direction is untouched, not merely
    // rescaled along with everything else. Note this protects v's own
    // direction specifically; it does not guarantee the CORRECTED matrix's
    // eventual minimum eigenvector is still v (the correction could in
    // principle open a new weak direction elsewhere) -- that is a separate,
    // reportable observation, not something this projection can prevent by
    // its own construction.
    const V6 v = positionMinEigVector(ekf.HtH);
    const M66 P_perp = M66::Identity() - v * v.transpose();
    const M66 delta_HtH_proj = P_perp * delta_HtH_total * P_perp;
    const V6  delta_Htz_proj = P_perp * delta_Htz_total;
    ekf.HtH += delta_HtH_proj;
    ekf.Htz += delta_Htz_proj;
    const double applied_trace = naive_trace_total + delta_HtH_proj.trace();
    stats.redund_info_ratio = (naive_trace_total > 0.0) ? (applied_trace / naive_trace_total) : 1.0;
    return stats;
  }

  throw std::runtime_error("lio/residual_redundancy/mode: unrecognized value '" + opts.mode + "'");
}

void applyPriorScalarControls(Eigen::MatrixXd& prior_cov, const PriorScalarOptions& opts)
{
  if (!opts.on()) return;  // identity at every default -- zero risk to the "off" path

  if (opts.p_inflate_alpha != 1.0) prior_cov *= opts.p_inflate_alpha;

  if (opts.p_floor_min_eig > 0.0) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(prior_cov);
    Eigen::VectorXd eigvals = es.eigenvalues();
    for (int i = 0; i < eigvals.size(); ++i)
      eigvals(i) = std::max(eigvals(i), opts.p_floor_min_eig);
    prior_cov = es.eigenvectors() * eigvals.asDiagonal() * es.eigenvectors().transpose();
  }

  if (opts.p_fading_lambda != 1.0) prior_cov /= opts.p_fading_lambda;
}

}  // namespace livo_recon
