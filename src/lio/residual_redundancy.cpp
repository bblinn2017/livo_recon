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
// added for these indices) and exact Woodbury-corrected contributions.
// Uses the exact three-parameter covariance carried by VoxelPlane. Returns
// n_raw==0 if its metadata/covariance is invalid, leaving the group unchanged.
GroupCorrection computeGroupCorrection(const std::vector<const Residual*>& group,
                                        double rho)
{
  GroupCorrection out;
  if (rho <= 0.0 || std::none_of(group.begin(), group.end(),
      [](const Residual* r) { return r->plane_var_term > 0.0; })) {
    out.degenerate = DegenerateReason::kPlaneVar;  // n_raw stays 0 -- caller skips
    return out;
  }

  const M3D plane_cov = 0.5 * (group.front()->plane_covariance +
                               group.front()->plane_covariance.transpose());
  Eigen::SelfAdjointEigenSolver<M3D> plane_es(plane_cov);
  if (plane_es.info() != Eigen::Success ||
      plane_es.eigenvalues().minCoeff() <
          -1e-10 * std::max(1.0, plane_es.eigenvalues().maxCoeff())) {
    out.degenerate = DegenerateReason::kResidualVar;
    return out;
  }
  const M3D plane_sqrt = plane_es.eigenvectors() *
      plane_es.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal();
  Eigen::MatrixXd U(group.size(), 3);

  std::vector<double> w_indep;
  w_indep.reserve(group.size());
  for (size_t i = 0; i < group.size(); ++i) {
    const Residual* r = group[i];
    if ((r->plane_covariance - group.front()->plane_covariance).norm() >
        1e-9 * std::max(1.0, plane_cov.norm())) {
      out.degenerate = DegenerateReason::kResidualVar;
      return out;
    }
    U.row(i) = std::sqrt(rho) *
        r->plane_jacobian.transpose() * plane_sqrt;
    const double shared = rho * r->plane_var_term;
    if (std::abs(U.row(i).squaredNorm() - shared) >
        1e-8 * std::max(1.0, std::abs(shared))) {
      out.degenerate = DegenerateReason::kResidualVar;
      return out;
    }
    const double sigma_indep2 = r->sigma_squared - shared;
    if (!(sigma_indep2 > 0.0)) {
      out.degenerate = DegenerateReason::kResidualVar;  // shared term consumes the whole variance
      return out;
    }
    w_indep.push_back(1.0 / sigma_indep2);
  }

  Eigen::Matrix<double,6,3> weighted_j_u = Eigen::Matrix<double,6,3>::Zero();
  V3D weighted_u_r = V3D::Zero();
  M3D core = M3D::Identity();
  for (size_t i = 0; i < group.size(); ++i) {
    const Residual& r = *group[i];
    const V6 j = jacobianOf(r);
    const double w = w_indep[i];
    out.naive_HtH.noalias() += (1.0 / r.sigma_squared) * (j * j.transpose());
    out.naive_Htz.noalias() += (1.0 / r.sigma_squared) * r.r * j;
    out.corrected_HtH.noalias() += w * (j * j.transpose());
    out.corrected_Htz.noalias() += w * r.r * j;
    const Eigen::RowVector3d ui = U.row(i);
    weighted_j_u.noalias() += w * j * ui;
    weighted_u_r.noalias() += w * ui.transpose() * r.r;
    core.noalias() += w * ui.transpose() * ui;
  }

  const Eigen::LDLT<M3D> core_ldlt(core);
  if (core_ldlt.info() != Eigen::Success || !core_ldlt.isPositive()) {
    out.degenerate = DegenerateReason::kResidualVar;
    return out;
  }
  out.corrected_HtH.noalias() -= weighted_j_u *
      core_ldlt.solve(weighted_j_u.transpose());
  out.corrected_Htz.noalias() -= weighted_j_u * core_ldlt.solve(weighted_u_r);

  out.n_raw = static_cast<int>(group.size());
  return out;
}

}  // namespace

ResidualRedundancyStats applyResidualRedundancyCorrection(
    const std::vector<Residual>& residuals,
    const ResidualRedundancyOptions& opts,
    EkfUpdate& ekf)
{
  if (opts.mode != "off" && opts.mode != "woodbury")
    throw std::runtime_error(
        "lio/residual_redundancy/mode supports only off or woodbury");
  if (opts.rho < 0.0 || opts.rho > 1.0)
    throw std::runtime_error(
        "lio/residual_redundancy/rho must be in [0,1]");

  ResidualRedundancyStats stats{};

  // Preserve first-observed group order so the correction's floating-point
  // summation is independent of pointer-hash/allocator ordering.
  std::unordered_map<const void*, size_t> plane_index;
  std::vector<std::vector<const Residual*>> by_plane;
  for (const Residual& r : residuals) {
    if (r.plane_id == nullptr) continue;
    auto inserted = plane_index.emplace(r.plane_id, by_plane.size());
    if (inserted.second) by_plane.emplace_back();
    by_plane[inserted.first->second].push_back(&r);
  }

  M66 delta_HtH_total = M66::Zero();
  V6  delta_Htz_total = V6::Zero();
  double naive_trace_total = 0.0, corrected_trace_total = 0.0;

  // CQ-31 item 7: this loop, and everything through redund_info_ratio below,
  // runs UNCONDITIONALLY -- including opts.mode == "off" -- so naive_
  // info_gain/woodbury_info_gain are always populated. Only the ekf.HtH/Htz
  // mutation further down is mode-gated.
  for (const auto& group : by_plane) {
    if (group.size() < 2) continue;
    ++stats.redund_groups_seen;

    const GroupCorrection gc = computeGroupCorrection(group, opts.rho);
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
    if (corrected_trace > naive_trace * (1.0 + 1e-12))
      ++stats.information_increase_groups;

    delta_HtH_total.noalias() += (gc.corrected_HtH - gc.naive_HtH);
    delta_Htz_total.noalias() += (gc.corrected_Htz - gc.naive_Htz);
  }

  // naive_info_gain/woodbury_info_gain are traces over the FULL 6x6, matching
  // h_pp_min_eig's own convention of reading off the position block from the
  // full matrix rather than a separately-tracked sub-accumulator.
  stats.naive_info_gain = naive_trace_total;
  stats.woodbury_info_gain = corrected_trace_total;
  stats.gamma_correction_trace = delta_HtH_total.trace();
  stats.gamma_correction_frobenius = delta_HtH_total.norm();
  stats.b_correction_norm = delta_Htz_total.norm();
  const Eigen::SelfAdjointEigenSolver<M66> correction_es(
      0.5 * (delta_HtH_total + delta_HtH_total.transpose()));
  if (correction_es.info() == Eigen::Success) {
    stats.gamma_correction_min_eigenvalue =
        correction_es.eigenvalues().minCoeff();
    stats.gamma_correction_max_eigenvalue =
        correction_es.eigenvalues().maxCoeff();
  }

  if (stats.redund_groups == 0) return stats;  // nothing to apply; info_ratio/gains stay at defaults

  stats.redund_info_ratio = (naive_trace_total > 0.0) ? (corrected_trace_total / naive_trace_total) : 1.0;

  if (!opts.on()) return stats;  // mode == "off": stats computed above, ekf untouched

  if (opts.mode == "woodbury") {
    ekf.HtH += delta_HtH_total;
    ekf.Htz += delta_Htz_total;
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
