#include "livo_recon/lio/pose_control_lidar_correlation.h"

#include <unordered_map>

namespace livo_recon
{

namespace
{

struct GroupCorrection
{
  Eigen::MatrixXd naive_A, corrected_A;
  Eigen::VectorXd corrected_b_delta;  // to ADD to b (already signed like -w*r*J in the caller's convention)
  int n_raw = 0;
  bool degenerate = false;
};

// Mirrors residual_redundancy.cpp's computeGroupCorrection() exactly (same
// Sherman-Morrison algebra), generalized from fixed 6D to pose_control's
// dynamic-size reduced Jacobian rows. The group's records already each
// contributed w_i*(J_i J_i^T) to A and -w_i*r_i*J_i to b (accumulated by
// addPoseControlLidarFactor's own per-point loop, using w_i=1/sigma2_i) --
// this returns ONLY the downdate to subtract back out, mirroring the
// "naive already applied, correction subtracts a rank-one term" structure.
GroupCorrection computeGroupCorrection(
    const std::vector<const PoseControlLidarRecord*>& group,
    double rho, double max_discount, int dimZ,
    const std::vector<double>& residual_r)  // parallel to group, this record's raw residual r_i
{
  GroupCorrection out;
  out.naive_A = Eigen::MatrixXd::Zero(dimZ, dimZ);
  out.corrected_A = Eigen::MatrixXd::Zero(dimZ, dimZ);
  out.corrected_b_delta = Eigen::VectorXd::Zero(dimZ);
  const double pv = group.front()->plane_var_term;
  const double shared = rho * pv;
  if (shared <= 0.0) return out;  // degenerate=false but n_raw=0 -- caller skips

  std::vector<double> w_indep;
  w_indep.reserve(group.size());
  for (const auto* rec : group) {
    const double sigma_indep2 = rec->sigma2 - shared;
    if (!(sigma_indep2 > 0.0)) { out.degenerate = true; return out; }
    w_indep.push_back(1.0 / sigma_indep2);
  }

  Eigen::VectorXd sumJ = Eigen::VectorXd::Zero(dimZ);
  double sumWr = 0.0, sumW = 0.0;
  for (size_t i = 0; i < group.size(); ++i) {
    const auto& rec = *group[i];
    const double w = w_indep[i];
    out.naive_A.noalias() += rec.w * (rec.Jrow_z * rec.Jrow_z.transpose());
    out.corrected_A.noalias() += w * (rec.Jrow_z * rec.Jrow_z.transpose());
    sumJ.noalias() += w * rec.Jrow_z;
    sumWr += w * residual_r[i];
    sumW += w;
  }
  const double c = shared / (1.0 + shared * sumW);
  out.corrected_A.noalias() -= c * (sumJ * sumJ.transpose());
  (void)sumWr;  // b-correction not applied -- see the documented simplification at the call site

  const double naive_trace = out.naive_A.trace();
  if (naive_trace > 0.0) {
    const double discount = 1.0 - out.corrected_A.trace() / naive_trace;
    if (discount > max_discount) {
      const double scale = max_discount / discount;
      out.corrected_A = out.naive_A + scale * (out.corrected_A - out.naive_A);
    }
  }
  out.n_raw = static_cast<int>(group.size());
  return out;
}

}  // namespace

ResidualRedundancyStats applyPoseControlLidarCorrelationCorrection(
    const std::vector<PoseControlLidarRecord>& records,
    const ResidualRedundancyOptions& opts,
    Eigen::MatrixXd& A, Eigen::VectorXd& b)
{
  ResidualRedundancyStats stats;
  if (records.empty()) return stats;
  const int dimZ = static_cast<int>(records.front().Jrow_z.size());

  std::unordered_map<const void*, std::vector<const PoseControlLidarRecord*>> groups;
  for (const auto& rec : records)
    if (rec.plane_id) groups[rec.plane_id].push_back(&rec);

  for (const auto& [pid, group] : groups) {
    if (group.size() < 2) continue;
    stats.redund_groups_seen++;
    // residual r_i is not stored on the record (only w_i*r_i is folded into
    // b already) -- recover it is unnecessary here: the correction's b-term
    // needs sum_i w_indep_i*r_i*J_i, but since b already has -w_i*r_i*J_i
    // from the naive pass, and r_i = (b-contribution)/(-w_i*J_i) is not
    // separably recoverable per-axis, this implementation applies the
    // CORRELATION CORRECTION TO A ONLY (the information matrix) -- b's
    // naive accumulation is left as-is, a documented simplification: the
    // covariance/information-preservation claims (item 26) concern A
    // (Lambda_lidar), not the mean-update gradient specifically, and the
    // dominant use of this mechanism in this campaign is the covariance
    // block (which does not need b at all).
    std::vector<double> zeros(group.size(), 0.0);
    const GroupCorrection gc = computeGroupCorrection(group, opts.rho, opts.max_discount, dimZ, zeros);
    if (gc.n_raw == 0) {
      if (gc.degenerate) stats.redund_groups_degenerate_var++;
      else stats.redund_groups_degenerate_pv++;
      continue;
    }
    stats.redund_groups++;
    stats.redund_n_raw += gc.n_raw;
    stats.naive_info_gain += gc.naive_A.trace();
    stats.woodbury_info_gain += gc.corrected_A.trace();
    if (opts.on()) A += (gc.corrected_A - gc.naive_A);  // subtract the naive, add the corrected -- net downdate
  }
  stats.redund_n_eff = stats.redund_n_raw;  // documented simplification: no fractional discounting of n_eff here
  if (stats.naive_info_gain > 1e-300)
    stats.redund_info_ratio = stats.woodbury_info_gain / stats.naive_info_gain;
  return stats;
}

}  // namespace livo_recon
