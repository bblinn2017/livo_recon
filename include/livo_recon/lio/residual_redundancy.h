#pragma once

#include <string>
#include <vector>

#include "livo_recon/utils/algo/ekf.h"
#include "livo_recon/utils/map/voxelmap_utils.h"

// ============================================================================
// Residual redundancy correction -- CQ-28, re-landing (config-gated, inert by
// default) the mechanism LioProcOptions' historical comment records as
// "woodbury_plane_correction": five prior schemes to correct for correlated/
// redundant residuals (many points matched to the same plane in one frame)
// were tried in 2026-08 and removed, all regressing or diverging. This one
// was "mathematically the most principled of all five" but still regressed
// (reduced_chi2 ~0.45-0.52, i.e. over-conservative) -- see
// refs/redundancy-prior-art in the register index for the full record.
//
// THE PROBLEM.  Residual::sigma_squared is floor_term + sigma_diag_squared
// (per-point-specific) + plane_var_term (the SAME value for every residual
// matched to one plane, since it comes from that plane's own fit
// uncertainty). accumulateLioResiduals() weights every residual by
// 1/sigma_squared independently, which is only correct if residuals sharing
// a plane are conditionally independent given the plane's true parameters --
// they are not: they share the SAME plane_var_term draw, so they are
// correlated, and summing them as independent double-counts that shared
// uncertainty once per residual, understating the true joint variance (i.e.
// overstating the information admitted) roughly in proportion to the
// group's size.
//
// THE CORRECTION (mode == "woodbury").  Model one plane's k matched
// residuals' joint noise as Sigma = D + rho*plane_var_term*ones(k,k), where
// D = diag(sigma_indep_1^2, ..., sigma_indep_k^2) is the per-residual
// INDEPENDENT-only variance (sigma_squared_i - rho*plane_var_term) and rho
// in [0,1] is how correlated the shared component is assumed to be (rho=1:
// the textbook fully-correlated case; rho=0: no correction, degenerates to
// today's diagonal treatment). By the Sherman-Morrison identity,
//   Sigma^-1 = D^-1 - c * (D^-1 1)(D^-1 1)^T,   c = rho*pv / (1 + rho*pv*sum(1/D_ii))
// so the group's correct joint information contribution to HtH/Htz is
//   J^T Sigma^-1 J = sum_i w_i J_i J_i^T - c * (sum_i w_i J_i)(sum_i w_i J_i)^T
// with w_i = 1/D_ii, instead of the naive sum_i (1/sigma_squared_i) J_i J_i^T
// accumulateLioResiduals() already computed for this group's residuals.
// Singleton groups (k==1) are UNCHANGED by this correction -- the rank-1
// term's Sherman-Morrison algebra collapses exactly back to
// 1/(sigma_indep^2+plane_var_term) when there is nothing to marginalize
// against -- so this module only ever touches groups with >= 2 residuals,
// leaving every frame's non-redundant majority untouched.
//
// mode == "woodbury_divpos" applies the same per-group correction, then
// rescales its TOTAL effect on ekf.HtH by one frame-level scalar GAMMA in
// [0,1] chosen (by a coarse grid search, not a continuous solve -- see the
// .cpp) so the position block's minimum eigenvalue (h_pp_min_eig) stays as
// close as possible to its pre-correction value, rather than letting the
// correction erode it freely. This targets a DIFFERENT historical failure
// mode ("count_weighted": H_pp_min_eig collapsed 4-4.6x under a naive
// group-size discount) than "woodbury"'s own regression (over-conservative
// reduced_chi2) -- it is an UNTESTED HYPOTHESIS per refs/redundancy-prior-
// art, not a proven fix; CQ-28 item 4's 1-D sweep is how it gets evaluated,
// not this module's own default behavior.
//
// mode == "off" (default): this entire module is never called -- see the
// call sites in lio_processing.cpp -- so there is zero risk of a
// floating-point reordering making "off" differ from pre-CQ-28 behavior by
// even one ULP.
// ============================================================================

namespace livo_recon
{

struct ResidualRedundancyOptions
{
  std::string mode = "off";  // "off" | "woodbury" | "woodbury_divpos"

  // Intra-plane correlation assumed for the shared plane_var_term component
  // (0 = no correction even when mode != "off"; 1 = the textbook fully-
  // correlated Woodbury identity, as derived above). CQ-28 item 4's sweep
  // knob #1 -- report the curve, do not pick a value here.
  double rho = 1.0;

  // Per-group cap: the correction may not remove more than this fraction of
  // a group's naive information-trace, a stability guard against a
  // degenerate group (very large plane_var_term) making the group's
  // contribution to HtH near-singular. CQ-28 item 4's sweep knob #2.
  double max_discount = 0.9;

  bool on() const { return mode != "off"; }
};

// Per-frame engagement/magnitude counters -- CQ-28 item 3. A fresh instance
// is returned by every applyResidualRedundancyCorrection() call (the caller
// only calls it when opts.on(), so there is no stale-state concern).
struct ResidualRedundancyStats
{
  int redund_groups = 0;           // plane groups with >= 2 matched residuals this frame
  int redund_n_raw = 0;            // total residuals belonging to those groups
  int redund_n_eff = 0;            // ceil(sum over groups of n_raw_g * corrected_trace_g/naive_trace_g)
  double redund_info_ratio = 1.0;  // admitted/naive information over grouped residuals only; 1.0 when redund_groups==0
};

// Groups `residuals` by plane_id; for every group with >= 2 members, computes
// the Woodbury-marginalized joint contribution (see file header) and adds
// the DELTA against what accumulateLioResiduals[Cuda]() already summed into
// `ekf.HtH`/`ekf.Htz` for that group's residuals (so this must be called
// AFTER accumulation, on the SAME residuals vector, and the caller must not
// call it at all when opts.mode == "off").
ResidualRedundancyStats applyResidualRedundancyCorrection(
    const std::vector<Residual>& residuals,
    const ResidualRedundancyOptions& opts,
    EkfUpdate& ekf);

}  // namespace livo_recon
