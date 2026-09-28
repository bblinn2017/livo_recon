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
// THE PROBLEM. Residual::sigma_squared is floor_term + sigma_diag_squared
// (per-point-specific) + plane_var_term. Points matched to one plane share
// the same three-parameter plane error, but have different J_nq projections
// and therefore different plane_var_term values. accumulateLioResiduals()
// weights every residual by
// 1/sigma_squared independently, which is only correct if residuals sharing
// a plane are conditionally independent given the plane's true parameters --
// they are not: they share the SAME plane-state draw, so they are
// correlated, and summing them as independent double-counts that shared
// uncertainty once per residual, understating the true joint variance (i.e.
// overstating the information admitted) roughly in proportion to the
// group's size.
//
// THE CORRECTION (mode == "woodbury"). Model one plane's k matched
// residuals as C = D + rho*J_plane*P_plane*J_plane^T. Factoring
// P_plane=L*L^T gives U=sqrt(rho)*J_plane*L and
//   C^-1=D^-1-D^-1 U (I+U^T D^-1 U)^-1 U^T D^-1.
// This rank-at-most-three update retains signed cross-covariances. Replacing
// it with sqrt(var_i*var_j) would incorrectly force perfect positive scalar
// correlation and is not the covariance represented by VoxelPlane.
// Singleton groups (k==1) are UNCHANGED by this correction -- the Woodbury
// algebra collapses exactly back to
// 1/(sigma_indep^2+plane_var_term) when there is nothing to marginalize
// against -- so this module only ever touches groups with >= 2 residuals,
// leaving every frame's non-redundant majority untouched.
//
// mode == "off" (default): the module still computes diagnostic comparison
// statistics, but it never mutates HtH/Htz. The independent accumulator runs
// first in its original residual order, so the estimator update is unchanged.
// ============================================================================

namespace livo_recon
{

struct ResidualRedundancyOptions
{
  // "off" | "woodbury". Deprecated conditioning workarounds were removed;
  // the live correlated mode is the direct covariance model above.
  std::string mode = "off";

  // Fraction of the shared three-parameter plane covariance treated as
  // correlated within a matched-plane group
  // (0 = no correction even when mode != "off"; 1 = the textbook fully-
  // correlated Woodbury identity, as derived above). CQ-28 item 4's sweep
  // knob #1 -- report the curve, do not pick a value here.
  double rho = 1.0;

  bool on() const { return mode != "off"; }
};

// CQ-31 item 5: three independently-switchable scalar P (prior covariance)
// controls, all default-off/identity, deliberately kept separate from
// ResidualRedundancyOptions above -- these act on the PRIOR directly, not on
// the measurement update, and are the CONTROL for the whole redundancy
// programme (see the card): if a blunt scalar inflation of P recovers most
// of what the selective Woodbury machinery buys, the machinery is paying for
// conditioning, not information geometry.
struct PriorScalarOptions
{
  // P_inv <- alpha * P_inv, i.e. P <- P / alpha (alpha >= 1 shrinks the
  // prior's confidence uniformly, matching kappa_eff's own +1 direction).
  // alpha == 1.0 is the identity (default, off).
  double p_inflate_alpha = 1.0;

  // Floor applied to P^-1's eigenvalues after the alpha scale (0 = no
  // floor). A floor on P^-1's eigenvalues is a CEILING on P's, i.e. this
  // bounds how uncertain the prior may ever claim to be.
  double p_floor_min_eig = 0.0;

  // P_inv <- P_inv / lambda, lambda < 1 grows the prior's covariance
  // ("fading"); lambda == 1.0 is the identity (default, off).
  double p_fading_lambda = 1.0;

  bool on() const
  {
    return p_inflate_alpha != 1.0 || p_floor_min_eig != 0.0 || p_fading_lambda != 1.0;
  }
};

// Per-frame engagement/magnitude counters -- CQ-28 item 3. A fresh instance
// is returned by every applyResidualRedundancyCorrection() call (the caller
// only calls it when opts.on(), so there is no stale-state concern).
struct ResidualRedundancyStats
{
  int redund_groups = 0;           // plane groups with >= 2 matched residuals this frame
  int redund_n_raw = 0;            // total residuals belonging to those groups
  double redund_info_ratio = 1.0;  // admitted/naive information over grouped residuals only; 1.0 when redund_groups==0

  // CQ-34 item 4: redund_groups==0 conflates "no group had >= 2 matched
  // residuals" with "a group did, but the correction declined it as
  // degenerate". redund_groups_seen counts group.size()>=2 BEFORE either
  // degeneracy test; the other two partition its shortfall against
  // redund_groups: seen == redund_groups + degenerate_pv + degenerate_var.
  int redund_groups_seen = 0;            // group.size() >= 2, before any degeneracy test
  int redund_groups_degenerate_pv = 0;   // rho*plane_var_term <= 0
  int redund_groups_degenerate_var = 0;  // some residual's sigma_squared - shared <= 0

  // CQ-31 item 7: LOG BESIDE, DO NOT DRIVE -- computed every frame regardless
  // of mode (including "off"), so claims/c-28's "no direct instrument" gap
  // finally has one. naive_info_gain is trace(sum over grouped residuals of
  // their already-accumulated 1/sigma_squared * J J^T) -- what
  // accumulateLioResiduals() actually admitted for these residuals;
  // woodbury_info_gain is the same trace after the correction this module
  // computes (whether or not opts.mode == "off" applies it to ekf.HtH). The
  // ratio of the two IS redund_info_ratio when mode != "off"; when mode ==
  // "off" these still let a filing quote "the correction WOULD have removed
  // X%" without switching it on.
  double naive_info_gain = 0.0;
  double woodbury_info_gain = 0.0;

  // Counterfactual exact correction (corrected minus independent) over all
  // admitted groups. Populated even in mode==off so Phase-3 experiments can
  // distinguish a scalar trace change from redistribution across state
  // directions without enabling the correction.
  double gamma_correction_trace = 0.0;
  double gamma_correction_frobenius = 0.0;
  double gamma_correction_min_eigenvalue = 0.0;
  double gamma_correction_max_eigenvalue = 0.0;
  double b_correction_norm = 0.0;

  // Exact signed correlations can add information in residual-difference
  // directions even when shared/common directions lose information. Count
  // such groups explicitly instead of describing a trace ratio as an
  // "effective residual count", which is not mathematically well defined.
  int information_increase_groups = 0;
};

// Groups `residuals` by plane_id; for every group with >= 2 members, computes
// the Woodbury-marginalized joint contribution (see file header). This must
// be called AFTER accumulateLioResiduals[Cuda](), on the SAME residuals
// vector, because it needs what that pass already summed into `ekf.HtH`/
// `ekf.Htz` as the "naive" baseline to correct against.
//
// CQ-31 item 7: SAFE, AND INTENDED, TO CALL EVERY FRAME INCLUDING mode ==
// "off" -- the diagnostic stats (naive_info_gain, woodbury_info_gain,
// redund_groups/n_raw/info_ratio) are always computed from `opts.rho`
// regardless of `opts.mode`, so a filing can quote "the
// correction would have removed X%" without switching it on. Only the
// MUTATION of `ekf.HtH`/`ekf.Htz` is gated on `opts.mode != "off"` --
// passing mode=="off" computes and returns stats but leaves ekf untouched,
// so "off" still reproduces pre-CQ-28 behavior to the ULP.
//
ResidualRedundancyStats applyResidualRedundancyCorrection(
    const std::vector<Residual>& residuals,
    const ResidualRedundancyOptions& opts,
    EkfUpdate& ekf);

// CQ-31 item 5's three scalar P controls, applied directly to the prior
// covariance P (NOT its inverse) so the implementation is one line each and
// unambiguous to verify by inspection:
//   p_inflate_alpha  (>= 1, default 1): P <- alpha * P.       Larger alpha
//                    -> a more uncertain (less confident) prior.
//   p_floor_min_eig  (>= 0, default 0, 0 == no floor): after the alpha
//                    scale, floor P's own eigenvalues at this value from
//                    BELOW -- a stability guard against the prior ever
//                    claiming to be more confident than min_eig allows,
//                    the same "prevent overconfidence" spirit as this
//                    codebase's existing sensor_noise_floor_eig0 and
//                    weight_floor mechanisms, applied to the prior instead
//                    of the measurement.
//   p_fading_lambda  (< 1 to have effect, default 1): P <- P / lambda,
//                    the textbook fading-memory-filter growth (lambda < 1
//                    inflates P further, on top of the alpha scale).
// NOTE ON THE CARD'S OWN NOTATION: item 5 writes these against "P-" (the
// glyph rendered ambiguously between "P" and "P inverse" in the source
// card). This implementation interprets all three as acting on P directly,
// matching the controls' own names ("p_INFLATE" inflates P; "p_FLOOR" is a
// floor, and a floor on P itself -- not its inverse -- is the direction
// that prevents overconfidence, consistent with every other floor already
// in this codebase). If the card meant P's inverse, the three formulas
// invert cleanly (alpha and lambda swap their scaling direction; the
// eigenvalue floor becomes a ceiling) -- flagged here explicitly so a
// reviewer can catch a mismatched intent before this is swept.
void applyPriorScalarControls(Eigen::MatrixXd& prior_cov, const PriorScalarOptions& opts);

}  // namespace livo_recon
