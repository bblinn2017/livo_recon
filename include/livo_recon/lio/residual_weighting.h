#pragma once

#include <string>
#include <vector>

#include "livo_recon/utils/map/voxelmap_utils.h"

// ============================================================================
// CQ-37: axes A and B of the four-axis weighting-scheme rebuild (see
// lio_processing.h's historical comment on LioProcOptions for the full
// record of why the five original schemes were removed in 2026-08, and
// CQ-37's own card in the coding-agent inbox for the four-axis decomposition
// this rebuild is organized around).
//
//   axis A  residual-set reduction   BEFORE accumulation, changes WHICH
//           lio/residual_weighting/collapse  { off, plane_averaged }
//           residuals exist at all.
//   axis B  per-residual reweight    BEFORE accumulation, changes
//           lio/residual_weighting/per_residual  { off, count_weighted,
//                                    count_weighted_renorm, info_gain }
//           sigma_squared (and plane_var_term, item 1c) PER RESIDUAL.
//   axis C  group marginalisation    AFTER accumulation -- UNCHANGED, see
//           lio/residual_redundancy/mode                residual_redundancy.h.
//   axis D  one global scalar        BEFORE accumulation -- see
//           lio/ekf/sigma_scale_mode                     lio_processing.h/cpp
//                                    (this file does not implement axis D;
//                                    it already lived beside
//                                    density_sigma_mode and CQ-37 extends
//                                    that call site directly).
//
// THIS IS A REBUILD FROM PROSE, NOT A RESTORATION (rule 51a). `git log --all
// -S` on every one of these schemes' names returns comment-only hits at
// every commit in this repository's history -- the original 2026-08
// implementations are not recoverable and never existed in this repo. A
// rebuilt scheme's numbers stand on their own; they cannot confirm or refute
// the historical 2026-08 figures quoted in lio_processing.h's comment, only
// be compared against them as a new, independent measurement.
//
// mode == "off" for both axes (the default): neither function below is ever
// called from a live path with anything to do -- see the call sites in
// lio_processing.cpp's buildResiduals()-adjacent code -- so "off" reproduces
// pre-CQ-37 behavior to the ULP, matching the same discipline
// residual_redundancy.h documents for axis C.
// ============================================================================

namespace livo_recon
{

struct ResidualWeightingOptions
{
  // "off" | "plane_averaged".
  std::string collapse = "off";
  // "off" | "count_weighted" | "count_weighted_renorm" | "info_gain".
  std::string per_residual = "off";

  bool collapseOn() const { return collapse != "off"; }
  bool perResidualOn() const { return per_residual != "off"; }
};

// Axis A engagement/magnitude, per frame.
struct CollapseStats
{
  int groups_collapsed  = 0;  // plane groups (size >= 2) collapsed to one residual
  int residuals_removed = 0;  // n_raw - n_collapsed over those groups (0 if collapse == off)
};

// Axis B engagement/magnitude, per frame.
struct PerResidualStats
{
  int    residuals_touched = 0;    // residuals whose sigma_squared this axis actually scaled
  double renorm_factor     = 1.0;  // count_weighted_renorm's own global correction (1.0 = not renorm, or off)
  double mean_applied_scale = 1.0; // mean of (sigma_squared_after / sigma_squared_before) over touched residuals
};

// Axis A ("collapse == plane_averaged"): groups `residuals` by plane_id;
// every group with >= 2 members is replaced by ONE residual at the group's
// centroid -- r, point_cross_normal and normal are the group's WEIGHTED mean
// (weight = 1/sigma_squared_i, so a tighter-fit member counts more, matching
// how accumulateLioResiduals() would have weighted them individually), and
// the collapsed residual's own sigma_squared is the group's JOINT variance:
// the harmonic mean of the naive per-residual variances divided by group
// size k (i.e. 1/sigma_collapsed^2 = sum_i(1/sigma_i^2)/k^2), which is the
// variance of the weighted mean under the (admittedly wrong, but explicit)
// assumption that all k residuals shared exactly one true value -- the same
// simplifying assumption "plane_averaged" made under its original name, see
// lio_processing.h's historical comment for its recorded failure mode
// (rotation starvation, now directly measurable via h_rr_min_eig, already
// logged in frame_stats.txt).
//
// plane_var_term is set to the group's OWN plane_var_term (identical across
// the group by construction -- see Residual::plane_var_term's own comment)
// rather than averaged; floor_term/sigma_diag_squared/s_prior_pose are left
// at -1.0 (not-set) on the collapsed residual since their per-residual
// S-decomposition no longer has a well-defined single source after
// collapsing -- frame_stats.txt's *_share columns already tolerate this via
// their own < 0.0 skip (see LioFrameDiag's own comment).
//
// DIVERGENCE CONDITION (CQ-84, measured by CQ-80/sections/round-80): the
// 1/sigma_collapsed^2 = sum_i(1/sigma_i^2)/k^2 formula above matches its
// own derivation exactly (confirmed on real data), but it is NOT bounded
// away from a collapse as the averaged-plane leverage k (group size) grows
// -- variance shrinks like 1/k^2 while it should only shrink like 1/k for
// a group that is genuinely k independent measurements of one value, so a
// large group's collapsed sigma_squared understates its true uncertainty
// once k is large enough. MEASURED exposure on eee_01: 37 of 3,981 frames
// (< 1%) have at least one group with k >= 4, max observed k = 7.15
// (fractional from the weighted-mean construction). Rare but real; the
// formula itself is unchanged, only this threshold/exposure is newly
// documented.
//
// Groups of size 1 (or residuals with plane_id == nullptr) pass through
// unmodified. Mutates `residuals` in place (erase-and-append pattern);
// caller must not hold iterators/pointers into it across this call.
CollapseStats applyResidualCollapse(std::vector<Residual>& residuals);

// Axis B: per-residual reweight, mode-dispatched.
//
// "count_weighted": sigma_squared_i *= k_i (k_i = this residual's plane
// group size, 1 for an ungrouped residual -- a no-op for those). Per CQ-37
// item 1c, plane_var_term_i is scaled by the SAME factor k_i, since
// buildResiduals() has already folded it into sigma_squared by this point
// and the two must move together or the sigma_squared = ... + plane_var_term
// decomposition frame_stats.txt reports becomes internally inconsistent.
//
// "count_weighted_renorm": as above, then ONE global multiplicative
// correction (stats.renorm_factor, computed in closed form -- see .cpp) is
// applied to every residual's weight so that sum_i(1/sigma_squared_i) over
// the WHOLE residual set is unchanged by this axis (trace(H_pp) preserved
// in the special case where every jacobian's position column has unit norm,
// which normal columns do by construction -- see residual_redundancy.cpp's
// own jacobianOf()). The per-residual GROUP-RELATIVE discount is preserved;
// only the aggregate budget is restored.
//
// "info_gain": a FRESH per-frame Sherman-Morrison sequential-information
// discount -- direction-aware (unlike the two group-size-based schemes
// above), and NOT the retired scheme's own implementation, which does not
// exist in this repository's history (rule 51a). A running COVARIANCE
// Sigma (6x6, tangent-space rot|pos, matching jacobianOf()'s layout) is
// seeded UNINFORMATIVELY at the start of EVERY frame call --
// Sigma_0 = kInfoGainSeedCov * I_6 -- which is this rebuild's fix for the
// retired scheme's recorded bug (seeding from the filter's own accumulated
// state_->cov() instead, causing a cross-frame feedback spiral -- see
// lio_processing.h's historical comment). Residuals are then processed IN
// THE ORDER `residuals` already holds them (buildResiduals()'s own scan
// order; not re-sorted), each one:
//   j_i = jacobianOf(r_i);  w_i = 1/sigma_squared_i
//   leverage_i = w_i * j_i^T * Sigma * j_i          (>= 0)
//   discount_i = 1 / (1 + leverage_i)                in (0, 1]
//   sigma_squared_i /= discount_i   (and plane_var_term_i, item 1c)
//   Sigma <- Sigma - w_i*(Sigma*j_i)*(Sigma*j_i)^T / (1 + leverage_i)
//            (the standard rank-1 Sherman-Morrison covariance downdate for
//            admitting one weight-w_i observation along direction j_i)
// A residual whose direction is novel relative to everything already seen
// this frame (leverage_i near 0, e.g. the first residual, or one pointing a
// direction nothing else has) gets discount_i near 1 (no discount); a
// residual duplicating a direction the running Sigma has already been
// informed about heavily (leverage_i large) gets discount_i near 0 (heavily
// downweighted) -- the intended "many residuals sharing one plane's normal
// get discounted against each other" behavior, but per-residual and
// continuous rather than the group-size heuristic's discrete k.
// discount_i is floored at kInfoGainMinDiscount purely for numerical safety
// (sigma_squared_i /= discount_i must not divide by something that
// underflows to 0), not as a modeling choice -- see the .cpp for the value.
PerResidualStats applyPerResidualReweight(std::vector<Residual>& residuals,
                                          const std::string& per_residual_mode);

}  // namespace livo_recon
