#pragma once

#include "livo_recon/node_context.h"
#include "livo_recon/utils/algo/ekf.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/log/profiler.h"

namespace livo_recon
{

class LioProcBase;
class VioProc;

struct CombinedProcOptions
{
  // History (16-26): see docs/livo_recon_changelog.md#include-livo_recon-processing-combined_processing.h-16
  bool   enable          = false;

  // 2026-10-27: these five fields are loaded from the SAME vio/ekf/* keys
  // VioProcOptions uses (see loadParameters()) rather than their own
  // combined/ekf/* copies -- no dataset config has ever set them
  // differently, so the two separately-loaded copies were pure
  // duplication risk, not a real independent knob. Still separate STRUCT
  // fields (combined's own joint loop reads opts_.* here, not VioProc's),
  // just sourced from one shared config key per concept.
  int    max_iterations  = 5;
  double min_norm_dtheta = 0.0;
  double min_norm_dt     = 0.0;
  double min_diff_error  = 1e-4;

  // Diagnostic-only -- see VioProcOptions::force_all_iterations's doc
  // comment, same rationale: when true, a non-improving VIO iteration is
  // no longer excluded from that iteration's joint update (see
  // processCombined()'s per-iteration VIO-only rejection, 2026-10-23).
  // LIO's own residual was never gated on this and still isn't.
  bool force_all_iterations = false;

  // Whole-frame VIO-fit rejection threshold, mirroring VioProcOptions::
  // max_avg_error -- including its 2026-10-23 unit change to
  // avg_abs_pixel_residual (mean |pixel error|, not the weighted/squared/
  // normalized-coordinate avg_err_vio the per-iteration check still uses).
  // If VIO contributed to this frame's joint solve and its final pixel
  // residual exceeds this, the ENTIRE joint result (both LIO's and VIO's
  // contribution) is discarded and LioProc::processLIO() is re-run alone
  // as the fallback (see processCombined()'s doc comment for why -- can't
  // decompose a joint solve after the fact to keep just LIO's share).
  double max_avg_error   = 5.0;

  // TEMPORARY (2026-10-27, LIO-wrong-direction investigation -- delete
  // alongside the rest of this investigation's plumbing): diagnostic DOF
  // split, testing the hypothesis that LIO corrects position well but
  // rotation poorly while VIO is the reverse. When true, right before
  // summing lio_part/vio_part into the joint normal equations, LIO's
  // rotation block (HtH rows/cols 0-2, Htz rows 0-2) is zeroed -- LIO
  // contributes ONLY a position correction -- and VIO's position block
  // (HtH rows/cols 3-5, Htz rows 3-5) is zeroed -- VIO contributes ONLY a
  // rotation correction. The resulting joint HtH is block-diagonal (no
  // rotation/position cross-coupling at all between the two sources).
  // Default false is behavior-preserving.
  bool   diag_split_dof_lio_pos_vio_rot = false;

  bool   log_debug_en    = false;

  // History (45-50): see docs/livo_recon_changelog.md#include-livo_recon-processing-combined_processing.h-45
  bool   log_consistency_en = false;

  // SW-5 bug-ledger fix: CombinedProc's primary loop never calls
  // LioProc::processLIO() (only accumulateForCombined()), so LioProc's own
  // "log/lio/nll"-gated nll.txt write inside processLIO() never fires
  // in combined mode -- a sweep config that sets log/lio/nll (the
  // LIO-only convention every existing sweep script uses) produces a
  // near-empty nll.txt with no warning. CombinedProc has its own
  // logConsistencyNll() call, gated on log_consistency_en above, that
  // already writes a per-channel ("lio"/"vio") row every frame from
  // accumulateForCombined()'s own residual -- it was simply never wired to
  // the flag any existing config actually sets. Read log/lio/nll here
  // too and OR it into the same gate (loadParameters()) so an existing
  // sweep config needs no change.
  bool   log_nll_en_compat = false;

  // log/combined/vio_contribution (default false): per-frame
  // avg_err_vio/n_meas_vio/vio_ever_contributed, written to
  // combined_vio_contribution.csv -- see diagnostics/vio_correction.h's
  // doc comment for why this (not a pose-delta) is the closest available
  // per-frame "how much did VIO contribute" signal in combined mode.
  bool   log_vio_contribution_en = false;
};

// Opt-in combined LIO+VIO EKF step. Each iteration: asks LioProc and
// VioProc (via their accumulateForCombined() entry points, which reuse the
// exact same lio_accumulator.h/vio_accumulator.h math their own sequential
// processLIO()/processVIO() paths use) for this frame's HtH/Htz
// contribution, sums whichever side(s) produced one, and performs ONE
// applyMeanUpdate/applyCovarianceUpdate on the total -- rather than LIO
// converging first and VIO correcting on top of an already-committed LIO
// posterior.
class CombinedProc
{
public:
  explicit CombinedProc(NodeContext& ctx);

  std::string loadParameters(ros::NodeHandle& pnh);

  bool enabled() const { return opts_.enable; }

  // lio_proc/vio_proc are LivoReconNode's own existing LioProc(Base)/VioProc
  // instances (passed in, not owned here) -- this class only orchestrates
  // their accumulate-only entry points, it doesn't duplicate LIO/VIO's own
  // per-frame bookkeeping (voxel map, tracker, etc.).
  // TEMPORARY (2026-10-27, vio_residual_diag investigation): out_avg_error/
  // out_avg_abs_pixel_residual/out_n_valid, when non-null, receive this
  // frame's final_avg_err_vio/final_pixel_residual/last_n_meas_vio (the
  // function's own already-correct true-post-refinement values -- see
  // processCombined()'s body) -- delete alongside the rest of this
  // investigation's plumbing. Default args keep existing call sites
  // byte-identical.
  std::string processCombined(MeasureGroup& mg, LioProcBase& lio_proc, VioProc& vio_proc,
                              float* out_avg_error = nullptr, float* out_avg_abs_pixel_residual = nullptr,
                              int* out_n_valid = nullptr);

private:
  CombinedProcOptions opts_;
  EkfUpdate ekf_;
  Eigen::MatrixXd prior_cov_;
  StateGroup state_propagat_;
  StateGroupPtr state_;
  ProfilerPtr profiler_;
  DataQueuesPtr data_queues_;
  int vio_contribution_frame_index_ = 0;
  // TEMPORARY (2026-10-27, correction_breakdown_diag investigation) -- delete alongside it.
  int correction_breakdown_frame_index_ = 0;
};

}  // namespace livo_recon
