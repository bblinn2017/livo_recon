#pragma once

#include "livo_recon/node_context.h"
#include "livo_recon/utils/algo/ekf.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/log/profiler.h"

namespace livo_recon
{

class LioProc;
class VioProc;

struct CombinedProcOptions
{
  // History (16-26): see docs/livo_recon_changelog.md#include-livo_recon-processing-combined_processing.h-16
  bool   enable          = false;

  int    max_iterations  = 5;
  double min_norm_dtheta = 0.0;
  double min_norm_dt     = 0.0;
  double min_diff_error  = -1.0;

  // Whole-frame VIO-fit rejection threshold, mirroring VioProcOptions::
  // max_avg_error. If VIO contributed to this frame's joint solve and its
  // own achieved avg_error exceeds this, the ENTIRE joint result (both
  // LIO's and VIO's contribution) is discarded and LioProc::processLIO()
  // is re-run alone as the fallback (see processCombined()'s doc comment
  // for why -- can't decompose a joint solve after the fact to keep just
  // LIO's share).
  double max_avg_error   = 5.0;

  bool   log_debug_en    = false;

  // History (45-50): see docs/livo_recon_changelog.md#include-livo_recon-processing-combined_processing.h-45
  bool   log_consistency_en = false;

  // SW-5 bug-ledger fix: CombinedProc's primary loop never calls
  // LioProc::processLIO() (only accumulateForCombined()), so LioProc's own
  // "lio/log_nll_en"-gated nll.txt write inside processLIO() never fires
  // in combined mode -- a sweep config that sets lio/log_nll_en (the
  // LIO-only convention every existing sweep script uses) produces a
  // near-empty nll.txt with no warning. CombinedProc has its own
  // logConsistencyNll() call, gated on log_consistency_en above, that
  // already writes a per-channel ("lio"/"vio") row every frame from
  // accumulateForCombined()'s own residual -- it was simply never wired to
  // the flag any existing config actually sets. Read lio/log_nll_en here
  // too and OR it into the same gate (loadParameters()) so an existing
  // sweep config needs no change.
  bool   log_nll_en_compat = false;
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

  // lio_proc/vio_proc are LivoReconNode's own existing LioProc/VioProc
  // instances (passed in, not owned here) -- this class only orchestrates
  // their accumulate-only entry points, it doesn't duplicate LIO/VIO's own
  // per-frame bookkeeping (voxel map, tracker, etc.).
  std::string processCombined(MeasureGroup& mg, LioProc& lio_proc, VioProc& vio_proc);

private:
  CombinedProcOptions opts_;
  EkfUpdate ekf_;
  Eigen::MatrixXd prior_cov_;
  StateGroup state_propagat_;
  StateGroupPtr state_;
  ProfilerPtr profiler_;
};

}  // namespace livo_recon
