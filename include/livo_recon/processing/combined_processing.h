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
  // Master switch -- when false (default), LivoReconNode::estimateState()
  // takes the original, unchanged sequential LIO-then-VIO path and never
  // touches CombinedProc at all. Opt-in per the user's explicit request
  // (2026-08-14): fuses HtH_lio+HtH_vio into ONE joint EKF update per
  // iteration instead of running LIO to convergence and only then starting
  // VIO's own separate iterated update from that posterior -- intended to
  // better balance translational/rotational correction from both
  // modalities at once. See lio/lio_accumulator.h and vio/vio_accumulator.h
  // for the shared accumulation code both this class and the ORIGINAL
  // sequential LioProc/VioProc now call -- this class adds no new residual
  // math of its own, it only sums the two existing accumulators' output.
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

  // G6 (2026-09-01). Logs both channels' NLL (nll.txt, channel="lio"/"vio")
  // for the SAME frame via the shared consistency_log.h writers -- see that
  // header's doc comment for why this was previously unaskable in combined
  // mode. Also logs a "rollback" row when the whole-frame VIO rejection
  // fires, so a rejected frame is recorded as one that happened rather than
  // silently vanishing from the log.
  bool   log_consistency_en = false;
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
