#include "livo_recon/processing/combined_processing.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/log/consistency_log.h"
#include "livo_recon/processing/lio_processing.h"
#include "livo_recon/processing/vio_processing.h"

#include <iomanip>
#include <limits>

namespace livo_recon
{

CombinedProc::CombinedProc(NodeContext& ctx)
  : state_(ctx.state), profiler_(ctx.profiler)
{}

std::string CombinedProc::loadParameters(ros::NodeHandle& pnh)
{
  paramWarn<bool>(pnh, "combined/enable",             opts_.enable,          false);
  paramWarn<int>(pnh, "combined/max_iterations",      opts_.max_iterations,  5);
  paramWarn<double>(pnh, "combined/ekf/min_norm_dtheta", opts_.min_norm_dtheta, 0.0);
  paramWarn<double>(pnh, "combined/ekf/min_norm_dt",     opts_.min_norm_dt,     0.0);
  paramWarn<double>(pnh, "combined/ekf/min_diff_error",  opts_.min_diff_error,  -1.0);
  paramWarn<double>(pnh, "combined/ekf/max_avg_error",   opts_.max_avg_error,   5.0);
  paramWarn<bool>(pnh, "combined/log_debug_en",       opts_.log_debug_en,   false);
  paramWarn<bool>(pnh, "combined/log_consistency_en", opts_.log_consistency_en, false);
  // See CombinedProcOptions::log_nll_en_compat -- lio/log_nll_en is read
  // here too so a sweep config using the LIO-only flag name still gets a
  // populated nll.txt in combined mode.
  paramWarn<bool>(pnh, "lio/log_nll_en",              opts_.log_nll_en_compat, false);

  std::ostringstream oss;
  oss << "[params/combined]"
      << "\n  enable:             " << (opts_.enable ? "true" : "false")
      << "\n  max_iterations:     " << opts_.max_iterations
      << "\n  ekf/min_norm_dtheta:" << opts_.min_norm_dtheta
      << "\n  ekf/min_norm_dt:    " << opts_.min_norm_dt
      << "\n  ekf/min_diff_error: " << opts_.min_diff_error
      << "\n  ekf/max_avg_error:  " << opts_.max_avg_error
      << "\n  log_consistency_en: " << (opts_.log_consistency_en ? "true" : "false")
      << "\n  log_nll_en_compat: " << (opts_.log_nll_en_compat ? "true" : "false");
  return oss.str();
}

std::string CombinedProc::processCombined(MeasureGroup& mg, LioProc& lio_proc, VioProc& vio_proc)
{
  TimedScope ts(profiler_, "combined/ekf");

  // Fallback snapshot -- see the rollback comment near the bottom. Taken
  // before ANY correction (LIO or VIO) is applied this frame, mirroring
  // VioProc::estimateStateCorrection()'s own state_before_frame pattern.
  const StateGroup state_before_frame = *state_;

  prior_cov_ = state_->cov();
  state_propagat_ = *state_;

  bool vio_ever_contributed = false;
  float last_avg_err_vio = std::numeric_limits<float>::quiet_NaN();
  std::string stop = "max_iter";
  int iter = 0;
  bool any_solved = false;

  // History (58-71): see docs/livo_recon_changelog.md#src-processing-combined_processing.cpp-58
  double best_avg_res_lio = std::numeric_limits<double>::max();
  float  best_avg_err_vio = std::numeric_limits<float>::max();
  // History (74-78): see docs/livo_recon_changelog.md#src-processing-combined_processing.cpp-74
  int last_n_meas_lio = 0;
  int last_n_meas_vio = 0;

  for (; iter < opts_.max_iterations; ++iter)
  {
    EkfUpdate lio_part, vio_part;
    double avg_res_lio = 0.0;
    float avg_err_vio = 0.0f;

    const bool have_lio = lio_proc.accumulateForCombined(mg, lio_part, avg_res_lio);
    const bool have_vio = vio_proc.accumulateForCombined(mg, vio_part, avg_err_vio);

    if (!have_lio && !have_vio) { stop = "no_residuals"; break; }

    if ((have_lio && avg_res_lio >= best_avg_res_lio) ||
        (have_vio && avg_err_vio >= best_avg_err_vio))
    {
      stop = "no_improve";
      break;
    }
    if (have_lio) { best_avg_res_lio = avg_res_lio; last_n_meas_lio = lio_part.n_meas; }
    if (have_vio) { best_avg_err_vio = avg_err_vio; last_n_meas_vio = vio_part.n_meas; }

    if (have_vio) { vio_ever_contributed = true; last_avg_err_vio = avg_err_vio; }

    ekf_.reset();
    if (have_lio) { ekf_.HtH += lio_part.HtH; ekf_.Htz += lio_part.Htz; }
    if (have_vio) { ekf_.HtH += vio_part.HtH; ekf_.Htz += vio_part.Htz; }

    if (opts_.log_debug_en)
    {
      const double tr_lio_rr = have_lio ? lio_part.HtH.block<3, 3>(0, 0).trace() : 0.0;
      const double tr_lio_pp = have_lio ? lio_part.HtH.block<3, 3>(3, 3).trace() : 0.0;
      const double tr_vio_rr = have_vio ? vio_part.HtH.block<3, 3>(0, 0).trace() : 0.0;
      const double tr_vio_pp = have_vio ? vio_part.HtH.block<3, 3>(3, 3).trace() : 0.0;
      ROS_INFO_STREAM(std::scientific << std::setprecision(4)
          << "[combined/ekf/dbg] t=" << mg.image.t << " iter=" << iter
          << "  have_lio=" << have_lio << " have_vio=" << have_vio
          << "  tr(HtH_lio_RR)=" << tr_lio_rr << "  tr(HtH_lio_PP)=" << tr_lio_pp
          << "  tr(HtH_vio_RR)=" << tr_vio_rr << "  tr(HtH_vio_PP)=" << tr_vio_pp
          << "  |Htz_lio|=" << lio_part.Htz.norm() << "  |Htz_vio|=" << vio_part.Htz.norm()
          << "  avg_res_lio=" << avg_res_lio << "  avg_err_vio=" << avg_err_vio);
    }

    ekf_.applyMeanUpdate(state_, prior_cov_, state_propagat_);
    any_solved = true;

    if (ekf_.dtheta.norm() < opts_.min_norm_dtheta && ekf_.dt.norm() < opts_.min_norm_dt)
    {
      stop = "norm_converged";
      ++iter;
      break;
    }
  }

  if (any_solved)
    ekf_.applyCovarianceUpdate(state_, prior_cov_);

  if (opts_.log_consistency_en || opts_.log_nll_en_compat) {
    // Both channels, same statistic, same frame. This is the measurement the
    // unification argument needs: if the two channels' NIS levels differ
    // materially at the same instant, they are calibrated inconsistently
    // relative to each other, which is a concrete argument for one filter.
    // If they agree, REA-LVIO's decoupling case is the better one and the
    // goal should be restated rather than pursued.
    logConsistencyNll("lio", mg.image.t, best_avg_res_lio, last_n_meas_lio,
                      ekf_.pivotRatio(), ekf_.kalmanGainNorm());
    logConsistencyNll("vio", mg.image.t, last_avg_err_vio, last_n_meas_vio,
                      ekf_.pivotRatio(), ekf_.kalmanGainNorm());
  }

  // History (138-152): see docs/livo_recon_changelog.md#src-processing-combined_processing.cpp-138
  float final_avg_err_vio = last_avg_err_vio;
  if (vio_ever_contributed)
  {
    EkfUpdate final_vio_part;
    float re_avg_err_vio = 0.0f;
    if (vio_proc.accumulateForCombined(mg, final_vio_part, re_avg_err_vio))
      final_avg_err_vio = re_avg_err_vio;
  }

  bool rejected = false;
  if (vio_ever_contributed && final_avg_err_vio > static_cast<float>(opts_.max_avg_error))
  {
    // Can't decompose the joint solve after the fact to keep just LIO's
    // share (HtH_lio/HtH_vio were already summed before the one solve) --
    // see combined_processing.h's doc comment. Discard the whole result
    // and re-run LIO alone, unmodified, as the fallback.
    *state_ = state_before_frame;
    if (opts_.log_consistency_en)
      logConsistencyNll("rollback", mg.image.t, final_avg_err_vio, 0, 0.0, 0.0);
    lio_proc.processLIO(mg);
    rejected = true;
  }

  // Anchor-pose history recording now happens centrally in
  // LivoReconNode::estimateState() (see Tracker::recordFramePose()) --
  // no per-mode commit call needed here anymore.

  std::ostringstream oss;
  oss << "[combined/ekf] iters=" << iter << "  stop=" << stop
      << "  vio_contributed=" << (vio_ever_contributed ? 1 : 0);
  if (rejected)
    oss << "  REJECTED(vio_avg_error=" << final_avg_err_vio << " > " << opts_.max_avg_error << ", ran LIO-only fallback)";
  return oss.str();
}

}  // namespace livo_recon
