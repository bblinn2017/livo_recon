#include "livo_recon/processing/combined_processing.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/diagnostics/log/consistency_log.h"
#include "livo_recon/diagnostics/vio_correction.h"
#include "livo_recon/diagnostics/correction_breakdown_diag.h"  // TEMPORARY, see its own header doc comment
#include "livo_recon/processing/lio_base.h"
#include "livo_recon/processing/vio_processing.h"

#include <iomanip>
#include <limits>

namespace livo_recon
{

CombinedProc::CombinedProc(NodeContext& ctx)
  : state_(ctx.state), profiler_(ctx.profiler), data_queues_(ctx.data_queues)
{}

std::string CombinedProc::loadParameters(ros::NodeHandle& pnh)
{
  // 2026-10-27: these six knobs used to be loaded a second time under
  // combined/*, independently of VioProcOptions's vio/ekf/* copies, even
  // though every dataset config has always given them the same value --
  // collapsed onto the single vio/ekf/* namespace so there is one key per
  // concept instead of two that could silently drift apart.
  paramWarn<bool>(pnh, "combined/enable",             opts_.enable,          false);
  paramWarn<int>(pnh, "vio/ekf/max_iterations",       opts_.max_iterations,  5);
  paramWarn<double>(pnh, "vio/ekf/min_norm_dtheta",   opts_.min_norm_dtheta, 0.0);
  paramWarn<double>(pnh, "vio/ekf/min_norm_dt",       opts_.min_norm_dt,     0.0);
  paramWarn<double>(pnh, "vio/ekf/min_diff_error",    opts_.min_diff_error,  1e-4);
  paramWarn<bool>(pnh, "vio/ekf/force_all_iterations", opts_.force_all_iterations, false);
  paramWarn<double>(pnh, "vio/ekf/max_avg_error",     opts_.max_avg_error,   5.0);
  paramWarn<bool>(pnh, "diag/combined/split_dof_lio_pos_vio_rot",
                   opts_.diag_split_dof_lio_pos_vio_rot, false);
  paramWarn<bool>(pnh, "log/combined/debug",       opts_.log_debug_en,   false);
  paramWarn<bool>(pnh, "log/combined/consistency", opts_.log_consistency_en, false);
  // See CombinedProcOptions::log_nll_en_compat -- log/lio/nll is read
  // here too so a sweep config using the LIO-only flag name still gets a
  // populated nll.txt in combined mode.
  paramWarn<bool>(pnh, "log/lio/nll",              opts_.log_nll_en_compat, false);
  paramWarn<bool>(pnh, "log/combined/vio_contribution", opts_.log_vio_contribution_en, false);

  std::ostringstream oss;
  oss << "[params/combined]"
      << "\n  enable:             " << (opts_.enable ? "true" : "false")
      << "\n  max_iterations:     " << opts_.max_iterations
      << "\n  ekf/min_norm_dtheta:" << opts_.min_norm_dtheta
      << "\n  ekf/min_norm_dt:    " << opts_.min_norm_dt
      << "\n  ekf/min_diff_error: " << opts_.min_diff_error
      << "\n  ekf/force_all_iterations: " << (opts_.force_all_iterations ? "true" : "false")
      << "\n  ekf/max_avg_error:  " << opts_.max_avg_error
      << "\n  log_consistency_en: " << (opts_.log_consistency_en ? "true" : "false")
      << "\n  log_vio_contribution_en: " << (opts_.log_vio_contribution_en ? "true" : "false")
      << "\n  log_nll_en_compat: " << (opts_.log_nll_en_compat ? "true" : "false");
  return oss.str();
}

std::string CombinedProc::processCombined(MeasureGroup& mg, LioProcBase& lio_proc, VioProc& vio_proc,
                                          float* out_avg_error, float* out_avg_abs_pixel_residual,
                                          int* out_n_valid)
{
  TimedScope ts(profiler_, "combined/ekf");

  // 2026-10-23: "post_imu" state_trace.csv row -- see LioProcBase::logStateTrace()'s own doc
  // comment for why this mode needs its own calls (processLIO() isn't called in the normal,
  // non-rejected path below, so the row LioProcIekf::processLIO() would otherwise log never
  // happens). The state here is exactly the post-IMU-propagation state processLIO() itself logs
  // from, since nothing between IMU propagation and this call touches the state.
  lio_proc.logStateTrace("post_imu", mg, 0, 0);

  // Fallback snapshot -- see the rollback comment near the bottom. Taken
  // before ANY correction (LIO or VIO) is applied this frame, mirroring
  // VioProc::estimateStateCorrection()'s own state_before_frame pattern.
  const StateGroup state_before_frame = *state_;

  prior_cov_ = state_->cov();
  state_propagat_ = *state_;

  bool vio_ever_contributed = false;
  float last_avg_err_vio = std::numeric_limits<float>::quiet_NaN();
  // Fallback for final_pixel_residual below if the post-loop sentinel
  // re-check (iter=99) ever fails -- mirrors last_avg_err_vio's own role
  // as final_avg_err_vio's fallback.
  float last_pixel_residual_vio = std::numeric_limits<float>::quiet_NaN();
  std::string stop = "max_iter";
  int iter = 0;
  bool any_solved = false;

  // 2026-10-23: LIO's own per-iteration "stop if worse" check is removed --
  // LIO never rejects on its own residual now (mirrors LIO-only's own
  // unconditional-iteration behavior, lio_iekf.cpp's R97A Q2 removal).
  // last_avg_res_lio is overwritten every iteration LIO actually
  // contributes (no longer gated on improvement -- there's no "best" to
  // track once nothing stops on it), purely for the post-loop consistency
  // log.
  double last_avg_res_lio = std::numeric_limits<double>::quiet_NaN();
  // History (74-78): see docs/livo_recon_changelog.md#src-processing-combined_processing.cpp-74
  int last_n_meas_lio = 0;
  int last_n_meas_vio = 0;
  // 2026-10-23: VIO is gated ONCE, at iteration 0, on an ABSOLUTE criterion (avg_abs_pixel_
  // residual_vio vs opts_.max_avg_error -- the SAME threshold the end-of-frame rollback check
  // below uses) instead of every iteration having to improve on a running best-so-far. History:
  // the old per-iteration "avg_err_vio >= best_avg_err_vio -> exclude VIO from THIS iteration"
  // check let VIO flip in and out of the joint solve mid-frame -- confirmed via a direct
  // investigation (capture_10_20_no3's stationary window) that excluding VIO for even one
  // iteration removes its restraint on LIO's own step, letting LIO's position correction grow
  // (observed one frame going from ~0.5mm/iteration with VIO included to 1.35mm the moment VIO
  // was excluded), which then pushes the state somewhere VIO's OWN reprojection disagrees with
  // even more (observed avg_err_vio jumping 0.06->3.14, pixel residual 1.1px->8.3px, consistent
  // with ~1.35mm of unconstrained translation at ~0.4m feature depth) -- a feedback loop that
  // both degrades the mean (unconstrained LIO-only steps are real, applied corrections) and
  // corrupts the reported covariance (applyCovarianceUpdate() only snapshots the LAST iteration's
  // information, so a frame that oscillated LIO-only/joint/LIO-only ends with a covariance that
  // reflects whichever one the last iteration happened to land on, not what actually moved the
  // mean). Fixing this (force_all_iterations=true, i.e. never excluding VIO once it has
  // residuals) cut mean position drift over the full stationary window by over half (1.71mm ->
  // 0.77mm) and made it the best of all three modes -- but force_all_iterations ALSO removes the
  // iteration-0 admission check entirely, which is NOT safe during real motion (motion blur/bad
  // tracking could produce a genuinely bad initial VIO fit that should be rejected outright, not
  // forced in) -- the stationary window can't exercise that failure mode at all, so a clean win
  // there doesn't prove it's safe in general. This is the middle ground: gate once, on an
  // absolute scale, then commit -- VIO admitted at iteration 0 is never re-litigated mid-frame,
  // but a genuinely bad iteration-0 fit is still rejected for the whole frame, same as before.
  bool vio_admitted = false;
  // TEMPORARY (2026-10-27, vio_residual_diag): the RAW iter-0 VIO residual,
  // captured regardless of whether it ends up admitted -- unlike
  // last_avg_err_vio/last_pixel_residual_vio (only ever set when
  // vio_ever_contributed), this always holds a value whenever VIO had ANY
  // candidate residuals at all, so a diagnostic caller can see what VIO
  // WOULD have seen even on a frame where it was excluded for the rest of
  // the loop. Delete alongside the rest of this investigation's plumbing.
  bool iter0_have_vio = false;
  float iter0_avg_err_vio = 0.0f, iter0_avg_abs_pixel_residual_vio = 0.0f;
  int iter0_n_meas_vio = 0;

  for (; iter < opts_.max_iterations; ++iter)
  {
    EkfUpdate lio_part, vio_part;
    double avg_res_lio = 0.0;
    float avg_err_vio = 0.0f;
    float avg_abs_pixel_residual_vio = 0.0f;

    const bool have_lio = lio_proc.accumulateForCombined(mg, lio_part, avg_res_lio, iter);
    bool have_vio = vio_proc.accumulateForCombined(mg, vio_part, avg_err_vio,
                                                   avg_abs_pixel_residual_vio, iter);

    if (iter == 0)
    {
      iter0_have_vio = have_vio;
      iter0_avg_err_vio = avg_err_vio;
      iter0_avg_abs_pixel_residual_vio = avg_abs_pixel_residual_vio;
      iter0_n_meas_vio = vio_part.n_meas;

      // TEMPORARY (2026-10-27, correction_breakdown_diag investigation):
      // (1) what LIO's and VIO's measurement evidence ALONE ask for, at
      // frame entry, unweighted by the prior (-HtH^-1*Htz -- see vio_
      // processing.cpp's matching comment for why this differs from
      // applyMeanUpdate()'s own measurement_term); (2) a single-iteration
      // counterfactual isolating VIO's marginal contribution to the joint
      // solve: re-run applyMeanUpdate() on a SCRATCH state copy (never
      // touches the real state_) once with LIO's info alone, once with
      // LIO+VIO summed, both from the SAME frame-entry snapshot -- the
      // difference is what including VIO changes about the committed
      // result, which can't be read off the real (summed-before-solving)
      // joint result directly. Delete alongside the rest of this
      // investigation's plumbing.
      if (out_avg_error || out_avg_abs_pixel_residual || out_n_valid)
      {
        V3D ask_lio_dtheta = V3D::Zero(), ask_lio_dt = V3D::Zero();
        if (have_lio)
        {
          const Eigen::Matrix<double, 6, 1> ask = -lio_part.HtH.ldlt().solve(lio_part.Htz);
          ask_lio_dtheta = ask.segment<3>(0);
          ask_lio_dt = ask.segment<3>(3);
        }
        V3D ask_vio_dtheta = V3D::Zero(), ask_vio_dt = V3D::Zero();
        if (have_vio)
        {
          const Eigen::Matrix<double, 6, 1> ask = -vio_part.HtH.ldlt().solve(vio_part.Htz);
          ask_vio_dtheta = ask.segment<3>(0);
          ask_vio_dt = ask.segment<3>(3);
        }

        auto scratch_lio_only = std::make_shared<StateGroup>(*state_);
        EkfUpdate cf_lio_only;
        if (have_lio) { cf_lio_only.HtH = lio_part.HtH; cf_lio_only.Htz = lio_part.Htz; }
        cf_lio_only.applyMeanUpdate(scratch_lio_only, prior_cov_, state_propagat_);

        auto scratch_joint = std::make_shared<StateGroup>(*state_);
        EkfUpdate cf_joint;
        if (have_lio) { cf_joint.HtH += lio_part.HtH; cf_joint.Htz += lio_part.Htz; }
        if (have_vio) { cf_joint.HtH += vio_part.HtH; cf_joint.Htz += vio_part.Htz; }
        cf_joint.applyMeanUpdate(scratch_joint, prior_cov_, state_propagat_);

        const double tr_rot_lio = have_lio ? lio_part.HtH.block<3, 3>(0, 0).trace() : 0.0;
        const double tr_pos_lio = have_lio ? lio_part.HtH.block<3, 3>(3, 3).trace() : 0.0;
        const double tr_rot_vio = have_vio ? vio_part.HtH.block<3, 3>(0, 0).trace() : 0.0;
        const double tr_pos_vio = have_vio ? vio_part.HtH.block<3, 3>(3, 3).trace() : 0.0;

        writeCorrectionBreakdownRow(lio_proc.frameIdx(), mg.image.t + data_queues_->start_time, "combined",
                                    ask_vio_dtheta, ask_vio_dt, have_vio ? vio_part.n_meas : 0,
                                    /*have_lio_ask=*/true, ask_lio_dtheta, ask_lio_dt, have_lio ? lio_part.n_meas : 0,
                                    /*have_counterfactual=*/true,
                                    cf_lio_only.dtheta, cf_lio_only.dt, cf_joint.dtheta, cf_joint.dt,
                                    tr_rot_lio, tr_pos_lio, tr_rot_vio, tr_pos_vio);
      }
    }

    if (!have_lio && !have_vio) { stop = "no_residuals"; break; }

    // See vio_admitted's own doc comment above. iter==0 decides ONCE whether VIO is usable at
    // all this frame, on an absolute scale; once admitted it is never excluded again mid-frame
    // merely for a worse-than-best-so-far iteration (force_all_iterations still bypasses the
    // iter-0 gate too, same as before, for offline analysis).
    if (iter == 0)
      vio_admitted = opts_.force_all_iterations ||
          (have_vio && avg_abs_pixel_residual_vio <= static_cast<float>(opts_.max_avg_error));
    if (!vio_admitted)
      have_vio = false;
    if (have_vio)
      last_n_meas_vio = vio_part.n_meas;
    if (have_lio) { last_avg_res_lio = avg_res_lio; last_n_meas_lio = lio_part.n_meas; }

    if (have_vio)
    {
      vio_ever_contributed = true;
      last_avg_err_vio = avg_err_vio;
      last_pixel_residual_vio = avg_abs_pixel_residual_vio;
    }

    if (!have_lio && !have_vio)
      continue;  // VIO rejected this iteration and LIO had nothing -- nothing to apply, try again next iteration

    // Diagnostic DOF split (see CombinedProcOptions::diag_split_dof_lio_pos_vio_rot's
    // doc comment) -- applied only to what actually gets summed into the real
    // joint solve below, AFTER the ask_lio/ask_vio/counterfactual diagnostics
    // above are computed from the unsplit lio_part/vio_part.
    if (opts_.diag_split_dof_lio_pos_vio_rot)
    {
      if (have_lio)
      {
        lio_part.HtH.block<3, 3>(0, 0).setZero();
        lio_part.HtH.block<3, 3>(0, 3).setZero();
        lio_part.HtH.block<3, 3>(3, 0).setZero();
        lio_part.Htz.segment<3>(0).setZero();
      }
      if (have_vio)
      {
        vio_part.HtH.block<3, 3>(3, 3).setZero();
        vio_part.HtH.block<3, 3>(3, 0).setZero();
        vio_part.HtH.block<3, 3>(0, 3).setZero();
        vio_part.Htz.segment<3>(3).setZero();
      }
    }

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
          << "  avg_res_lio=" << avg_res_lio << "  avg_err_vio=" << avg_err_vio
          << "  avg_abs_pixel_residual_vio=" << avg_abs_pixel_residual_vio);
    }

    ekf_.applyMeanUpdate(state_, prior_cov_, state_propagat_);
    any_solved = true;

    if (opts_.log_debug_en)
    {
      // 2026-10-23 (investigation): the STEP this iteration's applyMeanUpdate() just applied --
      // logged separately from the pre-update accumulate line above so it can be correlated with
      // the NEXT iteration's avg_err_vio (testing whether a rotation jump this iteration explains
      // a VIO reprojection-error spike next iteration).
      ROS_INFO_STREAM(std::scientific << std::setprecision(4)
          << "[combined/ekf/step] t=" << mg.image.t << " iter=" << iter
          << "  dtheta_deg=" << ekf_.dtheta.norm() * 180.0 / M_PI
          << "  dt_mm=" << ekf_.dt.norm() * 1000.0);
    }

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
    logConsistencyNll("lio", mg.image.t, last_avg_res_lio, last_n_meas_lio,
                      ekf_.pivotRatio(), ekf_.kalmanGainNorm());
    logConsistencyNll("vio", mg.image.t, last_avg_err_vio, last_n_meas_vio,
                      ekf_.pivotRatio(), ekf_.kalmanGainNorm());
  }

  // History (138-152): see docs/livo_recon_changelog.md#src-processing-combined_processing.cpp-138
  float final_avg_err_vio = last_avg_err_vio;
  // 2026-10-23: final_pixel_residual (not final_avg_err_vio) is what the
  // whole-frame max_avg_error gate below now compares against -- see
  // VioAccumulator::accumulate()'s doc comment for why this is a separate,
  // genuinely pixel-space quantity from avg_err_vio's weighted/squared/
  // normalized one.
  float final_pixel_residual = last_pixel_residual_vio;
  if (vio_ever_contributed)
  {
    EkfUpdate final_vio_part;
    float re_avg_err_vio = 0.0f;
    float re_pixel_residual = 0.0f;
    // Sentinel iter=99 (well outside 0..max_iterations-1, default max is 5):
    // this re-accumulate happens AFTER the loop above has already applied
    // its last correction, so it measures the truly-final post-update
    // residual -- distinct from every in-loop iter value, so a per-point
    // log consumer (log/vio/point_residuals) can tell "final" apart from
    // "last loop iteration" (whose own row is pre-that-iteration's update).
    if (vio_proc.accumulateForCombined(mg, final_vio_part, re_avg_err_vio, re_pixel_residual, 99))
    {
      final_avg_err_vio = re_avg_err_vio;
      final_pixel_residual = re_pixel_residual;
    }
  }

  bool rejected = false;
  if (vio_ever_contributed && final_pixel_residual > static_cast<float>(opts_.max_avg_error))
  {
    // Can't decompose the joint solve after the fact to keep just LIO's
    // share (HtH_lio/HtH_vio were already summed before the one solve) --
    // see combined_processing.h's doc comment. Discard the whole result
    // and re-run LIO alone, unmodified, as the fallback.
    *state_ = state_before_frame;
    if (opts_.log_consistency_en)
      logConsistencyNll("rollback", mg.image.t, final_avg_err_vio, 0, 0.0, 0.0);
    // processLIO() logs its own "post_imu"/"post_lio" state_trace.csv rows internally -- this
    // frame's "post_imu" row duplicates the one already logged at the top of this function (same
    // state either way, since nothing between the two calls changes it other than this rollback
    // restoring state_before_frame, which IS that same post-IMU state), harmless but redundant;
    // its "post_lio" row is the one this frame actually needs, so no separate call below for it.
    lio_proc.processLIO(mg);
    rejected = true;
  }
  else
  {
    // 2026-10-23: "post_lio" state_trace.csv row for the normal (non-rejected) path -- see
    // LioProcBase::logStateTrace()'s own doc comment. last_n_meas_lio + last_n_meas_vio is the
    // combined residual count from whichever iteration last contributed each; iter is the loop's
    // own completed-iterations count, mirroring processLIO()'s own call shape.
    lio_proc.logStateTrace("post_lio", mg, last_n_meas_lio + last_n_meas_vio, iter);
  }

  // Anchor-pose history recording now happens centrally in
  // LivoReconNode::estimateState() (see Tracker::recordFramePose()) --
  // no per-mode commit call needed here anymore.

  if (opts_.log_vio_contribution_en)
    writeCombinedVioContributionRow(++vio_contribution_frame_index_, mg.image.t + data_queues_->start_time,
                                    final_avg_err_vio, last_n_meas_vio, vio_ever_contributed);

  // TEMPORARY (2026-10-27, vio_residual_diag) -- see this function's header
  // doc comment. When VIO never got admitted this frame (vio_ever_
  // contributed=false), final_avg_err_vio/final_pixel_residual/
  // last_n_meas_vio are all still at their initial zero/unset values -- fall
  // back to the raw iter-0 attempt instead, so a diagnostic caller sees
  // "what VIO would have seen" rather than a misleading zero, matching
  // VioProc::processVIO()'s own fallback-to-raw-iter-0 behavior for
  // sequential/lio_only (see its matching comment).
  if (vio_ever_contributed)
  {
    if (out_avg_error) *out_avg_error = final_avg_err_vio;
    if (out_avg_abs_pixel_residual) *out_avg_abs_pixel_residual = final_pixel_residual;
    if (out_n_valid) *out_n_valid = last_n_meas_vio;
  }
  else
  {
    if (out_avg_error) *out_avg_error = iter0_have_vio ? iter0_avg_err_vio : 0.0f;
    if (out_avg_abs_pixel_residual) *out_avg_abs_pixel_residual = iter0_have_vio ? iter0_avg_abs_pixel_residual_vio : 0.0f;
    if (out_n_valid) *out_n_valid = iter0_have_vio ? iter0_n_meas_vio : 0;
  }

  std::ostringstream oss;
  oss << "[combined/ekf] iters=" << iter << "  stop=" << stop
      << "  vio_contributed=" << (vio_ever_contributed ? 1 : 0);
  if (rejected)
    oss << "  REJECTED(vio_avg_abs_pixel_residual=" << final_pixel_residual << " > " << opts_.max_avg_error << ", ran LIO-only fallback)";
  return oss.str();
}

}  // namespace livo_recon
