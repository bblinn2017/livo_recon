#include "livo_recon/processing/vio_processing.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/algo/omp_utils.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"
#include "livo_recon/diagnostics/correction_breakdown_diag.h"  // TEMPORARY, see its own header doc comment
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <omp.h>
#include <opencv2/imgproc.hpp>

namespace livo_recon
{

namespace
{

// Shared opt-in debug-log append helper for log_iterations/log_point_
// residuals/log_residual_counts/log_raw_tracker_output (all write to the
// SAME opts_.log_path -- see that option's doc comment). Truncated on
// first call PER PATH (not globally -- two differently-configured VioProc
// instances, unlikely in this codebase but not impossible, wouldn't stomp
// each other's first-call truncation), appended thereafter. Replaces the
// old debugLogVio()/kDebugVioLoggingEnabled mechanism (task #149 left it
// permanently compile-time-disabled) with a properly runtime-toggleable
// one.
// CQ-35: one persistent ofstream per distinct path, opened (truncating) the
// first time that path is seen and kept open for the process lifetime --
// replaces the old truncated_paths-set + fresh-ofstream-per-call pattern,
// which paid an open+close syscall pair on every single call.
void logIteration(const std::string& path, const std::string& msg)
{
  static std::map<std::string, std::ofstream> streams;
  auto it = streams.find(path);
  if (it == streams.end())
    it = streams.emplace(path, std::ofstream(path, std::ios::trunc)).first;
  it->second << msg << "\n";
  it->second.flush();
}

}  // namespace

// ─────────────────────────────────────────────
//  Constructor
// ─────────────────────────────────────────────

VioProc::VioProc(NodeContext& ctx)
  : tracker_(ctx.tracker), state_(ctx.state), profiler_(ctx.profiler),
    data_queues_(ctx.data_queues), accumulator_(ctx.state)
{}

std::string VioProc::loadParameters(ros::NodeHandle& pnh)
{
  paramWarn<bool>(pnh, "vio/enable",                opts_.enable,          true);
  paramWarn<int>(pnh, "vio/ekf/max_iterations",     opts_.max_iterations,  5);
  paramWarn<double>(pnh, "vio/ekf/min_norm_dtheta", opts_.min_norm_dtheta, 0.0);
  paramWarn<double>(pnh, "vio/ekf/min_norm_dt",     opts_.min_norm_dt,     0.0);
  paramWarn<double>(pnh, "vio/ekf/min_diff_error",  opts_.min_diff_error,  1e-4);
  paramWarn<bool>(pnh, "vio/ekf/force_all_iterations", opts_.force_all_iterations, false);
  paramWarn<double>(pnh, "vio/ekf/max_avg_error",      opts_.max_avg_error,      5.0);
  paramWarn<double>(pnh, "vio/ekf/epipolar_residual_weight", opts_.epipolar_residual_weight, 0.0556);
  paramWarn<double>(pnh, "vio/ekf/anchor_split_depth", opts_.anchor_split_depth,  0.0);
  paramWarn<bool>(pnh, "vio/ekf/bidirectional",        opts_.bidirectional,        true);
  paramWarn<bool>(pnh, "vio/ekf/distortion_weight_on",  opts_.distortion_weight_on,  false);
  paramWarn<double>(pnh, "vio/ekf/weight_pixel_noise_px", opts_.weight_pixel_noise_px, 0.5);

  paramWarn<bool>(pnh, "log/vio/iterations",       opts_.log_iterations,       false);
  paramWarn<bool>(pnh, "log/vio/point_residuals",  opts_.log_point_residuals,  false);
  paramWarn<bool>(pnh, "log/vio/residual_counts",  opts_.log_residual_counts,  false);
  paramWarn<bool>(pnh, "log/vio/consistency_corr", opts_.log_consistency_corr_en, false);
  paramWarn<bool>(pnh, "log/vio/raw_tracker_output", opts_.log_raw_tracker_output, false);
  paramWarn<bool>(pnh, "log/vio/raw_uv_dump", opts_.log_raw_uv_dump, false);
  paramWarn<double>(pnh, "log/vio/norm_l_low_frac", opts_.log_norm_l_low_frac, 0.3);
  paramWarn<std::string>(pnh, "log/vio/path",       opts_.log_path,             debugLogPath("livo_recon_myvio_log.txt"));

  std::ostringstream oss;
  oss << "[params/vio/ekf]"
      << "\n  enable:          " << (opts_.enable ? "true" : "false")
      << "\n  max_iterations:  " << opts_.max_iterations
      << "\n  min_norm_dtheta: " << opts_.min_norm_dtheta
      << "\n  min_norm_dt:     " << opts_.min_norm_dt
      << "\n  min_diff_error:  " << opts_.min_diff_error
      << "\n  force_all_iterations: " << (opts_.force_all_iterations ? "true" : "false")
      << "\n  max_avg_error:   " << opts_.max_avg_error
      << "\n  epipolar_residual_weight:     " << opts_.epipolar_residual_weight
      << "\n  anchor_split_depth:         " << opts_.anchor_split_depth
      << "\n  bidirectional:              " << (opts_.bidirectional ? "true" : "false")
      << "\n  distortion_weight_on:       " << (opts_.distortion_weight_on ? "true" : "false")
      << "\n  weight_pixel_noise_px:      " << opts_.weight_pixel_noise_px
      << "\n  log_iterations:        " << (opts_.log_iterations ? "true" : "false")
      << "\n  log_point_residuals:   " << (opts_.log_point_residuals ? "true" : "false")
      << "\n  log_residual_counts:   " << (opts_.log_residual_counts ? "true" : "false")
      << "\n  log_consistency_corr_en: " << (opts_.log_consistency_corr_en ? "true" : "false")
      << "\n  log_raw_tracker_output:" << (opts_.log_raw_tracker_output ? "true" : "false")
      << "\n  log_raw_uv_dump:       " << (opts_.log_raw_uv_dump ? "true" : "false")
      << "\n  log_norm_l_low_frac:   " << opts_.log_norm_l_low_frac
      << "\n  log_path:              " << opts_.log_path;

  return oss.str();
}

// History (107-111): see docs/livo_recon_changelog.md#src-processing-vio_processing.cpp-107

VioAccumulateOptions VioProc::accumulateOptions() const
{
  VioAccumulateOptions a;
  a.epipolar_residual_weight = opts_.epipolar_residual_weight;
  a.anchor_split_depth        = opts_.anchor_split_depth;
  a.bidirectional              = opts_.bidirectional;
  a.distortion_weight_on       = opts_.distortion_weight_on;
  a.weight_pixel_noise_px      = opts_.weight_pixel_noise_px;
  a.log_point_residuals        = opts_.log_point_residuals;
  a.log_residual_counts        = opts_.log_residual_counts;
  a.log_norm_l_low_frac        = opts_.log_norm_l_low_frac;
  a.log_path                   = opts_.log_path;
  a.log_consistency_corr_en    = opts_.log_consistency_corr_en;
  return a;
}

bool VioProc::solveSystem(const TrackedFrame& frame, float& last_error, float& last_pixel_residual,
                          double t_abs, int iter)
{
  TimedScope ts(profiler_, "vio/ekf/solve");

  ekf_.reset();
  float avg_error = 0.0f;
  float avg_abs_pixel_residual = 0.0f;
  int n_valid = 0;
  const bool ok = accumulator_.accumulate(frame, frame_anchors_, accumulateOptions(),
                                          ekf_, avg_error, avg_abs_pixel_residual, n_valid, t_abs, iter);
  last_n_residuals_ = n_valid;

  if (!ok) return false;

  // 2026-10-23: gate VIO admission ONCE, at the first iteration, on an ABSOLUTE criterion (the
  // same avg_abs_pixel_residual threshold the whole-frame rollback check below uses, opts_.
  // max_avg_error) -- not every iteration having to IMPROVE on the previous one (the old
  // behavior: `avg_error >= last_error -> reject AND stop the whole loop`, via estimateState
  // Correction()'s "no_improve" break on this function returning false). That conflated real
  // divergence with the ordinary small iteration-to-iteration wobble a converging Gauss-Newton
  // fit has -- fine during a clean, low-blur stationary window (where this matched force_all_
  // iterations closely enough not to matter), but during real motion (lower effective tracking
  // rate, motion blur) a transient wobble could trip this and abandon a frame that would have
  // converged fine. Now: iteration 0 decides whether VIO is admissible AT ALL this frame (a bad
  // initial fit is still rejected, same as before, just judged on an absolute scale instead of
  // "worse than iteration -1" which is meaningless for iter 0 anyway -- last_error starts at
  // FLT_MAX, so the old check was already a no-op at iter 0). Once admitted, later iterations are
  // never rejected for merely not improving on the immediately preceding one; the convergence
  // checks in estimateStateCorrection() (min_norm_dtheta/dt, rel_diff) are what stop the loop once
  // it's actually converged, and the end-of-frame max_avg_error check still catches a frame that
  // degraded badly by its final iteration.
  if (iter == 0 && avg_abs_pixel_residual > static_cast<float>(opts_.max_avg_error) &&
      !opts_.force_all_iterations)
    return false;

  last_error = avg_error;
  last_pixel_residual = avg_abs_pixel_residual;

  ekf_.applyMeanUpdate(state_, prior_cov_, state_propagat_);

  if (opts_.log_iterations)
  {
    const M3D HtH_RR = ekf_.HtH.block<3, 3>(0, 0);
    const M3D HtH_PP = ekf_.HtH.block<3, 3>(3, 3);
    const Eigen::Quaterniond q(state_->rot());
    std::ostringstream ss;
    ss << "[vio/iter]  t_abs=" << std::fixed << std::setprecision(6) << t_abs
       << "  iter=" << iter << "  n_valid=" << n_valid << "  avg_error=" << avg_error
       << "  avg_abs_pixel_residual=" << avg_abs_pixel_residual
       << std::scientific << std::setprecision(4)
       << "  trace(HtH_RR)=" << HtH_RR.trace() << "  trace(HtH_PP)=" << HtH_PP.trace()
       << "  dtheta_iter=[" << ekf_.dtheta.transpose() << "]"
       << "  dt_iter=[" << ekf_.dt.transpose() << "]"
       << "  pos_after=[" << state_->pos().transpose() << "]"
       << "  quat_after(wxyz)=[" << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << "]";
    logIteration(opts_.log_path, ss.str());
  }

  return true;
}

// Combined-mode entry point -- see vio_processing.h's doc comment. Unlike
// solveSystem() above, this does NOT reject against a previous-iteration
// error (CombinedProc's own joint loop owns that policy) and does NOT call
// applyMeanUpdate/applyCovarianceUpdate (CombinedProc sums this with
// LioProc's own accumulation first and applies once).
bool VioProc::accumulateForCombined(MeasureGroup& mg, EkfUpdate& out, float& avg_error,
                                    float& avg_abs_pixel_residual, int iter)
{
  if (mg.image.image.empty() || !tracker_->loaded()) return false;
  const TrackedFrame& frame = mg.tracked_frame;
  if (!frame.ok) return false;

  frame_anchors_ = frame.anchors;
  int n_valid = 0;
  const double t_abs = mg.image.t + data_queues_->start_time;  // see processVIO()'s matching comment
  const bool ok = accumulator_.accumulate(frame, frame_anchors_, accumulateOptions(),
                                          out, avg_error, avg_abs_pixel_residual, n_valid, t_abs, iter);
  last_n_residuals_ = n_valid;
  return ok;
}

// ─────────────────────────────────────────────
//  EKF iteration loop — repeated solveSystem calls to convergence
// ─────────────────────────────────────────────

std::string VioProc::estimateStateCorrection(const TrackedFrame& frame, double t_abs,
                                             float* out_avg_error, float* out_avg_abs_pixel_residual,
                                             int* out_n_valid, const StateGroup* pre_lio_state_for_ask,
                                             int diag_frame_idx)
{
  TimedScope ts_ekf(profiler_, "vio/ekf");

  // Full snapshot so a frame whose best achieved fit is still objectively
  // bad (see opts_.max_avg_error below) can be rolled back entirely,
  // instead of committing a garbage pose that would otherwise permanently
  // contaminate every subsequent frame's fixed "previous pose" reference.
  const StateGroup state_before_frame = *state_;

  if (opts_.log_iterations)
  {
    const Eigen::Quaterniond q_before(state_before_frame.rot());
    std::ostringstream ss;
    ss << "[vio/state_before]  t_abs=" << std::fixed << std::setprecision(6) << t_abs
       << std::scientific << std::setprecision(9)
       << "  pos_before=[" << state_before_frame.pos().transpose() << "]"
       << "  quat_before(wxyz)=[" << q_before.w() << " " << q_before.x() << " " << q_before.y() << " " << q_before.z() << "]";
    logIteration(opts_.log_path, ss.str());
  }

  // Fixed IEKF prior for this frame's ENTIRE inner loop -- see ekf.h's
  // applyMeanUpdate() doc comment. Set once here, read (never rewritten) by
  // every solveSystem() call below until the loop finishes and
  // applyCovarianceUpdate() is called exactly once.
  prior_cov_ = state_before_frame.cov();
  state_propagat_ = state_before_frame;

  // TEMPORARY (2026-10-27, correction_breakdown_diag investigation): what
  // does VIO's measurement evidence ALONE ask for, at frame entry, before
  // any Kalman-gain weighting against the prior? -- i.e. the unweighted
  // normal-equations solution -HtH^-1*Htz, NOT applyMeanUpdate()'s own
  // measurement_term (which already incorporates prior_cov via K1).
  // Reuses the existing out-param request as this feature's gate too.
  // Delete alongside the rest of this investigation's plumbing.
  //
  // Reference-point fix (2026-10-27): combined mode's matching ask
  // diagnostic (combined_processing.cpp) evaluates at the TRUE frame-entry
  // state (post-IMU-propagation, before LIO or VIO has touched anything
  // this frame). In sequential mode, state_ already holds LIO's converged
  // correction by the time this function runs -- so to make the two
  // "ask_vio" numbers comparable, temporarily swap *state_ to the caller-
  // supplied pre-LIO snapshot for this one read-only accumulate() call,
  // then restore it immediately (state_before_frame, captured above,
  // already IS that restore value -- it equals *state_ at function entry).
  // Safe: nothing else reads state_'s pose concurrently with this
  // single-threaded per-frame call (see vio_processing.h's doc comment).
  if (out_avg_error || out_avg_abs_pixel_residual || out_n_valid)
  {
    if (pre_lio_state_for_ask) *state_ = *pre_lio_state_for_ask;

    EkfUpdate ask_scratch;
    float ask_avg_error = 0.0f, ask_avg_abs_pixel_residual = 0.0f;
    int ask_n_valid = 0;
    const bool ask_ok = accumulator_.accumulate(frame, frame_anchors_, accumulateOptions(),
                                                ask_scratch, ask_avg_error, ask_avg_abs_pixel_residual,
                                                ask_n_valid, t_abs, /*iter=*/-1);

    if (pre_lio_state_for_ask) *state_ = state_before_frame;

    V3D ask_dtheta = V3D::Zero(), ask_dt = V3D::Zero();
    double tr_rot_vio = 0.0, tr_pos_vio = 0.0;
    if (ask_ok && ask_n_valid > 0)
    {
      const Eigen::Matrix<double, 6, 1> ask = -ask_scratch.HtH.ldlt().solve(ask_scratch.Htz);
      ask_dtheta = ask.segment<3>(0);
      ask_dt = ask.segment<3>(3);
      tr_rot_vio = ask_scratch.HtH.block<3, 3>(0, 0).trace();
      tr_pos_vio = ask_scratch.HtH.block<3, 3>(3, 3).trace();
    }
    writeCorrectionBreakdownRow(diag_frame_idx >= 0 ? diag_frame_idx : static_cast<int>(++correction_breakdown_frame_index_), t_abs, "sequential",
                                ask_dtheta, ask_dt, ask_n_valid,
                                /*have_lio_ask=*/false, V3D::Zero(), V3D::Zero(), 0,
                                /*have_counterfactual=*/false, V3D::Zero(), V3D::Zero(), V3D::Zero(), V3D::Zero(),
                                /*tr_HtH_rot_lio=*/0.0, /*tr_HtH_pos_lio=*/0.0, tr_rot_vio, tr_pos_vio);
  }

  V3D total_dtheta = V3D::Zero(), total_dt = V3D::Zero();
  float last_error = std::numeric_limits<float>::max();
  float last_pixel_residual = std::numeric_limits<float>::max();
  float prev_error = std::numeric_limits<float>::max();
  std::string stop = "max_iter";
  int n_successful = 0;

  for (int iter = 0; iter < opts_.max_iterations; ++iter)
  {
    if (!solveSystem(frame, last_error, last_pixel_residual, t_abs, iter))
      { stop = "no_improve"; break; }
    ++n_successful;
    total_dtheta += ekf_.dtheta;
    total_dt     += ekf_.dt;
    const float rel_diff = (prev_error - last_error) / std::max(prev_error, 1e-6f);
    prev_error = last_error;

    if (!opts_.force_all_iterations)
    {
      if (ekf_.dtheta.norm() < opts_.min_norm_dtheta && ekf_.dt.norm() < opts_.min_norm_dt)
      {
        std::ostringstream ss;
        ss << std::scientific << std::setprecision(1)
           << "norm(dth=" << ekf_.dtheta.norm() * (180.0 / M_PI) << "deg"
           << ",dt="      << ekf_.dt.norm() * 1000.0              << "mm)";
        stop = ss.str(); break;
      }
      if (rel_diff < static_cast<float>(opts_.min_diff_error))
        { stop = "rel_diff"; break; }
    }
  }

  // Posterior covariance write happens exactly ONCE here, using the SAME
  // fixed prior_cov_ every iteration above blended against (see ekf.h) --
  // not per-iteration like the old applyUpdate() did. No-op (skipped) when
  // n_successful == 0: no real measurement info was ever accumulated this
  // frame (ekf_'s H_full would be stale from a previous frame), so
  // there's nothing to blend and state_->cov() should stay exactly as it
  // was going into this frame. R97A Q2: vio/log/dry_run is removed.
  if (n_successful > 0)
    ekf_.applyCovarianceUpdate(state_, prior_cov_);

  // Absolute sanity check: reject the whole frame's correction if the best
  // fit it ever achieved is still objectively bad, rather than committing
  // it just because solveSystem's own (relative, iteration-to-iteration)
  // criterion happened to accept it. Restoring `state_before_frame`
  // (captured before this frame's prior_cov_/covariance write) discards the
  // just-written posterior covariance too, exactly like it discards the
  // mean correction. 2026-10-23: compares against last_pixel_residual (a
  // genuine mean |pixel error| across tracked points), not last_error
  // (weighted/squared/normalized-coordinate) -- see VioAccumulator::
  // accumulate()'s doc comment. max_avg_error's old default (5.0) was
  // calibrated for the OLD metric's scale and needs re-tuning for this one.
  bool rejected = false;
  if (n_successful > 0 && last_pixel_residual > static_cast<float>(opts_.max_avg_error))
  {
    *state_ = state_before_frame;
    total_dtheta.setZero();
    total_dt.setZero();
    rejected = true;
  }

  // Sentinel iter=99 (same convention as CombinedProc's own post-loop
  // re-check): re-evaluate residuals against the TRULY final state (after
  // the rollback decision above, so it reflects whatever state actually
  // persists past this frame) -- distinct from every in-loop iter value,
  // so log/vio/point_residuals can report a true "after everything"
  // residual instead of only "after the last accepted GN iteration" (which
  // solveSystem()'s own per-iteration rows already give, pre-that-
  // iteration's own update). Scratch EkfUpdate/avg_error/n_valid are
  // unused beyond driving the log write.
  // TEMPORARY (2026-10-27, vio_residual_diag): also run this re-check when
  // an out-param was requested, even if n_successful==0 (e.g. iter-0's own
  // admission gate rejected immediately) -- a caller asking for the true
  // final residual wants it regardless of whether any correction actually
  // got applied this frame. Delete the `|| want_diag_out` disjunct (and the
  // out-param writes below) alongside the rest of this investigation.
  const bool want_diag_out = out_avg_error || out_avg_abs_pixel_residual || out_n_valid;
  if (n_successful > 0 || want_diag_out)
  {
    EkfUpdate final_scratch;
    float final_avg_error = 0.0f;
    float final_avg_abs_pixel_residual = 0.0f;
    int final_n_valid = 0;
    accumulator_.accumulate(frame, frame_anchors_, accumulateOptions(),
                            final_scratch, final_avg_error, final_avg_abs_pixel_residual,
                            final_n_valid, t_abs, 99);
    if (out_avg_error) *out_avg_error = final_avg_error;
    if (out_avg_abs_pixel_residual) *out_avg_abs_pixel_residual = final_avg_abs_pixel_residual;
    if (out_n_valid) *out_n_valid = final_n_valid;
  }

  if (opts_.log_iterations)
  {
    std::ostringstream fss;
    fss << "[vio/frame_stop]  t_abs=" << std::fixed << std::setprecision(6) << t_abs
        << "  n_successful=" << n_successful
        << "  max_iterations=" << opts_.max_iterations
        << "  stop=" << stop
        << "  avg_error=" << (n_successful > 0 ? last_error : std::numeric_limits<float>::quiet_NaN())
        << "  avg_abs_pixel_residual=" << (n_successful > 0 ? last_pixel_residual : std::numeric_limits<float>::quiet_NaN())
        << "  rejected=" << (rejected ? 1 : 0);
    logIteration(opts_.log_path, fss.str());
  }

  std::ostringstream oss;
  oss << "  n=" << frame.uv_curr.size()
      << "  residuals=" << last_n_residuals_
      << "  " << n_successful << "(" << stop << ")";
  if (rejected)
    oss << " REJECTED(avg_abs_pixel_residual=" << last_pixel_residual << " > " << opts_.max_avg_error << ")";
  oss << std::scientific << std::setprecision(1)
      << "  |dtheta|=" << total_dtheta.norm() * (180.0 / M_PI) << " deg"
      << "  |dt|=" << total_dt.norm() * 1000.0 << " mm";

  // Confidence (state covariance) trace, prior (entering this frame, i.e.
  // what propagation+LIO handed off in sequential mode) vs posterior
  // (after this frame's correction, or unchanged if rejected/no_successful)
  // -- lets a diagnostic run show whether the filter is over- or under-
  // confident and how that shifts frame to frame, without needing the
  // opt-in log_iterations file. Always included (cheap: a few 3x3 traces),
  // gated on the SAME PrintCategory::VIO the rest of this string already
  // depends on for visibility.
  {
    const Eigen::MatrixXd& prior_cov = prior_cov_;
    const Eigen::MatrixXd& post_cov = state_->cov();
    oss << std::defaultfloat << std::setprecision(4)
        << "  cov_prior[R,P,V]=[" << prior_cov.block<3, 3>(StateGroup::idxR(), StateGroup::idxR()).trace()
        << "," << prior_cov.block<3, 3>(StateGroup::idxP(), StateGroup::idxP()).trace()
        << "," << prior_cov.block<3, 3>(StateGroup::idxV(), StateGroup::idxV()).trace() << "]"
        << "  cov_post[R,P,V]=[" << post_cov.block<3, 3>(StateGroup::idxR(), StateGroup::idxR()).trace()
        << "," << post_cov.block<3, 3>(StateGroup::idxP(), StateGroup::idxP()).trace()
        << "," << post_cov.block<3, 3>(StateGroup::idxV(), StateGroup::idxV()).trace() << "]";
  }

  return oss.str();
}

// ─────────────────────────────────────────────
//  Main entry point
// ─────────────────────────────────────────────

std::string VioProc::processVIO(MeasureGroup& mg, bool force_run,
                                float* out_avg_error, float* out_avg_abs_pixel_residual,
                                int* out_n_valid, bool* out_ok,
                                const StateGroup* pre_lio_state_for_ask,
                                int diag_frame_idx)
{
  TimedScope ts(profiler_, "vio");

  if (out_ok) *out_ok = false;

  if (!opts_.enable && !force_run)
    return "";

  if (mg.image.image.empty() || !tracker_->loaded())
    return "[vio/ekf]  skipped (no image or tracker not loaded)";

  // Already populated by CbkProc::syncMeasures() (task #145) -- see
  // MeasureGroup::tracked_frame's own doc comment.
  const TrackedFrame& frame = mg.tracked_frame;
  // Absolute (bag/wall-clock) timestamp, matching lio_processing.cpp's own
  // [lio] t_abs convention and FAST-LIVO2's [vio/iter]/[lio] logs -- 2026-
  // 08-16, was previously bag-RELATIVE (bare mg.image.t) here, unlike every
  // other debug log in this codebase, which made timestamp-matching this
  // log against LIO's own or against FAST-LIVO2's logs require manually
  // reconstructing the bag start offset.
  const double t_abs = mg.image.t + data_queues_->start_time;

  // See log_raw_tracker_output's doc comment -- BEFORE estimateStateCorrection()
  // ever touches these values, so this is the tracker's raw output
  // completely undisturbed by anything EKF-side.
  if (opts_.log_raw_tracker_output)
  {
    std::ostringstream ss;
    ss << "[vio/raw_tracker]  t_abs=" << std::fixed << std::setprecision(6) << t_abs
       << "  ok=" << (frame.ok ? 1 : 0)
       << "  n_tracked=" << frame.uv_curr.size()
       << "  n_anchors=" << frame.anchors.size()
       << "  reseeded=" << (frame.reseeded ? 1 : 0);
    logIteration(opts_.log_path, ss.str());
  }

  if (opts_.log_raw_uv_dump && frame.ok)
  {
    // One row per GENUINE tracked pair: the seed pixel this anchor was
    // picked at (anchors[idx].uv_seed -- the frame that seeded the
    // tracker, already distortion-corrected, see AnchorPoint::uv_seed)
    // and this frame's own tracked pixel for that same anchor -- the
    // following frame in the pair. su/sv -> uc/vc (frame.uv_curr[i],
    // also distortion-corrected) is the directly-comparable pair in a
    // CONSISTENT coordinate space -- unlike comparing uv_curr across
    // DIFFERENT frames' independently-reseeded idx slots (not a stable
    // point identity beyond one single step -- see Tracker::
    // stepAndReseedOnce() and CoTrackerBackend's class doc comment: every
    // frame reseeds its entire anchor set fresh). uc/vc (unlike
    // uv_curr_raw) is present in BOTH live and cache-replay TrackedFrame
    // records (an old field, always serialized), so this comparison works
    // identically whether tracking lived or replayed from cache. u/v
    // (uv_curr_raw, raw/pre-undistortion) is logged alongside only for a
    // live run that wants the true raw pixel too -- empty (0,0) when
    // replaying from a cache predating this field.
    std::ostringstream ss;
    for (size_t i = 0; i < frame.uv_curr.size(); ++i)
    {
      const int anchor_idx = (i < frame.indices.size()) ? frame.indices[i] : -1;
      const bool have_seed = anchor_idx >= 0 && static_cast<size_t>(anchor_idx) < frame.anchors.size();
      const cv::Point2f seed = have_seed ? frame.anchors[anchor_idx].uv_seed : cv::Point2f(-1.0f, -1.0f);
      const bool have_raw = i < frame.uv_curr_raw.size();
      const cv::Point2f raw = have_raw ? frame.uv_curr_raw[i] : cv::Point2f(0.0f, 0.0f);
      ss << "[vio/raw_uv]  t_abs=" << std::fixed << std::setprecision(6) << t_abs
         << "  idx=" << anchor_idx
         << "  su=" << std::setprecision(4) << seed.x
         << "  sv=" << seed.y
         << "  uc=" << frame.uv_curr[i].x
         << "  vc=" << frame.uv_curr[i].y
         << "  u=" << raw.x
         << "  v=" << raw.y << "\n";
    }
    logIteration(opts_.log_path, ss.str());
  }

  if (!frame.ok)
    return "[vio/ekf]  step failed";

  // See frame_anchors_'s doc comment (vio_processing.h) -- a snapshot, not
  // a live read of tracker_->anchors() (which may already reflect a NEWER
  // reseed by now).
  frame_anchors_ = frame.anchors;

  if (out_ok) *out_ok = true;
  std::string log = "[vio/ekf]" + estimateStateCorrection(frame, t_abs, out_avg_error,
                                                           out_avg_abs_pixel_residual, out_n_valid,
                                                           pre_lio_state_for_ask, diag_frame_idx);

  if (!frame.log.empty())
    log += "  " + frame.log;

  return log;
}

}  // namespace livo_recon
