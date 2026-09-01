#include "livo_recon/processing/vio_processing.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/algo/omp_utils.h"
#include "livo_recon/utils/log/debug_log_dir.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
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
void logIteration(const std::string& path, const std::string& msg)
{
  static std::set<std::string> truncated_paths;
  const bool first_call_for_path = truncated_paths.insert(path).second;
  std::ofstream ofs(path, first_call_for_path ? std::ios::trunc : std::ios::app);
  ofs << msg << "\n";
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
  paramWarn<double>(pnh, "vio/ekf/max_avg_error",      opts_.max_avg_error,      5.0);
  paramWarn<double>(pnh, "vio/ekf/pos_epipolar_residual_var", opts_.pos_epipolar_residual_var, 0.0556);
  paramWarn<double>(pnh, "vio/ekf/neg_epipolar_residual_var", opts_.neg_epipolar_residual_var, 0.03125);
  paramWarn<double>(pnh, "vio/ekf/pos_weight_scale", opts_.pos_weight_scale, 1.0);
  paramWarn<std::string>(pnh, "vio/ekf/residual_mode", opts_.residual_mode, "epipolar");
  paramWarn<double>(pnh, "vio/ekf/pos_ray_distance_residual_var", opts_.pos_ray_distance_residual_var, 1000.0);
  paramWarn<double>(pnh, "vio/ekf/neg_ray_distance_residual_var", opts_.neg_ray_distance_residual_var, 10.0);
  paramWarn<double>(pnh, "vio/ekf/anchor_split_depth", opts_.anchor_split_depth,  0.0);
  paramWarn<bool>(pnh, "vio/ekf/bidirectional",        opts_.bidirectional,        true);
  paramWarn<bool>(pnh, "vio/ekf/distortion_weight_on",  opts_.distortion_weight_on,  false);
  paramWarn<double>(pnh, "vio/ekf/weight_pixel_noise_px", opts_.weight_pixel_noise_px, 0.5);

  paramWarn<bool>(pnh, "vio/log/log_iterations",       opts_.log_iterations,       false);
  paramWarn<bool>(pnh, "vio/log/log_point_residuals",  opts_.log_point_residuals,  false);
  paramWarn<bool>(pnh, "vio/log/log_residual_counts",  opts_.log_residual_counts,  false);
  paramWarn<bool>(pnh, "vio/log/log_consistency_corr_en", opts_.log_consistency_corr_en, false);
  paramWarn<bool>(pnh, "vio/log/log_raw_tracker_output", opts_.log_raw_tracker_output, false);
  paramWarn<bool>(pnh, "vio/log/dry_run",              opts_.dry_run,              false);
  paramWarn<double>(pnh, "vio/log/log_norm_l_low_frac", opts_.log_norm_l_low_frac, 0.3);
  paramWarn<std::string>(pnh, "vio/log/log_path",       opts_.log_path,             debugLogPath("livo_recon_myvio_log.txt"));

  std::ostringstream oss;
  oss << "[params/vio/ekf]"
      << "\n  enable:          " << (opts_.enable ? "true" : "false")
      << "\n  max_iterations:  " << opts_.max_iterations
      << "\n  min_norm_dtheta: " << opts_.min_norm_dtheta
      << "\n  min_norm_dt:     " << opts_.min_norm_dt
      << "\n  min_diff_error:  " << opts_.min_diff_error
      << "\n  max_avg_error:   " << opts_.max_avg_error
      << "\n  pos_epipolar_residual_var: " << opts_.pos_epipolar_residual_var
      << "\n  neg_epipolar_residual_var: " << opts_.neg_epipolar_residual_var
      << "\n  pos_weight_scale:          " << opts_.pos_weight_scale
      << "\n  residual_mode:             " << opts_.residual_mode
      << "\n  pos_ray_distance_residual_var: " << opts_.pos_ray_distance_residual_var
      << "\n  neg_ray_distance_residual_var: " << opts_.neg_ray_distance_residual_var
      << "\n  anchor_split_depth:         " << opts_.anchor_split_depth
      << "\n  bidirectional:              " << (opts_.bidirectional ? "true" : "false")
      << "\n  distortion_weight_on:       " << (opts_.distortion_weight_on ? "true" : "false")
      << "\n  weight_pixel_noise_px:      " << opts_.weight_pixel_noise_px
      << "\n  log_iterations:        " << (opts_.log_iterations ? "true" : "false")
      << "\n  log_point_residuals:   " << (opts_.log_point_residuals ? "true" : "false")
      << "\n  log_residual_counts:   " << (opts_.log_residual_counts ? "true" : "false")
      << "\n  log_consistency_corr_en: " << (opts_.log_consistency_corr_en ? "true" : "false")
      << "\n  log_raw_tracker_output:" << (opts_.log_raw_tracker_output ? "true" : "false")
      << "\n  dry_run:               " << (opts_.dry_run ? "true" : "false")
      << "\n  log_norm_l_low_frac:   " << opts_.log_norm_l_low_frac
      << "\n  log_path:              " << opts_.log_path;

  return oss.str();
}

// History (107-111): see docs/livo_recon_changelog.md#src-processing-vio_processing.cpp-107

VioAccumulateOptions VioProc::accumulateOptions() const
{
  VioAccumulateOptions a;
  a.pos_epipolar_residual_var = opts_.pos_epipolar_residual_var;
  a.neg_epipolar_residual_var = opts_.neg_epipolar_residual_var;
  a.pos_weight_scale          = opts_.pos_weight_scale;
  a.residual_mode             = opts_.residual_mode;
  a.pos_ray_distance_residual_var = opts_.pos_ray_distance_residual_var;
  a.neg_ray_distance_residual_var = opts_.neg_ray_distance_residual_var;
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

bool VioProc::solveSystem(const TrackedFrame& frame, float& last_error, double t_abs, int iter)
{
  TimedScope ts(profiler_, "vio/ekf/solve");

  ekf_.reset();
  float avg_error = 0.0f;
  int n_valid = 0;
  const bool ok = accumulator_.accumulate(frame, frame_anchors_, accumulateOptions(),
                                          ekf_, avg_error, n_valid, t_abs, iter);
  last_n_residuals_ = n_valid;

  if (!ok) return false;

  // History (147-172): see docs/livo_recon_changelog.md#src-processing-vio_processing.cpp-147
  if (avg_error >= last_error) return false;

  last_error = avg_error;

  // See dry_run's doc comment -- HtH/Htz above are still the real
  // accumulated residuals/Jacobians (fully logged inside accumulate()),
  // but the actual state mutation is skipped when set.
  if (!opts_.dry_run)
    ekf_.applyMeanUpdate(state_, prior_cov_, state_propagat_);

  if (opts_.log_iterations)
  {
    const M3D HtH_RR = ekf_.HtH.block<3, 3>(0, 0);
    const M3D HtH_PP = ekf_.HtH.block<3, 3>(3, 3);
    const Eigen::Quaterniond q(state_->rot());
    std::ostringstream ss;
    ss << "[vio/iter]  t_abs=" << std::fixed << std::setprecision(6) << t_abs
       << "  iter=" << iter << "  n_valid=" << n_valid << "  avg_error=" << avg_error
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
bool VioProc::accumulateForCombined(MeasureGroup& mg, EkfUpdate& out, float& avg_error)
{
  if (mg.image.image.empty() || !tracker_->loaded()) return false;
  const TrackedFrame& frame = mg.tracked_frame;
  if (!frame.ok) return false;

  frame_anchors_ = frame.anchors;
  int n_valid = 0;
  const double t_abs = mg.image.t + data_queues_->start_time;  // see processVIO()'s matching comment
  const bool ok = accumulator_.accumulate(frame, frame_anchors_, accumulateOptions(),
                                          out, avg_error, n_valid, t_abs, 0);
  last_n_residuals_ = n_valid;
  return ok;
}

// ─────────────────────────────────────────────
//  EKF iteration loop — repeated solveSystem calls to convergence
// ─────────────────────────────────────────────

std::string VioProc::estimateStateCorrection(const TrackedFrame& frame, double t_abs)
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

  V3D total_dtheta = V3D::Zero(), total_dt = V3D::Zero();
  float last_error = std::numeric_limits<float>::max();
  float prev_error = std::numeric_limits<float>::max();
  std::string stop = "max_iter";
  int n_successful = 0;

  for (int iter = 0; iter < opts_.max_iterations; ++iter)
  {
    if (!solveSystem(frame, last_error, t_abs, iter))
      { stop = "no_improve"; break; }
    ++n_successful;
    total_dtheta += ekf_.dtheta;
    total_dt     += ekf_.dt;
    const float rel_diff = (prev_error - last_error) / std::max(prev_error, 1e-6f);
    prev_error = last_error;

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

  // Posterior covariance write happens exactly ONCE here, using the SAME
  // fixed prior_cov_ every iteration above blended against (see ekf.h) --
  // not per-iteration like the old applyUpdate() did. No-op (skipped) when
  // n_successful == 0 OR dry_run (see that option's doc comment): no real
  // measurement info was ever accumulated this frame (ekf_'s H_full would
  // be stale from a previous frame), so there's nothing to blend and
  // state_->cov() should stay exactly as it was going into this frame.
  if (n_successful > 0 && !opts_.dry_run)
    ekf_.applyCovarianceUpdate(state_, prior_cov_);

  // Absolute sanity check: reject the whole frame's correction if the best
  // fit it ever achieved is still objectively bad, rather than committing
  // it just because solveSystem's own (relative, iteration-to-iteration)
  // criterion happened to accept it. Restoring `state_before_frame`
  // (captured before this frame's prior_cov_/covariance write) discards the
  // just-written posterior covariance too, exactly like it discards the
  // mean correction.
  bool rejected = false;
  if (n_successful > 0 && last_error > static_cast<float>(opts_.max_avg_error))
  {
    *state_ = state_before_frame;
    total_dtheta.setZero();
    total_dt.setZero();
    rejected = true;
  }

  if (opts_.log_iterations)
  {
    std::ostringstream fss;
    fss << "[vio/frame_stop]  t_abs=" << std::fixed << std::setprecision(6) << t_abs
        << "  n_successful=" << n_successful
        << "  max_iterations=" << opts_.max_iterations
        << "  stop=" << stop
        << "  avg_error=" << (n_successful > 0 ? last_error : std::numeric_limits<float>::quiet_NaN())
        << "  rejected=" << (rejected ? 1 : 0);
    logIteration(opts_.log_path, fss.str());
  }

  std::ostringstream oss;
  oss << "  n=" << frame.uv_curr.size()
      << "  residuals=" << last_n_residuals_
      << "  " << n_successful << "(" << stop << ")";
  if (rejected)
    oss << " REJECTED(avg_error=" << last_error << " > " << opts_.max_avg_error << ")";
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

std::string VioProc::processVIO(MeasureGroup& mg)
{
  TimedScope ts(profiler_, "vio");

  if (!opts_.enable)
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

  if (!frame.ok)
    return "[vio/ekf]  step failed";

  // See frame_anchors_'s doc comment (vio_processing.h) -- a snapshot, not
  // a live read of tracker_->anchors() (which may already reflect a NEWER
  // reseed by now).
  frame_anchors_ = frame.anchors;

  std::string log = "[vio/ekf]" + estimateStateCorrection(frame, t_abs);

  if (!frame.log.empty())
    log += "  " + frame.log;

  return log;
}

}  // namespace livo_recon
