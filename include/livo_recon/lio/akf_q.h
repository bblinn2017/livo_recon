#pragma once

// R66: opt-in AKF-LIO-style adaptive process noise, simplified to the diagonal IMU noise block.
//
// AKF-LIO maps the filter's own state correction back into noise space: a correction dx = x_post - x_prior
// is the process-model error the measurements disagreed with, so a forgetting-factor average of the noise
// implied by dx estimates Q. This implementation is a DEPARTURE from the exact blockwise F_w^{-1} form:
// it aggregates the correction over one frame and inverts only the two diagonal noise blocks that the
// fixed model exposes (acc_variance, gyro_variance):
//
//   attitude error accumulated over the frame's N IMU steps:  Var(dtheta_axis) = var_gyr * sum_i dt_i^2
//   velocity  error accumulated over the frame:               Var(dv_axis)     = var_acc * sum_i dt_i^2
//   so per frame:  var_gyr_hat = |dtheta|^2 / (3 sum dt_i^2),  var_acc_hat = |dv|^2 / (3 sum dt_i^2)
//
// (dtheta, dv from StateGroup::boxminusFromPropagat, tangent-space rotation and velocity blocks).
// Limits (stated, not hidden): dx also contains measurement noise and map error, so the estimate is an upper
// bound on process noise; one frame gives one 3-vector sample per block (high variance, hence the forgetting
// factor and the ratio bounds); bias-rate adaptation is NOT implemented. Applied to the state through
// StateGroup::setNoiseParams, which only the "fixed" process-noise model reads (refused under other models).

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

#include "livo_recon/diagnostics/log/debug_log_dir.h"

namespace livo_recon
{

struct AkfQOptions
{
  bool   enable = false;
  double forgetting = 0.95;       // Qhat <- a*Qhat + (1-a)*sample
  int    warmup_frames = 20;      // frames observed before the estimate is applied
  double min_ratio = 0.01;        // applied value bounded to [min_ratio, max_ratio] x nominal
  double max_ratio = 100.0;
};

class AkfQ
{
public:
  void configure(const AkfQOptions& o) { o_ = o; }
  bool enabled() const { return o_.enable; }
  const AkfQOptions& options() const { return o_; }

  void setNominal(double var_acc, double var_gyr)
  {
    nom_acc_ = var_acc; nom_gyr_ = var_gyr;
    hat_acc_ = var_acc; hat_gyr_ = var_gyr;
    primed_ = true;
  }
  bool primed() const { return primed_; }

  // Returns true when a value is to be applied this frame (after warmup). dtheta_sq / dv_sq are squared norms.
  bool update(long scan_id, double t_abs, int n_steps, double sum_dt2, double dtheta_sq, double dv_sq)
  {
    ++frames_;
    bool applied = false, clamped = false;
    double s_gyr = 0.0, s_acc = 0.0;
    if (n_steps > 0 && sum_dt2 > 0.0)
    {
      s_gyr = dtheta_sq / (3.0 * sum_dt2);
      s_acc = dv_sq / (3.0 * sum_dt2);
      hat_gyr_ = o_.forgetting * hat_gyr_ + (1.0 - o_.forgetting) * s_gyr;
      hat_acc_ = o_.forgetting * hat_acc_ + (1.0 - o_.forgetting) * s_acc;
      ++used_frames_;
    }
    const double lo_a = o_.min_ratio * nom_acc_, hi_a = o_.max_ratio * nom_acc_;
    const double lo_g = o_.min_ratio * nom_gyr_, hi_g = o_.max_ratio * nom_gyr_;
    app_acc_ = std::min(std::max(hat_acc_, lo_a), hi_a);
    app_gyr_ = std::min(std::max(hat_gyr_, lo_g), hi_g);
    clamped = (app_acc_ != hat_acc_) || (app_gyr_ != hat_gyr_);
    if (clamped) ++clamped_frames_;
    if (used_frames_ > o_.warmup_frames)
    {
      applied = true;
      ++applied_frames_;
      if (applied_frames_ == 1) { min_acc_ = max_acc_ = app_acc_; min_gyr_ = max_gyr_ = app_gyr_; }
      min_acc_ = std::min(min_acc_, app_acc_); max_acc_ = std::max(max_acc_, app_acc_);
      min_gyr_ = std::min(min_gyr_, app_gyr_); max_gyr_ = std::max(max_gyr_, app_gyr_);
    }
    static PersistentLogStream log("akf_q.csv");
    bool first = false;
    std::ofstream& out = log.stream(&first);
    if (first)
      out << "scan_id,t_abs,n_steps,sum_dt2_s2,dtheta_sq_rad2,dv_sq_m2s2,sample_var_gyr,sample_var_acc,"
             "hat_var_gyr,hat_var_acc,applied_var_gyr,applied_var_acc,nominal_var_gyr,nominal_var_acc,"
             "applied_flag,clamped_flag\n";
    out << scan_id << ',' << std::setprecision(17) << t_abs << ',' << n_steps << ',' << sum_dt2 << ','
        << dtheta_sq << ',' << dv_sq << ',' << s_gyr << ',' << s_acc << ',' << hat_gyr_ << ',' << hat_acc_
        << ',' << app_gyr_ << ',' << app_acc_ << ',' << nom_gyr_ << ',' << nom_acc_ << ','
        << (applied ? 1 : 0) << ',' << (clamped ? 1 : 0) << '\n';
    out.flush();
    return applied;
  }

  double varAcc() const { return app_acc_; }
  double varGyr() const { return app_gyr_; }
  long frames() const { return frames_; }
  long appliedFrames() const { return applied_frames_; }
  long clampedFrames() const { return clamped_frames_; }
  double minAcc() const { return min_acc_; }
  double maxAcc() const { return max_acc_; }
  double minGyr() const { return min_gyr_; }
  double maxGyr() const { return max_gyr_; }

private:
  AkfQOptions o_;
  bool primed_ = false;
  double nom_acc_ = 0.0, nom_gyr_ = 0.0;
  double hat_acc_ = 0.0, hat_gyr_ = 0.0;
  double app_acc_ = 0.0, app_gyr_ = 0.0;
  double min_acc_ = 0.0, max_acc_ = 0.0, min_gyr_ = 0.0, max_gyr_ = 0.0;
  long frames_ = 0, used_frames_ = 0, applied_frames_ = 0, clamped_frames_ = 0;
};

}  // namespace livo_recon
