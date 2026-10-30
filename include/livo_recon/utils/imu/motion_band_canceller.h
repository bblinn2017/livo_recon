#pragma once
// Attenuation of a broadband noise hump in the mid-360's accelerometer, present
// ONLY while the device is moving (cbk/imu/motion_band/*).
//
// Investigation (docs/QSCALE_REFINE_RESTRUCTURE_PLAN.md): confirmed on calib_10_28.bag
// that the raw accelerometer's 8-12Hz band has a real but NON-mechanical character,
// unlike the prism vibration lines VibrationCanceller removes:
//   - it is NOT a sharp spectral line (peak/local-floor ratio ~3, vs ~53 for the
//     28.0119Hz prism line) -- a dense, broadband continuum across the whole band,
//     not a tone with a few close sub-peaks (no beating structure either)
//   - its amplitude is NOT a smooth function of motion intensity -- it is close to
//     a STEP: ~0.002-0.003 m/s^2 (per axis) while stationary, jumping to a roughly
//     CONSTANT ~0.045-0.066 m/s^2 plateau the instant gyro-magnitude crosses
//     ~0.025-0.03 rad/s, then staying flat across a further ~35x range of gyro
//     magnitude (confirmed stable 24-25x floor/plateau ratio across threshold
//     choices from 0.0225 to 0.05 rad/s)
// So unlike VibrationCanceller (which tracks a known tone's amplitude/phase via a
// recursive least-squares fit and subtracts its coherent prediction), this is a
// gated, non-coherent broadband attenuation: a per-axis causal bandpass isolates
// the band, a causal low-passed gyro-magnitude decides whether the device is
// "in motion", and when it is, a FIXED fraction of the isolated band component
// (1 - floor/plateau, i.e. enough to bring the plateau level back down to the
// floor level) is subtracted. No frequency/phase tracking is needed or possible
// here -- there is no coherent tone to track.
#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <vector>

namespace livo_recon
{

struct MotionBandCancellerOptions
{
  bool enable = false;
  double sample_rate_hz = 200.0;
  double band_lo_hz = 8.0, band_hi_hz = 12.0;
  // Motion-detection threshold on a low-passed gyro-magnitude signal (rad/s).
  // See this file's header comment: 0.025-0.05 all give the same ~24-25x
  // floor/plateau ratio -- not a sensitive choice, but must sit comfortably
  // above the stationary noise floor (~0.02 rad/s on this unit).
  double gyro_threshold_rad_s = 0.03;
  // Smoothing time constant for the gyro-magnitude motion detector (a single-
  // pole causal EMA, not the offline analysis's non-causal filtfilt -- avoids
  // flicker right at the threshold without needing future samples).
  double gyro_lpf_tau_s = 0.5;
  // Per-axis (x,y,z) 8-12Hz envelope amplitude (acc m/s^2) while stationary
  // and while in motion -- both measured directly from a real recording's
  // raw IMU stream (median envelope in the detected still/motion periods),
  // not assumed. attenuation[i] = 1 - floor[i]/plateau[i] is the fraction of
  // the isolated band component removed whenever in motion.
  std::vector<double> floor_amplitude{0.0, 0.0, 0.0};
  std::vector<double> plateau_amplitude{0.0, 0.0, 0.0};
};

class MotionBandCanceller
{
public:
  void configure(const MotionBandCancellerOptions& opts);
  bool active() const { return enabled_; }

  // Mutates acc in place (subtracts the gated band component); gyro is
  // read-only, used only to decide whether the device is in motion.
  void process(const Eigen::Vector3d& gyro, Eigen::Vector3d& acc);

private:
  // Causal 2nd-order (RBJ cookbook, constant-0dB-peak-gain) bandpass biquad.
  struct Biquad
  {
    double b0 = 0, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    double x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    double step(double x)
    {
      const double y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      x2 = x1; x1 = x;
      y2 = y1; y1 = y;
      return y;
    }
  };

  bool enabled_ = false;
  double gyro_alpha_ = 0.0;        // EMA coefficient for the motion-detection smoothing
  double gyro_threshold_ = 0.03;
  double gyro_intensity_ = 0.0;    // running smoothed |gyro|
  Eigen::Vector3d attenuation_ = Eigen::Vector3d::Zero();
  std::array<Biquad, 3> band_;
};

inline void MotionBandCanceller::configure(const MotionBandCancellerOptions& opts)
{
  enabled_ = opts.enable;
  if (!enabled_) return;

  const double dt = 1.0 / opts.sample_rate_hz;
  gyro_alpha_ = dt / (opts.gyro_lpf_tau_s + dt);
  gyro_threshold_ = opts.gyro_threshold_rad_s;
  gyro_intensity_ = 0.0;

  // RBJ cookbook bandpass (constant 0dB peak gain), Q = center/bandwidth.
  const double center_hz = 0.5 * (opts.band_lo_hz + opts.band_hi_hz);
  const double bandwidth_hz = opts.band_hi_hz - opts.band_lo_hz;
  const double w0 = 2.0 * M_PI * center_hz / opts.sample_rate_hz;
  const double q = center_hz / bandwidth_hz;
  const double alpha = std::sin(w0) / (2.0 * q);
  const double cos_w0 = std::cos(w0);
  const double a0 = 1.0 + alpha;
  const double b0 = alpha / a0, b1 = 0.0, b2 = -alpha / a0;
  const double a1 = -2.0 * cos_w0 / a0, a2 = (1.0 - alpha) / a0;

  for (int i = 0; i < 3; i++)
  {
    band_[i] = Biquad{};
    band_[i].b0 = b0; band_[i].b1 = b1; band_[i].b2 = b2; band_[i].a1 = a1; band_[i].a2 = a2;
    attenuation_[i] = (opts.plateau_amplitude[i] > 1e-12)
        ? std::max(0.0, 1.0 - opts.floor_amplitude[i] / opts.plateau_amplitude[i])
        : 0.0;
  }
}

inline void MotionBandCanceller::process(const Eigen::Vector3d& gyro, Eigen::Vector3d& acc)
{
  if (!enabled_) return;
  gyro_intensity_ += gyro_alpha_ * (gyro.norm() - gyro_intensity_);
  const bool in_motion = gyro_intensity_ > gyro_threshold_;
  for (int i = 0; i < 3; i++)
  {
    const double band_component = band_[i].step(acc[i]);
    if (in_motion) acc[i] -= attenuation_[i] * band_component;
  }
}

}  // namespace livo_recon
