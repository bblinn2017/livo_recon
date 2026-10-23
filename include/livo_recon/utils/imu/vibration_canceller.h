#pragma once
// Removal of the Mid-360's prism vibration from the IMU stream (cbk/imu/vibration/*).
//
// The rotating prisms shake the housing at a few narrow lines, seen in all six IMU channels: the main line
// (28.025 Hz on this unit), sidebands at +-0.790 Hz from a modulation of it, its 2nd harmonic and two weaker
// lines (13.400, 18.124 Hz). On the stationary windows of capture_10_20 and capture_10_20_no3 they carry
// 50-90 % of the per-sample accelerometer variance (no3: std 0.21 m/s^2 on z, vs 0.017 without them), but
// they are real motion of microns and microradians that cancels within a frame, so they inflate the
// stationary floors (and the motion term's excitation) without adding drift. Constants: derived offline
// by qscale_fit --vib-cancel (LiDAR_IMU_Init), see config/mid360.yaml.
//
// Per sample, causal (an adaptive line canceller):
//   - phases advance by the sample count round(dt * sample_rate): the IMU samples on its own clock, the
//     header stamps jitter (4.1 to 5.9 ms around 5 ms on this unit)
//   - each channel goes through a 2nd-order high pass (x minus its EMA, twice), which removes gravity and
//     the motion, then a joint exponentially weighted least-squares fit of cos / sin of every line (one
//     gain matrix: the regressors are the same for all six channels)
//   - the vibration that is subtracted is the fit's PREDICTION from the earlier samples (a priori), divided
//     by the high pass response. The current sample's own noise is never absorbed into the fit, so white
//     noise is not removed (no overfitting; a fit after the update would remove about 1 / (4 tau) Hz of
//     noise around every line)
//   - frequency loop: the phase drift of the main line's fitted amplitudes (summed over channels) corrects
//     it; sidebands (main +- offset) and harmonics follow it. Extra lines track slowly, clamped to
//     max_freq_dev_hz around their configured value. Sidebands start after sideband_delay_s: closer than
//     1 Hz to the main line they are not separable from it in less than about 2 / offset seconds
//   - a gap longer than max_gap_s restarts the canceller (the vibration phase is lost)
//   - apply = false: the fit runs (learns) but the sample is passed through unchanged. CbkProc uses this
//     until the stationary calibration is done, so the calibration sees the raw stream and removes the
//     lines itself with removeLinesBatch() (a cold canceller has not converged within its 0.5 s window),
//     and the canceller is already converged when it starts subtracting
// The amplitudes depend on the mount (a table: 0.04 m/s^2; a plastic bin lid: 0.11-0.23 m/s^2, z-dominant),
// so they are always learned online; only the frequencies are configured.
// Frequencies are in cycles per sample internally.
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "livo_recon/utils/algo/symmetrize.h"

namespace livo_recon
{

struct VibrationCancellerOptions
{
  bool enable = false;
  double sample_rate_hz = 200.0;     // the IMU's own sample clock (mean rate, not 1 / median stamp step)
  double main_hz = 0.0;              // main line (tracked)
  double sideband_offset_hz = 0.0;   // 0: no sidebands
  int harmonics = 1;                 // of the main line (1: none)
  std::vector<double> extra_hz;      // weaker lines (tracked slowly, clamped)
  double tau_fit_s = 4.0;            // least-squares forgetting time
  double highpass_hz = 8.0;          // both high pass stages
  double tau_freq_s = 20.0;          // main line frequency loop
  double tau_freq_extra_s = 60.0;    // extra lines
  double max_freq_dev_hz = 0.05;     // extra lines stay within this of their configured value
  double sideband_delay_s = 2.5;
  double max_gap_s = 0.5;

  // Diagnostic-only (2026-10-23 session): seed the fit's initial amplitude
  // (acc m/s^2, gyro rad/s; 6 values per line, x/y/z acc then x/y/z gyro)
  // for each of the 6 configured lines in configure()'s own build order
  // (MAIN, SIDEBAND-, SIDEBAND+, HARMONIC(2), EXTRA[0], EXTRA[1]), instead
  // of starting every line's fit at zero amplitude every reset. Phase is
  // NOT seeded (it changes with every power-up, per this header's own doc
  // comment -- there is nothing to seed it FROM) -- the seed places all
  // of the given amplitude on the cosine term (phase 0), an arbitrary but
  // no-worse-than-any-other starting guess, and the RLS gain (P_, left at
  // its normal uninformative kP0 prior -- NOT tightened around the seed)
  // corrects phase at its usual rate. Empty (default) is bit-identical to
  // before this option existed. Each inner vector must be empty or size 6
  // if the corresponding line exists (sideband/harmonic/extra seeds are
  // ignored if sideband_offset_hz/harmonics/extra_hz don't produce that
  // line); seeding a line that doesn't exist is a no-op, not an error.
  std::vector<double> seed_main;
  std::vector<double> seed_sideband_lo;
  std::vector<double> seed_sideband_hi;
  std::vector<double> seed_harmonic2;
  std::vector<double> seed_extra0;
  std::vector<double> seed_extra1;
};

class VibrationCanceller
{
public:
  enum class Kind { MAIN, SIDEBAND, HARMONIC, EXTRA };
  struct Line
  {
    Kind kind;
    double f0;      // configured, cycles/sample
    double f;       // current, cycles/sample
    double param;   // sideband: signed offset (cycles/sample); harmonic: order
    long start;     // first sample index with this line in the fit
  };

  VibrationCanceller() = default;
  explicit VibrationCanceller(const VibrationCancellerOptions& opt) { configure(opt); }

  void configure(const VibrationCancellerOptions& opt)
  {
    opt_ = opt;
    lines_.clear();
    if (!opt_.enable || !(opt_.main_hz > 0.0) || !(opt_.sample_rate_hz > 0.0)) return;
    const double fs = opt_.sample_rate_hz;
    const long sb_start = std::lround(opt_.sideband_delay_s * fs);
    lines_.push_back({Kind::MAIN, opt_.main_hz / fs, opt_.main_hz / fs, 0.0, 0});
    if (opt_.sideband_offset_hz > 0.0)
      for (double s : {-1.0, 1.0})
        lines_.push_back({Kind::SIDEBAND, (opt_.main_hz + s * opt_.sideband_offset_hz) / fs,
                          (opt_.main_hz + s * opt_.sideband_offset_hz) / fs, s * opt_.sideband_offset_hz / fs, sb_start});
    for (int h = 2; h <= opt_.harmonics; ++h)
      if (h * opt_.main_hz < 0.5 * fs)
        lines_.push_back({Kind::HARMONIC, h * opt_.main_hz / fs, h * opt_.main_hz / fs, static_cast<double>(h), 0});
    for (double f : opt_.extra_hz)
      if (f > 0.0 && f < 0.5 * fs) lines_.push_back({Kind::EXTRA, f / fs, f / fs, 0.0, 0});
    a_hp_ = 1.0 - std::exp(-2.0 * M_PI * opt_.highpass_hz / fs);
    lambda_ = std::exp(-1.0 / (opt_.tau_fit_s * fs));
    update_every_ = std::max(1L, std::lround(0.25 * fs));
    reset();
  }

  bool active() const { return !lines_.empty(); }
  // Any line was given a nonzero starting amplitude (seed_main etc. --
  // see VibrationCancellerOptions' own doc comment). Used by the caller
  // (CbkProc::imuCallback()) to start applying the a priori subtraction
  // from sample 0 when seeded, instead of waiting for calib_done: a
  // seeded beta_ is already a real estimate, not the "cold, unconverged"
  // state apply=false exists to avoid subtracting.
  bool seeded() const
  {
    for (const auto* v : {&opt_.seed_main, &opt_.seed_sideband_lo, &opt_.seed_sideband_hi,
                          &opt_.seed_harmonic2, &opt_.seed_extra0, &opt_.seed_extra1})
      if (!v->empty()) return true;
    return false;
  }
  const std::vector<Line>& lines() const { return lines_; }
  double frequencyHz(size_t l) const { return lines_[l].f * opt_.sample_rate_hz; }
  // fitted amplitude of line l in channel c (0-2 acc [m/s^2], 3-5 gyro [rad/s])
  double amplitude(size_t l, int c) const { return std::abs(amp(l, c) / hp_resp_[l]); }
  long resets() const { return resets_; }
  long samples() const { return n_; }

  // In place: acc [m/s^2], gyro [rad/s] of the sample at time t [s]. Samples must come in time order.
  // apply = false: update the fit only, leave acc / gyro unchanged.
  void process(double t, Eigen::Vector3d& acc, Eigen::Vector3d& gyro, bool apply = true)
  {
    if (lines_.empty()) return;
    double x[6] = {acc(0), acc(1), acc(2), gyro(0), gyro(1), gyro(2)};
    long step = 1;
    if (have_prev_)
    {
      const double dt = t - t_prev_;
      if (!(dt > 0.0) || dt > opt_.max_gap_s)
      {
        reset();
        ++resets_;
      }
      else
        step = std::max(1L, std::lround(dt * opt_.sample_rate_hz));
    }
    if (!have_prev_)
    {
      for (int c = 0; c < 6; ++c) { m1_[c] = x[c]; m2_[c] = 0.0; }
      have_prev_ = true;
      step = 0;
    }
    t_prev_ = t;
    n_ += step;

    const size_t L = lines_.size();
    for (size_t l = 0; l < L; ++l)
    {
      th_[l] = std::fmod(th_[l] + 2.0 * M_PI * lines_[l].f * static_cast<double>(step), 2.0 * M_PI);
      const bool on = n_ >= lines_[l].start;
      r_(2 * l) = on ? std::cos(th_[l]) : 0.0;
      r_(2 * l + 1) = on ? std::sin(th_[l]) : 0.0;
      if (!on)
      {
        // not yet in the fit: keep its block at the prior, uncorrelated
        P_.row(2 * l).setZero(); P_.row(2 * l + 1).setZero();
        P_.col(2 * l).setZero(); P_.col(2 * l + 1).setZero();
        P_(2 * l, 2 * l) = P_(2 * l + 1, 2 * l + 1) = kP0;
      }
    }

    const Eigen::VectorXd Pr = P_ * r_;
    const Eigen::VectorXd g = Pr / (lambda_ + r_.dot(Pr));
    for (int c = 0; c < 6; ++c)
    {
      m1_[c] += a_hp_ * (x[c] - m1_[c]);
      const double h1 = x[c] - m1_[c];
      m2_[c] += a_hp_ * (h1 - m2_[c]);
      const double hp = h1 - m2_[c];
      // a priori: the prediction from the earlier samples
      double vib = 0.0;
      for (size_t l = 0; l < L; ++l)
        if (n_ >= lines_[l].start) vib += std::real(amp(l, c) / hp_resp_[l] * std::polar(1.0, th_[l]));
      const double e = hp - r_.dot(beta_.col(c));
      beta_.col(c) += g * e;
      x[c] -= vib;
    }
    P_ = symmetrized((P_ - g * Pr.transpose()) / lambda_);
    if (!P_.allFinite() || !beta_.allFinite())
    {
      reset();
      ++resets_;
    }

    if (n_ >= next_update_)
    {
      if (n_ > std::lround(2.0 * opt_.tau_fit_s * opt_.sample_rate_hz)) updateFrequencies();
      snap_ = beta_;
      next_update_ = n_ + update_every_;
    }
    if (!apply) return;
    acc = Eigen::Vector3d(x[0], x[1], x[2]);
    gyro = Eigen::Vector3d(x[3], x[4], x[5]);
  }

private:
  static constexpr double kP0 = 1e2;

  std::complex<double> amp(size_t l, int c) const
  {
    // fit b_c cos + b_s sin = Re((b_c - i b_s) e^{i th})
    return {beta_(2 * l, c), -beta_(2 * l + 1, c)};
  }
  std::complex<double> snapAmp(size_t l, int c) const { return {snap_(2 * l, c), -snap_(2 * l + 1, c)}; }

  // 2nd-order high pass response at f (cycles/sample): per stage 1 - a / (1 - (1 - a) e^{-iw})
  std::complex<double> highpassResponse(double f) const
  {
    const std::complex<double> h = 1.0 - a_hp_ / (1.0 - (1.0 - a_hp_) * std::polar(1.0, -2.0 * M_PI * f));
    return h * h;
  }

  void updateFrequencies()
  {
    const double M = static_cast<double>(update_every_);
    for (size_t l = 0; l < lines_.size(); ++l)
    {
      Line& ln = lines_[l];
      if (ln.kind != Kind::MAIN && ln.kind != Kind::EXTRA) continue;
      std::complex<double> s = 0.0;
      for (int c = 0; c < 6; ++c) s += amp(l, c) * std::conj(snapAmp(l, c));
      if (!(std::abs(s) > 0.0)) continue;
      const double tau = ln.kind == Kind::MAIN ? opt_.tau_freq_s : opt_.tau_freq_extra_s;
      ln.f += (M / (tau * opt_.sample_rate_hz)) * std::arg(s) / (2.0 * M_PI * M);
      if (ln.kind == Kind::EXTRA)
      {
        const double dev = opt_.max_freq_dev_hz / opt_.sample_rate_hz;
        ln.f = std::min(ln.f0 + dev, std::max(ln.f0 - dev, ln.f));
      }
    }
    const double fm = lines_[0].f;
    for (Line& ln : lines_)
    {
      if (ln.kind == Kind::SIDEBAND) ln.f = fm + ln.param;
      if (ln.kind == Kind::HARMONIC) ln.f = ln.param * fm;
    }
    for (size_t l = 0; l < lines_.size(); ++l) hp_resp_[l] = highpassResponse(lines_[l].f);
  }

  void reset()
  {
    const size_t R = 2 * lines_.size();
    P_ = kP0 * Eigen::MatrixXd::Identity(R, R);
    beta_ = Eigen::MatrixXd::Zero(R, 6);
    snap_ = beta_;
    r_ = Eigen::VectorXd::Zero(R);
    th_.assign(lines_.size(), 0.0);
    hp_resp_.resize(lines_.size());
    int extra_seen = 0;
    for (size_t l = 0; l < lines_.size(); ++l)
    {
      lines_[l].f = lines_[l].kind == Kind::SIDEBAND ? lines_[0].f + lines_[l].param
                    : lines_[l].kind == Kind::HARMONIC ? lines_[l].param * lines_[0].f
                                                       : lines_[l].f;
      hp_resp_[l] = highpassResponse(lines_[l].f);

      const std::vector<double>* seed = nullptr;
      if (lines_[l].kind == Kind::MAIN) seed = &opt_.seed_main;
      else if (lines_[l].kind == Kind::SIDEBAND)
        seed = lines_[l].param < 0.0 ? &opt_.seed_sideband_lo : &opt_.seed_sideband_hi;
      else if (lines_[l].kind == Kind::HARMONIC && lines_[l].param == 2.0)
        seed = &opt_.seed_harmonic2;
      else if (lines_[l].kind == Kind::EXTRA)
      {
        seed = extra_seen == 0 ? &opt_.seed_extra0 : (extra_seen == 1 ? &opt_.seed_extra1 : nullptr);
        ++extra_seen;
      }
      if (seed && seed->size() == 6)
        for (int c = 0; c < 6; ++c)
          beta_(2 * l, c) = (*seed)[static_cast<size_t>(c)] * std::abs(hp_resp_[l]);
    }
    snap_ = beta_;
    have_prev_ = false;
    n_ = 0;
    next_update_ = update_every_;
    for (Line& ln : lines_)
      if (ln.kind == Kind::SIDEBAND) ln.start = std::lround(opt_.sideband_delay_s * opt_.sample_rate_hz);
  }

  VibrationCancellerOptions opt_;
  std::vector<Line> lines_;
  double a_hp_ = 0.0, lambda_ = 1.0;
  long update_every_ = 50;
  Eigen::MatrixXd P_, beta_, snap_;
  Eigen::VectorXd r_;
  std::vector<double> th_;
  std::vector<std::complex<double>> hp_resp_;
  double m1_[6] = {0, 0, 0, 0, 0, 0}, m2_[6] = {0, 0, 0, 0, 0, 0};
  double t_prev_ = 0.0;
  bool have_prev_ = false;
  long n_ = 0, next_update_ = 50, resets_ = 0;
};

// Batch removal of the lines from a stationary window (the calibration's), on the raw samples: a joint
// least-squares fit of a constant plus cos / sin of the main line, its harmonics and the extra lines (the
// sidebands only if the window is at least sideband_delay_s long), on the sample clock; the fitted
// sinusoids are subtracted, so the window mean becomes the fitted constant: the mean without the bias of
// partial vibration cycles (up to amplitude / (pi f T): 5 mm/s^2 for 0.23 m/s^2 over 0.5 s). Returns the number of sinusoid parameters p: the variance
// of the result over N samples is low by (N - 1 - p) / (N - 1) for white noise, so a variance computed
// from it is multiplied by (N - 1) / (N - 1 - p). Returns 0 (samples unchanged) if inactive or too short.
// Container: random access over samples with .t, .acc, .gyro (std::vector / std::deque<ImuSample>).
template <typename Container>
int removeLinesBatch(const VibrationCancellerOptions& opt, Container& s)
{
  if (!opt.enable || !(opt.main_hz > 0.0) || !(opt.sample_rate_hz > 0.0) || s.size() < 8) return 0;
  const double fs = opt.sample_rate_hz;
  const double span = s.back().t - s.front().t;
  std::vector<double> f{opt.main_hz};
  if (opt.sideband_offset_hz > 0.0 && span >= opt.sideband_delay_s)
  {
    f.push_back(opt.main_hz - opt.sideband_offset_hz);
    f.push_back(opt.main_hz + opt.sideband_offset_hz);
  }
  for (int h = 2; h <= opt.harmonics; ++h)
    if (h * opt.main_hz < 0.5 * fs) f.push_back(h * opt.main_hz);
  for (double e : opt.extra_hz)
    if (e > 0.0 && e < 0.5 * fs) f.push_back(e);
  const int p = 2 * static_cast<int>(f.size());
  const int N = static_cast<int>(s.size());
  if (N < 2 * (p + 1)) return 0;
  // sample index on the IMU clock (the stamps jitter)
  std::vector<double> n(N, 0.0);
  for (int k = 1; k < N; ++k)
    n[k] = n[k - 1] + std::max(1.0, std::round((s[k].t - s[k - 1].t) * fs));
  Eigen::MatrixXd X(N, p + 1), Y(N, 6);
  for (int k = 0; k < N; ++k)
  {
    X(k, 0) = 1.0;
    for (size_t l = 0; l < f.size(); ++l)
    {
      X(k, 1 + 2 * l) = std::cos(2.0 * M_PI * f[l] / fs * n[k]);
      X(k, 2 + 2 * l) = std::sin(2.0 * M_PI * f[l] / fs * n[k]);
    }
    Y.row(k) << s[k].acc.transpose(), s[k].gyro.transpose();
  }
  const Eigen::MatrixXd B = X.colPivHouseholderQr().solve(Y);
  if (!B.allFinite()) return 0;
  const Eigen::MatrixXd fit = X.rightCols(p) * B.bottomRows(p);   // the sinusoids only
  for (int k = 0; k < N; ++k)
  {
    s[k].acc -= fit.row(k).head<3>().transpose();
    s[k].gyro -= fit.row(k).tail<3>().transpose();
  }
  return p;
}

}  // namespace livo_recon
