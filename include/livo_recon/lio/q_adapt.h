#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// ============================================================================
// R72 (F-401): process-noise alternatives to the motion-dependent Q, as small ROS-free pure functions so each can be tested
// for algebraic identities in isolation (tests/test_q_adapt.cpp).  Two mechanisms live here (R74 F-427 removed the local-polynomial energy source and the additive WNOA disturbance):
//
//  (1) innovationStats()   per-scan LiDAR innovation statistics against the propagated prior (used by imu/process_noise/gain/mode =
//                          innovation_scale and nis_gate).
//  (2) QGain               a bounded scalar gain on the IMU step variance, updated once per scan from (1).  The gain is applied
//                          by ImuProc::propagate on the NEXT scan's steps (a one-scan lag, by construction).
// Every mechanism is OFF by default and off is bit-identical (the integration never evaluates any of this when off).
// Numeric defaults of the options below are PROVISIONAL (rule 1.7): they are starting values for the R72 experiment arms, not
// recommended settings; the manifest sets every one of them explicitly.
// ============================================================================

namespace livo_recon
{

// ---------------------------------------------------------------- (1) innovation statistics
struct InnovationRow
{
  double r = 0.0;        // plane residual (m), at the propagated prior state (GN iteration 0)
  double sigma2 = 1.0;   // residual measurement variance (m^2), as used by the accumulator
  Eigen::Matrix<double, 6, 1> h = Eigen::Matrix<double, 6, 1>::Zero();   // Jacobian row over (rotation 3, position 3)
};

struct InnovationStats
{
  int n = 0;
  double sum_r2 = 0.0;       // sum r^2
  double sum_sigma2 = 0.0;   // sum sigma^2
  double sum_hph = 0.0;      // sum h^T P h  (what the prior contributes to the innovation variance)
  double nis = 1.0;          // mean of r^2 / (sigma^2 + h^T P h); equals 1 for a consistent model
  double rho = 1.0;          // max(0, sum (r^2 - sigma^2)) / sum h^T P h ; equals 1 for a consistent model
  double lag1 = 0.0;         // lag-1 autocorrelation of e_i = r_i / sqrt(sigma_i^2 + h_i^T P h_i), in row order
  bool valid = false;        // false when there are no rows or sum_hph <= 0
};

inline InnovationStats innovationStats(const std::vector<InnovationRow>& rows, const Eigen::Matrix<double, 6, 6>& P6)
{
  InnovationStats s;
  s.n = static_cast<int>(rows.size());
  if (rows.empty()) return s;
  double nis_sum = 0.0, e2 = 0.0, e_lag = 0.0, e_prev = 0.0;
  bool have_prev = false;
  for (const InnovationRow& row : rows)
  {
    const double hph = row.h.dot(P6 * row.h);
    const double denom = row.sigma2 + hph;
    const double r2 = row.r * row.r;
    s.sum_r2 += r2;
    s.sum_sigma2 += row.sigma2;
    s.sum_hph += hph;
    nis_sum += r2 / denom;
    const double e = row.r / std::sqrt(denom);
    e2 += e * e;
    if (have_prev) e_lag += e * e_prev;
    e_prev = e;
    have_prev = true;
  }
  s.nis = nis_sum / static_cast<double>(s.n);
  s.lag1 = e2 > 0.0 ? e_lag / e2 : 0.0;
  if (s.sum_hph > 0.0)
  {
    s.rho = std::max(0.0, s.sum_r2 - s.sum_sigma2) / s.sum_hph;
    s.valid = true;
  }
  return s;
}

// ---------------------------------------------------------------- (2) bounded scalar gain
struct QGainOptions
{
  std::string mode = "off";   // "off" | "innovation_scale" | "nis_gate" | "nis_smooth"
  std::string apply = "both"; // "both" | "acc" | "gyr"   which IMU variance the gain multiplies
  double forgetting = 0.9;    // innovation_scale: g <- g * clamp(rho)^(1 - forgetting)
  double min_gain = 1.0;      // gain bounds (min 1 = inflate only)
  double max_gain = 100.0;
  double rho_clamp = 4.0;     // innovation_scale: rho is clamped to [1/rho_clamp, rho_clamp] before use
  double nis_hi = 1.5;        // nis_gate: inflate when NIS exceeds this
  double nis_lo = 0.0;        // R80 (F-476): nis_gate: deflate when NIS is below this (0 = never deflate: the R72 one-sided gate, bit for bit)
  double deflate = 1.0;       // R80 (F-476): nis_gate: gain multiplier on a deflating scan (< 1); the gain is then bounded below by min_gain (< 1 needed)
  double inflate = 1.25;      // nis_gate: gain multiplier on an inflating scan
  double decay = 0.95;        // nis_gate: g <- 1 + decay (g - 1) on a non-inflating scan
  double whiteness_max = 0.0; // nis_gate: when > 0, scans with |lag1| above this are NOT inflated (map-error signature)
  double nis_slope = 0.1;     // R81 (F-478): nis_smooth: ln g <- decay * ln g + nis_slope * clip(ln NIS, -nis_clip, +nis_clip); steady state g = NIS^(nis_slope / (1 - decay))
  double nis_clip = 0.0;      // R81 (F-478): nis_smooth: clip of ln NIS (> 0, no default: the value is Bryce's, see claude/r81-designs-pending.md)
  bool on() const { return mode != "off"; }
};

class QGain
{
public:
  void configure(const QGainOptions& o) { o_ = o; g_ = 1.0; }
  double gain() const { return g_; }
  double gainAcc() const { return o_.apply == "gyr" ? 1.0 : g_; }
  double gainGyr() const { return o_.apply == "acc" ? 1.0 : g_; }

  // One scan. Returns the new gain. An invalid statistic leaves the gain unchanged.
  double update(const InnovationStats& s)
  {
    ++scans_;
    if (!s.valid) return g_;
    last_ = g_;
    if (o_.mode == "innovation_scale")
    {
      const double rho = std::min(std::max(s.rho, 1.0 / o_.rho_clamp), o_.rho_clamp);
      g_ = g_ * std::pow(rho, 1.0 - o_.forgetting);
    }
    else if (o_.mode == "nis_smooth")
    {
      // R81 (F-478): the smooth version of the NIS gate. No threshold: every scan moves ln g by nis_slope * clip(ln NIS, +-nis_clip) after the
      // decay. NIS = 1 leaves ln g decaying toward 0; the gain is then bounded by [min_gain, max_gain] as for the other modes.
      const double ln_nis = std::min(std::max(std::log(std::max(s.nis, 1e-300)), -o_.nis_clip), o_.nis_clip);
      const double ln_g = o_.decay * std::log(g_) + o_.nis_slope * ln_nis;
      g_ = std::exp(ln_g);
      if (g_ > last_) ++inflating_;
      else if (g_ < last_) ++deflating_;
    }
    else if (o_.mode == "nis_gate")
    {
      const bool white_ok = !(o_.whiteness_max > 0.0) || std::abs(s.lag1) <= o_.whiteness_max;
      if (s.nis > o_.nis_hi && white_ok)
      {
        g_ = g_ * o_.inflate;
        ++inflating_;
      }
      else if (o_.nis_lo > 0.0 && s.nis < o_.nis_lo)
      {
        g_ = g_ * o_.deflate;   // R80 (F-476): the filter is under-confident on this scan (NIS below nis_lo): shrink the step variance
        ++deflating_;
      }
      else
      {
        if (s.nis > o_.nis_hi) ++whiteness_blocked_;
        g_ = 1.0 + o_.decay * (g_ - 1.0);
      }
    }
    g_ = std::min(std::max(g_, o_.min_gain), o_.max_gain);
    if (g_ != last_) ++changed_;
    max_seen_ = std::max(max_seen_, g_);
    min_seen_ = std::min(min_seen_, g_);
    return g_;
  }

  long scans() const { return scans_; }
  long changed() const { return changed_; }
  long inflating() const { return inflating_; }
  long deflating() const { return deflating_; }
  double minSeen() const { return min_seen_; }
  long whitenessBlocked() const { return whiteness_blocked_; }
  double maxSeen() const { return max_seen_; }

private:
  QGainOptions o_;
  double g_ = 1.0, last_ = 1.0, max_seen_ = 1.0, min_seen_ = 1.0;
  long scans_ = 0, changed_ = 0, inflating_ = 0, deflating_ = 0, whiteness_blocked_ = 0;
};

}  // namespace livo_recon
