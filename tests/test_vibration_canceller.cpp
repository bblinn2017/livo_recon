// Prism vibration canceller (include/livo_recon/utils/imu/vibration_canceller.h). Eigen + std only; own check macro
// (does not depend on NDEBUG). Synthetic IMU at the mid360 sample clock with jittered stamps, the mid360.yaml lines.
#include "livo_recon/utils/imu/vibration_canceller.h"

#include <cstdio>
#include <deque>
#include <random>
#include <vector>

using namespace livo_recon;

static int fails = 0;
#define CK(...) do { if (!(__VA_ARGS__)) { std::printf("FAIL line %d: %s\n", __LINE__, #__VA_ARGS__); ++fails; } } while (0)

struct Sample
{
  double t;
  Eigen::Vector3d acc, gyro;
};

static VibrationCancellerOptions mid360()
{
  VibrationCancellerOptions o;
  o.enable = true;
  o.sample_rate_hz = 199.998;
  o.main_hz = 28.025;
  o.sideband_offset_hz = 0.790;
  o.harmonics = 2;
  o.extra_hz = {13.400, 18.124};
  return o;
}

// noise (white, per sample), optional vibration like the mid360's (main line with 0.79 Hz modulation, harmonic,
// extra lines; random phases) and optional slow motion; gravity on z. vib holds the vibration alone.
static std::vector<Sample> synth(double seconds, bool with_vib, bool with_motion, unsigned seed,
                                 std::vector<Sample>* vib = nullptr, std::vector<Sample>* noise = nullptr)
{
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::uniform_real_distribution<double> ud(0.0, 2.0 * M_PI), jit(-0.9e-3, 0.9e-3);
  const double fs = 199.998, sa = 0.0141, sg = 1.3e-3;   // the vibration-free mid360 floors
  double ph[6][5];
  for (auto& row : ph)
    for (double& p : row) p = ud(rng);
  const double A[6] = {0.044, 0.043, 0.029, 1.9e-3, 6.4e-4, 1.0e-3};
  std::vector<Sample> s;
  const long N = static_cast<long>(seconds * fs);
  for (long n = 0; n < N; ++n)
  {
    const double tn = n / fs;
    Sample x{1000.0 + tn + jit(rng), Eigen::Vector3d(0, 0, 9.81), Eigen::Vector3d::Zero()};
    Sample v{x.t, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()}, w = v;
    for (int c = 0; c < 6; ++c)
    {
      const double am = 1.0 + 0.35 * std::cos(2 * M_PI * 0.790 * tn + ph[c][1]);
      double vb = A[c] * am * std::cos(2 * M_PI * 28.025 * tn + ph[c][0]) +
                  0.25 * A[c] * std::cos(2 * M_PI * 56.05 * tn + ph[c][2]) +
                  0.15 * A[c] * std::cos(2 * M_PI * 13.400 * tn + ph[c][3]) +
                  0.15 * A[c] * std::cos(2 * M_PI * 18.124 * tn + ph[c][4]);
      if (!with_vib) vb = 0.0;
      const double nz = (c < 3 ? sa : sg) * nd(rng);
      const double mo = with_motion ? (c < 3 ? 2.0 : 0.8) * std::sin(2 * M_PI * 1.0 * tn + c) : 0.0;
      (c < 3 ? x.acc(c) : x.gyro(c - 3)) += vb + nz + mo;
      (c < 3 ? v.acc(c) : v.gyro(c - 3)) = vb;
      (c < 3 ? w.acc(c) : w.gyro(c - 3)) = nz;
    }
    s.push_back(x);
    if (vib) vib->push_back(v);
    if (noise) noise->push_back(w);
  }
  return s;
}

static double var(const std::vector<double>& x)
{
  double m = 0, v = 0;
  for (double a : x) m += a / x.size();
  for (double a : x) v += (a - m) * (a - m) / (x.size() - 1);
  return v;
}

static std::vector<Sample> run(const std::vector<Sample>& in, VibrationCanceller& vc, double apply_from = -1.0)
{
  std::vector<Sample> out = in;
  for (Sample& s : out) vc.process(s.t, s.acc, s.gyro, s.t >= apply_from);
  return out;
}

int main()
{
  const double t_warm = 1000.0 + 10.0;   // after the fit and the frequency loop have settled

  // 1. no vibration: the a priori prediction never absorbs the current sample's noise -> nothing is removed
  {
    std::vector<Sample> noise;
    auto in = synth(60, false, false, 1, nullptr, &noise);
    VibrationCanceller vc(mid360());
    auto out = run(in, vc);
    for (int c = 0; c < 6; ++c)
    {
      std::vector<double> a, b;
      for (size_t k = 0; k < in.size(); ++k)
        if (in[k].t > t_warm)
        {
          a.push_back(c < 3 ? noise[k].acc(c) : noise[k].gyro(c - 3));
          b.push_back(c < 3 ? out[k].acc(c) - (c == 2 ? 9.81 : 0.0) : out[k].gyro(c - 3));
        }
      const double r = var(b) / var(a);
      std::printf("no vibration, channel %d: var(out) / var(noise) = %.4f\n", c, r);
      CK(r > 0.995 && r < 1.06);
    }
  }

  // 2.-4. vibration + noise + slow motion: lines removed, noise and motion and mean kept
  {
    std::vector<Sample> vib, noise;
    auto in = synth(60, true, true, 2, &vib, &noise);
    VibrationCanceller vc(mid360());
    auto out = run(in, vc);
    for (int c = 0; c < 6; ++c)
    {
      std::vector<double> left, injected, slow_in, slow_out;
      double mean_in = 0, mean_out = 0;
      size_t n = 0;
      for (size_t k = 0; k < in.size(); ++k)
      {
        if (in[k].t <= t_warm) continue;
        const double xi = c < 3 ? in[k].acc(c) : in[k].gyro(c - 3);
        const double xo = c < 3 ? out[k].acc(c) : out[k].gyro(c - 3);
        const double v = c < 3 ? vib[k].acc(c) : vib[k].gyro(c - 3);
        const double w = c < 3 ? noise[k].acc(c) : noise[k].gyro(c - 3);
        left.push_back(xo - (xi - v));   // what is left of the vibration (+ what was wrongly removed)
        injected.push_back(v);
        (void)w;
        mean_in += xi;
        mean_out += xo;
        ++n;
      }
      const double r = var(left) / var(injected);
      std::printf("vibration, channel %d: left %.4f of the injected vibration variance, mean shift %.2e\n", c, r,
                  (mean_out - mean_in) / n);
      CK(r < 0.10);
      CK(std::abs(mean_out - mean_in) / n < 2e-4);
    }
    // the 1 Hz motion passes: its amplitude in the output (projection over whole cycles) is unchanged
    for (int c = 0; c < 6; ++c)
    {
      double si = 0, so = 0;
      for (size_t k = 0; k < in.size(); ++k)
      {
        if (in[k].t <= t_warm) continue;
        const double tn = in[k].t - 1000.0, b = std::sin(2 * M_PI * 1.0 * tn + c);
        si += b * (c < 3 ? in[k].acc(c) : in[k].gyro(c - 3));
        so += b * (c < 3 ? out[k].acc(c) : out[k].gyro(c - 3));
      }
      std::printf("motion, channel %d: 1 Hz amplitude out / in = %.5f\n", c, so / si);
      CK(std::abs(so / si - 1.0) < 2e-3);
    }
  }

  // 5. apply = false passes the samples through unchanged but learns: switched on at 5 s it is converged
  {
    std::vector<Sample> vib;
    auto in = synth(20, true, false, 3, &vib);
    VibrationCanceller vc(mid360());
    auto out = run(in, vc, 1005.0);
    bool same = true;
    std::vector<double> left, injected;
    for (size_t k = 0; k < in.size(); ++k)
    {
      if (in[k].t < 1005.0) same = same && in[k].acc == out[k].acc && in[k].gyro == out[k].gyro;
      else if (in[k].t < 1006.0)
      {
        left.push_back(out[k].acc(2) - (in[k].acc(2) - vib[k].acc(2)));
        injected.push_back(vib[k].acc(2));
      }
    }
    std::printf("learn-only until 5 s: passed through %s; first second applied: left %.4f of the vibration\n",
                same ? "unchanged" : "CHANGED", var(left) / var(injected));
    CK(same);
    CK(var(left) / var(injected) < 0.15);
  }

  // 6. a gap longer than max_gap_s restarts it
  {
    auto in = synth(6, true, false, 4);
    for (size_t k = in.size() / 2; k < in.size(); ++k) in[k].t += 2.0;
    VibrationCanceller vc(mid360());
    run(in, vc);
    std::printf("gap: %ld reset(s)\n", vc.resets());
    CK(vc.resets() == 1);
  }

  // 7. removeLinesBatch on a 0.5 s calibration window (std::deque, like CalibProc::calib_imu_samples): the
  //    dof-corrected variance matches the noise alone; the mean is closer to the truth than the raw window's
  {
    double r_acc = 0, r_gyr = 0, err_in = 0, err_out = 0;
    const int trials = 40;
    for (int tr = 0; tr < trials; ++tr)
    {
      std::vector<Sample> noise;
      auto v = synth(0.49, true, false, 100 + tr, nullptr, &noise);
      std::deque<Sample> s(v.begin(), v.end());
      Eigen::Vector3d mean_in = Eigen::Vector3d::Zero(), mean_out = Eigen::Vector3d::Zero(),
                      truth(0, 0, 9.81);   // + the noise's own window mean
      for (size_t k = 0; k < s.size(); ++k) { mean_in += s[k].acc / s.size(); truth += noise[k].acc / s.size(); }
      const int p = removeLinesBatch(mid360(), s);
      CK(p == 8);   // main, harmonic, two extra lines; no sidebands in 0.5 s
      for (const Sample& x : s) mean_out += x.acc / s.size();
      const double scale = (s.size() - 1.0) / (s.size() - 1.0 - p);
      std::vector<double> a, b, g, h;
      for (size_t k = 0; k < s.size(); ++k)
      {
        a.push_back(s[k].acc(2)); b.push_back(noise[k].acc(2));
        g.push_back(s[k].gyro(0)); h.push_back(noise[k].gyro(0));
      }
      r_acc += scale * var(a) / var(b) / trials;
      r_gyr += scale * var(g) / var(h) / trials;
      err_in += (mean_in - truth).squaredNorm() / trials;
      err_out += (mean_out - truth).squaredNorm() / trials;
    }
    std::printf("batch, 0.5 s window (mean of 40): dof-corrected var / noise var: acc z %.3f, gyro x %.3f; "
                "rms mean error raw %.2e -> %.2e m/s^2\n", r_acc, r_gyr, std::sqrt(err_in), std::sqrt(err_out));
    // slightly above 1: the 0.79 Hz modulation's sidebands cannot be fitted in 0.5 s, so part of it stays
    // (conservative: never below the noise; CalibProc does not use this variance for the floor anyway)
    CK(r_acc > 0.97 && r_acc < 1.10);
    CK(r_gyr > 0.97 && r_gyr < 1.10);
    CK(err_out < err_in);
  }

  // disabled: nothing configured, samples untouched
  {
    VibrationCancellerOptions o = mid360();
    o.enable = false;
    VibrationCanceller vc(o);
    Eigen::Vector3d a(1, 2, 3), g(4, 5, 6);
    vc.process(0.0, a, g);
    CK(!vc.active() && a == Eigen::Vector3d(1, 2, 3) && g == Eigen::Vector3d(4, 5, 6));
  }

  std::printf(fails ? "%d FAILED\n" : "all passed\n", fails);
  return fails ? 1 : 0;
}
