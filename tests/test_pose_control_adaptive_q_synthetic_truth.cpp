// Adaptive-Q validation against KNOWN synthetic process-noise regimes.
// Exercises AdaptiveQ (adaptive_q.h/.cpp) directly against
// reduceImuResidualSamples()-reduced statistics -- the same reduction
// pose_control's own residual computation feeds it -- with residual
// samples synthesized from a KNOWN true noise process, not from a live
// estimator run.
#include "livo_recon/lio/adaptive_q.h"
#include "livo_recon/lio/spline.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace livo_recon;

namespace
{
int failures = 0;
void check(bool ok, const char* name, double value = 0.0, double tol = 0.0)
{
  std::printf("  [%s] %-90s %.6e tol %.6e\n", ok ? "PASS" : "FAIL", name, value, tol);
  if (!ok) ++failures;
}

SplineImuResidualStats syntheticWhiteResidualStats(double sigma_acc, double sigma_gyr, int n, std::mt19937& rng)
{
  std::normal_distribution<double> na(0.0, sigma_acc), ng(0.0, sigma_gyr);
  std::vector<V3D> ra(n), rw(n);
  for (int i = 0; i < n; ++i) { ra[i] = V3D(na(rng), na(rng), na(rng)); rw[i] = V3D(ng(rng), ng(rng), ng(rng)); }
  return reduceImuResidualSamples(ra, rw);
}

// Same white-noise floor PLUS a slowly-varying (low-frequency, i.e.
// strongly autocorrelated across the window) systematic component,
// standing in for genuine unmodeled motion / model error at CONSTANT true
// sensor noise -- residual magnitude grows without any change in the
// physical process noise itself.
SplineImuResidualStats syntheticModelErrorResidualStats(double sigma_acc, double sigma_gyr, double model_error_amp,
                                                        int n, std::mt19937& rng)
{
  std::normal_distribution<double> na(0.0, sigma_acc), ng(0.0, sigma_gyr);
  std::vector<V3D> ra(n), rw(n);
  for (int i = 0; i < n; ++i) {
    const double phase = M_PI * i / std::max(1, n - 1);   // half-cycle of a sine across the window -> strongly correlated
    const double drift = model_error_amp * std::sin(phase);
    ra[i] = V3D(na(rng) + drift, na(rng) + drift, na(rng) + drift);
    rw[i] = V3D(ng(rng), ng(rng), ng(rng));
  }
  return reduceImuResidualSamples(ra, rw);
}

struct Record { int k; double q_true; double q_used; double q_estimated_next; double measured; std::string status; };

std::vector<Record> runRegimeSequence(double q_low, double q_high, int n_per_regime, int n_samples_per_scan,
                                      std::mt19937& rng, double beta, double ema, int warmup)
{
  AdaptiveQ aq;
  AdaptiveQOptions opts;
  opts.enable = true; opts.beta_acc = beta; opts.beta_gyr = beta; opts.z_rate_limit = 1.0;  // no per-frame rate limit, isolates latency to ema alone
  opts.acf1_max = 1.0; opts.max_ratio = 100.0; opts.min_ratio = 0.01;
  opts.use_noise_floor = true; opts.warmup_frames = warmup; opts.ema = ema;
  aq.configure(opts);
  const double nom_acc = q_low, nom_gyr = 0.1 * q_low;
  aq.setNominal(nom_acc, nom_gyr);
  aq.setFloor(0.01 * q_low, 0.001 * q_low);   // far below either regime, never binds

  std::vector<Record> out;
  const int n_regimes = 3;   // low, high, low
  for (int r = 0; r < n_regimes; ++r) {
    const double q_true = (r == 1) ? q_high : q_low;
    for (int i = 0; i < n_per_regime; ++i) {
      const double q_used = aq.varAcc();   // value THIS scan's own prior would use (set by the PREVIOUS update() call)
      const auto st = syntheticWhiteResidualStats(std::sqrt(q_true), std::sqrt(0.1 * q_true), n_samples_per_scan, rng);
      aq.update(st);
      Record rec;
      rec.k = r * n_per_regime + i; rec.q_true = q_true; rec.q_used = q_used;
      rec.q_estimated_next = aq.varAcc(); rec.measured = aq.measuredAcc(); rec.status = aq.lastStatus();
      out.push_back(rec);
    }
  }
  return out;
}

void testAdaptiveQTracksKnownRegimeTransitions()
{
  std::mt19937 rng(2026);
  const double q_low = 0.02 * 0.02, q_high = 0.08 * 0.08;   // sigma 0.02 -> 0.08 -> 0.02
  const int n_per_regime = 60;
  const auto recs = runRegimeSequence(q_low, q_high, n_per_regime, 50, rng, /*beta=*/1.2, /*ema=*/0.7, /*warmup=*/5);

  for (const auto& r : recs)
    std::printf("  [k=%3d] q_true=%.6e q_used=%.6e q_est_next=%.6e measured=%.6e status=%s\n",
                r.k, r.q_true, r.q_used, r.q_estimated_next, r.measured, r.status.c_str());

  // Causality: q_used at scan k must equal q_estimated_next from scan k-1
  // exactly (the value read at the TOP of the loop, before update() runs).
  double worst_causality_violation = 0.0;
  for (size_t i = 1; i < recs.size(); ++i)
    worst_causality_violation = std::max(worst_causality_violation, std::abs(recs[i].q_used - recs[i - 1].q_estimated_next));
  check(worst_causality_violation < 1e-15,
        "causality: Q_estimated(k+1) never retroactively affects Q_used(k) (q_used(k)==q_estimated_next(k-1) exactly)",
        worst_causality_violation, 1e-15);

  // Steady-state error: average |q_estimated - q_true| over the LAST 15
  // scans of each regime (after any transient has settled).
  auto steadyStateRelErr = [&](int regime) {
    double sum = 0; int n = 0;
    const int start = regime * n_per_regime + (n_per_regime - 15);
    for (int i = start; i < start + 15; ++i) { sum += std::abs(recs[i].q_estimated_next - recs[i].q_true) / recs[i].q_true; ++n; }
    return sum / n;
  };
  const double ss_err_low1 = steadyStateRelErr(0), ss_err_high = steadyStateRelErr(1), ss_err_low2 = steadyStateRelErr(2);
  std::printf("  steady-state relative error: low1=%.4f high=%.4f low2=%.4f\n", ss_err_low1, ss_err_high, ss_err_low2);
  check(ss_err_low1 < 0.5, "steady-state error in the first low-Q regime is bounded", ss_err_low1, 0.5);
  check(ss_err_high < 0.5, "steady-state error in the high-Q regime is bounded", ss_err_high, 0.5);
  check(ss_err_low2 < 0.5, "steady-state error returning to the low-Q regime is bounded", ss_err_low2, 0.5);

  // Response latency: number of scans after the true transition until
  // q_estimated crosses 50% of the way from q_low toward q_high.
  const double halfway = 0.5 * (q_low + q_high);
  int latency = -1;
  for (int i = n_per_regime; i < 2 * n_per_regime; ++i)
    if (recs[i].q_estimated_next > halfway) { latency = i - n_per_regime; break; }
  std::printf("  response latency after low->high transition: %d scans (-1 = never crossed halfway)\n", latency);
  check(latency >= 0 && latency < n_per_regime,
        "adaptive Q crosses halfway toward the new true Q within the regime's own duration", latency, n_per_regime);

  // False adaptation during a CONSTANT-Q interval: coefficient of variation
  // of q_estimated over the middle 40 scans of the FIRST low-Q regime
  // (well before any transition, so any spread is pure false adaptation).
  {
    double mean = 0; int n = 0;
    for (int i = 10; i < 50; ++i) { mean += recs[i].q_estimated_next; ++n; }
    mean /= n;
    double var = 0;
    for (int i = 10; i < 50; ++i) { const double d = recs[i].q_estimated_next - mean; var += d * d; }
    var /= n;
    const double cv = std::sqrt(var) / mean;
    std::printf("  false-adaptation coefficient of variation during constant-Q interval: %.4f\n", cv);
    check(cv < 0.5, "adaptive Q does not wander excessively during a genuinely constant-Q interval", cv, 0.5);
  }
}

// The central Phase-8 distinction: adaptive Q reacts to residual MAGNITUDE,
// which model error inflates just as genuine process-noise increase does.
// True Q is held EXACTLY constant throughout; only a low-frequency
// (strongly autocorrelated) model-error component is injected mid-sequence.
void testAdaptiveQReactsToModelErrorAtConstantTrueQ()
{
  std::mt19937 rng(31415);
  const double q_true = 0.02 * 0.02;   // constant true process noise, NEVER changes
  const int n_per_phase = 60;
  AdaptiveQ aq;
  AdaptiveQOptions opts;
  opts.enable = true; opts.beta_acc = 1.2; opts.beta_gyr = 1.2; opts.z_rate_limit = 1.0;
  opts.acf1_max = 1.0; opts.use_noise_floor = true; opts.warmup_frames = 5; opts.ema = 0.7;
  aq.configure(opts);
  aq.setNominal(q_true, 0.1 * q_true);
  aq.setFloor(0.01 * q_true, 0.001 * q_true);

  std::vector<double> q_est, acf1;
  for (int phase = 0; phase < 3; ++phase) {
    // Phase 0: clean (no model error). Phase 1: model error injected
    // (unmodeled motion), TRUE Q UNCHANGED. Phase 2: model error removed.
    const double model_error_amp = (phase == 1) ? 8.0 * std::sqrt(q_true) : 0.0;
    for (int i = 0; i < n_per_phase; ++i) {
      const auto st = syntheticModelErrorResidualStats(std::sqrt(q_true), std::sqrt(0.1 * q_true), model_error_amp, 50, rng);
      aq.update(st);
      q_est.push_back(aq.varAcc());
      acf1.push_back(st.acf1_acc);
    }
  }

  double mean_clean = 0, mean_error = 0;
  for (int i = 10; i < n_per_phase; ++i) mean_clean += q_est[i];
  mean_clean /= (n_per_phase - 10);
  for (int i = n_per_phase + 10; i < 2 * n_per_phase; ++i) mean_error += q_est[i];
  mean_error /= (n_per_phase - 10);
  double mean_acf1_error = 0;
  for (int i = n_per_phase; i < 2 * n_per_phase; ++i) mean_acf1_error += acf1[i];
  mean_acf1_error /= n_per_phase;

  std::printf("  mean applied Q: clean-phase=%.6e  model-error-phase=%.6e  (true Q constant at %.6e throughout)\n",
              mean_clean, mean_error, q_true);
  std::printf("  mean acf1_acc during model-error phase: %.4f (high autocorrelation is the diagnostic signature "
              "that would, in principle, distinguish model error from genuine white process noise -- but "
              "acf1_max=1.0 in this codebase's own production default makes that gate INERT, see adaptive_q.h's "
              "own header comment: 'AdaptiveQ was a GYRO-ONLY estimator on every run this project has ever done' "
              "because the accelerometer channel's acf1 never cleared even a 0.35 bar on real data)\n", mean_acf1_error);

  check(mean_error > 2.0 * mean_clean,
        "CENTRAL FINDING: adaptive Q rises substantially in response to model error alone, with TRUE process "
        "noise held exactly constant -- confirming it measures residual MAGNITUDE (which model error inflates "
        "exactly as a genuine Q increase would), not physical process noise specifically",
        mean_error, 2.0 * mean_clean);
  check(mean_acf1_error > 0.3,
        "the model-error residual is genuinely non-white (high lag-1 autocorrelation) -- in principle "
        "distinguishable from genuine process noise by a whiteness gate, though this codebase's own acf1_max=1.0 "
        "production default does not currently act on this signal (see printed note)",
        mean_acf1_error, 0.3);
}

}  // namespace

int main()
{
  std::printf("Pose-control adaptive-Q synthetic known-truth validation suite\n");
  testAdaptiveQTracksKnownRegimeTransitions();
  testAdaptiveQReactsToModelErrorAtConstantTrueQ();
  std::printf("%d failure(s)\n", failures);
  return failures ? 1 : 0;
}
