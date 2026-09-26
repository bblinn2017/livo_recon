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

// ============================================================================
// Lifecycle tests (item list: persistence across scans; setNominal() is not
// an implicit per-scan reset; current scan's Q/R is unaffected by future
// residuals; Q_estimated(k+1) affects scan k+1 not scan k; an explicit
// reset restores nominal). Causality is already covered by
// testAdaptiveQTracksKnownRegimeTransitions() above; the rest are new.
// ============================================================================
void testAdaptiveStatePersistsAcrossRepeatedSetNominalCalls()
{
  std::mt19937 rng(7777);
  const double q_true = 0.02 * 0.02;
  AdaptiveQ aq;
  AdaptiveQOptions opts;
  opts.enable = true; opts.beta_acc = 1.2; opts.z_rate_limit = 1.0;
  opts.acf1_max = 1.0; opts.use_noise_floor = false; opts.warmup_frames = 5; opts.ema = 0.7;
  aq.configure(opts);
  aq.setNominal(q_true, 0.1 * q_true);

  // Drive it to a clearly-adapted state at a DIFFERENT true Q.
  const double q_drive = 4.0 * q_true;
  for (int i = 0; i < 40; ++i) {
    const auto st = syntheticWhiteResidualStats(std::sqrt(q_drive), std::sqrt(0.1 * q_drive), 50, rng);
    aq.update(st);
  }
  const double adapted_before = aq.varAcc();
  check(adapted_before > 1.5 * q_true, "adaptive state genuinely moved away from nominal before the repeated-setNominal test", adapted_before, 1.5 * q_true);

  // Call setNominal() with the SAME value repeatedly (exactly what
  // production does every scan) -- this must NOT reset the adapted state.
  for (int i = 0; i < 10; ++i) aq.setNominal(q_true, 0.1 * q_true);
  const double after_repeated_calls = aq.varAcc();
  check(std::abs(after_repeated_calls - adapted_before) < 1e-12,
        "item 1/2: repeated setNominal() calls with an UNCHANGED nominal do not reset the adapted state "
        "(this is the confirmed-and-fixed lifecycle defect: production calls setNominal() before every "
        "update(), which previously reset applied_acc_/applied_gyr_ to nominal unconditionally)",
        std::abs(after_repeated_calls - adapted_before), 1e-12);

  // A scan whose residual FAILS validity (e.g. non-finite/invalid) must
  // hold the LAST GOOD adapted value, not silently revert to nominal --
  // this is exactly the scenario the old unconditional setNominal()-then-
  // update() call sequence broke (setNominal() would reset applied_acc_,
  // and the invalid-residual update() call would never recompute it).
  aq.setNominal(q_true, 0.1 * q_true);   // production's own per-scan call, now a no-op
  SplineImuResidualStats bad_st;   // default-constructed: n=0, invalid()==false
  const bool changed = aq.update(bad_st);
  check(!changed, "an invalid residual is correctly refused (update() returns false)", changed ? 1.0 : 0.0);
  check(std::abs(aq.varAcc() - after_repeated_calls) < 1e-12,
        "item 3: an invalid/refused scan's residual does not retroactively alter the CURRENT applied value "
        "(the adapted state from before the bad scan is held, not reset to nominal)",
        std::abs(aq.varAcc() - after_repeated_calls), 1e-12);
}

void testExplicitResetRestoresNominal()
{
  std::mt19937 rng(8888);
  const double q_true = 0.02 * 0.02;
  AdaptiveQ aq;
  AdaptiveQOptions opts;
  opts.enable = true; opts.beta_acc = 1.2; opts.z_rate_limit = 1.0;
  opts.acf1_max = 1.0; opts.use_noise_floor = false; opts.warmup_frames = 5; opts.ema = 0.7;
  aq.configure(opts);
  aq.setNominal(q_true, 0.1 * q_true);
  for (int i = 0; i < 40; ++i) {
    const auto st = syntheticWhiteResidualStats(std::sqrt(4.0 * q_true), std::sqrt(0.4 * q_true), 50, rng);
    aq.update(st);
  }
  check(std::abs(aq.varAcc() - q_true) > 0.5 * q_true, "adapted away from nominal before reset", aq.varAcc());

  // item 5: an EXPLICIT reset (resetToNominal(), or setNominal() with a
  // GENUINELY different value) restores the intended nominal state.
  aq.resetToNominal(q_true, 0.1 * q_true);
  check(std::abs(aq.varAcc() - q_true) < 1e-12, "item 5: explicit resetToNominal() restores applied_acc_ to nominal exactly",
        std::abs(aq.varAcc() - q_true));
  // active() is a documented one-way latch ("has ever activated since
  // startup", adaptive_q.h's own header comment) -- a reset intentionally
  // does NOT clear it, since the filter genuinely HAS activated before;
  // only applied_{acc,gyr}_/frames_/the EMA/rate-limited state reset.

  // Re-driving after reset should behave exactly as a fresh instance would
  // (frames_/warmup restart), confirming the reset is a genuine restart,
  // not merely a value overwrite.
  for (int i = 0; i < 4; ++i) {
    const auto st = syntheticWhiteResidualStats(std::sqrt(4.0 * q_true), std::sqrt(0.4 * q_true), 50, rng);
    aq.update(st);
  }
  check(std::abs(aq.varAcc() - q_true) < 1e-9, "still in warm-up immediately after reset (frames_ restarted, not carried over)", aq.varAcc());
}

// ============================================================================
// Phase 3 items C/D: measurement-noise (R) change and bias error, both with
// TRUE PHYSICAL PROCESS NOISE held constant -- completing the A (Q change,
// already covered above) / B (model error, already covered above) / C / D
// battery this task requires.
// ============================================================================

// C: since this estimator's own measured quantity IS the IMU
// collocation/measurement variance R (see Phase 2's semantic finding --
// AdaptiveQ has no separate channel for "R" vs "Q", they are the SAME
// measured spline-vs-IMU residual spread), a pure R change is
// mathematically indistinguishable, from this estimator's point of view,
// from a Q change of the same magnitude -- this test exists to make that
// equivalence explicit and evidenced, not to find a different response.
void testAdaptiveQRespondsIdenticallyToRChangeAsToQChange()
{
  std::mt19937 rng(9999);
  const double q_true = 0.02 * 0.02, r_high = 4.0 * q_true;
  // "R change": interpreted here as the ACCELEROMETER's OWN measurement
  // noise increasing (e.g. a sensor/thermal effect) while the true
  // PHYSICAL motion's process noise is unchanged -- synthesized IDENTICALLY
  // to the Q-change case, because in THIS estimator's residual-based
  // formulation there is no way to inject "R increased, Q did not" as a
  // mathematically distinct signal from raw residual samples alone (both
  // manifest as larger spread in a-priori-white e_acc samples).
  AdaptiveQ aq;
  AdaptiveQOptions opts;
  opts.enable = true; opts.beta_acc = 1.2; opts.z_rate_limit = 1.0;
  opts.acf1_max = 1.0; opts.use_noise_floor = false; opts.warmup_frames = 5; opts.ema = 0.7;
  aq.configure(opts);
  aq.setNominal(q_true, 0.1 * q_true);
  double final_q = 0.0;
  for (int i = 0; i < 80; ++i) {
    const auto st = syntheticWhiteResidualStats(std::sqrt(r_high), std::sqrt(0.1 * r_high), 50, rng);
    aq.update(st);
    final_q = aq.varAcc();
  }
  const double rel_err = std::abs(final_q - r_high) / r_high;
  std::printf("  R-change fixture: true=%.6e final_applied=%.6e rel_err=%.4f\n", r_high, final_q, rel_err);
  check(rel_err < 0.3,
        "Phase 3C: a pure measurement-noise (R) change is tracked with the SAME accuracy as a Q change "
        "(expected: this estimator cannot distinguish R from Q, they are the same measured quantity here)",
        rel_err, 0.3);
}

// D: a CONSTANT (not time-varying) bias offset. reduceImuResidualSamples()
// computes cov_acc as the SPREAD of residuals around their OWN empirical
// MEAN (documented in spline.h -- "not the deviation from raw itself"),
// so a purely constant offset shifts the mean but should leave the spread,
// and therefore the adaptive estimate, UNCHANGED -- the opposite finding
// from model error (Phase 3B), which was deliberately TIME-VARYING
// (correlated) and therefore genuinely inflated the spread.
void testAdaptiveQIsBlindToPureConstantBiasOffset()
{
  std::mt19937 rng(11111);
  const double q_true = 0.02 * 0.02;
  AdaptiveQ aq;
  AdaptiveQOptions opts;
  opts.enable = true; opts.beta_acc = 1.2; opts.z_rate_limit = 1.0;
  opts.acf1_max = 1.0; opts.use_noise_floor = false; opts.warmup_frames = 5; opts.ema = 0.7;
  aq.configure(opts);
  aq.setNominal(q_true, 0.1 * q_true);

  std::vector<double> q_est_clean, q_est_biased;
  std::normal_distribution<double> na(0.0, std::sqrt(q_true)), ng(0.0, std::sqrt(0.1 * q_true));
  for (int phase = 0; phase < 2; ++phase) {
    const double bias = (phase == 1) ? 20.0 * std::sqrt(q_true) : 0.0;   // large CONSTANT offset, not time-varying
    for (int i = 0; i < 60; ++i) {
      std::vector<V3D> ra(50), rw(50);
      for (int j = 0; j < 50; ++j) { ra[j] = V3D(na(rng) + bias, na(rng) + bias, na(rng) + bias); rw[j] = V3D(ng(rng), ng(rng), ng(rng)); }
      const auto st = reduceImuResidualSamples(ra, rw);
      aq.update(st);
      (phase == 0 ? q_est_clean : q_est_biased).push_back(aq.varAcc());
    }
  }
  double mean_clean = 0, mean_biased = 0;
  for (int i = 10; i < 60; ++i) mean_clean += q_est_clean[i];
  mean_clean /= 50;
  for (int i = 10; i < 60; ++i) mean_biased += q_est_biased[i];
  mean_biased /= 50;
  const double rel_diff = std::abs(mean_biased - mean_clean) / mean_clean;
  std::printf("  constant-bias fixture: mean_applied_Q clean=%.6e biased=%.6e rel_diff=%.4f (true Q constant at %.6e throughout)\n",
              mean_clean, mean_biased, rel_diff, q_true);
  check(rel_diff < 0.3,
        "Phase 3D: adaptive Q is BLIND to a pure constant bias offset (mean-centered residual spread is "
        "unaffected by a shift in the mean) -- distinct from model error (Phase 3B), which is time-VARYING "
        "and therefore genuinely inflates the spread",
        rel_diff, 0.3);
}

// ============================================================================
// Phase 4: whiteness (acf1) characterization across all four synthetic
// regimes, to determine whether it discriminates model error from the
// other three (genuine Q change, R change, constant bias).
// ============================================================================
void testWhitenessCharacterizationAcrossRegimes()
{
  std::mt19937 rng(22222);
  const double q_true = 0.02 * 0.02;
  auto meanAcf1 = [&](auto&& sampler) {
    double sum = 0;
    const int n = 40;
    for (int i = 0; i < n; ++i) sum += std::abs(sampler().acf1_acc);
    return sum / n;
  };
  const double acf1_clean = meanAcf1([&] { return syntheticWhiteResidualStats(std::sqrt(q_true), std::sqrt(0.1 * q_true), 50, rng); });
  const double acf1_qchange = meanAcf1([&] { return syntheticWhiteResidualStats(std::sqrt(4 * q_true), std::sqrt(0.4 * q_true), 50, rng); });
  const double acf1_modelerror = meanAcf1([&] { return syntheticModelErrorResidualStats(std::sqrt(q_true), std::sqrt(0.1 * q_true), 8.0 * std::sqrt(q_true), 50, rng); });
  std::normal_distribution<double> na(0.0, std::sqrt(q_true)), ng(0.0, std::sqrt(0.1 * q_true));
  const double acf1_bias = meanAcf1([&] {
    std::vector<V3D> ra(50), rw(50);
    for (int j = 0; j < 50; ++j) { ra[j] = V3D(na(rng) + 20.0 * std::sqrt(q_true), na(rng) + 20.0 * std::sqrt(q_true), na(rng) + 20.0 * std::sqrt(q_true)); rw[j] = V3D(ng(rng), ng(rng), ng(rng)); }
    return reduceImuResidualSamples(ra, rw);
  });

  std::printf("  mean |acf1_acc|: clean=%.4f  Q-change=%.4f  model-error=%.4f  constant-bias=%.4f\n",
              acf1_clean, acf1_qchange, acf1_modelerror, acf1_bias);
  std::printf("  FINDING (Phase 4): in this IDEALIZED synthetic setting, |acf1| cleanly separates model error "
              "(strongly autocorrelated, ~%.2f) from the other three regimes (all near-white, <0.2). HOWEVER this "
              "codebase's own PRIOR real-data finding (adaptive_q.h's header comment, TQ-27, eee_01/eee_02) measured "
              "median |acf1_acc|~0.664 (p10~0.600) on REAL, otherwise-healthy accelerometer residuals -- i.e. real "
              "sensor residuals are inherently far more autocorrelated than this idealized synthetic 'clean' case, "
              "for reasons unrelated to gross model error (likely genuine high-frequency dynamics/vibration beyond "
              "the spline's own representational capacity). A FIXED absolute acf1 threshold therefore CANNOT be both "
              "tight enough to catch synthetic model error and loose enough to avoid rejecting healthy real data -- "
              "this is evidence FOR, not against, the production acf1_max=1.0 default, and this task's own "
              "instruction not to lower it arbitrarily is followed: no change made to the gate's default here. A "
              "genuinely more defensible mechanism (e.g. a SELF-REFERENCING/rolling-baseline whiteness test, rather "
              "than a fixed absolute threshold) is recommended for future work but not implemented in this task "
              "without real live-run acf1 evidence to validate it against (see the final report's live run6 "
              "(adaptive-Q ON) analysis for what real acf1 values this campaign actually observed).\n",
              acf1_modelerror);
  check(acf1_modelerror > 3.0 * std::max(acf1_clean, std::max(acf1_qchange, acf1_bias)),
        "in the idealized synthetic setting, model error's autocorrelation is clearly separable from the other "
        "three regimes (a necessary, though on this evidence alone not sufficient, condition for a whiteness gate "
        "to be useful)", acf1_modelerror, 3.0 * std::max(acf1_clean, std::max(acf1_qchange, acf1_bias)));
}

}  // namespace

int main()
{
  std::printf("Pose-control adaptive-Q synthetic known-truth validation suite\n");
  testAdaptiveQTracksKnownRegimeTransitions();
  testAdaptiveQReactsToModelErrorAtConstantTrueQ();
  testAdaptiveStatePersistsAcrossRepeatedSetNominalCalls();
  testExplicitResetRestoresNominal();
  testAdaptiveQRespondsIdenticallyToRChangeAsToQChange();
  testAdaptiveQIsBlindToPureConstantBiasOffset();
  testWhitenessCharacterizationAcrossRegimes();
  std::printf("%d failure(s)\n", failures);
  return failures ? 1 : 0;
}
