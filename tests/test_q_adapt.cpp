// R72 unit test (F-401): process-noise alternatives (include/livo_recon/lio/q_adapt.h). Isolated algebraic identities only; every
// identity that could pass for a wrong implementation has a paired negative control that must fail the same identity.
//   QA1 consistent model (r^2 = sigma^2 + h P h for every row): NIS == 1 and rho == 1 (1e-12). Control: rows scaled by 1.2 give
//       NIS == 1.44 (1e-12), so the check can tell a consistent model from an inconsistent one.
//   QA2 sum h P h scales exactly with P (P -> c P gives c * sum_hph) and rho scales as 1 / c.
//   QA3 invalid statistic: no rows, or P = 0, gives valid == false.
//   QA4 lag-1 autocorrelation: alternating-sign innovations give -(n-1)/n, constant-sign give +(n-1)/n.
//   QA5 innovation_scale gain: rho == 1 leaves the gain exactly 1; rho = 4 gives 4^(1-f); rho > clamp acts as the clamp; forgetting
//       == 1 never moves it; bounds hold.
//   QA6 nis_gate state machine: for NIS = 2,2,0.5,0.5 the gain is exactly 1.25, 1.5625, 1+0.95*0.5625, 1+0.95*(that) (1e-14);
//       the whiteness gate blocks inflation when |lag1| > whiteness_max and counts it.
//   QA7 apply selects which variance sees the gain.
#include "livo_recon/lio/q_adapt.h"

#include <cstdio>
#include <cstdlib>
#include <random>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;
using M6 = Eigen::Matrix<double, 6, 6>;
using V6 = Eigen::Matrix<double, 6, 1>;

static bool close(double a, double b, double rel) { return std::abs(a - b) <= rel * std::max(std::abs(a), std::abs(b)); }

static M6 randomSpd(std::mt19937& rng)
{
  std::normal_distribution<double> N(0, 1);
  M6 B;
  for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) B(i, j) = N(rng);
  return 1e-4 * (B * B.transpose() + 0.5 * M6::Identity());
}

static std::vector<InnovationRow> consistentRows(std::mt19937& rng, const M6& P, int n, double stretch)
{
  std::normal_distribution<double> N(0, 1);
  std::uniform_real_distribution<double> U(1e-4, 4e-4);
  std::vector<InnovationRow> rows;
  for (int i = 0; i < n; ++i) {
    InnovationRow row;
    for (int k = 0; k < 6; ++k) row.h(k) = N(rng);
    row.sigma2 = U(rng);
    const double hph = row.h.dot(P * row.h);
    row.r = stretch * std::sqrt(row.sigma2 + hph) * ((i % 2) ? 1.0 : -1.0);
    rows.push_back(row);
  }
  return rows;
}

int main()
{
  std::mt19937 rng(11);
  const M6 P = randomSpd(rng);

  {   // QA1
    const auto rows = consistentRows(rng, P, 200, 1.0);
    const InnovationStats s = innovationStats(rows, P);
    CHECK(s.valid && close(s.nis, 1.0, 1e-12), "QA1 consistent model: NIS == 1");
    CHECK(close(s.rho, 1.0, 1e-12), "QA1 consistent model: rho == 1");
    const auto bad = consistentRows(rng, P, 200, 1.2);
    const InnovationStats sb = innovationStats(bad, P);
    CHECK(close(sb.nis, 1.44, 1e-12), "QA1 control: rows scaled by 1.2 give NIS == 1.44 (the check fires)");
    CHECK(sb.rho > 1.4, "QA1 control: rho > 1.4 for the scaled rows");
  }
  {   // QA2
    const auto rows = consistentRows(rng, P, 100, 1.0);
    const InnovationStats a = innovationStats(rows, P);
    const InnovationStats b = innovationStats(rows, 3.0 * P);
    CHECK(close(b.sum_hph, 3.0 * a.sum_hph, 1e-12), "QA2 sum h P h scales with P");
    const double expect = std::max(0.0, a.sum_r2 - a.sum_sigma2) / (3.0 * a.sum_hph);
    CHECK(close(b.rho, expect, 1e-12), "QA2 rho scales as 1/c at fixed residuals");
  }
  {   // QA3
    const InnovationStats e = innovationStats({}, P);
    CHECK(!e.valid && e.n == 0, "QA3 no rows: invalid");
    const auto rows = consistentRows(rng, P, 10, 1.0);
    const InnovationStats z = innovationStats(rows, M6::Zero());
    CHECK(!z.valid, "QA3 P = 0: invalid");
  }
  {   // QA4
    std::vector<InnovationRow> alt, con;
    const int n = 10;
    for (int i = 0; i < n; ++i) {
      InnovationRow r; r.sigma2 = 1.0; r.h = V6::Zero(); r.h(0) = 1.0;
      r.r = (i % 2) ? 0.5 : -0.5; alt.push_back(r);
      r.r = 0.5; con.push_back(r);
    }
    CHECK(close(innovationStats(alt, P).lag1, -(n - 1.0) / n, 1e-12), "QA4 alternating innovations: lag1 == -(n-1)/n");
    CHECK(close(innovationStats(con, P).lag1, (n - 1.0) / n, 1e-12), "QA4 constant-sign innovations: lag1 == +(n-1)/n");
  }
  auto stat = [](double rho, double nis, double lag) { InnovationStats s; s.valid = true; s.rho = rho; s.nis = nis; s.lag1 = lag; s.n = 10; return s; };
  {   // QA5
    QGainOptions o; o.mode = "innovation_scale"; o.forgetting = 0.9; o.min_gain = 1.0; o.max_gain = 100.0; o.rho_clamp = 4.0;
    QGain g; g.configure(o);
    g.update(stat(1.0, 1.0, 0.0));
    CHECK(g.gain() == 1.0, "QA5 rho == 1 leaves the gain exactly 1");
    g.update(stat(4.0, 1.0, 0.0));
    CHECK(close(g.gain(), std::pow(4.0, 0.1), 1e-14), "QA5 rho = 4 gives 4^(1-f)");
    QGain h; h.configure(o);
    h.update(stat(400.0, 1.0, 0.0));
    CHECK(close(h.gain(), std::pow(4.0, 0.1), 1e-14), "QA5 rho above the clamp acts as the clamp");
    QGain k; QGainOptions o1 = o; o1.forgetting = 1.0; k.configure(o1);
    for (int i = 0; i < 5; ++i) k.update(stat(4.0, 1.0, 0.0));
    CHECK(k.gain() == 1.0, "QA5 forgetting == 1 never moves the gain");
    QGain m; QGainOptions o2 = o; o2.max_gain = 1.2; m.configure(o2);
    for (int i = 0; i < 50; ++i) m.update(stat(4.0, 1.0, 0.0));
    CHECK(m.gain() == 1.2, "QA5 max_gain bound holds");
    QGain d; d.configure(o);
    d.update(stat(0.25, 1.0, 0.0));
    CHECK(d.gain() == 1.0, "QA5 min_gain = 1 blocks deflation");
    QGainOptions o3 = o; o3.min_gain = 0.1; QGain d2; d2.configure(o3);
    d2.update(stat(0.25, 1.0, 0.0));
    CHECK(d2.gain() < 1.0, "QA5 min_gain < 1 allows deflation (the bound is what blocked it above)");
    QGain inv; inv.configure(o);
    InnovationStats bad; bad.valid = false;
    inv.update(bad);
    CHECK(inv.gain() == 1.0, "QA5 invalid statistic leaves the gain unchanged");
  }
  {   // QA6
    QGainOptions o; o.mode = "nis_gate"; o.nis_hi = 1.5; o.inflate = 1.25; o.decay = 0.95; o.min_gain = 1.0; o.max_gain = 100.0;
    QGain g; g.configure(o);
    g.update(stat(1, 2.0, 0)); CHECK(close(g.gain(), 1.25, 1e-14), "QA6 step 1: 1.25");
    g.update(stat(1, 2.0, 0)); CHECK(close(g.gain(), 1.5625, 1e-14), "QA6 step 2: 1.5625");
    g.update(stat(1, 0.5, 0)); CHECK(close(g.gain(), 1.0 + 0.95 * 0.5625, 1e-14), "QA6 step 3: decay toward 1");
    g.update(stat(1, 0.5, 0)); CHECK(close(g.gain(), 1.0 + 0.95 * (0.95 * 0.5625), 1e-14), "QA6 step 4: decay toward 1");
    CHECK(g.inflating() == 2 && g.scans() == 4, "QA6 counters");
    QGainOptions ow = o; ow.whiteness_max = 0.3;
    QGain w; w.configure(ow);
    w.update(stat(1, 2.0, 0.5));
    CHECK(w.gain() == 1.0 && w.whitenessBlocked() == 1, "QA6 whiteness gate blocks inflation and counts it");
    w.update(stat(1, 2.0, 0.1));
    CHECK(close(w.gain(), 1.25, 1e-14), "QA6 white innovations still inflate");
  }
  {   // QA7
    QGainOptions o; o.mode = "nis_gate"; o.inflate = 2.0; o.nis_hi = 1.0;
    QGain g; g.configure(o); g.update(stat(1, 2.0, 0));
    CHECK(g.gainAcc() == 2.0 && g.gainGyr() == 2.0, "QA7 apply=both");
    o.apply = "acc"; QGain a; a.configure(o); a.update(stat(1, 2.0, 0));
    CHECK(a.gainAcc() == 2.0 && a.gainGyr() == 1.0, "QA7 apply=acc");
    o.apply = "gyr"; QGain y; y.configure(o); y.update(stat(1, 2.0, 0));
    CHECK(y.gainAcc() == 1.0 && y.gainGyr() == 2.0, "QA7 apply=gyr");
  }

  {   // QA8 (R80, F-476): the deflating side of the NIS gate; nis_lo = 0 is the one-sided gate bit for bit
    QGainOptions o; o.mode = "nis_gate"; o.nis_hi = 1.5; o.inflate = 1.25; o.decay = 0.95; o.max_gain = 100.0;
    o.nis_lo = 0.75; o.deflate = 0.8; o.min_gain = 0.25;
    QGain g; g.configure(o);
    g.update(stat(1, 0.5, 0)); CHECK(close(g.gain(), 0.8, 1e-14), "QA8 NIS 0.5 < 0.75 deflates by 0.8");
    g.update(stat(1, 0.5, 0)); CHECK(close(g.gain(), 0.64, 1e-14), "QA8 deflation compounds");
    g.update(stat(1, 1.0, 0)); CHECK(close(g.gain(), 1.0 + 0.95 * (0.64 - 1.0), 1e-14), "QA8 NIS between the bounds decays toward 1 from below");
    for (int i = 0; i < 40; ++i) g.update(stat(1, 0.1, 0));
    CHECK(close(g.gain(), 0.25, 1e-14) && close(g.minSeen(), 0.25, 1e-14), "QA8 min_gain bounds the deflation");
    CHECK(g.deflating() == 42 && g.inflating() == 0, "QA8 counters");
    g.update(stat(1, 2.0, 0)); CHECK(close(g.gain(), 0.25 * 1.25, 1e-14), "QA8 inflation still multiplies the current gain");
    QGainOptions off = o; off.nis_lo = 0.0; off.deflate = 1.0; off.min_gain = 1.0;
    QGain h; h.configure(off); h.update(stat(1, 0.1, 0));
    CHECK(h.gain() == 1.0 && h.deflating() == 0, "QA8 nis_lo = 0 never deflates");
  }

  std::printf("ALL PASS\n");
  return 0;
}
