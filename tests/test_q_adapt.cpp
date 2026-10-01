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
//   QL1 data that IS a polynomial of the fitted degree has residual variance ~ 0 (< 1e-18).
//   QL2 data = polynomial + c * w, with w a vector orthogonal to the polynomial space scaled to |w|^2 = n - p, has variance c^2 (1e-9
//       relative). Controls: dividing RSS by n instead of n - p is off by the factor (n-p)/n, and the plain sample variance (no
//       polynomial removed) is far larger than c^2 on trending data - both would fail QL2.
//   QL3 refusals: n - p < 1 and zero time span return false.
//   QW1 WNOA blocks equal the integral of the propagated disturbance (3-point Gauss-Legendre, exact to degree 5) to 1e-14 relative.
//   QW2 semigroup: Q(dt1 + dt2) = F2 Q(dt1) F2^T + Q(dt2) with F2 = [1 dt2; 0 1]. Control: the naive sum Q(dt1) + Q(dt2) violates it.
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

  // ---- local polynomial
  const int n = 40, order = 2, p = order + 1;
  std::vector<double> t(n);
  for (int i = 0; i < n; ++i) t[i] = 0.1 + 0.0025 * i;
  auto tau = [&](int i) { return 2.0 * (t[i] - t.front()) / (t.back() - t.front()) - 1.0; };
  {   // QL1
    std::vector<Eigen::Vector3d> y(n);
    for (int i = 0; i < n; ++i) {
      const double x = tau(i);
      y[i] = Eigen::Vector3d(9.81 + 2.0 * x - 3.0 * x * x, -1.0 + 0.5 * x, 4.0 * x * x);
    }
    Eigen::Vector3d var;
    CHECK(localPolyResidualVariance(t, y, order, var), "QL1 fit succeeds");
    CHECK(var.maxCoeff() < 1e-18, "QL1 polynomial data has ~0 residual variance");
  }
  {   // QL2
    Eigen::MatrixXd A(n, p);
    for (int i = 0; i < n; ++i) { double pw = 1; for (int j = 0; j < p; ++j) { A(i, j) = pw; pw *= tau(i); } }
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
    std::normal_distribution<double> N(0, 1);
    const double cs[3] = {0.02, 0.003, 0.15};
    std::vector<Eigen::Vector3d> y(n);
    for (int i = 0; i < n; ++i) y[i].setZero();
    for (int axis = 0; axis < 3; ++axis) {
      Eigen::VectorXd e(n);
      for (int i = 0; i < n; ++i) e(i) = N(rng);
      Eigen::VectorXd w = e - A * qr.solve(e);
      w *= std::sqrt((n - p) / w.squaredNorm());
      for (int i = 0; i < n; ++i) {
        const double x = tau(i);
        y[i](axis) = (axis == 0 ? 9.81 + 40.0 * x : 3.0 * x * x) + cs[axis] * w(i);
      }
    }
    Eigen::Vector3d var;
    CHECK(localPolyResidualVariance(t, y, order, var), "QL2 fit succeeds");
    for (int axis = 0; axis < 3; ++axis)
      CHECK(close(var(axis), cs[axis] * cs[axis], 1e-9), "QL2 variance == c^2 for noise orthogonal to the polynomial space");
    const double c0 = cs[0];
    const double rss0 = var(0) * (n - p);
    CHECK(std::abs(rss0 / n - c0 * c0) > 1e-3 * c0 * c0, "QL2 control: RSS/n would be wrong by (n-p)/n (the check fires)");
    double mean = 0.0; for (int i = 0; i < n; ++i) mean += y[i](0); mean /= n;
    double sv = 0.0; for (int i = 0; i < n; ++i) sv += (y[i](0) - mean) * (y[i](0) - mean); sv /= (n - 1);
    CHECK(sv > 100.0 * c0 * c0, "QL2 control: plain sample variance on trending data is far above c^2 (the trend is what the fit removes)");
  }
  {   // QL3
    Eigen::Vector3d var(7, 7, 7);
    std::vector<double> t3 = {0.0, 0.1, 0.2};
    std::vector<Eigen::Vector3d> y3(3, Eigen::Vector3d::Zero());
    CHECK(!localPolyResidualVariance(t3, y3, 2, var) && var(0) == 7.0, "QL3 n - p < 1 refused, output untouched");
    std::vector<double> tz = {1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<Eigen::Vector3d> yz(5, Eigen::Vector3d::Zero());
    CHECK(!localPolyResidualVariance(tz, yz, 1, var), "QL3 zero time span refused");
  }
  {   // QW1, QW2
    const double qa = 0.37, qg = 0.011, dt = 0.0123;
    const WnoaBlocks b = wnoaBlocks(qa, qg, dt);
    const double xs[3] = {-std::sqrt(3.0 / 5.0), 0.0, std::sqrt(3.0 / 5.0)};
    const double ws[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
    double ipp = 0, ipv = 0, ivv = 0;
    for (int k = 0; k < 3; ++k) {
      const double s = 0.5 * dt * (xs[k] + 1.0);
      const double w = 0.5 * dt * ws[k];
      const double m = dt - s;   // position response to a velocity kick at s
      ipp += w * m * m; ipv += w * m; ivv += w * 1.0;
    }
    CHECK(close(b.pp, qa * ipp, 1e-14) && close(b.pv, qa * ipv, 1e-14) && close(b.vv, qa * ivv, 1e-14), "QW1 blocks equal the integral of the propagated disturbance");
    CHECK(close(b.rr, qg * dt, 1e-15), "QW1 rotation block is qc_gyr dt");
    const double d1 = 0.004, d2 = 0.009;
    auto Q = [&](double d) { const WnoaBlocks w = wnoaBlocks(qa, 0, d); Eigen::Matrix2d m; m << w.pp, w.pv, w.pv, w.vv; return m; };
    Eigen::Matrix2d F2; F2 << 1, d2, 0, 1;
    const Eigen::Matrix2d lhs = Q(d1 + d2), rhs = F2 * Q(d1) * F2.transpose() + Q(d2);
    CHECK((lhs - rhs).cwiseAbs().maxCoeff() < 1e-14 * lhs.cwiseAbs().maxCoeff(), "QW2 semigroup identity");
    const Eigen::Matrix2d naive = Q(d1) + Q(d2);
    CHECK((lhs - naive).cwiseAbs().maxCoeff() > 1e-3 * lhs.cwiseAbs().maxCoeff(), "QW2 control: the naive sum violates it (the check fires)");
  }
  std::printf("ALL PASS\n");
  return 0;
}
