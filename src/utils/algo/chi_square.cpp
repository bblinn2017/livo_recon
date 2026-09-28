#include "livo_recon/utils/algo/chi_square.h"

#include <cmath>
#include <limits>

namespace livo_recon
{
namespace
{

// Series expansion for P(a,x), valid and rapidly convergent for x < a+1.
double gammaSeries(double a, double x)
{
  if (x <= 0.0) return 0.0;
  const int max_iter = 200;
  const double eps = 1e-15;
  double ap = a;
  double sum = 1.0 / a;
  double del = sum;
  for (int n = 0; n < max_iter; ++n) {
    ap += 1.0;
    del *= x / ap;
    sum += del;
    if (std::abs(del) < std::abs(sum) * eps) break;
  }
  return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

// Continued fraction for Q(a,x) = 1 - P(a,x), valid for x >= a+1
// (Lentz's algorithm).
double gammaContinuedFractionQ(double a, double x)
{
  const int max_iter = 200;
  const double eps = 1e-15;
  const double tiny = 1e-300;
  double b = x + 1.0 - a;
  double c = 1.0 / tiny;
  double d = 1.0 / b;
  double h = d;
  for (int i = 1; i <= max_iter; ++i) {
    const double an = -i * (i - a);
    b += 2.0;
    d = an * d + b;
    if (std::abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (std::abs(c) < tiny) c = tiny;
    d = 1.0 / d;
    const double delta = d * c;
    h *= delta;
    if (std::abs(delta - 1.0) < eps) break;
  }
  return h * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

}  // namespace

double regularizedLowerIncompleteGamma(double a, double x)
{
  if (x < 0.0 || a <= 0.0) return std::numeric_limits<double>::quiet_NaN();
  if (x == 0.0) return 0.0;
  if (x < a + 1.0) return gammaSeries(a, x);
  return 1.0 - gammaContinuedFractionQ(a, x);
}

double chiSquareCdf(double x, int dof)
{
  if (dof <= 0) return std::numeric_limits<double>::quiet_NaN();
  if (x <= 0.0) return 0.0;
  return regularizedLowerIncompleteGamma(dof / 2.0, x / 2.0);
}

double chiSquareInverseCdf(double p, int dof)
{
  if (dof <= 0 || !(p > 0.0) || !(p < 1.0))
    return std::numeric_limits<double>::quiet_NaN();
  // Bisection on the monotonically increasing CDF. An upper bound that
  // comfortably contains any quantile used by this codebase (p<=0.999,
  // dof<=100): the mean is `dof`, and the CDF is effectively 1 well before
  // 10x the mean for these tail probabilities.
  double lo = 0.0;
  double hi = std::max(10.0, 20.0 * dof);
  while (chiSquareCdf(hi, dof) < p) hi *= 2.0;
  for (int i = 0; i < 200; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (chiSquareCdf(mid, dof) < p) lo = mid; else hi = mid;
    if (hi - lo < 1e-12 * std::max(1.0, hi)) break;
  }
  return 0.5 * (lo + hi);
}

}  // namespace livo_recon
