#pragma once

namespace livo_recon
{

// Regularized lower incomplete gamma function P(a, x) = gamma(a,x)/Gamma(a),
// via the standard series (x < a+1) / continued-fraction (x >= a+1) split
// (Numerical Recipes' gser/gcf). Used only to build the chi-square CDF below;
// exposed because it is independently a clean, isolated mathematical
// identity to test (P(a,0)=0, P(a,inf)->1, symmetry with the upper
// incomplete gamma).
double regularizedLowerIncompleteGamma(double a, double x);

// CDF of the chi-square distribution with `dof` degrees of freedom:
// chiSquareCdf(x, dof) = P(dof/2, x/2).
double chiSquareCdf(double x, int dof);

// Inverse CDF (quantile function): the value x such that
// chiSquareCdf(x, dof) == p, found by bisection on the (monotonic) CDF.
// `p` must be in (0, 1).
double chiSquareInverseCdf(double p, int dof);

}  // namespace livo_recon
