// R88, rewritten in R97 (the debiased moments() and the trusted-return sums were removed): the bin accumulator (include/livo_recon/map/binned/binned_bins.h).
// Eigen + std only; own check macro (does not depend on NDEBUG).
#include "livo_recon/map/binned/binned_bins.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace livo_recon::binned;

static int fails = 0;
#define CK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static double maxAbs(const M3& a, const M3& b) { return (a - b).cwiseAbs().maxCoeff(); }

int main()
{
  const double bs = 0.05;
  M3 C1 = M3::Identity() * 1e-4; C1(0, 1) = C1(1, 0) = 2e-5;
  M3 C2 = M3::Identity() * 3e-4;

  // Three bins; bin A holds the SAME point three times (a repeated beam), bin B two identical points, bin C one point.
  const V3 pa(1.01, 2.02, 0.51), pb(1.31, 2.02, 0.51), pc(1.01, 2.32, 0.81);
  std::vector<V3> pts = {pa, pa, pa, pb, pb, pc};
  std::vector<M3> cov = {C1, C2, C1, C2, C1, C2};
  BinSet bins(bs);
  int opened = 0;
  for (size_t i = 0; i < pts.size(); ++i) opened += bins.add(pts[i], cov[i]) ? 1 : 0;
  CK(opened == 3);
  CK(bins.numBins() == 3);
  CK(bins.numPoints() == 6);
  CK(!bins.add(pa, C1));            // a second return in an existing bin opens nothing
  CK(bins.numPoints() == 7);
  BinSet b2(bs);
  for (size_t i = 0; i < pts.size(); ++i) b2.add(pts[i], cov[i]);

  // average: one representative per occupied bin, in ascending key order; the point is the bin mean, the covariance C / k, the weight k.
  {
    Options o; o.cov_mode = CovMode::Average;
    const Reps r = b2.representatives(o);
    CK(r.p.size() == 3 && r.cov.size() == 3 && r.w.size() == 3);
    double wsum = 0.0;
    for (double w : r.w) wsum += w;
    CK(std::fabs(wsum - 6.0) < 1e-12);   // the weights sum to the number of returns
    // locate the bins by their representative point (the order is the key order, not the insertion order)
    int ia = -1, ib = -1, ic = -1;
    for (int i = 0; i < 3; ++i) {
      if ((r.p[i] - pa).norm() < 1e-12) ia = i;
      if ((r.p[i] - pb).norm() < 1e-12) ib = i;
      if ((r.p[i] - pc).norm() < 1e-12) ic = i;
    }
    CK(ia >= 0 && ib >= 0 && ic >= 0);
    if (ia >= 0 && ib >= 0 && ic >= 0) {
      CK(r.w[ia] == 3.0 && r.w[ib] == 2.0 && r.w[ic] == 1.0);
      CK(maxAbs(r.cov[ia], (2.0 * C1 + C2) / 3.0) < 1e-15);
      CK(maxAbs(r.cov[ib], (C2 + C1) / 2.0) < 1e-15);
      CK(maxAbs(r.cov[ic], C2) < 1e-15);
    }
  }
  // shrunk: the covariance is divided by k^2
  {
    Options o; o.cov_mode = CovMode::Shrunk;
    const Reps r = b2.representatives(o);
    int ia = -1;
    for (int i = 0; i < 3; ++i) if ((r.p[i] - pa).norm() < 1e-12) ia = i;
    CK(ia >= 0);
    if (ia >= 0) CK(maxAbs(r.cov[ia], (2.0 * C1 + C2) / 9.0) < 1e-15);
  }
  // a bin with spread inside it: the representative is the mean of its returns
  {
    BinSet s(bs);
    const V3 q1(1.010, 2.020, 0.510), q2(1.030, 2.020, 0.510);   // same 0.05 m bin
    s.add(q1, C1); s.add(q2, C1);
    CK(s.numBins() == 1);
    Options o;
    const Reps r = s.representatives(o);
    CK(r.p.size() == 1 && (r.p[0] - 0.5 * (q1 + q2)).norm() < 1e-12 && r.w[0] == 2.0);
  }
  // order independence: the same returns in another order give the same representatives to rounding
  {
    BinSet f(bs), g(bs);
    for (size_t i = 0; i < pts.size(); ++i) f.add(pts[i], cov[i]);
    for (size_t i = pts.size(); i-- > 0;) g.add(pts[i], cov[i]);
    Options o;
    const Reps a = f.representatives(o), b = g.representatives(o);
    bool same = a.p.size() == b.p.size();
    for (size_t i = 0; same && i < a.p.size(); ++i)
      same = (a.p[i] - b.p[i]).norm() < 1e-12 && maxAbs(a.cov[i], b.cov[i]) < 1e-15 && a.w[i] == b.w[i];
    CK(same);
  }
  // config names
  {
    CovMode c;
    CK(parseCovMode("average", c) && c == CovMode::Average);
    CK(parseCovMode("shrunk", c) && c == CovMode::Shrunk);
    CK(!parseCovMode("mean", c));
  }
  std::printf("test_binned_bins_r88 %s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
