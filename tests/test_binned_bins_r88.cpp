// R88: the bin accumulator (include/livo_recon/map/binned/binned_bins.h). Eigen + std only; own check macro (does not depend on NDEBUG).
#include "livo_recon/map/binned/binned_bins.h"

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace livo_recon::binned;

static int fails = 0;
#define CK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static double maxAbs(const M3& a, const M3& b) { return (a - b).cwiseAbs().maxCoeff(); }

struct Raw
{
  double n = 0; V3 sp = V3::Zero(); M3 spp = M3::Zero(); M3 scov = M3::Zero(); M3 scov_sensor = M3::Zero(); double svar = 0; M3 v[3][3]; M3 w[3];
  Raw() { for (int a = 0; a < 3; ++a) { w[a].setZero(); for (int b = 0; b < 3; ++b) v[a][b].setZero(); } }
};

// What VoxelPlane::addPoints() accumulates for raw points with weight 1 (sensor_only mode).
static Raw rawSums(const std::vector<V3>& p, const std::vector<M3>& c, const std::vector<bool>& tr, const V3& ref)
{
  Raw r;
  for (size_t i = 0; i < p.size(); ++i) {
    const V3 d = p[i] - ref;
    r.n += 1.0; r.sp += d; r.spp += d * d.transpose(); r.scov += c[i];
    if (tr[i]) { r.scov_sensor += c[i]; r.svar += c[i].trace() / 3.0; }
    for (int a = 0; a < 3; ++a) { r.w[a] += p[i](a) * c[i]; for (int b = 0; b < 3; ++b) r.v[a][b] += (p[i](a) * p[i](b)) * c[i]; }
  }
  return r;
}

int main()
{
  const double bs = 0.05;
  const V3 ref(0.3, -0.2, 0.1);
  M3 C1 = M3::Identity() * 1e-4; C1(0, 1) = C1(1, 0) = 2e-5;
  M3 C2 = M3::Identity() * 3e-4;

  // Three bins; bin A holds the SAME point three times (a repeated beam), bin B two identical points, bin C one point. Bootstrap (untrusted) returns in A, trusted elsewhere.
  const V3 pa(1.01, 2.02, 0.51), pb(1.31, 2.02, 0.51), pc(1.01, 2.32, 0.81);
  std::vector<V3> pts = {pa, pa, pa, pb, pb, pc};
  std::vector<M3> cov = {C1, C2, C1, C2, C1, C2};
  std::vector<bool> tr = {false, false, false, true, true, true};
  BinSet bins(bs);
  int opened = 0;
  for (size_t i = 0; i < pts.size(); ++i) opened += bins.add(pts[i], cov[i], tr[i]) ? 1 : 0;
  CK(opened == 3);
  CK(bins.numBins() == 3);
  CK(bins.numPoints() == 6);
  CK(!bins.add(pa, C1, true));            // a second return in an existing bin opens nothing
  CK(bins.numPoints() == 7);
  // rebuild without that extra point for the comparisons
  BinSet b2(bs);
  for (size_t i = 0; i < pts.size(); ++i) b2.add(pts[i], cov[i], tr[i]);

  // average + count on bins of identical points reproduces the raw sums exactly (the representative IS every member, covariance sums are the raw sums)
  {
    Options o; o.cov_mode = CovMode::Average;
    const Moments m = b2.moments(o, ref);
    const Raw r = rawSums(pts, cov, tr, ref);
    CK(std::fabs(m.n - r.n) < 1e-12);
    CK((m.sp - r.sp).cwiseAbs().maxCoeff() < 1e-12);
    CK(maxAbs(m.spp, r.spp) < 1e-12);
    CK(maxAbs(m.scov, r.scov) < 1e-15);
    CK(maxAbs(m.scov_sensor, r.scov_sensor) < 1e-15);
    CK(std::fabs(m.sum_sensor_var - r.svar) < 1e-15);
    bool vw = true;
    for (int a = 0; a < 3; ++a) { vw = vw && maxAbs(m.w[a], r.w[a]) < 1e-14; for (int b = 0; b < 3; ++b) vw = vw && maxAbs(m.v[a][b], r.v[a][b]) < 1e-14; }
    CK(vw);
  }
  // shrunk + count (R95: the uniform weight was removed): each bin's covariance is divided by k^2 and the bin is weighted by k, so the covariance sums are C / k per bin; n is the number of returns
  {
    Options o; o.cov_mode = CovMode::Shrunk;
    const Moments m = b2.moments(o, ref);
    const M3 expect_all = (2.0 * C1 + C2) / 3.0 + (C2 + C1) / 2.0 + C2;
    CK(maxAbs(m.scov, expect_all) < 1e-15);
    CK(maxAbs(m.scov_sensor, (C2 + C1) / 2.0 + C2) < 1e-15);   // trusted: bin A has none, bin B has both, bin C has C2
    CK(std::fabs(m.n - 6.0) < 1e-12);
  }
  // untrusted returns are in scov and not in scov_sensor
  {
    BinSet u(bs);
    u.add(pa, C1, false);
    Options o;
    const Moments m = u.moments(o, V3::Zero());
    CK(maxAbs(m.scov, C1) < 1e-15);
    CK(m.scov_sensor.cwiseAbs().maxCoeff() == 0.0);
    CK(m.sum_sensor_var == 0.0);
  }
  // a bin with spread inside it: the representative is the mean (position sums of count weighting equal the raw sums; the second moment differs by the within-bin scatter, which is not kept)
  {
    BinSet s(bs);
    const V3 q1(1.010, 2.020, 0.510), q2(1.030, 2.020, 0.510);   // same 0.05 m bin
    s.add(q1, C1, true); s.add(q2, C1, true);
    CK(s.numBins() == 1);
    Options o;   // average + count
    const Moments m = s.moments(o, V3::Zero());
    CK((m.sp - (q1 + q2)).norm() < 1e-12);
    const M3 raw_spp = q1 * q1.transpose() + q2 * q2.transpose();
    CK((raw_spp - m.spp).cwiseAbs().maxCoeff() > 1e-6);   // missing within-bin scatter, by design
  }
  // order independence: the same returns in another order give the same sums to rounding
  {
    BinSet f(bs), g(bs);
    for (size_t i = 0; i < pts.size(); ++i) f.add(pts[i], cov[i], tr[i]);
    for (size_t i = pts.size(); i-- > 0;) g.add(pts[i], cov[i], tr[i]);
    Options o;
    const Moments a = f.moments(o, ref), b = g.moments(o, ref);
    CK(maxAbs(a.spp, b.spp) < 1e-12);
    CK(maxAbs(a.scov, b.scov) < 1e-15);
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
