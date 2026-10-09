// R82 (D4, F-500): unit test of the debiased-path line tilt (include/livo_recon/lio/line_tilt_moments.h). Planner sandbox, not evidence. Rule 3.1: identities, not tuned values.
//   LT1 hard rows: the exact-moment form equals the per-point reference (a literal copy of VoxelPlane::computeLineTilt's loop) to 1e-9 relative: L[0], L[1] and X.
//   LT2 smooth rows: same for L[0..2].
//   LT3 with every point covariance equal, the mean-covariance form equals the exact form (the approximation is exact there).
//   LT4 with unequal covariances the mean-covariance form differs from the exact form (the check can tell the two apart), and its cross terms L(.,2) are zero.
//   LT5 the gap floor: a near-degenerate gap is lifted to the floor with the sign of the eigenvalue order, and counted.
#include <cstdio>
#include <random>
#include <vector>

#include "livo_recon/lio/line_tilt_moments.h"

using namespace livo_recon;
static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

typedef Eigen::Matrix3d M3;
typedef Eigen::Vector3d V3;

struct Pt { V3 p; M3 C; };

static M3 randCov(std::mt19937& rng, double scale)
{
  std::normal_distribution<double> N(0, 1);
  M3 B; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) B(i, j) = N(rng);
  return scale * (B * B.transpose() + 0.2 * M3::Identity());
}

// literal copy of computeLineTilt's accumulation loop (equal weights, combined covariance already in pt.C)
static LineTiltResult reference(const std::vector<Pt>& pts, const V3 u[3], const double e[3], bool smooth, double fl, const V3& center)
{
  LineTiltResult out; const int rows = smooth ? 3 : 2; const int N = static_cast<int>(pts.size()); const double inv_N = 1.0 / N;
  int ia[3], ib[3]; double gap_a[3] = {0, 0, 0}, gap_b[3] = {0, 0, 0}; long floored = 0;
  auto flooredGap = [&](int i, int a, bool count) { const double g = e[i] - e[a]; const double m = std::max(std::fabs(g), fl); if (count && std::fabs(g) < fl) ++floored; return (i < a) ? -m : m; };
  M3 L[3] = {M3::Zero(), M3::Zero(), M3::Zero()}; M3 X = M3::Zero();
  for (int i = 0; i < rows; ++i) { ia[i] = (i == 0) ? 1 : 0; ib[i] = (i == 2) ? 1 : 2; gap_a[i] = flooredGap(i, ia[i], smooth); gap_b[i] = flooredGap(i, ib[i], true); }
  for (int k = 0; k < N; ++k)
  {
    const double w = inv_N; const V3 z = pts[k].p - center; const double zz[3] = {u[0].dot(z), u[1].dot(z), u[2].dot(z)}; const M3& C = pts[k].C;
    V3 gbv[2] = {V3::Zero(), V3::Zero()}, gdv[2] = {V3::Zero(), V3::Zero()};
    for (int i = 0; i < rows; ++i)
    {
      const int a = ia[i], b = ib[i];
      const V3 gb = (w / gap_b[i]) * (zz[b] * u[i] + zz[i] * u[b]); const V3 gd = (-w) * u[i];
      if (!smooth && i < 2) { gbv[i] = gb; gdv[i] = gd; }
      const V3 Cgb = C * gb, Cgd = C * gd;
      L[i](1, 1) += gb.dot(Cgb); L[i](1, 2) += gb.dot(Cgd); L[i](2, 2) += gd.dot(Cgd);
      if (smooth) { const V3 ga = (w / gap_a[i]) * (zz[a] * u[i] + zz[i] * u[a]); const V3 Cga = C * ga; L[i](0, 0) += ga.dot(Cga); L[i](0, 1) += ga.dot(Cgb); L[i](0, 2) += ga.dot(Cgd); }
    }
    if (!smooth) { const V3 Cgb1 = C * gbv[1], Cgd1 = C * gdv[1]; X(1, 1) += gbv[0].dot(Cgb1); X(1, 2) += gbv[0].dot(Cgd1); X(2, 1) += gdv[0].dot(Cgb1); X(2, 2) += gdv[0].dot(Cgd1); }
  }
  for (int i = 0; i < rows; ++i) { L[i](1, 0) = L[i](0, 1); L[i](2, 0) = L[i](0, 2); L[i](2, 1) = L[i](1, 2); out.L[i] = L[i]; }
  out.X = X; out.floored = floored; return out;
}

static double relDiff(const M3& a, const M3& b) { return (a - b).norm() / std::max(1e-300, std::max(a.norm(), b.norm())); }

int main()
{
  std::mt19937 rng(21); std::normal_distribution<double> N01(0, 1);
  const int N = 60;
  // a line-like cloud along an arbitrary axis in a far-from-origin frame (stresses the shift to the mean)
  const M3 R = Eigen::AngleAxisd(0.9, V3(1, -2, 0.5).normalized()).toRotationMatrix();
  const V3 origin(120.0, -80.0, 15.0);
  std::vector<Pt> pts;
  for (int k = 0; k < N; ++k)
  {
    const V3 loc(0.012 * N01(rng), 0.02 * N01(rng), 0.8 * N01(rng));   // thin, thicker, long
    Pt q; q.p = origin + R * loc; q.C = randCov(rng, 4e-4 * (0.5 + (k % 7))); pts.push_back(q);
  }
  // world sums, eigenstructure of the centred scatter
  double n = N; V3 mean = V3::Zero(); for (auto& q : pts) mean += q.p; mean /= n;
  M3 raw = M3::Zero(); for (auto& q : pts) raw += (q.p - mean) * (q.p - mean).transpose(); raw /= n;
  Eigen::SelfAdjointEigenSolver<M3> es(raw); V3 u[3] = {es.eigenvectors().col(0), es.eigenvectors().col(1), es.eigenvectors().col(2)};
  const double e[3] = {es.eigenvalues()(0), es.eigenvalues()(1), es.eigenvalues()(2)};
  M3 M0 = M3::Zero(), M1[3], M2[3][3]; for (int a = 0; a < 3; ++a) { M1[a] = M3::Zero(); for (int b = 0; b < 3; ++b) M2[a][b] = M3::Zero(); }
  for (auto& q : pts) { M0 += q.C; for (int a = 0; a < 3; ++a) { M1[a] += q.p(a) * q.C; for (int b = 0; b < 3; ++b) M2[a][b] += q.p(a) * q.p(b) * q.C; } }
  const double fl = 4e-4;
  // The moments sit at |p| ~ 150 m: the shift to the mean cancels ~2e4-fold. The exact form is still compared at 1e-6 here (a stress case, honest about the cancellation).
  for (bool smooth : {false, true})
  {
    const LineTiltResult ref = reference(pts, u, e, smooth, fl, mean);
    const LineTiltResult mom = lineTiltFromMoments(u, e, smooth, fl, n, mean, M2, M1, M0);
    const int rows = smooth ? 3 : 2;
    for (int i = 0; i < rows; ++i) { const double d = relDiff(ref.L[i], mom.L[i]); if (d > 1e-6) std::printf("  L[%d] smooth=%d rel %.3e\n", i, (int)smooth, d); CHECK(d < 1e-6); }
    if (!smooth) { const double d = relDiff(ref.X, mom.X); if (d > 1e-6) std::printf("  X rel %.3e\n", d); CHECK(d < 1e-6); }
    CHECK(ref.floored == mom.floored);
  }
  // Same cloud shifted to the origin: the identity holds to 1e-9 (no cancellation).
  {
    std::vector<Pt> q0 = pts; for (auto& q : q0) q.p -= origin;
    const V3 mean0 = mean - origin;
    M3 N0 = M3::Zero(), A1[3], A2[3][3]; for (int a = 0; a < 3; ++a) { A1[a] = M3::Zero(); for (int b = 0; b < 3; ++b) A2[a][b] = M3::Zero(); }
    for (auto& q : q0) { N0 += q.C; for (int a = 0; a < 3; ++a) { A1[a] += q.p(a) * q.C; for (int b = 0; b < 3; ++b) A2[a][b] += q.p(a) * q.p(b) * q.C; } }
    for (bool smooth : {false, true})
    {
      const LineTiltResult ref = reference(q0, u, e, smooth, fl, mean0); const LineTiltResult mom = lineTiltFromMoments(u, e, smooth, fl, n, mean0, A2, A1, N0);
      for (int i = 0; i < (smooth ? 3 : 2); ++i) CHECK(relDiff(ref.L[i], mom.L[i]) < 1e-9);
      if (!smooth) CHECK(relDiff(ref.X, mom.X) < 1e-9);
    }
  }
  // LT3: equal covariances.
  {
    std::vector<Pt> eq = pts; const M3 Ce = randCov(rng, 1e-3); for (auto& q : eq) q.C = Ce;
    M3 E0 = M3::Zero(), E1[3], E2[3][3]; for (int a = 0; a < 3; ++a) { E1[a] = M3::Zero(); for (int b = 0; b < 3; ++b) E2[a][b] = M3::Zero(); }
    for (auto& q : eq) { E0 += q.C; for (int a = 0; a < 3; ++a) { E1[a] += (q.p(a) - origin(a)) * q.C; for (int b = 0; b < 3; ++b) E2[a][b] += (q.p(a) - origin(a)) * (q.p(b) - origin(b)) * q.C; } }
    const V3 m0 = mean - origin;
    const LineTiltResult mom = lineTiltFromMoments(u, e, false, fl, n, m0, E2, E1, E0);
    const LineTiltResult mc = lineTiltFromMeanCov(u, e, false, fl, n, raw, Ce);
    for (int i = 0; i < 2; ++i) CHECK(relDiff(mom.L[i], mc.L[i]) < 1e-8);
    CHECK(relDiff(mom.X, mc.X) < 1e-8);
  }
  // LT4: unequal covariances.
  {
    const LineTiltResult mom = lineTiltFromMoments(u, e, false, fl, n, mean, M2, M1, M0);
    const LineTiltResult mc = lineTiltFromMeanCov(u, e, false, fl, n, raw, M0 / n);
    CHECK(mc.L[0](1, 2) == 0.0 && mc.L[1](1, 2) == 0.0 && mc.L[0](2, 1) == 0.0);
    CHECK(relDiff(mom.L[0], mc.L[0]) > 1e-3 || relDiff(mom.L[1], mc.L[1]) > 1e-3);
    CHECK(mc.L[0](2, 2) > 0.0 && mom.L[0](2, 2) > 0.0 && std::fabs(mc.L[0](2, 2) - mom.L[0](2, 2)) < 1e-12 * 1e3);   // the offset term u^T M0 u / N^2 is identical in both forms
  }
  // LT5: gap floor, sign and count.
  {
    const double e2[3] = {1e-4, 1.2e-4, 0.5};   // gaps 2e-5 and 0.5: gap_b of row 0 (axis 2) is large, row 0 axis a is unused for hard rows; row 1 axis b = 2 large
    M3 Z[3][3]; M3 Zt[3]; for (int a = 0; a < 3; ++a) { Zt[a].setZero(); for (int b = 0; b < 3; ++b) Z[a][b].setZero(); }
    const LineTiltResult r = lineTiltFromSums(u, e2, true, fl, n, Z, Zt, M3::Zero());
    CHECK(r.floored >= 1);   // smooth: row 0 axis a = 1 has |gap| 2e-5 < 4e-4
    CHECK(line_tilt_detail::flooredGap(e2, 0, 1, fl, false, const_cast<long&>(r.floored)) == -fl);   // e0 < e1: negative, lifted to -fl
    long dummy = 0; CHECK(line_tilt_detail::flooredGap(e2, 2, 0, fl, false, dummy) > 0.0);
  }
  std::printf(g_fail ? "FAILED %d\n" : "ALL PASS\n", g_fail);
  return g_fail ? 1 : 0;
}
