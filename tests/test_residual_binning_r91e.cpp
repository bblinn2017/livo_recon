// R91E unit test: residual binning by constraint direction (include/livo_recon/lio/residual_binning.h). Isolated algebraic identities only (rule 3.1): no number from this
// file is evidence about the node. Planning agent's test, NOT built or run by the planner. Uses its own CHECK macro (no assert, so NDEBUG does not disable it).
//   RB1  mode=off: HtH and Htz are bit-identical and the stats are zero.
//   RB2  the corrected HtH' and Htz' equal H^T R^-1 H and H^T R^-1 z with the dense covariance R = diag(1/w) + tau^2 sum_k u_k u_k^T (u_k = signed projection on bin axis k,
//        zero outside the bin), to 1e-9 relative: the Woodbury form is the correlated-residual solve.
//   RB3  HtH' stays positive semidefinite for several tau (smallest eigenvalue >= -1e-12 * largest), including a large and a tiny tau.
//   RB4  saturation: n rows exactly along axis 0 with weight w: the information along axis 0 is s/(1 + tau^2 s) with s = n w, below 1/tau^2 for every n, and tends to it.
//   RB5  min_rows: a bin with fewer rows than min_rows is left alone (bit-identical HtH and Htz when it is the only bin).
//   RB6  distance rows (row_kind kRowLineDist) and rows with a non-positive or non-finite variance are skipped and counted, and do not move the result.
//   RB7  sign fold: h and -h give the same bin and projections of opposite sign; the corrected result is the same for h and -h rows with r and J flipped consistently.
//   RB8  determinism: the same input twice gives bit-identical output; the output does not depend on rows of other bins that were added after the rows of this bin.
//   RB9  the axis table: 13 unit vectors, no two parallel, and they are exactly the sign-folded directions of the 3x3x3 lattice.
//   RB10 the check can fire (F2): the uncorrected sums, the tau -> infinity form (HtH - v v^T / s) and a sign-flipped Htz correction each differ from the RB2 reference by a margin
//        far above the RB2 tolerance, so RB2 would fail for each of those wrong implementations.
#include "livo_recon/lio/residual_binning.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <tuple>
#include <vector>

#define CHECK(c, msg)                                                                                              \
  do {                                                                                                             \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); }    \
    else { std::printf("[PASS] %s\n", msg); }                                                                      \
  } while (0)

using namespace livo_recon;
using M6 = Eigen::Matrix<double, 6, 6>;
using V6 = Eigen::Matrix<double, 6, 1>;
using V3 = Eigen::Vector3d;

namespace {

struct MockResidual
{
  V3 normal = V3::Zero();
  V3 point_cross_normal = V3::Zero();
  double sigma_squared = 1.0;
  double r = 0.0;
  int row_kind = 0;
};

// What accumulateLioResiduals builds (same expressions, same order).
void accumulate(const std::vector<MockResidual>& rs, M6& H, V6& g)
{
  H.setZero(); g.setZero();
  for (const auto& x : rs) {
    const double w = 1.0 / x.sigma_squared, wr = w * x.r;
    H.block<3, 3>(0, 0).noalias() += w * x.point_cross_normal * x.point_cross_normal.transpose();
    H.block<3, 3>(0, 3).noalias() += w * x.point_cross_normal * x.normal.transpose();
    H.block<3, 3>(3, 0).noalias() += w * x.normal * x.point_cross_normal.transpose();
    H.block<3, 3>(3, 3).noalias() += w * x.normal * x.normal.transpose();
    g.segment<3>(0).noalias() += wr * x.point_cross_normal;
    g.segment<3>(3).noalias() += wr * x.normal;
  }
}

// Rows scattered around the given axes (index list), with random weights, residuals and rotation Jacobians.
std::vector<MockResidual> makeRows(std::mt19937& rng, const std::vector<int>& axes, int per_axis)
{
  std::normal_distribution<double> N(0, 1);
  std::uniform_real_distribution<double> U(0.5, 2.0);
  const double(&a)[residual_binning::kAxes][3] = residual_binning::axisTable();
  std::vector<MockResidual> out;
  for (int ax : axes) {
    for (int i = 0; i < per_axis; ++i) {
      V3 h(a[ax][0] + 0.08 * N(rng), a[ax][1] + 0.08 * N(rng), a[ax][2] + 0.08 * N(rng));
      h.normalize();
      MockResidual m;
      m.normal = h;
      m.point_cross_normal = V3(N(rng), N(rng), N(rng));
      m.sigma_squared = 1.0e-4 / U(rng);   // weights of order 1e4, so that tau^2 s is of order 0.1 to 1 (the node: tau^2 s about 0.3)
      m.r = 0.01 * N(rng);
      out.push_back(m);
    }
  }
  return out;
}

double relM(const M6& a, const M6& b) { return (a - b).norm() / std::max(b.norm(), 1e-300); }
double relV(const V6& a, const V6& b) { return (a - b).norm() / std::max(b.norm(), 1e-300); }

// Dense reference: H^T R^-1 H and H^T R^-1 z with R = diag(sigma_squared) + tau^2 sum_k u_k u_k^T over the bins with at least min_rows rows.
void dense(const std::vector<MockResidual>& rs, double tau, int min_rows, M6& H, V6& g)
{
  const int n = static_cast<int>(rs.size());
  Eigen::MatrixXd J(n, 6), R = Eigen::MatrixXd::Zero(n, n);
  Eigen::VectorXd z(n);
  std::vector<int> bin(n);
  std::vector<double> u(n);
  std::vector<int> cnt(residual_binning::kAxes, 0);
  for (int i = 0; i < n; ++i) {
    J.block<1, 3>(i, 0) = rs[i].point_cross_normal.transpose();
    J.block<1, 3>(i, 3) = rs[i].normal.transpose();
    z(i) = rs[i].r;
    R(i, i) = rs[i].sigma_squared;
    double uu = 0; bin[i] = residual_binning::binOf(rs[i].normal.x(), rs[i].normal.y(), rs[i].normal.z(), uu); u[i] = uu;
    ++cnt[bin[i]];
  }
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (bin[i] == bin[j] && cnt[bin[i]] >= min_rows) R(i, j) += tau * tau * u[i] * u[j];
  Eigen::LDLT<Eigen::MatrixXd> ldlt(R);
  const Eigen::MatrixXd RiJ = ldlt.solve(J);
  const Eigen::VectorXd Riz = ldlt.solve(z);
  H = J.transpose() * RiJ;
  g = J.transpose() * Riz;
}

}  // namespace

int main()
{
  std::mt19937 rng(91005);
  const double tau = 0.0006;

  // RB9 axis table
  {
    const double(&a)[residual_binning::kAxes][3] = residual_binning::axisTable();
    bool unit = true, nonpar = true;
    for (int i = 0; i < residual_binning::kAxes; ++i) {
      V3 ai(a[i][0], a[i][1], a[i][2]);
      unit = unit && std::abs(ai.norm() - 1.0) < 1e-14;
      for (int j = i + 1; j < residual_binning::kAxes; ++j) {
        V3 aj(a[j][0], a[j][1], a[j][2]);
        nonpar = nonpar && std::abs(ai.dot(aj)) < 1.0 - 1e-9;
      }
    }
    CHECK(unit, "RB9a all 13 axes are unit vectors");
    CHECK(nonpar, "RB9b no two axes are parallel");
    std::set<std::tuple<int, int, int>> lattice, table;
    for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) for (int z = -1; z <= 1; ++z) {
      if (x == 0 && y == 0 && z == 0) continue;
      int sx = x, sy = y, sz = z;
      if (sx < 0 || (sx == 0 && (sy < 0 || (sy == 0 && sz < 0)))) { sx = -sx; sy = -sy; sz = -sz; }
      lattice.insert(std::make_tuple(sx, sy, sz));
    }
    for (int i = 0; i < residual_binning::kAxes; ++i) {
      V3 ai(a[i][0], a[i][1], a[i][2]);
      const double m = ai.cwiseAbs().maxCoeff();
      int sx = static_cast<int>(std::lround(ai.x() / m)), sy = static_cast<int>(std::lround(ai.y() / m)), sz = static_cast<int>(std::lround(ai.z() / m));
      if (sx < 0 || (sx == 0 && (sy < 0 || (sy == 0 && sz < 0)))) { sx = -sx; sy = -sy; sz = -sz; }
      table.insert(std::make_tuple(sx, sy, sz));
    }
    CHECK(lattice.size() == 13 && table == lattice, "RB9c the table is the sign-folded 3x3x3 lattice directions");
  }

  ResidualBinningOptions on; on.mode = "direction"; on.shift_sd_m = tau; on.min_rows = 20;
  const std::vector<MockResidual> rows = makeRows(rng, {0, 2, 3, 9, 12}, 60);
  M6 H; V6 g;
  accumulate(rows, H, g);

  // RB1
  {
    ResidualBinningOptions off;
    M6 H0 = H; V6 g0 = g;
    const ResidualBinningStats st = applyResidualBinning(rows, off, H0, g0);
    CHECK(std::memcmp(H0.data(), H.data(), sizeof(double) * 36) == 0 && std::memcmp(g0.data(), g.data(), sizeof(double) * 6) == 0 && st.n_corrected_rows == 0 && st.n_bins_active == 0,
          "RB1 mode=off leaves HtH and Htz bit-identical and the stats zero");
  }

  // RB2
  M6 Href; V6 gref;
  dense(rows, tau, 20, Href, gref);
  M6 H1 = H; V6 g1 = g;
  const ResidualBinningStats st1 = applyResidualBinning(rows, on, H1, g1);
  CHECK(relM(H1, Href) < 1e-9, "RB2a HtH' equals H^T R^-1 H of the dense correlated covariance");
  CHECK(relV(g1, gref) < 1e-9, "RB2b Htz' equals H^T R^-1 z of the dense correlated covariance");
  CHECK(st1.n_bins_active >= 4 && st1.n_corrected_rows >= 200 && st1.n_rows == 300, "RB2c bins and rows are counted");

  // RB3
  for (double tt : {1e-6, 1e-4, 6e-4, 1e-2, 1.0}) {
    ResidualBinningOptions o = on; o.shift_sd_m = tt;
    M6 Hx = H; V6 gx = g;
    applyResidualBinning(rows, o, Hx, gx);
    Eigen::SelfAdjointEigenSolver<M6> es(0.5 * (Hx + Hx.transpose()));
    char msg[96]; std::snprintf(msg, sizeof msg, "RB3 HtH' positive semidefinite at tau = %g", tt);
    CHECK(es.eigenvalues().minCoeff() >= -1e-12 * es.eigenvalues().maxCoeff(), msg);
  }

  // RB4 saturation
  {
    const double w = 4.0e4;
    double prev = 0.0, lim = 1.0 / (tau * tau);
    bool below = true, grows = true;
    for (int n : {10, 100, 1000, 100000}) {
      std::vector<MockResidual> rs(n);
      for (auto& m : rs) { m.normal = V3(1, 0, 0); m.point_cross_normal = V3::Zero(); m.sigma_squared = 1.0 / w; m.r = 0.0; }
      M6 Hs; V6 gs; accumulate(rs, Hs, gs);
      ResidualBinningOptions o = on; o.min_rows = 1;
      applyResidualBinning(rs, o, Hs, gs);
      const double s = n * w, expect = s / (1.0 + tau * tau * s);
      const double info = Hs(3, 3);
      below = below && info < lim && std::abs(info - expect) / expect < 1e-9;
      grows = grows && info > prev;
      prev = info;
    }
    CHECK(below, "RB4a information along the bin axis equals s/(1 + tau^2 s) and stays below 1/tau^2");
    CHECK(grows && prev > 0.99 * lim, "RB4b it grows with n and tends to 1/tau^2");
  }

  // RB5 min_rows
  {
    std::vector<MockResidual> few = makeRows(rng, {0}, 10);
    M6 Hf; V6 gf; accumulate(few, Hf, gf);
    M6 Hg = Hf; V6 gg = gf;
    ResidualBinningOptions o = on; o.min_rows = 11;
    const ResidualBinningStats st = applyResidualBinning(few, o, Hg, gg);
    CHECK(std::memcmp(Hg.data(), Hf.data(), sizeof(double) * 36) == 0 && std::memcmp(gg.data(), gf.data(), sizeof(double) * 6) == 0 && st.n_bins_active == 0,
          "RB5a a bin below min_rows is left bit-identical");
    o.min_rows = 10;
    M6 Hh = Hf; V6 gh = gf;
    const ResidualBinningStats st2 = applyResidualBinning(few, o, Hh, gh);
    CHECK(st2.n_bins_active == 1 && (Hh - Hf).norm() > 0.0, "RB5b at min_rows equal to the count the bin is corrected");
  }

  // RB6 skipped rows
  {
    std::vector<MockResidual> mixed = rows;
    const int n0 = static_cast<int>(mixed.size());
    MockResidual d = mixed[0]; d.row_kind = kRowLineDist; mixed.push_back(d);
    MockResidual z = mixed[1]; z.sigma_squared = 0.0; mixed.push_back(z);
    MockResidual nn = mixed[2]; nn.sigma_squared = std::numeric_limits<double>::quiet_NaN(); mixed.push_back(nn);
    M6 Hm = H; V6 gm = g;      // H, g: accumulated over the first n0 rows only, so the skipped rows add nothing to the input either
    const ResidualBinningStats st = applyResidualBinning(mixed, on, Hm, gm);
    CHECK(std::memcmp(Hm.data(), H1.data(), sizeof(double) * 36) == 0 && std::memcmp(gm.data(), g1.data(), sizeof(double) * 6) == 0 && st.n_skipped_rows == 3 && st.n_rows == n0 + 3,
          "RB6 distance rows and rows with a bad variance are skipped, counted, and leave the result unchanged");
  }

  // RB7 sign fold
  {
    double u1 = 0, u2 = 0;
    const int b1 = residual_binning::binOf(0.3, -0.5, 0.81, u1), b2 = residual_binning::binOf(-0.3, 0.5, -0.81, u2);
    CHECK(b1 == b2 && u1 == -u2, "RB7a h and -h share a bin with opposite projections");
    std::vector<MockResidual> flipped = rows;
    for (auto& m : flipped) { m.normal = -m.normal; m.point_cross_normal = -m.point_cross_normal; m.r = -m.r; }   // the same row written with the opposite sign convention
    M6 Hf; V6 gf; accumulate(flipped, Hf, gf);
    applyResidualBinning(flipped, on, Hf, gf);
    CHECK(relM(Hf, H1) < 1e-12 && relV(gf, g1) < 1e-12, "RB7b a consistently sign-flipped row set gives the same corrected system");
  }

  // RB8 determinism
  {
    M6 Ha = H; V6 ga = g, gb = g; M6 Hb = H;
    applyResidualBinning(rows, on, Ha, ga);
    applyResidualBinning(rows, on, Hb, gb);
    CHECK(std::memcmp(Ha.data(), Hb.data(), sizeof(double) * 36) == 0 && std::memcmp(ga.data(), gb.data(), sizeof(double) * 6) == 0, "RB8a the same input twice gives bit-identical output");
    // rows of the other bins appended AFTER this bin's rows do not change this bin's own sums (accumulators are per bin, in row order)
    std::vector<MockResidual> head = makeRows(rng, {3}, 40), both = head;
    std::vector<MockResidual> tail = makeRows(rng, {9}, 40);
    both.insert(both.end(), tail.begin(), tail.end());
    M6 Hh; V6 gh; accumulate(head, Hh, gh);
    const ResidualBinningStats sh = applyResidualBinning(head, on, Hh, gh);
    M6 Hb2; V6 gb2; accumulate(both, Hb2, gb2);
    const ResidualBinningStats sb = applyResidualBinning(both, on, Hb2, gb2);
    CHECK(sh.bins[3].s == sb.bins[3].s && sh.bins[3].t == sb.bins[3].t && std::memcmp(sh.bins[3].v, sb.bins[3].v, sizeof(double) * 6) == 0,
          "RB8b a bin's sums do not depend on rows of other bins");
  }

  // RB10 the check fires
  {
    const double tol = 1e-9;
    CHECK(relM(H, Href) > 1e4 * tol, "RB10a the uncorrected sums differ from the reference far above the RB2 tolerance");
    // tau -> infinity form: HtH - v v^T / s, for every bin with >= min_rows
    ResidualBinningOptions o = on; o.shift_sd_m = 1e6;
    M6 Hi = H; V6 gi = g; applyResidualBinning(rows, o, Hi, gi);
    CHECK(relM(Hi, Href) > 1e4 * tol, "RB10b the tau -> infinity form differs from the reference far above the tolerance");
    // sign-flipped Htz correction
    M6 Hc = H; V6 gs = g;
    applyResidualBinning(rows, on, Hc, gs);   // gs is corrected; the wrong one adds the correction instead of subtracting it
    const V6 wrong = g + (g - gs);
    CHECK(relV(wrong, gref) > 1e4 * tol, "RB10c a sign-flipped Htz correction differs from the reference far above the tolerance");
  }
  std::printf("all R91E binning checks passed\n");
  return 0;
}
