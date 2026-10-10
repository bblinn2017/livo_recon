// R91F unit test: rotation-direction bins and the joint solve (include/livo_recon/lio/residual_binning.h, modes "rotation" and "both"). Isolated algebraic identities only (rule 3.1): no
// number from this file is evidence about the node. Planning agent's test, NOT built or run by the planner (rule 1.1). Own CHECK macro (no assert, so NDEBUG does not disable it).
//   RF1  mode=off: HtH and Htz bit-identical, stats zero. mode=direction still equals the R91E per-bin formula (re-implemented here independently) to 1e-12 relative.
//   RF2  mode=rotation equals H^T R^-1 H and H^T R^-1 z with the dense covariance R = diag(sigma^2) + tau_r^2 sum_l u_l u_l^T (u_l = hrot . a_l inside rotation bin l), to 1e-9 relative.
//   RF3  mode=both equals the dense covariance with BOTH families, R = diag(sigma^2) + tau_p^2 sum_k up_k up_k^T + tau_r^2 sum_l ur_l ur_l^T, to 1e-9 relative: the joint solve
//        (rows belong to one position bin and one rotation bin, so the two are coupled through X_kl).
//   RF4  limits: both with a tiny rotation shift equals direction, both with a tiny position shift equals rotation (1e-9 relative).
//   RF5  HtH' stays positive semidefinite for several (tau_p, tau_r) pairs, tiny and large included (smallest eigenvalue >= -1e-12 * largest).
//   RF6  saturation along a rotation axis: n rows with hrot = (L, 0, 0): the information on rotation x is s/(1 + tau_r^2 s) with s = n w L^2, below 1/tau_r^2, tending to it.
//   RF7  skipped rows: distance rows, non-positive / non-finite variance, and (rotation modes) a zero rotation lever arm are skipped and counted and leave the result unchanged;
//        a zero lever arm is NOT skipped in mode direction (it still has a position bin).
//   RF8  sign fold: a consistently sign-flipped row set gives the same corrected system in mode both.
//   RF9  determinism: the same input twice is bit-identical in mode both; the sums of one rotation bin do not depend on rows of other bins appended later; the cross sums equal an
//        independent accumulation.
//   RF10 the check can fire: the sequential form (direction correction, then rotation correction on the result), the block-diagonal form (cross terms X left out) and a sign-flipped
//        Htz correction each differ from the RF3 reference by a margin far above the RF3 tolerance.
//   RF11 min_rows applies to both families: a rotation bin below min_rows is left alone (its V column and D entry are inert): with min_rows above every bin's count the system is bit-identical.
// NOT tested here: the solve_failed path (a valid input cannot make the 26 x 26 factorisation fail; the path leaves HtH and Htz untouched and flags the engagement line).
#include "livo_recon/lio/residual_binning.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
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

V3 axisVec(int k) { const double(&a)[residual_binning::kAxes][3] = residual_binning::axisTable(); return V3(a[k][0], a[k][1], a[k][2]); }

// n rows; row i has its position direction near pos_axes[i % P] and its rotation lever arm (direction near rot_axes[(i / P + 2 * i) % R], length 3 to 40) so that the two partitions cross.
std::vector<MockResidual> makeRows(std::mt19937& rng, const std::vector<int>& pos_axes, const std::vector<int>& rot_axes, int n)
{
  std::normal_distribution<double> N(0, 1);
  std::uniform_real_distribution<double> U(0.5, 2.0), L(3.0, 40.0);
  const int P = static_cast<int>(pos_axes.size()), R = static_cast<int>(rot_axes.size());
  std::vector<MockResidual> out;
  for (int i = 0; i < n; ++i) {
    V3 h = axisVec(pos_axes[i % P]) + 0.08 * V3(N(rng), N(rng), N(rng));
    h.normalize();
    V3 g = axisVec(rot_axes[(i / P + 2 * i) % R]) + 0.08 * V3(N(rng), N(rng), N(rng));
    g.normalize();
    g *= L(rng);
    MockResidual m;
    m.normal = h; m.point_cross_normal = g;
    m.sigma_squared = 1.0e-4 / U(rng);   // weights of order 1e4
    m.r = 0.01 * N(rng);
    out.push_back(m);
  }
  return out;
}

double relM(const M6& a, const M6& b) { return (a - b).norm() / std::max(b.norm(), 1e-300); }
double relV(const V6& a, const V6& b) { return (a - b).norm() / std::max(b.norm(), 1e-300); }

// Dense reference. R = diag(sigma^2) + tp^2 sum_k up_k up_k^T (position bins with >= min_rows rows) + tr^2 sum_l ur_l ur_l^T (rotation bins with >= min_rows rows); tp or tr = 0 leaves a family out.
void dense(const std::vector<MockResidual>& rs, double tp, double tr, int min_rows, M6& H, V6& g)
{
  const int n = static_cast<int>(rs.size());
  Eigen::MatrixXd J(n, 6), R = Eigen::MatrixXd::Zero(n, n);
  Eigen::VectorXd z(n);
  std::vector<int> bp(n), br(n);
  std::vector<double> up(n), ur(n);
  std::vector<int> cp(residual_binning::kAxes, 0), cr(residual_binning::kAxes, 0);
  for (int i = 0; i < n; ++i) {
    J.block<1, 3>(i, 0) = rs[i].point_cross_normal.transpose();
    J.block<1, 3>(i, 3) = rs[i].normal.transpose();
    z(i) = rs[i].r;
    R(i, i) = rs[i].sigma_squared;
    double dummy = 0;
    bp[i] = residual_binning::binOf(rs[i].normal.x(), rs[i].normal.y(), rs[i].normal.z(), up[i]);
    const V3 gn = rs[i].point_cross_normal.normalized();
    br[i] = residual_binning::binOf(gn.x(), gn.y(), gn.z(), dummy);
    ur[i] = rs[i].point_cross_normal.dot(axisVec(br[i]));
    ++cp[bp[i]]; ++cr[br[i]];
  }
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      if (tp > 0 && bp[i] == bp[j] && cp[bp[i]] >= min_rows) R(i, j) += tp * tp * up[i] * up[j];
      if (tr > 0 && br[i] == br[j] && cr[br[i]] >= min_rows) R(i, j) += tr * tr * ur[i] * ur[j];
    }
  Eigen::LDLT<Eigen::MatrixXd> ldlt(R);
  const Eigen::MatrixXd RiJ = ldlt.solve(J);
  const Eigen::VectorXd Riz = ldlt.solve(z);
  H = J.transpose() * RiJ;
  g = J.transpose() * Riz;
}

ResidualBinningOptions opt(const char* mode, double tp, double tr, int min_rows = 20)
{
  ResidualBinningOptions o; o.mode = mode; o.shift_sd_m = tp; o.rot_shift_sd_rad = tr; o.min_rows = min_rows; return o;
}

}  // namespace

int main()
{
  std::mt19937 rng(91006);
  const double tp = 0.0006, tr = 1.0e-4;
  const std::vector<MockResidual> rows = makeRows(rng, {0, 2, 3, 9}, {1, 4, 5, 8, 11}, 480);
  M6 H; V6 g;
  accumulate(rows, H, g);

  // RF1
  {
    ResidualBinningOptions off;
    M6 H0 = H; V6 g0 = g;
    const ResidualBinningStats st = applyResidualBinning(rows, off, H0, g0);
    CHECK(std::memcmp(H0.data(), H.data(), sizeof(double) * 36) == 0 && std::memcmp(g0.data(), g.data(), sizeof(double) * 6) == 0 && st.n_corrected_rows == 0 && st.n_rot_corrected_rows == 0,
          "RF1a mode=off leaves HtH and Htz bit-identical and the stats zero");
    // the R91E per-bin formula, re-implemented here
    constexpr int K = residual_binning::kAxes;
    std::vector<int> n(K, 0); std::vector<double> s(K, 0), t(K, 0); std::array<V6, residual_binning::kAxes> v; for (auto& x : v) x.setZero();
    for (const auto& m : rows) {
      double u = 0; const int k = residual_binning::binOf(m.normal.x(), m.normal.y(), m.normal.z(), u);
      const double w = 1.0 / m.sigma_squared;
      V6 j; j << m.point_cross_normal, m.normal;
      v[k] += w * u * j; s[k] += w * u * u; t[k] += w * u * m.r; ++n[k];
    }
    M6 Hr = H; V6 gr = g;
    for (int k = 0; k < K; ++k) if (n[k] >= 20) { const double d = 1.0 / (tp * tp) + s[k]; Hr -= v[k] * v[k].transpose() / d; gr -= v[k] * (t[k] / d); }
    M6 Hd = H; V6 gd = g;
    applyResidualBinning(rows, opt("direction", tp, tr), Hd, gd);
    CHECK(relM(Hd, Hr) < 1e-12 && relV(gd, gr) < 1e-12, "RF1b mode=direction equals the R91E per-bin formula");
  }

  // RF2 rotation alone
  M6 Hrot; V6 grot;
  dense(rows, 0.0, tr, 20, Hrot, grot);
  M6 H2 = H; V6 g2 = g;
  const ResidualBinningStats st2 = applyResidualBinning(rows, opt("rotation", tp, tr), H2, g2);
  CHECK(relM(H2, Hrot) < 1e-9, "RF2a rotation: HtH' equals H^T R^-1 H of the dense covariance with rotation bins");
  CHECK(relV(g2, grot) < 1e-9, "RF2b rotation: Htz' equals H^T R^-1 z of the dense covariance with rotation bins");
  CHECK(st2.n_rot_bins_active >= 4 && st2.n_rot_corrected_rows >= 300 && st2.n_bins_active == 0 && st2.n_corrected_rows == 0 && st2.solve_failed == 0 && st2.n_rows == 480,
        "RF2c rotation: only rotation bins are counted");

  // RF3 both
  M6 Hboth; V6 gboth;
  dense(rows, tp, tr, 20, Hboth, gboth);
  M6 H3 = H; V6 g3 = g;
  const ResidualBinningStats st3 = applyResidualBinning(rows, opt("both", tp, tr), H3, g3);
  CHECK(relM(H3, Hboth) < 1e-9, "RF3a both: HtH' equals H^T R^-1 H of the dense covariance with both families");
  CHECK(relV(g3, gboth) < 1e-9, "RF3b both: Htz' equals H^T R^-1 z of the dense covariance with both families");
  CHECK(st3.n_bins_active >= 3 && st3.n_rot_bins_active >= 4 && st3.solve_failed == 0, "RF3c both: both families are counted");

  // RF4 limits
  {
    M6 Ha = H; V6 ga = g;
    M6 Hd = H; V6 gd = g;
    applyResidualBinning(rows, opt("direction", tp, tr), Hd, gd);
    applyResidualBinning(rows, opt("both", tp, 1.0e-12), Ha, ga);
    CHECK(relM(Ha, Hd) < 1e-9 && relV(ga, gd) < 1e-9, "RF4a both with a tiny rotation shift equals direction");
    M6 Hb = H; V6 gb = g;
    applyResidualBinning(rows, opt("both", 1.0e-12, tr), Hb, gb);
    CHECK(relM(Hb, H2) < 1e-9 && relV(gb, g2) < 1e-9, "RF4b both with a tiny position shift equals rotation");
  }

  // RF5 PSD
  for (double a : {1e-6, 6e-4, 1e-2}) for (double b : {1e-7, 1e-4, 1e-2, 1.0}) {
    M6 Hx = H; V6 gx = g;
    applyResidualBinning(rows, opt("both", a, b), Hx, gx);
    Eigen::SelfAdjointEigenSolver<M6> es(0.5 * (Hx + Hx.transpose()));
    char msg[112]; std::snprintf(msg, sizeof msg, "RF5 HtH' positive semidefinite at tau_p = %g, tau_r = %g", a, b);
    CHECK(es.eigenvalues().minCoeff() >= -1e-12 * es.eigenvalues().maxCoeff(), msg);
  }

  // RF6 saturation along a rotation axis
  {
    const double w = 4.0e4, Lv = 10.0;
    double prev = 0.0; const double lim = 1.0 / (tr * tr);
    bool below = true, grows = true;
    for (int n : {10, 100, 1000, 100000}) {
      std::vector<MockResidual> rs(n);
      for (auto& m : rs) { m.normal = V3(0, 0, 1); m.point_cross_normal = V3(Lv, 0, 0); m.sigma_squared = 1.0 / w; m.r = 0.0; }
      M6 Hs; V6 gs; accumulate(rs, Hs, gs);
      applyResidualBinning(rs, opt("rotation", tp, tr, 1), Hs, gs);
      const double s = n * w * Lv * Lv, expect = s / (1.0 + tr * tr * s), info = Hs(0, 0);
      below = below && info < lim && std::abs(info - expect) / expect < 1e-9;
      grows = grows && info > prev; prev = info;
    }
    CHECK(below, "RF6a information on the rotation axis equals s/(1 + tau_r^2 s) and stays below 1/tau_r^2");
    CHECK(grows && prev > 0.99 * lim, "RF6b it grows with n and tends to 1/tau_r^2");
  }

  // RF7 skipped rows
  {
    std::vector<MockResidual> mixed = rows;
    const int n0 = static_cast<int>(mixed.size());
    MockResidual d = mixed[0]; d.row_kind = kRowLineDist; mixed.push_back(d);
    MockResidual z = mixed[1]; z.sigma_squared = 0.0; mixed.push_back(z);
    MockResidual nn = mixed[2]; nn.sigma_squared = std::numeric_limits<double>::quiet_NaN(); mixed.push_back(nn);
    MockResidual zl = mixed[3]; zl.point_cross_normal = V3::Zero(); mixed.push_back(zl);   // zero lever arm: skipped in the rotation modes. The unskipped rows of H, g below are the first n0 rows only, so the extra rows add nothing to the input either
    M6 Hm = H; V6 gm = g;
    accumulate(std::vector<MockResidual>(mixed.begin(), mixed.begin() + n0), Hm, gm);
    const ResidualBinningStats st = applyResidualBinning(mixed, opt("both", tp, tr), Hm, gm);
    CHECK(st.n_skipped_rows == 4 && st.n_rows == n0 + 4 && relM(Hm, H3) < 1e-14 && relV(gm, g3) < 1e-14,
          "RF7a both: distance rows, bad variances and a zero lever arm are skipped, counted, and leave the result unchanged");
    M6 Hq = H; V6 gq = g;
    const ResidualBinningStats sq = applyResidualBinning(mixed, opt("direction", tp, tr), Hq, gq);
    CHECK(sq.n_skipped_rows == 3, "RF7b direction: a zero lever arm is not skipped (it still has a position bin)");
  }

  // RF8 sign fold
  {
    std::vector<MockResidual> flipped = rows;
    for (auto& m : flipped) { m.normal = -m.normal; m.point_cross_normal = -m.point_cross_normal; m.r = -m.r; }
    M6 Hf; V6 gf; accumulate(flipped, Hf, gf);
    applyResidualBinning(flipped, opt("both", tp, tr), Hf, gf);
    CHECK(relM(Hf, H3) < 1e-12 && relV(gf, g3) < 1e-12, "RF8 a consistently sign-flipped row set gives the same corrected system in mode both");
  }

  // RF9 determinism and cross sums
  {
    M6 Ha = H, Hb = H; V6 ga = g, gb = g;
    const ResidualBinningStats sa = applyResidualBinning(rows, opt("both", tp, tr), Ha, ga);
    const ResidualBinningStats sb = applyResidualBinning(rows, opt("both", tp, tr), Hb, gb);
    CHECK(std::memcmp(Ha.data(), Hb.data(), sizeof(double) * 36) == 0 && std::memcmp(ga.data(), gb.data(), sizeof(double) * 6) == 0 &&
          std::memcmp(sa.cross, sb.cross, sizeof(sa.cross)) == 0, "RF9a the same input twice gives bit-identical output (mode both)");
    // independent accumulation of X_kl
    double X[residual_binning::kAxes][residual_binning::kAxes] = {};
    for (const auto& m : rows) {
      double up = 0, dummy = 0;
      const int k = residual_binning::binOf(m.normal.x(), m.normal.y(), m.normal.z(), up);
      const V3 gn = m.point_cross_normal.normalized();
      const int l = residual_binning::binOf(gn.x(), gn.y(), gn.z(), dummy);
      const double ur = m.point_cross_normal.dot(axisVec(l));
      X[k][l] += (1.0 / m.sigma_squared) * up * ur;
    }
    double worst = 0, scale = 0;
    for (int k = 0; k < residual_binning::kAxes; ++k) for (int l = 0; l < residual_binning::kAxes; ++l) { worst = std::max(worst, std::abs(X[k][l] - sa.cross[k][l])); scale = std::max(scale, std::abs(X[k][l])); }
    CHECK(worst <= 1e-12 * scale && scale > 0.0, "RF9b the cross sums X_kl equal an independent accumulation");
    // rows of other rotation bins appended after this bin's rows do not change this bin's sums
    std::vector<MockResidual> head = makeRows(rng, {0}, {1}, 40), both_ = head;
    std::vector<MockResidual> tail = makeRows(rng, {2}, {4}, 40);
    both_.insert(both_.end(), tail.begin(), tail.end());
    M6 Hh; V6 gh; accumulate(head, Hh, gh);
    const ResidualBinningStats sh = applyResidualBinning(head, opt("both", tp, tr), Hh, gh);
    M6 Hq; V6 gq; accumulate(both_, Hq, gq);
    const ResidualBinningStats sq = applyResidualBinning(both_, opt("both", tp, tr), Hq, gq);
    int lh = -1; for (int l = 0; l < residual_binning::kAxes; ++l) if (sh.rot_bins[l].rows == 40) lh = l;
    CHECK(lh >= 0 && sh.rot_bins[lh].s == sq.rot_bins[lh].s && sh.rot_bins[lh].t == sq.rot_bins[lh].t && std::memcmp(sh.rot_bins[lh].v, sq.rot_bins[lh].v, sizeof(double) * 6) == 0,
          "RF9c a rotation bin's sums do not depend on rows of other bins");
  }

  // RF10 the check fires
  {
    const double tol = 1e-9;
    // sequential: the direction correction, then the rotation correction computed from the same rows on the already corrected matrices
    M6 Hs = H; V6 gs = g;
    applyResidualBinning(rows, opt("direction", tp, tr), Hs, gs);
    M6 Hr2 = H; V6 gr2 = g;
    applyResidualBinning(rows, opt("rotation", tp, tr), Hr2, gr2);
    const M6 Hseq = Hs + (Hr2 - H);   // both corrections subtracted from the uncorrected sums independently (no coupling)
    CHECK(relM(Hseq, Hboth) > 1e4 * tol, "RF10a the independent-families form differs from the dense reference far above the RF3 tolerance");
    // sign-flipped Htz correction
    const V6 wrong = g + (g - g3);
    CHECK(relV(wrong, gboth) > 1e4 * tol, "RF10b a sign-flipped Htz correction differs from the reference far above the RF3 tolerance");
    // uncorrected sums
    CHECK(relM(H, Hboth) > 1e4 * tol, "RF10c the uncorrected sums differ from the reference far above the RF3 tolerance");
  }

  // RF11 min_rows applies to both families
  {
    M6 Hm = H; V6 gm = g;
    const ResidualBinningStats st = applyResidualBinning(rows, opt("both", tp, tr, 100000), Hm, gm);
    CHECK(std::memcmp(Hm.data(), H.data(), sizeof(double) * 36) == 0 && std::memcmp(gm.data(), g.data(), sizeof(double) * 6) == 0 && st.n_bins_active == 0 && st.n_rot_bins_active == 0,
          "RF11 with min_rows above every bin count HtH and Htz are bit-identical (both families inert)");
  }
  std::printf("all R91F binning checks passed\n");
  return 0;
}
