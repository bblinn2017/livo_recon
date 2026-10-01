// R70 unit test (F-320): block information scale (include/livo_recon/lio/info_scale.h). Isolated algebraic identities only.
//   IS1 mode=off leaves HtH and Htz bit-identical.
//   IS2 F_rot = F_pos = F gives HtH/F and Htz/F (the global information scale), to 1e-12 relative.
//   IS3 block: HtH' = S HtH S exactly, and the measurement-only solution pinv(HtH) Htz is unchanged (to 1e-9 relative).
//   IS4 HtH' stays positive semidefinite (smallest eigenvalue >= -1e-12 * largest).
//   IS5 rank-deficient information (all rows in a 2-D subspace): discarded_modes == 4 and Htz' = HtH' x_ls with x_ls the
//       original minimum-norm measurement-only solution.
//   IS6 the check fires: the naive alternative Htz' = S Htz (no solution-preserving step) does NOT satisfy IS2, by a wide
//       margin, so IS2 would fail for that wrong implementation.
#include "livo_recon/lio/info_scale.h"

#include <cmath>
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

namespace {

// Information as the residual accumulator builds it: sum of w h h^T and sum of w r h.
void build(std::mt19937& rng, int n, int rank_limit, M6& H, V6& g)
{
  std::normal_distribution<double> N(0, 1);
  std::uniform_real_distribution<double> U(0.5, 2.0);
  H.setZero(); g.setZero();
  Eigen::Matrix<double, 6, 6> basis = Eigen::Matrix<double, 6, 6>::Zero();
  for (int k = 0; k < 6; ++k) for (int j = 0; j < 6; ++j) basis(k, j) = N(rng);
  for (int i = 0; i < n; ++i) {
    V6 c = V6::Zero();
    for (int k = 0; k < rank_limit; ++k) c(k) = N(rng);
    const V6 h = basis * c;
    const double w = U(rng), r = N(rng);
    H += w * h * h.transpose();
    g += w * r * h;
  }
}

double relErr(const V6& a, const V6& b) { return (a - b).norm() / std::max(b.norm(), 1e-300); }

}  // namespace

int main()
{
  std::mt19937 rng(12345);
  M6 H; V6 g;
  build(rng, 60, 6, H, g);

  { InfoScaleOptions o;   // mode off
    M6 H0 = H; V6 g0 = g;
    const InfoScaleStats st = applyBlockInfoScale(o, H0, g0);
    CHECK(H0 == H && g0 == g && st.discarded_modes == 0, "IS1 mode=off leaves HtH/Htz bit-identical"); }

  const double F = 7.5;
  { InfoScaleOptions o; o.mode = "block"; o.rotation = F; o.position = F;
    M6 H1 = H; V6 g1 = g;
    applyBlockInfoScale(o, H1, g1);
    CHECK((H1 - H / F).norm() / (H / F).norm() < 1e-12, "IS2a F_rot=F_pos=F: HtH/F");
    CHECK(relErr(g1, g / F) < 1e-9, "IS2b F_rot=F_pos=F: Htz/F"); }

  { InfoScaleOptions o; o.mode = "block"; o.rotation = 9.0; o.position = 2.0;
    M6 H1 = H; V6 g1 = g;
    applyBlockInfoScale(o, H1, g1);
    V6 s; s.head<3>().setConstant(1.0 / 3.0); s.tail<3>().setConstant(1.0 / std::sqrt(2.0));
    const M6 Sm = s.asDiagonal();
    const M6 Hexp = Sm * H * Sm;
    CHECK((H1 - Hexp).norm() / Hexp.norm() < 1e-12, "IS3a HtH' = S HtH S");
    const V6 x0 = H.ldlt().solve(g), x1 = H1.ldlt().solve(g1);
    CHECK(relErr(x1, x0) < 1e-9, "IS3b measurement-only solution unchanged");
    Eigen::SelfAdjointEigenSolver<M6> es(H1);
    CHECK(es.eigenvalues().minCoeff() >= -1e-12 * es.eigenvalues().maxCoeff(), "IS4 HtH' positive semidefinite"); }

  { M6 Hd; V6 gd;
    build(rng, 40, 2, Hd, gd);   // rank 2
    InfoScaleOptions o; o.mode = "block"; o.rotation = 5.0; o.position = 3.0;
    M6 H1 = Hd; V6 g1 = gd;
    const InfoScaleStats st = applyBlockInfoScale(o, H1, g1);
    CHECK(st.discarded_modes == 4, "IS5a rank-2 information: 4 modes discarded");
    // g' must equal HtH' times the original minimum-norm measurement-only solution
    Eigen::CompleteOrthogonalDecomposition<M6> c0(Hd);
    const V6 x_ls = c0.solve(gd);
    CHECK(relErr(H1 * x_ls, g1) < 1e-8, "IS5b rank-2: Htz' = HtH' x_ls (solution preserved)"); }

  { // IS6: the naive alternative must fail IS2 clearly
    V6 s; s.head<3>().setConstant(1.0 / std::sqrt(F)); s.tail<3>().setConstant(1.0 / std::sqrt(F));
    const V6 naive = s.asDiagonal() * g;
    CHECK(relErr(naive, g / F) > 0.5, "IS6 naive S*Htz differs from Htz/F by a wide margin (IS2b would catch it)"); }

  std::printf("ALL PASS\n");
  return 0;
}
