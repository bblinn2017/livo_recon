// R73 unit test (F-419): the symmetrize helper (include/livo_recon/utils/algo/symmetrize.h).
//   SY1 symmetrized(X) is exactly symmetric (asymmetry 0) and equals 0.5 (X + X^T).
//   SY2 symmetrized(X) of an already symmetric X returns X bit for bit.
//   SY3 aliasing: P = symmetrized(F*P*F^T + Q) (destination also on the right) equals the evaluated reference.
//   SY4 a 2000-step covariance chain P <- sym(F P F^T + Q) stays exactly symmetric; the same chain with the plain product is run too and its
//       asymmetry is printed (not asserted: it can legitimately be 0 on some builds).
//   SY5 fixed-size 18x18 and dynamic matrices both work (the two types the code base uses).
//   SY6 symmetrizeInPlace on a block-assigned matrix (calibration P0 style).
#include "livo_recon/utils/algo/symmetrize.h"

#include <cstdio>
#include <cstdlib>
#include <random>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

int main()
{
  std::mt19937 rng(11);
  std::normal_distribution<double> N(0, 1);
  const int n = 18;
  Eigen::MatrixXd X(n, n);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) X(i, j) = N(rng);
  const Eigen::MatrixXd S = symmetrized(X);
  CHECK(asymmetryMax(S) == 0.0, "SY1 symmetrized(X) is exactly symmetric");
  CHECK((S - 0.5 * (X + X.transpose())).cwiseAbs().maxCoeff() == 0.0, "SY1 equals 0.5 (X + X^T)");
  const Eigen::MatrixXd S2 = symmetrized(S);
  CHECK((S2 - S).cwiseAbs().maxCoeff() == 0.0, "SY2 symmetric input returned bit for bit");

  Eigen::MatrixXd B(n, n);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) B(i, j) = N(rng);
  Eigen::MatrixXd P = 1e-2 * (B * B.transpose()) + 1e-3 * Eigen::MatrixXd::Identity(n, n);
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(n, n) + 1e-2 * B;
  const Eigen::MatrixXd Q = 1e-6 * Eigen::MatrixXd::Identity(n, n);
  const Eigen::MatrixXd ref = symmetrized(Eigen::MatrixXd(F * P * F.transpose() + Q));
  P = symmetrized(F * P * F.transpose() + Q);
  CHECK((P - ref).cwiseAbs().maxCoeff() == 0.0, "SY3 aliasing-safe assignment");

  Eigen::MatrixXd Ps = P, Pp = P;
  for (int k = 0; k < 2000; ++k) {
    Ps = symmetrized(F * Ps * F.transpose() + Q);
    Pp = F * Pp * F.transpose() + Q;
    F = Eigen::MatrixXd::Identity(n, n) + 1e-3 * (F - Eigen::MatrixXd::Identity(n, n));
  }
  std::printf("[INFO] SY4 plain-product chain asymmetry after 2000 steps: %.3e\n", asymmetryMax(Pp));
  CHECK(asymmetryMax(Ps) == 0.0, "SY4 symmetrised chain stays exactly symmetric over 2000 steps");

  Eigen::Matrix<double, 18, 18> Xf = Eigen::Matrix<double, 18, 18>::Random();
  const Eigen::Matrix<double, 18, 18> Sf = symmetrized(Xf);
  CHECK(asymmetryMax(Sf) == 0.0, "SY5 fixed-size 18x18");
  Eigen::MatrixXd P0 = Eigen::MatrixXd::Zero(n, n);
  P0.block<3, 3>(0, 3) = Eigen::Matrix3d::Constant(0.3);
  symmetrizeInPlace(P0);
  CHECK(asymmetryMax(P0) == 0.0 && P0(0, 3) == 0.15 && P0(3, 0) == 0.15, "SY6 in-place on a block-assigned matrix");
  std::printf("ALL PASS\n");
  return 0;
}
