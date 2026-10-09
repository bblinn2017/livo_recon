// R89: exact-zero known blocks of P0 (include/livo_recon/lio/p0_pin.h). Eigen + std only; own check macro (does not depend on NDEBUG).
#include "livo_recon/lio/p0_pin.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

using namespace livo_recon;

static int fails = 0;
#define CK(...) do { if (!(__VA_ARGS__)) { std::printf("FAIL line %d: %s\n", __LINE__, #__VA_ARGS__); ++fails; } } while (0)

// The operation CalibProc::stabilizeP0() performed before R89, verbatim, on a matrix it owns.
static void oldStabilize(Eigen::MatrixXd& P, double floor, double& min_before, double& min_after)
{
  P = symmetrized(P);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(P);
  min_before = es.eigenvalues().minCoeff();
  if (min_before < floor)
  {
    const Eigen::VectorXd values = es.eigenvalues().cwiseMax(floor);
    P = es.eigenvectors() * values.asDiagonal() * es.eigenvectors().transpose();
    P = symmetrized(P);
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> check(P);
  min_after = check.eigenvalues().minCoeff();
  if (min_after < floor)
  {
    const double lift = floor - min_after + std::numeric_limits<double>::epsilon() * std::max(1.0, floor);
    P.diagonal().array() += lift;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> lifted(P);
    min_after = lifted.eigenvalues().minCoeff();
  }
}

// An 18 x 18 covariance in the production ordering theta[0:3] p[3:6] v[6:9] bg[9:12] ba[12:15] g[15:18], slightly indefinite.
static Eigen::MatrixXd makeP(unsigned seed)
{
  std::mt19937 g(seed);
  std::normal_distribution<double> n(0.0, 1.0);
  Eigen::MatrixXd A(18, 12);
  for (int i = 0; i < 18; ++i) for (int j = 0; j < 12; ++j) A(i, j) = 0.01 * n(g);
  Eigen::MatrixXd P = A * A.transpose();            // rank 12: six zero eigenvalues, rounding noise around zero
  P.diagonal().array() += 1e-6;
  P(3, 3) -= 3e-6;                                   // makes it slightly indefinite
  return symmetrized(P);
}

int main()
{
  const double floor = 1e-12;

  // P1: no known block => bit-identical to the pre-R89 operation.
  for (unsigned seed : {1u, 2u, 3u})
  {
    Eigen::MatrixXd P = makeP(seed), Q = P;
    double b0 = 0, a0 = 0;
    oldStabilize(Q, floor, b0, a0);
    const p0::StabilizeResult r = p0::stabilize(P, {}, floor);
    CK((P - Q).cwiseAbs().maxCoeff() == 0.0);
    CK(r.min_before == b0 && r.min_after == a0);
  }

  // P2: theta, p and v pinned at exact zero.
  std::vector<p0::KnownBlock> pins = {{0, 0.0}, {3, 0.0}, {6, 0.0}};
  const std::vector<int> pinned = p0::pinnedIndices(pins);
  CK(pinned.size() == 9 && pinned.front() == 0 && pinned.back() == 8);
  {
    Eigen::MatrixXd P = makeP(5);
    p0::applyKnownBlocks(P, pins);
    CK(P.topRows(9).cwiseAbs().maxCoeff() == 0.0);
    CK(P.leftCols(9).cwiseAbs().maxCoeff() == 0.0);
    // the free 9 x 9 block is untouched by the pinning
    const Eigen::MatrixXd freeBefore = P.bottomRightCorner(9, 9);
    p0::stabilize(P, pinned, floor);
    CK(P.topRows(9).cwiseAbs().maxCoeff() == 0.0);                 // exact zero, not 1e-12
    CK(P.leftCols(9).cwiseAbs().maxCoeff() == 0.0);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(P.bottomRightCorner(9, 9));
    CK(es.eigenvalues().minCoeff() >= floor * 0.999);              // floor acts on the free states
    // the free block equals what the old operation gives on that block alone
    Eigen::MatrixXd Q = freeBefore;
    double b0 = 0, a0 = 0;
    oldStabilize(Q, floor, b0, a0);
    CK((P.bottomRightCorner(9, 9) - Q).cwiseAbs().maxCoeff() == 0.0);
    CK((P - P.transpose()).cwiseAbs().maxCoeff() == 0.0);
    // the whole matrix is positive semi-definite
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> all(P);
    CK(all.eigenvalues().minCoeff() >= -1e-18);
  }

  // P3: applyKnownBlocks semantics.
  {
    Eigen::MatrixXd P = makeP(7);
    const Eigen::MatrixXd P0 = P;
    p0::applyKnownBlocks(P, {{0, -1.0}});                          // negative: not a known block
    CK((P - P0).cwiseAbs().maxCoeff() == 0.0);
    p0::applyKnownBlocks(P, {{3, 2.5e-7}});                        // positive constant: diagonal, no cross terms
    CK(P.block<3, 3>(3, 3) == 2.5e-7 * Eigen::Matrix3d::Identity());
    CK(P.block(3, 0, 3, 3).cwiseAbs().maxCoeff() == 0.0 && P.block(3, 6, 3, 12).cwiseAbs().maxCoeff() == 0.0);
    CK(P.block(0, 3, 3, 3).cwiseAbs().maxCoeff() == 0.0 && P.block(6, 3, 12, 3).cwiseAbs().maxCoeff() == 0.0);
    CK((P.bottomRightCorner(12, 12) - P0.bottomRightCorner(12, 12)).cwiseAbs().maxCoeff() == 0.0);   // other states untouched
    CK(p0::pinnedIndices({{3, 2.5e-7}, {0, -1.0}}).empty());      // only an exact zero pins
  }

  // P4 (toy propagation, not the production F and Q): a pinned position block is positive only after the second IMU step, and the
  // covariance is invertible from then on. This is why nothing may invert P0 or the covariance after a single step.
  {
    const double dt = 0.005;
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(9, 9);
    F.block<3, 3>(3, 6) = dt * Eigen::Matrix3d::Identity();
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(9, 9);
    Q.block<3, 3>(0, 0) = 1e-8 * Eigen::Matrix3d::Identity();       // gyro noise on attitude
    Q.block<3, 3>(6, 6) = 1e-6 * Eigen::Matrix3d::Identity();       // accelerometer noise on velocity
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(9, 9);
    P = symmetrized(Eigen::MatrixXd(F * P * F.transpose() + Q));
    CK(P.block<3, 3>(3, 3).cwiseAbs().maxCoeff() == 0.0);           // after one step: position still exactly zero
    CK(P.block<3, 3>(6, 6)(0, 0) > 0.0 && P.block<3, 3>(0, 0)(0, 0) > 0.0);
    P = symmetrized(Eigen::MatrixXd(F * P * F.transpose() + Q));
    CK(P.block<3, 3>(3, 3)(0, 0) > 0.0);                            // after two steps: positive
    Eigen::LDLT<Eigen::MatrixXd> ldlt(P);
    CK(ldlt.info() == Eigen::Success && ldlt.isPositive());
    CK(P.inverse().allFinite());
  }

  if (fails == 0) std::printf("test_p0_pin_r89: all checks passed\n");
  return fails == 0 ? 0 : 1;
}
