// R65 unit test (F-169): per-point pose covariance (StateGroup::poseCovAtPoint) and the backward sweep relCovBlocksBackward.
// Mirrors test_gravity_s2.cpp's pattern (ros first, then #define private public) so it stays ROS-runtime-free.
//   PC1 defaults (end_pose, posterior): poseCovAtPoint(p,t,true) and (p,t,false) equal poseCovAt(p) exactly.
//   PC2 propagated with an empty profile equals poseCovAt(p).
//   PC3 propagated: result = poseCovAt(p) + J B J^T, J built independently from -R[p]x (rotation) and I (position),
//       B the profile block selected by time; compared to 1e-12.
//   PC4 profile lookup: t before the first head -> block 0; t in [head_k, head_{k+1}) -> block k; t after the last head -> last.
//   PC5 insertion_source=prior: for_insertion=true uses the stored prior (R,P) block; for_insertion=false uses cov_.
//   PC6 relCovBlocksBackward equals an independent forward simulation (S <- F S F^T + Q from step k to the end) to 1e-12,
//       and the last step's block equals Q_last.
#include <ros/ros.h>
#define private public
#include "livo_recon/utils/state/state.h"
#undef private
#include "livo_recon/utils/state/rel_cov.h"

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

namespace {

M3D skew(const V3D& v) { M3D k; k << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0; return k; }

M6 randPsd6(std::mt19937& rng, double scale)
{
  std::normal_distribution<double> N(0, 1);
  M6 A; for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) A(i, j) = N(rng);
  return scale * (A * A.transpose() + 0.1 * M6::Identity());
}

// independent: J = [ -R[p]x , I ] on (rotation, position); returns J B J^T
M3D jbj(const M3D& R, const V3D& p, const M6& B)
{
  Eigen::Matrix<double, 3, 6> J; J.block<3,3>(0,0) = -R * skew(p); J.block<3,3>(0,3) = M3D::Identity();
  return J * B * J.transpose();
}

StateGroup makeState(std::mt19937& rng)
{
  StateGroup s;
  s.cov_ = Eigen::MatrixXd::Identity(s.dimState(), s.dimState());
  s.cov_.block<6,6>(0,0) = randPsd6(rng, 1e-3);
  Eigen::AngleAxisd aa(0.7, V3D(0.3, -0.5, 0.8).normalized());
  s.rot_ = aa.toRotationMatrix();
  return s;
}

}  // namespace

int main()
{
  std::mt19937 rng(7);
  const V3D p(1.3, -0.4, 2.1);

  {  // PC1
    StateGroup s = makeState(rng);
    const M3D ref = s.poseCovAt(p);
    CHECK((s.poseCovAtPoint(p, 0.05, true) - ref).norm() == 0.0, "PC1a defaults, insertion path bit-identical to poseCovAt");
    CHECK((s.poseCovAtPoint(p, 0.05, false) - ref).norm() == 0.0, "PC1b defaults, gate path bit-identical to poseCovAt");
    // independent check of poseCovAt itself against J P J^T
    const M6 Pb = s.cov_.block<6,6>(0,0);
    CHECK((ref - jbj(s.rot_, p, Pb)).norm() < 1e-12, "PC1c poseCovAt equals independent J P J^T");
  }
  {  // PC2
    StateGroup s = makeState(rng);
    s.setPointCovModel(true, false);
    const M3D ref = s.poseCovAt(p);
    CHECK((s.poseCovAtPoint(p, 0.05, false) - ref).norm() == 0.0, "PC2 propagated with empty profile equals poseCovAt");
  }
  M6 B0 = randPsd6(rng, 1e-4), B1 = randPsd6(rng, 3e-4), B2 = randPsd6(rng, 1e-5);
  {  // PC3, PC4
    StateGroup s = makeState(rng);
    s.setPointCovModel(true, false);
    s.setRelCovProfile({0.00, 0.05, 0.08}, {B0, B1, B2});
    const M3D base = s.poseCovAt(p);
    CHECK((s.poseCovAtPoint(p, 0.02, false) - (base + jbj(s.rot_, p, B0))).norm() < 1e-12, "PC3 propagated = base + J B0 J^T at t=0.02");
    CHECK((s.poseCovAtPoint(p, 0.02, true) - (base + jbj(s.rot_, p, B0))).norm() < 1e-12, "PC3b same at insertion (posterior base)");
    CHECK((s.poseCovAtPoint(p, -1.0, false) - (base + jbj(s.rot_, p, B0))).norm() < 1e-12, "PC4a t before first head -> block 0");
    CHECK((s.poseCovAtPoint(p, 0.05, false) - (base + jbj(s.rot_, p, B1))).norm() < 1e-12, "PC4b t == head_1 -> block 1");
    CHECK((s.poseCovAtPoint(p, 0.079, false) - (base + jbj(s.rot_, p, B1))).norm() < 1e-12, "PC4c t inside step 1 -> block 1");
    CHECK((s.poseCovAtPoint(p, 0.5, false) - (base + jbj(s.rot_, p, B2))).norm() < 1e-12, "PC4d t after last head -> last block");
  }
  {  // PC5
    StateGroup s = makeState(rng);
    s.setPointCovModel(false, true);
    Eigen::MatrixXd prior = Eigen::MatrixXd::Identity(s.dimState(), s.dimState());
    const M6 Pp = randPsd6(rng, 5e-3);
    prior.block<6,6>(0,0) = Pp;
    s.setPriorPoseBlock(prior);
    const M3D gate = s.poseCovAtPoint(p, 0.02, false);
    const M3D ins  = s.poseCovAtPoint(p, 0.02, true);
    CHECK((gate - s.poseCovAt(p)).norm() == 0.0, "PC5a gate uses cov_ (state's own block)");
    CHECK((ins - jbj(s.rot_, p, Pp)).norm() < 1e-12, "PC5b insertion uses the stored prior (R,P) block");
  }
  {  // PC6
    const int dim = 12, n = 5, idxR = 0;
    std::normal_distribution<double> N(0, 1);
    std::vector<Eigen::MatrixXd> F(n), Q(n);
    for (int k = 0; k < n; ++k) {
      F[k] = Eigen::MatrixXd::Identity(dim, dim) + 0.05 * Eigen::MatrixXd::NullaryExpr(dim, dim, [&]() { return N(rng); });
      Eigen::MatrixXd A = Eigen::MatrixXd::NullaryExpr(dim, dim, [&]() { return N(rng); });
      Q[k] = 1e-4 * (A * A.transpose());
    }
    const auto blocks = relCovBlocksBackward(F, Q, idxR);
    double worst = 0.0;
    for (int k = 0; k < n; ++k) {
      Eigen::MatrixXd S = Eigen::MatrixXd::Zero(dim, dim);
      for (int j = k; j < n; ++j) S = F[j] * S * F[j].transpose() + Q[j];
      worst = std::max(worst, (blocks[k] - S.block<6,6>(idxR, idxR)).norm());
    }
    CHECK(worst < 1e-12, "PC6a backward sweep equals independent forward simulation for every k");
    CHECK((blocks[n - 1] - Q[n - 1].block<6,6>(idxR, idxR)).norm() < 1e-14, "PC6b last block equals Q_last");
  }
  std::printf("ALL PASS\n");
  return 0;
}
