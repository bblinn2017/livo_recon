// R65 unit test (F-169): per-point pose covariance (StateGroup::poseCovAtPoint). R84 (F-521): the propagated mode, its backward sweep
// and the profile lookup tests (PC2 to PC4, PC6) were removed with the feature.
// Mirrors test_gravity_s2.cpp's pattern (ros first, then #define private public) so it stays ROS-runtime-free.
//   PC1 defaults (posterior): poseCovAtPoint(p,true) and (p,false) equal poseCovAt(p) exactly; poseCovAt equals an independent J P J^T.
//   PC5 insertion_source=prior: for_insertion=true uses the stored prior (R,P) block; for_insertion=false uses cov_.
#include <ros/ros.h>
#define private public
#include "livo_recon/utils/state/state.h"
#undef private

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
    CHECK((s.poseCovAtPoint(p, true) - ref).norm() == 0.0, "PC1a defaults, insertion path bit-identical to poseCovAt");
    CHECK((s.poseCovAtPoint(p, false) - ref).norm() == 0.0, "PC1b defaults, gate path bit-identical to poseCovAt");
    // independent check of poseCovAt itself against J P J^T
    const M6 Pb = s.cov_.block<6,6>(0,0);
    CHECK((ref - jbj(s.rot_, p, Pb)).norm() < 1e-12, "PC1c poseCovAt equals independent J P J^T");
  }
  {  // PC5
    StateGroup s = makeState(rng);
    s.setPointCovInsertionPrior(true);
    Eigen::MatrixXd prior = Eigen::MatrixXd::Identity(s.dimState(), s.dimState());
    const M6 Pp = randPsd6(rng, 5e-3);
    prior.block<6,6>(0,0) = Pp;
    s.setPriorPoseBlock(prior);
    const M3D gate = s.poseCovAtPoint(p, false);
    const M3D ins  = s.poseCovAtPoint(p, true);
    CHECK((gate - s.poseCovAt(p)).norm() == 0.0, "PC5a gate uses cov_ (state's own block)");
    CHECK((ins - jbj(s.rot_, p, Pp)).norm() < 1e-12, "PC5b insertion uses the stored prior (R,P) block");
  }
  std::printf("ALL PASS\n");
  return 0;
}
