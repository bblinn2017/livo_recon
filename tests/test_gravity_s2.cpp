// R63 coding-agent addition (not part of the planning agent's patch, labelled separately per
// R63_INSTRUCTIONS.md section 2): three cheap checks for the s2 gravity model added in R63.
// #define private public is used instead of a ros::NodeHandle/loadParameters() path so this
// stays ROS-free like the other tests in this file's family -- gravity_model_s2_ has no public
// setter (only loadParameters(), which needs a live NodeHandle). ros/ros.h (and everything it
// transitively pulls in, including <sstream>) must be fully included BEFORE the macro is active,
// or libstdc++'s own private/public-sensitive nested classes (e.g. basic_stringbuf's
// __xfer_bufptrs) fail to redeclare consistently -- state.h's own #include of it is then a no-op
// via the normal include guard.
#include <ros/ros.h>
#define private public
#include "livo_recon/utils/state/state.h"
#undef private

#include <cmath>
#include <cstdio>
#include <cstdlib>

// Every check is a CHECK() (not compiled out by NDEBUG, unlike assert()).
#define CHECK(c)                                                                        \
  do {                                                                                  \
    if (!(c)) {                                                                         \
      std::fprintf(stderr, "CHECK FAILED %s:%d: %s\n", __FILE__, __LINE__, #c);         \
      std::exit(1);                                                                     \
    }                                                                                   \
  } while (0)

namespace {
using namespace livo_recon;

StateGroup makeS2State()
{
  StateGroup s;
  s.gravity_model_s2_ = true;
  s.gravity_ = V3D(0.1, -0.2, -9.79);  // an arbitrary non-axis-aligned gravity vector
  s.resetGravityBasis();
  s.cov_ = Eigen::MatrixXd::Identity(s.dimState(), s.dimState());
  return s;
}

// (a) after applyDelta with random 2-vectors, |g| is unchanged to 1e-12 and B stays orthonormal
// and orthogonal to g.
void checkApplyDeltaPreservesNormAndBasis()
{
  StateGroup s = makeS2State();
  CHECK(s.dimState() == 17);
  const double g0_norm = s.gravity_.norm();

  std::srand(12345);
  for (int trial = 0; trial < 20; ++trial) {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(s.dimState());
    // gravity error coords are the last gravDim()=2 entries.
    const int g_off = s.dimState() - s.gravDim();
    dx(g_off + 0) = 0.05 * (2.0 * std::rand() / RAND_MAX - 1.0);
    dx(g_off + 1) = 0.05 * (2.0 * std::rand() / RAND_MAX - 1.0);
    s.applyDelta(dx);

    CHECK(std::abs(s.gravity_.norm() - g0_norm) < 1e-12);
    const auto& B = s.gravity_basis_;
    // orthonormal
    CHECK(std::abs(B.col(0).squaredNorm() - 1.0) < 1e-10);
    CHECK(std::abs(B.col(1).squaredNorm() - 1.0) < 1e-10);
    CHECK(std::abs(B.col(0).dot(B.col(1))) < 1e-10);
    // orthogonal to g
    CHECK(std::abs(B.col(0).dot(s.gravity_)) < 1e-9);
    CHECK(std::abs(B.col(1).dot(s.gravity_)) < 1e-9);
  }
}

// (b) boxminusFromPropagat(x boxplus delta) returns delta to 1e-9 for the gravity block (small
// delta), s2 mode.
void checkBoxminusInvertsApplyDeltaForGravity()
{
  StateGroup propagat = makeS2State();
  StateGroup x = propagat;  // copy: same gravity_/basis_ before any delta is applied

  Eigen::VectorXd dx = Eigen::VectorXd::Zero(x.dimState());
  const int g_off = x.dimState() - x.gravDim();
  const double d0 = 0.003, d1 = -0.0021;
  dx(g_off + 0) = d0;
  dx(g_off + 1) = d1;
  x.applyDelta(dx);

  // boxminusFromPropagat's convention: `a.boxminusFromPropagat(b)` returns the delta FROM a TO
  // b (confirmed via the rotation term: Log(this->rot^T * propagat.rot) is +dx_R when
  // propagat.rot = this->rot * Exp(dx_R)). x = propagat_original boxplus dx, so recovering dx
  // means calling it on the ORIGINAL state with x as the "propagat" argument.
  Eigen::VectorXd back = propagat.boxminusFromPropagat(x);
  CHECK(back.size() == x.dimState());
  CHECK(std::abs(back(g_off + 0) - d0) < 1e-9);
  CHECK(std::abs(back(g_off + 1) - d1) < 1e-9);
}

// (c) gravityJacobian() matches a central finite difference of applyDelta on the gravity vector
// to 1e-6.
void checkGravityJacobianMatchesFiniteDifference()
{
  StateGroup base = makeS2State();
  Eigen::MatrixXd J = base.gravityJacobian();
  CHECK(J.rows() == 3 && J.cols() == base.gravDim());

  const double h = 1e-6;
  const int g_off = base.dimState() - base.gravDim();
  for (int k = 0; k < base.gravDim(); ++k) {
    StateGroup plus = base;
    Eigen::VectorXd dxp = Eigen::VectorXd::Zero(base.dimState());
    dxp(g_off + k) = h;
    plus.applyDelta(dxp);

    StateGroup minus = base;
    Eigen::VectorXd dxm = Eigen::VectorXd::Zero(base.dimState());
    dxm(g_off + k) = -h;
    minus.applyDelta(dxm);

    const V3D fd = (plus.gravity_ - minus.gravity_) / (2.0 * h);
    const V3D col = J.col(k);
    CHECK((fd - col).norm() < 1e-6);
  }
}

}  // namespace

int main()
{
  checkApplyDeltaPreservesNormAndBasis();
  checkBoxminusInvertsApplyDeltaForGravity();
  checkGravityJacobianMatchesFiniteDifference();
  std::printf("test_gravity_s2: all checks passed\n");
  return 0;
}
