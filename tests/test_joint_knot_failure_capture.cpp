// R72 integration test (F-398): the real solver keeps what failed. Linked against joint_knot_test_core (auto-registered by
// tests/joint_knots/CMakeLists.txt, which globs test_joint_knot_*.cpp).
//   FK1 solveJointKnotInformation with a negative-definite prior information throws the existing runtime_error, and lastNonPdCapture()
//       says "posterior_A" with the symmetrised A = P^-1 + Gamma_L bit-exactly, all eigenvalues equal to -1.
//   FK2 jointKnotLastFailedSolve() holds the partial solve: A as above, vec and rhs of trajectory dimension, K1 empty (it failed).
//   FK3 control (the check fires): the same call with a positive-definite prior does not throw and leaves the earlier capture untouched.
//   FK4 buildJointKnotImuPrior with a scan-head covariance made indefinite throws, and the capture is labelled "scan_head" with a
//       negative smallest eigenvalue.
#include "livo_recon/lio/failure_capture.h"
#include "livo_recon/lio/joint_knot_estimator.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

using namespace livo_recon;

namespace
{
int fail(const char* name)
{
  std::printf("[FAIL] %s\n", name);
  return 1;
}

std::vector<Pose6D> propagation()
{
  std::vector<Pose6D> poses;
  for (int i = 0; i <= 10; ++i) {
    const double t = 0.01 * i;
    poses.emplace_back(t, V3D::Zero(), V3D::Zero(), V3D::Zero(),
                       V3D(1.0, 0.0, 0.0), V3D(t, 0.0, 0.0),
                       Exp(V3D(0.0, 0.0, 0.2 * t)), 0.01);
  }
  return poses;
}
}  // namespace

int main()
{
  StateGroup state;
  state.setNoiseParams(V3D::Constant(1e-3), V3D::Constant(1e-4));
  state.setPropagatedState(Exp(V3D(0.0, 0.0, 0.02)), V3D(0.1, 0.0, 0.0), V3D(1.0, 0.0, 0.0));
  JointKnotTrajectory trajectory;
  if (!trajectory.initialize(propagation(), 0.1, 4, state)) return fail("fixture trajectory");
  const int D = trajectory.dim();

  JointKnotPrior bad;
  bad.P = Eigen::MatrixXd::Identity(D, D);
  bad.P_inverse = -Eigen::MatrixXd::Identity(D, D);
  bool threw = false;
  try {
    solveJointKnotInformation(trajectory, trajectory, bad, {}, {}, JointKnotResidualTime::Measurement);
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find("not positive definite") != std::string::npos;
  }
  if (!threw) return fail("FK1 negative-definite prior information throws the existing error");
  const NonPdCapture& cap = lastNonPdCapture();
  if (!cap.valid || cap.where != "posterior_A") return fail("FK1 capture labelled posterior_A");
  if (cap.matrix.rows() != D || (cap.matrix + Eigen::MatrixXd::Identity(D, D)).cwiseAbs().maxCoeff() != 0.0)
    return fail("FK1 captured matrix is -I (P^-1 + zero information), bit-exact");
  if ((cap.eigenvalues.array() + 1.0).abs().maxCoeff() > 1e-14) return fail("FK1 eigenvalues are all -1");
  std::printf("[PASS] FK1 solver failure captured (posterior_A, A = -I, eigenvalues -1)\n");

  const JointKnotSolve& partial = jointKnotLastFailedSolve();
  if (partial.A.rows() != D || partial.vec.size() != D || partial.rhs.size() != D || partial.K1.size() != 0)
    return fail("FK2 partial solve holds A, vec, rhs and an empty K1");
  if ((partial.A - cap.matrix).cwiseAbs().maxCoeff() != 0.0) return fail("FK2 partial A equals the captured matrix");
  std::printf("[PASS] FK2 partial solve retained\n");

  JointKnotPrior good;
  good.P = Eigen::MatrixXd::Identity(D, D);
  good.P_inverse = Eigen::MatrixXd::Identity(D, D);
  const JointKnotSolve ok = solveJointKnotInformation(trajectory, trajectory, good, {}, {}, JointKnotResidualTime::Measurement);
  if (ok.K1.rows() != D || lastNonPdCapture().where != "posterior_A" || (lastNonPdCapture().matrix - cap.matrix).norm() != 0.0)
    return fail("FK3 PD prior solves and leaves the earlier capture untouched");
  std::printf("[PASS] FK3 control: a PD prior does not throw and takes no capture\n");

  const std::vector<Pose6D> poses = propagation();
  std::vector<Eigen::MatrixXd> transitions(poses.size(), Eigen::MatrixXd::Identity(state.dimState(), state.dimState()));
  std::vector<Eigen::MatrixXd> process(poses.size(), Eigen::MatrixXd::Identity(state.dimState(), state.dimState()) * 1e-7);
  for (Eigen::MatrixXd& q : process) q.block<3, 3>(state.idxG(), state.idxG()).setZero();
  Eigen::MatrixXd bad_head = state.cov();
  bad_head(0, 0) = -1.0;
  bool threw_head = false;
  try {
    buildJointKnotImuPrior(trajectory, state, poses, transitions, process, bad_head);
  } catch (const std::runtime_error& e) {
    threw_head = std::string(e.what()).find("scan-head") != std::string::npos;
  }
  if (!threw_head) return fail("FK4 indefinite scan-head covariance throws");
  if (lastNonPdCapture().where != "scan_head" || !(lastNonPdCapture().eigenvalues(0) < 0.0))
    return fail("FK4 capture labelled scan_head with a negative smallest eigenvalue");
  std::printf("[PASS] FK4 scan-head failure captured\n");
  std::printf("ALL PASS\n");
  return 0;
}
