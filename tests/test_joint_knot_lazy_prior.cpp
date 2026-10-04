// R74 integration test (F-410): the coupled estimator builds the joint IMU prior only when the scan has LiDAR residuals. Linked against
// joint_knot_test_core (auto-registered by tests/joint_knots/CMakeLists.txt, which globs test_joint_knot_*.cpp).
//   LP1 zero residuals: the builder is NOT called, the helper returns false, and a covariance held beside it is bit-identical afterwards.
//   LP2 one or more residuals: the builder is called exactly once; a second call with the prior already built does not call it again.
//   LP3 control (the check fires): with a scan-head covariance made indefinite (the synthetic singular prior, same device as FK4) the REAL
//       buildJointKnotImuPrior throws when it is called; so the eager path would have aborted.
//   LP4 the same real builder through the helper: zero residuals -> no throw and the state covariance is bit-identical; one residual -> it is
//       reached and throws (the helper does not swallow the failure when there is something to solve).
#include "livo_recon/lio/lazy_prior.h"
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
  // LP1
  {
    int calls = 0;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6) * 3.0;
    const Eigen::MatrixXd cov_before = cov;
    const bool built = buildPriorIfResiduals(0, false, [&]() { ++calls; cov.setZero(); });
    if (built || calls != 0) return fail("LP1 zero residuals: builder not called, helper returns false");
    if ((cov - cov_before).cwiseAbs().maxCoeff() != 0.0) return fail("LP1 covariance bit-identical");
    std::printf("[PASS] LP1 zero residuals build nothing and leave the covariance unchanged\n");
  }
  // LP2
  {
    int calls = 0;
    bool built = buildPriorIfResiduals(1500, false, [&]() { ++calls; });
    if (!built || calls != 1) return fail("LP2 residuals present: builder called once");
    built = buildPriorIfResiduals(1500, true, [&]() { ++calls; });
    if (!built || calls != 1) return fail("LP2 already built: builder not called again");
    built = buildPriorIfResiduals(0, true, [&]() { ++calls; });
    if (!built || calls != 1) return fail("LP2 already built and no residuals: still available, builder not called");
    std::printf("[PASS] LP2 builds once when residuals exist and never twice\n");
  }
  // fixture: the same trajectory and transitions as test_joint_knot_failure_capture
  StateGroup state;
  state.setNoiseParams(V3D::Constant(1e-3), V3D::Constant(1e-4));
  state.setPropagatedState(Exp(V3D(0.0, 0.0, 0.02)), V3D(0.1, 0.0, 0.0), V3D(1.0, 0.0, 0.0));
  JointKnotTrajectory trajectory;
  if (!trajectory.initialize(propagation(), 0.1, 4, state)) return fail("fixture trajectory");
  const std::vector<Pose6D> poses = propagation();
  std::vector<Eigen::MatrixXd> transitions(poses.size(), Eigen::MatrixXd::Identity(state.dimState(), state.dimState()));
  std::vector<Eigen::MatrixXd> process(poses.size(), Eigen::MatrixXd::Identity(state.dimState(), state.dimState()) * 1e-7);
  for (Eigen::MatrixXd& q : process) q.block<3, 3>(state.idxG(), state.idxG()).setZero();
  Eigen::MatrixXd bad_head = state.cov();
  bad_head(0, 0) = -1.0;
  const auto real_build = [&]() { buildJointKnotImuPrior(trajectory, state, poses, transitions, process, bad_head); };

  // LP3
  {
    bool threw = false;
    try { real_build(); } catch (const std::runtime_error&) { threw = true; }
    if (!threw) return fail("LP3 control: the singular prior makes the real builder throw");
    std::printf("[PASS] LP3 control: an indefinite scan-head covariance makes buildJointKnotImuPrior throw (the eager path would abort)\n");
  }
  // LP4
  {
    const Eigen::MatrixXd cov_before = state.cov();
    bool threw = false;
    try { buildPriorIfResiduals(0, false, real_build); } catch (const std::runtime_error&) { threw = true; }
    if (threw) return fail("LP4 zero residuals: no throw");
    if ((state.cov() - cov_before).cwiseAbs().maxCoeff() != 0.0) return fail("LP4 zero residuals: state covariance bit-identical");
    bool threw1 = false;
    try { buildPriorIfResiduals(1, false, real_build); } catch (const std::runtime_error&) { threw1 = true; }
    if (!threw1) return fail("LP4 one residual: the builder is reached and throws");
    std::printf("[PASS] LP4 zero residuals: no throw and covariance unchanged; one residual: the failure is not swallowed\n");
  }
  std::printf("ALL PASS\n");
  return 0;
}
