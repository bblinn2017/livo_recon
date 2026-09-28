#include "livo_recon/lio/joint_knot_estimator.h"

#include <cmath>
#include <cstdio>
#include <iostream>

using namespace livo_recon;

namespace
{
// This repo builds permanent tests under CMAKE_BUILD_TYPE=Release, which
// defines NDEBUG and silently compiles out the argument of assert() --
// including any side effect it has. Every check in this file therefore uses
// this NDEBUG-immune fail()-return pattern (the established convention in
// tests/reference_math/*.cpp), never a bare assert(), particularly not for a
// call whose side effect the rest of the test depends on.
int fail(const char* name, double value = 0.0, double tol = 0.0)
{
  std::printf("[FAIL] %s %.6e tol %.6e\n", name, value, tol);
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
}

int main()
{
  // The coupled gating-state axis must be able to add state uncertainty to
  // admission without changing the measurement variance used by the solve.
  const V3D gate_n = V3D(1.0, 2.0, -1.0).normalized();
  const M3D gate_sensor = (V3D(0.01, 0.02, 0.03)).asDiagonal();
  const M3D gate_state = (V3D(0.10, 0.20, 0.30)).asDiagonal();
  const double sensor_only_gate = pointPlaneGateMeasurementVariance(
      gate_n, gate_sensor, gate_state, false);
  const double state_aware_gate = pointPlaneGateMeasurementVariance(
      gate_n, gate_sensor, gate_state, true);
  if (std::abs(sensor_only_gate - gate_n.dot(gate_sensor * gate_n)) >= 1e-15)
    return fail("sensor-only gate excludes state covariance",
                sensor_only_gate - gate_n.dot(gate_sensor * gate_n), 1e-15);
  if (std::abs(state_aware_gate -
      gate_n.dot((gate_sensor + gate_state) * gate_n)) >= 1e-15)
    return fail("gating-state gate includes state covariance exactly once",
                state_aware_gate - gate_n.dot((gate_sensor + gate_state) * gate_n), 1e-15);

  StateGroup state;
  state.setNoiseParams(V3D::Constant(1e-3), V3D::Constant(1e-4));
  state.setPropagatedState(Exp(V3D(0.0, 0.0, 0.02)),
                           V3D(0.1, 0.0, 0.0), V3D(1.0, 0.0, 0.0));
  JointKnotTrajectory trajectory;
  if (!trajectory.initialize(propagation(), 0.1, 4, state))
    return fail("trajectory.initialize succeeds for a valid 4-knot request");
  if (trajectory.knotCount() != 4)
    return fail("knotCount == 4", trajectory.knotCount(), 4);
  if (trajectory.dim() != 54)  // 9*(4-1) + 6*4 knot biases + shared g.
    return fail("dim == 9*(N-1)+6*N+3", trajectory.dim(), 54);
  if (trajectory.knots()[0].t != 0.0) return fail("knot 0 timestamp", trajectory.knots()[0].t, 0.0);
  if (trajectory.knots()[1].t != 0.04) return fail("knot 1 timestamp", trajectory.knots()[1].t, 0.04);
  if (trajectory.knots()[2].t != 0.07) return fail("knot 2 timestamp", trajectory.knots()[2].t, 0.07);
  if (trajectory.knots()[3].t != 0.1) return fail("knot 3 (tail) timestamp", trajectory.knots()[3].t, 0.1);

  const JointKnot head_before = trajectory.knots().front();
  Eigen::VectorXd delta = Eigen::VectorXd::Zero(trajectory.dim());
  delta.segment<3>(trajectory.knotOffset(1) + 3) = V3D(0.1, 0.2, 0.3);
  delta.segment<3>(trajectory.gravityOffset()) = V3D(0.01, 0.02, 0.03);
  trajectory.applyDelta(delta);
  if ((trajectory.knots().front().p - head_before.p).norm() != 0.0)
    return fail("knot 0 position untouched by applyDelta", (trajectory.knots().front().p - head_before.p).norm(), 0.0);
  if (Log(M3D(head_before.R.transpose() * trajectory.knots().front().R)).norm() != 0.0)
    return fail("knot 0 rotation untouched by applyDelta", Log(M3D(head_before.R.transpose() * trajectory.knots().front().R)).norm(), 0.0);
  if ((trajectory.knots().front().v - head_before.v).norm() != 0.0)
    return fail("knot 0 velocity untouched by applyDelta", (trajectory.knots().front().v - head_before.v).norm(), 0.0);

  const auto& knots = trajectory.knots();
  for (const JointKnot& knot : knots) {
    const JointKnotEvaluation at_knot = trajectory.evaluate(knot.t);
    if ((at_knot.p - knot.p).norm() >= 1e-10)
      return fail("Hermite interpolation reproduces endpoint p", (at_knot.p - knot.p).norm(), 1e-10);
    if ((at_knot.v - knot.v).norm() >= 1e-10)
      return fail("Hermite interpolation reproduces endpoint v", (at_knot.v - knot.v).norm(), 1e-10);
    if (Log(M3D(knot.R.transpose() * at_knot.R)).norm() >= 1e-10)
      return fail("geodesic interpolation reproduces endpoint attitude", Log(M3D(knot.R.transpose() * at_knot.R)).norm(), 1e-10);
  }

  const Eigen::MatrixXd J = trajectory.worldPointJacobian(
      0.05, V3D(1.0, 2.0, 3.0), JointKnotResidualTime::Measurement);
  if (J.rows() != 3 || J.cols() != trajectory.dim())
    return fail("measurement-time Jacobian has shape (3, dim)");
  const int physical_dim = 9 * (trajectory.knotCount() - 1);
  if (J.rightCols(trajectory.dim() - physical_dim).norm() != 0.0)
    return fail("LiDAR never observes bias or gravity directly",
                J.rightCols(trajectory.dim() - physical_dim).norm(), 0.0);

  // Independent analytic check of the Hermite translation/velocity columns
  // in the middle interval.  This does not call the implementation's finite
  // difference machinery to construct the expected answer.
  const double t_jac = 0.055;
  const int ia = 1, ib = 2;
  const double dt_jac = knots[ib].t - knots[ia].t;
  const double u = (t_jac - knots[ia].t) / dt_jac;
  const double u2 = u*u, u3 = u2*u;
  const double h00 = 2*u3 - 3*u2 + 1;
  const double h10 = u3 - 2*u2 + u;
  const double h01 = -2*u3 + 3*u2;
  const double h11 = u3 - u2;
  const Eigen::MatrixXd J_mid = trajectory.worldPointJacobian(
      t_jac, V3D(1.3, -0.4, 2.1), JointKnotResidualTime::Measurement);
  const M3D I = M3D::Identity();
  if ((J_mid.block<3,3>(0, trajectory.knotOffset(ia) + 3) - h00*I).norm() >= 1e-7)
    return fail("measurement Jacobian analytic left-position block",
        (J_mid.block<3,3>(0, trajectory.knotOffset(ia) + 3) - h00*I).norm(), 1e-7);
  if ((J_mid.block<3,3>(0, trajectory.knotOffset(ia) + 6) - h10*dt_jac*I).norm() >= 1e-7)
    return fail("measurement Jacobian analytic left-velocity block",
        (J_mid.block<3,3>(0, trajectory.knotOffset(ia) + 6) - h10*dt_jac*I).norm(), 1e-7);
  if ((J_mid.block<3,3>(0, trajectory.knotOffset(ib) + 3) - h01*I).norm() >= 1e-7)
    return fail("measurement Jacobian analytic right-position block",
        (J_mid.block<3,3>(0, trajectory.knotOffset(ib) + 3) - h01*I).norm(), 1e-7);
  if ((J_mid.block<3,3>(0, trajectory.knotOffset(ib) + 6) - h11*dt_jac*I).norm() >= 1e-7)
    return fail("measurement Jacobian analytic right-velocity block",
        (J_mid.block<3,3>(0, trajectory.knotOffset(ib) + 6) - h11*dt_jac*I).norm(), 1e-7);

  // Independent scalar point-to-plane directional derivative, using a
  // different step from production and perturbing every mutable column.
  const V3D q_jac(1.3, -0.4, 2.1);
  const V3D normal = V3D(0.2, -0.7, 0.4).normalized();
  constexpr double oracle_eps = 3e-6;
  for (int c = 0; c < trajectory.dim(); ++c) {
    JointKnotTrajectory plus = trajectory, minus = trajectory;
    Eigen::VectorXd d = Eigen::VectorXd::Zero(trajectory.dim());
    d(c) = oracle_eps; plus.applyDelta(d);
    d(c) = -oracle_eps; minus.applyDelta(d);
    const double numerical = normal.dot(
        plus.worldPoint(t_jac, q_jac, JointKnotResidualTime::Measurement) -
        minus.worldPoint(t_jac, q_jac, JointKnotResidualTime::Measurement)) /
        (2.0 * oracle_eps);
    const double predicted = (normal.transpose() * J_mid)(c);
    if (std::abs(numerical - predicted) >= 2e-7)
      return fail("measurement scalar-residual directional derivative",
                  std::abs(numerical - predicted), 2e-7);
  }
  const V3D nominal_measurement = trajectory.worldPoint(
      0.05, V3D(1.0,2.0,3.0), JointKnotResidualTime::Measurement);
  const V3D nominal_tail = trajectory.worldPoint(
      0.05, V3D(1.0,2.0,3.0), JointKnotResidualTime::Tail);
  if ((nominal_measurement - nominal_tail).norm() >= 1e-12)
    return fail("measurement-time and tail-time nominal world points agree", (nominal_measurement - nominal_tail).norm(), 1e-12);
  const Eigen::MatrixXd J_tail = trajectory.worldPointJacobian(
      0.05, V3D(1.0,2.0,3.0), JointKnotResidualTime::Tail);
  if (J_tail.leftCols(trajectory.knotOffset(3)).norm() != 0.0)
    return fail("tail-time direct Jacobian support is tail-only (left columns zero)", J_tail.leftCols(trajectory.knotOffset(3)).norm(), 0.0);
  if (J_tail.rightCols(trajectory.dim() - physical_dim).norm() != 0.0)
    return fail("tail-time Jacobian has zero bias/gravity columns",
                J_tail.rightCols(trajectory.dim() - physical_dim).norm(), 0.0);
  const JointKnotEvaluation tail_nominal = trajectory.evaluate(
      trajectory.knots().back().t);
  const V3D q_tail_fixed = tail_nominal.R.transpose() *
      (nominal_measurement - tail_nominal.p);
  for (int c = 0; c < trajectory.dim(); ++c) {
    JointKnotTrajectory plus = trajectory, minus = trajectory;
    Eigen::VectorXd d = Eigen::VectorXd::Zero(trajectory.dim());
    d(c) = oracle_eps; plus.applyDelta(d);
    d(c) = -oracle_eps; minus.applyDelta(d);
    const JointKnotEvaluation xp = plus.evaluate(plus.knots().back().t);
    const JointKnotEvaluation xm = minus.evaluate(minus.knots().back().t);
    const double numerical = normal.dot(
        (xp.R * q_tail_fixed + xp.p) -
        (xm.R * q_tail_fixed + xm.p)) / (2.0 * oracle_eps);
    const double predicted = (normal.transpose() * J_tail)(c);
    if (std::abs(numerical - predicted) >= 2e-7)
      return fail("tail scalar-residual directional derivative",
                  std::abs(numerical - predicted), 2e-7);
  }

  const std::vector<Pose6D> poses = propagation();
  std::vector<Eigen::MatrixXd> transitions(poses.size(),
      Eigen::MatrixXd::Identity(state.dimState(), state.dimState()));
  std::vector<Eigen::MatrixXd> process_covariances(poses.size(),
      Eigen::MatrixXd::Identity(state.dimState(), state.dimState()) * 1e-7);
  for (Eigen::MatrixXd& q : process_covariances)
    q.block<3,3>(state.idxG(), state.idxG()).setZero();
  for (size_t m = 0; m < poses.size(); ++m) {
    const double dt = poses[m].dt;
    transitions[m].block<3,3>(StateGroup::idxP(), StateGroup::idxV()) = dt * Eye3d;
    transitions[m].block<3,3>(StateGroup::idxR(), state.idxBG()) = -dt * Eye3d;
    transitions[m].block<3,3>(StateGroup::idxV(), state.idxBA()) = -dt * Eye3d;
    transitions[m].block<3,3>(StateGroup::idxV(), state.idxG()) = dt * Eye3d;
  }
  const JointKnotPrior prior = buildJointKnotImuPrior(
      trajectory, state, poses, transitions, process_covariances, state.cov());
  if (prior.P.rows() != trajectory.dim())
    return fail("composed joint prior P has trajectory dimension");
  if ((prior.P - prior.P.transpose()).norm() >= 1e-10)
    return fail("composed joint prior P is symmetric", (prior.P - prior.P.transpose()).norm(), 1e-10);
  if ((prior.P * prior.P_inverse - Eigen::MatrixXd::Identity(
      trajectory.dim(), trajectory.dim())).norm() >= 1e-5)
    return fail("P * P_inverse is identity within tolerance",
        (prior.P * prior.P_inverse - Eigen::MatrixXd::Identity(trajectory.dim(), trajectory.dim())).norm(), 1e-5);
  // Independent stacked-state oracle for every block of the extended P.
  // This is deliberately assembled from Cov(x_i,x_j)=P_i Phi(j<-i)^T,
  // rather than calling any helper from buildJointKnotImuPrior().
  constexpr int E = 18;
  std::vector<Eigen::Matrix<double,E,E>> propagated(4);
  std::vector<Eigen::Matrix<double,E,E>> phi_step(4);
  propagated[0] = state.cov();
  for (int k = 1; k < 4; ++k) {
    const int count = k == 1 ? 4 : 3;
    const int first_record = k == 1 ? 0 : (k == 2 ? 4 : 7);
    phi_step[k] = Eigen::Matrix<double,E,E>::Identity();
    Eigen::Matrix<double,E,E> q = Eigen::Matrix<double,E,E>::Zero();
    for (int m = 0; m < count; ++m) {
      const Eigen::Matrix<double,E,E> fm = transitions[first_record + m];
      const Eigen::Matrix<double,E,E> qm = process_covariances[first_record + m];
      q = fm * q * fm.transpose() + qm;
      phi_step[k] = fm * phi_step[k];
    }
    propagated[k] = phi_step[k] * propagated[k-1] * phi_step[k].transpose() + q;
  }
  struct OracleBlock { int joint; int knot; int production; };
  std::vector<OracleBlock> blocks;
  for (int k = 1; k < 4; ++k)
    for (int c = 0; c < 3; ++c)
      blocks.push_back({trajectory.knotOffset(k) + 3*c, k, 3*c});
  for (int k = 0; k < 4; ++k) {
    blocks.push_back({trajectory.biasOffset(k), k, state.idxBG()});
    blocks.push_back({trajectory.biasOffset(k) + 3, k, state.idxBA()});
  }
  blocks.push_back({trajectory.gravityOffset(), 0, state.idxG()});
  auto oracle_cross = [&](int a, int b) -> Eigen::Matrix<double,E,E> {
    Eigen::Matrix<double,E,E> phi = Eigen::Matrix<double,E,E>::Identity();
    if (a == b) return propagated[a];
    if (a < b) {
      for (int k = a + 1; k <= b; ++k) phi = phi_step[k] * phi;
      return propagated[a] * phi.transpose();
    }
    for (int k = b + 1; k <= a; ++k) phi = phi_step[k] * phi;
    return phi * propagated[b];
  };
  Eigen::MatrixXd oracle = Eigen::MatrixXd::Zero(trajectory.dim(), trajectory.dim());
  for (const OracleBlock& row : blocks)
    for (const OracleBlock& col : blocks)
      oracle.block<3,3>(row.joint, col.joint) =
          oracle_cross(row.knot, col.knot).block<3,3>(
              row.production, col.production);
  if ((prior.P - oracle).norm() >= 1e-10)
    return fail("extended P matches independent stacked-state oracle",
                (prior.P - oracle).norm(), 1e-10);

  // Full production bias random walk must survive in the knot-specific
  // model. Later biases gain variance and adjacent biases are not identical.
  const M3D bg0 = prior.P.block<3,3>(trajectory.biasOffset(0),
                                     trajectory.biasOffset(0));
  const M3D bg3 = prior.P.block<3,3>(trajectory.biasOffset(3),
                                     trajectory.biasOffset(3));
  if ((bg3 - bg0).trace() <= 0.0)
    return fail("gyro-bias random walk increases tail bias variance",
                (bg3 - bg0).trace(), 0.0);
  const M3D ba0 = prior.P.block<3,3>(trajectory.biasOffset(0) + 3,
                                     trajectory.biasOffset(0) + 3);
  const M3D ba3 = prior.P.block<3,3>(trajectory.biasOffset(3) + 3,
                                     trajectory.biasOffset(3) + 3);
  if ((ba3 - ba0).trace() <= 0.0)
    return fail("accelerometer-bias random walk increases tail bias variance",
                (ba3 - ba0).trace(), 0.0);

  // Before LiDAR, selecting the tail physical state, tail biases and shared
  // gravity from the extended prior must reproduce ordinary propagation.
  const Eigen::MatrixXd tail_prior = extractTailStateCovariance(
      trajectory, prior.P, state);
  if ((tail_prior - propagated.back()).norm() >= 1e-10)
    return fail("extended-prior tail marginal matches production propagation",
                (tail_prior - propagated.back()).norm(), 1e-10);

  JointKnotTrajectory current = trajectory;
  Eigen::VectorXd displaced = Eigen::VectorXd::Zero(current.dim());
  displaced.segment<3>(current.knotOffset(2) + 3) = V3D(0.01, -0.02, 0.03);
  current.applyDelta(displaced);
  const JointKnotSolve restore = solveJointKnotInformation(
      current, trajectory, prior, {}, {}, JointKnotResidualTime::Measurement);
  if ((restore.delta - current.boxminus(trajectory)).norm() >= 1e-8)
    return fail("empty-LiDAR solve produces the fixed-prior restoring correction",
        (restore.delta - current.boxminus(trajectory)).norm(), 1e-8);
  if ((restore.delta - restore.lidar_delta - restore.prior_delta).norm() >= 1e-12)
    return fail("diagnostic correction split sums exactly to delta",
        (restore.delta - restore.lidar_delta - restore.prior_delta).norm(), 1e-12);

  std::cout << "joint-knot estimator invariants passed\n";
  return 0;
}
