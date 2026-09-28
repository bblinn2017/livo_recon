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
  // Gravity alignment constrains the two tangent directions perpendicular to
  // gravity while leaving yaw (rotation about gravity) separately tunable.
  // Use a non-identity attitude so this catches accidental world/body-axis
  // confusion rather than validating only the identity special case.
  const M3D aligned_R = Exp(V3D(0.7, -1.1, 0.4));
  const V3D yaw_axis_body =
      (aligned_R.transpose() * V3D(0.0, 0.0, -1.0)).normalized();
  const double tilt_variance = 1e-6, yaw_variance = 1e-2;
  const M3D aligned_cov = StateGroup::gravityAlignedAttitudeCovariance(
      aligned_R, tilt_variance, yaw_variance);
  if (std::abs(yaw_axis_body.dot(aligned_cov * yaw_axis_body) -
               yaw_variance) >= 1e-14)
    return fail("gravity-aligned yaw covariance eigenvalue",
        yaw_axis_body.dot(aligned_cov * yaw_axis_body) - yaw_variance, 1e-14);
  const V3D tilt_axis_1 = yaw_axis_body.unitOrthogonal().normalized();
  const V3D tilt_axis_2 = yaw_axis_body.cross(tilt_axis_1).normalized();
  if (std::abs(tilt_axis_1.dot(aligned_cov * tilt_axis_1) -
               tilt_variance) >= 1e-14 ||
      std::abs(tilt_axis_2.dot(aligned_cov * tilt_axis_2) -
               tilt_variance) >= 1e-14)
    return fail("gravity-aligned tilt covariance eigenvalues",
        std::max(std::abs(tilt_axis_1.dot(aligned_cov * tilt_axis_1) - tilt_variance),
                 std::abs(tilt_axis_2.dot(aligned_cov * tilt_axis_2) - tilt_variance)),
        1e-14);
  if ((aligned_cov - aligned_cov.transpose()).norm() >= 1e-15)
    return fail("gravity-aligned attitude covariance symmetric",
                (aligned_cov - aligned_cov.transpose()).norm(), 1e-15);

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

  // Phase-3 algebra oracle: independent mode must retain the ordinary
  // diagonal accumulation, while Woodbury mode must equal an explicit dense
  // inverse of C = D + shared*11^T for one matched-plane group.
  constexpr int lidar_dim = 5;
  int plane_token = 0;
  const void* plane_id = &plane_token;
  std::vector<JointLidarRow> rows;
  const double shared = 0.08;
  const std::vector<double> independent_var{0.21, 0.34, 0.27};
  const std::vector<double> residual_value{0.12, -0.07, 0.19};
  Eigen::MatrixXd H_dense(3, lidar_dim);
  H_dense << 1.0, 0.2, -0.3, 0.0, 0.5,
             0.4, -0.8, 0.1, 0.7, 0.0,
             -0.2, 0.3, 0.9, -0.4, 0.6;
  for (int i = 0; i < 3; ++i)
    rows.push_back(JointLidarRow{
        H_dense.row(i), residual_value[i], independent_var[i] + shared,
        shared, plane_id});
  ResidualRedundancyOptions independent_options;
  independent_options.mode = "off";
  const JointLidarInformation independent = accumulateJointLidarInformation(
      rows, lidar_dim, independent_options);
  Eigen::MatrixXd expected_independent = Eigen::MatrixXd::Zero(lidar_dim, lidar_dim);
  Eigen::VectorXd expected_independent_b = Eigen::VectorXd::Zero(lidar_dim);
  for (int i = 0; i < 3; ++i) {
    const double w = 1.0 / (independent_var[i] + shared);
    expected_independent.noalias() += w * H_dense.row(i).transpose() * H_dense.row(i);
    expected_independent_b.noalias() += -w * H_dense.row(i).transpose() * residual_value[i];
  }
  if ((independent.Gamma - expected_independent).norm() >= 1e-12 ||
      (independent.b - expected_independent_b).norm() >= 1e-12)
    return fail("independent joint LiDAR accumulation matches diagonal oracle");

  ResidualRedundancyOptions woodbury_options;
  woodbury_options.mode = "woodbury";
  woodbury_options.rho = 1.0;
  woodbury_options.max_discount = 1.0;
  const JointLidarInformation woodbury = accumulateJointLidarInformation(
      rows, lidar_dim, woodbury_options);
  Eigen::Matrix3d C = Eigen::Matrix3d::Constant(shared);
  for (int i = 0; i < 3; ++i) C(i, i) += independent_var[i];
  const Eigen::Vector3d r_dense(
      residual_value[0], residual_value[1], residual_value[2]);
  const Eigen::MatrixXd expected_woodbury =
      H_dense.transpose() * C.inverse() * H_dense;
  const Eigen::VectorXd expected_woodbury_b =
      -H_dense.transpose() * C.inverse() * r_dense;
  if ((woodbury.Gamma - expected_woodbury).norm() >= 1e-11 ||
      (woodbury.b - expected_woodbury_b).norm() >= 1e-11)
    return fail("Woodbury joint LiDAR accumulation matches dense C inverse");
  if ((woodbury.Gamma - independent.Gamma).norm() <= 1e-6)
    return fail("correlated and independent joint LiDAR information differ");
  if (woodbury.redundancy_stats.redund_groups != 1 ||
      woodbury.redundancy_stats.redund_n_raw != 3)
    return fail("joint LiDAR redundancy engagement is reported");
  if (woodbury.redundancy_stats.max_discount_bound_groups != 0)
    return fail("max_discount_bound_groups is 0 when max_discount=1.0 never binds",
        woodbury.redundancy_stats.max_discount_bound_groups, 0.0);

  // A group of duplicate (identical H, high shared-variance-fraction)
  // measurements has a large TRUE discount (naively triple-counting one
  // measurement as three independent ones) -- verified externally at
  // discount ~= 0.643 for these exact numbers. A max_discount below that
  // must actually clamp, and the clamped result must land exactly on the
  // max_discount-scaled blend between naive and corrected, not merely
  // differ from the unclamped correction.
  std::vector<JointLidarRow> dup_rows;
  Eigen::RowVectorXd H_dup = Eigen::RowVectorXd::Zero(lidar_dim);
  H_dup(0) = 1.0;
  const double dup_shared = 0.09, dup_independent_var = 0.01;
  for (int i = 0; i < 3; ++i)
    dup_rows.push_back(JointLidarRow{
        H_dup, 0.1, dup_independent_var + dup_shared, dup_shared, plane_id});
  ResidualRedundancyOptions unclamped_dup_options;
  unclamped_dup_options.mode = "woodbury";
  unclamped_dup_options.rho = 1.0;
  unclamped_dup_options.max_discount = 1.0;
  const JointLidarInformation unclamped_dup = accumulateJointLidarInformation(
      dup_rows, lidar_dim, unclamped_dup_options);
  if (unclamped_dup.redundancy_stats.max_discount_bound_groups != 0)
    return fail("max_discount=1.0 does not bind on the duplicate-measurement group",
        unclamped_dup.redundancy_stats.max_discount_bound_groups, 0.0);

  ResidualRedundancyOptions clamped_options = unclamped_dup_options;
  clamped_options.max_discount = 0.5;
  const JointLidarInformation clamped = accumulateJointLidarInformation(
      dup_rows, lidar_dim, clamped_options);
  if (clamped.redundancy_stats.max_discount_bound_groups != 1)
    return fail("max_discount=0.5 clamps the duplicate-measurement group and increments the bound counter",
        clamped.redundancy_stats.max_discount_bound_groups, 1.0);

  ResidualRedundancyOptions independent_dup_options;
  independent_dup_options.mode = "off";
  const JointLidarInformation naive_dup = accumulateJointLidarInformation(
      dup_rows, lidar_dim, independent_dup_options);
  const double scale = clamped_options.max_discount /
      (1.0 - unclamped_dup.Gamma.trace() / naive_dup.Gamma.trace());
  const Eigen::MatrixXd expected_clamped_gamma =
      naive_dup.Gamma + scale * (unclamped_dup.Gamma - naive_dup.Gamma);
  if ((clamped.Gamma - expected_clamped_gamma).norm() >= 1e-9)
    return fail("clamped Gamma lands exactly on the max_discount-scaled naive/corrected blend",
        (clamped.Gamma - expected_clamped_gamma).norm(), 0.0);

  // R45 repair regression test: plane_var_term = J_nq*plane_var_*J_nq^T is
  // evaluated per-POINT (voxelplane.cpp), so real matched-plane groups have
  // DIFFERING plane_var_term across members -- the original patch asserted
  // they must be equal and threw on every real coupled run. The fix treats
  // each row's own plane_var_term as its personal loading onto one shared
  // latent factor (general rank-1 C = D + v*v^T, v_i = sqrt(rho*pvt_i)) and
  // must no longer throw, matching an explicit dense inverse exactly.
  const double nonuniform_rho = 0.7;
  const std::vector<double> nonuniform_independent_var{0.15, 0.22, 0.31, 0.18};
  const std::vector<double> nonuniform_plane_var_term{0.05, 0.09, 0.02, 0.07};
  const std::vector<double> nonuniform_residual{0.08, -0.11, 0.05, 0.13};
  Eigen::MatrixXd H_nonuniform(4, lidar_dim);
  H_nonuniform << 0.6, -0.2, 0.4, 0.1, 0.0,
                  0.1, 0.5, -0.3, 0.2, 0.4,
                  -0.4, 0.3, 0.2, -0.5, 0.1,
                  0.2, 0.1, 0.5, 0.3, -0.2;
  int nonuniform_plane_token = 0;
  const void* nonuniform_plane_id = &nonuniform_plane_token;
  std::vector<JointLidarRow> nonuniform_rows;
  for (int i = 0; i < 4; ++i)
    nonuniform_rows.push_back(JointLidarRow{
        H_nonuniform.row(i), nonuniform_residual[i],
        nonuniform_independent_var[i] + nonuniform_rho * nonuniform_plane_var_term[i],
        nonuniform_plane_var_term[i], nonuniform_plane_id});
  ResidualRedundancyOptions nonuniform_options;
  nonuniform_options.mode = "woodbury";
  nonuniform_options.rho = nonuniform_rho;
  nonuniform_options.max_discount = 1.0;
  const JointLidarInformation nonuniform = accumulateJointLidarInformation(
      nonuniform_rows, lidar_dim, nonuniform_options);
  Eigen::Vector4d v_nonuniform, sigma_nonuniform;
  for (int i = 0; i < 4; ++i) {
    v_nonuniform(i) = std::sqrt(nonuniform_rho * nonuniform_plane_var_term[i]);
    sigma_nonuniform(i) = nonuniform_independent_var[i];
  }
  Eigen::Matrix4d C_nonuniform = v_nonuniform * v_nonuniform.transpose();
  for (int i = 0; i < 4; ++i) C_nonuniform(i, i) += sigma_nonuniform(i);
  const Eigen::Vector4d r_nonuniform(
      nonuniform_residual[0], nonuniform_residual[1],
      nonuniform_residual[2], nonuniform_residual[3]);
  const Eigen::MatrixXd expected_nonuniform_gamma =
      H_nonuniform.transpose() * C_nonuniform.inverse() * H_nonuniform;
  const Eigen::VectorXd expected_nonuniform_b =
      -H_nonuniform.transpose() * C_nonuniform.inverse() * r_nonuniform;
  if ((nonuniform.Gamma - expected_nonuniform_gamma).norm() >= 1e-9)
    return fail("non-uniform plane_var_term Woodbury matches dense C inverse (Gamma)",
        (nonuniform.Gamma - expected_nonuniform_gamma).norm(), 0.0);
  if ((nonuniform.b - expected_nonuniform_b).norm() >= 1e-9)
    return fail("non-uniform plane_var_term Woodbury matches dense C inverse (b)",
        (nonuniform.b - expected_nonuniform_b).norm(), 0.0);
  if (nonuniform.redundancy_stats.redund_groups != 1)
    return fail("non-uniform plane_var_term group is admitted, not rejected as degenerate",
        nonuniform.redundancy_stats.redund_groups, 1.0);

  std::cout << "joint-knot estimator invariants passed\n";
  return 0;
}
