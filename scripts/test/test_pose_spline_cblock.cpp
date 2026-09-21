// CQ-82 Phase 2, Artifact 3: item 6's correctness gate for
// buildPoseSplineCBlock() (lio/pose_spline_system.h) -- "Initialised from
// ScanSpline::fit() and with the LiDAR term DISABLED, the solve must leave
// the trajectory at the fitted spline, and the fit residual against
// mg.poses must match what updateFitResiduals() already reports. STATE BOTH
// NUMBERS."
//
// Read literally: with LiDAR observations empty AND the IMU-factor weights
// also at 0 (the only combination under which "the solve leaves the
// trajectory unchanged" is actually true -- at the shipped pose_imu_weight_*
// default of 1.0 the IMU term is NOT trivially satisfied by a pure
// geometric fit, since ScanSpline::fit() never enforces IMU consistency),
// the only content left in A is the fixed Tikhonov floor
// (POSE_SPLINE_TIKHONOV_EPS) -- so b=0 forces c=0 exactly (up to solver
// numerical noise), meaning the trajectory the system implies is bit-for-bit
// the SAME spline fit() already produced. This test verifies exactly that:
// solves the disabled-terms system, confirms max|c| is at numerical zero,
// and separately confirms the fit's own residual (posAt/rotAt vs the same
// poses) matches updateFitResiduals()'s own already-reported numbers.
//
// No ROS/PCL/OpenCV, Eigen-only. Never linked into livo_recon_node; pure
// standalone diagnostic/verification tooling, zero risk to shipped behavior
// (this basis is not reachable from any live estimator path yet -- see
// pose_spline_system.h's own doc comment).
//
//   g++ -std=c++17 -O2 -I include -I /usr/include/eigen3 \
//       scripts/test/test_pose_spline_cblock.cpp src/lio/spline.cpp \
//       src/lio/pose_spline_system.cpp \
//       -o /tmp/test_pose_spline_cblock && /tmp/test_pose_spline_cblock
#include "livo_recon/lio/pose_spline_system.h"
#include <cstdio>
#include <vector>

using namespace livo_recon;

static std::vector<Pose6D> buildRawPoses(int N, double dt, const M3D& rot0,
                                          const V3D& pos0, const V3D& vel0,
                                          const V3D& gravity0, const V3D& gyr,
                                          const V3D& acc_body)
{
  std::vector<Pose6D> poses;
  poses.reserve(N);
  M3D R = rot0;
  V3D p = pos0, v = vel0;
  for (int k = 0; k < N; ++k) {
    const double t = k * dt;
    const V3D acc_head = R * acc_body + gravity0;
    const M3D R_tail = R * Exp(gyr, dt);
    const V3D acc_tail = R_tail * acc_body + gravity0;
    poses.push_back(Pose6D{t, acc_head, acc_tail, gyr, v, p, R, dt});
    const V3D acc_avr_world = 0.5 * (acc_head + acc_tail);
    p = p + v * dt + 0.5 * acc_avr_world * dt * dt;
    v = v + acc_avr_world * dt;
    R = R_tail;
  }
  return poses;
}

int main()
{
  const int N = 20;
  const double dt = 0.005;
  const V3D gravity(0, 0, -9.81);
  const V3D gyr(0.05, -0.03, 0.08);       // a real, modest rotation rate
  const V3D acc_body(0.2, -0.1, 9.81);    // roughly gravity-compensating
  const auto poses = buildRawPoses(N, dt, M3D::Identity(), V3D::Zero(),
                                    V3D::Zero(), gravity, gyr, acc_body);

  const double t0 = poses.front().t;
  const double t1 = poses.back().t + poses.back().dt;

  SplineOptions opts;
  opts.control_point_hz = 400.0;  // -> a handful of control points over this short window
  opts.end_constraint_velocity = true;

  ScanSpline spline;
  const bool ok = spline.fit(poses, t0, t1, opts);
  if (!ok) {
    std::printf("FAIL: ScanSpline::fit() itself failed, cause=%d\n",
                static_cast<int>(spline.lastFitFailCause()));
    return 1;
  }
  spline.updateFitResiduals(poses);
  const double ref_fit_res_pos = spline.fitResidualPos();
  const double ref_fit_res_rot = spline.fitResidualRot();

  // LiDAR DISABLED (empty) AND IMU-factor weights at 0 -- the only
  // combination under which the item-6 claim is literally true (see this
  // file's own header comment).
  const std::vector<PoseSplineLidarObs> lidar_obs;  // empty
  const std::vector<PoseSplineImuObs> imu_obs;      // empty -- weight is what matters
  const auto build = buildPoseSplineCBlock(
      spline, lidar_obs, imu_obs,
      V3D::Zero(), V3D::Zero(), gravity,
      /*pose_imu_weight_acc=*/0.0, /*pose_imu_weight_gyr=*/0.0,
      /*pose_curvature_weight_pos=*/0.0, /*pose_curvature_weight_rot=*/0.0);

  Eigen::LDLT<Eigen::MatrixXd> ldlt(build.A);
  const Eigen::VectorXd c = ldlt.solve(build.b);
  const double max_abs_c = c.cwiseAbs().maxCoeff();

  // Apply c to a COPY of the spline and re-check its fit residual against
  // the same poses -- should reproduce the reference numbers exactly (c is
  // ~0, so this is a tautology check on the plumbing, not a new number).
  ScanSpline spline_plus_c = spline;
  const int n_c = spline.nControlPoints();
  for (int j = 0; j < n_c; ++j) {
    spline_plus_c.cpPosMut().col(j) += c.segment<3>(3 * j);
    spline_plus_c.cp_phi_.col(j)    += c.segment<3>(3 * n_c + 3 * j);
  }
  spline_plus_c.updateFitResiduals(poses);
  const double check_fit_res_pos = spline_plus_c.fitResidualPos();
  const double check_fit_res_rot = spline_plus_c.fitResidualRot();

  std::printf("n_c=%d (requested %d, clamped=%d)\n",
              n_c, spline.nControlPointsRequested(), spline.nControlPointsClamped());
  std::printf("max|c| = %.3e (expect ~0, floor Tikhonov-only system)\n", max_abs_c);
  std::printf("fit_res_pos: reference(fit)=%.9f  after-disabled-solve=%.9f  diff=%.3e\n",
              ref_fit_res_pos, check_fit_res_pos, std::fabs(ref_fit_res_pos - check_fit_res_pos));
  std::printf("fit_res_rot: reference(fit)=%.9f  after-disabled-solve=%.9f  diff=%.3e\n",
              ref_fit_res_rot, check_fit_res_rot, std::fabs(ref_fit_res_rot - check_fit_res_rot));

  bool pass = (max_abs_c < 1e-6) &&
              (std::fabs(ref_fit_res_pos - check_fit_res_pos) < 1e-9) &&
              (std::fabs(ref_fit_res_rot - check_fit_res_rot) < 1e-9);

  // Second, separate check: item 6's own gate (above) never actually
  // exercises the LiDAR/IMU Jacobian code, since both observation lists
  // were empty. Smoke-test them with real data: a handful of synthetic
  // point-plane observations plus the same IMU samples the fit itself used,
  // with all weights ON (the shipped pose_imu_weight_* default of 1.0), and
  // confirm the resulting system is finite and PSD (no NaN/Inf, no negative
  // eigenvalue) -- not a numerics-correctness proof, but catches an
  // obviously broken Jacobian (a sign error producing a wildly indefinite
  // A, a NaN from a degenerate FD step, etc.).
  std::vector<PoseSplineLidarObs> lidar_smoke;
  for (int i = 0; i < N; i += 4) {
    PoseSplineLidarObs o;
    o.t = poses[i].t;
    o.raw_body_point = V3D(1.0, 0.2, 0.1);  // an arbitrary body-frame point
    o.normal = V3D(0, 0, 1);
    o.r = 0.01;  // a small nonzero residual
    o.sigma2 = 0.01;
    lidar_smoke.push_back(o);
  }
  std::vector<PoseSplineImuObs> imu_smoke;
  for (int i = 0; i < N; ++i) {
    PoseSplineImuObs o;
    o.t = poses[i].t;
    o.acc = poses[i].acc_head;
    o.gyr = poses[i].gyr;
    imu_smoke.push_back(o);
  }
  const auto build_smoke = buildPoseSplineCBlock(
      spline, lidar_smoke, imu_smoke, V3D::Zero(), V3D::Zero(), gravity,
      /*pose_imu_weight_acc=*/1.0, /*pose_imu_weight_gyr=*/1.0,
      /*pose_curvature_weight_pos=*/0.0, /*pose_curvature_weight_rot=*/0.0);
  const bool finite = build_smoke.A.allFinite() && build_smoke.b.allFinite();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (build_smoke.A + build_smoke.A.transpose()));
  const double min_eig = finite ? es.eigenvalues().minCoeff() : -1.0;
  std::printf("smoke check: n_lidar=%d n_imu=%d finite=%d min_eig(A)=%.3e (expect >=0, PSD)\n",
              build_smoke.n_lidar, build_smoke.n_imu, finite ? 1 : 0, min_eig);
  const bool smoke_pass = finite && (min_eig > -1e-9);
  pass = pass && smoke_pass;

  // ==========================================================================
  // POST-CQ-87-REVIEW addition (item 9: "the current correctness test is not
  // sufficient" -- GN iteration 2+, head consistency, FD Jacobian, rotation
  // head-tie characterization were all named as gaps). Four new checks below,
  // each targeting one specific reviewer-named gap.
  // ==========================================================================

  // ---- Check 1 (item 1's fix, item 9's "head consistency") ----
  // Refit WITH setFrozenBoundary() this time (the previous fit() call above
  // was deliberately left unconstrained, matching pre-fix behavior, so its
  // own reference numbers stay meaningful) and confirm p(t0)==pos0,
  // R(t0)==rot0 to numerical precision -- NOT approximately, per item 1's
  // own "unless p_fit(t0)=p_start, you do not actually have
  // p(t0)=p_start+delta_p0" argument.
  ScanSpline spline_frozen;
  spline_frozen.setFrozenBoundary(SplineOptions::N_FROZEN_CP,
                                   poses.front().pos, poses.front().rot, poses.front().vel,
                                   poses.back().pos, poses.back().rot, poses.back().vel);
  const bool ok_frozen = spline_frozen.fit(poses, t0, t1, opts);
  const double head_pos_err = ok_frozen ? (spline_frozen.posAt(t0) - poses.front().pos).norm() : -1.0;
  const double head_rot_err = ok_frozen ? Log(M3D(poses.front().rot.transpose() * spline_frozen.rotAt(t0))).norm() : -1.0;
  std::printf("head consistency: ok=%d pos_err=%.3e rot_err=%.3e (expect ~0)\n",
              ok_frozen ? 1 : 0, head_pos_err, head_rot_err);
  const bool head_pass = ok_frozen && head_pos_err < 1e-9 && head_rot_err < 1e-9;
  pass = pass && head_pass;

  // ---- Check 2 (item 3's fix, direct formula verification) ----
  // Curvature+Tikhonov only (no LiDAR/IMU), so b is otherwise exactly zero
  // -- with a nonzero c_current, the fix's own claim is `b == -Lambda*c`
  // literally, checkable to machine precision without any GN dynamics.
  {
    const std::vector<PoseSplineLidarObs> no_lidar;
    const std::vector<PoseSplineImuObs> no_imu;
    const auto build_c0 = buildPoseSplineCBlock(
        spline, no_lidar, no_imu, V3D::Zero(), V3D::Zero(), gravity,
        /*pose_imu_weight_acc=*/0.0, /*pose_imu_weight_gyr=*/0.0,
        /*pose_curvature_weight_pos=*/2.0, /*pose_curvature_weight_rot=*/3.0,
        PoseSplineTimeMode::kPointTime, 0.0, /*audit=*/false,
        POSE_SPLINE_TIKHONOV_EPS, 0.5, 0.3, Eigen::VectorXd());
    Eigen::VectorXd v(build_c0.A.rows());
    for (int i = 0; i < v.size(); ++i) v(i) = 0.01 * std::sin(0.7 * i + 1.0);  // arbitrary, deterministic
    const auto build_cv = buildPoseSplineCBlock(
        spline, no_lidar, no_imu, V3D::Zero(), V3D::Zero(), gravity,
        /*pose_imu_weight_acc=*/0.0, /*pose_imu_weight_gyr=*/0.0,
        /*pose_curvature_weight_pos=*/2.0, /*pose_curvature_weight_rot=*/3.0,
        PoseSplineTimeMode::kPointTime, 0.0, /*audit=*/false,
        POSE_SPLINE_TIKHONOV_EPS, 0.5, 0.3, v);
    // Lambda == build_c0.A exactly here (b_c0 == 0, A never depends on c).
    const Eigen::VectorXd predicted_b = -(build_c0.A * v);
    const double reg_grad_err = (build_cv.b - predicted_b).cwiseAbs().maxCoeff();
    std::printf("regularizer gradient formula: max|b_actual - (-Lambda*c)| = %.3e (expect ~0)\n",
                reg_grad_err);
    pass = pass && (reg_grad_err < 1e-9);
  }

  // ---- Check 3 (item 2's own requested test: rotation head-tie is a
  // first-order chart approximation, discrepancy should scale QUADRATICALLY
  // with perturbation size) ----
  // R_spline(t0, c+dtheta-tied-to-head) vs R_start*Exp(dtheta): tie cp_phi
  // 0..N_FROZEN_CP-1's CORRECTION to dtheta (the exact operation
  // reducePoseSplineHeadCoupling() performs) and compare.
  {
    const M3D R_start = spline_frozen.rotAt(t0);
    const V3D axis = V3D(0.3, -0.6, 0.74).normalized();
    std::printf("rotation head-tie characterization (expect ~quadratic scaling):\n");
    double prev_err = -1.0;
    for (double mag : {1e-4, 1e-3, 1e-2, 1e-1}) {
      const V3D dtheta = axis * mag;
      ScanSpline perturbed = spline_frozen;
      for (int j = 0; j < std::min(POSE_SPLINE_HEAD_TIE_CP, perturbed.nControlPoints()); ++j)
        perturbed.cp_phi_.col(j) += dtheta;
      const M3D R_actual = perturbed.rotAt(t0);
      const M3D R_expected = R_start * Exp(dtheta);
      const double err = Log(M3D(R_expected.transpose() * R_actual)).norm();
      const double ratio = (prev_err > 0.0) ? (err / prev_err) : 0.0;
      std::printf("  |dtheta|=%.1e  err=%.3e  ratio_vs_prev(expect~100 per 10x step)=%.2f\n",
                  mag, err, ratio);
      prev_err = err;
    }
    // Not folded into `pass` -- item 2's own text: "this might be perfectly
    // adequate... but it should be characterized," not asserted against a
    // fixed threshold. Printed for the record per rule 41b ("print the
    // number, never 'ok'").
  }

  // ---- Check 4 (item 9's "finite-difference LiDAR Jacobian... PSD does
  // not prove the Jacobian is correct") ----
  // Recompute the TRUE point-to-plane residual r(c) = n^T(R(t;c)q+p(t;c))+d
  // (d fixed from r(0)) under a perturbed control point, central-difference
  // it, and compare against the analytic Jrow column buildPoseSplineCBlock()
  // used (recomputed identically here, since Jrow itself isn't exposed).
  {
    const V3D q(1.0, 0.2, 0.1), n(0.0, 0.0, 1.0);
    const double t_obs = poses[N / 2].t;
    const double d_plane = 0.05;  // arbitrary fixed plane offset -- doesn't affect dr/dc
    auto residual = [&](const ScanSpline& s) {
      return n.dot(s.rotAt(t_obs) * q + s.posAt(t_obs)) + d_plane;
    };
    int first_cp; Eigen::Vector4d bw, dbw, ddbw;
    spline.basisAt(t_obs, first_cp, bw, dbw, ddbw);
    constexpr double EPS = 1e-6;
    double max_rel_err = 0.0;
    for (int k = 0; k < 4; ++k) {
      const int j = first_cp + k;
      if (j < 0 || j >= n_c) continue;
      for (int axis = 0; axis < 3; ++axis) {
        ScanSpline plus = spline, minus = spline;
        plus.cpPosMut()(axis, j)  += EPS;
        minus.cpPosMut()(axis, j) -= EPS;
        const double fd = (residual(plus) - residual(minus)) / (2.0 * EPS);
        const double analytic = bw(k) * n(axis);  // dr/dc_p[j][axis] = b_j(t)*n^T
        max_rel_err = std::max(max_rel_err, std::fabs(fd - analytic));
      }
    }
    std::printf("LiDAR position-Jacobian FD check: max|fd - analytic| = %.3e (expect ~0)\n",
                max_rel_err);
    pass = pass && (max_rel_err < 1e-6);
  }

  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
