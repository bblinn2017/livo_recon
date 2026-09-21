#include "livo_recon/lio/pose_spline_system.h"
#include "livo_recon/utils/algo/math.h"

#include <cmath>

namespace livo_recon
{

namespace
{

// Central finite difference: how much does `spline`'s IMU-factor residual
// (acc or omega, at time t) change per unit perturbation of one control
// point, in one of its 3 tangent/position directions? Returns the 3-vector
// d(residual)/d(delta), delta a scalar perturbation along axis `axis` of
// control point `cp_idx`, in either the position (is_pos=true) or tangent-
// rotation (is_pos=false) control-point array.
V3D fdImuJacobianCol(const ScanSpline& spline, int cp_idx, int axis, bool is_pos,
                     double t, bool want_acc, const V3D& bias_acc, const V3D& bias_gyr,
                     const V3D& gravity)
{
  constexpr double EPS = 1e-6;
  ScanSpline plus = spline, minus = spline;
  if (is_pos) {
    plus.cpPosMut()(axis, cp_idx) += EPS;
    minus.cpPosMut()(axis, cp_idx) -= EPS;
  } else {
    plus.cp_phi_(axis, cp_idx) += EPS;
    minus.cp_phi_(axis, cp_idx) -= EPS;
  }
  V3D r_plus, r_minus;
  if (want_acc) {
    r_plus  = plus.rotAt(t).transpose()  * (plus.accAt(t)  - gravity) + bias_acc;
    r_minus = minus.rotAt(t).transpose() * (minus.accAt(t) - gravity) + bias_acc;
  } else {
    r_plus  = plus.omegaBodyAt(t)  + bias_gyr;
    r_minus = minus.omegaBodyAt(t) + bias_gyr;
  }
  return (r_plus - r_minus) / (2.0 * EPS);
}

}  // namespace

PoseSplineCBlockBuild buildPoseSplineCBlock(
    const ScanSpline& spline,
    const std::vector<PoseSplineLidarObs>& lidar_obs,
    const std::vector<PoseSplineImuObs>& imu_obs,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    double pose_imu_weight_acc, double pose_imu_weight_gyr,
    double pose_curvature_weight_pos, double pose_curvature_weight_rot,
    PoseSplineTimeMode time_mode, double end_time_t1, bool audit)
{
  const int n_c = spline.nControlPoints();
  const int ncol_c = 6 * n_c;  // [c_p(3n_c); c_phi(3n_c)]

  PoseSplineCBlockBuild build;
  build.A = Eigen::MatrixXd::Identity(ncol_c, ncol_c) * POSE_SPLINE_TIKHONOV_EPS;
  build.b = Eigen::VectorXd::Zero(ncol_c);
  Eigen::MatrixXd& A = build.A;
  Eigen::VectorXd& b = build.b;

  Eigen::MatrixXd A_lidar_only;
  Eigen::MatrixXd phic_spread_sum;
  double phic_spread_sumsq = 0.0;
  int phic_spread_n = 0;
  if (audit) A_lidar_only = Eigen::MatrixXd::Zero(ncol_c, ncol_c);

  // ---- smoothness (second-difference, separately weighted) ----
  if ((pose_curvature_weight_pos > 0.0 || pose_curvature_weight_rot > 0.0) && n_c >= 3) {
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n_c - 2, n_c);
    for (int k = 0; k < n_c - 2; ++k) { D(k, k) = 1.0; D(k, k + 1) = -2.0; D(k, k + 2) = 1.0; }
    const Eigen::MatrixXd Curv = D.transpose() * D;
    for (int i = 0; i < n_c; ++i)
      for (int j = 0; j < n_c; ++j) {
        if (pose_curvature_weight_pos > 0.0)
          A.block<3, 3>(3 * i, 3 * j) += pose_curvature_weight_pos * Curv(i, j) * M3D::Identity();
        if (pose_curvature_weight_rot > 0.0)
          A.block<3, 3>(3 * n_c + 3 * i, 3 * n_c + 3 * j) +=
              pose_curvature_weight_rot * Curv(i, j) * M3D::Identity();
      }
  }

  // ---- LiDAR term: linear in c_p, chain rule in c_phi via Jr(phi) ----
  build.n_lidar = static_cast<int>(lidar_obs.size());
  for (const auto& obs : lidar_obs) {
    // CQ-86 item 0: point_time (default) evaluates the basis weight/rotation
    // chain at this residual's OWN capture time obs.t, as the pose arm has
    // always done. end_time instead pins that same evaluation to the FIXED
    // scan-end time end_time_t1 for every residual -- only obs.normal and
    // obs.raw_body_point still vary per point -- the direct structural
    // mirror of raw_imu's own jacobian_time_mode=end_time construction.
    const double basis_t = (time_mode == PoseSplineTimeMode::kEndTime) ? end_time_t1 : obs.t;
    int first_cp; Eigen::Vector4d bw, dbw, ddbw;
    spline.basisAt(basis_t, first_cp, bw, dbw, ddbw);
    const V3D hk = obs.raw_body_point.cross(spline.rotAt(basis_t).transpose() * obs.normal);
    const M3D Jr_phi = Jr(spline.phiAt(basis_t));
    const Eigen::RowVector3d dr_dtheta = (hk.transpose() * Jr_phi);
    const double w = 1.0 / std::max(obs.sigma2, 1e-18);

    Eigen::Matrix<double, 1, Eigen::Dynamic> Jrow(1, ncol_c);
    Jrow.setZero();
    for (int k = 0; k < 4; ++k) {
      const int j = first_cp + k;
      if (j < 0 || j >= n_c) continue;
      Jrow.segment<3>(3 * j)           = bw(k) * obs.normal.transpose();
      Jrow.segment<3>(3 * n_c + 3 * j) = bw(k) * dr_dtheta;
    }
    A.noalias() += w * (Jrow.transpose() * Jrow);
    b.noalias() -= w * Jrow.transpose() * obs.r;

    if (audit) {
      // CQ-86 item 0: the residual-loop-only contribution (rank_cblock's
      // own input) and the Jrow spread (phic_spread's own input) -- both
      // UNWEIGHTED by sigma2, matching CQ-66's own H6_raw_accum/phic_spread
      // convention of measuring the raw sensitivity structure, not the
      // information-weighted one.
      A_lidar_only.noalias() += Jrow.transpose() * Jrow;
      if (phic_spread_sum.size() == 0) phic_spread_sum = Eigen::MatrixXd::Zero(1, ncol_c);
      phic_spread_sum.noalias() += Jrow;
      phic_spread_sumsq += Jrow.squaredNorm();
      ++phic_spread_n;
    }
  }

  if (audit) {
    build.A_lidar_only = A_lidar_only;
    if (phic_spread_n > 0) {
      const Eigen::MatrixXd mean_jrow = phic_spread_sum / static_cast<double>(phic_spread_n);
      const double var = phic_spread_sumsq / static_cast<double>(phic_spread_n) - mean_jrow.squaredNorm();
      build.phic_spread = std::sqrt(std::max(var, 0.0));
    }
  }

  // ---- IMU-as-measurement-factor: per raw sample, FD Jacobian ----
  build.n_imu = static_cast<int>(imu_obs.size());
  // A fixed, generous floor noise so the term is well-scaled even before any
  // real calibration is threaded through -- this builder is a standalone
  // correctness-gate harness (see the header's own doc comment), not yet
  // wired to state_'s own calibrated sigma_a/sigma_g.
  constexpr double SIGMA_A = 0.5, SIGMA_G = 0.3;
  const double w_acc = pose_imu_weight_acc / (SIGMA_A * SIGMA_A);
  const double w_gyr = pose_imu_weight_gyr / (SIGMA_G * SIGMA_G);
  for (const auto& obs : imu_obs) {
    if (pose_imu_weight_acc > 0.0) {
      const V3D r_a = spline.rotAt(obs.t).transpose() * (spline.accAt(obs.t) - gravity) +
                      bias_acc - obs.acc;
      Eigen::Matrix<double, 3, Eigen::Dynamic> J(3, ncol_c);
      J.setZero();
      // Position columns are linear -- analytic: d(pddot)/dc_p[j] = ddb_j(t)*I3,
      // rotated by R(t)^T (the acc term's own frame). Rotation columns: FD.
      int first_cp; Eigen::Vector4d bw, dbw, ddbw;
      spline.basisAt(obs.t, first_cp, bw, dbw, ddbw);
      const M3D Rt_T = spline.rotAt(obs.t).transpose();
      for (int k = 0; k < 4; ++k) {
        const int j = first_cp + k;
        if (j < 0 || j >= n_c) continue;
        J.block<3, 3>(0, 3 * j) = ddbw(k) * Rt_T;
        for (int axis = 0; axis < 3; ++axis)
          J.block<3, 1>(0, 3 * n_c + 3 * j + axis) =
              fdImuJacobianCol(spline, j, axis, false, obs.t, true, bias_acc, bias_gyr, gravity);
      }
      A.noalias() += w_acc * (J.transpose() * J);
      b.noalias() -= w_acc * J.transpose() * r_a;
    }
    if (pose_imu_weight_gyr > 0.0) {
      const V3D r_g = spline.omegaBodyAt(obs.t) + bias_gyr - obs.gyr;
      Eigen::Matrix<double, 3, Eigen::Dynamic> J(3, ncol_c);
      J.setZero();
      int first_cp; Eigen::Vector4d bw, dbw, ddbw;
      spline.basisAt(obs.t, first_cp, bw, dbw, ddbw);
      for (int k = 0; k < 4; ++k) {
        const int j = first_cp + k;
        if (j < 0 || j >= n_c) continue;
        // omega_body has no c_p dependence at all -- left zero.
        for (int axis = 0; axis < 3; ++axis)
          J.block<3, 1>(0, 3 * n_c + 3 * j + axis) =
              fdImuJacobianCol(spline, j, axis, false, obs.t, false, bias_acc, bias_gyr, gravity);
      }
      A.noalias() += w_gyr * (J.transpose() * J);
      b.noalias() -= w_gyr * J.transpose() * r_g;
    }
  }

  return build;
}

}  // namespace livo_recon
