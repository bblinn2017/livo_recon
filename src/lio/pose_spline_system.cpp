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
    PoseSplineTimeMode time_mode, double end_time_t1, bool audit,
    double tikhonov_eps, double sigma_acc, double sigma_gyr,
    const Eigen::VectorXd& c_current)
{
  const int n_c = spline.nControlPoints();
  const int ncol_c = 6 * n_c;  // [c_p(3n_c); c_phi(3n_c)]

  PoseSplineCBlockBuild build;
  build.A = Eigen::MatrixXd::Identity(ncol_c, ncol_c) * tikhonov_eps;
  build.b = Eigen::VectorXd::Zero(ncol_c);
  Eigen::MatrixXd& A = build.A;
  Eigen::VectorXd& b = build.b;
  // POST-CQ-87-REVIEW FIX (item 3): mirrors every addition to `A` made by
  // the curvature/Tikhonov prior ONLY (never the LiDAR/IMU measurement
  // terms) -- see this function's own declaration-site doc comment for
  // the derivation. Applied to `b` once, after the prior's own
  // contribution to `A` is fully assembled below.
  Eigen::MatrixXd Lambda_reg = Eigen::MatrixXd::Identity(ncol_c, ncol_c) * tikhonov_eps;

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
        if (pose_curvature_weight_pos > 0.0) {
          const M3D block = pose_curvature_weight_pos * Curv(i, j) * M3D::Identity();
          A.block<3, 3>(3 * i, 3 * j) += block;
          Lambda_reg.block<3, 3>(3 * i, 3 * j) += block;
        }
        if (pose_curvature_weight_rot > 0.0) {
          const M3D block = pose_curvature_weight_rot * Curv(i, j) * M3D::Identity();
          A.block<3, 3>(3 * n_c + 3 * i, 3 * n_c + 3 * j) += block;
          Lambda_reg.block<3, 3>(3 * n_c + 3 * i, 3 * n_c + 3 * j) += block;
        }
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
  // POST-CQ-87-REVIEW FIX (item 4): sigma_acc/sigma_gyr are now caller-
  // supplied (the live call site passes state_->varAcc()/varGyr(), the
  // SAME calibrated noise the raw_imu arm solves against) rather than a
  // fixed placeholder -- see this function's own declaration-site doc
  // comment. Floored to avoid a division blowup if a caller ever passes
  // (near-)zero.
  const double sigma_a_eff = std::max(sigma_acc, 1e-6);
  const double sigma_g_eff = std::max(sigma_gyr, 1e-6);
  const double w_acc = pose_imu_weight_acc / (sigma_a_eff * sigma_a_eff);
  const double w_gyr = pose_imu_weight_gyr / (sigma_g_eff * sigma_g_eff);
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

  // POST-CQ-87-REVIEW FIX (item 3), applied last (order doesn't matter --
  // this is a plain sum into `b`, same as every LiDAR/IMU contribution
  // above): center the curvature/Tikhonov prior at the CURRENT accumulated
  // correction rather than at 0. Empty c_current (default) means "no
  // correction, reproduce old behavior" -- exactly right for iteration 1
  // (c=0) and for any caller (e.g. this file's own correctness-gate test)
  // that never accumulates a c at all.
  if (c_current.size() == ncol_c) {
    b.noalias() -= Lambda_reg * c_current;
  }

  return build;
}

// CQ-87 items 1/2/3 -- see this function's own declaration-site doc
// comment (pose_spline_system.h) for the full derivation and the
// mechanism chosen. Implemented as a direct T^T A T reduction via
// per-column destination indices, then the prior is seeded into the
// resulting [delta_phi0;delta_pos0] block -- both O(n_c^2), trivial cost
// at this project's n_c range (4-13).
PoseSplineReducedSystem reducePoseSplineHeadCoupling(
    const PoseSplineCBlockBuild& build, int n_c,
    const Eigen::Matrix<double, 6, 6>& pi_ss,
    const Eigen::Matrix<double, 6, 1>& s_vec)
{
  constexpr int kTie = POSE_SPLINE_HEAD_TIE_CP;
  const int n_free = std::max(0, n_c - kTie);
  const int ncol_orig = 6 * n_c;
  const int new_dim = 6 * n_free + 6;

  PoseSplineReducedSystem out;
  out.n_free = n_free;
  out.A = Eigen::MatrixXd::Zero(new_dim, new_dim);
  out.b = Eigen::VectorXd::Zero(new_dim);
  if (n_c <= 0) return out;

  // dest[orig_col] = destination row/col in the reduced system. The tied
  // head columns (c_p and c_phi, control points 0..kTie-1) all map to the
  // SAME 3 destination columns each (delta_phi0 or delta_pos0) -- that
  // repetition, summed via the loop below, is exactly the T^T A T
  // reduction; free columns (control points kTie..n_c-1) each get their
  // own unique destination.
  std::vector<int> dest(ncol_orig, -1);
  int fp = 0;
  for (int j = kTie; j < n_c; ++j)
    for (int a = 0; a < 3; ++a) dest[3 * j + a] = fp++;
  int fr = 0;
  for (int j = kTie; j < n_c; ++j)
    for (int a = 0; a < 3; ++a) dest[3 * n_c + 3 * j + a] = 3 * n_free + fr++;
  for (int j = 0; j < std::min(kTie, n_c); ++j)
    for (int a = 0; a < 3; ++a) {
      dest[3 * j + a]           = 6 * n_free + 3 + a;  // delta_pos0 block
      dest[3 * n_c + 3 * j + a] = 6 * n_free + a;       // delta_phi0 block
    }

  for (int r = 0; r < ncol_orig; ++r) {
    const int dr = dest[r];
    if (dr < 0) continue;
    out.b(dr) += build.b(r);
    for (int c = 0; c < ncol_orig; ++c) {
      const int dc = dest[c];
      if (dc < 0) continue;
      out.A(dr, dc) += build.A(r, c);
    }
  }

  // The prior: SAME normal-equations convention the raw_imu arm's own
  // Pi_ss/s_vec seeding uses (lio_coupled.cpp:1903/1910) -- added ON TOP
  // of the reduced LiDAR/IMU/curvature/Tikhonov contributions already
  // summed into this block above (addition order doesn't matter for a
  // sum, so this is exactly equivalent to seeding first, as the raw_imu
  // arm's own code does).
  out.A.block<6, 6>(6 * n_free, 6 * n_free) += pi_ss;
  out.b.segment<6>(6 * n_free) -= pi_ss * s_vec;

  return out;
}

}  // namespace livo_recon
