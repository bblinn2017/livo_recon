#include "livo_recon/lio/pose_control_adaptive_q.h"
#include "livo_recon/utils/algo/omp_utils.h"

#include <Eigen/Eigenvalues>
#include <algorithm>

namespace livo_recon
{

SplineImuResidualStats computePoseControlImuResidual(
    const PoseControlSpline& spline, const std::vector<ImuSample>& imu,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity)
{
  if (imu.size() < 8) return SplineImuResidualStats();

  // Filter to the in-window subset FIRST (a fixed, index-addressable
  // sequence, still in chronological order), then compute each sample's
  // residual in a parallel loop that writes to a fixed, disjoint index
  // per sample -- item 9: "per-thread local statistics, one merge after
  // the parallel loop". There is no reduction race here at all (each
  // thread only ever writes its own indices, and the final reduction
  // (reduceImuResidualSamples(), which needs the ORIGINAL chronological
  // order for its lag-k autocorrelation) reads the fully-populated,
  // correctly-ordered array afterward, single-threaded) -- this is safer
  // than an append-and-concatenate pattern for an order-sensitive
  // reduction. In practice a scan's in-window IMU sample count is tens,
  // not thousands (unlike the LiDAR residual loop, which genuinely
  // benefits from cappedOmpThreads()'s full thread count) -- the loop is
  // still threaded here for the required pattern/correctness property,
  // not because it is expected to move the needle on wall-clock time.
  std::vector<ImuSample> windowed;
  windowed.reserve(imu.size());
  const double t0 = spline.t0(), t1 = spline.t1();
  for (const auto& s : imu) if (s.t >= t0 && s.t <= t1) windowed.push_back(s);

  const int n = static_cast<int>(windowed.size());
  std::vector<V3D> ra(n), rw(n);
  const int threads = std::max(1, std::min(cappedOmpThreads(), std::max(1, n)));
  #pragma omp parallel for num_threads(threads) schedule(static)
  for (int i = 0; i < n; ++i)
  {
    const ImuSample& s = windowed[i];
    const M3D R = spline.rotAt(s.t);
    const V3D a_pred = R.transpose() * (spline.accAt(s.t) - gravity) + bias_acc;
    const V3D w_pred = spline.omegaBodyAt(s.t) + bias_gyr;
    ra[i] = a_pred - s.acc;
    rw[i] = w_pred - s.gyro;
  }

  return reduceImuResidualSamples(ra, rw);
}

namespace
{
// Clamp a symmetric 3x3 to the PSD cone (item 24: "project C_extra onto the
// PSD cone") and return its trace/3 -- the isotropic scalar this estimator's
// downstream AdaptiveQ::update() consumes.
double clampedTraceOver3(const Eigen::Matrix3d& C)
{
  const Eigen::Matrix3d Csym = 0.5 * (C + C.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Csym);
  const Eigen::Vector3d ev = es.eigenvalues().cwiseMax(0.0);
  return ev.sum() / 3.0;
}
}  // namespace

void applyPoseControlAdaptiveQBiasGravityCorrection(
    SplineImuResidualStats& st, const M3D& R_rep,
    const Eigen::Matrix3d& P_ba, const Eigen::Matrix3d& P_bg,
    const Eigen::Matrix3d& P_g, const Eigen::Matrix3d& P_ba_g_cross)
{
  if (st.n <= 0) return;

  // C_pred_acc = J_ba P_ba J_ba^T + J_g P_g J_g^T + 2*sym(J_ba P_ba_g J_g^T),
  // J_ba = I, J_g = -R_rep^T (see this file's header comment for the
  // derivation).
  const Eigen::Matrix3d Rt = R_rep.transpose();
  const Eigen::Matrix3d C_pred_acc =
      P_ba + Rt * P_g * Rt.transpose() - P_ba_g_cross * R_rep - Rt * P_ba_g_cross.transpose();
  const Eigen::Matrix3d C_emp_acc = Eigen::Matrix3d::Identity() * st.cov_acc;
  st.cov_acc = clampedTraceOver3(C_emp_acc - C_pred_acc);

  // C_pred_gyr = P_bg (J_bg = I, no gravity dependence).
  const Eigen::Matrix3d C_emp_gyr = Eigen::Matrix3d::Identity() * st.cov_gyr;
  st.cov_gyr = clampedTraceOver3(C_emp_gyr - P_bg);
}

namespace
{
inline M3D skew3v(const V3D& v)
{
  M3D S;
  S <<     0, -v.z(),  v.y(),
        v.z(),      0, -v.x(),
       -v.y(),  v.x(),      0;
  return S;
}
}  // namespace

void computePoseControlImuResidualStateJacobian(
    const PoseControlSpline& spline, const PoseControlFreeLayout& layout,
    const PoseControlHeadNullspace& hns, double t, const V3D& gravity,
    Eigen::MatrixXd& J_acc_eta, Eigen::MatrixXd& J_gyr_eta,
    Eigen::Matrix<double, 3, 9>& J_acc_head, Eigen::Matrix<double, 3, 9>& J_gyr_head)
{
  const int rawDim = hns.rawDim();
  Eigen::MatrixXd J_acc_raw = Eigen::MatrixXd::Zero(3, rawDim);
  Eigen::MatrixXd J_gyr_raw = Eigen::MatrixXd::Zero(3, rawDim);

  const auto jac = spline.jacobianAt(t);
  const M3D R = spline.rotAt(t);
  const V3D v = spline.accAt(t) - gravity;
  // e_acc = R(t)^T*v + bias_acc - a_measured. Under the SAME body-frame
  // right-perturbation convention as the LiDAR factor's own dr_dtheta
  // (R -> R*Exp(theta)): R(t)^T*v -> Exp(theta)^T*(R^T*v)
  // ~= (R^T*v) - theta x (R^T*v), so d(R^T*v)/d(theta) = skew(R^T*v).
  // Sign/convention verified by finite difference in the registered test
  // (testStateJacobianFiniteDifference), not asserted from derivation
  // alone.
  const V3D Rtv = R.transpose() * v;
  const M3D d_acc_d_theta = skew3v(Rtv);

  bool touches_head_pos = false;
  for (int k = 0; k < 4; ++k)
  {
    const int abs_k = jac.s + k;
    const int colp = layout.colPos(abs_k);
    const int colph = layout.colPhi(abs_k);
    if (colp >= 0)
      J_acc_raw.block<3, 3>(0, colp) = R.transpose() * PoseControlSpline::dAccDcp(jac, k);
    if (colph >= 0)
    {
      const M3D dtheta_dcphi = spline.dThetaDcphi(jac, k, t);
      J_acc_raw.block<3, 3>(0, colph) = d_acc_d_theta * dtheta_dcphi;
      J_gyr_raw.block<3, 3>(0, colph) = spline.dOmegaDcphi(jac, k, t);
    }
    if (abs_k < 3) touches_head_pos = true;
  }

  J_acc_eta = J_acc_raw * hns.Z;
  J_gyr_eta = J_gyr_raw * hns.Z;

  // Head columns [theta0(0:3); p0(3:6); v0(6:9)] -- SAME layout as
  // Sigma_full_post's topLeftCorner(9,9) and pose_control_imu_prior_
  // builder.cpp's own Jhead. e_gyr's head sensitivity is exactly zero
  // (see this file's header comment) -- left as the zero matrix, not
  // assumed away silently.
  J_acc_head.setZero();
  J_gyr_head.setZero();
  J_acc_head.block<3, 3>(0, 0) = d_acc_d_theta * poseControlHeadRotJacobian(spline, t);
  if (touches_head_pos)
  {
    const PoseControlHeadPosSensitivity hs = poseControlHeadPosSensitivity(spline);
    M3D da_dp0 = M3D::Zero(), da_dv0 = M3D::Zero();
    for (int k = 0; k < 4; ++k)
    {
      const int abs_k = jac.s + k;
      if (abs_k >= 3) continue;
      const double m0 = hs.Minv(abs_k, 0), m1 = hs.Minv(abs_k, 1);
      da_dp0 += m0 * PoseControlSpline::dAccDcp(jac, k);
      da_dv0 += m1 * PoseControlSpline::dAccDcp(jac, k);
    }
    J_acc_head.block<3, 3>(0, 3) = R.transpose() * da_dp0;
    J_acc_head.block<3, 3>(0, 6) = R.transpose() * da_dv0;
  }
}

void applyPoseControlAdaptiveQTrajectoryStateCorrection(
    SplineImuResidualStats& st,
    const Eigen::MatrixXd& J_acc_eta, const Eigen::MatrixXd& J_gyr_eta,
    const Eigen::Matrix<double, 3, 9>& J_acc_head, const Eigen::Matrix<double, 3, 9>& J_gyr_head,
    const Eigen::MatrixXd& Sigma_head_eta)
{
  if (st.n <= 0) return;
  const int dEta = static_cast<int>(J_acc_eta.cols());
  if (J_gyr_eta.cols() != dEta) return;
  if (Sigma_head_eta.rows() != 9 + dEta || Sigma_head_eta.cols() != 9 + dEta) return;

  Eigen::MatrixXd J_full_acc(3, 9 + dEta), J_full_gyr(3, 9 + dEta);
  J_full_acc << J_acc_head, J_acc_eta;
  J_full_gyr << J_gyr_head, J_gyr_eta;

  const Eigen::Matrix3d C_pred_traj_acc = J_full_acc * Sigma_head_eta * J_full_acc.transpose();
  const Eigen::Matrix3d C_emp_acc = Eigen::Matrix3d::Identity() * st.cov_acc;
  st.cov_acc = clampedTraceOver3(C_emp_acc - C_pred_traj_acc);

  const Eigen::Matrix3d C_pred_traj_gyr = J_full_gyr * Sigma_head_eta * J_full_gyr.transpose();
  const Eigen::Matrix3d C_emp_gyr = Eigen::Matrix3d::Identity() * st.cov_gyr;
  st.cov_gyr = clampedTraceOver3(C_emp_gyr - C_pred_traj_gyr);
}

}  // namespace livo_recon
