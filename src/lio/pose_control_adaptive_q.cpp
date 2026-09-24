#include "livo_recon/lio/pose_control_adaptive_q.h"

#include <Eigen/Eigenvalues>
#include <algorithm>

namespace livo_recon
{

SplineImuResidualStats computePoseControlImuResidual(
    const PoseControlSpline& spline, const std::vector<ImuSample>& imu,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity)
{
  if (imu.size() < 8) return SplineImuResidualStats();

  std::vector<V3D> ra; ra.reserve(imu.size());
  std::vector<V3D> rw; rw.reserve(imu.size());

  const double t0 = spline.t0(), t1 = spline.t1();
  for (const auto& s : imu)
  {
    if (s.t < t0 || s.t > t1) continue;
    const M3D R = spline.rotAt(s.t);
    const V3D a_pred = R.transpose() * (spline.accAt(s.t) - gravity) + bias_acc;
    const V3D w_pred = spline.omegaBodyAt(s.t) + bias_gyr;
    ra.push_back(a_pred - s.acc);
    rw.push_back(w_pred - s.gyro);
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

}  // namespace livo_recon
