#include "livo_recon/diagnostics/pose_control/pose_control_physical_diagnostics.h"
#include "livo_recon/lio/pose_control_adaptive_q.h"
#include "livo_recon/utils/algo/omp_utils.h"

#include <Eigen/Eigenvalues>
#include <algorithm>

namespace livo_recon
{

namespace
{
inline M3D skew3v_(const V3D& v)
{
  M3D S;
  S <<     0, -v.z(),  v.y(),
        v.z(),      0, -v.x(),
       -v.y(),  v.x(),      0;
  return S;
}
}  // namespace

PoseControlPhysicalSample evaluatePoseControlPhysicalSample(
    const PoseControlSpline& spline, const PoseControlFreeLayout& layout,
    const PoseControlHeadNullspace& hns, double t, const V3D& gravity)
{
  PoseControlPhysicalSample out;
  out.t = t;
  out.p = spline.posAt(t);
  out.v = spline.velAt(t);
  out.a = spline.accAt(t);
  out.R = spline.rotAt(t);
  out.omega = spline.omegaBodyAt(t);

  const int rawDim = hns.rawDim();
  Eigen::MatrixXd dp_raw = Eigen::MatrixXd::Zero(3, rawDim);
  Eigen::MatrixXd dv_raw = Eigen::MatrixXd::Zero(3, rawDim);
  Eigen::MatrixXd da_raw = Eigen::MatrixXd::Zero(3, rawDim);
  Eigen::MatrixXd domega_raw = Eigen::MatrixXd::Zero(3, rawDim);
  Eigen::MatrixXd dtheta_raw = Eigen::MatrixXd::Zero(3, rawDim);

  const auto jac = spline.jacobianAt(t);
  for (int k = 0; k < 4; ++k)
  {
    const int abs_k = jac.s + k;
    const int colp = layout.colPos(abs_k);
    const int colph = layout.colPhi(abs_k);
    if (colp >= 0)
    {
      dp_raw.block<3, 3>(0, colp) = PoseControlSpline::dPosDcp(jac, k);
      dv_raw.block<3, 3>(0, colp) = PoseControlSpline::dVelDcp(jac, k);
      da_raw.block<3, 3>(0, colp) = PoseControlSpline::dAccDcp(jac, k);
    }
    if (colph >= 0)
    {
      domega_raw.block<3, 3>(0, colph) = spline.dOmegaDcphi(jac, k, t);
      dtheta_raw.block<3, 3>(0, colph) = spline.dThetaDcphi(jac, k, t);
    }
  }

  out.dp_deta = dp_raw * hns.Z;
  out.dv_deta = dv_raw * hns.Z;
  out.da_deta = da_raw * hns.Z;
  out.domega_deta = domega_raw * hns.Z;
  out.dtheta_deta = dtheta_raw * hns.Z;
  (void)gravity;  // a(t) here is the WORLD-frame spline acceleration itself (gravity not subtracted) -- see e_acc's own convention for the specific-force version.
  return out;
}

namespace
{
}  // namespace

std::vector<ImuSplineResidualSample> computePoseControlImuSplineResidualSamples(
    const PoseControlSpline& spline, const std::vector<ImuSample>& imu,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity)
{
  std::vector<ImuSplineResidualSample> windowed_samples;
  const double t0 = spline.t0(), t1 = spline.t1();
  std::vector<ImuSample> windowed;
  windowed.reserve(imu.size());
  for (const auto& s : imu) if (s.t >= t0 && s.t <= t1) windowed.push_back(s);

  const int n = static_cast<int>(windowed.size());
  windowed_samples.resize(n);
  const int threads = std::max(1, std::min(cappedOmpThreads(), std::max(1, n)));
  #pragma omp parallel for num_threads(threads) schedule(static)
  for (int i = 0; i < n; ++i)
  {
    const ImuSample& s = windowed[i];
    ImuSplineResidualSample out;
    out.t = s.t;
    const M3D R = spline.rotAt(s.t);
    out.a_meas = s.acc;
    out.a_spline_body = R.transpose() * (spline.accAt(s.t) - gravity);
    out.e_acc = out.a_spline_body + bias_acc - s.acc;
    out.omega_meas = s.gyro;
    out.omega_spline_body = spline.omegaBodyAt(s.t);
    out.e_gyr = out.omega_spline_body + bias_gyr - s.gyro;
    windowed_samples[i] = out;
  }
  return windowed_samples;
}

void computePoseControlImuMeasurementJacobianZ(
    const PoseControlSpline& spline, const PoseControlFreeLayout& layout,
    const PoseControlHeadNullspace& hns, double t, const V3D& gravity,
    int dimZ, int off_bg, int off_ba, int off_g,
    Eigen::MatrixXd& H_acc, Eigen::MatrixXd& H_gyr)
{
  const int rawDim = hns.rawDim();
  Eigen::MatrixXd J_acc_raw = Eigen::MatrixXd::Zero(3, rawDim);
  Eigen::MatrixXd J_gyr_raw = Eigen::MatrixXd::Zero(3, rawDim);

  const auto jac = spline.jacobianAt(t);
  const M3D R = spline.rotAt(t);
  const V3D v = spline.accAt(t) - gravity;
  const V3D Rtv = R.transpose() * v;
  const M3D d_acc_d_theta = skew3v_(Rtv);

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
  }

  const Eigen::MatrixXd J_acc_eta = J_acc_raw * hns.Z;
  const Eigen::MatrixXd J_gyr_eta = J_gyr_raw * hns.Z;
  const int dEta = hns.freeDim();

  H_acc = Eigen::MatrixXd::Zero(3, dimZ);
  H_gyr = Eigen::MatrixXd::Zero(3, dimZ);
  H_acc.leftCols(dEta) = J_acc_eta;
  H_gyr.leftCols(dEta) = J_gyr_eta;
  if (off_ba >= 0) H_acc.block(0, off_ba, 3, 3) = M3D::Identity();
  if (off_g  >= 0) H_acc.block(0, off_g,  3, 3) = -R.transpose();
  if (off_bg >= 0) H_gyr.block(0, off_bg, 3, 3) = M3D::Identity();
}

ImuMeasurementInformation computePoseControlImuMeasurementInformation(
    const Eigen::MatrixXd& H_acc, const Eigen::MatrixXd& H_gyr,
    const V3D& R_acc_diag, const V3D& R_gyr_diag,
    int n_acc, int n_gyr, const V3D& mean_e_acc, const V3D& mean_e_gyr)
{
  ImuMeasurementInformation out;
  const Eigen::Vector3d Rinv_acc = R_acc_diag.cwiseInverse();
  const Eigen::Vector3d Rinv_gyr = R_gyr_diag.cwiseInverse();

  out.Lambda_acc = static_cast<double>(std::max(0, n_acc)) *
      (H_acc.transpose() * Rinv_acc.asDiagonal() * H_acc);
  out.Lambda_gyr = static_cast<double>(std::max(0, n_gyr)) *
      (H_gyr.transpose() * Rinv_gyr.asDiagonal() * H_gyr);
  out.Lambda_imu_meas = out.Lambda_acc + out.Lambda_gyr;

  // NEGATIVE sign to match this codebase's own GN b-vector convention
  // (delta_z = A^-1 * b directly, no leading minus at solve time -- every
  // existing factor here, e.g. the curvature block's "b_raw += -w*r" and
  // the joint prior's "b += -Lambda_prior_z*r_prior_z", builds b as
  // -Jacobian^T * R^-1 * residual, not +).
  out.b_acc = -static_cast<double>(std::max(0, n_acc)) *
      (H_acc.transpose() * Rinv_acc.asDiagonal() * mean_e_acc);
  out.b_gyr = -static_cast<double>(std::max(0, n_gyr)) *
      (H_gyr.transpose() * Rinv_gyr.asDiagonal() * mean_e_gyr);
  out.b_imu_meas = out.b_acc + out.b_gyr;

  const Eigen::MatrixXd Lsym = 0.5 * (out.Lambda_imu_meas + out.Lambda_imu_meas.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Lsym);
  const Eigen::VectorXd ev = es.eigenvalues();
  out.trace_lambda_imu_meas = Lsym.trace();
  out.min_eig = ev.size() > 0 ? ev(0) : 0.0;
  out.max_eig = ev.size() > 0 ? ev(ev.size() - 1) : 0.0;
  out.condition = (out.min_eig > 1e-12) ? out.max_eig / out.min_eig
                                         : std::numeric_limits<double>::infinity();
  const double thresh = 1e-9 * std::max(1.0, out.max_eig);
  int rank = 0;
  for (int i = 0; i < ev.size(); ++i) if (ev(i) > thresh) ++rank;
  out.effective_rank = rank;
  return out;
}

ResidualToQAccounting computePoseControlResidualToQAccounting(
    double cov_acc_empirical, double cov_gyr_empirical,
    const M3D& R_rep, const Eigen::Matrix3d& P_ba, const Eigen::Matrix3d& P_bg,
    const Eigen::Matrix3d& P_g, const Eigen::Matrix3d& P_ba_g_cross,
    const Eigen::MatrixXd& J_acc_eta, const Eigen::MatrixXd& J_gyr_eta,
    const Eigen::MatrixXd& P_eta, double R_acc_sensor, double R_gyr_sensor)
{
  ResidualToQAccounting out;
  out.C_empirical_acc = cov_acc_empirical;
  out.C_empirical_gyr = cov_gyr_empirical;
  out.C_sensor_acc = R_acc_sensor;
  out.C_sensor_gyr = R_gyr_sensor;

  const Eigen::Matrix3d Rt = R_rep.transpose();
  const Eigen::Matrix3d C_pred_bg_acc =
      P_ba + Rt * P_g * Rt.transpose() - P_ba_g_cross * R_rep - Rt * P_ba_g_cross.transpose();
  Eigen::Matrix3d C_pred_traj_acc = Eigen::Matrix3d::Zero();
  if (J_acc_eta.cols() == P_eta.rows() && P_eta.rows() > 0)
    C_pred_traj_acc = J_acc_eta * P_eta * J_acc_eta.transpose();
  out.C_pred_state_acc = (C_pred_bg_acc + C_pred_traj_acc).trace() / 3.0;

  Eigen::Matrix3d C_pred_traj_gyr = Eigen::Matrix3d::Zero();
  if (J_gyr_eta.cols() == P_eta.rows() && P_eta.rows() > 0)
    C_pred_traj_gyr = J_gyr_eta * P_eta * J_gyr_eta.transpose();
  out.C_pred_state_gyr = (P_bg + C_pred_traj_gyr).trace() / 3.0;

  out.C_extra_acc = out.C_empirical_acc - out.C_pred_state_acc - out.C_sensor_acc;
  out.C_extra_acc_psd = std::max(0.0, out.C_extra_acc);
  out.C_extra_gyr = out.C_empirical_gyr - out.C_pred_state_gyr - out.C_sensor_gyr;
  out.C_extra_gyr_psd = std::max(0.0, out.C_extra_gyr);
  return out;
}

}  // namespace livo_recon
