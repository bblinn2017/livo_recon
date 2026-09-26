#include "livo_recon/lio/pose_control_imu_prior_builder.h"
#include "livo_recon/utils/algo/omp_utils.h"

#include <algorithm>

namespace livo_recon
{

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

// ============================================================================
// PRODUCTION: buildPoseControlContinuousImuPrior() -- see the header's own
// derivation comment for the full residual/Jacobian/discretization
// rationale. Per-thread accumulator mirrors PoseControlLidarThreadAccum
// (pose_control_lidar_factor.cpp) exactly.
// ============================================================================
namespace
{
struct ContinuousImuThreadAccum
{
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  double E_imu = 0.0;
  Eigen::Matrix<double, 9, 9> A_hh = Eigen::Matrix<double, 9, 9>::Zero();
  Eigen::MatrixXd A_hf;
};
}  // namespace

PoseControlContinuousImuPriorStats buildPoseControlContinuousImuPrior(
    const PoseControlSpline& spline, const PoseControlFreeLayout& layout,
    const std::vector<ImuSample>& imu_samples,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    const V3D& var_acc, const V3D& var_gyr,
    Eigen::MatrixXd& A, Eigen::VectorXd& b,
    PoseControlPriorHeadBlock* head_block,
    std::vector<ImuSplineResidualSample>* out_samples)
{
  PoseControlContinuousImuPriorStats stats;
  const int dimZ = layout.dim();
  const double t0 = spline.t0(), t1 = spline.t1();
  std::vector<ImuSample> windowed;
  windowed.reserve(imu_samples.size());
  for (const auto& s : imu_samples) if (s.t >= t0 && s.t <= t1) windowed.push_back(s);
  const int n = static_cast<int>(windowed.size());
  stats.n_samples = n;
  if (n == 0) return stats;

  // R-vs-Q semantics (item 10, see LioProcCoupledOptions::pose_imu_var_acc's
  // own header comment): var_acc/var_gyr here are used as a MEASUREMENT
  // variance R (Wdiag=1/R below), not a propagated Q -- they coincide with
  // the continuous process-noise density (Q_density = R*dt_imu) only
  // because this model treats each accelerometer/gyro sample as a direct
  // noisy observation of the process's own driving noise term.
  const Eigen::Matrix<double, 6, 1> Wdiag = (Eigen::Matrix<double, 6, 1>() <<
      1.0 / std::max(var_acc.x(), 1e-12), 1.0 / std::max(var_acc.y(), 1e-12), 1.0 / std::max(var_acc.z(), 1e-12),
      1.0 / std::max(var_gyr.x(), 1e-12), 1.0 / std::max(var_gyr.y(), 1e-12), 1.0 / std::max(var_gyr.z(), 1e-12)).finished();

  if (out_samples) out_samples->resize(n);

  const int threads = std::max(1, std::min(cappedOmpThreads(), std::max(1, n)));
  std::vector<ContinuousImuThreadAccum> acc(threads);
  for (auto& t : acc) {
    t.A = Eigen::MatrixXd::Zero(dimZ, dimZ);
    t.b = Eigen::VectorXd::Zero(dimZ);
    if (head_block) t.A_hf = Eigen::MatrixXd::Zero(9, dimZ);
  }

  const int off_bg = layout.colBG(), off_ba = layout.colBA(), off_g = layout.colG();
  const PoseControlHeadPosSensitivity hs = poseControlHeadPosSensitivity(spline);

  #pragma omp parallel num_threads(threads)
  {
    const int tid = omp_get_thread_num();
    ContinuousImuThreadAccum& acc_t = acc[tid];

    #pragma omp for schedule(static)
    for (int i = 0; i < n; ++i)
    {
      const ImuSample& s = windowed[i];
      const double t = s.t;
      const M3D R = spline.rotAt(t);
      const V3D a_world = spline.accAt(t);
      const V3D v_specific = a_world - gravity;              // world-frame (a - g)
      const V3D a_spline_body = R.transpose() * v_specific;  // R(t)^T*(a-g)
      const V3D e_acc = a_spline_body + bias_acc - s.acc;
      const V3D omega_body = spline.omegaBodyAt(t);
      const V3D e_gyr = omega_body + bias_gyr - s.gyro;

      if (out_samples) {
        ImuSplineResidualSample rec;
        rec.t = t; rec.a_meas = s.acc; rec.a_spline_body = a_spline_body; rec.e_acc = e_acc;
        rec.omega_meas = s.gyro; rec.omega_spline_body = omega_body; rec.e_gyr = e_gyr;
        (*out_samples)[i] = rec;
      }

      Eigen::Matrix<double, 6, 1> r; r << e_acc, e_gyr;
      Eigen::MatrixXd Jrow = Eigen::MatrixXd::Zero(6, dimZ);
      if (off_ba >= 0) Jrow.block<3, 3>(0, off_ba) = M3D::Identity();     // de_acc/dba
      if (off_g  >= 0) Jrow.block<3, 3>(0, off_g)  = -R.transpose();     // de_acc/dg
      if (off_bg >= 0) Jrow.block<3, 3>(3, off_bg) = M3D::Identity();    // de_gyr/dbg

      const M3D d_acc_d_theta = skew3v(a_spline_body);  // de_acc/d(local body-frame rotation perturbation at t)
      bool touches_head_pos = false;
      const auto jac = spline.jacobianAt(t);
      for (int k = 0; k < 4; ++k) {
        const int abs_k = jac.s + k;
        const int colp = layout.colPos(abs_k);
        const int colph = layout.colPhi(abs_k);
        if (colp >= 0) Jrow.block<3, 3>(0, colp) = R.transpose() * PoseControlSpline::dAccDcp(jac, k);
        if (colph >= 0) {
          Jrow.block<3, 3>(0, colph) = d_acc_d_theta * spline.dThetaDcphi(jac, k, t);
          Jrow.block<3, 3>(3, colph) = spline.dOmegaDcphi(jac, k, t);
        }
        if (abs_k < 3) touches_head_pos = true;
      }

      if (head_block) {
        Eigen::Matrix<double, 6, 9> Jhead = Eigen::Matrix<double, 6, 9>::Zero();
        // theta0 (GLOBAL, e_acc only -- e_gyr's head sensitivity is
        // EXACTLY zero, see the header's own derivation).
        Jhead.block<3, 3>(0, 0) = d_acc_d_theta * poseControlHeadRotJacobian(spline, t);
        if (touches_head_pos) {
          M3D dp_dp0, dp_dv0, dv_dp0, dv_dv0;
          poseControlHeadPosJacobians(spline, hs, t, dp_dp0, dp_dv0, dv_dp0, dv_dv0);
          // p0/v0 reach e_acc ONLY through cp_p[0..2]'s own p0/v0-
          // sensitivity (Minv), chained through dAccDcp for k in {0,1,2}
          // whose abs_k<3 -- poseControlHeadPosJacobians already folds the
          // Minv weighting for p(t)/v(t); the equivalent for a(t) is
          // derived the same way directly against dAccDcp below (a(t)'s
          // p0/v0 sensitivity shares the SAME Minv/locality structure as
          // p(t)/v(t), just one more derivative -- dAccDcp instead of
          // dPosDcp/dVelDcp).
          M3D da_dp0 = M3D::Zero(), da_dv0 = M3D::Zero();
          for (int k = 0; k < 4; ++k) {
            const int abs_k = jac.s + k;
            if (abs_k >= 3) continue;
            const double m0 = hs.Minv(abs_k, 0), m1 = hs.Minv(abs_k, 1);
            da_dp0 += m0 * PoseControlSpline::dAccDcp(jac, k);
            da_dv0 += m1 * PoseControlSpline::dAccDcp(jac, k);
          }
          Jhead.block<3, 3>(0, 3) = R.transpose() * da_dp0;
          Jhead.block<3, 3>(0, 6) = R.transpose() * da_dv0;
        }
        acc_t.A_hh += Jhead.transpose() * Wdiag.asDiagonal() * Jhead;
        acc_t.A_hf += Jhead.transpose() * Wdiag.asDiagonal() * Jrow;
      }

      acc_t.A += Jrow.transpose() * Wdiag.asDiagonal() * Jrow;
      acc_t.b += -Jrow.transpose() * Wdiag.asDiagonal() * r;
      acc_t.E_imu += 0.5 * (r.transpose() * Wdiag.asDiagonal() * r)(0);
    }
  }

  for (auto& t : acc) {
    A += t.A;
    b += t.b;
    stats.E_imu += t.E_imu;
    if (head_block) {
      head_block->A_hh += t.A_hh;
      if (head_block->A_hf.size() == 0) head_block->A_hf = Eigen::MatrixXd::Zero(9, dimZ);
      head_block->A_hf += t.A_hf;
    }
  }
  return stats;
}

}  // namespace livo_recon
