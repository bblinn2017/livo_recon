#include "livo_recon/diagnostics/calibration_p0_writer.h"

#include "livo_recon/diagnostics/log/debug_log_dir.h"
#include "livo_recon/utils/algo/math.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <iomanip>

namespace livo_recon
{

void writeCalibrationP0Diagnostic(
    const StateGroup& state, const std::string& mode, int sample_count,
    int autocov_lags, const M3D& acc_mean_cov, const M3D& gyro_mean_cov)
{
  static PersistentLogStream log("calibration_p0.txt");
  std::ofstream& out = log.stream();
  out << std::setprecision(17);

  const Eigen::MatrixXd P = 0.5 * (state.cov() + state.cov().transpose());
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(P);

  out << "schema calibration_p0_v1\n"
      << "phase post_stationary_calibration_pre_first_imu\n"
      << "mode " << mode << '\n'
      << "sample_count " << sample_count << '\n'
      << "autocov_lags " << autocov_lags << '\n'
      << "state_dim " << state.dimState() << '\n'
      << "numerical_stabilization min_before " << state.p0MinEigenvalueBefore()
      << " floor " << state.p0EigenvalueFloor()
      << " min_after " << state.p0MinEigenvalueAfter() << '\n'
      << "ordering theta[0:3] p[3:6] v[6:9]";
  if (state.estBG()) out << " bg[" << state.idxBG() << ':' << state.idxBG()+3 << ']';
  if (state.estBA()) out << " ba[" << state.idxBA() << ':' << state.idxBA()+3 << ']';
  if (state.estGravity()) out << " g[" << state.idxG() << ':' << state.idxG()+state.gravDim() << ']'
                              << " gravity_model " << state.gravityModelName();
  out << "\nconfigured_init_variances tilt " << state.initCovRotTilt()
      << " yaw " << state.initCovRotYaw()
      << " p " << state.initCovPos()
      << " v " << state.initCovVel()
      << " bg " << state.initCovBg()
      << " ba " << state.initCovBa()
      << " g " << state.initCovGravity() << '\n'
      << "state_rotation\n" << state.rot() << '\n'
      << "state_position " << state.pos().transpose() << '\n'
      << "state_velocity " << state.vel().transpose() << '\n'
      << "state_bias_gyro " << state.biasGyr().transpose() << '\n'
      << "state_bias_accel " << state.biasAcc().transpose() << '\n'
      << "state_gravity " << state.gravity().transpose() << '\n'
      << "acc_mean_covariance\n" << acc_mean_cov << '\n'
      << "gyro_mean_covariance\n" << gyro_mean_cov << '\n'
      << "P0_theta_theta\n" << P.block<3,3>(state.idxR(), state.idxR()) << '\n'
      << "P0_p_p\n" << P.block<3,3>(state.idxP(), state.idxP()) << '\n'
      << "P0_v_v\n" << P.block<3,3>(state.idxV(), state.idxV()) << '\n';
  if (state.estBG())
    out << "P0_bg_bg\n" << P.block(state.idxBG(), state.idxBG(), 3, 3) << '\n';
  if (state.estBA())
  {
    out << "P0_ba_ba\n" << P.block(state.idxBA(), state.idxBA(), 3, 3) << '\n'
        << "P0_theta_ba\n" << P.block(state.idxR(), state.idxBA(), 3, 3) << '\n';
  }
  if (state.estGravity())
    out << "P0_g_g\n" << P.block(state.idxG(), state.idxG(), state.gravDim(), state.gravDim()) << '\n';
  out << "P0_full\n" << P << '\n';
  if (eig.info() == Eigen::Success)
    out << "P0_eigenvalues " << eig.eigenvalues().transpose() << '\n';
  else
    out << "P0_eigenvalues ERROR eigensolve_failed\n";
  out << "end_calibration_p0_v1\n";
  out.flush();
}

namespace
{
void meanAndSampleCov(const std::deque<ImuSample>& s, bool accel, size_t lo, size_t hi,
                      V3D& mean, M3D& cov)
{
  mean = V3D::Zero();
  cov = M3D::Zero();
  const size_t n = hi > lo ? hi - lo : 0;
  if (n == 0) return;
  for (size_t i = lo; i < hi; ++i) mean += accel ? s[i].acc : s[i].gyro;
  mean /= static_cast<double>(n);
  if (n < 2) return;
  for (size_t i = lo; i < hi; ++i)
  {
    const V3D d = (accel ? s[i].acc : s[i].gyro) - mean;
    cov += d * d.transpose();
  }
  cov /= static_cast<double>(n - 1);
}
}  // namespace

void writeGravityAlignmentDiagnostic(
    const std::deque<ImuSample>& samples, const M3D& acc_mean_cov,
    const M3D& gyro_mean_cov, const M3D& R_before, const M3D& R_init,
    const StateGroup& state_after, const V3D& true_acc_bias,
    bool apply_accel_bias, bool apply_gyro_bias)
{
  static PersistentLogStream log("gravity_alignment.txt");
  std::ofstream& out = log.stream();
  out << std::setprecision(17);
  const size_t n = samples.size();
  V3D a_all, g_all, a_h1, a_h2, g_h1, g_h2;
  M3D a_cov, g_cov, tmp;
  meanAndSampleCov(samples, true, 0, n, a_all, a_cov);
  meanAndSampleCov(samples, false, 0, n, g_all, g_cov);
  meanAndSampleCov(samples, true, 0, n / 2, a_h1, tmp);
  meanAndSampleCov(samples, true, n / 2, n, a_h2, tmp);
  meanAndSampleCov(samples, false, 0, n / 2, g_h1, tmp);
  meanAndSampleCov(samples, false, n / 2, n, g_h2, tmp);

  const double a_norm = a_all.norm();
  const double g_mag = state_after.gravity().norm();
  // Tilt sd implied by the accelerometer-mean covariance alone: the two
  // eigenvalues of the mean covariance restricted to the plane orthogonal to
  // the mean acceleration, divided by |mean accel|.
  V3D tilt_sd = V3D::Zero();
  if (a_norm > 1e-9)
  {
    const V3D u = a_all / a_norm;
    const M3D T = M3D::Identity() - u * u.transpose();
    Eigen::SelfAdjointEigenSolver<M3D> es(T * acc_mean_cov * T);
    if (es.info() == Eigen::Success)
      for (int i = 0; i < 3; ++i)
        tilt_sd(i) = std::sqrt(std::max(0.0, es.eigenvalues()(i))) / a_norm;
  }
  out << "schema gravity_alignment_v1\n"
      << "sample_count " << n << '\n'
      << "t_first_rel " << (n ? samples.front().t : 0.0)
      << " t_last_rel " << (n ? samples.back().t : 0.0) << '\n'
      << "apply_to_state accel_bias " << (apply_accel_bias ? 1 : 0)
      << " gyro_bias " << (apply_gyro_bias ? 1 : 0) << '\n'
      << "acc_mean_body " << a_all.transpose() << '\n'
      << "acc_mean_norm " << a_norm << '\n'
      << "state_gravity_norm " << g_mag << '\n'
      << "acc_norm_excess_over_gravity " << (a_norm - g_mag) << '\n'
      << "acc_mean_first_half " << a_h1.transpose() << '\n'
      << "acc_mean_second_half " << a_h2.transpose() << '\n'
      << "acc_sample_covariance\n" << a_cov << '\n'
      << "acc_mean_covariance_hac\n" << acc_mean_cov << '\n'
      << "tilt_sd_from_acc_mean_cov_rad (2 nonzero eigen-directions, sorted ascending) "
      << tilt_sd.transpose() << '\n'
      << "gyro_mean_body " << g_all.transpose() << '\n'
      << "gyro_mean_first_half " << g_h1.transpose() << '\n'
      << "gyro_mean_second_half " << g_h2.transpose() << '\n'
      << "gyro_sample_covariance\n" << g_cov << '\n'
      << "gyro_mean_covariance_hac\n" << gyro_mean_cov << '\n'
      << "rotation_before_alignment\n" << R_before << '\n'
      << "rotation_after_alignment\n" << R_init << '\n'
      << "tilt_from_alignment_rad " << Log(R_before.transpose() * R_init).norm() << '\n'
      << "true_acc_bias_stripped_of_gravity " << true_acc_bias.transpose() << '\n'
      << "state_bias_accel_after " << state_after.biasAcc().transpose() << '\n'
      << "state_bias_gyro_after " << state_after.biasGyr().transpose() << '\n'
      << "state_gravity_after " << state_after.gravity().transpose() << '\n'
      << "end_gravity_alignment_v1\n";
  out.flush();
}

}  // namespace livo_recon
