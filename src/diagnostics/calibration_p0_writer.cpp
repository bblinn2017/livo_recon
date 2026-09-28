#include "livo_recon/diagnostics/calibration_p0_writer.h"

#include "livo_recon/diagnostics/log/debug_log_dir.h"

#include <Eigen/Eigenvalues>
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
  if (state.estGravity()) out << " g[" << state.idxG() << ':' << state.idxG()+3 << ']';
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
    out << "P0_g_g\n" << P.block(state.idxG(), state.idxG(), 3, 3) << '\n';
  out << "P0_full\n" << P << '\n';
  if (eig.info() == Eigen::Success)
    out << "P0_eigenvalues " << eig.eigenvalues().transpose() << '\n';
  else
    out << "P0_eigenvalues ERROR eigensolve_failed\n";
  out << "end_calibration_p0_v1\n";
  out.flush();
}

}  // namespace livo_recon
