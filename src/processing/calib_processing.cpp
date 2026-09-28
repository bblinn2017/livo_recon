#include "livo_recon/processing/calib_processing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <Eigen/Eigenvalues>
#include "livo_recon/diagnostics/calibration_p0_writer.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/state/state.h"

namespace livo_recon
{

CalibProc::CalibProc(NodeContext& ctx)
  : data_queues_(ctx.data_queues), measures_(ctx.measures), state_(ctx.state),
    profiler_(ctx.profiler)
{}

std::string CalibProc::loadParameters(ros::NodeHandle& pnh)
{
  for (const char* legacy : {"calib/use_calib", "calib/use_calib_var",
                             "calib/use_calib_bias", "calib/num_samples",
                             "calib/stall_calls_max"})
  {
    if (pnh.hasParam(legacy))
      throw std::invalid_argument(std::string("removed ambiguous calibration key: ") +
          legacy + "; use calib/stationary, calib/p0, and imu/process_noise");
  }
  paramWarn<int>(pnh, "calib/stationary/num_samples", opts_.num_samples, 200);
  paramWarn<int>(pnh, "calib/stationary/stall_calls_max", opts_.stall_calls_max, 300);
  paramWarn<bool>(pnh, "calib/stationary/apply_to_state/gyro_bias",
                  opts_.apply_gyro_bias, true);
  paramWarn<bool>(pnh, "calib/stationary/apply_to_state/accel_bias",
                  opts_.apply_accel_bias, true);
  paramWarn<std::string>(pnh, "calib/p0/mode", opts_.p0_mode, "configured");
  paramWarn<int>(pnh, "calib/p0/autocov_lags", opts_.p0_autocov_lags, 20);
  paramWarn<double>(pnh, "calib/p0/known_pos_variance",
                    opts_.p0_known_pos_variance, 1e-12);
  paramWarn<double>(pnh, "calib/p0/known_vel_variance",
                    opts_.p0_known_vel_variance, 1e-12);
  paramWarn<double>(pnh, "calib/p0/tilt_ba_ambiguity_accel_std",
                    opts_.p0_tilt_ba_ambiguity_accel_std, -1.0);
  paramWarn<double>(pnh, "calib/p0/bg_model_floor",
                    opts_.p0_bg_model_floor, -1.0);
  paramWarn<double>(pnh, "calib/p0/ba_radial_model_floor",
                    opts_.p0_ba_radial_model_floor, -1.0);
  paramWarn<double>(pnh, "calib/p0/numerical_eigenvalue_floor",
                    opts_.p0_numerical_eigenvalue_floor, 1e-12);

  if (opts_.num_samples < 1)
    opts_.num_samples = 1;
  opts_.p0_autocov_lags = std::max(0, opts_.p0_autocov_lags);
  if (opts_.p0_mode != "configured" && opts_.p0_mode != "calibration_derived")
    throw std::invalid_argument("calib/p0/mode must be configured or calibration_derived");
  if (!std::isfinite(opts_.p0_known_pos_variance) ||
      !std::isfinite(opts_.p0_known_vel_variance) ||
      opts_.p0_known_pos_variance <= 0.0 || opts_.p0_known_vel_variance <= 0.0)
    throw std::invalid_argument("calib/p0 known-state variances must be finite and positive");
  if (!std::isfinite(opts_.p0_tilt_ba_ambiguity_accel_std) ||
      !std::isfinite(opts_.p0_bg_model_floor) ||
      !std::isfinite(opts_.p0_ba_radial_model_floor) ||
      !std::isfinite(opts_.p0_numerical_eigenvalue_floor) ||
      opts_.p0_numerical_eigenvalue_floor <= 0.0)
    throw std::invalid_argument("calib/p0 ambiguity and floor parameters must be finite");
  std::ostringstream oss;
  oss << "[params/calib/stationary]"
      << "\n  required: true"
      << "\n  num_samples: " << opts_.num_samples
      << "\n  stall_calls_max: " << opts_.stall_calls_max
      << "\n  apply_to_state/gyro_bias: "
      << (opts_.apply_gyro_bias ? "true" : "false")
      << "\n  apply_to_state/accel_bias: "
      << (opts_.apply_accel_bias ? "true" : "false");
  oss << "\n  p0/mode:       " << opts_.p0_mode
      << "\n  p0/autocov_lags: " << opts_.p0_autocov_lags
      << "\n  p0/known_pos_variance: " << opts_.p0_known_pos_variance
      << "\n  p0/known_vel_variance: " << opts_.p0_known_vel_variance
      << "\n  p0/tilt_ba_ambiguity_accel_std: "
      << opts_.p0_tilt_ba_ambiguity_accel_std
      << "\n  p0/model_floors(bg,ba_radial): " << opts_.p0_bg_model_floor << ", "
      << opts_.p0_ba_radial_model_floor;
  oss << "\n  p0/numerical_eigenvalue_floor: "
      << opts_.p0_numerical_eigenvalue_floor;
  return oss.str();
}

void CalibProc::stabilizeP0()
{
  Eigen::MatrixXd& P = state_->covMut();
  P = 0.5 * (P + P.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(P);
  if (es.info() != Eigen::Success)
    throw std::runtime_error("P0 stabilization eigensolve failed");
  const double min_before = es.eigenvalues().minCoeff();
  if (min_before < opts_.p0_numerical_eigenvalue_floor)
  {
    const Eigen::VectorXd values =
        es.eigenvalues().cwiseMax(opts_.p0_numerical_eigenvalue_floor);
    P = es.eigenvectors() * values.asDiagonal() * es.eigenvectors().transpose();
    P = 0.5 * (P + P.transpose());
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> check(P);
  if (check.info() != Eigen::Success)
    throw std::runtime_error("stabilized P0 verification eigensolve failed");
  double min_after = check.eigenvalues().minCoeff();
  if (min_after < opts_.p0_numerical_eigenvalue_floor)
  {
    const double lift = opts_.p0_numerical_eigenvalue_floor - min_after +
        std::numeric_limits<double>::epsilon() *
        std::max(1.0, opts_.p0_numerical_eigenvalue_floor);
    P.diagonal().array() += lift;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> lifted(P);
    if (lifted.info() != Eigen::Success)
      throw std::runtime_error("lifted P0 verification eigensolve failed");
    min_after = lifted.eigenvalues().minCoeff();
  }
  state_->setP0Stabilization(min_before,
      opts_.p0_numerical_eigenvalue_floor, min_after);
  ROS_INFO_STREAM("[calib/P0] SPD stabilization: min_before=" << min_before
      << " floor=" << opts_.p0_numerical_eigenvalue_floor
      << " min_after=" << min_after);
}

bool CalibProc::collectSamples()
{
  std::deque<ImuSample> imu_samples;
  std::vector<PointXYZT> points;

  while (calib_imu_samples.size() < static_cast<size_t>(opts_.num_samples) &&
         data_queues_->ready())
  {
    ++calib_groups_seen_;
    ++calib_images_consumed_;
    calib_last_img_ = data_queues_->popImage();

    points.clear();
    data_queues_->popLidar(points, calib_last_img_.t);
    imu_samples.clear();
    data_queues_->popImu(imu_samples, calib_last_img_.t);

    // See DataQueues::dry_run_lidar_queue's doc comment. Must be drained in
    // lockstep with the primary lidar_queue above, even though calibration
    // has no use for the dry-run points themselves (discarded immediately)
    // -- otherwise it silently accumulates every dry-run scan that arrives
    // during the whole calibration window (nothing else pops it), and the
    // first real frame after calibration then dumps that entire backlog
    // into one MeasureGroup via popDryRunLidar(), spanning far more time
    // than that frame's own IMU-propagated pose coverage. Confirmed this
    // session: deskewPoints() FATAL-aborted ("24868 point(s) unmatched by
    // poses coverage") on exactly this backlog on the very first frame.
    std::vector<PointXYZT> dry_run_points;
    data_queues_->popDryRunLidar(dry_run_points, calib_last_img_.t);

    calib_points.insert(
        calib_points.end(),
        std::make_move_iterator(points.begin()),
        std::make_move_iterator(points.end()));

    if (imu_samples.size() < 2)
    {
      // Counted, not silent -- this discard is the most likely reason the
      // sample count never grows, and it was invisible.
      ++calib_groups_short_imu_;
      continue;
    }
    calib_imu_samples.insert(
        calib_imu_samples.end(),
        imu_samples.begin(),
        imu_samples.end() - 1);
  }

  return calib_imu_samples.size() >= static_cast<size_t>(opts_.num_samples);
}

void CalibProc::computeBiasAndNoise(V3D& acc_bias, V3D& gyro_bias,
                                    V3D& var_acc, V3D& var_gyr) const
{
  const int N = static_cast<int>(calib_imu_samples.size());

  acc_bias  = V3D::Zero();
  gyro_bias = V3D::Zero();
  for (int i = 0; i < N; ++i)
  {
    acc_bias  += calib_imu_samples[i].acc;
    gyro_bias += calib_imu_samples[i].gyro;
  }
  acc_bias  /= static_cast<double>(N);
  gyro_bias /= static_cast<double>(N);

  M3D acc_cov  = M3D::Zero();
  M3D gyro_cov = M3D::Zero();
  for (int i = 0; i < N; ++i)
  {
    const V3D da = calib_imu_samples[i].acc  - acc_bias;
    const V3D dg = calib_imu_samples[i].gyro - gyro_bias;
    acc_cov  += da * da.transpose();
    gyro_cov += dg * dg.transpose();
  }
  acc_cov  /= static_cast<double>(N - 1);
  gyro_cov /= static_cast<double>(N - 1);

  // CQ-59 item 4: measured on eee_01/eee_02 (10s LIO-only smoke runs,
  // calib_processing's own diag print below) -- acc max/min ratio 9.8-17.4,
  // gyr max/min ratio 7.98-39.4. Neither is anywhere near 1, so per the
  // card's own criterion this IS worth a flag: var_acc/var_gyr now carry
  // the REAL per-axis diagonal instead of an isotropic V3D::Constant(mean)
  // broadcast. This is behavior-PRESERVING for every existing consumer --
  // grepped every varAccFloor()/varGyrFloor() call site (lio_coupled.cpp,
  // lio_decoupled.cpp, adaptive_q.h's own caller): all of them immediately
  // call .mean() on the result, and mean([d0,d1,d2]) == trace/3 exactly,
  // so nothing downstream changes unless it reads the raw V3D directly --
  // which only the new prior_per_axis_sigma path (estimateCoupledCorrection())
  // now does.
  ROS_INFO_STREAM("[calib] per-axis floor (diag, not mean-reduced):"
      << "  acc=[" << acc_cov(0,0) << ", " << acc_cov(1,1) << ", " << acc_cov(2,2) << "]"
      << "  gyr=[" << gyro_cov(0,0) << ", " << gyro_cov(1,1) << ", " << gyro_cov(2,2) << "]"
      << "  acc_ratio(max/min)=" << (std::max({acc_cov(0,0), acc_cov(1,1), acc_cov(2,2)})
                                    / std::max(std::min({acc_cov(0,0), acc_cov(1,1), acc_cov(2,2)}), 1e-18))
      << "  gyr_ratio(max/min)=" << (std::max({gyro_cov(0,0), gyro_cov(1,1), gyro_cov(2,2)})
                                    / std::max(std::min({gyro_cov(0,0), gyro_cov(1,1), gyro_cov(2,2)}), 1e-18)));

  var_acc = V3D(acc_cov(0,0), acc_cov(1,1), acc_cov(2,2));
  var_gyr = V3D(gyro_cov(0,0), gyro_cov(1,1), gyro_cov(2,2));
}

M3D CalibProc::covarianceOfMean(bool accelerometer) const
{
  const int n = static_cast<int>(calib_imu_samples.size());
  if (n < 2) return M3D::Zero();

  V3D mean = V3D::Zero();
  for (const auto& s : calib_imu_samples)
    mean += accelerometer ? s.acc : s.gyro;
  mean /= static_cast<double>(n);

  // Bartlett-window HAC estimate of Cov(mean). Unlike sample_cov/N this
  // retains the temporal correlation present in a stationary IMU stream.
  // The Bartlett taper keeps the finite-sample estimate PSD in exact
  // arithmetic; the eigensolver projection removes round-off negatives.
  const int lmax = std::min(opts_.p0_autocov_lags, n - 1);
  M3D spectral = M3D::Zero();
  for (int lag = 0; lag <= lmax; ++lag)
  {
    M3D gamma = M3D::Zero();
    for (int i = lag; i < n; ++i)
    {
      const V3D xi = (accelerometer ? calib_imu_samples[i].acc
                                    : calib_imu_samples[i].gyro) - mean;
      const V3D xj = (accelerometer ? calib_imu_samples[i-lag].acc
                                    : calib_imu_samples[i-lag].gyro) - mean;
      gamma += xi * xj.transpose();
    }
    gamma /= static_cast<double>(n);
    if (lag == 0)
      spectral += gamma;
    else
    {
      const double w = 1.0 - static_cast<double>(lag) /
                                 static_cast<double>(lmax + 1);
      spectral += w * (gamma + gamma.transpose());
    }
  }
  M3D cov_mean = 0.5 * (spectral + spectral.transpose()) /
                 static_cast<double>(n);
  Eigen::SelfAdjointEigenSolver<M3D> es(cov_mean);
  if (es.info() != Eigen::Success) throw std::runtime_error("P0 mean covariance eigensolve failed");
  return es.eigenvectors() * es.eigenvalues().cwiseMax(0.0).asDiagonal() *
         es.eigenvectors().transpose();
}

void CalibProc::applyCalibrationDerivedP0(
    const V3D& acc_mean, const M3D& R_init, const M3D& acc_mean_cov,
    const M3D& gyro_mean_cov)
{
  auto rotationAndBias = [&](const V3D& mean, M3D& R, V3D& ba) {
    R = computeInitialRotation(mean);
    ba = mean + R.transpose() * state_->gravity();
  };

  // Numerical Jacobian is intentional: it differentiates the exact
  // production gravity-alignment convention, including the right/body
  // attitude perturbation consumed by StateGroup::applyDelta().
  Eigen::Matrix<double, 6, 3> J = Eigen::Matrix<double, 6, 3>::Zero();
  for (int axis = 0; axis < 3; ++axis)
  {
    const double eps = std::max(1e-7, std::abs(acc_mean(axis)) * 1e-7);
    V3D plus = acc_mean, minus = acc_mean;
    plus(axis) += eps;
    minus(axis) -= eps;
    M3D Rp, Rm;
    V3D bap, bam;
    rotationAndBias(plus, Rp, bap);
    rotationAndBias(minus, Rm, bam);
    J.block<3,1>(0, axis) =
        (Log(R_init.transpose() * Rp) - Log(R_init.transpose() * Rm)) /
        (2.0 * eps);
    J.block<3,1>(3, axis) = (bap - bam) / (2.0 * eps);
  }

  Eigen::Matrix<double, 6, 6> P_acc = J * acc_mean_cov * J.transpose();
  P_acc = 0.5 * (P_acc + P_acc.transpose());
  const double bg_floor = opts_.p0_bg_model_floor >= 0.0
      ? opts_.p0_bg_model_floor : state_->initCovBg();
  const double ba_radial_floor = opts_.p0_ba_radial_model_floor >= 0.0
      ? opts_.p0_ba_radial_model_floor : state_->initCovBa();

  // Stationary specific force constrains only the SUM
  //     delta_a = A * delta_theta + delta_ba,  A = [a_gravity]_x.
  // Model its transverse null-space ambiguity with one latent acceleration
  // vector e: A*dtheta=-e, dba=+e. Thus both alternatives
  // have identical acceleration-equivalent variance and their negative
  // cross-covariance preserves the fact that one compensates the other.
  const double gravity_mag = state_->gravity().norm();
  if (!std::isfinite(gravity_mag) || gravity_mag < 1e-6)
    throw std::runtime_error("calibration-derived P0 requires nonzero finite gravity");
  const V3D gravity_axis_body =
      (R_init.transpose() * (-state_->gravity())).normalized();
  const M3D tangent = M3D::Identity() -
      gravity_axis_body * gravity_axis_body.transpose();
  const double inherited_accel_var = std::max(
      gravity_mag * gravity_mag * state_->initCovRotTilt(), state_->initCovBa());
  const double ambiguity_std = opts_.p0_tilt_ba_ambiguity_accel_std >= 0.0
      ? opts_.p0_tilt_ba_ambiguity_accel_std : std::sqrt(inherited_accel_var);
  const M3D ambiguity_accel_cov =
      ambiguity_std * ambiguity_std * tangent;
  M3D A;
  A << SKEW_SYM_MATRX(gravity_mag * gravity_axis_body);
  // On the tangent plane pinv([g*u]_x) = -[g*u]_x/g^2.
  const M3D A_pinv = -A / (gravity_mag * gravity_mag);
  const M3D P_theta_amb =
      A_pinv * ambiguity_accel_cov * A_pinv.transpose();
  const M3D P_ba_amb = ambiguity_accel_cov;
  const M3D P_theta_ba_amb =
      -A_pinv * ambiguity_accel_cov;
  P_acc.block<3,3>(0,0) += P_theta_amb;
  P_acc.block<3,3>(3,3) += P_ba_amb + ba_radial_floor *
      gravity_axis_body * gravity_axis_body.transpose();
  P_acc.block<3,3>(0,3) += P_theta_ba_amb;
  P_acc.block<3,3>(3,0) += P_theta_ba_amb.transpose();

  Eigen::MatrixXd& P = state_->covMut();
  P.block<3,3>(StateGroup::idxP(), StateGroup::idxP()) =
      opts_.p0_known_pos_variance * M3D::Identity();
  P.block<3,3>(StateGroup::idxV(), StateGroup::idxV()) =
      opts_.p0_known_vel_variance * M3D::Identity();

  // Accelerometer mean observes tilt, not yaw. Add the configured yaw
  // variance along the right-perturbation gravity axis.
  const V3D yaw_axis_body =
      (R_init.transpose() * V3D(0.0, 0.0, -1.0)).normalized();
  P.block<3,3>(StateGroup::idxR(), StateGroup::idxR()) =
      P_acc.block<3,3>(0,0) + state_->initCovRotYaw() *
          yaw_axis_body * yaw_axis_body.transpose();

  if (state_->estBA())
  {
    P.block(state_->idxBA(), state_->idxBA(), 3, 3) = P_acc.block<3,3>(3,3);
    P.block(StateGroup::idxR(), state_->idxBA(), 3, 3) = P_acc.block<3,3>(0,3);
    P.block(state_->idxBA(), StateGroup::idxR(), 3, 3) = P_acc.block<3,3>(3,0);
  }
  if (state_->estBG())
    P.block(state_->idxBG(), state_->idxBG(), 3, 3) =
        gyro_mean_cov + bg_floor * M3D::Identity();

  // Gravity magnitude/direction is deliberately left at state/cov/gravity:
  // one stationary accelerometer mean cannot separate gravity from b_a.
  P = 0.5 * (P + P.transpose());

  ROS_INFO_STREAM("[calib/P0] balanced tilt-b_a ambiguity: accel_std="
      << ambiguity_std << " m/s^2, equivalent_each_accel_std="
      << ambiguity_std << " m/s^2, equivalent_each_tilt_std="
      << ambiguity_std / gravity_mag << " rad");
}

M3D CalibProc::computeInitialRotation(const V3D& acc_bias) const
{
  const V3D acc_dir  = acc_bias / acc_bias.norm();
  const V3D world_up = V3D(0, 0, 1);
  const V3D rot_axis = acc_dir.cross(world_up);
  const double sine  = rot_axis.norm();
  const double cosn  = acc_dir.dot(world_up);
  if (sine < 1e-8)
    return (cosn > 0.0) ? M3D::Identity() : M3D(Eigen::AngleAxisd(M_PI, V3D::UnitX()));
  return M3D(Eigen::AngleAxisd(std::atan2(sine, cosn), rot_axis / sine));
}

std::string CalibProc::estimateFromBuffer()
{
  TimedScope ts(profiler_, "calib");

  std::ostringstream oss;
  oss << "Calibrating IMU... " << calib_imu_samples.size() << " samples collected.";
  if (!collectSamples())
  {
    // No progress since the last call means the buffer is not filling, and
    // repeating the same line forever is what made this look like a hang.
    // Bound it and fail with the breakdown that says WHY -- see
    // CalibProcOptions::stall_calls_max.
    if (calib_imu_samples.size() == calib_last_count_)
      ++calib_stall_calls_;
    else
      { calib_stall_calls_ = 0; calib_last_count_ = calib_imu_samples.size(); }

    if (opts_.stall_calls_max > 0 && calib_stall_calls_ >= opts_.stall_calls_max)
    {
      std::ostringstream err;
      err << "[calib] STALLED: " << calib_imu_samples.size() << "/"
          << opts_.num_samples << " IMU samples after " << calib_stall_calls_
          << " consecutive calls with zero growth. images_consumed="
          << calib_images_consumed_
          << "  groups_seen=" << calib_groups_seen_
          << "  groups_dropped_for_fewer_than_2_imu=" << calib_groups_short_imu_
          << ". If groups_dropped is close to groups_seen the IMU stream is "
             "arriving slower than one sample per image window and no amount "
             "of waiting will fix it; if groups_seen stopped growing the "
             "queues went dry. Either way this is a failure, not a wait.";
      ROS_FATAL_STREAM(err.str());
      throw std::runtime_error(err.str());
    }

    oss << "  [stalled " << calib_stall_calls_ << "/" << opts_.stall_calls_max
        << ", dropped " << calib_groups_short_imu_ << " of "
        << calib_groups_seen_ << " groups for <2 IMU samples]";
    return oss.str();
  }

  V3D acc_bias, gyro_bias, var_acc, var_gyr;
  computeBiasAndNoise(acc_bias, gyro_bias, var_acc, var_gyr);

  const M3D R_init = computeInitialRotation(acc_bias);
  const M3D acc_mean_cov = covarianceOfMean(true);
  const M3D gyro_mean_cov = covarianceOfMean(false);
  // acc_bias = true_bias + R_init^T * [0,0,9.81]; strip gravity to get true sensor bias
  const V3D true_acc_bias = acc_bias + R_init.transpose() * state_->gravity();
  state_->setCalibResult(R_init,
                         opts_.apply_gyro_bias ? gyro_bias : state_->biasGyr(),
                         opts_.apply_accel_bias ? true_acc_bias : state_->biasAcc());
  // Stationary variance is a measured sensor-noise floor, not a complete
  // dynamic process-noise model. Keep it separate from imu/process_noise.
  state_->setNoiseFloor(var_acc, var_gyr);
  if (opts_.p0_mode == "calibration_derived")
    applyCalibrationDerivedP0(acc_bias, R_init, acc_mean_cov, gyro_mean_cov);
  stabilizeP0();
  state_->captureInitialCovariance(opts_.p0_mode);
  writeCalibrationP0Diagnostic(
      *state_, opts_.p0_mode, static_cast<int>(calib_imu_samples.size()),
      opts_.p0_autocov_lags, acc_mean_cov, gyro_mean_cov);

  // TQ-34, Bryce 2026-09-18: groups_seen/groups_dropped/stall_calls were
  // only ever printed on the STALL path (above) -- a normal, successful
  // calibration never surfaced them, even though the counters are right
  // here. R_init's own roll/pitch weren't surfaced anywhere either
  // (computeInitialRotation()'s result was consumed by setCalibResult()
  // and then only visible indirectly through biasAcc()). Both added to
  // this existing success log line rather than a new file -- this run's
  // TQ-34 diagnostic reads them straight out of run.log.
  const V3D r_init_rpy = R_init.eulerAngles(0, 1, 2) * (180.0 / M_PI);
  oss << "\nIMU calibration done with " << calib_imu_samples.size() << " samples."
      << "\n  acc bias:   " << state_->biasAcc().transpose()
      << "\n  gyro bias:  " << state_->biasGyr().transpose()
      << "\n  acc noise:  " << state_->varAcc().transpose()
      << "\n  gyro noise: " << state_->varGyr().transpose()
      << "\n  acc floor:  " << state_->varAccFloor().transpose()
      << "\n  gyro floor: " << state_->varGyrFloor().transpose()
      << "\n  gravity:    " << state_->gravity().transpose()
      << "\n  P0 mode:    " << opts_.p0_mode
      << "\n  acc mean covariance:\n" << acc_mean_cov
      << "\n  gyro mean covariance:\n" << gyro_mean_cov
      << "\n  initialized P0:\n" << state_->cov()
      << "\n  groups_seen=" << calib_groups_seen_
      << " groups_dropped_for_fewer_than_2_imu=" << calib_groups_short_imu_
      << " stall_calls=" << calib_stall_calls_
      << "\n  R_init_roll_deg=" << r_init_rpy(0) << " R_init_pitch_deg=" << r_init_rpy(1);
  const std::string done_msg = oss.str();

  data_queues_->setStartTime(calib_last_img_.t);
  calib_last_img_.t = 0.;
  for (auto& p : calib_points) p.t = 0.;
  calib_imu_samples.clear();

  measures_->curr_time.set(0.);
  measures_->pushMeasureGroup(
      MeasureGroup{std::move(calib_last_img_), std::move(calib_points), std::move(calib_imu_samples)});
  measures_->calib_done.set(true);
  return done_msg;
}

}  // namespace livo_recon
