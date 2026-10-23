#include "livo_recon/processing/calib_processing.h"
#include "livo_recon/utils/algo/symmetrize.h"
#include "livo_recon/lio/p0_pin.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <Eigen/Eigenvalues>
#include "livo_recon/diagnostics/calibration_p0_writer.h"
#include "livo_recon/diagnostics/bootstrap_diagnostic_writer.h"
#include "livo_recon/lio/deskew.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/algo/hashing.h"
#include "livo_recon/utils/state/state.h"

namespace livo_recon
{

Eigen::MatrixXd gravityCovarianceFromCalibration(
    const V3D& u, const M3D& R_init, const M3D& acc_mean_cov, double s_r2)
{
  const M3D T = M3D::Identity() - u * u.transpose();
  const M3D C_tan = symmetrized(M3D(T * (R_init * acc_mean_cov * R_init.transpose()) * T));
  return C_tan + s_r2 * u * u.transpose();
}

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
  // 2026-10-20 (Bryce's call): calib/stationary/num_samples (the sample-count stop condition) is
  // removed -- num_frames is now the only stop condition and must be set and > 0.
  if (pnh.hasParam("calib/stationary/num_samples"))
    throw std::invalid_argument(
        "calib/stationary/num_samples was removed in R106+ (Bryce, 2026-10-20): "
        "use calib/stationary/num_frames instead (must be > 0)");
  paramWarn<int>(pnh, "calib/stationary/num_frames", opts_.num_frames, 0);
  vib_ = loadVibrationConfig(pnh);
  if (opts_.num_frames <= 0)
    throw std::invalid_argument("calib/stationary/num_frames must be set and > 0");
  paramWarn<std::string>(pnh, "imu/ds/mode", opts_.ds_mode, "first");
  if (opts_.ds_mode != "off" && opts_.ds_mode != "first" && opts_.ds_mode != "average")
    throw std::runtime_error("imu/ds/mode must be off, first, or average");
  paramWarn<double>(pnh, "imu/ds/ds_leaf_size", opts_.ds_leaf_size, 0.15);
  // Bootstrap points carry the sensor model covariance (same formula as the live deskew path: imu/sensor/range_err, imu/sensor/angle_err_deg).
  if (pnh.hasParam("calib/bootstrap/point_cov"))
    throw std::invalid_argument("removed key: calib/bootstrap/point_cov (bootstrap points always use the sensor model covariance)");
  {
    double range_err = 0.05;
    paramWarn<double>(pnh, "imu/sensor/range_err", range_err, 0.05);
    opts_.sensor_sigma_r2 = range_err * range_err;
    double angle_err_deg = 0.2;
    paramWarn<double>(pnh, "imu/sensor/angle_err_deg", angle_err_deg, 0.2);
    const double sin_angle_err = std::sin(std::max(1e-6, angle_err_deg * M_PI / 180.0));
    opts_.sensor_sigma_a2 = sin_angle_err * sin_angle_err;
  }
  if (opts_.ds_mode != "off" && !(opts_.ds_leaf_size > 0.0))
    throw std::runtime_error("imu/ds/ds_leaf_size must be > 0 when downsampling is enabled");
  paramWarn<std::string>(pnh, "calib/stationary/accel_excess_model",
                         opts_.accel_excess_model, std::string("bias"));
  if (opts_.accel_excess_model != "bias" && opts_.accel_excess_model != "scale")
    throw std::invalid_argument(
        "calib/stationary/accel_excess_model must be bias or scale");
  for (const char* removed_key : {"calib/p0/tilt_ba_ambiguity_accel_std", "calib/p0/ba_radial_model_floor"})
    if (pnh.hasParam(removed_key))
      throw std::invalid_argument(
          std::string("removed key: ") + removed_key + " -- both were ablated away 2026-10-23 "
          "(zeroing each changed frame-1 position drift by <0.5%, not adding real information); "
          "P_ba is now exactly the calibration's own measured J*acc_mean_cov*J^T, nothing added");
  paramWarn<double>(pnh, "imu/process_noise/floor/acc_variance",
                    opts_.floor_acc_variance, 1.288e-03);
  paramWarn<double>(pnh, "imu/process_noise/floor/gyro_variance",
                    opts_.floor_gyro_variance, 2.865e-06);
  if (!std::isfinite(opts_.floor_acc_variance) || opts_.floor_acc_variance <= 0.0 ||
      !std::isfinite(opts_.floor_gyro_variance) || opts_.floor_gyro_variance <= 0.0)
    throw std::invalid_argument("imu/process_noise/floor/{acc,gyro}_variance must be finite and > 0");
  paramWarn<int>(pnh, "calib/p0/autocov_lags", opts_.p0_autocov_lags, 20);
  paramWarn<double>(pnh, "calib/p0/theta_variance", opts_.p0_theta_variance, -1.0);
  paramWarn<double>(pnh, "calib/p0/bg_model_floor",
                    opts_.p0_bg_model_floor, 1.0e-5);
  paramWarn<double>(pnh, "calib/p0/numerical_eigenvalue_floor",
                    opts_.p0_numerical_eigenvalue_floor, 1e-12);
  paramWarn<std::string>(pnh, "calib/p0/gravity_cov_mode", opts_.p0_gravity_cov_mode, std::string("fixed"));
  if (opts_.p0_gravity_cov_mode != "fixed" && opts_.p0_gravity_cov_mode != "calibration")
    throw std::invalid_argument("calib/p0/gravity_cov_mode must be fixed or calibration");

  opts_.p0_autocov_lags = std::max(0, opts_.p0_autocov_lags);
  if (!std::isfinite(opts_.p0_theta_variance))
    throw std::invalid_argument("calib/p0/theta_variance must be finite (-1 keeps the exact-zero pin)");
  if (!std::isfinite(opts_.p0_bg_model_floor) || opts_.p0_bg_model_floor < 0.0 ||
      !std::isfinite(opts_.p0_numerical_eigenvalue_floor) ||
      opts_.p0_numerical_eigenvalue_floor <= 0.0)
    throw std::invalid_argument("calib/p0 floor parameters must be finite and not negative");
  std::ostringstream oss;
  oss << "[params/calib/stationary]"
      << "\n  required: true"
      << "\n  num_frames: " << opts_.num_frames;
  oss << "\n  bootstrap point covariance: sensor model"
      << "\n  sensor sigma_r2, sigma_a2: " << opts_.sensor_sigma_r2 << ", " << opts_.sensor_sigma_a2;
  oss << "\n  process_noise/floor(acc,gyro): " << opts_.floor_acc_variance << ", " << opts_.floor_gyro_variance;
  oss << "\n  p0/autocov_lags: " << opts_.p0_autocov_lags
      << "\n  p0/theta_variance: " << opts_.p0_theta_variance
      << "\n  p0/model_floor(bg): " << opts_.p0_bg_model_floor;
  oss << "\n  p0/numerical_eigenvalue_floor: "
      << opts_.p0_numerical_eigenvalue_floor;
  oss << "\n  p0/gravity_cov_mode: " << opts_.p0_gravity_cov_mode;
  return oss.str();
}

// R97A Q4 item 3: calib/p0/mode (and known_{pos,vel,theta}_variance) are removed -- position and velocity are
// always pinned exactly zero (the yaml's own "calibration_derived" values, unconditionally now). Attitude
// (theta) is pinned exactly zero too UNLESS p0_theta_variance (diagnostic-only, see its own doc comment) is
// set >= 0, in which case it becomes a known CONSTANT (not pinned, see KnownBlock's own variance semantics)
// of that isotropic value instead.
std::vector<p0::KnownBlock> CalibProc::knownP0Blocks() const
{
  const double theta_var = opts_.p0_theta_variance >= 0.0 ? opts_.p0_theta_variance : 0.0;
  return { {StateGroup::idxR(), theta_var}, {StateGroup::idxP(), 0.0}, {StateGroup::idxV(), 0.0} };
}

void CalibProc::stabilizeP0()
{
  const std::vector<p0::KnownBlock> blocks = knownP0Blocks();
  const std::vector<int> pinned = p0::pinnedIndices(blocks);
  const p0::StabilizeResult r = p0::stabilize(state_->covMut(), pinned, opts_.p0_numerical_eigenvalue_floor);
  const double min_before = r.min_before;
  const double min_after = r.min_after;
  state_->setP0Stabilization(min_before,
      opts_.p0_numerical_eigenvalue_floor, min_after);
  ROS_INFO_STREAM("[calib/P0] SPD stabilization: min_before=" << min_before
      << " floor=" << opts_.p0_numerical_eigenvalue_floor
      << " min_after=" << min_after
      << " pinned_states=" << pinned.size());
}

bool CalibProc::collectSamples()
{
  std::deque<ImuSample> imu_samples;
  std::vector<PointXYZT> points;

  // 2026-10-20: num_frames is the only stop condition now (loadParameters() requires it > 0);
  // the old num_samples-count fallback is removed.
  while (calib_frame_stop_.groups_qualifying < opts_.num_frames && data_queues_->ready())
  {
    ++calib_images_consumed_;
    calib_last_img_ = data_queues_->popImage();

    points.clear();
    data_queues_->popLidar(points, calib_last_img_.t);
    imu_samples.clear();
    data_queues_->popImu(imu_samples, calib_last_img_.t);

    // R97A Q2: the dry-run second decimation stream (and its queue) is removed.

    // Preserve the scan boundary for the explicit post-calibration map
    // bootstrap path. Keep the incumbent aggregate cloud too: aggregate
    // mode below remains the exact regression control.
    calib_observations_raw_.push_back(points);

    const bool qualifies = calibFrameStopConsumeGroup(calib_frame_stop_, static_cast<int>(imu_samples.size()));
    calib_groups_seen_ = calib_frame_stop_.groups_seen;
    calib_groups_short_imu_ = calib_frame_stop_.groups_short_imu;
    if (!qualifies)
      // Counted, not silent -- this discard is the most likely reason the
      // sample count never grows, and it was invisible.
      continue;
    calib_imu_samples.insert(
        calib_imu_samples.end(),
        imu_samples.begin(),
        imu_samples.end() - 1);
  }

  return calib_frame_stop_.groups_qualifying >= opts_.num_frames;
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

  // Stationary noise floor: isotropic (same value all 3 axes), config-
  // controlled (imu/process_noise/floor/*), measured directly from a
  // dedicated 20.6s stationary IMU calibration run (mid360 unit) rather
  // than derived from this calibration's own short collectSamples()
  // window (calib/stationary/num_frames, ~0.5s of IMU samples). That
  // short-window derivation (removed here) was found to overestimate the
  // true stationary floor by ~19x for accel and ~2x for gyro against this
  // dedicated measurement -- too few samples for a reliable variance
  // estimate.
  var_acc = V3D::Constant(opts_.floor_acc_variance);
  var_gyr = V3D::Constant(opts_.floor_gyro_variance);

  // cbk/imu/vibration/enable: the isotropic floor above was measured on a run that still has the
  // prism vibration lines in it. The stream the filter actually propagates has the lines removed
  // (CbkProc runs every live IMU sample through the canceller before it reaches the filter), so the
  // floor must be the vibration-free one from the config instead, not the isotropic one above.
  if (vib_.canceller.enable)
  {
    var_acc = V3D(vib_.floor_acc_variance[0], vib_.floor_acc_variance[1], vib_.floor_acc_variance[2]);
    var_gyr = V3D(vib_.floor_gyro_variance[0], vib_.floor_gyro_variance[1], vib_.floor_gyro_variance[2]);
  }

  ROS_INFO_STREAM("[calib] stationary noise floor (isotropic, config-set, not derived from this run's own samples"
      << (vib_.canceller.enable ? ", vibration-free: cbk/imu/vibration/floor_*" : "") << "):"
      << "  acc=[" << var_acc.x() << ", " << var_acc.y() << ", " << var_acc.z() << "]"
      << "  gyr=[" << var_gyr.x() << ", " << var_gyr.y() << ", " << var_gyr.z() << "]");
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
  M3D cov_mean = symmetrized(spectral) /
                 static_cast<double>(n);
  Eigen::SelfAdjointEigenSolver<M3D> es(cov_mean);
  if (es.info() != Eigen::Success) throw std::runtime_error("P0 mean covariance eigensolve failed");
  return symmetrized(M3D(es.eigenvectors() * es.eigenvalues().cwiseMax(0.0).asDiagonal() *
                         es.eigenvectors().transpose()));   // R74 F-419
}

void CalibProc::applyCalibrationDerivedP0(
    const M3D& R_init, const M3D& acc_mean_cov,
    const M3D& gyro_mean_cov)
{
  // 2026-10-23 (this session, R-ba-zero): P0's ba BLOCK (the one-time
  // initial covariance set here, at calibration completion -- NOT a
  // permanent pin; ba is not in knownP0Blocks(), so its covariance still
  // evolves normally frame-to-frame afterward via ordinary EKF prediction/
  // process noise, exactly like any other state) is now unconditionally
  // Zero. Previously this was J*acc_mean_cov*J^T (J = d(true_acc_bias)/
  // d(acc_mean) via central finite difference through
  // computeInitialRotation()) -- the predecessor to that (a hardcoded
  // tilt/b_a-ambiguity term plus a radial floor) was already removed
  // earlier the same session after an ablation found <0.5% effect on
  // frame-1 drift. A direct sweep of a diagnostic p0_ba_scale multiplier
  // (0, 0.1, 0.5, 1, 10, 25, 100) on this same P0 ba block found the SAME
  // result at the other end: propagated position covariance at
  // frame_idx_=6 barely moved across five orders of magnitude (0.316-
  // 0.321mm pooled std) -- it has essentially no leverage on position
  // covariance growth over these frames, in either direction. It DID have
  // a small, monotonic effect on the actual realized trajectory (more
  // initial ba uncertainty -> more drift, 1.79mm at scale=0 up to 2.57mm
  // at scale=100), so zero is strictly the best of everything tested: less
  // drift, and no measurable covariance cost since this block wasn't
  // informing anything anyway. Removed entirely rather than left at a
  // config-set zero, since the whole Jacobian machinery existed only to
  // produce this now-discarded value -- acc_mean/R_init are still needed
  // by true_acc_bias/gravity's own construction above this function, just
  // not here anymore.
  const M3D P0_ba = M3D::Zero();
  const double bg_floor = opts_.p0_bg_model_floor;

  Eigen::MatrixXd& P = state_->covMut();
  P.block(state_->idxBA(), state_->idxBA(), 3, 3) = P0_ba;
  P.block(state_->idxBG(), state_->idxBG(), 3, 3) =
      gyro_mean_cov + bg_floor * M3D::Identity();

  // R89: known blocks (exact zero position, velocity and attitude). Cross terms of those rows and columns are cleared:
  // removing states from a positive semi-definite matrix leaves it positive semi-definite.
  p0::applyKnownBlocks(P, knownP0Blocks());

  // R97A R4: calib/p0/gravity_cov_mode. "fixed" (default) leaves the gravity block exactly at
  // state/cov/gravity (the identity path below is unreachable at the default -- nothing above or
  // below this point touches P.block(idxG, idxG, ...)). Gravity magnitude/direction is otherwise
  // deliberately left at state/cov/gravity: one stationary accelerometer mean cannot separate
  // gravity from b_a (the radial part, |g| vs b_a along gravity, is NOT separable at rest).
  p0_gravity_before_ = P.block(state_->idxG(), state_->idxG(), state_->gravDim(), state_->gravDim());
  if (opts_.p0_gravity_cov_mode == "calibration")
  {
    // Cross-covariance between gravity and every other block (bg, ba, theta, p, v) stays zero --
    // the tilt/ba/gravity coupling is a known omission, not modeled here.
    const Eigen::MatrixXd P_g = gravityCovarianceFromCalibration(
        state_->gravity().normalized(), R_init, acc_mean_cov, state_->initCovGravity());
    P.block(state_->idxG(), state_->idxG(), state_->gravDim(), state_->gravDim()) = P_g;
  }

  P = symmetrized(P);
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
    // R97A Q4 item 1: the stall-abort (calib/stationary/stall_calls_max) is removed. The calibration loop's own caller
    // (LivoReconNode::run()/runOffline()) is what ends it: runOffline() breaks out the moment feedNextMessage() reports the
    // bag has no more data, so an offline run with too few stationary samples ends there, not here, with calibration simply
    // never completing. A live/online run (ros::ok() && !calib_done, fed by rosbag play rather than feedNextMessage()) has no
    // such bag-exhaustion signal of its own and WILL spin here forever on a bag with too few samples -- it depends on the
    // outer process being killed once playback ends.
    oss << "  [progress: " << calib_groups_short_imu_ << " of "
        << calib_groups_seen_ << " groups dropped for <2 IMU samples]";
    return oss.str();
  }

  // cbk/imu/vibration: the calibration window is raw (the canceller only learns until calibration is
  // done); remove the vibration lines from it in batch so the stationary mean (the bias) and its
  // covariance (covarianceOfMean) are not biased by a partial vibration cycle.
  const int vib_params = removeLinesBatch(vib_.canceller, calib_imu_samples);
  if (vib_.canceller.enable)
    ROS_INFO_STREAM("[calib] vibration lines removed from the calibration window: " << vib_params
        << " parameters over " << calib_imu_samples.size() << " samples");

  V3D acc_bias, gyro_bias, var_acc, var_gyr;
  computeBiasAndNoise(acc_bias, gyro_bias, var_acc, var_gyr);

  double r63_acc_scale = 1.0;
  if (opts_.accel_excess_model == "scale")
  {
    // acc_bias here is the stationary MEAN specific force (name is historical).
    const double mean_norm = acc_bias.norm();
    const double g_norm = state_->gravity().norm();
    if (!std::isfinite(mean_norm) || mean_norm < 1e-6)
      throw std::runtime_error("accel_excess_model=scale requires a nonzero mean accel");
    r63_acc_scale = g_norm / mean_norm;
    acc_bias *= r63_acc_scale;
    var_acc *= r63_acc_scale * r63_acc_scale;
    ROS_INFO_STREAM("[calib] accel_excess_model=scale: |mean acc| " << mean_norm
                    << " -> " << g_norm << " (scale " << r63_acc_scale
                    << "); accel bias left at its initial value");
  }
  const M3D R_init = computeInitialRotation(acc_bias);
  M3D acc_mean_cov = covarianceOfMean(true);
  acc_mean_cov *= r63_acc_scale * r63_acc_scale;
  const M3D gyro_mean_cov = covarianceOfMean(false);
  // acc_bias = true_bias + R_init^T * [0,0,9.81]; strip gravity to get true sensor bias
  const V3D true_acc_bias = acc_bias + R_init.transpose() * state_->gravity();
  const M3D r62_R_before_alignment = state_->rot();
  state_->setCalibResult(R_init, gyro_bias, true_acc_bias);
  // Stationary variance is a measured sensor-noise floor, not a complete
  // dynamic process-noise model. Keep it separate from imu/process_noise.
  state_->setNoiseFloor(var_acc, var_gyr);
  state_->setAccScale(r63_acc_scale);
  applyCalibrationDerivedP0(R_init, acc_mean_cov, gyro_mean_cov);
  stabilizeP0();
  state_->captureInitialCovariance("calibration_derived");
  writeGravityAlignmentDiagnostic(
      calib_imu_samples, acc_mean_cov, gyro_mean_cov, r62_R_before_alignment,
      R_init, *state_, true_acc_bias);
  writeCalibrationP0Diagnostic(
      *state_, static_cast<int>(calib_imu_samples.size()),
      opts_.p0_autocov_lags, acc_mean_cov, gyro_mean_cov, p0_gravity_before_);

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
      << "\n  P0 mode:    calibration_derived"
      << "\n  acc mean covariance:\n" << acc_mean_cov
      << "\n  gyro mean covariance:\n" << gyro_mean_cov
      << "\n  initialized P0:\n" << state_->cov()
      << "\n  groups_seen=" << calib_groups_seen_
      << " groups_dropped_for_fewer_than_2_imu=" << calib_groups_short_imu_
      << "\n  R_init_roll_deg=" << r_init_rpy(0) << " R_init_pitch_deg=" << r_init_rpy(1)
      // R97A R1: calib/stationary/num_frames > 0 stops by qualifying-group count instead of
      // IMU-sample count; "samples" here is the raw (untrimmed) IMU-sample sum over those
      // groups, distinct from calib_imu_samples.size() (which drops one sample per group at
      // the group boundary) -- the number asked for by R1's own worked example.
      << "\ncalib/stationary: frames = " << calib_frame_stop_.groups_qualifying
      << " samples = " << (opts_.num_frames > 0 ? calib_frame_stop_.raw_samples_qualifying
                                                 : static_cast<long>(calib_imu_samples.size()));
  const std::string done_msg = oss.str();

  data_queues_->setStartTime(calib_last_img_.t);
  calib_imu_samples.clear();
  measures_->curr_time.set(0.);

  prepareBootstrapObservations();
  // Map seeding is now an explicit post-calibration lifecycle operation.
  // Do not manufacture a synthetic MeasureGroup or route calibration points
  // through IMU/LIO/VIO/EVO.
  measures_->calib_done.set(true);
  return done_msg;
}

void CalibProc::prepareBootstrapObservations()
{
  bootstrap_observations_.clear();
  bootstrap_observations_.reserve(calib_observations_raw_.size());
  const DsMode ds_mode = opts_.ds_mode == "average" ? DsMode::AVERAGE : DsMode::FIRST;

  for (size_t obs_idx = 0; obs_idx < calib_observations_raw_.size(); ++obs_idx) {
    const auto& raw = calib_observations_raw_[obs_idx];
    std::vector<PointXYZCov> prepared;
    prepared.reserve(raw.size());
    for (const auto& p : raw) {
      const M3D cov_imu = state_->lidarToImu(getBodyCov(p.p, opts_.sensor_sigma_r2, opts_.sensor_sigma_a2));
      PointXYZCov q{state_->lidarToImu(p.p), cov_imu};
      q.t = 0.0;
      q.raw_body_point = q.point;
      prepared.push_back(q);
    }

    if (opts_.ds_mode == "off") {
      for (auto& q : prepared) q.bootstrap_observation_id = static_cast<int>(obs_idx);
      bootstrap_observations_.push_back(std::move(prepared));
    } else {
      std::vector<PointXYZCov> downsampled;
      voxelDownsample(prepared, downsampled, PointXYZCovKeyFn{opts_.ds_leaf_size}, ds_mode);
      for (auto& q : downsampled) q.bootstrap_observation_id = static_cast<int>(obs_idx);
      bootstrap_observations_.push_back(std::move(downsampled));
    }
  }

  writeBootstrapPreparationDiagnostic(calib_observations_raw_, bootstrap_observations_);
}

}  // namespace livo_recon
