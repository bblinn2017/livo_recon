#include "livo_recon/processing/calib_processing.h"
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
  paramWarn<int>(pnh, "calib/num_samples",   opts_.num_samples,   200);
  paramWarn<bool>(pnh, "calib/use_calib",     opts_.use_calib,     true);
  paramWarn<bool>(pnh, "calib/use_calib_var",  opts_.use_calib_var,  false);
  paramWarn<bool>(pnh, "calib/use_calib_bias", opts_.use_calib_bias, true);

  if (opts_.num_samples < 1)
    opts_.num_samples = 1;

  std::ostringstream oss;
  oss << "[params/calib]"
      << "\n  num_samples:    " << opts_.num_samples
      << "\n  use_calib:      " << (opts_.use_calib      ? "true" : "false")
      << "\n  use_calib_var:  " << (opts_.use_calib_var  ? "true" : "false")
      << "\n  use_calib_bias: " << (opts_.use_calib_bias ? "true" : "false");
  return oss.str();
}

std::string CalibProc::skipCalibration()
{
  if (!data_queues_->ready())
    return "Waiting for first frame (calib skipped — using provided init state)...";

  ImageData img = data_queues_->popImage();
  std::vector<PointXYZT> points;
  data_queues_->popLidar(points, img.t);
  std::deque<ImuSample> imu_samples;
  data_queues_->popImu(imu_samples, img.t);
  // See collectSamples()'s matching comment -- same backlog-drain reasoning
  // applies here (this is calibration's other entry point, use_calib=false).
  std::vector<PointXYZT> dry_run_points;
  data_queues_->popDryRunLidar(dry_run_points, img.t);

  data_queues_->setStartTime(img.t);
  img.t = 0.;
  for (auto& p : points) p.t = 0.;
  imu_samples.clear();
  measures_->curr_time.set(0.);
  measures_->pushMeasureGroup(
      MeasureGroup{std::move(img), std::move(points), std::move(imu_samples)});
  measures_->calib_done.set(true);

  std::ostringstream oss;
  oss << "Calibration skipped — using provided init state."
      << "\n  acc bias:   " << state_->biasAcc().transpose()
      << "\n  gyro bias:  " << state_->biasGyr().transpose()
      << "\n  acc noise:  " << state_->varAcc().transpose()
      << "\n  gyro noise: " << state_->varGyr().transpose()
      << "\n  gravity:    " << state_->gravity().transpose();
  return oss.str();
}

bool CalibProc::collectSamples()
{
  std::deque<ImuSample> imu_samples;
  std::vector<PointXYZT> points;

  while (calib_imu_samples.size() < static_cast<size_t>(opts_.num_samples) &&
         data_queues_->ready())
  {
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
      continue;
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

  var_acc = V3D::Constant(acc_cov.trace()  / 3.0);
  var_gyr = V3D::Constant(gyro_cov.trace() / 3.0);
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

  if (!opts_.use_calib)
    return skipCalibration();

  std::ostringstream oss;
  oss << "Calibrating IMU... " << calib_imu_samples.size() << " samples collected.";
  if (!collectSamples())
    return oss.str();

  V3D acc_bias, gyro_bias, var_acc, var_gyr;
  computeBiasAndNoise(acc_bias, gyro_bias, var_acc, var_gyr);

  const M3D R_init = computeInitialRotation(acc_bias);
  // acc_bias = true_bias + R_init^T * [0,0,9.81]; strip gravity to get true sensor bias
  const V3D true_acc_bias = acc_bias + R_init.transpose() * state_->gravity();
  state_->setCalibResult(R_init,
                         opts_.use_calib_bias ? gyro_bias   : state_->biasGyr(),
                         opts_.use_calib_bias ? true_acc_bias : state_->biasAcc());
  // use_calib_var defaults false -- see CalibProcOptions's docs. var_acc/
  // var_gyr (computed above) are discarded in that case; state.yaml's
  // cov/acc,gyr (hardcoded, per-sensor) stays in effect untouched.
  // ALWAYS publish the measured variance as a FLOOR, whatever
  // use_calib_var says -- see StateGroup::setNoiseFloor()'s doc comment for
  // why the bound and the operating value must stay separate quantities.
  // Previously this number was computed on every run and then discarded
  // whenever use_calib_var was false, which is both shipped configs.
  state_->setNoiseFloor(var_acc, var_gyr);
  if (opts_.use_calib_var)
    state_->setNoiseParams(var_acc, var_gyr);

  oss << "\nIMU calibration done with " << calib_imu_samples.size() << " samples."
      << "\n  acc bias:   " << state_->biasAcc().transpose()
      << "\n  gyro bias:  " << state_->biasGyr().transpose()
      << "\n  acc noise:  " << state_->varAcc().transpose()
      << "\n  gyro noise: " << state_->varGyr().transpose()
      << "\n  acc floor:  " << state_->varAccFloor().transpose()
      << "\n  gyro floor: " << state_->varGyrFloor().transpose()
      << "\n  gravity:    " << state_->gravity().transpose();
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
