#pragma once

#include "livo_recon/node_context.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/log/profiler.h"
#include "livo_recon/lio/p0_pin.h"
#include <vector>

namespace livo_recon
{

// R97A R1: the group-counting policy `collectSamples()` uses, factored out as a pure function so
// it can be unit-tested against a synthetic stream of group sizes without a live DataQueues/ROS
// stack. One call = one group with `imu_count` IMU samples in it; `num_frames` <= 0 means "no
// frame-count stop" (the caller only wants to know whether the group qualifies). Returns whether
// this group counts towards num_frames (>= 2 IMU samples).
struct CalibFrameStopState
{
  long groups_seen = 0;
  long groups_short_imu = 0;
  long groups_qualifying = 0;
  // Raw (untrimmed) sum of imu_count over qualifying groups -- the value the
  // "calib/stationary: frames = <n> samples = <m>" log line reports under
  // num_frames > 0 (distinct from calib_imu_samples.size(), which is this
  // sum minus one IMU sample per qualifying group -- the existing boundary
  // trim collectSamples() performs when inserting into calib_imu_samples).
  long raw_samples_qualifying = 0;
};

inline bool calibFrameStopConsumeGroup(CalibFrameStopState& s, int imu_count)
{
  ++s.groups_seen;
  if (imu_count < 2)
  {
    ++s.groups_short_imu;
    return false;
  }
  ++s.groups_qualifying;
  s.raw_samples_qualifying += imu_count;
  return true;
}

struct CalibProcOptions
{
  int  num_samples       = 200;
  // R97A R1: when > 0, the calibration stops after this many measure groups
  // (image windows) with >= 2 IMU samples have been consumed, instead of
  // stopping at num_samples IMU samples. 0 (default) is today's behaviour.
  int  num_frames        = 0;
  std::string ds_mode = "first";
  double ds_leaf_size = 0.15;

  // R67 (F-265). Sensor covariance of the bootstrap (calibration) points.
  //   "sensor_model" (default): the same model the live scans use in deskewPoints(): getBodyCov(p_lidar, sigma_r2, sigma_a2)
  //                             rotated to the IMU frame with the LiDAR-IMU extrinsic. There is no motion inside a stationary
  //                             calibration window, so the deskew rotation is the identity and no time-based term applies.
  //   "zero":                   the R53-R66 behaviour (M3D::Zero()); kept ONLY as the byte-identity control of the R67 patch.
  //                             To be removed after R67 (rule 3.3), not a tuning option.
  // sigma_r2 / sigma_a2 are parsed from imu/sensor/range_err and imu/sensor/angle_err_deg with the same defaults and the same
  // formula as LioProcBase (lio_base.cpp), so the two paths cannot disagree.
  double sensor_sigma_r2 = 0.05 * 0.05;
  double sensor_sigma_a2 = 0.0;

  // R97A Q4: stall_calls_max (the abort on consecutive zero-growth calls) is removed.
  // apply_to_state/gyro_bias and accel_bias are always on (the yaml sets both true).
  // R63: how the stationary accelerometer-magnitude excess (|mean acc| - G)
  // is explained. "bias" (default, historical): absorbed by the z accel
  // bias state. "scale": a fixed scale factor G/|mean acc| is applied to
  // every IMU sample (as FAST-LIO2/FAST-LIVO2 do) and the bias stays at its
  // configured initial value.
  std::string accel_excess_model = "bias";

  // R97A Q4: calib/p0/mode (and its "configured" branch) is removed. P0's
  // position, velocity and attitude blocks are always exactly zero -- the
  // pinned, excluded-from-the-floor path "calibration_derived" always
  // selected with known_pos/vel/theta_variance == 0.
  int p0_autocov_lags = 20;
  // Standard deviation of the ONE transverse specific-force ambiguity
  // shared equally (in acceleration-equivalent units) by gravity tilt and
  // accelerometer bias.
  double p0_tilt_ba_ambiguity_accel_std = 0.03;
  double p0_bg_model_floor = 1.0e-5;
  double p0_ba_radial_model_floor = 1.0e-5;
  // Purely numerical covariance floor. This is not a physical uncertainty
  // model: it only makes the symmetric P0 strictly positive definite for
  // information-form solves when a physically valid construction is PSD.
  double p0_numerical_eigenvalue_floor = 1e-12;

  // A stationary calibration window measures the sensor's noise *floor*,
  // not the process noise real dynamic motion needs. The measured floor is
  // always stored separately and never replaces imu/process_noise/fixed.
  // Using it unmodified as EKF process noise makes the filter
  // drastically overconfident in IMU-only propagation between corrections
  // (confirmed: NTU VIRAL's eee_01 calibrates to acc=0.00434, gyr=0.0000636
  // -- ~100-1000x smaller than the values needed for stable tracking on
  // that dataset, and directly caused a residual-matching death spiral
  // once corrections thinned out). Verified with Allan variance too: a more
  // rigorous noise estimate agrees with the simple calibration within
  // ~15-20%, so this isn't a calibration-quality problem -- the required
  // margin reflects model uncertainty (unmodeled dynamics, mounting
  // vibration, linearization error) that no stationary recording can ever
  // measure, calibrated carefully or not.
  //
  // So var_acc/var_gyr remain whatever imu/process_noise/fixed configures. The
  // stationary window is mandatory, always gravity-aligns attitude, and can
  // independently apply or merely report each estimated bias.
};

class CalibProc
{
public:
  explicit CalibProc(NodeContext& ctx);

  std::string loadParameters(ros::NodeHandle& pnh);

  std::string estimateFromBuffer();
  const std::vector<std::vector<PointXYZCov>>& getBootstrapObservations() const
  { return bootstrap_observations_; }
  const ImageData& calibrationLastImage() const { return calib_last_img_; }
private:
  bool        collectSamples();
  void        computeBiasAndNoise(V3D& acc_bias, V3D& gyro_bias,
                                  V3D& var_acc, V3D& var_gyr) const;
  M3D         covarianceOfMean(bool accelerometer) const;
  void        applyCalibrationDerivedP0(const V3D& acc_mean,
                                        const M3D& R_init,
                                        const M3D& acc_mean_cov,
                                        const M3D& gyro_mean_cov);
  void        stabilizeP0();
  std::vector<p0::KnownBlock> knownP0Blocks() const;
  M3D         computeInitialRotation(const V3D& acc_bias) const;
  void        prepareBootstrapObservations();

  // Why calibration is starving, counted rather than guessed. The prime
  // suspect is the `imu_samples.size() < 2` discard in collectSamples(): if
  // every image window yields fewer than two IMU samples, EVERY group is
  // dropped and the count never grows, which looks identical from outside to
  // "the bag has no IMU".
  long   calib_groups_seen_      = 0;
  long   calib_groups_short_imu_ = 0;
  long   calib_images_consumed_  = 0;
  // R97A R1: num_frames stop-condition bookkeeping (see CalibFrameStopState above).
  CalibFrameStopState calib_frame_stop_;

  std::vector<std::vector<PointXYZT>> calib_observations_raw_;
  std::vector<std::vector<PointXYZCov>> bootstrap_observations_;
  std::deque<ImuSample>  calib_imu_samples;
  ImageData              calib_last_img_;

  CalibProcOptions opts_;

  DataQueuesPtr data_queues_;
  MeasuresPtr measures_;
  StateGroupPtr state_;
  ProfilerPtr profiler_;
};

}  // namespace livo_recon
