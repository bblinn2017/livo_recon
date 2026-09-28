#pragma once

#include "livo_recon/node_context.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/log/profiler.h"

namespace livo_recon
{

struct CalibProcOptions
{
  int  num_samples       = 200;

  // Calibration used to have no way to fail. collectSamples() returns false
  // whenever the buffer is short, estimateFromBuffer() reports "Calibrating
  // IMU... N samples collected." and the node calls it again next frame --
  // forever, if the condition that starves it never clears. That is the
  // "hang" on exp01_construction_ground_level and once on site1_handheld_4's
  // pca arm: not an infinite inner loop but a precondition that is never met,
  // with no timeout and no diagnosis. After this many consecutive calls with
  // ZERO growth in the sample count, abort loudly with the counts that
  // explain why. 0 disables the bound and restores the old behaviour.
  int  stall_calls_max   = 300;
  bool apply_gyro_bias = true;
  bool apply_accel_bias = true;

  // Initial-covariance policy. "configured" is the legacy path and leaves
  // state/cov/* byte-for-byte in control. "calibration_derived" replaces
  // the observable attitude/bias blocks after the stationary calibration;
  // yaw, gravity and explicit model floors remain configured because a
  // single stationary specific-force vector cannot identify them.
  std::string p0_mode = "configured";
  int p0_autocov_lags = 20;
  double p0_known_pos_variance = 1e-12;
  double p0_known_vel_variance = 1e-12;
  // Standard deviation of the ONE transverse specific-force ambiguity
  // shared equally (in acceleration-equivalent units) by gravity tilt and
  // accelerometer bias. Negative conservatively derives it from the larger
  // of the configured tilt and b_a uncertainties.
  double p0_tilt_ba_ambiguity_accel_std = -1.0;
  double p0_bg_model_floor = -1.0;
  double p0_ba_radial_model_floor = -1.0;

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
private:
  bool        collectSamples();
  void        computeBiasAndNoise(V3D& acc_bias, V3D& gyro_bias,
                                  V3D& var_acc, V3D& var_gyr) const;
  M3D         covarianceOfMean(bool accelerometer) const;
  void        applyCalibrationDerivedP0(const V3D& acc_mean,
                                        const M3D& R_init,
                                        const M3D& acc_mean_cov,
                                        const M3D& gyro_mean_cov);
  M3D         computeInitialRotation(const V3D& acc_bias) const;

  // Why calibration is starving, counted rather than guessed. The prime
  // suspect is the `imu_samples.size() < 2` discard in collectSamples(): if
  // every image window yields fewer than two IMU samples, EVERY group is
  // dropped and the count never grows, which looks identical from outside to
  // "the bag has no IMU".
  long   calib_groups_seen_      = 0;
  long   calib_groups_short_imu_ = 0;
  long   calib_images_consumed_  = 0;
  int    calib_stall_calls_      = 0;
  size_t calib_last_count_       = 0;

  std::vector<PointXYZT> calib_points;
  std::deque<ImuSample>  calib_imu_samples;
  ImageData              calib_last_img_;

  CalibProcOptions opts_;

  DataQueuesPtr data_queues_;
  MeasuresPtr measures_;
  StateGroupPtr state_;
  ProfilerPtr profiler_;
};

}  // namespace livo_recon
