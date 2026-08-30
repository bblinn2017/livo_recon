#pragma once

#include "livo_recon/node_context.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/log/profiler.h"

namespace livo_recon
{

struct CalibProcOptions
{
  int  num_samples       = 200;
  bool use_calib         = true;
  bool use_calib_var      = false;
  bool use_calib_bias     = true;

  // use_calib_var default false: a stationary calibration window measures
  // the sensor's noise *floor*, not the process noise real dynamic motion
  // needs -- using it unmodified as EKF process noise makes the filter
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
  // So: var_acc/var_gyr are left as whatever state.yaml's cov/acc,gyr
  // hardcodes (per-sensor/per-dataset, tuned empirically -- see e.g.
  // ntu_viral/state.yaml), not derived from calibration at all. Calibration
  // (use_calib) still runs and still sets bias (use_calib_bias) and the
  // initial gravity-aligned rotation (computeInitialRotation(), from the
  // same stationary acc_bias) -- only the noise/variance side is skipped.
};

class CalibProc
{
public:
  explicit CalibProc(NodeContext& ctx);

  std::string loadParameters(ros::NodeHandle& pnh);

  std::string estimateFromBuffer();
private:
  std::string skipCalibration();
  bool        collectSamples();
  void        computeBiasAndNoise(V3D& acc_bias, V3D& gyro_bias,
                                  V3D& var_acc, V3D& var_gyr) const;
  M3D         computeInitialRotation(const V3D& acc_bias) const;

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
