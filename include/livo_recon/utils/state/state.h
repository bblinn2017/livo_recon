#pragma once

#include <memory>
#include <string>
#include <vector>
#include <ros/ros.h>
#include <Eigen/Geometry>
#include <xmlrpcpp/XmlRpcValue.h>
#include <sstream>
#include "livo_recon/utils/algo/math.h"
#include "livo_recon/utils/data/data_wrappers.h"

namespace livo_recon
{

class StateGroup
{
public:
  StateGroup();
  StateGroup& operator=(const StateGroup& o);

  // History (22-29): see docs/livo_recon_changelog.md#include-livo_recon-utils-state-state.h-22
  // R97A Q4 item 5 made rotation/position/velocity/gyro-bias/accelerometer-bias/gravity all
  // always-estimated, compile-time-constant-width blocks. 2026-10-23 (this session) partially
  // reopened that for gravity ONLY: state/gravity_estimable (default false) now controls whether
  // gravity is a live EKF state (18-dim total, gravity's own P0/process-noise/correction all real)
  // or a fixed constant set once at calibration and never touched again, with NO row/column in the
  // covariance at all (15-dim total) -- not merely pinned at zero covariance (which this project
  // already uses for position/velocity/attitude's P0, and already proved behaviorally IDENTICAL to
  // removing a state entirely when that state also has zero process noise, which gravity always
  // has -- see this flag's own default-false rationale below). The OTHER four blocks (R/P/V/BG/BA)
  // are unaffected and stay compile-time constants; only idxG() position is still a compile-time
  // constant (15 -- "where gravity's block WOULD start"), because Eigen allows a zero-row/column
  // block at exactly the end of a matrix (`.block(rows(), 0, 0, cols)` is valid, the STL-"end"
  // case) -- gravDim()/dimState() are the only two that actually depend on the flag.
  static constexpr int idxR()  { return 0; }
  static constexpr int idxP()  { return 3; }
  static constexpr int idxV()  { return 6; }
  static constexpr int idxBG() { return 9; }
  static constexpr int idxBA() { return 12; }
  static constexpr int idxG()  { return 15; }
  // 0 when gravity is fixed (state/gravity_estimable=false, the default), else 3 (vector3; the s2
  // tangent-gravity model was removed, gravity is always the free 3-vector when it IS a state).
  int gravDim() const { return gravity_estimable_ ? 3 : 0; }
  int dimState() const { return idxG() + gravDim(); }
  bool gravityEstimable() const { return gravity_estimable_; }
  // d(gravity vector)/d(gravity error coordinates) = I3.
  Eigen::MatrixXd gravityJacobian() const;
  // R63: fixed accelerometer scale factor applied to every IMU sample
  // (1.0 unless calib/stationary/accel_excess_model == scale). Not a state.
  double accScale() const { return acc_scale_; }
  void setAccScale(double s) { acc_scale_ = s; }

  // R97A Q4 item 5: estBG()/estBA()/estGravity() are removed -- all three are always true.

  // ── Setters ───────────────────────────────────────────────────────────────

  void setCalibResult(const M3D& rot, const V3D& bias_gyr, const V3D& bias_acc);
  void setNoiseParams(const V3D& var_acc, const V3D& var_gyr);

  // R97A Q3 item 3: the Q gain (imu/process_noise/gain/*) is removed.

  // The sensor's OWN measured noise, from CalibProc's static window --
  // stored ALWAYS and never used directly as the operating value. Replacing
  // the operating variance with this, per the config's own sweep comment,
  // would put the gyro
  // roughly two orders of magnitude inside the region that sweep found
  // catastrophic (it fails below ~90x the datasheet sigma and runs at
  // >=150x).  So the two are deliberately different quantities: this is a
  // BOUND, kept from the removed AdaptiveQ (the lower
  // anchor of its estimate).  Zero until CalibProc sets it.
  void setNoiseFloor(const V3D& var_acc_floor, const V3D& var_gyr_floor);
  void captureInitialCovariance(const std::string& mode)
  {
    initial_covariance_actual_ = cov_;
    initial_covariance_mode_ = mode;
  }
  void setP0Stabilization(double min_before, double floor, double min_after)
  {
    p0_min_eigenvalue_before_ = min_before;
    p0_eigenvalue_floor_ = floor;
    p0_min_eigenvalue_after_ = min_after;
  }
  void setPropagatedState(const M3D& rot, const V3D& pos, const V3D& vel);
  Eigen::MatrixXd& covMut() { return cov_; }
  // R66: mean-bias writers, used only by the opt-in Gauss-Markov bias model (imu/process_noise/bias_model).
  V3D& biasGyrMut() { return bias_gyr_; }
  V3D& biasAccMut() { return bias_acc_; }
  // TEMPORARY (2026-10-27, pin_state_diag investigation): mutable gravity access,
  // used only by the diagnostic that overwrites the live state from a reference
  // state_trace.csv row. No existing writer touches gravity_ otherwise -- delete
  // alongside the rest of that investigation's plumbing.
  V3D& gravityMut() { return gravity_; }
  void applyDelta(const Eigen::VectorXd& dx);

  // Manifold-consistent (SO(3)-aware for rotation, plain subtraction
  // elsewhere) deviation of `propagat` from the CURRENT state -- mirrors
  // FAST-LIVO2's StatesGroup::operator-()/"state_propagat - state_"
  // convention exactly: Log(this->rot^T * propagat.rot()) for the rotation
  // block, propagat.X - this->X elsewhere. Needed by EkfUpdate::
  // applyMeanUpdate()'s fixed-prior pull-back term (see ekf.h) -- absent
  // any measurement evidence (K1@Htz -> 0), the per-iteration correction
  // reverts to exactly this vector, restoring the frame-entry prior mean.
  // StateGroup has no operator- of its own (unlike StatesGroup), so this is
  // the direct equivalent rather than an operator overload.
  Eigen::VectorXd boxminusFromPropagat(const StateGroup& propagat) const;

  // ── State parameter getters ───────────────────────────────────────────────

  const M3D& rot()        const { return rot_; }
  const V3D& pos()        const { return pos_; }
  const V3D& vel()        const { return vel_; }
  const V3D& biasGyr()    const { return bias_gyr_; }
  const V3D& biasAcc()    const { return bias_acc_; }
  const V3D& gravity()    const { return gravity_; }
  const Eigen::MatrixXd& cov() const { return cov_; }
  const V3D& varAcc()     const { return var_acc_; }
  const V3D& varGyr()     const { return var_gyr_; }
  const V3D& varAccFloor() const { return var_acc_floor_; }
  const V3D& varGyrFloor() const { return var_gyr_floor_; }
  const V3D& covBiasGyr() const { return cov_bias_gyr_; }
  const V3D& covBiasAcc() const { return cov_bias_acc_; }

  // R97A Q4 item 6: initCovRotTilt/RotYaw/Pos/Vel/Bg/Ba() are removed with their `state/cov/*` keys (section above). Only
  // gravity survives (an arm of this round changes how the gravity covariance is built).
  double initCovGravity() const { return init_cov_gravity_; }
  const Eigen::MatrixXd& initialCovarianceActual() const
  {
    return initial_covariance_actual_;
  }
  const std::string& initialCovarianceMode() const
  {
    return initial_covariance_mode_;
  }
  double p0MinEigenvalueBefore() const { return p0_min_eigenvalue_before_; }
  double p0EigenvalueFloor() const { return p0_eigenvalue_floor_; }
  double p0MinEigenvalueAfter() const { return p0_min_eigenvalue_after_; }

  // ── R / t getters ─────────────────────────────────────────────────────────

  const M3D& Ric() const { return R_ic_; }
  const V3D& tic() const { return t_ic_; }
  const M3D& Rcw() const { return Rcw_; }
  const V3D& tcw() const { return tcw_; }

  // ── Point / covariance transform functions ────────────────────────────────

  V3D worldToBody(const V3D& p) const { return rot_.transpose() * (p - pos_); }
  M3D worldToBody(const M3D& R) const { return rot_.transpose() * R * rot_; }
  V3D bodyToWorld(const V3D& p) const { return rot_ * p + pos_; }
  M3D bodyToWorld(const M3D& R) const { return rot_ * R * rot_.transpose(); }
  // Transforms a body-frame point + its own (independent, per-point) sensor
  // covariance into world frame -- rotation only, no state uncertainty
  // mixed in. See poseCovAt() for that contribution, kept separate so a
  // caller (e.g. LioProc::buildResiduals()) can combine them as needed
  // (full sum for gating, sensor-only for residual weighting -- see
  // WorldPointCov in voxelmap_utils.h) instead of getting them
  // pre-mixed and unrecoverable.
  PointXYZCov toWorld(const PointXYZCov& pt_body) const;

  // The current state's pose (rotation+position) uncertainty's contribution
  // to a world-frame point's covariance, for a point at body-frame offset
  // p_body from the sensor origin -- i.e. how much extra positional
  // uncertainty this point inherits purely from not knowing the robot's
  // pose exactly (independent of the point's own sensor noise). Always PSD
  // (it's J*P_pose*J^T for the rigid-transform Jacobian J and the state's
  // own (R,P) covariance block, itself PSD).
  M3D poseCovAt(const V3D& p_body) const;

  // R93A (Bryce 2026-10-10): poseCovAtPoint, setPointCovInsertionPrior and setPriorPoseBlock (R65 F-169, the `prior` insertion source) are removed; the gate and map insertion both use poseCovAt().

  PointXYZCov toCamera(const PointXYZCov& pt_body) const;
  V3D lidarToImu(const V3D& p)  const { return R_li_ * p + t_li_; }
  M3D lidarToImu(const M3D& C)  const { return R_li_ * C * R_li_.transpose(); }
  V3D imuToCamera(const V3D& p) const { return R_ic_ * p + t_ic_; }
  const M3D& R_l2c() const { return R_lc_; }
  const V3D& t_l2c() const { return t_lc_; }
  V3D worldToCamera(const V3D& p) const { return Rcw_ * p + tcw_; }

  // ── Camera intrinsic getters ──────────────────────────────────────────────

  int    cameraWidth()  const { return camera_.width; }
  int    cameraHeight() const { return camera_.height; }
  double cameraFx()     const { return camera_.fx; }
  double cameraFy()     const { return camera_.fy; }
  double cameraCx()     const { return camera_.cx; }
  double cameraCy()     const { return camera_.cy; }
  const std::vector<double>& cameraDistortionCoeffs() const { return camera_.distortion_coeffs; }

  // ── Misc helpers ──────────────────────────────────────────────────────────

  std::string loadParameters(ros::NodeHandle& pnh);
  std::string toString() const;

private:
  struct CameraParams
  {
    int width = 0, height = 0;
    double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
    std::vector<double> distortion_coeffs;
  };

  double acc_scale_ = 1.0;
  // state/gravity_estimable (default false): see gravDim()'s own doc comment above.
  bool gravity_estimable_ = false;

  // EKF state
  M3D rot_;
  V3D pos_, vel_, bias_gyr_, bias_acc_, gravity_;
  Eigen::MatrixXd cov_;   // dimState() × dimState()
  V3D var_acc_, var_gyr_, cov_bias_gyr_, cov_bias_acc_;
  V3D var_acc_floor_ = V3D::Zero(), var_gyr_floor_ = V3D::Zero();
  double init_cov_gravity_ = 1e-5;
  Eigen::MatrixXd initial_covariance_actual_;
  std::string initial_covariance_mode_ = "uncaptured";
  double p0_min_eigenvalue_before_ = 0.0;
  double p0_eigenvalue_floor_ = 0.0;
  double p0_min_eigenvalue_after_ = 0.0;

  // Extrinsics
  M3D R_li_, R_il_, R_lc_, R_ic_;
  V3D t_li_, t_il_, t_lc_, t_ic_;
  CameraParams camera_;

  // Cached world → camera
  M3D Rcw_;  // R_ic_ * rot_.transpose()
  V3D tcw_;  // t_ic_ - Rcw_ * pos_

  void updateDerivedTransforms();
  void initCov(double cov_gravity);
};

using StateGroupPtr = std::shared_ptr<StateGroup>;

}  // namespace livo_recon
