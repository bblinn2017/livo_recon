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
  static constexpr int idxR() { return 0; }
  static constexpr int idxP() { return 3; }
  static constexpr int idxV() { return 6; }
  int idxBG() const { return est_bg_      ? 9                                     : -1; }
  int idxBA() const { return est_ba_      ? 9 + (est_bg_ ? 3 : 0)                 : -1; }
  int idxG()  const { return est_gravity_ ? 9 + (est_bg_ ? 3 : 0) + (est_ba_ ? 3 : 0) : -1; }
  int dimState() const { return 9 + (est_bg_ ? 3 : 0) + (est_ba_ ? 3 : 0) + (est_gravity_ ? 3 : 0); }

  bool estBG()      const { return est_bg_; }
  bool estBA()      const { return est_ba_; }
  bool estGravity() const { return est_gravity_; }

  // ── Setters ───────────────────────────────────────────────────────────────

  void setCalibResult(const M3D& rot, const V3D& bias_gyr, const V3D& bias_acc);
  void setNoiseParams(const V3D& var_acc, const V3D& var_gyr);
  void setPropagatedState(const M3D& rot, const V3D& pos, const V3D& vel);
  Eigen::MatrixXd& covMut() { return cov_; }
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
  const V3D& covBiasGyr() const { return cov_bias_gyr_; }
  const V3D& covBiasAcc() const { return cov_bias_acc_; }

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

  PointXYZCov toCamera(const PointXYZCov& pt_body) const;
  V3D lidarToImu(const V3D& p)  const { return R_li_ * p + t_li_; }
  M3D lidarToImu(const M3D& C)  const { return R_li_ * C * R_li_.transpose(); }
  V3D imuToCamera(const V3D& p) const { return R_ic_ * p + t_ic_; }
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
    std::string distortion_model = "plumb_bob";
    std::vector<double> distortion_coeffs;
  };

  // Estimation flags (set before cov_ is sized)
  bool est_bg_      = true;
  bool est_ba_      = true;
  bool est_gravity_ = true;
  bool est_cov_acc_ = false;
  bool est_cov_gyr_ = false;

  // EKF state
  M3D rot_;
  V3D pos_, vel_, bias_gyr_, bias_acc_, gravity_;
  Eigen::MatrixXd cov_;   // dimState() × dimState()
  V3D var_acc_, var_gyr_, cov_bias_gyr_, cov_bias_acc_;

  // Extrinsics
  M3D R_li_, R_il_, R_lc_, R_ic_;
  V3D t_li_, t_il_, t_lc_, t_ic_;
  CameraParams camera_;

  // Cached world → camera
  M3D Rcw_;  // R_ic_ * rot_.transpose()
  V3D tcw_;  // t_ic_ - Rcw_ * pos_

  void updateDerivedTransforms();
  void initCov(double cov_rot, double cov_pos, double cov_vel,
               double cov_bg, double cov_ba, double cov_gravity);
};

using StateGroupPtr = std::shared_ptr<StateGroup>;

}  // namespace livo_recon
