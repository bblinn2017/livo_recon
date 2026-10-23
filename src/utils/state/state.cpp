#include "livo_recon/utils/state/state.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/log/renamed_keys.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace livo_recon
{

// R97A Q4 item 5/6: gravityAlignedAttitudeCovariance() is removed -- its only call site (setCalibResult() below) wrote a
// value that CalibProc::applyCalibrationDerivedP0() always immediately overwrites to exactly zero (the attitude block is
// always pinned; see p0::applyKnownBlocks()), so it was dead.

void StateGroup::initCov(double cov_gravity)
{
  // R97A Q4 item 6: rotation, position, velocity, gyro-bias and accelerometer-bias blocks are always overwritten before
  // first use (calibration is mandatory and its P0 construction always pins rot/pos/vel to exactly zero and always
  // replaces the bias blocks), so this pre-calibration placeholder covariance only needs to size cov_ and seed gravity's.
  const int dim = dimState();
  cov_ = Eigen::MatrixXd::Zero(dim, dim);
  cov_.block(idxG(), idxG(), gravDim(), gravDim()) =
      cov_gravity * Eigen::MatrixXd::Identity(gravDim(), gravDim());
}

Eigen::MatrixXd StateGroup::gravityJacobian() const
{
  return Eigen::MatrixXd::Identity(3, 3);
}

StateGroup::StateGroup()
{
  rot_ = M3D::Identity();
  pos_ = V3D::Zero();
  vel_ = V3D::Zero();
  bias_gyr_ = V3D::Zero();
  bias_acc_ = V3D::Zero();
  gravity_  = V3D::Zero();

  initCov(1e-5);

  var_acc_ = V3D::Zero();
  var_gyr_ = V3D::Zero();
  cov_bias_gyr_ = V3D::Zero();
  cov_bias_acc_ = V3D::Zero();

  R_li_.setIdentity(); t_li_.setZero();
  R_il_.setIdentity(); t_il_.setZero();
  R_lc_.setIdentity(); t_lc_.setZero();
  R_ic_.setIdentity(); t_ic_.setZero();

  updateDerivedTransforms();
}

StateGroup& StateGroup::operator=(const StateGroup& o)
{
  if (this != &o) {
    acc_scale_ = o.acc_scale_;
    gravity_estimable_ = o.gravity_estimable_;
    rot_ = o.rot_; pos_ = o.pos_; vel_ = o.vel_;
    bias_gyr_ = o.bias_gyr_; bias_acc_ = o.bias_acc_; gravity_ = o.gravity_;
    cov_ = o.cov_;
    var_acc_ = o.var_acc_; var_gyr_ = o.var_gyr_;
    // R73b: the stationary calibration floors were never copied (found by r73_smoke_shadow_iso_60s: the shadow state is assigned from the driver every scan and
    // an isotropic motion-Q shadow threw "motion-dependent Q requires positive stationary calibration floors"). Copying them is inert for the driver.
    var_acc_floor_ = o.var_acc_floor_; var_gyr_floor_ = o.var_gyr_floor_;
    cov_bias_gyr_ = o.cov_bias_gyr_; cov_bias_acc_ = o.cov_bias_acc_;
    init_cov_gravity_ = o.init_cov_gravity_;
    initial_covariance_actual_ = o.initial_covariance_actual_;
    initial_covariance_mode_ = o.initial_covariance_mode_;
    p0_min_eigenvalue_before_ = o.p0_min_eigenvalue_before_;
    p0_eigenvalue_floor_ = o.p0_eigenvalue_floor_;
    p0_min_eigenvalue_after_ = o.p0_min_eigenvalue_after_;
    R_li_ = o.R_li_; t_li_ = o.t_li_;
    R_il_ = o.R_il_; t_il_ = o.t_il_;
    R_lc_ = o.R_lc_; t_lc_ = o.t_lc_;
    R_ic_ = o.R_ic_; t_ic_ = o.t_ic_;
    camera_ = o.camera_;
    updateDerivedTransforms();
  }
  return *this;
}

PointXYZCov StateGroup::toWorld(const PointXYZCov& pt_body) const
{
  return {
    rot_ * pt_body.point + pos_,
    rot_ * pt_body.sensor_cov * rot_.transpose()
  };
}

M3D StateGroup::poseCovAt(const V3D& p_body) const
{
  M3D p_skew;
  p_skew << SKEW_SYM_MATRX(p_body);

  const M3D P_RR = cov_.block<3,3>(idxR(), idxR());
  const M3D P_PP = cov_.block<3,3>(idxP(), idxP());
  const M3D P_RP = cov_.block<3,3>(idxR(), idxP());

  // Propagates the state's own (R,P) covariance block through the rigid
  // transform's Jacobian J=-R*K (K=p_skew) w.r.t. rotation, plus the
  // identity Jacobian w.r.t. position: J*P_RR*J^T = R*K*P_RR*K*R^T (K
  // symmetric skew-square sign flips this to a plain product, see the
  // original combined derivation this was split out of), plus P_PP, plus
  // the R/P cross term. Cross term computed once;
  // P_RP^T*J^T = (J*P_RP)^T = -(Rp*P_RP)^T.
  const M3D Rp  = rot_ * p_skew;   // = -J_rot
  const M3D JRP = Rp * P_RP;       // J*P_RP = -Rp*P_RP, so cross = -(JRP + JRP^T)

  return -rot_ * p_skew * P_RR * p_skew * rot_.transpose()
       + P_PP
       - JRP - JRP.transpose();
}

PointXYZCov StateGroup::toCamera(const PointXYZCov& pt_body) const
{
  return {
    R_ic_ * pt_body.point + t_ic_,
    R_ic_ * pt_body.sensor_cov * R_ic_.transpose()
  };
}

void StateGroup::updateDerivedTransforms()
{
  Rcw_ = R_ic_ * rot_.transpose();
  tcw_ = t_ic_ - Rcw_ * pos_;
}

void StateGroup::setCalibResult(const M3D& rot, const V3D& bias_gyr, const V3D& bias_acc)
{
  rot_      = rot;
  vel_      = V3D::Zero();
  bias_gyr_ = bias_gyr;
  bias_acc_ = bias_acc;
  // R97A Q4 item 6: the attitude block used to be seeded here (gravityAlignedAttitudeCovariance()); it is now left alone --
  // CalibProc::applyCalibrationDerivedP0() always pins it to exactly zero immediately afterward.
  updateDerivedTransforms();
}

void StateGroup::setNoiseParams(const V3D& var_acc, const V3D& var_gyr)
{
  var_acc_ = var_acc;
  var_gyr_ = var_gyr;
}

void StateGroup::setNoiseFloor(const V3D& var_acc_floor, const V3D& var_gyr_floor)
{
  var_acc_floor_ = var_acc_floor;
  var_gyr_floor_ = var_gyr_floor;
}

void StateGroup::setPropagatedState(const M3D& rot, const V3D& pos, const V3D& vel)
{
  rot_ = rot; pos_ = pos; vel_ = vel;
  updateDerivedTransforms();
}

void StateGroup::applyDelta(const Eigen::VectorXd& dx)
{
  rot_      = rot_ * Exp(dx.segment<3>(idxR()));
  pos_      += dx.segment<3>(idxP());
  vel_      += dx.segment<3>(idxV());
  bias_gyr_ += dx.segment<3>(idxBG());
  bias_acc_ += dx.segment<3>(idxBA());
  // gravity_estimable_=false: gravDim()==0, dx has no segment at idxG() at all (dx.size() ==
  // dimState() == 15) -- gravity_ stays exactly what calibration/config set it to, forever.
  if (gravity_estimable_) gravity_ += dx.segment<3>(idxG());
  updateDerivedTransforms();
}

Eigen::VectorXd StateGroup::boxminusFromPropagat(const StateGroup& propagat) const
{
  Eigen::VectorXd vec = Eigen::VectorXd::Zero(dimState());
  const M3D rotd = rot_.transpose() * propagat.rot_;
  vec.segment<3>(idxR()) = Log(rotd);
  vec.segment<3>(idxP()) = propagat.pos_ - pos_;
  vec.segment<3>(idxV()) = propagat.vel_ - vel_;
  vec.segment<3>(idxBG()) = propagat.bias_gyr_ - bias_gyr_;
  vec.segment<3>(idxBA()) = propagat.bias_acc_ - bias_acc_;
  if (gravity_estimable_) vec.segment<3>(idxG()) = propagat.gravity_ - gravity_;
  return vec;
}

// History (156-158): see docs/livo_recon_changelog.md#src-utils-state-state.cpp-156

std::string StateGroup::loadParameters(ros::NodeHandle& pnh)
{
  for (const char* legacy : {"state/cov/acc", "state/cov/gyr",
                             "imu/bias_gyr_rw", "imu/bias_acc_rw"})
  {
    if (pnh.hasParam(legacy))
      throw std::invalid_argument(std::string("removed mixed-responsibility IMU key: ") +
          legacy + "; use imu/process_noise/fixed or bias_random_walk");
  }
  refuseRenamedKeys(pnh);   // R97A Q4: catches state/est/{bg,ba,gravity}, state/cov/*, state/init_state/{vel,ba,bg}, etc.
  // R97A Q4 item 5: state/est/{bg,ba,gravity} are removed -- all three are always true.
  if (pnh.hasParam("state/gravity_model"))
    throw std::invalid_argument(
        "state/gravity_model was removed: the s2 tangent-gravity model is gone, gravity is "
        "always the free 3-vector (vector3) -- remove this key");
  // 2026-10-23: state/gravity_estimable -- see gravDim()'s own doc comment in state.h. Read here,
  // before initCov() below, since initCov() sizes cov_ by dimState() which depends on this.
  // Default false: gravity has zero process noise everywhere in this codebase (confirmed --
  // imu_processing.cpp's covariance propagation has no cov_w.block(idxG(),idxG(),...) term), so a
  // live-estimated gravity with any nonzero P0 is only ever as correctable as that P0 allows, and
  // a full-bag scale-less ATE comparison (gravity P0 1e-5 vs 0, capture_10_20_no3/no4/no5) found
  // no measurable difference either way on these bags -- fixing it outright (no P0, no row/column
  // in the covariance at all) costs nothing observed and removes 3 live state dimensions' worth of
  // Jacobian/Kalman-gain work every scan.
  paramWarn<bool>(pnh, "state/gravity_estimable", gravity_estimable_, false);

  std::vector<double> R_l2i, t_l2i, R_l2c, t_l2c;
  paramWarn<std::vector<double>>(pnh, "extrinsics/R_l2i", R_l2i, std::vector<double>());
  paramWarn<std::vector<double>>(pnh, "extrinsics/t_l2i", t_l2i, std::vector<double>());
  paramWarn<std::vector<double>>(pnh, "extrinsics/R_l2c", R_l2c, std::vector<double>());
  paramWarn<std::vector<double>>(pnh, "extrinsics/t_l2c", t_l2c, std::vector<double>());

  if (R_l2i.size() != 9 || t_l2i.size() != 3)
  {
    ROS_ERROR_STREAM("Invalid lidar->imu extrinsics");
    return {};
  }

  for (int i = 0; i < 9; ++i) R_li_(i / 3, i % 3) = R_l2i[i];
  for (int i = 0; i < 3; ++i) t_li_(i)             = t_l2i[i];
  R_il_ = R_li_.transpose();
  t_il_ = -R_il_ * t_li_;

  if (R_l2c.size() == 9 && t_l2c.size() == 3)
  {
    for (int i = 0; i < 9; ++i) R_lc_(i / 3, i % 3) = R_l2c[i];
    for (int i = 0; i < 3; ++i) t_lc_(i)             = t_l2c[i];
  }
  else
  {
    ROS_WARN_STREAM("Lidar->camera extrinsics not provided or invalid");
  }

  R_ic_ = R_lc_ * R_il_;
  t_ic_ = R_lc_ * t_il_ + t_lc_;

  paramWarn<int>(pnh, "camera/width",            camera_.width,            0);
  paramWarn<int>(pnh, "camera/height",           camera_.height,           0);
  paramWarn<double>(pnh, "camera/fx",               camera_.fx,               0.0);
  paramWarn<double>(pnh, "camera/fy",               camera_.fy,               0.0);
  paramWarn<double>(pnh, "camera/cx",               camera_.cx,               0.0);
  paramWarn<double>(pnh, "camera/cy",               camera_.cy,               0.0);

  XmlRpc::XmlRpcValue dist;
  markParamConsumed("camera/distortion_coeffs");
  if (pnh.getParam("camera/distortion_coeffs", dist) &&
      dist.getType() == XmlRpc::XmlRpcValue::TypeArray)
  {
    camera_.distortion_coeffs.clear();
    for (int i = 0; i < dist.size(); ++i)
    {
      if (dist[i].getType() == XmlRpc::XmlRpcValue::TypeDouble)
        camera_.distortion_coeffs.push_back(static_cast<double>(dist[i]));
      else if (dist[i].getType() == XmlRpc::XmlRpcValue::TypeInt)
        camera_.distortion_coeffs.push_back(static_cast<int>(dist[i]));
    }
  }
  else
  {
    camera_.distortion_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};
  }

  updateDerivedTransforms();

  // R97A Q4 item 6: state/cov/{rot_tilt,rot_yaw,pos,vel,bg,ba} are removed -- CalibProc::applyCalibrationDerivedP0() always
  // pins position/velocity/attitude to exactly zero and always replaces the bias blocks, so none of these six keys had a
  // consumer left after Q4 items 3-5. state/cov/gravity is kept -- it still seeds initCov() below,
  // but only meaningful (and read) when state/gravity_estimable is true; refused if set otherwise
  // (gravity_estimable_=false has no gravity row/column in cov_ at all for this to seed).
  if (gravity_estimable_)
  {
    paramWarn<double>(pnh, "state/cov/gravity", init_cov_gravity_, 1e-5);
  }
  else if (pnh.hasParam("state/cov/gravity"))
  {
    throw std::invalid_argument(
        "state/cov/gravity is set but state/gravity_estimable is false -- gravity has no "
        "covariance block to seed when it isn't a live state");
  }
  initCov(init_cov_gravity_);

  // R97A Q4 item 6: state/init_state/{vel,ba,bg} are removed -- CalibProc::estimateFromBuffer() always overwrites vel_
  // (to zero) and bias_gyr_/bias_acc_ (from the mandatory stationary calibration) before first use. state/init_state/gravity
  // is kept (the only one with a live consumer: the uncalibrated starting gravity vector calibration aligns against).
  std::vector<double> g_vec;
  paramWarn<std::vector<double>>(pnh, "state/init_state/gravity", g_vec,  {0., 0., -G_m_s2});
  gravity_  = V3D(g_vec[0],  g_vec[1],  g_vec[2]);

  // Fixed IMU propagation noise. Stationary calibration records a separate
  // sensor floor and never silently replaces these operating values.
  double var_acc, var_gyr;
  paramWarn<double>(pnh, "imu/process_noise/fixed/acc_variance", var_acc, 1e-4);
  paramWarn<double>(pnh, "imu/process_noise/fixed/gyro_variance", var_gyr, 1e-4);
  var_acc_ = V3D::Constant(var_acc);
  var_gyr_ = V3D::Constant(var_gyr);
  // CQ-55 item 10: state/cov/bg and state/cov/ba used to seed BOTH the
  // initial covariance (units rad^2/s^2, "we just calibrated and are not
  // sure") AND the random-walk RATE (units rad^2/s^3, imu_processing.cpp's
  // covBiasGyr()*dt term) -- two independent physical facts that cannot
  // correctly share one value. Measured: with the initial value (1e-5 in
  // config/ntu_viral.yaml) used as the rate, the licensed 1-sigma
  // gyro-bias walk over a 69.4s window is 543x a MEMS sensor's own quoted
  // in-run stability. Split into their own keys.
  // R97A Q3 item 1: with the Gauss-Markov bias model removed, random walk
  // is the only bias model; these two keys default to the code constant
  // 1e-5 (the yaml value) rather than mirroring state/cov/bg|ba (which Q4
  // removes). Identity at ntu_viral.yaml: both yamls set these keys
  // explicitly, so the default never applies there. Report: any OTHER
  // yaml that relied on the old default (sharing state/cov/bg|ba) now
  // gets 1e-5 instead.
  double bias_gyr_rw = 1.0e-5, bias_acc_rw = 1.0e-5;
  paramWarn<double>(pnh, "imu/process_noise/bias_random_walk/gyro_variance_rate",
                    bias_gyr_rw, 1.0e-5);
  paramWarn<double>(pnh, "imu/process_noise/bias_random_walk/accel_variance_rate",
                    bias_acc_rw, 1.0e-5);
  if (!std::isfinite(bias_gyr_rw) || bias_gyr_rw <= 0.0)
    throw std::invalid_argument(
        "imu/process_noise/bias_random_walk/gyro_variance_rate must be finite and > 0");
  if (!std::isfinite(bias_acc_rw) || bias_acc_rw <= 0.0)
    throw std::invalid_argument(
        "imu/process_noise/bias_random_walk/accel_variance_rate must be finite and > 0");
  cov_bias_acc_ = V3D::Constant(bias_acc_rw);
  cov_bias_gyr_ = V3D::Constant(bias_gyr_rw);

  std::ostringstream oss;
  oss << "[params/transform]"
      << "\n  R_l2i:\n" << R_li_
      << "\n  t_l2i: " << t_li_.transpose()
      << "\n  R_l2c:\n" << R_lc_
      << "\n  t_l2c: " << t_lc_.transpose()
      << "\n  camera: " << camera_.width << "x" << camera_.height
      << "  fx=" << camera_.fx << "  fy=" << camera_.fy
      << "  cx=" << camera_.cx << "  cy=" << camera_.cy
      << "  gravity_model=vector3"
      << "  gravity_estimable=" << (gravity_estimable_ ? "true" : "false")
      << "  dimState=" << dimState();
  return oss.str();
}

std::string StateGroup::toString() const
{
  const Eigen::Quaterniond q(rot_);
  std::ostringstream oss;
  oss << "[state]"
      << "\n  pos:      " << pos_.transpose()
      << "\n  vel:      " << vel_.transpose()
      << "\n  quat:     " << q.w() << " " << q.x() << " " << q.y() << " " << q.z()
      << "\n  bias_gyr: " << bias_gyr_.transpose()
      << "\n  bias_acc: " << bias_acc_.transpose()
      << "\n  gravity:  " << gravity_.transpose();
  return oss.str();
}

}  // namespace livo_recon
