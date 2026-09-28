#include "livo_recon/utils/state/state.h"
#include "livo_recon/utils/log/param_warn.h"

#include <stdexcept>

namespace livo_recon
{

M3D StateGroup::gravityAlignedAttitudeCovariance(
    const M3D& aligned_rot, double tilt_variance, double yaw_variance)
{
  const V3D yaw_axis_body =
      (aligned_rot.transpose() * V3D(0.0, 0.0, -1.0)).normalized();
  return tilt_variance * M3D::Identity() +
      (yaw_variance - tilt_variance) *
          yaw_axis_body * yaw_axis_body.transpose();
}

void StateGroup::initCov(double cov_rot, double cov_pos, double cov_vel,
                          double cov_bg, double cov_ba, double cov_gravity)
{
  const int dim = dimState();
  cov_ = Eigen::MatrixXd::Zero(dim, dim);
  cov_.block<3,3>(idxR(), idxR()) = cov_rot     * M3D::Identity();
  cov_.block<3,3>(idxP(), idxP()) = cov_pos     * M3D::Identity();
  cov_.block<3,3>(idxV(), idxV()) = cov_vel     * M3D::Identity();
  if (est_bg_)      cov_.block(idxBG(), idxBG(), 3, 3) = cov_bg      * M3D::Identity();
  if (est_ba_)      cov_.block(idxBA(), idxBA(), 3, 3) = cov_ba      * M3D::Identity();
  if (est_gravity_) cov_.block(idxG(),  idxG(),  3, 3) = cov_gravity * M3D::Identity();
}

StateGroup::StateGroup()
{
  rot_ = M3D::Identity();
  pos_ = V3D::Zero();
  vel_ = V3D::Zero();
  bias_gyr_ = V3D::Zero();
  bias_acc_ = V3D::Zero();
  gravity_  = V3D::Zero();

  initCov(1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-5);

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
    est_bg_      = o.est_bg_;
    est_ba_      = o.est_ba_;
    est_gravity_ = o.est_gravity_;
    rot_ = o.rot_; pos_ = o.pos_; vel_ = o.vel_;
    bias_gyr_ = o.bias_gyr_; bias_acc_ = o.bias_acc_; gravity_ = o.gravity_;
    cov_ = o.cov_;
    var_acc_ = o.var_acc_; var_gyr_ = o.var_gyr_;
    cov_bias_gyr_ = o.cov_bias_gyr_; cov_bias_acc_ = o.cov_bias_acc_;
    init_cov_rot_tilt_ = o.init_cov_rot_tilt_;
    init_cov_rot_yaw_ = o.init_cov_rot_yaw_;
    init_cov_pos_ = o.init_cov_pos_;
    init_cov_vel_ = o.init_cov_vel_;
    init_cov_gravity_ = o.init_cov_gravity_;
    init_cov_bg_ = o.init_cov_bg_;
    init_cov_ba_ = o.init_cov_ba_;
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
  // Gravity alignment observes tilt but cannot observe yaw. Since applyDelta
  // uses a right/body-frame perturbation, the tangent-space yaw axis is the
  // world-gravity direction expressed in the aligned body frame.
  cov_.block<3,3>(idxR(), idxR()) = gravityAlignedAttitudeCovariance(
      rot_, init_cov_rot_tilt_, init_cov_rot_yaw_);
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
  if (est_bg_)      bias_gyr_ += dx.segment<3>(idxBG());
  if (est_ba_)      bias_acc_ += dx.segment<3>(idxBA());
  if (est_gravity_) gravity_  += dx.segment<3>(idxG());
  updateDerivedTransforms();
}

Eigen::VectorXd StateGroup::boxminusFromPropagat(const StateGroup& propagat) const
{
  Eigen::VectorXd vec = Eigen::VectorXd::Zero(dimState());
  const M3D rotd = rot_.transpose() * propagat.rot_;
  vec.segment<3>(idxR()) = Log(rotd);
  vec.segment<3>(idxP()) = propagat.pos_ - pos_;
  vec.segment<3>(idxV()) = propagat.vel_ - vel_;
  if (est_bg_)      vec.segment<3>(idxBG()) = propagat.bias_gyr_ - bias_gyr_;
  if (est_ba_)      vec.segment<3>(idxBA()) = propagat.bias_acc_ - bias_acc_;
  if (est_gravity_) vec.segment<3>(idxG())  = propagat.gravity_ - gravity_;
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
  paramWarn<bool>(pnh, "state/est/bg",      est_bg_,      true);
  paramWarn<bool>(pnh, "state/est/ba",      est_ba_,      true);
  paramWarn<bool>(pnh, "state/est/gravity", est_gravity_, true);
  paramWarn<bool>(pnh, "state/est/cov_acc", est_cov_acc_, false);
  paramWarn<bool>(pnh, "state/est/cov_gyr", est_cov_gyr_, false);

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
  paramWarn<std::string>(pnh, "camera/distortion_model", camera_.distortion_model, "plumb_bob");

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

  // Initial state covariance
  paramWarn<double>(pnh, "state/cov/rot_tilt", init_cov_rot_tilt_, 1e-4);
  paramWarn<double>(pnh, "state/cov/rot_yaw", init_cov_rot_yaw_, 1e-4);
  paramWarn<double>(pnh, "state/cov/pos",     init_cov_pos_,     1e-4);
  paramWarn<double>(pnh, "state/cov/vel",     init_cov_vel_,     1e-4);
  paramWarn<double>(pnh, "state/cov/bg",      init_cov_bg_,      1e-6);
  paramWarn<double>(pnh, "state/cov/ba",      init_cov_ba_,      1e-6);
  paramWarn<double>(pnh, "state/cov/gravity", init_cov_gravity_, 1e-5);
  // Before gravity alignment there is not yet a meaningful yaw axis. Seed
  // the temporary isotropic block with the tilt value; setCalibResult()
  // replaces it with the exact aligned tilt/yaw covariance before LIO.
  initCov(init_cov_rot_tilt_, init_cov_pos_, init_cov_vel_, init_cov_bg_,
          init_cov_ba_, init_cov_gravity_);

  // Initial dynamic state
  std::vector<double> g_vec, v_vec, ba_vec, bg_vec;
  paramWarn<std::vector<double>>(pnh, "state/init_state/gravity", g_vec,  {0., 0., -G_m_s2});
  paramWarn<std::vector<double>>(pnh, "state/init_state/vel",     v_vec,  {0., 0., 0.});
  paramWarn<std::vector<double>>(pnh, "state/init_state/ba",      ba_vec, {0., 0., 0.});
  paramWarn<std::vector<double>>(pnh, "state/init_state/bg",      bg_vec, {0., 0., 0.});
  gravity_  = V3D(g_vec[0],  g_vec[1],  g_vec[2]);
  vel_      = V3D(v_vec[0],  v_vec[1],  v_vec[2]);
  bias_acc_ = V3D(ba_vec[0], ba_vec[1], ba_vec[2]);
  bias_gyr_ = V3D(bg_vec[0], bg_vec[1], bg_vec[2]);

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
  // q_alpha_bias*covBiasGyr()*dt term) -- two independent physical facts
  // that cannot correctly share one value. Measured: with the initial
  // value (1e-5 in config/ntu_viral.yaml) used as the rate, the licensed
  // 1-sigma gyro-bias walk over a 69.4s window is 543x a MEMS sensor's own
  // quoted in-run stability. Split into their own keys, each DEFAULTING TO
  // THE SAME VALUE the (still-present) initial-covariance key uses, so this
  // is provably md5-inert unless a config explicitly sets the new key --
  // rule 26 item 1, no new default chosen here.
  double bias_gyr_rw = init_cov_bg_, bias_acc_rw = init_cov_ba_;
  paramWarn<double>(pnh, "imu/process_noise/bias_random_walk/gyro_variance_rate",
                    bias_gyr_rw, init_cov_bg_);
  paramWarn<double>(pnh, "imu/process_noise/bias_random_walk/accel_variance_rate",
                    bias_acc_rw, init_cov_ba_);
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
      << "\n  distortion_model: " << camera_.distortion_model
      << "\n  est/bg=" << (est_bg_ ? "true" : "false")
      << "  est/ba=" << (est_ba_ ? "true" : "false")
      << "  est/gravity=" << (est_gravity_ ? "true" : "false")
      << "  est/cov_acc=" << (est_cov_acc_ ? "true" : "false")
      << "  est/cov_gyr=" << (est_cov_gyr_ ? "true" : "false")
      << "  dimState=" << dimState()
      << "\n  initial_attitude_cov/tilt=" << init_cov_rot_tilt_
      << "  initial_attitude_cov/yaw=" << init_cov_rot_yaw_;
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
