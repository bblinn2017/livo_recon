#include "livo_recon/processing/imu_processing.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/state/state.h"
#include "livo_recon/utils/log/debug_log_dir.h"

#include <fstream>
#include <iomanip>

namespace livo_recon
{

namespace
{

// Debug trace of IMU propagation's per-measure-group state and covariance,
// for diagnosing where trajectory divergence originates (see lio_processing
// .cpp's debugLogLio() for the LIO-side counterpart, and evo_processing
// .cpp's per-stage ATE/RTE/ROE/ARE for the downstream effect on accuracy).
// Truncated at the start of each process (first call), appended thereafter.
// Absolute (bag/wall-clock) timestamps throughout, matching evo_processing
// .cpp's /tmp/evo.txt and FAST-LIVO2's own logs, so all of these can be
// compared directly against each other. Remove once done debugging.
void debugLogImu(const std::string& msg)
{
  static bool first_call = true;
  std::ofstream ofs(debugLogPath("imu.txt"), first_call ? std::ios::trunc : std::ios::app);
  first_call = false;
  ofs << msg << "\n";
}

}  // namespace

ImuProc::ImuProc(NodeContext& ctx)
  : state_(ctx.state), profiler_(ctx.profiler), data_queues_(ctx.data_queues)
{}

std::string ImuProc::loadParameters(ros::NodeHandle& pnh)
{
  // sensor/range_err, sensor/angle_err_deg, ds/*, undistort/
  // time_based_process_noise moved to LioProc::loadParameters()
  // (2026-08-18) -- deskewing/downsampling now live there, see
  // LioProcOptions::deskew/ds_leaf_size/ds_mode's doc comments. Still the
  // SAME rosparam keys, just read by a different class now.
  paramWarn<bool>(pnh, "imu/second_order",   opts_.second_order, true);
  paramWarn<double>(pnh, "imu/q_alpha", opts_.q_alpha, 1.0);
  paramWarn<bool>(pnh, "imu/log_debug_en", opts_.log_debug_en, false);

  std::ostringstream oss;
  oss << "[params/imu]"
      << "\n  second_order:         " << (opts_.second_order ? "true" : "false")
      << "\n  q_alpha:              " << opts_.q_alpha;
  return oss.str();
}

void ImuProc::propagate(MeasureGroup& mg)
{
  if (mg.imu_samples.empty())
  {
    last_imu_sample_.acc  = -state_->rot().transpose() * state_->gravity();
    last_imu_sample_.gyro = V3D::Zero();
    return;
  }

  const double t_curr = mg.image.t;

  V3D acc_avr, angvel_avr, acc_avr_world;

  M3D acc_avr_skew, Exp_f;
  M3D acc_noise_world;

  const int dim = state_->dimState();
  Eigen::MatrixXd F_x(dim, dim);
  Eigen::MatrixXd cov_w(dim, dim);

  M3D rot_imu(state_->rot());
  V3D pos_imu(state_->pos()), vel_imu(state_->vel());

  const M3D var_acc_diag = state_->varAcc().asDiagonal();

  ImuSample head = last_imu_sample_;
  auto it = mg.imu_samples.begin();

  mg.poses.reserve(mg.imu_samples.size());
  for (; it != mg.imu_samples.end(); ++it)
  {
    ImuSample tail = *it;
    bool is_last = false;

    // ---- interpolate last sample at t_curr ----
    if (tail.t > t_curr)
    {
      double alpha = (t_curr - head.t) / (tail.t - head.t);

      tail.t = t_curr;
      tail.acc  = (1.0 - alpha) * head.acc  + alpha * tail.acc;
      tail.gyro = (1.0 - alpha) * head.gyro + alpha * tail.gyro;

      is_last = true;
    }

    double dt = tail.t - head.t;
    double dt2 = dt * dt;

    // ---- average measurements ----
    acc_avr    = 0.5 * (head.acc  + tail.acc) - state_->biasAcc();
    angvel_avr = 0.5 * (head.gyro + tail.gyro) - state_->biasGyr();

    // ---- save head state for pose storage ----
    const M3D rot_at_head = rot_imu;
    const V3D pos_at_head = pos_imu;
    const V3D vel_at_head = vel_imu;

    // ---- covariance propagation ----
    acc_avr_skew << SKEW_SYM_MATRX(acc_avr);
    acc_noise_world = rot_imu * var_acc_diag * rot_imu.transpose();
    Exp_f = Exp(angvel_avr, dt);

    F_x.setIdentity();
    cov_w.setZero();

    // Rotation
    F_x.block<3,3>(StateGroup::idxR(), StateGroup::idxR()) = Exp_f.transpose();
    if (state_->estBG())
      F_x.block(StateGroup::idxR(), state_->idxBG(), 3, 3) = -Eye3d * dt;

    // Position
    F_x.block<3,3>(StateGroup::idxP(), StateGroup::idxV()) = Eye3d * dt;
    if (opts_.second_order)
    {
      F_x.block<3,3>(StateGroup::idxP(), StateGroup::idxR()) += -0.5 * rot_imu * acc_avr_skew * dt2;
      if (state_->estGravity())
        F_x.block(StateGroup::idxP(), state_->idxG(), 3, 3) = 0.5 * Eye3d * dt2;
    }

    // Velocity
    F_x.block<3,3>(StateGroup::idxV(), StateGroup::idxR()) = -rot_imu * acc_avr_skew * dt;
    if (state_->estBA())
      F_x.block(StateGroup::idxV(), state_->idxBA(), 3, 3) = -rot_imu * dt;
    if (state_->estGravity())
      F_x.block(StateGroup::idxV(), state_->idxG(), 3, 3) = Eye3d * dt;

    // Rotation noise
    cov_w.block<3,3>(StateGroup::idxR(), StateGroup::idxR()).diagonal() = state_->varGyr() * dt2;

    // Velocity noise
    cov_w.block<3,3>(StateGroup::idxV(), StateGroup::idxV()) = acc_noise_world * dt2;

    // Bias Gyro random-walk noise
    if (state_->estBG())
      cov_w.block(state_->idxBG(), state_->idxBG(), 3, 3).diagonal() =
          (state_->covBiasGyr() * dt).eval();

    // Bias Acc random-walk noise
    if (state_->estBA())
      cov_w.block(state_->idxBA(), state_->idxBA(), 3, 3).diagonal() =
          (state_->covBiasAcc() * dt).eval();

    if (opts_.second_order)
    {
      // Position noise from dp = 0.5*R*n_a*dt²; cross-term with dv = R*n_a*dt
      cov_w.block<3,3>(StateGroup::idxP(), StateGroup::idxP()) = 0.25 * dt2 * dt2 * acc_noise_world;
      cov_w.block<3,3>(StateGroup::idxP(), StateGroup::idxV()) = 0.5  * dt  * dt2 * acc_noise_world;
      cov_w.block<3,3>(StateGroup::idxV(), StateGroup::idxP()) = 0.5  * dt  * dt2 * acc_noise_world;
    }

    // T0-E: q_alpha scales the WHOLE process-noise contribution uniformly
    // -- 1.0 (default) is bit-identical to the pre-T0-E formula.
    state_->covMut() = F_x * state_->cov() * F_x.transpose() + opts_.q_alpha * cov_w;

    // ---- state propagation ----
    // World-frame acc is the average of acc transformed by head and tail rotations
    // respectively, giving a better estimate than using either alone.
    const V3D acc_world_head = rot_imu * (head.acc - state_->biasAcc()) + state_->gravity();
    rot_imu = rot_imu * Exp_f;
    const V3D acc_world_tail = rot_imu * (tail.acc - state_->biasAcc()) + state_->gravity();
    acc_avr_world = 0.5 * (acc_world_head + acc_world_tail);

    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_avr_world * dt2;
    vel_imu = vel_imu + acc_avr_world * dt;

    // ---- store pose at head time with head state ----
    mg.poses.emplace_back(Pose6D{
        head.t,
        acc_world_head,
        acc_world_tail,
        angvel_avr,
        vel_at_head,
        pos_at_head,
        rot_at_head,
        dt
    });

    head = tail;

    if (is_last)
      break;
  }

  last_imu_sample_ = head;
  mg.imu_samples.clear();

  state_->setPropagatedState(rot_imu, pos_imu, vel_imu);

  if (opts_.log_debug_en)
  {
    const double t_abs = mg.image.t + data_queues_->start_time;
    const Eigen::Quaterniond q(state_->rot());
    const auto& P = state_->cov();
    const M3D P_RR = P.block<3, 3>(StateGroup::idxR(), StateGroup::idxR());
    const M3D P_PP = P.block<3, 3>(StateGroup::idxP(), StateGroup::idxP());
    const M3D P_VV = P.block<3, 3>(StateGroup::idxV(), StateGroup::idxV());
    const M3D P_PV = P.block<3, 3>(StateGroup::idxP(), StateGroup::idxV());

    std::ostringstream dbg;
    dbg << std::fixed << std::setprecision(6)
        << "[imu]  t_abs=" << t_abs
        << "  n_imu=" << mg.poses.size()
        << "  pos=[" << state_->pos().transpose() << "]"
        << "  vel=[" << state_->vel().transpose() << "]"
        << "  quat(wxyz)=[" << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << "]"
        << "  bias_gyr=[" << state_->biasGyr().transpose() << "]"
        << "  bias_acc=[" << state_->biasAcc().transpose() << "]"
        << "  gravity=[" << state_->gravity().transpose() << "]"
        << std::scientific
        << "  trace(P_RR)=" << P_RR.trace()
        << "  trace(P_PP)=" << P_PP.trace()
        << "  trace(P_VV)=" << P_VV.trace()
        << "  norm(P_PV)=" << P_PV.norm();
    if (state_->estBG())
      dbg << "  trace(P_BGBG)=" << P.block<3, 3>(state_->idxBG(), state_->idxBG()).trace();
    if (state_->estBA())
      dbg << "  trace(P_BABA)=" << P.block<3, 3>(state_->idxBA(), state_->idxBA()).trace();

    // Full post-IMU-propagation (pre-LIO-correction) state + covariance --
    // see lio_processing.cpp's matching post-LIO dump for why (raw-value
    // cross-system comparison, not just derived trace/norm scalars). This
    // is state_->cov() as it stood right after propagate()/undistortLidar(),
    // before any residual-based correction this measure group.
    dbg << "  dim=" << P.rows() << "  cov_flat=[";
    for (int r = 0; r < P.rows(); r++)
      for (int c = 0; c < P.cols(); c++)
        dbg << P(r, c) << (r == P.rows() - 1 && c == P.cols() - 1 ? "" : " ");
    dbg << "]";

    debugLogImu(dbg.str());
  }
}

void ImuProc::processIMU(MeasureGroup& mg)
{
  TimedScope ts(profiler_, "imu/propagate");
  propagate(mg);
}


}  // namespace livo_recon
