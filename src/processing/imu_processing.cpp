#include "livo_recon/processing/imu_processing.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/state/state.h"
#include "livo_recon/utils/log/debug_log_dir.h"

#include <fstream>
#include <iomanip>
#include <mutex>

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

// T7-a (2026-09-01): Myers-Tapley process-noise-estimator accumulators.
// See imu_processing.h's imuProcQhatRead() doc comment for the recursion
// this implements and why it's a recursion, not a sum, across substeps.
bool g_qhat_enabled = false;
bool g_qhat_primed  = false;
Eigen::MatrixXd g_qhat_accum_cov_w;   // A <- F A F^T + cov_w, over one frame
Eigen::MatrixXd g_qhat_p_after;       // P after the frame's last propagation
std::mutex g_qhat_mtx;

}  // namespace

bool imuProcQhatRead(Eigen::MatrixXd& phi_p_phit, Eigen::MatrixXd& accum_cov_w)
{
  std::lock_guard<std::mutex> lock(g_qhat_mtx);
  if (!g_qhat_enabled || !g_qhat_primed) return false;
  accum_cov_w = g_qhat_accum_cov_w;
  // Phi P+_{k-1} Phi^T = (propagated P) - (accumulated process noise)
  phi_p_phit = g_qhat_p_after - g_qhat_accum_cov_w;
  g_qhat_accum_cov_w.setZero();
  g_qhat_primed = false;
  return true;
}

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
  paramWarn<double>(pnh, "imu/q_alpha_acc", opts_.q_alpha_acc, 1.0);
  paramWarn<double>(pnh, "imu/q_alpha_gyr", opts_.q_alpha_gyr, 1.0);
  paramWarn<double>(pnh, "imu/q_alpha_bias", opts_.q_alpha_bias, 1.0);
  paramWarn<bool>(pnh, "imu/log_debug_en", opts_.log_debug_en, false);
  paramWarn<bool>(pnh, "imu/log_qhat_en", opts_.log_qhat_en, false);
  // Read directly rather than being wired from LioProcOptions: the two
  // classes are constructed independently and this keeps the single source
  // of truth in the parameter server, where the sweep harness writes it.
  // The copy is skipped entirely when the spline is off.
  paramWarn<bool>(pnh, "spline/enable", opts_.keep_raw_samples, false);
  { std::lock_guard<std::mutex> lock(g_qhat_mtx); g_qhat_enabled = opts_.log_qhat_en; }

  std::ostringstream oss;
  oss << "[params/imu]"
      << "\n  second_order:         " << (opts_.second_order ? "true" : "false")
      << "\n  q_alpha_acc:          " << opts_.q_alpha_acc
      << "\n  q_alpha_gyr:          " << opts_.q_alpha_gyr
      << "\n  q_alpha_bias:         " << opts_.q_alpha_bias
      << "\n  log_qhat_en:          " << (opts_.log_qhat_en ? "true" : "false")
      << "\n  keep_raw_samples:     " << (opts_.keep_raw_samples ? "true" : "false");
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

  // Snapshot the raw stream BEFORE the loop consumes it -- the spline-vs-
  // IMU residual (lio/spline.h) needs the unaveraged, un-bias-corrected
  // samples, and mg.imu_samples is cleared at the end of this function.
  // The loop below also interpolates the final sample onto t_curr; that
  // interpolated tail is appended after the loop so the residual covers
  // the whole scan window rather than stopping at the last raw sample.
  if (opts_.keep_raw_samples)
  {
    mg.imu_samples_raw.clear();
    mg.imu_samples_raw.reserve(mg.imu_samples.size() + 1);
    for (const auto& s : mg.imu_samples)
      if (s.t <= t_curr) mg.imu_samples_raw.push_back(s);
  }

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
    cov_w.block<3,3>(StateGroup::idxR(), StateGroup::idxR()).diagonal() =
        opts_.q_alpha_gyr * state_->varGyr() * dt2;

    // Velocity noise
    cov_w.block<3,3>(StateGroup::idxV(), StateGroup::idxV()) =
        opts_.q_alpha_acc * acc_noise_world * dt2;

    // Bias Gyro random-walk noise
    if (state_->estBG())
      cov_w.block(state_->idxBG(), state_->idxBG(), 3, 3).diagonal() =
          (opts_.q_alpha_bias * state_->covBiasGyr() * dt).eval();

    // Bias Acc random-walk noise
    if (state_->estBA())
      cov_w.block(state_->idxBA(), state_->idxBA(), 3, 3).diagonal() =
          (opts_.q_alpha_bias * state_->covBiasAcc() * dt).eval();

    if (opts_.second_order)
    {
      // Position noise from dp = 0.5*R*n_a*dt²; cross-term with dv = R*n_a*dt
      // -- both derived from the same accel noise, so scaled by q_alpha_acc.
      cov_w.block<3,3>(StateGroup::idxP(), StateGroup::idxP()) = opts_.q_alpha_acc * 0.25 * dt2 * dt2 * acc_noise_world;
      cov_w.block<3,3>(StateGroup::idxP(), StateGroup::idxV()) = opts_.q_alpha_acc * 0.5  * dt  * dt2 * acc_noise_world;
      cov_w.block<3,3>(StateGroup::idxV(), StateGroup::idxP()) = opts_.q_alpha_acc * 0.5  * dt  * dt2 * acc_noise_world;
    }

    // T0-E/T7 step 2: each block above is already pre-scaled by its own
    // q_alpha_{acc,gyr,bias} -- cov_w is added unscaled here. All three
    // at 1.0 (default) is bit-identical to the pre-split formula.
    state_->covMut() = F_x * state_->cov() * F_x.transpose() + cov_w;

    if (opts_.log_qhat_en) {
      std::lock_guard<std::mutex> lock(g_qhat_mtx);
      if (g_qhat_accum_cov_w.rows() != dim) {
        g_qhat_accum_cov_w = Eigen::MatrixXd::Zero(dim, dim);
      }
      g_qhat_accum_cov_w = F_x * g_qhat_accum_cov_w * F_x.transpose() + cov_w;
      g_qhat_p_after = state_->cov();
      g_qhat_primed = true;
    }

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

  if (opts_.keep_raw_samples)
  {
    // `head` is the loop's final tail, already interpolated onto t_curr on
    // the is_last path.  Append it only if it actually extends the window.
    if (mg.imu_samples_raw.empty() || head.t > mg.imu_samples_raw.back().t + 1e-12)
      mg.imu_samples_raw.push_back(head);
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
