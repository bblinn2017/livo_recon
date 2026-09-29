#include "livo_recon/processing/imu_processing.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/state/state.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"
#include "livo_recon/diagnostics/motion_q_writer.h"
#include "livo_recon/diagnostics/state_trace.h"

#include <fstream>
#include <iomanip>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <stdexcept>

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
// CQ-36: PersistentLogStream -- see its own doc comment for the CQ-35
// regression this fixes (a bare static ofstream froze onto the first-ever
// resolved path and never flushed).
void debugLogImu(const std::string& msg)
{
  static PersistentLogStream log("imu.txt");
  std::ofstream& ofs = log.stream();
  ofs << msg << "\n";
  ofs.flush();
}

// R62: one row per IMU step of the first three propagations, so the first
// interval's head sample, bias, gravity and world acceleration are visible.
void writeImuFirstScanRow(size_t propagation_index, size_t step, double t_off,
                          const ImuSample& head, const ImuSample& tail, double dt,
                          const V3D& ba, const V3D& bg, const V3D& g,
                          const V3D& dynamic_acc, const V3D& acc_world_head,
                          const V3D& acc_world_tail, const V3D& acc_avr_world,
                          const V3D& vel_before, const V3D& vel_after,
                          const V3D& pos_after)
{
  static PersistentLogStream log("imu_first_scans.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first)
    out << "propagation_index,step,head_t_abs,tail_t_abs,dt,"
           "head_acc_x,head_acc_y,head_acc_z,tail_acc_x,tail_acc_y,tail_acc_z,"
           "head_gyro_x,head_gyro_y,head_gyro_z,tail_gyro_x,tail_gyro_y,tail_gyro_z,"
           "ba_x,ba_y,ba_z,bg_x,bg_y,bg_z,g_x,g_y,g_z,"
           "dynamic_acc_x,dynamic_acc_y,dynamic_acc_z,"
           "acc_world_head_x,acc_world_head_y,acc_world_head_z,"
           "acc_world_tail_x,acc_world_tail_y,acc_world_tail_z,"
           "acc_avr_world_x,acc_avr_world_y,acc_avr_world_z,"
           "vel_before_x,vel_before_y,vel_before_z,vel_after_x,vel_after_y,vel_after_z,"
           "pos_after_x,pos_after_y,pos_after_z\n";
  out << std::setprecision(17) << propagation_index << ',' << step << ','
      << head.t + t_off << ',' << tail.t + t_off << ',' << dt;
  for (const V3D* v : {&head.acc, &tail.acc, &head.gyro, &tail.gyro, &ba, &bg, &g,
                       &dynamic_acc, &acc_world_head, &acc_world_tail, &acc_avr_world,
                       &vel_before, &vel_after, &pos_after})
    out << ',' << (*v)(0) << ',' << (*v)(1) << ',' << (*v)(2);
  out << '\n';
  out.flush();
}

// History (32-34): see docs/livo_recon_changelog.md#src-processing-imu_processing.cpp-32
bool g_qhat_enabled = false;
bool g_qhat_primed  = false;
Eigen::MatrixXd g_qhat_accum_cov_w;   // A <- F A F^T + cov_w, over one frame
Eigen::MatrixXd g_qhat_p_after;       // P after the frame's last propagation
// CQ-71 item 0: P BEFORE this frame's propagation begins -- the fourth
// leg of the information budget (P_before/P_after_IMU/FPF^T/Q_eff were
// three-quarters already sitting in this file's existing qhat machinery;
// only the starting snapshot was missing). Captured once per frame, at
// the top of propagate(), under the SAME opts_.log_qhat_en gate as the
// rest of this mechanism -- no new flag, since it's the same measurement.
Eigen::MatrixXd g_qhat_p_before;
std::mutex g_qhat_mtx;

}  // namespace

bool imuProcQhatRead(Eigen::MatrixXd& phi_p_phit, Eigen::MatrixXd& accum_cov_w,
                      Eigen::MatrixXd& p_before)
{
  std::lock_guard<std::mutex> lock(g_qhat_mtx);
  if (!g_qhat_enabled || !g_qhat_primed) return false;
  accum_cov_w = g_qhat_accum_cov_w;
  // Phi P+_{k-1} Phi^T = (propagated P) - (accumulated process noise)
  phi_p_phit = g_qhat_p_after - g_qhat_accum_cov_w;
  p_before = g_qhat_p_before;
  g_qhat_accum_cov_w.setZero();
  g_qhat_primed = false;
  return true;
}

bool imuProcQhatPeekPBefore(Eigen::MatrixXd& p_before)
{
  std::lock_guard<std::mutex> lock(g_qhat_mtx);
  if (!g_qhat_enabled || !g_qhat_primed) return false;
  p_before = g_qhat_p_before;
  return true;
}

bool imuProcQhatPeekAll(Eigen::MatrixXd& p_before, Eigen::MatrixXd& phi_p_phit,
                        Eigen::MatrixXd& q_eff, Eigen::MatrixXd& p_after)
{
  std::lock_guard<std::mutex> lock(g_qhat_mtx);
  if (!g_qhat_enabled || !g_qhat_primed) return false;
  p_before = g_qhat_p_before;
  q_eff = g_qhat_accum_cov_w;
  p_after = g_qhat_p_after;
  phi_p_phit = p_after - q_eff;
  return true;
}

ImuProc::ImuProc(NodeContext& ctx)
  : state_(ctx.state), profiler_(ctx.profiler), data_queues_(ctx.data_queues)
{}

std::string ImuProc::loadParameters(ros::NodeHandle& pnh)
{
  // History (61-65): see docs/livo_recon_changelog.md#src-processing-imu_processing.cpp-61
  paramWarn<bool>(pnh, "imu/second_order",   opts_.second_order, true);
  paramWarn<double>(pnh, "imu/q_alpha_acc", opts_.q_alpha_acc, 1.0);
  paramWarn<double>(pnh, "imu/q_alpha_gyr", opts_.q_alpha_gyr, 1.0);
  paramWarn<double>(pnh, "imu/q_alpha_bias", opts_.q_alpha_bias, 1.0);
  paramWarn<bool>(pnh, "imu/log_debug_en", opts_.log_debug_en, false);
  paramWarn<bool>(pnh, "imu/log_qhat_en", opts_.log_qhat_en, false);
  paramWarn<std::string>(pnh, "imu/process_noise/model",
                         opts_.process_noise_model, std::string("fixed"));
  paramWarn<double>(pnh, "imu/process_noise/motion/beta", opts_.motion_beta, 0.0);
  paramWarn<double>(pnh, "imu/process_noise/motion/acc_scale",
                    opts_.motion_acc_scale, 1.0);
  paramWarn<double>(pnh, "imu/process_noise/motion/gyro_scale",
                    opts_.motion_gyr_scale, 1.0);
  paramWarn<double>(pnh, "imu/process_noise/motion/acc_floor_scale", opts_.motion_acc_floor_scale, 1.0);
  paramWarn<double>(pnh, "imu/process_noise/motion/gyro_floor_scale", opts_.motion_gyr_floor_scale, 1.0);
  paramWarn<double>(pnh, "imu/process_noise/motion/acc_max_dynamic_variance",
                    opts_.motion_acc_max_dynamic_variance, 0.5);
  paramWarn<double>(pnh, "imu/process_noise/motion/gyro_max_dynamic_variance",
                    opts_.motion_gyr_max_dynamic_variance, 0.3);
  {
    // R63 (F-118): the zero-initialised head sample is removed. The key is
    // no longer an option; a config that still sets it is refused rather
    // than silently ignored.
    if (pnh.hasParam("imu/first_scan_head"))
      throw std::invalid_argument(
          "imu/first_scan_head was removed in R63: the first interval head "
          "is always seeded from the first IMU sample");
  }
  paramWarn<bool>(pnh, "eval/imu_first_scans_en", opts_.first_scans_log_en, false);
  if (opts_.process_noise_model != "fixed" &&
      opts_.process_noise_model != "isotropic" &&
      opts_.process_noise_model != "axis_aware")
    throw std::invalid_argument(
        "imu/process_noise/model must be fixed, isotropic, or axis_aware");
  if (!std::isfinite(opts_.motion_beta) || opts_.motion_beta < 0.0 ||
      opts_.motion_beta >= 1.0)
    throw std::invalid_argument("imu/process_noise/motion/beta must be in [0,1)");
  for (double v : {opts_.motion_acc_floor_scale, opts_.motion_gyr_floor_scale})
    if (!std::isfinite(v) || v <= 0.0)
      throw std::invalid_argument("imu/process_noise/motion/{acc,gyro}_floor_scale must be finite and > 0");
  if (opts_.process_noise_model == "fixed" &&
      (opts_.motion_acc_floor_scale != 1.0 || opts_.motion_gyr_floor_scale != 1.0))
    throw std::invalid_argument(
        "imu/process_noise/motion/*_floor_scale != 1 has no effect under process_noise/model=fixed (no floor term)");
  for (double v : {opts_.motion_acc_scale, opts_.motion_gyr_scale,
                   opts_.motion_acc_max_dynamic_variance,
                   opts_.motion_gyr_max_dynamic_variance})
    if (!std::isfinite(v) || v < 0.0)
      throw std::invalid_argument("IMU motion-noise scales/caps must be finite and nonnegative");
  // Read directly rather than being wired from LioProcOptions: the two
  // classes are constructed independently and this keeps the single source
  // of truth in the parameter server, where the sweep harness writes it.
  // The copy is skipped entirely when the spline is off.
  //
  // BUG FIX (found 2026-09-15, during a TQ-14 item-8 investigation into why
  // q_active read exactly 0.0 on every frame across the entire TQ-12/TQ-6/
  // TQ-7 dataset): this used to read the boolean "spline/enable" key, which
  // the spline restructure DELETED -- replaced by the string "spline/mode".
  // Since "spline/enable" is never set on the param server anymore, this
  // paramWarn<bool> silently fell back to its default (false) on every
  // single run, so imu_samples_raw was NEVER retained regardless of
  // spline/mode -- finalizeSplineAndQ()'s `if (!mg.imu_samples_raw.empty())`
  // guard was always false, last_spline_stats_ stayed default-constructed
  // (n=0), SplineImuResidualStats::valid() (n>=8) was always false, and
  // AdaptiveQ::update() hit its very first check ("no_residual") on every
  // call, before warmup/whiteness/floor were ever evaluated -- active_
  // never became true anywhere, so state_->setNoiseParams() (gated on
  // adaptive_q_.active()) was NEVER called, meaning adaptive_q/enable=true
  // had ZERO effect on any run's actual trajectory: it silently ran
  // identically to adaptive_q/enable=false. Fixed by reading the current
  // "spline/mode" string directly (mirroring LioProcOptions::splineOn()'s
  // own "mode != raw_imu" definition) instead of the deleted boolean key.
  std::string spline_mode = "raw_imu";
  paramWarn<std::string>(pnh, "spline/mode", spline_mode, std::string("raw_imu"));
  // The joint-knot coupled estimator consumes the propagation record in
  // mg.poses and has no private raw-IMU/coefficient mode.  Raw samples are
  // therefore retained only for the decoupled spline/Adaptive-Q pipeline.
  opts_.keep_raw_samples = (spline_mode != "raw_imu");
  { std::lock_guard<std::mutex> lock(g_qhat_mtx); g_qhat_enabled = opts_.log_qhat_en; }

  std::ostringstream oss;
  oss << "[params/imu]"
      << "\n  second_order:         " << (opts_.second_order ? "true" : "false")
      << "\n  q_alpha_acc:          " << opts_.q_alpha_acc
      << "\n  q_alpha_gyr:          " << opts_.q_alpha_gyr
      << "\n  q_alpha_bias:         " << opts_.q_alpha_bias
      << "\n  process_noise/model:  " << opts_.process_noise_model
      << "\n  motion/beta:          " << opts_.motion_beta
      << "\n  motion/acc_scale:     " << opts_.motion_acc_scale
      << "\n  motion/gyro_scale:    " << opts_.motion_gyr_scale
      << "\n  motion/acc_floor_scale: " << opts_.motion_acc_floor_scale
      << "\n  motion/gyro_floor_scale:" << opts_.motion_gyr_floor_scale
      << "\n  motion/acc_cap:       " << opts_.motion_acc_max_dynamic_variance
      << "\n  motion/gyro_cap:      " << opts_.motion_gyr_max_dynamic_variance
      << "\n  first_scan_head:      always seeded from first sample (R63)"
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

  // R63: fixed accelerometer scale (calib/stationary/accel_excess_model ==
  // scale). Applied once to this group's raw samples so the head/tail
  // bookkeeping, the raw-sample snapshot and every consumer downstream see
  // the same scaled stream. 1.0 (bit-identical, loop skipped) otherwise.
  if (state_->accScale() != 1.0)
  {
    for (auto& smp : mg.imu_samples) smp.acc *= state_->accScale();
    acc_scale_applied_samples_ += static_cast<long>(mg.imu_samples.size());
    if (!acc_scale_logged_)
    {
      acc_scale_logged_ = true;
      ROS_INFO_STREAM("[imu] accel scale " << state_->accScale()
                      << " applied to the IMU stream (engagement counter starts)");
    }
  }

  if (state_->gravityS2() && !s2_logged_)
  {
    s2_logged_ = true;
    ROS_INFO_STREAM("[imu] gravity_model=s2 active: state dim " << state_->dimState()
                    << ", gravity Jacobian columns " << state_->gravDim()
                    << ", |g|=" << state_->gravity().norm());
  }

  // CQ-71 item 0: snapshot P BEFORE this frame's propagation touches it.
  // Same gate as the rest of the qhat machinery below; a plain read, never
  // written back, so it cannot perturb state_->covMut() itself.
  if (opts_.log_qhat_en) {
    std::lock_guard<std::mutex> lock(g_qhat_mtx);
    g_qhat_p_before = state_->cov();
  }

  V3D acc_avr, angvel_avr, acc_avr_world;

  M3D acc_avr_skew, Exp_f;
  M3D acc_noise_world;

  const int dim = state_->dimState();
  Eigen::MatrixXd F_x(dim, dim);
  Eigen::MatrixXd cov_w(dim, dim);

  M3D rot_imu(state_->rot());
  V3D pos_imu(state_->pos()), vel_imu(state_->vel());

  ++propagation_index_;
  const bool dynamic_q = opts_.process_noise_model != "fixed";
  size_t q_samples = 0, acc_clamped = 0, gyr_clamped = 0;
  V3D q_acc_sum = V3D::Zero(), q_gyr_sum = V3D::Zero();
  V3D debiased_acc_energy_sum = V3D::Zero();
  V3D debiased_gyr_energy_sum = V3D::Zero();
  double acc_motion_sum = 0.0, gyr_motion_sum = 0.0;

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

  // R63 (F-118): the very first interval's head sample is the first IMU
  // sample of the first non-empty measure group. (Before R63 the head was a
  // default-constructed zero sample, which halved gravity for one step.)
  if (!head_seeded_)
  {
    head_seeded_ = true;
    last_imu_sample_.acc  = mg.imu_samples.front().acc;
    last_imu_sample_.gyro = mg.imu_samples.front().gyro;
    ++first_head_seed_count_;
    ROS_INFO_STREAM("[imu] first interval head seeded from first sample (engagement "
                    << first_head_seed_count_ << "): head acc=["
                    << last_imu_sample_.acc.transpose() << "]");
  }
  ImuSample head = last_imu_sample_;
  auto it = mg.imu_samples.begin();

  mg.poses.reserve(mg.imu_samples.size());
  mg.pose_covariances.clear();
  mg.imu_state_transitions.clear();
  mg.imu_process_covariances.clear();
  mg.pose_covariances.reserve(mg.imu_samples.size());
  mg.imu_state_transitions.reserve(mg.imu_samples.size());
  mg.imu_process_covariances.reserve(mg.imu_samples.size());
  // R61: exact split of the scan's covariance growth: P_end = Phi P_start Phi^T + Qacc, Qacc <- F Qacc F^T + Q per step.
  const Eigen::MatrixXd r61_P_start = state_->cov();
  Eigen::MatrixXd r61_Q_acc = Eigen::MatrixXd::Zero(dim, dim);
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

    // At rest acc_avr == -R^T g, hence this is the gravity-removed dynamic
    // specific force in the IMU/body frame. Angular rate is already
    // bias-corrected above.
    const V3D dynamic_acc = acc_avr + rot_imu.transpose() * state_->gravity();
    const V3D dynamic_gyr = angvel_avr;
    V3D var_acc_step = state_->varAcc();
    V3D var_gyr_step = state_->varGyr();
    V3D acc_dynamic_variance = V3D::Zero();
    V3D gyr_dynamic_variance = V3D::Zero();
    V3D debiased_acc_energy = V3D::Zero();
    V3D debiased_gyr_energy = V3D::Zero();
    if (dynamic_q)
    {
      if ((state_->varAccFloor().array() <= 0.0).any() ||
          (state_->varGyrFloor().array() <= 0.0).any())
        throw std::runtime_error(
            "motion-dependent Q requires positive stationary calibration floors");
      if (opts_.process_noise_model == "isotropic")
      {
        const V3D acc_energy = V3D::Constant(dynamic_acc.squaredNorm() / 3.0);
        const V3D gyr_energy = V3D::Constant(dynamic_gyr.squaredNorm() / 3.0);
        if (!motion_noise_primed_)
        {
          filtered_acc_excitation_energy_ = acc_energy;
          filtered_gyr_excitation_energy_ = gyr_energy;
        }
        else
        {
          filtered_acc_excitation_energy_ = opts_.motion_beta *
              filtered_acc_excitation_energy_ + (1.0 - opts_.motion_beta) * acc_energy;
          filtered_gyr_excitation_energy_ = opts_.motion_beta *
              filtered_gyr_excitation_energy_ + (1.0 - opts_.motion_beta) * gyr_energy;
        }
        // The isotropic signal has one shared energy, so subtract the mean
        // calibrated per-axis floor. This removes stationary sensor power
        // before the scale is applied instead of counting it twice.
        debiased_acc_energy.setConstant(std::max(
            0.0, filtered_acc_excitation_energy_.x() - state_->varAccFloor().mean()));
        debiased_gyr_energy.setConstant(std::max(
            0.0, filtered_gyr_excitation_energy_.x() - state_->varGyrFloor().mean()));
      }
      else
      {
        const V3D acc_energy = dynamic_acc.array().square().matrix();
        const V3D gyr_energy = dynamic_gyr.array().square().matrix();
        if (!motion_noise_primed_)
        {
          filtered_acc_excitation_energy_ = acc_energy;
          filtered_gyr_excitation_energy_ = gyr_energy;
        }
        else
        {
          filtered_acc_excitation_energy_ = opts_.motion_beta *
              filtered_acc_excitation_energy_ + (1.0 - opts_.motion_beta) * acc_energy;
          filtered_gyr_excitation_energy_ = opts_.motion_beta *
              filtered_gyr_excitation_energy_ + (1.0 - opts_.motion_beta) * gyr_energy;
        }
        debiased_acc_energy = (filtered_acc_excitation_energy_ -
            state_->varAccFloor()).cwiseMax(0.0);
        debiased_gyr_energy = (filtered_gyr_excitation_energy_ -
            state_->varGyrFloor()).cwiseMax(0.0);
      }
      motion_noise_primed_ = true;
      acc_dynamic_variance = opts_.motion_acc_scale * opts_.motion_acc_scale *
          debiased_acc_energy;
      gyr_dynamic_variance = opts_.motion_gyr_scale * opts_.motion_gyr_scale *
          debiased_gyr_energy;
      for (int axis = 0; axis < 3; ++axis)
      {
        if (acc_dynamic_variance(axis) > opts_.motion_acc_max_dynamic_variance)
          ++acc_clamped;
        if (gyr_dynamic_variance(axis) > opts_.motion_gyr_max_dynamic_variance)
          ++gyr_clamped;
        acc_dynamic_variance(axis) = std::min(
            acc_dynamic_variance(axis), opts_.motion_acc_max_dynamic_variance);
        gyr_dynamic_variance(axis) = std::min(
            gyr_dynamic_variance(axis), opts_.motion_gyr_max_dynamic_variance);
      }
      var_acc_step = state_->varAccFloor() * opts_.motion_acc_floor_scale + acc_dynamic_variance;
      var_gyr_step = state_->varGyrFloor() * opts_.motion_gyr_floor_scale + gyr_dynamic_variance;
    }
    ++q_samples;
    q_acc_sum += var_acc_step;
    q_gyr_sum += var_gyr_step;
    debiased_acc_energy_sum += debiased_acc_energy;
    debiased_gyr_energy_sum += debiased_gyr_energy;
    acc_motion_sum += dynamic_acc.norm();
    gyr_motion_sum += dynamic_gyr.norm();
    writeMotionQSampleDiagnostic(
        propagation_index_, q_samples - 1, tail.t + data_queues_->start_time,
        opts_.process_noise_model, dynamic_acc, dynamic_gyr,
        debiased_acc_energy, debiased_gyr_energy, var_acc_step, var_gyr_step);

    // ---- save head state for pose storage ----
    const M3D rot_at_head = rot_imu;
    const V3D pos_at_head = pos_imu;
    const V3D vel_at_head = vel_imu;
    const Eigen::MatrixXd cov_at_head = state_->cov();

    // ---- covariance propagation ----
    acc_avr_skew << SKEW_SYM_MATRX(acc_avr);
    acc_noise_world = rot_imu * var_acc_step.asDiagonal() * rot_imu.transpose();
    Exp_f = Exp(angvel_avr, dt);

    F_x.setIdentity();
    cov_w.setZero();
    // R63: d(g)/d(gravity error coordinates); I3 for vector3 (bit-identical
    // to the pre-R63 blocks), -[g]x B for s2. g and B do not change during
    // propagation, so this is constant over the whole call.
    const Eigen::MatrixXd J_g = state_->estGravity() ? Eigen::MatrixXd(state_->gravityJacobian())
                                                     : Eigen::MatrixXd(Eigen::MatrixXd::Zero(3, 0));

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
        F_x.block(StateGroup::idxP(), state_->idxG(), 3, state_->gravDim()) = 0.5 * J_g * dt2;
    }

    // Velocity
    F_x.block<3,3>(StateGroup::idxV(), StateGroup::idxR()) = -rot_imu * acc_avr_skew * dt;
    if (state_->estBA())
      F_x.block(StateGroup::idxV(), state_->idxBA(), 3, 3) = -rot_imu * dt;
    if (state_->estGravity())
      F_x.block(StateGroup::idxV(), state_->idxG(), 3, state_->gravDim()) = J_g * dt;

    // Rotation noise
    cov_w.block<3,3>(StateGroup::idxR(), StateGroup::idxR()).diagonal() =
        opts_.q_alpha_gyr * var_gyr_step * dt2;

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
    r61_Q_acc = F_x * r61_Q_acc * F_x.transpose() + cov_w;

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
    if (opts_.first_scans_log_en && propagation_index_ <= 3)
      writeImuFirstScanRow(propagation_index_, q_samples - 1, data_queues_->start_time,
                           head, tail, dt, state_->biasAcc(), state_->biasGyr(),
                           state_->gravity(), dynamic_acc, acc_world_head,
                           acc_world_tail, acc_avr_world, vel_at_head, vel_imu,
                           pos_imu);

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
    mg.pose_covariances.push_back(cov_at_head);
    mg.imu_state_transitions.push_back(F_x);
    mg.imu_process_covariances.push_back(cov_w);

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
  mg.n_imu_samples = static_cast<int>(mg.imu_samples.size());
  mg.imu_samples.clear();

  state_->setPropagatedState(rot_imu, pos_imu, vel_imu);
  writeImuCovGrowthRow(propagation_index_, mg.image.t + data_queues_->start_time, r61_P_start, state_->cov(), r61_Q_acc);

  // One compact row per scan; the first post-calibration scan is also logged
  // sample-by-sample above. Both are intentionally small enough for campaign
  // returns.
  const double inv_n = q_samples ? 1.0 / static_cast<double>(q_samples) : 0.0;
  writeMotionQScanDiagnostic(
      propagation_index_, mg.image.t + data_queues_->start_time,
      opts_.process_noise_model, opts_.motion_beta,
      opts_.motion_acc_scale, opts_.motion_gyr_scale, q_samples,
      acc_motion_sum * inv_n, gyr_motion_sum * inv_n,
      q_acc_sum * inv_n, q_gyr_sum * inv_n,
      debiased_acc_energy_sum * inv_n, debiased_gyr_energy_sum * inv_n,
      acc_clamped, gyr_clamped);

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
