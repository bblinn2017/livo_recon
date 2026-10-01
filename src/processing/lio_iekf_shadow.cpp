// R73 (F-414, F-415): shadow LIO update, per-scan update records and scan_log.csv for the iekf estimator. Everything here is
// write-only with respect to the driver run; the shadow pass saves and restores every member the shared residual machinery touches.
#include "livo_recon/processing/lio_iekf.h"

#include "livo_recon/diagnostics/log/debug_log_dir.h"
#include "livo_recon/map/voxelmap.h"
#include "livo_recon/utils/algo/math.h"
#include "livo_recon/utils/algo/symmetrize.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace livo_recon
{
namespace
{
std::string joinD(const std::vector<double>& v)
{
  std::ostringstream o;
  o << std::setprecision(10);
  for (size_t i = 0; i < v.size(); ++i) o << (i ? ";" : "") << v[i];
  return o.str();
}
std::string joinI(const std::vector<int>& v)
{
  std::ostringstream o;
  for (size_t i = 0; i < v.size(); ++i) o << (i ? ";" : "") << v[i];
  return o.str();
}
std::string hex64(unsigned long long v)
{
  std::ostringstream o;
  o << std::hex << std::setw(16) << std::setfill('0') << v;
  return o.str();
}
double blockTrace(const Eigen::MatrixXd& P, int idx)
{
  if (P.rows() < idx + 3 || P.cols() < idx + 3) return std::nan("");
  return P.block<3, 3>(idx, idx).trace();
}
}  // namespace

void LioProcIekf::recordIteration(UpdateRecord& rec, int iter, const V3D& dtheta, const V3D& dt) const
{
  rec.step_rot.push_back(residuals_.empty() ? 0.0 : dtheta.norm());   // no residuals: no solve, dtheta/dt are stale
  rec.step_pos.push_back(residuals_.empty() ? 0.0 : dt.norm());
  rec.n_resid_iter.push_back(static_cast<int>(residuals_.size()));
  rec.n_iter = iter + 1;
  if (residuals_.empty()) return;
  const Eigen::Matrix<double, 6, 6> H = ekf_.HtH;
  const Eigen::Matrix<double, 6, 1> z = ekf_.Htz;
  rec.HtHlast = H;
  rec.Htzlast = z;
  if (iter == 0)
  {
    double s = 0.0;
    for (const auto& r : residuals_) s += (r.r * r.r) / r.sigma_squared;
    rec.n_resid0 = static_cast<int>(residuals_.size());
    rec.chi2_iter0 = s / static_cast<double>(residuals_.size());
    rec.HtH0 = H;
    rec.Htz0 = z;
  }
}

// The shadow pass. See include/livo_recon/lio/shadow_options.h for the semantics. It runs BEFORE the driver's own update of the same
// scan, against the same (read-only) voxel map, starting from the shadow state that ShadowLio::propagateFromDriver rolled back to the
// driver's previous posterior and propagated with the shadow's own process noise.
void LioProcIekf::runShadowUpdate(const MeasureGroup& mg)
{
  shadow_rec_.reset();
  auto* vm = dynamic_cast<VoxelMap*>(voxel_map_.get());
  shadow_digest_before_ = vm ? vm->mapDigestExact() : 0ULL;

  // Save every member the shared machinery writes (the numerics of the driver's own pass must see none of this).
  StateGroupPtr real_state = state_;
  const Eigen::MatrixXd real_prior_cov = prior_cov_;
  const StateGroup real_state_propagat = state_propagat_;
  const std::vector<Residual> real_residuals = residuals_;
  const EkfUpdate real_ekf = ekf_;
  const ResidualRedundancyStats real_redundancy = redundancy_stats_;
  const long sc_it = step_cap_iterations_, sc_bound = step_cap_bound_;
  const double sc_sum = step_cap_scale_sum_, sc_min = step_cap_scale_min_, sc_rot = step_cap_rot_raw_max_, sc_pos = step_cap_pos_raw_max_;
  const long is_it = info_scale_iterations_, is_disc = info_scale_discarded_modes_;
  const double is_tr = info_scale_trace_ratio_sum_;
  const long rn_it = renorm_iterations_, rn_ap = renorm_applied_;
  const double rn_sum = renorm_c_sum_, rn_max = renorm_c_max_;
  const int n_mc = n_miss_coverage_, n_mm = n_miss_mismatch_, n_gc = n_statistical_gate_candidates_, n_gr = n_statistical_gate_rejections_;
  const int n_t0c = n_tier0_miss_coverage_, n_t0m = n_tier0_miss_mismatch_;
  const int real_diag_iter = diagnostic_gn_iteration_;

  shadow_active_ = true;
  state_ = shadow_->state();
  setDiagnosticGnIteration(-1);   // every iteration-gated diagnostic in the shared machinery stays off

  shadow_rec_.valid = true;
  shadow_rec_.start_pos = mg.pos_before_imu;
  shadow_rec_.start_rv = Log(mg.rot_before_imu);
  prior_cov_ = state_->cov();
  applyPriorScalarControls(prior_cov_, opts_.prior_scalar);
  state_propagat_ = *state_;
  shadow_rec_.prop_pos = state_->pos();
  shadow_rec_.prop_rv = Log(state_->rot());
  shadow_rec_.prior_cov = prior_cov_;

  V3D dtheta = V3D::Zero(), dt = V3D::Zero();
  double prev_error = std::numeric_limits<double>::infinity();
  std::string stop = "max_iter";
  bool any_solved = false;
  int last_iter = -1;
  for (int iter = 0; iter < opts_.max_iterations; ++iter)
  {
    const double error = estimateStateCorrection(mg.points, dtheta, dt, /*allow_consistency_log=*/false);
    last_iter = iter;
    recordIteration(shadow_rec_, iter, dtheta, dt);
    if (iter == 0) dumpResidualsIfListed(0, /*shadow=*/true);
    if (!residuals_.empty()) any_solved = true;
    const double prev = prev_error;
    prev_error = error;
    if (dtheta.norm() < opts_.min_norm_dtheta && dt.norm() < opts_.min_norm_dt) { stop = "norm"; break; }
    if ((prev - error) / std::max(prev, 1e-6) < opts_.min_diff_error) { stop = "rel_diff"; break; }
  }
  if (opts_.residual_dump_last_iter && last_iter > 0) dumpResidualsIfListed(last_iter, /*shadow=*/true);
  if (any_solved) ekf_.applyCovarianceUpdate(state_, prior_cov_, 1.0);
  shadow_rec_.stop = stop;
  shadow_rec_.post_pos = state_->pos();
  shadow_rec_.post_rv = Log(state_->rot());
  shadow_rec_.post_cov = state_->cov();
  ++shadow_scans_;
  if (!any_solved) ++shadow_scans_no_residuals_;

  // Restore.
  shadow_active_ = false;
  state_ = real_state;
  prior_cov_ = real_prior_cov;
  state_propagat_ = real_state_propagat;
  residuals_ = real_residuals;
  ekf_ = real_ekf;
  redundancy_stats_ = real_redundancy;
  step_cap_iterations_ = sc_it; step_cap_bound_ = sc_bound; step_cap_scale_sum_ = sc_sum; step_cap_scale_min_ = sc_min;
  step_cap_rot_raw_max_ = sc_rot; step_cap_pos_raw_max_ = sc_pos;
  info_scale_iterations_ = is_it; info_scale_discarded_modes_ = is_disc; info_scale_trace_ratio_sum_ = is_tr;
  renorm_iterations_ = rn_it; renorm_applied_ = rn_ap; renorm_c_sum_ = rn_sum; renorm_c_max_ = rn_max;
  n_miss_coverage_ = n_mc; n_miss_mismatch_ = n_mm; n_statistical_gate_candidates_ = n_gc; n_statistical_gate_rejections_ = n_gr;
  n_tier0_miss_coverage_ = n_t0c; n_tier0_miss_mismatch_ = n_t0m;
  setDiagnosticGnIteration(real_diag_iter);

  shadow_digest_after_ = vm ? vm->mapDigestExact() : 0ULL;
  if (shadow_digest_after_ != shadow_digest_before_)
  {
    ++shadow_map_changed_scans_;
    ROS_ERROR_STREAM("[shadow] the voxel map digest CHANGED during the shadow pass of frame " << voxel_map_->frame_idx_
                     << " (before " << hex64(shadow_digest_before_) << ", after " << hex64(shadow_digest_after_)
                     << "): the shadow pass is not read-only; the shadow arms are invalid");
  }
}

// shadow_scan.csv: one row per (window scan, role). shadow_cov.csv: the full prior and posterior covariance of each role.
void LioProcIekf::writeShadowRows(const MeasureGroup& mg, unsigned long long digest_before, unsigned long long digest_after)
{
  static PersistentLogStream log("shadow_scan.csv");
  static PersistentLogStream clog("shadow_cov.csv");
  bool first = false, cfirst = false;
  std::ofstream& o = log.stream(&first);
  std::ofstream& co = clog.stream(&cfirst);
  if (first)
  {
    o << "frame,t_abs,role,q_acc_x,q_acc_y,q_acc_z,q_gyr_x,q_gyr_y,q_gyr_z,q_samples,acc_clamped_axes,gyr_clamped_axes,"
         "start_pos_x,start_pos_y,start_pos_z,start_rv_x,start_rv_y,start_rv_z,"
         "prop_pos_x,prop_pos_y,prop_pos_z,prop_rv_x,prop_rv_y,prop_rv_z,"
         "post_pos_x,post_pos_y,post_pos_z,post_rv_x,post_rv_y,post_rv_z,"
         "prior_trace_rot,prior_trace_pos,prior_trace_vel,post_trace_rot,post_trace_pos,post_trace_vel,"
         "n_iter,n_resid_iter0,reduced_chi2_iter0,n_resid_by_iter,step_rot_by_iter,step_pos_by_iter,stop";
    for (int i = 0; i < 6; ++i)
      for (int j = i; j < 6; ++j) o << ",HtH0_" << i << j;
    for (int i = 0; i < 6; ++i) o << ",Htz0_" << i;
    for (int i = 0; i < 6; ++i)
      for (int j = i; j < 6; ++j) o << ",HtHlast_" << i << j;
    for (int i = 0; i < 6; ++i) o << ",Htzlast_" << i;
    o << ",map_digest_before_pass,map_digest_after_pass\n";
  }
  if (cfirst) co << "frame,role,kind,dim,row_major_values\n";
  const double t_abs = mg.image.t + data_queues_->start_time;
  const int frame = voxel_map_->frame_idx_;
  for (int role = 0; role < 2; ++role)
  {
    const UpdateRecord& r = role == 0 ? drv_rec_ : shadow_rec_;
    if (!r.valid) continue;
    const char* name = role == 0 ? "driver" : "shadow";
    const ImuProc& imu = role == 0 ? shadow_->driverImu() : shadow_->imu();
    const ImuProc::ScanQ& q = imu.lastScanQ();
    o << frame << ',' << std::setprecision(17) << t_abs << ',' << name << ',' << q.q_acc.x() << ',' << q.q_acc.y() << ',' << q.q_acc.z()
      << ',' << q.q_gyr.x() << ',' << q.q_gyr.y() << ',' << q.q_gyr.z() << ',' << q.samples << ',' << q.acc_clamped_axes << ','
      << q.gyr_clamped_axes;
    for (const V3D* v : {&r.start_pos, &r.start_rv, &r.prop_pos, &r.prop_rv, &r.post_pos, &r.post_rv})
      o << ',' << (*v).x() << ',' << (*v).y() << ',' << (*v).z();
    o << std::setprecision(10);
    o << ',' << blockTrace(r.prior_cov, StateGroup::idxR()) << ',' << blockTrace(r.prior_cov, StateGroup::idxP()) << ','
      << blockTrace(r.prior_cov, StateGroup::idxV()) << ',' << blockTrace(r.post_cov, StateGroup::idxR()) << ','
      << blockTrace(r.post_cov, StateGroup::idxP()) << ',' << blockTrace(r.post_cov, StateGroup::idxV());
    o << ',' << r.n_iter << ',' << r.n_resid0 << ',' << r.chi2_iter0 << ',' << joinI(r.n_resid_iter) << ',' << joinD(r.step_rot) << ','
      << joinD(r.step_pos) << ',' << r.stop;
    for (int i = 0; i < 6; ++i)
      for (int j = i; j < 6; ++j) o << ',' << r.HtH0(i, j);
    for (int i = 0; i < 6; ++i) o << ',' << r.Htz0(i);
    for (int i = 0; i < 6; ++i)
      for (int j = i; j < 6; ++j) o << ',' << r.HtHlast(i, j);
    for (int i = 0; i < 6; ++i) o << ',' << r.Htzlast(i);
    if (role == 1) o << ',' << hex64(digest_before) << ',' << hex64(digest_after);
    else o << ",,";
    o << '\n';
    for (int kind = 0; kind < 2; ++kind)
    {
      const Eigen::MatrixXd& P = kind == 0 ? r.prior_cov : r.post_cov;
      co << frame << ',' << name << ',' << (kind == 0 ? "prior" : "post") << ',' << P.rows() << ',' << std::setprecision(12);
      for (int a = 0; a < P.rows(); ++a)
        for (int b = 0; b < P.cols(); ++b) co << (a || b ? " " : "") << P(a, b);
      co << '\n';
    }
  }
  o.flush();
  co.flush();
}

void LioProcIekf::writeScanLogRow(const MeasureGroup& mg, const VoxelMapScanStats& st) const
{
  static PersistentLogStream log("scan_log.csv");
  bool first = false;
  std::ofstream& o = log.stream(&first);
  if (first)
    o << "frame,t_abs,n_root_voxels,n_nodes,n_plane_nodes,n_retired,voxels_created,planes_changed,map_digest_exact,"
         "map_digest_rounded_1e-3,map_frozen_scans,n_resid_iter0,n_iter,reduced_chi2_iter0,n_resid_by_iter,step_rot_by_iter,"
         "step_pos_by_iter,post_trace_rot,post_trace_pos,post_trace_vel\n";
  const auto* vm = dynamic_cast<const VoxelMap*>(voxel_map_.get());
  o << voxel_map_->frame_idx_ << ',' << std::setprecision(17) << (mg.image.t + data_queues_->start_time) << std::setprecision(10) << ','
    << st.n_root_voxels << ',' << st.n_nodes << ',' << st.n_plane_nodes << ',' << st.n_retired << ',' << st.voxels_created << ','
    << st.planes_changed << ',' << hex64(st.digest_exact) << ',' << hex64(st.digest_rounded) << ',' << (vm ? vm->frozenScans() : 0L) << ','
    << drv_rec_.n_resid0 << ',' << drv_rec_.n_iter << ',' << drv_rec_.chi2_iter0 << ',' << joinI(drv_rec_.n_resid_iter) << ','
    << joinD(drv_rec_.step_rot) << ',' << joinD(drv_rec_.step_pos) << ',' << blockTrace(drv_rec_.post_cov, StateGroup::idxR()) << ','
    << blockTrace(drv_rec_.post_cov, StateGroup::idxP()) << ',' << blockTrace(drv_rec_.post_cov, StateGroup::idxV()) << '\n';
  o.flush();
}
}  // namespace livo_recon
