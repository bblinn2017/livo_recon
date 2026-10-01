#include "livo_recon/processing/lio_base.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/log/config_resolve.h"
#include "livo_recon/utils/algo/omp_utils.h"
#include "livo_recon/utils/algo/hashing.h"
#include "livo_recon/map/voxelmap.h"
#include "livo_recon/lio/voxelplane.h"   // voxelPlaneInformationFitCount(), flushVarianceShareLog/flushConsistencyCorrLog
#include "livo_recon/lio/lio_accumulator.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"
#include "livo_recon/diagnostics/state_trace.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

namespace livo_recon
{

namespace
{

// CQ-46: per-residual pair-correlation diagnostic. Mirrors voxelplane.cpp's
// t_corr_buf/g_corr_with_covariates idiom exactly (thread_local append
// buffer, one flush per thread after the OMP region's implicit barrier) for
// the same reason: writing per-candidate, not just per-accepted-residual,
// through a shared ofstream would serialize buildResiduals()'s
// embarrassingly-parallel loop. Gated behind log_pair_corr_en (default
// false) AND allow_consistency_log (true only on a scan's first IEKF
// iteration) -- neither condition is met by any shipped config, so this is
// dead weight, not a live cost, at every default.
thread_local std::string t_pair_corr_buf;

void appendPairCorrResidual(int scan_id, const Residual& res)
{
  std::ostringstream oss;
  oss << std::setprecision(9)
      << scan_id << ','
      << res.world_point.x() << ',' << res.world_point.y() << ',' << res.world_point.z() << ','
      << res.normal.x() << ',' << res.normal.y() << ',' << res.normal.z() << ','
      << res.point_cross_normal.x() << ',' << res.point_cross_normal.y() << ',' << res.point_cross_normal.z() << ','
      << reinterpret_cast<uint64_t>(res.plane_id) << ','
      << res.r << ',' << res.floor_term << ',' << res.sigma_diag_squared << ','
      << res.plane_var_term << ',' << res.s_prior_pose << '\n';
  t_pair_corr_buf += oss.str();
}

void flushPairCorrLog()
{
  if (t_pair_corr_buf.empty()) return;
  static PersistentLogStream log("cq46_residuals.txt");
  bool first_call;
  std::ofstream& ofs = log.stream(&first_call);
  if (first_call)
    ofs << "scan_id,px,py,pz,nx,ny,nz,jx,jy,jz,plane_id,r,floor_term,sigma_diag_squared,plane_var_term,s_prior_pose\n";
  ofs << t_pair_corr_buf;
  ofs.flush();
  t_pair_corr_buf.clear();
}

void logPairCorrPrior(int scan_id, const Eigen::Matrix<double, 6, 6>& prior_cov_rp)
{
  static PersistentLogStream log("cq46_prior.txt");
  bool first_call;
  std::ofstream& ofs = log.stream(&first_call);
  if (first_call) {
    ofs << "scan_id";
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j)
        ofs << ",p" << i << j;
    ofs << "\n";
  }
  ofs << scan_id << std::setprecision(9);
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      ofs << ',' << prior_cov_rp(i, j);
  ofs << "\n";
  ofs.flush();
}



void writeFirstFrameSolveMatrix(std::ofstream& ofs, const char* name, const Eigen::MatrixXd& M)
{
  ofs << "matrix " << name << " rows=" << M.rows() << " cols=" << M.cols() << "\n";
  ofs << std::setprecision(17);
  for (int r = 0; r < M.rows(); ++r) {
    for (int c = 0; c < M.cols(); ++c) {
      if (c) ofs << ' ';
      ofs << M(r, c);
    }
    ofs << '\n';
  }
}

void writeFirstFrameSolveVector(std::ofstream& ofs, const char* name, const Eigen::VectorXd& v)
{
  ofs << "vector " << name << " size=" << v.size() << "\n";
  ofs << std::setprecision(17);
  for (int i = 0; i < v.size(); ++i) {
    if (i) ofs << ' ';
    ofs << v(i);
  }
  ofs << '\n';
}

void logFirstFrameIekfSolve(
    int frame_idx, int scan_id, int iteration, double t_abs,
    const char* architecture, const std::vector<PointXYZCov>& pts,
    const std::vector<Residual>& residuals, const StateGroup& before,
    const StateGroup& after, const StateGroup& propagat,
    const Eigen::MatrixXd& prior_cov, const EkfUpdate& ekf,
    const Eigen::Matrix<double,6,6>& HtH,
    const Eigen::Matrix<double,6,1>& Htz,
    int max_frame)
{
  if (frame_idx < 1 || frame_idx > max_frame) return;
  static PersistentLogStream csv_log("pose_control_first_frame_solve.csv");
  bool first = false;
  std::ofstream& csv = csv_log.stream(&first);
  if (first) csv << "architecture,frame_idx,scan_id,iteration,t_abs,n_points,n_residuals,"
                     "delta_px,delta_py,delta_pz,delta_pnorm,delta_rx,delta_ry,delta_rz,delta_rnorm,"
                     "HtH_trace,Htz_norm,prior_cov_trace,A_trace,K1_norm,solution_norm\n";
  const V3D dp = after.pos() - before.pos();
  const V3D dtheta = Log(before.rot().transpose() * after.rot());
  const Eigen::VectorXd& solution = ekf.lastSolution();
  csv << std::setprecision(17)
      << architecture << ',' << frame_idx << ',' << scan_id << ',' << iteration << ',' << t_abs << ','
      << pts.size() << ',' << residuals.size() << ','
      << dp.x() << ',' << dp.y() << ',' << dp.z() << ',' << dp.norm() << ','
      << dtheta.x() << ',' << dtheta.y() << ',' << dtheta.z() << ',' << dtheta.norm() << ','
      << HtH.trace() << ',' << Htz.norm() << ',' << prior_cov.trace() << ','
      << (ekf.lastA().size() ? ekf.lastA().trace() : 0.0) << ','
      << (ekf.lastK1().size() ? ekf.lastK1().norm() : 0.0) << ','
      << (solution.size() ? solution.norm() : 0.0) << '\n';
  csv.flush();
  static PersistentLogStream dump_log("pose_control_first_frame_solve_matrices.txt");
  std::ofstream& dump = dump_log.stream();
  dump << "=== iekf_solve_snapshot ===\n"
       << "architecture=" << architecture << "\n"
       << "frame_idx=" << frame_idx << "\n"
       << "scan_id=" << scan_id << "\n"
       << "iteration=" << iteration << "\n"
       << "t_abs=" << std::setprecision(17) << t_abs << "\n"
       << "n_points=" << pts.size() << "\n"
       << "n_residuals=" << residuals.size() << "\n"
       << "state_before_pos=" << before.pos().transpose() << "\n"
       << "state_after_pos=" << after.pos().transpose() << "\n"
       << "delta_pos=" << dp.transpose() << "\n"
       << "delta_rot=" << dtheta.transpose() << "\n"
       << "propagated_pos=" << propagat.pos().transpose() << "\n"
       << "propagated_rot_log=" << Log(propagat.rot()).transpose() << "\n";
  // FIX (found this round): the supplied patch replaced this diagnostic's
  // write of `vec` with the exact production value (ekf.lastVec()), per its
  // own explicit rationale that a hand-recomputed vec here (previously
  // `before.boxminusFromPropagat(propagat)`, using the pre-iteration
  // snapshot `before` rather than production's own live `state` pointer at
  // the actual solve point) "could use a different state... and would
  // defeat the audit" -- but left the now-dead local declaration and its
  // redundant boxminus computation in place. Removed as dead code, exactly
  // the kind of duplicate-computation this patch exists to eliminate.
  writeFirstFrameSolveMatrix(dump, "prior_cov", prior_cov);
  writeFirstFrameSolveMatrix(dump, "H_full", ekf.lastHFull());
  writeFirstFrameSolveMatrix(dump, "A_exact", ekf.lastA());
  writeFirstFrameSolveMatrix(dump, "K1_exact", ekf.lastK1());
  writeFirstFrameSolveMatrix(dump, "HtH", Eigen::MatrixXd(HtH));
  writeFirstFrameSolveVector(dump, "Htz", Eigen::VectorXd(Htz));
  writeFirstFrameSolveVector(dump, "prior_boxminus_propagated", ekf.lastVec());
  writeFirstFrameSolveVector(dump, "prior_boxminus_propagated_rp", ekf.lastVecRp());
  writeFirstFrameSolveMatrix(dump, "G_exact", ekf.lastG());
  writeFirstFrameSolveVector(dump, "measurement_term_exact", ekf.lastMeasurementTerm());
  writeFirstFrameSolveVector(dump, "prior_term_exact", ekf.lastPriorTerm());
  writeFirstFrameSolveVector(dump, "solution_exact", solution);
  for (size_t i = 0; i < residuals.size(); ++i) {
    const auto& r = residuals[i];
    dump << "residual " << i << " t=" << r.t << " r=" << r.r
         << " sigma_squared=" << r.sigma_squared << " plane_var_term=" << r.plane_var_term
         << " plane_id=" << reinterpret_cast<uintptr_t>(r.plane_id)
         << " h_rot=" << r.point_cross_normal.transpose() << " h_pos=" << r.normal.transpose()
         << " world_point=" << r.world_point.transpose()
         << " raw_body_point=" << r.raw_body_point.transpose() << '\n';
  }
  dump << "=== end_iekf_solve_snapshot ===\n";
  dump.flush();
}


// R69 (F-300): write-only per-iteration record of the LiDAR information actually used by the update.
// One row per (frame, iteration). HtH is the 6x6 (rot, pos) information, upper triangle row-major; Htz its right-hand side.
// sigma_squared is the admitted residual variance as used (after any bootstrap_inflate factor); plane_var_term is the
// plane-fit part of it, so base = sigma_squared - plane_var_term.
void logUpdateDiag(int frame_idx, int iteration, double t_abs, const std::vector<Residual>& residuals, const EkfUpdate& ekf)
{
  static PersistentLogStream log("lio_update_diag.csv");
  bool first = false;
  std::ofstream& o = log.stream(&first);
  if (first) {
    o << "frame,iter,t_abs,n_resid,sum_inv_sigma2,mean_sigma2,mean_plane_var_term,mean_base_sigma2,mean_r,mean_r2_over_sigma2";
    for (int i = 0; i < 6; ++i) for (int j = i; j < 6; ++j) o << ",HtH_" << i << j;
    for (int i = 0; i < 6; ++i) o << ",Htz_" << i;
    o << ",dtheta_x,dtheta_y,dtheta_z,dt_x,dt_y,dt_z\n";
  }
  const double n = static_cast<double>(residuals.size());
  double s_inv = 0.0, s_sig = 0.0, s_pv = 0.0, s_r = 0.0, s_chi = 0.0;
  for (const auto& r : residuals) {
    s_inv += 1.0 / r.sigma_squared; s_sig += r.sigma_squared; s_pv += r.plane_var_term;
    s_r += r.r; s_chi += r.r * r.r / r.sigma_squared;
  }
  const double dn = n > 0.0 ? n : 1.0;
  o << std::setprecision(10) << frame_idx << ',' << iteration << ',' << std::setprecision(17) << t_abs << ',' << residuals.size()
    << ',' << s_inv << ',' << s_sig / dn << ',' << s_pv / dn << ',' << (s_sig - s_pv) / dn << ',' << s_r / dn << ',' << s_chi / dn;
  for (int i = 0; i < 6; ++i) for (int j = i; j < 6; ++j) o << ',' << ekf.HtH(i, j);
  for (int i = 0; i < 6; ++i) o << ',' << ekf.Htz(i);
  o << ',' << ekf.dtheta.x() << ',' << ekf.dtheta.y() << ',' << ekf.dtheta.z()
    << ',' << ekf.dt.x() << ',' << ekf.dt.y() << ',' << ekf.dt.z() << '\n';
  o.flush();
}

// R69 (F-300): write-only statistics of the live scan's per-point sensor covariance (body frame, as passed to the
// residual builder), first iteration of every frame. sigma_trace3 = trace/3 per point.
void logLivePointCov(int frame_idx, const std::vector<PointXYZCov>& pts)
{
  static PersistentLogStream log("live_point_cov.csv");
  bool first = false;
  std::ofstream& o = log.stream(&first);
  if (first) {
    o << "frame,n_pts,mean_range,min_sigma_trace3,mean_sigma_trace3,max_sigma_trace3,mean_C00,mean_C01,mean_C02,mean_C11,mean_C12,mean_C22\n";
  }
  double mr = 0.0, mn = 1e300, mx = 0.0, sm = 0.0;
  double c[6] = {0, 0, 0, 0, 0, 0};
  for (const auto& p : pts) {
    const double t3 = p.sensor_cov.trace() / 3.0;
    mr += p.point.norm(); sm += t3; mn = std::min(mn, t3); mx = std::max(mx, t3);
    c[0] += p.sensor_cov(0, 0); c[1] += p.sensor_cov(0, 1); c[2] += p.sensor_cov(0, 2);
    c[3] += p.sensor_cov(1, 1); c[4] += p.sensor_cov(1, 2); c[5] += p.sensor_cov(2, 2);
  }
  const double n = static_cast<double>(pts.size());
  const double dn = n > 0.0 ? n : 1.0;
  if (pts.empty()) mn = 0.0;
  o << frame_idx << ',' << pts.size() << ',' << std::setprecision(10) << mr / dn << ',' << mn << ',' << sm / dn << ',' << mx;
  for (int k = 0; k < 6; ++k) o << ',' << c[k] / dn;
  o << '\n';
  o.flush();
}

// R70 (F-340): write-only per-frame geometry of the admitted residuals (iteration 0). Backend independent.
// n_planes / top1 / top5: distinct matched planes (by plane_id) and the share of residuals owned by the largest one / five.
// nn_eig*: eigenvalues of (1/n) sum n n^T (unweighted normal spread, ascending; a flat scene has nn_eig0 ~ 0 across x or y).
// pos_info_eig* / rot_info_eig*: eigenvalues (ascending) of the 3x3 position and rotation blocks of sum w h h^T with
// w = 1/sigma_squared as used (so they include any bootstrap_inflate factor but not the block information scale).
// abs_r_med / abs_r_p90: |r| quantiles. range_med: median body-frame range of the matched points. tier1_frac: share found by the
// neighbour fallback.
void logResidualGeometry(int frame_idx, const std::vector<Residual>& residuals)
{
  static PersistentLogStream log("residual_geometry.csv");
  bool first = false;
  std::ofstream& o = log.stream(&first);
  if (first)
    o << "frame,n_resid,n_planes,top1_share,top5_share,nn_eig0,nn_eig1,nn_eig2,pos_info_eig0,pos_info_eig1,pos_info_eig2,"
         "rot_info_eig0,rot_info_eig1,rot_info_eig2,abs_r_med,abs_r_p90,range_med,tier1_frac\n";
  const size_t n = residuals.size();
  o << frame_idx << ',' << n;
  if (n == 0) { o << std::string(16, ',') << '\n'; o.flush(); return; }
  std::map<const void*, int> owners;
  M3D nn = M3D::Zero(), Ip = M3D::Zero(), Ir = M3D::Zero();
  std::vector<double> ar, rg;
  ar.reserve(n); rg.reserve(n);
  size_t t1 = 0;
  for (const auto& r : residuals) {
    ++owners[r.plane_id];
    nn += r.normal * r.normal.transpose();
    const double w = 1.0 / r.sigma_squared;
    Ip += w * r.normal * r.normal.transpose();
    Ir += w * r.point_cross_normal * r.point_cross_normal.transpose();
    ar.push_back(std::abs(r.r));
    rg.push_back(r.raw_body_point.norm());
    if (r.match_tier != 0) ++t1;
  }
  std::vector<int> cnt;
  for (const auto& kv : owners) cnt.push_back(kv.second);
  std::sort(cnt.begin(), cnt.end(), std::greater<int>());
  int top5 = 0;
  for (size_t i = 0; i < cnt.size() && i < 5; ++i) top5 += cnt[i];
  auto quant = [](std::vector<double>& v, double q) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(q * (v.size() - 1) + 0.5))];
  };
  auto eig3 = [](const M3D& m) { Eigen::SelfAdjointEigenSolver<M3D> es(m); return es.eigenvalues(); };
  const V3D e_nn = eig3(nn / static_cast<double>(n)), e_p = eig3(Ip), e_r = eig3(Ir);
  o << ',' << owners.size() << ',' << std::setprecision(10) << static_cast<double>(cnt[0]) / n << ',' << static_cast<double>(top5) / n;
  for (int k = 0; k < 3; ++k) o << ',' << e_nn(k);
  for (int k = 0; k < 3; ++k) o << ',' << e_p(k);
  for (int k = 0; k < 3; ++k) o << ',' << e_r(k);
  o << ',' << quant(ar, 0.5) << ',' << quant(ar, 0.9) << ',' << quant(rg, 0.5) << ',' << static_cast<double>(t1) / n << '\n';
  o.flush();
}


// R71 (F-363): write-only per-residual dump for the frames listed in lio/residual_dump_frames (iteration 0, i.e. at the
// propagated state), plus the linearisation state. Lets the information each residual claims (w_i h_i h_i^T) be compared
// offline with what the residuals jointly deliver (leverage, per-plane groups, split-half and cross-frame solutions).
// h_rot / h_pos are the rotation and position Jacobian rows (point_cross_normal, normal); w_i = 1 / sigma_squared.
void logResidualDump(int frame_idx, int iteration, const StateGroup& st, const std::vector<Residual>& residuals)
{
  static PersistentLogStream log("residual_dump.csv");
  static PersistentLogStream slog("residual_dump_state.csv");
  bool first = false, sfirst = false;
  std::ofstream& o = log.stream(&first);
  std::ofstream& so = slog.stream(&sfirst);
  if (first)
    o << "frame,iter,idx,source_index,t,r,sigma_squared,plane_var_term,plane_key,node_id,tier,vis,"
         "hrot_x,hrot_y,hrot_z,hpos_x,hpos_y,hpos_z,wx,wy,wz,bx,by,bz\n";
  if (sfirst) so << "frame,iter,n_resid,px,py,pz,rvx,rvy,rvz\n";
  const V3D rv = Log(st.rot());
  so << frame_idx << ',' << iteration << ',' << residuals.size() << ',' << std::setprecision(17) << st.pos().x() << ','
     << st.pos().y() << ',' << st.pos().z() << ',' << rv.x() << ',' << rv.y() << ',' << rv.z() << '\n';
  o << std::setprecision(9);
  for (size_t i = 0; i < residuals.size(); ++i) {
    const auto& r = residuals[i];
    o << frame_idx << ',' << iteration << ',' << i << ',' << r.source_index << ',' << r.t << ',' << r.r << ',' << r.sigma_squared << ','
      << r.plane_var_term << ',' << reinterpret_cast<uintptr_t>(r.plane_id) << ',' << r.plane_node_id << ',' << r.match_tier << ',' << r.vis_state
      << ',' << r.point_cross_normal.x() << ',' << r.point_cross_normal.y() << ',' << r.point_cross_normal.z()
      << ',' << r.normal.x() << ',' << r.normal.y() << ',' << r.normal.z()
      << ',' << r.world_point.x() << ',' << r.world_point.y() << ',' << r.world_point.z()
      << ',' << r.raw_body_point.x() << ',' << r.raw_body_point.y() << ',' << r.raw_body_point.z() << '\n';
  }
  o.flush(); so.flush();
}
}  // namespace

LioProcBase::LioProcBase(NodeContext& ctx)
  : state_(ctx.state), voxel_map_(ctx.voxel_map), profiler_(ctx.profiler), data_queues_(ctx.data_queues)
{}

// CQ-49 item 1: extracted ONCE from the pre-split LioProc::loadParameters(),
// called from both derived classes' own loadParameters() overrides against
// their OWN ConfigResolver -- see each override for the estimator-specific
// keys (estimator/coupled/* for coupled)
// this does NOT read. Order relative to the original function is changed
// (this reads its keys before the derived override reads its own), which is
// safe: item 0's byte-identical gate is on odometry.txt, and the
// [config/effective] echo's key ORDER is explicitly excluded from that
// check (CQ-49 item 3f(iv)).
void LioProcBase::loadSharedParameters(ConfigResolver& cfg, ros::NodeHandle& pnh)
{
  paramWarn<int>(pnh, "lio/ekf/max_iterations",     opts_.max_iterations,  5);
  paramWarn<double>(pnh, "lio/ekf/min_norm_dtheta", opts_.min_norm_dtheta, 0.0);
  paramWarn<double>(pnh, "lio/ekf/min_norm_dt",     opts_.min_norm_dt,     0.0);
  paramWarn<double>(pnh, "lio/ekf/min_diff_error",  opts_.min_diff_error,  -1.0);
  paramWarn<bool>(pnh, "lio/log_debug_en",          opts_.log_debug_en,   false);
  paramWarn<bool>(pnh, "lio/log_pair_corr_en",      opts_.log_pair_corr_en, false);
  paramWarn<bool>(pnh, "lio/log_eigenspectrum_en",  opts_.log_eigenspectrum_en, false);
  paramWarn<bool>(pnh, "lio/log_consistency_scan_en", opts_.log_consistency_scan_en, false);
  paramWarn<bool>(pnh, "lio/log_nll_en", opts_.log_nll_en, false);
  paramWarn<bool>(pnh, "lio/update_diag_en", opts_.update_diag_en, false);          // R69: lio_update_diag.csv (write-only)
  paramWarn<bool>(pnh, "lio/live_point_cov_en", opts_.live_point_cov_en, false);    // R69: live_point_cov.csv (write-only)
  paramWarn<int>(pnh, "lio/first_frame_dump_frames", opts_.first_frame_dump_frames, 1);  // R69: solve dump for frames 1..K (1 = unchanged)
  paramWarn<int>(pnh, "lio/dry_run_point_filter_num", opts_.dry_run_point_filter_num, 0);
  paramWarn<bool>(pnh, "eval/nees_per_dof_en", opts_.nees_per_dof_en, false);
  paramWarn<int>(pnh, "eval/nees_tier1_window_scans", opts_.nees_tier1_window_scans, 489);
  paramWarn<bool>(pnh, "eval/state_trace_en", opts_.state_trace_en, true);
  paramWarn<std::string>(pnh, "eval/state_trace_run_id", opts_.state_trace_run_id, "run");
  paramWarn<bool>(pnh, "cuda/enable",               cuda_enable_,          false);

  // History (134-136): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-134
  double range_err;
  paramWarn<double>(pnh, "imu/sensor/range_err", range_err, 0.05);
  opts_.deskew.sigma_r2 = range_err * range_err;
  double angle_err_deg;
  paramWarn<double>(pnh, "imu/sensor/angle_err_deg", angle_err_deg, 0.2);
  const double sin_angle_err = std::sin(std::max(1e-6, angle_err_deg * M_PI / 180.0));
  opts_.deskew.sigma_a2 = sin_angle_err * sin_angle_err;

  // Independent control or exact per-VoxelPlane covariance marginalization.
  cfg.mode("lio/residual_redundancy/mode", opts_.residual_redundancy.mode, "off",
           { "off", "woodbury" });
  const bool rr = opts_.residual_redundancy.mode != "off";
  cfg.nested<double>(rr, "lio/residual_redundancy/mode!=off", "lio/residual_redundancy/rho",
                     opts_.residual_redundancy.rho, 1.0);
  cfg.nestedMode(rr, "lio/residual_redundancy/mode!=off", "lio/residual_redundancy/renorm",
                 opts_.residual_redundancy.renorm, "off", { "off", "hold_pos_min_eig" });

  // R70 (F-320): block information scale (rotation vs position). Sub-options are refused unless mode=block; block mode with
  // both scales 1 would be inert and is refused (rule 2.4 / R4c).
  cfg.mode("lio/info_scale/mode", opts_.info_scale.mode, "off", { "off", "block" });
  {
    const bool blk = opts_.info_scale.mode == "block";
    cfg.nested<double>(blk, "lio/info_scale/mode=block", "lio/info_scale/rotation", opts_.info_scale.rotation, 1.0);
    cfg.nested<double>(blk, "lio/info_scale/mode=block", "lio/info_scale/position", opts_.info_scale.position, 1.0);
    if (blk) {
      if (!(opts_.info_scale.rotation >= 1.0) || !(opts_.info_scale.position >= 1.0))
        throw std::invalid_argument("lio/info_scale/rotation and /position must be >= 1");
      if (opts_.info_scale.rotation == 1.0 && opts_.info_scale.position == 1.0)
        throw std::invalid_argument("lio/info_scale/mode=block with rotation == position == 1 is inert (use mode=off)");
    }
  }
  // R72 (F-403, F-391): hard cap on the iekf step. Sub-options refused unless mode=hard. Caps must be positive.
  cfg.mode("lio/ekf/step_cap/mode", opts_.step_cap.mode, "off", { "off", "hard" });
  {
    const bool hard = opts_.step_cap.mode == "hard";
    cfg.nested<double>(hard, "lio/ekf/step_cap/mode=hard", "lio/ekf/step_cap/rotation_rad", opts_.step_cap.rotation_rad, 1e-2);
    cfg.nested<double>(hard, "lio/ekf/step_cap/mode=hard", "lio/ekf/step_cap/position_m", opts_.step_cap.position_m, 1e-2);
    if (hard && (!(opts_.step_cap.rotation_rad > 0.0) || !(opts_.step_cap.position_m > 0.0)))
      throw std::invalid_argument("lio/ekf/step_cap/rotation_rad and /position_m must be > 0");
    if (hard && cuda_enable_)
      throw std::invalid_argument("lio/ekf/step_cap/mode=hard is not implemented with cuda/enable=true");
    ekf_.step_cap = opts_.step_cap;   // off: EkfUpdate never evaluates it (bit-identical)
  }
  {   // R71: per-residual dump frames, a comma-separated list of frame indices in ONE string key (job overrides carry scalars only); empty = off
    std::string dump_frames;
    paramWarn<std::string>(pnh, "lio/residual_dump_frames", dump_frames, std::string());
    opts_.residual_dump_frames.clear();
    std::stringstream ss(dump_frames);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (tok.empty()) continue;
      char* end = nullptr;
      const long v = std::strtol(tok.c_str(), &end, 10);
      if (end == tok.c_str() || *end != '\0' || v < 1)
        throw std::invalid_argument("lio/residual_dump_frames: '" + tok + "' is not a positive integer frame index (comma-separated list, e.g. \"1,2,3,10\")");
      opts_.residual_dump_frames.push_back(static_cast<int>(v));
    }
  }
  paramWarn<bool>(pnh, "lio/residual_geometry_en", opts_.residual_geometry_en, false);   // R70: residual_geometry.csv (write-only)

  // R61: open-loop IMU propagation (see OpenLoopOptions). reset_period_s is only meaningful, and only accepted,
  // under propagate_only.
  cfg.mode("lio/open_loop/mode", opts_.open_loop.mode, "off", { "off", "propagate_only" });
  cfg.nested<double>(opts_.open_loop.mode == "propagate_only", "lio/open_loop/mode=propagate_only",
                     "lio/open_loop/reset_period_s", opts_.open_loop.reset_period_s, 0.0);

  // R65 (F-169): per-point pose covariance model. end_pose (default) = R64 behaviour, bit-identical. propagated adds, per
  // point, the noise accrued from the point's time to the scan end. insertion_source=prior makes map insertion use the
  // frame's prior (R,P) block, the same basis the residual gate sees; posterior (default) = R64.
  {
    std::string pc_mode, pc_ins;
    cfg.mode("lio/point_cov/mode", pc_mode, "end_pose", { "end_pose", "propagated" });
    cfg.mode("lio/point_cov/insertion_source", pc_ins, "posterior", { "posterior", "prior" });
    state_->setPointCovModel(pc_mode == "propagated", pc_ins == "prior");
  }

  // R65 (F-128): measurement-noise inflation for the first N LIO scans (see LioProcOptions::bootstrap_inflate_scans).
  cfg.get<int>("lio/bootstrap_inflate/scans", opts_.bootstrap_inflate_scans, 0);
  if (opts_.bootstrap_inflate_scans < 0)
    throw std::invalid_argument("lio/bootstrap_inflate/scans must be >= 0");
  cfg.nested<double>(opts_.bootstrap_inflate_scans > 0, "lio/bootstrap_inflate/scans>0", "lio/bootstrap_inflate/factor",
                     opts_.bootstrap_inflate_factor, 1.0);
  if (!(opts_.bootstrap_inflate_factor >= 1.0))
    throw std::invalid_argument("lio/bootstrap_inflate/factor must be >= 1");

  // R64: startup hold (see LioProcOptions::startup_hold_scans). 0 = off.
  cfg.get<int>("lio/startup_hold/scans", opts_.startup_hold_scans, 0);
  if (opts_.startup_hold_scans < 0)
    throw std::invalid_argument("lio/startup_hold/scans must be >= 0");

  // CQ-31 item 5: three independently-switchable scalar P controls, all
  // default-identity.
  cfg.get<double>("lio/p_inflate/alpha", opts_.prior_scalar.p_inflate_alpha, 1.0);
  cfg.get<double>("lio/p_floor/min_eig", opts_.prior_scalar.p_floor_min_eig, 0.0);
  cfg.get<double>("lio/p_fading/lambda", opts_.prior_scalar.p_fading_lambda, 1.0);

  // Downsampling is one axis with three states, not a mode plus a magic
  // zero.
  cfg.mode("imu/ds/mode", opts_.ds_mode, "first", { "off", "first", "average" });
  cfg.nested<double>(opts_.dsOn(), "imu/ds/mode != off",
                     "imu/ds/ds_leaf_size", opts_.ds_leaf_size, 0.15);
  if (!opts_.dsOn()) opts_.ds_leaf_size = 0.0;
  else if (!(opts_.ds_leaf_size > 0.0))
    cfg.requireCombination(
        "imu/ds/ds_leaf_size must be > 0 when imu/ds/mode is '" + opts_.ds_mode +
        "' -- a zero leaf under an ON mode is the sentinel this mode set "
        "exists to remove; write imu/ds/mode: off instead");
  cfg.mode("imu/undistort/time_based_process_noise",
           opts_.deskew.time_based_process_noise, "var_acc",
           { "none", "state", "var_acc" });

}

// CQ-49: the shared finish -- refuse anything left unclaimed in the given
// namespaces (each derived class passes its OWN set: {"spline","adaptive_q",...}
// (removed namespaces, refused) for iekf, {"estimator"} for coupled), record the accumulation
// precision, and either refuse (throw) or return the effective config.
std::string LioProcBase::finalizeConfig(ConfigResolver& cfg, std::initializer_list<const char*> allowed_unclaimed_namespaces)
{
  cfg.refuseUnclaimed(allowed_unclaimed_namespaces);

  // The build's accumulation precision, recorded rather than assumed. The
  // CUDA path returns float-precision HtH/Htz, so nll.txt is NOT comparable
  // between a CPU and a GPU build.
  cfg.derived("build/accumulation_precision",
              cuda_enable_ ? "float (CUDA HtH/Htz) -- nll.txt NOT comparable "
                             "with a CPU build"
                           : "double (CPU HtH/Htz)");

  if (!cfg.ok())
  {
    ROS_FATAL_STREAM("\n" << cfg.report());
    throw std::runtime_error(
        "[config] refused: " + std::to_string(cfg.errors().size()) +
        " option(s) set into a dead scope or outside their allowed set -- "
        "see the [config/REFUSED] block above");
  }
  return cfg.report();
}

// Residual and EKF code
void LioProcBase::ensureStationaryReference(const MeasureGroup& mg)
{
  if (stationary_reference_valid_) return;
  stationary_reference_R_ = mg.rot_before_imu;
  stationary_reference_p_ = mg.pos_before_imu;
  stationary_reference_v_ = mg.vel_before_imu;
  stationary_reference_valid_ = true;
  writeStateReference(stationary_reference_R_, stationary_reference_p_, stationary_reference_v_,
                      mg.pose_covariances.empty() ? state_->cov() : mg.pose_covariances.front());
}


void LioProcBase::logStateTrace(const char* phase, const MeasureGroup& mg, int residual_count,
                                int completed_iterations)
{
  if (!opts_.state_trace_en) return;
  writeStateTraceRow(opts_.state_trace_run_id, static_cast<int>(voxel_map_->frame_idx_),
                     mg.image.t + data_queues_->start_time, phase, *state_, residual_count, completed_iterations,
                     ol_window_, ol_reset_now_);
}

std::string LioProcBase::openLoopFinishScan(MeasureGroup& mg)
{
  const double t_abs = mg.image.t + data_queues_->start_time;
  if (!ol_captured_) {
    // First scan that reached ensureStationaryReference(): its head covariance is the first-frame pre-IMU P0.
    ol_P0_ = mg.pose_covariances.empty() ? state_->cov() : mg.pose_covariances.front();
    ol_window_ = 0;
    ol_window_t0_ = t_abs;
    ol_captured_ = true;
  }
  ol_reset_now_ = 0;
  // The post_imu row was already written by the caller for the CURRENT window. Reset (if due) starts the next one.
  if (opts_.open_loop.reset_period_s > 0.0 && (t_abs - ol_window_t0_) >= opts_.open_loop.reset_period_s) {
    state_->setPropagatedState(stationary_reference_R_, stationary_reference_p_, stationary_reference_v_);
    state_->covMut() = ol_P0_;
    ++ol_window_;
    ol_window_t0_ = t_abs;
    ol_reset_now_ = 1;
  }
  logStateTrace("post_lio", mg, 0, 0);
  mg.pos_after_lio = state_->pos();
  mg.rot_after_lio = state_->rot();
  mg.vel_after_lio = state_->vel();
  mg.cov_after_lio = state_->cov();
  return "[open_loop] propagate_only window=" + std::to_string(ol_window_) + " reset=" + std::to_string(ol_reset_now_);
}

void LioProcBase::buildResiduals(
  const std::vector<PointXYZCov>& pts,
  std::vector<Residual>& residuals,
  bool allow_consistency_log,
  bool include_state_uncertainty_in_gate,
  const std::vector<M3D>* pose_cov_overrides) const {
  // Single-threaded, before the OMP region starts below -- see
  // VoxelMap::setAllowConsistencyLog()'s doc comment.
  if (auto* vm = dynamic_cast<VoxelMap*>(voxel_map_.get()))
    vm->setAllowConsistencyLog(allow_consistency_log);

  const int n = (int)pts.size();
  const int threads = cappedOmpThreads();

  build_thread_residuals_.resize(threads);
  for (auto& v : build_thread_residuals_) v.clear();
  build_thread_miss_.assign(threads, {0, 0});
  build_thread_gate_audit_.assign(threads, {0, 0});
  build_thread_tier0_miss_.assign(threads, {0, 0});

  // Frame-constant context for T0-D's corr.csv S column (H P- H^T + R) --
  // see WorldPointCov::body_point/rot_transpose/prior_cov_rp's doc
  // comment. Computed once here (not per point) and copied onto every
  // pt_world below; cheap regardless of whether logging is actually on
  // this call, so no separate gate is needed.
  const M3D rot_transpose = state_->rot().transpose();
  Eigen::Matrix<double, 6, 6> prior_cov_rp = Eigen::Matrix<double, 6, 6>::Zero();
  if (prior_cov_.rows() >= StateGroup::idxR() + 6 && prior_cov_.cols() >= StateGroup::idxR() + 6)
    prior_cov_rp = prior_cov_.block<6, 6>(StateGroup::idxR(), StateGroup::idxR());

  // CQ-35: an explicit `omp parallel` region wrapping an `omp for`, rather
  // than a combined `omp parallel for`, so that flushVarianceShareLog() can
  // be called once per thread AFTER the implicit barrier at the end of
  // `omp for` -- every thread is guaranteed to have finished its share of
  // the loop (and therefore finished appending to its own thread_local
  // buffer) before any thread reaches the flush call, and it is still the
  // SAME team of threads that did the logging, so draining "this thread's
  // buffer" here is well-defined (see voxelplane.h's doc comment on why
  // this must be called from inside the same parallel region).
  #pragma omp parallel num_threads(threads)
  {
    #pragma omp for schedule(static)
    for (int i = 0; i < n; ++i)
    {
      const PointXYZCov sensor_world = state_->toWorld(pts[i]);
      M3D pose_cov = M3D::Zero();
      if (include_state_uncertainty_in_gate) {
        if (pose_cov_overrides && i < static_cast<int>(pose_cov_overrides->size()))
          pose_cov = (*pose_cov_overrides)[i];
        else
          pose_cov = state_->poseCovAtPoint(pts[i].point, pts[i].t, false);   // R65: default path == poseCovAt
      }
      WorldPointCov pt_world{sensor_world.point, sensor_world.sensor_cov, pose_cov};
      pt_world.include_pose_cov_in_gate = include_state_uncertainty_in_gate;
      pt_world.body_point = pts[i].point;
      pt_world.rot_transpose = rot_transpose;
      pt_world.prior_cov_rp = prior_cov_rp;
      GateAudit gate_audit;
      pt_world.gate_audit = &gate_audit;
      Residual res{};
      bool tier0_had_plane = false;
      bool tier0_missed = true;
      // had_converged_neighbor: was a duplicate hasConvergedNeighbor(pt_world.point)
      // call on the else branch below (a second full neighborhood_size box
      // scan of the SAME box findPlaneResidual()'s own tier1/tier2 fallback
      // just scanned to look for a match) -- findPlaneResidual() now
      // accumulates the identical answer for free from cells it already
      // visits, via this out-param, instead of scanning the box twice on
      // every total miss.
      bool had_converged_neighbor = false;
      const bool admitted = voxel_map_->findPlaneResidual(
          pt_world, res, &tier0_had_plane, &had_converged_neighbor);
      if (admitted) {
        res.source_index = i;
        res.point_cross_normal = pts[i].point.cross(state_->rot().transpose() * res.normal);
        res.world_point = pt_world.point;
        res.sigma_squared += res.plane_var_term;
        if (opts_.bootstrap_inflate_scans > 0 && voxel_map_->frame_idx_ >= 1 &&
            voxel_map_->frame_idx_ <= static_cast<decltype(voxel_map_->frame_idx_)>(opts_.bootstrap_inflate_scans))
          res.sigma_squared *= opts_.bootstrap_inflate_factor;   // R65 (F-128)
        res.t = pts[i].t;   // diagnostic only (spline refinement removed)
        res.raw_body_point = pts[i].raw_body_point;  // CQ-50
        build_thread_residuals_[omp_get_thread_num()].push_back(res);
        if (allow_consistency_log && opts_.log_pair_corr_en)
          appendPairCorrResidual(voxel_map_->frame_idx_, res);
        tier0_missed = (res.match_tier != 0);
      } else {
        const int idx = had_converged_neighbor ? 1 : 0;
        ++build_thread_miss_[omp_get_thread_num()][idx];
      }
      auto& gate_counts = build_thread_gate_audit_[omp_get_thread_num()];
      if (gate_audit.reached_statistical_gate) ++gate_counts[0];
      if (!admitted && gate_audit.rejected_by_statistical_gate) ++gate_counts[1];
      if (tier0_missed) {
        const int idx0 = tier0_had_plane ? 1 : 0;
        ++build_thread_tier0_miss_[omp_get_thread_num()][idx0];
      }
    }
    // Implicit barrier at the end of `omp for` above already happened --
    // every thread's residual-building work (and its diagnostic logging,
    // gated behind log_variance_shares_en) is done by the time any thread
    // reaches here.
    flushVarianceShareLog();
    // CQ-38: same call-site reasoning as flushVarianceShareLog() above --
    // debugLogConsistencyCorr() (called from computeResidual(), same OMP
    // region) now buffers per-thread too. See voxelplane.h's doc comment.
    flushConsistencyCorrLog();
    // CQ-60: same once-per-thread obligation as flushVarianceShareLog()/
    // flushConsistencyCorrLog() above -- merges this thread's t_corr_scan
    // into the shared g_corr_scan, no lock held longer than one struct-add.
    mergeCorrScanThreadLocal();
    if (allow_consistency_log && opts_.log_pair_corr_en)
      flushPairCorrLog();
  }

  if (allow_consistency_log && opts_.log_pair_corr_en)
    logPairCorrPrior(voxel_map_->frame_idx_, prior_cov_rp);
  // CQ-60: single-threaded, after the parallel region above has fully
  // closed -- every thread's contribution is merged into g_corr_scan by
  // now (mergeCorrScanThreadLocal() ran for all of them, inside the
  // region, before its implicit join). One row per buildResiduals() call.
  flushCorrScanRow();

  residuals.clear();
  for (const auto& local : build_thread_residuals_)
    residuals.insert(residuals.end(), local.begin(), local.end());

  // One deliberately narrow join artifact for the voxel-model audit.  The
  // map snapshot is taken immediately after calibration frame 0 is inserted;
  // frame 1 is therefore the first scan that queries exactly that map.  A
  // stable node_id lets the offline analysis join every accepted residual to
  // the precise plane row without serialising allocator addresses.  Keep this
  // behind the same one-shot option as the snapshot so normal runs remain
  // byte-for-byte free of this potentially point-level diagnostic.
  const auto* snapshot_vm = dynamic_cast<const VoxelMap*>(voxel_map_.get());
  if (snapshot_vm && snapshot_vm->opts()->snapshot_post_calibration &&
      voxel_map_->frame_idx_ == 1) {
    static PersistentLogStream log("first_post_calibration_plane_associations.csv");
    bool first_call = false;
    std::ofstream& out = log.stream(&first_call);
    if (first_call) {
      out << "scan_id,iteration,source_index,point_time,plane_node_id,match_tier,"
             "residual,sigma_squared,plane_var_term,floor_term,sigma_diag_squared,"
             "s_prior_pose,world_x,world_y,world_z,normal_x,normal_y,normal_z,"
             "hrot_x,hrot_y,hrot_z,plane_jac_x,plane_jac_y,plane_jac_z\n";
    }
    out << std::setprecision(17);
    for (const auto& r : residuals) {
      out << voxel_map_->frame_idx_ << ',' << diagnostic_gn_iteration_ << ','
          << r.source_index << ',' << r.t << ',' << r.plane_node_id << ','
          << r.match_tier << ',' << r.r << ',' << r.sigma_squared << ','
          << r.plane_var_term << ',' << r.floor_term << ','
          << r.sigma_diag_squared << ',' << r.s_prior_pose << ','
          << r.world_point.x() << ',' << r.world_point.y() << ','
          << r.world_point.z() << ',' << r.normal.x() << ',' << r.normal.y()
          << ',' << r.normal.z() << ',' << r.point_cross_normal.x() << ','
          << r.point_cross_normal.y() << ',' << r.point_cross_normal.z() << ','
          << r.plane_jacobian.x() << ',' << r.plane_jacobian.y() << ','
          << r.plane_jacobian.z() << '\n';
    }
  }

  n_miss_coverage_ = 0;
  n_miss_mismatch_ = 0;
  for (const auto& m : build_thread_miss_) {
    n_miss_coverage_ += m[0];
    n_miss_mismatch_ += m[1];
  }
  n_statistical_gate_candidates_ = 0;
  n_statistical_gate_rejections_ = 0;
  for (const auto& counts : build_thread_gate_audit_) {
    n_statistical_gate_candidates_ += counts[0];
    n_statistical_gate_rejections_ += counts[1];
  }

  n_tier0_miss_coverage_ = 0;
  n_tier0_miss_mismatch_ = 0;
  for (const auto& m : build_thread_tier0_miss_) {
    n_tier0_miss_coverage_ += m[0];
    n_tier0_miss_mismatch_ += m[1];
  }
}

// History (242-246): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-242
void LioProcBase::solveSystem_cuda(const std::vector<Residual>& residuals) const {
  accumulateLioResidualsCuda(residuals, ekf_, cuda_buf_);
  // CQ-34: called UNCONDITIONALLY -- residual_redundancy.h's own header
  // already promises this is safe at mode=="off" (the module's internal
  // gate on ekf.HtH/Htz mutation is what keeps "off" byte-identical, not
  // this call site).
  redundancyAndRenorm(residuals);
  applyInfoScaleToEkf(residuals);
  ekf_.applyMeanUpdate(state_, prior_cov_, state_propagat_);
}

void LioProcBase::redundancyAndRenorm(const std::vector<Residual>& residuals) const
{
  // R70 (F-317): the redundancy correction as before, plus the optional renormalisation that holds the weakest position
  // eigenvalue of HtH at its independent-accumulator value. renorm == "off" (default) takes exactly the previous path.
  const bool renorm = opts_.residual_redundancy.renorm != "off" && opts_.residual_redundancy.on();
  Eigen::Matrix<double, 3, 3> Hpp_naive;
  if (renorm) Hpp_naive = ekf_.HtH.block<3, 3>(3, 3);
  { TimedScope ts(profiler_, "lio/ekf/redundancy");
    redundancy_stats_ = applyResidualRedundancyCorrection(residuals, opts_.residual_redundancy, ekf_); }
  if (!renorm) return;
  ++renorm_iterations_;
  const double ln = Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>>(Hpp_naive).eigenvalues()(0);
  const double lc = Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>>(
                        Eigen::Matrix<double, 3, 3>(ekf_.HtH.block<3, 3>(3, 3))).eigenvalues()(0);
  double c = 1.0;
  if (lc > 0.0 && ln > lc) {
    c = ln / lc;
    ekf_.HtH *= c;
    ekf_.Htz *= c;
    ++renorm_applied_;
    renorm_c_sum_ += c;
    renorm_c_max_ = std::max(renorm_c_max_, c);
  }
  static PersistentLogStream log("redundancy_renorm.csv");
  bool first = false;
  std::ofstream& o = log.stream(&first);
  if (first) o << "frame,iter,n_resid,hpp_min_eig_naive,hpp_min_eig_corrected,c\n";
  o << std::setprecision(10) << voxel_map_->frame_idx_ << ',' << diagnostic_gn_iteration_ << ',' << residuals.size() << ','
    << ln << ',' << lc << ',' << c << '\n';
}

void LioProcBase::applyInfoScaleToEkf(const std::vector<Residual>& residuals) const
{
  if (!opts_.info_scale.on()) return;   // default: HtH/Htz untouched (bit-identical)
  const InfoScaleStats st = applyBlockInfoScale(opts_.info_scale, ekf_.HtH, ekf_.Htz);
  ++info_scale_iterations_;
  info_scale_discarded_modes_ += st.discarded_modes;
  if (st.trace_before > 0.0) info_scale_trace_ratio_sum_ += st.trace_after / st.trace_before;
  static PersistentLogStream log("info_scale.csv");
  bool first = false;
  std::ofstream& o = log.stream(&first);
  if (first) o << "frame,iter,n_resid,F_rot,F_pos,trace_before,trace_after,gnorm_before,gnorm_after,discarded_modes\n";
  o << std::setprecision(10) << voxel_map_->frame_idx_ << ',' << diagnostic_gn_iteration_ << ',' << residuals.size() << ','
    << opts_.info_scale.rotation << ',' << opts_.info_scale.position << ',' << st.trace_before << ',' << st.trace_after << ','
    << st.gnorm_before << ',' << st.gnorm_after << ',' << st.discarded_modes << '\n';
}

void LioProcBase::recordStepCap() const
{
  if (!opts_.step_cap.on()) return;   // default: nothing evaluated, nothing written (bit-identical)
  const StepCapResult& r = ekf_.lastStepCap();
  ++step_cap_iterations_;
  step_cap_rot_raw_max_ = std::max(step_cap_rot_raw_max_, r.rot_norm);
  step_cap_pos_raw_max_ = std::max(step_cap_pos_raw_max_, r.pos_norm);
  if (r.bound) {
    ++step_cap_bound_;
    step_cap_scale_sum_ += r.scale;
    step_cap_scale_min_ = std::min(step_cap_scale_min_, r.scale);
  }
  static PersistentLogStream log("step_cap.csv");
  bool first = false;
  std::ofstream& o = log.stream(&first);
  if (first) o << "frame,iter,rot_raw,pos_raw,scale,bound,cap_rot,cap_pos\n";
  o << std::setprecision(10) << voxel_map_->frame_idx_ << ',' << diagnostic_gn_iteration_ << ',' << r.rot_norm << ','
    << r.pos_norm << ',' << r.scale << ',' << (r.bound ? 1 : 0) << ',' << opts_.step_cap.rotation_rad << ','
    << opts_.step_cap.position_m << '\n';
}

void LioProcBase::solveSystem(const std::vector<Residual>& residuals) const {
  accumulateLioResiduals(residuals, ekf_);

  // CQ-34: see solveSystem_cuda()'s comment above -- same fix, same reason.
  redundancyAndRenorm(residuals);

  applyInfoScaleToEkf(residuals);   // R70 (F-320): no-op unless lio/info_scale/mode=block

  // Mean-only update against the frame's FIXED prior (prior_cov_/
  // state_propagat_, snapshotted once in processLIO() before this frame's
  // iteration loop began) -- does not touch state_->cov().
  ekf_.applyMeanUpdate(state_, prior_cov_, state_propagat_);
  recordStepCap();   // R72 (F-403): no-op unless lio/ekf/step_cap/mode=hard
}

double LioProcBase::estimateStateCorrection(
  const std::vector<PointXYZCov>& pts,
  V3D &dtheta,
  V3D &dt,
  bool allow_consistency_log) {

  {
    TimedScope ts(profiler_, "lio/ekf/build_residuals");
    buildResiduals(pts, residuals_, allow_consistency_log);
  }
  if (residuals_.empty())
    return 0.0;

  double avg_res = 0.0;
  for (const auto& r : residuals_)
    avg_res += std::abs(r.r);
  avg_res /= residuals_.size();

  const V3D p_before_update = state_->pos();
  const V3D v_before_update = state_->vel();
  const M3D R_before_update = state_->rot();
  StateGroup state_before_update;
  state_before_update = *state_;
  {
    TimedScope ts(profiler_, "lio/ekf/solve");
    if (cuda_enable_)
      solveSystem_cuda(residuals_);
    else
      solveSystem(residuals_);
    dtheta = ekf_.dtheta;
    dt     = ekf_.dt;
  }

  if (voxel_map_->frame_idx_ >= 1 && voxel_map_->frame_idx_ <= opts_.first_frame_dump_frames &&
      diagnostic_gn_iteration_ >= 0) {
    const double t_abs = pts.empty() ? data_queues_->start_time : pts.front().t + data_queues_->start_time;
    logFirstFrameIekfSolve(
        voxel_map_->frame_idx_, 0, diagnostic_gn_iteration_, t_abs,
        firstFrameArchitectureName(), pts, residuals_, state_before_update, *state_,
        state_propagat_, prior_cov_, ekf_, ekf_.HtH, ekf_.Htz, opts_.first_frame_dump_frames);
  }

  if (opts_.update_diag_en && diagnostic_gn_iteration_ >= 0) {   // R69: write-only per-iteration LiDAR update record
    const double t_abs_ud = pts.empty() ? data_queues_->start_time : pts.front().t + data_queues_->start_time;
    logUpdateDiag(voxel_map_->frame_idx_, diagnostic_gn_iteration_, t_abs_ud, residuals_, ekf_);
  }
  if (opts_.live_point_cov_en && diagnostic_gn_iteration_ == 0)   // R69: write-only live-scan sensor covariance statistics
    logLivePointCov(voxel_map_->frame_idx_, pts);
  if (opts_.residual_geometry_en && diagnostic_gn_iteration_ == 0)   // R70: write-only residual geometry
    logResidualGeometry(voxel_map_->frame_idx_, residuals_);
  if (!opts_.residual_dump_frames.empty() && diagnostic_gn_iteration_ == 0 &&
      std::find(opts_.residual_dump_frames.begin(), opts_.residual_dump_frames.end(), voxel_map_->frame_idx_) != opts_.residual_dump_frames.end())
    logResidualDump(voxel_map_->frame_idx_, diagnostic_gn_iteration_, *state_, residuals_);   // R71: write-only per-residual dump

  if (opts_.log_debug_en && diagnostic_gn_iteration_ >= 0) {
    static PersistentLogStream log("iekf_factor_iteration.csv");
    bool first_call;
    std::ofstream& ofs = log.stream(&first_call);
    if (first_call) {
      ofs << "scan_id,iter,timestamp,p_before_x,p_before_y,p_before_z,p_after_x,p_after_y,p_after_z,"
             "v_before_x,v_before_y,v_before_z,v_after_x,v_after_y,v_after_z,dp_x,dp_y,dp_z,dp_norm,"
             "dtheta_x,dtheta_y,dtheta_z,dtheta_norm,rank,lambda_min,lambda_max";
      for (int i = 0; i < 36; ++i) ofs << ",A" << i;
      for (int i = 0; i < 6; ++i) ofs << ",b" << i;
      ofs << "\n";
    }
    const M3D dR = R_before_update.transpose() * state_->rot();
    Eigen::AngleAxisd aa(dR);
    const V3D dtheta_phys = aa.angle() * aa.axis();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> es(
        0.5 * (ekf_.HtH + ekf_.HtH.transpose()));
    int rank = 0;
    double lmin = 0.0, lmax = 0.0;
    if (es.info() == Eigen::Success && es.eigenvalues().size() > 0) {
      lmin = es.eigenvalues()(0);
      lmax = es.eigenvalues()(5);
      const double thresh = 1e-12 * std::max(1.0, std::abs(lmax));
      for (int i = 0; i < 6; ++i) if (es.eigenvalues()(i) > thresh) ++rank;
    }
    ofs << std::setprecision(12)
        << voxel_map_->frame_idx_ << ',' << diagnostic_gn_iteration_ << ','
        << (pts.empty() ? 0.0 : pts.front().t) << ','
        << p_before_update.x() << ',' << p_before_update.y() << ',' << p_before_update.z() << ','
        << state_->pos().x() << ',' << state_->pos().y() << ',' << state_->pos().z() << ','
        << v_before_update.x() << ',' << v_before_update.y() << ',' << v_before_update.z() << ','
        << state_->vel().x() << ',' << state_->vel().y() << ',' << state_->vel().z() << ','
        << (state_->pos()-p_before_update).x() << ',' << (state_->pos()-p_before_update).y() << ','
        << (state_->pos()-p_before_update).z() << ',' << (state_->pos()-p_before_update).norm() << ','
        << dtheta_phys.x() << ',' << dtheta_phys.y() << ',' << dtheta_phys.z() << ','
        << dtheta_phys.norm() << ',' << rank << ',' << lmin << ',' << lmax;
    for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) ofs << ',' << ekf_.HtH(r,c);
    for (int i = 0; i < 6; ++i) ofs << ',' << ekf_.Htz(i);
    ofs << "\n";
  }

  return avg_res;
}

bool LioProcBase::accumulateForCombined(MeasureGroup& mg, EkfUpdate& out, double& avg_res)
{
  if (voxel_map_->isEmpty()) return false;

  {
    TimedScope ts(profiler_, "lio/ekf/build_residuals");
    buildResiduals(mg.points, residuals_);
  }
  if (residuals_.empty()) return false;

  avg_res = 0.0;
  for (const auto& r : residuals_)
    avg_res += std::abs(r.r);
  avg_res /= residuals_.size();

  {
    TimedScope ts(profiler_, "lio/ekf/accumulate_combined");
    if (cuda_enable_)
      accumulateLioResidualsCuda(residuals_, out, cuda_buf_);
    else
      accumulateLioResiduals(residuals_, out);
  }
  return true;
}

void LioProcBase::logEigenspectrum18(int scan_id, double t_abs, const char* mode_label) const
{
  if (!opts_.log_eigenspectrum_en) return;
  if (state_->idxBG() < 0 || state_->idxBA() < 0 || state_->idxG() < 0) return;
  const Eigen::MatrixXd& P = state_->cov();
  if (P.rows() < 18 || P.cols() < 18) return;

  const Eigen::Matrix<double, 18, 18> P18 = P.block<18, 18>(0, 0);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 18, 18>> es(P18);
  const auto& eigvals = es.eigenvalues();     // ascending
  const auto& eigvecs = es.eigenvectors();

  // CQ-74 item 5a: which state block (rot/pos/vel/bg/ba/gravity) the
  // SMALLEST eigenvector loads onto, by largest-component -- StateGroup's
  // own fixed 18-dim layout (idxR=0,idxP=3,idxV=6,idxBG=9,idxBA=12,idxG=15,
  // each a contiguous 3-block).
  static const char* BLOCK_NAMES[6] = {"rot", "pos", "vel", "bg", "ba", "gravity"};
  const Eigen::VectorXd v_min = eigvecs.col(0);
  int max_block = 0;
  double max_block_norm = -1.0;
  for (int b = 0; b < 6; ++b) {
    const double block_norm = v_min.segment<3>(3 * b).squaredNorm();
    if (block_norm > max_block_norm) { max_block_norm = block_norm; max_block = b; }
  }

  static PersistentLogStream log("cq74_eigenspectrum.txt");
  bool first;
  std::ofstream& ofs = log.stream(&first);
  if (first) {
    ofs << "mode,scan_id,t_abs,lambda_min,lambda_max,weak_axis_block";
    for (int i = 0; i < 18; ++i) ofs << ",eig" << i;
    ofs << "\n";
  }
  ofs << mode_label << "," << scan_id << "," << std::fixed << std::setprecision(6) << t_abs
      << std::defaultfloat << "," << eigvals(0) << "," << eigvals(17) << ","
      << BLOCK_NAMES[max_block];
  for (int i = 0; i < 18; ++i) ofs << "," << eigvals(i);
  ofs << "\n";
  ofs.flush();
}

}  // namespace livo_recon
