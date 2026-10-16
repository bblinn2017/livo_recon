#include "livo_recon/utils/log/bag_clock.h"
#include "livo_recon/processing/lio_base.h"
#include "livo_recon/diagnostics/dump_rules.h"
#include "livo_recon/diagnostics/residual_dump_format.h"
#include "livo_recon/diagnostics/zsq_summary.h"
#include "livo_recon/utils/algo/symmetrize.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/log/config_resolve.h"
#include "livo_recon/utils/algo/omp_utils.h"
#include "livo_recon/utils/algo/hashing.h"
#include "livo_recon/map/voxelmap.h"
#include "livo_recon/lio/voxelplane.h"   // voxelPlaneInformationFitCount(), flushVarianceShareLog/flushConsistencyCorrLog
#include "livo_recon/lio/lio_accumulator.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"
#include "livo_recon/diagnostics/state_trace.h"

#include <atomic>
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
    int max_frame, bool slim = false)
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
       << (slim ? "slim=1\n" : "")   // R91C: only the slim form carries this line; the full form is byte-identical to R69..R91
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
  if (!slim) writeFirstFrameSolveMatrix(dump, "H_full", ekf.lastHFull());
  writeFirstFrameSolveMatrix(dump, "A_exact", ekf.lastA());
  if (!slim) writeFirstFrameSolveMatrix(dump, "K1_exact", ekf.lastK1());
  writeFirstFrameSolveMatrix(dump, "HtH", Eigen::MatrixXd(HtH));
  writeFirstFrameSolveVector(dump, "Htz", Eigen::VectorXd(Htz));
  writeFirstFrameSolveVector(dump, "prior_boxminus_propagated", ekf.lastVec());
  writeFirstFrameSolveVector(dump, "prior_boxminus_propagated_rp", ekf.lastVecRp());
  if (!slim) writeFirstFrameSolveMatrix(dump, "G_exact", ekf.lastG());
  writeFirstFrameSolveVector(dump, "measurement_term_exact", ekf.lastMeasurementTerm());
  writeFirstFrameSolveVector(dump, "prior_term_exact", ekf.lastPriorTerm());
  writeFirstFrameSolveVector(dump, "solution_exact", solution);
  for (size_t i = 0; i < residuals.size() && !slim; ++i) {   // R91C: the slim form has no per-residual lines (the residual dump carries them)
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
    o << "frame,iter,t_abs,t_rel_s,n_resid,sum_inv_sigma2,mean_sigma2,mean_plane_var_term,mean_base_sigma2,mean_r,mean_r2_over_sigma2";
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
  o << std::setprecision(10) << frame_idx << ',' << iteration << ',' << std::setprecision(17) << t_abs << ',' << tRelString(t_abs) << ',' << residuals.size()
    << ',' << s_inv << ',' << s_sig / dn << ',' << s_pv / dn << ',' << (s_sig - s_pv) / dn << ',' << s_r / dn << ',' << s_chi / dn;
  for (int i = 0; i < 6; ++i) for (int j = i; j < 6; ++j) o << ',' << ekf.HtH(i, j);
  for (int i = 0; i < 6; ++i) o << ',' << ekf.Htz(i);
  o << ',' << ekf.dtheta.x() << ',' << ekf.dtheta.y() << ',' << ekf.dtheta.z()
    << ',' << ekf.dt.x() << ',' << ekf.dt.y() << ',' << ekf.dt.z() << '\n';
  o.flush();
}

// R77 (F-451): write-only per-scan record of the rows the update used, first iteration of every frame.
// Row kinds (lio/line_residual.h LineRowKind): 0 plane; 1, 2 hard line u0, u1; 3, 4, 5 smooth u0, u1, u2. Every admitted point has exactly one u0-type row (kind 0, 1 or 3), so
// n_points = kind0 + kind1 + kind3 and n_pts_with_line_rows = kind1 + kind3. sum_fit_var_line / sum_tilt_var / sum_sigma2_line are over the rows of kind 1..5 (tilt is the part of the
// fit variance added by line_residual/variance = plane_equivalent, 0 otherwise); sum_sigma2_plane over the kind 0 rows. The last six columns are the cumulative R77 engagement counters
// of voxelPlaneR77Counters(): they read the map updates through the previous scan.
void logLineRows(int frame_idx, int iteration, const std::vector<Residual>& residuals)
{
  static PersistentLogStream log("line_rows_per_scan.csv");
  bool first = false;
  std::ofstream& o = log.stream(&first);
  if (first) {
    o << "scan_id,iter,n_rows,n_points,n_plane_rows,n_line_u0,n_line_u1,n_smooth_u0,n_smooth_u1,n_smooth_u2,n_pts_with_line_rows,sum_fit_var_line,sum_tilt_var,sum_sigma2_line,sum_sigma2_plane,"
         "gap_fits,gap_d1,gap_d2,tilt_voxels,tilt_floored_axes,tilt_fallbacks,n_line_dist\n";   // R80: n_line_dist appended
  }
  long cnt[7] = {0, 0, 0, 0, 0, 0, 0};
  double s_fit = 0.0, s_tilt = 0.0, s_sig_line = 0.0, s_sig_plane = 0.0;
  for (const auto& r : residuals) {
    const int k = (r.row_kind >= 0 && r.row_kind <= 6) ? r.row_kind : 0;
    ++cnt[k];
    if (k == 0) { s_sig_plane += r.sigma_squared; continue; }
    s_fit += r.plane_var_term; s_tilt += r.tilt_var_term; s_sig_line += r.sigma_squared;
  }
  const VoxelPlaneR77Counters c = voxelPlaneR77Counters();
  o << frame_idx << ',' << iteration << ',' << residuals.size() << ',' << (cnt[0] + cnt[1] + cnt[3]) << ',' << cnt[0] << ',' << cnt[1] << ',' << cnt[2] << ',' << cnt[3] << ',' << cnt[4] << ',' << cnt[5]
    << ',' << (cnt[1] + cnt[3]) << ',' << std::setprecision(10) << s_fit << ',' << s_tilt << ',' << s_sig_line << ',' << s_sig_plane
    << ',' << c.gap_fits << ',' << c.gap_d1 << ',' << c.gap_d2 << ',' << c.tilt_voxels << ',' << c.tilt_floored_axes << ',' << c.tilt_fallbacks << ',' << cnt[6] << '\n';
  o.flush();
}

// R95: write-only per-scan summary of z squared = r^2 / sigma_squared (diagnostics/zsq_summary.h defines the columns). g_zsq_rows counts the rows written (engagement counter).
std::atomic<long> g_zsq_rows{0};
void logZsqSummary(int frame_idx, int iteration, const std::vector<Residual>& residuals)
{
  static PersistentLogStream log("zsq_per_scan.csv");
  bool first = false;
  std::ofstream& o = log.stream(&first);
  if (first) o << zsq_summary::header();
  o << std::setprecision(10);
  zsq_summary::writeRow(o, frame_idx, iteration, residuals);
  o.flush();
  g_zsq_rows.fetch_add(1, std::memory_order_relaxed);
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


// R71 (F-363): write-only per-residual dump for the frames listed in log/lio/residual_dump/frames (iteration 0, i.e. at the
// propagated state), plus the linearisation state. Lets the information each residual claims (w_i h_i h_i^T) be compared
// offline with what the residuals jointly deliver (leverage, per-plane groups, split-half and cross-frame solutions).
// h_rot / h_pos are the rotation and position Jacobian rows (point_cross_normal, normal); w_i = 1 / sigma_squared.
void logResidualDump(int frame_idx, int iteration, const StateGroup& st, const std::vector<Residual>& residuals, bool voxel_info = false, bool line_geom = false, bool slim = false)
{
  static PersistentLogStream log("residual_dump.csv");
  static PersistentLogStream slog("residual_dump_state.csv");
  bool first = false, sfirst = false;
  std::ofstream& o = log.stream(&first);
  std::ofstream& so = slog.stream(&sfirst);
  if (first) residual_dump_format::writeHeader(o, voxel_info, line_geom, slim);   // R91C: the format moved to diagnostics/residual_dump_format.h (default byte-identical, tests/test_residual_dump_format_r91c.cpp)
  if (sfirst) so << "frame,iter,n_resid,px,py,pz,rvx,rvy,rvz\n";
  const V3D rv = Log(st.rot());
  so << frame_idx << ',' << iteration << ',' << residuals.size() << ',' << std::setprecision(17) << st.pos().x() << ','
     << st.pos().y() << ',' << st.pos().z() << ',' << rv.x() << ',' << rv.y() << ',' << rv.z() << '\n';
  o << std::setprecision(9);
  for (size_t i = 0; i < residuals.size(); ++i) residual_dump_format::writeRow(o, frame_idx, iteration, i, residuals[i], voxel_info, line_geom, slim);
  o.flush(); so.flush();
}
}  // namespace

void LioProcBase::logZsqIteration(int iteration) const
{
  if (!opts_.zsq_log_en) return;
  logZsqSummary(static_cast<int>(voxel_map_->frame_idx_), iteration, residuals_);
}

long LioProcBase::zsqRowsWritten() { return g_zsq_rows.load(std::memory_order_relaxed); }

void LioProcBase::dumpResidualsIfListed(int iteration) const
{
  if (opts_.residual_dump_frames.empty()) return;
  if (std::find(opts_.residual_dump_frames.begin(), opts_.residual_dump_frames.end(), voxel_map_->frame_idx_) ==
      opts_.residual_dump_frames.end())
    return;
  logResidualDump(voxel_map_->frame_idx_, iteration, *state_, residuals_, opts_.residual_dump_voxel_info, opts_.residual_dump_line_geometry, opts_.residual_dump_slim);
}

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
  paramWarn<bool>(pnh, "log/lio/debug",          opts_.log_debug_en,   false);
  paramWarn<bool>(pnh, "log/lio/pair_corr",      opts_.log_pair_corr_en, false);
  paramWarn<bool>(pnh, "log/lio/eigenspectrum",  opts_.log_eigenspectrum_en, false);
  paramWarn<bool>(pnh, "log/lio/consistency_scan", opts_.log_consistency_scan_en, false);
  paramWarn<bool>(pnh, "log/lio/nll", opts_.log_nll_en, false);
  paramWarn<bool>(pnh, "log/lio/update_diag", opts_.update_diag_en, false);          // R69: lio_update_diag.csv (write-only)
  paramWarn<bool>(pnh, "log/lio/live_point_cov", opts_.live_point_cov_en, false);    // R69: live_point_cov.csv (write-only)
  paramWarn<int>(pnh, "log/lio/first_frame_dump/frames", opts_.first_frame_dump_frames, 1);  // R69: solve dump for frames 1..K (1 = unchanged)
  {   // R91C: further frames for the solve snapshot; slim form refused unless the list is given. Empty / false is byte-identical.
    std::string ff_list;
    paramWarn<std::string>(pnh, "log/lio/first_frame_dump/frame_list", ff_list, std::string());
    opts_.first_frame_dump_list = dump_rules::parseFrameList(ff_list, "log/lio/first_frame_dump/frame_list");
    cfg.nested<bool>(!opts_.first_frame_dump_list.empty(), "log/lio/first_frame_dump/frame_list is non-empty", "log/lio/first_frame_dump/slim",
                     opts_.first_frame_dump_slim, false);
  }
  // R97A Q2: lio/dry_run_point_filter_num and lio/iekf_point_stream are removed.
  paramWarn<bool>(pnh, "log/nees/per_dof", opts_.nees_per_dof_en, false);
  paramWarn<int>(pnh, "log/nees/tier1_window_scans", opts_.nees_tier1_window_scans, 489);
  paramWarn<bool>(pnh, "log/state_trace/enable", opts_.state_trace_en, true);
  paramWarn<std::string>(pnh, "log/state_trace/run_id", opts_.state_trace_run_id, "run");
  paramWarn<bool>(pnh, "log/pose_cov_trace/enable", opts_.pose_cov_trace_en, false);   // R92
  paramWarn<double>(pnh, "log/pose_cov_trace/end_s", opts_.pose_cov_trace_end_s, -1.0);   // R92
  if (opts_.pose_cov_trace_end_s >= 0.0 && !opts_.pose_cov_trace_en)
    throw std::invalid_argument("log/pose_cov_trace/end_s needs log/pose_cov_trace/enable = true; the key is dead without it");
  paramWarn<int>(pnh, "log/state_trace/pre_imu_frames", opts_.state_trace_pre_imu_frames, 0);   // R86B: pre_imu rows for frames 1..K (0 = none)
  if (opts_.state_trace_pre_imu_frames < 0) throw std::invalid_argument("log/state_trace/pre_imu_frames must be >= 0");
  paramWarn<bool>(pnh, "cuda/enable",               cuda_enable_,          false);

  // History (134-136): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-134
  double range_err;
  paramWarn<double>(pnh, "imu/sensor/range_err", range_err, 0.05);
  opts_.deskew.sigma_r2 = range_err * range_err;
  double angle_err_deg;
  paramWarn<double>(pnh, "imu/sensor/angle_err_deg", angle_err_deg, 0.2);
  const double sin_angle_err = std::sin(std::max(1e-6, angle_err_deg * M_PI / 180.0));
  opts_.deskew.sigma_a2 = sin_angle_err * sin_angle_err;

  // R97A Q2: lio/residual_redundancy/* (mode, rho, renorm) is removed.

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
  // R91E / R91F: residual binning by constraint direction. mode direction = position bins (R91E), rotation = rotation bins (R91F), both = joint (R91F). Sub-keys exist only under the
  // modes that read them: axes and min_rows under any mode but off; shift_sd_m under direction and both; rot_shift_sd_rad under rotation and both. A key set outside its scope is an error
  // (nested()), so a rotation job that still carries shift_sd_m is refused. Each shift must be > 0 (0 would be inert: use mode=off).
  cfg.mode("lio/residual_binning/mode", opts_.residual_binning.mode, "off", { "off", "direction", "rotation", "both" });
  {
    const bool rb = opts_.residual_binning.mode != "off";
    const bool rbp = opts_.residual_binning.usesPosition(), rbr = opts_.residual_binning.usesRotation();
    cfg.nestedMode(rb, "lio/residual_binning/mode != off", "lio/residual_binning/axes", opts_.residual_binning.axes, "lattice13", { "lattice13" });
    cfg.nested<double>(rbp, "lio/residual_binning/mode = direction or both", "lio/residual_binning/shift_sd_m", opts_.residual_binning.shift_sd_m, 0.0006);
    cfg.nested<double>(rbr, "lio/residual_binning/mode = rotation or both", "lio/residual_binning/rot_shift_sd_rad", opts_.residual_binning.rot_shift_sd_rad, 1.0e-4);
    cfg.nested<int>(rb, "lio/residual_binning/mode != off", "lio/residual_binning/min_rows", opts_.residual_binning.min_rows, 20);
    if (rb) {
      if (rbp && (!std::isfinite(opts_.residual_binning.shift_sd_m) || !(opts_.residual_binning.shift_sd_m > 0.0)))
        throw std::invalid_argument("lio/residual_binning/shift_sd_m must be a finite number > 0 (a shift of 0 is the identity: use mode=off)");
      if (rbr && (!std::isfinite(opts_.residual_binning.rot_shift_sd_rad) || !(opts_.residual_binning.rot_shift_sd_rad > 0.0)))
        throw std::invalid_argument("lio/residual_binning/rot_shift_sd_rad must be a finite number > 0 (a shift of 0 is the identity: use mode=off)");
      if (opts_.residual_binning.min_rows < 1)
        throw std::invalid_argument("lio/residual_binning/min_rows must be >= 1");
      if (opts_.info_scale.mode != "off")
        cfg.requireCombination("lio/residual_binning/mode != off cannot be combined with lio/info_scale/mode != off: both correct the same HtH and Htz "
                               "and the order of the corrections is not specified.");
      std::string lr_rows = "per_normal";
      pnh.getParam("voxel_map/plane/line_residual/rows", lr_rows);
      bool combined_en = false;
      pnh.getParam("combined/enable", combined_en);
      if (combined_en)
        cfg.requireCombination("lio/residual_binning/mode != off cannot be combined with combined/enable = true: the joint LIO+VIO step builds its own HtH and Htz through "
                               "accumulateForCombined() and does not call the binning correction (the binning covers the LIO-only solve).");
      // R97A Q2: line residuals are always on now (voxel_map/plane/line_residual/enabled is removed).
      if (lr_rows == "distance")
        cfg.requireCombination("lio/residual_binning/mode != off needs voxel_map/plane/line_residual/rows = per_normal: the Jacobian direction of a distance row is set by the noise draw, "
                               "so it cannot key a bin (distance rows would be skipped, leaving the line voxels, which carry nearly all the information, uncorrected).");
    }
  }
  cfg.get<bool>("log/lio/scan", opts_.scan_log_en, false);   // R73 (F-414, F-415)
  {   // R71: per-residual dump frames, a comma-separated list of frame indices in ONE string key (job overrides carry scalars only); empty = off
    std::string dump_frames;
    paramWarn<std::string>(pnh, "log/lio/residual_dump/frames", dump_frames, std::string());
    opts_.residual_dump_frames.clear();
    std::stringstream ss(dump_frames);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (tok.empty()) continue;
      char* end = nullptr;
      const long v = std::strtol(tok.c_str(), &end, 10);
      if (end == tok.c_str() || *end != '\0' || v < 1)
        throw std::invalid_argument("log/lio/residual_dump/frames: '" + tok + "' is not a positive integer frame index (comma-separated list, e.g. \"1,2,3,10\")");
      opts_.residual_dump_frames.push_back(static_cast<int>(v));
    }
    // R73 (F-417): last-iteration dump; refused unless a frame list is given.
    cfg.nested<bool>(!opts_.residual_dump_frames.empty(), "log/lio/residual_dump/frames is non-empty", "log/lio/residual_dump/last_iter",
                     opts_.residual_dump_last_iter, false);
    // R86B: every-iteration dump; refused unless a frame list is given. Default false is byte-identical.
    cfg.nested<bool>(!opts_.residual_dump_frames.empty(), "log/lio/residual_dump/frames is non-empty", "log/lio/residual_dump/all_iters",
                     opts_.residual_dump_all_iters, false);
    // R81 (F-485 2B): two more columns (voxel_point_count, voxel_distinct_frames) in residual_dump.csv; refused unless a frame list is given. Default false is byte-identical.
    cfg.nested<bool>(!opts_.residual_dump_frames.empty(), "log/lio/residual_dump/frames is non-empty", "log/lio/residual_dump/voxel_info",
                     opts_.residual_dump_voxel_info, false);
    // R91C: line axis columns / 14-column slim form of residual_dump.csv; refused unless a frame list is given. Default false is byte-identical.
    cfg.nested<bool>(!opts_.residual_dump_frames.empty(), "log/lio/residual_dump/frames is non-empty", "log/lio/residual_dump/line_geometry",
                     opts_.residual_dump_line_geometry, false);
    cfg.nested<bool>(!opts_.residual_dump_frames.empty(), "log/lio/residual_dump/frames is non-empty", "log/lio/residual_dump/slim",
                     opts_.residual_dump_slim, false);
  }
  paramWarn<bool>(pnh, "log/lio/residual_geometry", opts_.residual_geometry_en, false);   // R70: residual_geometry.csv (write-only)
  paramWarn<bool>(pnh, "log/lio/line_rows", opts_.line_rows_log_en, false);   // R77 (F-451): line_rows_per_scan.csv (write-only)
  paramWarn<bool>(pnh, "log/lio/zsq", opts_.zsq_log_en, false);   // R95: zsq_per_scan.csv (write-only)

  // R97A Q2: lio/open_loop/mode and reset_period_s are removed.

  // R93A (Bryce 2026-10-10): lio/point_cov/insertion_source is removed (the posterior block is the only insertion source; `prior` is gone) and refused if set. The time-dependent point-covariance terms
  // (lio/point_cov/mode, lio/point_cov/window_s, lio/time_cov/*) were removed on 2026-10-07 and are refused too.
  // lio/gate_state_uncertainty (default true): whether the admission gate adds the point's pose covariance (see LioProcOptions::gate_state_uncertainty).
  cfg.get<bool>("lio/gate_state_uncertainty", opts_.gate_state_uncertainty, true);

  // R97A Q2: lio/bootstrap_inflate/scans, lio/bootstrap_inflate/factor and lio/startup_hold/scans are removed.

  // R97A Q2: lio/p_inflate/alpha, lio/p_floor/min_eig and lio/p_fading/lambda are removed.

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
           { "none", "state", "var_acc", "relpose_q", "relpose_full" });
  // R97A Q3 item 5: the rotation-warp branch (imu/undistort/rot_warp_scale, rot_warp_last_frame) is removed; 1.0 was the baseline path.
}

// CQ-49: the shared finish -- refuse anything left unclaimed in the given
// namespaces (the iekf passes its own set: removed namespaces, refused), record the accumulation
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


void LioProcBase::logPreImuState(const MeasureGroup& mg)
{
  if (!dump_rules::preImuTraceWanted(opts_.state_trace_pre_imu_frames, static_cast<long>(voxel_map_->frame_idx_))) return;
  logStateTrace("pre_imu", mg, 0, 0);
}

void LioProcBase::logStateTrace(const char* phase, const MeasureGroup& mg, int residual_count,
                                int completed_iterations)
{
  if (opts_.pose_cov_trace_en && std::string(phase) == "post_lio")   // R92: slim posterior pose trace, independent of state_trace_en
  {
    const double t_abs = mg.image.t + data_queues_->start_time;
    if (opts_.pose_cov_trace_end_s < 0.0 || !bagStartIsSet() || (t_abs - bagStart()) <= opts_.pose_cov_trace_end_s)
      writePoseCovTraceRow(opts_.state_trace_run_id, static_cast<int>(voxel_map_->frame_idx_), t_abs, *state_, residual_count, completed_iterations);
  }
  if (!opts_.state_trace_en) return;
  writeStateTraceRow(opts_.state_trace_run_id, static_cast<int>(voxel_map_->frame_idx_),
                     mg.image.t + data_queues_->start_time, phase, *state_, residual_count, completed_iterations,
                     ol_window_, ol_reset_now_);
}

// R97A Q2: openLoopFinishScan() is removed with lio/open_loop/*; ol_window_/ol_reset_now_
// (state_trace.csv's columns) stay at 0 always, matching every job's actual behavior.

void LioProcBase::buildResiduals(
  const std::vector<PointXYZCov>& pts,
  std::vector<Residual>& residuals,
  bool allow_consistency_log,
  const std::vector<M3D>* pose_cov_overrides) const {
  const bool include_state_uncertainty_in_gate = opts_.gate_state_uncertainty;   // R93A: was an argument (always true); now lio/gate_state_uncertainty
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
      PointXYZCov sensor_world = state_->toWorld(pts[i]);
      M3D pose_cov = M3D::Zero();
      if (include_state_uncertainty_in_gate) {
        if (pose_cov_overrides && i < static_cast<int>(pose_cov_overrides->size()))
          pose_cov = (*pose_cov_overrides)[i];
        else
          pose_cov = state_->poseCovAt(pts[i].point);
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
        // R97A Q2: lio/bootstrap_inflate/scans and factor are removed.
        res.t = pts[i].t;   // diagnostic only (spline refinement removed)
        res.raw_body_point = pts[i].raw_body_point;  // CQ-50
        build_thread_residuals_[omp_get_thread_num()].push_back(res);
        // R74 (F-430, F-431): the further rows of this point (line u1 row, smooth u1/u2 rows), each its own Residual, right after the u0 row.
        // Same point, same plane id, same world point; own direction, residual, variance (fit part in plane_var_term, as for the u0 row).
        for (int e = 0; e < res.n_extra; ++e) {
          Residual x = res;
          x.n_extra = 0;
          x.row_kind = res.extra_kind[e];
          x.normal = res.extra_normal[e];
          x.r = res.extra_r[e];
          x.plane_var_term = res.extra_fit_var[e];
          x.tilt_var_term = res.extra_tilt_var[e];   // R77 (F-453)
          x.sigma_squared = res.extra_sigma[e] + x.plane_var_term;
          x.sigma_diag_squared = res.extra_sigma[e] - (res.floor_term > 0.0 ? res.floor_term : 0.0);
          x.point_cross_normal = pts[i].point.cross(state_->rot().transpose() * x.normal);
          x.plane_jacobian = V3D::Zero();
          x.plane_covariance = M3D::Zero();
          build_thread_residuals_[omp_get_thread_num()].push_back(x);
        }
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
  // R97A Q2: lio/residual_redundancy/* (and the renorm it gated) is removed.
  applyResidualBinningToEkf(residuals);   // R91E/R91F: no-op unless lio/residual_binning/mode != off
  applyInfoScaleToEkf(residuals);
  ekf_.applyMeanUpdate(state_, prior_cov_, state_propagat_);
}

void LioProcBase::applyResidualBinningToEkf(const std::vector<Residual>& residuals) const
{
  if (!opts_.residual_binning.on()) return;   // default: HtH/Htz untouched (bit-identical)
  ResidualBinningStats st;
  { TimedScope ts(profiler_, "lio/ekf/binning");
    st = applyResidualBinning(residuals, opts_.residual_binning, ekf_.HtH, ekf_.Htz); }
  ++residual_binning_iterations_;
  residual_binning_rows_sum_ += st.n_corrected_rows;
  residual_binning_skipped_sum_ += st.n_skipped_rows;
  residual_binning_bins_sum_ += st.n_bins_active;
  residual_binning_trace_before_sum_ += st.trace_before;
  residual_binning_trace_after_sum_ += st.trace_after;
  residual_binning_rot_rows_sum_ += st.n_rot_corrected_rows;   // R91F
  residual_binning_rot_bins_sum_ += st.n_rot_bins_active;
  residual_binning_solve_failed_sum_ += st.solve_failed;
  const bool ext = opts_.residual_binning.mode != "direction";   // R91F: modes rotation and both write the extended columns; mode direction writes exactly the R91E files
  const bool use_p = opts_.residual_binning.usesPosition(), use_r = opts_.residual_binning.usesRotation();
  static PersistentLogStream log("residual_binning.csv");
  bool first = false;
  std::ofstream& o = log.stream(&first);
  if (first) {
    o << "frame,iter,n_resid,n_corrected_rows,n_skipped_rows,n_bins_active,shift_sd_m,min_rows,trace_before,trace_after,gnorm_before,gnorm_after";
    if (ext) o << ",mode,rot_shift_sd_rad,n_rot_corrected_rows,n_rot_bins_active,solve_failed";   // shift_sd_m / rot_shift_sd_rad are written as 0 when the mode does not use that family
    o << '\n';
  }
  o << std::setprecision(17) << voxel_map_->frame_idx_ << ',' << diagnostic_gn_iteration_ << ',' << residuals.size() << ',' << st.n_corrected_rows << ',' << st.n_skipped_rows << ','
    << st.n_bins_active << ',' << (use_p ? opts_.residual_binning.shift_sd_m : 0.0) << ',' << opts_.residual_binning.min_rows << ',' << st.trace_before << ',' << st.trace_after << ','
    << st.gnorm_before << ',' << st.gnorm_after;
  if (ext)
    o << ',' << opts_.residual_binning.mode << ',' << (use_r ? opts_.residual_binning.rot_shift_sd_rad : 0.0) << ',' << st.n_rot_corrected_rows << ',' << st.n_rot_bins_active << ',' << st.solve_failed;
  o << '\n';
  // Per-bin sums (v, s, t) for the frames that get a solve snapshot, so the correction can be recomputed offline from the dumped rows and the uncorrected HtH, Htz recovered.
  if (diagnostic_gn_iteration_ >= 0 &&
      dump_rules::firstFrameSolveDumpMode(voxel_map_->frame_idx_, opts_.first_frame_dump_frames, opts_.first_frame_dump_list, opts_.first_frame_dump_slim) != 0) {
    static PersistentLogStream blog("residual_binning_bins.csv");
    bool bfirst = false;
    std::ofstream& b = blog.stream(&bfirst);
    if (bfirst) b << "frame,iter,bin,ax,ay,az,rows,applied,s,t,v0,v1,v2,v3,v4,v5" << (ext ? ",family" : "") << '\n';
    const double(&axt)[residual_binning::kAxes][3] = residual_binning::axisTable();
    b << std::setprecision(17);
    for (int fam = 0; fam < 2; ++fam) {   // 0 = position bins (modes direction, both), 1 = rotation bins (modes rotation, both)
      if (fam == 0 && !use_p) continue;
      if (fam == 1 && !use_r) continue;
      for (int k = 0; k < residual_binning::kAxes; ++k) {
        const residual_binning::BinOut& bo = fam == 0 ? st.bins[k] : st.rot_bins[k];
        b << voxel_map_->frame_idx_ << ',' << diagnostic_gn_iteration_ << ',' << k << ',' << axt[k][0] << ',' << axt[k][1] << ',' << axt[k][2] << ',' << bo.rows << ',' << (bo.applied ? 1 : 0)
          << ',' << bo.s << ',' << bo.t;
        for (int j = 0; j < 6; ++j) b << ',' << bo.v[j];
        if (ext) b << ',' << (fam == 0 ? 'p' : 'r');
        b << '\n';
      }
    }
    if (use_p && use_r) {   // R91F: the coupling X_kl of the two families (sum over the rows of position bin k and rotation bin l of w u_pos u_rot)
      static PersistentLogStream xlog("residual_binning_cross.csv");
      bool xfirst = false;
      std::ofstream& x = xlog.stream(&xfirst);
      if (xfirst) x << "frame,iter,pos_bin,rot_bin,x\n";
      x << std::setprecision(17);
      for (int k = 0; k < residual_binning::kAxes; ++k)
        for (int l = 0; l < residual_binning::kAxes; ++l)
          x << voxel_map_->frame_idx_ << ',' << diagnostic_gn_iteration_ << ',' << k << ',' << l << ',' << st.cross[k][l] << '\n';
    }
  }
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

void LioProcBase::solveSystem(const std::vector<Residual>& residuals) const {
  accumulateLioResiduals(residuals, ekf_);

  applyResidualBinningToEkf(residuals);   // R91E/R91F: no-op unless lio/residual_binning/mode != off
  applyInfoScaleToEkf(residuals);   // R70 (F-320): no-op unless lio/info_scale/mode=block

  // Mean-only update against the frame's FIXED prior (prior_cov_/
  // state_propagat_, snapshotted once in processLIO() before this frame's
  // iteration loop began) -- does not touch state_->cov().
  ekf_.applyMeanUpdate(state_, prior_cov_, state_propagat_);
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

  const int ff_mode = dump_rules::firstFrameSolveDumpMode(voxel_map_->frame_idx_, opts_.first_frame_dump_frames, opts_.first_frame_dump_list, opts_.first_frame_dump_slim);   // R91C
  if (ff_mode != 0 && diagnostic_gn_iteration_ >= 0) {
    const double t_abs = pts.empty() ? data_queues_->start_time : pts.front().t + data_queues_->start_time;
    logFirstFrameIekfSolve(
        voxel_map_->frame_idx_, 0, diagnostic_gn_iteration_, t_abs,
        firstFrameArchitectureName(), pts, residuals_, state_before_update, *state_,
        state_propagat_, prior_cov_, ekf_, ekf_.HtH, ekf_.Htz,
        (voxel_map_->frame_idx_ <= opts_.first_frame_dump_frames) ? opts_.first_frame_dump_frames : voxel_map_->frame_idx_,   // R91C: a listed frame beyond 1..K passes its own index so the guard inside admits it
        ff_mode == 2);
  }

  if (opts_.update_diag_en && diagnostic_gn_iteration_ >= 0) {   // R69: write-only per-iteration LiDAR update record
    const double t_abs_ud = pts.empty() ? data_queues_->start_time : pts.front().t + data_queues_->start_time;
    logUpdateDiag(voxel_map_->frame_idx_, diagnostic_gn_iteration_, t_abs_ud, residuals_, ekf_);
  }
  if (opts_.live_point_cov_en && diagnostic_gn_iteration_ == 0)   // R69: write-only live-scan sensor covariance statistics
    logLivePointCov(voxel_map_->frame_idx_, pts);
  if (opts_.residual_geometry_en && diagnostic_gn_iteration_ == 0)   // R70: write-only residual geometry
    logResidualGeometry(voxel_map_->frame_idx_, residuals_);
  if (opts_.line_rows_log_en && diagnostic_gn_iteration_ == 0)   // R77 (F-451): write-only per-scan row counts by kind
    logLineRows(voxel_map_->frame_idx_, diagnostic_gn_iteration_, residuals_);
  if (opts_.zsq_log_en && diagnostic_gn_iteration_ == 0)   // R95: write-only z squared summary, iteration 0 (the last iteration is written by the IEKF loop through logZsqIteration)
    logZsqIteration(0);
  if (!opts_.residual_dump_frames.empty() && dump_rules::residualDumpIterationWanted(opts_.residual_dump_all_iters, diagnostic_gn_iteration_) &&
      std::find(opts_.residual_dump_frames.begin(), opts_.residual_dump_frames.end(), voxel_map_->frame_idx_) != opts_.residual_dump_frames.end())
    logResidualDump(voxel_map_->frame_idx_, diagnostic_gn_iteration_, *state_, residuals_, opts_.residual_dump_voxel_info, opts_.residual_dump_line_geometry, opts_.residual_dump_slim);   // R71: write-only per-residual dump

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
        symmetrized(ekf_.HtH));
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
