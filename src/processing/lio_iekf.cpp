#include "livo_recon/processing/lio_iekf.h"
#include "livo_recon/diagnostics/dump_rules.h"
#include "livo_recon/utils/algo/symmetrize.h"
#include "livo_recon/diagnostics/stationary_diagnostic_writer.h"
#include "livo_recon/processing/imu_processing.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/log/config_resolve.h"
#include "livo_recon/utils/log/renamed_keys.h"
#include "livo_recon/diagnostics/log/consistency_log.h"
#include "livo_recon/utils/algo/omp_utils.h"
#include "livo_recon/utils/algo/hashing.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"
#include "livo_recon/utils/algo/math.h"
#include "livo_recon/map/voxelmap.h"
#include "livo_recon/lio/voxelplane.h"   // voxelPlaneInformationFitCount()
#include "livo_recon/lio/lio_accumulator.h"
#include "livo_recon/utils/algo/covariance_math.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <cstdint>

namespace livo_recon
{

namespace
{

static uint64_t iekfFirstFrameHashDouble(uint64_t h, double v)
{
  const auto* p = reinterpret_cast<const unsigned char*>(&v);
  for (size_t i = 0; i < sizeof(v); ++i) { h ^= static_cast<uint64_t>(p[i]); h *= 1099511628211ULL; }
  return h;
}

static uint64_t iekfFirstFrameHashPoints(const std::vector<PointXYZCov>& pts)
{
  uint64_t h = 1469598103934665603ULL;
  for (const auto& pt : pts) {
    h = iekfFirstFrameHashDouble(h, pt.point.x()); h = iekfFirstFrameHashDouble(h, pt.point.y()); h = iekfFirstFrameHashDouble(h, pt.point.z()); h = iekfFirstFrameHashDouble(h, pt.t);
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) h=iekfFirstFrameHashDouble(h,pt.sensor_cov(r,c));
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) h=iekfFirstFrameHashDouble(h,pt.pos_cov(r,c));
  }
  return h;
}

static uint64_t iekfFirstFrameHashResiduals(const std::vector<Residual>& residuals)
{
  uint64_t h = 1469598103934665603ULL;
  for (const auto& r : residuals) {
    h = iekfFirstFrameHashDouble(h,r.r);
    for(int i=0;i<3;++i) h=iekfFirstFrameHashDouble(h,r.normal(i));
    for(int i=0;i<3;++i) h=iekfFirstFrameHashDouble(h,r.world_point(i));
    h=iekfFirstFrameHashDouble(h,r.t); h=iekfFirstFrameHashDouble(h,r.sigma_squared);
  }
  return h;
}

static std::string iekfFirstFrameHex(uint64_t h) { std::ostringstream oss; oss << std::hex << h; return oss.str(); }

// ADDED (symbol/scope audit): the supplied patch's instructions explicitly
// state "For the first map-backed frame, the coupled AND iekf paths
// now dump [the covariance budget]," but the patch itself only defined and
// wired logFirstFrameCovarianceBudget() into lio_coupled.cpp -- it is
// never mentioned in the patch's own file list for lio_iekf.cpp. This
// is a mirrored, byte-identical port of lio_coupled.cpp's own function
// (same computation, same columns, no new derivation) so the iekf
// path satisfies the instructions' explicit
// requirement rather than silently omitting it.
static double firstFrameCovMinEig(const Eigen::MatrixXd& X)
{
  if (X.rows() == 0 || X.cols() == 0 || X.rows() != X.cols()) return std::numeric_limits<double>::quiet_NaN();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(symmetrized(X));
  return es.info() == Eigen::Success ? es.eigenvalues().minCoeff() : std::numeric_limits<double>::quiet_NaN();
}
static double firstFrameCovMaxEig(const Eigen::MatrixXd& X)
{
  if (X.rows() == 0 || X.cols() == 0 || X.rows() != X.cols()) return std::numeric_limits<double>::quiet_NaN();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(symmetrized(X));
  return es.info() == Eigen::Success ? es.eigenvalues().maxCoeff() : std::numeric_limits<double>::quiet_NaN();
}
static void logFirstFrameCovarianceBudget(const char* architecture, const char* phase, int scan_id, int iteration,
                                          const MeasureGroup& mg, const StateGroup& state, double t_abs)
{
  if (scan_id != 0) return;
  static PersistentLogStream csv_log("pose_control_first_frame_covariance_budget.csv");
  static PersistentLogStream dump_log("pose_control_first_frame_covariance_budget.txt");
  bool first = false;
  std::ofstream& csv = csv_log.stream(&first);
  if (first) csv << "architecture,phase,scan_id,iteration,t_abs,t_rel,dim,p_before_trace,p_after_imu_trace,p_after_lio_trace,delta_p_trace_imu,delta_p_trace_lio,min_eig_before,max_eig_before,min_eig_after_imu,max_eig_after_imu,min_eig_after_lio,max_eig_after_lio,q_eff_trace,fpf_trace,q_reconstruction_rel_error,p_rp_norm,p_rv_norm,p_pv_norm\n";
  const Eigen::MatrixXd& Pb = mg.cov_before_imu;
  const Eigen::MatrixXd& Pi = mg.cov_after_imu;
  const Eigen::MatrixXd Pl = mg.cov_after_lio.rows() ? mg.cov_after_lio : state.cov();
  const Eigen::MatrixXd dPi = (Pb.rows()==Pi.rows() && Pb.cols()==Pi.cols()) ? Pi-Pb : Eigen::MatrixXd();
  const Eigen::MatrixXd dPl = (Pi.rows()==Pl.rows() && Pi.cols()==Pl.cols()) ? Pl-Pi : Eigen::MatrixXd();
  Eigen::MatrixXd p_before_q,fpf,qeff,p_after_q;
  const bool have_q = (std::string(phase) == "pre_update") && imuProcQhatPeekAll(p_before_q,fpf,qeff,p_after_q);
  const double qerr = have_q && p_after_q.rows()==Pi.rows() && p_after_q.cols()==Pi.cols()
      ? (fpf+qeff-p_after_q).norm()/std::max(1e-300,p_after_q.norm()) : std::numeric_limits<double>::quiet_NaN();
  double rp=std::numeric_limits<double>::quiet_NaN(), rr=rp, rv=rp, pv=rp;
  if (Pi.rows() >= StateGroup::idxV()+3) {
    const int ir=StateGroup::idxR(), ip=StateGroup::idxP(), iv=StateGroup::idxV();
    rp=Pi.block<3,3>(ir,ip).norm(); rr=Pi.block<3,3>(ir,iv).norm(); rv=Pi.block<3,3>(ip,iv).norm(); pv=Pi.block<3,3>(ip,ir).norm();
  }
  const int ip = StateGroup::idxP();
  const auto positionTrace = [ip](const Eigen::MatrixXd& P) {
    return P.rows() >= ip + 3 && P.cols() >= ip + 3
        ? P.block<3,3>(ip, ip).trace() : 0.0;
  };
  csv << std::setprecision(17) << architecture << ',' << phase << ',' << scan_id << ',' << iteration << ',' << t_abs << ',' << mg.image.t << ','
      << Pi.rows() << ',' << positionTrace(Pb) << ',' << positionTrace(Pi) << ',' << positionTrace(Pl) << ','
      << positionTrace(dPi) << ',' << positionTrace(dPl) << ','
      << firstFrameCovMinEig(Pb) << ',' << firstFrameCovMaxEig(Pb) << ',' << firstFrameCovMinEig(Pi) << ',' << firstFrameCovMaxEig(Pi) << ','
      << firstFrameCovMinEig(Pl) << ',' << firstFrameCovMaxEig(Pl) << ',' << (have_q?qeff.trace():std::numeric_limits<double>::quiet_NaN()) << ','
      << (have_q?fpf.trace():std::numeric_limits<double>::quiet_NaN()) << ',' << qerr << ',' << rp << ',' << rr << ',' << rv << '\n';
  csv.flush();
  std::ofstream& dump=dump_log.stream();
  dump << "=== covariance_budget " << architecture << " " << phase << " iter=" << iteration << " t_abs=" << t_abs << " ===\n";
  dump << "P_before_IMU\n" << Pb << "\nP_after_IMU\n" << Pi << "\nP_after_LIO\n" << Pl << "\n";
  dump << "delta_P_IMU\n" << dPi << "\ndelta_P_LIO\n" << dPl << "\n";
  if (have_q) { Eigen::MatrixXd q_info = generalPseudoInverse(qeff, 1e-12); dump << "Q_eff\n" << qeff << "\nQ_eff_information_pinv\n" << q_info << "\nF_P_Ft\n" << fpf << "\nQhat_P_before\n" << p_before_q << "\nQhat_P_after\n" << p_after_q << "\n"; }
  dump.flush();
}

static void logFirstFrameStateChain(const char* architecture, const char* phase, int scan_id, int iteration,
                                    const MeasureGroup& mg, const StateGroup& lio_state, double t_abs)
{
  if (scan_id != 0) return;
  static PersistentLogStream log("pose_control_first_frame_state_chain.csv");
  bool first;
  std::ofstream& ofs = log.stream(&first);
  if (first) {
    ofs << "architecture,phase,scan_id,iteration,t_abs,t_rel,"
           "x0_px,x0_py,x0_pz,ximu_px,ximu_py,ximu_pz,xlio_px,xlio_py,xlio_pz,"
           "x0_r_x,x0_r_y,x0_r_z,ximu_r_x,ximu_r_y,ximu_r_z,xlio_r_x,xlio_r_y,xlio_r_z,"
           "x0_vx,x0_vy,x0_vz,ximu_vx,ximu_vy,ximu_vz,xlio_vx,xlio_vy,xlio_vz,"
           "imu_dp_x,imu_dp_y,imu_dp_z,lio_dp_x,lio_dp_y,lio_dp_z,delta_lio_from_imu_x,delta_lio_from_imu_y,delta_lio_from_imu_z,"
           "imu_dr_x,imu_dr_y,imu_dr_z,lio_dr_x,lio_dr_y,lio_dr_z,delta_lio_rot_x,delta_lio_rot_y,delta_lio_rot_z,"
           "imu_dv_x,imu_dv_y,imu_dv_z,lio_dv_x,lio_dv_y,lio_dv_z,delta_lio_vel_x,delta_lio_vel_y,delta_lio_vel_z,"
           "ideal_stationary_dp_x,ideal_stationary_dp_y,ideal_stationary_dp_z,"
           "ideal_stationary_dr_x,ideal_stationary_dr_y,ideal_stationary_dr_z,"
           "pos_residual_norm_after_lio,rot_residual_norm_after_lio,vel_residual_norm_after_lio,"
           "pos_correction_norm,rot_correction_norm,vel_correction_norm,"
           "pos_correction_alignment_to_ideal,rot_correction_alignment_to_ideal,"
           "pos_correction_fraction_of_imu_error,rot_correction_fraction_of_imu_error\n";
  }

  const V3D x0p = mg.pos_before_imu;
  const V3D ximp = mg.pos_after_imu;
  const V3D xliop = lio_state.pos();
  const V3D x0r = Log(mg.rot_before_imu);
  const V3D ximr = Log(mg.rot_after_imu);
  const V3D xlior = Log(lio_state.rot());
  const V3D x0v = mg.vel_before_imu;
  const V3D ximv = mg.vel_after_imu;
  const V3D xliov = lio_state.vel();

  const V3D imu_dp = ximp - x0p;
  const V3D lio_dp = xliop - x0p;
  const V3D delta_lio_dp = xliop - ximp;
  const V3D imu_dr = Log(mg.rot_before_imu.transpose() * mg.rot_after_imu);
  const V3D lio_dr = Log(mg.rot_before_imu.transpose() * lio_state.rot());
  const V3D delta_lio_dr = Log(mg.rot_after_imu.transpose() * lio_state.rot());
  const V3D imu_dv = ximv - x0v;
  const V3D lio_dv = xliov - x0v;
  const V3D delta_lio_dv = xliov - ximv;
  const V3D ideal_dp = -imu_dp;
  const V3D ideal_dr = -imu_dr;
  const double eps = 1e-12;
  const auto align = [&](const V3D& a, const V3D& b) -> double {
    const double na = a.norm(), nb = b.norm();
    return (na > eps && nb > eps) ? a.dot(b) / (na * nb) : 0.0;
  };
  const double pos_frac = (imu_dp.norm() > eps) ? delta_lio_dp.norm() / imu_dp.norm() : 0.0;
  const double rot_frac = (imu_dr.norm() > eps) ? delta_lio_dr.norm() / imu_dr.norm() : 0.0;

  ofs << architecture << ',' << phase << ',' << scan_id << ',' << iteration << ',' << std::setprecision(17) << t_abs << ',' << mg.image.t << ','
      << x0p.x() << ',' << x0p.y() << ',' << x0p.z() << ','
      << ximp.x() << ',' << ximp.y() << ',' << ximp.z() << ','
      << xliop.x() << ',' << xliop.y() << ',' << xliop.z() << ','
      << x0r.x() << ',' << x0r.y() << ',' << x0r.z() << ','
      << ximr.x() << ',' << ximr.y() << ',' << ximr.z() << ','
      << xlior.x() << ',' << xlior.y() << ',' << xlior.z() << ','
      << x0v.x() << ',' << x0v.y() << ',' << x0v.z() << ','
      << ximv.x() << ',' << ximv.y() << ',' << ximv.z() << ','
      << xliov.x() << ',' << xliov.y() << ',' << xliov.z() << ','
      << imu_dp.x() << ',' << imu_dp.y() << ',' << imu_dp.z() << ','
      << lio_dp.x() << ',' << lio_dp.y() << ',' << lio_dp.z() << ','
      << delta_lio_dp.x() << ',' << delta_lio_dp.y() << ',' << delta_lio_dp.z() << ','
      << imu_dr.x() << ',' << imu_dr.y() << ',' << imu_dr.z() << ','
      << lio_dr.x() << ',' << lio_dr.y() << ',' << lio_dr.z() << ','
      << delta_lio_dr.x() << ',' << delta_lio_dr.y() << ',' << delta_lio_dr.z() << ','
      << imu_dv.x() << ',' << imu_dv.y() << ',' << imu_dv.z() << ','
      << lio_dv.x() << ',' << lio_dv.y() << ',' << lio_dv.z() << ','
      << delta_lio_dv.x() << ',' << delta_lio_dv.y() << ',' << delta_lio_dv.z() << ','
      << ideal_dp.x() << ',' << ideal_dp.y() << ',' << ideal_dp.z() << ','
      << ideal_dr.x() << ',' << ideal_dr.y() << ',' << ideal_dr.z() << ','
      << lio_dp.norm() << ',' << lio_dr.norm() << ',' << lio_dv.norm() << ','
      << delta_lio_dp.norm() << ',' << delta_lio_dr.norm() << ',' << delta_lio_dv.norm() << ','
      << align(delta_lio_dp, ideal_dp) << ',' << align(delta_lio_dr, ideal_dr) << ','
      << pos_frac << ',' << rot_frac << '\n';
  ofs.flush();
}

static void logIekfFirstFrame(const char* phase, int scan_id, int iteration, const MeasureGroup& mg,
                                   const StateGroup& state, double t_abs, int n_residuals, uint64_t point_hash,
                                   uint64_t residual_hash, int map_points, int active_voxels,
                                   const V3D& reference_pos, const M3D& reference_rot)
{
  if (scan_id != 0) return;
  static PersistentLogStream log("pose_control_first_frame.csv");
  bool first; std::ofstream& ofs=log.stream(&first);
  if(first) ofs<<"architecture,phase,scan_id,iteration,t_abs,t_rel,n_points,n_residuals,map_points,active_voxels,point_hash,residual_hash,p_x,p_y,p_z,rlog_x,rlog_y,rlog_z,v_x,v_y,v_z,position_norm,rotation_norm,reference_p_x,reference_p_y,reference_p_z,reference_rlog_x,reference_rlog_y,reference_rlog_z,relative_p_x,relative_p_y,relative_p_z,relative_rlog_x,relative_rlog_y,relative_rlog_z,x0_p_x,x0_p_y,x0_p_z,x0_rlog_x,x0_rlog_y,x0_rlog_z,x0_v_x,x0_v_y,x0_v_z,ximu_p_x,ximu_p_y,ximu_p_z,ximu_rlog_x,ximu_rlog_y,ximu_rlog_z,ximu_v_x,ximu_v_y,ximu_v_z\n";
  const V3D rlog=Log(state.rot());
  const V3D reference_rlog=Log(reference_rot);
  const V3D relative_rlog=Log(reference_rot.transpose()*state.rot());
  const V3D relative_pos=state.pos()-reference_pos;
  const double t_rel=mg.image.t;
  const V3D x0_rlog=Log(mg.rot_before_imu); const V3D ximu_rlog=Log(mg.rot_after_imu);
  ofs<<"iekf,"<<phase<<","<<scan_id<<","<<iteration<<","<<std::setprecision(12)<<t_abs<<","<<t_rel<<","<<mg.points.size()<<","<<n_residuals<<","<<map_points<<","<<active_voxels<<","<<iekfFirstFrameHex(point_hash)<<","<<iekfFirstFrameHex(residual_hash)<<","<<state.pos().x()<<","<<state.pos().y()<<","<<state.pos().z()<<","<<rlog.x()<<","<<rlog.y()<<","<<rlog.z()<<","<<state.vel().x()<<","<<state.vel().y()<<","<<state.vel().z()<<","<<state.pos().norm()<<","<<rlog.norm()<<","<<reference_pos.x()<<","<<reference_pos.y()<<","<<reference_pos.z()<<","<<reference_rlog.x()<<","<<reference_rlog.y()<<","<<reference_rlog.z()<<","<<relative_pos.x()<<","<<relative_pos.y()<<","<<relative_pos.z()<<","<<relative_rlog.x()<<","<<relative_rlog.y()<<","<<relative_rlog.z()<<","<<mg.pos_before_imu.x()<<","<<mg.pos_before_imu.y()<<","<<mg.pos_before_imu.z()<<","<<x0_rlog.x()<<","<<x0_rlog.y()<<","<<x0_rlog.z()<<","<<mg.vel_before_imu.x()<<","<<mg.vel_before_imu.y()<<","<<mg.vel_before_imu.z()<<","<<mg.pos_after_imu.x()<<","<<mg.pos_after_imu.y()<<","<<mg.pos_after_imu.z()<<","<<ximu_rlog.x()<<","<<ximu_rlog.y()<<","<<ximu_rlog.z()<<","<<mg.vel_after_imu.x()<<","<<mg.vel_after_imu.y()<<","<<mg.vel_after_imu.z()<<"\n";
  ofs.flush();
}

// Debug trace of LIO's per-measure-group state correction and covariance,
// for diagnosing where trajectory divergence originates (see imu_processing
// .cpp's debugLogImu() for the IMU-side counterpart). Truncated at the
// start of each process (first call), appended thereafter. Absolute (bag/
// wall-clock) timestamps throughout, matching evo_processing.cpp's
// /tmp/evo.txt, imu_processing.cpp's /tmp/imu.txt, and FAST-LIVO2's own
// logs, so all of these can be compared directly against each other.
// Remove once done debugging.
// CQ-36: PersistentLogStream -- see its own doc comment for the CQ-35
// regression this fixes -- applied to every debugLogXxx() helper in this
// file (see voxelmap.cpp's debugLogFrameStats() for the same fix applied
// project-wide).
void debugLogLio(const std::string& msg)
{
  static PersistentLogStream log("lio.txt");
  std::ofstream& ofs = log.stream();
  ofs << msg << "\n";
  ofs.flush();
}

// History (40-42): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-40
void debugLogIterError(const std::string& msg)
{
  static PersistentLogStream log("iter_error.txt");
  std::ofstream& ofs = log.stream();
  ofs << msg << "\n";
  ofs.flush();
}

// R97A Q2: debugLogLioDryRun() / lio_dryrun.txt are removed with runDryRunShadowPass().

// History (63-70): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-63

// History (72-75): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-72

// History (77-89): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-77
void debugLogQhat(int scan_id, double t_abs, const Eigen::VectorXd& dx,
                  const Eigen::VectorXd& expected_diag,
                  const Eigen::VectorXd& acw_diag)
{
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  const int n = static_cast<int>(dx.size());
  // CQ-36: PersistentLogStream -- see its own doc comment for the CQ-35
  // regression this fixes.
  static PersistentLogStream log("qhat.csv");
  bool first_call;
  std::ofstream& ofs = log.stream(&first_call);
  if (first_call) {
    ofs << "scan_id,t,dim";
    for (int i = 0; i < n; ++i) ofs << ",dx" << i;
    for (int i = 0; i < n; ++i) ofs << ",exp" << i;
    for (int i = 0; i < n; ++i) ofs << ",acw" << i;
    ofs << "\n";
  }
  ofs << scan_id << "," << t_abs << "," << n;
  for (int i = 0; i < n; ++i) ofs << "," << dx(i);
  for (int i = 0; i < n; ++i) ofs << "," << expected_diag(i);
  for (int i = 0; i < n; ++i) ofs << "," << acw_diag(i);
  ofs << "\n";
  ofs.flush();
}

}  // namespace

LioProcIekf::LioProcIekf(NodeContext& ctx)
  : LioProcBase(ctx)
{}

std::string LioProcIekf::loadParameters(ros::NodeHandle& pnh)
{
  // One resolver for the whole function: modes validated against a named
  // set, nested keys read only inside a live scope and REFUSED if set into a
  // dead one, and the EFFECTIVE configuration returned rather than the
  // requested one.  See config_resolve.h.
  refuseRenamedKeys(pnh);   // R73 F-404: old Q-mode key names are refused with a rename message
  ConfigResolver cfg(pnh);

  loadSharedParameters(cfg, pnh);
  iekf_tier1_nees_ = Tier1NeesBuffer(opts_.nees_per_dof_en ? opts_.nees_tier1_window_scans : 0);

  // The scan spline and AdaptiveQ were removed (Bryce, 2026-09-30). Any spline/*, adaptive_q/* or akf_q/* key (R66 AKF-style Q adaptation, removed in R97) left in a config is refused by finalizeConfig() below.
  // R97A Q3 item 3: the innovation-driven Q gain (imu/process_noise/gain/*) is removed; it was off at the yaml.

  // Anything left over in the spline/*, adaptive_q/* and akf_q/* namespaces (all removed) is a key nobody consumes.
  // Deliberately NOT extended to voxel_map/: several of its keys are still
  // read by paramWarn() in voxelmap.cpp, so they are unclaimed here and would
  // be reported as dead when they are merely read elsewhere. Widen this only
  // as those move onto a resolver.
  // R97A Q5: log/* and run/* are added -- every other reader's paramWarn() calls run before this (the shared last-step
  // check), so an unknown key under either namespace (a typo, or a stale old name outside this patch's own replacements)
  // is now refused the same way spline/adaptive_q/akf_q/lio/ekf/voxel_map already are.
  return finalizeConfig(cfg, { "spline", "adaptive_q", "akf_q", "lio/ekf", "voxel_map", "log", "run" });
}

// LIO Processing

void LioProcIekf::deskewAndDownsample(MeasureGroup& mg)
{
  TimedScope ts(profiler_, "lio/deskew");

  // R97A Q3 item 5: the rotation-warp branch (imu/undistort/rot_warp_scale, rot_warp_last_frame) is removed; 1.0 was the baseline path.
  std::vector<PointXYZCov> deskewed;
  deskewPoints(state_, mg.poses, mg.image.t, mg.lidar_points, opts_.deskew, deskewed);

  if (opts_.dsOn()) {
    DsMode mode = (opts_.ds_mode == "average") ? DsMode::AVERAGE : DsMode::FIRST;
    voxelDownsample(deskewed, mg.points, PointXYZCovKeyFn{opts_.ds_leaf_size}, mode);
  } else {
    mg.points = std::move(deskewed);
  }

  // R97A Q2: the dry-run second decimation stream (lio/dry_run_point_filter_num) is removed.
}


LioProcIekf::~LioProcIekf()
{
  // R71 (F-350): the iekf processor never wrote engagement.txt, so gates G13/H7(a) tested nothing.
  const std::string report = engagementReport();
  ROS_WARN_STREAM("\n" << report);
  std::ofstream out(debugLogPath("engagement.txt"), std::ios::trunc);
  if (out) out << report << '\n';
}

std::string LioProcIekf::engagementReport() const
{
  std::ostringstream o;
  o << "[engagement]  did each flag actually do anything?";

  // The plane model is a VoxelMap-side flag, but its engagement belongs in
  // the same report -- a run that selected "information" and never built one
  // is an inert cell for exactly the same reason any flag with a zero
  // engagement counter is, and the scorer only reads this file. voxel_map_ is a
  // MapBackendPtr -- plane_var_mode/
  // weight_floor_mode are VoxelOpts-only concepts, so this block only fires for the voxel backend.
  if (auto* vm = dynamic_cast<VoxelMap*>(voxel_map_.get()))
  {
    const long info_fits = voxelPlaneInformationFitCount();
    o << "\n  voxel_map/plane/plane_var_mode = "
      << vm->opts()->plane_var_mode;
    if (vm->opts()->plane_var_mode == "information" ||
        vm->opts()->plane_var_mode == "information_directional")
    {
      o << " count=" << info_fits;
      if (info_fits == 0)
        o << "   <-- INERT: the information model was selected and no plane "
             "was ever fitted under it";
    }
    o << "\n  voxel_map/plane/weight_floor/mode = "
      << vm->opts()->weight_floor_mode;
    // R90: the fit and the admission settings the map ran with. R97A: the binned map is the only map, so backend is always "binned"
    // (harness tools parse this exact line -- see 00_COMMON.md rule 5).
    o << "\n  voxel_map/fit_config: backend = binned"
      << " min_init = " << vm->opts()->min_init_points << " min_update = " << vm->opts()->min_update_points << " max_points = " << vm->opts()->max_points;
    {   // R82 (D4, F-500), unconditional since R93A: the fit-time classification and the one gap floor are always on; their counters are always written
      const VoxelPlaneR82Counters c82 = voxelPlaneR82Counters();
      const VoxelPlaneR77Counters c77 = voxelPlaneR77Counters();
      o << "\n  voxel_map/plane fit-time model (always on): line_classified=" << c82.line_classified << " line_accepted=" << c82.line_accepted << " e1_tiny_kept=" << c82.e1_tiny_kept << " e0_clamped=" << c82.e0_clamped
        << " gap_fits=" << c77.gap_fits << " gap_d1=" << c77.gap_d1 << " gap_d2=" << c77.gap_d2;
      if (vm->opts()->line_form == "hard" && c82.line_classified == 0)
        o << "   <-- INERT: hard lines are enabled and no fit was classified a line";
    }
    {   // R93A (Bryce 2026-10-10): lio/gate_state_uncertainty -- pose covariance in the admission gate (engagement counter and magnitude; a map backend that does not count gate rows is not reported)
      const VoxelPlaneR93Counters c93 = voxelPlaneR93Counters();
      o << "\n  lio/gate_state_uncertainty = " << (opts_.gate_state_uncertainty ? "true" : "false") << " gate_rows=" << c93.gate_rows << " gate_pose_rows=" << c93.gate_pose_rows
        << " mean_pose_var_added_m2=" << (c93.gate_pose_rows > 0 ? c93.gate_pose_var_sum / static_cast<double>(c93.gate_pose_rows) : 0.0);
      if (opts_.gate_state_uncertainty && c93.gate_rows > 0 && !(c93.gate_pose_var_sum > 0.0))
        o << "   <-- INERT: the gate was set to include the pose covariance and no row received any";
      else if (!opts_.gate_state_uncertainty && c93.gate_pose_rows != 0)
        o << "   <-- ERROR: the gate was set to exclude the pose covariance and " << c93.gate_pose_rows << " rows received it";
    }
    {   // R95 (Bryce 2026-10-11): log/lio/zsq -- the per-scan z squared summary (engagement counter)
      const long zr = zsqRowsWritten();
      o << "\n  log/lio/zsq = " << (opts_.zsq_log_en ? "true" : "false") << " zsq_rows_written=" << zr;
      if (opts_.zsq_log_en && zr == 0)
        o << "   <-- INERT: the key is on and no row was written";
      else if (!opts_.zsq_log_en && zr != 0)
        o << "   <-- ERROR: the key is off and " << zr << " rows were written";
    }
    {   // R88: written for the (only, since R97A) binned map type
      const auto st88 = vm->stats();
      const long long bp = st88 ? st88->binned_points.load() : 0, bb = st88 ? st88->binned_new_bins.load() : 0, bf = st88 ? st88->binned_fits.load() : 0, bt = st88 ? st88->binned_touched_at_fit.load() : 0;
      o << "\n  voxel_map/backend = binned: bin_size_fraction = " << vm->opts()->binned_bin_fraction << " cov_mode = " << (vm->opts()->binned_cov_mode == 1 ? "shrunk" : "average")
        << " weight = count binned_points=" << bp << " binned_new_bins=" << bb << " binned_fits=" << bf
        << " update_count = touched binned_touched_at_fit=" << bt;
      if (bp == 0)
        o << "   <-- INERT: the binned map was selected and no return was put in a bin";
      else if (bf == 0)
        o << "   <-- INERT: the binned map was selected and no fit was made from bins";
    }
    if (vm->opts()->plane_var_diag_en)   // R82 (D2, F-502): written only when the key is on
    {
      const PlaneVarDiagSnapshot d = voxelPlaneVarDiagSnapshot();
      o << "\n  log/voxel_map/plane_var_diag rows=" << d.rows;
      if (d.rows > 0)
      {
        const double qs[5] = {0.10, 0.25, 0.50, 0.70, 0.90};
        o << " fit_q10_q25_q50_q70_q90=";
        for (int k = 0; k < 5; ++k) o << (k ? "," : "") << pvdQuantile(d.fit, d.rows, qs[k]);
        o << " post_q10_q25_q50_q70_q90=";
        for (int k = 0; k < 5; ++k) o << (k ? "," : "") << pvdQuantile(d.post, d.rows, qs[k]);
        o << " fit_below_1e-5_3e-5_1e-4_4e-4=";
        for (int k = 0; k < kPvdFloors; ++k) o << (k ? "," : "") << static_cast<double>(d.fit_below[k]) / d.rows;
        o << " mean_inv_n=" << d.sum_inv_n / d.rows << " (quantiles: log bins, 20 per decade, exact to 6 percent; fractions exact)";
      }
      else
        o << "   <-- INERT: plane_var_diag_en was set and no row reached the variance term";
    }
    // R97A: voxel_map/plane/plane_var_min and plane_var_scan_s2 are removed; this report block (VoxelPlaneR81Counters) is dead with them.
  }

  // R97A Q2: lio/residual_redundancy/* (and its renorm report block) is removed.
  if (opts_.info_scale.on())
  {
    o << "\n  lio/info_scale mode=block F_rot=" << opts_.info_scale.rotation << " F_pos=" << opts_.info_scale.position
      << " iterations=" << info_scale_iterations_ << " discarded_modes=" << info_scale_discarded_modes_
      << " mean_trace_ratio=" << (info_scale_iterations_ > 0 ? info_scale_trace_ratio_sum_ / info_scale_iterations_ : 0.0);
    if (info_scale_iterations_ == 0)
      o << "   <-- INERT: block information scale selected and never applied";
  }
  if (opts_.residual_binning.on())   // R91E / R91F
  {
    const double its = static_cast<double>(residual_binning_iterations_);
    const bool rbp = opts_.residual_binning.usesPosition(), rbr = opts_.residual_binning.usesRotation();
    o << "\n  lio/residual_binning mode=" << opts_.residual_binning.mode << " axes=" << opts_.residual_binning.axes;
    if (rbp) o << " shift_sd_m=" << opts_.residual_binning.shift_sd_m;
    if (rbr) o << " rot_shift_sd_rad=" << opts_.residual_binning.rot_shift_sd_rad;
    o << " min_rows=" << opts_.residual_binning.min_rows
      << " iterations=" << residual_binning_iterations_ << " mean_corrected_rows=" << (its > 0 ? residual_binning_rows_sum_ / its : 0.0)
      << " mean_bins_active=" << (its > 0 ? residual_binning_bins_sum_ / its : 0.0) << " skipped_rows=" << residual_binning_skipped_sum_
      << " mean_trace_before=" << (its > 0 ? residual_binning_trace_before_sum_ / its : 0.0) << " mean_trace_after=" << (its > 0 ? residual_binning_trace_after_sum_ / its : 0.0);
    if (rbr)   // R91F
      o << " mean_rot_corrected_rows=" << (its > 0 ? residual_binning_rot_rows_sum_ / its : 0.0) << " mean_rot_bins_active=" << (its > 0 ? residual_binning_rot_bins_sum_ / its : 0.0)
        << " solve_failed=" << residual_binning_solve_failed_sum_;
    if (residual_binning_iterations_ == 0)
      o << "   <-- INERT: residual binning selected and never applied";
    else if ((rbp && residual_binning_rows_sum_ == 0) || (rbr && residual_binning_rot_rows_sum_ == 0))
      o << "   <-- NOT ENGAGED: no bin of a selected family ever reached min_rows";
    if (residual_binning_skipped_sum_ > 0)
      o << "   <-- NOTE: rows were skipped (distance rows, rows with a bad variance, or a zero rotation lever arm); expected 0 with rows = per_normal";
    if (residual_binning_solve_failed_sum_ > 0)
      o << "   <-- NOTE: the 26 x 26 factorisation failed " << residual_binning_solve_failed_sum_ << " time(s); the correction was not applied in those iterations";
  }

  if (opts_.scan_log_en) o << "\n  log/lio/scan=true  (scan_log.csv written)";
  o << "\n[engagement] a flag marked INERT above did not test anything; the "
       "cell is invalid, not null.";
  return o.str();
}

// R97A Q2: runDryRunShadowPass() is removed (lio/dry_run_point_filter_num).

std::string LioProcIekf::processLIO(MeasureGroup& mg)
{
  // P8.  Seed prior_* to the CURRENT (pre-frame) state before any early
  // return below can skip the real snapshot further down. Without this, an
  // early return (e.g. an empty map) leaves mg.prior_* at MeasureGroup's
  // bare defaults (zero/identity) while publishOdometry() still writes a
  // pose_pair.csv row unconditionally -- gain(t) on that row would read as
  // a huge, meaningless jump instead of the true zero (no update happened).
  mg.prior_pos = state_->pos();
  mg.prior_rot = state_->rot();
  mg.prior_vel = state_->vel();

  // FIX (found this round, mirrors the same fix in lio_coupled.cpp): frame_idx_
  // increments INSIDE updateMap(), so frame_idx_==0 is always the empty-map
  // bootstrap scan, never a map-backed frame -- verified directly (isEmpty()
  // true, 0 residuals). frame_idx_==1 is the true first non-empty map-backed
  // frame the instructions ask for; logged as scan_id=0 (relabeled) to match
  // the CSV vocabulary the instructions use throughout.
  // FIX (found during round-19's real-data run, dangling-if without
  // braces): the patch's second statement here was NOT inside the
  // `if (frame_idx_==1)` guard -- logFirstFrameStateChain fired on EVERY
  // frame, confirmed by the live run's own CSV (pre_update rows spanning
  // the whole bag, not just the first frame). Braced both calls together.
  if (voxel_map_->frame_idx_ == 1) {
    logIekfFirstFrame("pre_update", 0, -1, mg, *state_, mg.image.t + data_queues_->start_time, 0, iekfFirstFrameHashPoints(mg.points), 1469598103934665603ULL, voxel_map_->last_n_map_pts_, voxel_map_->last_n_active_voxels_, state_->pos(), state_->rot());
    logFirstFrameStateChain("iekf", "pre_update", 0, -1, mg, *state_, mg.image.t + data_queues_->start_time);
    logFirstFrameCovarianceBudget(firstFrameArchitectureName(), "pre_update", 0, -1, mg, *state_, mg.image.t + data_queues_->start_time);
  }

  if (voxel_map_->isEmpty()) return {};
  ensureStationaryReference(mg);

  logStateTrace("post_imu", mg, 0, 0);
  // R73 (F-415): per-scan map statistics (before this scan's passes, i.e. the map as inserted through the previous scan) and the
  // driver's update record. All write-only; default off is bit-identical.
  const bool capture_update = opts_.scan_log_en;
  drv_rec_.reset();
  if (opts_.scan_log_en)
    if (auto* vm_stats = dynamic_cast<VoxelMap*>(voxel_map_.get())) { scan_stats_ = vm_stats->computeScanStats(); scan_stats_valid_ = true; }

  {
    TimedScope ts(profiler_, "lio/ekf");
    V3D dtheta, dt;
    V3D total_dtheta = V3D::Zero(), total_dt = V3D::Zero();
    double prev_error = std::numeric_limits<double>::infinity();
    std::string stop = "max_iter";
    int iter = 0;

    // pose_control_lidar_information_footprint_validation Phase 2A: per-
    // iteration physical tail state before/after this scan's REAL IEKF
    // loop (the dry-run pass above is explicitly excluded -- see
    // that pass's own doc comment, "this whole pass is discarded"). Reuses
    // the opts_.log_debug_en gate -- diagnostic-only, zero cost when off, no new config key.
    static PersistentLogStream gn_iter_log("iekf_gn_iteration.csv");

    // Fixed IEKF prior for this frame's ENTIRE inner loop -- see ekf.h's
    // applyMeanUpdate() doc comment. Set once here, read (never rewritten)
    // by every solveSystem()/solveSystem_cuda() call below until the loop
    // finishes and applyCovarianceUpdate() is called exactly once.
    prior_cov_ = state_->cov();
    // R97A Q2: the scalar P controls (lio/p_inflate/alpha, lio/p_floor/min_eig, lio/p_fading/lambda) are removed.
    state_propagat_ = *state_;
    if (capture_update)   // R73: the driver's update record
    {
      drv_rec_.valid = true;
      drv_rec_.start_pos = mg.pos_before_imu;
      drv_rec_.start_rv = Log(mg.rot_before_imu);
      drv_rec_.prop_pos = state_->pos();
      drv_rec_.prop_rv = Log(state_->rot());
      drv_rec_.prior_cov = prior_cov_;
    }
    // P1.  Captured here, before this frame's update runs at all -- pairs
    // with the POST value already computed further down (see that block's
    // "const auto& P = state_->cov()") to give the covariance delta, i.e.
    // how much this scan actually learned, which a level alone cannot.
    trP_pos_pre_ = prior_cov_.block<3, 3>(StateGroup::idxP(), StateGroup::idxP()).trace();
    // P8.  state_propagat_ IS the propagated (pre-update) state at this
    // exact point -- snapshot it into mg so PubProc::publishOdometry() can
    // export it alongside the posterior pose, without processLIO() needing
    // to know anything about odometry export itself.
    mg.prior_pos = state_propagat_.pos();
    mg.prior_rot = state_propagat_.rot();
    mg.prior_vel = state_propagat_.vel();
    // R97A Q2: lio/iekf_point_stream is removed; the IEKF always reads mg.points (the same set the map update uses).
    const std::vector<PointXYZCov>& iekf_points = mg.points;
    bool any_solved = false;
    for (; iter < opts_.max_iterations; iter++) {
      const StateGroup iteration_before = *state_;
      // Phase 2A: exact physical tail state BEFORE this iteration's own
      // solve -- state_ IS the tail/current estimator state the IEKF
      // updates directly (confirmed by source audit: solveSystem() below
      // writes ekf_.dtheta/ekf_.dt, and LioProcBase::estimateStateCorrection
      // applies them to state_ internally via the ESIKF update this
      // codebase uses throughout).
      const bool log_iter = opts_.log_debug_en;
      V3D p_tail_before, v_tail_before; M3D R_tail_before;
      if (log_iter) { p_tail_before = state_->pos(); R_tail_before = state_->rot(); v_tail_before = state_->vel(); }

      // T0-D wants the first-iteration (pre-update, un-relinearized)
      // innovation only -- later iterations relinearize at an
      // already-partially-corrected state, which is not the quantity NIS
      // is defined over.
      setDiagnosticGnIteration(iter);
      double error = estimateStateCorrection(iekf_points, dtheta, dt, /*allow_consistency_log=*/iter == 0);
      if (capture_update) recordIteration(drv_rec_, iter, dtheta, dt);   // R73
      writeStationaryIterationDiagnostics(
          firstFrameArchitectureName(), firstFrameArchitectureName(),
          voxel_map_->frame_idx_, iter,
          mg.image.t + data_queues_->start_time, stationary_reference_R_,
          stationary_reference_p_, stationary_reference_v_, iteration_before,
          *state_, residuals_);
      if (voxel_map_->frame_idx_ == 1) {  // see the frame_idx_ fix comment above
        logIekfFirstFrame("post_iteration", 0, iter, mg, *state_, mg.image.t + data_queues_->start_time, static_cast<int>(residuals_.size()), iekfFirstFrameHashPoints(mg.points), iekfFirstFrameHashResiduals(residuals_), voxel_map_->last_n_map_pts_, voxel_map_->last_n_active_voxels_, mg.prior_pos, mg.prior_rot);
        logFirstFrameStateChain("iekf", "post_iteration", 0, iter, mg, *state_, mg.image.t + data_queues_->start_time);
        logFirstFrameCovarianceBudget(firstFrameArchitectureName(), "post_iteration", 0, iter, mg, *state_, mg.image.t + data_queues_->start_time);
      }

      if (log_iter) {
        const V3D p_tail_after = state_->pos();
        const M3D R_tail_after = state_->rot();
        const V3D v_tail_after = state_->vel();
        const V3D dp_tail = p_tail_after - p_tail_before;
        const V3D dtheta_tail = Log(M3D(R_tail_before.transpose() * R_tail_after));
        const V3D dv_tail = v_tail_after - v_tail_before;
        // Linearized (predicted) LiDAR cost reduction for THIS iteration's
        // own applied step dx=[dtheta;dt], from the SAME residuals_/
        // ekf_.HtH/ekf_.Htz this iteration's solveSystem() just built --
        // NOT a re-evaluated exact residual cost (that would require
        // re-projecting every point through the new pose, not done here).
        // Standard GN/LM predicted-reduction form: dE = dx^T*Htz -
        // 0.5*dx^T*HtH*dx, documented explicitly as linearized/predicted.
        Eigen::Matrix<double, 6, 1> dx; dx << dtheta, dt;   // (theta;pos) ordering matches ekf_.HtH/Htz's own (see idxR=0,idxP=3)
        const double predicted_delta_E_lidar = (dx.transpose() * ekf_.Htz)(0) - 0.5 * (dx.transpose() * ekf_.HtH * dx)(0);
        bool first;
        std::ofstream& ofs = gn_iter_log.stream(&first);
        if (first) ofs << "scan_id,iter,n_residuals,t_abs_iter,scan_timestamp,"
                          "tail_p_before_x,tail_p_before_y,tail_p_before_z,tail_p_after_x,tail_p_after_y,tail_p_after_z,"
                          "tail_v_before_x,tail_v_before_y,tail_v_before_z,tail_v_after_x,tail_v_after_y,tail_v_after_z,"
                          "dp_tail_x,dp_tail_y,dp_tail_z,dp_tail_norm,"
                          "dtheta_tail_x,dtheta_tail_y,dtheta_tail_z,dtheta_tail_norm,"
                          "dv_tail_x,dv_tail_y,dv_tail_z,dv_tail_norm,"
                          "predicted_delta_E_lidar,avg_abs_residual,lidar_tail_lambda_min,lidar_tail_lambda_max,lidar_tail_rank,lidar_tail_condition\n";
        const double t_abs_iter = mg.image.t + data_queues_->start_time;
        // scan_timestamp duplicates t_abs_iter under the column name the
        // factor-isolation patch's offline analysis scripts and the pose-control
        // side's own gn_iteration rows use -- kept as an alias rather than a
        // rename so existing t_abs_iter-keyed consumers (analyze_tail_correction_gt.py,
        // the prior task's build_final_csv.py) are unaffected.
        const V3D dv_tail_vec = v_tail_after - v_tail_before;
        const Eigen::Matrix<double,6,6> H6 = ekf_.HtH;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> es6(symmetrized(H6));
        const Eigen::VectorXd ev6 = es6.eigenvalues();
        const double thresh6 = 1e-12 * std::max(std::abs(ev6(5)), 1e-300);
        int rank6 = 0;
        for (int i6 = 0; i6 < 6; ++i6) if (ev6(i6) > thresh6) ++rank6;
        const double cond6 = ev6(0) > thresh6 ? ev6(5) / ev6(0) : std::numeric_limits<double>::infinity();
        ofs << voxel_map_->frame_idx_ << "," << iter << "," << residuals_.size() << ","
            << std::setprecision(12) << t_abs_iter << "," << t_abs_iter << std::setprecision(6) << ","
            << p_tail_before.x() << "," << p_tail_before.y() << "," << p_tail_before.z() << ","
            << p_tail_after.x() << "," << p_tail_after.y() << "," << p_tail_after.z() << ","
            << v_tail_before.x() << "," << v_tail_before.y() << "," << v_tail_before.z() << ","
            << v_tail_after.x() << "," << v_tail_after.y() << "," << v_tail_after.z() << ","
            << dp_tail.x() << "," << dp_tail.y() << "," << dp_tail.z() << "," << dp_tail.norm() << ","
            << dtheta_tail.x() << "," << dtheta_tail.y() << "," << dtheta_tail.z() << "," << dtheta_tail.norm() << ","
            << dv_tail_vec.x() << "," << dv_tail_vec.y() << "," << dv_tail_vec.z() << "," << dv_tail.norm() << ","
            << predicted_delta_E_lidar << "," << error << ","
            << ev6(0) << "," << ev6(5) << "," << rank6 << "," << cond6 << "\n";
        ofs.flush();
      }
      if (!residuals_.empty()) any_solved = true;
      total_dtheta += dtheta;
      total_dt     += dt;

      // History (590-607): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-590
      if (iter == 0 && opts_.log_nll_en)
      {
        const int n_res = static_cast<int>(residuals_.size());
        double nll = 0.0;
        if (n_res > 0) {
          // Only defined when solveSystem() actually ran this iteration
          // (estimateStateCorrection() returns early on an empty
          // residuals_ WITHOUT calling solveSystem(), which is the only
          // thing that (re-)accumulates ekf_.HtH/Htz -- calling
          // nllQuadraticAndLogdet() in the n_res==0 branch would silently
          // read a STALE HtH/Htz left over from a previous frame).
          double sum_chi2 = 0.0, sum_log_sigma2 = 0.0;
          for (const auto& r : residuals_) {
            sum_chi2 += (r.r * r.r) / r.sigma_squared;
            sum_log_sigma2 += std::log(r.sigma_squared);
          }
          nll = 0.5 * (n_res * std::log(2.0 * M_PI) + sum_log_sigma2 + sum_chi2
                       + ekf_.nllQuadraticAndLogdet(prior_cov_));
        }
        const double pivot_ratio = (n_res > 0) ? ekf_.pivotRatio() : std::numeric_limits<double>::quiet_NaN();
        const double kalman_gain_norm = (n_res > 0) ? ekf_.kalmanGainNorm() : std::numeric_limits<double>::quiet_NaN();
        logConsistencyNll("lio", mg.image.t + data_queues_->start_time, nll, n_res, pivot_ratio, kalman_gain_norm);
      }

      const double prev = prev_error;
      prev_error = error;

      if (opts_.log_debug_en)
      {
        const double t_abs = mg.image.t + data_queues_->start_time;
        std::ostringstream iss;
        iss << "t_abs=" << std::fixed << std::setprecision(6) << t_abs
            << "  iter=" << iter << "  n_residuals=" << residuals_.size()
            << "  avg_abs_r=" << std::scientific << std::setprecision(6) << error
            << "  rel_diff=" << ((prev - error) / std::max(prev, 1e-6));
        debugLogIterError(iss.str());
      }

      // R97A Q2: the min_norm_dtheta/min_norm_dt/min_diff_error early stops are removed; every scan
      // now runs all max_iterations iterations (stop stays "max_iter").
    }

    // Posterior covariance write happens exactly ONCE here, using the SAME
    // fixed prior_cov_ every real iteration above blended against (see
    // ekf.h) -- not per-iteration like the old applyUpdate() did. No-op
    // (skipped) when any_solved is false: no real measurement info was ever
    // accumulated this frame (ekf_'s H_full would be stale from a previous
    // frame), so there's nothing to blend and state_->cov() should stay
    // exactly as it was going into this frame.
    //
    if (any_solved)
      ekf_.applyCovarianceUpdate(state_, prior_cov_, 1.0);
    if (capture_update)   // R73
    {
      drv_rec_.stop = stop;
      drv_rec_.post_pos = state_->pos();
      drv_rec_.post_rv = Log(state_->rot());
      drv_rec_.post_cov = state_->cov();
    }
    if (dump_rules::lastIterDumpWanted(opts_.residual_dump_last_iter, opts_.residual_dump_all_iters))   // R73 (F-417); R86B: not with every-iteration dump: the residuals of the last executed iteration
    {
      const int last_iter_dump = std::min(iter, opts_.max_iterations - 1);
      if (last_iter_dump > 0) dumpResidualsIfListed(last_iter_dump);
    }
    if (opts_.zsq_log_en)   // R95: the z squared summary of the last executed iteration, the residuals_ of that iteration (as the last-iteration dump above); iteration 0 was written in estimateStateCorrection()
    {
      const int last_iter_zsq = std::min(iter, opts_.max_iterations - 1);
      if (last_iter_zsq > 0) logZsqIteration(last_iter_zsq);
    }

    // The covariance is intentionally updated once, after all fixed-prior
    // mean iterations.  Earlier per-iteration diagnostics therefore show
    // the unchanged prior by design.  Record the actual posterior here so
    // the iekf estimator returns a matched
    // P_prior/P_after pair for the first post-calibration LIO frame.
    if (voxel_map_->frame_idx_ == 1) {
      mg.cov_after_lio = state_->cov();
      logFirstFrameCovarianceBudget(
          firstFrameArchitectureName(), "post_covariance", 0, iter,
          mg, *state_, mg.image.t + data_queues_->start_time);
    }

    // T7-a. dx must be read here and not inside the loop: it is the FRAME's
    // correction, and reading it mid-iteration would report one Gauss-Newton
    // step rather than the process-model error the measurement disagreed with.
    {
      Eigen::MatrixXd phi_p_phit, accum_cov_w, p_before;
      if (imuProcQhatRead(phi_p_phit, accum_cov_w, p_before)) {
        const Eigen::VectorXd dx = state_->boxminusFromPropagat(state_propagat_);
        const Eigen::MatrixXd& P_post = state_->cov();
        if (phi_p_phit.rows() == P_post.rows()) {
          const double t_abs = mg.image.t + data_queues_->start_time;
          debugLogQhat(voxel_map_->frame_idx_, t_abs, dx, (phi_p_phit - P_post).diagonal(),
                       accum_cov_w.diagonal());

          // CQ-71 item 0: the information budget. P_after_IMU reconstructed
          // as phi_p_phit + accum_cov_w (F P F^T + Q_eff, by construction --
          // see imuProcQhatRead()'s own doc comment). eig(H^TWH) is NOT
          // separately tracked anywhere in this codebase -- reconstructed
          // here from ekf.h's own A = H_full + prior_cov^-1 identity:
          // H_full = P_after_LIO^-1 - P_after_IMU^-1 exactly.
          if (p_before.rows() == P_post.rows() && p_before.rows() > 0) {
            const Eigen::MatrixXd P_after_imu = phi_p_phit + accum_cov_w;
            auto eigStats = [](const Eigen::MatrixXd& M, double& trP, double& logdetP,
                                double& lam_min, double& lam_max) {
              Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(symmetrized(M));
              const Eigen::VectorXd ev = es.eigenvalues();
              trP = M.trace();
              logdetP = ev.array().log().sum();
              lam_min = ev(0);
              lam_max = ev(ev.size() - 1);
            };
            double trP_before, ld_before, lmin_before, lmax_before;
            double trP_imu, ld_imu, lmin_imu, lmax_imu;
            double trP_fpf, ld_fpf, lmin_fpf, lmax_fpf;
            double trP_qeff, ld_qeff, lmin_qeff, lmax_qeff;
            double trP_lio, ld_lio, lmin_lio, lmax_lio;
            eigStats(p_before, trP_before, ld_before, lmin_before, lmax_before);
            eigStats(P_after_imu, trP_imu, ld_imu, lmin_imu, lmax_imu);
            eigStats(phi_p_phit, trP_fpf, ld_fpf, lmin_fpf, lmax_fpf);
            eigStats(accum_cov_w, trP_qeff, ld_qeff, lmin_qeff, lmax_qeff);
            eigStats(P_post, trP_lio, ld_lio, lmin_lio, lmax_lio);

            double trP_h = std::numeric_limits<double>::quiet_NaN();
            double ld_h = std::numeric_limits<double>::quiet_NaN();
            double lmin_h = std::numeric_limits<double>::quiet_NaN();
            double lmax_h = std::numeric_limits<double>::quiet_NaN();
            Eigen::LDLT<Eigen::MatrixXd> ldlt_imu(P_after_imu), ldlt_lio(P_post);
            if (ldlt_imu.info() == Eigen::Success && ldlt_lio.info() == Eigen::Success) {
              const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(P_post.rows(), P_post.rows());
              const Eigen::MatrixXd H_eff = ldlt_lio.solve(I) - ldlt_imu.solve(I);
              eigStats(H_eff, trP_h, ld_h, lmin_h, lmax_h);
            }

            static PersistentLogStream ib_log("info_budget.txt");
            bool ib_first;
            std::ofstream& ib_ofs = ib_log.stream(&ib_first);
            if (ib_first)
              ib_ofs << "scan_id,t_abs,h_eff_valid,"
                        "trP_before,logdet_before,lmin_before,lmax_before,"
                        "trP_after_imu,logdet_after_imu,lmin_after_imu,lmax_after_imu,"
                        "trP_fpf,logdet_fpf,lmin_fpf,lmax_fpf,"
                        "trP_qeff,logdet_qeff,lmin_qeff,lmax_qeff,"
                        "trP_after_lio,logdet_after_lio,lmin_after_lio,lmax_after_lio,"
                        "trP_h,logdet_h,lmin_h,lmax_h\n";
            ib_ofs << voxel_map_->frame_idx_ << "," << std::setprecision(12) << t_abs << ","
                   << 1 << ","
                   << trP_before << "," << ld_before << "," << lmin_before << "," << lmax_before << ","
                   << trP_imu << "," << ld_imu << "," << lmin_imu << "," << lmax_imu << ","
                   << trP_fpf << "," << ld_fpf << "," << lmin_fpf << "," << lmax_fpf << ","
                   << trP_qeff << "," << ld_qeff << "," << lmin_qeff << "," << lmax_qeff << ","
                   << trP_lio << "," << ld_lio << "," << lmin_lio << "," << lmax_lio << ","
                   << trP_h << "," << ld_h << "," << lmin_h << "," << lmax_h << "\n";
            ib_ofs.flush();
          }
        }
      }
    }

    if (opts_.log_consistency_scan_en)
    {
      const double t_abs = mg.image.t + data_queues_->start_time;
      const double dt = (last_scan_t_abs_ < 0.0) ? 0.0 : (t_abs - last_scan_t_abs_);
      last_scan_t_abs_ = t_abs;

      const auto& P = state_->cov();
      const double trP_pos = P.block<3, 3>(StateGroup::idxP(), StateGroup::idxP()).trace();
      const double trP_vel = P.block<3, 3>(StateGroup::idxV(), StateGroup::idxV()).trace();
      const double trP_att = P.block<3, 3>(StateGroup::idxR(), StateGroup::idxR()).trace();

      // mg.imu_samples is already cleared by ImuProc::propagate() (see its
      // doc comment there) by the time processLIO() runs -- mg.poses is
      // the per-IMU-step record propagate() leaves behind instead. `gyr`
      // is the bias-corrected body-frame angular velocity used for each
      // step's rotation propagation; acc_head/acc_tail are WORLD-frame
      // true linear acceleration (gravity already added back, see Pose6D's
      // doc comment) -- their average norm is near 0 for a near-stationary/
      // constant-velocity segment and grows with genuine dynamic motion,
      // which is the excitation signal T0-D's dynamics-slope panel wants
      // (a cleaner proxy than raw specific-force magnitude, which would be
      // dominated by the ~9.8 m/s^2 gravity constant).
      double sum_omega = 0.0, sum_acc = 0.0;
      for (const auto& p : mg.poses) {
        sum_omega += p.gyr.norm();
        sum_acc   += (0.5 * (p.acc_head + p.acc_tail)).norm();
      }
      const double n_poses = std::max<size_t>(1, mg.poses.size());
      const V3D& ba = state_->biasAcc();
      const V3D& bg = state_->biasGyr();
      logConsistencyScan("lio", voxel_map_->frame_idx_, t_abs, dt, trP_pos, trP_vel, trP_att,
                          sum_omega / n_poses, sum_acc / n_poses,
                          ba.x(), ba.y(), ba.z(), bg.x(), bg.y(), bg.z());
    }

    std::ostringstream oss;
    oss << "[lio/ekf] iters=" << iter + 1 << "  stop=" << stop
        << std::scientific << std::setprecision(1)
        << "  |dtheta|=" << total_dtheta.norm() * (180.0 / M_PI) << " deg"
        << "  |dt|=" << total_dt.norm() * 1000.0 << " mm";

    // Hand the per-frame LIO diagnostics to the map so frame_stats.txt can
    // carry them.  Computed unconditionally and cheaply (two 3x3 eigen
    // solves on a matrix the update already built) because gating them on
    // log_debug_en is exactly how they ended up unavailable to every sweep
    // that needed them -- H_pp_eig has existed for months, inside a
    // free-text debug line no scorer parses.
    {
      LioFrameDiag diag;
      diag.n_residuals = static_cast<int>(residuals_.size());
      diag.n_points_after_pfn = static_cast<int>(mg.lidar_points.size());
      diag.n_points_after_ds  = static_cast<int>(mg.points.size());
      diag.n_imu_samples      = mg.n_imu_samples;
      diag.n_miss_coverage = n_miss_coverage_;
      diag.n_miss_mismatch = n_miss_mismatch_;
      diag.n_tier0_miss_coverage = n_tier0_miss_coverage_;
      diag.n_tier0_miss_mismatch = n_tier0_miss_mismatch_;
      // CQ-19(a): P's own decomposition (prior_cov_, pre-update -- state_->
      // cov()'s POST-update counterpart is read further below, once the
      // covariance update has actually run). Computed unconditionally, NOT
      // gated on residuals_.empty() -- prior_cov_ exists regardless of
      // whether any plane matched this frame, and gating it the same way
      // the HtH-derived diagnostics below must be gated left every
      // zero-residual frame's P columns at their -1.0 "unavailable"
      // default even though the prior itself was perfectly well-defined
      // (confirmed on site1_handheld_1's smoke cell: 150/2044 frames had
      // residuals_.empty() and silently failed the "eigenvalues sum to
      // trP_pos_pre" sanity check as a result).
      Eigen::Vector3d p_pos_pre_min_eigenvector = Eigen::Vector3d::Zero();
      bool have_p_pos_pre = false;
      if (prior_cov_.rows() >= StateGroup::idxP() + 3 && prior_cov_.cols() >= StateGroup::idxP() + 3) {
        const M3D P_pp_pre = prior_cov_.block<3, 3>(StateGroup::idxP(), StateGroup::idxP());
        Eigen::SelfAdjointEigenSolver<M3D> es_p_pre(P_pp_pre);
        diag.p_pos_eig_min_pre = es_p_pre.eigenvalues()(0);
        diag.p_pos_eig_mid_pre = es_p_pre.eigenvalues()(1);
        diag.p_pos_eig_max_pre = es_p_pre.eigenvalues()(2);
        p_pos_pre_min_eigenvector = es_p_pre.eigenvectors().col(0);
        have_p_pos_pre = true;

        const M3D P_rr_pre = prior_cov_.block<3, 3>(StateGroup::idxR(), StateGroup::idxR());
        Eigen::SelfAdjointEigenSolver<M3D> es_r_pre(P_rr_pre);
        diag.p_rot_trace_pre   = P_rr_pre.trace();
        diag.p_rot_eig_min_pre = es_r_pre.eigenvalues()(0);
        // CQ-74 item 1: the rest of the same already-computed spectrum --
        // only the min was kept before.
        diag.p_rot_eig_mid_pre = es_r_pre.eigenvalues()(1);
        diag.p_rot_eig_max_pre = es_r_pre.eigenvalues()(2);

        diag.p_pos_vel_fro_pre =
            prior_cov_.block<3, 3>(StateGroup::idxP(), StateGroup::idxV()).norm();
        if (state_->idxBA() >= 0 &&
            prior_cov_.rows() >= state_->idxBA() + 3 && prior_cov_.cols() >= state_->idxBA() + 3) {
          diag.p_pos_bias_fro_pre =
              prior_cov_.block<3, 3>(StateGroup::idxP(), state_->idxBA()).norm();
        }
      }

      if (!residuals_.empty()) {
        const M3D H_pp_d = ekf_.HtH.block<3, 3>(3, 3);
        const M3D H_rr_d = ekf_.HtH.block<3, 3>(0, 0);
        Eigen::SelfAdjointEigenSolver<M3D> es_pp(H_pp_d), es_rr(H_rr_d);
        diag.h_pp_min_eig = es_pp.eigenvalues()(0);
        diag.h_pp_max_eig = es_pp.eigenvalues()(2);  // CQ-81 item B
        diag.h_rr_min_eig = es_rr.eigenvalues()(0);
        diag.h_rr_trace   = H_rr_d.trace();
        // CQ-36 item 1: read directly off ekf_.HtH, unlike sum_weight below
        // (which sums 1/sigma_squared over residuals_ and is therefore
        // structurally blind to any HtH-level correction axis C applies --
        // see residual_redundancy.h). trace(H_pp) == sum_weight EXACTLY
        // whenever no HtH-level correction is active (mode=="off"), so the
        // pair is its own control: equal at off, divergent otherwise.
        // h_rr_trace above (H_rr_d.trace(), already logged) IS this same
        // quantity for the rotation block -- no separate htth_rot_trace
        // column is added; the item-3 one-column table in the round report
        // says so explicitly rather than adding a byte-identical duplicate.
        diag.htth_pos_trace = H_pp_d.trace();
        double sw = 0.0;
        for (const auto& r : residuals_) if (r.sigma_squared > 0.0) sw += 1.0 / r.sigma_squared;
        diag.sum_weight = sw;

        // CQ-18 item (2): S = floor_term + sigma_diag_squared + plane_var_term
        // + s_prior_pose, per residual -- summed for sum_S/the four shares,
        // and used per-residual for nis (nu^2/S) / nis_est (nu^2/(S-s_prior_pose)),
        // matching voxelplane.cpp's per-candidate corr_scan.csv definitions
        // exactly but averaged over THIS frame's accepted residuals instead
        // of a whole scan's candidates.
        {
          double sum_floor = 0.0, sum_sdiag = 0.0, sum_pvar = 0.0, sum_prior_pose = 0.0;
          double sum_nis = 0.0, sum_nis_est = 0.0;
          int n_nis = 0, n_nis_est = 0;
          for (const auto& r : residuals_) {
            if (r.floor_term < 0.0 || r.sigma_diag_squared < 0.0 || r.s_prior_pose < 0.0) continue;
            const double S_i = r.floor_term + r.sigma_diag_squared + r.plane_var_term + r.s_prior_pose;
            if (!(S_i > 0.0)) continue;
            sum_floor      += r.floor_term;
            sum_sdiag      += r.sigma_diag_squared;
            sum_pvar       += r.plane_var_term;
            sum_prior_pose += r.s_prior_pose;
            const double nis_i = r.r * r.r / S_i;
            if (std::isfinite(nis_i)) { sum_nis += nis_i; ++n_nis; }
            const double S_est_i = r.floor_term + r.sigma_diag_squared + r.plane_var_term;
            if (S_est_i > 0.0) {
              const double nis_est_i = r.r * r.r / S_est_i;
              if (std::isfinite(nis_est_i)) { sum_nis_est += nis_est_i; ++n_nis_est; }
            }
          }
          const double sum_S = sum_floor + sum_sdiag + sum_pvar + sum_prior_pose;
          if (sum_S > 0.0) {
            diag.sum_S = sum_S;
            diag.floor_share      = sum_floor / sum_S;
            diag.sdiag_share      = sum_sdiag / sum_S;
            diag.pvar_share       = sum_pvar / sum_S;
            diag.prior_pose_share = sum_prior_pose / sum_S;
          }
          if (n_nis > 0) diag.nis = sum_nis / n_nis;
          if (n_nis_est > 0) diag.nis_est = sum_nis_est / n_nis_est;
        }

        // CQ-19(b): the one dot product that signs rho_ref, using HtH_pp's
        // own weakest eigenvector (es_pp above; SelfAdjointEigenSolver
        // orders eigenvalues ascending, so column 0 is the weakest
        // direction, matching diag.h_pp_min_eig) against P_pp's own
        // weakest eigenvector, computed unconditionally just above.
        if (have_p_pos_pre) {
          diag.cos_pmin_hmin = std::abs(p_pos_pre_min_eigenvector.dot(es_pp.eigenvectors().col(0)));
        }

        // P1.  ask/got/refusal.  ekf_.HtH is 6x6 (rows 0-2 rot, 3-5 pos) and is
        // singular whenever a direction is unconstrained, so solve in the
        // eigenbasis and drop modes below a relative tolerance rather than
        // inverting.  A dropped mode is a direction the scan did not constrain;
        // it contributes nothing to either quantity, which is correct.
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es6(ekf_.HtH);
        const auto& ev = es6.eigenvalues();
        const double tol = 1e-12 * std::max(1.0, ev(5));
        const Eigen::Matrix<double, 6, 1> z = es6.eigenvectors().transpose() * ekf_.Htz;
        // Paired with THIS iteration's own dtheta/dt, not the frame's
        // cumulative total_dtheta/total_dt -- ekf_.HtH/Htz are only ever the
        // LAST relinearization (this diagnostic runs once, after the inner
        // loop), and dtheta/dt is set from ekf_.dtheta/dt at that same last
        // solveSystem() call (see the assignment right after it), so this is
        // the correctly paired iteration. Pairing against the cumulative
        // total instead double-counts every earlier iteration's own already-
        // applied correction into "got", routinely pushing got > ask and
        // refusal negative even on ordinary frames -- confirmed on the P1-P4
        // smoke test (66% of frames outside [0,1] before this fix).
        Eigen::Matrix<double, 6, 1> dxv;
        dxv << dtheta, dt;
        const Eigen::Matrix<double, 6, 1> y = es6.eigenvectors().transpose() * dxv;
        double ask = 0.0, got = 0.0;
        for (int i = 0; i < 6; ++i) if (ev(i) > tol) {
          ask += z(i) * z(i) / ev(i);
          got += ev(i) * y(i) * y(i);
        }
        diag.ask = ask;  diag.got = got;
        diag.refusal = (ask > 0.0) ? (1.0 - got / ask)
                                    : std::numeric_limits<double>::quiet_NaN();
        // TQ-20 item 1: kappa_eff = 1/sqrt(1-rho_ref)-1, rho_ref==refusal
        // above (same formula, 1-got/ask) -- got>0.0 is the same
        // availability condition refusal's own ask>0.0 branch relies on,
        // since got/ask both come from the same eigenbasis sum above.
        diag.kappa_eff = (got > 0.0) ? (std::sqrt(ask / got) - 1.0) : -1.0;
        diag.htz_rot_norm = ekf_.Htz.segment<3>(0).norm();
        diag.htz_pos_norm = ekf_.Htz.segment<3>(3).norm();
        diag.iters        = iter + 1;
        diag.dx_rot_deg   = total_dtheta.norm() * (180.0 / M_PI);
        diag.dx_pos_mm    = total_dt.norm() * 1000.0;

        // TQ-20 item 1/6: the 6 generalized eigenvalues of (HtH, P) --
        // needs the SAME combined 6x6 prior block buildResiduals()'s own
        // prior_cov_rp slices (rot 0-3, pos 3-6, matching HtH's layout),
        // re-sliced here since that one is frame-constant context, not
        // stored on diag. have_p_pos_pre (computed above, this same
        // iteration) already proved prior_cov_ is at least idxP()+3 --
        // re-check idxR()+6 explicitly since the combined slice spans
        // both blocks and idxR()/idxP() are not asserted adjacent in
        // general (only true by this codebase's current StateGroup
        // layout, which line 283's prior_cov_rp slice already assumes).
        if (have_p_pos_pre &&
            prior_cov_.rows() >= StateGroup::idxR() + 6 &&
            prior_cov_.cols() >= StateGroup::idxR() + 6) {
          const Eigen::Matrix<double, 6, 6> P_prior_6 =
              prior_cov_.block<6, 6>(StateGroup::idxR(), StateGroup::idxR());
          Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>>
              ges(ekf_.HtH, P_prior_6);
          if (ges.info() == Eigen::Success) {
            const auto& gev = ges.eigenvalues();
            diag.kappa_gev0 = gev(0); diag.kappa_gev1 = gev(1); diag.kappa_gev2 = gev(2);
            diag.kappa_gev3 = gev(3); diag.kappa_gev4 = gev(4); diag.kappa_gev5 = gev(5);
            diag.kappa_gev_ok = true;
          }
        }
      }
      // Captured at the TOP of processLIO(), before the update -- see note.
      diag.trP_pos_pre = trP_pos_pre_;
      // CQ-19(a): the POST-update counterpart to the pre-update p_pos_eig_*
      // above -- state_->cov() already reflects applyCovarianceUpdate()'s
      // result by this point in processLIO() (that call runs well above,
      // unconditionally on any_solved), independent of whether residuals_
      // was non-empty this frame.
      {
        const Eigen::MatrixXd& P_post = state_->cov();
        if (P_post.rows() >= StateGroup::idxP() + 3 && P_post.cols() >= StateGroup::idxP() + 3) {
          const M3D P_pp_post = P_post.block<3, 3>(StateGroup::idxP(), StateGroup::idxP());
          Eigen::SelfAdjointEigenSolver<M3D> es_p_post(P_pp_post);
          diag.p_pos_eig_min_post = es_p_post.eigenvalues()(0);
          diag.p_pos_eig_mid_post = es_p_post.eigenvalues()(1);
          diag.p_pos_eig_max_post = es_p_post.eigenvalues()(2);
          diag.trP_pos_post = P_pp_post.trace();
        }
        // CQ-74 item 1: the rotation block's POST-update counterpart --
        // previously had no post-update columns at all.
        if (P_post.rows() >= StateGroup::idxR() + 3 && P_post.cols() >= StateGroup::idxR() + 3) {
          const M3D P_rr_post = P_post.block<3, 3>(StateGroup::idxR(), StateGroup::idxR());
          Eigen::SelfAdjointEigenSolver<M3D> es_r_post(P_rr_post);
          diag.p_rot_trace_post   = P_rr_post.trace();
          diag.p_rot_eig_min_post = es_r_post.eigenvalues()(0);
          diag.p_rot_eig_mid_post = es_r_post.eigenvalues()(1);
          diag.p_rot_eig_max_post = es_r_post.eigenvalues()(2);
        }
      }
      // R97A Q2: lio/residual_redundancy/* is removed; diag.redund_*/naive_info_gain/
      // woodbury_info_gain/gamma_correction_*/b_correction_norm stay at their LioFrameDiag
      // struct defaults (0 / 1.0, matching mode=="off" exactly).
      // CQ-31 item 8: reduced_chi2 was computed twice already in this file
      // (lines ~1048, ~1648) but only ever streamed to a log line -- TQ-23
      // could not report it per-cell for exactly that reason. Same formula,
      // computed unconditionally (not gated behind opts_.log_debug_en) so
      // it is always available: sum(r.r^2/r.sigma_squared) / n_residuals,
      // which averages to ~1 for a correctly-calibrated residual model.
      {
        double sum_chi2 = 0.0;
        for (const auto& r : residuals_) sum_chi2 += (r.r * r.r) / r.sigma_squared;
        diag.reduced_chi2 = residuals_.empty() ? 0.0 : sum_chi2 / residuals_.size();
      }

      if (auto* vm = dynamic_cast<VoxelMap*>(voxel_map_.get())) vm->noteLioFrameDiag(diag);
      logEigenspectrum18(voxel_map_->frame_idx_, mg.image.t + data_queues_->start_time, "iekf");
    }

    if (opts_.log_debug_en)
    {
      const double t_abs = mg.image.t + data_queues_->start_time;
      const Eigen::Quaterniond q(state_->rot());
      // Pre-correction pose, captured by estimateState() right after
      // processIMU() (pure propagation, no LIO update applied yet) --
      // lets a reader compute this frame's actual correction vector
      // (state_->pos() - mg.pos_after_imu) directly, instead of only
      // inferring it indirectly via aligned-ATE deltas in evo.txt.
      const Eigen::Quaterniond q_before(mg.rot_after_imu);
      const auto& P = state_->cov();
      const M3D P_RR = P.block<3, 3>(StateGroup::idxR(), StateGroup::idxR());
      const M3D P_PP = P.block<3, 3>(StateGroup::idxP(), StateGroup::idxP());
      const M3D P_VV = P.block<3, 3>(StateGroup::idxV(), StateGroup::idxV());
      // Position-velocity cross-covariance -- this is the ONLY channel
      // through which a position-only measurement (LIO's HtH/Htz only
      // ever touch the R/P block, never V directly) can correct velocity
      // at all: the update's effect on V is entirely mediated by how
      // correlated P and V already are in the prior. Its norm answers
      // "how much of this frame's position correction, if any, leaked into
      // a velocity correction" -- near 0 means velocity got essentially no
      // benefit from this update, regardless of how good the position fix was.
      const M3D P_PV = P.block<3, 3>(StateGroup::idxP(), StateGroup::idxV());

      // Plane/residual derivation summary for this frame's final iteration
      // (residuals_ reflects the last buildResiduals() call inside the loop
      // above) -- mean_normal_norm close to 1 means nearly every residual
      // this frame shares almost the same plane normal (dominated by one
      // surface -- expected while facing a single wall, but also the
      // signature of degenerate/spurious planes all landing on the same
      // artifact direction, e.g. along the LIDAR's own scan lines during a
      // near-stationary period); close to 0 means normals are diverse
      // (multiple, differently-oriented real surfaces contributing).
      V3D mean_normal = V3D::Zero();
      double mean_abs_r = 0.0, max_abs_r = 0.0;
      // sigma/weight stats (VARIANCE throughout, not std-dev, despite the
      // "sigma" name -- see Residual::sigma_squared's docs) -- testing
      // whether livo_recon's per-residual measurement noise (res.
      // sigma_squared = 1e-3 + sigma_diag_squared, sigma_diag_squared =
      // point sensor noise projected onto the plane normal + the plane's
      // own fit uncertainty) is systematically smaller (more confident)
      // than FAST-LIVO2's structurally-identical R_inv formula, which would
      // explain livo_recon's tighter aggregate H_full/posterior even after
      // matching residual density via ds_leaf_size.
      double sum_sigma_squared = 0.0, min_sigma_squared = std::numeric_limits<double>::infinity(), max_sigma_squared = 0.0;
      double sum_weight = 0.0;
      // Tier breakdown (see Residual::match_tier's docs): how many of this
      // frame's residuals came from the primary voxel vs. the single
      // directional neighbor (also present in FAST-LIVO2) vs. the full
      // neighborhood-box search (livo_recon-only) -- debugging whether a
      // spike in tier-2 usage correlates with degraded match quality
      // (rising mean_sigma_squared/max_abs_r) during dynamic motion.
      int n_tier0 = 0, n_tier1 = 0, n_tier2 = 0;
      // History (774-785): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-774
      double sum_chi2 = 0.0;
      for (const auto& r : residuals_)
      {
        mean_normal += r.normal;
        const double abs_r = std::abs(r.r);
        mean_abs_r += abs_r;
        max_abs_r = std::max(max_abs_r, abs_r);

        sum_sigma_squared += r.sigma_squared;
        min_sigma_squared = std::min(min_sigma_squared, r.sigma_squared);
        max_sigma_squared = std::max(max_sigma_squared, r.sigma_squared);
        sum_weight += 1.0 / r.sigma_squared;
        sum_chi2 += (r.r * r.r) / r.sigma_squared;

        if (r.match_tier == 0) ++n_tier0;
        else if (r.match_tier == 1) ++n_tier1;
        else ++n_tier2;
      }
      if (!residuals_.empty())
      {
        mean_normal /= static_cast<double>(residuals_.size());
        mean_abs_r  /= static_cast<double>(residuals_.size());
      }
      const double mean_sigma_squared = residuals_.empty() ? 0.0 : sum_sigma_squared / residuals_.size();
      const double reduced_chi2 = residuals_.empty() ? 0.0 : sum_chi2 / residuals_.size();

      // Per-axis structure of this frame's position information sub-block
      // (H^T*W*H, position rows/cols only) -- unlike the scalar sum_weight
      // above, this tests whether the aggregate weight is concentrated
      // (anisotropic, one huge eigenvalue) vs. well-distributed across all
      // three translation axes. ekf_.HtH is 6x6 (local layout: rows/cols
      // 0-2 rotation, 3-5 position) as accumulated by solveSystem() this
      // iteration.
      const M3D H_pp = ekf_.HtH.block<3, 3>(3, 3);
      Eigen::SelfAdjointEigenSolver<M3D> h_es(H_pp);
      const V3D h_pp_eig = h_es.eigenvalues();

      // Rotation analog of H_pp_eig above: eigenvalues of the rotation-rows/
      // cols sub-block of this frame's H^T*W*H. A small minimum eigenvalue
      // here means at least one rotational DOF is weakly constrained by
      // this frame's residual set (an aperture-problem/gauge-freedom
      // signature -- e.g. residuals concentrated on too few distinct plane
      // orientations to pin down all 3 rotational DOF), regardless of how
      // low mean_sigma_squared/max_abs_r look per-residual. Logged to test whether
      // livo_recon's rotation estimate is chronically under-constrained
      // relative to FAST-LIVO2's (same field added there, see voxel_map.cpp)
      // or only becomes so transiently during fast motion.
      const M3D H_rr = ekf_.HtH.block<3, 3>(0, 0);
      Eigen::SelfAdjointEigenSolver<M3D> h_rr_es(H_rr);
      const V3D h_rr_eig = h_rr_es.eigenvalues();

      // History (837-854): see docs/livo_recon_changelog.md#src-processing-lio_processing.cpp-837
      const V3D v_pp_weak   = h_es.eigenvectors().col(0);
      const V3D v_pp_strong = h_es.eigenvectors().col(2);
      const V3D v_rr_weak   = h_rr_es.eigenvectors().col(0);
      const V3D v_rr_strong = h_rr_es.eigenvectors().col(2);
      double sum_frac_weak_pp = 0.0, sum_frac_strong_pp = 0.0;
      double sum_frac_weak_rr = 0.0, sum_frac_strong_rr = 0.0;
      for (const auto& res : residuals_)
      {
        const double hp_sqnorm = res.normal.squaredNorm();
        if (hp_sqnorm > 1e-12) {
          sum_frac_weak_pp   += std::pow(res.normal.dot(v_pp_weak),   2) / hp_sqnorm;
          sum_frac_strong_pp += std::pow(res.normal.dot(v_pp_strong), 2) / hp_sqnorm;
        }
        const double hr_sqnorm = res.point_cross_normal.squaredNorm();
        if (hr_sqnorm > 1e-12) {
          sum_frac_weak_rr   += std::pow(res.point_cross_normal.dot(v_rr_weak),   2) / hr_sqnorm;
          sum_frac_strong_rr += std::pow(res.point_cross_normal.dot(v_rr_strong), 2) / hr_sqnorm;
        }
      }
      const double n_res_d = std::max<double>(1.0, static_cast<double>(residuals_.size()));
      const double mean_frac_weak_pp   = sum_frac_weak_pp   / n_res_d;
      const double mean_frac_strong_pp = sum_frac_strong_pp / n_res_d;
      const double mean_frac_weak_rr   = sum_frac_weak_rr   / n_res_d;
      const double mean_frac_strong_rr = sum_frac_strong_rr / n_res_d;

      // VoxelStats (converged/disabled/init_frames_sum/etc) describes a
      // convergence-lock-in concept only VoxelMap has, so this diagnostic is gated on
      // an actual VoxelMap rather than promoted to the MapBackend interface.
      std::ostringstream voxel_stats_dbg;
      if (auto* vm = dynamic_cast<VoxelMap*>(voxel_map_.get())) {
        voxel_stats_dbg << "  init_frames_sum=" << vm->stats()->init_frames_sum.load()
                        << "  init_count=" << vm->stats()->init_count.load()
                        << "  update_frames_sum=" << vm->stats()->update_frames_sum.load()
                        << "  update_count=" << vm->stats()->update_count.load();
      } else {
        voxel_stats_dbg << "  init_frames_sum=n/a  init_count=n/a  update_frames_sum=n/a  update_count=n/a";
      }

      std::ostringstream dbg;
      dbg << std::fixed << std::setprecision(6)
          << "[lio]  t_abs=" << t_abs
          << "  iters=" << iter + 1
          << "  stop=" << stop
          << "  n_residuals=" << residuals_.size()
          << "  tier0=" << n_tier0 << "  tier1=" << n_tier1 << "  tier2=" << n_tier2
          << "  miss_coverage=" << n_miss_coverage_ << "  miss_mismatch=" << n_miss_mismatch_
          << "  tier0_miss_coverage=" << n_tier0_miss_coverage_ << "  tier0_miss_mismatch=" << n_tier0_miss_mismatch_
          << "  n_map_pts=" << voxel_map_->last_n_map_pts_ << "  n_active_voxels=" << voxel_map_->last_n_active_voxels_
          << voxel_stats_dbg.str()
          << "  pos=[" << state_->pos().transpose() << "]"
          << "  vel=[" << state_->vel().transpose() << "]"
          << "  quat(wxyz)=[" << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << "]"
          << "  pos_before=[" << mg.pos_after_imu.transpose() << "]"
          << "  quat_before(wxyz)=[" << q_before.w() << " " << q_before.x() << " " << q_before.y() << " " << q_before.z() << "]"
          << "  total_dtheta=[" << total_dtheta.transpose() << "]"
          << "  total_dt=[" << total_dt.transpose() << "]"
          << "  mean_normal=[" << mean_normal.transpose() << "]"
          << "  mean_normal_norm=" << mean_normal.norm()
          << "  mean_abs_r=" << mean_abs_r
          << "  max_abs_r=" << max_abs_r
          << std::scientific
          << "  reduced_chi2=" << reduced_chi2
          << "  mean_sigma_squared=" << mean_sigma_squared
          << "  min_sigma_squared=" << (residuals_.empty() ? 0.0 : min_sigma_squared)
          << "  max_sigma_squared=" << max_sigma_squared
          << "  sum_weight=" << sum_weight
          << "  H_pp_eig=[" << h_pp_eig.transpose() << "]"
          << "  H_rr_eig=[" << h_rr_eig.transpose() << "]"
          << "  mean_frac_weak_pp=" << mean_frac_weak_pp
          << "  mean_frac_strong_pp=" << mean_frac_strong_pp
          << "  mean_frac_weak_rr=" << mean_frac_weak_rr
          << "  mean_frac_strong_rr=" << mean_frac_strong_rr
          << "  trace(P_RR)=" << P_RR.trace()
          << "  trace(P_PP)=" << P_PP.trace()
          << "  trace(P_VV)=" << P_VV.trace()
          << "  norm(P_PV)=" << P_PV.norm();

      // Full post-LIO state + covariance dump, for direct raw-value
      // comparison against FAST-LIVO2's equivalent (see LIVMapper.cpp's
      // handleLIO() debug block) -- everything above this point already
      // summarizes P via trace/norm of individual blocks, but a from-
      // scratch cross-system comparison (e.g. per-axis correction size vs.
      // per-axis covariance, or reconstructing the full Kalman gain) needs
      // the raw entries, not just those derived scalars. cov_dim is
      // state_->dimState() (9 always: R,P,V, plus BG/BA/gravity if
      // estimated); cov_flat is P flattened ROW-MAJOR (Eigen's default
      // storage is column-major, so this is NOT a raw memory dump).
      dbg << "  dim=" << P.rows()
          << "  bias_gyr=[" << state_->biasGyr().transpose() << "]"
          << "  bias_acc=[" << state_->biasAcc().transpose() << "]"
          << "  gravity=[" << state_->gravity().transpose() << "]"
          << "  cov_flat=[";
      for (int r = 0; r < P.rows(); r++)
        for (int c = 0; c < P.cols(); c++)
          dbg << P(r, c) << (r == P.rows() - 1 && c == P.cols() - 1 ? "" : " ");
      dbg << "]";

      debugLogLio(dbg.str());
    }

    if (opts_.scan_log_en && scan_stats_valid_) writeScanLogRow(mg, scan_stats_);          // R73

    logStateTrace("post_lio", mg, static_cast<int>(residuals_.size()), iter);

    // CQ-60 item 5: Tier 1's shared NEES machinery -- independent of
    // log_debug_en, mirrors LioProcCoupled's own hook exactly.
    if (opts_.nees_per_dof_en) {
      const Eigen::MatrixXd& P = state_->cov();
      const int iP = StateGroup::idxP(), iR = StateGroup::idxR();
      if (P.rows() >= iP + 3 && P.cols() >= iP + 3 && P.rows() >= iR + 3 && P.cols() >= iR + 3) {
        Eigen::Matrix<double, 6, 6> P6;
        P6.block<3, 3>(0, 0) = P.block<3, 3>(iR, iR);
        P6.block<3, 3>(3, 3) = P.block<3, 3>(iP, iP);
        P6.block<3, 3>(0, 3) = P.block<3, 3>(iR, iP);
        // CQ-62: was P.block(iP,iR).transpose() -- the SAME extraction
        // bug as LioProcCoupled's own equivalent hook (see its comment
        // for the full derivation). P is symmetric, so the extra
        // transpose silently re-flips this back to P.block(iR,iP) -- the
        // same value already placed at (0,3), not its true transpose --
        // making P6 not actually a principal submatrix of P at all.
        P6.block<3, 3>(3, 0) = P.block<3, 3>(iP, iR);
        const double t_abs_nees = mg.image.t + data_queues_->start_time;
        iekf_tier1_nees_.addScan("tier1_iekf", voxel_map_->frame_idx_, t_abs_nees,
                                       state_->rot(), state_->pos(), P6);
      }
    }

    return oss.str();
  }
}

// Voxel map update
}
