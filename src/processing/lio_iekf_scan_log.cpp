// R73 (F-415): per-scan update records and scan_log.csv for the iekf estimator. Everything here is write-only with respect to the run.
// (R97: the shadow LIO update that shared this file was removed; the file was renamed from lio_iekf_shadow.cpp.)
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
