#include "livo_recon/diagnostics/vio_correction.h"
#include "livo_recon/utils/log/bag_clock.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"

#include <algorithm>
#include <cmath>
#include <iomanip>

namespace livo_recon
{
namespace
{
double rotationAngleDeg(const Eigen::Matrix3d& R_a, const Eigen::Matrix3d& R_b)
{
  const Eigen::Matrix3d R_delta = R_a.transpose() * R_b;
  const double cos_theta = std::clamp((R_delta.trace() - 1.0) / 2.0, -1.0, 1.0);
  return std::acos(cos_theta) * 180.0 / M_PI;
}
}  // namespace

void writeVioCorrectionRow(int scan_id, double t_abs,
                           const Eigen::Vector3d& pos_after_lio, const Eigen::Matrix3d& rot_after_lio,
                           const Eigen::Vector3d& pos_after_vio, const Eigen::Matrix3d& rot_after_vio)
{
  static PersistentLogStream log("vio_correction.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first)
    out << "scan_id,t_abs,t_rel_s,dpx,dpy,dpz,dpos_norm,drot_deg\n";
  const Eigen::Vector3d dpos = pos_after_vio - pos_after_lio;
  const double drot_deg = rotationAngleDeg(rot_after_lio, rot_after_vio);
  out << std::setprecision(17) << scan_id << ',' << t_abs << ',' << tRelString(t_abs) << ','
      << std::setprecision(9) << dpos.x() << ',' << dpos.y() << ',' << dpos.z() << ','
      << dpos.norm() << ',' << drot_deg << '\n';
  out.flush();
}

void writeCombinedVioContributionRow(int scan_id, double t_abs, float avg_err_vio, int n_meas_vio,
                                     bool vio_ever_contributed)
{
  static PersistentLogStream log("combined_vio_contribution.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first)
    out << "scan_id,t_abs,t_rel_s,avg_err_vio,n_meas_vio,vio_ever_contributed\n";
  out << std::setprecision(17) << scan_id << ',' << t_abs << ',' << tRelString(t_abs) << ','
      << std::setprecision(9) << avg_err_vio << ',' << n_meas_vio << ',' << (vio_ever_contributed ? 1 : 0) << '\n';
  out.flush();
}

}  // namespace livo_recon
