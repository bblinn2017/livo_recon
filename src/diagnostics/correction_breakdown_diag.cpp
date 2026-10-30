#include "livo_recon/diagnostics/correction_breakdown_diag.h"
#include "livo_recon/utils/log/bag_clock.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"

#include <iomanip>

namespace livo_recon
{

void writeCorrectionBreakdownRow(
    int scan_id, double t_abs, const std::string& mode,
    const Eigen::Vector3d& ask_vio_dtheta, const Eigen::Vector3d& ask_vio_dt, int n_meas_vio,
    bool have_lio_ask, const Eigen::Vector3d& ask_lio_dtheta, const Eigen::Vector3d& ask_lio_dt, int n_meas_lio,
    bool have_counterfactual,
    const Eigen::Vector3d& counterfactual_lio_only_dtheta, const Eigen::Vector3d& counterfactual_lio_only_dt,
    const Eigen::Vector3d& counterfactual_joint_dtheta, const Eigen::Vector3d& counterfactual_joint_dt,
    double tr_HtH_rot_lio, double tr_HtH_pos_lio, double tr_HtH_rot_vio, double tr_HtH_pos_vio)
{
  static PersistentLogStream log("correction_breakdown_diag.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first)
    out << "scan_id,t_abs,t_rel_s,mode,"
        << "ask_vio_dtheta_x,ask_vio_dtheta_y,ask_vio_dtheta_z,ask_vio_dt_x,ask_vio_dt_y,ask_vio_dt_z,n_meas_vio,"
        << "have_lio_ask,ask_lio_dtheta_x,ask_lio_dtheta_y,ask_lio_dtheta_z,ask_lio_dt_x,ask_lio_dt_y,ask_lio_dt_z,n_meas_lio,"
        << "have_counterfactual,"
        << "cf_lio_only_dtheta_x,cf_lio_only_dtheta_y,cf_lio_only_dtheta_z,cf_lio_only_dt_x,cf_lio_only_dt_y,cf_lio_only_dt_z,"
        << "cf_joint_dtheta_x,cf_joint_dtheta_y,cf_joint_dtheta_z,cf_joint_dt_x,cf_joint_dt_y,cf_joint_dt_z,"
        << "tr_HtH_rot_lio,tr_HtH_pos_lio,tr_HtH_rot_vio,tr_HtH_pos_vio\n";
  out << std::setprecision(17) << scan_id << ',' << t_abs << ',' << tRelString(t_abs) << ',' << mode << ','
      << std::setprecision(9)
      << ask_vio_dtheta.x() << ',' << ask_vio_dtheta.y() << ',' << ask_vio_dtheta.z() << ','
      << ask_vio_dt.x() << ',' << ask_vio_dt.y() << ',' << ask_vio_dt.z() << ',' << n_meas_vio << ','
      << (have_lio_ask ? 1 : 0) << ','
      << ask_lio_dtheta.x() << ',' << ask_lio_dtheta.y() << ',' << ask_lio_dtheta.z() << ','
      << ask_lio_dt.x() << ',' << ask_lio_dt.y() << ',' << ask_lio_dt.z() << ',' << n_meas_lio << ','
      << (have_counterfactual ? 1 : 0) << ','
      << counterfactual_lio_only_dtheta.x() << ',' << counterfactual_lio_only_dtheta.y() << ',' << counterfactual_lio_only_dtheta.z() << ','
      << counterfactual_lio_only_dt.x() << ',' << counterfactual_lio_only_dt.y() << ',' << counterfactual_lio_only_dt.z() << ','
      << counterfactual_joint_dtheta.x() << ',' << counterfactual_joint_dtheta.y() << ',' << counterfactual_joint_dtheta.z() << ','
      << counterfactual_joint_dt.x() << ',' << counterfactual_joint_dt.y() << ',' << counterfactual_joint_dt.z() << ','
      << tr_HtH_rot_lio << ',' << tr_HtH_pos_lio << ',' << tr_HtH_rot_vio << ',' << tr_HtH_pos_vio << '\n';
  out.flush();
}

}  // namespace livo_recon
