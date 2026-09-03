#include "livo_recon/utils/log/consistency_log.h"
#include "livo_recon/utils/log/debug_log_dir.h"

#include <fstream>
#include <mutex>

namespace livo_recon {

// History (9-13): see docs/livo_recon_changelog.md#src-utils-log-consistency_log.cpp-9
void logConsistencyScan(const char* channel, int scan_id, double t, double dt,
                        double trP_pos, double trP_vel, double trP_att,
                        double omega_norm, double acc_norm,
                        double ba_x, double ba_y, double ba_z,
                        double bg_x, double bg_y, double bg_z)
{
  static bool first_call = true;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  std::ofstream ofs(debugLogPath("scan.csv"), first_call ? std::ios::trunc : std::ios::app);
  if (first_call)
    ofs << "channel,scan_id,t,dt,trP_pos,trP_vel,trP_att,omega_norm,acc_norm,"
           "ba_x,ba_y,ba_z,bg_x,bg_y,bg_z\n";
  first_call = false;
  ofs << channel << "," << scan_id << "," << t << "," << dt << "," << trP_pos << "," << trP_vel << ","
      << trP_att << "," << omega_norm << "," << acc_norm << ","
      << ba_x << "," << ba_y << "," << ba_z << "," << bg_x << "," << bg_y << "," << bg_z << "\n";
}

// History (29-33): see docs/livo_recon_changelog.md#src-utils-log-consistency_log.cpp-29
void logConsistencyNll(const char* channel, double t_abs, double nll, int n_residuals,
                       double pivot_ratio, double kalman_gain_norm)
{
  static bool first_call = true;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  std::ofstream ofs(debugLogPath("nll.txt"), first_call ? std::ios::trunc : std::ios::app);
  if (first_call)
    ofs << "channel,t,nll,n_residuals,pivot_ratio,kalman_gain_norm\n";
  first_call = false;
  ofs << channel << "," << t_abs << "," << nll << "," << n_residuals << "," << pivot_ratio
      << "," << kalman_gain_norm << "\n";
}

// History (48-53): see docs/livo_recon_changelog.md#src-utils-log-consistency_log.cpp-48
void logConsistencyVioCorr(double t_abs, int iter, double nu, double S, const char* residual_mode)
{
  static bool first_call = true;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  std::ofstream ofs(debugLogPath("vio_corr.csv"), first_call ? std::ios::trunc : std::ios::app);
  if (first_call)
    ofs << "t,iter,nu,S,residual_mode\n";
  first_call = false;
  ofs << t_abs << "," << iter << "," << nu << "," << S << "," << residual_mode << "\n";
}

}  // namespace livo_recon
