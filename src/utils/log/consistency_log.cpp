#include "livo_recon/utils/log/consistency_log.h"
#include "livo_recon/utils/log/debug_log_dir.h"

#include <fstream>
#include <mutex>

namespace livo_recon {

// T0-D (2026-08-31): scan.csv for scripts/analysis/consistency.py -- see
// LioProcOptions::log_consistency_scan_en's doc comment for the column
// list. G6 (2026-09-01): hoisted out of lio_processing.cpp's anonymous
// namespace so CombinedProc can also log its own scan rows, distinguished
// by the new leading `channel` column ("lio"/"vio"/"rollback").
void logConsistencyScan(const char* channel, int scan_id, double t, double dt,
                        double trP_pos, double trP_vel, double trP_att,
                        double omega_norm, double acc_norm)
{
  static bool first_call = true;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  std::ofstream ofs(debugLogPath("scan.csv"), first_call ? std::ios::trunc : std::ios::app);
  if (first_call)
    ofs << "channel,scan_id,t,dt,trP_pos,trP_vel,trP_att,omega_norm,acc_norm\n";
  first_call = false;
  ofs << channel << "," << scan_id << "," << t << "," << dt << "," << trP_pos << "," << trP_vel << ","
      << trP_att << "," << omega_norm << "," << acc_norm << "\n";
}

// T0-E (2026-08-31): nll.txt -- see LioProcOptions::log_nll_en's doc
// comment for the column/formula. G6 (2026-09-01): hoisted out of
// lio_processing.cpp's anonymous namespace, same reasoning as
// logConsistencyScan() above -- CombinedProc logs "lio"/"vio"/"rollback"
// rows for the same frame, so the two channels can finally be compared.
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

// G6 (2026-09-01): VIO-side correspondence log, mirroring corr.csv's
// nu/S columns so both channels are scored by the same statistic. Logged
// BEFORE any further rejection at the call site (vio_accumulator.cpp),
// matching corr.csv's own "log candidates before the gate" convention --
// T0-D measured a NIS computed over only the survivors of its own gate as
// biased low (2.59 -> 1.85 on a fixture) when this was done wrong.
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
