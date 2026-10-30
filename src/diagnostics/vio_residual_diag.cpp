#include "livo_recon/diagnostics/vio_residual_diag.h"
#include "livo_recon/utils/log/bag_clock.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"

#include <iomanip>

namespace livo_recon
{

void writeVioResidualDiagRow(int scan_id, double t_abs, bool ok, float avg_error,
                             float avg_abs_pixel_residual, int n_valid)
{
  static PersistentLogStream log("vio_residual_diag.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first)
    out << "scan_id,t_abs,t_rel_s,ok,avg_error,avg_abs_pixel_residual,n_valid\n";
  out << std::setprecision(17) << scan_id << ',' << t_abs << ',' << tRelString(t_abs) << ','
      << (ok ? 1 : 0) << ',' << std::setprecision(9) << avg_error << ',' << avg_abs_pixel_residual
      << ',' << n_valid << '\n';
  out.flush();
}

}  // namespace livo_recon
