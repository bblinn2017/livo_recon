#include "livo_recon/diagnostics/lidar_image_bounds.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"

#include <fstream>

namespace livo_recon {

void writeLidarImageBoundsRow(int scan_id, double t_abs, int n_points,
                              int n_behind_camera, int n_out_of_bounds,
                              int n_visible)
{
  static PersistentLogStream log("lidar_image_bounds.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first)
    out << "scan_id,t_abs,n_points,n_behind_camera,n_out_of_bounds,n_visible,frac_visible\n";
  out << scan_id << ',' << t_abs << ',' << n_points << ',' << n_behind_camera << ','
      << n_out_of_bounds << ',' << n_visible << ','
      << (n_points ? static_cast<double>(n_visible) / n_points : 0.0) << '\n';
  out.flush();
}

}  // namespace livo_recon
