#include "livo_recon/utils/log/bag_clock.h"
#include "livo_recon/diagnostics/r66_writers.h"

#include "livo_recon/diagnostics/log/debug_log_dir.h"

#include <fstream>
#include <iomanip>

namespace livo_recon
{

void writeFrameWindowRow(std::size_t frame_index, double t_abs_s, std::size_t n_lidar_raw,
                         std::size_t n_points_kept, int n_imu_samples)
{
  static PersistentLogStream log("frame_window.csv");
  static double prev_t = 0.0;
  static bool have_prev = false;
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first)
  {
    have_prev = false;
    out << "frame_index,t_abs_s,t_rel_s,frame_dt_s,n_lidar_raw,n_points_kept,n_imu_samples\n";
  }
  const double dt = have_prev ? (t_abs_s - prev_t) : 0.0;
  prev_t = t_abs_s;
  have_prev = true;
  out << frame_index << ',' << std::setprecision(17) << t_abs_s << ',' << tRelString(t_abs_s) << ',' << dt << ',' << n_lidar_raw << ','
      << n_points_kept << ',' << n_imu_samples << '\n';
  out.flush();
}

}  // namespace livo_recon
