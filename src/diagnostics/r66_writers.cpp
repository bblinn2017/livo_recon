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
    out << "frame_index,t_abs_s,frame_dt_s,n_lidar_raw,n_points_kept,n_imu_samples\n";
  }
  const double dt = have_prev ? (t_abs_s - prev_t) : 0.0;
  prev_t = t_abs_s;
  have_prev = true;
  out << frame_index << ',' << std::setprecision(17) << t_abs_s << ',' << dt << ',' << n_lidar_raw << ','
      << n_points_kept << ',' << n_imu_samples << '\n';
  out.flush();
}

void writeCoupledScanStatus(long scan_id, double t_abs_s, const std::string& report, int iterations_this_scan,
                            int thrown, const std::string& what)
{
  static PersistentLogStream log("coupled_scan_status.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first) out << "scan_id,t_abs_s,report,iterations_this_scan,thrown,exception_message\n";
  auto clean = [](std::string x) {
    for (char& c : x) if (c == ',' || c == '\n' || c == '\r') c = ';';
    return x;
  };
  out << scan_id << ',' << std::setprecision(17) << t_abs_s << ',' << clean(report) << ',' << iterations_this_scan
      << ',' << thrown << ',' << clean(what) << '\n';
  out.flush();
}

}  // namespace livo_recon
