#include "livo_recon/diagnostics/lidar_window_dump.h"
#include "livo_recon/utils/data/data_wrappers.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"

#include <iomanip>

namespace livo_recon
{

void writeLidarWindowPoints(int scan_id, double t_abs, const std::vector<PointXYZCov>& points)
{
  static PersistentLogStream log("lidar_window_dump.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first)
    out << "scan_id,t_abs,x,y,z\n";
  out << std::setprecision(17);
  for (const auto& p : points)
    out << scan_id << ',' << t_abs << ',' << std::setprecision(9) << p.point.x() << ','
        << p.point.y() << ',' << p.point.z() << '\n';
  out.flush();
}

}  // namespace livo_recon
