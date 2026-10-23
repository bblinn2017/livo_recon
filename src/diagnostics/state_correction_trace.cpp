#include "livo_recon/diagnostics/state_correction_trace.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"

#include <Eigen/Geometry>
#include <fstream>
#include <iomanip>

namespace livo_recon {

namespace {
void writeQuat(std::ofstream& out, const M3D& R) {
  const Eigen::Quaterniond q(R);
  out << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z();
}
}  // namespace

void writeStateCorrectionTraceRow(
    int scan_id, double t_abs,
    const V3D& pos_before_imu, const M3D& rot_before_imu, const V3D& vel_before_imu,
    const V3D& pos_after_imu, const M3D& rot_after_imu, const V3D& vel_after_imu,
    const V3D& pos_after_lio, const M3D& rot_after_lio, const V3D& vel_after_lio,
    const V3D& pos_after_vio, const M3D& rot_after_vio)
{
  static PersistentLogStream log("state_correction_trace.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first)
    out << "scan_id,t_abs,"
           "px_before_imu,py_before_imu,pz_before_imu,qw_before_imu,qx_before_imu,qy_before_imu,qz_before_imu,vx_before_imu,vy_before_imu,vz_before_imu,"
           "px_after_imu,py_after_imu,pz_after_imu,qw_after_imu,qx_after_imu,qy_after_imu,qz_after_imu,vx_after_imu,vy_after_imu,vz_after_imu,"
           "px_after_lio,py_after_lio,pz_after_lio,qw_after_lio,qx_after_lio,qy_after_lio,qz_after_lio,vx_after_lio,vy_after_lio,vz_after_lio,"
           "px_after_vio,py_after_vio,pz_after_vio,qw_after_vio,qx_after_vio,qy_after_vio,qz_after_vio\n";
  out << std::setprecision(17) << scan_id << ',' << t_abs << ',';
  out << std::setprecision(9);
  out << pos_before_imu.x() << ',' << pos_before_imu.y() << ',' << pos_before_imu.z() << ',';
  writeQuat(out, rot_before_imu);
  out << ',' << vel_before_imu.x() << ',' << vel_before_imu.y() << ',' << vel_before_imu.z() << ',';
  out << pos_after_imu.x() << ',' << pos_after_imu.y() << ',' << pos_after_imu.z() << ',';
  writeQuat(out, rot_after_imu);
  out << ',' << vel_after_imu.x() << ',' << vel_after_imu.y() << ',' << vel_after_imu.z() << ',';
  out << pos_after_lio.x() << ',' << pos_after_lio.y() << ',' << pos_after_lio.z() << ',';
  writeQuat(out, rot_after_lio);
  out << ',' << vel_after_lio.x() << ',' << vel_after_lio.y() << ',' << vel_after_lio.z() << ',';
  out << pos_after_vio.x() << ',' << pos_after_vio.y() << ',' << pos_after_vio.z() << ',';
  writeQuat(out, rot_after_vio);
  out << '\n';
  out.flush();
}

}  // namespace livo_recon
