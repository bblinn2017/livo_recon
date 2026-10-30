#include "livo_recon/utils/log/bag_clock.h"
#include "livo_recon/diagnostics/motion_q_writer.h"

#include "livo_recon/diagnostics/log/debug_log_dir.h"

#include <fstream>
#include <iomanip>

namespace livo_recon
{

void writeMotionQSampleDiagnostic(
    size_t scan_index, size_t sample_index, double t_abs,
    const std::string& model, const V3D& dynamic_acc, const V3D& dynamic_gyr,
    const V3D& debiased_acc_energy, const V3D& debiased_gyr_energy,
    const V3D& q_acc, const V3D& q_gyr)
{
  if (scan_index != 1) return;
  static PersistentLogStream log("motion_q_first_frame.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first)
    out << "sample_index,t_abs,model,dynamic_acc_x,dynamic_acc_y,dynamic_acc_z,"
           "dynamic_gyr_x,dynamic_gyr_y,dynamic_gyr_z,q_acc_x,q_acc_y,q_acc_z,"
           "q_gyr_x,q_gyr_y,q_gyr_z,debiased_acc_energy_x,"
           "debiased_acc_energy_y,debiased_acc_energy_z,"
           "debiased_gyr_energy_x,debiased_gyr_energy_y,"
           "debiased_gyr_energy_z\n";
  out << std::setprecision(17) << sample_index << ',' << t_abs << ',' << model
      << ',' << dynamic_acc.x() << ',' << dynamic_acc.y() << ',' << dynamic_acc.z()
      << ',' << dynamic_gyr.x() << ',' << dynamic_gyr.y() << ',' << dynamic_gyr.z()
      << ',' << q_acc.x() << ',' << q_acc.y() << ',' << q_acc.z()
      << ',' << q_gyr.x() << ',' << q_gyr.y() << ',' << q_gyr.z()
      << ',' << debiased_acc_energy.x() << ',' << debiased_acc_energy.y()
      << ',' << debiased_acc_energy.z() << ',' << debiased_gyr_energy.x()
      << ',' << debiased_gyr_energy.y() << ',' << debiased_gyr_energy.z() << '\n';
  out.flush();
}

void writeMotionQStepDiagnostic(
    size_t scan_index, size_t step_index, double t_abs,
    const V3D& q_acc, const V3D& q_gyr)
{
  static PersistentLogStream log("motion_q_step.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first)
    out << "scan_index,step_index,t_abs,q_acc_x,q_acc_y,q_acc_z,q_gyr_x,q_gyr_y,q_gyr_z\n";
  out << std::setprecision(17) << scan_index << ',' << step_index << ',' << t_abs << ','
      << q_acc.x() << ',' << q_acc.y() << ',' << q_acc.z() << ','
      << q_gyr.x() << ',' << q_gyr.y() << ',' << q_gyr.z() << '\n';
  out.flush();
}

void writeMotionQScanDiagnostic(
    size_t scan_index, double t_abs, const std::string& model,
    const V3D& acc_scale, const V3D& gyr_scale,
    size_t sample_count, double mean_dynamic_acc_norm,
    double mean_dynamic_gyr_norm, const V3D& mean_q_acc,
    const V3D& mean_q_gyr, const V3D& mean_debiased_acc_energy,
    const V3D& mean_debiased_gyr_energy)
{
  static PersistentLogStream log("motion_q_scan.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first)
    // 2026-10-28: acc_scale/gyro_scale expanded from one scalar column each to per-axis
    // (anisotropic process_noise support) -- _x/_y/_z suffixes, same convention as every other
    // per-axis column here. All 3 are identical (isotropic) unless the config sets a 3-vector.
    out << "scan_index,t_abs,t_rel_s,model,acc_scale_x,acc_scale_y,acc_scale_z,"
           "gyro_scale_x,gyro_scale_y,gyro_scale_z,n_samples,mean_dynamic_acc_norm,"
           "mean_dynamic_gyr_norm,mean_q_acc_x,mean_q_acc_y,mean_q_acc_z,"
           "mean_q_gyr_x,mean_q_gyr_y,mean_q_gyr_z,mean_debiased_acc_energy_x,"
           "mean_debiased_acc_energy_y,mean_debiased_acc_energy_z,"
           "mean_debiased_gyr_energy_x,mean_debiased_gyr_energy_y,"
           "mean_debiased_gyr_energy_z\n";
  out << std::setprecision(17) << scan_index << ',' << t_abs << ',' << tRelString(t_abs) << ',' << model
      << ',' << acc_scale.x() << ',' << acc_scale.y() << ',' << acc_scale.z()
      << ',' << gyr_scale.x() << ',' << gyr_scale.y() << ',' << gyr_scale.z() << ','
      << sample_count << ',' << mean_dynamic_acc_norm
      << ',' << mean_dynamic_gyr_norm << ',' << mean_q_acc.x() << ','
      << mean_q_acc.y() << ',' << mean_q_acc.z() << ',' << mean_q_gyr.x()
      << ',' << mean_q_gyr.y() << ',' << mean_q_gyr.z() << ','
      << mean_debiased_acc_energy.x() << ',' << mean_debiased_acc_energy.y()
      << ',' << mean_debiased_acc_energy.z() << ','
      << mean_debiased_gyr_energy.x() << ',' << mean_debiased_gyr_energy.y()
      << ',' << mean_debiased_gyr_energy.z() << '\n';
  out.flush();
}

}  // namespace livo_recon
