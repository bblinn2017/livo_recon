#include "livo_recon/diagnostics/stationary_diagnostic_writer.h"

#include "livo_recon/diagnostics/log/debug_log_dir.h"
#include "livo_recon/utils/algo/math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>

namespace livo_recon
{
namespace
{

uint64_t fnv1a(uint64_t hash, const void* data, size_t size)
{
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

uint64_t acceptedIndexHash(const std::vector<Residual>& residuals)
{
  std::vector<int> indices;
  indices.reserve(residuals.size());
  for (const Residual& residual : residuals) indices.push_back(residual.source_index);
  std::sort(indices.begin(), indices.end());
  uint64_t hash = 1469598103934665603ULL;
  for (const int index : indices) hash = fnv1a(hash, &index, sizeof(index));
  return hash;
}

uint64_t solveInputHash(const std::vector<Residual>& residuals)
{
  std::vector<const Residual*> ordered;
  ordered.reserve(residuals.size());
  for (const Residual& residual : residuals) ordered.push_back(&residual);
  std::sort(ordered.begin(), ordered.end(), [](const Residual* a, const Residual* b) {
    return a->source_index < b->source_index;
  });
  uint64_t hash = 1469598103934665603ULL;
  for (const Residual* residual : ordered) {
    hash = fnv1a(hash, &residual->source_index, sizeof(residual->source_index));
    hash = fnv1a(hash, &residual->r, sizeof(residual->r));
    hash = fnv1a(hash, &residual->sigma_squared, sizeof(residual->sigma_squared));
    hash = fnv1a(hash, residual->normal.data(), 3 * sizeof(double));
  }
  return hash;
}

void writeVector(std::ofstream& out, const V3D& value)
{
  out << value.x() << ',' << value.y() << ',' << value.z() << ',' << value.norm();
}

}  // namespace

void writeStationaryIterationDiagnostics(
    const std::string& test_id, const std::string& architecture,
    int scan_id, int iteration, double t_abs,
    const M3D& reference_R, const V3D& reference_p, const V3D& reference_v,
    const StateGroup& before, const StateGroup& after,
    const std::vector<Residual>& residuals)
{
  static PersistentLogStream log("stationary_iteration.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first) {
    out << "test_id,architecture,scan_id,iteration,t_abs,residual_count,"
           "reference_px,reference_py,reference_pz,reference_vx,reference_vy,reference_vz,"
           "before_px,before_py,before_pz,before_pos_distance,"
           "after_px,after_py,after_pz,after_pos_distance,"
           "ideal_dp_x,ideal_dp_y,ideal_dp_z,ideal_dp_norm,"
           "realized_dp_x,realized_dp_y,realized_dp_z,realized_dp_norm,"
           "before_attitude_error_x,before_attitude_error_y,before_attitude_error_z,before_attitude_distance,"
           "after_attitude_error_x,after_attitude_error_y,after_attitude_error_z,after_attitude_distance,"
           "ideal_dtheta_x,ideal_dtheta_y,ideal_dtheta_z,ideal_dtheta_norm,"
           "realized_dtheta_x,realized_dtheta_y,realized_dtheta_z,realized_dtheta_norm,"
           "before_vx,before_vy,before_vz,before_velocity_distance,"
           "after_vx,after_vy,after_vz,after_velocity_distance,"
           "ideal_dv_x,ideal_dv_y,ideal_dv_z,ideal_dv_norm,"
           "realized_dv_x,realized_dv_y,realized_dv_z,realized_dv_norm,"
           "position_distance_reduction,attitude_distance_reduction,velocity_distance_reduction,"
           "before_gravity_tilt_error,after_gravity_tilt_error,gravity_tilt_error_reduction,"
           "before_speed,after_speed,reference_velocity_norm\n";
  }
  const V3D before_p_error = before.pos() - reference_p;
  const V3D after_p_error = after.pos() - reference_p;
  const V3D before_r_error = Log(M3D(reference_R.transpose() * before.rot()));
  const V3D after_r_error = Log(M3D(reference_R.transpose() * after.rot()));
  const V3D before_v_error = before.vel() - reference_v;
  const V3D after_v_error = after.vel() - reference_v;
  const V3D ideal_dp = reference_p - before.pos();
  const V3D realized_dp = after.pos() - before.pos();
  const V3D ideal_dtheta = Log(M3D(before.rot().transpose() * reference_R));
  const V3D realized_dtheta = Log(M3D(before.rot().transpose() * after.rot()));
  const V3D ideal_dv = reference_v - before.vel();
  const V3D realized_dv = after.vel() - before.vel();
  const V3D gravity_world = V3D(0.0, 0.0, -1.0);
  const V3D reference_gravity_body = reference_R.transpose() * gravity_world;
  const V3D before_gravity_body = before.rot().transpose() * gravity_world;
  const V3D after_gravity_body = after.rot().transpose() * gravity_world;
  const auto vectorAngle = [](const V3D& a, const V3D& b) {
    return std::acos(std::max(-1.0, std::min(1.0,
        a.normalized().dot(b.normalized()))));
  };
  const double before_tilt = vectorAngle(reference_gravity_body, before_gravity_body);
  const double after_tilt = vectorAngle(reference_gravity_body, after_gravity_body);

  out << test_id << ',' << architecture << ',' << scan_id << ',' << iteration << ','
      << std::setprecision(17) << t_abs << ',' << residuals.size() << ',';
  out << reference_p.x() << ',' << reference_p.y() << ',' << reference_p.z() << ','
      << reference_v.x() << ',' << reference_v.y() << ',' << reference_v.z() << ',';
  writeVector(out, before_p_error); out << ',';
  writeVector(out, after_p_error); out << ',';
  writeVector(out, ideal_dp); out << ',';
  writeVector(out, realized_dp); out << ',';
  writeVector(out, before_r_error); out << ',';
  writeVector(out, after_r_error); out << ',';
  writeVector(out, ideal_dtheta); out << ',';
  writeVector(out, realized_dtheta); out << ',';
  writeVector(out, before_v_error); out << ',';
  writeVector(out, after_v_error); out << ',';
  writeVector(out, ideal_dv); out << ',';
  writeVector(out, realized_dv); out << ','
      << before_p_error.norm() - after_p_error.norm() << ','
      << before_r_error.norm() - after_r_error.norm() << ','
      << before_v_error.norm() - after_v_error.norm() << ','
      << before_tilt << ',' << after_tilt << ',' << before_tilt - after_tilt << ','
      << before.vel().norm() << ',' << after.vel().norm() << ','
      << reference_v.norm() << '\n';
  out.flush();
}

void writeGatingIterationDiagnostics(
    const std::string& test_id, const std::string& architecture,
    int scan_id, int iteration, double t_abs, bool gating_state_uncertainty,
    int input_points, int statistical_candidates,
    int statistical_gate_rejections, int coverage_misses, int mismatch_misses,
    double mean_gating_cov_trace, double max_gating_cov_trace,
    const std::vector<Residual>& residuals)
{
  static PersistentLogStream log("gating_all_scans.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first) {
    out << "test_id,architecture,scan_id,iteration,t_abs,gating_state_uncertainty,"
           "input_points,statistical_candidates,statistical_gate_rejections,"
           "coverage_misses,mismatch_misses,accepted_count,accepted_index_hash,"
           "solve_input_hash,accepted_index_sum,accepted_index_xor,"
           "mean_gating_cov_trace,max_gating_cov_trace,solve_uses_gating_covariance\n";
  }
  long long index_sum = 0;
  uint64_t index_xor = 0;
  for (const Residual& residual : residuals) {
    index_sum += residual.source_index;
    index_xor ^= static_cast<uint64_t>(static_cast<uint32_t>(residual.source_index));
  }
  out << test_id << ',' << architecture << ',' << scan_id << ',' << iteration << ','
      << std::setprecision(17) << t_abs << ','
      << (gating_state_uncertainty ? 1 : 0) << ',' << input_points << ','
      << statistical_candidates << ',' << statistical_gate_rejections << ','
      << coverage_misses << ',' << mismatch_misses << ',' << residuals.size() << ','
      << acceptedIndexHash(residuals) << ',' << solveInputHash(residuals) << ','
      << index_sum << ',' << index_xor << ',' << mean_gating_cov_trace << ','
      << max_gating_cov_trace << ",0\n";
  out.flush();
}

}  // namespace livo_recon
