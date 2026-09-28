#include "livo_recon/diagnostics/bootstrap_diagnostic_writer.h"

#include "livo_recon/diagnostics/log/debug_log_dir.h"

#include <iomanip>

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

}  // namespace

uint64_t hashBootstrapPopulation(const std::vector<PointXYZCov>& flattened)
{
  uint64_t hash = 1469598103934665603ULL;  // FNV-1a 64-bit offset basis
  for (const PointXYZCov& p : flattened) {
    hash = fnv1a(hash, p.point.data(), 3 * sizeof(double));
    hash = fnv1a(hash, p.raw_body_point.data(), 3 * sizeof(double));
    hash = fnv1a(hash, p.sensor_cov.data(), 9 * sizeof(double));
    hash = fnv1a(hash, &p.t, sizeof(p.t));
  }
  return hash;
}

void writeBootstrapPreparationDiagnostic(
    const std::vector<std::vector<PointXYZT>>& raw_observations,
    const std::vector<std::vector<PointXYZCov>>& prepared_observations)
{
  static PersistentLogStream log("bootstrap_preparation.txt");
  std::ofstream& out = log.stream();
  out << std::setprecision(17);

  size_t total_raw = 0, total_prepared = 0;
  std::vector<PointXYZCov> flattened;
  for (const auto& obs : prepared_observations) total_prepared += obs.size();
  flattened.reserve(total_prepared);
  for (const auto& obs : prepared_observations)
    flattened.insert(flattened.end(), obs.begin(), obs.end());
  for (const auto& obs : raw_observations) total_raw += obs.size();

  out << "schema bootstrap_preparation_v1\n"
      << "num_observations " << raw_observations.size() << '\n';
  for (size_t i = 0; i < raw_observations.size(); ++i) {
    const size_t prepared_count = i < prepared_observations.size() ? prepared_observations[i].size() : 0;
    out << "observation " << i << " raw_count " << raw_observations[i].size()
        << " prepared_count " << prepared_count << '\n';
  }
  out << "total_raw_point_count " << total_raw << '\n'
      << "total_prepared_point_count " << total_prepared << '\n'
      << "flattened_prepared_population_hash " << hashBootstrapPopulation(flattened) << '\n'
      << "end_bootstrap_preparation_v1\n";
  out.flush();
}

void writeBootstrapMapSeedDiagnostic(
    const std::string& mode,
    int frame_idx_before, int frame_idx_after,
    const StateGroup& state_before, const StateGroup& state_after,
    uint64_t flattened_population_hash, size_t flattened_population_count)
{
  static PersistentLogStream log("bootstrap_map_seed.txt");
  std::ofstream& out = log.stream();
  out << std::setprecision(17);

  const double p0_trace_before = state_before.cov().trace();
  const double p0_trace_after = state_after.cov().trace();

  out << "schema bootstrap_map_seed_v1\n"
      << "bootstrap_mode " << mode << '\n'
      << "frame_idx_before " << frame_idx_before << '\n'
      << "frame_idx_after " << frame_idx_after << '\n'
      << "flattened_population_count " << flattened_population_count << '\n'
      << "flattened_population_hash " << flattened_population_hash << '\n'
      << "state_position_before " << state_before.pos().transpose() << '\n'
      << "state_position_after " << state_after.pos().transpose() << '\n'
      << "state_rotation_before\n" << state_before.rot() << '\n'
      << "state_rotation_after\n" << state_after.rot() << '\n'
      << "state_velocity_before " << state_before.vel().transpose() << '\n'
      << "state_velocity_after " << state_after.vel().transpose() << '\n'
      << "P0_trace_before " << p0_trace_before << '\n'
      << "P0_trace_after " << p0_trace_after << '\n'
      << "state_unchanged "
      << ((state_before.pos() - state_after.pos()).norm() == 0.0 &&
          (state_before.vel() - state_after.vel()).norm() == 0.0 &&
          (state_before.rot() - state_after.rot()).norm() == 0.0 &&
          p0_trace_before == p0_trace_after ? "true" : "false")
      << '\n'
      << "end_bootstrap_map_seed_v1\n";
  out.flush();
}

}  // namespace livo_recon
