#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace livo_recon
{

// R66 item 0b: one row per MeasureGroup (frame), written after deskew + downsample.
// Columns: frame_index, t_abs_s, frame_dt_s, n_lidar_raw, n_points_kept, n_imu_samples.
// frame_dt_s is the difference to the previous written frame (0 for the first row).
void writeFrameWindowRow(std::size_t frame_index, double t_abs_s, std::size_t n_lidar_raw,
                         std::size_t n_points_kept, int n_imu_samples);

// R66 (C-D4, C-D6): one row per coupled-estimator scan. report is the estimator's own return string (commas are
// replaced by ';'), iterations_this_scan the GN iterations completed, thrown=1 when the solve threw and the scan
// was skipped under estimator/coupled/on_solver_error=skip_scan, what the exception text (empty otherwise).
void writeCoupledScanStatus(long scan_id, double t_abs_s, const std::string& report, int iterations_this_scan,
                            int thrown, const std::string& what);

}  // namespace livo_recon
