#pragma once

// R95 (Bryce 2026-10-11): the per-scan summary of z squared = r^2 / sigma_squared, written as zsq_per_scan.csv under lio/zsq_log_en (default false: nothing is written, bit-identical).
// Header-only and ROS-free so that tests/test_zsq_summary_r95.cpp can check the format and the arithmetic alone; templated on the residual type (livo_recon::Residual in the node, a mock in the test).
//
// One row per (scan, iteration). The node writes iteration 0 (the residuals at the propagated state) and the last executed iteration (the residuals of the last iteration, as the R73 last-iteration dump does);
// a scan whose only iteration is 0 has one row. Definitions (a "valid" row has finite r, finite sigma_squared and sigma_squared > 0; invalid rows are counted in n_invalid and are in no other column):
//   z2 = r^2 / sigma_squared, where sigma_squared is the variance the solve uses as the row weight (w = 1 / sigma_squared); this is the quantity of the nll in lio_iekf.cpp and of the R93 reducer.
//   n_rows       all rows (valid + invalid)
//   sum_r2       sum of r^2 over valid rows;  sum_z2 sum of z2 over valid rows
//   z2_p50, z2_p90, z2_max   nearest-rank quantiles (index = floor(q (n - 1) + 0.5) of the ascending valid z2) and the maximum; 0 with no valid row
//   n_z2_gt4, n_z2_gt9       valid rows with z2 > 4, z2 > 9
//   n_plane, sum_z2_plane, n_plane_gt9   rows of row_kind 0 (plane rows; a kind outside 0..6 counts here, as in line_rows_per_scan.csv)
//   n_line, sum_z2_line, n_line_gt9      rows of row_kind 1..6 (hard line u0, u1; smooth u0, u1, u2; distance)
//   n_gate, sum_z2_gate, n_gate_gt9      valid rows whose sigma_gate_squared (the variance the admission gate used, floor + sensor + pose + plane terms) is finite and > 0: z2g = r^2 / sigma_gate_squared
// The caller sets std::setprecision(10) once. Write-only: nothing in the estimator reads it.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <vector>

namespace livo_recon
{
namespace zsq_summary
{
inline const char* header()
{
  return "scan_id,iter,n_rows,n_invalid,sum_r2,sum_z2,z2_p50,z2_p90,z2_max,n_z2_gt4,n_z2_gt9,n_plane,sum_z2_plane,n_plane_gt9,n_line,sum_z2_line,n_line_gt9,n_gate,sum_z2_gate,n_gate_gt9\n";
}

template <class R>
inline void writeRow(std::ostream& o, int scan_id, int iteration, const std::vector<R>& residuals)
{
  long n_invalid = 0, n_gt4 = 0, n_gt9 = 0, n_plane = 0, n_plane_gt9 = 0, n_line = 0, n_line_gt9 = 0, n_gate = 0, n_gate_gt9 = 0;
  double sum_r2 = 0.0, sum_z2 = 0.0, sum_plane = 0.0, sum_line = 0.0, sum_gate = 0.0;
  std::vector<double> z;
  z.reserve(residuals.size());
  for (const auto& r : residuals) {
    const double s = r.sigma_squared;
    if (!(std::isfinite(r.r) && std::isfinite(s) && s > 0.0)) { ++n_invalid; continue; }
    const double r2 = r.r * r.r;
    const double z2 = r2 / s;
    sum_r2 += r2;
    sum_z2 += z2;
    z.push_back(z2);
    if (z2 > 4.0) ++n_gt4;
    if (z2 > 9.0) ++n_gt9;
    const int k = (r.row_kind >= 0 && r.row_kind <= 6) ? r.row_kind : 0;
    if (k == 0) { ++n_plane; sum_plane += z2; if (z2 > 9.0) ++n_plane_gt9; }
    else        { ++n_line;  sum_line  += z2; if (z2 > 9.0) ++n_line_gt9; }
    const double sg = r.sigma_gate_squared;
    if (std::isfinite(sg) && sg > 0.0) {
      const double zg = r2 / sg;
      ++n_gate;
      sum_gate += zg;
      if (zg > 9.0) ++n_gate_gt9;
    }
  }
  double p50 = 0.0, p90 = 0.0, zmax = 0.0;
  if (!z.empty()) {
    std::sort(z.begin(), z.end());
    const std::size_t last = z.size() - 1;
    p50 = z[std::min(last, static_cast<std::size_t>(0.5 * static_cast<double>(last) + 0.5))];
    p90 = z[std::min(last, static_cast<std::size_t>(0.9 * static_cast<double>(last) + 0.5))];
    zmax = z[last];
  }
  o << scan_id << ',' << iteration << ',' << residuals.size() << ',' << n_invalid << ',' << sum_r2 << ',' << sum_z2 << ',' << p50 << ',' << p90 << ',' << zmax << ',' << n_gt4 << ',' << n_gt9
    << ',' << n_plane << ',' << sum_plane << ',' << n_plane_gt9 << ',' << n_line << ',' << sum_line << ',' << n_line_gt9 << ',' << n_gate << ',' << sum_gate << ',' << n_gate_gt9 << '\n';
}

}  // namespace zsq_summary
}  // namespace livo_recon
