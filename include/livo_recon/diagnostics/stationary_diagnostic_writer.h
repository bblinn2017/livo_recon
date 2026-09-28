#pragma once

#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/map/voxelmap_utils.h"
#include "livo_recon/utils/state/state.h"

#include <string>
#include <vector>

namespace livo_recon
{

void writeStationaryIterationDiagnostics(
    const std::string& test_id, const std::string& architecture,
    int scan_id, int iteration, double t_abs,
    const M3D& reference_R, const V3D& reference_p, const V3D& reference_v,
    const StateGroup& before, const StateGroup& after,
    const std::vector<Residual>& residuals);

void writeGatingIterationDiagnostics(
    const std::string& test_id, const std::string& architecture,
    int scan_id, int iteration, double t_abs, bool gating_state_uncertainty,
    int input_points, int statistical_candidates,
    int statistical_gate_rejections, int coverage_misses, int mismatch_misses,
    double mean_gating_cov_trace, double max_gating_cov_trace,
    const std::vector<Residual>& residuals);

}  // namespace livo_recon
