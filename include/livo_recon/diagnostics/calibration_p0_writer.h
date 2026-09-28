#pragma once

#include <string>

#include "livo_recon/utils/state/state.h"

namespace livo_recon
{

// One machine-readable text artifact per run, written after stationary
// calibration and before the first propagation. This is the actual P0 used by
// the estimator, not a reconstruction from configuration values.
void writeCalibrationP0Diagnostic(
    const StateGroup& state, const std::string& mode, int sample_count,
    int autocov_lags, const M3D& acc_mean_cov, const M3D& gyro_mean_cov);

}  // namespace livo_recon
