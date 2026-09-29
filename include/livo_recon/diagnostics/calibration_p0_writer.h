#pragma once

#include <deque>
#include <string>

#include "livo_recon/utils/data/data_wrappers.h"
#include "livo_recon/utils/state/state.h"

namespace livo_recon
{

// One machine-readable text artifact per run, written after stationary
// calibration and before the first propagation. This is the actual P0 used by
// the estimator, not a reconstruction from configuration values.
void writeCalibrationP0Diagnostic(
    const StateGroup& state, const std::string& mode, int sample_count,
    int autocov_lags, const M3D& acc_mean_cov, const M3D& gyro_mean_cov);

// R62: the raw quantities behind gravity alignment, so tilt P0 can be tested
// against the accelerometer itself. One small text file per run
// (gravity_alignment.txt), written right after setCalibResult(): raw mean and
// sample/mean covariance of the calibration accelerometer and gyro, the mean
// over each half of the window (stationarity check), |mean accel| against the
// state's gravity magnitude, the rotation before and after alignment, the
// bias stripped of gravity, and what the state actually received.
void writeGravityAlignmentDiagnostic(
    const std::deque<ImuSample>& samples, const M3D& acc_mean_cov,
    const M3D& gyro_mean_cov, const M3D& R_before, const M3D& R_init,
    const StateGroup& state_after, const V3D& true_acc_bias,
    bool apply_accel_bias, bool apply_gyro_bias);

}  // namespace livo_recon
