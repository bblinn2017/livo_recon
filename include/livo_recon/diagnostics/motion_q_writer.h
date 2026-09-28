#pragma once

#include <cstddef>
#include <string>

#include "livo_recon/utils/algo/math.h"

namespace livo_recon
{

// Shared compact diagnostic interface for every estimator mode using IMU
// propagation. Keeping file ownership here prevents propagation math from
// accumulating mode-specific CSV formatting.
void writeMotionQSampleDiagnostic(
    size_t scan_index, size_t sample_index, double t_abs,
    const std::string& model, const V3D& dynamic_acc, const V3D& dynamic_gyr,
    const V3D& debiased_acc_energy, const V3D& debiased_gyr_energy,
    const V3D& q_acc, const V3D& q_gyr);

void writeMotionQScanDiagnostic(
    size_t scan_index, double t_abs, const std::string& model, double beta,
    double acc_scale, double gyr_scale,
    size_t sample_count, double mean_dynamic_acc_norm,
    double mean_dynamic_gyr_norm, const V3D& mean_q_acc,
    const V3D& mean_q_gyr, const V3D& mean_debiased_acc_energy,
    const V3D& mean_debiased_gyr_energy, size_t acc_clamped_axes,
    size_t gyr_clamped_axes);

}  // namespace livo_recon
