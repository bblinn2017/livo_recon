#pragma once

#include <Eigen/Dense>

namespace livo_recon_test
{

// Propagates the [position; velocity] covariance of a continuous-time model
// driven by white acceleration noise with isotropic spectral density q.
Eigen::Matrix<double, 6, 6> propagatePVWhiteAcceleration(
    const Eigen::Matrix<double, 6, 6>& P0, double dt, double q);

// Returns the exact continuous-time white-acceleration process covariance
// over an interval dt for the [position; velocity] state.
Eigen::Matrix<double, 6, 6> whiteAccelerationProcessCovariance(double dt, double q);

}  // namespace livo_recon_test
