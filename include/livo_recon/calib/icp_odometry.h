// Pure point-to-plane Gauss-Newton registration, with NO Kalman gain and
// NO process-noise-dependent prior covariance anywhere in the solve.
//
// Why this exists (see docs/QSCALE_REFINE_RESTRUCTURE_PLAN.md): Phase 1 of
// the Q-scale calibration restructure originally ran livo_recon's own
// LioProcIekf for its seed trajectory. That is an iterated EKF -- its
// correction step computes a Kalman gain from the propagated prior
// covariance, which depends on imu/process_noise/motion/{acc,gyro}_scale,
// the very constants this whole pipeline exists to calibrate. Even though
// the IMU's MEAN propagation (used for deskewing and the inter-scan pose
// guess) does not depend on those scales -- only their COVARIANCE does --
// routing the seed trajectory through an EKF correction step still makes
// it soft-dependent on an unvalidated prior, undermining the calibration
// (the user's objection, independently raised twice this investigation:
// "I think it's kind of nonsensical to use the EKF model here because
// we're directly trying to predict the EKF model constants").
//
// icpRegister() replaces that correction step with plain weighted
// least-squares point-to-plane minimization (iterated, like the EKF's own
// inner loop, but never blended with any prior/covariance term) -- the
// zero-information limit of StateGroup's own EkfUpdate::applyMeanUpdate()
// (see that function's doc comment: as prior_cov -> infinity, its
// solution -> -(HtH)^-1 * Htz exactly, which is what this computes
// directly, skipping the state/prior-covariance machinery entirely since
// there is no prior to blend with).
#pragma once

#include "livo_recon/common_lib.h"
#include "livo_recon/map/map_backend.h"
#include "livo_recon/utils/data/data_wrappers.h"

#include <vector>

namespace livo_recon
{
namespace calib
{

struct IcpOptions
{
  int max_iterations = 5;
  double convergence_rot = 1e-5;   // rad, per-iteration step norm
  double convergence_pos = 1e-5;   // m, per-iteration step norm
  double damping = 1e-9;           // tiny LM-style diagonal regularizer, numerical stability only
  int min_matches = 10;            // bail out (leave R, p unchanged from this iteration) below this
};

struct IcpReport
{
  int iterations = 0;
  int n_matched = 0;
  bool converged = false;
  bool degenerate = false;   // true if min_matches was never reached
};

// Registers `points` (deskewed, scan-end body frame -- PointXYZCov::point/
// sensor_cov, exactly what LioProcBase::deskewAndDownsample() produces)
// against `map` (already built, read-only -- NEVER updated here) via
// iterated point-to-plane Gauss-Newton, refining (R, p) IN PLACE from
// whatever initial guess the caller seeded them with (typically the IMU's
// own mean-propagated pose for this scan, which is safe to use -- see this
// file's header comment).
IcpReport icpRegister(const std::vector<PointXYZCov>& points, MapBackend& map, M3D& R, V3D& p,
                       const IcpOptions& opt = IcpOptions{});

}  // namespace calib
}  // namespace livo_recon
