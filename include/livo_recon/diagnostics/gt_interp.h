#pragma once

#include <vector>
#include "livo_recon/utils/algo/math.h"

// ============================================================================
// Ground-truth-to-query-timestamp linear interpolation.
//
// NTU-VIRAL ground truth (Leica) is POSITION-ONLY and DISCRETE. This utility
// interpolates a dense GT position stream onto an arbitrary query timestamp
// (e.g. a pose-control scan's own converged time) -- the OPPOSITE direction
// from EvoProc's own scoring convention (evo_processing.h interpolates the
// ESTIMATE onto GT's timestamps, matching the official NTU-VIRAL evaluator).
// This utility exists for covariance-calibration analysis, which needs GT
// position at the ESTIMATOR's own timestamps, not the reverse.
//
// Never extrapolates: a query strictly before the first or after the last
// GT sample, or one whose bracketing GT samples are farther apart than
// max_gap, is reported unavailable rather than interpolated.
// ============================================================================

namespace livo_recon
{

struct GtSample3
{
  double t = 0.0;
  V3D pos = V3D::Zero();
};

struct GtInterpResult
{
  bool available = false;
  V3D pos = V3D::Zero();
  double t_lo = 0.0, t_hi = 0.0;
  double fraction = 0.0;   // 0 at t_lo, 1 at t_hi
  double gap = 0.0;        // t_hi - t_lo
};

// `samples` MUST be sorted ascending by t. max_gap bounds how far apart the
// two bracketing samples may be before the interpolation is refused (a large
// gap means the linear interpolation would span too much real, unobserved
// motion to be trustworthy).
GtInterpResult interpolateGtToQueryTime(const std::vector<GtSample3>& samples, double t_query, double max_gap);

}  // namespace livo_recon
