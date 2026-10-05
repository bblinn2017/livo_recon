#pragma once
// R74 (F-430, F-431, F-432): residual rows of a voxel plane that may really be a LINE (a single LiDAR ring on a wall: points along a curve,
// no constraint along it, a constraint in the plane perpendicular to it, and no way to tell "up" from "forward" in that plane).
// ROS-free pure functions so tests/test_line_residual.cpp can check them; the glue is in VoxelPlane::computeResidual() and
// LioProcBase::buildResiduals(). Everything here is OFF by default (mode Off returns 0 rows and the caller never enters this code).
//
// e0 <= e1 <= e2 are the eigenvalues of the voxel's point scatter (VoxelPlane::eigen_values_), u0, u1, u2 the unit eigenvectors
// (u0 = plane_.normal, u1 = y_normal_, u2 = x_normal_), d = p - plane centre (world frame), N = the number of points of the fit.
//
// Three runnable variants (Bryce 2026-10-01):
//   HardAbsolute  (variant 1): the voxel is a LINE when its second eigenvalue is also small in the absolute sense, e1 < plane_threshold
//                 (the same 2.5e-3 m^2 constant that already tests e0). A line gives TWO scalar rows (u0 and u1), each with its own
//                 residual u_i . d and its own fit variance e_i / N. A plane (e1 >= plane_threshold) gives 0 rows here: the caller keeps the
//                 existing single point-to-plane row, bit for bit.
//   HardRelative  (variant 2): the same two-row form, but the voxel is a LINE when e1 is NOT distinguishable from e0:
//                 e1 <= (1 + k sqrt(2 / (N - 1))) e0, with k = rel_sigmas (3 as accepted). N < 2 or e0 <= 0: not a line.
//   HardRelativeNoise (R75 variant 4, Bryce 2026-10-02): the HardRelative test with e0 floored at the sensor noise variance:
//                 e1 <= (1 + k sqrt(2 / (N - 1))) max(e0, sigma^2), sigma^2 = LineOptions::sensor_noise_var (= imu/sensor/range_err squared, the constant Smooth uses),
//                 k = rel_sigmas (3). Reason: a degenerate voxel has e0 near 0, which made HardRelative's bound near 0 (it labelled about 2 points of 10,000 as a line in R74).
//                 N < 2 is not a line; sigma^2 <= 0 reduces it to HardRelative. Same two rows as the other hard variants.
//   HardShape     (R75 variant 5, Bryce 2026-10-02): the voxel is a LINE when its two smaller eigenvalues are small against the largest: e1 <= tau e2, tau = shape_ratio (0.1).
//                 Scale free; ignores N and sensor noise; e2 <= 0 is not a line. Same two rows as the other hard variants.
//   Smooth        (variant 3, F-431): no class switch. EVERY matched point gets THREE scalar rows, one per eigen-direction (the point-to-
//                 distribution / Mahalanobis residual with a diagonal covariance in the eigenbasis). The variance of the fit part of row i is
//                 max(e_i, sensor_noise_var) (1 + 1/N): the predictive variance of a new point about the voxel distribution (scatter plus the
//                 variance of the fitted centre), each eigenvalue floored at the sensor noise variance.
// Approximation (all variants): the rows of one point share that point's noise; their cross-covariance is ignored (diagonal weights).
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <string>

#include "livo_recon/lio/plane_test_values.h"

namespace livo_recon
{
enum class LineMode { Off = 0, HardAbsolute = 1, HardRelative = 2, Smooth = 3, HardRelativeNoise = 4, HardShape = 5 };

struct LineOptions
{
  LineMode mode = LineMode::Off;
  double plane_threshold = 2.5e-3;   // m^2; HardAbsolute tests e1 against it
  double rel_sigmas = 3.0;           // HardRelative: the "few" of 1 + k sqrt(2/(N-1))
  double sensor_noise_var = 0.0;     // m^2; Smooth floors every eigenvalue at it; HardRelativeNoise floors e0 at it
  double shape_ratio = 0.1;          // HardShape: line when e1 <= shape_ratio * e2
};

struct LineRow
{
  Eigen::Vector3d normal = Eigen::Vector3d::Zero();   // the row's direction u_i (the "plane normal" of this row)
  double r = 0.0;                                      // u_i . d
  double fit_var = 0.0;                                // the fit part of the row variance (m^2); the point's own noise is added by the caller
};

// Row kinds written in the residual dump (Residual::row_kind). 0 is the ordinary single point-to-plane row.
// R80 (F-453, Bryce 2026-10-05): kRowLineDist = 6, the single distance row of a hard line voxel (rows = distance): the shortest line-to-point distance, along the direction from the line to the point.
enum LineRowKind { kRowPlane = 0, kRowLineU0 = 1, kRowLineU1 = 2, kRowSmoothU0 = 3, kRowSmoothU1 = 4, kRowSmoothU2 = 5, kRowLineDist = 6 };

// e2 is only read by HardShape (default -1: HardShape then says not a line); the other variants ignore it, so every R74 call site keeps its meaning.
inline bool isLineVoxel(const LineOptions& o, double e0, double e1, int n_points, double e2 = -1.0)
{
  switch (o.mode)
  {
    case LineMode::HardAbsolute: return e1 < o.plane_threshold;
    case LineMode::HardRelative:
    {
      const double thr = relativeGapThreshold(n_points, o.rel_sigmas);
      if (!std::isfinite(thr) || !(e0 > 0.0)) return false;
      return e1 <= thr * e0;
    }
    case LineMode::HardRelativeNoise:
    {
      const double thr = relativeGapThreshold(n_points, o.rel_sigmas);
      if (!std::isfinite(thr)) return false;
      const double floor_e0 = std::max(e0, o.sensor_noise_var);
      if (!(floor_e0 > 0.0)) return false;
      return e1 <= thr * floor_e0;
    }
    case LineMode::HardShape: return (e2 > 0.0) && e1 <= o.shape_ratio * e2;
    default: return false;
  }
}

// Fills out[0..n-1] and returns n: 0 = use the ordinary single point-to-plane row; 2 = line rows (u0, u1); 3 = smooth rows (u0, u1, u2).
inline int lineRows(const LineOptions& o, const Eigen::Vector3d& u0, const Eigen::Vector3d& u1, const Eigen::Vector3d& u2,
                    const Eigen::Vector3d& eig, const Eigen::Vector3d& d, int n_points, LineRow out[3])
{
  const double n = static_cast<double>(std::max(n_points, 1));
  const Eigen::Vector3d* u[3] = {&u0, &u1, &u2};
  if (o.mode == LineMode::Smooth)
  {
    for (int i = 0; i < 3; ++i)
    {
      out[i].normal = *u[i];
      out[i].r = u[i]->dot(d);
      out[i].fit_var = std::max(eig(i), o.sensor_noise_var) * (1.0 + 1.0 / n);
    }
    return 3;
  }
  if ((o.mode == LineMode::HardAbsolute || o.mode == LineMode::HardRelative || o.mode == LineMode::HardRelativeNoise || o.mode == LineMode::HardShape) &&
      isLineVoxel(o, eig(0), eig(1), n_points, eig(2)))
  {
    for (int i = 0; i < 2; ++i)
    {
      out[i].normal = *u[i];
      out[i].r = u[i]->dot(d);
      out[i].fit_var = std::max(eig(i), 0.0) / n;
    }
    return 2;
  }
  return 0;
}

inline bool parseLineMode(const std::string& mode, const std::string& threshold, LineMode& out)
{
  if (mode != "hard" && (threshold == "relative_noise" || threshold == "shape")) { return false; }
  if (mode == "off") { out = LineMode::Off; return true; }
  if (mode == "smooth") { out = LineMode::Smooth; return true; }
  if (mode == "hard" && threshold == "absolute") { out = LineMode::HardAbsolute; return true; }
  if (mode == "hard" && threshold == "relative") { out = LineMode::HardRelative; return true; }
  if (mode == "hard" && threshold == "relative_noise") { out = LineMode::HardRelativeNoise; return true; }
  if (mode == "hard" && threshold == "shape") { out = LineMode::HardShape; return true; }
  return false;
}

// R76 (F-445, Bryce 2026-10-02): the configuration is a flag plus mutually exclusive choices.
//   enabled (bool, default false)  form (hard | smooth, default hard)  rule (absolute | relative | relative_noise | shape, default absolute; only form hard).
// Returns true and sets out when the combination is legal; otherwise false and why says what to change. enabled == false -> Off (rule and form are not looked at:
// the caller has already refused a key that is set while the flag is off). Pure so tests/test_line_residual_r76.cpp can check it.
inline bool resolveLineConfig(bool enabled, const std::string& form, const std::string& rule, LineMode& out, std::string* why = nullptr)
{
  auto fail = [&](const std::string& m) { if (why) *why = m; return false; };
  if (!enabled) { out = LineMode::Off; return true; }
  if (form == "smooth")
  {
    if (!rule.empty() && rule != "absolute") return fail("rule is read only for form hard; remove it or set form hard");
    out = LineMode::Smooth; return true;
  }
  if (form != "hard") return fail("form must be hard or smooth");
  if (rule == "absolute") { out = LineMode::HardAbsolute; return true; }
  if (rule == "relative") { out = LineMode::HardRelative; return true; }
  if (rule == "relative_noise") { out = LineMode::HardRelativeNoise; return true; }
  if (rule == "shape") { out = LineMode::HardShape; return true; }
  return fail("rule must be absolute, relative, relative_noise or shape");
}
}  // namespace livo_recon
