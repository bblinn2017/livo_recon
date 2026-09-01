#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>

#include "livo_recon/utils/algo/math.h"
#include "livo_recon/utils/data/data_wrappers.h"

// ============================================================================
// ScanSpline -- a per-scan cubic B-spline trajectory, fitted to the IMU-
// propagated pose sequence, re-anchored to the IEKF's corrected state on every
// inner iteration, and used for (a) per-iteration deskewing and (b) a direct
// residual of spline-derived acceleration / angular velocity against the RAW
// IMU stream.  (b) is the point: that residual is a direct measurement of the
// IMU's own noise, which is the same physical quantity cov_w's accelerometer
// and gyro blocks are built from.  See AdaptiveQ (lio/adaptive_q.h).
//
// HISTORY / SCOPE.  A different iterative-deskew mechanism (a cubic HERMITE
// spline over the whole scan, with an angular-velocity state idxW() and a
// 12-dim "wide" EkfUpdate) was removed on 2026-08-24 with an unresolved
// regression on 3 HILTI sequences; its design doc and code snapshot are cited
// in deskew.h/state.h/ekf.h but are NOT present in this repository.  This is
// deliberately NOT that mechanism:
//   - B-spline, not Hermite: control points are FITTED to the whole pose
//     sequence by least squares, not interpolated through endpoint tangents,
//     so a noisy endpoint tangent cannot swing the interior of the scan.
//   - No new filter states.  EkfUpdate stays 6x6 (R,P).  The spline is
//     re-anchored RIGIDLY to the corrected state each iteration (anchorTo()),
//     so the IEKF correction propagates into the whole scan without the state
//     vector growing and without touching the estimator's tuned structure.
//   - The IMU enters as a RESIDUAL against the spline's analytic derivatives,
//     not as a propagation input.
// Everything here is gated by SplineOptions::enable, default false, so with
// the feature off the binary is bit-identical to the pre-change build.
//
// PARAMETERISATION.  Uniform cubic (order 4) B-spline over the scan window
// [t0, t1], n_cp control points, n_seg = n_cp - 3 segments, segment duration
// delta = (t1 - t0) / n_seg.  n_cp is the ablation knob: it controls how much
// the trajectory may bend between constraints, which is the same physical
// statement Q makes, and the residual in (b) is a monotone function of it.
//
//   position:  p(t)  = sum_{j=0..3} b_j(u) * cp_p_[s+j]           (world frame)
//   rotation:  R(t)  = R_anchor_ * Exp( phi(t) ),
//              phi(t) = sum_{j=0..3} b_j(u) * cp_phi_[s+j]
//
// Rotation is carried as a tangent-space spline about a fixed anchor rather
// than as a cumulative SO(3) spline.  Over ONE scan (~0.1 s) the total
// rotation is small -- 15 deg at 2.6 rad/s -- so the O(|phi|^3) truncation is
// far below the gyro noise we are trying to measure, and it buys an exactly
// LINEAR least-squares fit and exact analytic derivatives.  It would NOT be
// valid over a multi-second window; it is valid here because the anchor is
// re-chosen every scan.  splineRotationChordDeg() reports the actual angle so
// the assumption is monitored rather than assumed (see AdaptiveQ's log).
//
// DERIVATIVES.  With R(t) = R_a Exp(phi(t)),
//     d/dt Exp(phi) = Exp(phi) * [ Jr(phi) * phidot ]_x
// so the BODY-frame angular velocity is exactly
//     omega_body(t) = Jr(phi(t)) * phidot(t)
// which matches the convention ImuProc::propagate() uses (Exp(angvel, dt)
// right-multiplied onto rot_imu, angvel = gyro - bias_gyr).
// World-frame acceleration is pddot(t) directly, and the specific force the
// accelerometer should report is
//     a_pred(t) = R(t)^T ( pddot(t) - gravity ) + bias_acc
// which is the inverse of propagate()'s own
//     acc_world = rot * (acc - bias_acc) + gravity.
// ============================================================================

namespace livo_recon
{

struct SplineOptions
{
  // Master switch.  false (default) = every hook in this file and its
  // callers is skipped and behaviour is identical to the pre-change build.
  bool enable = false;

  // Number of control points for the scan window.  Minimum 4 (one cubic
  // segment).  THE ablation parameter -- see the file header.  8 over a
  // 100 ms scan is a 70 Hz effective control rate, in the same band
  // CT-VoxelMap hand-sets per dataset (35-80 Hz).
  int n_control_points = 8;

  // Alternative to n_control_points: if > 0, choose n_cp from a target
  // control-point rate in Hz over the actual scan duration, clamped to
  // [4, n_control_points_max].  0 (default) = use n_control_points as-is.
  double control_point_hz = 0.0;
  int    n_control_points_max = 32;

  // Tikhonov weight on the fit's second difference of control points.
  // Keeps A^T A conditioned when n_cp approaches the pose-sample count and
  // damps ringing at the scan ends.  Pure numerics -- it must stay small
  // enough not to bias the residual the Q estimate is read from, which is
  // what fit_reg_max_frac guards.
  double fit_regularization = 1e-6;

  // Refuse the fit (and fall back to the legacy one-shot deskew for this
  // frame) if the regularizer contributes more than this fraction of the
  // normal-matrix trace -- i.e. if the fit is being held up by the prior
  // rather than by the data.  Prevents a silently over-smoothed spline
  // feeding a silently under-estimated Q.
  double fit_reg_max_frac = 0.05;

  // Re-run the deskew against the spline on every IEKF inner iteration
  // (the whole point of the mechanism).  false = fit and deskew once per
  // frame, which isolates "spline vs piecewise-IMU deskew" from "iterated
  // vs one-shot deskew" as separate ablation arms.
  bool redeskew_each_iteration = true;

  // Refit the spline's POSITION control points against the current
  // iteration's point-to-plane residuals before re-deskewing.  This is
  // what makes a LiDAR point influence the control points in its own
  // temporal neighbourhood (cubic local support: each point touches
  // exactly 4) rather than only through the rigid 6-dof state correction.
  // Default OFF: it is the most invasive part of the mechanism and the
  // part most likely to reproduce the 2026-08-24 regression, so it ships
  // as its own arm rather than folded into the baseline.
  bool lidar_refine_cp = false;
  double lidar_refine_damping = 1e-2;   // Levenberg damping, relative to trace
  double lidar_refine_prior_w = 1.0;    // pull toward the IMU-only fit
  int    lidar_refine_iters = 1;

  // Per-scan CSV of the fit and the IMU residual (spline.csv in the debug
  // log dir).  Cheap: one line per frame.
  bool log_en = false;
};

// One scan's fitted trajectory.  Cheap to copy; sized by n_cp.
class ScanSpline
{
public:
  // Fit position and rotation control points to `poses` (ImuProc::
  // propagate()'s output, ascending in time) over [t0, t1].  `t1` should be
  // the frame reference time (mg.image.t) so the spline covers every point.
  // Returns false if there is not enough support to fit (fewer distinct
  // pose samples than control points, degenerate window, or the
  // regularizer exceeding fit_reg_max_frac) -- the caller must then fall
  // back to the legacy path rather than using a garbage spline.
  bool fit(const std::vector<Pose6D>& poses, double t0, double t1,
           const SplineOptions& opts);

  bool valid() const { return valid_; }
  int  nControlPoints() const { return n_cp_; }
  double t0() const { return t0_; }
  double t1() const { return t1_; }

  // Evaluation.  `t` is clamped to [t0_, t1_] -- points at the exact scan
  // boundary are common and must not fall off the end of the knot vector.
  M3D rotAt(double t) const;
  V3D posAt(double t) const;
  V3D velAt(double t) const;                 // world frame, d/dt p
  V3D accAt(double t) const;                 // world frame, d2/dt2 p
  V3D omegaBodyAt(double t) const;           // body frame, Jr(phi) * phidot

  // Rigidly transform the whole spline so that its pose at `t_ref` equals
  // (R_ref, p_ref).  This is how the IEKF's 6-dof correction reaches every
  // control point between inner iterations, without the control points
  // entering the filter state.  Cheap: one rotation of n_cp vectors.
  void anchorTo(double t_ref, const M3D& R_ref, const V3D& p_ref);

  // Total rotation swept across the scan, in degrees -- the health metric
  // for the small-angle tangent parameterisation documented in the header.
  double rotationChordDeg() const;

  // Basis weights and their first/second time-derivatives at `t`, together
  // with the index of the first of the 4 control points they apply to.
  // Exposed because the LiDAR control-point refinement needs the same
  // local support the evaluator uses.
  void basisAt(double t, int& first_cp, Eigen::Vector4d& b,
               Eigen::Vector4d& db, Eigen::Vector4d& ddb) const;

  // Position control points, for the LiDAR refinement step.
  Eigen::Matrix<double, 3, Eigen::Dynamic>& cpPosMut() { return cp_p_; }
  const Eigen::Matrix<double, 3, Eigen::Dynamic>& cpPos() const { return cp_p_; }

  // Diagnostics from the last fit.
  double fitResidualPos() const { return fit_res_pos_; }   // RMS, m
  double fitResidualRot() const { return fit_res_rot_; }   // RMS, rad
  double fitRegFrac()     const { return fit_reg_frac_; }

private:
  bool   valid_ = false;
  int    n_cp_  = 0;
  int    n_seg_ = 0;
  double t0_ = 0.0, t1_ = 0.0, delta_ = 0.0, inv_delta_ = 0.0;

  M3D    R_anchor_ = M3D::Identity();
  Eigen::Matrix<double, 3, Eigen::Dynamic> cp_p_;     // 3 x n_cp, world
  Eigen::Matrix<double, 3, Eigen::Dynamic> cp_phi_;   // 3 x n_cp, tangent

  double fit_res_pos_ = 0.0, fit_res_rot_ = 0.0, fit_reg_frac_ = 0.0;

  V3D phiAt(double t) const;
  V3D phiDotAt(double t) const;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// ---------------------------------------------------------------------------
// The IMU-vs-spline residual, accumulated over one scan.
//
// r_a(t_m) = R(t_m)^T ( pddot(t_m) - gravity ) + bias_acc - acc_m
// r_w(t_m) = omega_body(t_m) + bias_gyr - gyro_m
//
// `cov_*` are the isotropic (trace/3) empirical variances -- the same
// reduction calib_processing.cpp applies, so the numbers are directly
// comparable to the calibration floor and to state_->varAcc()/varGyr().
// `acf1_*` are the lag-1 autocorrelations of the residual sequence, the
// whiteness test that decides whether the control-point count is high
// enough for the residual to BE noise rather than unmodelled motion.
// ---------------------------------------------------------------------------
struct SplineImuResidualStats
{
  int    n = 0;
  double cov_acc = 0.0;    // (m/s^2)^2, isotropic
  double cov_gyr = 0.0;    // (rad/s)^2, isotropic
  double acf1_acc = 0.0;   // in [-1, 1]
  double acf1_gyr = 0.0;
  double max_abs_acc = 0.0;
  double max_abs_gyr = 0.0;
  bool   valid() const { return n >= 8; }
};

// `imu` must be the RAW samples for this scan, in ascending time, with
// timestamps on the same base as the spline's.  `bias_acc`/`bias_gyr`/
// `gravity` come from the current state.  Samples outside [t0, t1] are
// skipped rather than clamped -- extrapolating the spline would put
// unmodelled motion into the very statistic we are calling noise.
SplineImuResidualStats computeSplineImuResidual(
    const ScanSpline& spline,
    const std::vector<ImuSample>& imu,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity);

}  // namespace livo_recon
