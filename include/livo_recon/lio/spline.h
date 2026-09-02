#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>

#include "livo_recon/utils/algo/math.h"
#include "livo_recon/utils/data/data_wrappers.h"

// ============================================================================
// ScanSpline -- a per-scan cubic B-spline trajectory, fitted to the IMU-
// propagated pose sequence, refined by the scan's own LiDAR returns,
// re-anchored to the IEKF's corrected state on every inner iteration, and used
// for (a) per-iteration deskewing and (b) a direct residual of spline-derived
// acceleration / angular velocity against the RAW IMU stream.  (b) is the
// point: that residual is a direct measurement of the IMU's own noise, which
// is the same physical quantity cov_w's accelerometer and gyro blocks are
// built from.  See AdaptiveQ (lio/adaptive_q.h).
//
// HISTORY / SCOPE.  A different iterative-deskew mechanism (a cubic HERMITE
// spline over the whole scan, with an angular-velocity state idxW() and a
// 12-dim "wide" EkfUpdate) was removed on 2026-08-24 with an unresolved
// regression on 3 HILTI sequences; it has since been established that that
// spline was very low resolution.  This is deliberately NOT that mechanism:
//   - B-spline, not Hermite: control points are FITTED to the whole pose
//     sequence by least squares, not interpolated through endpoint tangents.
//   - No new filter states.  EkfUpdate stays 6x6 (R,P).
//   - The IMU enters as a RESIDUAL against the spline's analytic derivatives.
// Everything here is gated by SplineOptions::enable, default false.
//
// ---------------------------------------------------------------------------
// ROTATION.  Two parameterisations, selectable, and the choice between them is
// MEASURED rather than argued -- see scripts/test/test_spline_rotation.cpp.
//
// "tangent" (DEFAULT) -- a cubic spline of phi(t) in ONE tangent chart about a
// mid-scan anchor:
//     R(t) = R_anchor * Exp(phi(t)),   phi(t) = sum_j b_j(u) cp_phi_[s+j]
//     omega_body = Jr(phi) * phidot                                  (exact)
// The omega formula is exact for this parameterisation.  The parameterisation
// itself is only an approximation of a cubic on SO(3): splining linearly in a
// single chart is EXACT when the rotation axis is fixed and carries an
// O(|phi|^3) BCH error when the axis moves.  Its fit is an exact linear least
// squares.
//
// "cumulative" -- the standard cumulative B-spline on SO(3), the form every
// published CT-LIO system uses (Sommer et al. 2020, and after them CLINS,
// Coco-LIC, CT-VoxelMap, SLICT2, RESPLE):
//     R(t) = R_s * A_1 * A_2 * A_3,   A_j = Exp( Btilde_j(u) * d_j )
//     d_j  = Log( R_{s+j-1}^T R_{s+j} ),  Btilde_j(u) = sum_{k>=j} b_k(u)
// Each A_j has a FIXED axis (only the scalar Btilde_j varies with t), so
// Adot_j = A_j [Btildedot_j d_j]_x with no right-Jacobian term, and pushing
// all three through R^T Rdot gives the exact body rate
//     omega_body = (A_2 A_3)^T w_1 + A_3^T w_2 + w_3,
//     w_j = (dBtilde_j/du) * (1/delta) * d_j.
// Its fit is Gauss-Newton with the EXACT Jacobian of that product (Sommer's
// recurrence; see fitRotationCumulative()), initialised from the tangent fit.
//
// WHY CUMULATIVE IS THE DEFAULT, as of 2026-09-02 -- and this reverses the
// previous default along with the reasoning that set it.
//
// The earlier default was "tangent", on a moving-axis synthetic fixture where
// tangent came out ahead at large rotation.  That comparison was rigged, in a
// way that took a second look to see: tangent is an EXACT linear least-squares
// solve, while fitRotationCumulative() was refining by Gauss-Newton using the
// ORDINARY basis b_j(u)*I as its Jacobian -- a first-order stand-in for the
// true derivative of the cumulative product.  So a correct implementation of
// the non-standard choice was being compared against a sloppy implementation
// of the standard one, and the non-standard one was kept because it won.
//
// Two further facts, both from the same re-examination:
//   - At the rotation a real scan contains they are EQUAL.  SP-4a measured max
//     rot_chord_deg ~6 deg on eee_01; at 8.5 deg and n_cp=8 the two agree to
//     three digits.  The gap only opens at rotations these bags do not reach.
//   - The claim that cumulative was "fully converged -- rot_fit_iters 2, 4, 8,
//     16 and 32 agree to five digits" was established at n_cp=16.  At the
//     shipped n_cp=8 it fails above ~34 deg: most of the apparent tangent
//     advantage at 128 and 166 deg was under-convergence of our own solver.
//
// So the default is now the form every published CT-LIO system uses, the
// Jacobian is exact, and the burden of proof sits where it belongs: on the
// departure.  "tangent" is retained as the ablation arm, and whether it is
// worth keeping is now an empirical question on real sequences (queue item
// SP-R), not a synthetic one.  No synthetic number appears in this decision
// any more except as the reason to stop trusting synthetic numbers here.
// ---------------------------------------------------------------------------
//
// PARAMETERISATION.  Uniform cubic (order 4) B-spline over the scan window
// [t0, t1], n_cp control points, n_seg = n_cp - 3 segments, segment duration
// delta = (t1 - t0) / n_seg.  n_cp is the resolution knob: it controls how
// much the trajectory may bend between constraints, which is the same physical
// statement Q makes.
//
//   position:  p(t) = sum_{j=0..3} b_j(u) * cp_p_[s+j]            (world frame)
//
// World-frame acceleration is pddot(t) directly, and the specific force the
// accelerometer should report is
//     a_pred(t) = R(t)^T ( pddot(t) - gravity ) + bias_acc
// which is the inverse of ImuProc::propagate()'s own
//     acc_world = rot * (acc - bias_acc) + gravity.
// ============================================================================

namespace livo_recon
{

struct SplineOptions
{
  // Master switch.  false (default) = every hook in this file and its
  // callers is skipped and behaviour is identical to the pre-change build.
  bool enable = false;

  // "cumulative" (default) | "tangent" -- see the ROTATION block above.  The
  // default is the literature's form; "tangent" is the ablation arm.  Any
  // value other than "cumulative" is treated as "tangent".
  std::string rot_mode = "cumulative";

  // Number of control points for the scan window.  Minimum 4 (one cubic
  // segment).  8 over a 100 ms scan is a ~70 Hz effective control rate, in
  // the same band CT-VoxelMap hand-sets per dataset (35-80 Hz).
  int n_control_points = 8;

  // Alternative to n_control_points: choose n_cp from a target control-point
  // rate in Hz over the ACTUAL scan duration, clamped to
  // [4, n_control_points_max].  0 (default) = use n_control_points as-is.
  //
  // Prefer this whenever results are to be compared ACROSS sequences: a
  // fixed n_cp is a different control rate on every sequence with a
  // different scan duration, so an n_cp chosen on one bag does not transfer
  // to another.  control_point_hz is the physically comparable axis.
  double control_point_hz = 0.0;
  int    n_control_points_max = 32;

  // Tikhonov weight on the second difference of control points.  Keeps A^T A
  // conditioned when n_cp approaches the pose-sample count and damps ringing
  // at the scan ends.  Must stay small enough not to bias the residual the Q
  // estimate is read from, which is what fit_reg_max_frac guards.
  double fit_regularization = 1e-6;
  double fit_reg_max_frac = 0.05;

  // Gauss-Newton refinements of the control ROTATIONS after the tangent-space
  // initialisation, in "cumulative" mode only.  0 means "initialise from the
  // tangent fit and stop", which is NOT a usable spline -- the Greville
  // re-encoding is lossy, so at least one refinement is required.
  //
  // Raised from 2 to 4 when the exact Jacobian landed.  The old default was
  // set against the first-order Jacobian and was measured to be insufficient
  // at n_cp=8 above ~34 deg of chord; an exact Jacobian should converge much
  // faster, but that has not been measured on real data yet, and each
  // iteration is one LDLT of a banded 3*n_cp system (24x24 at n_cp=8) per
  // frame -- far cheaper than being wrong.  Lower it once SP-R has a number.
  int rot_fit_iters = 4;

  // Re-run the deskew against the spline on every IEKF inner iteration, and
  // once more after the loop converges so the map is built from points
  // placed by the FINAL state rather than the second-to-last one.
  bool redeskew_each_iteration = true;

  // ── LiDAR refinement of the control points ──────────────────────────────
  // Let each LiDAR return, at its own timestamp, pull the POSITION control
  // points in its own temporal neighbourhood.  Cubic local support means one
  // point touches exactly 4 of them, and coupling propagates through
  // overlapping supports -- which is the difference between "the whole scan
  // moves rigidly with the 6-dof state correction" and "the intra-scan SHAPE
  // is informed by the map".
  //
  // The residual is LINEAR in the position control points:
  //     r_i = n^T ( R(t_i) p_i^L + p(t_i) - q ),   p(t) = sum_j b_j(u) cp[s+j]
  //     dr_i/dcp[s+j] = b_j(u_i) * n^T
  // so one weighted linear solve is the exact Gauss-Newton step at fixed
  // associations.  The system is 3*n_cp square (24x24 at n_cp=8) and banded.
  //
  // DIVISION OF LABOUR, and it is deliberate: the ESIKF owns the scan-END
  // pose; this refinement owns only the intra-scan SHAPE.  The caller
  // re-anchors immediately afterwards (anchorTo), which restores the endpoint
  // exactly, so the two estimators never fight over the same quantity.
  // Rotation control points are NOT refined here -- they stay with the
  // IMU + ESIKF.
  bool   lidar_refine_cp = false;
  double lidar_refine_damping = 1e-2;    // Levenberg, relative to trace(H)/n
  double lidar_refine_prior_w = 1.0;     // pull toward the pre-refinement fit
  int    lidar_refine_iters = 1;
  // Reject the WHOLE step if any control point would move further than this
  // (metres).  A system documented to move ATE 18.88% on a semantically inert
  // 0.1% nudge does not get an unbounded shape correction; failing safe to
  // the unrefined spline is cheap and the rejection is counted, not hidden.
  double lidar_refine_max_step = 0.10;

  // ── Raw-IMU term in the FIT itself ──────────────────────────────────────
  // Until now the acc/gyro residual was diagnostic only: the fit was pure
  // least squares to the propagated POSES, and the residual was measured
  // afterwards for AdaptiveQ.  These weights add the residual to the fit.
  //
  // Both are LINEAR in their control points at fixed rotation:
  //     accel:  pddot(t) = sum_j ddb_j(u) cp_p[s+j] / delta^2
  //             target   = R(t) (acc_raw - b_a) + g
  //     gyro :  phidot(t) = sum_j db_j(u) cp_phi[s+j] / delta
  //             target   = Jr(phi(t))^-1 (gyro_raw - b_g)
  // so each contributes ordinary normal-equation rows.  The gyro target
  // depends on phi, so it is relinearised imu_fit_iters times.
  //
  // WEIGHTS ARE DIMENSIONLESS AND TRACE-NORMALISED, the same idiom
  // fit_regularization uses: the IMU block is scaled so that its normal-matrix
  // trace equals w times the POSE block's trace.  w = 1 therefore means "the
  // IMU term carries as much total weight as the whole pose sequence",
  // independently of sample counts, n_cp and units.
  //
  // DEFAULT 0 (off), and that default is a judgement, not laziness.  Fitting
  // to positions penalises the DOUBLE INTEGRAL of accel error, which is
  // heavily low-pass -- the fit is nearly blind to the high-frequency accel
  // noise that AdaptiveQ's sigma_a is trying to measure, which is exactly why
  // sigma_a comes out ~15% low instead of collapsing.  Weighting the accel
  // residual into the fit erodes that separation: the spline starts absorbing
  // the very quantity it is about to be asked to report.  The DOF ratio
  // bounds how far this can go (n_cp ~ 8 against ~21 samples cannot
  // interpolate the IMU), but it does not remove the bias -- it caps it.
  // Raise these deliberately, and read sigma_a_hat against a known truth when
  // you do.
  double imu_fit_w_acc = 0.0;
  double imu_fit_w_gyr = 0.0;
  int    imu_fit_iters = 2;

  // ── Re-integration under the ESIKF's bias corrections ───────────────────
  // The pose sequence the spline is fitted to was dead-reckoned by
  // ImuProc::propagate() using the biases as they stood BEFORE this frame's
  // update, and propagate() is never re-run inside the IEKF loop.  So without
  // this the spline's SHAPE is frozen at the pre-update bias while every
  // inner iteration moves that bias.  anchorTo() cannot repair it: a bias
  // delta produces a shape change (rotation drifting linearly in t, position
  // quadratically), and anchorTo is a rigid 6-dof transform.
  //
  // With this on, reintegratePoses() replays the stored pose recurrence under
  // the CURRENT bias and gravity before each re-fit.  It needs no raw IMU
  // samples: Pose6D already carries the per-step world accelerations and the
  // bias-corrected mean body rate, from which the body-frame measurement is
  // recoverable exactly.  Zero deltas short-circuit to an exact no-op.
  bool   reintegrate_each_iteration = false;
  // Skip the replay when the correction is smaller than this (rad/s and
  // m/s^2 respectively).  The point is not to save the ~20 integration steps
  // -- it is that a re-fit moves the spline out from under the LiDAR
  // residuals the refinement was about to be linearised at, so it should
  // happen only when it buys something.
  double reintegrate_min_dbg = 1e-9;
  double reintegrate_min_dba = 1e-9;

  // Per-scan CSV of the fit and the IMU residual (spline_q.csv).
  bool log_en = false;
};

// Raw IMU for the fit's optional acc/gyro term.  Separate from the pose
// sequence because it is the UNAVERAGED, un-bias-corrected stream --
// ImuProc::keep_raw_samples preserves exactly that as mg.imu_samples_raw.
struct SplineImuFitData
{
  const std::vector<ImuSample>* samples = nullptr;
  V3D bias_acc = V3D::Zero();
  V3D bias_gyr = V3D::Zero();
  V3D gravity  = V3D::Zero();

  bool usable() const { return samples != nullptr && samples->size() >= 4; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// One LiDAR observation, reduced to what the control-point refinement needs.
// Built from the residuals the IEKF just accumulated, so the refinement is
// linearised exactly where those residuals were measured.
struct SplineLidarObs
{
  double t = 0.0;        // capture time, same base as the spline's
  V3D    normal = V3D::Zero();
  double r = 0.0;        // signed point-to-plane residual at the CURRENT fit
  double sigma2 = 1.0;   // that residual's variance

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// One scan's fitted trajectory.  Cheap to copy; sized by n_cp.
class ScanSpline
{
public:
  // `imu` is optional and only read when opts.imu_fit_w_acc/w_gyr > 0.
  bool fit(const std::vector<Pose6D>& poses, double t0, double t1,
           const SplineOptions& opts, const SplineImuFitData* imu = nullptr);

  // Per-FRAME reset of the refinement counters.  Deliberately NOT done inside
  // fit(): with spline.reintegrate_each_iteration on, fit() runs once per IEKF
  // iteration, and resetting there would make spline_q.csv's refine_applied /
  // refine_rejects report only the last iteration instead of the frame.
  void resetRefineStats() { refine_rejects_ = 0; refine_applied_ = 0; last_refine_step_ = 0.0; }

  bool valid() const { return valid_; }
  int  nControlPoints() const { return n_cp_; }
  double t0() const { return t0_; }
  double t1() const { return t1_; }
  bool cumulative() const { return cumulative_; }
  // n_cp as REQUESTED vs as actually used.  A cubic fit needs strictly more
  // pose samples than control points, so the fit silently shrinks n_cp to
  // n_samples-1.  At a 10 Hz scan and a 200 Hz IMU that ceiling is ~19,
  // which means every requested n_cp above it collapses onto the same fit --
  // a sweep that asks for 20, 24 and 32 gets three identical cells.  Both
  // numbers go in spline_q.csv so that is visible instead of silent.
  int  nControlPointsRequested() const { return n_cp_req_; }
  bool nControlPointsClamped() const { return n_cp_req_ != n_cp_; }

  // Evaluation.  `t` is clamped to [t0_, t1_].
  M3D rotAt(double t) const;
  V3D posAt(double t) const;
  V3D velAt(double t) const;                 // world frame, d/dt p
  V3D accAt(double t) const;                 // world frame, d2/dt2 p
  V3D omegaBodyAt(double t) const;           // body frame

  // Rigidly transform the whole spline so its pose at `t_ref` equals
  // (R_ref, p_ref).  How the IEKF's 6-dof correction reaches every control
  // point, and how the endpoint is restored after a LiDAR refinement.
  void anchorTo(double t_ref, const M3D& R_ref, const V3D& p_ref);

  // One damped Gauss-Newton step on the POSITION control points against
  // `obs`.  Returns true if a step was applied; false if there was nothing to
  // do, the system was singular, or the step exceeded lidar_refine_max_step
  // (in which case the spline is left exactly as it was).  Call anchorTo()
  // immediately afterwards -- see SplineOptions::lidar_refine_cp.
  bool refineWithLidar(const std::vector<SplineLidarObs>& obs,
                       const SplineOptions& opts);

  double rotationChordDeg() const;

  void basisAt(double t, int& first_cp, Eigen::Vector4d& b,
               Eigen::Vector4d& db, Eigen::Vector4d& ddb) const;

  Eigen::Matrix<double, 3, Eigen::Dynamic>& cpPosMut() { return cp_p_; }
  const Eigen::Matrix<double, 3, Eigen::Dynamic>& cpPos() const { return cp_p_; }

  // Diagnostics from the last fit / refinement.
  double fitResidualPos() const { return fit_res_pos_; }   // RMS, m
  double fitResidualRot() const { return fit_res_rot_; }   // RMS, rad
  double fitRegFrac()     const { return fit_reg_frac_; }
  double lastRefineStep() const { return last_refine_step_; }   // max |dcp|, m
  int    refineRejects()  const { return refine_rejects_; }
  int    refineApplied()  const { return refine_applied_; }

private:
  bool   valid_ = false;
  bool   cumulative_ = true;
  int    n_cp_  = 0;
  int    n_seg_ = 0;
  double t0_ = 0.0, t1_ = 0.0, delta_ = 0.0, inv_delta_ = 0.0;

  M3D    R_anchor_ = M3D::Identity();
  Eigen::Matrix<double, 3, Eigen::Dynamic> cp_p_;     // 3 x n_cp, world
  Eigen::Matrix<double, 3, Eigen::Dynamic> cp_phi_;   // 3 x n_cp, tangent mode
  std::vector<M3D> cp_R_;                             // n_cp, cumulative mode

  double fit_res_pos_ = 0.0, fit_res_rot_ = 0.0, fit_reg_frac_ = 0.0;
  double last_refine_step_ = 0.0;
  int    refine_rejects_ = 0, refine_applied_ = 0;
  int    n_cp_req_ = 0;

  V3D phiAt(double t) const;
  V3D phiDotAt(double t) const;
  void cumBasisAt(double t, int& first_cp, Eigen::Vector3d& Bt,
                  Eigen::Vector3d& dBt) const;
  bool fitRotationCumulative(const std::vector<Pose6D>& poses, int gn_iters);

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct SplineImuResidualStats
{
  int    n = 0;
  double cov_acc = 0.0;    // (m/s^2)^2, isotropic
  double cov_gyr = 0.0;    // (rad/s)^2, isotropic
  double acf1_acc = 0.0;
  double acf1_gyr = 0.0;
  double max_abs_acc = 0.0;
  double max_abs_gyr = 0.0;
  bool   valid() const { return n >= 8; }
};

// Replay a stored pose recurrence under corrected biases and gravity.
//
// `in` is the sequence ImuProc::propagate() produced; Pose6D carries, per
// step, the WORLD accelerations at the step's head and tail and the mean body
// rate already corrected by the OLD gyro bias, which is enough to recover the
// body-frame measurement exactly:
//
//     a_head_body = R_k^T (acc_head - g_old)          == acc_raw_head - ba_old
//     a_tail_body = (R_k Exp(gyr_k, dt))^T (acc_tail - g_old)
//
// The recurrence is then re-run with (bias + d_bias) and (g_old + d_gravity),
// starting from in.front()'s pose and velocity.  The initial condition is
// deliberately NOT corrected: the ESIKF's correction is defined at the scan
// END and is carried by anchorTo(), so correcting both ends would apply it
// twice.  This function fixes the SHAPE only, which is the part anchorTo
// cannot reach.
//
// Returns false and leaves `out` untouched when every delta is below its
// threshold -- an exact no-op rather than a round-trip through R^T ... R that
// would perturb the last bits for nothing.
bool reintegratePoses(const std::vector<Pose6D>& in,
                      const V3D& d_bias_acc,
                      const V3D& d_bias_gyr,
                      const V3D& d_gravity,
                      const V3D& gravity_old,
                      double min_dba, double min_dbg,
                      std::vector<Pose6D>& out);

SplineImuResidualStats computeSplineImuResidual(
    const ScanSpline& spline,
    const std::vector<ImuSample>& imu,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity);

}  // namespace livo_recon
