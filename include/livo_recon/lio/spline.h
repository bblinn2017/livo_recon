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
  // ── THE ONE MODE KNOB.  Bryce, 2026-09-06. ───────────────────────────────
  // This replaces `enable`, `per_iteration` (five levels) and
  // `boundary_anchor_mode` (three levels).  Those were 30 nominal
  // combinations carrying five distinct behaviours, and SP-4a"/4b" spent
  // half a 276-job grid landing on ATE-identical duplicates discovering it.
  //
  //   "raw_imu"        No spline is fitted.  The points are deskewed ONCE,
  //                    before the first IEKF iteration, from the IMU-
  //                    propagated poses.  Every other key in this struct --
  //                    and every adaptive_q/* key -- is REFUSED at startup.
  //   "spline"         The spline is fitted once per frame with BOTH
  //                    endpoints clamped (see below), and the points are
  //                    re-deskewed against it every IEKF iteration.  No
  //                    LiDAR refinement of the control points.
  //   "spline+refine"  As above, and each iteration the control points'
  //                    interior shape is refined against the SAME residuals
  //                    the IEKF accumulates from -- so both are linearised
  //                    at the same trajectory and neither is stale.
  //
  // BOTH ENDPOINTS ARE CLAMPED BY CONSTRUCTION, AND anchorTo() IS GONE.
  // spline(t0) IS the previous scan's final pose and spline(t1) IS this
  // iteration's corrected pose -- not approximately, not after a rigid
  // correction, but as an identity the fit is solved under.  The uniform
  // cubic basis at u=0 is [1/6, 4/6, 1/6, 0] and at u=1 is
  // [0, 1/6, 4/6, 1/6], each summing to 1, so freezing three control points
  // at each end forces both endpoints exactly.  Cumulative rotation obeys
  // the mirror identity for the same reason.
  //
  // WHY THAT MATTERS AND IT IS NOT TIDINESS.  anchorTo() applied ONE rigid
  // correction to the point at t0 and the point at t1 alike -- but the
  // correction is drift accumulated ACROSS THIS SCAN, and at t0 the state
  // was already corrected by the previous scan's own update.  So it moved a
  // point that was already right, and boundary_dpos measured exactly that:
  // p50 0.00798 m, p90 0.0181 m (B-0).
  std::string mode = "spline";
  static constexpr const char* MODES[] = { "raw_imu", "spline", "spline+refine" };
  bool splineOn() const { return mode != "raw_imu"; }
  bool refineOn() const { return mode == "spline+refine"; }

  // Rotation is ALWAYS the cumulative form (Bryce, 2026-09-06).  The tangent
  // parameterisation was the ablation arm and is removed.
  //
  // GN refinements of the control ROTATIONS after the tangent-space
  // initialisation.  NOT a tuning knob and no longer a config key: 0 does not
  // mean "cheaper", it means the Greville re-encoding's loss is never undone
  // and the spline is unusable.  4 was set when the exact Jacobian landed;
  // 2 was measured insufficient at n_cp = 8 above ~34 deg of chord.
  static constexpr int ROT_FIT_ITERS = 4;

  // Control-point rate in Hz over the ACTUAL scan duration.  n_cp is derived
  // from it, never set directly -- a fixed n_cp is a different control rate
  // on every sequence with a different scan duration, so it does not
  // transfer between bags and Hz is the physically comparable axis.
  //
  // *** 100 Hz AGAINST A 200 Hz IMU, AND THE FLOOR IS NOT ARBITRARY. ***
  // Both endpoints are clamped, which spends SIX control points -- three at
  // each end -- so the free interior shape is n_cp - 6.  At a 10 Hz scan,
  // 100 Hz gives n_cp = 13 and therefore 7 free.  Below ~70 Hz (n_cp = 10)
  // the interior has fewer than 4 free control points and the shape is
  // essentially dictated by the two clamps.  fit() additionally clamps n_cp
  // to n_samples - 1, about 19 at 10 Hz scan / 200 Hz IMU.
  double control_point_hz = 100.0;

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
  // DIVISION OF LABOUR, and it is now STRUCTURAL rather than restored after
  // the fact: the ESIKF owns both endpoints, which are frozen control points
  // the refinement's solved step is forced to zero on; this refinement owns
  // only the intra-scan SHAPE.  The two estimators cannot fight over the
  // same quantity because the refinement cannot reach it.
  // Rotation control points are NOT refined here -- they stay with the
  // IMU + ESIKF.
  // Only `iters` survives (Bryce, 2026-09-06).  prior_w and max_step are
  // gone as CONFIG KEYS; a fixed Tikhonov term stays inside refineWithLidar()
  // because without ANY regulariser the normal equations are rank-deficient
  // in every direction the plane normals do not span -- a corridor or a
  // facade being exactly that -- and the solve would fail into the unrefined
  // spline in precisely the scenes refinement is for.
  int lidar_refine_iters = 1;

  // Per-scan CSV of the fit and the IMU residual (spline_q.csv).
  // ── dumping the trajectory as a FUNCTION, for analysis only ──────────
  //
  // results_lio.txt records one pose per scan, so any comparison against a
  // ground-truth timestamp interpolates the estimate across a ~0.1 s gap --
  // and on a spline run that is throwing away the very thing the spline is:
  // a queryable function of time.  "dense" evaluates posAt()/rotAt() on a
  // fixed grid across each scan's own window and writes them, so an analysis
  // can land on GT timestamps with interpolation error set by the grid rather
  // than by the scan rate.
  //
  // Deliberately NOT a control-point dump.  Reconstructing the cumulative
  // SO(3) blending in Python would be a second implementation of the exact
  // Jacobian's own basis, to be kept in sync forever, and its bugs would read
  // as trajectory error.  Evaluating the C++ we actually ship has no such
  // failure mode.
  //
  // ANALYSIS ONLY.  This is not a scoring path: scores come from
  // results_lio.txt as they always have, so nothing here can flatter a run.
  // ~200 Hz over a 120 s sequence is ~24k rows, a couple of MB -- cheap
  // enough to leave on for a diagnostic row and pointless for a sweep.
  std::string traj_log_mode = "off";
  static constexpr const char* TRAJ_LOG_MODES[] = { "off", "dense" };
  bool trajLogOn() const { return traj_log_mode != "off"; }
  double traj_log_hz = 200.0;

  bool log_en = false;

  // Both endpoints are clamped unconditionally -- see `mode` above.  Three
  // control points at each end, which is the clamped-B-spline identity, and
  // the count is not configurable: 1 ("single_cp") did NOT force the endpoint
  // exactly and existed only as the weaker ablation arm.
  static constexpr int N_FROZEN_CP = 3;
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
  bool fit(const std::vector<Pose6D>& poses, double t0, double t1,
           const SplineOptions& opts);

  // Per-FRAME reset of the refinement counters.  Deliberately NOT done inside
  // fit(): with spline.reintegrate_each_iteration on, fit() runs once per IEKF
  // iteration, and resetting there would make spline_q.csv's refine_applied /
  // refine_rejects report only the last iteration instead of the frame.
  void resetRefineStats()
  { refine_rejects_ = 0; refine_applied_ = 0; last_refine_step_ = 0.0;
    refine_dcp_max_ = 0.0; refine_dcp_rms_ = 0.0; }

  // Freeze the first n control points (0=off, 1="single_cp", 3="exact" --
  // SplineOptions::N_FROZEN_CP) to a single repeated value
  // (pos, rot) BEFORE fit() runs. n=3 repeats the value degree(=3) times,
  // the standard "clamped B-spline" identity: since the basis weights at
  // u=0 are [1/6, 4/6, 1/6, 0] and sum to 1, cp_0=cp_1=cp_2=X forces
  // spline(t0)=X bit-exactly regardless of what any later control point
  // does. n=1 (only cp_0=X) does NOT force this exactly -- cp_1/cp_2 still
  // carry 5/6 of the weight and stay free -- it is deliberately the WEAKER
  // ablation arm. Call before fit(); persists across fit()/
  // refineWithLidar()/fitRotationCumulative() until cleared (n=0).
  void setFrozenBoundary(int n_frozen,
                         const V3D& pos0, const M3D& rot0,
                         const V3D& pos1, const M3D& rot1)
  { n_frozen_cp_ = n_frozen;
    frozen_pos_ = pos0; frozen_rot_ = rot0;
    frozen_pos1_ = pos1; frozen_rot1_ = rot1; }
  int nFrozenCp() const { return n_frozen_cp_; }

  // Move the TAIL clamp to a new scan-end pose without re-fitting, so the
  // refinement already applied to the interior survives.  The correction is
  // distributed PROPORTIONALLY TO ELAPSED TIME across the control points:
  // zero at the head clamp (which stays pinned to the previous scan's end)
  // and full at the tail.  That is the same random-walk assumption Q itself
  // encodes -- process noise accumulates with time, so a control point
  // halfway through the scan has accumulated half the drift and takes half
  // the correction.
  //
  // *** THE DISTRIBUTION IS A MODELLING CHOICE AND IT IS UNVALIDATED. ***
  // Both endpoints being pinned over-determines the trajectory relative to
  // preserving the IMU's own relative increments -- the difference IS the
  // drift the filter corrected -- so something must absorb it and no
  // measurement in this project says what.  Time-proportional is the
  // principled default, not a measured one.  It is one line, below.
  void moveTailClamp(const V3D& pos1, const M3D& rot1);

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

  // One damped Gauss-Newton step on the POSITION control points against
  // `obs`.  Returns true if a step was applied; false if there was nothing to
  // do, the system was singular, or the step was not finite
  // (in which case the spline is left exactly as it was).
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

  // MAGNITUDE, not just acceptance.  refineApplied() counts Gauss-Newton
  // iterations that were allowed through; lastRefineStep() is the LAST
  // iteration's max step and is overwritten each time.  Neither answers "how
  // far did the refinement actually move the trajectory", which is the only
  // question that separates "ran and did nothing" from "ran and mattered" --
  // and with a fixed internal Tikhonov term, a mechanism that is
  // half prior by construction can report an acceptance on every frame while
  // moving the control points by microns.  These are the NET displacement
  // from the pre-refinement fit, accumulated across iterations, per frame.
  double refineDcpMax() const { return refine_dcp_max_; }   // m
  double refineDcpRms() const { return refine_dcp_rms_; }   // m

  int    n_frozen_cp_ = 0;
  V3D    frozen_pos_  = V3D::Zero();
  M3D    frozen_rot_  = M3D::Identity();
  V3D    frozen_pos1_ = V3D::Zero();
  M3D    frozen_rot1_ = M3D::Identity();

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
  V3D    bias_acc_delta_ = V3D::Zero();
  V3D    bias_gyr_delta_ = V3D::Zero();
  V3D    gravity_delta_  = V3D::Zero();
  // See refineDcpMax().  Called on every exit path of refineWithLidar()
  // that may have touched cp_p_, including the reject paths.
  void accumulateRefineDisplacement(
      const Eigen::Matrix<double, 3, Eigen::Dynamic>& cp_prior);

  double last_refine_step_ = 0.0;
  double refine_dcp_max_ = 0.0, refine_dcp_rms_ = 0.0;
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

SplineImuResidualStats computeSplineImuResidual(
    const ScanSpline& spline,
    const std::vector<ImuSample>& imu,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity);

}  // namespace livo_recon
