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
// CQ-39 item (2), 2026-09-17: this block used to describe a cumulative-
// rotation-fit path (fitRotationCumulative(), a cp_R_ member) as the
// shipped default. That path was REMOVED by 826619a "revert to linear
// tangent splines" (2026-09-06, an ancestor of HEAD) -- there is no
// fitRotationCumulative() definition anywhere in spline.cpp, and cp_phi_
// (this class's actual rotation control-point storage, see line 37 above)
// is the tangent parameterization. TANGENT IS WHAT SHIPS. Caught after a
// claim built on "the cumulative path" turned out to describe dead code --
// a comment describing a removed path as the default is exactly the kind
// of defect that causes that.
//
// The SP-4a measurement that motivated the (since-reverted) switch is still
// true and still load-bearing, so it's kept here rather than deleted along
// with the conclusion it was used for: at the rotation a real scan
// contains, tangent and a Gauss-Newton cumulative fit are EQUAL. SP-4a
// measured max rot_chord_deg ~6 deg on eee_01; at 8.5 deg and n_cp=8 the two
// agree to three digits. The two approaches only diverge at rotations these
// bags do not reach -- which is also why reverting to tangent cost nothing
// measurable on real sequences.
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

  // ROTATION IS THE TANGENT FORM AND THERE IS NO OTHER (Bryce, 2026-09-06).
  //
  //   R(t) = R_anchor * Exp( sum_j b_j(u) cp_phi_[s+j] )
  //
  // phi(t) is LINEAR in its control points, so rotation solves in the SAME
  // linear system as position -- one LDLT, exact, no surrogate and nothing
  // to repair.  The cumulative form could not: R(t) = cp_R[s] * prod
  // Exp(Btilde_j d_j) with d_j = Log(cp_R[s+j-1]^T cp_R[s+j]) is a product
  // of exponentials of RELATIVE rotations, each coupling two control points,
  // so it is not of the form A x = b.  It was therefore fitted in this same
  // tangent chart and then RE-ENCODED by sampling at the Greville abscissae,
  // which loses twice over -- a B-spline does not pass through its control
  // points, and the two parameterisations agree only to first order -- and
  // Gauss-Newton existed solely to repair that loss.
  //
  // WHY THE STANDARD FORM IS NOT KEPT.  This header's own measurement:
  // "At the rotation a real scan contains they are EQUAL.  SP-4a measured
  // max rot_chord_deg ~6 deg on eee_01; at 8.5 deg and n_cp=8 the two agree
  // to three digits.  The gap only opens at rotations these bags do not
  // reach."  SP-4a"/4b" did measure cumulative ahead on real data by 17-20%
  // on exp05, which sits AT T0-G's 18.88% nondeterminism bar rather than
  // clear of it -- consistent-direction evidence on two arms, not a
  // measured margin.  That is the one result this removal overrides.
  //
  // THE COST IS A CHART ASSUMPTION AND IT IS BOUNDED BY SCAN DURATION.
  // R_anchor is the MIDDLE pose, halving the |phi| the chart must carry, so
  // at the measured 26.86 deg max chord |phi| <= ~13.4 deg -- nowhere near
  // the pi injectivity radius.  Longer scans or faster motion erode that,
  // which is why fit() refuses above CHART_MAX_PHI_RAD rather than
  // silently distorting.
  //
  // 1.0 rad = 57.3 deg, about 4x the largest chord these sequences produce
  // and well inside the pi injectivity radius.
  //
  // CQ-22 (2026-09-14) stood this down from a refusal to a WARNING: a
  // refusal falls through to deskewPoints(), stratum A's own treatment, so
  // refusing inside a spline+refine cell silently mixed two of TQ-12's
  // three levels. Nothing here reverses that call.
  //
  // CQ-23 (2026-09-14) found the warning was carrying two different
  // questions and split it. eee_01 scan 3133 hit max_abs_cp_phi =
  // 64.5624 rad -- ~3700 degrees, twenty times the pi injectivity radius --
  // while its own rot_chord_deg (0.6996 rad) was unremarkable next to its
  // neighbours' (~0.3 rad). That is not a chart degrading as rotation
  // grows; it is a solve that produced garbage: Eigen::LDLT reported
  // Success on both the position and tangent-rotation systems
  // (ldlt_p.info()==ldlt_r.info()==Success), but vectorD().minCoeff() was
  // EXACTLY 0 on both -- a numerically singular normal-equations matrix
  // Success did not catch. CHART_MAX_PHI_RAD (below) answers "is the chart
  // degrading" and should stay a warning. CHART_HARD_PHI_RAD answers "did
  // the solve diverge" and belongs with fit()'s other hard-failure causes,
  // where falling back to deskewPoints() is correct and the cell is
  // honestly marked, exactly like kSolveFailed/kNonFinite above it.
  static constexpr double CHART_MAX_PHI_RAD = 1.0;
  //
  // NOT SET BY THIS CARD/COMMIT -- CQ-23's own DELIVERS item (3) is a
  // proposal, not a committed value ("Bryce sets the number"). Defaulted
  // to a value that makes this ceiling a no-op (never fires) until Bryce
  // sets a real one, so shipping this mechanism does not silently start
  // refusing frames nobody has reviewed the threshold for. CQ-23's own
  // filed evidence: the largest max_abs_cp_phi any FITTING frame reached
  // across 3 sequences (excluding scan 3133 itself, which this mechanism
  // now refuses) was 0.300360 rad (site1_handheld_1); the one observed
  // divergent value was 64.5624 rad. A proposed CHART_HARD_PHI_RAD = M_PI
  // (~3.14159 rad, the exponential map's actual injectivity radius) sits
  // ~10.5x above the largest fitting value and ~20x below the observed
  // divergence, with no data point anywhere in that gap -- see the CQ-23
  // RESULTS entry for the full distribution this proposal is based on.
  static constexpr double CHART_HARD_PHI_RAD = 1.0e9;
  //
  // CQ-24 (2026-09-14): CHART_HARD_PHI_RAD catches a bad ANSWER; this
  // catches a bad SOLVE, and only one of the two is bounded.
  // Eigen::LDLT::info() reports allocation/input validity, not rank -- it
  // read Success on eee_01 scan 3133's exactly-singular
  // (vectorD().minCoeff()==0) normal-equations matrix. The chart guard
  // caught that ONE case only because its garbage happened to be enormous
  // in phi (64.5624 rad); a near-singular solve producing MODERATELY wrong
  // control points passes every existing guard silently.
  //
  // CQ-24 item (2) measured BOTH candidates before choosing, and the
  // absolute floor won, reversing this card's own a-priori expectation
  // ("an absolute pivot floor is wrong on a matrix whose scale moves with
  // the residual count; the ratio is what transfers"). Measured on 3
  // sequences: dmin's healthy range is remarkably STABLE across sequences
  // (p50 0.827-0.841, p01 0.525-0.820 -- all comfortably inside
  // [0.5, 1.0]), while the condition-ratio's healthy MAXIMUM varies by
  // ~2x across sequences (496 on eee_01, 920 on exp05, 463 on site1) --
  // a fixed ratio threshold picked safely above one sequence's noise
  // floor has a thinner, less predictable margin on another. This is
  // NOT an accident: AtA_p and AtA_r are the SAME left-hand-side matrix
  // (only Atb_p/Atb_r, the right-hand sides, differ -- see fit()), so
  // their pivots (and therefore both metrics) are IDENTICAL between the
  // position and rotation solves on every frame. See the CQ-24 RESULTS
  // entry for the full distributions this paragraph describes.
  //
  // CQ-41 (2026-09-18): the identity-pinned boundary rows this paragraph
  // used to credit for the healthy p50 0.827-0.841 range are gone --
  // freezeAbsoluteScalarSystem() was deleted, replaced by a KKT-
  // constrained solve (see fit()'s own comment). dmin_p/dmax_p are still
  // read off a separate LDLT of AtA alone (item 3a), so PIVOT_MIN_FLOOR's
  // meaning is unchanged, but dmin_p is now genuinely DATA-dependent
  // rather than dominated by fixed identity structure -- measured
  // (CQ-41 filing) at two orders of magnitude below the old range
  // (p50 ~0.0006 vs the ~0.83 above, eee_01, velocity:true default),
  // confirming item (3b)'s own prediction that removing the freeze would
  // expose real end conditioning rather than hide it.
  //
  // NOT SET BY THIS CARD/COMMIT, same discipline as CHART_HARD_PHI_RAD:
  // -1.0 is the true no-op (every real pivot is >= 0 by construction, so
  // dmin < -1.0 can never fire).  Proposed value: 0.1 -- roughly 5x below
  // the smallest healthy p01 observed (0.525) and far above the one
  // observed singular case (exactly 0).
  static constexpr double PIVOT_MIN_FLOOR = -1.0;
  //

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

  // CQ-43 item (2), Bryce 2026-09-18: refineSplineFromResiduals() only ever
  // runs INSIDE the IEKF loop, against that iteration's pre-update
  // residuals -- the converged state's own final correction reaches the
  // spline interior solely via moveTailClamp()'s ramp, never a refit (see
  // the card's own finding). When true, after the frame's state is fully
  // converged, rebuild residuals against the FINAL trajectory and refine
  // once more -- READ-ONLY with respect to state_ (no estimateStateCorrection(),
  // no covariance update, no new moveTailClamp() target), so it can only
  // change spline_'s own interior shape and, downstream, the deskewed
  // points VoxelMap::updateMap() builds the map from. DEFAULT false --
  // this is a behavior change (the map), not a diagnostic, and item (6)
  // says explicitly not to flip this default from this card's own numbers.
  bool final_pass = false;

  // CQ-41 follow-up (2026-09-18): second-difference (discrete curvature)
  // regulariser on the refinement step, relative to trace(H)/dim (same
  // convention as the fixed positional Tikhonov above). Config key:
  // spline/refine/curvature_weight. 0.0 (default) preserves prior behavior
  // exactly -- opt in explicitly. A/B on eee_01 (velocity:true, weight=1.0):
  // max_abs_acc p50 2061->98 m/s^2 (-95%), ATE 0.0268->0.0253m (improved),
  // constraint (pos_err at boundary) unchanged (~1e-15). See
  // refineWithLidar()'s own comment for the mechanism (curvature between
  // adjacent control points is what accAt()'s 1/delta^2 amplifies; the
  // plain magnitude-toward-prior Tikhonov above cannot distinguish a smooth
  // correction from an oscillating one that costs the same under an
  // isotropic-magnitude penalty).
  double refine_curvature_weight = 0.0;

  // CQ-41 follow-up (2026-09-18), third term: anchor the refined spline's
  // ACCELERATION to what the raw accelerometer actually measured, weighted
  // by the calibration noise floor (1/var_acc_floor), rather than only
  // regularising the control polygon's own shape (curvature, above) or
  // magnitude (REFINE_TIKHONOV). Complementary to curvature_weight, not a
  // replacement -- curvature penalises internal smoothness; this anchors
  // absolute level to physical reality. Config key:
  // spline/refine/imu_acc_weight. 0.0 (default) preserves prior behavior.
  double refine_imu_acc_weight = 0.0;

  // Bryce, 2026-09-18: experiment -- does letting the refinement solve for
  // a small PER-SCAN accelerometer bias CORRECTION (on top of the fixed
  // state_->biasAcc() the imu_acc term above otherwise anchors to) let the
  // endpoint snapping resolve through a bias shift instead of curvature,
  // while still fitting well? Only meaningful when refine_imu_acc_weight >
  // 0 (there is otherwise no term this correction could affect). The
  // correction is regularised toward zero with variance
  // cov_bias_acc * scan_duration -- the SAME process-noise budget the
  // EKF's own bias random walk already allows per scan (see
  // ImuProc::propagate()'s q_alpha_bias term) -- specifically so this
  // cannot invent a per-scan bias that the EKF's own uncertainty model
  // would consider implausible. NOT fed back into state_->biasAcc() (that
  // coupling is the open question, not yet resolved) -- purely diagnostic
  // for now: see ScanSpline::lastDeltaBiasAcc().
  bool refine_imu_acc_solve_bias = false;

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

  // Both endpoints are constrained unconditionally -- see `mode` above.
  // N_FROZEN_CP names how many control points setFrozenBoundary()'s caller
  // reserves at each end for the KKT constraint construction (fit()'s own
  // doc comment) -- it is not itself a freeze count any more (CQ-41: the
  // repeated-triple freeze that forced acceleration/angular rate to zero is
  // gone), just the historical name for "boundary values are available this
  // call" (0 = first scan of a run, no boundary yet).
  static constexpr int N_FROZEN_CP = 3;

  // CQ-41, Bryce 2026-09-18: "they should be unconstrained as they're
  // variables we're trying to estimate and compare to the imu measurements."
  // The endpoint constraint is now a LINEAR EQUALITY solved via a KKT
  // system (see fit()'s own doc comment), not a repeated-control-point
  // freeze -- the freeze's side effect of also forcing acceleration and
  // angular velocity to exactly zero at both ends (an unintended
  // consequence of the repeated-triple identity, never a chosen behavior --
  // setFrozenBoundary() never had a velocity parameter) is now structurally
  // impossible: there is no key, enum value or code path anywhere in this
  // class that can constrain acceleration, angular velocity, or angular
  // acceleration. Only position and velocity (this flag) and attitude
  // (always, both settings) are ever constrained.
  //   true  (DEFAULT, Bryce-authorised 2026-09-18): position AND velocity
  //         constrained at both ends (k=4 constraint rows on the position
  //         channel); attitude constrained at both ends (k=2, unconditional).
  //   false: position only at both ends (k=2); attitude unchanged (k=2).
  bool end_constraint_velocity = true;
};

// CQ-22 item (4): one named cause per fit() early-return site, so a run's
// failure count can be attributed rather than lumped into one shared
// counter.
//
// CQ-23: kChartGuard is live again. CQ-22 stood the chart guard down to a
// pure warning (no return false at all); CQ-23 split it into
// CHART_MAX_PHI_RAD (still a warning, chartGuardWarned()) and
// CHART_HARD_PHI_RAD (a real ninth failure cause -- a solve that diverged,
// not a chart that degraded -- see SplineOptions::CHART_HARD_PHI_RAD's own
// comment for why these are different questions).
enum class FitFailCause
{
  kNone = 0,
  kTooFewPoses,        // poses.size() < 2
  kBadWindow,          // !(t1 > t0)
  kTooFewSamples,      // n_samples < 5
  kNCpTooSmall,        // n_cp < 4 after clamping to n_samples - 1
  kDegenerateDelta,    // !(delta_ > 1e-9)
  kUnderdetermined,    // fewer in-window samples than n_cp_ + 1
  kSolveFailed,        // LDLT::info() != Success
  kNonFinite,          // solved control points not all finite
  kChartGuard,         // max_abs_cp_phi > CHART_HARD_PHI_RAD (CQ-23)
  kPivotGuard,         // condition ratio > PIVOT_MAX_CONDITION_RATIO (CQ-24)
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
    refine_dcp_max_ = 0.0; refine_dcp_rms_ = 0.0; last_delta_bias_acc_ = V3D::Zero(); }

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
  // CQ-41 item (4): vel0/vel1 added -- the head velocity comes from the
  // PREVIOUS scan's converged state_->vel() (LioProc::prev_scan_end_vel_,
  // populated the same place/time as prev_scan_end_pos_/prev_scan_end_rot_),
  // the tail velocity from state_->vel() at the CURRENT IEKF iteration
  // (moveTailClamp() updates it every iteration, same as pos1/rot1 always
  // have). Both are ignored by the k=2 (velocity:false) constraint set but
  // always stored, since which setting is active can change frame to frame.
  void setFrozenBoundary(int n_frozen,
                         const V3D& pos0, const M3D& rot0, const V3D& vel0,
                         const V3D& pos1, const M3D& rot1, const V3D& vel1)
  { n_frozen_cp_ = n_frozen;
    frozen_pos_ = pos0; frozen_rot_ = rot0; frozen_vel_ = vel0;
    frozen_pos1_ = pos1; frozen_rot1_ = rot1; frozen_vel1_ = vel1; }
  int nFrozenCp() const { return n_frozen_cp_; }

  // CQ-41 item (3c): move the TAIL constraint to a new scan-end pose
  // without re-fitting, so refinement already applied to the interior
  // survives -- via a constraint-increment solve reusing fit()'s own KKT
  // factorization (see spline.cpp's implementation comment for the linear
  // system). REPLACES the old time-proportional ramp, kept only at HEAD~1.
  // vel1 is ignored when the constrained fit this scan's cache came from
  // used end_constraint_velocity=false.
  void moveTailClamp(const V3D& pos1, const M3D& rot1, const V3D& vel1);

  bool valid() const { return valid_; }
  int  nControlPoints() const { return n_cp_; }
  double t0() const { return t0_; }
  double t1() const { return t1_; }
  // n_cp as REQUESTED vs as actually used.  A cubic fit needs strictly more
  // pose samples than control points, so the fit silently shrinks n_cp to
  // n_samples-1.  At a 10 Hz scan and a 200 Hz IMU that ceiling is ~19,
  // which means every requested n_cp above it collapses onto the same fit --
  // a sweep that asks for 20, 24 and 32 gets three identical cells.  Both
  // numbers go in spline_q.csv so that is visible instead of silent.
  int  nControlPointsRequested() const { return n_cp_req_; }
  bool nControlPointsClamped() const { return n_cp_req_ != n_cp_; }

  // Evaluation.  `t` is clamped to [t0_, t1_].
  //
  // PREFER poseAt() WHERE BOTH ARE WANTED.  rotAt() and posAt() each do their
  // own segment lookup, so calling both costs the index arithmetic twice --
  // and the per-point deskew calls both, for every point, on every IEKF
  // iteration.
  void poseAt(double t, M3D& R, V3D& p) const;

  // ── SEGMENT-MAJOR EVALUATION, the deskewPoints() pattern ────────────────
  // deskewPoints() hoists everything that depends only on the POSE BRACKET
  // out of its per-point body and refreshes it when the bracket changes.
  // The spline's bracket is the SEGMENT: within one segment the four control
  // points and the three relative rotations are fixed, and only the local
  // coordinate u varies per point.  SegView is that hoist, made explicit.
  //
  // A scan's points arrive in time order, so a linear sweep changes segment
  // n_seg times (about 10) rather than n_points times (thousands).  The CSR
  // traversal is voxel-major and therefore not monotone in t, so it refreshes
  // more often -- still never more than once per point, which is what the
  // unhoisted code paid unconditionally.
  struct SegView
  {
    int  s = -1;              // segment index this view is built for
    V3D  cp[4];               // position control points s..s+3
    V3D  phi[4];              // rotation control points s..s+3
  };

  // Segment index and local coordinate for `t`, with `t` clamped to the
  // window.  Cheap: one multiply, one floor, two compares.
  void locate(double t, int& s, double& u) const;

  // Refresh `v` only if it is not already built for segment `s`.
  void buildSegView(int s, SegView& v) const;

  // Evaluate using a view already built for this point's segment.
  void poseAtSeg(const SegView& v, double u, M3D& R, V3D& p) const;

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
  // log_debug_en: reuses LioProcOptions::log_debug_en (threaded by the
  // caller, LioProc::refineSplineFromResiduals()) -- not a new config key.
  // CQ-39 item (1): gates a per-pass proof log (it, max_step, g.norm(),
  // H.trace(), step.norm()) that this inner loop's H/g/step are IDENTICAL
  // every pass when lidar_refine_iters > 1, since `obs` never changes
  // inside the loop -- see the inner loop's own comment.
  // imu_raw/bias_acc/gravity/var_acc_floor/cov_bias_acc: only consulted
  // when opts.refine_imu_acc_weight > 0 -- see that field's doc comment.
  // Cheap to pass unconditionally (the caller already has all five on hand
  // for computeSplineImuResidual()/ImuProc::propagate()) rather than making
  // them optional.
  bool refineWithLidar(const std::vector<SplineLidarObs>& obs,
                       const SplineOptions& opts,
                       const std::vector<ImuSample>& imu_raw,
                       const V3D& bias_acc, const V3D& gravity,
                       double var_acc_floor, const V3D& cov_bias_acc,
                       bool log_debug_en = false);

  // Diagnostic only (see SplineOptions::refine_imu_acc_solve_bias) -- the
  // per-scan accelerometer bias correction the LAST refineWithLidar() pass
  // solved for, or zero if solve_bias was off / never ran. NOT applied to
  // state_->biasAcc() anywhere.
  const V3D& lastDeltaBiasAcc() const { return last_delta_bias_acc_; }

  // CQ-43 item 0: how far the LAST moveTailClamp() call actually moved the
  // tail constraint target, in the position/rotation channels respectively
  // -- -1.0 sentinel if that call returned early (kkt_k_p_<=0, no boundary
  // this scan) rather than 0.0, so "moved by exactly zero" and "never
  // measured" stay distinguishable.
  double lastTailMoveDpNorm()   const { return last_tail_move_dp_norm_; }
  double lastTailMoveDphiNorm() const { return last_tail_move_dphi_norm_; }

  // Bryce, 2026-09-18: diagnostic -- does the LiDAR-implied trajectory
  // agree with the TAIL boundary target (state_->pos()/vel(), what the
  // real fit is pinned to), or is there a persistent disagreement (which
  // would point at the EKF's propagated boundary target itself being
  // subtly wrong -- e.g. from accel/gyro bias mis-estimation -- rather
  // than the curvature being an artifact of forcing exact equality)? One
  // Gauss-Newton step against the SAME LiDAR data term refineWithLidar()
  // uses, starting from the current (converged, both-ends-constrained)
  // cp_p_, but with ONLY the head frozen (to the trusted, already-
  // converged previous scan's end pose) -- reads where the tail naturally
  // wants to land. Purely read-only: does not modify cp_p_ or any other
  // spline state. Requires a boundary to exist (nFrozenCp() > 0) and
  // enough LiDAR observations, same preconditions as refineWithLidar().
  // imu_raw/bias_acc/gravity/var_acc_floor: same as refineWithLidar()'s own
  // -- included here too (when opts.refine_imu_acc_weight > 0) so this is a
  // fair "regularized + constrained" vs "regularized + free" comparison,
  // not "regularized + constrained" vs "unregularized + free". Curvature
  // (refine_curvature_weight) is included the same way.
  // TQ-36, Bryce 2026-09-18: cp_free_out/fit_res_pos_free_out are additive
  // outputs -- cp_free_out is the FULL (3 x n_cp) free-tail control-point
  // matrix (not just its t1 evaluation), so a caller can read the
  // per-control-point delta d_i = cp_free_out.col(i) - cp_p_.col(i) for
  // EVERY i, not only the boundary. fit_res_pos_free_out is
  // updateFitResiduals()'s own RMS-vs-poses formula (posAt(ps.t)-ps.pos,
  // squared, averaged, sqrt), evaluated against cp_free_out instead of
  // cp_p_ -- needs `poses` (the same mg.poses fit() was built from) since
  // it can't reuse the member posAt() (that reads live cp_p_, not the
  // hypothetical free-tail solution).
  bool diagnosticFreeTailFit(const std::vector<SplineLidarObs>& obs,
                             const SplineOptions& opts,
                             const std::vector<ImuSample>& imu_raw,
                             const V3D& bias_acc, const V3D& gravity,
                             double var_acc_floor,
                             const std::vector<Pose6D>& poses,
                             V3D& pos1_free, V3D& vel1_free, V3D& acc1_free,
                             M3D& cov_pos1_free,
                             Eigen::MatrixXd& cp_free_out,
                             double& fit_res_pos_free_out) const;

  double rotationChordDeg() const;

  // CQ-21 item (5): max_i |cp_phi_[i]| -- the EXACT quantity the chart
  // guard checks against CHART_MAX_PHI_RAD/CHART_HARD_PHI_RAD, not the
  // end-to-end chord rotationChordDeg() reports.
  //
  // CQ-23 item (2): stored at computation time (see fit()), not
  // recomputed live from cp_phi_ gated on valid_ -- a HARD-refused fit
  // (CHART_HARD_PHI_RAD exceeded) sets valid_=false, and the whole point
  // of this change is that the value which caused a refusal must still be
  // observable, not hidden by the refusal itself. 0.0 only for a fit that
  // never reached the point of computing cp_phi_ at all (an early-return
  // cause before the LDLT solve).
  double maxAbsCpPhi() const { return last_max_abs_cp_phi_; }

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

  // CQ-22 item (4): which of fit()'s early-return sites fired last time
  // fit() returned false. kNone after a successful fit.
  FitFailCause lastFitFailCause() const { return fail_cause_; }
  // CQ-22 item (4): true if the last fit() exceeded CHART_MAX_PHI_RAD (the
  // soft threshold) at any control point. Independent of
  // lastFitFailCause() -- this can be true on a fit that still succeeded.
  bool chartGuardWarned() const { return chart_guard_warned_; }
  // CQ-23 item (4): true if the last fit() exceeded CHART_HARD_PHI_RAD.
  // When true, lastFitFailCause() == kChartGuard and valid_ == false.
  bool chartGuardHard() const { return chart_guard_hard_; }
  // CQ-24 item (1)/(2): the LDLT pivot floor/ceiling for both solves this
  // fit, stored regardless of info() or of which threshold (if any)
  // ultimately refuses -- -1.0 if fit() never reached the LDLT solve.
  double dminPos() const { return dmin_p_; }
  double dmaxPos() const { return dmax_p_; }
  double dminRot() const { return dmin_r_; }
  double dmaxRot() const { return dmax_r_; }
  // CQ-24 item (3): true if the last fit() exceeded
  // PIVOT_MAX_CONDITION_RATIO. When true, lastFitFailCause() ==
  // kPivotGuard and valid_ == false.
  bool pivotGuardFired() const { return pivot_guard_; }

  // CQ-22 item (5): recompute fit_res_pos_/fit_res_rot_ against `poses`
  // using the CURRENT control points. Call this after the last
  // moveTailClamp() of the frame (not from inside fit(), which runs before
  // any clamp move) so the residual reflects the converged tail, not the
  // frozen-at-propagation one. RMS over the same in-window poses fit()
  // itself uses. No-op (0.0/0.0) if the fit is not valid.
  void updateFitResiduals(const std::vector<Pose6D>& poses);

  int    n_frozen_cp_ = 0;
  V3D    frozen_pos_  = V3D::Zero();
  M3D    frozen_rot_  = M3D::Identity();
  V3D    frozen_vel_  = V3D::Zero();
  V3D    frozen_pos1_ = V3D::Zero();
  M3D    frozen_rot1_ = M3D::Identity();
  V3D    frozen_vel1_ = V3D::Zero();

  // CQ-41 item (3c): the KKT factorization built by fit() for the position
  // and rotation channels, cached so moveTailClamp() can reuse it for a
  // constraint-increment solve instead of the old ad hoc time-ramp --
  // "The KKT factorisation depends only on AtA and C, never on the RHS, so
  // it can be REUSED." Sized (n_cp_+k) x (n_cp_+k), k = kPosConstraints()
  // for position (4 if end_constraint_velocity else 2) or 2 for rotation
  // (always -- attitude is constrained in both settings). Invalid (empty)
  // whenever n_frozen_cp_==0 (first scan of a run, no boundary -- fit() is
  // unconstrained OLS, k=0, nothing to cache).
  Eigen::LDLT<Eigen::MatrixXd> kkt_ldlt_p_, kkt_ldlt_r_;
  int kkt_k_p_ = 0, kkt_k_r_ = 0;
  // Raw (unfactored) position-channel constraint rows, cached alongside the
  // factorization -- refineWithLidar() needs the RAW C to block-expand into
  // Cb (item 6), not the KKT factorization itself (that system's H differs
  // from fit()'s AtA).
  Eigen::MatrixXd kkt_C_p_;

  bool   valid_ = false;
  int    n_cp_  = 0;
  int    n_seg_ = 0;
  double t0_ = 0.0, t1_ = 0.0, delta_ = 0.0, inv_delta_ = 0.0;

  M3D    R_anchor_ = M3D::Identity();
  Eigen::Matrix<double, 3, Eigen::Dynamic> cp_p_;     // 3 x n_cp, world
  Eigen::Matrix<double, 3, Eigen::Dynamic> cp_phi_;   // 3 x n_cp, tangent mode

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
  V3D    last_delta_bias_acc_ = V3D::Zero();
  double last_tail_move_dp_norm_ = -1.0, last_tail_move_dphi_norm_ = -1.0;
  int    n_cp_req_ = 0;

  FitFailCause fail_cause_ = FitFailCause::kNone;
  bool         chart_guard_warned_ = false;
  bool         chart_guard_hard_ = false;
  double       last_max_abs_cp_phi_ = 0.0;
  bool         pivot_guard_ = false;
  double       dmin_p_ = -1.0, dmax_p_ = -1.0, dmin_r_ = -1.0, dmax_r_ = -1.0;

  V3D phiAt(double t) const;
  V3D phiDotAt(double t) const;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct SplineImuResidualStats
{
  int    n = 0;
  double cov_acc = 0.0;    // (m/s^2)^2, isotropic -- SPREAD of (pred-raw) around
                           // its own mean, not the deviation from raw itself.
  double cov_gyr = 0.0;    // (rad/s)^2, isotropic -- same caveat.
  double acf1_acc = 0.0;
  double acf1_gyr = 0.0;
  double max_abs_acc = 0.0;  // also mean-subtracted -- see cov_acc's caveat.
  double max_abs_gyr = 0.0;
  // Mean-residual magnitude: ||mean_i(pred_i - raw_i)|| -- a genuine, honest
  // "how far is the predicted acc/gyro from what the IMU actually measured"
  // number, distinct from cov_acc/max_abs_acc above (both of which subtract
  // this mean out first and therefore cannot see it). A large mean_abs_acc
  // with small cov_acc means predicted and raw disagree consistently, not
  // just noisily.
  double mean_abs_acc = 0.0;
  double mean_abs_gyr = 0.0;
  bool   valid() const { return n >= 8; }
};

SplineImuResidualStats computeSplineImuResidual(
    const ScanSpline& spline,
    const std::vector<ImuSample>& imu,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity);

}  // namespace livo_recon
