#pragma once

#include "livo_recon/lio/spline.h"

// ============================================================================
// CQ-82 Phase 2, Artifacts 1/3: the pose-basis coefficient-block builder.
//
// STANDALONE BY DESIGN. This is not yet wired into estimateCoupledCorrection()
// -- the shared dispatcher's state persistence (coupled_c_acc_/coupled_c_gyr_
// and their many raw_imu-specific touch points: bg-projection, delta
// application, effective-config reporting) has raw_imu-specific meaning
// throughout, and safely branching a genuinely different basis through ~600
// lines of that function without risking Phase 1's byte-identical guarantee
// is a separate, larger integration task. This file gives CQ-82's own
// Artifact 3 (item 6's correctness gate) a real, testable implementation of
// the pose basis's own normal-equations construction, verifiable in complete
// isolation (see tests/test_pose_spline_cblock.cpp) -- zero risk to
// any shipped behavior, since nothing here is called from the live node.
//
// LiDAR term (linear in c_p, chain rule in c_phi via Jr(phi)):
//   r_i = n_i^T ( R(t_i) q_i + p(t_i) ) + d_i
//   dr_i/dc_p[j]   = b_j(t_i) * n_i^T
//   dr_i/dc_phi[j] = ( dr_i/dtheta_i ) * Jr(phi_i) * b_j(t_i)
//   dr_i/dtheta_i, the rotation-jacobian row, reuses the SAME convention this
//   codebase's own point_time residual already uses (raw_body_point.cross(
//   R(t)^T * normal) -- verified right-multiplicative, CQ-71's own
//   delta_phi0->dtheta ratio of exactly 1.000000) -- matched, not re-derived.
//
// IMU-as-measurement-factor (per raw IMU sample k, NOT a prior):
//   r_a,k     = R(t_k)^T ( pddot(t_k) - gravity ) + bias_acc - acc_k
//   r_omega,k = omega_spline(t_k) + bias_gyr - gyr_k
//   weighted by pose_imu_weight_{acc,gyr}. The acceleration term is LINEAR
//   in c_p (pddot(t) = sum_j ddb_j(t) cp_p[j]) and computed analytically;
//   every rotation-dependent piece (R(t_k)'s own dependence on c_phi for the
//   acc term, and the whole omega term's dependence on both c_p [none] and
//   c_phi) is differentiated by CENTRAL FINITE DIFFERENCE on a scratch
//   ScanSpline copy -- deliberately, to avoid hand-deriving a second-order
//   tangent-chart chain rule (phi(t) through Jr, through R(t)'s own use in
//   pddot's frame rotation) under real time pressure; FD is an established,
//   already-used pattern in this codebase's own test suite (see
//   test_coupled_phix_full18.cpp) for exactly this kind of cross-check.
//
// Smoothness (second-difference, separately weighted -- "ONE SCALAR FOR BOTH
// WOULD BE THE SAME UNITS ERROR IN A DIFFERENT PLACE" per the card):
//   pose_curvature_weight_pos on c_p's D^T D block, pose_curvature_weight_rot
//   on c_phi's, D the same [+1,-2,+1] second-difference operator
//   buildImuCorrectionSystem()'s own Curv already uses. No natural sigma
//   normalization exists for a position/attitude control point the way
//   sigma_a/sigma_g exist for accel/gyro corrections, so this term is NOT
//   sigma-scaled (unlike the raw_imu basis's smoothness_weight_{acc,gyr}) --
//   the raw weight IS the precision. Both default 0.0 -- md5-inert (this
//   whole file is unreachable from any shipped path regardless).
//
// Dimensionality: c = [c_p (3*n_c); c_phi (3*n_c)], matching the card's own
// "18+6*n_c... only the meaning of the 6*n_c block changes" instruction --
// this builder returns exactly that 6*n_c-square system's own A/b, the
// c-block only (the 18-dim s-block/Pi_ss prior is unchanged from the
// raw_imu path and out of scope for this standalone builder).
// ============================================================================

namespace livo_recon
{

// One LiDAR point-to-plane observation against the pose basis, reduced to
// what buildPoseSplineCBlock() needs. raw_body_point is q_i (the body-frame
// point BEFORE any deskew warp), matching hcol_reldiff's own
// raw_body_point.cross(...) convention.
struct PoseSplineLidarObs
{
  double t = 0.0;
  V3D    raw_body_point = V3D::Zero();
  V3D    normal = V3D::Zero();
  double r = 0.0;        // current signed point-to-plane residual, at c=0
  double sigma2 = 1.0;
};

// One raw IMU sample, body frame, for the IMU-as-measurement-factor term.
struct PoseSplineImuObs
{
  double t = 0.0;
  V3D    acc = V3D::Zero();
  V3D    gyr = V3D::Zero();
};

struct PoseSplineCBlockBuild
{
  Eigen::MatrixXd A;              // 6*n_c x 6*n_c: [c_p(3n_c); c_phi(3n_c)]
  Eigen::VectorXd b;
  int    n_lidar = 0;
  int    n_imu   = 0;
  // CQ-86 item 0: the LiDAR residual loop's OWN contribution to the c-block,
  // isolated BEFORE the Tikhonov floor / smoothness prior are added (the
  // pose-arm analogue of the raw_imu arm's "A.block(c) - Lambda" isolation
  // in estimateCoupledCorrection() -- see lio_coupled.cpp's CQ-66 comment).
  // Populated only when `psd_audit_en`-gated instrumentation is requested by
  // the caller (see the `audit` param below); otherwise left empty (size 0)
  // so the normal live solve path pays nothing for this.
  Eigen::MatrixXd A_lidar_only;
  // Single-pass spread (Var = E[||X||^2] - ||E[X]||^2, Frobenius) of the
  // per-residual LiDAR Jacobian row Jrow (1 x ncol_c) across all lidar_obs --
  // the pose-arm analogue of CQ-66 item 3's phic_spread. CQ-87 item 7,
  // CORRECTING A CLAIM THIS COMMENT USED TO MAKE: "end_time" does NOT
  // guarantee this is exactly 0 the way raw_imu's own end_time phic_spread
  // is. CQ-86 item 0 MEASURED IT: 0.589624 at n_c=7 and 0.589624 at n_c=13
  // (sections/round-81) -- eight orders of magnitude from raw_imu's own
  // ~6.04e-9 (CQ-65). The reason: under end_time, only b_j/Jr/rotAt are
  // pinned to the fixed t1 -- obs.normal and obs.raw_body_point still vary
  // per residual, so Jrow = b_j(t1) * n_i^T still varies per point. This is
  // DIFFERENT from raw_imu's own Phic_pt, which carries no per-residual
  // normal at all, so ITS end_time guarantee is real and this arm's is not.
  // CONSEQUENCE: the two arms' phic_spread values are NOT the same
  // quantity -- CQ-86's own rank_cblock comparison against CQ-66 still
  // stands (both measure the SAME thing, the c-block's own information
  // rank), but a phic_spread comparison between the two arms does not.
  double phic_spread = 0.0;
};

// CQ-86 item 0: at which time is the LiDAR term's basis-weight vector (and
// therefore the rotation-Jacobian chain through Jr(phi)) evaluated for a
// given residual? "point_time" (default, matches the shipped/only behavior
// through CQ-82 round 5) uses each residual's OWN capture time obs.t --
// dr_i/dc_p[j] = b_j(obs.t) n_i^T, so the basis-weight vector varies per
// point. "end_time" instead evaluates b_j/Jr/rotAt at the FIXED scan-end
// time `end_time_t1` for every residual (only the normal n_i and the raw
// body point still vary), mirroring raw_imu's own jacobian_time_mode=
// end_time construction -- this is the direct structural test of whether
// the pose arm's rank advantage over raw_imu depends on retaining each
// point's own capture time.
enum class PoseSplineTimeMode { kPointTime, kEndTime };

// Builds the pose-basis coefficient-block normal equations for a correction
// c ON TOP OF `spline`'s CURRENT control points (spline is read-only here --
// never mutated; scratch copies are used internally for the FD Jacobian).
// gravity/bias_acc/bias_gyr are FIXED inputs (read from state_ in the live
// path; the s-block, not this c-block, owns their correction). A small fixed
// Tikhonov term (TIKHONOV_EPS below) is always added to the diagonal so A is
// PD even when every weight is 0 -- required for item 6's own correctness
// gate (LiDAR AND IMU terms both disabled) to have a well-posed solve at all.
// `time_mode`/`end_time_t1` are CQ-86 item 0's diagnostic-only additions --
// default kPointTime with end_time_t1 unused reproduces every prior call
// site's behavior exactly (md5-inert to anything not passing kEndTime).
// `audit`, if true, additionally populates A_lidar_only/phic_spread above
// (two extra passes' worth of bookkeeping, opt-in for the same reason
// psd_audit_en gates the raw_imu arm's own equivalent).
// `tikhonov_eps` -- CQ-87 item 5: now an explicit parameter, THREADED
// THROUGH rather than read from config inside this function (this file
// stays standalone/config-free, per its own header doc comment and item
// 6's correctness gate's own dependence on that). Default
// POSE_SPLINE_TIKHONOV_EPS reproduces every prior call site exactly.
//
// POST-CQ-87-REVIEW FIX (item 3): `c_current`, if non-empty (must then be
// exactly 6*n_c long), is the correction ALREADY ACCUMULATED this scan
// (coupled_c_pos_/coupled_c_rot_ flattened, same [c_p;c_phi] layout as
// `A`/`b`). The smoothness (curvature) and Tikhonov terms are a quadratic
// prior E(c) = 0.5 c^T Lambda c on the correction itself -- for a GN step
// linearised around the CURRENT c=c_current, the correct normal-equations
// contribution is `Lambda*delta_c = -Lambda*c_current` (added to `A`,`b`
// respectively), not `Lambda*delta_c = 0`. Previously the curvature/
// Tikhonov blocks were added to `A` only, never to `b` -- correct at c=0
// (the first GN iteration, where every prior call site left this
// starting) but WRONG at c=c_current!=0 (iteration 2+): the solver kept
// changing the regularizer's STIFFNESS every iteration without applying
// its corresponding RESTORING FORCE, i.e. the prior silently stopped
// being centered at 0 after iteration 1. Leaving `c_current` empty
// (default -- matches every pre-fix call site, including this file's own
// correctness-gate test) reproduces the old (c=0-only-correct) behavior
// exactly; a live per-GN-iteration caller (lio_coupled.cpp) always passes
// the real accumulated c.
//
// POST-CQ-87-REVIEW FIX (item 4): `sigma_acc`/`sigma_gyr` replace the
// previously-hardcoded SIGMA_A=0.5/SIGMA_G=0.3 floor constants -- this
// arm now solves against the SAME calibrated IMU noise the raw_imu arm
// uses (state_->varAcc()/varGyr(), see the live call site), making an A/B
// comparison between the two arms fair. Defaults (0.5/0.3) reproduce
// every prior call site's numbers exactly for any caller that doesn't
// pass real values (this file's own test).
PoseSplineCBlockBuild buildPoseSplineCBlock(
    const ScanSpline& spline,
    const std::vector<PoseSplineLidarObs>& lidar_obs,
    const std::vector<PoseSplineImuObs>& imu_obs,
    const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
    double pose_imu_weight_acc, double pose_imu_weight_gyr,
    double pose_curvature_weight_pos, double pose_curvature_weight_rot,
    PoseSplineTimeMode time_mode = PoseSplineTimeMode::kPointTime,
    double end_time_t1 = 0.0, bool audit = false,
    double tikhonov_eps = 1e-6,
    double sigma_acc = 0.5, double sigma_gyr = 0.3,
    const Eigen::VectorXd& c_current = Eigen::VectorXd());

// The fixed regularizer floor's DEFAULT value -- see
// buildPoseSplineCBlock()'s own doc comment. Small enough to be
// negligible whenever any real term is active, large enough to keep A
// invertible when every weight is 0 (item 6's gate). CQ-87 item 5: this
// remains the single source of truth for estimator/coupled/
// pose_tikhonov_eps's own shipped default (lio_coupled.cpp's
// loadParameters()) -- the config key is now what callers actually pass,
// this constant is only what it defaults to.
inline constexpr double POSE_SPLINE_TIKHONOV_EPS = 1e-6;

// ============================================================================
// CQ-87 items 1/2/3: real coupling between the pose c-block and the
// filter's own state -- "under spline_mode=pose there is no state prior
// in the solve, no posterior covariance write, and the head is still
// untied, so coupled mode is not coupled, it is a per-scan spline fit
// with no memory." This function is the fix for all three, in one place,
// because they are one mechanism: tying the head IS how the prior enters
// the solve, and the reduced system's own marginal covariance IS the
// posterior.
//
// MECHANISM CHOSEN, AND WHY (item 1's own "say which forms you
// considered" requirement): item 1 asks for this "via
// setFrozenBoundary()" -- that function constrains ScanSpline::fit()'s
// own one-time KKT solve at scan START, not the per-GN-iteration
// correction loop buildPoseSplineCBlock() rebuilds every iteration; it
// cannot be called here without re-deriving a KKT solve for a system
// that already exists in normal-equations form. Instead this function
// applies the SAME underlying identity setFrozenBoundary()/
// SplineOptions::N_FROZEN_CP already exploits: a clamped cubic B-spline's
// basis weights at u=0 are [1/6, 4/6, 1/6, 0] and sum to 1, so tying the
// first POSE_SPLINE_HEAD_TIE_CP (=3, matching N_FROZEN_CP) control
// points' CORRECTIONS together (cp0=cp1=cp2=X) forces the CORRECTION to
// p(t0) to be exactly X, bit-exact, regardless of every other control
// point -- i.e. p(t0) = p_start + X. This is a linear reduction (T^T A T)
// on the existing normal equations, not a second solve: cheaper, and it
// keeps buildPoseSplineCBlock() itself untouched and still directly
// testable by item 6's own gate.
//
// The reduced 6-dim block [delta_phi0; delta_pos0] this produces is
// EXACTLY the raw_imu arm's own s-block head variables
// (lio_coupled.cpp's coupled_delta_phi0_/coupled_delta_pos0_,
// s_vec.segment<3>(0)/segment<3>(3)) -- reused, not duplicated, so a
// caller seeds `pi_ss`/`s_vec` from the SAME Omega=state_->cov().inverse()
// the raw_imu arm already computes, in the SAME [rot;pos] order, and gets
// the SAME normal-equations prior convention (A_block = pi_ss,
// b_block -= pi_ss * s_vec -- see lio_coupled.cpp:1903/1910) applied to
// the pose arm's head. pi_ss=Zero()/s_vec=Zero() reduces the system
// without adding a prior (the head is tied but with no statistical pull),
// which is NOT the default path (item 2 wants a real prior) but is kept
// reachable for testing the reduction in isolation from state_->cov().
//
// n_c must be >= POSE_SPLINE_HEAD_TIE_CP + 1 for a nonempty free block;
// callers must guard n_c (this project's own minimum n_c is 4, so this
// is never violated in practice, but is not re-checked here -- a caller
// passing a smaller n_c gets an empty-but-well-defined reduced system).
struct PoseSplineReducedSystem
{
  Eigen::MatrixXd A;   // 6*n_free + 6 square, n_free = n_c - POSE_SPLINE_HEAD_TIE_CP
  Eigen::VectorXd b;
  int n_free = 0;
  // Column layout, for unpacking a solved delta: [0, 3*n_free) = free
  // c_p (control points POSE_SPLINE_HEAD_TIE_CP..n_c-1, in order);
  // [3*n_free, 6*n_free) = free c_phi, same order; [6*n_free, 6*n_free+3)
  // = delta_phi0; [6*n_free+3, 6*n_free+6) = delta_pos0.
};

inline constexpr int POSE_SPLINE_HEAD_TIE_CP = 3;

PoseSplineReducedSystem reducePoseSplineHeadCoupling(
    const PoseSplineCBlockBuild& build, int n_c,
    const Eigen::Matrix<double, 6, 6>& pi_ss,
    const Eigen::Matrix<double, 6, 1>& s_vec);

}  // namespace livo_recon
