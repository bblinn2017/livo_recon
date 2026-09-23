#pragma once

#include <vector>
#include <Eigen/Dense>

#include "livo_recon/utils/algo/math.h"
#include "livo_recon/utils/data/data_wrappers.h"

// ============================================================================
// PoseControlSpline -- the pose-control-point-ONLY trajectory state requested
// 2026-09-22 ("Implement the actual requested formulation as a separate
// pose-only trajectory state. Do not reuse the 9N physical-knot state for
// the new mode.").
//
// State, for N control points:
//     c = [c_p (3N); c_phi (3N)]           -- state_dimension = 6N, EXACTLY.
// There is no cp[.].vel, no delta_vel storage, no velocity_mode constraint
// machinery anywhere in this class -- velocity/angular velocity do not exist
// as variables here at all, only as functions:
//     p(t)     = sum_j b_j(t) c_p[j]
//     v(t)     = sum_j bdot_j(t) c_p[j]
//     a(t)     = sum_j bddot_j(t) c_p[j]
//     R(t)     = R_anchor * Exp(phi(t)),   phi(t) = sum_j b_j(t) c_phi[j]
//     omega_body(t) = Jr(phi(t)) * phidot(t)
//
// BASIS: the exact same uniform cubic B-spline blending functions
// ScanSpline::basisU() (spline.cpp) already uses -- NOT Catmull-Rom/Hermite.
// Verified identical coefficients (b/db/ddb below), independently
// re-typed here rather than shared, because this class's Jacobian API
// (jacobianAt() below) is new and does not exist on ScanSpline.
//
// CONVENTION (matches ScanSpline / the codebase's existing right-
// perturbative SO(3) convention, re-verified 2026-09-22):
//     R(t) = R_anchor * Exp(phi(t))    =>    Rdot = R * [omega_body]_x
//     omega_body(t) = Jr(phi(t)) * phidot(t)      (exact, not an approx.)
// Right-perturbation Jacobian of R(t) w.r.t. a control point:
//     R(t; c_phi[k] + eps e) = R_anchor Exp(phi(t) + eps b_k(t) e)
//                             ~= R(t) * Exp( eps * Jr(phi(t)) * b_k(t) * e )
//     => d(theta_bodyperturb(t))/d(c_phi[k]) = Jr(phi(t)) * b_k(t) * I3
// This is the SAME chain-through-Jr(phi) convention pose_spline_system.h's
// buildPoseSplineCBlock() already uses for its (already-analytic) LiDAR
// rotation Jacobian -- matched, not re-derived.
// ============================================================================

namespace livo_recon
{

// The 4 uniform-cubic-B-spline basis weights and their du/d2u derivatives,
// evaluated at local coordinate u in [0,1). Bit-identical to
// ScanSpline's internal basisU() (spline.cpp) -- see this file's header
// comment for why it is re-typed rather than shared.
inline void poseControlBasisU(double u, Eigen::Vector4d& b, Eigen::Vector4d& db,
                               Eigen::Vector4d& ddb)
{
  const double u2 = u * u, u3 = u2 * u, om = 1.0 - u;
  b[0]  = om * om * om / 6.0;
  b[1]  = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0;
  b[2]  = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0;
  b[3]  = u3 / 6.0;
  db[0] = -0.5 * om * om;
  db[1] = 0.5 * (3.0 * u2 - 4.0 * u);
  db[2] = 0.5 * (-3.0 * u2 + 2.0 * u + 1.0);
  db[3] = 0.5 * u2;
  ddb[0] = om;
  ddb[1] = 3.0 * u - 2.0;
  ddb[2] = -3.0 * u + 1.0;
  ddb[3] = u;
}

// The analytic Jacobian of [p(t); v(t); a(t); theta_pert(t); omega_body(t)]
// with respect to the 4 active control points at time t (segment s..s+3).
// theta_pert is the RIGHT-PERTURBATION tangent coordinate (see header
// comment) -- NOT phi(t) itself once R_anchor != I; but since these
// Jacobians are always used relative to R(t) itself (perturbing FROM the
// current trial), d(theta_pert)/dc_phi[k] = Jr(phi(t)) * b_k(t) is exact.
//
// domega/dc_phi[k]: omega_body(t) = Jr(phi(t)) * phidot(t) is a NONLINEAR
// function of phi(t) (through Jr) as well as a linear function of phidot(t).
// The full derivative is
//     d(omega)/dc_phi[k] = dJr/dphi[phidot] * b_k(t)  +  Jr(phi(t)) * bdot_k(t)
// The first term is the derivative of Jr along phidot, contracted with the
// perturbation direction b_k(t)*e_a; PoseControlSpline::domegaDcphi() below
// computes it via the standard right-Jacobian identity
//     d(Jr(phi)v)/dphi[dphi] = -0.5*[v]_x*dphi + O(|phi|) higher-order terms
// approximated to leading order (exact at phi=0, the dominant regime for
// this codebase's <15 deg per-scan chords -- CHART_MAX_PHI_RAD=1.0 rad
// documents the same regime assumption already accepted for ScanSpline).
// Validated empirically against central FD in
// scripts/test/test_pose_control_spline.cpp (unit test B) rather than
// asserted -- see that file's reported max_abs_omega_error.
struct PoseControlJac
{
  int s = -1;                     // segment index (control points s..s+3)
  Eigen::Vector4d b, db, ddb;      // basis weights + du/d2u derivatives
  double inv_delta = 0.0, inv_delta2 = 0.0;
};

class PoseControlSpline
{
public:
  // Allocates N control points uniformly spanning [t0,t1] (n_seg = N-3
  // segments, matching ScanSpline's own convention). Control points are
  // left at zero -- caller (fitHeadTail() or a direct assignment) sets them.
  void init(int N, double t0, double t1);

  int    N() const { return N_; }
  int    nSeg() const { return n_seg_; }
  double t0() const { return t0_; }
  double t1() const { return t1_; }
  double delta() const { return delta_; }

  void locate(double t, int& s, double& u) const;

  V3D posAt(double t) const;
  V3D velAt(double t) const;
  V3D accAt(double t) const;
  V3D phiAt(double t) const;
  V3D phiDotAt(double t) const;
  M3D rotAt(double t) const;
  V3D omegaBodyAt(double t) const;
  void poseAt(double t, M3D& R, V3D& p) const;

  // Builds the basis/Jacobian bookkeeping (segment index, basis weights,
  // 1/delta powers) for time t once, reused by every derivative-Jacobian
  // accessor below (dPosDcp/dVelDcp/... all take the SAME PoseControlJac).
  PoseControlJac jacobianAt(double t) const;

  // 3x3 block d(quantity(t))/d(c_p[jac.s + k]) or d(.)/d(c_phi[jac.s + k]),
  // k in [0,3]. Position/velocity/acceleration depend ONLY on c_p; theta/
  // omega depend ONLY on c_phi -- the cross blocks are exactly zero
  // (position and rotation control points are decoupled in this basis),
  // which is why this class never builds a 6x6 joint block.
  static M3D dPosDcp(const PoseControlJac& j, int k)   { return j.b[k] * M3D::Identity(); }
  static M3D dVelDcp(const PoseControlJac& j, int k)   { return (j.db[k] * j.inv_delta) * M3D::Identity(); }
  static M3D dAccDcp(const PoseControlJac& j, int k)   { return (j.ddb[k] * j.inv_delta2) * M3D::Identity(); }
  M3D dThetaDcphi(const PoseControlJac& j, int k, double t) const
  { return j.b[k] * Jr(phiAt(t)); }
  // EXACT (2026-09-22 correction, item 13): the full analytic derivative
  // of omega(t)=Jr(phi(t))*phidot(t) w.r.t. c_phi[jac.s+k], including the
  // derivative of Jr(phi) itself (previously only a leading-order-in-phi
  // approximation -- see dOmegaDcphiLeadingOrder() below, kept only as a
  // secondary cross-check). Validated against FD in
  // test_pose_control_spline.cpp (FD is the ORACLE only, never used here).
  M3D dOmegaDcphi(const PoseControlJac& j, int k, double t) const;
  // The original leading-order-in-phi approximation, kept for the
  // diagnostic comparison in the unit test (NOT used by the estimator).
  M3D dOmegaDcphiLeadingOrder(const PoseControlJac& j, int k, double t) const;

  Eigen::Matrix<double, 3, Eigen::Dynamic>       cp_p;    // 3 x N, world position
  Eigen::Matrix<double, 3, Eigen::Dynamic>       cp_phi;  // 3 x N, tangent-chart rotation
  M3D R_anchor = M3D::Identity();

private:
  int    N_ = 0, n_seg_ = 0;
  double t0_ = 0.0, t1_ = 0.0, delta_ = 0.0, inv_delta_ = 0.0;
};

// ============================================================================
// HARD head constraint (spec item 4, 2026-09-22): "Do NOT use a soft
// covariance prior to merely discourage head movement... hard-constrain the
// spline so p(t0)=p0, R(t0)=R0, v(t0)=v0, omega(t0)=omega0... zero
// optimization DOFs corresponding to head motion." Implemented as genuine
// linear EQUALITY constraint rows (a KKT-augmented solve), not elimination
// of control-point variables and not a soft prior -- this is what keeps
// raw_parameter_dimension == reduced_parameter_dimension == 6N (spec item
// 15): the state truly is 6N control-point coordinates throughout; the KKT
// multipliers are bookkeeping for the equality rows, not hidden state.
//
// R_ANCHOR CONVENTION: the caller sets spline.R_anchor = R0 (the scan-start
// attitude) BEFORE calling buildHeadConstraintRows() -- this makes the
// target phi(t0) exactly 0, so the rotation constraint is
//     phi(t0) = 0,   phidot(t0) = omega0        (since Jr(0) = I exactly)
// rather than needing Jr(phi0)^-1 for a nonzero phi0 target.
//
// TAIL IS DELIBERATELY UNCONSTRAINED (spec item 5): only 12 rows are ever
// built here (2 constraints [value, rate] x 3 axes x 2 channels [pos, rot],
// all touching only cp_p[0..2]/cp_phi[0..2] -- the basis weight/rate-weight
// of control point 3 is exactly 0 at u=0, same identity ScanSpline's own
// buildEndConstraints() exploits at BOTH ends).  p(t1)/v(t1)/R(t1)/
// omega(t1) are evaluated from the OPTIMIZED spline after solving -- never
// clamped, never read from an independently-tracked tail-velocity variable.
//
// Flattened state-vector layout used by C/delta_c below: index(k,axis) =
// 3*k+axis for c_p's block [0, 3N); 3N + 3*k+axis for c_phi's block
// [3N, 6N). This is the SAME layout every caller (process factor, LiDAR
// factor, covariance) must use when assembling A/b.
void buildHeadConstraintRows(const PoseControlSpline& spline,
                              const V3D& p0, const V3D& v0, const V3D& omega0,
                              Eigen::MatrixXd& C, Eigen::VectorXd& d);

// Solves [A C^T; C 0][delta_c; lambda] = [b; d] via LDLT (indefinite,
// symmetric) and returns delta_c (size 6N) -- the correction that,
// applied to the CURRENT trial control points, satisfies the head
// constraint rows exactly (to solver tolerance) while otherwise
// minimizing the quadratic cost A/b already encodes (LiDAR + process
// factor normal equations, assembled elsewhere). A is 6N x 6N, b is 6N,
// C is 12 x 6N, d is 12 -- exactly buildHeadConstraintRows()'s output
// re-expressed as a residual against the CURRENT trial (see that
// function's caller in the estimator for how d is built each GN
// iteration: d = target - (current value), so delta_c=0 solves a
// system already exactly at the head target).
bool solvePoseControlKkt(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                          const Eigen::MatrixXd& C, const Eigen::VectorXd& d,
                          Eigen::VectorXd& delta_c);

// ============================================================================
// HEAD ELIMINATION -- 2026-09-22 follow-up ("The head is NOT an optimization
// variable... Do NOT add a soft prior and then allow the optimizer to
// change the head... delta_x0 = 0 BY CONSTRUCTION"). This SUPERSEDES the
// KKT-augmented head treatment above for the live estimator: cp[0..2] and
// cp_phi[0..2] (the only 3 control points p(t0)/v(t0)/R(t0) depend on --
// basis weight at u=0 is [1/6,4/6,1/6,0], so cp[3.] never touches t0) are
// SOLVED ONCE from the incoming fixed head data and then HELD CONSTANT for
// the entire GN loop -- they are simply not columns in the optimization
// state at all, not "free but constrained via equality rows".
//
// Head data is [p0,v0,a0] / [phi0=0,omega0,alpha0] -- 3 equations per axis
// (value/rate/2nd-derivative at u=0) for the 3 unknowns cp[0],cp[1],cp[2]
// (resp. cp_phi[0..2]), uniquely solvable (a full-rank 3x3 system, the
// clamped-cubic-B-spline boundary identity). a0/alpha0 (acceleration/
// angular-acceleration) are NOT part of StateGroup -- default to ZERO
// (the standard "no better information" boundary assumption) unless the
// caller has a genuine estimate.
void solveHeadControlPoints(const PoseControlSpline& spline,
                             const V3D& p0, const V3D& v0, const V3D& a0,
                             const V3D& omega0, const V3D& alpha0,
                             V3D cp_p_head[3], V3D cp_phi_head[3]);

// Number of control points ELIMINATED at the head (fixed, not GN
// variables) -- always 3, named for readability at call sites.
inline constexpr int POSE_CONTROL_HEAD_FIXED_CP = 3;

// ============================================================================
// HEAD-COVARIANCE SENSITIVITY -- how p(t)/v(t)/R(t) respond to a
// perturbation of the FIXED incoming head x0=[theta0,p0,v0] itself (as
// opposed to a perturbation of any GN variable). Needed for the
// Schur-complement covariance step (pose_control_covariance.h) to build
// the head/free cross-block -- NOT part of the GN mean solve (the head
// never moves there), only the COVARIANCE propagation.
//
// POSITION is a LOCAL effect, mediated entirely through cp_p[0..2]
// (solveHeadControlPoints()'s own 3x3 M matrix): d(cp_p[k])/d(p0) =
// Minv(k,0)*I, d(cp_p[k])/d(v0) = Minv(k,1)*I (a0 is FIXED at 0, not a
// variable, so its Minv column is irrelevant here) -- chained through the
// ordinary basis weights, so dp(t)/d(p0)=dp(t)/d(v0)=0 EXACTLY for t
// beyond cp[2]'s support (segment 3 onward), a genuine locality property,
// not an approximation.
//
// ROTATION is a GLOBAL effect and has NO control-point-mediated
// component at all: cp_phi[0..2] are solved from [phi0=0 (fixed BY
// DEFINITION, since R_anchor:=R0 always), omega0, alpha0=0] -- phi0's
// target is 0 regardless of what R0 numerically is, so cp_phi[0..2]
// themselves do not depend on theta0. Instead, perturbing R0 rigidly
// perturbs R_anchor itself, which shifts R(t)=R_anchor*Exp(phi(t)) for
// EVERY t in the scan (not just near the head) via the EXACT (not
// leading-order) conjugation identity R*Exp(v)*R^T=Exp(R*v):
//     R(t;theta0+eps) = R(t;theta0) * Exp( R(t)^T R_anchor * eps )
//     => d(theta_pert(t))/d(theta0) = R(t)^T * R_anchor      (exact)
// which is nonzero for every t across the whole scan, including the tail.
struct PoseControlHeadPosSensitivity
{
  Eigen::Matrix3d Minv;   // solveHeadControlPoints()'s own per-axis 3x3 map
};
PoseControlHeadPosSensitivity poseControlHeadPosSensitivity(const PoseControlSpline& spline);
// dp(t)/d(p0) and dp(t)/d(v0), dv(t)/d(p0), dv(t)/d(v0) -- each 3x3
// (scalar-times-identity, since Minv acts identically per axis), zero for
// t beyond cp[2]'s support.
void poseControlHeadPosJacobians(const PoseControlSpline& spline,
                                  const PoseControlHeadPosSensitivity& hs, double t,
                                  M3D& dp_dp0, M3D& dp_dv0, M3D& dv_dp0, M3D& dv_dv0);

// d(theta_pert(t))/d(theta0), exact, for any t (see header comment above).
M3D poseControlHeadRotJacobian(const PoseControlSpline& spline, double t);

// ============================================================================
// NULLSPACE HEAD ELIMINATION -- 2026-09-22 correction, SUPERSEDES both the
// KKT-augmented mechanism above (12 rows including a phidot(t0)=omega0 row
// that must NOT exist -- omega isn't a StateGroup variable) and the
// control-point-FIXING mechanism above (solveHeadControlPoints(), which
// over-constrains by using 3 conditions per channel -- p0/v0/a0=0 and
// phi0/omega0/alpha0=0 -- freezing cp[0..2]/cp_phi[0..2] ENTIRELY and
// removing 9 raw DOF that should stay available to the GN solve).
//
// The correct head is EXACTLY 9 linear equality constraints on the raw 6N
// spline state, nothing more:
//     B(t0)   . c_p   = p0     (3 rows, value)
//     dB/dt(t0) . c_p = v0     (3 rows, rate)
//     B(t0)   . c_phi = 0      (3 rows, phi(t0)=0, since R_anchor:=R0)
// NO phidot(t0) row (that would fix omega(t0), which must be free to change
// as attitude control points move), NO 2nd-derivative row (that would fix
// acceleration(t0)). Both channels' constraint matrices only touch
// cp_p[0..2]/cp_phi[0..2] (basis weight of cp[3] is exactly 0 at u=0).
//
// c = c_particular + Z*eta: Z (6N x (6N-9)) is an orthonormal basis for the
// constraint nullspace (SVD-based, mirrors the EXACT deterministic-
// constraint nullspace-elimination pattern already used elsewhere in this
// codebase's own pose_knots exact_deterministic_constraint machinery --
// same technique, not reinvented), c_particular is any one solution
// satisfying the 9 constraints (minimum-norm). eta (6N-9 dim, 69 for N=13)
// is the actual free-position-and-rotation GN variable -- this is what
// "the mean solve operates directly in the nullspace of these 9
// constraints" means: eta, not c, is optimized; c is always recovered as
// c_particular + Z*eta before being read back into the spline.
struct PoseControlHeadNullspace
{
  Eigen::MatrixXd Z;              // 6N x (6N-9)
  Eigen::VectorXd c_particular;   // 6N, satisfies the 9 constraints exactly
  int rawDim() const { return static_cast<int>(Z.rows()); }
  int freeDim() const { return static_cast<int>(Z.cols()); }
};

// spline.R_anchor must already be set to R0 by the caller (phi0 target 0).
PoseControlHeadNullspace buildPoseControlHeadNullspace(
    const PoseControlSpline& spline, const V3D& p0, const V3D& v0);

// Flattens spline.cp_p/cp_phi into the SAME [c_p(3N);c_phi(3N)] layout
// buildHeadConstraintRows() used, for projecting an initial guess onto eta
// (eta0 = Z^T * (c_initial - c_particular), valid since Z has orthonormal
// columns) or reading c back out after eta changes.
Eigen::VectorXd poseControlFlatten(const PoseControlSpline& spline);
void poseControlUnflatten(const Eigen::VectorXd& c, PoseControlSpline& spline);

}  // namespace livo_recon
