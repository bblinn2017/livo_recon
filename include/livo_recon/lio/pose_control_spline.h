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
  // See this class's header comment -- leading-order term only, validated
  // against FD in the unit test, not assumed exact.
  M3D dOmegaDcphi(const PoseControlJac& j, int k, double t) const;

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

}  // namespace livo_recon
