#pragma once

#include "livo_recon/utils/algo/math.h"
#include "livo_recon/utils/data/data_wrappers.h"

#include <vector>

// ============================================================================
// User instruction 2026-09-21 ("do all of them", 32-item redesign): the
// pose-spline's control points should be ACTUAL PHYSICAL TRAJECTORY STATES
// at known times, not B-spline basis coefficients (cp,j != p(tj) in
// pose_spline_system.h's own formulation). This file is that: a knot j
// directly holds p_j=p(t_j), R_j=R(t_j), v_j=v(t_j), and Cov[theta_j,p_j,v_j]
// -- items 1-4.
//
// SCOPE OF THIS FIRST IMPLEMENTATION (matches the user's own item 31,
// "make the first test extremely controlled... first prove physical knot
// representation + IMU factors + LiDAR works, then add smoothness"):
//   - IMU factors are the standard discrete-time process-noise factor
//     between ADJACENT knots (item 16: "replaces the current
//     fdImuJacobianCol() approach" -- there is no separate per-sample
//     IMU-as-measurement residual in this formulation; the raw IMU stream
//     between t_j and t_{j+1} IS the factor connecting knot j to knot j+1).
//   - Bias/gravity are held FIXED at their scan-start (ESIKF) values (item
//     22: "hold them fixed... exactly as the current pose arm does. Once
//     the pose spline works, you can add joint bias updates") -- so the
//     knot state is [theta,p,v] (9-dim), matching StateGroup::idxR()/
//     idxP()/idxV()'s own [0,3,6) layout exactly (this is deliberate: the
//     9x9 F/Q construction below is a MARGINAL BLOCK of the exact same
//     formula ImuProc::propagate() uses -- see buildImuStep9x9()'s own doc
//     comment -- restricted to [R,P,V], since bias/gravity contribute no
//     OWN uncertainty this round by construction).
//   - Rotation interpolation between knots is the geodesic R(t) =
//     R_j*Exp(u*Log(R_j^T R_{j+1})) -- exact at both endpoints, analytic
//     angular velocity via Jr(u*phi_rel)*phi_rel/dt. Position/velocity use
//     the standard cubic Hermite (exact at both endpoints in position AND
//     velocity, analytic acceleration) -- item 3's "interpolating
//     Hermite-like representation".
// ============================================================================

namespace livo_recon
{

// One physical trajectory knot -- item 4/25 (position+attitude+velocity,
// the bias/gravity block held fixed this round per item 22).
struct PoseKnot
{
  double t = 0.0;
  V3D    pos = V3D::Zero();
  M3D    rot = M3D::Identity();
  V3D    vel = V3D::Zero();
  // [theta,p,v] order, matching StateGroup::idxR()=0/idxP()=3/idxV()=6 --
  // items 4/12/13: this IS Cov(x_j,x_j) directly, no basis-coefficient
  // translation needed, unlike pose_spline_system.h's cp_p_/cp_phi_.
  Eigen::Matrix<double, 9, 9> P = Eigen::Matrix<double, 9, 9>::Zero();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Item 5's boxed formula, P_{j+1}^- = F_j P_j^+ F_j^T + Q_j, for ONE raw IMU
// sample interval -- a marginal [theta,p,v] 9x9 BLOCK of the EXACT SAME
// F_x/cov_w construction ImuProc::propagate() builds (imu_processing.cpp,
// the acc_avr_skew/Exp_f/second_order terms), copied rather than a live
// shared call site (imu_processing.cpp is the primary/default estimator's
// hot path for every mode; a live refactor there re-risks the whole
// existing md5 gate for a knob only this new mode uses -- item 5 itself
// permits either "call... or factor it into a shared routine", and this
// takes the lower-blast-radius of the two). Bias/gravity ROWS/COLUMNS are
// simply absent (item 22: held fixed, contribute no own uncertainty here)
// -- everything else (rotation/position/velocity terms, second_order
// gating, q_alpha_* scaling) matches imu_processing.cpp line-for-line.
void buildImuStep9x9(const M3D& rot_imu, const V3D& acc_avr, const V3D& angvel_avr,
                     double dt, const V3D& var_acc, const V3D& var_gyr,
                     double q_alpha_acc, double q_alpha_gyr, bool second_order,
                     Eigen::Matrix<double, 9, 9>& F9, Eigen::Matrix<double, 9, 9>& Q9);

class PoseKnotSpline
{
public:
  // Item 3: initializes knot times t_j = t0 + j*dt (dt=(t1-t0)/(n_knots-1))
  // and p_j/R_j/v_j from the EXISTING IMU-propagated trajectory (mg.poses,
  // linear/SLERP-interpolated onto each t_j -- the ESIKF's own raw chain
  // output, unchanged). Item 18: knot 0's covariance is seeded from the
  // caller's P0 (state_->cov()'s own [theta,p,v] block at scan start), NOT
  // a hard clamp -- see estimateCoupledPoseKnotSpline()'s own prior-factor
  // construction for how that stays a SOFT prior, not an equality
  // constraint (the physical-knot formulation makes the old clamped-
  // B-spline head-tie machinery unnecessary: knot 0 IS p_0 directly, no
  // basis-coefficient gap to close). Item 5/6: knots 1..N-1's covariance
  // is causally forward-propagated ONCE here (P_j^-, pre-LiDAR) by walking
  // `imu_raw` sample-by-sample between each pair of knot times, using
  // buildImuStep9x9() -- the LiDAR-updated P_j^+ is the solver's own job,
  // not this class's.
  bool init(const std::vector<Pose6D>& imu_poses, double t0, double t1, int n_knots,
            const V3D& p0, const M3D& R0, const V3D& v0,
            const Eigen::Matrix<double, 9, 9>& P0,
            const std::vector<ImuSample>& imu_raw,
            const V3D& bias_acc, const V3D& bias_gyr, const V3D& gravity,
            double q_alpha_acc, double q_alpha_gyr, const V3D& var_acc, const V3D& var_gyr,
            bool second_order);

  int nKnots() const { return static_cast<int>(knots_.size()); }
  const PoseKnot& knot(int j) const { return knots_[j]; }
  PoseKnot& knotMut(int j) { return knots_[j]; }
  bool valid() const { return valid_; }

  // Item 15/26: continuous interpolation, used by both the LiDAR factor
  // (point_time: H_i = H(t_i), item 8/9) and diagnostics/logging (item 27).
  V3D positionAt(double t) const;
  V3D velocityAt(double t) const;
  V3D accelerationAt(double t) const;
  M3D rotationAt(double t) const;
  V3D angularVelocityAt(double t) const;

  // Which knot pair (j,j+1) a query time falls in, and the normalized
  // fraction u=(t-t_j)/(t_{j+1}-t_j) in [0,1] -- item 8's own "interpolate
  // local trajectory state" step. Clamped to [0,N-2] so a time outside
  // [t0,t1] still returns a well-defined (extrapolating) bracket rather
  // than an out-of-range index.
  void bracket(double t, int& j, double& u) const;

  // Per-segment (knot j -> knot j+1) accumulated state-transition/process-
  // noise, computed ONCE in init() from the INITIAL IMU-propagated chain --
  // item 5/6's own process factor, cached here so estimateCoupledPoseKnot-
  // Spline() (lio_coupled.cpp) doesn't re-walk raw IMU samples every GN
  // iteration. DELIBERATE SIMPLIFICATION (named, not hidden): item 11 asks
  // for residuals/Jacobians to be recomputed "around the current spline
  // each iteration" -- this solver DOES recompute the process-factor
  // RESIDUAL (x_{j+1}-F9*x_j) fresh every iteration from the current
  // accumulated knot deltas, but reuses these same F9/Q9 matrices
  // (evaluated once, at the initial/frozen IMU-chain rotation) rather than
  // re-deriving a fresh per-raw-sample F9/Q9 chain every GN iteration --
  // that full re-linearization would need re-walking every raw IMU sample
  // in every segment on every iteration, a materially larger cost not
  // taken on for this first implementation.
  const Eigen::Matrix<double, 9, 9>& segF9(int j) const { return seg_F9_[j]; }
  const Eigen::Matrix<double, 9, 9>& segQ9(int j) const { return seg_Q9_[j]; }

private:
  std::vector<PoseKnot> knots_;
  std::vector<Eigen::Matrix<double, 9, 9>> seg_F9_, seg_Q9_;
  bool valid_ = false;
};

}  // namespace livo_recon
