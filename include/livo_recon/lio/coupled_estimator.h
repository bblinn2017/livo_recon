#pragma once

#include <vector>
#include <Eigen/Dense>

#include "livo_recon/utils/algo/math.h"
#include "livo_recon/utils/data/data_wrappers.h"

// ============================================================================
// CQ-44: THE COUPLED ESTIMATOR.
//
// delta_a(t) = sum_j beta_j(t) c^a_j,  delta_omega(t) = sum_j beta_j(t) c^omega_j
// -- a small basis of MEASUREMENT corrections over one scan window [t0,t1],
// same uniform-cubic-B-spline basis shape as ScanSpline's own (see basisU(),
// spline.cpp), but parameterising the IMU correction directly rather than a
// position spline, and NOT tied to ScanSpline::n_cp/control_point_hz --
// round-64 already measured that knob buys nothing (40-130Hz, flat ATE) and
// item 2 of the card explicitly says not to reuse it.
//
// propagateCoupled() re-integrates ONE scan's already-computed raw pose
// chain (LioProc::processLIO()'s own mg.poses, i.e. ImuProc::propagate()'s
// bias-only output) with these two additive corrections, and accumulates
// the CAUSAL CUMULATIVE Jacobian of every intermediate/endpoint state to
// every coefficient -- see the .cpp for the derivation. At c=0 this
// reproduces the raw chain exactly (position/velocity/rotation identical to
// machine precision); that identity is this module's own cheapest
// correctness check and is verified in CoupledEstimator's own build/test.
//
// SCOPE, STATED PLAINLY (not silently assumed): the Jacobian recursion below
// reuses the SAME first-order linearisation ImuProc::propagate()'s own F_x
// already uses for its covariance propagation (Exp_f^T for rotation,
// -R*dt/-Eye3d*dt for the bias blocks) -- it is not a higher-order exact
// derivative of the nonlinear propagation map, and does not claim to be.
// Where the existing F_x omits a term (there is no F_x(idxP,idxBA) block in
// imu_processing.cpp -- grepped, confirmed absent), this module omits the
// analogous direct-position accel-correction term too, for consistency with
// the linearisation this project already ships and trusts for its own P
// propagation, not as a new simplification introduced here.
// ============================================================================

namespace livo_recon
{

// Uniform cubic B-spline basis weight of control point `j` (0..n_c-1) at
// time `t`, over window [t0,t1] with n_c control points (n_seg = n_c-3
// segments, n_c >= 4). Zero outside j's 4-wide local support. Independent of
// ScanSpline's own basisU()/basisAt() (which are private to spline.cpp) --
// duplicated rather than exposed, since the two bases parameterise
// different physical quantities and are deliberately allowed to diverge in
// resolution (see the file header).
double basisWeight(int j, int n_c, double t0, double t1, double t);

// One scan's corrected trajectory + the causal-cumulative sensitivity of
// every intermediate/endpoint (rotation-tangent, position, velocity) state
// to the 2*3*n_c correction coefficients.
struct CoupledPropagation
{
  // Same shape/semantics as MeasureGroup::poses (ImuProc::propagate()'s own
  // output): poses[k].t/dt/rot/pos/vel are the state AT HEAD TIME of
  // segment k under THIS correction; acc_head/acc_tail are the CORRECTED
  // world-frame accelerations bracketing the segment (i.e. directly
  // reusable by deskewPoints(), which expects exactly this shape).
  std::vector<Pose6D> poses;
  M3D rot1 = M3D::Identity();
  V3D pos1 = V3D::Zero();
  V3D vel1 = V3D::Zero();

  // phi_head[k]: 9 x (6*n_c) sensitivity of (phi_R, p, v) -- phi_R the
  // LEFT-tangent rotation error, i.e. d(Log(R_true^T R_nominal)) under a
  // small perturbation of c -- AT poses[k].t (state BEFORE segment k is
  // applied) to the coefficients, column order [c_acc_0..c_acc_{n_c-1} (3
  // each), c_gyr_0..c_gyr_{n_c-1} (3 each)]. phi_head.size() == poses.size()
  // + 1; the last entry is the sensitivity of (rot1,pos1,vel1) at t1.
  std::vector<Eigen::Matrix<double, 9, Eigen::Dynamic>> phi_head;
};

// Recompute the scan from (rot0,pos0,vel0) at raw_poses.front().t through to
// scan_end_time, given `raw_poses` (the UNCORRECTED chain from
// ImuProc::propagate(), providing each segment's own head/tail rotation --
// see the .cpp for how the original body-frame accel measurement is
// recovered from it) and `raw_rot1` (the raw chain's own tail-most
// rotation, needed for the LAST segment's tail term -- LioProc's own
// state_propagat_.rot(), captured right after ImuProc::propagate() and
// before any correction).
void propagateCoupled(const std::vector<Pose6D>& raw_poses,
                      const M3D& raw_rot1,
                      double scan_end_time,
                      const M3D& rot0, const V3D& pos0, const V3D& vel0,
                      const V3D& gravity,
                      const std::vector<V3D>& c_acc, const std::vector<V3D>& c_gyr,
                      int n_c, CoupledPropagation& out);

// Linear interpolation of phi_head at an arbitrary time t (used to build the
// per-point residual Jacobian dr/dc -- a point's own capture time almost
// never lands exactly on a pose sample). Piecewise-linear between the two
// bracketing phi_head entries; matches the same bracket-and-alpha the
// existing deskewPoints() lambda already uses for acc_world interpolation
// (deskew.cpp), so the two never disagree about which segment a point time
// falls in.
Eigen::Matrix<double, 9, Eigen::Dynamic> interpolatePhi(
    const CoupledPropagation& prop, double t);

}  // namespace livo_recon
