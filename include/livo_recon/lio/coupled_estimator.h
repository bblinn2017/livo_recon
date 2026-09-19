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
// item 2 of the card explicitly says not to reuse it, since it's a separate
// resolution knob for a differently-purposed basis (a measurement CORRECTION
// over one scan, not a position/orientation trajectory). NOTE: an earlier
// version of this comment cited round-64's finding that control_point_hz
// "buys nothing" (flat ATE, 40-130Hz) as supporting evidence -- that
// measurement predates the CQ-21 valid_ bug fix that made ScanSpline's own
// mechanism actually take effect, so it does not establish anything about
// resolution sensitivity post-fix and should not be treated as a validated
// reason on its own. n_c's independence from control_point_hz is still the
// right design (different basis, different purpose), just not for that
// reason.
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

  // CQ-44 items 3c/3d, REVISED 2026-09-19 (item 3e(v)/3f bug 2 fix):
  // phi_x_head[k], 9x18, the SAME (phi_R,p,v) state's sensitivity to the
  // FULL delta_x(t0) = [delta_phi0, delta_p0, delta_v0, delta_bg, delta_ba,
  // delta_g] (column blocks of 3, in that order). Item 3c originally held
  // delta_phi(t0)/delta_p(t0) at exactly zero (correcting this scan's OWN
  // start pose retro-corrects the previous scan's already-committed,
  // already-map-built-from end pose) -- but item 3e/3f found that omitting
  // their COLUMNS here (not just their mean) also removes
  // Phi_pose*P_pose(t0)*Phi_pose^T from the covariance propagation, so every
  // scan discarded ALL accumulated pose uncertainty (the P-collapse). The
  // fix solves the full 18 in BOTH mean and covariance -- a one-scan
  // fixed-lag smoother; the previous scan's published pose is genuinely
  // revised by delta_phi0/delta_p0's converged value (named consequence,
  // not silently absorbed -- the map already built from the old pose is NOT
  // retroactively updated). Seeded at t0 with phi_R<-delta_phi0,
  // p<-delta_p0 and v<-delta_v0 each as identity (their own initial
  // condition, everything else's t0 sensitivity is zero), then accumulated
  // by the SAME Fx recursion phi_head uses, plus this segment's OWN direct
  // bias/gravity sensitivity (the same R<-bg/V<-ba/V<-g/P<-g blocks
  // ImuProc::propagate()'s own F_x already carries for covariance
  // propagation, imu_processing.cpp:188-206 -- P<-ba is omitted for the
  // identical reason phi_head's own G omits any direct accel-correction
  // position term, see the file header). Same length/indexing as phi_head.
  std::vector<Eigen::Matrix<double, 9, 18>> phi_x_head;
};

// Recompute the scan from (rot0,pos0,vel0) at raw_poses.front().t through to
// scan_end_time, given `raw_poses` (the UNCORRECTED chain from
// ImuProc::propagate(), providing each segment's own head/tail rotation --
// see the .cpp for how the original body-frame accel measurement is
// recovered from it) and `raw_rot1` (the raw chain's own tail-most
// rotation, needed for the LAST segment's tail term -- LioProc's own
// state_propagat_.rot(), captured right after ImuProc::propagate() and
// before any correction). `delta_bg`/`delta_ba` (CQ-44 items 3c/3d) are this
// GN iteration's own accumulated bias correction ON TOP OF the bias already
// baked into raw_poses by ImuProc::propagate() -- i.e. subtracted from the
// raw chain's own (already-bias-corrected) body-frame measurements, exactly
// as a further "-delta_bg"/"-delta_ba" would if the bias state itself had
// been updated by this amount before propagate() ran; `vel0`/`gravity`
// likewise already have this iteration's delta_v0/delta_g folded in by the
// caller (they are ordinary initial-condition/parameter inputs, no separate
// "delta" plumbing needed for them). AS OF item 3e(v)/3f bug 2, `rot0`/`pos0`
// ALSO already have this iteration's own accumulated delta_phi0/delta_p0
// folded in by the caller (rot0 = raw_rot0 * Exp(delta_phi0), matching
// StateGroup::applyDelta()'s own right-multiplicative body-frame convention;
// pos0 = raw_pos0 + delta_p0) -- the pose is no longer held fixed at the raw
// chain's own t0 value.
// `gravity0` (CQ-44 item 3f, BUG 3) is the FIXED, pre-scan gravity that
// raw_poses' own acc_head/acc_tail (Pose6D::acc_head is world-frame,
// = R_head*a_body + g -- see data_wrappers.h) were originally built with
// (i.e. LioProc's coupled_g0_pre_, the snapshot taken BEFORE this scan's own
// delta_g accumulates) -- used ONLY to strip acc_head/acc_tail back to the
// body-frame measurement. `gravity` (this iteration's g0_pre + delta_g) is
// used ONLY to reconstitute the corrected world-frame accel. Stripping with
// the CORRECTED gravity instead of the ORIGINAL one (the bug) makes
// delta_g cancel to first order between the strip and the reconstitute
// steps, leaving the trajectory with no real sensitivity to it even though
// Gx's own P<-g/V<-g blocks claim it does.
void propagateCoupled(const std::vector<Pose6D>& raw_poses,
                      const M3D& raw_rot1,
                      double scan_end_time,
                      const M3D& rot0, const V3D& pos0, const V3D& vel0,
                      const V3D& gravity, const V3D& gravity0,
                      const V3D& delta_bg, const V3D& delta_ba,
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

// Same interpolation, for phi_x_head (items 3c/3d, 18 columns per 3e(v)/3f).
Eigen::Matrix<double, 9, 18> interpolatePhiX(
    const CoupledPropagation& prop, double t);

}  // namespace livo_recon
