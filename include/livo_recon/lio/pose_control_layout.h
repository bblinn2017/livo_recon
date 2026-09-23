#pragma once

// ============================================================================
// The REDUCED optimization-variable layout for the pose-control-point
// estimator, per the 2026-09-22 follow-up: z = [c_free; sT].
//   c_free: control points [POSE_CONTROL_HEAD_FIXED_CP .. N-1] (position AND
//           rotation) -- cp[0..2]/cp_phi[0..2] are FIXED (see
//           solveHeadControlPoints()) and are simply not columns here at all.
//   sT:     the free non-trajectory tail StateGroup components actually
//           enabled by config -- bg/ba/g, in that fixed order, each present
//           iff its own StateGroup flag (est_bg_/est_ba_/est_gravity_) says
//           so. Mirrors StateGroup::idxBG()/idxBA()/idxG()'s own convention
//           (a -1/absent sentinel when disabled) rather than hardcoding a
//           tail dimension.
// ============================================================================

namespace livo_recon
{

struct PoseControlFreeLayout
{
  int N = 0;                       // total control points (spline's own N)
  bool has_bg = true, has_ba = true, has_g = true;

  int nFreeCp() const { return N - 3; }               // POSE_CONTROL_HEAD_FIXED_CP == 3
  int dimCFree() const { return 6 * nFreeCp(); }        // c_p_free + c_phi_free
  int dimST() const { return (has_bg ? 3 : 0) + (has_ba ? 3 : 0) + (has_g ? 3 : 0); }
  int dim() const { return dimCFree() + dimST(); }

  // -1 if control point k (0-indexed, absolute) is FIXED (part of the head,
  // k<3) -- no column exists for it. Otherwise the column offset of its
  // FIRST axis in the flattened [c_p_free(3*nFreeCp); c_phi_free(3*nFreeCp)]
  // block (before the sT offset below).
  int colPos(int k) const { return (k < 3) ? -1 : 3 * (k - 3); }
  int colPhi(int k) const { return (k < 3) ? -1 : 3 * nFreeCp() + 3 * (k - 3); }

  // sT column offsets (relative to dimCFree()), -1 if that component is
  // disabled by config. Order: bg, ba, g -- matching StateGroup's own
  // idxBG()<idxBA()<idxG() packing convention.
  int colBG() const { return has_bg ? dimCFree() : -1; }
  int colBA() const { return has_ba ? dimCFree() + (has_bg ? 3 : 0) : -1; }
  int colG()  const { return has_g  ? dimCFree() + (has_bg ? 3 : 0) + (has_ba ? 3 : 0) : -1; }
};

}  // namespace livo_recon
