#pragma once

// ============================================================================
// The optimization-variable layout for the pose-control-point estimator.
//
// 2026-09-22 correction: the head is now handled by TRUE NULLSPACE
// ELIMINATION over the RAW 6N control-point space (see
// buildPoseControlHeadNullspace() in pose_control_spline.h), not by
// removing cp[0..2]/cp_phi[0..2] as columns here. This layout's `fix_head`
// flag therefore defaults to FALSE: colPos()/colPhi() return ordinary raw
// indices for EVERY control point (including k<3) -- factors are built
// against the FULL RAW 78-dim control-point space (+9 for sT, N=13), and
// the caller (the estimator) projects the resulting normal equations
// through Z (the nullspace basis) to get the true 69-dim eta system. This
// is what lets LiDAR/process factors touching cp[0..2] contribute their
// information to eta's own nullspace-basis columns automatically, via the
// SAME projection, rather than needing separate head-block bookkeeping in
// every factor.
//
// `fix_head=true` reproduces the OLD (superseded) behavior for reference/
// comparison only -- not used by the live estimator.
//
//   sT: the free non-trajectory tail StateGroup components actually
//       enabled by config -- bg/ba/g, in that fixed order, each present
//       iff its own StateGroup flag (est_bg_/est_ba_/est_gravity_) says
//       so. Mirrors StateGroup::idxBG()/idxBA()/idxG()'s own convention
//       (a -1/absent sentinel when disabled) rather than hardcoding a
//       tail dimension.
// ============================================================================

namespace livo_recon
{

struct PoseControlFreeLayout
{
  int N = 0;                       // total control points (spline's own N)
  bool has_bg = true, has_ba = true, has_g = true;
  bool fix_head = false;           // see header comment -- default is the
                                    // CURRENT (nullspace-elimination-based)
                                    // behavior; true reproduces the
                                    // superseded column-skipping scheme.

  int nFreeCp() const { return fix_head ? N - 3 : N; }
  int dimCFree() const { return 6 * nFreeCp(); }
  int dimST() const { return (has_bg ? 3 : 0) + (has_ba ? 3 : 0) + (has_g ? 3 : 0); }
  int dim() const { return dimCFree() + dimST(); }

  // -1 only when fix_head=true and k<3 (the superseded scheme). Otherwise
  // the column offset of control point k's first axis in the flattened
  // [c_p(3*nFreeCp); c_phi(3*nFreeCp)] block (before the sT offset below).
  int colPos(int k) const { return (fix_head && k < 3) ? -1 : 3 * (fix_head ? k - 3 : k); }
  int colPhi(int k) const { return (fix_head && k < 3) ? -1 : 3 * nFreeCp() + 3 * (fix_head ? k - 3 : k); }

  // sT column offsets (relative to dimCFree()), -1 if that component is
  // disabled by config. Order: bg, ba, g -- matching StateGroup's own
  // idxBG()<idxBA()<idxG() packing convention.
  int colBG() const { return has_bg ? dimCFree() : -1; }
  int colBA() const { return has_ba ? dimCFree() + (has_bg ? 3 : 0) : -1; }
  int colG()  const { return has_g  ? dimCFree() + (has_bg ? 3 : 0) + (has_ba ? 3 : 0) : -1; }
};

}  // namespace livo_recon
