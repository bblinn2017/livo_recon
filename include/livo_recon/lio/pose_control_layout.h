#pragma once

// Layout of the raw spline control points and optional tail state variables.
// The live pose-control estimator projects raw control-point variables through
// its fixed-head nullspace before optimization.

namespace livo_recon
{

struct PoseControlFreeLayout
{
  int N = 0;                       // total control points (spline's own N)
  bool has_bg = true, has_ba = true, has_g = true;
  bool fix_head = false;

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
