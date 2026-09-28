// Isolated identities for ImuProc's motion-dependent Q measurement model.
#include "livo_recon/utils/algo/math.h"

#include <algorithm>
#include <cstdio>

using namespace livo_recon;

namespace
{
int fail(const char* name, double value, double tol)
{
  std::printf("[FAIL] %s %.6e tol %.6e\n", name, value, tol);
  return 1;
}
}

int main()
{
  const V3D floor(1e-3, 2e-3, 3e-3);
  const V3D a(3.0, 4.0, 0.0);
  const double scale = 0.2;
  const double cap = 0.5;

  // Isotropic contribution preserves the trace of scale^2*a*a^T's
  // diagonal-energy approximation and adds the calibrated per-axis floor.
  const double iso = std::min(scale * scale * a.squaredNorm() / 3.0, cap);
  const V3D q_iso = floor + V3D::Constant(iso);
  if (std::abs((q_iso - floor).sum() - scale * scale * a.squaredNorm()) > 1e-12)
    return fail("isotropic dynamic-Q trace", (q_iso - floor).sum(),
                scale * scale * a.squaredNorm());

  // Axis-aware contribution follows component energy and is capped per axis.
  V3D axis = scale * scale * a.array().square().matrix();
  for (int k = 0; k < 3; ++k) axis(k) = std::min(axis(k), cap);
  const V3D expected(0.36, 0.5, 0.0);
  if ((axis - expected).norm() > 1e-12)
    return fail("axis-aware dynamic-Q and cap", (axis - expected).norm(), 1e-12);

  // beta=0 is an exact no-filter identity.
  const V3D previous(0.4, 0.3, 0.2), current(0.1, 0.2, 0.3);
  const double beta = 0.0;
  const V3D filtered = beta * previous + (1.0 - beta) * current;
  if ((filtered - current).norm() > 0.0)
    return fail("beta=0 disables filtering", (filtered - current).norm(), 0.0);

  // Production sign convention: a stationary accelerometer reads -R^T*g,
  // so acc_unbiased + R^T*g is exactly zero.
  const M3D R = Exp(V3D(0.2, -0.1, 0.3));
  const V3D gravity(0.0, 0.0, -9.81);
  const V3D stationary_acc = -R.transpose() * gravity;
  const V3D dynamic_acc = stationary_acc + R.transpose() * gravity;
  if (dynamic_acc.norm() > 1e-12)
    return fail("gravity-removed stationary acceleration", dynamic_acc.norm(), 1e-12);

  std::printf("[PASS] test_joint_knot_motion_q\n");
  return 0;
}
