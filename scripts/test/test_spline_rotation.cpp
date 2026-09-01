// Moving-axis rotation fixture -- the test the original suite did not have.
//
// test_spline.cpp's rotation truth was Exp(axis * theta(t)) with a FIXED axis.
// That is exactly the case where a cubic spline in a single tangent chart is
// an EXACT representation, so it could not distinguish the tangent
// parameterisation from the cumulative SO(3) one. This fixture uses a genuine
// moving axis:
//
//     R(t) = Exp(e_z * alpha(t)) * Exp(e_x * beta(t))
//     omega_body = Exp(e_x beta)^T e_z alphadot  +  e_x betadot     (exact)
//
// and sweeps the total rotation swept per scan, so the tangent mode's error is
// reported as a function of the quantity the production log already measures
// (rot_chord_deg) rather than asserted.
#include "livo_recon/lio/spline.h"
#include <cstdio>
#include <vector>
using namespace livo_recon;

struct Truth {
  double A, B, c1, c2, w1, w2;
  double alpha(double t)    const { return A*std::sin(w1*t) + c1*t; }
  double alphadot(double t) const { return A*w1*std::cos(w1*t) + c1; }
  double beta(double t)     const { return B*std::cos(w2*t) + c2*t; }
  double betadot(double t)  const { return -B*w2*std::sin(w2*t) + c2; }
  M3D rot(double t) const {
    return Exp(V3D(0,0,1) * alpha(t)) * Exp(V3D(1,0,0) * beta(t));
  }
  V3D omega(double t) const {
    return Exp(V3D(1,0,0) * beta(t)).transpose() * V3D(0,0,1) * alphadot(t)
         + V3D(1,0,0) * betadot(t);
  }
};

static std::vector<Pose6D> makePoses(const Truth& T, double t0, double t1, int n) {
  std::vector<Pose6D> ps;
  for (int i = 0; i < n; ++i) {
    double t = t0 + (t1 - t0) * i / double(n - 1);
    Pose6D p{}; p.t = t; p.rot = T.rot(t); p.pos = V3D(0.3*t, -0.2*t, 0.05*t);
    p.dt = (t1 - t0) / (n - 1);
    ps.push_back(p);
  }
  return ps;
}

static void run(const char* label, double scale, int n_cp) {
  const double t0 = 0.0, t1 = 0.1;
  Truth T{0.35*scale, 0.28*scale, 0.8*scale, 0.6*scale, 24.0, 31.0};
  auto poses = makePoses(T, t0, t1, 40);

  double chord = Log(T.rot(t0).transpose() * T.rot(t1)).norm() * 180.0 / M_PI;

  double res[2][3];
  const char* modes[2] = {"tangent", "cumulative"};
  for (int m = 0; m < 2; ++m) {
    SplineOptions o; o.n_control_points = n_cp; o.rot_mode = modes[m];
    ScanSpline s;
    if (!s.fit(poses, t0, t1, o)) { printf("  %s fit FAILED\n", modes[m]); return; }
    double max_r = 0, max_w = 0;
    for (double t = t0; t <= t1 + 1e-12; t += 0.001) {
      max_r = std::max(max_r, Log(T.rot(t).transpose() * s.rotAt(t)).norm());
      max_w = std::max(max_w, (s.omegaBodyAt(t) - T.omega(t)).norm());
    }
    res[m][0] = max_r; res[m][1] = max_w; res[m][2] = s.fitResidualRot();
  }
  printf("  %-10s chord %6.2f deg | rot err  tan %.3e  cum %.3e  (%5.1fx) "
         "| omega err  tan %.3e  cum %.3e  (%5.1fx)\n",
         label, chord,
         res[0][0], res[1][0], res[1][0] > 0 ? res[0][0]/res[1][0] : 0.0,
         res[0][1], res[1][1], res[1][1] > 0 ? res[0][1]/res[1][1] : 0.0);
}

int main() {
  printf("Moving-axis rotation: tangent chart vs cumulative SO(3), n_cp=8\n");
  printf("  (eee_01 measured max rot_chord_deg ~6 deg in SP-4a)\n\n");
  for (double s : {0.25, 0.5, 1.0, 2.0, 4.0, 8.0}) {
    char buf[32]; snprintf(buf, sizeof buf, "x%.2f", s);
    run(buf, s, 8);
  }
  printf("\nSame sweep at n_cp=16:\n");
  for (double s : {0.25, 1.0, 4.0, 8.0}) {
    char buf[32]; snprintf(buf, sizeof buf, "x%.2f", s);
    run(buf, s, 16);
  }
  return 0;
}
