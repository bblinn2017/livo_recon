// Runs VibrationCanceller (include/livo_recon/utils/imu/vibration_canceller.h, the code livo_recon runs) over an IMU
// text dump (scripts/vibration/dump_imu.py), offline, no ROS. Defaults are config/mid360.yaml's cbk/imu/vibration.
// build: g++ -O2 -std=c++17 -I include -I /usr/include/eigen3 scripts/vibration/vib_apply.cpp -o vib_apply
// usage: vib_apply IN OUT [enable=0|1] [main=HZ] [offset=HZ] [harmonics=N] [tau=S] [delay=S] [fs=HZ] [noextra=1]
//        [apply_from=S]   (S after the first sample: learn only before it, like CbkProc before calib_done)
#include "livo_recon/utils/imu/vibration_canceller.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char** argv)
{
  if (argc < 3) { std::fprintf(stderr, "usage: vib_apply IN OUT [key=value ...]\n"); return 1; }
  livo_recon::VibrationCancellerOptions o;
  o.enable = true; o.sample_rate_hz = 199.998; o.main_hz = 28.025; o.sideband_offset_hz = 0.790; o.harmonics = 2;
  o.extra_hz = {13.400, 18.124};
  double apply_from = 0.0;
  for (int i = 3; i < argc; ++i)
  {
    const std::string kv = argv[i];
    const auto p = kv.find('=');
    if (p == std::string::npos) { std::fprintf(stderr, "expected key=value: %s\n", argv[i]); return 1; }
    const std::string k = kv.substr(0, p);
    const double v = std::atof(kv.c_str() + p + 1);
    if (k == "enable") o.enable = v != 0;
    else if (k == "main") o.main_hz = v;
    else if (k == "offset") o.sideband_offset_hz = v;
    else if (k == "harmonics") o.harmonics = static_cast<int>(v);
    else if (k == "tau") o.tau_fit_s = v;
    else if (k == "delay") o.sideband_delay_s = v;
    else if (k == "fs") o.sample_rate_hz = v;
    else if (k == "noextra") o.extra_hz.clear();
    else if (k == "apply_from") apply_from = v;
    else { std::fprintf(stderr, "unknown key %s\n", k.c_str()); return 1; }
  }
  livo_recon::VibrationCanceller vc(o);
  std::ifstream in(argv[1]);
  FILE* out = std::fopen(argv[2], "w");
  if (!in || !out) { std::fprintf(stderr, "cannot open %s or %s\n", argv[1], argv[2]); return 1; }
  const double G = 9.81;   // CbkProc: cbk/imu/acc_type "g" -> *G_m_s2 before the canceller
  std::string line;
  double t_first = -1;
  while (std::getline(in, line))
  {
    std::istringstream ss(line);
    double t, w[3], a[3];
    if (!(ss >> t >> w[0] >> w[1] >> w[2] >> a[0] >> a[1] >> a[2])) continue;
    if (t_first < 0) t_first = t;
    Eigen::Vector3d acc(a[0] * G, a[1] * G, a[2] * G), gyro(w[0], w[1], w[2]);
    vc.process(t, acc, gyro, t - t_first >= apply_from);
    std::fprintf(out, "%.9f %.9f %.9f %.9f %.9f %.9f %.9f\n", t, gyro(0), gyro(1), gyro(2), acc(0) / G, acc(1) / G,
                 acc(2) / G);
  }
  std::fclose(out);
  std::printf("%ld samples, %ld reset(s)\n  line [Hz]   amplitude acc [m/s^2]          gyro [rad/s]\n", vc.samples(),
              vc.resets());
  for (size_t l = 0; l < vc.lines().size(); ++l)
    std::printf("  %8.3f   [%.4f %.4f %.4f]   [%.2e %.2e %.2e]\n", vc.frequencyHz(l), vc.amplitude(l, 0),
                vc.amplitude(l, 1), vc.amplitude(l, 2), vc.amplitude(l, 3), vc.amplitude(l, 4), vc.amplitude(l, 5));
  return 0;
}
