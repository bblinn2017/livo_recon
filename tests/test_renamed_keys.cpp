// R73 unit test (F-404): retired configuration key names are refused with a message that names the new key.
//   RK1 every retired name maps to a message that contains both the old and the new name.
//   RK2 the new names and unrelated keys give an empty message (not refused).
//   RK3 the renamed table covers the ten q_adapt keys (10 entries; the three continuous_prior entries of R73 moved to the removed table).
//   RK4 (R74 F-427, R82 F-476 and F-403, R83 F-401 and F-508, R84 F-521, R85 F-520) every removed key gives a message that contains the key and its release; the removed table has 28 entries (R92: voxel_map/binned/update_count_mode).
//   RK5 a removed key is never also a renamed-key target or source in the renamed table (no key can be both renamed and removed),
//       and the new gain keys are not removed.
#include "livo_recon/utils/log/renamed_key_table.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

int main()
{
  CHECK(renamedKeyTable().size() == 10, "RK3 renamed table has 10 entries");
  for (const auto& kv : renamedKeyTable()) {
    const std::string m = renamedKeyMessage(kv.first);
    if (m.find(kv.first) == std::string::npos || m.find(kv.second) == std::string::npos) CHECK(false, "RK1 message names old and new key");
    if (!renamedKeyMessage(kv.second).empty()) CHECK(false, "RK2 new name is not refused");
  }
  CHECK(true, "RK1 all 10 retired names refused with the new name in the message");
  CHECK(renamedKeyMessage("lio/zupt/mode").empty(), "RK2 lio/zupt/mode is removed, not renamed");
  CHECK(renamedKeyMessage("imu/process_noise/model").empty(), "RK2 imu/process_noise/model is not retired");
  CHECK(renamedKeyMessage("lio/q_adapt").empty(), "RK2 a prefix alone is not a retired key");
  CHECK(removedKeyTable().size() == 28, "RK4 removed table has 28 entries (8 from R74, 5 from R82, 9 from R83, 4 from R84, 1 from R85, 1 from R92)");
  for (const auto& k : removedKeyTable()) {
    const std::string m = removedKeyMessage(k);
    if (m.find(k) == std::string::npos || m.find("removed in R7") == std::string::npos && m.find("removed in R82") == std::string::npos && m.find("removed in R83") == std::string::npos && m.find("removed in R84") == std::string::npos && m.find("removed in R85") == std::string::npos && m.find("removed in R92") == std::string::npos) CHECK(false, "RK4 message names the key and its release");
  }
  CHECK(true, "RK4 all 28 removed keys refused with a message that names the key");
  CHECK(removedKeyMessage("lio/zupt/mode").find("R83 (F-401, F-508)") != std::string::npos, "RK6 lio/zupt/mode names R83 F-401 F-508");
  CHECK(removedKeyMessage("lio/zupt/calib_margin_gyro").find("R83") != std::string::npos && removedKeyMessage("lio/zupt/sigma_v").find("R83") != std::string::npos, "RK6 every lio/zupt key is refused");
  CHECK(removedKeyMessage("lio/zupt").empty(), "RK5 the lio/zupt prefix alone is not a removed key");
  CHECK(removedKeyMessage("lio/point_cov/mode").find("R84 (F-521)") != std::string::npos
        && removedKeyMessage("lio/point_cov/window_s").find("R84 (F-521)") != std::string::npos
        && removedKeyMessage("lio/time_cov/mode").find("R84 (F-521)") != std::string::npos
        && removedKeyMessage("lio/time_cov/sigma_s").find("R84 (F-521)") != std::string::npos, "RK7 the removed temporal point-covariance keys name R84 F-521");
  CHECK(removedKeyMessage("lio/point_cov/sample_window_s").find("R85 (F-520)") != std::string::npos, "RK9 lio/point_cov/sample_window_s is refused and names R85 F-520");
  CHECK(removedKeyMessage("voxel_map/binned/update_count_mode").find("removed in R92") != std::string::npos, "RK10 voxel_map/binned/update_count_mode is refused and names R92");
  CHECK(removedKeyMessage("voxel_map/binned/cov_mode").empty() && removedKeyMessage("voxel_map/binned/bin_size_fraction").empty() && removedKeyMessage("voxel_map/binned/weight_mode").empty(), "RK10 the other binned keys are not removed");
  CHECK(removedKeyMessage("lio/point_cov/insertion_source").empty(), "RK8 lio/point_cov/insertion_source is kept");
  CHECK(removedKeyMessage("lio/ekf/step_cap/mode").find("R82 (F-403)") != std::string::npos, "RK6 step_cap/mode names R82 F-403");
  CHECK(removedKeyMessage("imu/process_noise/gain/nis_lo").find("R82 (F-476)") != std::string::npos, "RK6 nis_lo names R82 F-476");
  CHECK(removedKeyMessage("imu/process_noise/disturbance/mode").find("R74 (F-427)") != std::string::npos, "RK6 R74 keys keep their R74 message");
  CHECK(removedKeyMessage("imu/process_noise/gain/nis_hi").empty() && removedKeyMessage("imu/process_noise/gain/inflate").empty(), "RK6 nis_gate keys that stay are not removed");
  for (const auto& kv : renamedKeyTable())
    for (const auto& k : removedKeyTable())
      if (kv.first == k || kv.second == k) CHECK(false, "RK5 a key is both renamed and removed");
  CHECK(true, "RK5 no overlap between renamed and removed tables");
  CHECK(removedKeyMessage("imu/process_noise/gain/mode").empty(), "RK5 gain/mode is not removed");
  CHECK(removedKeyMessage("imu/process_noise/model").empty(), "RK5 process_noise/model is not removed");
  CHECK(removedKeyMessage("imu/process_noise/disturbance").empty(), "RK5 a prefix alone is not a removed key");
  CHECK(removedKeyMessage("imu/process_noise/motion/beta").empty(), "RK5 motion/beta (kept) is not removed");
  std::printf("ALL PASS\n");
  return 0;
}
