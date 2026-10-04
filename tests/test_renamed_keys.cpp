// R73 unit test (F-404): retired configuration key names are refused with a message that names the new key.
//   RK1 every retired name maps to a message that contains both the old and the new name.
//   RK2 the new names and unrelated keys give an empty message (not refused).
//   RK3 the renamed table covers the ten q_adapt keys (10 entries; the three continuous_prior entries of R73 moved to the removed table).
//   RK4 (R74 F-427) every removed key gives a message that contains the key and "removed in R74"; the removed table has 8 entries.
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
  CHECK(renamedKeyMessage("lio/zupt/mode").empty(), "RK2 lio/zupt/mode is not retired");
  CHECK(renamedKeyMessage("imu/process_noise/model").empty(), "RK2 imu/process_noise/model is not retired");
  CHECK(renamedKeyMessage("lio/q_adapt").empty(), "RK2 a prefix alone is not a retired key");
  CHECK(removedKeyTable().size() == 8, "RK4 removed table has 8 entries");
  for (const auto& k : removedKeyTable()) {
    const std::string m = removedKeyMessage(k);
    if (m.find(k) == std::string::npos || m.find("removed in R74") == std::string::npos) CHECK(false, "RK4 message names the key and R74");
  }
  CHECK(true, "RK4 all 8 removed keys refused with a message that names the key");
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
