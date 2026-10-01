// R73 unit test (F-404): retired configuration key names are refused with a message that names the new key.
//   RK1 every retired name maps to a message that contains both the old and the new name.
//   RK2 the new names and unrelated keys give an empty message (not refused).
//   RK3 the table covers all ten q_adapt keys and the three continuous_prior keys (13 entries).
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
  CHECK(renamedKeyTable().size() == 13, "RK3 table has 13 entries");
  for (const auto& kv : renamedKeyTable()) {
    const std::string m = renamedKeyMessage(kv.first);
    if (m.find(kv.first) == std::string::npos || m.find(kv.second) == std::string::npos) CHECK(false, "RK1 message names old and new key");
    if (!renamedKeyMessage(kv.second).empty()) CHECK(false, "RK2 new name is not refused");
  }
  CHECK(true, "RK1 all 13 retired names refused with the new name in the message");
  CHECK(renamedKeyMessage("lio/zupt/mode").empty(), "RK2 lio/zupt/mode is not retired");
  CHECK(renamedKeyMessage("imu/process_noise/model").empty(), "RK2 imu/process_noise/model is not retired");
  CHECK(renamedKeyMessage("lio/q_adapt").empty(), "RK2 a prefix alone is not a retired key");
  std::printf("ALL PASS\n");
  return 0;
}
