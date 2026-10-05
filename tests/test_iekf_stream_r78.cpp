// R78 unit test: which point set the IEKF uses (include/livo_recon/lio/iekf_stream.h). Pure selection logic, ROS-free, header only.
//   IK1 primary (the default): returns the primary vector itself (same object, so bit-identical input), no flag set, whatever the second set holds.
//   IK2 second with a non-empty second set: returns the second vector itself, second_used true, fell_back false.
//   IK3 second with an empty second set: falls back to the primary vector, fell_back true, second_used false.
//   IK4 the checks can fail: an implementation that always returned the primary set would fail IK2 (the returned object must differ from the primary one
//       and carry the second set's size); an implementation that never fell back would fail IK3 (empty result).
//   IK5 the flags are reset on every call (a previous fell_back does not leak into the next frame).
#include "livo_recon/lio/iekf_stream.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

int main()
{
  std::vector<int> primary, second, empty;
  for (int i = 0; i < 30; ++i) primary.push_back(i);
  for (int i = 0; i < 30; i += 3) second.push_back(i);
  IekfStreamPick pick;

  const std::vector<int>& a = pickIekfPoints(false, primary, second, pick);
  CHECK(&a == &primary && a.size() == 30, "IK1 primary returns the primary vector itself");
  CHECK(!pick.second_used && !pick.fell_back, "IK1 primary sets no flag");
  const std::vector<int>& a2 = pickIekfPoints(false, primary, empty, pick);
  CHECK(&a2 == &primary && !pick.fell_back, "IK1 primary ignores an empty second set and does not count a fallback");

  const std::vector<int>& b = pickIekfPoints(true, primary, second, pick);
  CHECK(&b == &second && b.size() == 10, "IK2 second returns the second vector itself (10 of 30)");
  CHECK(pick.second_used && !pick.fell_back, "IK2 second sets second_used only");
  CHECK(&b != &primary && b.size() != primary.size(), "IK4a an always-primary implementation would fail IK2: the result is a different object of a different size");

  const std::vector<int>& c = pickIekfPoints(true, primary, empty, pick);
  CHECK(&c == &primary && c.size() == 30, "IK3 empty second set falls back to the primary vector");
  CHECK(pick.fell_back && !pick.second_used, "IK3 fallback sets fell_back only");
  CHECK(c.size() != empty.size(), "IK4b a never-falling-back implementation would return the empty set and fail IK3");

  const std::vector<int>& d = pickIekfPoints(true, primary, second, pick);
  CHECK(&d == &second && !pick.fell_back, "IK5 the flags are reset on every call");

  std::printf("test_iekf_stream_r78: all checks passed\n");
  return 0;
}
