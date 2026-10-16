// R91C: the frame-list parser and the solve-snapshot decision (include/livo_recon/diagnostics/dump_rules.h). Header-only; no ROS.
#include "livo_recon/diagnostics/dump_rules.h"
#include <cstdio>
#include <stdexcept>

using namespace livo_recon::dump_rules;

static int fails = 0;
#define CK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static bool throws(const char* text)
{
  try { parseFrameList(text, "lio/test_key"); } catch (const std::invalid_argument&) { return true; }
  return false;
}

int main()
{
  // parser: empty is empty; a list keeps its order; spaces are ignored; a trailing comma is tolerated.
  CK(parseFrameList("", "k").empty());
  { const auto v = parseFrameList("100,101, 150,485", "k"); CK(v.size() == 4 && v[0] == 100 && v[1] == 101 && v[2] == 150 && v[3] == 485); }
  { const auto v = parseFrameList("7,", "k"); CK(v.size() == 1 && v[0] == 7); }
  // parser refuses: a non-integer, zero, negative, a repeat, a float, too many frames.
  CK(throws("100,abc"));
  CK(throws("0"));
  CK(throws("-5"));
  CK(throws("100,100"));
  CK(throws("1.5"));
  { std::string big; for (int i = 1; i <= kMaxDumpListFrames + 1; ++i) { if (i > 1) big += ","; big += std::to_string(i); } CK(throws(big.c_str())); }
  { std::string ok; for (int i = 1; i <= kMaxDumpListFrames; ++i) { if (i > 1) ok += ","; ok += std::to_string(i); } CK(!throws(ok.c_str())); }
  // the error message names the key.
  try { parseFrameList("x", "log/lio/first_frame_dump/frame_list"); CK(false); }
  catch (const std::invalid_argument& e) { CK(std::string(e.what()).find("log/lio/first_frame_dump/frame_list") != std::string::npos); }

  // decision: default (K = 1, empty list) = frame 1 full, nothing else (as before R91C).
  const std::vector<int> none;
  CK(firstFrameSolveDumpMode(1, 1, none, false) == 1);
  CK(firstFrameSolveDumpMode(2, 1, none, false) == 0);
  CK(firstFrameSolveDumpMode(0, 1, none, false) == 0);
  CK(firstFrameSolveDumpMode(100, 1, none, true) == 0);   // slim without a list is refused at load; here it must still do nothing
  // K = 3: frames 1..3 full.
  CK(firstFrameSolveDumpMode(3, 3, none, false) == 1);
  CK(firstFrameSolveDumpMode(4, 3, none, false) == 0);
  // a list adds frames: slim off = full snapshot, slim on = slim snapshot; frames 1..K stay full under slim.
  const std::vector<int> lst = {100, 150, 2};
  CK(firstFrameSolveDumpMode(100, 1, lst, false) == 1);
  CK(firstFrameSolveDumpMode(100, 1, lst, true) == 2);
  CK(firstFrameSolveDumpMode(150, 1, lst, true) == 2);
  CK(firstFrameSolveDumpMode(101, 1, lst, true) == 0);
  CK(firstFrameSolveDumpMode(1, 1, lst, true) == 1);
  CK(firstFrameSolveDumpMode(2, 5, lst, true) == 1);      // in 1..K and in the list: full
  CK(firstFrameSolveDumpMode(2, 1, lst, true) == 2);      // in the list only: slim
  std::printf("test_dump_rules_r91c %s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
