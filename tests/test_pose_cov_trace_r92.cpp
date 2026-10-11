// R92: the slim posterior pose trace writer (writePoseCovTraceRow, diagnostics/state_trace.h). Planner sandbox on a synthetic covariance: NOT evidence about any bag. Identities only (rule 3.1).
//   PT1 the header names run_id,scan_id,t_abs,t_rel_s,qw..qz,px..pz,residual_count,completed_iterations and the 21 upper-triangle columns C_i_j (i <= j < 6); every row has as many fields as the header.
//   PT2 the 6 x 6 block maps rotation (state 0-2) to pose 0-2 and position (state 3-5) to pose 3-5: C_0_0 = P(0,0), C_0_3 = P(0,3), C_2_4 = P(2,4), C_3_5 = P(3,5), C_5_5 = P(5,5) of a covariance whose entry P(i,j) = 1000 min(i,j) + max(i,j) + 0.5 (so no two entries are equal).
//   PT3 control: no column holds a velocity or bias entry (P(6,6) = 6006.5 and P(9,9) = 9009.5 appear in no field), and two rows (two calls) give two data lines.
// Links src/diagnostics/state_trace.cpp, debug_log_dir.cpp and state_lib; no ROS node.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "livo_recon/diagnostics/log/debug_log_dir.h"
#include "livo_recon/diagnostics/state_trace.h"

using namespace livo_recon;
static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

static std::vector<std::string> split(const std::string& s)
{
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) v.push_back(t);
  if (!s.empty() && s.back() == ',') v.push_back("");
  return v;
}

int main()
{
  const char* dir = "/tmp/livo_pose_cov_trace_r92_test";
  setDebugLogDir(dir);
  StateGroup s;
  Eigen::MatrixXd P = Eigen::MatrixXd::Zero(18, 18);
  for (int i = 0; i < 18; ++i)
    for (int j = 0; j < 18; ++j) P(i, j) = 1000.0 * std::min(i, j) + std::max(i, j) + 0.5;
  s.covMut() = P;
  writePoseCovTraceRow("t", 7, 100.25, s, 42, 3);
  writePoseCovTraceRow("t", 8, 100.35, s, 43, 4);
  std::ifstream in(std::string(dir) + "/pose_cov_trace.csv");
  std::string line;
  std::vector<std::string> lines;
  while (std::getline(in, line)) lines.push_back(line);
  CHECK(lines.size() == 3);   // header + 2 rows (PT3)
  if (lines.size() < 3) { std::printf("test_pose_cov_trace_r92 FAIL\n"); return 1; }
  const auto h = split(lines[0]);
  const std::vector<std::string> head13 = {"run_id", "scan_id", "t_abs", "t_rel_s", "qw", "qx", "qy", "qz", "px", "py", "pz", "residual_count", "completed_iterations"};
  CHECK(h.size() == 13 + 21);
  for (std::size_t i = 0; i < head13.size() && i < h.size(); ++i) CHECK(h[i] == head13[i]);
  std::map<std::string, double> val;
  const auto r = split(lines[1]);
  CHECK(r.size() == h.size() && split(lines[2]).size() == h.size());   // PT1
  for (std::size_t i = 13; i < h.size() && i < r.size(); ++i) val[h[i]] = std::atof(r[i].c_str());
  CHECK(val.size() == 21);
  CHECK(val.count("C_0_0") && val["C_0_0"] == P(0, 0));
  CHECK(val.count("C_0_3") && val["C_0_3"] == P(0, 3));
  CHECK(val.count("C_2_4") && val["C_2_4"] == P(2, 4));
  CHECK(val.count("C_3_5") && val["C_3_5"] == P(3, 5));
  CHECK(val.count("C_5_5") && val["C_5_5"] == P(5, 5));   // PT2
  CHECK(val.count("C_1_4") && val["C_1_4"] == P(1, 4) && val["C_1_4"] != P(1, 3));
  for (const auto& kv : val) CHECK(std::fabs(kv.second - 6006.5) > 1e-9 && std::fabs(kv.second - 9009.5) > 1e-9);   // PT3
  CHECK(r[0] == "t" && r[1] == "7" && r[11] == "42" && r[12] == "3");
  std::printf("test_pose_cov_trace_r92 %s\n", g_fail ? "FAIL" : "PASS");
  return g_fail ? 1 : 0;
}
