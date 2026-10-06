// R80 (F-389): unit test of the per-message LiDAR log helpers and of the overlap rule pushLidar() uses.
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "livo_recon/utils/data/lidar_msg_log.h"

using namespace livo_recon;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

static std::vector<std::string> split(const std::string& s)
{
  std::vector<std::string> out; std::stringstream ss(s); std::string t;
  while (std::getline(ss, t, ',')) out.push_back(t);
  return out;
}

int main()
{
  // Rule: strictly earlier first point than the previous accepted last point is dropped; equal is kept.
  CHECK(lidarMessageOverlapsPrevious(10.0, 10.05));
  CHECK(!lidarMessageOverlapsPrevious(10.05, 10.05));
  CHECK(!lidarMessageOverlapsPrevious(10.1, 10.05));
  CHECK(!lidarMessageOverlapsPrevious(5.0, 0.0));   // before the first accepted message latest_lidar_time = 0

  // Header columns == row columns.
  const std::string hdr = lidarMsgLogHeader();
  CHECK(!hdr.empty() && hdr.back() == '\n');
  const size_t ncol = split(hdr.substr(0, hdr.size() - 1)).size();
  CHECK(ncol == 17);   // R80C: retime_shift_s, n_trimmed appended

  // Accepted message, first point delayed 5 ms, span 99.9 ms.
  LidarMsgRow r;
  r.msg_index = 3; r.seq = 77; r.header_stamp = 1000.000000; r.scan_time = 999.900000; r.prev_header_stamp = 999.9;
  r.n_raw = 20000; r.n_kept = 9000; r.t_first = 999.905; r.t_last = 1000.0049; r.prev_accepted_last_t = 999.8;
  r.verdict = LidarMsgVerdict::Accepted;
  auto f = split(formatLidarMsgRow(r).substr(0, formatLidarMsgRow(r).size() - 1));
  CHECK(f.size() == ncol);
  CHECK(std::fabs(std::stod(f[4]) - 0.1) < 1e-6);        // gap_header_s
  CHECK(std::fabs(std::stod(f[9]) - 0.005) < 1e-6);      // off_first_s
  CHECK(std::fabs(std::stod(f[10]) - 0.1049) < 1e-6);    // off_last_s
  CHECK(std::fabs(std::stod(f[11]) - 0.0999) < 1e-6);    // span_s
  CHECK(std::fabs(std::stod(f[13]) - (999.8 - 999.905)) < 1e-6);  // overlap_s negative = no overlap
  CHECK(f[14] == "0");
  CHECK(f[15] == "0.000000000" && f[16] == "0");   // R80C: default row = discard policy (no shift, nothing trimmed)

  // Overlapping drop: previous ended 3 ms after this one starts.
  LidarMsgRow d = r; d.prev_accepted_last_t = 999.908; d.verdict = LidarMsgVerdict::DroppedOverlap;
  auto g = split(formatLidarMsgRow(d).substr(0, formatLidarMsgRow(d).size() - 1));
  CHECK(std::fabs(std::stod(g[13]) - 0.003) < 1e-6);
  CHECK(g[14] == "1");
  CHECK(lidarMessageOverlapsPrevious(d.t_first, d.prev_accepted_last_t));

  // Not built (no points): nan fields, verdict 3; no prior message: gap nan, overlap nan.
  LidarMsgRow e; e.header_stamp = 1.0; e.scan_time = 0.9; e.n_raw = 10; e.verdict = LidarMsgVerdict::NoPoints;
  auto h = split(formatLidarMsgRow(e).substr(0, formatLidarMsgRow(e).size() - 1));
  CHECK(h.size() == ncol && h[4] == "nan" && h[7] == "nan" && h[9] == "nan" && h[13] == "nan" && h[14] == "3");
  LidarMsgRow first = r; first.prev_header_stamp = std::numeric_limits<double>::quiet_NaN(); first.prev_accepted_last_t = 0.0;
  auto k = split(formatLidarMsgRow(first).substr(0, formatLidarMsgRow(first).size() - 1));
  CHECK(k[4] == "nan" && k[12] == "nan" && k[13] == "nan");

  if (g_fail) { std::printf("%d failures\n", g_fail); return 1; }
  std::printf("lidar_msg_log_r80: all checks passed\n");
  return 0;
}
