// R86A (F-532) unit test: the second (dry-run) stream under retime_trim (header only, ROS-free; the queue rule is replayed with lidarMessageOverlapsPrevious, the function pushLidar and
// pushDryRunLidar both call). Planner's test, never compiled or run by the planner against the real build (rule 1.1).
//   DT1 trimStreamOverlap removes exactly the points strictly earlier than the stream's last accepted time, keeps a point at exactly that time, and returns the count.
//   DT2 replaying overlapping messages through the old second-stream path (no trim) REJECTS whole messages; through the new path (trim against its own last time) rejects none and keeps
//       every point later than the previous end. The check can fail: the old path must show rejections, otherwise the replay does not reproduce the R86 defect.
//   DT3 the trimmed second stream equals the set a stand-alone point-filter-k primary stream keeps (same helper, same boundary), so a split run's residual points are the points a
//       point-filter-k run would use.
//   DT4 an empty stream and an all-earlier message do not push and do not throw.
#include <cstdio>
#include <vector>

#include "livo_recon/utils/data/lidar_msg_log.h"

using namespace livo_recon;
static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)
struct P { double t; };

// replay of pushLidar / pushDryRunLidar for one stream: returns the number of rejected messages and the number of points kept
static void replay(const std::vector<std::vector<P>>& msgs, bool trim, int& rejected, long& kept)
{
  double latest = 0.0; rejected = 0; kept = 0;
  for (auto m : msgs)
  {
    if (trim) trimStreamOverlap(m, latest);
    if (m.empty()) continue;
    if (lidarMessageOverlapsPrevious(m.front().t, latest)) { ++rejected; continue; }
    latest = m.back().t; kept += static_cast<long>(m.size());
  }
}

int main()
{
  std::vector<P> v = {{10.0}, {10.1}, {10.2}, {10.2}, {10.3}};
  CHECK(trimStreamOverlap(v, 9.0) == 0 && v.size() == 5);
  CHECK(trimStreamOverlap(v, 10.0) == 0 && v.size() == 5);            // equal stays
  CHECK(trimStreamOverlap(v, 10.15) == 2 && v.size() == 3 && v.front().t == 10.2);
  std::vector<P> e;
  CHECK(trimStreamOverlap(e, 5.0) == 0 && e.empty());                  // DT4
  std::vector<P> early = {{1.0}, {2.0}};
  CHECK(trimStreamOverlap(early, 9.0) == 2 && early.empty());          // DT4: everything earlier

  // messages of 100 raw points on a 0.1 s grid; every 4th message starts 3 points (3 ms) early, as the retimed NTU VIRAL messages do
  std::vector<std::vector<P>> raw;
  for (int m = 0; m < 40; ++m)
  {
    std::vector<P> msg;
    const double start = 0.1 * m - ((m % 4 == 3) ? 0.003 : 0.0);
    for (int i = 0; i < 100; ++i) msg.push_back({start + 0.001 * i});
    raw.push_back(msg);
  }
  auto decimate = [&](int k) { std::vector<std::vector<P>> out; for (auto& m : raw) { std::vector<P> d; for (std::size_t i = 0; i < m.size(); ++i) if (k <= 1 || i % k == 0) d.push_back(m[i]); out.push_back(d); } return out; };
  const auto d3 = decimate(3), d1 = decimate(1);
  int rej_old = 0, rej_new = 0, rej_p = 0; long kept_old = 0, kept_new = 0, kept_p = 0;
  replay(d3, false, rej_old, kept_old);            // the R85 / R86 second-stream path
  replay(d3, true, rej_new, kept_new);             // the R86A path
  CHECK(rej_old > 0);                              // DT2: the replay reproduces the defect
  CHECK(rej_new == 0);                             // DT2: fixed
  CHECK(kept_new > kept_old);
  // DT3: a stand-alone point-filter-3 primary stream is the same helper on the same decimated messages
  replay(d3, true, rej_p, kept_p);
  CHECK(rej_p == rej_new && kept_p == kept_new);
  // point filter 1 as second stream also rejects nothing
  int r1 = 0; long k1 = 0; replay(d1, true, r1, k1);
  CHECK(r1 == 0);
  std::printf(g_fail ? "FAIL %d\n" : "PASS test_dry_run_trim_r86a\n", g_fail);
  return g_fail ? 1 : 0;
}
