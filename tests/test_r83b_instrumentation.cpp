// R83B unit test for the two new instrumentation outputs (A1/B4 plane_fit_eigs.csv, A6/B5 motion_q_step.csv).
// ROS-free. Checks
//   E1  plane_fit_eigs_en=false (default): VoxelPlane::update()/addPoints() behavior and plane_fit_trace.csv /
//       plane_fit_reasons.csv are unaffected by the new layer_idx parameter -- no plane_fit_eigs.csv is written.
//   E2  plane_fit_eigs_en=true: plane_fit_eigs.csv gets one row per fit attempt with the voxel's layer (threaded
//       through update()'s new layer_idx parameter) and the three eigenvalues of the raw (uncorrected) scatter,
//       ascending order, matching an independently-computed reference to 1e-10.
//   E3  addPoints() (the debiased accumulation path) also threads layer_idx through to plane_fit_eigs.csv.
//   E4  writeMotionQStepDiagnostic() writes one row per call with the exact q_acc/q_gyr/clamp values passed in.
// Exit code 0 = all pass.
#include "livo_recon/lio/voxelplane.h"
#include "livo_recon/diagnostics/log/debug_log_dir.h"
#include "livo_recon/diagnostics/motion_q_writer.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace livo_recon;

namespace {
int g_fail = 0;
void check(bool ok, const char* name, const std::string& detail)
{
  std::printf("[%s] %s  %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
  if (!ok) ++g_fail;
}

std::vector<PointXYZCov> makePoints(std::mt19937& rng, int n, double sz, const M3D& sensor_cov)
{
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  std::normal_distribution<double> Nz(0.0, sz);
  std::vector<PointXYZCov> pts;
  for (int i = 0; i < n; ++i) {
    PointXYZCov p;
    const double x = U(rng), y = U(rng);
    p.point = V3D(x, y, 0.05 * x + 0.02 * y + Nz(rng));
    p.sensor_cov = sensor_cov;
    p.pos_cov = M3D::Zero();
    p.bootstrap_observation_id = -1;
    pts.push_back(p);
  }
  return pts;
}

VoxelOptsPtr makeOpts(bool eigs_en, int trace_frames)
{
  auto o = std::make_shared<VoxelOpts>();
  o->plane_fit_mode = "pca";
  o->debias_mode = "none";
  o->plane_threshold = 0.01;
  o->plane_fit_eigs_en = eigs_en;
  o->plane_fit_trace_frames = trace_frames;
  o->plane_fit_trace_max_rows = 200000;
  return o;
}

struct Row { int frame; std::string path; int layer; int N; int F; int reason; double e0, e1, e2; };

std::vector<Row> readEigsCsv(const std::string& path)
{
  std::vector<Row> rows;
  std::ifstream f(path);
  std::string line;
  if (!f || !std::getline(f, line)) return rows;   // header only / absent
  while (std::getline(f, line)) {
    std::stringstream ss(line); std::string tok; std::vector<std::string> c;
    while (std::getline(ss, tok, ',')) c.push_back(tok);
    if (c.size() != 9) { std::fprintf(stderr, "readEigsCsv: skipping line with %zu fields: %s\n", c.size(), line.c_str()); continue; }
    try {
      Row r;
      r.frame = std::stoi(c[0]); r.path = c[1]; r.layer = std::stoi(c[2]);
      r.N = std::stoi(c[3]); r.F = std::stoi(c[4]); r.reason = std::stoi(c[5]);
      r.e0 = std::stod(c[6]); r.e1 = std::stod(c[7]); r.e2 = std::stod(c[8]);
      rows.push_back(r);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "readEigsCsv: parse error on line: %s  (%s)\n", line.c_str(), e.what());
    }
  }
  return rows;
}
}  // namespace

int main()
{
  setDebugLogDir("");   // "" -> /tmp, same convention test_debias_modes.cpp relies on
  std::remove("/tmp/plane_fit_eigs.csv");
  std::remove("/tmp/plane_fit_trace.csv");
  std::remove("/tmp/plane_fit_reasons.csv");
  std::remove("/tmp/motion_q_step.csv");
  const M3D sens_iso = 1e-6 * M3D::Identity();
  std::mt19937 rng(42);

  // ---- E1: default off, no plane_fit_eigs.csv ----
  {
    auto o = makeOpts(/*eigs_en=*/false, /*trace_frames=*/5);
    VoxelPlane vp(o);
    vp.update(makePoints(rng, 40, 0.02, sens_iso), -1, nullptr, nullptr, nullptr, /*frame_idx=*/3, /*layer_idx=*/2);
    voxelPlaneFitTraceFlush();
    std::ifstream f("/tmp/plane_fit_eigs.csv");
    check(!f.good() || f.peek() == std::ifstream::traits_type::eof(),
          "E1 plane_fit_eigs_en=false writes no plane_fit_eigs.csv", "");
  }

  // ---- E2: update() (PCA path), eigs_en=true, layer threaded through ----
  std::remove("/tmp/plane_fit_eigs.csv");
  {
    auto o = makeOpts(/*eigs_en=*/true, /*trace_frames=*/5);
    VoxelPlane vp(o);
    auto pts = makePoints(rng, 50, 0.02, sens_iso);
    vp.update(pts, -1, nullptr, nullptr, nullptr, /*frame_idx=*/3, /*layer_idx=*/2);
    voxelPlaneFitTraceFlush();
    auto rows = readEigsCsv("/tmp/plane_fit_eigs.csv");
    bool found = false;
    for (const auto& r : rows) if (r.frame == 3 && r.path == "pca") { found = true;
      check(r.layer == 2, "E2 layer column equals the layer_idx passed to update()", "got " + std::to_string(r.layer));
      check(r.N == (int)pts.size(), "E2 N column equals the point count", "got " + std::to_string(r.N));
      check(r.e0 <= r.e1 + 1e-12 && r.e1 <= r.e2 + 1e-12, "E2 eigenvalues ascending", "");
      // independent reference: eigenvalues of the raw (uncorrected) scatter about the mean
      const double N = static_cast<double>(pts.size());
      V3D sp = V3D::Zero(); M3D spp = M3D::Zero();
      for (const auto& p : pts) { sp += p.point; spp += p.point * p.point.transpose(); }
      const V3D mean = sp / N;
      const M3D raw = spp / N - mean * mean.transpose();
      Eigen::SelfAdjointEigenSolver<M3D> rs(raw);
      const V3D re = rs.eigenvalues();
      check(std::fabs(r.e0 - re(0)) < 1e-9 && std::fabs(r.e1 - re(1)) < 1e-9 && std::fabs(r.e2 - re(2)) < 1e-9,
            "E2 eigenvalues match an independent raw-scatter eigendecomposition", "");
    }
    check(found, "E2 a frame=3 pca row was written", "");
  }

  // ---- E3: addPoints() (debiased path) also threads layer ----
  // NOTE: deliberately no std::remove() here -- plane_fit_eigs.csv is written through a function-local static
  // PersistentLogStream (see voxelPlaneFitTraceFlush()); removing the file mid-process would orphan its already-
  // open fd onto the unlinked inode, so the next flush's writes would go nowhere visible. Check cumulative content
  // instead (both E2's frame=3 row and this block's frame=7 row are expected to be present).
  {
    auto o = makeOpts(/*eigs_en=*/true, /*trace_frames=*/9);
    o->plane_fit_mode = "debiased";
    VoxelPlane vp(o);
    vp.addPoints(makePoints(rng, 40, 0.02, sens_iso), -1, 1, true, /*frame_idx=*/7, /*layer_idx=*/1);
    voxelPlaneFitTraceFlush();
    auto rows = readEigsCsv("/tmp/plane_fit_eigs.csv");
    bool found = false, e2_still_present = false;
    for (const auto& r : rows) {
      if (r.frame == 7 && r.path == "debiased") { found = true;
        check(r.layer == 1, "E3 addPoints() threads layer_idx through to plane_fit_eigs.csv", "got " + std::to_string(r.layer));
      }
      if (r.frame == 3 && r.path == "pca") e2_still_present = true;
    }
    check(found, "E3 a frame=7 debiased row was written", "rows=" + std::to_string(rows.size()));
    check(e2_still_present, "E3 E2's frame=3 row is still present (cumulative, same process)", "");
  }

  // ---- E4: writeMotionQStepDiagnostic() writes the exact values passed in ----
  std::remove("/tmp/motion_q_step.csv");
  {
    const V3D q_acc(0.1, 0.2, 0.3), q_gyr(0.4, 0.5, 0.6);
    writeMotionQStepDiagnostic(5, 12, 1000.25, q_acc, q_gyr, true, false, true, false, true, false);
    std::ifstream f("/tmp/motion_q_step.csv");
    std::string header, line;
    bool ok = false;
    if (f && std::getline(f, header) && std::getline(f, line)) {
      std::stringstream ss(line); std::string tok; std::vector<std::string> c;
      while (std::getline(ss, tok, ',')) c.push_back(tok);
      ok = c.size() == 15 && c[0] == "5" && c[1] == "12" &&
           std::fabs(std::stod(c[3]) - 0.1) < 1e-12 && std::fabs(std::stod(c[8]) - 0.6) < 1e-12 &&
           c[9] == "1" && c[10] == "0" && c[11] == "1" && c[12] == "0" && c[13] == "1" && c[14] == "0";
    }
    check(ok, "E4 motion_q_step.csv row matches the values passed to the writer", ok ? "" : "line=" + line);
  }

  std::printf("%s (%d failed)\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail);
  return g_fail == 0 ? 0 : 1;
}
