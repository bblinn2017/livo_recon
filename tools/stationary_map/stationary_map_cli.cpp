// R52: C++ benchmark driver for the stationary_map_research library.
// Reads a preprocessed observation cache (manifest.csv + per-observation
// .bin sidecars written by preprocess_stationary_dump.py -- deliberately
// NOT parsing .npz directly; see that script's own comment for why) and
// drives makeBackend(family, options) at a sequence of observation-count
// checkpoints, dumping every accepted patch's full diagnostic fields plus
// a per-checkpoint summary row.
//
// Usage:
//   stationary_map_cli --family F --input DIR --patches-out CSV
//       --summary-out CSV [--checkpoints 1,2,3,5,10,20,...] [--leaf L]
//       [--min-points N] [--plane-eig-max V] [--min-secondary-eig V]
//       [--max-planarity R] [--merge-criterion combined_fit|pairwise]
//       [--debiased [--debias-context CSV] [--sensor-var V] [--sensor-noise-floor-eig0]]
//       [--merge-angle-deg A] [--merge-offset M] [--merge-gap G] ...
// Defaults reproduce the production VoxelMap plane-validity test (rank-2 and
// absolute eig0 < 0.01). The R52 rule is --plane-eig-max 1e30
// --min-secondary-eig 0 --max-planarity 0.10 --merge-criterion pairwise.
#include "livo_recon/map/stationary/stationary_map.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace livo_recon::stationary_map;

namespace {

struct ManifestRow { int obs_id; double timestamp; std::string file; bool in_calibration; };

std::vector<ManifestRow> loadManifest(const std::string& dir) {
  std::vector<ManifestRow> rows;
  std::ifstream f(dir + "/manifest.csv");
  std::string line;
  std::getline(f, line);  // header
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string obs_id, ts, points, file, sha, in_calib;
    std::getline(ss, obs_id, ','); std::getline(ss, ts, ',');
    std::getline(ss, points, ','); std::getline(ss, file, ',');
    std::getline(ss, sha, ','); std::getline(ss, in_calib, ',');
    ManifestRow r; r.obs_id = std::stoi(obs_id); r.timestamp = std::stod(ts);
    r.file = file; r.in_calibration = (in_calib == "1");
    rows.push_back(r);
  }
  std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.obs_id < b.obs_id; });
  return rows;
}

std::vector<V3> loadBin(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open cache file: " + path);
  std::int64_t n = 0;
  f.read(reinterpret_cast<char*>(&n), sizeof(n));
  if (n < 0 || n > 100000000) throw std::runtime_error("implausible point count " + std::to_string(n) + " in " + path);
  std::vector<double> flat(n * 3);
  if (n > 0) f.read(reinterpret_cast<char*>(flat.data()), n * 3 * sizeof(double));
  std::vector<V3> pts; pts.reserve(n);
  for (std::int64_t i = 0; i < n; ++i) pts.emplace_back(flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]);
  return pts;
}

std::vector<std::string> splitCsvArg(const std::string& s) {
  std::vector<std::string> out; std::stringstream ss(s); std::string tok;
  while (std::getline(ss, tok, ',')) out.push_back(tok);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string family, input, patches_out, summary_out;
  std::string checkpoints_arg = "1,2,3,5,10,20,50,100,200,500,1000,999999999";
  Options o;
  // ntu_viral.yaml voxel_map/plane/plane_threshold (the R49-R51 control jobs used it);
  // the library's own default is the code default 0.01.
  o.plane_eig_max = 2.5e-3;
  std::string debias_context;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--family") family = next();
    else if (a == "--input") input = next();
    else if (a == "--patches-out") patches_out = next();
    else if (a == "--summary-out") summary_out = next();
    else if (a == "--checkpoints") checkpoints_arg = next();
    else if (a == "--leaf") o.leaf = std::stod(next());
    else if (a == "--min-points") o.min_points = std::stoul(next());
    else if (a == "--merge-angle-deg") o.merge_angle_deg = std::stod(next());
    else if (a == "--merge-offset") o.merge_offset = std::stod(next());
    else if (a == "--merge-gap") o.merge_gap = std::stod(next());
    else if (a == "--plane-eig-max") o.plane_eig_max = std::stod(next());
    else if (a == "--min-secondary-eig") o.min_secondary_eig = std::stod(next());
    else if (a == "--max-planarity") o.max_planarity = std::stod(next());
    else if (a == "--merge-criterion") o.merge_criterion = next();
    else if (a == "--debiased") o.debiased = true;
    else if (a == "--sensor-noise-floor-eig0") o.sensor_noise_floor_eig0 = true;
    else if (a == "--sensor-var") o.sensor_var = std::stod(next());
    else if (a == "--debias-context") debias_context = next();
    else if (a == "--robust-reservoir") o.robust_reservoir = std::stoul(next());
    else if (a == "--robust-ransac-dist") o.robust_ransac_dist = std::stod(next());
    else if (a == "--robust-ransac-iters") o.robust_ransac_iters = std::stoi(next());
    else if (a == "--split-planarity-max") o.split_planarity_max = std::stod(next());
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
  }
  if (family.empty() || input.empty() || patches_out.empty() || summary_out.empty()) {
    std::fprintf(stderr, "missing required args\n"); return 1;
  }

  if (!debias_context.empty()) {
    // One line of 36 comma-separated numbers, row-major: R(9), P_RR(9), P_PP(9), P_RP(9).
    std::ifstream cf(debias_context);
    std::string line, tok;
    std::getline(cf, line);
    std::stringstream ss(line);
    std::vector<double> v;
    while (std::getline(ss, tok, ',')) v.push_back(std::stod(tok));
    if (v.size() != 36) { std::fprintf(stderr, "--debias-context needs 36 numbers, got %zu\n", v.size()); return 1; }
    auto m3 = [&](int off) { M3 m; for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) m(r, c) = v[off + 3 * r + c]; return m; };
    o.pose.R = m3(0); o.pose.P_RR = m3(9); o.pose.P_PP = m3(18); o.pose.P_RP = m3(27);
  }
  if (o.debiased && debias_context.empty())
    std::fprintf(stderr, "warning: --debiased without --debias-context: pose correction is zero\n");

  std::vector<std::uint64_t> checkpoints;
  for (auto& s : splitCsvArg(checkpoints_arg)) checkpoints.push_back(std::stoull(s));

  auto manifest = loadManifest(input);
  auto backend = makeBackend(family, o);

  std::ofstream patches_f(patches_out);
  patches_f << std::setprecision(17);
  patches_f << "checkpoint,patch_id,surface_id,n_children,child_ids,point_count,center_x,center_y,center_z,"
               "normal_x,normal_y,normal_z,d,eig0,eig1,eig2,planarity,bb_min_x,bb_min_y,bb_min_z,bb_max_x,bb_max_y,bb_max_z,"
               "rejected_points,reservoir_pending,rank2\n";
  std::ofstream summary_f(summary_out);
  summary_f << "checkpoint,backend,observations_ingested,total_raw_points_ingested,patches,surfaces,"
               "merges,splits,unmerges,cumulative_insert_ms,cells_total,cells_rank_deficient\n";

  std::size_t cp_idx = 0;
  std::uint64_t total_points = 0;
  for (std::size_t i = 0; i < manifest.size(); ++i) {
    // manifest.csv's "file" column names the .npz (the Python-tool-facing
    // sidecar); the CLI reads the .bin sidecar written alongside it by
    // preprocess_stationary_dump.py -- see that script's own comment.
    std::string bin_file = manifest[i].file;
    auto npz_pos = bin_file.rfind(".npz");
    if (npz_pos != std::string::npos) bin_file.replace(npz_pos, 4, ".bin");
    auto pts = loadBin(input + "/" + bin_file);
    total_points += pts.size();
    backend->insert(static_cast<std::uint64_t>(manifest[i].obs_id), pts);
    std::uint64_t obs_count = i + 1;
    if (cp_idx < checkpoints.size() && (obs_count >= checkpoints[cp_idx] || i + 1 == manifest.size())) {
      auto snap = backend->snapshot();
      summary_f << obs_count << "," << family << "," << obs_count << "," << total_points << ","
                << snap.patches << "," << snap.surfaces << "," << snap.merges << "," << snap.splits << ","
                << snap.unmerges << "," << snap.insert_ms << ","
                << snap.cells_total << "," << snap.cells_rank_deficient << "\n";
      for (const auto& p : snap.data) {
        patches_f << obs_count << "," << p.id << "," << p.surface_id << "," << p.children.size() << ",\"";
        for (std::size_t c = 0; c < p.children.size(); ++c) { if (c) patches_f << ";"; patches_f << p.children[c]; }
        patches_f << "\"," << p.stats.n << "," << p.fit.center.x() << "," << p.fit.center.y() << "," << p.fit.center.z() << ","
                  << p.fit.normal.x() << "," << p.fit.normal.y() << "," << p.fit.normal.z() << "," << p.fit.d << ","
                  << p.fit.eigenvalues[0] << "," << p.fit.eigenvalues[1] << "," << p.fit.eigenvalues[2] << "," << p.fit.planarity << ","
                  << p.bb_min.x() << "," << p.bb_min.y() << "," << p.bb_min.z() << ","
                  << p.bb_max.x() << "," << p.bb_max.y() << "," << p.bb_max.z() << ","
                  << p.rejected_points << "," << p.reservoir_pending << "," << (p.fit.rank2 ? 1 : 0) << "\n";
      }
      while (cp_idx < checkpoints.size() && obs_count >= checkpoints[cp_idx]) ++cp_idx;
      if (i + 1 == manifest.size()) break;
    }
  }
  std::fprintf(stderr, "%s: %zu observations, %llu points, done\n", family.c_str(), manifest.size(),
               static_cast<unsigned long long>(total_points));
  return 0;
}
