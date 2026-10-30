#include "livo_recon/diagnostics/pin_state_diag.h"

#include <Eigen/Dense>

#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace livo_recon
{

namespace
{
std::vector<std::string> splitCsvLine(const std::string& line)
{
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string cell;
  while (std::getline(ss, cell, ',')) out.push_back(cell);
  return out;
}
}  // namespace

bool loadPinState(const std::string& csv_path, int scan_id, const std::string& phase, StateGroup& state)
{
  std::ifstream in(csv_path);
  if (!in.is_open()) return false;

  std::string header_line;
  if (!std::getline(in, header_line)) return false;
  const std::vector<std::string> header = splitCsvLine(header_line);
  std::unordered_map<std::string, int> col;
  for (size_t i = 0; i < header.size(); ++i) col[header[i]] = static_cast<int>(i);

  auto need = [&](const std::string& name) -> int
  {
    auto it = col.find(name);
    return it == col.end() ? -1 : it->second;
  };

  const int c_scan_id = need("scan_id");
  const int c_phase = need("phase");
  const int c_dim = need("dim");
  if (c_scan_id < 0 || c_phase < 0 || c_dim < 0) return false;

  std::string line;
  std::vector<std::string> row;
  bool found = false;
  while (std::getline(in, line))
  {
    row = splitCsvLine(line);
    if (static_cast<int>(row.size()) <= c_scan_id || static_cast<int>(row.size()) <= c_phase) continue;
    if (row[c_phase] != phase) continue;
    if (std::stoi(row[c_scan_id]) != scan_id) continue;
    found = true;
    break;
  }
  if (!found) return false;

  const int dim = std::stoi(row[c_dim]);
  if (dim != state.dimState()) return false;

  const int c_qw = need("qw"), c_qx = need("qx"), c_qy = need("qy"), c_qz = need("qz");
  const int c_px = need("px"), c_py = need("py"), c_pz = need("pz");
  const int c_vx = need("vx"), c_vy = need("vy"), c_vz = need("vz");
  const int c_bgx = need("bgx"), c_bgy = need("bgy"), c_bgz = need("bgz");
  const int c_bax = need("bax"), c_bay = need("bay"), c_baz = need("baz");
  const int c_gx = need("gx"), c_gy = need("gy"), c_gz = need("gz");
  if (c_qw < 0 || c_px < 0 || c_vx < 0 || c_bgx < 0 || c_bax < 0 || c_gx < 0) return false;

  const auto f = [&](int c) { return std::stod(row[c]); };

  const Eigen::Quaterniond q(f(c_qw), f(c_qx), f(c_qy), f(c_qz));
  const M3D rot = q.normalized().toRotationMatrix();
  const V3D pos(f(c_px), f(c_py), f(c_pz));
  const V3D vel(f(c_vx), f(c_vy), f(c_vz));

  Eigen::MatrixXd P(dim, dim);
  for (int i = 0; i < dim; ++i)
    for (int j = i; j < dim; ++j)
    {
      const int c = need("P_" + std::to_string(i) + "_" + std::to_string(j));
      if (c < 0 || c >= static_cast<int>(row.size()) || row[c].empty()) return false;
      P(i, j) = std::stod(row[c]);
      P(j, i) = P(i, j);
    }

  state.setPropagatedState(rot, pos, vel);
  state.biasGyrMut() = V3D(f(c_bgx), f(c_bgy), f(c_bgz));
  state.biasAccMut() = V3D(f(c_bax), f(c_bay), f(c_baz));
  if (dim > StateGroup::idxG()) state.gravityMut() = V3D(f(c_gx), f(c_gy), f(c_gz));
  state.covMut() = P;
  return true;
}

}  // namespace livo_recon
