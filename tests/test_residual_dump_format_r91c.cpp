// R91C: residual_dump.csv format (include/livo_recon/diagnostics/residual_dump_format.h). The default path must be byte-identical to the pre-R91C code, which is copied
// below verbatim (OLD_*) as the reference; the new columns (geom) and the slim form are checked against explicit expected text. Header-only; no ROS, no Eigen (mock types).
#include "livo_recon/diagnostics/residual_dump_format.h"
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

struct V3 { double a, b, c; double x() const { return a; } double y() const { return b; } double z() const { return c; } };
struct MockRes
{
  long source_index = 0; double t = 0, r = 0, sigma_squared = 0, plane_var_term = 0; const void* plane_id = nullptr; long plane_node_id = 0; int match_tier = 0, vis_state = 0;
  V3 point_cross_normal{0, 0, 0}, normal{0, 0, 0}, world_point{0, 0, 0}, raw_body_point{0, 0, 0};
  int row_kind = 0; double floor_term = 0, sigma_diag_squared = 0, s_prior_pose = 0, sigma_gate_squared = 0;
  int voxel_point_count = -1, voxel_distinct_frames = -1, voxel_layer = -1;
  V3 line_axis{0, 0, 0}; double line_axial = 0;
};

// ---- the pre-R91C code of logResidualDump(), copied verbatim from the R91 tree (header string and row loop) ----
static void oldHeader(std::ostream& o, bool voxel_info)
{
  o << "frame,iter,idx,source_index,t,r,sigma_squared,plane_var_term,plane_key,node_id,tier,vis,"
       "hrot_x,hrot_y,hrot_z,hpos_x,hpos_y,hpos_z,wx,wy,wz,bx,by,bz,row_kind,floor_term,sigma_diag_squared,s_prior_pose,sigma_gate_squared"
    << (voxel_info ? ",voxel_point_count,voxel_distinct_frames,voxel_layer" : "") << '\n';
}
static void oldRows(std::ostream& o, int frame_idx, int iteration, const std::vector<MockRes>& residuals, bool voxel_info)
{
  o << std::setprecision(9);
  for (size_t i = 0; i < residuals.size(); ++i) {
    const auto& r = residuals[i];
    o << frame_idx << ',' << iteration << ',' << i << ',' << r.source_index << ',' << r.t << ',' << r.r << ',' << r.sigma_squared << ','
      << r.plane_var_term << ',' << reinterpret_cast<uintptr_t>(r.plane_id) << ',' << r.plane_node_id << ',' << r.match_tier << ',' << r.vis_state
      << ',' << r.point_cross_normal.x() << ',' << r.point_cross_normal.y() << ',' << r.point_cross_normal.z()
      << ',' << r.normal.x() << ',' << r.normal.y() << ',' << r.normal.z()
      << ',' << r.world_point.x() << ',' << r.world_point.y() << ',' << r.world_point.z()
      << ',' << r.raw_body_point.x() << ',' << r.raw_body_point.y() << ',' << r.raw_body_point.z() << ',' << r.row_kind
      << ',' << r.floor_term << ',' << r.sigma_diag_squared << ',' << r.s_prior_pose << ',' << r.sigma_gate_squared;
    if (voxel_info) o << ',' << r.voxel_point_count << ',' << r.voxel_distinct_frames << ',' << r.voxel_layer;
    o << '\n';
  }
}

static int fails = 0;
#define CK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static std::vector<MockRes> sample()
{
  static int dummy[2];
  std::vector<MockRes> v(3);
  v[0].source_index = 11; v[0].t = 0.0123456789; v[0].r = -0.012345678901; v[0].sigma_squared = 1.2345678e-4; v[0].plane_var_term = 3.3e-6; v[0].plane_id = &dummy[0];
  v[0].plane_node_id = 987654321; v[0].match_tier = 1; v[0].vis_state = 2; v[0].point_cross_normal = {0.1, -0.2, 0.3}; v[0].normal = {0.0, 0.6, 0.8};
  v[0].world_point = {1.5, -2.25, 3.125}; v[0].raw_body_point = {4.0, 5.5, -6.75}; v[0].row_kind = 0; v[0].floor_term = 1e-5; v[0].sigma_diag_squared = 2e-5;
  v[0].s_prior_pose = 7e-7; v[0].sigma_gate_squared = 9e-5; v[0].voxel_point_count = 40; v[0].voxel_distinct_frames = 3; v[0].voxel_layer = 0;
  v[0].line_axis = {1, 0, 0}; v[0].line_axial = 0.0125;
  v[1] = v[0]; v[1].source_index = 12; v[1].row_kind = 6; v[1].r = 0.0302; v[1].normal = {0.6, 0.8, 0.0}; v[1].line_axis = {0, 0, 1}; v[1].line_axial = -0.0431;
  v[2] = v[1]; v[2].source_index = 13; v[2].plane_node_id = -1; v[2].voxel_point_count = -1;
  return v;
}

int main()
{
  const auto rs = sample();
  for (int vi = 0; vi < 2; ++vi) {
    const bool voxel_info = (vi == 1);
    // default path byte-identical to the old code (header and rows)
    std::ostringstream a, b;
    oldHeader(a, voxel_info); oldRows(a, 5, 2, rs, voxel_info);
    livo_recon::residual_dump_format::writeHeader(b, voxel_info, false, false);
    b << std::setprecision(9);
    for (size_t i = 0; i < rs.size(); ++i) livo_recon::residual_dump_format::writeRow(b, 5, 2, i, rs[i], voxel_info, false, false);
    CK(a.str() == b.str());
    // geom: the old text plus four columns on each line, nothing else changed
    std::ostringstream g;
    livo_recon::residual_dump_format::writeHeader(g, voxel_info, true, false);
    g << std::setprecision(9);
    for (size_t i = 0; i < rs.size(); ++i) livo_recon::residual_dump_format::writeRow(g, 5, 2, i, rs[i], voxel_info, true, false);
    std::istringstream ia(a.str()), ig(g.str()); std::string la, lg; int n = 0;
    while (std::getline(ia, la) && std::getline(ig, lg)) {
      ++n;
      CK(lg.compare(0, la.size(), la) == 0);
      const std::string tail = lg.substr(la.size());
      if (n == 1) CK(tail == ",line_axis_x,line_axis_y,line_axis_z,line_axial");
      if (n == 2) CK(tail == ",1,0,0,0.0125");
      if (n == 3) CK(tail == ",0,0,1,-0.0431");
    }
    CK(n == 4);
  }
  // slim: 14 columns (+3 voxel, +4 geom), exact text
  {
    std::ostringstream s;
    livo_recon::residual_dump_format::writeHeader(s, true, true, true);
    s << std::setprecision(9);
    livo_recon::residual_dump_format::writeRow(s, 5, 2, 1, rs[1], true, true, true);
    const std::string want =
        "frame,iter,idx,r,sigma_squared,plane_var_term,node_id,row_kind,hrot_x,hrot_y,hrot_z,hpos_x,hpos_y,hpos_z,voxel_point_count,voxel_distinct_frames,voxel_layer,line_axis_x,line_axis_y,line_axis_z,line_axial\n"
        "5,2,1,0.0302,0.00012345678,3.3e-06,987654321,6,0.1,-0.2,0.3,0.6,0.8,0,40,3,0,0,0,1,-0.0431\n";
    CK(s.str() == want);
    if (s.str() != want) std::printf("got:\n%s\n", s.str().c_str());
    // header and rows have the same number of columns
    std::istringstream is(s.str()); std::string h, r; std::getline(is, h); std::getline(is, r);
    int ch = 1, cr = 1; for (char c : h) ch += (c == ','); for (char c : r) cr += (c == ',');
    CK(ch == cr && ch == 21);
  }
  // the full header has 29 columns (voxel off), 32 (voxel on), 36 (voxel + geom); a row has the same number
  for (int k = 0; k < 3; ++k) {
    const bool vox = k >= 1, geom = (k == 2);
    std::ostringstream s; livo_recon::residual_dump_format::writeHeader(s, vox, geom, false); s << std::setprecision(9);
    livo_recon::residual_dump_format::writeRow(s, 1, 0, 0, rs[0], vox, geom, false);
    std::istringstream is(s.str()); std::string h, r; std::getline(is, h); std::getline(is, r);
    int ch = 1, cr = 1; for (char c : h) ch += (c == ','); for (char c : r) cr += (c == ',');
    CK(ch == cr); CK(ch == (k == 0 ? 29 : k == 1 ? 32 : 36));
  }
  std::printf("test_residual_dump_format_r91c %s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
