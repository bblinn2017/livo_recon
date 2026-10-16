#pragma once

// R91C. The text format of residual_dump.csv, moved out of lio_base.cpp so that a ROS-free unit test can compare it with the pre-R91C format byte for byte
// (tests/test_residual_dump_format_r91c.cpp). Templates on the residual type (livo_recon::Residual in the node, a mock with the same member names in the test).
//
// Default (geom = false, slim = false): header and rows are byte-identical to R71..R91 (with the voxel columns only under voxel_info, as since R81).
// geom = true  (log/lio/residual_dump/line_geometry): four columns appended after the voxel columns: line_axis_x, line_axis_y, line_axis_z, line_axial. line_axis is the unit
//     vector u2 = VoxelPlane::x_normal_ of the row's voxel (the third eigenvector of the point scatter: the LINE AXIS when the voxel is a line, the largest in-plane direction
//     otherwise); line_axial = u2 . (p - plane centre), p the world point. For a distance row (row_kind 6) hpos is the unit vector q from the line to the point and r = |d - (u2.d) u2|,
//     so the 2-D perpendicular displacement of the point from the line is r * q (no further column is needed for it).
// slim = true  (log/lio/residual_dump/slim): 14 columns instead of 29, enough to rebuild HtH and Htz (w = 1 / sigma_squared, r, hrot, hpos) and to join to the voxel map by node_id:
//     frame,iter,idx,r,sigma_squared,plane_var_term,node_id,row_kind,hrot_x..z,hpos_x..z. Dropped: source_index, t, plane_key, tier, vis, wx..wz, bx..bz, floor_term,
//     sigma_diag_squared, s_prior_pose, sigma_gate_squared. The voxel and geometry columns follow in the same order as in the full format.
// The caller sets std::setprecision(9) once, as before.
#include <cstdint>
#include <ostream>
#include <string>

namespace livo_recon
{
namespace residual_dump_format
{
inline void writeHeader(std::ostream& o, bool voxel_info, bool geom, bool slim)
{
  if (slim)
    o << "frame,iter,idx,r,sigma_squared,plane_var_term,node_id,row_kind,hrot_x,hrot_y,hrot_z,hpos_x,hpos_y,hpos_z";
  else
    o << "frame,iter,idx,source_index,t,r,sigma_squared,plane_var_term,plane_key,node_id,tier,vis,"
         "hrot_x,hrot_y,hrot_z,hpos_x,hpos_y,hpos_z,wx,wy,wz,bx,by,bz,row_kind,floor_term,sigma_diag_squared,s_prior_pose,sigma_gate_squared";
  if (voxel_info) o << ",voxel_point_count,voxel_distinct_frames,voxel_layer";
  if (geom) o << ",line_axis_x,line_axis_y,line_axis_z,line_axial";
  o << '\n';
}

template <class R>
inline void writeRow(std::ostream& o, int frame_idx, int iteration, size_t i, const R& r, bool voxel_info, bool geom, bool slim)
{
  if (slim) {
    o << frame_idx << ',' << iteration << ',' << i << ',' << r.r << ',' << r.sigma_squared << ',' << r.plane_var_term << ',' << r.plane_node_id << ',' << r.row_kind
      << ',' << r.point_cross_normal.x() << ',' << r.point_cross_normal.y() << ',' << r.point_cross_normal.z()
      << ',' << r.normal.x() << ',' << r.normal.y() << ',' << r.normal.z();
  } else {
    o << frame_idx << ',' << iteration << ',' << i << ',' << r.source_index << ',' << r.t << ',' << r.r << ',' << r.sigma_squared << ','
      << r.plane_var_term << ',' << reinterpret_cast<uintptr_t>(r.plane_id) << ',' << r.plane_node_id << ',' << r.match_tier << ',' << r.vis_state
      << ',' << r.point_cross_normal.x() << ',' << r.point_cross_normal.y() << ',' << r.point_cross_normal.z()
      << ',' << r.normal.x() << ',' << r.normal.y() << ',' << r.normal.z()
      << ',' << r.world_point.x() << ',' << r.world_point.y() << ',' << r.world_point.z()
      << ',' << r.raw_body_point.x() << ',' << r.raw_body_point.y() << ',' << r.raw_body_point.z() << ',' << r.row_kind
      << ',' << r.floor_term << ',' << r.sigma_diag_squared << ',' << r.s_prior_pose << ',' << r.sigma_gate_squared;
  }
  if (voxel_info) o << ',' << r.voxel_point_count << ',' << r.voxel_distinct_frames << ',' << r.voxel_layer;
  if (geom) o << ',' << r.line_axis.x() << ',' << r.line_axis.y() << ',' << r.line_axis.z() << ',' << r.line_axial;
  o << '\n';
}
}  // namespace residual_dump_format
}  // namespace livo_recon
