// R82 (D4, F-500): classifyVoxel decision table. Planner sandbox, not evidence.
#include <cstdio>
#include "livo_recon/lio/voxel_class.h"
using namespace livo_recon;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)
int main()
{
  LineOptions off;                                   // mode Off
  LineOptions hard; hard.mode = LineMode::HardAbsolute; hard.plane_threshold = 2.5e-3;
  LineOptions smooth; smooth.mode = LineMode::Smooth;
  const double thr = 2.5e-3;
  // VC1 line model off: old rules exactly.
  CHECK(classifyVoxel(off, 1e-4, 5e-1, 1.0, 20, thr).cls == VoxelClass::Plane);
  CHECK(classifyVoxel(off, 1e-9, 1e-9, 1.0, 20, thr).cls == VoxelClass::Reject);                       // e1 < 1e-8 rejects
  CHECK(classifyVoxel(off, 1e-9, 1e-9, 1.0, 20, thr).why == VoxelRejectWhy::Degenerate);
  CHECK(classifyVoxel(off, 3e-3, 5e-1, 1.0, 20, thr).why == VoxelRejectWhy::Blob);
  // VC2 hard form: a thin line (e1 < 1e-8 <= e2) is kept as a line, and flagged.
  VoxelClassResult a = classifyVoxel(hard, 1e-10, 1e-9, 1.0, 20, thr);
  CHECK(a.cls == VoxelClass::Line && a.e1_tiny_kept);
  // VC3 hard form: no direction at all (e2 < 1e-8) rejects.
  CHECK(classifyVoxel(hard, 1e-12, 1e-11, 1e-9, 20, thr).why == VoxelRejectWhy::Degenerate);
  // VC4 hard absolute: e1 below the threshold is a line, above is a plane.
  CHECK(classifyVoxel(hard, 1e-4, 1e-3, 1.0, 20, thr).cls == VoxelClass::Line);
  CHECK(classifyVoxel(hard, 1e-4, 1e-2, 1.0, 20, thr).cls == VoxelClass::Plane);
  // VC5 blob rejects before the line rule is asked (e0 >= threshold).
  CHECK(classifyVoxel(hard, 4e-3, 4e-3, 1.0, 20, thr).why == VoxelRejectWhy::Blob);
  // VC6 smooth form: no classes, every accepted voxel is a plane; the old degenerate test applies.
  CHECK(classifyVoxel(smooth, 1e-4, 1e-3, 1.0, 20, thr).cls == VoxelClass::Plane);
  CHECK(classifyVoxel(smooth, 1e-9, 1e-9, 1.0, 20, thr).why == VoxelRejectWhy::Degenerate);
  // VC7 non-finite rejects.
  CHECK(classifyVoxel(hard, std::nan(""), 1e-3, 1.0, 20, thr).why == VoxelRejectWhy::NonFinite);
  // VC8 hard shape: line when e1 <= tau e2 and e0 < threshold.
  LineOptions shape; shape.mode = LineMode::HardShape; shape.shape_ratio = 0.1;
  CHECK(classifyVoxel(shape, 1e-4, 5e-2, 1.0, 20, thr).cls == VoxelClass::Line);
  CHECK(classifyVoxel(shape, 1e-4, 5e-1, 1.0, 20, thr).cls == VoxelClass::Plane);
  std::printf(fails ? "FAILED %d\n" : "ALL PASS\n", fails);
  return fails != 0;
}
