#pragma once
// Surface map core: a flat grid of fixed-size cells, each holding O(1) additive plane statistics (the same
// accumulators VoxelPlane's "debiased" path keeps), plus a merge/split layer that groups neighbouring cell planes
// into "surfaces" (the stationary harness's `gaussian_surface` family, threshold merge test, combined-fit no-chaining
// check, footprint support test). Dependency-free (Eigen + std only) so it can be unit-tested without ROS.
//
// Frame: whatever frame the caller feeds points in (the live backend feeds WORLD points). Every angle/offset test is
// sign invariant (eigenvector signs are arbitrary).
//
// Pose covariance is NOT part of this layer (plane_fit_pose_cov_mode = sensor_only): the covariance accumulators use
// the per-point SENSOR covariance only.
//
// Update model (dirty set): addPoint() marks the touched cell dirty; update() refits only dirty cells, recomputes the
// surfaces that contain them, tests merge candidates only for pairs with at least one dirty cell, and re-checks splits
// only for touched surfaces. All loops run in ascending cell-id order, so results do not depend on thread count.
#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

namespace livo_recon
{
namespace surface
{

using V3 = Eigen::Vector3d;
using M3 = Eigen::Matrix3d;

struct CellKey
{
  int x = 0, y = 0, z = 0;
  bool operator==(const CellKey& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct CellKeyHash
{
  std::size_t operator()(const CellKey& k) const
  {
    std::uint64_t h = static_cast<std::uint32_t>(k.x) * 73856093ULL;
    h ^= static_cast<std::uint32_t>(k.y) * 19349663ULL;
    h ^= static_cast<std::uint32_t>(k.z) * 83492791ULL;
    return static_cast<std::size_t>(h);
  }
};

struct Options
{
  double cell_size = 0.25;
  // Cell validity (VoxelPlane::refitDebiased chain).
  int    min_points = 5;              // fewer accumulated points -> no plane
  double plane_threshold = 2.5e-3;    // eig0 (debiased) must be below this
  double min_secondary_eig = 1e-8;    // rank-2 test on eig1, eig2
  double denom_floor_scale = 0.1;     // eigengap denominators must exceed max(1e-8, scale*plane_threshold)
  double plane_var_ceiling = 1.0;     // trace(plane_var) above this -> not a plane
  // Merge / split (metres and degrees).
  double merge_angle_deg = 5.0;
  double merge_offset = 0.05;
  double merge_gap = 0.75;            // 3-D distance between the two cell plane centres
  bool   support_test = true;         // footprint test: gap between the two cells' point bounding boxes
  double support_gap = 0.25;
  double split_planarity_max = 0.20;  // combined fit with planarity above this is "degraded" (forces a split)
};

// Additive plane statistics. sp/spp are about `ref` (fixed per cell = its centre) for numerical health; scov, v, w are
// origin-referenced exactly like VoxelPlane's Scov_/V_/W_ (they only enter through differences that are formed once).
struct SurfaceStats
{
  double n = 0.0;
  V3 ref = V3::Zero();
  V3 sp = V3::Zero();     // sum (p - ref)
  M3 spp = M3::Zero();    // sum (p - ref)(p - ref)^T
  M3 scov = M3::Zero();   // sum sensor covariance
  M3 v[3][3];             // v[a][b] = sum p(a) p(b) Cov
  M3 w[3];                // w[a]    = sum p(a) Cov
  SurfaceStats();
  void setRef(const V3& r) { ref = r; }    // only meaningful while n == 0
  void add(const V3& p, const M3& sensor_cov);
  void merge(const SurfaceStats& o);       // exact: re-references o onto this->ref (adopts o if this is empty)
  V3 mean() const { return ref + sp / n; }
  M3 covariance() const;                   // debiased scatter: spp/n - m m^T - scov/n
};

struct Fit
{
  bool valid = false;
  bool denom_rejected = false;
  bool ceiling_rejected = false;
  double n = 0.0;
  V3 center = V3::Zero();
  V3 normal = V3::Zero();
  V3 x_axis = V3::Zero();   // eigenvector of the LARGEST eigenvalue (VoxelPlane::x_normal_)
  V3 y_axis = V3::Zero();   // eigenvector of the middle eigenvalue (VoxelPlane::y_normal_)
  V3 eig = V3::Zero();      // ascending; eig(0) clipped at 0
  double d = 0.0;           // -normal . center
  double radius = 0.0;      // sqrt(eig(2)) as float, like VoxelPlane::radius_
  double planarity = 0.0;   // eig0 / sum(eig)
  M3 plane_var = M3::Zero();  // over [theta1, theta2, h]: VoxelPlane's eigengap plane_var
};

Fit fitStats(const SurfaceStats& s, const Options& o);
double planeAngleDeg(const V3& na, const V3& nb);        // sign invariant
double planeOffset(const Fit& a, const Fit& b);          // sign invariant: max(|na.dc|, |nb.dc|)
double boxGap(const V3& amin, const V3& amax, const V3& bmin, const V3& bmax);

struct Cell
{
  std::uint64_t id = 0;       // 1-based, creation order
  CellKey key;
  SurfaceStats st;
  Fit fit;
  V3 bb_min = V3::Constant(1e300);
  V3 bb_max = V3::Constant(-1e300);
  std::uint64_t surface_id = 0;
  bool dirty = false;
};

struct Surface
{
  std::uint64_t id = 0;                     // = id of its seed cell
  std::vector<std::uint64_t> children;      // ascending cell ids
  SurfaceStats combined;                    // sum of the VALID children's stats
  Fit fit;
};

struct UpdateStats
{
  std::size_t dirty_cells = 0;
  std::size_t cells_valid_dirty = 0;
  std::uint64_t pair_tests = 0;        // neighbour pairs looked at
  std::uint64_t gate_rejects = 0;      // angle / offset / centre gap
  std::uint64_t support_rejects = 0;
  std::uint64_t combined_rejects = 0;  // combined-fit (no-chaining) check refused
  std::uint64_t merges = 0;
  std::uint64_t splits = 0;
  std::uint64_t denom_rejected = 0;    // dirty cells whose eigengap denominator guard tripped
};

struct Summary
{
  std::size_t cells = 0, cells_valid = 0;
  std::size_t surfaces = 0;            // all surfaces (singletons included)
  std::size_t surfaces_multi = 0;      // surfaces with >= 2 valid child cells
  std::size_t patches_in_multi = 0;    // valid cells belonging to those
  std::size_t largest = 0;             // most valid cells in one surface
  double points_valid = 0.0;           // points inside valid cells
  double points_total = 0.0;
};

class SurfaceLayer
{
public:
  explicit SurfaceLayer(const Options& o);
  const Options& options() const { return o_; }

  // Insert one point (in the layer's frame) with its sensor covariance; marks the cell dirty.
  void addPoint(const V3& p, const M3& sensor_cov);
  // Refit dirty cells and run the dirty-set merge/split pass. Clears the dirty set.
  UpdateStats update();

  bool empty() const { return cells_.empty(); }
  std::size_t numCells() const { return cells_.size(); }
  std::size_t numSurfaces() const { return surfaces_.size(); }
  std::size_t numDirty() const { return dirty_.size(); }
  std::uint64_t totalMerges() const { return merges_; }
  std::uint64_t totalSplits() const { return splits_; }
  Summary summary() const;                         // O(cells + surfaces); diagnostics only

  CellKey keyOf(const V3& p) const;
  const Cell* findCell(const CellKey& k) const;
  const Cell& cellById(std::uint64_t id) const { return cells_[id - 1]; }
  const Surface* findSurface(std::uint64_t id) const;
  // The plane a query in `c` should be judged against: the surface plane when c belongs to a valid multi-cell surface,
  // else the cell's own plane, else nullptr. `owner` (optional) receives the address of the Surface or Cell used.
  const Fit* planeFor(const Cell& c, const void** owner = nullptr) const;

  // Deterministic iteration (ascending id) for diagnostics.
  template <class F> void forEachCell(F&& f) const { for (const Cell& c : cells_) f(c); }
  std::vector<std::uint64_t> surfaceIds() const;   // ascending

private:
  struct Consistency
  {
    bool ok = true;
    std::uint64_t worst = 0;
    double worst_score = -1.0;
    double worst_ang = 0.0, worst_off = 0.0, planarity = 0.0;
  };
  Cell& getCell(const CellKey& k);
  void recomputeSurface(Surface& s);
  Consistency evaluate(const SurfaceStats& u, const std::vector<std::uint64_t>& children, bool invalid_violates) const;
  bool tryMerge(std::uint64_t ida, std::uint64_t idb, UpdateStats& us);
  void reevaluateSplits(std::uint64_t sid, UpdateStats& us);

  Options o_;
  std::deque<Cell> cells_;                                   // stable addresses, id = index + 1
  std::unordered_map<CellKey, std::uint32_t, CellKeyHash> index_;
  std::unordered_map<std::uint64_t, Surface> surfaces_;      // node based: stable addresses
  std::vector<std::uint64_t> dirty_;
  std::vector<CellKey> offsets_;                             // neighbour offsets (ordered), excludes (0,0,0)
  std::uint64_t merges_ = 0, splits_ = 0;
};

}  // namespace surface
}  // namespace livo_recon
