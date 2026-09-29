#pragma once
#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
namespace livo_recon::stationary_map {
using V3=Eigen::Vector3d; using M3=Eigen::Matrix3d;
struct PlaneStats { std::uint64_t n=0; V3 sum=V3::Zero(); M3 sum_outer=M3::Zero(); void add(const V3&p){++n;sum+=p;sum_outer+=p*p.transpose();} void merge(const PlaneStats&o){n+=o.n;sum+=o.sum;sum_outer+=o.sum_outer;} V3 mean()const{return n?V3(sum/double(n)):V3::Zero();} M3 covariance()const{if(n<2)return M3::Zero();auto m=mean();return sum_outer/double(n)-m*m.transpose();}};
struct PlaneFit { bool valid=false; V3 center=V3::Zero(),normal=V3::UnitZ(); double d=0,planarity=0; Eigen::Vector3d eigenvalues=Eigen::Vector3d::Zero(); };
PlaneFit fitIncrementalPca(const PlaneStats& s,double max_ratio=0.08);

// R52 completion: bounded per-leaf reservoir + dominant-plane RANSAC for
// the robust_voxel/robust_mergeable families (see stationary_map.cpp's
// RobustCell -- this header only exposes the pieces the CLI/tests need).
struct RansacResult { bool found=false; V3 normal=V3::UnitZ(); double d=0; std::vector<int> inlier_idx; };
RansacResult ransacPlane(const std::vector<V3>& pts, double dist_thresh, int iters, std::uint64_t seed);

struct Patch { std::uint64_t id=0; PlaneStats stats; PlaneFit fit; V3 bb_min=V3::Constant(1e30),bb_max=V3::Constant(-1e30); std::uint64_t surface_id=0; std::vector<std::uint64_t> children;
  // R52: robust-family diagnostics (0 for non-robust families).
  std::uint64_t rejected_points=0, reservoir_pending=0; };
struct MergeEvent { std::string kind; std::uint64_t surface_id=0; std::vector<std::uint64_t> members; std::string reason; };
struct Snapshot { std::string backend; std::size_t patches=0,surfaces=0; std::uint64_t merges=0,splits=0,unmerges=0; double insert_ms=0; std::vector<Patch> data; std::vector<MergeEvent> events; };
class Backend { public: virtual ~Backend()=default; virtual std::string name()const=0; virtual void insert(std::uint64_t obs,const std::vector<V3>&pts)=0; virtual Snapshot snapshot()const=0; };
struct Options { double leaf=0.25; std::size_t min_points=12; double merge_angle_deg=5,merge_offset=0.05,merge_gap=0.75; double split_ratio=0.10; std::size_t robust_reservoir=96; double robust_ransac_dist=0.02; int robust_ransac_iters=200; double split_planarity_max=0.20; };
std::unique_ptr<Backend> makeBackend(const std::string& family,const Options& o);
}
