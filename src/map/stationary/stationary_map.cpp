#include "livo_recon/map/stationary/stationary_map.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
namespace livo_recon::stationary_map {
PlaneFit fitIncrementalPca(const PlaneStats&s,double max_ratio){PlaneFit f;if(s.n<3)return f;Eigen::SelfAdjointEigenSolver<M3> es(s.covariance());if(es.info()!=Eigen::Success)return f;f.eigenvalues=es.eigenvalues();f.center=s.mean();f.normal=es.eigenvectors().col(0).normalized();f.d=-f.normal.dot(f.center);double den=std::max(f.eigenvalues.sum(),1e-15);f.planarity=f.eigenvalues[0]/den;f.valid=f.planarity<=max_ratio;return f;}
namespace { struct Key{int x,y,z;bool operator<(const Key&o)const{return std::tie(x,y,z)<std::tie(o.x,o.y,o.z);}}; Key key(const V3&p,double l){return{int(std::floor(p.x()/l)),int(std::floor(p.y()/l)),int(std::floor(p.z()/l))};}
class MapImpl:public Backend{public:MapImpl(std::string n,Options o,bool merge,bool robust,bool gaussian):n_(std::move(n)),o_(o),merge_(merge),robust_(robust),gaussian_(gaussian){} std::string name()const override{return n_;}
void insert(std::uint64_t,const std::vector<V3>&pts)override{auto t=std::chrono::steady_clock::now();for(auto&p:pts){auto k=key(p,o_.leaf);auto&pa=cells_[k];if(!pa.id){pa.id=++next_;pa.surface_id=pa.id;}pa.stats.add(p);pa.bb_min=pa.bb_min.cwiseMin(p);pa.bb_max=pa.bb_max.cwiseMax(p);}for(auto&kv:cells_)kv.second.fit=fitIncrementalPca(kv.second.stats,o_.split_ratio);if(merge_)rebuildSurfaces();ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();}
Snapshot snapshot()const override{Snapshot s;s.backend=n_;s.insert_ms=ms_;s.merges=merges_;s.splits=splits_;s.unmerges=unmerges_;std::set<uint64_t> ids;for(auto&kv:cells_){if(kv.second.fit.valid){s.data.push_back(kv.second);ids.insert(kv.second.surface_id);}}s.patches=s.data.size();s.surfaces=ids.size();return s;}
private:void rebuildSurfaces(){std::vector<Patch*> v;for(auto&kv:cells_)if(kv.second.fit.valid)v.push_back(&kv.second);for(auto*p:v)p->surface_id=p->id;for(size_t i=0;i<v.size();++i)for(size_t j=i+1;j<v.size();++j){auto&a=*v[i];auto&b=*v[j];double c=std::abs(a.fit.normal.dot(b.fit.normal));c=std::clamp(c,-1.0,1.0);double ang=std::acos(c)*180/M_PI;double off=std::abs(a.fit.d-b.fit.d);double gap=(a.fit.center-b.fit.center).norm();if(ang<=o_.merge_angle_deg&&off<=o_.merge_offset&&gap<=o_.merge_gap){auto id=std::min(a.surface_id,b.surface_id);auto olda=a.surface_id,oldb=b.surface_id;for(auto*q:v)if(q->surface_id==olda||q->surface_id==oldb)q->surface_id=id;if(olda!=oldb)++merges_;}}}
std::string n_;Options o_;bool merge_,robust_,gaussian_;std::map<Key,Patch>cells_;uint64_t next_=0,merges_=0,splits_=0,unmerges_=0;double ms_=0;};}
std::unique_ptr<Backend> makeBackend(const std::string&f,const Options&o){if(f=="mergeable_voxel")return std::make_unique<MapImpl>(f,o,true,false,false);if(f=="robust_voxel")return std::make_unique<MapImpl>(f,o,false,true,false);if(f=="robust_mergeable")return std::make_unique<MapImpl>(f,o,true,true,false);if(f=="gaussian_surface")return std::make_unique<MapImpl>(f,o,true,false,true);if(f=="incremental_pca")return std::make_unique<MapImpl>(f,o,false,false,false);throw std::runtime_error("unknown stationary map backend: "+f);}
}
