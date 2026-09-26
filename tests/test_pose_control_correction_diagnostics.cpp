#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
namespace {
int failures=0;
void check(bool ok,const char* name,double v=0,double tol=0){std::printf("  [%s] %-75s %.6e tol %.6e\n",ok?"PASS":"FAIL",name,v,tol);if(!ok)++failures;}
struct Projection { double cosine=0, parallel=0, perpendicular_norm=0, pre_error=0, post_error=0, error_reduction=0; };
Projection project(const Eigen::Vector3d& b,const Eigen::Vector3d& a,const Eigen::Vector3d& gt){
 const Eigen::Vector3d d=a-b, g=gt-b; Projection o; o.pre_error=g.norm(); o.post_error=(a-gt).norm(); o.error_reduction=o.pre_error-o.post_error;
 if(d.norm()>1e-15&&g.norm()>1e-15){const auto u=g/g.norm();o.cosine=d.dot(g)/(d.norm()*g.norm());o.parallel=d.dot(u);o.perpendicular_norm=(d-o.parallel*u).norm();} return o;
}
void aligned(){auto p=project(Eigen::Vector3d::Zero(),Eigen::Vector3d(.25,0,0),Eigen::Vector3d(1,0,0));check(std::abs(p.cosine-1)<1e-12,"aligned correction cosine +1",p.cosine,1e-12);check(std::abs(p.parallel-.25)<1e-12,"aligned parallel projection",p.parallel,1e-12);check(p.perpendicular_norm<1e-12,"aligned perpendicular is zero",p.perpendicular_norm,1e-12);check(std::abs(p.error_reduction-.25)<1e-12,"aligned correction reduces GT error by its magnitude",p.error_reduction,1e-12);}
void orth(){auto p=project(Eigen::Vector3d::Zero(),Eigen::Vector3d(0,.5,0),Eigen::Vector3d(1,0,0));check(std::abs(p.cosine)<1e-12,"orthogonal correction cosine zero",p.cosine,1e-12);check(std::abs(p.parallel)<1e-12,"orthogonal correction parallel zero",p.parallel,1e-12);check(std::abs(p.perpendicular_norm-.5)<1e-12,"orthogonal component preserves magnitude",p.perpendicular_norm,1e-12);check(p.error_reduction<0,"orthogonal correction increases GT distance",p.error_reduction,0);}
void away(){auto p=project(Eigen::Vector3d::Zero(),Eigen::Vector3d(-.25,0,0),Eigen::Vector3d(1,0,0));check(std::abs(p.cosine+1)<1e-12,"away correction cosine -1",p.cosine,1e-12);check(p.parallel<0,"away correction parallel negative",p.parallel,0);check(p.error_reduction<0,"away correction increases GT error",p.error_reduction,0);}
void identity(){const Eigen::Vector3d b(.2,-.1,.3),a(.8,.5,-.4),gt(-.3,.9,.2);auto p=project(b,a,gt);const double lhs=(a-b).squaredNorm(),rhs=p.parallel*p.parallel+p.perpendicular_norm*p.perpendicular_norm;check(std::abs(lhs-rhs)<1e-12,"parallel/perpendicular decomposition preserves squared norm",std::abs(lhs-rhs),1e-12);}
}
int main(){std::printf("Pose-control correction-direction diagnostic math tests\n");aligned();orth();away();identity();std::printf("%d failure(s)\n",failures);return failures?1:0;}
