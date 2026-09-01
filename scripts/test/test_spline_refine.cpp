// refineWithLidar(): does a plane-constrained Gauss-Newton step on the
// POSITION control points actually recover a known intra-scan shape error,
// and do its guards hold?
#include "livo_recon/lio/spline.h"
#include <cstdio>
#include <vector>
#include <random>
using namespace livo_recon;
static int fails=0;
static void ck(bool ok,const char*n,const char*d=""){printf("  [%s] %s %s\n",ok?" ok ":"FAIL",n,d);if(!ok)fails++;}

static std::vector<Pose6D> mk(double t0,double t1,int n){
  std::vector<Pose6D> ps;
  for(int i=0;i<n;++i){double t=t0+(t1-t0)*i/double(n-1);Pose6D p{};p.t=t;
    p.rot=Exp(V3D(0.2,-0.3,0.9).normalized()*(0.8*t));
    p.pos=V3D(0.9*std::sin(2.1*t)+0.3*t,0.6*std::cos(1.7*t),0.25*std::sin(3.3*t));
    p.dt=(t1-t0)/(n-1);ps.push_back(p);} return ps;}

int main(){
  const double t0=0,t1=0.1;
  auto poses=mk(t0,t1,40);

  // Truth spline, then a copy whose interior control points are displaced --
  // an intra-scan SHAPE error of a known size, with the endpoints left alone.
  SplineOptions base; base.n_control_points=8; base.rot_mode="tangent";
  ScanSpline truth; truth.fit(poses,t0,t1,base);

  // Three orthogonal plane families, so the observations span R^3 and the
  // problem is not rank-deficient (the corridor case is tested separately).
  const V3D nrm[3]={V3D(1,0,0),V3D(0,1,0),V3D(0,0,1)};
  std::vector<SplineLidarObs> obs;
  std::mt19937 rng(7); std::uniform_real_distribution<double> ut(t0,t1);
  std::vector<double> ts; for(int i=0;i<600;++i) ts.push_back(ut(rng));

  SplineOptions ro=base; ro.lidar_refine_cp=true; ro.lidar_refine_prior_w=1e-6;
  ro.lidar_refine_damping=1e-6; ro.lidar_refine_iters=3; ro.lidar_refine_max_step=1.0;

  ScanSpline bent=truth;
  for(int i=2;i<6;++i) bent.cpPosMut().col(i)+=V3D(0.02,-0.015,0.01);
  double err0=0; for(double t=t0;t<=t1+1e-12;t+=0.002) err0=std::max(err0,(bent.posAt(t)-truth.posAt(t)).norm());

  for(int it=0;it<ro.lidar_refine_iters;++it){
    obs.clear();
    for(size_t k=0;k<ts.size();++k){
      const V3D& n=nrm[k%3];
      SplineLidarObs o; o.t=ts[k]; o.normal=n; o.sigma2=1e-4;
      o.r=n.dot(bent.posAt(ts[k])-truth.posAt(ts[k]));   // n^T(p_now - p_true)
      obs.push_back(o);
    }
    SplineOptions one=ro; one.lidar_refine_iters=1;
    bent.refineWithLidar(obs,one);
  }
  double err1=0; for(double t=t0;t<=t1+1e-12;t+=0.002) err1=std::max(err1,(bent.posAt(t)-truth.posAt(t)).norm());
  char b[200]; snprintf(b,sizeof b,"shape err %.3e -> %.3e m (%.0fx), %d steps applied",err0,err1,err0/std::max(err1,1e-15),bent.refineApplied());
  ck(err1 < err0*0.02, "refinement recovers a known intra-scan shape error", b);

  // Guard: an oversized step is rejected whole, spline untouched.
  { ScanSpline s=truth; for(int i=2;i<6;++i) s.cpPosMut().col(i)+=V3D(5.0,0,0);
    auto before=s.cpPos();
    std::vector<SplineLidarObs> o2;
    for(size_t k=0;k<ts.size();++k){const V3D& n=nrm[k%3];SplineLidarObs o;o.t=ts[k];o.normal=n;o.sigma2=1e-4;
      o.r=n.dot(s.posAt(ts[k])-truth.posAt(ts[k]));o2.push_back(o);}
    SplineOptions g=ro; g.lidar_refine_iters=1; g.lidar_refine_max_step=0.10;
    bool applied=s.refineWithLidar(o2,g);
    ck(!applied && s.cpPos().isApprox(before,0.0) && s.refineRejects()==1,
       "oversized step rejected whole, control points bit-unchanged"); }

  // Guard: rank-deficient observations (one plane family only -- the corridor
  // case) must not blow up the unconstrained directions. The prior holds them.
  { ScanSpline s=truth; for(int i=2;i<6;++i) s.cpPosMut().col(i)+=V3D(0.02,-0.015,0.01);
    std::vector<SplineLidarObs> o3;
    for(size_t k=0;k<ts.size();++k){SplineLidarObs o;o.t=ts[k];o.normal=V3D(1,0,0);o.sigma2=1e-4;
      o.r=V3D(1,0,0).dot(s.posAt(ts[k])-truth.posAt(ts[k]));o3.push_back(o);}
    SplineOptions g=ro; g.lidar_refine_iters=1; g.lidar_refine_prior_w=1.0;
    s.refineWithLidar(o3,g);
    double dx=0,dyz=0;
    for(double t=t0;t<=t1+1e-12;t+=0.002){V3D d=s.posAt(t)-truth.posAt(t);
      dx=std::max(dx,std::abs(d.x())); dyz=std::max(dyz,std::hypot(d.y(),d.z()));}
    snprintf(b,sizeof b,"constrained axis %.3e m, unconstrained %.3e m",dx,dyz);
    ck(std::isfinite(dx)&&std::isfinite(dyz)&&dyz<0.03, "single-normal (corridor) case stays bounded",b); }

  // Guard: off by default is a hard no-op.
  { ScanSpline s=truth; auto before=s.cpPos();
    SplineOptions g=base;   // lidar_refine_cp defaults false
    ck(!s.refineWithLidar(obs,g) && s.cpPos().isApprox(before,0.0), "disabled is a hard no-op"); }

  printf("\n%s (%d failures)\n",fails?"FAILURES":"ALL CHECKS PASSED",fails);
  return fails?1:0;
}
