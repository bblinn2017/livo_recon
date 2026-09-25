#include "livo_recon/lio/pose_control_spline.h"
#include "livo_recon/lio/pose_control_layout.h"
#include "livo_recon/lio/pose_control_imu_prior_builder.h"
#include "livo_recon/lio/pose_control_covariance.h"
#include "livo_recon/diagnostics/pose_control/pose_control_physical_diagnostics.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace livo_recon;
namespace {
int failures = 0;
void check(bool ok, const char* name, double value=0.0, double tol=0.0) {
  std::printf("  [%s] %-72s %.6e tol %.6e\n", ok?"PASS":"FAIL", name, value, tol);
  if (!ok) ++failures;
}
PoseControlSpline makeSpline(int N=7) {
  PoseControlSpline s; s.init(N,0.0,0.1); s.R_anchor=M3D::Identity();
  for(int k=0;k<N;++k){
    s.cp_p.col(k)=V3D(0.01*k+1e-4*k*k,-0.002*k+5e-5*k*k,0.001*k-3e-5*k*k);
    s.cp_phi.col(k)=V3D(0.001*k,-0.0005*k,0.0002*k);
  }
  return s;
}
std::vector<ImuSample> makeImu(const PoseControlSpline& s,double dt){
  const V3D g(0,0,-9.81); std::vector<ImuSample> out;
  for(double t=s.t0();t<s.t1()-0.25*dt;t+=dt){ ImuSample m; m.t=t; m.acc=s.rotAt(t).transpose()*(s.accAt(t)-g); m.gyro=s.omegaBodyAt(t); out.push_back(m); }
  return out;
}
PoseControlFreeLayout rawLayout(int N){ PoseControlFreeLayout l; l.N=N; l.has_bg=l.has_ba=l.has_g=false; return l; }
Eigen::VectorXd deterministicDirection(int d){ Eigen::VectorXd x(d); for(int i=0;i<d;++i)x(i)=std::sin(0.37*(i+1))+0.3*std::cos(0.11*(i+2)); return x.normalized(); }
double infoAlong(const PoseControlSpline&s,double dt,const V3D&va,const V3D&vw,const Eigen::VectorXd&d){
  auto l=rawLayout(s.N()); Eigen::MatrixXd A=Eigen::MatrixXd::Zero(l.dim(),l.dim()); Eigen::VectorXd b=Eigen::VectorXd::Zero(l.dim());
  buildPoseControlContinuousImuPrior(s,l,makeImu(s,dt),V3D::Zero(),V3D::Zero(),V3D(0,0,-9.81),va,vw,A,b,nullptr,nullptr); return d.dot(A*d);
}
double objective(const PoseControlSpline&s,const std::vector<ImuSample>&imu,const V3D&va,const V3D&vw){
  const V3D g(0,0,-9.81); double E=0;
  for(const auto&m:imu){ const V3D ea=s.rotAt(m.t).transpose()*(s.accAt(m.t)-g)-m.acc; const V3D ew=s.omegaBodyAt(m.t)-m.gyro;
    E += 0.5*(ea.array().square()/va.array()).sum()+0.5*(ew.array().square()/vw.array()).sum(); }
  return E;
}
void testHeadNullspace(){
  auto s=makeSpline(13); const V3D p0(0.3,-0.2,0.1),v0(1.0,-0.4,0.2); auto h=buildPoseControlHeadNullspace(s,p0,v0);
  Eigen::VectorXd eta(h.freeDim()); for(int i=0;i<eta.size();++i) eta(i)=0.05*std::sin(0.29*(i+1));
  poseControlUnflatten(h.c_particular+h.Z*eta,s);
  const double ep=(s.posAt(s.t0())-p0).norm(), ev=(s.velAt(s.t0())-v0).norm(), er=s.phiAt(s.t0()).norm();
  check(ep<1e-11,"head nullspace preserves p(t0) exactly",ep,1e-11);
  check(ev<1e-10,"head nullspace preserves v(t0) exactly",ev,1e-10);
  check(er<1e-11,"head nullspace preserves phi(t0)=0 exactly",er,1e-11);
  check(h.Z.cols()==6*s.N()-9,"head nullspace has exactly 6N-9 free dimensions",double(h.Z.cols()-(6*s.N()-9)),0.0);
  check((h.Z.transpose()*h.Z-Eigen::MatrixXd::Identity(h.freeDim(),h.freeDim())).norm()<1e-11,"head nullspace basis is orthonormal",(h.Z.transpose()*h.Z-Eigen::MatrixXd::Identity(h.freeDim(),h.freeDim())).norm(),1e-11);
}
void testFiniteDifferenceHessian(){
  auto s=makeSpline(7); auto l=rawLayout(s.N()); const double dt=0.002; auto imu=makeImu(s,dt); const V3D va=V3D::Constant(4e-4),vw=V3D::Constant(2.5e-5);
  Eigen::MatrixXd A=Eigen::MatrixXd::Zero(l.dim(),l.dim()); Eigen::VectorXd b=Eigen::VectorXd::Zero(l.dim()); buildPoseControlContinuousImuPrior(s,l,imu,V3D::Zero(),V3D::Zero(),V3D(0,0,-9.81),va,vw,A,b,nullptr,nullptr);
  const Eigen::VectorXd d=deterministicDirection(l.dim()), c0=poseControlFlatten(s); const double eps=2e-7;
  PoseControlSpline sp=s,sm=s; poseControlUnflatten(c0+eps*d,sp); poseControlUnflatten(c0-eps*d,sm);
  const double num=(objective(sp,imu,va,vw)+objective(sm,imu,va,vw)-2*objective(s,imu,va,vw))/(eps*eps); const double ana=d.dot(A*d); const double rel=std::abs(num-ana)/std::max({1.0,std::abs(num),std::abs(ana)});
  check(rel<2e-3,"production IMU Hessian matches independent objective finite difference",rel,2e-3);
}
void testDensityRefinement(){
  auto s=makeSpline(7); auto l=rawLayout(s.N()); const Eigen::VectorXd d=deterministicDirection(l.dim()); const V3D qa=V3D::Constant(4e-4),qw=V3D::Constant(2.5e-5); const std::vector<double>dts={0.004,0.002,0.001,0.0005}; std::vector<double>I;
  for(double dt:dts) I.push_back(infoAlong(s,dt,qa/dt,qw/dt,d));
  const double ref=I.back(); double worst=0; for(double x:I) worst=std::max(worst,std::abs(x/ref-1.0));
  std::printf("    density refinement I/ref: %.6f %.6f %.6f %.6f\n",I[0]/ref,I[1]/ref,I[2]/ref,I[3]/ref);
  check(worst<0.08,"continuous-time density information is stable over 8x timestep refinement",worst,0.08);
}
void testPhysicalCovariancePSD(){
  auto s=makeSpline(7); auto l=rawLayout(s.N()); auto h=buildPoseControlHeadNullspace(s,s.posAt(s.t0()),s.velAt(s.t0())); const double dt=0.001; const V3D qA=V3D::Constant(4e-4),qW=V3D::Constant(2.5e-5);
  Eigen::MatrixXd A=Eigen::MatrixXd::Zero(l.dim(),l.dim()); Eigen::VectorXd b=Eigen::VectorXd::Zero(l.dim()); buildPoseControlContinuousImuPrior(s,l,makeImu(s,dt),V3D::Zero(),V3D::Zero(),V3D(0,0,-9.81),qA/dt,qW/dt,A,b,nullptr,nullptr);
  Eigen::MatrixXd Aeta=h.Z.transpose()*A*h.Z, P=generalPseudoInverse(Aeta,1e-9); double worstNeg=0,worstSym=0;
  for(double f: {0.0,0.25,0.5,0.75,1.0}){ auto x=evaluatePoseControlPhysicalSample(s,l,h,s.t0()+f*(s.t1()-s.t0()),V3D(0,0,-9.81)); Eigen::MatrixXd J=Eigen::MatrixXd::Zero(12,h.freeDim()); J.block(0,0,3,h.freeDim())=x.dp_deta; J.block(3,0,3,h.freeDim())=x.dv_deta; J.block(6,0,3,h.freeDim())=x.da_deta; J.block(9,0,3,h.freeDim())=x.domega_deta; Eigen::MatrixXd C=J*P*J.transpose(); worstSym=std::max(worstSym,(C-C.transpose()).norm()); Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5*(C+C.transpose())); double scale=std::max(1.0,es.eigenvalues().cwiseAbs().maxCoeff()); worstNeg=std::max(worstNeg,std::max(0.0,-es.eigenvalues().minCoeff()/scale)); }
  check(worstSym<1e-8,"mapped physical covariance is symmetric",worstSym,1e-8); check(worstNeg<1e-9,"mapped physical covariance is PSD up to scale-aware roundoff",worstNeg,1e-9);
}
void testHighFrequencyAcrossN(){
  for(int N: {4,7,13}){ auto s=makeSpline(N); auto l=rawLayout(N); const V3D va=V3D::Constant(4e-4),vw=V3D::Constant(2.5e-5); Eigen::MatrixXd A=Eigen::MatrixXd::Zero(l.dim(),l.dim()); Eigen::VectorXd b=Eigen::VectorXd::Zero(l.dim()); buildPoseControlContinuousImuPrior(s,l,makeImu(s,0.001),V3D::Zero(),V3D::Zero(),V3D(0,0,-9.81),va,vw,A,b,nullptr,nullptr); Eigen::VectorXd hi=Eigen::VectorXd::Zero(l.dim()),lo=Eigen::VectorXd::Zero(l.dim()); for(int k=3;k<N;++k){hi(l.colPos(k))=((k&1)?-1.0:1.0);lo(l.colPos(k))=1.0;} double ch=hi.dot(A*hi),cl=lo.dot(A*lo),ratio=ch/std::max(cl,1e-12); std::printf("    N=%d high/low position-mode IMU cost ratio %.6e\n",N,ratio); check(std::isfinite(ratio)&&ratio>0,"high/low mode ratio is finite and positive",ratio,0); if(N>=7) check(ratio>2.0,"extra spline capacity makes alternating mode materially more expensive",ratio,2.0); }
}
}
int main(){ std::printf("Pose-control batched prior validation\n"); testHeadNullspace(); testFiniteDifferenceHessian(); testDensityRefinement(); testPhysicalCovariancePSD(); testHighFrequencyAcrossN(); std::printf("%d failure(s)\n",failures); return failures?1:0; }
